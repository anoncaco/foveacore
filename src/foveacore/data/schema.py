"""FoveaCore Phase 0 T3 공개 스키마 및 정책 상수.

공개 split manifest에는 라벨을 복원할 수 있는 필드를 포함하지 않습니다.
클래스·writer·source 메타데이터는 후속 validator 전용 저장소에서 관리합니다.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "1.0"


class _StringEnum(str, Enum):
    """JSON 및 기존 문자열 기반 API와 호환되는 열거형."""

    def __str__(self) -> str:
        return self.value


class SplitName(_StringEnum):
    SUPPORT_POOL = "support_pool"
    DEV_STREAM = "dev_stream"
    MAIN_STREAM = "main_stream"
    OFFICIAL_TEST = "official_test"
    RESERVED_SEALED = "reserved_sealed"
    P2_NON_DIGIT = "p2_non_digit"
    NON_DIGIT_DEV = "non_digit_dev"
    DEV_OOD = "dev_ood"
    TEST_OOD = "test_ood"
    PHASE7_CHAR_TRANSFER = "phase7_char_transfer"


class SealLevel(_StringEnum):
    OPEN = "OPEN"
    SEALED = "SEALED"


class ExecutionMode(_StringEnum):
    INFER = "INFER"
    ADAPT = "ADAPT"
    EVALUATE = "EVALUATE"


class ExecutionProtocol(_StringEnum):
    STRICT_1SHOT = "STRICT_1SHOT"
    CONTINUAL = "CONTINUAL"
    REFERENCE_P3 = "REFERENCE_P3"


SPLIT_NAMES = frozenset(split.value for split in SplitName)
SEALED_LEVELS = frozenset(level.value for level in SealLevel)
EXECUTION_MODES = frozenset(mode.value for mode in ExecutionMode)
EXECUTION_PROTOCOLS = frozenset(protocol.value for protocol in ExecutionProtocol)

SEALED_SPLITS = frozenset(
    {
        SplitName.OFFICIAL_TEST.value,
        SplitName.RESERVED_SEALED.value,
        SplitName.DEV_OOD.value,
        SplitName.TEST_OOD.value,
        SplitName.PHASE7_CHAR_TRANSFER.value,
    }
)

P2_SPLITS = frozenset(
    {SplitName.P2_NON_DIGIT.value, SplitName.NON_DIGIT_DEV.value}
)
OOD_TRANSFER_SPLITS = frozenset(
    {
        SplitName.DEV_OOD.value,
        SplitName.TEST_OOD.value,
        SplitName.PHASE7_CHAR_TRANSFER.value,
    }
)

# T3 문서의 8쌍을 단일 정의합니다.
OOD_TRANSFER_DISJOINT_RULES = (
    (SplitName.P2_NON_DIGIT.value, SplitName.DEV_OOD.value),
    (SplitName.P2_NON_DIGIT.value, SplitName.TEST_OOD.value),
    (SplitName.P2_NON_DIGIT.value, SplitName.PHASE7_CHAR_TRANSFER.value),
    (SplitName.NON_DIGIT_DEV.value, SplitName.DEV_OOD.value),
    (SplitName.NON_DIGIT_DEV.value, SplitName.TEST_OOD.value),
    (SplitName.NON_DIGIT_DEV.value, SplitName.PHASE7_CHAR_TRANSFER.value),
    (SplitName.DEV_OOD.value, SplitName.TEST_OOD.value),
    (SplitName.TEST_OOD.value, SplitName.PHASE7_CHAR_TRANSFER.value),
)

# 기존 원문 5쌍은 레거시로만 보존합니다.
LEGACY_DISJOINT_RULES = (
    OOD_TRANSFER_DISJOINT_RULES[:3] + OOD_TRANSFER_DISJOINT_RULES[6:]
)
# 기존 import 경로는 호환하되, 불완전한 5쌍 검증을 유도하지 않도록 공식 8쌍으로 연결합니다.
DISJOINT_RULES = OOD_TRANSFER_DISJOINT_RULES
DISJOINT_GROUPS = tuple(
    (name, (split,)) for name, split in (
        ("P2", SplitName.P2_NON_DIGIT.value),
        ("dev_ood", SplitName.DEV_OOD.value),
        ("test_ood", SplitName.TEST_OOD.value),
        ("phase7_char_transfer", SplitName.PHASE7_CHAR_TRANSFER.value),
    )
)

PUBLIC_ENTRY_FIELDS = frozenset(
    {"sample_id", "split_name", "sha256", "phash", "sealed_level"}
)
FORBIDDEN_PUBLIC_ENTRY_FIELDS = frozenset(
    {"label", "class_id", "writer_id", "alphabet_id", "source_id"}
)
VALIDATION_METADATA_FIELDS = frozenset(
    {"class_id", "writer_id", "alphabet_id", "source_id"}
)
FREEZE_HASH_FIELDS = (
    "data_split_hash",
    "config_hash",
    "checkpoint_hash",
)
REQUIRED_FREEZE_FIELDS = frozenset(
    {"schema_version", "manifest_sha256", *FREEZE_HASH_FIELDS}
)


class SchemaError(ValueError):
    """T3 공개 스키마 또는 요청 스냅샷이 계약을 위반했습니다."""


def _enum_value(enum_cls, value, field_name: str):
    if isinstance(value, enum_cls):
        return value.value
    if not isinstance(value, str) or value not in {
        member.value for member in enum_cls
    }:
        allowed = ", ".join(member.value for member in enum_cls)
        raise SchemaError(f"{field_name}은(는) 다음 중 하나여야 합니다: {allowed}")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name}은(는) 비어있지 않은 문자열이어야 합니다.")
    return value


def _optional_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _id_set(values: Any, field_name: str) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise SchemaError(f"{field_name}은(는) sample_id 집합이어야 합니다.")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise SchemaError(f"{field_name}은(는) sample_id 집합이어야 합니다.") from exc
    normalized = set()
    for index, value in enumerate(items):
        normalized.add(_required_string(value, f"{field_name}[{index}]"))
    return frozenset(normalized)


def _freeze_mapping(value: Any, field_name: str) -> MappingProxyType:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name}은(는) mapping이어야 합니다.")

    frozen = {}
    for key, item in value.items():
        if isinstance(key, str):
            frozen_key = key
        elif isinstance(key, (int, float, bool)):
            frozen_key = key
        else:
            raise SchemaError(f"{field_name}에 지원하지 않는 키 타입이 있습니다.")
        if isinstance(item, Mapping):
            frozen[frozen_key] = _freeze_mapping(item, f"{field_name}.{key}")
        elif isinstance(item, (list, tuple)):
            frozen[frozen_key] = tuple(
                _freeze_mapping(v, f"{field_name}.{key}")
                if isinstance(v, Mapping)
                else tuple(_freeze_mapping(x, f"{field_name}.{key}") for x in v)
                if isinstance(v, (list, tuple))
                else v
                for v in item
            )
        else:
            frozen[frozen_key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class SplitEntry:
    """모델에 공개되는 라벨 없는 분할 엔트리."""

    sample_id: str
    split_name: str
    sha256: str
    phash: str
    sealed_level: str

    def __post_init__(self):
        for name in ("sample_id", "sha256", "phash"):
            _required_string(getattr(self, name), name)
        object.__setattr__(
            self,
            "split_name",
            _enum_value(SplitName, self.split_name, "split_name"),
        )
        object.__setattr__(
            self,
            "sealed_level",
            _enum_value(SealLevel, self.sealed_level, "sealed_level"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "sha256": self.sha256,
            "phash": self.phash,
            "sealed_level": self.sealed_level,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitEntry":
        if not isinstance(data, Mapping):
            raise SchemaError("SplitEntry 데이터는 mapping이어야 합니다.")
        missing = PUBLIC_ENTRY_FIELDS.difference(data)
        if missing:
            raise SchemaError(f"SplitEntry 필수 필드 누락: {sorted(missing)}")
        forbidden = FORBIDDEN_PUBLIC_ENTRY_FIELDS.intersection(data)
        if forbidden:
            raise SchemaError(f"공개 SplitEntry 금지 필드: {sorted(forbidden)}")
        return cls(**{name: data[name] for name in PUBLIC_ENTRY_FIELDS})


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    split_name: str
    sealed_level: str
    entries: tuple[SplitEntry, ...]

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version 불일치: {self.schema_version} != {SCHEMA_VERSION}"
            )
        split_name = _enum_value(SplitName, self.split_name, "split_name")
        sealed_level = _enum_value(SealLevel, self.sealed_level, "sealed_level")
        object.__setattr__(self, "split_name", split_name)
        object.__setattr__(self, "sealed_level", sealed_level)

        if isinstance(self.entries, (str, bytes)):
            raise SchemaError("entries는 SplitEntry 시퀀스여야 합니다.")
        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise SchemaError("entries는 SplitEntry 시퀀스여야 합니다.") from exc

        seen = set()
        for entry in entries:
            if isinstance(entry, Mapping):
                entry = SplitEntry.from_dict(entry)
            if not isinstance(entry, SplitEntry):
                raise SchemaError("entries에는 SplitEntry만 포함할 수 있습니다.")
            if entry.split_name != split_name:
                raise SchemaError(
                    f"split_name 불일치: manifest={split_name}, entry={entry.split_name}"
                )
            if entry.sealed_level != sealed_level:
                raise SchemaError(
                    f"sealed_level 불일치: manifest={sealed_level}, "
                    f"entry {entry.sample_id}={entry.sealed_level}"
                )
            if entry.sample_id in seen:
                raise SchemaError(f"중복 sample_id: {entry.sample_id}")
            seen.add(entry.sample_id)
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_name": self.split_name,
            "sealed_level": self.sealed_level,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        if not isinstance(data, Mapping):
            raise SchemaError("SplitManifest 데이터는 mapping이어야 합니다.")
        required = {"schema_version", "split_name", "sealed_level", "entries"}
        missing = required.difference(data)
        if missing:
            raise SchemaError(f"SplitManifest 필수 필드 누락: {sorted(missing)}")
        forbidden = {"labels", "ground_truth"}.intersection(data)
        if forbidden:
            raise SchemaError(f"공개 SplitManifest 금지 필드: {sorted(forbidden)}")
        return cls(
            schema_version=data["schema_version"],
            split_name=data["split_name"],
            sealed_level=data["sealed_level"],
            entries=tuple(
                SplitEntry.from_dict(entry) if isinstance(entry, Mapping) else entry
                for entry in data["entries"]
            ),
        )


@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """원문 필드명을 유지하는 동결 manifest 스냅샷."""

    schema_version: str
    manifest_sha256: str
    data_split_hash: str
    config_hash: str
    checkpoint_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version 불일치: {self.schema_version} != {SCHEMA_VERSION}"
            )
        for name in REQUIRED_FREEZE_FIELDS - {"schema_version"}:
            _required_string(getattr(self, name), name)
        if isinstance(self.metadata, Mapping):
            metadata = _freeze_mapping(self.metadata, "metadata")
        else:
            raise SchemaError("metadata는 mapping이어야 합니다.")
        overlap = REQUIRED_FREEZE_FIELDS.intersection(metadata)
        if overlap:
            raise SchemaError(f"metadata가 manifest 필드를 덮어씁니다: {sorted(overlap)}")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "data_split_hash": self.data_split_hash,
            "config_hash": self.config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            **dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FreezeManifest":
        if not isinstance(data, Mapping):
            raise SchemaError("FreezeManifest 데이터는 mapping이어야 합니다.")
        missing = REQUIRED_FREEZE_FIELDS.difference(data)
        if missing:
            raise SchemaError(f"FreezeManifest 필수 필드 누락: {sorted(missing)}")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in REQUIRED_FREEZE_FIELDS
        }
        return cls(
            schema_version=data["schema_version"],
            manifest_sha256=data["manifest_sha256"],
            data_split_hash=data["data_split_hash"],
            config_hash=data["config_hash"],
            checkpoint_hash=data["checkpoint_hash"],
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class EvaluationGrant:
    """특정 평가에서 승인된 봉인 split 집합."""

    evaluation_id: str
    phase: int
    approved_splits: frozenset[str]

    def __post_init__(self):
        object.__setattr__(
            self, "evaluation_id", _required_string(self.evaluation_id, "evaluation_id")
        )
        if isinstance(self.phase, bool) or not isinstance(self.phase, int):
            raise SchemaError("phase은 정수여야 합니다.")
        if self.phase < 0:
            raise SchemaError("phase은 0 이상이어야 합니다.")
        splits = _id_set(self.approved_splits, "approved_splits")
        unknown = splits.difference(SPLIT_NAMES)
        if unknown:
            raise SchemaError(f"알 수 없는 승인 split: {sorted(unknown)}")
        object.__setattr__(self, "approved_splits", splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "phase": self.phase,
            "approved_splits": sorted(self.approved_splits),
        }


@dataclass(frozen=True, slots=True)
class LabelAccessRequest:
    """라벨 접근 판정에 필요한 요청 스냅샷. 실제 라벨은 포함하지 않습니다."""

    execution_mode: str
    protocol: str
    split_name: str
    sample_ids: frozenset[str]
    episode_id: str | None = None
    current_episode_support_ids: frozenset[str] = frozenset()
    approved_stream_ids: frozenset[str] = frozenset()
    query_ids: frozenset[str] = frozenset()
    evaluation_grant: EvaluationGrant | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "execution_mode",
            _enum_value(ExecutionMode, self.execution_mode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "protocol",
            _enum_value(ExecutionProtocol, self.protocol, "protocol"),
        )
        object.__setattr__(
            self,
            "split_name",
            _enum_value(SplitName, self.split_name, "split_name"),
        )
        for name in (
            "sample_ids",
            "current_episode_support_ids",
            "approved_stream_ids",
            "query_ids",
        ):
            object.__setattr__(self, name, _id_set(getattr(self, name), name))
        object.__setattr__(
            self, "episode_id", _optional_id(self.episode_id, "episode_id")
        )
        if self.evaluation_grant is not None and not isinstance(
            self.evaluation_grant, EvaluationGrant
        ):
            raise SchemaError("evaluation_grant는 EvaluationGrant여야 합니다.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "protocol": self.protocol,
            "split_name": self.split_name,
            "sample_ids": sorted(self.sample_ids),
            "episode_id": self.episode_id,
            "current_episode_support_ids": sorted(self.current_episode_support_ids),
            "approved_stream_ids": sorted(self.approved_stream_ids),
            "query_ids": sorted(self.query_ids),
            "evaluation_grant": (
                self.evaluation_grant.to_dict()
                if self.evaluation_grant is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelAccessRequest":
        if not isinstance(data, Mapping):
            raise SchemaError("LabelAccessRequest 데이터는 mapping이어야 합니다.")
        required = {
            "execution_mode",
            "protocol",
            "split_name",
            "sample_ids",
        }
        missing = required.difference(data)
        if missing:
            raise SchemaError(f"LabelAccessRequest 필수 필드 누락: {sorted(missing)}")
        grant = data.get("evaluation_grant")
        if isinstance(grant, Mapping):
            grant = EvaluationGrant(
                evaluation_id=grant.get("evaluation_id"),
                phase=grant.get("phase"),
                approved_splits=grant.get("approved_splits", ()),
            )
        return cls(
            execution_mode=data["execution_mode"],
            protocol=data["protocol"],
            split_name=data["split_name"],
            sample_ids=data["sample_ids"],
            episode_id=data.get("episode_id"),
            current_episode_support_ids=data.get("current_episode_support_ids", ()),
            approved_stream_ids=data.get("approved_stream_ids", ()),
            query_ids=data.get("query_ids", ()),
            evaluation_grant=grant,
        )


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """허용 여부와 사유만 담은 판정 결과. 라벨을 포함하지 않습니다."""

    allowed: bool
    reason: str
    allowed_sample_ids: frozenset[str] = frozenset()
    denied_sample_ids: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.allowed, bool):
            raise SchemaError("allowed는 bool이어야 합니다.")
        object.__setattr__(self, "reason", _required_string(self.reason, "reason"))
        for name in ("allowed_sample_ids", "denied_sample_ids"):
            object.__setattr__(self, name, _id_set(getattr(self, name), name))
        overlap = self.allowed_sample_ids & self.denied_sample_ids
        if overlap:
            raise SchemaError(f"허용/차단 sample_id 충돌: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "allowed_sample_ids": sorted(self.allowed_sample_ids),
            "denied_sample_ids": sorted(self.denied_sample_ids),
        }


__all__ = [
    "AccessDecision",
    "DISJOINT_GROUPS",
    "DISJOINT_RULES",
    "EvaluationGrant",
    "EXECUTION_MODES",
    "EXECUTION_PROTOCOLS",
    "ExecutionMode",
    "ExecutionProtocol",
    "FORBIDDEN_PUBLIC_ENTRY_FIELDS",
    "FREEZE_HASH_FIELDS",
    "FreezeManifest",
    "LEGACY_DISJOINT_RULES",
    "LabelAccessRequest",
    "OOD_TRANSFER_DISJOINT_RULES",
    "OOD_TRANSFER_SPLITS",
    "P2_SPLITS",
    "PUBLIC_ENTRY_FIELDS",
    "REQUIRED_FREEZE_FIELDS",
    "SCHEMA_VERSION",
    "SEALED_LEVELS",
    "SEALED_SPLITS",
    "SchemaError",
    "SealLevel",
    "SPLIT_NAMES",
    "SplitEntry",
    "SplitManifest",
    "SplitName",
    "VALIDATION_METADATA_FIELDS",
]
