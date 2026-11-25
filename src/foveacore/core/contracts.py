"""FoveaCore Phase 0 공통 인터페이스 계약 (T2, v1.9.3 부록 1)."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from numbers import Real
from types import MappingProxyType

import numpy as np
import torch

from foveacore.config import SENSOR_PROFILES
from foveacore.core.exceptions import (
    ImmutabilityViolationError,
    ensure_finite,
)

SCHEMA_VERSION = "1.0"
SENSOR_PROFILE_VERSION = "1.0"
COORDINATE_SYSTEM = "unit_01"
_IMAGE_MODALITIES = frozenset({"image", "visual"})


@dataclass(frozen=True)
class SensorProfileContract:
    """기본/확장 센서 프로파일을 호환 불가능한 별도 계약으로 관리한다."""

    profile_id: str
    profile_version: str
    channels: int
    grid_size: int

    @property
    def contract_id(self):
        return f"{self.profile_id}@{self.profile_version}"

    @property
    def feature_shape(self):
        return (self.channels, self.grid_size, self.grid_size)


SENSOR_PROFILE_CONTRACTS = MappingProxyType(
    {
        name: SensorProfileContract(
            name,
            SENSOR_PROFILE_VERSION,
            channels=profile["channels"],
            grid_size=profile["grid_size"],
        )
        for name, profile in SENSOR_PROFILES.items()
    }
)


def _positive_hw(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name}는 (height, width) 길이 2 시퀀스여야 합니다.")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in value):
        raise ValueError(f"{name}의 원소는 양의 정수여야 합니다.")
    return (int(value[0]), int(value[1]))


def validate_image_provenance(provenance):
    """이미지 Observation provenance의 센서·좌표 계약을 검증한다."""
    if not isinstance(provenance, Mapping):
        raise ValueError("이미지 Observation의 provenance는 mapping이어야 합니다.")
    required = {
        "original_hw",
        "processed_hw",
        "preprocess_profile_id",
        "sensor_profile",
        "grid_size",
    }
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"이미지 provenance 필수 키 누락: {sorted(missing)}")

    _positive_hw(provenance["original_hw"], "original_hw")
    _positive_hw(provenance["processed_hw"], "processed_hw")
    profile_id = provenance["preprocess_profile_id"]
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("preprocess_profile_id는 비어있지 않은 문자열이어야 합니다.")

    sensor_profile = provenance["sensor_profile"]
    if not isinstance(sensor_profile, str) or sensor_profile not in SENSOR_PROFILE_CONTRACTS:
        raise ValueError(f"알 수 없는 sensor_profile: {sensor_profile!r}")
    contract = SENSOR_PROFILE_CONTRACTS[sensor_profile]
    grid_size = provenance["grid_size"]
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise ValueError("grid_size는 정수여야 합니다.")
    if int(grid_size) != contract.grid_size:
        raise ValueError(
            f"grid_size 불일치: {sensor_profile}은 {contract.grid_size}이어야 합니다({grid_size})."
        )
    version = provenance.get("sensor_profile_version", contract.profile_version)
    if version != contract.profile_version:
        raise ValueError(f"sensor_profile_version 불일치: {version!r}")
    contract_id = provenance.get("profile_contract_id", contract.contract_id)
    if contract_id != contract.contract_id:
        raise ValueError(f"profile_contract_id 불일치: {contract_id!r}")
    if provenance.get("coordinate_system", COORDINATE_SYSTEM) != COORDINATE_SYSTEM:
        raise ValueError(f"coordinate_system은 {COORDINATE_SYSTEM}이어야 합니다.")
    return provenance


def make_image_provenance(
    *,
    original_hw,
    processed_hw,
    preprocess_profile_id,
    sensor_profile="F3",
    grid_size=None,
    **metadata,
):
    """검증된 이미지 provenance 매핑을 생성한다."""
    protected = {
        "original_hw",
        "processed_hw",
        "preprocess_profile_id",
        "sensor_profile",
        "grid_size",
        "sensor_profile_version",
        "profile_contract_id",
        "coordinate_system",
    }
    overlap = protected & set(metadata)
    if overlap:
        raise ValueError(f"provenance 기본값을 덮어쓸 수 없습니다: {sorted(overlap)}")
    if not isinstance(sensor_profile, str) or sensor_profile not in SENSOR_PROFILE_CONTRACTS:
        raise ValueError(f"알 수 없는 sensor_profile: {sensor_profile!r}")
    contract = SENSOR_PROFILE_CONTRACTS[sensor_profile]
    provenance = {
        "original_hw": _positive_hw(original_hw, "original_hw"),
        "processed_hw": _positive_hw(processed_hw, "processed_hw"),
        "preprocess_profile_id": preprocess_profile_id,
        "sensor_profile": sensor_profile,
        "sensor_profile_version": contract.profile_version,
        "profile_contract_id": contract.contract_id,
        "grid_size": contract.grid_size if grid_size is None else grid_size,
        "coordinate_system": COORDINATE_SYSTEM,
        **metadata,
    }
    validate_image_provenance(provenance)
    return provenance


def _validate_image_features(features, sensor_profile):
    """이미지 features가 CHW 또는 BCHW이고 프로파일 형상과 일치하는지 검증한다."""
    if features is None:
        return
    if isinstance(features, ImmutableTensor):
        shape = tuple(features.tensor().shape)
    else:
        try:
            shape = tuple(np.shape(features))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "이미지 Observation features는 CHW 또는 BCHW 배열이어야 합니다."
            ) from exc
    if len(shape) == 4:
        shape = shape[1:]
    elif len(shape) != 3:
        raise ValueError(f"이미지 Observation features는 CHW 또는 BCHW여야 합니다: {shape}")
    contract = SENSOR_PROFILE_CONTRACTS[sensor_profile]
    if shape != contract.feature_shape:
        raise ValueError(
            f"features 형상 불일치: {sensor_profile}은 {contract.feature_shape}이어야 합니다({shape})."
        )


def validate_unit_coordinates(value, name="location"):
    """이미지 좌표의 모든 숫자 값이 [0, 1] 범위인지 검증한다."""
    if value is None:
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} 좌표는 [0, 1] 범위여야 합니다: {value!r}")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            validate_unit_coordinates(item, name)
        return
    if isinstance(value, np.ndarray):
        validate_unit_coordinates(value.tolist(), name)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_unit_coordinates(item, name)
        return
    raise ValueError(f"{name} 좌표는 숫자 또는 중첩 시퀀스여야 합니다: {value!r}")


def _immutable_ndarray(value):
    """숫자 배열을 불변 bytes 버퍼로 복사해 쓰기 재활성화를 차단한다."""
    if value.dtype.hasobject:
        arr = value.copy()
        arr.setflags(write=False)
        return arr
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _freeze(obj):
    """내부 데이터를 방어 복사하고 불변 형태로 변환한다."""
    if isinstance(obj, Mapping):
        if "__numpy__" in obj and set(obj) <= {"__numpy__", "dtype"}:
            arr = np.array(obj["__numpy__"], dtype=obj.get("dtype"))
            return _immutable_ndarray(arr)
        if "__tensor__" in obj and set(obj) == {"__tensor__"}:
            return ImmutableTensor.from_dict(obj["__tensor__"])
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return _immutable_ndarray(obj)
    if isinstance(obj, torch.Tensor):
        return ImmutableTensor(obj)
    if isinstance(obj, _SerializableContract):
        return obj
    return obj


def _to_plain(obj):
    """불변 객체를 JSON 직렬화 가능한 일반 형태로 변환한다."""
    if isinstance(obj, (MappingProxyType, Mapping)):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__numpy__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, ImmutableTensor):
        return {"__tensor__": obj.to_dict()}
    if isinstance(obj, _SerializableContract):
        return obj.to_dict()
    return obj


class ImmutableTensor:
    """Tensor 방어 복사 + SHA-256 스냅샷 기반 변경 탐지(T2)."""

    def __init__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("ImmutableTensor는 torch.Tensor를 받아야 합니다.")
        self._tensor = tensor.detach().clone()
        self._snapshot = self._digest()

    def _digest(self):
        t = self._tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        h = hashlib.sha256()
        h.update(raw)
        h.update(f"|{tuple(t.shape)}|{t.dtype}".encode("utf-8"))
        return h.hexdigest()

    def tensor(self):
        """무결성을 검증한 뒤 조회용 방어 복사를 반환한다."""
        self.assert_unchanged()
        return self._tensor.detach().clone()

    def snapshot_hash(self):
        self.assert_unchanged()
        return self._snapshot

    def assert_unchanged(self):
        if self._digest() != self._snapshot:
            raise ImmutabilityViolationError("ImmutableTensor 내부 상태 변경 감지(T2)")

    def to_dict(self):
        t = self.tensor()
        return {
            "schema_version": SCHEMA_VERSION,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "data": t.cpu().tolist(),
            "snapshot": self._snapshot,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("ImmutableTensor 직렬화 데이터는 mapping이어야 합니다.")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {data.get('schema_version')!r}")
        shape = data.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape
        ):
            raise ValueError("ImmutableTensor shape가 올바르지 않습니다.")
        dtype_name = str(data.get("dtype", "")).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"지원하지 않는 Tensor dtype: {data.get('dtype')!r}")
        tensor = torch.tensor(data["data"], dtype=dtype).reshape(shape)
        obj = cls(tensor)
        snapshot = data.get("snapshot")
        if snapshot is not None and snapshot != obj._snapshot:
            raise ImmutabilityViolationError("ImmutableTensor snapshot 검증 실패(T2)")
        return obj


def immutable_tensor(tensor):
    return ImmutableTensor(tensor)


class _SerializableContract:
    def to_dict(self):
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["schema_version"] = SCHEMA_VERSION
        return _to_plain(data)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__} 직렬화 데이터는 mapping이어야 합니다.")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {version} != {SCHEMA_VERSION}")
        return cls(**{k: v for k, v in data.items() if k != "schema_version"})


def _validate_modality(modality):
    if not isinstance(modality, str) or not modality:
        raise ValueError("modality는 비어있지 않은 문자열이어야 합니다.")


def _validate_nonnegative_cost(value, name):
    ensure_finite(value, name)
    if value < 0:
        raise ValueError(f"{name}는 0 이상이어야 합니다.")


@dataclass(frozen=True)
class Observation(_SerializableContract):
    """Observation (v1.9.3 부록 1)."""

    modality: str
    source_id: str
    location: object = None
    features: object = None
    observed_mask: object = None
    confidence: float = 1.0
    cost: float = 0.0
    provenance: object = None

    def __post_init__(self):
        _validate_modality(self.modality)
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id는 비어있지 않은 문자열이어야 합니다.")
        ensure_finite(self.confidence, "confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence는 [0, 1] 범위여야 합니다.")
        _validate_nonnegative_cost(self.cost, "cost")
        if self.modality.lower() in _IMAGE_MODALITIES:
            validate_image_provenance(self.provenance)
            validate_unit_coordinates(self.location)
        object.__setattr__(self, "location", _freeze(self.location))
        object.__setattr__(self, "features", _freeze(self.features))
        object.__setattr__(self, "observed_mask", _freeze(self.observed_mask))
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        if self.modality.lower() in _IMAGE_MODALITIES:
            _validate_image_features(self.features, self.provenance["sensor_profile"])


@dataclass(frozen=True)
class Action(_SerializableContract):
    """Action (v1.9.3 부록 1): 센서 원본 조회 명령."""

    modality: str
    target: object = None
    resolution: object = None
    cost: float = 0.0
    window: object = None

    def __post_init__(self):
        _validate_modality(self.modality)
        _validate_nonnegative_cost(self.cost, "cost")
        object.__setattr__(self, "target", _freeze(self.target))
        object.__setattr__(self, "resolution", _freeze(self.resolution))
        object.__setattr__(self, "window", _freeze(self.window))


@dataclass(frozen=True)
class Structure(_SerializableContract):
    """Structure (v1.9.3 부록 1)."""

    entity: object = None
    attribute: object = None
    relation: object = None
    embedding: object = None
    uncertainty: float = 0.0
    source_evidence: object = None

    def __post_init__(self):
        ensure_finite(self.uncertainty, "uncertainty")
        if self.uncertainty < 0:
            raise ValueError("uncertainty는 0 이상이어야 합니다.")
        for name in ("entity", "attribute", "relation", "embedding", "source_evidence"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class Hypothesis(_SerializableContract):
    """Hypothesis (v1.9.3 부록 1)."""

    concept_id: str = ""
    belief: float = 0.5
    supporting_evidence: object = None
    contradicting_evidence: object = None

    def __post_init__(self):
        if not isinstance(self.concept_id, str):
            raise ValueError("concept_id는 문자열이어야 합니다.")
        ensure_finite(self.belief, "belief")
        if not 0.0 <= self.belief <= 1.0:
            raise ValueError("belief는 [0, 1] 범위여야 합니다.")
        object.__setattr__(self, "supporting_evidence", _freeze(self.supporting_evidence))
        object.__setattr__(self, "contradicting_evidence", _freeze(self.contradicting_evidence))


@dataclass(frozen=True)
class Prediction(_SerializableContract):
    """Prediction (v1.9.3 부록 1)."""

    modality_specific_distribution: object = None
    candidate_observation: object = None
    expected_cost: float = 0.0

    def __post_init__(self):
        _validate_nonnegative_cost(self.expected_cost, "expected_cost")
        object.__setattr__(
            self, "modality_specific_distribution", _freeze(self.modality_specific_distribution)
        )
        object.__setattr__(self, "candidate_observation", _freeze(self.candidate_observation))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("Prediction 직렬화 데이터는 mapping이어야 합니다.")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {version} != {SCHEMA_VERSION}")
        payload = {k: v for k, v in data.items() if k != "schema_version"}
        candidate = payload.get("candidate_observation")
        if isinstance(candidate, Mapping) and "modality" in candidate:
            payload["candidate_observation"] = Observation.from_dict(candidate)
        return cls(**payload)


@dataclass(frozen=True)
class Memory(_SerializableContract):
    """Memory (v1.9.3 부록 1)."""

    concept_record: object = None
    evidence_links: object = None
    modality_views: object = None
    versions: object = None
    uncertainty: object = None

    def __post_init__(self):
        for name in ("concept_record", "evidence_links", "modality_views", "versions", "uncertainty"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
