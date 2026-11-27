"""T3 validator 전용 기밀 메타데이터 저장소.

공개 split manifest와 분리하며 class·writer·source 격리 검증에만 사용합니다.
모델·실험 루프에는 공개 엔트리와 이 메타데이터를 결합하는 API를 제공하지 않습니다.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from foveacore.data.schema import (
    SPLIT_NAMES,
    VALIDATION_METADATA_FIELDS,
    SchemaError,
    _enum_value,
    _required_string,
    SplitName,
)


class ValidationMetadataError(ValueError):
    """검증 메타데이터가 공개 샘플 또는 스키마와 일치하지 않습니다."""


@dataclass(frozen=True, slots=True)
class ValidationMetadata:
    split_name: str
    sample_id: str
    class_id: str
    writer_id: str | None
    alphabet_id: str | None
    source_id: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "split_name",
            _enum_value(SplitName, self.split_name, "split_name"),
        )
        object.__setattr__(
            self, "sample_id", _required_string(self.sample_id, "sample_id")
        )
        for name in ("class_id", "source_id"):
            object.__setattr__(
                self, name, _required_string(getattr(self, name), name)
            )
        if self.writer_id is not None:
            object.__setattr__(
                self, "writer_id", _required_string(self.writer_id, "writer_id")
            )
        if self.alphabet_id is not None:
            object.__setattr__(
                self,
                "alphabet_id",
                _required_string(self.alphabet_id, "alphabet_id"),
            )

    def to_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "sample_id": self.sample_id,
            "class_id": self.class_id,
            "writer_id": self.writer_id,
            "alphabet_id": self.alphabet_id,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ValidationMetadata":
        if not isinstance(data, Mapping):
            raise ValidationMetadataError("메타데이터는 mapping이어야 합니다.")
        try:
            return cls(
                split_name=data["split_name"],
                sample_id=data["sample_id"],
                class_id=data["class_id"],
                writer_id=data.get("writer_id"),
                alphabet_id=data.get("alphabet_id"),
                source_id=data["source_id"],
            )
        except KeyError as exc:
            raise ValidationMetadataError(
                f"검증 메타데이터 필수 필드 누락: {exc.args[0]}"
            ) from exc


def _records_from_json(data) -> list[ValidationMetadata]:
    if isinstance(data, Mapping) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValidationMetadataError(
            "검증 메타데이터 JSON은 records 목록이어야 합니다."
        )
    return [ValidationMetadata.from_dict(record) for record in data]


class ValidationMetadataStore:
    """sample_id와 split_name으로 기밀 메타데이터를 분리 보관합니다."""

    def __init__(self, records: Iterable[ValidationMetadata | Mapping] = ()):
        self._records: dict[tuple[str, str], ValidationMetadata] = {}
        for record in records:
            if isinstance(record, Mapping):
                record = ValidationMetadata.from_dict(record)
            if not isinstance(record, ValidationMetadata):
                raise ValidationMetadataError(
                    "records에는 ValidationMetadata 또는 mapping만 포함할 수 있습니다."
                )
            key = (record.split_name, record.sample_id)
            existing = self._records.get(key)
            if existing is not None and existing != record:
                raise ValidationMetadataError(
                    f"sample_id 메타데이터 충돌: {record.split_name}/{record.sample_id}"
                )
            self._records[key] = record

    @classmethod
    def from_records(
        cls, records: Iterable[ValidationMetadata | Mapping]
    ) -> "ValidationMetadataStore":
        return cls(records)

    @classmethod
    def load(cls, path: Path | str) -> "ValidationMetadataStore":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationMetadataError(f"메타데이터 JSON 파싱 실패: {p}") from exc
        return cls(_records_from_json(data))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        records = [
            record.to_dict()
            for record in sorted(
                self._records.values(),
                key=lambda item: (item.split_name, item.sample_id),
            )
        ]
        p.write_text(
            json.dumps({"records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, split_name: str, sample_id: str) -> ValidationMetadata:
        key = (split_name, sample_id)
        try:
            return self._records[key]
        except KeyError as exc:
            raise ValidationMetadataError(
                f"검증 메타데이터 누락: {split_name}/{sample_id}"
            ) from exc

    def contains(self, split_name: str, sample_id: str) -> bool:
        return (split_name, sample_id) in self._records

    def values_for_split(self, split_name: str) -> tuple[ValidationMetadata, ...]:
        if split_name not in SPLIT_NAMES:
            raise ValidationMetadataError(f"알 수 없는 split_name: {split_name}")
        return tuple(
            record
            for (name, _), record in sorted(self._records.items())
            if name == split_name
        )

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "ValidationMetadata",
    "ValidationMetadataError",
    "ValidationMetadataStore",
    "VALIDATION_METADATA_FIELDS",
]
