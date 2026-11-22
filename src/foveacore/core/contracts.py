"""FoveaCore Phase 0 공통 인터페이스 계약 (T2, v1.9.3 부록 1)."""

import hashlib
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

import numpy as np
import torch

from foveacore.core.exceptions import ImmutabilityViolationError

SCHEMA_VERSION = "1.0"


def _freeze(obj):
    """내부 데이터 불변 변환(T2)."""
    if isinstance(obj, dict):
        if "__numpy__" in obj:
            arr = np.array(obj["__numpy__"], dtype=obj.get("dtype", None))
            arr = arr.copy()
            arr.setflags(write=False)
            return arr
        if "__tensor__" in obj:
            return ImmutableTensor.from_dict(obj["__tensor__"])
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, np.ndarray):
        arr = obj.copy()
        arr.setflags(write=False)
        return arr
    if isinstance(obj, torch.Tensor):
        return ImmutableTensor(obj)
    return obj


def _to_plain(obj):
    """불변 객체를 JSON 직렬화 가능한 일반 형태로 변환."""
    if isinstance(obj, MappingProxyType):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__numpy__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, ImmutableTensor):
        return {"__tensor__": obj.to_dict()}
    return obj


class ImmutableTensor:
    """torch.Tensor 방어 복사 + SHA-256 바이트 체크섬 스냅샷 기반 변경 탐지(T2)."""

    def __init__(self, tensor):
        self._tensor = tensor.detach().clone()
        self._snapshot = self._digest()

    def _digest(self):
        t = self._tensor
        h = hashlib.sha256()
        h.update(t.detach().cpu().numpy().tobytes())
        h.update(f"|{tuple(t.shape)}|{t.dtype}".encode("utf-8"))
        return h.hexdigest()

    def tensor(self):
        """조회용 방어 복사 반환."""
        return self._tensor.detach().clone()

    def snapshot_hash(self):
        return self._snapshot

    def assert_unchanged(self):
        """스냅샷 이후 내부 상태 변경 감지 시 ImmutabilityViolationError raise."""
        if self._digest() != self._snapshot:
            raise ImmutabilityViolationError("ImmutableTensor 내부 상태 변경 감지(T2)")

    def to_dict(self):
        t = self._tensor
        return {
            "schema_version": SCHEMA_VERSION,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "data": t.detach().cpu().tolist(),
            "snapshot": self._snapshot,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data.pop("schema_version", None)
        t = torch.tensor(data["data"], dtype=getattr(torch, data["dtype"].replace("torch.", "")))
        t = t.reshape(data["shape"])
        obj = cls(t)
        # restore snapshot if provided
        if "snapshot" in data:
            obj._snapshot = data["snapshot"]
        return obj


def immutable_tensor(tensor):
    """생성 시 `.detach().clone()` 방어 복사를 적용하는 팩토리(T2)."""
    return ImmutableTensor(tensor)


@dataclass(frozen=True)
class Observation:
    """Observation (v1.9.3 부록 1): modality, source_id, location/span, features, observed_mask, confidence, cost, provenance"""

    modality: str
    source_id: str
    location: object = None
    features: object = None
    observed_mask: object = None
    confidence: float = 1.0
    cost: float = 0.0
    provenance: object = None

    def __post_init__(self):
        object.__setattr__(self, "location", _freeze(self.location))
        object.__setattr__(self, "features", _freeze(self.features))
        object.__setattr__(self, "observed_mask", _freeze(self.observed_mask))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)


@dataclass(frozen=True)
class Action:
    """Action (v1.9.3 부록 1): modality, target, resolution/window, cost — 오직 센서 원본 데이터 조회 허용"""

    modality: str
    target: object = None
    resolution: object = None
    cost: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "target", _freeze(self.target))
        object.__setattr__(self, "resolution", _freeze(self.resolution))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)


@dataclass(frozen=True)
class Structure:
    """Structure (v1.9.3 부록 1): entity/attribute/relation, embedding, uncertainty, source_evidence"""

    entity: object = None
    attribute: object = None
    relation: object = None
    embedding: object = None
    uncertainty: float = 0.0
    source_evidence: object = None

    def __post_init__(self):
        object.__setattr__(self, "entity", _freeze(self.entity))
        object.__setattr__(self, "attribute", _freeze(self.attribute))
        object.__setattr__(self, "relation", _freeze(self.relation))
        object.__setattr__(self, "embedding", _freeze(self.embedding))
        object.__setattr__(self, "source_evidence", _freeze(self.source_evidence))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)


@dataclass(frozen=True)
class Hypothesis:
    """Hypothesis (v1.9.3 부록 1): concept_id, belief, supporting/contradicting_evidence"""

    concept_id: str = ""
    belief: float = 0.5
    supporting_evidence: object = None
    contradicting_evidence: object = None

    def __post_init__(self):
        object.__setattr__(self, "supporting_evidence", _freeze(self.supporting_evidence))
        object.__setattr__(self, "contradicting_evidence", _freeze(self.contradicting_evidence))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)


@dataclass(frozen=True)
class Prediction:
    """Prediction (v1.9.3 부록 1): modality_specific_distribution, candidate_observation, expected_cost"""

    modality_specific_distribution: object = None
    candidate_observation: object = None
    expected_cost: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "modality_specific_distribution", _freeze(self.modality_specific_distribution))
        object.__setattr__(self, "candidate_observation", _freeze(self.candidate_observation))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)


@dataclass(frozen=True)
class Memory:
    """Memory (v1.9.3 부록 1): concept_record, evidence_links, modality_views, versions, uncertainty"""

    concept_record: object = None
    evidence_links: object = None
    modality_views: object = None
    versions: object = None
    uncertainty: object = None

    def __post_init__(self):
        object.__setattr__(self, "concept_record", _freeze(self.concept_record))
        object.__setattr__(self, "evidence_links", _freeze(self.evidence_links))
        object.__setattr__(self, "modality_views", _freeze(self.modality_views))
        object.__setattr__(self, "versions", _freeze(self.versions))
        object.__setattr__(self, "uncertainty", _freeze(self.uncertainty))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema_version"] = SCHEMA_VERSION
        return _to_plain(d)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        sv = data.pop("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise ValueError(f"schema_version 불일치: {sv} != {SCHEMA_VERSION}")
        return cls(**data)
