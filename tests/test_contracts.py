"""core/contracts.py 공통 인터페이스 + 불변성 계약 테스트 (T2)."""

import json
from types import MappingProxyType

import numpy as np
import pytest
import torch

from foveacore.core.contracts import (
    SCHEMA_VERSION,
    Action,
    Hypothesis,
    ImmutableTensor,
    Memory,
    Observation,
    Prediction,
    Structure,
    _freeze,
)
from foveacore.core.exceptions import ImmutabilityViolationError


def test_schema_version():
    assert SCHEMA_VERSION == "1.0"


def test_freeze_dict_and_list():
    frozen = _freeze({"a": [1, 2], "b": "x"})
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["a"], tuple)
    assert frozen["a"] == (1, 2)


def test_freeze_ndarray_readonly():
    arr = _freeze(np.array([1, 2, 3]))
    assert isinstance(arr, np.ndarray)
    assert not arr.flags.writeable


def test_observation_contract_and_json_roundtrip():
    obs = Observation(modality="visual", source_id="s1", features=[1.0, 2.0],
                      observed_mask=np.array([True]))
    assert isinstance(obs.features, tuple)
    assert isinstance(obs.observed_mask, np.ndarray)
    assert not obs.observed_mask.flags.writeable
    d = obs.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert Observation.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_all_contracts_json_roundtrip():
    cases = [
        Observation(modality="visual", source_id="s1", features=[1.0], location=[[0, 0]]),
        Action(modality="visual", target=[[0, 0]], resolution=[28, 28], cost=1.0),
        Structure(entity="E1", attribute="attr", relation="rel", embedding=[0.1, 0.2],
                  uncertainty=0.05, source_evidence=["ev1"]),
        Hypothesis(concept_id="c1", belief=0.7, supporting_evidence=["ev1"],
                   contradicting_evidence=[]),
        Prediction(modality_specific_distribution={"visual": [0.9, 0.1]},
                   expected_cost=0.5),
        Memory(concept_record={"id": "c1"}, evidence_links=[], modality_views=[],
               versions=[], uncertainty=0.1),
    ]
    for obj in cases:
        d = obj.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert type(obj).from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_immutable_tensor_detects_internal_change():
    it = ImmutableTensor(torch.tensor([1.0, 2.0, 3.0]))
    it.assert_unchanged()
    it._tensor[0] = 99.0
    with pytest.raises(ImmutabilityViolationError):
        it.assert_unchanged()
