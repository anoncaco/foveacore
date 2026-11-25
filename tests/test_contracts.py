"""core/contracts.py 공통 인터페이스 + 불변성 계약 테스트 (T2)."""

from dataclasses import FrozenInstanceError
import json
from types import MappingProxyType

import numpy as np
import pytest
import torch

from foveacore.core.contracts import (
    SCHEMA_VERSION,
    SENSOR_PROFILE_CONTRACTS,
    Action,
    Hypothesis,
    ImmutableTensor,
    Memory,
    Observation,
    Prediction,
    Structure,
    _freeze,
    _immutable_ndarray,
    make_image_provenance,
    validate_image_provenance,
    validate_unit_coordinates,
)
from foveacore.core.exceptions import (
    ImmutabilityViolationError,
    NumericalInstabilityError,
)


def _image_provenance(**overrides):
    values = {
        "original_hw": (28, 28),
        "processed_hw": (28, 28),
        "preprocess_profile_id": "mnist-f3-v1",
        "sensor_profile": "F3",
        "max_side": 1024,
        "resize_method": "adaptive_average",
    }
    values.update(overrides)
    return make_image_provenance(**values)


def test_schema_version():
    assert SCHEMA_VERSION == "1.0"


def test_sensor_profiles_are_incompatible_contracts():
    basic = SENSOR_PROFILE_CONTRACTS["F3"]
    hires = SENSOR_PROFILE_CONTRACTS["F3_HIRES"]
    assert basic.contract_id != hires.contract_id
    assert basic.feature_shape == (4, 5, 5)
    assert hires.feature_shape == (4, 97, 97)


def test_freeze_dict_and_list_copies_mapping_proxy_input():
    source = {"items": [1, {"x": 2}]}
    frozen = _freeze(MappingProxyType(source))
    source["items"][1]["x"] = 99
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["items"], tuple)
    assert frozen["items"][1]["x"] == 2


def test_freeze_ndarray_is_immutable_defensive_copy():
    source = np.array([1, 2, 3])
    frozen = _freeze(source)
    source[0] = 99
    assert frozen.tolist() == [1, 2, 3]

    base = frozen
    while isinstance(base, np.ndarray):
        assert not base.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            base.setflags(write=True)
        base = base.base
    assert isinstance(base, bytes)

    with pytest.raises(ValueError, match="read-only"):
        frozen[0] = 77
    assert frozen.tolist() == [1, 2, 3]


def test_immutable_ndarray_rejects_object_dtype():
    with pytest.raises(TypeError):
        _immutable_ndarray(np.array([1, "a"], dtype=object))


def test_freeze_rejects_object_ndarray_creation():
    with pytest.raises(TypeError):
        _freeze(np.array([1, "a"], dtype=object))


def test_structure_rejects_object_array_creation():
    with pytest.raises(TypeError):
        Structure(embedding=np.array([1, "a"], dtype=object))


def test_object_ndarray_deserialization_rejected():
    with pytest.raises(TypeError):
        _freeze({"__numpy__": [1, "a"], "dtype": "object"})


def test_observation_ndarray_cannot_be_reactivated_for_writing():
    obs = Observation(
        modality="visual",
        source_id="s1",
        features=np.arange(100, dtype=np.float32).reshape(1, 4, 5, 5),
        provenance=_image_provenance(),
    )
    before = obs.to_dict()
    with pytest.raises(ValueError, match="WRITEABLE"):
        obs.features.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        obs.features[0, 0, 0, 0] = -1
    assert obs.to_dict() == before


def test_image_observation_contract_and_strict_json_roundtrip():
    provenance = _image_provenance()
    features = np.zeros((1, 4, 5, 5), dtype=np.float32)
    obs = Observation(
        modality="visual",
        source_id="s1",
        location={"bbox": [[0.0, 0.0], [1.0, 1.0]]},
        features=features,
        observed_mask=np.array([True]),
        provenance=provenance,
    )
    assert isinstance(obs.features, np.ndarray)
    assert not obs.features.flags.writeable
    assert isinstance(obs.provenance, MappingProxyType)
    assert obs.provenance["original_hw"] == (28, 28)
    assert obs.provenance["sensor_profile"] == "F3"
    assert obs.provenance["grid_size"] == 5
    data = obs.to_dict()
    payload = json.dumps(data, allow_nan=False)
    assert Observation.from_dict(json.loads(payload)).to_dict() == data


def test_image_observation_accepts_chw_and_bchw_features():
    cases = [
        ("F3", 5, (4, 5, 5)),
        ("F3", 5, (1, 4, 5, 5)),
        ("F3_HIRES", 97, (4, 97, 97)),
        ("F3_HIRES", 97, (2, 4, 97, 97)),
    ]
    for sensor_profile, grid_size, shape in cases:
        obs = Observation(
            modality="visual",
            source_id="s1",
            features=np.zeros(shape, dtype=np.float32),
            provenance=_image_provenance(sensor_profile=sensor_profile, grid_size=grid_size),
        )
        assert obs.features.shape == shape


@pytest.mark.parametrize(
    ("sensor_profile", "grid_size", "shape"),
    [
        ("F3", 5, (1, 4, 97, 97)),
        ("F3_HIRES", 97, (1, 4, 5, 5)),
    ],
)
def test_image_observation_rejects_profile_feature_shape_mismatch(
    sensor_profile, grid_size, shape
):
    with pytest.raises(ValueError, match="features 형상"):
        Observation(
            modality="visual",
            source_id="s1",
            features=np.zeros(shape, dtype=np.float32),
            provenance=_image_provenance(sensor_profile=sensor_profile, grid_size=grid_size),
        )


def test_image_features_shape_is_validated_on_json_restore():
    obs = Observation(
        modality="visual",
        source_id="s1",
        features=np.zeros((1, 4, 5, 5), dtype=np.float32),
        provenance=_image_provenance(),
    )
    data = obs.to_dict()
    data["features"] = {
        "__numpy__": np.zeros((1, 4, 97, 97), dtype=np.float32).tolist(),
        "dtype": "float32",
    }
    with pytest.raises(ValueError, match="features 형상"):
        Observation.from_dict(data)


def test_image_observation_rejects_missing_provenance():
    with pytest.raises(ValueError, match="mapping"):
        Observation(modality="image", source_id="s1")


@pytest.mark.parametrize(
    "changes",
    [
        {"grid_size": 97},
        {"sensor_profile_version": "9.0"},
        {"profile_contract_id": "F3_HIRES@1.0"},
        {"coordinate_system": "pixels"},
        {"original_hw": (0, 28)},
        {"processed_hw": (28,)},
        {"preprocess_profile_id": ""},
        {"sensor_profile": "unknown"},
    ],
)
def test_image_provenance_rejects_invalid_contract(changes):
    provenance = _image_provenance()
    provenance.update(changes)
    with pytest.raises(ValueError):
        validate_image_provenance(provenance)


def test_high_resolution_image_provenance_is_separate():
    provenance = _image_provenance(sensor_profile="F3_HIRES", grid_size=97)
    assert provenance["profile_contract_id"] == "F3_HIRES@1.0"
    assert provenance["grid_size"] == 97


@pytest.mark.parametrize("location", [-0.1, 1.1, float("nan"), "center"])
def test_image_observation_rejects_non_unit_coordinates(location):
    with pytest.raises(ValueError, match="좌표"):
        Observation(
            modality="visual",
            source_id="s1",
            location=location,
            provenance=_image_provenance(),
        )


def test_unit_coordinate_validation_accepts_nested_unit_values():
    validate_unit_coordinates({"bbox": [[0, 0], [1, 1]], "gaze": [0.5, 0.25]})


def test_all_contracts_json_roundtrip():
    image_provenance = _image_provenance()
    cases = [
        Observation(
            modality="visual",
            source_id="s1",
            features=np.zeros((1, 4, 5, 5), dtype=np.float32),
            location=[[0, 0]],
            provenance=image_provenance,
        ),
        Action(
            modality="visual",
            target=[[0, 0]],
            resolution=[28, 28],
            window=[20, 20],
            cost=1.0,
        ),
        Structure(
            entity="E1",
            attribute="attr",
            relation="rel",
            embedding=[0.1, 0.2],
            uncertainty=0.05,
            source_evidence=["ev1"],
        ),
        Hypothesis(
            concept_id="c1",
            belief=0.7,
            supporting_evidence=["ev1"],
            contradicting_evidence=[],
        ),
        Prediction(
            modality_specific_distribution={"visual": [0.9, 0.1]},
            candidate_observation=Observation(
                modality="visual", source_id="s2", provenance=image_provenance
            ),
            expected_cost=0.5,
        ),
        Memory(
            concept_record={"id": "c1"},
            evidence_links=[],
            modality_views=[],
            versions=[],
            uncertainty=0.1,
        ),
    ]
    for obj in cases:
        data = obj.to_dict()
        payload = json.dumps(data, allow_nan=False)
        restored = type(obj).from_dict(json.loads(payload))
        assert restored.to_dict() == data


def test_prediction_restores_candidate_observation_type():
    obs = Observation(modality="visual", source_id="s1", provenance=_image_provenance())
    restored = Prediction.from_dict(Prediction(candidate_observation=obs).to_dict())
    assert isinstance(restored.candidate_observation, Observation)


def test_nested_contract_data_is_immutable():
    source = {"items": [1, 2]}
    obs = Observation(
        modality="visual",
        source_id="s1",
        provenance=_image_provenance(extra=source),
    )
    source["items"].append(3)
    assert obs.provenance["extra"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        obs.provenance["grid_size"] = 97
    with pytest.raises(FrozenInstanceError):
        obs.cost = 1.0


def test_immutable_tensor_defensively_copies_input_and_output():
    source = torch.tensor([1.0, 2.0, 3.0])
    wrapped = ImmutableTensor(source)
    source[0] = 99.0
    returned = wrapped.tensor()
    returned[0] = 77.0
    assert wrapped.tensor().tolist() == [1.0, 2.0, 3.0]


def test_immutable_tensor_detects_internal_change_on_access():
    wrapped = ImmutableTensor(torch.tensor([1.0, 2.0, 3.0]))
    wrapped.assert_unchanged()
    wrapped._tensor[0] = 99.0
    with pytest.raises(ImmutabilityViolationError):
        wrapped.tensor()
    with pytest.raises(ImmutabilityViolationError):
        wrapped.to_dict()


def test_immutable_tensor_rejects_tampered_snapshot():
    data = ImmutableTensor(torch.tensor([1.0, 2.0])).to_dict()
    data["data"][0] = 99.0
    with pytest.raises(ImmutabilityViolationError, match="snapshot"):
        ImmutableTensor.from_dict(data)


def test_immutable_tensor_bfloat16_json_roundtrip():
    wrapped = ImmutableTensor(torch.tensor([1.5, 2.5], dtype=torch.bfloat16))
    data = wrapped.to_dict()
    restored = ImmutableTensor.from_dict(json.loads(json.dumps(data, allow_nan=False)))
    assert torch.equal(restored.tensor(), wrapped.tensor())
    restored.assert_unchanged()


def test_immutable_tensor_rejects_schema_mismatch():
    data = ImmutableTensor(torch.tensor([1.0])).to_dict()
    data["schema_version"] = "9.0"
    with pytest.raises(ValueError, match="schema_version"):
        ImmutableTensor.from_dict(data)


def test_contract_numeric_fields_reject_instability():
    with pytest.raises(NumericalInstabilityError):
        Observation(
            modality="visual",
            source_id="s1",
            confidence=float("nan"),
            provenance=_image_provenance(),
        )
    with pytest.raises(ValueError, match="belief"):
        Hypothesis(belief=1.1)
    with pytest.raises(ValueError, match="cost"):
        Action(modality="visual", cost=-1.0)


@pytest.mark.parametrize(
    "obj",
    [
        Observation(modality="visual", source_id="s1", provenance=_image_provenance()),
        Action(modality="visual"),
        Structure(),
        Hypothesis(),
        Prediction(),
        Memory(),
    ],
)
def test_all_contracts_reject_schema_mismatch(obj):
    data = obj.to_dict()
    data["schema_version"] = "9.0"
    with pytest.raises(ValueError, match="schema_version"):
        type(obj).from_dict(data)
