"""config.py 스키마 검증 테스트 (T1)."""

from pathlib import Path

import pytest

from foveacore.config import load_phase0_config, validate_phase0_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase0.yaml"


def _load_valid_config():
    return load_phase0_config(CONFIG_PATH)


def test_valid_config():
    cfg = _load_valid_config()
    assert validate_phase0_config(cfg) == []


def test_input_sensor_contract_registered():
    """T1 요구사항: 입력·센서 계약값이 phase0.yaml에 선등록되어야 한다."""
    cfg = _load_valid_config()
    assert cfg["input"] == {
        "max_side": 1024,
        "resize_method": "adaptive_average",
    }
    assert cfg["sensor"] == {
        "profile": "F3",
        "grid_size": 5,
        "channels": 4,
        "stroke_clue_available": True,
    }


def test_missing_key_is_error():
    errors = validate_phase0_config({"version": "1.9.3"})
    assert errors
    assert any("schema_version" in error for error in errors)


@pytest.mark.parametrize("section", ["input", "sensor"])
def test_missing_contract_section_is_error(section):
    cfg = _load_valid_config()
    del cfg[section]
    errors = validate_phase0_config(cfg)
    assert any(f"필수 키 누락: {section}" in error for error in errors)


@pytest.mark.parametrize("value", [0, -1, True, 1025])
def test_input_max_side_must_be_1024(value):
    cfg = _load_valid_config()
    cfg["input"]["max_side"] = value
    errors = validate_phase0_config(cfg)
    assert any("input.max_side" in error for error in errors)


@pytest.mark.parametrize("value", ["nearest", "", None, []])
def test_input_resize_method_is_validated(value):
    cfg = _load_valid_config()
    cfg["input"]["resize_method"] = value
    errors = validate_phase0_config(cfg)
    assert any("input.resize_method" in error for error in errors)


@pytest.mark.parametrize("value", ["BAD", [], {}])
def test_bad_protocol_is_error(value):
    cfg = _load_valid_config()
    cfg["experiment_protocol"] = value
    errors = validate_phase0_config(cfg)
    assert any("experiment_protocol" in error for error in errors)


@pytest.mark.parametrize("value", ["BAD", [], {}])
def test_bad_pretraining_policy_is_error(value):
    cfg = _load_valid_config()
    cfg["pretraining_policy"] = value
    errors = validate_phase0_config(cfg)
    assert any("pretraining_policy" in error for error in errors)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"grid_size": 97}, "sensor.grid_size"),
        ({"channels": 3}, "sensor.channels"),
    ],
)
def test_f3_profile_rejects_mismatched_contract(changes, field):
    cfg = _load_valid_config()
    cfg["sensor"].update(changes)
    errors = validate_phase0_config(cfg)
    assert any(field in error for error in errors)


def test_high_resolution_profile_is_separate():
    """고해상도 97x97은 F3가 아닌 별도 프로파일로만 허용한다."""
    cfg = _load_valid_config()
    cfg["sensor"] = {
        "profile": "F3_HIRES",
        "grid_size": 97,
        "channels": 4,
        "stroke_clue_available": True,
    }
    assert validate_phase0_config(cfg) == []


def test_unknown_sensor_profile_is_error():
    cfg = _load_valid_config()
    cfg["sensor"]["profile"] = "UNKNOWN"
    errors = validate_phase0_config(cfg)
    assert any("sensor.profile" in error for error in errors)


def test_smoke_way_must_be_10():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "phase0_full_smoke"
    cfg["smoke"]["way"] = 5
    errors = validate_phase0_config(cfg)
    assert any("smoke.way" in error for error in errors)


def test_stroke_clue_available_bool_accepted():
    cfg = _load_valid_config()
    assert validate_phase0_config(cfg) == []
    assert cfg["sensor"]["stroke_clue_available"] is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_stroke_clue_available_invalid_type(value):
    cfg = _load_valid_config()
    cfg["sensor"]["stroke_clue_available"] = value
    errors = validate_phase0_config(cfg)
    assert any("stroke_clue_available" in error for error in errors)


def test_stroke_clue_available_profile_mismatch():
    cfg = _load_valid_config()
    cfg["sensor"]["stroke_clue_available"] = False
    errors = validate_phase0_config(cfg)
    assert any("stroke_clue_available" in error for error in errors)


def test_stroke_clue_available_missing():
    cfg = _load_valid_config()
    del cfg["sensor"]["stroke_clue_available"]
    errors = validate_phase0_config(cfg)
    assert any("sensor.stroke_clue_available" in error or "stroke_clue_available" in error for error in errors)


def test_strict_1shot_3way_passes():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "general"
    cfg["datasets"]["support_per_episode"] = 3
    cfg["smoke"]["way"] = 3
    assert validate_phase0_config(cfg) == []


def test_official_scope_way_and_support_both_5_rejected():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "phase0_full_smoke"
    cfg["smoke"]["way"] = 5
    cfg["datasets"]["support_per_episode"] = 5
    errors = validate_phase0_config(cfg)
    assert any("smoke.way" in error for error in errors)
    assert any("support_per_episode" in error for error in errors)


def test_official_scope_support_only_5_rejected():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "phase0_full_smoke"
    cfg["datasets"]["support_per_episode"] = 5
    errors = validate_phase0_config(cfg)
    assert any("support_per_episode" in error for error in errors)


def test_general_scope_3way_passes():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "general"
    cfg["datasets"]["support_per_episode"] = 3
    cfg["smoke"]["way"] = 3
    assert validate_phase0_config(cfg) == []


def test_general_scope_5way_passes():
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = "general"
    cfg["datasets"]["support_per_episode"] = 5
    cfg["smoke"]["way"] = 5
    assert validate_phase0_config(cfg) == []


def test_smoke_scope_missing_is_error():
    cfg = _load_valid_config()
    del cfg["smoke"]["scope"]
    errors = validate_phase0_config(cfg)
    assert any("smoke.scope" in error for error in errors)


@pytest.mark.parametrize("value", [[], 1, "unknown", None])
def test_smoke_scope_invalid_value_is_error(value):
    cfg = _load_valid_config()
    cfg["smoke"]["scope"] = value
    errors = validate_phase0_config(cfg)
    assert any("smoke.scope" in error for error in errors)
