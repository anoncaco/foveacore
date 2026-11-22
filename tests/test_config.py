"""config.py 스키마 검증 테스트 (T1)."""

from foveacore.config import load_phase0_config, validate_phase0_config


def test_valid_config():
    cfg = load_phase0_config("configs/phase0.yaml")
    assert validate_phase0_config(cfg) == []


def test_missing_key_is_error():
    assert validate_phase0_config({"version": "1.9.3"})


def test_bad_protocol_is_error():
    cfg = load_phase0_config("configs/phase0.yaml")
    cfg["experiment_protocol"] = "BAD"
    errors = validate_phase0_config(cfg)
    assert any("experiment_protocol" in e for e in errors)


def test_smoke_way_must_be_10():
    cfg = load_phase0_config("configs/phase0.yaml")
    cfg["smoke"]["way"] = 5
    errors = validate_phase0_config(cfg)
    assert any("smoke.way" in e for e in errors)
