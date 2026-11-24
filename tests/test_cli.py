"""CLI 진입점 및 오류 처리 테스트 (T1)."""

from pathlib import Path

import pytest
import yaml

import foveacore.__main__ as cli
from foveacore.forbidden_packages import assert_no_forbidden_packages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase0.yaml"


def _write_config(tmp_path, **changes):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for dotted_key, value in changes.items():
        if "." in dotted_key:
            section, key = dotted_key.split(".", 1)
            cfg[section][key] = value
        else:
            cfg[dotted_key] = value
    path = tmp_path / "phase0.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def test_validate_command_passes(capsys):
    assert cli.main(["validate", "--config", str(CONFIG_PATH)]) == 0
    assert "validate PASS" in capsys.readouterr().out


def test_validate_rejects_bad_smoke_way(tmp_path, capsys):
    path = _write_config(tmp_path, **{"smoke.way": 9})
    assert cli.main(["validate", "--config", str(path)]) == 1
    assert "smoke.way" in capsys.readouterr().out


def test_build_splits_rejects_bad_config_before_io(tmp_path, capsys):
    path = _write_config(tmp_path, **{"smoke.way": 9})
    out_dir = tmp_path / "must-not-be-created"
    assert cli.main(["build-splits", "--config", str(path), "--out", str(out_dir)]) == 1
    assert "smoke.way" in capsys.readouterr().out
    assert not out_dir.exists()


def test_unhashable_protocol_reports_config_error(tmp_path, capsys):
    path = _write_config(tmp_path, **{"experiment_protocol": []})
    assert cli.main(["validate", "--config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "experiment_protocol" in output
    assert "TypeError" not in output


def test_smoke_baseline_entrypoint_registered(capsys):
    assert cli.main(["smoke-baseline"]) == 2
    assert "미구현" in capsys.readouterr().out


def test_forbidden_package_cli_error_is_clean(monkeypatch, capsys):
    def fail():
        raise RuntimeError("금지 패키지 발견: ['openai']")

    monkeypatch.setattr(cli, "assert_no_forbidden_packages", fail)
    assert cli.main(["validate", "--config", str(CONFIG_PATH)]) == 1
    output = capsys.readouterr().out
    assert "ERROR:" in output
    assert "openai" in output


def test_forbidden_package_guard_available():
    assert callable(assert_no_forbidden_packages)


BROKEN_YAML = "sensor: [\n"


def test_validate_reports_yaml_error_without_traceback(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text(BROKEN_YAML, encoding="utf-8")
    assert cli.main(["validate", "--config", str(path)]) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ERROR:" in captured.out
    assert str(path) in captured.out
    assert "Traceback" not in output


def test_build_splits_reports_yaml_error_before_creating_out(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text(BROKEN_YAML, encoding="utf-8")
    out_dir = tmp_path / "must-not-be-created"
    assert cli.main(["build-splits", "--config", str(path), "--out", str(out_dir)]) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ERROR:" in captured.out
    assert str(path) in captured.out
    assert "Traceback" not in output
    assert not out_dir.exists()
