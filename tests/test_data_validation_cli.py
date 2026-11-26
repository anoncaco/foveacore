"""T3 CLI 데이터 검증의 오류·경고 출력 계약."""

from pathlib import Path

from foveacore.__main__ import _validate_data_integrity, main
from foveacore.data.splits import build_split_manifest
from foveacore.data.validation_metadata import ValidationMetadataStore


SEALED = {
    "official_test": "SEALED",
    "reserved_sealed": "SEALED",
    "dev_ood": "SEALED",
    "test_ood": "SEALED",
    "phase7_char_transfer": "SEALED",
}


def _write_empty_slots(out_dir):
    for split_name in (
        "support_pool",
        "dev_stream",
        "main_stream",
        "official_test",
        "reserved_sealed",
        "p2_non_digit",
        "non_digit_dev",
        "dev_ood",
        "test_ood",
        "phase7_char_transfer",
    ):
        manifest = build_split_manifest(
            [], split_name, SEALED.get(split_name, "OPEN"),
            out_path=out_dir / f"split_{split_name}.json",
        )
    (out_dir / "ground_truth_vault.json").write_text('{"labels": {}}', encoding="utf-8")
    ValidationMetadataStore().save(out_dir / "validation_metadata.json")


def test_validate_data_integrity_returns_separate_errors_and_warnings(tmp_path):
    _write_empty_slots(tmp_path)
    errors, warnings = _validate_data_integrity(tmp_path)
    assert errors == []
    assert warnings == []


def test_validate_data_integrity_reports_alphabet_overlap_as_warning(tmp_path):
    _write_empty_slots(tmp_path)
    for split_name, sample_id in (("dev_ood", "a1"), ("test_ood", "b1")):
        build_split_manifest(
            [{"sample_id": sample_id, "sha256": f"sha-{sample_id}", "phash": f"p-{sample_id}"}],
            split_name,
            "SEALED",
            out_path=tmp_path / f"split_{split_name}.json",
        )
    ValidationMetadataStore.from_records(
        [
            {
                "split_name": "dev_ood", "sample_id": "a1", "class_id": "class-a",
                "writer_id": "writer-a", "alphabet_id": "latin", "source_id": "source-a",
            },
            {
                "split_name": "test_ood", "sample_id": "b1", "class_id": "class-b",
                "writer_id": "writer-b", "alphabet_id": "latin", "source_id": "source-b",
            },
        ]
    ).save(tmp_path / "validation_metadata.json")

    errors, warnings = _validate_data_integrity(tmp_path)
    assert errors == []
    assert any("alphabet_id 중복 경고" in warning for warning in warnings)


def _config_path():
    return Path(__file__).resolve().parents[1] / "configs" / "phase0.yaml"


def test_validate_cli_accepts_complete_empty_split_slots(tmp_path, capsys):
    _write_empty_slots(tmp_path)
    assert main(
        ["validate", "--config", str(_config_path()), "--data-dir", str(tmp_path)]
    ) == 0
    assert "validate PASS" in capsys.readouterr().out


def test_validate_cli_reports_missing_manifest_without_traceback(tmp_path, capsys):
    assert main(
        ["validate", "--config", str(_config_path()), "--data-dir", str(tmp_path)]
    ) == 1
    output = capsys.readouterr().out
    assert "분할 매니페스트 누락" in output
    assert "Traceback" not in output


def test_validate_cli_fails_closed_on_missing_metadata(tmp_path, capsys):
    _write_empty_slots(tmp_path)
    build_split_manifest(
        [{"sample_id": "ood-1", "sha256": "sha-ood-1", "phash": "phash-ood-1"}],
        "dev_ood",
        "SEALED",
        out_path=tmp_path / "split_dev_ood.json",
    )
    (tmp_path / "validation_metadata.json").unlink()

    assert main(
        ["validate", "--config", str(_config_path()), "--data-dir", str(tmp_path)]
    ) == 1
    output = capsys.readouterr().out
    assert "검증 메타데이터 누락" in output
    assert "Traceback" not in output
