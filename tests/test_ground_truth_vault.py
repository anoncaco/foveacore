"""GroundTruthVault 저장·검증·비공개 읽기 계약 테스트."""

import json
from pathlib import Path

import pytest

from foveacore.data import ground_truth_vault
from foveacore.data.ground_truth_vault import GroundTruthVault, GroundTruthVaultError


def test_vault_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "ground_truth_vault.json"
    vault = GroundTruthVault({"mn-000001": "5", "mn-000002": "3"})
    vault.save(path)

    loaded = GroundTruthVault.load(path)
    assert loaded._read_labels(frozenset({"mn-000001", "mn-000002"})) == {
        "mn-000001": "5",
        "mn-000002": "3",
    }
    assert loaded.contains("mn-000001")
    assert not loaded.contains("mn-999999")


def test_vault_has_no_direct_runtime_lookup_api():
    for name in ("get", "get_label", "_get", "_labels"):
        assert not hasattr(GroundTruthVault, name)
    assert not hasattr(ground_truth_vault, "AdaptAccessGuard")


@pytest.mark.parametrize("value", [None, "", "   ", 1, True])
def test_vault_rejects_invalid_labels(value):
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault({"sample-1": value})
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault().set("sample-1", value)


@pytest.mark.parametrize("value", [None, "", "   ", 1, True])
def test_vault_rejects_invalid_sample_ids(value):
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault({value: "5"})
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault().set(value, "5")
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault().contains(value)


def test_vault_rejects_non_dict_constructor_input():
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault([("sample-1", "5")])


def test_load_supports_wrapped_and_raw_mappings(tmp_path: Path):
    wrapped = tmp_path / "wrapped.json"
    raw = tmp_path / "raw.json"
    wrapped.write_text(json.dumps({"labels": {"sample-1": "5"}}), encoding="utf-8")
    raw.write_text(json.dumps({"sample-1": "5"}), encoding="utf-8")

    assert GroundTruthVault.load(wrapped)._read_labels(frozenset({"sample-1"})) == {
        "sample-1": "5"
    }
    assert GroundTruthVault.load(raw)._read_labels(frozenset({"sample-1"})) == {
        "sample-1": "5"
    }


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "[]",
        '{"labels": []}',
        '{"labels": {"sample-1": ""}}',
    ],
)
def test_load_rejects_malformed_vault(tmp_path: Path, content: str):
    path = tmp_path / "vault.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GroundTruthVaultError):
        GroundTruthVault.load(path)


def test_save_is_deterministic_for_different_insertion_order(tmp_path: Path):
    first = GroundTruthVault({"b": "2", "a": "1"})
    second = GroundTruthVault({"a": "1", "b": "2"})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first.save(first_path)
    second.save(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_read_labels_fails_closed_and_returns_a_copy():
    vault = GroundTruthVault({"b": "2", "a": "1"})
    labels = vault._read_labels(frozenset({"b", "a"}))
    assert list(labels) == ["a", "b"]
    labels["a"] = "changed"
    assert vault._read_labels(frozenset({"a"})) == {"a": "1"}

    with pytest.raises(GroundTruthVaultError, match="없는 sample_id"):
        vault._read_labels(frozenset({"missing"}))
    with pytest.raises(GroundTruthVaultError, match="frozenset"):
        vault._read_labels(["a"])
    assert vault._read_labels(frozenset()) == {}
