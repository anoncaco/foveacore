"""seal_guard.py freeze_manifest 해시 검증 테스트 (T3)."""

import pytest

from foveacore.data.seal_guard import (
    SealGuardError,
    canonical_json,
    compute_freeze_manifest_hash,
    verify_integrity,
)


def test_canonical_json_deterministic():
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b
    assert " " not in a


def test_compute_freeze_manifest_hash_excludes_own_field():
    manifest = {
        "schema_version": "1.0",
        "data_split_hash": "abc",
        "manifest_sha256": "placeholder",
    }
    h1 = compute_freeze_manifest_hash(manifest)
    manifest["manifest_sha256"] = "different-placeholder"
    h2 = compute_freeze_manifest_hash(manifest)
    assert h1 == h2  # own field excluded


def test_verify_integrity_pass():
    body = {
        "schema_version": "1.0",
        "data_split_hash": "abc",
        "config_hash": "cfg",
        "checkpoint_hash": "ckpt",
    }
    h = compute_freeze_manifest_hash(body)
    manifest = dict(body)
    manifest["manifest_sha256"] = h
    verify_integrity(manifest, sidecar_hash=h, artifact_hashes={
        "data_split_hash": "abc",
        "config_hash": "cfg",
        "checkpoint_hash": "ckpt",
    })


def test_verify_integrity_hash_mismatch():
    manifest = {
        "schema_version": "1.0",
        "data_split_hash": "abc",
        "manifest_sha256": "wrong-hash",
    }
    with pytest.raises(SealGuardError):
        verify_integrity(manifest)


def test_verify_integrity_sidecar_mismatch():
    manifest = {
        "schema_version": "1.0",
        "data_split_hash": "abc",
        "manifest_sha256": compute_freeze_manifest_hash(
            {"schema_version": "1.0", "data_split_hash": "abc"}
        ),
    }
    with pytest.raises(SealGuardError):
        verify_integrity(manifest, sidecar_hash="different-sidecar")


def test_verify_integrity_artifact_mismatch():
    manifest = {
        "schema_version": "1.0",
        "data_split_hash": "abc",
        "manifest_sha256": compute_freeze_manifest_hash(
            {"schema_version": "1.0", "data_split_hash": "abc"}
        ),
    }
    with pytest.raises(SealGuardError):
        verify_integrity(manifest, artifact_hashes={"data_split_hash": "actual-hash"})
