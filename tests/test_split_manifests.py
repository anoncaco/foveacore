"""T3 2단계 공개 manifest 빌더·로더·등록기 테스트."""

import json

import pytest

from foveacore.data.splits import (
    SplitManifestError,
    build_split_manifest,
    load_split_manifest,
    register_manifest,
    validate_split_manifest,
)


SEALED_BY_SPLIT = {
    "support_pool": "OPEN",
    "dev_stream": "OPEN",
    "main_stream": "OPEN",
    "official_test": "SEALED",
    "reserved_sealed": "SEALED",
    "p2_non_digit": "OPEN",
    "non_digit_dev": "OPEN",
    "dev_ood": "SEALED",
    "test_ood": "SEALED",
    "phase7_char_transfer": "SEALED",
}


def _sample(sample_id, **metadata):
    return {
        "sample_id": sample_id,
        "sha256": f"sha-{sample_id}",
        "phash": f"phash-{sample_id}",
        **metadata,
    }


def _all_slots(entries=None):
    entries = entries or {}
    return {
        split_name: build_split_manifest(
            entries.get(split_name, []), split_name, sealed_level
        )
        for split_name, sealed_level in SEALED_BY_SPLIT.items()
    }


def test_builder_emits_all_five_public_entry_fields():
    manifest = build_split_manifest([_sample("s1")], "support_pool")
    assert set(manifest["entries"][0]) == {
        "sample_id",
        "split_name",
        "sha256",
        "phash",
        "sealed_level",
    }


def test_builder_drops_label_and_validation_metadata():
    manifest = build_split_manifest(
        [
            _sample(
                "s1",
                label="0",
                class_id="0",
                writer_id="w1",
                alphabet_id="latin",
                source_id="mnist",
                structure_id="shape-1",
            )
        ],
        "support_pool",
    )
    assert manifest["entries"] == [
        {
            "sample_id": "s1",
            "split_name": "support_pool",
            "sha256": "sha-s1",
            "phash": "phash-s1",
            "sealed_level": "OPEN",
        }
    ]


@pytest.mark.parametrize("field", ["sample_id", "sha256", "phash"])
def test_builder_rejects_missing_required_values(field):
    sample = _sample("s1")
    del sample[field]
    with pytest.raises(SplitManifestError, match=field):
        build_split_manifest([sample], "support_pool")


def test_loader_validates_public_manifest(tmp_path):
    path = tmp_path / "split_support_pool.json"
    build_split_manifest([_sample("s1")], "support_pool", out_path=path)
    assert load_split_manifest(path) == build_split_manifest(
        [_sample("s1")], "support_pool"
    )


def test_loader_rejects_label_in_public_manifest(tmp_path):
    path = tmp_path / "split_support_pool.json"
    manifest = build_split_manifest([_sample("s1")], "support_pool")
    manifest["entries"][0]["label"] = "0"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SplitManifestError, match="금지 필드"):
        load_split_manifest(path)


def test_register_manifest_accepts_all_ten_slots_including_empty_ood():
    manifests = _all_slots(
        {
            "support_pool": [_sample("s1")],
            "dev_ood": [_sample("dev-1")],
            "test_ood": [_sample("test-1")],
        }
    )
    assert register_manifest(manifests) == []


def test_register_manifest_fails_when_a_required_slot_is_missing():
    manifests = _all_slots()
    del manifests["non_digit_dev"]
    errors = register_manifest(manifests)
    assert any("non_digit_dev" in error for error in errors)


def test_register_manifest_rejects_unknown_slot():
    manifests = _all_slots()
    manifests["unknown"] = build_split_manifest([], "support_pool")
    assert any("등록할 수 없는 split_name" in error for error in register_manifest(manifests))


def test_register_manifest_rejects_key_and_split_name_mismatch():
    manifests = _all_slots()
    manifests["support_pool"] = build_split_manifest([], "dev_stream")
    errors = register_manifest(manifests)
    assert any("manifest 키와 split_name 불일치" in error for error in errors)


def test_validate_manifest_rejects_entry_split_and_seal_mismatch():
    manifest = build_split_manifest([_sample("s1")], "official_test", "SEALED")
    manifest["entries"][0]["sealed_level"] = "OPEN"
    assert any("sealed_level 불일치" in error for error in validate_split_manifest(manifest))


def test_validate_manifest_returns_all_duplicate_errors():
    manifest = build_split_manifest([_sample("s1"), _sample("s1")], "main_stream")
    errors = validate_split_manifest(manifest)
    assert any("중복 sample_id" in error for error in errors)
