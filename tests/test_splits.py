"""splits.py 분할 매니페스트 테스트 (T3)."""

import json
from pathlib import Path

import pytest

from foveacore.data.splits import (
    DISJOINT_RULES,
    SPLIT_NAMES,
    SplitManifestError,
    build_all_mnist_splits,
    build_split_manifest,
    load_split_manifest,
    validate_split_ids_disjoint,
    validate_split_manifest,
)


def test_build_split_manifest_required_fields():
    m = build_split_manifest(
        [{"sample_id": "mn-000001", "sha256": "a", "phash": "b"}], "support_pool", "OPEN"
    )
    assert m["schema_version"] == "1.0"
    assert m["split_name"] == "support_pool"
    assert m["sealed_level"] == "OPEN"
    assert len(m["entries"]) == 1
    assert m["entries"][0]["sample_id"] == "mn-000001"
    assert m["entries"][0]["sha256"]
    assert m["entries"][0]["phash"]
    assert m["entries"][0]["sealed_level"] == "OPEN"


def test_validate_split_manifest_pass():
    m = build_split_manifest(
        [{"sample_id": f"mn-{i:06d}", "sha256": f"h{i}", "phash": f"p{i}"} for i in range(3)], "dev_stream", "OPEN"
    )
    assert validate_split_manifest(m) == []


def test_validate_split_manifest_duplicate_id():
    m = build_split_manifest(
        [{"sample_id": "mn-000001", "sha256": "a", "phash": "b"}, {"sample_id": "mn-000001", "sha256": "a", "phash": "b"}], "main_stream", "OPEN"
    )
    errors = validate_split_manifest(m)
    assert any("중복 sample_id" in e for e in errors)


def test_validate_split_manifest_missing_fields():
    m = build_split_manifest([{"sample_id": "mn-000001", "sha256": "a", "phash": "b"}], "main_stream", "OPEN")
    m["entries"][0].pop("sha256")
    errors = validate_split_manifest(m)
    assert any("sha256 누락" in e for e in errors)


def test_invalid_split_name_raises():
    with pytest.raises(SplitManifestError):
        build_split_manifest([{"sample_id": "x"}], "bad_split")


def test_save_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "split_support_pool.json"
    m = build_split_manifest(
        [{"sample_id": "mn-000001", "sha256": "a", "phash": "b"}], "official_test", "SEALED", out_path=out
    )
    loaded = load_split_manifest(out)
    assert loaded == m


def test_validate_split_ids_disjoint_pass():
    manifests = {
        "dev_ood": validate_split_manifest(
            build_split_manifest(
                [{"sample_id": f"ood-{i}", "sha256": f"o{i}", "phash": f"q{i}"} for i in range(3)], "dev_ood", "SEALED"
            )
        ) or build_split_manifest(
            [{"sample_id": f"ood-{i}", "sha256": f"o{i}", "phash": f"q{i}"} for i in range(3)], "dev_ood", "SEALED"
        ),
        "test_ood": build_split_manifest(
            [{"sample_id": f"test-{i}", "sha256": f"t{i}", "phash": f"r{i}"} for i in range(3)], "test_ood", "SEALED"
        ),
    }
    assert validate_split_ids_disjoint(manifests) == []


def test_validate_split_ids_disjoint_violation():
    shared = build_split_manifest(
        [{"sample_id": "shared-001", "sha256": "s", "phash": "p"}], "dev_ood", "SEALED"
    )
    manifests = {
        "dev_ood": shared,
        "test_ood": build_split_manifest(
            [{"sample_id": "shared-001", "sha256": "s", "phash": "p"}], "test_ood", "SEALED"
        ),
    }
    errors = validate_split_ids_disjoint(manifests)
    assert any("DISJOINT 위반" in e for e in errors)


def test_build_all_mnist_splits(tmp_path: Path):
    from foveacore.data.mnist import mnist_available

    raw = tmp_path / "raw"
    if not mnist_available(raw):
        pytest.skip("MNIST 원본 없음 (download_mnist 필요)")
    manifests = build_all_mnist_splits(raw, tmp_path / "out")
    assert set(manifests.keys()) == SPLIT_NAMES
    assert (tmp_path / "out" / "split_manifest_v1.json").exists()
    total = sum(len(m["entries"]) for m in manifests.values())
    assert total == 10000 + 15000 + 35000 + 7000 + 3000  # 70000
