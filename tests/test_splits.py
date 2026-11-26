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
    assert any("분할 간 ID 중복" in e for e in errors)


def test_build_all_mnist_splits(tmp_path: Path):
    from foveacore.data.ground_truth_vault import GroundTruthVault
    from foveacore.data.mnist import (
        MNIST_FILES,
        mnist_available,
        sha256_file,
        verify_mnist_checksums,
    )

    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not mnist_available(raw):
        pytest.fail(
            f"MNIST 원본이 없습니다: {raw}. "
            "download_mnist()로 로컬 원본을 확보한 뒤 다시 실행하세요."
        )

    assert verify_mnist_checksums(raw) == []
    for filename, info in MNIST_FILES.items():
        assert sha256_file(raw / filename) == info["sha256"]

    split_seed = 1001
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    first = build_all_mnist_splits(raw, first_out, split_seed=split_seed)
    second = build_all_mnist_splits(raw, second_out, split_seed=split_seed)

    assert set(first) == set(second) == SPLIT_NAMES
    expected_counts = {
        "support_pool": 10_000,
        "dev_stream": 15_000,
        "main_stream": 35_000,
        "official_test": 7_000,
        "reserved_sealed": 3_000,
    }
    for split_name, expected_count in expected_counts.items():
        assert len(first[split_name]["entries"]) == expected_count
    for split_name in SPLIT_NAMES.difference(expected_counts):
        assert first[split_name]["entries"] == []

    sample_ids = [
        entry["sample_id"]
        for manifest in first.values()
        for entry in manifest["entries"]
    ]
    assert len(sample_ids) == len(set(sample_ids)) == 70_000

    first_assignment = {
        entry["sample_id"]: split_name
        for split_name, manifest in first.items()
        for entry in manifest["entries"]
    }
    second_assignment = {
        entry["sample_id"]: split_name
        for split_name, manifest in second.items()
        for entry in manifest["entries"]
    }
    assert first_assignment == second_assignment
    assert first == second

    first_vault = GroundTruthVault.load(first_out / "ground_truth_vault.json")
    second_vault = GroundTruthVault.load(second_out / "ground_truth_vault.json")
    all_ids = frozenset(sample_ids)
    assert first_vault._read_labels(all_ids) == second_vault._read_labels(all_ids)


def _mock_small_mnist(monkeypatch):
    import numpy as np

    import foveacore.data.mnist as mnist_module

    train_images = np.zeros((60, 28, 28), dtype=np.uint8)
    test_images = np.zeros((10, 28, 28), dtype=np.uint8)
    for index in range(60):
        train_images[index, 0, 0] = index
    for index in range(10):
        test_images[index, 0, 0] = index + 100
    train_labels = np.arange(60, dtype=np.uint8) % 10
    test_labels = np.arange(10, dtype=np.uint8) % 10

    monkeypatch.setattr(mnist_module, "mnist_available", lambda _raw: True)
    monkeypatch.setattr(mnist_module, "verify_mnist_checksums", lambda _raw: [])
    monkeypatch.setattr(
        mnist_module,
        "load_mnist",
        lambda _raw: (train_images, train_labels, test_images, test_labels),
    )


def test_build_all_mnist_splits_with_synthetic_raw_writes_all_outputs(tmp_path, monkeypatch):
    _mock_small_mnist(monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    manifests = build_all_mnist_splits(raw, out)

    assert set(manifests) == SPLIT_NAMES
    assert all((out / f"split_{split_name}.json").exists() for split_name in SPLIT_NAMES)
    assert (out / "split_manifest_v1.json").exists()
    assert (out / "ground_truth_vault.json").exists()
    assert (out / "validation_metadata.json").exists()
    assert (out / "phash_candidates.json").exists()


def test_build_all_mnist_splits_validates_global_ids_before_saving(tmp_path, monkeypatch):
    _mock_small_mnist(monkeypatch)
    import foveacore.data.splits as splits_module

    monkeypatch.setattr(
        splits_module,
        "validate_split_ids_disjoint",
        lambda _manifests: ["forced global isolation failure"],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="사전 저장 검증"):
        build_all_mnist_splits(raw, out)
    assert not out.exists()


def test_build_all_mnist_splits_validates_sealed_policy_before_saving(tmp_path, monkeypatch):
    _mock_small_mnist(monkeypatch)
    import foveacore.data.seal_guard as seal_guard_module

    monkeypatch.setattr(
        seal_guard_module,
        "validate_sealed_policy",
        lambda _manifests: ["forced sealed policy failure"],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="사전 저장 검증"):
        build_all_mnist_splits(raw, out)
    assert not out.exists()


def test_build_all_mnist_splits_validates_metadata_before_saving(tmp_path, monkeypatch):
    _mock_small_mnist(monkeypatch)
    from foveacore.data.ood_validator import OODValidator

    monkeypatch.setattr(
        OODValidator,
        "validate_disjoint",
        lambda _self: ["forced metadata isolation failure"],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="데이터 분할 격리 검증 실패"):
        build_all_mnist_splits(raw, out)
    assert not out.exists()
