"""FoveaCore Phase 0 데이터 분할 매니페스트 (T3).

split_manifest_v1.json 스키마(라벨 미포함):
  {
    "schema_version": "1.0",
    "split_name": "support_pool | dev_stream | main_stream | official_test |
                   reserved_sealed | p2_non_digit | non_digit_dev |
                   dev_ood | test_ood | phase7_char_transfer",
    "sealed_level": "OPEN | SEALED",          # 봉인 등급
    "entries": [
      {"sample_id": "mn-000123", "sha256": "...", "phash": "...", "sealed_level": "OPEN"},
      ...
    ]
  }
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

SPLIT_NAMES = {
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
}

SEALED_LEVELS = {"OPEN", "SEALED"}

# 분할 간 ID 분리가 엄격히 요구되는 디조인트 그룹(클래스 및 writer/source 모두)
DISJOINT_GROUPS = [
    ("P2", ("p2_non_digit",)),
    ("dev_ood", ("dev_ood",)),
    ("test_ood", ("test_ood",)),
    ("phase7_char_transfer", ("phase7_char_transfer",)),
]
DISJOINT_RULES = [
    ("p2_non_digit", "dev_ood"),
    ("p2_non_digit", "test_ood"),
    ("p2_non_digit", "phase7_char_transfer"),
    ("dev_ood", "test_ood"),
    ("test_ood", "phase7_char_transfer"),
]


class SplitManifestError(ValueError):
    """split_manifest_v1.json 검증 실패."""


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split_manifest(
    samples: List[Dict[str, Any]],
    split_name: str,
    sealed_level: str = "OPEN",
    out_path: Path | None = None,
) -> Dict[str, Any]:
    """분할 매니페스트(dict)를 구성하고, out_path가 주어지면 JSON 파일로 저장."""
    if split_name not in SPLIT_NAMES:
        raise SplitManifestError(f"잘못된 split_name: {split_name}")
    if sealed_level not in SEALED_LEVELS:
        raise SplitManifestError(f"잘못된 sealed_level: {sealed_level}")

    entries = []
    for s in samples:
        sid = s["sample_id"]
        sha = s.get("sha256")
        ph = s.get("phash")
        if not sha or not ph:
            raise SplitManifestError(f"sha256/phash 누락: {sid}")
        entry = {
            "sample_id": sid,
            "sha256": sha,
            "phash": ph,
            "sealed_level": sealed_level,
        }
        for meta_key in ("writer_id", "alphabet_id", "source_id", "class_id", "structure_id"):
            if meta_key in s:
                entry[meta_key] = s.get(meta_key)
        entries.append(entry)

    manifest = {
        "schema_version": "1.0",
        "split_name": split_name,
        "sealed_level": sealed_level,
        "entries": entries,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def load_split_manifest(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_split_manifest(manifest: Dict[str, Any]) -> List[str]:
    """검증 오류 목록(빈 리스트 = PASS)."""
    errors: List[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("schema_version이 1.0이 아닙니다.")
    if manifest.get("split_name") not in SPLIT_NAMES:
        errors.append(f"잘못된 split_name: {manifest.get('split_name')}")
    if manifest.get("sealed_level") not in SEALED_LEVELS:
        errors.append(f"잘못된 sealed_level: {manifest.get('sealed_level')}")

    ids: Set[str] = set()
    for entry in manifest.get("entries", []):
        sid = entry.get("sample_id")
        if sid is None:
            errors.append("sample_id 누락")
            continue
        if sid in ids:
            errors.append(f"중복 sample_id: {sid}")
        ids.add(sid)
        if entry.get("sha256") is None:
            errors.append(f"sha256 누락: {sid}")
        if entry.get("phash") is None:
            errors.append(f"phash 누락: {sid}")
        if entry.get("sealed_level") not in SEALED_LEVELS:
            errors.append(f"잘못된 sealed_level: {sid}")

    return errors


def validate_split_ids_disjoint(manifests: Dict[str, Dict[str, Any]]) -> List[str]:
    """분할 간 ID 중복 검사 + DISJOINT_RULES 적용."""
    errors: List[str] = []
    id_to_splits: Dict[str, List[str]] = {}
    for split_name, manifest in manifests.items():
        for entry in manifest.get("entries", []):
            sid = entry.get("sample_id")
            if sid:
                id_to_splits.setdefault(sid, []).append(split_name)

    for sid, splits in id_to_splits.items():
        if len(splits) > 1:
            errors.append(f"분할 간 ID 중복: {sid} -> {splits}")

    for a, b in DISJOINT_RULES:
        ids_a = {
            e.get("sample_id")
            for e in manifests.get(a, {}).get("entries", [])
            if e.get("sample_id")
        }
        ids_b = {
            e.get("sample_id")
            for e in manifests.get(b, {}).get("entries", [])
            if e.get("sample_id")
        }
        overlap = ids_a & ids_b
        if overlap:
            errors.append(f"DISJOINT 위반: {a} ∩ {b} = {sorted(overlap)}")

    return errors


def build_all_mnist_splits(
    raw_dir: Path | str,
    out_dir: Path | str,
    split_seed: int = 1001,
    vault_path: Path | str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """MNIST raw 디렉터리 기반 전체 분할 매니페스트를 구성(파일 저장).

    실제 IDX 파일을 파싱해 이미지 픽셀 기반 SHA-256·pHash를 계산하고,
    split_seed로 결정론적 분할을 수행한다. GroundTruthVault도 함께 생성.
    MNIST 원본이 없으면 FileNotFoundError.
    """
    from foveacore.data.mnist import (
        load_mnist,
        mnist_available,
        phash_image,
        sha256_image,
    )

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mnist_available(raw_dir):
        raise FileNotFoundError(
            f"MNIST 원본이 없습니다: {raw_dir}. download_mnist()로 먼저 확보하세요."
        )

    from foveacore.data.mnist import verify_mnist_checksums

    checksum_errors = verify_mnist_checksums(raw_dir)
    if checksum_errors:
        raise ValueError(f"MNIST 체크섬 검증 실패: {checksum_errors}")

    train_images, train_labels, test_images, test_labels = load_mnist(raw_dir)

    rng = random.Random(split_seed)
    train_indices = list(range(len(train_images)))
    rng.shuffle(train_indices)
    # 10000 / 15000 / 35000
    support_idx = train_indices[:10000]
    dev_idx = train_indices[10000:25000]
    main_idx = train_indices[25000:60000]

    test_indices = list(range(len(test_images)))
    rng2 = random.Random(split_seed + 1)
    rng2.shuffle(test_indices)
    official_idx = test_indices[:7000]
    reserved_idx = test_indices[7000:10000]

    vault_labels: Dict[str, str] = {}

    def make_entries(indices, img_arr, label_arr, prefix):
        entries = []
        for i in indices:
            sid = f"{prefix}-{i:05d}"
            entries.append(
                {
                    "sample_id": sid,
                    "sha256": sha256_image(img_arr[i]),
                    "phash": phash_image(img_arr[i]),
                    "sealed_level": None,  # filled by build_split_manifest default
                }
            )
            vault_labels[sid] = str(int(label_arr[i]))
        return entries

    manifests: Dict[str, Dict[str, Any]] = {}

    split_specs = [
        ("support_pool", "OPEN", make_entries(support_idx, train_images, train_labels, "mn-train")),
        ("dev_stream", "OPEN", make_entries(dev_idx, train_images, train_labels, "mn-train")),
        ("main_stream", "OPEN", make_entries(main_idx, train_images, train_labels, "mn-train")),
        ("official_test", "SEALED", make_entries(official_idx, test_images, test_labels, "mn-test")),
        ("reserved_sealed", "SEALED", make_entries(reserved_idx, test_images, test_labels, "mn-test")),
    ]

    for split_name, sealed_level, entries in split_specs:
        # set sealed_level per entry
        for e in entries:
            e["sealed_level"] = sealed_level
        manifests[split_name] = build_split_manifest(
            entries, split_name, sealed_level,
            out_path=out_dir / f"split_{split_name}.json",
        )

    for split_name in ("p2_non_digit", "non_digit_dev", "dev_ood", "test_ood", "phase7_char_transfer"):
        sealed = "SEALED" if split_name in ("dev_ood", "test_ood", "phase7_char_transfer") else "OPEN"
        manifests[split_name] = build_split_manifest(
            [], split_name, sealed, out_path=out_dir / f"split_{split_name}.json"
        )

    # vault 저장
    if vault_path is None:
        vault_path = out_dir / "ground_truth_vault.json"
    from foveacore.data.ground_truth_vault import GroundTruthVault
    GroundTruthVault(vault_labels).save(vault_path)

    # OOD 교차 SHA-256 중복 검사 (T3-05)
    from foveacore.data.ood_validator import OODValidator

    dup_errors = OODValidator(manifests).validate_cross_split_image_duplicates()
    if dup_errors:
        raise ValueError(f"이미지 SHA-256 교차 중복: {dup_errors}")

    # pHash 근접 중복 보고서 (T3-16)
    candidates = OODValidator(manifests).find_phash_candidates(threshold=10)
    (out_dir / "phash_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_manifest = {
        "schema_version": "1.0",
        "splits": {k: v for k, v in manifests.items()},
    }
    (out_dir / "split_manifest_v1.json").write_text(
        json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifests
