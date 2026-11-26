"""FoveaCore Phase 0 데이터 분할 매니페스트 (T3).

split_manifest_v1.json 스키마(라벨 미포함):
  {
    "schema_version": "1.0",
    "split_name": "support_pool | dev_stream | main_stream | official_test |
                   reserved_sealed | p2_non_digit | non_digit_dev |
                   dev_ood | test_ood | phase7_char_transfer",
    "sealed_level": "OPEN | SEALED",          # 봉인 등급
    "entries": [
      {"sample_id": "mn-000123", "split_name": "support_pool",
       "sha256": "...", "phash": "...", "sealed_level": "OPEN"},
      ...
    ]
  }
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from foveacore.data.schema import (
    SCHEMA_VERSION,
    SEALED_LEVELS,
    SPLIT_NAMES,
    SchemaError,
    SealLevel,
    SplitEntry,
    SplitManifest,
    SplitName,
    _enum_value,
    _required_string,
)

# 기존 5쌍 DISJOINT_RULES import 경로는 공식 8쌍으로 호환합니다.
from foveacore.data.schema import DISJOINT_GROUPS, DISJOINT_RULES


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
    """공개 SplitManifest를 구성합니다. 라벨·검증 메타데이터는 저장하지 않습니다."""
    try:
        normalized_split = _enum_value(SplitName, split_name, "split_name")
        normalized_seal = _enum_value(SealLevel, sealed_level, "sealed_level")
        entries = []
        for sample in samples:
            sid = _required_string(sample.get("sample_id"), "sample_id")
            entries.append(
                SplitEntry(
                    sample_id=sid,
                    split_name=normalized_split,
                    sha256=_required_string(sample.get("sha256"), f"{sid}.sha256"),
                    phash=_required_string(sample.get("phash"), f"{sid}.phash"),
                    sealed_level=normalized_seal,
                )
            )
        # 객체 스키마는 중복 ID를 거부하지만, 기존 builder 계약은 중복을
        # manifest에 보존해 validate_split_manifest()가 오류 목록을 반환하게 합니다.
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "split_name": normalized_split,
            "sealed_level": normalized_seal,
            "entries": [entry.to_dict() for entry in entries],
        }
    except (SchemaError, TypeError, AttributeError) as exc:
        raise SplitManifestError(str(exc)) from exc

    if out_path is not None:
        errors = validate_split_manifest(manifest)
        if errors:
            raise SplitManifestError("; ".join(errors))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


def load_split_manifest(path: Path | str) -> Dict[str, Any]:
    """공개 SplitManifest JSON을 읽고 스키마 검증 후 dict로 반환합니다."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return SplitManifest.from_dict(data).to_dict()
    except (SchemaError, json.JSONDecodeError) as exc:
        raise SplitManifestError(f"분할 매니페스트 로딩 실패({p}): {exc}") from exc


def validate_split_manifest(manifest: Dict[str, Any]) -> List[str]:
    """공개 manifest의 검증 오류 목록을 반환합니다(빈 목록 = PASS)."""
    try:
        SplitManifest.from_dict(manifest)
    except (SchemaError, TypeError, ValueError) as exc:
        message = str(exc)
        if "sha256" in message:
            return [f"sha256 누락 또는 잘못된 값: {message}"]
        return [message]
    return []


def register_manifest(
    manifests: Dict[str, Dict[str, Any]],
    *,
    require_all_splits: bool = True,
) -> List[str]:
    """10종 공개 manifest를 등록하고 키·스키마·필수 슬롯 누락을 검사합니다."""
    errors: List[str] = []
    if not isinstance(manifests, dict):
        return ["manifests는 split_name → manifest 매핑이어야 합니다."]

    for key, manifest in manifests.items():
        if not isinstance(key, str):
            errors.append(f"split_name 키는 문자열이어야 합니다: {key!r}")
            continue
        if key not in SPLIT_NAMES:
            errors.append(f"등록할 수 없는 split_name: {key}")
        errors.extend(validate_split_manifest(manifest))
        if isinstance(manifest, dict) and manifest.get("split_name") != key:
            errors.append(
                f"manifest 키와 split_name 불일치: key={key}, "
                f"split_name={manifest.get('split_name')}"
            )
    if require_all_splits:
        for split_name in sorted(SPLIT_NAMES.difference(manifests)):
            errors.append(f"필수 분할 매니페스트 누락: {split_name}")
    return errors


def validate_split_ids_disjoint(manifests: Dict[str, Dict[str, Any]]) -> List[str]:
    """모든 분할의 sample_id 및 이미지 SHA-256 교차 중복을 검사합니다."""
    errors: List[str] = []
    id_to_splits: Dict[str, List[str]] = {}
    hash_to_splits: Dict[str, List[str]] = {}
    for split_name, manifest in manifests.items():
        for entry in manifest.get("entries", []):
            sid = entry.get("sample_id")
            if sid:
                id_to_splits.setdefault(sid, []).append(split_name)
            digest = entry.get("sha256")
            if digest:
                hash_to_splits.setdefault(digest, []).append(split_name)

    for sid, splits in id_to_splits.items():
        if len(splits) > 1:
            errors.append(f"분할 간 ID 중복: {sid} -> {splits}")
    for digest, splits in hash_to_splits.items():
        if len(splits) > 1:
            errors.append(f"분할 간 이미지 SHA-256 중복: {digest} -> {splits}")
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
        verify_mnist_checksums,
    )

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if not mnist_available(raw_dir):
        raise FileNotFoundError(
            f"MNIST 원본이 없습니다: {raw_dir}. download_mnist()로 먼저 확보하세요."
        )

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
    validation_records = []

    def make_entries(split_name, indices, img_arr, label_arr, prefix):
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
            validation_records.append(
                {
                    "split_name": split_name,
                    "sample_id": sid,
                    "class_id": str(int(label_arr[i])),
                    "writer_id": f"{prefix}-writer",
                    "alphabet_id": "digits",
                    "source_id": "mnist",
                }
            )
        return entries

    manifests: Dict[str, Dict[str, Any]] = {}

    split_specs = [
        ("support_pool", "OPEN", support_idx, "mn-train"),
        ("dev_stream", "OPEN", dev_idx, "mn-train"),
        ("main_stream", "OPEN", main_idx, "mn-train"),
        ("official_test", "SEALED", official_idx, "mn-test"),
        ("reserved_sealed", "SEALED", reserved_idx, "mn-test"),
    ]

    for split_name, sealed_level, indices, prefix in split_specs:
        entries = make_entries(
            split_name,
            indices,
            train_images if prefix == "mn-train" else test_images,
            train_labels if prefix == "mn-train" else test_labels,
            prefix,
        )
        # set sealed_level per entry
        for e in entries:
            e["sealed_level"] = sealed_level
        manifests[split_name] = build_split_manifest(
            entries, split_name, sealed_level
        )

    for split_name in (
        "p2_non_digit",
        "non_digit_dev",
        "dev_ood",
        "test_ood",
        "phase7_char_transfer",
    ):
        if split_name in (
            "support_pool",
            "dev_stream",
            "main_stream",
            "p2_non_digit",
            "non_digit_dev",
        ):
            sealed = "OPEN"
        elif split_name in (
            "official_test",
            "reserved_sealed",
            "dev_ood",
            "test_ood",
            "phase7_char_transfer",
        ):
            sealed = "SEALED"
        else:
            raise ValueError(f"알 수 없는 분할 슬롯: {split_name}")
        manifests[split_name] = build_split_manifest([], split_name, sealed)

    registration_errors = register_manifest(manifests)
    if registration_errors:
        raise ValueError(f"분할 매니페스트 등록 검증 실패: {registration_errors}")

    from foveacore.data.seal_guard import validate_sealed_policy

    pre_save_errors = validate_split_ids_disjoint(manifests)
    pre_save_errors.extend(validate_sealed_policy(manifests))
    if pre_save_errors:
        raise ValueError(f"분할 사전 저장 검증 실패: {pre_save_errors}")

    from foveacore.data.ground_truth_vault import GroundTruthVault
    from foveacore.data.ood_validator import OODValidator
    from foveacore.data.validation_metadata import ValidationMetadataStore

    validator = OODValidator(
        manifests, ValidationMetadataStore.from_records(validation_records)
    )
    isolation_errors = validator.validate_disjoint()
    isolation_errors.extend(validator.validate_cross_split_image_duplicates())
    if isolation_errors:
        raise ValueError(f"데이터 분할 격리 검증 실패: {isolation_errors}")

    # 모든 검증이 끝난 뒤에만 공개 per-split manifest를 저장합니다.
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, manifest in manifests.items():
        (out_dir / f"split_{split_name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 검증 완료 후 공개 manifest와 분리된 산출물을 저장합니다.
    if vault_path is None:
        vault_path = out_dir / "ground_truth_vault.json"
    GroundTruthVault(vault_labels).save(vault_path)
    ValidationMetadataStore.from_records(validation_records).save(
        out_dir / "validation_metadata.json"
    )

    candidates = validator.find_phash_candidates(threshold=10)
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
