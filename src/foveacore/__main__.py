"""FoveaCore CLI 진입점 (T1/T3): validate, build-splits, smoke-baseline."""

import argparse
import json
from pathlib import Path

import yaml

from foveacore.config import load_phase0_config, validate_phase0_config
from foveacore.data.mnist import (
    MNIST_FILES,
    download_mnist,
    mnist_available,
    sha256_file,
    verify_mnist_checksums,
)
from foveacore.data.ood_validator import OODValidator
from foveacore.data.seal_guard import SEALED_SPLITS, validate_sealed_policy
from foveacore.data.splits import (
    SPLIT_NAMES,
    build_all_mnist_splits,
    load_split_manifest,
    register_manifest,
    validate_split_ids_disjoint,
    validate_split_manifest,
)
from foveacore.data.validation_metadata import (
    ValidationMetadataError,
    ValidationMetadataStore,
)
from foveacore.forbidden_packages import assert_no_forbidden_packages
from foveacore.utils.seed import DEFAULT_SEEDS, initialize_gpu_runtime


def _validate_data_integrity(out_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_dir = out_dir / "raw"
    # MNIST 체크섬 (T3-01/02)
    if raw_dir.exists():
        errors.extend(verify_mnist_checksums(raw_dir))

    # 필수 산출물 존재
    required_splits = SPLIT_NAMES
    manifests: dict[str, dict] = {}
    for split_name in required_splits:
        p = out_dir / f"split_{split_name}.json"
        if not p.exists():
            errors.append(f"분할 매니페스트 누락: {p}")
            continue
        try:
            manifests[split_name] = load_split_manifest(p)
            errors.extend(validate_split_manifest(manifests[split_name]))
        except Exception as exc:
            errors.append(f"분할 매니페스트 파싱 실패({split_name}): {exc}")

    vault_path = out_dir / "ground_truth_vault.json"
    if not vault_path.exists():
        errors.append(f"정답 저장소 누락: {vault_path}")

    errors.extend(register_manifest(manifests))
    errors.extend(validate_sealed_policy(manifests))
    errors.extend(validate_split_ids_disjoint(manifests))
    metadata_path = out_dir / "validation_metadata.json"
    metadata_store = None
    if metadata_path.exists():
        try:
            metadata_store = ValidationMetadataStore.load(metadata_path)
        except ValidationMetadataError as exc:
            errors.append(str(exc))
    else:
        populated_splits = any(manifest.get("entries") for manifest in manifests.values())
        if not populated_splits:
            metadata_store = ValidationMetadataStore()
        else:
            errors.append(f"검증 메타데이터 누락: {metadata_path}")
    v = OODValidator(manifests, metadata_store)
    errors.extend(v.validate_disjoint())
    errors.extend(v.validate_cross_split_image_duplicates())
    warnings.extend(v.validate_alphabet_overlap())

    return errors, warnings


def main(argv=None) -> int:
    try:
        initialize_gpu_runtime()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    parser = argparse.ArgumentParser(prog="foveacore")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--config", default="configs/phase0.yaml")
    p_validate.add_argument("--data-dir", default=None)

    p_splits = sub.add_parser("build-splits")
    p_splits.add_argument("--config", default="configs/phase0.yaml")
    p_splits.add_argument("--out", default="data")

    p_smoke = sub.add_parser("smoke-baseline")
    p_smoke.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args(argv)

    try:
        assert_no_forbidden_packages()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.command == "validate":
        try:
            cfg = load_phase0_config(args.config)
        except yaml.YAMLError as exc:
            print(f"ERROR: {args.config}: {exc}")
            return 1
        errors = validate_phase0_config(cfg)
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        if args.data_dir is not None:
            data_errors, data_warnings = _validate_data_integrity(Path(args.data_dir))
            for warning in data_warnings:
                print(f"WARNING: {warning}")
            if data_errors:
                for err in data_errors:
                    print(f"ERROR: {err}")
                return 1
        print(f"validate PASS: {args.config}")
        return 0

    if args.command == "build-splits":
        cfg_path = Path(args.config)
        if cfg_path.exists():
            try:
                cfg = load_phase0_config(cfg_path)
            except yaml.YAMLError as exc:
                print(f"ERROR: {cfg_path}: {exc}")
                return 1
            errors = validate_phase0_config(cfg)
            if errors:
                for err in errors:
                    print(f"ERROR: {err}")
                return 1

        out_dir = Path(args.out)
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 체크섬 사전 검증 (T3-02)
        pre_errors = verify_mnist_checksums(raw_dir)
        if pre_errors and any("체크섬 불일치" in e for e in pre_errors):
            for err in pre_errors:
                print(f"ERROR: {err}")
            return 1

        try:
            checksums = download_mnist(raw_dir)
            print(f"build-splits: MNIST 원본 다운로드 완료 ({len(checksums)}개 파일)")
        except RuntimeError as exc:
            msg = str(exc)
            if "체크섬 불일치" in msg:
                print(f"ERROR: {msg}")
                return 1
            # 파일이 없고 네트워크 실패인 경우
            if not mnist_available(raw_dir):
                print(f"build-splits: MNIST 다운로드 실패({exc})")
                return 1
            print(f"build-splits: MNIST 다운로드 스킵({exc})")

        try:
            manifests = build_all_mnist_splits(
                raw_dir, out_dir, split_seed=DEFAULT_SEEDS["split_seed"]
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

        disjoint_errors = validate_split_ids_disjoint(manifests)
        if disjoint_errors:
            for err in disjoint_errors:
                print(f"ERROR: {err}")
            return 1
        print(f"build-splits: {out_dir}/split_manifest_v1.json 생성(T3) 완료")
        return 0

    if args.command == "smoke-baseline":
        print("smoke-baseline: 미구현 (T5/T8에서 구현 예정)")
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())