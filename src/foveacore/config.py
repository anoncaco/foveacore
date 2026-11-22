"""FoveaCore Phase 0 설정 로더·스키마 검증기 (T1)."""

from pathlib import Path

import yaml

ALLOWED_PROTOCOLS = {"STRICT_1SHOT", "CONTINUAL", "REFERENCE_P3"}
ALLOWED_POLICIES = {"P0", "P1", "P2", "P3"}

REQUIRED_KEYS = {
    "version",
    "schema_version",
    "experiment_protocol",
    "pretraining_policy",
    "datasets",
    "splits_synthetic",
    "smoke",
    "forbidden_packages",
}

REQUIRED_DATASET_KEYS = {
    "mnist_train",
    "mnist_test",
    "support_per_episode",
    "dev_stream",
    "main_stream",
    "p2_non_digit",
    "non_digit_dev",
    "dev_ood",
    "test_ood",
    "phase7_char_transfer",
}

REQUIRED_SYNTHETIC_KEYS = {"train", "dev", "test"}

REQUIRED_SMOKE_KEYS = {
    "quick_episodes",
    "full_episodes",
    "way",
    "n_query_per_class",
}


def load_phase0_config(path):
    """YAML 설정 파일을 dict로 로드합니다."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_phase0_config(cfg):
    """스키마 검증: 오류 목록(빈 리스트 = PASS). 문서 기준 필드/경계 검사."""
    errors = []
    if not isinstance(cfg, dict):
        return ["설정이 dict가 아닙니다."]

    for key in sorted(REQUIRED_KEYS):
        if key not in cfg:
            errors.append(f"필수 키 누락: {key}")

    if errors:
        return errors

    # version / schema_version
    if not isinstance(cfg.get("version"), str) or not cfg["version"]:
        errors.append("version이 비어있지 않은 문자열이어야 합니다.")
    if not isinstance(cfg.get("schema_version"), str) or not cfg["schema_version"]:
        errors.append("schema_version이 비어있지 않은 문자열이어야 합니다.")

    # experiment protocol / pretraining policy
    if cfg["experiment_protocol"] not in ALLOWED_PROTOCOLS:
        errors.append(
            f"experiment_protocol 부적절: {cfg['experiment_protocol']!r} "
            f"(허용: {sorted(ALLOWED_PROTOCOLS)})"
        )
    if cfg["pretraining_policy"] not in ALLOWED_POLICIES:
        errors.append(
            f"pretraining_policy 부적절: {cfg['pretraining_policy']!r} "
            f"(허용: {sorted(ALLOWED_POLICIES)})"
        )

    # datasets
    datasets = cfg["datasets"]
    if not isinstance(datasets, dict):
        errors.append("datasets가 dict여야 합니다.")
    else:
        for key in sorted(REQUIRED_DATASET_KEYS):
            if key not in datasets:
                errors.append(f"datasets.{key} 누락")
        if "mnist_train" in datasets:
            if not _is_pos_int(datasets.get("mnist_train")):
                errors.append("datasets.mnist_train가 양의 정수가 아닙니다.")
            elif datasets.get("mnist_train") != 60000:
                errors.append("datasets.mnist_train은 60000이어야 합니다.")
        if "mnist_test" in datasets:
            if not _is_pos_int(datasets.get("mnist_test")):
                errors.append("datasets.mnist_test가 양의 정수가 아닙니다.")
            elif datasets.get("mnist_test") != 10000:
                errors.append("datasets.mnist_test는 10000이어야 합니다.")
        if "dev_stream" in datasets and datasets.get("dev_stream") is not None:
            if not _is_pos_int(datasets.get("dev_stream")):
                errors.append("datasets.dev_stream가 양의 정수 또는 null이어야 합니다.")
            elif datasets.get("dev_stream") != 15000:
                errors.append("datasets.dev_stream은 15000이어야 합니다.")
        if "main_stream" in datasets and datasets.get("main_stream") is not None:
            if not _is_pos_int(datasets.get("main_stream")):
                errors.append("datasets.main_stream가 양의 정수 또는 null이어야 합니다.")
            elif datasets.get("main_stream") != 35000:
                errors.append("datasets.main_stream은 35000이어야 합니다.")
        if "support_per_episode" in datasets:
            v = datasets.get("support_per_episode")
            if not _is_pos_int(v):
                errors.append("datasets.support_per_episode가 양의 정수가 아닙니다.")
            elif cfg["experiment_protocol"] == "STRICT_1SHOT" and v != 10:
                errors.append("STRICT_1SHOT에서 support_per_episode는 10이어야 합니다.")
        for stream_key in ("dev_stream", "main_stream"):
            v = datasets.get(stream_key)
            if v is not None and not _is_pos_int(v):
                errors.append(f"datasets.{stream_key}가 양의 정수 또는 null이어야 합니다.")
        for nullable_key in ("p2_non_digit", "non_digit_dev", "dev_ood", "test_ood", "phase7_char_transfer"):
            v = datasets.get(nullable_key)
            if v is not None and not _is_pos_int(v):
                errors.append(f"datasets.{nullable_key}가 양의 정수 또는 null이어야 합니다.")

    # splits_synthetic
    synthetic = cfg["splits_synthetic"]
    if not isinstance(synthetic, dict):
        errors.append("splits_synthetic이 dict여야 합니다.")
    else:
        for key in sorted(REQUIRED_SYNTHETIC_KEYS):
            if key not in synthetic:
                errors.append(f"splits_synthetic.{key} 누락")
        vals = [synthetic.get(k) for k in ("train", "dev", "test")]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            if any(v <= 0 for v in vals):
                errors.append("splits_synthetic 비율은 양수여야 합니다.")
            elif sum(vals) != 100:
                errors.append("splits_synthetic train/dev/test 합이 100이어야 합니다.")
            elif not (vals[0] == 70 and vals[1] == 15 and vals[2] == 15):
                errors.append("splits_synthetic은 70:15:15여야 합니다.")
        else:
            for k in ("train", "dev", "test"):
                if not isinstance(synthetic.get(k), int) or isinstance(synthetic.get(k), bool):
                    errors.append(f"splits_synthetic.{k}가 정수가 아닙니다.")

    # smoke
    smoke = cfg["smoke"]
    if not isinstance(smoke, dict):
        errors.append("smoke가 dict여야 합니다.")
    else:
        for key in sorted(REQUIRED_SMOKE_KEYS):
            if key not in smoke:
                errors.append(f"smoke.{key} 누락")
        if smoke.get("way") != 10:
            errors.append("smoke.way은 통합 프로젝트(Phase 0~7) 기준 10이어야 합니다.")
        nq = smoke.get("n_query_per_class")
        if not _is_pos_int(nq):
            errors.append("smoke.n_query_per_class가 양의 정수여야 합니다.")
        elif nq != 20:
            errors.append("smoke.n_query_per_class은 20이어야 합니다.")
        qe = smoke.get("quick_episodes")
        if not _is_pos_int(qe):
            errors.append("smoke.quick_episodes가 양의 정수여야 합니다.")
        elif not (1 <= qe <= 3):
            errors.append("smoke.quick_episodes는 1~3이어야 합니다.")
        fe = smoke.get("full_episodes")
        if not _is_pos_int(fe):
            errors.append("smoke.full_episodes가 양의 정수여야 합니다.")
        elif fe != 30:
            errors.append("smoke.full_episodes는 30이어야 합니다.")

    # forbidden_packages
    fp = cfg["forbidden_packages"]
    if not isinstance(fp, list) or not fp:
        errors.append("forbidden_packages가 비어있지 않은 리스트여야 합니다.")
    elif not all(isinstance(p, str) and p for p in fp):
        errors.append("forbidden_packages 항목은 비어있지 않은 문자열이어야 합니다.")

    return errors