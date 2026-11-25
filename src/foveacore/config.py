"""FoveaCore Phase 0 설정 로더·스키마 검증기 (T1)."""

from pathlib import Path

import yaml

ALLOWED_PROTOCOLS = {"STRICT_1SHOT", "CONTINUAL", "REFERENCE_P3"}
ALLOWED_POLICIES = {"P0", "P1", "P2", "P3"}
INPUT_MAX_SIDE = 1024
ALLOWED_RESIZE_METHODS = {"adaptive_average"}

# 기본 F3 계약과 고해상도 확장 프로파일을 분리한다.
# grid_size/channels는 프로파일과 반드시 일치해야 한다.
SENSOR_PROFILES = {
    "F3": {"grid_size": 5, "channels": 4, "stroke_clue_available": True},
    "F3_HIRES": {"grid_size": 97, "channels": 4, "stroke_clue_available": True},
}

REQUIRED_INPUT_KEYS = {"max_side", "resize_method"}
REQUIRED_SENSOR_KEYS = {"profile", "grid_size", "channels", "stroke_clue_available"}

REQUIRED_KEYS = {
    "version",
    "schema_version",
    "experiment_protocol",
    "pretraining_policy",
    "input",
    "sensor",
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
    "scope",
    "quick_episodes",
    "full_episodes",
    "way",
    "n_query_per_class",
}

ALLOWED_SMOKE_SCOPES = {"phase0_full_smoke", "general"}


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
    protocol = cfg["experiment_protocol"]
    policy = cfg["pretraining_policy"]
    if not isinstance(protocol, str) or protocol not in ALLOWED_PROTOCOLS:
        errors.append(
            f"experiment_protocol 부적절: {protocol!r} "
            f"(허용: {sorted(ALLOWED_PROTOCOLS)})"
        )
    if not isinstance(policy, str) or policy not in ALLOWED_POLICIES:
        errors.append(
            f"pretraining_policy 부적절: {policy!r} "
            f"(허용: {sorted(ALLOWED_POLICIES)})"
        )

    # input / sensor contract (T1)
    input_cfg = cfg["input"]
    if not isinstance(input_cfg, dict):
        errors.append("input이 dict여야 합니다.")
    else:
        for key in sorted(REQUIRED_INPUT_KEYS):
            if key not in input_cfg:
                errors.append(f"input.{key} 누락")
        max_side = input_cfg.get("max_side")
        if not _is_pos_int(max_side):
            errors.append("input.max_side가 양의 정수가 아닙니다.")
        elif max_side != INPUT_MAX_SIDE:
            errors.append(f"input.max_side는 {INPUT_MAX_SIDE}이어야 합니다.")
        resize_method = input_cfg.get("resize_method")
        if not isinstance(resize_method, str) or resize_method not in ALLOWED_RESIZE_METHODS:
            errors.append(
                f"input.resize_method 부적절: {resize_method!r} "
                f"(허용: {sorted(ALLOWED_RESIZE_METHODS)})"
            )

    sensor = cfg["sensor"]
    if not isinstance(sensor, dict):
        errors.append("sensor가 dict여야 합니다.")
    else:
        for key in sorted(REQUIRED_SENSOR_KEYS):
            if key not in sensor:
                errors.append(f"sensor.{key} 누락")
        profile = sensor.get("profile")
        if not isinstance(profile, str) or profile not in SENSOR_PROFILES:
            errors.append(
                f"sensor.profile 부적절: {profile!r} "
                f"(허용: {sorted(SENSOR_PROFILES)})"
            )
        else:
            expected = SENSOR_PROFILES[profile]
            grid_size = sensor.get("grid_size")
            channels = sensor.get("channels")
            stroke_clue_available = sensor.get("stroke_clue_available")
            if not _is_pos_int(grid_size):
                errors.append("sensor.grid_size가 양의 정수가 아닙니다.")
            elif grid_size != expected["grid_size"]:
                errors.append(
                    f"sensor.grid_size는 {profile} 프로파일에서 "
                    f"{expected['grid_size']}이어야 합니다."
                )
            if not _is_pos_int(channels):
                errors.append("sensor.channels가 양의 정수가 아닙니다.")
            elif channels != expected["channels"]:
                errors.append(
                    f"sensor.channels는 {profile} 프로파일에서 "
                    f"{expected['channels']}이어야 합니다."
                )
            if not isinstance(stroke_clue_available, bool):
                errors.append("sensor.stroke_clue_available가 bool이어야 합니다.")
            elif stroke_clue_available != expected["stroke_clue_available"]:
                errors.append(
                    f"sensor.stroke_clue_available는 {profile} 프로파일에서 "
                    f"{expected['stroke_clue_available']}이어야 합니다."
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
            else:
                smoke_cfg = cfg.get("smoke") if isinstance(cfg.get("smoke"), dict) else {}
                scope = smoke_cfg.get("scope") if isinstance(smoke_cfg, dict) else None
                smoke_way = smoke_cfg.get("way") if isinstance(smoke_cfg, dict) else None
                smoke_way_is_int = isinstance(smoke_way, int) and not isinstance(smoke_way, bool)
                if scope == "phase0_full_smoke":
                    if v != 10:
                        errors.append(
                            "공식 full smoke(scope=phase0_full_smoke)에서 "
                            "datasets.support_per_episode는 10이어야 합니다 "
                            "(MNIST 공식 full smoke 요구)."
                        )
                elif cfg["experiment_protocol"] == "STRICT_1SHOT":
                    if scope == "general":
                        if smoke_way_is_int and v != smoke_way:
                            errors.append(
                                "STRICT_1SHOT scope=general에서 "
                                "datasets.support_per_episode는 smoke.way와 일치해야 합니다."
                            )
                    elif smoke_way_is_int and v != smoke_way:
                        # scope가 유효하지 않으면 smoke 쪽 검증에서 이미 오류를 보고한다.
                        errors.append(
                            "STRICT_1SHOT에서 datasets.support_per_episode는 "
                            "smoke.way와 일치해야 합니다."
                        )
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
        scope = smoke.get("scope")
        if not isinstance(scope, str) or scope not in ALLOWED_SMOKE_SCOPES:
            errors.append(
                f"smoke.scope 부적절: {scope!r} "
                f"(허용: {sorted(ALLOWED_SMOKE_SCOPES)})"
            )
        way = smoke.get("way")
        if not _is_pos_int(way):
            errors.append("smoke.way가 양의 정수가 아닙니다.")
        else:
            # MNIST 데이터 수량·full_episodes·n_query 고정 검증은 별도로 유지한다.
            if scope == "phase0_full_smoke" and way != 10:
                errors.append(
                    "공식 full smoke(scope=phase0_full_smoke)에서 "
                    "smoke.way는 10이어야 합니다 (MNIST 공식 full smoke 요구)."
                )
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