"""FoveaCore Phase 0 시드 관리·GPU 결정론·FP32/TF32 차단기 (T2)."""

import os
import random

import numpy as np

SEED_NAMES = (
    "split_seed",
    "episode_seed",
    "generator_seed",
    "model_seed",
    "stream_order_seed",
)
DEFAULT_SEEDS = {
    "split_seed": 1001,
    "episode_seed": 2001,
    "generator_seed": 3001,
    "model_seed": 4001,
    "stream_order_seed": 5001,
}

_EXPECTED_CUBLAS = ":4096:8"


def verify_cublas_workspace_config():
    """CUBLAS_WORKSPACE_CONFIG=:4096:8 설정 상태 검증(CUDA 초기화 전 호출 권장)."""
    return os.environ.get("CUBLAS_WORKSPACE_CONFIG") == _EXPECTED_CUBLAS


def set_cublas_workspace_config():
    if not verify_cublas_workspace_config():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _EXPECTED_CUBLAS
    return verify_cublas_workspace_config()


def ensure_cublas_before_cuda():
    """CUDA 초기화 여부를 검사한 뒤 cuBLAS 환경변수를 설정(T2-01)."""
    import torch

    cuda_initialized = False
    try:
        cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        cuda_initialized = False

    if cuda_initialized:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _EXPECTED_CUBLAS:
            raise RuntimeError(
                "CUDA가 이미 초기화된 후 CUBLAS_WORKSPACE_CONFIG를 변경할 수 없습니다(T2-01)."
            )
        return True

    ok = set_cublas_workspace_config()
    if not ok:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG 설정 실패(T2-01).")
    return ok


def set_gpu_determinism(model_seed=None):
    """GPU 결정론·FP32·TF32 차단 강제기(T2). model_seed 지정 가능(T2-02)."""
    import torch

    if model_seed is None:
        model_seed = DEFAULT_SEEDS["model_seed"]

    ensure_cublas_before_cuda()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception as exc:
        raise RuntimeError(f"GPU 결정론 설정 실패(T2-02): {exc}") from exc

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    # 적용 결과 검증
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic_algorithms 활성화 실패(T2-02)")
    if torch.backends.cudnn.deterministic is not True:
        raise RuntimeError("cudnn.deterministic 활성화 실패(T2-02)")
    if torch.backends.cudnn.benchmark is not False:
        raise RuntimeError("cudnn.benchmark 비활성화 실패(T2-02)")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 차단 실패(T2-02)")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("matmul precision highest 설정 실패(T2-02)")
    return True


def is_gpu_deterministic():
    import torch
    return torch.are_deterministic_algorithms_enabled()


UNSTABLE_CUDA_OPS = {
    "grid_sample": "CUDA 비결정론 연산(grid_sample)",
    "scatter_add": "CUDA 비결정론 연산(scatter_add)",
}


def find_unstable_ops(used_ops):
    """비결정론 CUDA 연산(`grid_sample`·`scatter_add` 등 Phase 1~5 예정 연산 포함) 검출 훅."""
    return [op for op in UNSTABLE_CUDA_OPS if op in used_ops]


def assert_no_unstable_ops(used_ops):
    """비결정론 CUDA 연산 사용 시 RuntimeError(Phase 0~7 금지)."""
    found = find_unstable_ops(used_ops)
    if found:
        raise RuntimeError(f"비결정론 CUDA 연산 사용 금지(Phase 0~7): {found}")


# 용도별 독립 RNG 보관소(T2-01)
_INDEPENDENT_RNGS = {}


def get_rng(name, kind="python"):
    """용도별 독립 RNG 반환(T2-01).

    kind: "python"(random.Random), "numpy"(np.random.Generator), "torch"(torch.Generator)
    """
    if name not in SEED_NAMES:
        raise ValueError(f"알 수 없는 시드 이름: {name}")
    if name not in DEFAULT_SEEDS:
        raise ValueError(f"시드 기본값 없음: {name}")
    key = (name, kind)
    if key in _INDEPENDENT_RNGS:
        return _INDEPENDENT_RNGS[key]
    seed = DEFAULT_SEEDS[name]
    if kind == "python":
        import random as _r
        rng = _r.Random(seed)
    elif kind == "numpy":
        rng = np.random.default_rng(seed)
    elif kind == "torch":
        import torch
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        raise ValueError(f"지원하지 않는 RNG 종류: {kind}")
    _INDEPENDENT_RNGS[key] = rng
    return rng


def apply_seeds(seeds=None):
    """5종 시드를 용도별 독립 RNG에 적용(T2-01)."""
    merged = dict(DEFAULT_SEEDS)
    if seeds:
        merged.update(seeds)
    # 용도별 독립 적용: split_seed→split RNG, episode_seed→episode RNG, ...
    import random as _r
    _INDEPENDENT_RNGS.clear()
    for name in SEED_NAMES:
        val = merged[name]
        _INDEPENDENT_RNGS[(name, "python")] = _r.Random(val)
        _INDEPENDENT_RNGS[(name, "numpy")] = np.random.default_rng(val)
    import torch
    for name in SEED_NAMES:
        g = torch.Generator()
        g.manual_seed(merged[name])
        _INDEPENDENT_RNGS[(name, "torch")] = g
    torch.manual_seed(merged["model_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(merged["model_seed"])
    return merged
