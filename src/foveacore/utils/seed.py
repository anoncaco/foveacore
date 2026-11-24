"""FoveaCore Phase 0 시드 관리·GPU 결정론·FP32/TF32 차단기 (T2)."""

from collections.abc import Mapping
from contextlib import contextmanager
import os
import random

import numpy as np

from foveacore.core.exceptions import NumericalInstabilityError

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
_MAX_SEED = (1 << 64) - 1
_SUPPORTED_RNG_KINDS = frozenset({"python", "numpy", "torch"})


def _validate_seed(name, value):
    """시드는 bool이 아닌 비음수 정수이며 PyTorch 시드 범위를 만족해야 한다."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}는 정수여야 합니다: {value!r}")
    value = int(value)
    if not 0 <= value <= _MAX_SEED:
        raise ValueError(f"{name}는 0..{_MAX_SEED} 범위여야 합니다: {value}")
    return value


def _normalize_seeds(seeds=None):
    merged = dict(DEFAULT_SEEDS)
    if seeds is None:
        return merged
    if not isinstance(seeds, Mapping):
        raise TypeError("seeds는 mapping이어야 합니다.")
    unknown = set(seeds) - set(SEED_NAMES)
    if unknown:
        raise ValueError(f"알 수 없는 시드 이름: {sorted(unknown)}")
    for name, value in seeds.items():
        merged[name] = _validate_seed(name, value)
    return merged


def verify_cublas_workspace_config():
    """CUBLAS_WORKSPACE_CONFIG=:4096:8 설정 상태 검증."""
    return os.environ.get("CUBLAS_WORKSPACE_CONFIG") == _EXPECTED_CUBLAS


def set_cublas_workspace_config():
    """cuBLAS 결정론 작업공간 설정(호출 전 CUDA 초기화 여부는 ensure 함수에서 검사)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _EXPECTED_CUBLAS
    return verify_cublas_workspace_config()


def ensure_cublas_before_cuda():
    """CUDA 초기화 전 cuBLAS 환경변수를 설정하고, 초기화 후 변경을 차단한다."""
    import torch

    # is_available()는 드라이버 조회를 수반할 수 있으므로 초기화 여부만 확인한다.
    if torch.cuda.is_initialized():
        if not verify_cublas_workspace_config():
            raise RuntimeError(
                "CUDA가 이미 초기화된 후 CUBLAS_WORKSPACE_CONFIG를 변경할 수 없습니다(T2)."
            )
        return True
    if not set_cublas_workspace_config():
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG 설정 실패(T2).")
    return True


def set_gpu_determinism(model_seed=None):
    """GPU 결정론·FP32·TF32 차단을 강제하고 적용 결과를 검증한다."""
    import torch

    model_seed = _validate_seed("model_seed", DEFAULT_SEEDS["model_seed"] if model_seed is None else model_seed)
    ensure_cublas_before_cuda()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
    except Exception as exc:
        raise RuntimeError(f"GPU 결정론 설정 실패(T2): {exc}") from exc

    status = gpu_determinism_status()
    failed = [name for name, value in status.items() if value is not True]
    if failed:
        raise RuntimeError(f"GPU 결정론 적용 검증 실패(T2): {failed}")
    return True


def gpu_determinism_status():
    """결정론 강제기 6개 항목의 실제 적용 상태를 반환한다."""
    import torch

    return {
        "cublas_workspace_config": verify_cublas_workspace_config(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": torch.backends.cudnn.deterministic is True,
        "cudnn_benchmark_disabled": torch.backends.cudnn.benchmark is False,
        "matmul_precision_highest": torch.get_float32_matmul_precision() == "highest",
        "tf32_disabled": not (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ),
    }


def is_gpu_deterministic():
    return all(gpu_determinism_status().values())


# Phase 1~5에서 사용할 수 있는 CUDA 비결정론 연산의 정규화 이름.
UNSTABLE_CUDA_OPS = {
    "grid_sample": "CUDA 비결정론 연산(grid_sample)",
    "scatter_add": "CUDA 비결정론 연산(scatter_add)",
    "index_add": "CUDA 비결정론 연산(index_add)",
    "scatter": "CUDA 비결정론 연산(scatter)",
    "index_copy": "CUDA 비결정론 연산(index_copy)",
    "embedding_dense_backward": "CUDA 비결정론 연산(embedding_dense_backward)",
    "cumsum": "CUDA 비결정론 연산(cumsum)",
}
_UNSTABLE_ALIASES = {
    "grid_sampler": "grid_sample",
    "grid_sampler_2d": "grid_sample",
    "grid_sampler_3d": "grid_sample",
}


def _canonical_op_name(op):
    name = str(getattr(op, "__name__", op))
    if name.startswith("aten::"):
        name = name[len("aten::"):]
    name = name.split(".", 1)[0].rstrip("_")
    return _UNSTABLE_ALIASES.get(name, name)


def find_unstable_ops(used_ops):
    """문자열 연산 이름에서 비결정론 CUDA 연산을 정규화해 찾는다."""
    if isinstance(used_ops, str):
        used_ops = (used_ops,)
    normalized = {_canonical_op_name(op) for op in used_ops}
    return [name for name in UNSTABLE_CUDA_OPS if name in normalized]


def assert_no_unstable_ops(used_ops):
    found = find_unstable_ops(used_ops)
    if found:
        raise RuntimeError(f"비결정론 CUDA 연산 사용 금지(Phase 0~7): {found}")


def _has_cuda_tensor(torch_module, value):
    if isinstance(value, torch_module.Tensor):
        return value.is_cuda
    if isinstance(value, Mapping):
        return any(_has_cuda_tensor(torch_module, v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_cuda_tensor(torch_module, v) for v in value)
    return False


@contextmanager
def detect_unstable_cuda_ops():
    """TorchDispatchMode로 실제 CUDA 연산을 기록하고 위반 시 RuntimeError를 낸다.

    CPU 연산은 결정론 보장 대상이 아니므로 기록하지 않는다. yield되는 집합에는
    감지된 정규화된 연산 이름만 들어간다.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    used_ops = set()

    class _Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = _canonical_op_name(func)
            if name in UNSTABLE_CUDA_OPS and _has_cuda_tensor(
                torch, (args, kwargs or {})
            ):
                used_ops.add(name)
            return func(*args, **(kwargs or {}))

    with _Recorder():
        yield used_ops
    assert_no_unstable_ops(used_ops)


def assert_reproducible(operation, *args, repeats=2, **kwargs):
    """동일 연산을 반복 실행해 결과가 결정론적인지 검증하는 테스트 훅."""
    import torch

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats는 2 이상의 정수여야 합니다.")
    baseline = operation(*args, **kwargs)
    for _ in range(repeats - 1):
        current = operation(*args, **kwargs)
        if isinstance(baseline, torch.Tensor):
            equal = torch.equal(baseline, current)
        elif isinstance(baseline, np.ndarray):
            equal = np.array_equal(baseline, current, equal_nan=True)
        else:
            equal = baseline == current
        if not equal:
            raise NumericalInstabilityError("반복 실행 결과가 결정론적이지 않습니다(T2).")
    return baseline


# 용도별 독립 RNG 보관소(T2).
_INDEPENDENT_RNGS = {}


def create_rng(kind, seed):
    """명시적 시드로 독립 RNG를 생성한다(캐시하지 않음)."""
    if kind not in _SUPPORTED_RNG_KINDS:
        raise ValueError(f"지원하지 않는 RNG 종류: {kind}")
    seed = _validate_seed("seed", seed)
    if kind == "python":
        return random.Random(seed)
    if kind == "numpy":
        return np.random.default_rng(seed)
    import torch

    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def get_rng(name, kind="python"):
    """용도별 독립 RNG를 반환한다(apply_seeds 이후에는 적용된 시드 사용)."""
    if name not in SEED_NAMES:
        raise ValueError(f"알 수 없는 시드 이름: {name}")
    if kind not in _SUPPORTED_RNG_KINDS:
        raise ValueError(f"지원하지 않는 RNG 종류: {kind}")
    key = (name, kind)
    if key not in _INDEPENDENT_RNGS:
        _INDEPENDENT_RNGS[key] = create_rng(kind, DEFAULT_SEEDS[name])
    return _INDEPENDENT_RNGS[key]


def apply_seeds(seeds=None):
    """5종 시드를 검증해 용도별 독립 RNG와 전역 모델 RNG에 적용한다."""
    merged = _normalize_seeds(seeds)
    import torch

    ensure_cublas_before_cuda()
    _INDEPENDENT_RNGS.clear()
    for name in SEED_NAMES:
        for kind in _SUPPORTED_RNG_KINDS:
            _INDEPENDENT_RNGS[(name, kind)] = create_rng(kind, merged[name])
    torch.manual_seed(merged["model_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(merged["model_seed"])
    return merged
