"""FoveaCore Phase 0 환경 정보 수집기 (T2).

OS/CPU/GPU(`gpu_name`)/CUDA(`cuda_version`)/cuDNN(`cudnn_version`)/
PyTorch(`torch_version`)/정밀도(`precision`, `allow_tf32`)/결정론 설정/
스레드/Git/패키지 수집.
"""

import os
import platform
import socket
import subprocess
from importlib import metadata

from foveacore.utils.seed import DEFAULT_SEEDS

SCHEMA_VERSION = "1.0"


def _torch():
    import torch
    return torch


def get_gpu_name():
    try:
        return _torch().cuda.get_device_name(0)
    except Exception:
        return None


def get_cuda_version():
    try:
        return _torch().version.cuda
    except Exception:
        return None


def get_cudnn_version():
    try:
        return _torch().backends.cudnn.version()
    except Exception:
        return None


def get_precision_settings():
    t = _torch()
    matmul_tf32 = t.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = t.backends.cudnn.allow_tf32
    return {
        "precision_dtype": "fp32",
        "precision": t.get_float32_matmul_precision(),
        "matmul_precision": t.get_float32_matmul_precision(),
        "allow_tf32_matmul": matmul_tf32,
        "allow_tf32_cudnn": cudnn_tf32,
        "allow_tf32": matmul_tf32 or cudnn_tf32,
    }


def get_git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def is_git_dirty():
    # git_dirty 기준(STATE.md): src/, configs/, locks/, pyproject.toml 내 수정·미추적(untracked) 파일 존재
    # 조회 실패는 clean(False)과 구분하기 위해 None 반환(T2-10).
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs", "locks", "pyproject.toml"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _collect_packages():
    pkgs = set()
    for d in metadata.distributions():
        name = d.metadata.get("Name")
        version = d.version
        if name and version:
            pkgs.add(f"{name}=={version}")
    return sorted(pkgs)


def collect_provenance(seeds=None):
    """전체 환경 정보 수집 → dict(manifest에 포함되는 필드 그룹)."""
    from foveacore.utils.seed import DEFAULT_SEEDS

    t = _torch()
    applied = dict(DEFAULT_SEEDS)
    if seeds:
        applied.update(seeds)

    if t.cuda.is_available():
        compute_device = f"cuda:{t.cuda.current_device()}"
    else:
        compute_device = "cpu"

    p = {
        "schema_version": SCHEMA_VERSION,
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": t.__version__,
        "compute_device": compute_device,
        "cuda_available": bool(t.cuda.is_available()),
        "cuda_device_count": (t.cuda.device_count() if t.cuda.is_available() else 0),
        "seeds": applied,
        "deterministic_mode": t.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": t.backends.cudnn.deterministic,
        "cudnn_benchmark": t.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": t.get_num_threads(),
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
        "packages": _collect_packages(),
        "gpu_name": get_gpu_name(),
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
    }
    p.update(get_precision_settings())
    return p
