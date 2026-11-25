"""FoveaCore Phase 0 환경 정보 수집기 (T2).

OS/CPU/GPU(`gpu_name`)/CUDA(`cuda_version`)/cuDNN(`cudnn_version`)/
PyTorch(`torch_version`)/정밀도(`precision`, `allow_tf32`)/결정론 설정/
스레드/Git/패키지 수집.
"""

import hashlib
import os
import platform
import socket
import subprocess
from importlib import metadata
from pathlib import Path

from foveacore.utils.seed import ensure_gpu_runtime_initialized, get_applied_seeds

SCHEMA_VERSION = "1.0"


def _project_root():
    """실행 위치와 무관하게 저장소 루트를 계산한다."""
    return Path(__file__).resolve().parents[3]


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
            cwd=_project_root(), capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def is_git_dirty():
    # git_dirty 기준(STATE.md): src/, configs/, locks/, pyproject.toml 내 수정·미추적(untracked) 파일 존재
    # 조회 실패는 clean(False)과 구분하기 위해 None 반환(T2-10).
    project_root = _project_root()
    try:
        out = subprocess.run(
            [
                "git", "status", "--porcelain", "--untracked-files=all",
                "--", "src", "configs", "locks", "pyproject.toml",
            ],
            cwd=project_root, capture_output=True, text=True, check=True,
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


def _dependency_lock_hash():
    """locks/phase0_requirements.lock 파일의 SHA-256 해시 반환.
    실행 위치와 무관하게 프로젝트 루트의 잠금 파일을 읽음.
    """
    project_root = _project_root()
    lock_path = project_root / "locks" / "phase0_requirements.lock"
    if lock_path.exists():
        return hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return None


def collect_provenance(seeds=None):
    """운영 초기화 이후 전체 환경 정보를 수집한다(내부에서 결정론을 재설정하지 않음)."""
    ensure_gpu_runtime_initialized()
    from foveacore.utils.seed import _normalize_seeds

    t = _torch()
    applied = get_applied_seeds()
    if seeds is not None:
        requested = _normalize_seeds(seeds)
        if requested != applied:
            raise ValueError("seeds가 실제로 적용된 시드와 일치하지 않습니다(T2).")
        applied = requested

    if t.cuda.is_available():
        compute_device = f"cuda:{t.cuda.current_device()}"
    else:
        compute_device = "cpu"

    try:
        torch_interop_threads = t.get_num_interop_threads()
    except Exception:
        torch_interop_threads = None

    p = {
        "schema_version": SCHEMA_VERSION,
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
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
        "torch_interop_threads": torch_interop_threads,
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
        "packages": _collect_packages(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "gpu_name": get_gpu_name(),
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
    }
    p.update(get_precision_settings())
    return p
