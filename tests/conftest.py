"""Phase 0 테스트의 전역 RNG·PyTorch·환경변수 상태 격리(T2)."""

import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from foveacore.utils import seed as seed_utils

_ENV_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_fresh_python():
    """CUDA 환경변수를 비운 새 인터프리터에서 코드를 실행한다."""
    def run(source):
        env = os.environ.copy()
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    return run


def _snapshot_rngs():
    snapshot = {}
    for key, rng in seed_utils._INDEPENDENT_RNGS.items():
        if isinstance(rng, random.Random):
            state = rng.getstate()
        elif isinstance(rng, np.random.Generator):
            state = copy.deepcopy(rng.bit_generator.state)
        elif isinstance(rng, torch.Generator):
            state = rng.get_state().clone()
        else:
            state = copy.deepcopy(rng)
        snapshot[key] = state
    return snapshot


def _restore_rngs(snapshot, applied_seeds):
    seed_utils._INDEPENDENT_RNGS.clear()
    for key, state in snapshot.items():
        rng = seed_utils.create_rng(key[1], applied_seeds[key[0]])
        if isinstance(rng, random.Random):
            rng.setstate(state)
        elif isinstance(rng, np.random.Generator):
            rng.bit_generator.state = copy.deepcopy(state)
        elif isinstance(rng, torch.Generator):
            rng.set_state(state)
        seed_utils._INDEPENDENT_RNGS[key] = rng


@pytest.fixture(autouse=True)
def isolate_global_state(monkeypatch):
    """단위 테스트가 바꾼 전역 상태를 복원하고 CUDA 초기화는 통합 테스트로 분리한다."""
    cuda_was_initialized = torch.cuda.is_initialized()
    cuda_rng_states = torch.cuda.get_rng_state_all() if cuda_was_initialized else None

    env_before = {key: os.environ.get(key) for key in _ENV_KEYS}
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    matmul_precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    applied_seeds = seed_utils.get_applied_seeds()
    independent_rngs = _snapshot_rngs()
    gpu_runtime_initialized = seed_utils._GPU_RUNTIME_INITIALIZED

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    yield

    for key, value in env_before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.set_rng_state(torch_rng_state)
    if cuda_was_initialized:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    seed_utils._APPLIED_SEEDS.clear()
    seed_utils._APPLIED_SEEDS.update(applied_seeds)
    seed_utils._GPU_RUNTIME_INITIALIZED = gpu_runtime_initialized
    _restore_rngs(independent_rngs, applied_seeds)
