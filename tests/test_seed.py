"""utils/seed.py 시드·GPU 결정론·TF32 차단 테스트 (T2)."""

import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from foveacore.core.exceptions import NumericalInstabilityError
from foveacore.utils.seed import (
    DEFAULT_SEEDS,
    SEED_NAMES,
    apply_seeds,
    assert_no_unstable_ops,
    assert_reproducible,
    create_rng,
    detect_unstable_cuda_ops,
    ensure_cublas_before_cuda,
    ensure_gpu_runtime_initialized,
    find_unstable_ops,
    get_applied_seeds,
    get_rng,
    gpu_determinism_status,
    initialize_gpu_runtime,
    is_gpu_deterministic,
    set_cublas_workspace_config,
    set_gpu_determinism,
    verify_cublas_workspace_config,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_cuda_script(source):
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


def test_seed_names_are_the_five():
    assert SEED_NAMES == (
        "split_seed",
        "episode_seed",
        "generator_seed",
        "model_seed",
        "stream_order_seed",
    )


def test_apply_seeds_returns_five_seed_keys():
    seeds = apply_seeds()
    assert set(seeds) == set(SEED_NAMES)


def test_apply_seeds_records_normalized_snapshot():
    expected = dict(DEFAULT_SEEDS, model_seed=88)
    apply_seeds({"model_seed": 88})
    recorded = get_applied_seeds()
    assert recorded == expected
    recorded["model_seed"] = 0
    assert get_applied_seeds() == expected


def test_partial_seed_override_is_reproducible():
    overrides = {"episode_seed": 17}
    first = apply_seeds(overrides)
    assert first["episode_seed"] == 17
    assert first["split_seed"] == DEFAULT_SEEDS["split_seed"]
    first_values = [get_rng(name, "python").random() for name in SEED_NAMES]

    apply_seeds(overrides)
    second_values = [get_rng(name, "python").random() for name in SEED_NAMES]
    assert first_values == second_values


def test_independent_rngs_do_not_advance_each_other():
    apply_seeds()
    get_rng("split_seed", "python").random()
    assert get_rng("episode_seed", "python").random() == random.Random(2001).random()


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (-1, ValueError),
        (1 << 64, ValueError),
        ("1", TypeError),
        (1.5, TypeError),
    ],
)
def test_apply_seeds_rejects_invalid_values(value, error):
    with pytest.raises(error):
        apply_seeds({"model_seed": value})


def test_apply_seeds_rejects_unknown_names_and_non_mapping():
    with pytest.raises(ValueError, match="알 수 없는 시드"):
        apply_seeds({"unknown_seed": 1})
    with pytest.raises(TypeError, match="mapping"):
        apply_seeds([1, 2, 3])


def test_create_rng_supports_three_kinds():
    assert isinstance(create_rng("python", 1), random.Random)
    assert isinstance(create_rng("numpy", 1), np.random.Generator)
    assert isinstance(create_rng("torch", 1), torch.Generator)
    with pytest.raises(ValueError, match="RNG 종류"):
        create_rng("other", 1)


def test_get_rng_rejects_unknown_name_and_kind():
    with pytest.raises(ValueError, match="시드 이름"):
        get_rng("unknown")
    with pytest.raises(ValueError, match="RNG 종류"):
        get_rng("split_seed", "other")


def test_set_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert set_cublas_workspace_config()
    assert verify_cublas_workspace_config()


def test_ensure_cublas_sets_before_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert ensure_cublas_before_cuda()
    assert verify_cublas_workspace_config()


def test_ensure_cublas_rejects_change_after_initialization(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="이미 초기화"):
        ensure_cublas_before_cuda()


def test_initialize_gpu_runtime_enforces_setup_order(monkeypatch):
    calls = []

    def set_cublas():
        calls.append("cublas")
        return True

    def set_determinism(model_seed=None):
        calls.append(("determinism", model_seed))
        return True

    monkeypatch.setattr("foveacore.utils.seed.ensure_cublas_before_cuda", set_cublas)
    monkeypatch.setattr("foveacore.utils.seed.set_gpu_determinism", set_determinism)
    assert initialize_gpu_runtime(88) is True
    assert calls == ["cublas", ("determinism", 88)]


def test_set_gpu_determinism_marks_runtime_initialized():
    set_gpu_determinism()
    assert ensure_gpu_runtime_initialized() is True


def test_set_gpu_determinism_flags(monkeypatch):
    calls = []
    original = torch.use_deterministic_algorithms

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", record)
    assert set_gpu_determinism() is True
    assert calls[-1]["warn_only"] is False
    assert all(gpu_determinism_status().values())
    assert is_gpu_deterministic() is True
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scatter_add", "scatter_add"),
        ("aten::scatter_add_.default", "scatter_add"),
        ("aten::grid_sampler_2d.default", "grid_sample"),
        ("index_copy_", "index_copy"),
        ("other_op", None),
    ],
)
def test_unstable_ops_normalization(name, expected):
    assert find_unstable_ops({name}) == ([] if expected is None else [expected])


def test_unstable_ops_guard_allows_cpu_operations():
    target = torch.zeros(3)
    with detect_unstable_cuda_ops() as used_ops:
        target.scatter_add_(0, torch.tensor([0, 2]), torch.tensor([1.0, 2.0]))
    assert used_ops == set()
    assert_no_unstable_ops(set())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_unstable_ops_guard_detects_cuda_grid_sample():
    result = _run_cuda_script(
        """
        import torch
        from foveacore.utils.seed import detect_unstable_cuda_ops

        image = torch.zeros(1, 1, 4, 4, device="cuda")
        grid = torch.zeros(1, 2, 2, 3, device="cuda")
        try:
            with detect_unstable_cuda_ops():
                torch.nn.functional.grid_sample(image, grid, align_corners=False)
        except RuntimeError as exc:
            assert "grid_sample" in str(exc), exc
        else:
            raise AssertionError("비결정론 CUDA 연산이 감지되지 않았습니다.")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_gpu_determinism_cuda_integration_in_fresh_process():
    result = _run_cuda_script(
        """
        import os
        import torch

        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert not torch.cuda.is_initialized()
        from foveacore.utils.seed import is_gpu_deterministic, set_gpu_determinism

        set_gpu_determinism()
        torch.ones(1, device="cuda")
        assert torch.cuda.is_initialized()
        assert is_gpu_deterministic()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_assert_reproducible_accepts_equal_results():
    assert assert_reproducible(lambda x: x + 1, 2, repeats=3) == 3


def test_assert_reproducible_accepts_stable_reused_buffers():
    array = np.zeros(3, dtype=np.float32)

    def numpy_operation():
        array[...] = [1.0, 2.0, 3.0]
        return array

    tensor = torch.zeros(3)

    def tensor_operation():
        tensor.copy_(torch.tensor([1.0, 2.0, 3.0]))
        return tensor

    assert assert_reproducible(numpy_operation, repeats=3).tolist() == [1.0, 2.0, 3.0]
    assert assert_reproducible(tensor_operation, repeats=3).tolist() == [1.0, 2.0, 3.0]


def test_assert_reproducible_rejects_inplace_mutation():
    array = np.zeros(3, dtype=np.float32)

    def numpy_operation():
        array[...] += 1
        return array

    tensor = torch.zeros(3)

    def tensor_operation():
        tensor.add_(1)
        return tensor

    with pytest.raises(NumericalInstabilityError, match="결정론적"):
        assert_reproducible(numpy_operation, repeats=3)
    with pytest.raises(NumericalInstabilityError, match="결정론적"):
        assert_reproducible(tensor_operation, repeats=3)


def test_assert_reproducible_rejects_changing_results():
    values = iter([1, 2])
    with pytest.raises(NumericalInstabilityError, match="결정론적"):
        assert_reproducible(lambda: next(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_model_seed_consistency_fresh_process():
    result = _run_cuda_script(
        """
        import torch
        from foveacore.utils.seed import (
            DEFAULT_SEEDS,
            apply_seeds,
            get_applied_seeds,
            initialize_gpu_runtime,
        )

        assert get_applied_seeds()["model_seed"] == DEFAULT_SEEDS["model_seed"]
        initialize_gpu_runtime(88)
        assert get_applied_seeds()["model_seed"] == 88
        assert torch.initial_seed() == 88
        if torch.cuda.is_available():
            assert torch.cuda.initial_seed() == 88

        apply_seeds({"model_seed": 88})
        initialize_gpu_runtime()
        assert get_applied_seeds()["model_seed"] == 88
        assert torch.initial_seed() == 88
        if torch.cuda.is_available():
            assert torch.cuda.initial_seed() == 88
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_initialize_gpu_runtime_explicit_model_seed_updates_applied_and_torch():
    initialize_gpu_runtime(88)
    assert get_applied_seeds()["model_seed"] == 88
    assert torch.initial_seed() == 88
    assert ensure_gpu_runtime_initialized() is True


def test_initialize_gpu_runtime_none_uses_last_applied_model_seed():
    apply_seeds({"model_seed": 88})
    before = get_applied_seeds()
    initialize_gpu_runtime()
    assert get_applied_seeds() == before
    assert torch.initial_seed() == 88


def test_initialize_gpu_runtime_default_matches_default_seed():
    assert get_applied_seeds()["model_seed"] == DEFAULT_SEEDS["model_seed"]


def test_failed_initialization_does_not_update_applied_seeds_or_rng(monkeypatch):
    apply_seeds({"model_seed": 7})
    before_applied = get_applied_seeds()
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()

    monkeypatch.setattr(
        "foveacore.utils.seed.gpu_determinism_status", lambda: {"x": False}
    )
    with pytest.raises(RuntimeError):
        initialize_gpu_runtime(99)
    assert get_applied_seeds() == before_applied
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_invalid_explicit_model_seed_does_not_mutate_state():
    apply_seeds({"model_seed": 7})
    before_applied = get_applied_seeds()
    with pytest.raises((TypeError, ValueError)):
        initialize_gpu_runtime("bad")
    assert get_applied_seeds() == before_applied


def test_warn_only_true_makes_ensure_gpu_runtime_fail():
    initialize_gpu_runtime()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(RuntimeError) as exc:
        ensure_gpu_runtime_initialized()
    assert "warn_only" in str(exc.value)
