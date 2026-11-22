"""utils/seed.py 시드·GPU 결정론·TF32 차단 테스트 (T2)."""

import pytest

from foveacore.utils.seed import (
    SEED_NAMES,
    apply_seeds,
    assert_no_unstable_ops,
    find_unstable_ops,
    set_gpu_determinism,
    set_cublas_workspace_config,
    verify_cublas_workspace_config,
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
    assert set(seeds.keys()) == set(SEED_NAMES)


def test_set_cublas_workspace_config():
    assert set_cublas_workspace_config()
    assert verify_cublas_workspace_config()


def test_set_gpu_determinism_flags():
    import torch

    assert set_gpu_determinism() is True
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_unstable_ops_hook():
    assert find_unstable_ops({"scatter_add"}) == ["scatter_add"]
    assert find_unstable_ops({"other_op"}) == []
    assert_no_unstable_ops(set())
    with pytest.raises(RuntimeError):
        assert_no_unstable_ops({"grid_sample"})
