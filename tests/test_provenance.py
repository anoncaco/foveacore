"""utils/provenance.py 환경 정보 수집 테스트 (T2)."""

from foveacore.utils.provenance import collect_provenance


def test_collect_provenance_required_keys():
    p = collect_provenance()
    for key in (
        "schema_version", "os", "hostname", "torch_version", "compute_device",
        "seeds", "deterministic_mode", "cudnn_deterministic", "cudnn_benchmark",
        "cublas_workspace_config", "torch_num_threads", "git_commit", "git_dirty",
        "packages", "gpu_name", "cuda_version", "cudnn_version",
        "precision", "allow_tf32", "allow_tf32_matmul", "allow_tf32_cudnn",
    ):
        assert key in p, f"필수 키 누락: {key}"


def test_seeds_present():
    p = collect_provenance()
    assert set(p["seeds"].keys()) == {
        "split_seed", "episode_seed", "generator_seed", "model_seed", "stream_order_seed",
    }
