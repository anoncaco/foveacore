"""utils/provenance.py 환경 정보 수집 테스트 (T2)."""

import hashlib
import subprocess
from pathlib import Path

from foveacore.utils.provenance import collect_provenance


def test_collect_provenance_required_keys():
    p = collect_provenance()
    for key in (
        "schema_version", "os", "hostname", "torch_version", "compute_device",
        "seeds", "deterministic_mode", "cudnn_deterministic", "cudnn_benchmark",
        "cublas_workspace_config", "torch_num_threads", "git_commit", "git_dirty",
        "packages", "dependency_lock_hash", "gpu_name", "cuda_version",
        "cudnn_version", "precision", "allow_tf32", "allow_tf32_matmul",
        "allow_tf32_cudnn",
    ):
        assert key in p, f"필수 키 누락: {key}"


def test_seeds_present():
    p = collect_provenance()
    assert set(p["seeds"].keys()) == {
        "split_seed", "episode_seed", "generator_seed", "model_seed", "stream_order_seed",
    }


def test_dependency_lock_hash_present():
    """collect_provenance()에 dependency_lock_hash 키가 포함되어야 함."""
    p = collect_provenance()
    assert "dependency_lock_hash" in p, "dependency_lock_hash 키 누락"


def test_dependency_lock_hash_matches_lock_file():
    """dependency_lock_hash가 locks/phase0_requirements.lock의 SHA-256과 일치해야 함."""
    p = collect_provenance()
    lock_path = Path("locks/phase0_requirements.lock")
    expected = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert p["dependency_lock_hash"] == expected, (
        f"dependency_lock_hash 불일치: provenance={p['dependency_lock_hash']}, "
        f"lock_file={expected}"
    )


def test_t0_gitignore_excludes_agent_files():
    """OpenCode 에이전트 파일이 Git에 추적되지 않음 (.gitignore로만 무시됨)."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, check=True,
    )
    tracked = result.stdout.splitlines()
    for f in [".opencode/", "AGENTS.md", "opencode.jsonc"]:
        assert f not in tracked, f"{f}가 Git에 추적되고 있음"
