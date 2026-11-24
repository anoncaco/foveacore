"""utils/provenance.py 환경 정보 수집 테스트 (T2)."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from foveacore.utils import provenance
from foveacore.utils.provenance import collect_provenance, is_git_dirty


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
    project_root = Path(__file__).resolve().parents[1]
    lock_path = project_root / "locks" / "phase0_requirements.lock"
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


def test_dependency_lock_hash_from_subdirectory():
    """하위 디렉터리에서도 프로젝트 루트의 lock 파일을 읽어야 함."""
    import os

    project_root = Path(__file__).resolve().parents[1]
    lock_path = project_root / "locks" / "phase0_requirements.lock"
    expected = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    original_cwd = Path.cwd()
    tests_dir = Path(__file__).resolve().parent
    try:
        os.chdir(tests_dir)
        p = collect_provenance()
        assert p["dependency_lock_hash"] == expected, (
            f"하위 디렉터리에서 hash 불일치: "
            f"provenance={p['dependency_lock_hash']}, lock_file={expected}"
        )
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """격리된 임시 Git 저장소와 프로젝트 루트 경로를 제공한다."""
    root = tmp_path / "repo"
    for directory in ("src", "configs", "locks", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'probe'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=T1 Test", "-c", "user.email=t1@example.invalid",
            "commit", "-qm", "fixture",
        ],
        cwd=root,
        check=True,
    )
    monkeypatch.setattr(provenance, "__file__", str(root / "src" / "foveacore" / "utils" / "provenance.py"))
    return root


@pytest.mark.parametrize("path", ["src/probe.py", "configs/probe.yaml", "locks/probe.lock", "pyproject.toml"])
def test_git_dirty_detects_contract_paths(git_repo, path):
    target = git_repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n", encoding="utf-8")
    assert is_git_dirty() is True


def test_git_dirty_ignores_unrelated_paths(git_repo):
    (git_repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    (git_repo / "tests" / "probe_test.py").write_text("unrelated\n", encoding="utf-8")
    assert is_git_dirty() is False


def test_git_dirty_from_subdirectory(git_repo, monkeypatch):
    (git_repo / "src" / "untracked_probe.py").write_text("untracked\n", encoding="utf-8")
    monkeypatch.chdir(git_repo / "tests")
    assert is_git_dirty() is True


def test_git_dirty_clean_repository(git_repo):
    assert is_git_dirty() is False


def test_git_dirty_returns_none_outside_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "src" / "foveacore" / "utils" / "provenance.py"))
    assert is_git_dirty() is None
