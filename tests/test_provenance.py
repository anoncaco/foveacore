"""utils/provenance.py 환경 정보 수집 테스트 (T2)."""

import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

from foveacore.utils import provenance, seed as seed_utils
from foveacore.utils.provenance import collect_provenance, get_git_commit, is_git_dirty
from foveacore.utils.seed import DEFAULT_SEEDS, apply_seeds


@pytest.fixture(autouse=True)
def operational_runtime():
    seed_utils.initialize_gpu_runtime()
    yield


def test_collect_provenance_rejects_reversed_order_before_torch_query(monkeypatch):
    monkeypatch.setattr(seed_utils, "_GPU_RUNTIME_INITIALIZED", False)

    def unexpected_torch_query():
        raise AssertionError("운영 초기화 전에 Torch/CUDA 조회가 발생했습니다.")

    monkeypatch.setattr(provenance, "_torch", unexpected_torch_query)
    with pytest.raises(RuntimeError, match="운영 초기화"):
        collect_provenance()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_gpu_operational_order_in_fresh_process(run_fresh_python):
    normal = run_fresh_python(
        """
        import os
        import torch
        from foveacore.utils.provenance import collect_provenance
        from foveacore.utils.seed import initialize_gpu_runtime

        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert not torch.cuda.is_initialized()
        initialize_gpu_runtime()
        manifest = collect_provenance()
        assert manifest["cublas_workspace_config"] == ":4096:8"
        assert manifest["deterministic_mode"] is True
        assert manifest["cuda_available"] is True
        assert manifest["gpu_name"]
        assert torch.cuda.is_initialized()
        """
    )
    assert normal.returncode == 0, normal.stdout + normal.stderr

    reversed_order = run_fresh_python(
        """
        import os
        import torch
        from foveacore.utils.provenance import collect_provenance

        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert not torch.cuda.is_initialized()
        try:
            collect_provenance()
        except RuntimeError as exc:
            assert "운영 초기화" in str(exc), exc
        else:
            raise AssertionError("순서가 뒤집힌 provenance 수집이 성공했습니다.")
        assert not torch.cuda.is_initialized()
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        """
    )
    assert reversed_order.returncode == 0, reversed_order.stdout + reversed_order.stderr


def test_collect_provenance_required_keys():
    p = collect_provenance()
    for key in (
        "schema_version", "os", "hostname", "cpu", "cpu_count", "torch_version", "compute_device",
        "seeds", "deterministic_mode", "cudnn_deterministic", "cudnn_benchmark",
        "cublas_workspace_config", "torch_num_threads", "torch_interop_threads",
        "git_commit", "git_dirty", "packages", "dependency_lock_hash", "gpu_name",
        "cuda_version", "cudnn_version", "precision", "allow_tf32",
        "allow_tf32_matmul", "allow_tf32_cudnn",
    ):
        assert key in p, f"필수 키 누락: {key}"


def test_seeds_present():
    p = collect_provenance()
    assert set(p["seeds"].keys()) == {
        "split_seed", "episode_seed", "generator_seed", "model_seed", "stream_order_seed",
    }


def test_collect_provenance_reads_last_applied_seeds():
    expected = dict(DEFAULT_SEEDS, model_seed=88)
    apply_seeds({"model_seed": 88})
    assert collect_provenance()["seeds"] == expected
    assert collect_provenance({"model_seed": 88})["seeds"] == expected


@pytest.mark.parametrize(
    ("seeds", "error"),
    [
        ({"unknown_seed": 1}, ValueError),
        ({"model_seed": "88"}, TypeError),
        ({"model_seed": 89}, ValueError),
        ([1, 2], TypeError),
    ],
)
def test_collect_provenance_rejects_non_applied_seeds(seeds, error):
    apply_seeds({"model_seed": 88})
    with pytest.raises(error):
        collect_provenance(seeds)


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


def test_git_commit_from_subdirectory(monkeypatch):
    """Git 해시 조회도 실행 디렉터리와 무관해야 한다."""
    project_root = Path(__file__).resolve().parents[1]
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    monkeypatch.chdir(Path(__file__).resolve().parent)
    assert get_git_commit() == expected


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


def test_git_dirty_detects_untracked_when_status_config_disabled(git_repo):
    """사용자 설정 status.showUntrackedFiles=no에서도 미추적 파일을 감지해야 한다."""
    subprocess.run(
        ["git", "config", "status.showUntrackedFiles", "no"],
        cwd=git_repo,
        check=True,
    )
    (git_repo / "configs" / "untracked_probe.yaml").write_text("probe: 1\n", encoding="utf-8")
    assert is_git_dirty() is True


def test_git_dirty_returns_none_outside_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "src" / "foveacore" / "utils" / "provenance.py"))
    assert is_git_dirty() is None
