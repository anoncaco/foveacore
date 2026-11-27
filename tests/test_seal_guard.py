"""SealGuard 해시·기준점·record 기초 계약 테스트."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from foveacore.data.schema import EvaluationGrant, FreezeManifest
from foveacore.data.seal_guard import (
    IndependentBaseline,
    SealGuard,
    SealGuardError,
    VerificationRecord,
    canonical_json,
    compute_freeze_manifest_hash,
    load_freeze_manifest,
    load_sidecar_hash,
    verify_git_reference,
)


@dataclass
class FreezeFixture:
    manifest_path: Path
    sidecar_path: Path
    artifact_paths: dict[str, Path]
    baseline: IndependentBaseline
    grant: EvaluationGrant
    guard: SealGuard


def _build_freeze_fixture(tmp_path: Path) -> FreezeFixture:
    artifact_paths = {
        "data_split_hash": tmp_path / "split.json",
        "config_hash": tmp_path / "config.json",
        "checkpoint_hash": tmp_path / "checkpoint.bin",
    }
    contents = {
        "data_split_hash": b'{"split":"official_test"}',
        "config_hash": b'{"seed":1001}',
        "checkpoint_hash": b"checkpoint-bytes",
    }
    artifact_hashes = {}
    for field_name, path in artifact_paths.items():
        path.write_bytes(contents[field_name])
        artifact_hashes[field_name] = hashlib.sha256(contents[field_name]).hexdigest()

    body = {"schema_version": "1.0", **artifact_hashes}
    manifest_hash = compute_freeze_manifest_hash(body)
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps({**body, "manifest_sha256": manifest_hash}),
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "freeze_manifest.sha256"
    sidecar_path.write_text(manifest_hash + "\n", encoding="utf-8")
    baseline = IndependentBaseline(manifest_sha256=manifest_hash)
    grant = EvaluationGrant(
        evaluation_id="evaluation-1",
        phase=6,
        approved_splits=frozenset({"official_test"}),
    )
    guard = SealGuard({manifest_path: baseline})
    return FreezeFixture(
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
        artifact_paths=artifact_paths,
        baseline=baseline,
        grant=grant,
        guard=guard,
    )


def test_canonical_json_is_deterministic_and_rejects_nan():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(SealGuardError):
        canonical_json({"value": float("nan")})


def test_canonical_json_handles_nested_frozen_metadata():
    body = {
        "schema_version": "1.0",
        "data_split_hash": "a" * 64,
        "config_hash": "b" * 64,
        "checkpoint_hash": "c" * 64,
        "manifest_sha256": "d" * 64,
        "metadata": {"nested": {"b": 2, "a": 1}, "items": [{"z": 0}]},
    }
    manifest = FreezeManifest.from_dict(body)
    assert canonical_json(manifest.to_dict()) == (
        '{"checkpoint_hash":"'
        + "c" * 64
        + '","config_hash":"'
        + "b" * 64
        + '","data_split_hash":"'
        + "a" * 64
        + '","manifest_sha256":"'
        + "d" * 64
        + '","metadata":{"items":[{"z":0}],"nested":{"a":1,"b":2}},'
        + '"schema_version":"1.0"}'
    )


def test_compute_hash_excludes_self_and_accepts_freeze_manifest():
    body = {
        "schema_version": "1.0",
        "data_split_hash": "a" * 64,
        "config_hash": "b" * 64,
        "checkpoint_hash": "c" * 64,
    }
    first = compute_freeze_manifest_hash({**body, "manifest_sha256": "1" * 64})
    second = compute_freeze_manifest_hash({**body, "manifest_sha256": "2" * 64})
    assert first == second
    assert first == compute_freeze_manifest_hash(
        FreezeManifest.from_dict({**body, "manifest_sha256": first})
    )


def test_load_sidecar_hash_accepts_only_single_sha256(tmp_path: Path):
    sidecar = tmp_path / "freeze.sha256"
    sidecar.write_text("a" * 64 + "\n", encoding="utf-8")
    assert load_sidecar_hash(sidecar) == "a" * 64
    for content in ("", "abc", "A" * 64, "a" * 64 + " extra"):
        sidecar.write_text(content, encoding="utf-8")
        with pytest.raises(SealGuardError):
            load_sidecar_hash(sidecar)


def test_load_freeze_manifest_rejects_malformed_json_and_non_dict(tmp_path: Path):
    path = tmp_path / "freeze.json"
    for content in ("not-json", "[]"):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(SealGuardError):
            load_freeze_manifest(path)


def test_independent_baseline_requires_complete_git_configuration(tmp_path: Path):
    with pytest.raises(SealGuardError):
        IndependentBaseline("a" * 64, git_project_root=tmp_path)
    with pytest.raises(SealGuardError):
        IndependentBaseline("not-a-hash")
    with pytest.raises(SealGuardError):
        IndependentBaseline(
            "a" * 64,
            git_project_root=tmp_path,
            git_reference="HEAD",
            expected_git_commit="not-a-commit",
        )


def test_verify_git_reference_fails_closed_without_git(tmp_path: Path):
    with pytest.raises(SealGuardError, match=".git"):
        verify_git_reference(tmp_path, "HEAD", "a" * 40)
    with pytest.raises(SealGuardError):
        verify_git_reference(tmp_path, "HEAD", 1)


def test_verification_record_is_immutable_and_bound_to_baseline(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        fixture.grant,
    )
    assert record.manifest_sha256 == fixture.baseline.manifest_sha256
    with pytest.raises(TypeError):
        record.artifact_paths["data_split_hash"] = Path("changed")

    other_baseline = IndependentBaseline(manifest_sha256="f" * 64)
    with pytest.raises(SealGuardError, match="기준점"):
        VerificationRecord(
            manifest_path=record.manifest_path,
            manifest_sha256=record.manifest_sha256,
            sidecar_path=record.sidecar_path,
            sidecar_sha256=record.sidecar_sha256,
            baseline=other_baseline,
            artifact_paths=record.artifact_paths,
            artifact_hashes=record.artifact_hashes,
            evaluation_id=record.evaluation_id,
            phase=record.phase,
            approved_splits=record.approved_splits,
        )


def test_verification_record_requires_all_artifact_fields(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        fixture.grant,
    )
    incomplete_paths = dict(record.artifact_paths)
    incomplete_paths.pop("config_hash")
    with pytest.raises(SealGuardError, match="세 artifact"):
        VerificationRecord(
            manifest_path=record.manifest_path,
            manifest_sha256=record.manifest_sha256,
            sidecar_path=record.sidecar_path,
            sidecar_sha256=record.sidecar_sha256,
            baseline=record.baseline,
            artifact_paths=incomplete_paths,
            artifact_hashes=record.artifact_hashes,
            evaluation_id=record.evaluation_id,
            phase=record.phase,
            approved_splits=record.approved_splits,
        )


def test_verify_freeze_manifest_returns_bound_record(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        fixture.grant,
    )
    assert record.manifest_path == fixture.manifest_path.resolve()
    assert record.sidecar_path == fixture.sidecar_path.resolve()
    assert record.manifest_sha256 == fixture.baseline.manifest_sha256
    assert record.evaluation_id == fixture.grant.evaluation_id
    assert record.phase == 6
    assert record.approved_splits == frozenset({"official_test"})
    assert set(record.artifact_hashes) == {
        "data_split_hash",
        "config_hash",
        "checkpoint_hash",
    }


@pytest.mark.parametrize(
    "field_name",
    ["data_split_hash", "config_hash", "checkpoint_hash"],
)
def test_verify_freeze_manifest_recomputes_each_real_artifact(tmp_path: Path, field_name: str):
    fixture = _build_freeze_fixture(tmp_path)
    fixture.artifact_paths[field_name].write_bytes(b"changed-after-freeze")
    with pytest.raises(SealGuardError, match="artifact 해시 불일치"):
        fixture.guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_requires_all_artifact_paths(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    incomplete = dict(fixture.artifact_paths)
    incomplete.pop("checkpoint_hash")
    with pytest.raises(SealGuardError, match="세 artifact"):
        fixture.guard.verify(
            fixture.manifest_path,
            incomplete,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_rejects_sidecar_mismatch(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    fixture.sidecar_path.write_text("f" * 64, encoding="utf-8")
    with pytest.raises(SealGuardError, match="sidecar 해시 불일치"):
        fixture.guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_rejects_independent_baseline_mismatch(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    other_baseline = IndependentBaseline(manifest_sha256="f" * 64)
    other_guard = SealGuard({fixture.manifest_path: other_baseline})
    with pytest.raises(SealGuardError, match="독립 manifest 기준점"):
        other_guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_rejects_internal_hash_mismatch(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = "f" * 64
    fixture.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SealGuardError, match="내재 해시 불일치"):
        fixture.guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_rejects_rewritten_manifest_and_sidecar(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    new_hash = hashlib.sha256(b"new-artifact").hexdigest()
    body = {
        "schema_version": "1.0",
        "data_split_hash": new_hash,
        "config_hash": new_hash,
        "checkpoint_hash": new_hash,
    }
    new_manifest_hash = compute_freeze_manifest_hash(body)
    fixture.manifest_path.write_text(
        json.dumps({**body, "manifest_sha256": new_manifest_hash}),
        encoding="utf-8",
    )
    fixture.sidecar_path.write_text(new_manifest_hash, encoding="utf-8")
    for path in fixture.artifact_paths.values():
        path.write_bytes(b"new-artifact")
    with pytest.raises(SealGuardError, match="독립 manifest 기준점"):
        fixture.guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def test_verify_freeze_manifest_rejects_wrong_phase_and_unregistered_baseline(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    wrong_phase = EvaluationGrant(
        evaluation_id="evaluation-early",
        phase=5,
        approved_splits=frozenset({"official_test"}),
    )
    with pytest.raises(SealGuardError, match="Phase 6"):
        fixture.guard.verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            wrong_phase,
        )
    with pytest.raises(SealGuardError, match="사전등록"):
        SealGuard({}).verify(
            fixture.manifest_path,
            fixture.artifact_paths,
            fixture.sidecar_path,
            fixture.grant,
        )


def _verified_record(fixture: FreezeFixture):
    return fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        fixture.grant,
    )


def _assert_access(
    fixture: FreezeFixture,
    record: VerificationRecord,
    *,
    phase=6,
    execution_mode="EVALUATE",
    split_name="official_test",
    grant=None,
):
    fixture.guard.assert_access(
        phase=phase,
        execution_mode=execution_mode,
        split_name=split_name,
        evaluation_grant=fixture.grant if grant is None else grant,
        verification_record=record,
    )


def test_assert_access_allows_verified_official_evaluation(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    _assert_access(fixture, _verified_record(fixture))


@pytest.mark.parametrize("phase", [0, 5])
def test_assert_access_blocks_phase_before_six(tmp_path: Path, phase: int):
    fixture = _build_freeze_fixture(tmp_path)
    with pytest.raises(SealGuardError, match="Phase 6"):
        _assert_access(
            fixture,
            _verified_record(fixture),
            phase=phase,
        )


@pytest.mark.parametrize("execution_mode", ["ADAPT", "INFER"])
def test_assert_access_blocks_non_evaluate_modes(tmp_path: Path, execution_mode: str):
    fixture = _build_freeze_fixture(tmp_path)
    with pytest.raises(SealGuardError, match="EVALUATE"):
        _assert_access(
            fixture,
            _verified_record(fixture),
            execution_mode=execution_mode,
        )


def test_assert_access_blocks_reserved_in_first_official_evaluation(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    grant = EvaluationGrant(
        evaluation_id="first-official",
        phase=6,
        approved_splits=frozenset({"reserved_sealed"}),
    )
    record = fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        grant,
    )
    with pytest.raises(SealGuardError, match="첫 공식 평가"):
        _assert_access(
            fixture,
            record,
            split_name="reserved_sealed",
            grant=grant,
        )


def test_assert_access_allows_reserved_only_with_phase_seven_grant(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    grant = EvaluationGrant(
        evaluation_id="followup-evaluation",
        phase=7,
        approved_splits=frozenset({"reserved_sealed"}),
    )
    record = fixture.guard.verify(
        fixture.manifest_path,
        fixture.artifact_paths,
        fixture.sidecar_path,
        grant,
    )
    _assert_access(
        fixture,
        record,
        phase=7,
        split_name="reserved_sealed",
        grant=grant,
    )


def test_assert_access_blocks_unapproved_ood_split(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    with pytest.raises(SealGuardError, match="미승인"):
        _assert_access(
            fixture,
            _verified_record(fixture),
            phase=7,
            split_name="dev_ood",
            grant=EvaluationGrant(
                evaluation_id="ood-evaluation",
                phase=7,
                approved_splits=frozenset({"official_test"}),
            ),
        )


def test_assert_access_rejects_record_from_different_evaluation(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    other_grant = EvaluationGrant(
        evaluation_id="other-evaluation",
        phase=6,
        approved_splits=frozenset({"official_test"}),
    )
    with pytest.raises(SealGuardError, match="평가 승인 정보"):
        _assert_access(fixture, record, grant=other_grant)


def test_assert_access_rejects_record_with_different_approved_splits(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    other_grant = EvaluationGrant(
        evaluation_id="evaluation-1",
        phase=6,
        approved_splits=frozenset({"official_test", "dev_ood"}),
    )
    with pytest.raises(SealGuardError, match="평가 승인 정보"):
        _assert_access(fixture, record, grant=other_grant)


def test_assert_access_reverifies_artifact_after_record_creation(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    fixture.artifact_paths["checkpoint_hash"].write_bytes(b"replaced")
    with pytest.raises(SealGuardError, match="artifact 해시 불일치"):
        _assert_access(fixture, record)


def test_assert_access_reverifies_sidecar_after_record_creation(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    fixture.sidecar_path.write_text("f" * 64, encoding="utf-8")
    with pytest.raises(SealGuardError, match="sidecar 해시 불일치"):
        _assert_access(fixture, record)


def test_assert_access_rejects_record_from_untrusted_baseline(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    other_guard = SealGuard({})
    with pytest.raises(SealGuardError, match="기준점"):
        other_guard.assert_access(
            phase=6,
            execution_mode="EVALUATE",
            split_name="official_test",
            evaluation_grant=fixture.grant,
            verification_record=record,
        )


def test_assert_access_has_no_freeze_verified_bypass(tmp_path: Path):
    fixture = _build_freeze_fixture(tmp_path)
    with pytest.raises(TypeError):
        fixture.guard.assert_access(
            phase=6,
            execution_mode="EVALUATE",
            split_name="official_test",
            evaluation_grant=fixture.grant,
            verification_record=_verified_record(fixture),
            freeze_verified=True,
        )


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _build_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "git-repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "seal-test@example.com")
    _run_git(repo, "config", "user.name", "Seal Test")
    (repo / "tracked.txt").write_text("initial", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-m", "initial")
    return repo, _run_git(repo, "rev-parse", "HEAD")


def _build_git_freeze_fixture(tmp_path: Path) -> FreezeFixture:
    fixture = _build_freeze_fixture(tmp_path)
    repo, commit = _build_git_repo(tmp_path)
    baseline = IndependentBaseline(
        manifest_sha256=fixture.baseline.manifest_sha256,
        git_project_root=repo,
        git_reference="HEAD",
        expected_git_commit=commit,
    )
    fixture.baseline = baseline
    fixture.guard = SealGuard({fixture.manifest_path: baseline})
    return fixture


def test_git_baseline_verifies_and_allows_access(tmp_path: Path):
    fixture = _build_git_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    assert record.git_commit == fixture.baseline.expected_git_commit
    _assert_access(fixture, record)


def test_git_tag_reference_is_accepted(tmp_path: Path):
    fixture = _build_git_freeze_fixture(tmp_path)
    repo = fixture.baseline.git_project_root
    commit = fixture.baseline.expected_git_commit
    _run_git(repo, "tag", "seal-v1", commit)
    baseline = IndependentBaseline(
        manifest_sha256=fixture.baseline.manifest_sha256,
        git_project_root=repo,
        git_reference="seal-v1",
        expected_git_commit=commit,
    )
    fixture.baseline = baseline
    fixture.guard = SealGuard({fixture.manifest_path: baseline})
    _assert_access(fixture, _verified_record(fixture))


def test_git_head_movement_breaks_fixed_baseline(tmp_path: Path):
    fixture = _build_git_freeze_fixture(tmp_path)
    record = _verified_record(fixture)
    repo = fixture.baseline.git_project_root
    (repo / "tracked.txt").write_text("second", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-m", "second")
    with pytest.raises(SealGuardError, match="Git commit 기준점 불일치"):
        _assert_access(fixture, record)


def test_git_query_failure_is_blocked(tmp_path: Path):
    fixture = _build_git_freeze_fixture(tmp_path)
    baseline = IndependentBaseline(
        manifest_sha256=fixture.baseline.manifest_sha256,
        git_project_root=fixture.baseline.git_project_root,
        git_reference="missing-reference",
        expected_git_commit=fixture.baseline.expected_git_commit,
    )
    fixture.baseline = baseline
    fixture.guard = SealGuard({fixture.manifest_path: baseline})
    with pytest.raises(SealGuardError, match="Git 기준점 조회 실패"):
        _verified_record(fixture)


def test_git_root_mismatch_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _build_git_freeze_fixture(tmp_path)
    real_run = subprocess.run

    def run_with_wrong_root(arguments, *args, **kwargs):
        if "--show-toplevel" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=str(tmp_path / "wrong-root"),
                stderr="",
            )
        return real_run(arguments, *args, **kwargs)

    monkeypatch.setattr(
        "foveacore.data.seal_guard.subprocess.run",
        run_with_wrong_root,
    )
    with pytest.raises(SealGuardError, match="Git root 불일치"):
        _verified_record(fixture)
