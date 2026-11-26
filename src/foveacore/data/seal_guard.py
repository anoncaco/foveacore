"""FoveaCore Phase 0 freeze manifest·봉인 접근 검증 (T3)."""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List

from foveacore.data.schema import (
    FREEZE_HASH_FIELDS,
    REQUIRED_FREEZE_FIELDS,
    SEALED_SPLITS,
    SPLIT_NAMES,
    EvaluationGrant,
    FreezeManifest,
    SchemaError,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SealGuardError(ValueError):
    """freeze manifest·artifact·봉인 접근 검증 실패."""


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise SealGuardError(f"{field_name}은 64자리 소문자 SHA-256이어야 합니다.")
    return value


def canonical_json(obj: Any) -> str:
    """키 정렬·공백 제거·NaN 금지 Canonical JSON."""
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SealGuardError("Canonical JSON 직렬화 실패") from exc


def _manifest_dict(manifest: Mapping[str, Any] | FreezeManifest) -> Dict[str, Any]:
    if isinstance(manifest, FreezeManifest):
        return manifest.to_dict()
    if isinstance(manifest, Mapping):
        return dict(manifest)
    raise SealGuardError("manifest는 mapping 또는 FreezeManifest이어야 합니다.")


def compute_freeze_manifest_hash(manifest: Mapping[str, Any] | FreezeManifest) -> str:
    """manifest_sha256 자신을 제외한 본문의 SHA-256."""
    body = {
        key: value
        for key, value in _manifest_dict(manifest).items()
        if key != "manifest_sha256"
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def load_freeze_manifest(path: Path | str) -> Dict[str, Any]:
    """freeze manifest JSON을 dict로 읽습니다."""
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SealGuardError(f"freeze manifest 로딩 실패: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise SealGuardError("freeze manifest 최상위는 dict여야 합니다.")
    return data


def load_sidecar_hash(path: Path | str) -> str:
    """단일 SHA-256 문자열 sidecar를 읽고 검증합니다."""
    sidecar_path = Path(path)
    try:
        value = sidecar_path.read_text(encoding="utf-8").strip()
    except (UnicodeDecodeError, OSError) as exc:
        raise SealGuardError(f"sidecar 로딩 실패: {sidecar_path}") from exc
    return _require_sha256(value, "sidecar_hash")


def _sha256_file(path: Path | str) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SealGuardError(f"artifact 해시 계산 실패: {file_path}") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IndependentBaseline:
    """manifest와 별도로 고정한 해시 및 선택적 Git 기준점."""

    manifest_sha256: str
    git_project_root: Path | None = None
    git_reference: str | None = None
    expected_git_commit: str | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(self.manifest_sha256, "baseline.manifest_sha256"),
        )
        git_values = (
            self.git_project_root,
            self.git_reference,
            self.expected_git_commit,
        )
        if any(value is not None for value in git_values) and not all(
            value is not None for value in git_values
        ):
            raise SealGuardError("Git 기준점은 root·reference·expected commit이 모두 필요합니다.")
        if self.git_project_root is not None:
            if not isinstance(self.git_project_root, (str, Path)):
                raise SealGuardError("Git project root는 경로여야 합니다.")
            try:
                root = Path(self.git_project_root).resolve(strict=True)
            except OSError as exc:
                raise SealGuardError("Git project root를 확인할 수 없습니다.") from exc
            if not root.is_dir():
                raise SealGuardError("Git project root는 디렉터리여야 합니다.")
            object.__setattr__(self, "git_project_root", root)
        if self.git_reference is not None and (
            not isinstance(self.git_reference, str)
            or not self.git_reference.strip()
            or self.git_reference.startswith("-")
        ):
            raise SealGuardError("git_reference는 유효한 문자열이어야 합니다.")
        if self.expected_git_commit is not None and (
            not isinstance(self.expected_git_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.expected_git_commit)
            is None
        ):
            raise SealGuardError("expected_git_commit은 40/64자리 commit 해시여야 합니다.")


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """검증된 manifest·artifact·기준점·평가 승인 정보의 불변 결합."""

    manifest_path: Path
    manifest_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    baseline: IndependentBaseline
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    evaluation_id: str
    phase: int
    approved_splits: frozenset[str]
    git_commit: str | None = None

    def __post_init__(self):
        for field_name in ("manifest_path", "sidecar_path"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)):
                raise SealGuardError(f"{field_name}는 경로여야 합니다.")
            object.__setattr__(self, field_name, Path(value).resolve())
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(self.manifest_sha256, "record.manifest_sha256"),
        )
        object.__setattr__(
            self,
            "sidecar_sha256",
            _require_sha256(self.sidecar_sha256, "record.sidecar_sha256"),
        )
        if not isinstance(self.baseline, IndependentBaseline):
            raise SealGuardError("record.baseline은 IndependentBaseline이어야 합니다.")
        if self.manifest_sha256 != self.baseline.manifest_sha256:
            raise SealGuardError("record manifest 해시가 독립 기준점과 다릅니다.")
        if self.sidecar_sha256 != self.manifest_sha256:
            raise SealGuardError("record sidecar 해시가 manifest 해시와 다릅니다.")
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise SealGuardError("evaluation_id는 비어 있지 않은 문자열이어야 합니다.")
        if isinstance(self.phase, bool) or not isinstance(self.phase, int) or self.phase not in (6, 7):
            raise SealGuardError("record phase는 6 또는 7이어야 합니다.")
        if not isinstance(self.approved_splits, frozenset) or any(
            not isinstance(value, str) for value in self.approved_splits
        ):
            raise SealGuardError("approved_splits는 문자열 frozenset이어야 합니다.")
        unknown = self.approved_splits.difference(SPLIT_NAMES)
        if unknown:
            raise SealGuardError(f"record에 알 수 없는 승인 split: {sorted(unknown)}")

        expected_keys = frozenset(FREEZE_HASH_FIELDS)
        if not isinstance(self.artifact_paths, Mapping) or frozenset(self.artifact_paths) != expected_keys:
            raise SealGuardError("artifact_paths는 세 artifact 해시 필드를 모두 포함해야 합니다.")
        if not isinstance(self.artifact_hashes, Mapping) or frozenset(self.artifact_hashes) != expected_keys:
            raise SealGuardError("artifact_hashes는 세 artifact 해시 필드를 모두 포함해야 합니다.")
        normalized_paths = {}
        normalized_hashes = {}
        for field_name in FREEZE_HASH_FIELDS:
            path = self.artifact_paths[field_name]
            if not isinstance(path, (str, Path)):
                raise SealGuardError(f"artifact_paths.{field_name}는 경로여야 합니다.")
            normalized_paths[field_name] = Path(path).resolve()
            normalized_hashes[field_name] = _require_sha256(
                self.artifact_hashes[field_name],
                f"artifact_hashes.{field_name}",
            )
        object.__setattr__(
            self,
            "artifact_paths",
            MappingProxyType({key: normalized_paths[key] for key in FREEZE_HASH_FIELDS}),
        )
        object.__setattr__(
            self,
            "artifact_hashes",
            MappingProxyType(
                {key: normalized_hashes[key] for key in FREEZE_HASH_FIELDS}
            ),
        )

        if self.baseline.git_project_root is None:
            if self.git_commit is not None:
                raise SealGuardError("Git 기준점이 없는데 git_commit이 설정됐습니다.")
        else:
            if self.git_commit != self.baseline.expected_git_commit:
                raise SealGuardError("record git_commit이 독립 Git 기준점과 다릅니다.")


def verify_git_reference(
    project_root: Path | str,
    reference: str,
    expected_commit: str,
) -> str:
    """명시한 Git root와 참조가 사전에 고정한 commit으로 해석되는지 검증합니다."""
    if not isinstance(reference, str) or not reference.strip() or reference.startswith("-"):
        raise SealGuardError("Git reference가 유효하지 않습니다.")
    if (
        not isinstance(expected_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected_commit) is None
    ):
        raise SealGuardError("expected_commit은 40/64자리 commit 해시여야 합니다.")
    if not isinstance(project_root, (str, Path)):
        raise SealGuardError("project_root는 경로여야 합니다.")
    root = Path(project_root).resolve()
    if not (root / ".git").exists():
        raise SealGuardError(f".git 없음: {root}")

    def run_git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SealGuardError("Git 기준점 조회 실패") from exc
        if completed.returncode != 0:
            raise SealGuardError(
                f"Git 기준점 조회 실패: {completed.stderr.strip() or completed.returncode}"
            )
        return completed.stdout.strip()

    actual_root = Path(run_git("rev-parse", "--show-toplevel")).resolve()
    if actual_root != root:
        raise SealGuardError(f"Git root 불일치: expected={root}, actual={actual_root}")
    commit = run_git("rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}")
    if commit != expected_commit:
        raise SealGuardError(f"Git commit 기준점 불일치: expected={expected_commit}, actual={commit}")
    return commit


def verify_freeze_manifest(
    manifest_path: Path | str,
    artifact_paths: Mapping[str, Path | str],
    sidecar_path: Path | str,
    baseline: IndependentBaseline,
    evaluation_grant: EvaluationGrant,
) -> VerificationRecord:
    """manifest·sidecar·독립 기준점·실제 artifact를 모두 검증합니다."""
    if not isinstance(baseline, IndependentBaseline):
        raise SealGuardError("baseline은 IndependentBaseline이어야 합니다.")
    if not isinstance(evaluation_grant, EvaluationGrant):
        raise SealGuardError("evaluation_grant는 EvaluationGrant이어야 합니다.")
    if evaluation_grant.phase not in (6, 7):
        raise SealGuardError("freeze 검증은 Phase 6~7 평가에서만 수행할 수 있습니다.")

    if not isinstance(manifest_path, (str, Path)):
        raise SealGuardError("manifest_path는 경로여야 합니다.")
    if not isinstance(sidecar_path, (str, Path)):
        raise SealGuardError("sidecar_path는 경로여야 합니다.")
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest_data = load_freeze_manifest(resolved_manifest_path)
    try:
        manifest = FreezeManifest.from_dict(manifest_data)
    except SchemaError as exc:
        raise SealGuardError(f"freeze manifest 스키마 오류: {exc}") from exc

    computed_hash = compute_freeze_manifest_hash(manifest)
    if manifest.manifest_sha256 != computed_hash:
        raise SealGuardError("freeze manifest 내재 해시 불일치")
    if baseline.manifest_sha256 != computed_hash:
        raise SealGuardError("독립 manifest 기준점 해시 불일치")

    resolved_sidecar_path = Path(sidecar_path).resolve()
    sidecar_hash = load_sidecar_hash(resolved_sidecar_path)
    if sidecar_hash != computed_hash:
        raise SealGuardError("sidecar 해시 불일치")

    git_commit = None
    if baseline.git_project_root is not None:
        git_commit = verify_git_reference(
            baseline.git_project_root,
            baseline.git_reference,
            baseline.expected_git_commit,
        )

    expected_keys = frozenset(FREEZE_HASH_FIELDS)
    if not isinstance(artifact_paths, Mapping) or frozenset(artifact_paths) != expected_keys:
        raise SealGuardError("artifact_paths는 세 artifact 파일을 모두 포함해야 합니다.")
    normalized_paths: Dict[str, Path] = {}
    actual_hashes: Dict[str, str] = {}
    for field_name in FREEZE_HASH_FIELDS:
        path = artifact_paths[field_name]
        if not isinstance(path, (str, Path)):
            raise SealGuardError(f"artifact_paths.{field_name}는 경로여야 합니다.")
        normalized_paths[field_name] = Path(path).resolve()
        actual_hash = _sha256_file(normalized_paths[field_name])
        expected_hash = _require_sha256(
            getattr(manifest, field_name),
            f"manifest.{field_name}",
        )
        if expected_hash != actual_hash:
            raise SealGuardError(f"실제 artifact 해시 불일치: {field_name}")
        actual_hashes[field_name] = actual_hash

    return VerificationRecord(
        manifest_path=resolved_manifest_path,
        manifest_sha256=computed_hash,
        sidecar_path=resolved_sidecar_path,
        sidecar_sha256=sidecar_hash,
        baseline=baseline,
        artifact_paths=normalized_paths,
        artifact_hashes=actual_hashes,
        evaluation_id=evaluation_grant.evaluation_id,
        phase=evaluation_grant.phase,
        approved_splits=evaluation_grant.approved_splits,
        git_commit=git_commit,
    )


def validate_sealed_policy(manifests: Dict[str, Any]) -> List[str]:
    """봉인 필수 split과 엔트리의 SEALED 등급 정책을 검사합니다."""
    errors: List[str] = []
    for name in sorted(SEALED_SPLITS):
        manifest = manifests.get(name)
        if manifest is None:
            continue
        if not isinstance(manifest, dict):
            errors.append(f"봉인 manifest 형식 오류: {name}")
            continue
        level = manifest.get("sealed_level")
        if level is not None and level != "SEALED":
            errors.append(f"봉인 정책 위반: {name}는 SEALED이어야 합니다 (현재 {level})")
        for entry in manifest.get("entries", []):
            if not isinstance(entry, dict):
                errors.append(f"봉인 entry 형식 오류: {name}")
                continue
            if entry.get("sealed_level") not in (None, "SEALED"):
                errors.append(
                    f"봉인 정책 위반: {name} entry "
                    f"{entry.get('sample_id')} not SEALED"
                )
    return errors


class SealGuard:
    """신뢰된 독립 기준점과 검증 record로 봉인 접근을 제어합니다."""

    def __init__(self, baselines: Mapping[str | Path, IndependentBaseline]):
        if not isinstance(baselines, Mapping):
            raise SealGuardError("baselines는 mapping이어야 합니다.")
        normalized: Dict[Path, IndependentBaseline] = {}
        for manifest_path, baseline in baselines.items():
            if not isinstance(manifest_path, (str, Path)):
                raise SealGuardError("baseline manifest 경로가 유효하지 않습니다.")
            if not isinstance(baseline, IndependentBaseline):
                raise SealGuardError("baseline 값은 IndependentBaseline이어야 합니다.")
            normalized[Path(manifest_path).resolve()] = baseline
        self._baselines = normalized

    def verify(
        self,
        manifest_path: Path | str,
        artifact_paths: Mapping[str, Path | str],
        sidecar_path: Path | str,
        evaluation_grant: EvaluationGrant,
    ) -> VerificationRecord:
        """이 guard에 고정된 기준점으로 freeze manifest를 검증합니다."""
        if not isinstance(manifest_path, (str, Path)):
            raise SealGuardError("manifest_path는 경로여야 합니다.")
        resolved_path = Path(manifest_path).resolve()
        baseline = self._baselines.get(resolved_path)
        if baseline is None:
            raise SealGuardError(f"사전등록되지 않은 manifest 기준점: {resolved_path}")
        return verify_freeze_manifest(
            resolved_path,
            artifact_paths,
            sidecar_path,
            baseline,
            evaluation_grant,
        )

    def assert_access(
        self,
        *,
        phase: int,
        execution_mode: str,
        split_name: str,
        evaluation_grant: EvaluationGrant | None,
        verification_record: VerificationRecord | None,
    ) -> None:
        """평가 승인과 record를 재검증한 뒤 읽기 전용 접근을 허용합니다."""
        if isinstance(phase, bool) or not isinstance(phase, int) or phase not in (6, 7):
            raise SealGuardError("봉인 접근은 Phase 6~7에서만 허용됩니다.")
        if execution_mode != "EVALUATE":
            raise SealGuardError("봉인 접근은 EVALUATE에서만 허용됩니다.")
        if split_name not in SPLIT_NAMES:
            raise SealGuardError(f"알 수 없는 split: {split_name}")
        if not isinstance(evaluation_grant, EvaluationGrant):
            raise SealGuardError("evaluation_grant가 필요합니다.")
        if evaluation_grant.phase != phase:
            raise SealGuardError("요청 phase와 평가 승인 phase가 다릅니다.")
        if split_name not in evaluation_grant.approved_splits:
            raise SealGuardError(f"평가 미승인 split: {split_name}")
        if phase == 6 and split_name != "official_test":
            raise SealGuardError("첫 공식 평가에서는 official_test만 열 수 있습니다.")
        if not isinstance(verification_record, VerificationRecord):
            raise SealGuardError("verification_record가 필요합니다.")
        if (
            verification_record.evaluation_id != evaluation_grant.evaluation_id
            or verification_record.phase != phase
            or verification_record.approved_splits
            != evaluation_grant.approved_splits
        ):
            raise SealGuardError("verification_record의 평가 승인 정보가 일치하지 않습니다.")

        baseline = self._baselines.get(verification_record.manifest_path)
        if baseline is None or verification_record.baseline != baseline:
            raise SealGuardError("verification_record의 독립 기준점이 신뢰 목록과 다릅니다.")

        current = verify_freeze_manifest(
            verification_record.manifest_path,
            verification_record.artifact_paths,
            verification_record.sidecar_path,
            baseline,
            evaluation_grant,
        )
        if current != verification_record:
            raise SealGuardError("접근 시 재검증 record가 기존 record와 다릅니다.")