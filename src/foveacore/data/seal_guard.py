"""FoveaCore Phase 0 SealGuard (T3).

freeze_manifest.json 해시 검증 기준점.

자기참조 해시 모순을 막기 위해:
  1. normalize: freeze_manifest.json 내 manifest_sha256 자체 필드 제외 후
     키 정렬·공백 제거된 Canonical JSON으로 직렬화해 SHA-256 계산
  2. 외부 기준점(freeze_manifest.sha256 sidecar, Git 태그/커밋, 사전등록 해시) 대조
  3. 내부 아티팩트 교차 검증(data_split_hash, config_hash, checkpoint_hash)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set

# 봉인 필수 필드 (T3-08)
REQUIRED_FREEZE_FIELDS = {
    "schema_version",
    "data_split_hash",
    "config_hash",
    "checkpoint_hash",
    "manifest_sha256",
}

# Phase 0~5에서 접근 금지(봉인) 분할 (T3-07/09)
SEALED_SPLITS = {"official_test", "reserved_sealed", "dev_ood", "test_ood", "phase7_char_transfer"}


class SealGuardError(ValueError):
    """freeze_manifest 해시 검증 실패."""


def canonical_json(obj: Any) -> str:
    """키 정렬·공백 제거된 Canonical JSON 직렬화."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_freeze_manifest_hash(manifest: Dict[str, Any]) -> str:
    """manifest_sha256 자체 필드를 제외한 본문의 SHA-256."""
    filtered = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(filtered).encode("utf-8")).hexdigest()


def load_freeze_manifest(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def verify_integrity(
    manifest: Dict[str, Any],
    sidecar_hash: str | None = None,
    artifact_hashes: Dict[str, str] | None = None,
) -> None:
    """freeze_manifest 무결성 검증.

    - 내재 해시(manifest_sha256)가 계산값과 일치하는지
    - sidecar_hash(독립 기준점)가 주어지면 대조
    - artifact_hashes(실제 파일 SHA-256)가 주어지면 freeze_manifest 내 기록값과 교차 검증
    """
    # 필수 필드 검증 (T3-08)
    for field in REQUIRED_FREEZE_FIELDS:
        if manifest.get(field) is None:
            raise SealGuardError(f"freeze_manifest 필수 필드 누락: {field}")

    stored_hash = manifest.get("manifest_sha256")
    computed_hash = compute_freeze_manifest_hash(manifest)

    if stored_hash != computed_hash:
        raise SealGuardError(
            f"freeze_manifest 내재 해시 불일치: stored={stored_hash}, computed={computed_hash}"
        )

    # 독립 기준점 필수 (T3-11)
    if sidecar_hash is None:
        raise SealGuardError("독립 기준점(sidecar) 해시가 필요합니다(T3-11).")
    if sidecar_hash != computed_hash:
        raise SealGuardError(
            f"freeze_manifest 외부 기준점 해시 불일치: sidecar={sidecar_hash}, computed={computed_hash}"
        )

    # 아티팩트 해시 필수 (T3-11)
    if not artifact_hashes:
        raise SealGuardError("아티팩트 해시가 필요합니다(T3-11).")
    for key, expected in artifact_hashes.items():
        actual = manifest.get(key)
        if actual is None:
            raise SealGuardError(f"freeze_manifest 내 {key} 누락")
        if actual != expected:
            raise SealGuardError(
                f"freeze_manifest {key} 교차 검증 실패: manifest={actual}, actual={expected}"
            )


def validate_sealed_policy(manifests: Dict[str, Any]) -> List[str]:
    """split별 봉인 등급 정책 강제 (T3-09). SEALED_SPLITS는 SEALED여야 함."""
    errors: List[str] = []
    for name in SEALED_SPLITS:
        m = manifests.get(name)
        if m is None:
            continue
        level = m.get("sealed_level") if isinstance(m, dict) else None
        if isinstance(m, dict) and "entries" in m:
            level = m.get("sealed_level")
        # manifest may be per-split dict with sealed_level
        if level is None and isinstance(m, dict):
            level = m.get("sealed_level")
        if level not in (None,) and level != "SEALED":
            errors.append(f"봉인 정책 위반: {name}는 SEALED여야 합니다 (현재 {level})")
        # entries sealed_level check
        for e in (m.get("entries", []) if isinstance(m, dict) else []):
            if e.get("sealed_level") not in (None, "SEALED"):
                errors.append(f"봉인 정책 위반: {name} entry {e.get('sample_id')} not SEALED")
    return errors


class SealGuard:
    """Phase·모드 기반 봉인 접근 제어 (T3-07)."""

    # Phase 0~5에서 봉인 데이터 접근 금지
    PHASE_SEALED_UNTIL = 6

    def check_access(self, phase: int, execution_mode: str, split_name: str,
                     freeze_verified: bool = False) -> bool:
        # ADAPT는 봉인 분할 항상 차단 (T3-10)
        if execution_mode == "ADAPT" and split_name in SEALED_SPLITS:
            return False
        if execution_mode == "INFER" and split_name in SEALED_SPLITS:
            return False
        # Phase 6+ 공식 평가: freeze manifest 검증 필수 (T3-09)
        if phase >= self.PHASE_SEALED_UNTIL and split_name in SEALED_SPLITS:
            if execution_mode not in ("EVALUATE",):
                return False
            if not freeze_verified:
                return False
        if split_name in SEALED_SPLITS and phase < self.PHASE_SEALED_UNTIL:
            return False
        return True

    def assert_access(self, phase: int, execution_mode: str, split_name: str,
                      freeze_verified: bool = False) -> None:
        if not self.check_access(phase, execution_mode, split_name, freeze_verified):
            raise SealGuardError(
                f"봉인 접근 차단: phase={phase}, mode={execution_mode}, split={split_name}"
            )
