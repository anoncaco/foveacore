"""FoveaCore Phase 0 정답 저장소(ground_truth_vault.json) + ADAPT 접근 제어 (T3).

split_manifest_v1.json: 라벨 미포함(sample_id, split_name, sha256, phash, sealed_level)
ground_truth_vault.json: sample_id -> label 격리 저장소
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

EXECUTION_MODES = {"INFER", "ADAPT", "EVALUATE"}

# ADAPT가 접근 가능한 승인 학습 분할 (v1.9.3 Phase 0)
ADAPT_ALLOWED_SPLITS = {"support_pool", "dev_stream", "main_stream"}

# 라벨 조회가 무조건 차단되는 분할
ADAPT_BLOCKED_SPLITS = {"official_test", "reserved_sealed", "test_ood"}


class GroundTruthVault:
    """sample_id -> label 격리 저장소. 라벨은 분할 매니페스트와 분리 보관."""

    def __init__(self, vault: Dict[str, str] | None = None):
        self._vault: Dict[str, str] = dict(vault or {})

    @classmethod
    def load(cls, path: Path | str) -> "GroundTruthVault":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data.get("labels", data))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"labels": self._vault}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get(self, sample_id: str) -> str | None:
        return self._vault.get(sample_id)

    def set(self, sample_id: str, label: str) -> None:
        self._vault[sample_id] = label

    def contains(self, sample_id: str) -> bool:
        return sample_id in self._vault

    def _labels(self) -> Dict[str, str]:
        return dict(self._vault)


class AdaptAccessGuard:
    """(ExecutionMode × split_name) 기반 ADAPT 라벨 접근 제어(T3).

    - INFER: 전면 차단(라벨 조회 불가)
    - ADAPT: 승인된 학습 분할(support_pool/dev_stream/main_stream)만 허용,
      official_test/reserved_sealed/test_ood 및 미확정 query_ids 라벨 조회 차단
    """

    def __init__(self, vault: GroundTruthVault):
        self._vault = vault
        # split_name -> set of sample_ids (T3-10 ID 소속 검증)
        self._split_members: Dict[str, Set[str]] = {}

    def register_split(self, split_name: str, sample_ids: List[str]) -> None:
        """split에 속한 sample_id 집합 등록(T3-10)."""
        self._split_members[split_name] = set(sample_ids)

    def _is_member(self, split_name: str, sid: str) -> bool:
        members = self._split_members.get(split_name)
        if members is None:
            return False  # 미등록 split은 기본 거부 (T3-07)
        return sid in members

    def may_access_label(
        self,
        execution_mode: str,
        split_name: str,
        sample_ids: List[str],
    ) -> Dict[str, str | None]:
        """요청 샘플에 대해 라벨 조회 가능 여부를 dict로 반환(None = 차단)."""
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"잘못된 ExecutionMode: {execution_mode}")

        result: Dict[str, str | None] = {}

        if execution_mode == "INFER":
            # INFER는 라벨 조회 전면 차단
            for sid in sample_ids:
                result[sid] = None
            return result

        if execution_mode == "EVALUATE":
            # EVALUATE는 봉인되지 않은 분할에서만 라벨 조회 허용(정책상)
            if split_name in ADAPT_BLOCKED_SPLITS:
                for sid in sample_ids:
                    result[sid] = None
                return result
            for sid in sample_ids:
                if not self._is_member(split_name, sid):
                    result[sid] = None
                else:
                    result[sid] = self._vault._get(sid)
            return result

        # ADAPT
        if split_name not in ADAPT_ALLOWED_SPLITS:
            for sid in sample_ids:
                result[sid] = None
            return result
        for sid in sample_ids:
            if not self._is_member(split_name, sid):
                result[sid] = None
            else:
                result[sid] = self._vault._get(sid)
        return result

    def assert_no_blocked_access(
        self,
        execution_mode: str,
        split_name: str,
        sample_ids: List[str],
    ) -> None:
        """차단 대상 분할/모드에서 라벨 조회를 시도하면 RuntimeError."""
        if not sample_ids:
            return
        if execution_mode == "INFER":
            raise RuntimeError(
                f"INFER 라벨 접근 차단 위반: mode={execution_mode}, split={split_name}, "
                f"samples={sample_ids}"
            )
        if execution_mode == "ADAPT" and split_name not in ADAPT_ALLOWED_SPLITS:
            raise RuntimeError(
                f"ADAPT 라벨 접근 차단 위반: mode={execution_mode}, split={split_name}, "
                f"samples={sample_ids}"
            )
        if execution_mode == "EVALUATE" and split_name in ADAPT_BLOCKED_SPLITS:
            raise RuntimeError(
                f"EVALUATE 라벨 접근 차단 위반: mode={execution_mode}, split={split_name}, "
                f"samples={sample_ids}"
            )
        # ID 소속 검증(T3-10)
        for sid in sample_ids:
            if not self._is_member(split_name, sid):
                raise RuntimeError(
                    f"샘플 ID 소속 위반: {sid} not in {split_name}"
                )
