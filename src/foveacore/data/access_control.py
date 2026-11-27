"""T3 라벨 접근 판정과 접근 후 라벨 해석."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict

from foveacore.data.ground_truth_vault import GroundTruthVault, GroundTruthVaultError
from foveacore.data.seal_guard import SealGuard, VerificationRecord

from foveacore.data.schema import (
    AccessDecision,
    ExecutionMode,
    ExecutionProtocol,
    LabelAccessRequest,
    SPLIT_NAMES,
)


ADAPT_STREAM_SPLITS = frozenset({"dev_stream", "main_stream"})


class LabelAccessError(RuntimeError):
    """라벨 접근 거부 또는 해석 실패."""


@dataclass(frozen=True, slots=True)
class AdaptScope:
    """현재 episode support와 승인된 stream ID만 보유하는 ADAPT 컨텍스트."""

    episode_id: str | None
    current_episode_support_ids: frozenset[str]
    approved_stream_ids: frozenset[str]

    def __post_init__(self):
        if self.episode_id is not None and (
            not isinstance(self.episode_id, str) or not self.episode_id.strip()
        ):
            raise LabelAccessError("episode_id는 비어 있지 않은 문자열이어야 합니다.")
        for field_name in (
            "current_episode_support_ids",
            "approved_stream_ids",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise LabelAccessError(f"{field_name}은 유효한 sample_id frozenset이어야 합니다.")

    def authorize(
        self,
        *,
        split_name: str,
        sample_ids: frozenset[str],
        protocol: str,
    ) -> bool:
        if (
            not isinstance(sample_ids, frozenset)
            or not sample_ids
            or any(not isinstance(value, str) or not value.strip() for value in sample_ids)
        ):
            return False
        if split_name == "support_pool":
            return (
                self.episode_id is not None
                and sample_ids.issubset(self.current_episode_support_ids)
            )
        if split_name in ADAPT_STREAM_SPLITS:
            return (
                protocol == ExecutionProtocol.CONTINUAL.value
                and sample_ids.issubset(self.approved_stream_ids)
            )
        return False


class LabelAccessPolicy:
    """등록 분할 정보를 사용해 라벨 접근 허용 여부만 판정합니다."""

    def __init__(self, registered_splits: Mapping[str, Iterable[str]]):
        if not isinstance(registered_splits, Mapping):
            raise LabelAccessError("registered_splits는 mapping이어야 합니다.")
        normalized: Dict[str, frozenset[str]] = {}
        id_to_split: Dict[str, str] = {}
        for split_name, sample_ids in registered_splits.items():
            if split_name not in SPLIT_NAMES:
                raise LabelAccessError(f"알 수 없는 등록 split: {split_name}")
            if isinstance(sample_ids, (str, bytes)):
                raise LabelAccessError(f"{split_name} sample_ids는 iterable이어야 합니다.")
            try:
                ids = frozenset(sample_ids)
            except TypeError as exc:
                raise LabelAccessError(f"{split_name} sample_ids를 집합으로 변환할 수 없습니다.") from exc
            if any(not isinstance(value, str) or not value.strip() for value in ids):
                raise LabelAccessError(f"{split_name}에 유효하지 않은 sample_id가 있습니다.")
            for sample_id in ids:
                previous_split = id_to_split.get(sample_id)
                if previous_split is not None:
                    raise LabelAccessError(
                        f"sample_id 중복 등록: {sample_id} -> "
                        f"{previous_split}, {split_name}"
                    )
                id_to_split[sample_id] = split_name
            normalized[split_name] = ids
        self._registered_splits = normalized
        self._id_to_split = id_to_split

    @staticmethod
    def _deny(request: LabelAccessRequest, reason: str) -> AccessDecision:
        return AccessDecision(
            allowed=False,
            reason=reason,
            denied_sample_ids=request.sample_ids,
        )

    def decide(self, request: LabelAccessRequest) -> AccessDecision:
        if not isinstance(request, LabelAccessRequest):
            return AccessDecision(allowed=False, reason="잘못된 LabelAccessRequest")
        if not request.sample_ids:
            return LabelAccessPolicy._deny(request, "빈 sample_ids 요청")

        registered_ids = self._registered_splits.get(request.split_name)
        if registered_ids is None:
            return LabelAccessPolicy._deny(request, f"미등록 split: {request.split_name}")
        unregistered = request.sample_ids - registered_ids
        if unregistered:
            return LabelAccessPolicy._deny(
                request,
                f"미등록 sample_id: {request.split_name} -> {sorted(unregistered)}",
            )
        unregistered_query = {
            sample_id
            for sample_id in request.query_ids
            if sample_id not in self._id_to_split
        }
        if unregistered_query:
            return LabelAccessPolicy._deny(
                request,
                f"미등록 query sample_id: {sorted(unregistered_query)}",
            )

        if request.execution_mode == ExecutionMode.INFER.value:
            return LabelAccessPolicy._deny(request, "INFER 라벨 접근은 전면 차단")
        query_overlap = request.sample_ids & request.query_ids
        if query_overlap:
            return LabelAccessPolicy._deny(
                request,
                f"미확정 query 접근 차단: {sorted(query_overlap)}",
            )

        if request.execution_mode == ExecutionMode.EVALUATE.value:
            return LabelAccessPolicy._decide_evaluate(self, request)
        if request.execution_mode == ExecutionMode.ADAPT.value:
            return LabelAccessPolicy._decide_adapt(self, request)
        return LabelAccessPolicy._deny(
            request, f"지원하지 않는 execution_mode: {request.execution_mode}"
        )

    def _decide_evaluate(self, request: LabelAccessRequest) -> AccessDecision:
        grant = request.evaluation_grant
        if grant is None:
            return LabelAccessPolicy._deny(request, "EVALUATE evaluation_grant 필요")
        if grant.phase not in (6, 7):
            return LabelAccessPolicy._deny(request, f"EVALUATE 허용 phase 아님: {grant.phase}")
        if request.split_name not in grant.approved_splits:
            return LabelAccessPolicy._deny(request, f"평가 미승인 split: {request.split_name}")
        return AccessDecision(
            allowed=True,
            reason="EVALUATE 승인; SealGuard 검증 필요",
            allowed_sample_ids=request.sample_ids,
        )

    def _decide_adapt(self, request: LabelAccessRequest) -> AccessDecision:
        support_ids = self._registered_splits.get("support_pool", frozenset())
        stream_ids = (
            self._registered_splits.get("dev_stream", frozenset())
            | self._registered_splits.get("main_stream", frozenset())
        )
        unknown_support = request.current_episode_support_ids - support_ids
        unknown_stream = request.approved_stream_ids - stream_ids
        if unknown_support or unknown_stream:
            return LabelAccessPolicy._deny(
                request,
                "ADAPT 승인 컨텍스트에 미등록 sample_id: "
                f"support={sorted(unknown_support)}, stream={sorted(unknown_stream)}",
            )
        scope = AdaptScope(
            episode_id=request.episode_id,
            current_episode_support_ids=request.current_episode_support_ids,
            approved_stream_ids=request.approved_stream_ids,
        )
        if not scope.authorize(
            split_name=request.split_name,
            sample_ids=request.sample_ids,
            protocol=request.protocol,
        ):
            return LabelAccessPolicy._deny(
                request,
                f"ADAPT 승인 범위 밖: split={request.split_name}, protocol={request.protocol}",
            )
        return AccessDecision(
            allowed=True,
            reason="ADAPT 승인 범위 허용",
            allowed_sample_ids=request.sample_ids,
        )


def resolve_labels(
    request: LabelAccessRequest,
    vault: GroundTruthVault,
    policy: LabelAccessPolicy,
    *,
    seal_guard: Any = None,
    verification_record: Any = None,
) -> Dict[str, str]:
    """접근 판정과 선택적 SealGuard 검증 후에만 vault 라벨을 해석합니다."""
    if not isinstance(vault, GroundTruthVault):
        raise LabelAccessError("vault는 GroundTruthVault여야 합니다.")
    if not isinstance(policy, LabelAccessPolicy):
        raise LabelAccessError("policy는 LabelAccessPolicy여야 합니다.")

    decision = LabelAccessPolicy.decide(policy, request)
    if not decision.allowed:
        raise LabelAccessError(decision.reason)

    if request.execution_mode == ExecutionMode.EVALUATE.value:
        grant = request.evaluation_grant
        if not isinstance(seal_guard, SealGuard):
            raise LabelAccessError("EVALUATE 실제 SealGuard가 필요합니다.")
        if not isinstance(verification_record, VerificationRecord):
            raise LabelAccessError("EVALUATE VerificationRecord가 필요합니다.")
        try:
            SealGuard.assert_access(
                seal_guard,
                phase=grant.phase,
                execution_mode=request.execution_mode,
                split_name=request.split_name,
                evaluation_grant=grant,
                verification_record=verification_record,
            )
        except Exception as exc:
            raise LabelAccessError("EVALUATE SealGuard 검증 실패") from exc

    try:
        return vault._read_labels(decision.allowed_sample_ids)
    except GroundTruthVaultError as exc:
        raise LabelAccessError(str(exc)) from exc