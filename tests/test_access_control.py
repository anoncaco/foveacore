"""T3 ADAPT·EVALUATE 라벨 접근 정책과 resolver 테스트."""

import hashlib
import json

import pytest

from foveacore.data.access_control import (
    AdaptScope,
    LabelAccessError,
    LabelAccessPolicy,
    resolve_labels,
)
from foveacore.data.ground_truth_vault import GroundTruthVault
from foveacore.data.schema import AccessDecision, EvaluationGrant, LabelAccessRequest
from foveacore.data.seal_guard import (
    IndependentBaseline,
    SealGuard,
    compute_freeze_manifest_hash,
)


REGISTERED_SPLITS = {
    "support_pool": {"support-1", "support-2"},
    "dev_stream": {"dev-1", "dev-2"},
    "main_stream": {"main-1", "main-2"},
    "official_test": {"official-1"},
    "reserved_sealed": {"reserved-1"},
    "p2_non_digit": {"p2-1"},
    "non_digit_dev": {"non-digit-1"},
    "dev_ood": {"dev-ood-1"},
    "test_ood": {"test-ood-1"},
    "phase7_char_transfer": {"phase7-1"},
}


def _policy(splits=None):
    return LabelAccessPolicy(REGISTERED_SPLITS if splits is None else splits)


def _request(
    *,
    execution_mode="ADAPT",
    protocol="STRICT_1SHOT",
    split_name="support_pool",
    sample_ids=("support-1",),
    **kwargs,
):
    return LabelAccessRequest(
        execution_mode=execution_mode,
        protocol=protocol,
        split_name=split_name,
        sample_ids=frozenset(sample_ids),
        **kwargs,
    )


def test_infer_blocks_registered_sample_and_decision_has_no_labels():
    decision = _policy().decide(_request(execution_mode="INFER"))
    assert decision.allowed is False
    assert decision.denied_sample_ids == frozenset({"support-1"})
    assert decision.allowed_sample_ids == frozenset()
    assert set(decision.to_dict()) == {
        "allowed",
        "reason",
        "allowed_sample_ids",
        "denied_sample_ids",
    }


def test_support_requires_current_episode_membership():
    policy = _policy()
    allowed = policy.decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1"}),
        )
    )
    assert allowed.allowed is True
    assert allowed.allowed_sample_ids == frozenset({"support-1"})

    no_episode = policy.decide(
        _request(current_episode_support_ids=frozenset({"support-1"}))
    )
    outside_episode = policy.decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-2"}),
        )
    )
    assert no_episode.allowed is False
    assert outside_episode.allowed is False


@pytest.mark.parametrize("split_name", ["dev_stream", "main_stream"])
def test_strict_1shot_blocks_even_approved_stream(split_name):
    sample_id = "dev-1" if split_name == "dev_stream" else "main-1"
    decision = _policy().decide(
        _request(
            split_name=split_name,
            sample_ids=(sample_id,),
            approved_stream_ids=frozenset({sample_id}),
        )
    )
    assert decision.allowed is False


@pytest.mark.parametrize("split_name", ["dev_stream", "main_stream"])
def test_continual_allows_only_approved_stream(split_name):
    sample_id = "dev-1" if split_name == "dev_stream" else "main-1"
    policy = _policy()
    approved = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name=split_name,
            sample_ids=(sample_id,),
            approved_stream_ids=frozenset({sample_id}),
        )
    )
    unapproved = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name=split_name,
            sample_ids=(sample_id,),
        )
    )
    assert approved.allowed is True
    assert unapproved.allowed is False


def test_non_digit_dev_is_blocked():
    decision = _policy().decide(
        _request(
            protocol="CONTINUAL",
            split_name="non_digit_dev",
            sample_ids=("non-digit-1",),
            approved_stream_ids=frozenset({"non-digit-1"}),
        )
    )
    assert decision.allowed is False


def test_unresolved_query_blocks_support_and_stream():
    policy = _policy()
    support = policy.decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1"}),
            query_ids=frozenset({"support-1"}),
        )
    )
    stream = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name="dev_stream",
            sample_ids=("dev-1",),
            approved_stream_ids=frozenset({"dev-1"}),
            query_ids=frozenset({"dev-1"}),
        )
    )
    assert support.allowed is False
    assert stream.allowed is False


def test_confirmed_query_requires_separate_stream_approval():
    policy = _policy()
    unresolved = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name="dev_stream",
            sample_ids=("dev-1",),
            query_ids=frozenset({"dev-1"}),
        )
    )
    approved_after_confirmation = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name="dev_stream",
            sample_ids=("dev-1",),
            approved_stream_ids=frozenset({"dev-1"}),
        )
    )
    assert unresolved.allowed is False
    assert approved_after_confirmation.allowed is True


def test_batch_request_is_all_or_nothing():
    decision = _policy().decide(
        _request(
            sample_ids=("support-1", "support-2"),
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1"}),
        )
    )
    assert decision.allowed is False
    assert decision.allowed_sample_ids == frozenset()
    assert decision.denied_sample_ids == frozenset({"support-1", "support-2"})


def test_unregistered_split_and_sample_fail_closed():
    without_official = dict(REGISTERED_SPLITS)
    without_official.pop("official_test")
    unregistered_split = _policy(without_official).decide(
        _request(
            execution_mode="EVALUATE",
            split_name="official_test",
            sample_ids=("official-1",),
        )
    )
    unregistered_sample = _policy().decide(_request(sample_ids=("missing",)))
    assert unregistered_split.allowed is False
    assert unregistered_sample.allowed is False


def test_policy_rejects_cross_split_duplicate_registration():
    with pytest.raises(LabelAccessError, match="중복"):
        _policy({"support_pool": {"shared"}, "dev_stream": {"shared"}})


def test_cross_split_query_without_requested_overlap_is_allowed():
    decision = _policy().decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1"}),
            query_ids=frozenset({"official-1"}),
        )
    )
    assert decision.allowed is True


def test_unregistered_query_fails_closed():
    decision = _policy().decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1"}),
            query_ids=frozenset({"missing-query"}),
        )
    )
    assert decision.allowed is False


def test_unregistered_adapt_context_fails_closed():
    policy = _policy()
    unknown_support = policy.decide(
        _request(
            episode_id="episode-1",
            current_episode_support_ids=frozenset({"support-1", "missing"}),
        )
    )
    unknown_stream = policy.decide(
        _request(
            protocol="CONTINUAL",
            split_name="dev_stream",
            sample_ids=("dev-1",),
            approved_stream_ids=frozenset({"dev-1", "missing"}),
        )
    )
    assert unknown_support.allowed is False
    assert unknown_stream.allowed is False


def test_adapt_scope_and_policy_reject_invalid_inputs():
    with pytest.raises(LabelAccessError):
        AdaptScope("episode-1", {"support-1"}, frozenset())
    with pytest.raises(LabelAccessError):
        LabelAccessPolicy({"unknown_split": {"sample-1"}})
    with pytest.raises(LabelAccessError):
        LabelAccessPolicy({"support_pool": "support-1"})


class SpyVault(GroundTruthVault):
    def __init__(self, labels):
        super().__init__(labels)
        self.read_calls = 0

    def _read_labels(self, sample_ids):
        self.read_calls += 1
        return super()._read_labels(sample_ids)


def _grant(phase=6, approved_splits=("official_test",)):
    return EvaluationGrant(
        evaluation_id="evaluation-1",
        phase=phase,
        approved_splits=frozenset(approved_splits),
    )


def _evaluate_request(*, grant=None, split_name="official_test", sample_ids=("official-1",)):
    return _request(
        execution_mode="EVALUATE",
        split_name=split_name,
        sample_ids=sample_ids,
        evaluation_grant=_grant() if grant is None else grant,
    )


def _spy_vault():
    return SpyVault(
        {
            "support-1": "5",
            "support-2": "6",
            "dev-1": "7",
            "official-1": "8",
            "reserved-1": "9",
        }
    )


def test_evaluate_requires_phase_and_approved_split():
    policy = _policy()
    no_grant = policy.decide(
        _request(
            execution_mode="EVALUATE",
            split_name="official_test",
            sample_ids=("official-1",),
        )
    )
    wrong_phase = policy.decide(_evaluate_request(grant=_grant(phase=5)))
    unapproved = policy.decide(
        _evaluate_request(
            grant=_grant(phase=6, approved_splits=("official_test",)),
            split_name="reserved_sealed",
            sample_ids=("reserved-1",),
        )
    )
    allowed = policy.decide(_evaluate_request())
    assert no_grant.allowed is False
    assert wrong_phase.allowed is False
    assert unapproved.allowed is False
    assert allowed.allowed is True


def test_resolve_labels_rejects_policy_subclass_guard_bypass():
    class AllowAllPolicy(LabelAccessPolicy):
        def decide(self, request):
            return AccessDecision(
                allowed=True,
                reason="bypass",
                allowed_sample_ids=request.sample_ids,
            )

    vault = _spy_vault()
    with pytest.raises(LabelAccessError):
        resolve_labels(_request(execution_mode="INFER"), vault, AllowAllPolicy(REGISTERED_SPLITS))
    assert vault.read_calls == 0


def test_resolve_labels_reads_vault_only_after_adapt_approval():
    vault = _spy_vault()
    request = _request(
        episode_id="episode-1",
        current_episode_support_ids=frozenset({"support-1"}),
    )
    assert resolve_labels(request, vault, _policy()) == {"support-1": "5"}
    assert vault.read_calls == 1


def test_resolve_labels_does_not_read_vault_when_adapt_is_denied():
    vault = _spy_vault()
    with pytest.raises(LabelAccessError):
        resolve_labels(_request(execution_mode="INFER"), vault, _policy())
    assert vault.read_calls == 0


def test_evaluate_requires_real_seal_guard_and_record_before_vault_read(tmp_path):
    vault = _spy_vault()
    request = _evaluate_request()
    guard, record, _ = _real_seal_context(tmp_path)

    class NoopSealGuard:
        def assert_access(self, **kwargs):
            return None

    with pytest.raises(LabelAccessError):
        resolve_labels(request, vault, _policy())
    with pytest.raises(LabelAccessError):
        resolve_labels(request, vault, _policy(), seal_guard=guard)
    with pytest.raises(LabelAccessError):
        resolve_labels(
            request,
            vault,
            _policy(),
            seal_guard=NoopSealGuard(),
            verification_record=record,
        )
    with pytest.raises(LabelAccessError):
        resolve_labels(
            request,
            vault,
            _policy(),
            seal_guard=guard,
            verification_record=object(),
        )
    assert vault.read_calls == 0


def test_evaluate_rejects_seal_guard_subclass_override_bypass(tmp_path):
    guard, record, _ = _real_seal_context(tmp_path)

    class NoopSealGuard(SealGuard):
        def assert_access(self, **kwargs):
            return None

    vault = _spy_vault()
    with pytest.raises(LabelAccessError, match="SealGuard"):
        resolve_labels(
            _evaluate_request(),
            vault,
            _policy(),
            seal_guard=NoopSealGuard({}),
            verification_record=record,
        )
    assert vault.read_calls == 0


def test_evaluate_calls_seal_guard_before_reading_vault(tmp_path, monkeypatch):
    vault = _spy_vault()
    guard, record, _ = _real_seal_context(tmp_path)
    request = _evaluate_request()
    original_assert_access = SealGuard.assert_access
    calls = []

    def assert_access_before_vault_read(self, **kwargs):
        assert vault.read_calls == 0
        calls.append(kwargs)
        return original_assert_access(self, **kwargs)

    monkeypatch.setattr(SealGuard, "assert_access", assert_access_before_vault_read)

    labels = resolve_labels(
        request,
        vault,
        _policy(),
        seal_guard=guard,
        verification_record=record,
    )
    assert labels == {"official-1": "8"}
    assert vault.read_calls == 1
    assert calls == [
        {
            "phase": 6,
            "execution_mode": "EVALUATE",
            "split_name": "official_test",
            "evaluation_grant": request.evaluation_grant,
            "verification_record": record,
        }
    ]


def test_evaluate_seal_failure_does_not_read_vault(tmp_path, monkeypatch):
    guard, record, _ = _real_seal_context(tmp_path)

    def fail_assert_access(self, **kwargs):
        raise RuntimeError("seal failed")

    monkeypatch.setattr(SealGuard, "assert_access", fail_assert_access)
    vault = _spy_vault()
    with pytest.raises(LabelAccessError, match="SealGuard"):
        resolve_labels(
            _evaluate_request(),
            vault,
            _policy(),
            seal_guard=guard,
            verification_record=record,
        )
    assert vault.read_calls == 0


def test_evaluate_rejects_mismatched_record_before_vault_read(tmp_path):
    guard, record, _ = _real_seal_context(tmp_path)
    grant = EvaluationGrant(
        evaluation_id="evaluation-2",
        phase=6,
        approved_splits=frozenset({"official_test"}),
    )
    vault = _spy_vault()
    with pytest.raises(LabelAccessError, match="SealGuard"):
        resolve_labels(
            _evaluate_request(grant=grant),
            vault,
            _policy(),
            seal_guard=guard,
            verification_record=record,
        )
    assert vault.read_calls == 0


def test_resolve_labels_does_not_return_partial_labels_for_missing_vault_id():
    vault = SpyVault({"support-1": "5"})
    request = _request(
        sample_ids=("support-1", "support-2"),
        episode_id="episode-1",
        current_episode_support_ids=frozenset({"support-1", "support-2"}),
    )
    with pytest.raises(LabelAccessError, match="없는 sample_id"):
        resolve_labels(request, vault, _policy())
    assert vault.read_calls == 1


def _real_seal_context(tmp_path):
    artifact_paths = {
        "data_split_hash": tmp_path / "split.json",
        "config_hash": tmp_path / "config.json",
        "checkpoint_hash": tmp_path / "checkpoint.bin",
    }
    artifact_hashes = {}
    for field_name, path in artifact_paths.items():
        content = field_name.encode("utf-8")
        path.write_bytes(content)
        artifact_hashes[field_name] = hashlib.sha256(content).hexdigest()
    body = {"schema_version": "1.0", **artifact_hashes}
    manifest_hash = compute_freeze_manifest_hash(body)
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps({**body, "manifest_sha256": manifest_hash}),
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "freeze_manifest.sha256"
    sidecar_path.write_text(manifest_hash, encoding="utf-8")
    baseline = IndependentBaseline(manifest_sha256=manifest_hash)
    guard = SealGuard({manifest_path: baseline})
    grant = _grant()
    record = guard.verify(
        manifest_path,
        artifact_paths,
        sidecar_path,
        grant,
    )
    return guard, record, artifact_paths


def test_resolve_labels_with_real_seal_guard(tmp_path):
    guard, record, _ = _real_seal_context(tmp_path)
    vault = _spy_vault()
    labels = resolve_labels(
        _evaluate_request(),
        vault,
        _policy(),
        seal_guard=guard,
        verification_record=record,
    )
    assert labels == {"official-1": "8"}
    assert vault.read_calls == 1


def test_resolve_labels_rejects_artifact_change_after_real_verification(tmp_path):
    guard, record, artifact_paths = _real_seal_context(tmp_path)
    artifact_paths["checkpoint_hash"].write_bytes(b"changed-after-verification")
    vault = _spy_vault()
    with pytest.raises(LabelAccessError, match="SealGuard"):
        resolve_labels(
            _evaluate_request(),
            vault,
            _policy(),
            seal_guard=guard,
            verification_record=record,
        )
    assert vault.read_calls == 0
