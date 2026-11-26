"""T3 1단계 공개 스키마·요청 계약 테스트."""

import json
from dataclasses import FrozenInstanceError

import pytest

from foveacore.data.schema import (
    DISJOINT_RULES,
    FREEZE_HASH_FIELDS,
    LEGACY_DISJOINT_RULES,
    OOD_TRANSFER_DISJOINT_RULES,
    PUBLIC_ENTRY_FIELDS,
    REQUIRED_FREEZE_FIELDS,
    SPLIT_NAMES,
    AccessDecision,
    EvaluationGrant,
    ExecutionMode,
    ExecutionProtocol,
    FreezeManifest,
    LabelAccessRequest,
    SchemaError,
    SealLevel,
    SplitEntry,
    SplitManifest,
    SplitName,
)


def _entry(sample_id="sample-1", split_name="support_pool", sealed_level="OPEN"):
    return {
        "sample_id": sample_id,
        "split_name": split_name,
        "sha256": f"sha-{sample_id}",
        "phash": f"phash-{sample_id}",
        "sealed_level": sealed_level,
    }


def test_split_names_have_exactly_ten_public_slots():
    assert SPLIT_NAMES == {
        "support_pool",
        "dev_stream",
        "main_stream",
        "official_test",
        "reserved_sealed",
        "p2_non_digit",
        "non_digit_dev",
        "dev_ood",
        "test_ood",
        "phase7_char_transfer",
    }


def test_enums_are_string_compatible():
    assert SplitName.SUPPORT_POOL == "support_pool"
    assert str(ExecutionMode.ADAPT) == "ADAPT"
    assert ExecutionProtocol.CONTINUAL == "CONTINUAL"
    assert SealLevel.SEALED == "SEALED"


def test_eight_ood_transfer_pairs_and_legacy_five_pair_alias():
    assert len(OOD_TRANSFER_DISJOINT_RULES) == 8
    assert len(LEGACY_DISJOINT_RULES) == 5
    assert DISJOINT_RULES is OOD_TRANSFER_DISJOINT_RULES
    assert {
        ("non_digit_dev", "dev_ood"),
        ("non_digit_dev", "test_ood"),
        ("non_digit_dev", "phase7_char_transfer"),
    }.issubset(OOD_TRANSFER_DISJOINT_RULES)


def test_split_entry_has_only_required_public_fields():
    entry = SplitEntry.from_dict(_entry())
    assert set(entry.to_dict()) == PUBLIC_ENTRY_FIELDS
    assert json.loads(json.dumps(entry.to_dict())) == entry.to_dict()


@pytest.mark.parametrize(
    "field", ["label", "class_id", "writer_id", "alphabet_id", "source_id"]
)
def test_split_entry_rejects_label_reconstructing_fields(field):
    data = {**_entry(), field: "0"}
    with pytest.raises(SchemaError, match="금지 필드"):
        SplitEntry.from_dict(data)


@pytest.mark.parametrize("missing", sorted(PUBLIC_ENTRY_FIELDS))
def test_split_entry_rejects_missing_required_fields(missing):
    data = _entry()
    del data[missing]
    with pytest.raises(SchemaError, match="필수 필드 누락"):
        SplitEntry.from_dict(data)


def test_split_entry_is_immutable():
    entry = SplitEntry.from_dict(_entry())
    with pytest.raises(FrozenInstanceError):
        entry.sample_id = "changed"


def test_split_manifest_roundtrip_preserves_public_boundary():
    manifest = SplitManifest.from_dict(
        {
            "schema_version": "1.0",
            "split_name": "support_pool",
            "sealed_level": "OPEN",
            "entries": [_entry()],
        }
    )
    restored = SplitManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    assert not hasattr(restored.entries[0], "label")


def test_split_manifest_rejects_entry_manifest_mismatch():
    with pytest.raises(SchemaError, match="split_name 불일치"):
        SplitManifest.from_dict(
            {
                "schema_version": "1.0",
                "split_name": "support_pool",
                "sealed_level": "OPEN",
                "entries": [_entry(split_name="dev_stream")],
            }
        )


def test_split_manifest_rejects_entry_seal_mismatch():
    with pytest.raises(SchemaError, match="sealed_level 불일치"):
        SplitManifest.from_dict(
            {
                "schema_version": "1.0",
                "split_name": "official_test",
                "sealed_level": "SEALED",
                "entries": [_entry(split_name="official_test", sealed_level="OPEN")],
            }
        )


def test_split_manifest_rejects_duplicate_sample_ids():
    with pytest.raises(SchemaError, match="중복 sample_id"):
        SplitManifest.from_dict(
            {
                "schema_version": "1.0",
                "split_name": "dev_stream",
                "sealed_level": "OPEN",
                "entries": [
                    _entry(split_name="dev_stream"),
                    _entry(split_name="dev_stream"),
                ],
            }
        )


def test_freeze_manifest_roundtrip_preserves_required_field_names():
    freeze = FreezeManifest.from_dict(
        {
            "schema_version": "1.0",
            "manifest_sha256": "manifest-hash",
            "data_split_hash": "split-hash",
            "config_hash": "config-hash",
            "checkpoint_hash": "checkpoint-hash",
            "experiment_id": "exp-1",
        }
    )
    restored = FreezeManifest.from_dict(json.loads(json.dumps(freeze.to_dict())))
    assert restored == freeze
    assert REQUIRED_FREEZE_FIELDS <= freeze.to_dict().keys()
    assert set(FREEZE_HASH_FIELDS) == {
        "data_split_hash",
        "config_hash",
        "checkpoint_hash",
    }


@pytest.mark.parametrize("missing", sorted(REQUIRED_FREEZE_FIELDS))
def test_freeze_manifest_rejects_missing_required_fields(missing):
    data = {
        "schema_version": "1.0",
        "manifest_sha256": "manifest-hash",
        "data_split_hash": "split-hash",
        "config_hash": "config-hash",
        "checkpoint_hash": "checkpoint-hash",
    }
    del data[missing]
    with pytest.raises(SchemaError, match="필수 필드 누락"):
        FreezeManifest.from_dict(data)


def test_freeze_manifest_metadata_cannot_override_hash_fields():
    data = {
        "schema_version": "1.0",
        "manifest_sha256": "manifest-hash",
        "data_split_hash": "split-hash",
        "config_hash": "config-hash",
        "checkpoint_hash": "checkpoint-hash",
    }
    with pytest.raises(SchemaError, match="manifest 필드를 덮어씁니다"):
        FreezeManifest(**data, metadata={"manifest_sha256": "override"})


def test_label_access_request_normalizes_ids_without_labels():
    request = LabelAccessRequest(
        execution_mode=ExecutionMode.ADAPT,
        protocol=ExecutionProtocol.CONTINUAL,
        split_name=SplitName.MAIN_STREAM,
        sample_ids=["sample-1", "sample-1"],
        episode_id="episode-1",
        current_episode_support_ids=["support-1"],
        approved_stream_ids=["sample-1"],
        query_ids=["query-1"],
    )
    restored = LabelAccessRequest.from_dict(
        json.loads(json.dumps(request.to_dict()))
    )
    assert restored == request
    assert restored.sample_ids == frozenset({"sample-1"})
    assert "label" not in restored.to_dict()
    assert "labels" not in restored.to_dict()


def test_label_access_request_rejects_invalid_protocol_and_mode():
    with pytest.raises(SchemaError, match="execution_mode"):
        LabelAccessRequest("UNKNOWN", "CONTINUAL", "main_stream", {"s1"})
    with pytest.raises(SchemaError, match="protocol"):
        LabelAccessRequest("ADAPT", "UNKNOWN", "main_stream", {"s1"})


def test_evaluation_grant_is_split_scoped_and_phase_bound():
    grant = EvaluationGrant("official-1", 6, {"official_test"})
    assert grant.phase == 6
    assert grant.approved_splits == frozenset({"official_test"})
    with pytest.raises(SchemaError, match="알 수 없는 승인 split"):
        EvaluationGrant("official-1", 6, {"unknown_split"})


def test_access_decision_contains_no_label_payload_and_disallows_overlap():
    decision = AccessDecision(
        allowed=False,
        reason="query label access blocked",
        allowed_sample_ids=(),
        denied_sample_ids={"query-1"},
    )
    assert set(decision.to_dict()) == {
        "allowed",
        "reason",
        "allowed_sample_ids",
        "denied_sample_ids",
    }
    assert "label" not in decision.to_dict()
    with pytest.raises(SchemaError, match="충돌"):
        AccessDecision(True, "conflict", {"s1"}, {"s1"})
