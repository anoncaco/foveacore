"""ood_validator.py OOD/전환 disjoint 검증기 테스트 (T3)."""

import pytest

from foveacore.data.ood_validator import DISJOINT_RULES, OODValidator
from foveacore.data.schema import LEGACY_DISJOINT_RULES, OOD_TRANSFER_DISJOINT_RULES
from foveacore.data.validation_metadata import ValidationMetadataStore


def test_disjoint_rules_pass():
    manifests = {
        "p2_non_digit": {"entries": [{"sample_id": f"p2-{i}"} for i in range(3)]},
        "dev_ood": {"entries": [{"sample_id": f"ood-{i}"} for i in range(3)]},
        "test_ood": {"entries": [{"sample_id": f"test-{i}"} for i in range(3)]},
        "phase7_char_transfer": {
            "entries": [{"sample_id": f"p7-{i}"} for i in range(3)]
        },
    }
    v = OODValidator(manifests)
    assert any("저장소가 필요" in error for error in v.validate_disjoint())


def test_disjoint_rules_violation():
    manifests = {
        "p2_non_digit": {"entries": [{"sample_id": "shared-001"}]},
        "dev_ood": {"entries": [{"sample_id": "shared-001"}]},
    }
    v = OODValidator(manifests)
    errors = v.validate_disjoint()
    assert any("저장소가 필요" in error for error in errors)


def test_validate_no_overlap_with_pass():
    manifests = {
        "dev_ood": {"entries": [{"sample_id": f"ood-{i}"} for i in range(3)]},
    }
    v = OODValidator(manifests)
    assert v.validate_no_overlap_with("dev_ood", {"p2-1", "p2-2"}) == []


def test_validate_no_overlap_with_violation():
    manifests = {
        "dev_ood": {"entries": [{"sample_id": "shared-001"}]},
    }
    v = OODValidator(manifests)
    errors = v.validate_no_overlap_with("dev_ood", {"shared-001"})
    assert len(errors) == 1


def test_disjoint_rules_count():
    assert len(LEGACY_DISJOINT_RULES) == 5
    assert DISJOINT_RULES is OOD_TRANSFER_DISJOINT_RULES
    assert len(DISJOINT_RULES) == 8


def test_legacy_validators_do_not_filter_missing_metadata_errors():
    manifests = {"dev_ood": {"entries": [{"sample_id": "ood-missing-metadata"}]}}
    validator = OODValidator(manifests, ValidationMetadataStore())

    for validate in (
        validator.validate_class_writer_source_overlap,
        validator.validate_writer_source_isolation,
    ):
        errors = validate()
        assert any("검증 메타데이터 누락" in error for error in errors)
    assert validator.validate_alphabet_overlap() == []
