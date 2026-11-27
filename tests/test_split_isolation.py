"""T3 3단계 전체·OOD 격리 및 검증 메타데이터 테스트."""

import pytest

from foveacore.data.ood_validator import OODValidator
from foveacore.data.schema import OOD_TRANSFER_DISJOINT_RULES
from foveacore.data.splits import build_split_manifest, validate_split_ids_disjoint
from foveacore.data.validation_metadata import (
    ValidationMetadata,
    ValidationMetadataError,
    ValidationMetadataStore,
)


def _manifest(split_name, sample_ids):
    sealed = "SEALED" if split_name in {
        "official_test", "reserved_sealed", "dev_ood", "test_ood",
        "phase7_char_transfer",
    } else "OPEN"
    return build_split_manifest(
        [
            {
                "sample_id": sample_id,
                "sha256": f"sha-{split_name}-{sample_id}",
                "phash": f"phash-{split_name}-{sample_id}",
            }
            for sample_id in sample_ids
        ],
        split_name,
        sealed,
    )


def _record(split_name, sample_id, class_id=None, writer_id=None, source_id=None,
            alphabet_id=None):
    return ValidationMetadata(
        split_name=split_name,
        sample_id=sample_id,
        class_id=class_id or f"class-{split_name}-{sample_id}",
        writer_id=writer_id or f"writer-{split_name}-{sample_id}",
        source_id=source_id or f"source-{split_name}",
        alphabet_id=alphabet_id,
    )


def test_ood_transfer_has_exactly_eight_pairs():
    assert len(OOD_TRANSFER_DISJOINT_RULES) == 8
    assert {
        ("non_digit_dev", "dev_ood"),
        ("non_digit_dev", "test_ood"),
        ("non_digit_dev", "phase7_char_transfer"),
    }.issubset(OOD_TRANSFER_DISJOINT_RULES)


@pytest.mark.parametrize("a,b", OOD_TRANSFER_DISJOINT_RULES)
@pytest.mark.parametrize("field", ["class_id", "writer_id", "source_id"])
def test_each_ood_pair_rejects_each_metadata_axis(a, b, field):
    manifests = {a: _manifest(a, ["sample-a"]), b: _manifest(b, ["sample-b"])}
    records = [
        _record(a, "sample-a", **{field: f"shared-{field}"}),
        _record(b, "sample-b", **{field: f"shared-{field}"}),
    ]
    errors = OODValidator(
        manifests, ValidationMetadataStore.from_records(records)
    ).validate_disjoint()
    assert any(
        f"{field} DISJOINT 위반: {a} ∩ {b}" in error for error in errors
    )


def test_ood_pair_with_unknown_writer_fails_closed():
    manifests = {
        "dev_ood": _manifest("dev_ood", ["a1"]),
        "test_ood": _manifest("test_ood", ["b1"]),
    }
    records = [
        ValidationMetadata(
            split_name="dev_ood",
            sample_id="a1",
            class_id="class-a",
            writer_id=None,
            alphabet_id=None,
            source_id="source-a",
        ),
        ValidationMetadata(
            split_name="test_ood",
            sample_id="b1",
            class_id="class-b",
            writer_id=None,
            alphabet_id=None,
            source_id="source-b",
        ),
    ]

    errors = OODValidator(
        manifests, ValidationMetadataStore.from_records(records)
    ).validate_disjoint()

    assert any("dev_ood.writer_id" in error for error in errors)
    assert any("test_ood.writer_id" in error for error in errors)


def test_class_overlap_fails_even_when_ids_images_writers_sources_differ():
    a, b = "dev_ood", "test_ood"
    manifests = {a: _manifest(a, ["a1"]), b: _manifest(b, ["b1"])}
    store = ValidationMetadataStore.from_records(
        [_record(a, "a1", class_id="shared"), _record(b, "b1", class_id="shared")]
    )
    errors = OODValidator(manifests, store).validate_disjoint()
    assert any("class_id DISJOINT 위반" in error for error in errors)


def test_mnist_internal_class_overlap_is_not_an_ood_pair():
    manifests = {
        "support_pool": _manifest("support_pool", ["train-1"]),
        "dev_stream": _manifest("dev_stream", ["train-2"]),
    }
    store = ValidationMetadataStore.from_records(
        [
            _record("support_pool", "train-1", class_id="0"),
            _record("dev_stream", "train-2", class_id="0"),
        ]
    )
    assert OODValidator(manifests, store).validate_disjoint() == []


def test_p2_splits_do_not_require_pairwise_class_isolation():
    manifests = {
        "p2_non_digit": _manifest("p2_non_digit", ["p2-1"]),
        "non_digit_dev": _manifest("non_digit_dev", ["nd-1"]),
    }
    store = ValidationMetadataStore.from_records(
        [
            _record("p2_non_digit", "p2-1", class_id="shared"),
            _record("non_digit_dev", "nd-1", class_id="shared"),
        ]
    )
    assert OODValidator(manifests, store).validate_disjoint() == []


def test_phash_candidates_cross_prefix_buckets_within_hamming_threshold():
    manifests = {
        "dev_ood": _manifest("dev_ood", ["a1"]),
        "test_ood": _manifest("test_ood", ["b1"]),
    }
    manifests["dev_ood"]["entries"][0]["phash"] = "0000000000000000"
    manifests["test_ood"]["entries"][0]["phash"] = "0001000000000000"

    candidates = OODValidator(manifests).find_phash_candidates(threshold=1)

    assert candidates == [
        {
            "split_a": "dev_ood",
            "id_a": "a1",
            "split_b": "test_ood",
            "id_b": "b1",
            "distance": 1,
        }
    ]


def test_empty_ood_slots_need_no_metadata():
    manifests = {
        "p2_non_digit": _manifest("p2_non_digit", []),
        "non_digit_dev": _manifest("non_digit_dev", []),
        "dev_ood": _manifest("dev_ood", []),
        "test_ood": _manifest("test_ood", []),
        "phase7_char_transfer": _manifest("phase7_char_transfer", []),
    }
    assert OODValidator(manifests, ValidationMetadataStore()).validate_disjoint() == []


def test_nonempty_split_with_missing_metadata_fails_closed():
    manifests = {"dev_ood": _manifest("dev_ood", ["ood-1"])}
    errors = OODValidator(manifests, ValidationMetadataStore()).validate_disjoint()
    assert any("검증 메타데이터 누락: dev_ood" in error for error in errors)


def test_metadata_for_unknown_public_sample_fails_closed():
    manifests = {"dev_ood": _manifest("dev_ood", ["ood-1"])}
    store = ValidationMetadataStore.from_records(
        [_record("dev_ood", "ood-2")]
    )
    errors = OODValidator(manifests, store).validate_disjoint()
    assert any("공개 manifest에 없는 메타데이터" in error for error in errors)


def test_metadata_store_is_required_for_metadata_validation():
    manifests = {"dev_ood": _manifest("dev_ood", ["ood-1"])}
    validator = OODValidator(manifests)
    assert any("저장소가 필요" in error for error in validator.validate_metadata_binding())
    assert any("저장소가 필요" in error for error in validator.validate_disjoint())
    assert any("저장소가 필요" in error for error in validator.validate_class_writer_source_overlap())
    assert any("저장소가 필요" in error for error in validator.validate_writer_source_isolation())
    # 경고 계산은 별도 오류를 만들지 않고, 바인딩 오류는 validator 오류 경로로만 보고합니다.
    assert validator.validate_alphabet_overlap() == []


def test_all_splits_reject_cross_split_duplicate_ids():
    manifests = {
        "support_pool": _manifest("support_pool", ["shared"]),
        "official_test": _manifest("official_test", ["shared"]),
    }
    assert any("분할 간 ID 중복" in error for error in validate_split_ids_disjoint(manifests))


def test_all_splits_reject_cross_split_duplicate_image_hashes():
    shared = _manifest("support_pool", ["s1"])
    other = _manifest("official_test", ["s2"])
    other["entries"][0]["sha256"] = shared["entries"][0]["sha256"]
    errors = validate_split_ids_disjoint({"support_pool": shared, "official_test": other})
    assert any("이미지 SHA-256 중복" in error for error in errors)


def test_duplicate_image_within_one_split_is_rejected():
    manifest = _manifest("support_pool", ["s1", "s2"])
    manifest["entries"][1]["sha256"] = manifest["entries"][0]["sha256"]
    errors = OODValidator({"support_pool": manifest}).validate_cross_split_image_duplicates()
    assert any("이미지 SHA-256 중복" in error for error in errors)


def test_alphabet_overlap_is_a_warning_not_a_blocking_axis():
    a, b = "dev_ood", "test_ood"
    manifests = {a: _manifest(a, ["a1"]), b: _manifest(b, ["b1"])}
    store = ValidationMetadataStore.from_records(
        [
            _record(a, "a1", alphabet_id="latin"),
            _record(b, "b1", alphabet_id="latin"),
        ]
    )
    validator = OODValidator(manifests, store)
    assert validator.validate_disjoint() == []
    assert any("alphabet_id 중복 경고" in warning for warning in validator.validate_alphabet_overlap())


def test_metadata_store_roundtrip_preserves_separate_records(tmp_path):
    records = [_record("dev_ood", "a1"), _record("test_ood", "b1")]
    path = tmp_path / "validation_metadata.json"
    ValidationMetadataStore.from_records(records).save(path)
    restored = ValidationMetadataStore.load(path)
    assert restored.get("dev_ood", "a1") == records[0]
    assert restored.get("test_ood", "b1") == records[1]
    assert len(restored) == 2
    assert not hasattr(restored, "as_mapping")


def test_metadata_store_rejects_conflicting_duplicate_sample():
    with pytest.raises(ValidationMetadataError, match="충돌"):
        ValidationMetadataStore.from_records(
            [_record("dev_ood", "a1"), _record("dev_ood", "a1", class_id="other")]
        )
