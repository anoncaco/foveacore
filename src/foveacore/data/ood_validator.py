"""FoveaCore Phase 0 OOD/전이 격리 검증기 (T3).

이미지 SHA-256 중복과 8개 OOD·전이 쌍의 class·writer·source 격리를
검증합니다. 전체 sample_id 중복은 foveacore.data.splits.validate_split_ids_disjoint()가 담당합니다. 메타데이터는 공개 manifest가 아닌 별도 저장소에서 읽습니다.
"""

from typing import Any, Dict, List, Set

from foveacore.data.schema import (
    LEGACY_DISJOINT_RULES,
    OOD_TRANSFER_DISJOINT_RULES,
    SPLIT_NAMES,
)
from foveacore.data.validation_metadata import ValidationMetadataStore

# 기존 import 경로는 공식 8쌍으로 호환합니다. 원문 5쌍은 LEGACY_DISJOINT_RULES입니다.
DISJOINT_RULES = OOD_TRANSFER_DISJOINT_RULES


class OODValidator:
    """OOD 및 문자 전이 데이터 분할의 disjoint(분리) 규칙 검증기."""

    def __init__(
        self,
        manifests: Dict[str, Dict[str, Any]],
        metadata_store: ValidationMetadataStore | None = None,
    ):
        self._manifests = manifests
        self._metadata_store = metadata_store

    def _sample_ids(self, split_name: str) -> Set[str]:
        entries = self._manifests.get(split_name, {}).get("entries", [])
        return {e.get("sample_id") for e in entries if e.get("sample_id")}

    def _entries(self, split_name: str) -> List[Dict[str, Any]]:
        return self._manifests.get(split_name, {}).get("entries", [])

    def _split_ids(self, split_name: str) -> Set[str]:
        if self._metadata_store is None:
            raise ValueError("검증 메타데이터 저장소가 필요합니다.")
        try:
            return {
                record.sample_id
                for record in self._metadata_store.values_for_split(split_name)
            }
        except ValueError:
            return set()

    def validate_metadata_binding(self) -> List[str]:
        """샘플이 있는데 메타데이터가 없거나 미확인 샘플이 있으면 실패합니다."""
        if self._metadata_store is None:
            return [
                "검증 메타데이터 저장소가 필요합니다. "
                "metadata_store 없이 메타데이터 격리를 검증할 수 없습니다."
            ]
        errors: List[str] = []
        public_ids_by_split = {
            split_name: frozenset(
                entry.get("sample_id")
                for entry in manifest.get("entries", [])
                if entry.get("sample_id")
            )
            for split_name, manifest in self._manifests.items()
        }
        for split_name, ids in public_ids_by_split.items():
            metadata_ids = self._split_ids(split_name)
            missing = ids - metadata_ids
            if missing:
                errors.append(
                    f"검증 메타데이터 누락: {split_name} -> {sorted(missing)}"
                )
        for split_name in SPLIT_NAMES:
            metadata_ids = self._split_ids(split_name)
            public_ids = public_ids_by_split.get(split_name, frozenset())
            missing = metadata_ids - public_ids
            if missing:
                errors.append(
                    f"공개 manifest에 없는 메타데이터: {split_name} -> {sorted(missing)}"
                )
        return errors

    def validate_disjoint(self) -> List[str]:
        """8개 OOD·전이 쌍의 ID와 class·writer·source 교집합을 검사합니다.

        전체 분할의 sample_id·이미지 SHA-256 중복은
        validate_split_ids_disjoint()/validate_cross_split_image_duplicates()가 담당합니다.
        """
        errors: List[str] = self.validate_metadata_binding()
        if errors:
            return errors
        for a, b in OOD_TRANSFER_DISJOINT_RULES:
            ids_a = self._sample_ids(a)
            ids_b = self._sample_ids(b)
            overlap = ids_a & ids_b
            if overlap:
                errors.append(f"OOD/전환 DISJOINT 위반: {a} ∩ {b} = {sorted(overlap)}")
            errors.extend(self._validate_pair_metadata(a, b))
        return errors

    def _validate_pair_metadata(self, a: str, b: str) -> List[str]:
        """한 쌍에 대해 class_id·writer_id·source_id를 축별로 검사합니다."""
        if self._metadata_store is None:
            raise ValueError("검증 메타데이터 저장소가 필요합니다.")
        errors: List[str] = []
        for field in ("class_id", "writer_id", "source_id"):
            values_a = {
                getattr(self._metadata_store.get(a, sid), field)
                for sid in self._sample_ids(a)
            }
            values_b = {
                getattr(self._metadata_store.get(b, sid), field)
                for sid in self._sample_ids(b)
            }
            overlap = values_a & values_b
            if overlap:
                errors.append(
                    f"OOD/전환 {field} DISJOINT 위반: {a} ∩ {b} = {sorted(overlap)}"
                )
        return errors

    def validate_no_overlap_with(self, split_name: str, other_ids: Set[str]) -> List[str]:
        """특정 분할이 주어진 ID 집합과 중복되지 않는지 검사."""
        ids = self._sample_ids(split_name)
        overlap = ids & other_ids
        if overlap:
            return [f"OOD/전환 중복: {split_name} ∩ 주어진ID = {sorted(overlap)}"]
        return []

    def validate_cross_split_image_duplicates(self) -> List[str]:
        """모든 분할의 이미지 SHA-256 교차 중복 및 분할 내 중복을 검사합니다."""
        errors: List[str] = []
        hash_to_locations: Dict[str, List[str]] = {}
        for split_name, manifest in self._manifests.items():
            for entry in manifest.get("entries", []):
                digest = entry.get("sha256")
                if digest:
                    hash_to_locations.setdefault(digest, []).append(
                        f"{split_name}:{entry.get('sample_id')}"
                    )
        for digest, locations in hash_to_locations.items():
            if len(locations) > 1:
                errors.append(f"이미지 SHA-256 중복: {digest} -> {locations}")
        return errors

    def _phash_distance(self, a: str, b: str) -> int:
        try:
            va = int(a, 16)
            vb = int(b, 16)
        except (ValueError, TypeError):
            return 10**9
        return bin(va ^ vb).count("1")

    def find_phash_candidates(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """pHash 근접 중복 후보 검출(버킷 기반)."""
        candidates: List[Dict[str, Any]] = []
        items = []
        for split_name, manifest in self._manifests.items():
            for entry in manifest.get("entries", []):
                if entry.get("phash"):
                    items.append((split_name, entry.get("sample_id"), entry["phash"]))

        buckets: Dict[str, List[int]] = {}
        for idx, (_, _, ph) in enumerate(items):
            key = ph[:4] if len(ph) >= 4 else ph
            buckets.setdefault(key, []).append(idx)

        seen = set()
        for idxs in buckets.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if (i, j) in seen or (j, i) in seen:
                        continue
                    seen.add((i, j))
                    distance = self._phash_distance(items[i][2], items[j][2])
                    if distance <= threshold and items[i][0] != items[j][0]:
                        candidates.append(
                            {
                                "split_a": items[i][0],
                                "id_a": items[i][1],
                                "split_b": items[j][0],
                                "id_b": items[j][1],
                                "distance": distance,
                            }
                        )
        return candidates

    @staticmethod
    def _has_metadata_binding_errors(errors: List[str]) -> bool:
        """메타데이터 바인딩 실패를 필터링 없이 전파해야 하는지 판단합니다."""
        return any(
            error.startswith(
                (
                    "검증 메타데이터 저장소가 필요합니다",
                    "검증 메타데이터 누락:",
                    "공개 manifest에 없는 메타데이터:",
                )
            )
            for error in errors
        )

    def validate_class_writer_source_overlap(self) -> List[str]:
        """기존 메서드명을 유지하며 8쌍의 메타데이터 격리 오류를 반환합니다."""
        errors = self.validate_disjoint()
        if self._metadata_store is None or self._has_metadata_binding_errors(errors):
            return errors
        return [
            error
            for error in errors
            if any(
                f" {field} DISJOINT 위반:" in error
                for field in ("class_id", "writer_id", "source_id")
            )
        ]

    def validate_alphabet_overlap(self) -> List[str]:
        """alphabet_id 중복은 자동 차단이 아닌 보조 경고로 보고합니다."""
        if self._metadata_store is None or self.validate_metadata_binding():
            return []
        warnings: List[str] = []
        for a, b in OOD_TRANSFER_DISJOINT_RULES:
            values_a = {
                self._metadata_store.get(a, sid).alphabet_id
                for sid in self._sample_ids(a)
                if self._metadata_store.get(a, sid).alphabet_id is not None
            }
            values_b = {
                self._metadata_store.get(b, sid).alphabet_id
                for sid in self._sample_ids(b)
                if self._metadata_store.get(b, sid).alphabet_id is not None
            }
            overlap = values_a & values_b
            if overlap:
                warnings.append(
                    f"OOD/전환 alphabet_id 중복 경고: {a} ∩ {b} = {sorted(overlap)}"
                )
        return warnings

    @staticmethod
    def validate_synthetic_structure_ids(manifest: Dict[str, Any]) -> List[str]:
        """합성 데이터 structure_id 중복 검사 (호환 메서드)."""
        ids = [
            entry.get("structure_id")
            for entry in manifest.get("entries", [])
            if entry.get("structure_id") is not None
        ]
        if len(ids) != len(set(ids)):
            return ["합성 데이터 structure_id 중복"]
        return []

    def validate_writer_source_isolation(self) -> List[str]:
        """레거시 writer/source 검사를 8쌍 기준으로 위임합니다."""
        errors = self.validate_disjoint()
        if self._metadata_store is None or self._has_metadata_binding_errors(errors):
            return errors
        return [
            error
            for error in errors
            if " writer_id DISJOINT 위반:" in error
            or " source_id DISJOINT 위반:" in error
        ]

    @staticmethod
    def validate_synthetic_split(ratios: Dict[str, int]) -> List[str]:
        """합성 데이터 70:15:15 분할 규약 검증."""
        errors: List[str] = []
        for key in ("train", "dev", "test"):
            if key not in ratios:
                errors.append(f"splits_synthetic.{key} 누락")
        if not errors:
            values = [ratios.get(key) for key in ("train", "dev", "test")]
            if not all(isinstance(value, int) for value in values):
                errors.append("splits_synthetic 비율이 정수가 아닙니다.")
            elif sum(values) != 100:
                errors.append("splits_synthetic 합이 100이어야 합니다.")
            elif ratios.get("train") != 70 or ratios.get("dev") != 15 or ratios.get("test") != 15:
                errors.append("splits_synthetic은 70:15:15여야 합니다.")
        return errors
