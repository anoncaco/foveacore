"""FoveaCore Phase 0 OOD/전환 disjoint 검증기 (T3).

P2 ∩ dev_ood == ∅, P2 ∩ test_ood == ∅, P2 ∩ phase7_char_transfer == ∅,
dev_ood ∩ test_ood == ∅, test_ood ∩ phase7_char_transfer == ∅
(클래스 및 writer/source 모두)
"""

from typing import Any, Dict, List, Set

DISJOINT_RULES = [
    ("p2_non_digit", "dev_ood"),
    ("p2_non_digit", "test_ood"),
    ("p2_non_digit", "phase7_char_transfer"),
    ("dev_ood", "test_ood"),
    ("test_ood", "phase7_char_transfer"),
]


class OODValidator:
    """OOD 및 문자 전이 데이터 분할의 disjoint(분리) 규칙 검증기."""

    def __init__(self, manifests: Dict[str, Dict[str, Any]]):
        self._manifests = manifests

    def _sample_ids(self, split_name: str) -> Set[str]:
        entries = self._manifests.get(split_name, {}).get("entries", [])
        return {e.get("sample_id") for e in entries if e.get("sample_id")}

    def validate_disjoint(self) -> List[str]:
        """분리 규칙 위반 목록(빈 리스트 = PASS)."""
        errors: List[str] = []
        for a, b in DISJOINT_RULES:
            ids_a = self._sample_ids(a)
            ids_b = self._sample_ids(b)
            overlap = ids_a & ids_b
            if overlap:
                errors.append(f"OOD/전환 DISJOINT 위반: {a} ∩ {b} = {sorted(overlap)}")
        return errors

    def validate_no_overlap_with(self, split_name: str, other_ids: Set[str]) -> List[str]:
        """특정 분할이 주어진 ID 집합과 중복되지 않는지 검사."""
        ids = self._sample_ids(split_name)
        overlap = ids & other_ids
        if overlap:
            return [f"OOD/전환 중복: {split_name} ∩ 주어진ID = {sorted(overlap)}"]
        return []

    def _entries(self, split_name: str) -> List[Dict[str, Any]]:
        return self._manifests.get(split_name, {}).get("entries", [])

    def validate_cross_split_image_duplicates(self) -> List[str]:
        """이미지 SHA-256 교차 중복 검사 (T3-14)."""
        errors: List[str] = []
        hash_to_splits: Dict[str, List[str]] = {}
        for split_name, manifest in self._manifests.items():
            for e in (manifest.get("entries", []) if isinstance(manifest, dict) else []):
                h = e.get("sha256")
                if h:
                    hash_to_splits.setdefault(h, []).append(f"{split_name}:{e.get('sample_id')}")
        for h, locs in hash_to_splits.items():
            if len(locs) > 1:
                errors.append(f"이미지 SHA-256 교차 중복: {h} -> {locs}")
        return errors

    def _phash_distance(self, a: str, b: str) -> int:
        try:
            va = int(a, 16)
            vb = int(b, 16)
        except (ValueError, TypeError):
            return 10**9
        return bin(va ^ vb).count("1")

    def find_phash_candidates(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """pHash 근접 중복 후보 검출 (T3-16). 버킷 기반 효율화 (T3-17)."""
        candidates: List[Dict[str, Any]] = []
        items = []
        for split_name, manifest in self._manifests.items():
            for e in (manifest.get("entries", []) if isinstance(manifest, dict) else []):
                if e.get("phash"):
                    items.append((split_name, e.get("sample_id"), e.get("phash")))

        # 버킷: pHash 앞 4 hex chars (16 bits) → 동일 버킷 내에서만 비교
        buckets: Dict[str, List[int]] = {}
        for idx, (_, _, ph) in enumerate(items):
            key = ph[:4] if len(ph) >= 4 else ph
            buckets.setdefault(key, []).append(idx)

        seen = set()
        for key, idxs in buckets.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if i == j:
                        continue
                    if (i, j) in seen or (j, i) in seen:
                        continue
                    seen.add((i, j))
                    d = self._phash_distance(items[i][2], items[j][2])
                    if d <= threshold and items[i][0] != items[j][0]:
                        candidates.append({
                            "split_a": items[i][0], "id_a": items[i][1],
                            "split_b": items[j][0], "id_b": items[j][1],
                            "distance": d,
                        })
        return candidates

    def validate_class_writer_source_overlap(self) -> List[str]:
        """P2와 OOD 사이 클래스·writer·source 중복 검출 (T3-13)."""
        errors: List[str] = []
        p2_splits = {"p2_non_digit"}
        ood_splits = {"dev_ood", "test_ood", "phase7_char_transfer"}
        for key in ("class_id", "writer_id", "alphabet_id", "source_id"):
            p2_vals = set()
            ood_vals = set()
            for split_name, manifest in self._manifests.items():
                for e in (manifest.get("entries", []) if isinstance(manifest, dict) else []):
                    val = e.get(key)
                    if val is None:
                        continue
                    if split_name in p2_splits:
                        p2_vals.add(val)
                    if split_name in ood_splits:
                        ood_vals.add(val)
            overlap = p2_vals & ood_vals
            if overlap:
                errors.append(f"P2-OOD {key} 중복: {sorted(overlap)}")
        return errors

    @staticmethod
    def validate_synthetic_structure_ids(manifest: Dict[str, Any]) -> List[str]:
        """합성 데이터 structure_id 중복 검사 (T3-15)."""
        errors: List[str] = []
        ids = []
        for e in manifest.get("entries", []):
            sid = e.get("structure_id")
            if sid is not None:
                ids.append(sid)
        if len(ids) != len(set(ids)):
            errors.append("합성 데이터 structure_id 중복")
        return errors

    def validate_writer_source_isolation(self) -> List[str]:
        """writer_id/alphabet_id/source 기반 격리 검증 (T3-16)."""
        errors: List[str] = []
        # writer_id별 split 소속
        writer_splits: Dict[Any, Set[str]] = {}
        for split_name, manifest in self._manifests.items():
            for e in (manifest.get("entries", []) if isinstance(manifest, dict) else []):
                for key in ("writer_id", "alphabet_id", "source_id"):
                    val = e.get(key)
                    if val is not None:
                        writer_splits.setdefault((key, val), set()).add(split_name)
        for (key, val), splits in writer_splits.items():
            # OOD/transfer splits must not share writer/source with digit splits
            if len(splits) > 1 and not (len(splits) == 1):
                # only flag if a sealed/OOD split shares with an open digit split
                ood = splits & {"dev_ood", "test_ood", "phase7_char_transfer"}
                digit = splits & {"support_pool", "dev_stream", "main_stream", "official_test", "reserved_sealed"}
                if ood and digit:
                    errors.append(f"writer/source 격리 위반: {key}={val} in {sorted(splits)}")
        return errors

    @staticmethod
    def validate_synthetic_split(ratios: Dict[str, int]) -> List[str]:
        """합성 데이터 70:15:15 분할 규약 검증 (T3-17)."""
        errors: List[str] = []
        for k in ("train", "dev", "test"):
            if k not in ratios:
                errors.append(f"splits_synthetic.{k} 누락")
        if not errors:
            vals = [ratios.get(k) for k in ("train", "dev", "test")]
            if not all(isinstance(v, int) for v in vals):
                errors.append("splits_synthetic 비율이 정수가 아닙니다.")
            elif sum(vals) != 100:
                errors.append("splits_synthetic 합이 100이어야 합니다.")
            elif ratios.get("train") != 70 or ratios.get("dev") != 15 or ratios.get("test") != 15:
                errors.append("splits_synthetic은 70:15:15여야 합니다.")
        return errors
