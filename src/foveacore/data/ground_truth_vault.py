"""FoveaCore Phase 0 ground-truth vault (T3).

ground_truth_vault.json은 sample_id -> label을 데이터 준비 단계에서 격리 저장한다.
런타임 라벨 소비는 data.access_control.resolve_labels()만 사용한다.
"""

import json
from pathlib import Path
from typing import Any, Dict


class GroundTruthVaultError(ValueError):
    """Ground-truth vault 값 검증 오류."""


def _required_vault_string(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise GroundTruthVaultError(f"{field_name}은 비어 있지 않은 문자열이어야 합니다.")


class GroundTruthVault:
    """sample_id -> label 격리 저장소. 공개 manifest에는 라벨을 저장하지 않는다."""

    __slots__ = ("_vault",)

    def __init__(self, vault: Dict[str, str] | None = None):
        if vault is None:
            vault = {}
        if not isinstance(vault, dict):
            raise GroundTruthVaultError("vault는 dict여야 합니다.")
        self._vault: Dict[str, str] = {}
        for sample_id, label in vault.items():
            sid = _required_vault_string(sample_id, "sample_id")
            lab = _required_vault_string(label, "label")
            self._vault[sid] = lab

    @classmethod
    def load(cls, path: Path | str) -> "GroundTruthVault":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GroundTruthVaultError("ground-truth vault JSON 파싱 실패") from exc
        if not isinstance(data, dict):
            raise GroundTruthVaultError("최상위 vault 값은 dict여야 합니다.")
        labels = data.get("labels", data)
        if not isinstance(labels, dict):
            raise GroundTruthVaultError("labels는 dict여야 합니다.")
        return cls(labels)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"labels": self._vault},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def set(self, sample_id: str, label: str) -> None:
        sid = _required_vault_string(sample_id, "sample_id")
        lab = _required_vault_string(label, "label")
        self._vault[sid] = lab

    def contains(self, sample_id: str) -> bool:
        sid = _required_vault_string(sample_id, "sample_id")
        return sid in self._vault

    def _read_labels(self, sample_ids: frozenset[str]) -> Dict[str, str]:
        if not isinstance(sample_ids, frozenset):
            raise GroundTruthVaultError("sample_ids는 frozenset이어야 합니다.")
        missing = sample_ids.difference(self._vault)
        if missing:
            raise GroundTruthVaultError(f"vault에 없는 sample_id: {sorted(missing)}")
        return {sample_id: self._vault[sample_id] for sample_id in sorted(sample_ids)}