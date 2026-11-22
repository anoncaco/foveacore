"""ground_truth_vault.py 정답 저장소 + ADAPT 접근 제어 테스트 (T3)."""

import json
from pathlib import Path

import pytest

from foveacore.data.ground_truth_vault import (
    AdaptAccessGuard,
    EXECUTION_MODES,
    GroundTruthVault,
)


def test_vault_save_load_roundtrip(tmp_path: Path):
    p = tmp_path / "ground_truth_vault.json"
    vault = GroundTruthVault({"mn-000001": "5", "mn-000002": "3"})
    vault.save(p)
    loaded = GroundTruthVault.load(p)
    assert loaded._get("mn-000001") == "5"
    assert loaded._get("mn-000002") == "3"
    assert loaded.contains("mn-000001")
    assert not loaded.contains("mn-999999")


def test_infer_label_access_blocked():
    vault = GroundTruthVault({"mn-000001": "5"})
    guard = AdaptAccessGuard(vault)
    result = guard.may_access_label("INFER", "support_pool", ["mn-000001"])
    assert result == {"mn-000001": None}


def test_adapt_allowed_split_access():
    vault = GroundTruthVault({"mn-000001": "5"})
    guard = AdaptAccessGuard(vault)
    guard.register_split("support_pool", ["mn-000001"])
    result = guard.may_access_label("ADAPT", "support_pool", ["mn-000001"])
    assert result == {"mn-000001": "5"}


def test_adapt_blocked_split_access():
    vault = GroundTruthVault({"mn-000001": "5"})
    guard = AdaptAccessGuard(vault)
    for blocked in ("official_test", "reserved_sealed"):
        result = guard.may_access_label("ADAPT", blocked, ["mn-000001"])
        assert result == {"mn-000001": None}, f"{blocked} must be blocked"


def test_assert_no_blocked_access_raises():
    vault = GroundTruthVault({"mn-000001": "5"})
    guard = AdaptAccessGuard(vault)
    with pytest.raises(RuntimeError):
        guard.assert_no_blocked_access("INFER", "support_pool", ["mn-000001"])
