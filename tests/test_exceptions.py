"""core/exceptions.py 명시적 수치 불안정 검증 테스트 (T2)."""

import subprocess
import sys

import pytest
import torch

from foveacore.core.exceptions import (
    NumericalInstabilityError,
    ensure_finite,
    ensure_finite_tensor,
)


def test_ensure_finite_accepts_finite_scalar():
    assert ensure_finite(1.5, "confidence") == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ensure_finite_rejects_non_finite_scalar(value):
    with pytest.raises(NumericalInstabilityError):
        ensure_finite(value)


@pytest.mark.parametrize("value", [True, "1.0", None])
def test_ensure_finite_rejects_non_real_scalar(value):
    with pytest.raises(TypeError):
        ensure_finite(value)


def test_ensure_finite_tensor_rejects_nan_and_inf():
    with pytest.raises(NumericalInstabilityError):
        ensure_finite_tensor(torch.tensor([1.0, float("nan")]))
    with pytest.raises(NumericalInstabilityError):
        ensure_finite_tensor(torch.tensor([1.0, float("inf")]))


def test_ensure_finite_tensor_accepts_finite_tensor():
    tensor = torch.tensor([1.0, 2.0])
    assert ensure_finite_tensor(tensor) is tensor


def test_finite_validation_survives_python_optimized_mode():
    code = (
        "from foveacore.core.exceptions import ensure_finite; "
        "ensure_finite(float('nan'))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "NumericalInstabilityError" in result.stderr
