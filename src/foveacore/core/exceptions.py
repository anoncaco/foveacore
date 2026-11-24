"""FoveaCore Phase 0 수치·상태 불변성 예외 계약 (T2)."""

import math
from numbers import Real


class NumericalInstabilityError(RuntimeError):
    """수치 불안정(NaN/Inf, 비결정론 결과 등) 검출 시 raise.

    `assert`(`python -O`에서 무시) 대신 명시적 raise를 사용한다(v1.9.3 Phase 0).
    """


class StateMutationError(RuntimeError):
    """INFER 중 모델 파라미터/버퍼/영속 내부 상태 변경 감지.

    T4(ReadOnlyInferenceGuard)에서 강제된다.
    """


class ImmutabilityViolationError(RuntimeError):
    """불변 객체(ImmutableTensor 등) 내부 상태 변경 감지.

    생성·조회 시 `.detach().clone()` 방어 복사 + 스냅샷 기반 감지(T2).
    """


def ensure_finite(value, name="value"):
    """스칼라 수치가 유한한지 명시적으로 검증한다(최적화 모드에서도 유지)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name}는 실수여야 합니다: {value!r}")
    if not math.isfinite(float(value)):
        raise NumericalInstabilityError(f"{name}가 유한하지 않습니다: {value!r}")
    return value


def ensure_finite_tensor(tensor, name="tensor"):
    """Tensor의 모든 원소가 유한한지 검증한다."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}는 torch.Tensor여야 합니다.")
    if not bool(torch.isfinite(tensor).all()):
        raise NumericalInstabilityError(f"{name}에 NaN/Inf가 포함되어 있습니다.")
    return tensor
