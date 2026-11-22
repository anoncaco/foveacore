"""FoveaCore Phase 0 수치·상태 불변성 예외 계약 (T2)."""


class NumericalInstabilityError(RuntimeError):
    """수치 불안정(NaN/Inf 등) 검출 시 raise.

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
