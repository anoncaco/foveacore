# Phase 0 문서

Phase 0의 기준 계획과 Task별 확정 설계·검증 문서를 모아둡니다.

| 문서 | 역할 |
|---|---|
| [`phase0_detail_plan.md`](phase0_detail_plan.md) | Phase 0 전체 원칙·결정 잠금·T0~T5 요구사항과 완료 기준 |
| [`t0_project_foundation.md`](t0_project_foundation.md) | 패키지·src 레이아웃·의존성 잠금·추적 규칙 |
| [`t1_configuration_cli_and_profiles.md`](t1_configuration_cli_and_profiles.md) | 설정 검증·CLI·센서 프로파일·금지 패키지·Git provenance |
| [`t2_determinism_provenance_and_contracts.md`](t2_determinism_provenance_and_contracts.md) | 5종 시드·GPU 결정론·provenance·6대 공통 계약 |
| [`t3_data_splits_and_seal_guard.md`](t3_data_splits_and_seal_guard.md) | T3 재작성 설계·분할 격리·접근제어·동결 해시·수용 테스트 |

문서의 해석 우선순위는 다음과 같습니다.

1. `phase0_detail_plan.md`와 상위 연구 계획서의 요구사항
2. Task별 설계 문서에 명시된 수용 기준
3. 코드와 테스트에서 확인되는 실제 구현 동작

요구사항과 구현이 달라진 경우에는 임의로 명세를 바꾸지 않고, 해당 Task 문서와 테스트를 함께 검토·갱신합니다.
