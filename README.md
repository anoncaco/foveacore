<!-- 지시: 구현 및 검증된 기능만 기록한다.
설치 및 실행 방법이 변경되면 갱신한다. -->

# FoveaCore

중심와 능동 시각과 내부 시뮬레이션 기반
초경량 1-Shot 인지 모델 연구 프로젝트.

## 연구 명세
- 연구 계획서 v1.9.3

## 프로젝트 구조 및 생성 문서

Phase 0~7 동안 생성되는 주요 문서·산출물:

| 문서 | 역할 | 생성 시점 |
|---|---|---|
| `README.md` (이 문서) | 프로젝트 개요·구조·생성 문서 포인터 | T0 |
| `STATE.md` | 현재 단계·버전(v1.9.3)·보류 사항 | T0 |
| `locks/phase0_requirements.lock` | T0 시점 의존성 `pip freeze` 스냅샷 | T0 |
| `configs/phase0.yaml` | Phase 0 통합 설정(seed·분할·Smoke 규격) | T0/T1 |
| `data/raw/` | MNIST 원본 (SHA-256 검증 필수) | T3 |
| `artifacts/<run_id>/` | run별 manifest·trace·events | T2~T8 |
| `tests/` | pytest 검증 스위트 | T1/T2~T8 |

## 개발 환경
- Python 가상환경: .venv
- 현재 단계: Phase 0
