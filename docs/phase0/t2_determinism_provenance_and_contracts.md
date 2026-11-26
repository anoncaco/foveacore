# T2 시드·GPU 결정론·provenance·공통 계약 구현 문서

> Phase 0 T2의 확정 동작과 검증 기준을 기록합니다. 수치 검증과 상태 불변성을 `assert`에 의존하지 않는 계약으로 관리합니다.
> 상세 요구사항은 [`phase0_detail_plan.md`](phase0_detail_plan.md)를 참조합니다.

## 1. 목표와 범위

- 목적별 5종 시드를 분리하고 독립 RNG을 제공합니다.
- CUDA 초기화 전 cuBLAS workspace를 설정하고 결정론·FP32·TF32 차단을 강제합니다.
- 반복 실행의 재현성을 확인하고 비결정론 연산을 진단합니다.
- 실행 환경·시드·Git·패키지·정밀도 정보를 provenance로 수집합니다.
- v1.9.3 부록 1의 6대 인터페이스를 불변으로 만들고 JSON 왕복을 지원합니다.

## 2. 시드 및 GPU 운영 계약

### 2.1 시드 5종

| 시드 | 용도 |
|---|---|
| `split_seed` | 데이터셋·OOD·전이 분할 |
| `episode_seed` | Support/Query 에피소드 샘플링 |
| `generator_seed` | 합성 도형·반례·이동·가림 생성 |
| `model_seed` | 모델 초기화·학습 셔플 |
| `stream_order_seed` | 지속 학습 경험 순서 |

- 시드 값은 `0 ~ 2^64-1`의 정수로 정규화합니다.
- Python·NumPy·Torch RNG를 목적별로 분리합니다.
- 한 목적의 RNG를 소비해도 다른 목적의 RNG 상태는 변하지 않습니다.
- `apply_seeds()`는 실제 적용된 5종 시드 스냅샷을 반환·보관합니다.
- provenance에 명시한 시드는 실제 적용된 값과 일치해야 합니다.

### 2.2 GPU 초기화 순서

1. `ensure_cublas_before_cuda()`에서 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정합니다.
2. CUDA 초기화 이후 workspace 값을 바꾸려는 시도를 거부합니다.
3. `initialize_gpu_runtime()` 또는 `set_gpu_determinism()`에서 deterministic algorithms와 cuDNN·정밀도 설정을 적용합니다.
4. provenance 수집은 runtime 초기화 이후에 수행합니다.

강제 설정은 다음과 같습니다.

```text
torch.use_deterministic_algorithms(True, warn_only=False)
cudnn.deterministic = True
cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CUBLAS_WORKSPACE_CONFIG = :4096:8
```

## 3. 공개 API와 입력·출력

| 모듈 | 함수 / 클래스 | 입력 | 출력 / 역할 |
|---|---|---|---|
| `utils/seed.py` | `apply_seeds()` | 선택적 시드 override | 정규화된 적용 시드 사전 |
|  | `get_applied_seeds()` | 없음 | 마지막 적용 시드의 방어 복사 |
|  | `create_rng()` / `get_rng()` | RNG 종류·시드 이름 | Python/NumPy/Torch 독립 RNG |
|  | `ensure_cublas_before_cuda()` | 없음 | workspace 설정 또는 오류 |
|  | `initialize_gpu_runtime()` | 선택적 model seed | runtime 초기화·결정론 적용 |
|  | `set_gpu_determinism()` | 선택적 model seed | 결정론·정밀도·TF32 설정 |
|  | `gpu_determinism_status()` | 없음 | 현재 결정론 설정 스냅샷 |
|  | `is_gpu_deterministic()` | 없음 | 결정론 설정 일치 여부 |
|  | `find_unstable_ops()` / `assert_no_unstable_ops()` | 사용 연산 이름 | 비결정론 연산 목록 또는 예외 |
|  | `detect_unstable_cuda_ops()` | 없음 | CUDA tensor를 사용하는 최근 연산 중 위험 연산 진단 |
|  | `assert_reproducible()` | operation·인자·반복 횟수 | 반복 결과 일치 또는 예외 |
| `utils/provenance.py` | `collect_provenance()` | 선택적 시드 | 환경·Git·패키지·결정론 JSON 사전 |
| `core/exceptions.py` | `ensure_finite()` / `ensure_finite_tensor()` | 숫자 또는 tensor | 유한값 통과, 비유한값에서 명시적 예외 |
| `core/contracts.py` | `make_image_provenance()` | 이미지·전처리·센서 정보 | 검증된 이미지 provenance |
|  | `immutable_tensor()` | tensor | 방어 복사된 `ImmutableTensor` |
|  | 6대 Contract 클래스 | 필드·provenance | 불변 객체 및 JSON 복원 |

## 4. provenance 계약

`collect_provenance()`는 최소 다음을 수집합니다.

- OS·CPU·스레드 정보
- GPU 이름, CUDA 버전, cuDNN 버전, PyTorch 버전
- FP32 정밀도와 TF32 허용 여부
- deterministic algorithms·cuDNN·CUBLAS 설정
- 실제 적용된 5종 시드
- Git 커밋과 계약 경로의 dirty 여부
- 설치 패키지와 의존성 잠금 해시

Git 조회와 잠금 해시 계산은 CWD와 무관해야 합니다. runtime 초기화 전에 GPU 정보를 질의하지 않으며, 적용되지 않은 시드를 provenance에 주입하면 오류입니다.

## 5. 공통 계약 및 불변성

6대 인터페이스는 다음과 같습니다.

```text
Observation
Action
Structure
Hypothesis
Prediction
Memory
```

공통 규칙은 다음과 같습니다.

- `schema_version = "1.0"`을 유지합니다.
- dict는 `MappingProxyType`, list/set은 불변 컨테이너로 변환합니다.
- NumPy 배열은 방어 복사 후 write-protect합니다.
- object dtype 배열과 비JSON 타입을 무단 직렬화하지 않습니다.
- Torch tensor는 `.detach().clone()` 방어 복사와 SHA-256 스냅샷으로 내부 변경을 탐지합니다.
- 내부 상태 변경이 감지되면 `ImmutabilityViolationError`를 발생시킵니다.
- 숫자 경계 검증은 `python -O`에서도 유지되도록 `assert` 대신 명시적 예외를 사용합니다.

### 이미지 Observation

- provenance에 `original_hw`, `processed_hw`, `preprocess_profile_id`를 포함합니다.
- 위치 좌표는 `[0,1]` 정규화를 사용합니다.
- `sensor_profile`과 `grid_size`로 features shape를 검증합니다.
- `F3`와 `F3_HIRES`는 서로 호환되지 않는 별개 프로파일로 관리합니다.
- `stroke_clue_available`는 관측 provenance에 중복 저장하지 않고 `SENSOR_PROFILE_CONTRACTS`에서 조회합니다.
- JSON 왕복 후에도 `SENSOR_PROFILE_CONTRACTS[obs.provenance["sensor_profile"]].stroke_clue_available`가 동일해야 합니다.

## 6. 검증 기준

```bash
.venv/bin/python -m pytest \
  tests/test_seed.py \
  tests/test_provenance.py \
  tests/test_exceptions.py \
  tests/test_contracts.py -q
.venv/bin/python -O -m pytest tests/test_exceptions.py -q
```

주요 요구사항과 테스트 근거는 다음과 같습니다.

| 요구사항 | 테스트 근거 |
|---|---|
| 5종 시드·독립 RNG·스냅샷 | `tests/test_seed.py:53`, `tests/test_seed.py:68`, `tests/test_seed.py:89` |
| CUBLAS 초기화 순서·변경 거부 | `tests/test_seed.py:132`, `tests/test_seed.py:145` |
| deterministic algorithms·cuDNN·TF32 | `tests/test_seed.py`의 결정론 관련 테스트 |
| 비결정론 연산·반복 실행 진단 | `tests/test_seed.py`의 `detect_unstable_cuda_ops`, `assert_reproducible` 테스트 |
| provenance 필수 필드·적용 시드 일치 | `tests/test_provenance.py:75`, `tests/test_provenance.py:95` |
| Git·잠금 해시의 루트 기준 조회 | `tests/test_provenance.py:117`, `tests/test_provenance.py:203` |
| 수치 검증의 최적화 모드 내성 | `tests/test_exceptions.py:44` |
| 6대 계약 JSON 왕복·스키마 검증 | `tests/test_contracts.py:241`, `tests/test_contracts.py:381` |
| Tensor·ndarray 불변성과 변경 탐지 | `tests/test_contracts.py:315`, `tests/test_contracts.py:430` |
| 이미지 provenance·프로파일·stroke clue | `tests/test_contracts.py:121`, `tests/test_contracts.py:538` |

## 7. 알려진 한계와 후속 작업

- CUDA 디바이스가 없는 환경에서는 실제 GPU 연산 재현성을 완전히 검증할 수 없습니다. GPU 사용 가능 환경에서의 결정론 테스트를 별도로 유지해야 합니다.
- Phase 0에서 정의한 계약은 Phase 1~5의 모델·센서·스트림 구현이 따라야 할 인터페이스이며, 해당 본체 구현은 후속 Phase 책임입니다.
- 정책 결합 검증, 읽기 전용 추론 guard, 전체 상태 스냅샷 복원, 미래 접근 차단 커서는 T4 이후 범위입니다.
- 실제 GPU 풀링·센서 통합은 Phase 1 게이트에서 검증합니다.
