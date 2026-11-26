# T1 설정·CLI·센서 프로파일·금지 패키지 구현 문서

> Phase 0 T1의 확정 동작과 검증 기준을 기록합니다. 구현 과정에서 확정된 설정 오류 처리와 호환성 규칙을 코드·테스트와 함께 관리합니다.
> 상세 요구사항은 [`phase0_detail_plan.md`](phase0_detail_plan.md)를 참조합니다.

## 1. 목표와 범위

- Phase 0 실험 설정을 YAML에서 읽고 타입·값·조합을 검증합니다.
- 입력 크기와 센서 프로파일의 호환성을 선등록합니다.
- `validate`, `build-splits`, `smoke-baseline` CLI 진입점과 오류 종료 규칙을 제공합니다.
- 텍스트 모델·LLM 라이브러리가 설치·로드되지 않은 환경을 진입점에서 검사합니다.
- provenance가 프로젝트 루트 기준으로 Git 상태와 잠금 파일을 읽도록 합니다.

## 2. 설정 계약

### 2.1 기본 입력·센서 계약

| 항목 | 고정값 |
|---|---|
| `input.max_side` | `1024` |
| `input.resize_method` | `adaptive_average` |
| 기본 센서 프로파일 | `F3` |
| 기본 `grid_size` | `5` |
| 기본 `channels` | `4` |
| `stroke_clue_available` | `true` |

고해상도 프로파일은 기본 F3와 별개 버전으로 관리합니다.

| 프로파일 | grid_size | channels | 호환성 |
|---|---:|---:|---|
| `F3` | 5 | 4 | 기본 프로파일 |
| `F3_HIRES` | 97 | 4 | 별도 고해상도 프로파일 |

`F3`에 97 격자를 지정하거나 `F3_HIRES`에 5 격자를 지정하는 등의 불일치는 오류입니다. 프로파일 선언과 설정값은 `stroke_clue_available`까지 일치해야 합니다.

### 2.2 프로토콜·정책·Smoke

- 허용 프로토콜: `STRICT_1SHOT`, `CONTINUAL`, `REFERENCE_P3`
- 허용 사전학습 정책: `P0`, `P1`, `P2`, `P3`
- 문자열이 아닌 프로토콜·정책 값은 `TypeError`로 중단하지 않고 설정 오류 목록을 반환합니다.
- `smoke.scope`는 `phase0_full_smoke` 또는 `general`입니다.
- `phase0_full_smoke`의 MNIST 공식 설정은 10-way와 지원 예제 10개를 함께 요구합니다.
- `general` 범위는 일반 3-way·5-way 설정을 허용합니다.
- 일부 필드만 5-way로 바꾼 혼합 설정은 거부합니다.

## 3. 공개 API와 입력·출력

| 모듈 | 함수 / 클래스 | 입력 | 출력 / 역할 |
|---|---|---|---|
| `config.py` | `load_phase0_config()` | YAML 경로 | 파싱된 설정 사전 |
|  | `validate_phase0_config()` | 설정 사전 | 오류 문자열 목록, 빈 목록이 통과 |
|  | `SENSOR_PROFILES` | 없음 | 프로파일별 격자·채널·stroke clue 선언 |
| `forbidden_packages.py` | `assert_no_forbidden_packages()` | 없음 | 금지 패키지가 없으면 통과, 있으면 `RuntimeError` |
| `__main__.py` | `main()` | CLI 인자 | 종료 코드, 설정·데이터 검증 결과 출력 |
| `utils/provenance.py` | `collect_provenance()` | 선택적 시드 | 실행 환경·Git·패키지·결정론 스냅샷 |
|  | `is_git_dirty()` | 없음 | 계약 경로 변경 여부 또는 저장소 밖에서 `None` |

### CLI 계약

| 명령 | 역할 | 실패 시 |
|---|---|---|
| `validate` | 설정과 선택한 데이터 디렉터리의 무결성 검사 | `ERROR:` 출력 후 exit 1 |
| `build-splits` | 설정 검증 후 데이터 분할 생성 | I/O 전에 설정 오류를 차단하고 exit 1 |
| `smoke-baseline` | 향후 baseline 실행 진입점 등록 | 현재 등록 확인만 T1 범위, 실제 실행은 후속 작업 |

YAML 파싱 오류와 금지 패키지·CUBLAS 초기화 오류는 traceback 대신 정리된 `ERROR:` 메시지와 exit 1로 보고합니다.

## 4. Git·provenance 경계

- Git 상태는 실행 CWD가 아니라 프로젝트 루트 기준으로 조회합니다.
- `git_dirty`는 `src/`, `configs/`, `locks/`, `pyproject.toml`의 변경·미추적 파일을 대상으로 합니다.
- 무관한 경로의 변경은 `git_dirty`에 영향을 주지 않습니다.
- Git 저장소 밖에서는 단정적으로 `False`를 반환하지 않고 `None`을 사용합니다.
- Git 상태 조회 설정이 비활성화된 경우에도 미추적 파일을 놓치지 않아야 합니다.
- 의존성 잠금 해시도 서브디렉터리에서 동일하게 계산되어야 합니다.

## 5. 검증 기준

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_cli.py \
  tests/test_forbidden_packages.py tests/test_provenance.py -q
.venv/bin/python -m foveacore validate --config configs/phase0.yaml
.venv/bin/python -m foveacore build-splits --help
```

주요 요구사항과 테스트 대응은 다음과 같습니다.

| 요구사항 | 테스트 근거 |
|---|---|
| 입력·센서 계약 선등록 | `tests/test_config.py:22` |
| max_side·resize_method 검증 | `tests/test_config.py:52`, `tests/test_config.py:60` |
| F3/F3_HIRES 격자·채널 분리 | `tests/test_config.py:90`, `tests/test_config.py:97` |
| 프로파일별 stroke clue 선언 일치 | `tests/test_config.py:124` |
| 공식 10-way·일반 3-way/5-way 구분 | `tests/test_config.py:152` |
| CLI 설정 오류·YAML 오류 처리 | `tests/test_cli.py:28`, `tests/test_cli.py:89` |
| 금지 패키지 설치·로드·예외 진단 | `tests/test_forbidden_packages.py:12` |
| Git 경로·서브디렉터리·미추적 파일 | `tests/test_provenance.py:203` |
| 잠금 해시·서브디렉터리 조회 | `tests/test_provenance.py:117`, `tests/test_provenance.py:146` |

## 6. 알려진 한계와 후속 작업

- T1은 Phase 0 설정과 진입점 계약을 확정하며, 실제 Foveal Sensor 풀링이나 baseline 학습을 구현하지 않습니다.
- `smoke-baseline` 실제 실행과 에피소드 샘플링은 후속 T5/T8 및 관련 작업에서 연결됩니다.
- `build-splits`의 상세 격리·봉인·정답 분리 정책은 T3 문서를 따릅니다.
- GPU 초기화 순서와 결정론·TF32 설정의 세부 검증은 T2 문서를 따릅니다.
