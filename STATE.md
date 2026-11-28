# STATE.md

- **현재 단계:** Phase 0 (C-Gate·G1-Gate PASS)
- **기준 문서:** FoveaCore 연구·구현 계획서 v1.9.3
- **Phase 0 문서:** [`docs/phase0/README.md`](docs/phase0/README.md)
- **버전:** 0.1.0

## 파이썬 환경 규칙

- 라이브러리는 프로젝트 루트의 `.venv`를 사용한다(pip/실행은 `.venv/bin/...`).

## Git 전략

- `git_dirty` = `git status --porcelain` 기준 `src/`, `configs/`, `locks/`, `pyproject.toml` 내 수정·미추적(untracked) 파일 존재 시 True.

## Git 운영 규칙

### 브랜치 흐름 (단방향)

```
dev (구현) → verify (검증/수정) → main (최종 통합)
```

- `dev`: T0~T4 등 단계별 기능 개발 브랜치
- `verify`: `dev`에서 분기하여 검증 및 수정 수행
- `main`: 검증 통과 후 최종 통합만 수행 (보호 브랜치)

### 커밋 규칙

- 한 단계(T0, T1, T2, T3...) = 최소 1커밋
- 여러 태스크를 한 커밋에 스쿼시하지 않음
- 기준 커밋 해시는 "해당 코드가 확정된 직전 커밋 해시"를 기록 (커밋 직후 해시가 아님)

### 변경 영수증 양식 (각 Task 완료 시 STATE.md에 기록)

```
### T{n} 완료 노트 (`<직전 커밋 해시>`)

기준 커밋: `git rev-parse --short HEAD` (직전 커밋)

생성/수정 파일 (`git show --stat <커밋>` 결과):
- `<파일 경로 1>`
- `<파일 경로 2>`

요구사항 구현 방식 구분:
- [코드·테스트 반영] — 실제 .py 및 tests/에 구현된 항목
- [수동 기록 / 후속 이관] — 아직 코드에 연결 안 되고 텍스트만 적은 항목

검증 명령 & 실제 결과:
- `<명령>` → `<결과>`
```


## T0 완료 노트

- 기준 커밋: `45e1eee` (현재 HEAD, T0~T3 포함)
- 관련 커밋: `05b9c2b` (feat(phase0): complete phase 0 (T0-T3) all hardening)
- `pip install -e .` 완료, `import foveacore` 검증(0.1.0, `src/foveacore/__init__.py`).
- `locks/phase0_requirements.lock`(39 lines), `dependency_lock_hash` = `7afc502593648a28ef09c8bd964340ae75d1d41e1ac0e5e3d3b2485cce877ce1` (editable 경로 상대화 후 재산출).

### T0 산출물 파일 목록

| # | 파일 | T0 역할 | 검증 방법 |
|---|------|---------|-----------|
| 1 | `pyproject.toml` | 서브패키지 자동 탐색, 버전 `0.1.0` | `[tool.setuptools.packages.find] where=["src"]` |
| 2 | `locks/phase0_requirements.lock` | 의존성 잠금 (39줄, `-e .` 상대 경로) | `sha256sum` 일치 |
| 3 | `.gitignore` | Python/캐시/환경 제외, 데이터 예외 | `git check-ignore` |
| 4 | `README.md` | 연구 계획서 `v1.9.3` 표기 | `v1.9.3` 표기 존재 |
| 5 | `STATE.md` | 단계·버전·완료 노트 | 파일 존재, Git 추적 중 |


## T1 재구축 완료 노트 (`d9a338c`)

- 기준 커밋: `d9a338c` (재구축 직전 HEAD).
- 커밋: `fix(phase0): T1 재구축`.
- 변경 파일:
  - `STATE.md`
  - `configs/phase0.yaml`
  - `src/foveacore/config.py`
  - `src/foveacore/__main__.py`
  - `src/foveacore/forbidden_packages.py`
  - `src/foveacore/utils/provenance.py`
  - `tests/test_config.py`
  - `tests/test_cli.py` (신규)
  - `tests/test_forbidden_packages.py`
  - `tests/test_provenance.py`

### 요구사항 구현 방식 구분

- [코드·테스트 반영] `input.max_side=1024`, `input.resize_method=adaptive_average`, `sensor.profile=F3`, `sensor.grid_size=5`, `sensor.channels=4`를 설정에 선등록하고 스키마 검증 추가.
- [코드·테스트 반영] 고해상도 `grid_size=97`을 별도 `F3_HIRES` 프로파일로 분리하고 F3와의 격자·채널 불일치를 거부.
- [코드·테스트 반영] `experiment_protocol`/`pretraining_policy`의 비문자열 값도 `TypeError` 없이 설정 오류로 반환.
- [코드·테스트 반영] `validate`, `build-splits`, `smoke-baseline` 진입점 및 오류 종료 처리 검증. `smoke-baseline` 실제 실행은 [후속 이관] T5/T8.
- [코드·테스트 반영] `git_dirty`를 프로젝트 루트 기준 `git status --porcelain`로 수정해 서브디렉터리에서도 `src/`, `configs/`, `locks/`, `pyproject.toml` 변경을 감지.
- [코드·테스트 반영] 이미 로드된 금지 패키지(`__spec__=None`)와 `find_spec` 예외를 안전하게 진단하고 CLI는 `ERROR:`와 exit 1로 처리.

### 요구사항 ↔ 테스트

| 요구사항 | 테스트 |
|----------|--------|
| 입력·센서 계약 선등록 | `tests/test_config.py::test_input_sensor_contract_registered` |
| max_side/resize_method 검증 | `test_input_max_side_must_be_1024`, `test_input_resize_method_is_validated` |
| F3 격자·채널 및 고해상도 프로파일 분리 | `test_f3_profile_rejects_mismatched_contract`, `test_high_resolution_profile_is_separate` |
| CLI 진입점·설정 오류 | `tests/test_cli.py` |
| git_dirty 경로·서브디렉터리 | `tests/test_provenance.py::test_git_dirty_*` |
| 금지 패키지 설치·로드 진단 | `tests/test_forbidden_packages.py` |

### 검증 명령 & 실제 결과

- `.venv/bin/python -m pytest tests/ -q` → **85 passed, 1 skipped** (exit 0, 로그: `tmp_ai/logs/t1_rebuild_pytest.log`).
- `.venv/bin/python -m foveacore validate --config configs/phase0.yaml` → `validate PASS`, exit 0.
- 잘못된 `max_side=0`, `resize_method=nearest`, F3+97 격자, F3+3채널, `smoke.way=9`, `experiment_protocol=[]` → 모두 `ERROR:`와 exit 1 (로그: `tmp_ai/logs/t1_rebuild_cli.log`).
- `tests/` CWD에서 `src/` 미추적 파일 생성 → `git_dirty=True`.
- 로드된 `openai` 모듈의 `__spec__=None` → `RuntimeError` 진단.
- `git diff --check` → 통과(exit 0).

## T2 완료 노트 (현재 커밋)

- `utils/seed.py`: 5종 시드(split/episode/generator/model/stream_order), `CUBLAS_WORKSPACE_CONFIG=:4096:8` 설정·검증, `set_gpu_determinism()`(deterministic_algorithms True/warn_only=False, cudnn deterministic/benchmark, highest precision, TF32 차단), 비결정론 CUDA 연산(`grid_sample`·`scatter_add`) 검출 훅.
- `utils/provenance.py`: `collect_provenance()` — OS/CPU/GPU(`gpu_name`)/CUDA(`cuda_version`)/cuDNN(`cudnn_version`)/PyTorch(`torch_version`)/정밀도(`precision`, `allow_tf32`)/결정론 설정/스레드/Git/패키지 수집.
- `core/exceptions.py`: `NumericalInstabilityError`(assert 대신 명시적 raise), `StateMutationError`, `ImmutabilityViolationError`.
- `core/contracts.py`: v1.9.3 부록 1 6대 인터페이스(Observation/Action/Structure/Hypothesis/Prediction/Memory) — 불변 변환(`MappingProxyType`/`tuple`/read-only ndarray), `ImmutableTensor`(`.detach().clone()` 방어 복사 + SHA-256 스냅샷 변경 탐지), JSON 왕복 직렬화.
- 검증: `pytest tests/ -q`(19 passed). 결정론 플래그: cudnn.deterministic=True, cudnn.benchmark=False, tf32_matmul=False, tf32_cudnn=False, precision=highest, CUBLAS=:4096:8.

## T3 결함 수정 완료 노트 (`61b4e5b`)

- 기준 커밋: `9fb9a91` (`verify`, 결함 수정 직전 HEAD).
- 커밋·푸시: `61b4e5b`, `origin/verify`.
- 수정 결함:
  - P1: resolver의 policy 오버라이드 guard 우회 차단.
  - P2: 등록 시 전역 `sample_id` 중복 거부 및 query 전역 등록 검증·요청 ID 겹침 시 차단.
  - P3: `split_manifest_v1.json` 내 라벨·기밀 필드 주입 검증 추가.
  - P4: 중첩 `MappingProxyType`/tuple 메타데이터 Canonical JSON 정규화.
  - P5: pHash 근접 후보 검색을 Hamming 거리 BK-tree로 전환해 버킷 누락 제거.
  - P6: 실제 MNIST 공개 엔트리 70,000개의 `split_name`을 상위 manifest와 일치시켜 재생성.
  - P7: MNIST `writer_id`를 `None`으로 미확인 표기하고, OOD 8쌍에서는 writer 미확인 시 fail-closed 처리.
- 생성/수정 파일:
  - `src/foveacore/__main__.py`
  - `src/foveacore/data/access_control.py`
  - `src/foveacore/data/ood_validator.py`
  - `src/foveacore/data/seal_guard.py`
  - `src/foveacore/data/splits.py`
  - `src/foveacore/data/validation_metadata.py`
  - `tests/test_access_control.py`, `tests/test_data_validation_cli.py`, `tests/test_seal_guard.py`, `tests/test_split_isolation.py`, `tests/test_splits.py`
  - `data/split_{support_pool,dev_stream,main_stream,official_test,reserved_sealed}.json`
- 검증 명령 & 실제 결과:
  - `.venv/bin/python -m pytest -q` → **406 passed** (589.16초, exit 0).
  - `.venv/bin/python -m foveacore validate --config configs/phase0.yaml --data-dir data` → `validate PASS`, exit 0.
  - `.venv/bin/python -m foveacore build-splits --out data` → 70,000 엔트리 재생성, exit 0.
  - BK-tree 무작위 30세트·threshold 0/1/3/10 brute-force 대조 → PASS.
  - `git diff --check` → PASS.
- 독립 감사: `auditor` 3개 단위 요청했으나 공급자 무료 사용량 한도·동시 요청 600 초과·모델 unavailable로 미실행. 메인 모델 fallback 자체 검증만 완료.

## T3 접근제어 후속 보강 (`61b4e5b`, 미커밋)

- 정상 support 요청은 다른 split에 등록된 미확정 query 때문에 차단되지 않도록, query는 전역 등록만 확인하고 요청 `sample_ids`와 겹칠 때만 차단.
- EVALUATE는 실제 `SealGuard`와 `VerificationRecord`만 허용하고, `SealGuard.assert_access()` 기본 구현을 직접 호출해 가드 오버라이드 우회를 차단.
- 생성/수정 파일: `src/foveacore/data/access_control.py`, `tests/test_access_control.py`, `STATE.md`.
- 검증: `.venv/bin/python -m pytest -q` → **408 passed** (596.38초, exit 0), `git diff --check` → PASS.
- 독립 감사는 재실행하지 않았고, 해당 후속 수정은 미커밋 상태.

## Phase 0 공통 Gate 검증 (2026-11-29)

- **검증 대상:** verify `2f07e0976f762d65310a409aa0580b0917a59880`
- **기준 working_rules.md SHA-256:** `c32f631d7dc8ee92f7ff21ca8e28b9944eff09b8964e598dadc6b8a9307e1561`
- **대조 범위:** 문서·Git 이력·산출물 존재 여부 대조
- **종합 판정:** **C-Gate PASS**

| 게이트 | 항목 | 결과 |
|---|---|---|
| C-01 | Phase0/REQ-P0-01~10/v1.9.3/target 확인 | PASS |
| C-02 | Index 25종 O/X 완료·필수 문서 존재·X 문서 부재 | PASS |
| C-03 | origin/verify..verify 변경이 .gitignore와 docs/phase0에만 존재 | PASS |
| C-04 | R3 변경 이력에 수용 기준 변경 없음·사용자 지시에 따른 재구성 | PASS |
| C-05 | 진행을 막는 미확정 사항 없음·auditor 미완료는 별도 검증 한계 | PASS |
| C-06 | 대상 커밋과 로컬 파일 해시 식별·별도 검증 환경 미사용 | PASS |
| C-07 | 커밋·머지·푸시가 사용자 지시에 따른 것과 실제 해시 확인 | PASS |

- **한계:** 독립 auditor 감사 미완료.
- **관찰 사항:** R3의 REQ-P0-07~10 상태가 설계 전으로 남아 실제와 불일치.
- **기록 전 Git 상태:** verify, tracked clean.

## Phase 0 1단계 Gate 재검증 (2026-11-29)

- **검증 대상:** verify `9a6b0b38326b82be03858dc88e249df574d6001b` + 미커밋 `R3.md` 변경(SHA-256 `aaf703ccdefb4588b972e7edd2fa5e312accc44a5f5ce2c3fd65262a4b9e6984`) 및 본 `STATE.md` 기록
- **검증 기준:** `working_rules.md` `G1-01`~`G1-07`
- **대조 범위:** Index·R3·`v1.9.3` Phase 0 본문 및 부록 1·4·5·9·작업 이력
- **종합 판정:** **G1-Gate PASS**

| 게이트 | 항목 | 결과 |
| :--- | :--- | :--- |
| G1-01 | Index와 요구사항 문서 존재 | PASS |
| G1-02 | 산출물 전 항목 O/X·사유·위치 기록 | PASS |
| G1-03 | Phase 0 요구사항 전수 추출·타 Phase 범위 미혼입 | PASS |
| G1-04 | 출처·요약·수용 기준·예정 연결 위치·요구사항별 선행 의존성·참조 계약·기존 상태 기록 | PASS |
| G1-05 | 선행 계약·공통 제약·외부 라이브러리 제약 기록 | PASS |
| G1-06 | 진행을 막는 미확정 사항 없음 명시 | PASS |
| G1-07 | 소급 재구성 예외 적용·무단 기준 변경 없음 | PASS |

- **G1-07 소급 재구성 예외:** Phase 0은 기존 구현 위에 문서를 사후 재구성한 사례이므로 1·2·3단계 산출물이 함께 존재한다. 대화 이력을 근거로 PASS 처리하며, Git 이력만으로 단계 수행 시점을 입증할 수 없다는 한계가 있다. 신규 Phase부터는 단계별 검증 후 커밋을 분리한다.
- **한계:** 독립 auditor 감사 미완료.
- **Git 상태:** verify, `R3.md`·`STATE.md` 미커밋, 커밋·푸시 없음.

## 보류 사항

- T4 — 정책 결합 검증, ReadOnlyInferenceGuard, 스냅샷 복원, 미래 차단 커서, 에피소드 샘플러 (T4 미착수).
