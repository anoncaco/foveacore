# STATE.md

- **현재 단계:** Phase 0 (T3 완료)
- **기준 문서:** FoveaCore 연구·구현 계획서 v1.9.3
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


## T1 완료 노트 (`6cd9193`)

- `configs/phase0.yaml` 작성(STRICT_1SHOT, P1, 분할·Smoke 수량 스키마).
- 스키마 검증기: `foveacore.config.load_phase0_config` / `validate_phase0_config`.
- CLI: `src/foveacore/__main__.py` — `validate`(PASS)/`build-splits`/`smoke-baseline`.
- 금지 패키지 검증기: `foveacore.forbidden_packages`(`transformers`·`openai`·`anthropic`·`llama_cpp`·`vllm`·`langchain` 미임포트 확인).
- 검증: `pytest tests/ -q`(6 passed), `python -m foveacore validate --config configs/phase0.yaml` → `validate PASS` exit=0.

## T2 완료 노트 (현재 커밋)

- `utils/seed.py`: 5종 시드(split/episode/generator/model/stream_order), `CUBLAS_WORKSPACE_CONFIG=:4096:8` 설정·검증, `set_gpu_determinism()`(deterministic_algorithms True/warn_only=False, cudnn deterministic/benchmark, highest precision, TF32 차단), 비결정론 CUDA 연산(`grid_sample`·`scatter_add`) 검출 훅.
- `utils/provenance.py`: `collect_provenance()` — OS/CPU/GPU(`gpu_name`)/CUDA(`cuda_version`)/cuDNN(`cudnn_version`)/PyTorch(`torch_version`)/정밀도(`precision`, `allow_tf32`)/결정론 설정/스레드/Git/패키지 수집.
- `core/exceptions.py`: `NumericalInstabilityError`(assert 대신 명시적 raise), `StateMutationError`, `ImmutabilityViolationError`.
- `core/contracts.py`: v1.9.3 부록 1 6대 인터페이스(Observation/Action/Structure/Hypothesis/Prediction/Memory) — 불변 변환(`MappingProxyType`/`tuple`/read-only ndarray), `ImmutableTensor`(`.detach().clone()` 방어 복사 + SHA-256 스냅샷 변경 탐지), JSON 왕복 직렬화.
- 검증: `pytest tests/ -q`(19 passed). 결정론 플래그: cudnn.deterministic=True, cudnn.benchmark=False, tf32_matmul=False, tf32_cudnn=False, precision=highest, CUBLAS=:4096:8.

## 보류 사항

- T4 — 정책 결합 검증, ReadOnlyInferenceGuard, 스냅샷 복원, 미래 차단 커서, 에피소드 샘플러 (T4 미착수).
