# STATE.md

- **현재 단계:** Phase 0 (T3 완료)
- **기준 문서:** FoveaCore 연구·구현 계획서 v1.9.3
- **버전:** 0.1.0

## 파이썬 환경 규칙

- 라이브러리는 프로젝트 루트의 `.venv`를 사용한다(pip/실행은 `.venv/bin/...`).

## Git 전략

- `git_dirty` = `git status --porcelain` 기준 `src/`, `configs/`, `locks/`, `pyproject.toml` 내 수정·미추적(untracked) 파일 존재 시 True.

## T0 완료 노트

- Git baseline commit `b699237`(13 files), 후 `chore(phase0): restore repo, lock gpu env & base CLI`(99b6342).
- `pip install -e .` 완료, `import foveacore` 검증(0.1.0, `src/foveacore/__init__.py`).
- `locks/phase0_requirements.lock`(39 lines), `dependency_lock_hash` = `7afc502593648a28ef09c8bd964340ae75d1d41e1ac0e5e3d3b2485cce877ce1` (editable 경로 상대화 후 재산출).

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
