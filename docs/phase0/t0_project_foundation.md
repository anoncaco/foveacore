# T0 프로젝트 기반·의존성 잠금 구현 문서

> Phase 0 T0의 확정 동작과 검증 기준을 기록합니다. 코드를 대체하는 문서가 아니라 구현 계약과 검증 근거를 제공합니다.
> 상세 요구사항은 [`phase0_detail_plan.md`](phase0_detail_plan.md)를 참조합니다.

## 1. 목표와 범위

- Python 패키지를 `src/` 레이아웃으로 설치 가능하게 구성합니다.
- 연구 계획서 v1.9.3과 프로젝트 버전을 연결합니다.
- Phase 0 의존성을 고정하고 설치 환경의 재현성을 확보합니다.
- 캐시·가상환경·에이전트 작업 파일을 Git 추적에서 제외합니다.

T0는 프로젝트 기반과 환경 재현성만을 다루며, 설정·CLI·GPU 결정론의 세부 정책은 각각 T1·T2 문서를 참조합니다.

## 2. 확정 동작

| 항목 | 구현 내용 | 근거 |
|---|---|---|
| 패키지 정체성 | 프로젝트 이름 `foveacore`, 버전 `0.1.0`, Python `>=3.10` | `pyproject.toml:5` |
| src 레이아웃 | `package-dir = {"" = "src"}`, `src` 아래 자동 패키지 탐색 | `pyproject.toml:11` |
| 빌드 백엔드 | `setuptools.build_meta`, `setuptools>=61` | `pyproject.toml:1` |
| 의존성 잠금 | `locks/phase0_requirements.lock`에 현재 환경의 고정 패키지와 editable 설치 항목 기록 | `locks/phase0_requirements.lock:1` |
| editable 설치 | 프로젝트 루트에서 `-e .`로 설치하며 잠금 파일에도 상대 경로 유지 | `locks/phase0_requirements.lock:5` |
| 추적 제외 | Python 캐시·가상환경·데이터 캐시와 에이전트 임시 파일을 Git에서 제외 | `.gitignore`, `tests/test_provenance.py:135` |
| 연구 기준 명시 | README에 연구 계획서 v1.9.3과 현재 Phase 0 상태를 명시 | `README.md:9` |

## 3. 공개 API와 산출물

| 파일 | 역할 | 입력 → 출력 |
|---|---|---|
| `pyproject.toml` | 패키지·빌드 메타데이터 정의 | 프로젝트 메타데이터 → 설치 가능한 배포 패키지 |
| `src/foveacore/__init__.py` | 루트 패키지 진입점 | import → 버전이 설정된 `foveacore` 모듈 |
| `locks/phase0_requirements.lock` | Phase 0 설치 환경 스냅샷 | 설치 환경 → 고정 의존성 목록 |
| `.gitignore` | Git 추적 범위 정의 | 작업 파일 경로 → 추적 제외 규칙 |
| `README.md` | 프로젝트 개요·문서·실행 진입점 안내 | 프로젝트 구조 → 사용자 안내 |
| `STATE.md` | 단계·버전·완료 이력·보류 사항 | 작업 결과 → 프로젝트 상태 기록 |

## 4. 운영 규칙

- 라이브러리 설치는 프로젝트 루트의 `.venv`에서 수행합니다.
- 잠금 파일을 변경한 경우 `dependency_lock_hash`도 함께 갱신합니다.
- 잠금 파일의 editable 경로는 절대 경로가 아닌 프로젝트 루트 기준 상대 경로를 유지합니다.
- 원본 데이터·실험 산출물·개인 환경 파일은 Git에 커밋하지 않습니다.
- 의존성 변경은 별도 검증 후 반영하며, 설치 환경과 잠금 파일이 어긋난 상태로 작업을 완료하지 않습니다.

## 5. 검증 기준

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -c "import foveacore; print(foveacore.__version__)"
sha256sum locks/phase0_requirements.lock
.venv/bin/python -m pytest \
  tests/test_provenance.py::test_dependency_lock_hash_present \
  tests/test_provenance.py::test_dependency_lock_hash_matches_lock_file \
  tests/test_provenance.py::test_t0_gitignore_excludes_agent_files -q
```

통과 조건은 다음과 같습니다.

- editable 설치와 `import foveacore`가 성공합니다.
- 패키지 버전이 `0.1.0`입니다.
- 잠금 파일 해시가 provenance가 계산한 `dependency_lock_hash`와 일치합니다.
- `.gitignore`가 에이전트 작업 파일과 환경 부산물을 제외합니다.
- Git diff 검사에서 공백 오류가 없습니다.

## 6. 알려진 한계와 후속 작업

- 잠금 파일은 설치 환경의 스냅샷이므로 Python·플랫폼·CUDA 환경이 달라지면 동일 잠금의 재현성이 보장되지 않을 수 있습니다.
- GPU 연산의 결정론과 수치 정밀도 정책은 T0 범위가 아니며 T2에서 검증합니다.
- 설정 스키마·CLI·센서 프로파일·금지 패키지 정책은 T1에서 다룹니다.
- MNIST 원본 확보와 체크섬 검증은 T3의 실제 분할 인수 테스트에서 연결됩니다.
