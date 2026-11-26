# T3 데이터 분할·정답 분리·봉인 해제 설계 및 수용 기준

> Phase 0 T3의 정식 설계 문서입니다. 구현·테스트·리뷰 및 회귀 검증의 기준으로 사용합니다.
> 기준 문서: `foveal_active_vision_1shot_research_plan_v1.9.3.md` 및 `phase0_detail_plan.md`의 T3입니다.

## 1. 구현 목표

기존 dev 구현을 참고하되 다음 네 경계를 중심으로 재작성합니다.

1. 공개 매니페스트와 정답·검증 메타데이터 분리
2. MNIST 및 OOD·전이 분할의 정확한 격리 검증
3. 프로토콜·현재 episode·승인 범위를 반영한 라벨 접근제어
4. 독립 기준점·실제 파일 해시·평가 승인을 모두 확인하는 봉인 해제

### 완료 산출물

- 10종 분할 스키마와 결정론적 MNIST 분할
- 라벨이 없는 공개 매니페스트 및 분리된 정답 저장소
- 전체 분할 ID·이미지 해시 중복 검사
- OOD·전이 8쌍의 클래스·writer·source 격리 검사
- 공통 차단 조건이 우선인 ADAPT 라벨 접근제어
- Canonical JSON 기반 동결 해시 및 독립 기준점 검증
- 평가별 승인 분할만 읽기 전용으로 여는 SealGuard
- 기존 `validate`, `build-splits` CLI와의 호환성
- 정상·부정 테스트 및 실제 MNIST 분할 재현성 테스트

## 2. 분할 정의와 스키마

### 2.1 필수 등록 분할 10종

```text
MNIST:
  support_pool      초기 Support 풀 10,000
  dev_stream        개발 스트림 15,000 (CONTINUAL 전용)
  main_stream       본 실험 학습 스트림 35,000
  official_test     제1 공식 평가 7,000 (Phase 6~7)
  reserved_sealed   제2 예비 봉인 3,000 (후속 독립 확증)

비숫자/OOD/전이:
  p2_non_digit
  non_digit_dev     엄격 1-shot 보정용
  dev_ood
  test_ood
  phase7_char_transfer (10클래스)
```

빈 OOD/전이 슬롯도 등록된 유효한 manifest입니다. Phase 0에서는 합성·비숫자 fixture로 검증기를 테스트하고, 실제 EMNIST/Omniglot 데이터 등록 시 동일 검증기를 적용합니다.

### 2.2 공개 `SplitEntry` 필수 필드

```text
sample_id
split_name
sha256
phash
sealed_level
```

- `phash`는 유사 이미지 중복 후보 검사용으로 반드시 보존합니다.
- 공개 엔트리에 `label`, `class_id`, `writer_id`, `alphabet_id`, `source_id`를 포함하면 실패합니다.
- `split_name`은 엔트리와 상위 manifest가 반드시 일치해야 합니다.
- `sealed_level`은 엔트리와 상위 manifest가 반드시 일치해야 합니다.

### 2.3 공개/기밀 경계

| 저장소 | 내용 | 접근 주체 |
|---|---|---|
| 공개 split manifest | `sample_id`, `split_name`, `sha256`, `phash`, `sealed_level` | 모델·데이터 로더 |
| 검증 메타데이터 | `class_id`, `writer_id`, `alphabet_id`, `source_id` | validator 내부 |
| ground truth vault | `sample_id → label` | 접근제어를 통과한 resolver만 |

- 검증 메타데이터는 공개 manifest에 복사하지 않습니다.
- 모델·실험 루프에는 공개 엔트리와 기밀 메타데이터를 결합하는 API를 제공하지 않습니다.
- validator는 `sample_id`로 공개 엔트리와 내부 메타데이터를 대응합니다.
- 빈 OOD 슬롯은 메타데이터가 없어도 유효합니다.
- 샘플이 있는데 해당 `sample_id`의 검증 메타데이터가 없으면 빈 교집합으로 PASS 처리하지 않고 실패합니다.

## 3. 분할 격리 규칙

### 3.1 모든 분할 공통

- 모든 분할 간 `sample_id` 중복 0건
- 모든 분할 간 이미지 `sha256` 중복 0건
- `p2_non_digit`과 `non_digit_dev` 사이에도 이 두 축은 적용됩니다.
- pHash 근접 후보는 별도 보고하며, 본 문서에서는 근접 후보 자체를 절대 중복으로 간주하지 않습니다.

### 3.2 클래스·writer·source 검증 8쌍

P2 그룹을 `P2 = {p2_non_digit, non_digit_dev}`로 정의합니다. 두 분할은 같은 비숫자 보정용 계열이므로 **서로 간에는 별도 격리 조건을 두지 않지만**, `dev_ood`, `test_ood`, `phase7_char_transfer`와는 같은 격리 규칙을 각각 적용합니다.

```text
 1. p2_non_digit      ↔ dev_ood
 2. p2_non_digit      ↔ test_ood
 3. p2_non_digit      ↔ phase7_char_transfer
 4. non_digit_dev     ↔ dev_ood
 5. non_digit_dev     ↔ test_ood
 6. non_digit_dev     ↔ phase7_char_transfer
 7. dev_ood           ↔ test_ood
 8. test_ood          ↔ phase7_char_transfer
```

각 쌍에 대해 `class_id`, `writer_id`, `source_id`를 축별로 검사합니다. 어느 한 축이라도 교집합이 있으면 실패합니다.

- 기존 원문의 5쌍을 유지하면서 `non_digit_dev` 관련 3쌍을 추가한 것입니다.
- MNIST 내부 분할은 숫자 클래스 `0~9`를 공유하므로 클래스·writer·source 교차 검사 대상이 아닙니다.
- MNIST는 writer ID가 없으므로 공식 train/test 경계와 전체 ID·SHA-256 검사로 격리를 확인하고, 독립 필체 검증의 한계를 명시합니다.
- `alphabet_id`는 P2/OOD/Phase 7 데이터셋에서 writer/source와 별도로 수집·보존합니다. **자동 차단 축은 `class_id`, `writer_id`, `source_id`로 고정하고, `alphabet_id` 중복은 보조 경고로 보고합니다.**

## 4. 라벨 접근제어

### 4.1 요청과 판정

`LabelAccessRequest`는 최소 다음을 포함합니다.

```text
execution_mode
protocol (STRICT_1SHOT | CONTINUAL)
split_name
sample_ids
episode_id
current_episode_support_ids
approved_stream_ids
query_ids
evaluation_grant (EVALUATE인 경우)
```

`query_ids`는 **현재 미확정 query 집합**으로 정의합니다.

### 4.2 차단 우선순위와 허용 판정

공통 차단 조건을 먼저 적용합니다.

```text
1. 잘못된 요청 또는 미등록 split/sample → 차단
2. INFER → 차단
3. 현재 미확정 query_ids에 포함 → 차단
4. 승인되지 않은 샘플 → 차단
5. 그 외의 모드·프로토콜·분할 조건 미충족 → 차단
```

공통 차단을 통과한 뒤 다음 중 하나를 만족하면 허용합니다.

```text
현재 에피소드의 support_pool support
OR
CONTINUAL에서 승인된 dev_stream/main_stream 샘플
```

- `STRICT_1SHOT`에서는 `dev_stream`, `main_stream` 라벨을 차단합니다.
- 전체 `support_pool` 구성원이라는 이유만으로 접근할 수 없으며, 현재 episode support에 명시적으로 포함되어야 합니다.
- support와 미확정 query가 겹치면 차단이 우선합니다.
- query가 확정됐다는 사실만으로 권한이 생기지 않습니다. 이후 별도로 학습 승인된 샘플이면 그 승인 범위에 따라 허용합니다.
- `non_digit_dev` 라벨 접근은 이번 구현에서 허용하지 않습니다. 필요하면 명세를 먼저 변경합니다.
- `official_test`, `reserved_sealed`, `test_ood` 등은 ADAPT에서 차단합니다.

### 4.3 EVALUATE

- Phase 0~5의 봉인 분할 접근을 차단합니다.
- Phase 6~7에서 해당 평가에 승인된 분할만 SealGuard 검증 후 읽기 전용으로 엽니다.
- 첫 공식 평가에서는 `official_test`만 승인합니다.
- `reserved_sealed`은 후속 모델의 독립 확증용이므로 첫 공식 평가와 함께 열지 않습니다.
- `dev_ood`, `test_ood`도 평가 목적에 명시적으로 승인되지 않았다면 열지 않습니다.

### 4.4 라벨 조회 경로

1. `LabelAccessPolicy.decide()`가 허용 여부와 차단 사유가 담긴 `AccessDecision`을 반환합니다.
2. 허용 판정 후에만 `resolve_labels()`가 vault에서 라벨을 읽습니다.
3. 모델·실험 루프에는 `GroundTruthVault.get()`과 같은 직접 조회 경로를 제공하지 않습니다.
4. EVALUATE의 봉인 분할 라벨 조회도 같은 resolver를 거치며, 내부에서 SealGuard의 split 승인과 검증 record를 확인합니다.
5. vault 저장·로딩은 데이터 준비 단계로 한정하고, 런타임 라벨 소비 경로와 분리합니다.

## 5. 동결 해시와 봉인 해제

### 5.1 `FreezeManifest` 필수 필드

원문 필드명을 그대로 유지합니다.

```text
manifest_sha256
data_split_hash
config_hash
checkpoint_hash
```

### 5.2 검증 규칙

검증은 다음을 **모두** 만족해야 합니다.

1. `manifest_sha256` 자신을 제외한 나머지 본문을 키 정렬·공백 제거 Canonical JSON으로 직렬화합니다.
2. 계산한 manifest SHA-256과 내부 `manifest_sha256`가 일치해야 합니다.
3. 계산한 manifest SHA-256과 sidecar 해시가 일치해야 합니다.
4. 계산한 manifest SHA-256과 **별도로 고정한 독립 기준점**이 일치해야 합니다.
5. 실제 데이터 분할·config·checkpoint 파일 3종의 SHA-256을 계산하여 manifest의 `data_split_hash`, `config_hash`, `checkpoint_hash`와 각각 대조해야 합니다.
6. 모든 검증이 끝난 뒤에만 특정 manifest·파일·기준점에 묶인 검증 record를 생성합니다.

sidecar 일치만으로는 부족하며, sidecar 일치 **AND** 독립 기준점 일치 **AND** 실제 파일 3종 해시 일치가 필요합니다.

### 5.3 독립 기준점

- 사전등록 해시, Git 태그 또는 Git 커밋을 독립 기준점으로 사용할 수 있습니다.
- Git을 기준점으로 선택한 경우 조회 실패는 차단입니다.
- Git 조회는 실행 위치와 무관하게 명시한 프로젝트 루트를 작업 디렉터리로 사용합니다.
- `.git`이 없거나 실제 Git 루트가 지정 루트와 다르면 `SealGuardError`를 발생시킵니다.
- 태그·커밋 조회 실패 또는 기준점 불일치도 `SealGuardError`입니다.
- sidecar와 Git 기준점을 함께 재작성해도, 사전에 고정한 값이 다르면 실패해야 합니다.

### 5.4 검증 결과의 수명과 결합

- 호출자가 `freeze_verified=True`를 전달하는 것만으로 검증을 생략할 수 없습니다.
- 검증 record는 manifest 경로·manifest hash와 실제 아티팩트 경로·해시·평가 승인 정보에 묶습니다.
- 봉인 접근 시 실제 파일 해시를 다시 확인하므로 검증 후 파일을 교체하면 열리지 않습니다.
- 다른 manifest, 다른 실행, 다른 평가 승인에는 이전 검증 record를 재사용할 수 없습니다.
- 봉인 접근은 읽기 전용이어야 하며, 접근 후 파일 변경도 후속 검증에서 탐지되어야 합니다.

## 6. CLI 연결 및 호환성

`data/mnist.py`는 유지합니다.

기존 `__main__.py`가 직접 사용하는 다음 심볼과 진입점을 유지하거나 호환 래퍼를 제공합니다.

```text
download_mnist
verify_mnist_checksums
mnist_available
SPLIT_NAMES
build_all_mnist_splits
validate_split_manifest
validate_split_ids_disjoint
OODValidator
validate_sealed_policy
validate
build-splits
```

`validate`는 최소 다음을 수행합니다.

- MNIST 원본이 존재하면 원본 4개 파일 체크섬 검증
- 필수 10종 manifest 존재·등록 확인
- 공개 엔트리 필수 필드 및 금지 필드 검사
- 전체 ID·이미지 SHA-256 교차 중복 검사
- OOD·전이 8쌍의 클래스·writer·source 검증
- 봉인 등급 정책 검사

`build-splits`는 공개 manifest와 vault를 생성하고, 10종 manifest·전체 격리·봉인 정책을 저장 전에 검증합니다.

## 7. 작성할 함수·클래스 목록

| 모듈 | 함수 / 클래스 | 입력 | 출력 / 역할 |
|---|---|---|---|
| `data/schema.py` | `SplitName`, `SealedLevel`, `ExecutionMode`, `ExecutionProtocol` | 상수 | 허용값 단일 정의 |
|  | `SplitEntry`, `SplitManifest` | 엔트리·manifest 데이터 | 공개 스키마 및 필수·금지 필드 검증 |
|  | `FreezeManifest` | manifest 데이터 | 필수 해시 필드 유지 |
|  | `LabelAccessRequest`, `AccessDecision` | 접근 요청·판정 | 요청 스냅샷, 허용 여부·사유 |
| `data/splits.py` | `build_split_manifest()` | 샘플·split명·봉인등급 | 라벨 없는 공개 manifest |
|  | `load_split_manifest()` | JSON 경로 | `SplitManifest` |
|  | `validate_split_manifest()` | manifest | 오류 목록 |
|  | `register_manifest()` | 전체 manifest 모음 | 필수 분할 등록·구조 검증 결과 |
|  | `validate_split_ids_disjoint()` | manifest 모음 | 전체 ID·이미지 SHA-256 중복 오류 |
|  | `build_all_mnist_splits()` | raw 경로·출력 경로·split_seed | 10종 manifest·vault, 결정론적 ID |
| `data/validation_metadata.py` | `ValidationMetadataStore` | 기밀 class·writer·source 메타데이터 | validator 전용 저장·`sample_id` 대응 조회 |
| `data/ood_validator.py` | `OODValidator.validate_disjoint()` | manifest·검증 메타데이터 | 8쌍 축별 오류 목록 |
|  | `validate_synthetic_split()` | 합성 분할 fixture | 비숫자 fixture 규칙 검증 |
| `data/ground_truth_vault.py` | vault 저장·로딩 | JSON 또는 ID·라벨 | 데이터 준비 단계의 정답 격리 저장 |
| `data/access_control.py` | `LabelAccessPolicy.decide()` | `LabelAccessRequest` | 라벨을 포함하지 않는 `AccessDecision` |
|  | `AdaptScope.authorize()` | episode support·승인 stream ID | 제한된 승인 컨텍스트 |
|  | `resolve_labels()` | 요청·vault·SealGuard | 허용된 ID의 라벨, 차단 시 예외 |
| `data/seal_guard.py` | `canonical_json()` | JSON 객체 | 정규화 문자열 |
|  | `compute_freeze_manifest_hash()` | `FreezeManifest` | self 필드 제외 SHA-256 |
|  | `load_sidecar_hash()` | sidecar 경로 | 기준 해시 |
|  | `verify_git_reference()` | 프로젝트 루트·Git 참조·사전등록 해시 | 기준점 검증 또는 예외 |
|  | `verify_freeze_manifest()` | manifest·파일·기준점 | 검증 record 또는 예외 |
|  | `SealGuard.assert_access()` | phase·모드·split·평가 승인·검증 record | 읽기 전용 접근 허용 또는 예외 |
| `__main__.py` | `validate`, `build-splits` 연결부 | CLI 인자 | 검증 보고·분할 생성·종료 코드 |
| `tests/` | 정상·부정 fixture 테스트 | 수용 기준 fixture | PASS/FAIL 단언 |

## 8. 필수 테스트

### 8.1 분할·스키마

- 10종 필수 분할 등록 확인
- 필수 분할 누락 시 실패
- 빈 OOD/전이 슬롯 등록은 통과
- `SplitEntry` 필수 5필드 확인
- 공개 manifest의 `label`, `class_id`, `writer_id`, `alphabet_id`, `source_id` 포함 시 실패
- 엔트리와 manifest의 `split_name`, `sealed_level` 불일치 시 실패
- 전체 분할 `sample_id` 중복 시 실패
- 전체 분할 이미지 `sha256` 중복 시 실패

### 8.2 OOD·전이 격리

- 8개 쌍 × `class_id`, `writer_id`, `source_id` 각각의 중복 부정 테스트
- ID·이미지·writer·source가 달라도 class가 같으면 실패
- MNIST 내부 class 중복은 통과
- P2 두 분할 사이의 메타데이터 중복은 통과
- 샘플이 있는데 `sample_id` 대응 검증 메타데이터가 누락되면 실패
- 빈 OOD 슬롯은 메타데이터 없이 통과
- `alphabet_id` 중복은 보조 경고로 보고

### 8.3 접근제어

- `INFER` 전면 차단
- `STRICT_1SHOT`의 stream 라벨 차단
- `CONTINUAL`의 승인 stream만 허용
- episode 밖 `support_pool` 샘플 차단
- `non_digit_dev` 차단
- 미확정 query 차단
- 확정 query도 별도 학습 승인 전에는 차단
- support와 미확정 query가 동시에 포함되면 차단
- 확정 후 별도 학습 승인된 샘플은 허용
- 미등록 split/sample fail-closed
- `AccessDecision`에는 라벨이 없고, 허용된 경우에만 `resolve_labels()`가 vault를 읽음
- 모델·실험 루프의 직접 vault 조회 경로 없음

### 8.4 봉인 해제

- **정상 해제:** 승인된 Phase 6~7 `EVALUATE`에서 해시가 모두 맞으면 `official_test`를 읽을 수 있음
- 같은 검증 record로 `reserved_sealed`은 열리지 않음
- Phase 0~5 봉인 분할 차단
- 첫 공식 평가에서 `reserved_sealed` 차단
- 평가 미승인 OOD 분할 차단
- 아티팩트만 변경 시 실제 파일 해시 대조에서 실패
- 아티팩트·manifest·sidecar를 함께 다시 만들고 고정 기준점과 다르면 실패
- 독립 기준점 없이 `freeze_verified=True`만 전달 시 실패
- `.git` 없음, Git 루트 불일치, Git 조회 실패 시 예외
- 검증 record를 다른 manifest·파일·실행·평가 승인에 재사용 시 실패
- 검증 후 파일 교체 시 접근 거부

### 8.5 실제 MNIST 분할

기존 `tests/test_splits.py::test_build_all_mnist_splits`는 원본이 없으면 skip 되고 총 개수만 확인합니다. 이를 실제 원본이 있을 때 실행되는 필수 인수 테스트로 보강합니다.

- MNIST 원본 4개 파일의 SHA-256 체크섬 검증
- train 60,000 분할:
  - `support_pool = 10,000`
  - `dev_stream = 15,000`
  - `main_stream = 35,000`
- test 10,000 분할:
  - `official_test = 7,000`
  - `reserved_sealed = 3,000`
- 세 MNIST 학습·평가 분할의 ID가 서로 겹치지 않음
- 같은 `split_seed`로 두 번 생성한 결과의 `sample_id → split_name` 및 순서가 동일함
- 같은 원본·seed에서 공개 엔트리와 vault 라벨이 재현 가능함
- 원본이 없을 때는 조용히 skip하지 않고, 로컬 원본 경로를 안내하는 명시적 실패로 보고합니다.

## 9. 구현 순서

1. **스키마·열거형**
   - 10종 split, seal level, execution mode/protocol
   - 공개/기밀 필드 경계와 필수 필드
2. **매니페스트 빌더·등록기**
   - 금지 필드, split 일치, 10종 등록, 빈 슬롯 처리
3. **전체 및 OOD 격리 validator**
   - 전체 ID·SHA-256
   - 8개 쌍의 class·writer·source와 메타데이터 누락 검사
4. **Vault·접근제어**
   - 공통 차단 조건, OR 허용 조건, protocol, query, episode support
   - 판정과 라벨 해석 분리
5. **SealGuard**
   - Canonical hash, sidecar AND 독립 기준점 AND 실제 파일 해시
   - 평가 split 승인 및 파일 결합 검증 record
6. **CLI 연결**
   - 기존 명령·옵션·종료 코드 유지
7. **정상·부정 테스트 및 실제 MNIST 인수 테스트**
8. **T3 완료 검증 후 이 임시 문서 삭제**

## 10. 기존 코드에서 재작성 시 확인할 지점

- `src/foveacore/data/splits.py:97`가 `class_id` 등을 공개 엔트리에 복사합니다.
- `src/foveacore/data/ground_truth_vault.py:76`의 반환값이 차단과 실제 라벨을 섞고, episode support/query/protocol 조건이 없습니다.
- `src/foveacore/data/ood_validator.py:107`의 메타데이터 검사가 P2를 `p2_non_digit` 단독으로 한정합니다.
- `src/foveacore/data/seal_guard.py:50`는 호출자가 준 artifact hash만 비교하며 실제 파일을 읽지 않습니다.
- `src/foveacore/__main__.py:55`는 새 8쌍·메타데이터 누락·전체 격리 규칙을 모두 호출하지 않습니다.
- `tests/test_splits.py:100`의 실제 MNIST 테스트는 원본 부재 시 skip 됩니다.
