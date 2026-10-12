# CP4 장치 검증 오류 수정: 2026-09-24 18:08 PDT

## 원인과 변경

실패 잡 `3207149` 원본 로그/JSON은 `failure3207149.txt`에 보존했다. Accelerate가 GPU로 옮긴 배치를 `validate_b6_batch`가 검증할 때, 공용 `b6_track_spec.validate_batch`의 `np.asarray(chain_mask/chain_idx/chain_feat)`가 CUDA 텐서의 암묵적 NumPy 변환을 시도했다. 학습이 시작되기 전에 양 rank가 실패했다.

수정은 위 세 변환을 제거하고, 이미 shape/dtype이 검증된 NumPy 배열 또는 torch 텐서에 동일한 masking/comparison/any 검사를 직접 수행하는 것이다. CPU 복사, 검사 생략, 모델·손실·데이터 계약 변경은 없다.

- 로컬 `/tmp/b6-integration`, 커밋 `a48579d240e1912d39974a5af6a7a78cd529433b`.
- 변경 파일: `src/transgenic/evidence/b6_track_spec.py`, `tests/test_b6_batch_device.py`.
- NumPy/torch CPU/실제 CUDA에 대해 학습·추론 정상 배치와 잘못된 dummy mask, masked idx, 음수/상한 idx, masked feature를 검사한다.

## 검증

- 로컬 실제 Quadro K620/CUDA(2 GB)에서 수정 전 **7 failed, 14 passed**. 7개 CUDA 사례 모두 운영 로그와 같은 TypeError.
- 수정 후 관련 spec/dataset/계약/장치 시험 **46 passed, 13 subtests passed**.
- Transformers4.57.6 환경의 추가 trainer/bench 포함 **78 passed, 91 subtests passed**. 로컬 CUDA7사례 실제 실행.
- 실제 DeltaAI SIF CPU에서 spec/장치 시험 **24 passed, 7 skipped**. skip7은 로그인 노드에 GPU가 없어 CUDA 사례를 실행하지 않은 것이며, 로컬 CUDA 검증과 구분한다.
- 독립 검토 `openai-codex/gpt-5.5`, agent `4e699b2b-6b19-44e`: **APPROVE, findings none**, 다음 bounded 2GPU CP4 한정. 검토자 환경 자체 시험은 torch 미가용으로 7passed/14skipped였으며, 본 세션의 실제 CUDA 결과를 대체하지 않는다.

## 배포와 재실행

- 이전 두 원격 디렉터리 및 실패 결과 보존. 새 디렉터리 `/work/nvme/bilv/wcyim/transgenic/repo_b6_devicefix_20260924`.
- 원격 스냅숏 `6446692c34c7fd72f993452197c9cbc5e3ba9bd6`.
- 217개 추적 소스·설정·시험 파일 SHA256 대조 통과. 매니페스트 자체 SHA256 `1274ffda8b89ca217bff235a6aaa75167b78824359ef79e10102ab8c04bfc0ca`.
- 메모리 벤치3207080에서 측정한 model/bench 파일 해시 불변 확인. 검증기만 수정돼 모델 메모리 벤치를 반복하지 않았다.
- 제출 전 B6 활성잡0, 잔액1024 확인. noclobber `CP4_DEVICEFIX_SUBMISSION_ATTEMPT` 선기록 후 sbatch 1회, rc0.
- 새 **CP4 wp3-fixtures 잡 `3207331`**, GPU2, ghx4-interactive, 상한1h30m.
- 18:08 PDT 실측 PENDING(Priority), 예상 시작 N/A.
- 로그 `/work/nvme/bilv/wcyim/transgenic/repo_b6_devicefix_20260924/b6-int4-3207331.out`.
- 보고서 `/work/nvme/bilv/wcyim/transgenic/runs/b6_int4/run_3207331/report.json`.

현재 예약 확인 대상은 3207331이다. CP4 보고서의 compile/straight/preempt/resume/generation parity를 검증하고, 통과 후 synthetic CP4를 실행한다. synthetic 미제출, B5·운영학습·main 병합·푸시 무변경.
