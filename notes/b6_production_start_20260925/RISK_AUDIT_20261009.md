# WP5 파일럿 추가 위험 점검 — 2026-10-09

현재 **저장 공간으로 인한 시작 거부 위험 1건**, **종료·복구의 조건부 버그 2건**을 확인했다. 현재 잡에서 새 학습 오류가 발생한 것은 아니다. 원격 코드·잡·파일은 변경하지 않았으며 로컬 감사 기록만 추가했다.

## 범위와 현재 상태

- 실제 제출 코드: `b6-compile-static-len`, `233574ae6e8b4f0e35738556ecffff8a2f823680`, 로컬 `/tmp/b6-compile-static`, DeltaAI `repo_b6_233574a`.
- 최근 수정 비교: `git diff 826e68e...233574a` — 컴파일 수정에서 확인된 신규 결함 0건. 아래 종료 처리 문제들은 그 이전부터 존재한다.
- 12:56 CDT 실측: B6 `3345917`, C2 `3345919` Priority 대기; `3345918`, `3345920` 각각 afterany 의존 대기. B5 `3309786` Priority 대기(예상 10-10 15:20 CDT, 변동 가능). 잔액 771 GPU-h.
- 스모크 `3338555`, 입력 검증 `3345916` 모두 COMPLETED 0:0. verify 기록은 이전 `3306593`과 동일.
- 런처·config·runtime·trainer의 원격/로컬 SHA-256 일치. Slurm에 저장된 B6/C2 첫 링크 스크립트의 SHA도 런처와 일치.
- 실제 스모크 체크포인트는 **247 updates**다. 이전 240 보고는 10-step 간격의 마지막 학습 로그 수치였으며, `latest_state/meta.json`은 247을 기록한다.

## 1. 우선 보완: 100 GiB 시작 검사에 여유가 5.09 GiB뿐

근거: `revision/scripts/b6/storage_preflight.py:9,43–48`, `revision/scripts/b6/wp5_pilot.sbatch:71` (모두 위 커밋 기준).

| 실측/기준 | 값 |
|---|---:|
| NVMe soft quota | 500 GiB |
| 현재 quota 여유 | 105.090736 GiB |
| 런처 시작 검사 최소 여유 | 100 GiB |
| 검사 기준 초과분 | 5.090736 GiB |
| 실측 B6 latest_state 한 벌 | 5.334316 GiB |
| 해당 크기의 새 파일이 생긴 뒤 여유 | 99.756420 GiB |

실제 `check_storage()`를 현재 관측값으로 호출하면 PASS다. OS 조회값만 모의해 체크포인트 한 벌이 추가되는 경우를 재현하면 **insufficient CP4 storage로 거부**된다. 실제 원격 파일을 추가하거나 quota를 바꾸지 않았다.

B5는 같은 풀에서 epoch checkpoint를 계속 보존한다(현재 전체 약184 GiB; epoch14 한 벌 14,123,535,700 bytes). 먼저 실행한 파일럿의 체크포인트, B5의 다음 epoch 저장 또는 저장 중 임시 사본이 다른 링크의 사전 검사 시점과 겹치면 시작이 거부될 수 있다. 단순 `df`의 여유 약276 GiB로 판단하면 프로젝트 quota를 놓친다.

코드상 B6/C2는 각각 epoch1 checkpoint, rank별 pending gradients, latest_state, final step을 보존한다. 실측 모델/optimizer 크기로 추정한 두 팔 합계는 약42.7 GiB, latest_state 동시 교체 중 약49.2 GiB다. pending-gradient 크기는 FP32 Adam moment 크기로 추정했으며 생산 실측값이 아니다. B5의 추가 저장량은 별도다.

대응: **기존 100 GiB 검사를 우회하지 말고** 앞으로 생성할 체크포인트와 동시 실행까지 포함해 공간을 확보한다. 확인된 정리 검토 후보는 `/work/nvme/bilv/wcyim/apptainer_cache/cache/blob` 약43 GiB다. 실행에 사용하는 `.sif`는 별도 `containers/`에 있다. 캐시는 이번 감사에서 삭제하지 않았고, 43 GiB만으로 전체 B5 잔여 epoch와 모든 링크의 시작 여유가 보장되지는 않는다. B5 체크포인트는 기존 보존 결정에 따라 유지한다.

재현: [storage_repro.py](audit_20261009/storage_repro.py), [출력](audit_20261009/storage_repro.log).

## 2. P2: 종료 검증 실패가 성공으로 보고될 수 있음

근거: `train/train_HyenaTransgenic_B6.py:70,409–411,503–505`, `revision/scripts/b6/wp5_pilot.sbatch:113–116`.

트레이너는 최종 step과 `TRAINING_DONE`을 먼저 쓰고, 짧은 링크의 미실시 컴파일 검증을 그 뒤에 한다. 기본 warmup은 3 updates다. 마지막 재개 링크에 1–2 updates만 남은 경우 검증 오류가 발생하면 완료 표식이 남는다. 런처는 `TRAIN_RC`보다 완료 표식을 먼저 확인하므로 **트레이너 오류 코드 1을 성공 코드 0으로 바꿀 수 있다**.

기본 warmup3을 그대로 두고 마지막 2 updates를 재개하는 작은 실제 트레이너 fixture에서 검증 예외를 주입했다. 두 완료 표식이 남았고, 수정하지 않은 런처의 종료 분기를 `TRAIN_RC=1`로 실행해 exit0을 확인했다. 이는 오류 처리 결함의 재현이며, 실제 GPU에서 컴파일 검증 오류가 저절로 발생한다는 증거는 아니다.

대응: 성공 표식보다 필수 검증을 먼저 끝내고, 런처는 비정상 trainer 반환값을 먼저 거부해야 한다. 실제 제출본에 수정은 하지 않았다.

## 3. P2: 최종 체크포인트 게시 직후 중단 시 자동 재개 실패

근거: `train/train_HyenaTransgenic_B6.py:266–292,400–411`.

`step_04099` 게시와 `TRAINING_DONE` 작성은 별도 단계다. 그 사이 I/O 오류나 강제 종료가 나면 valid step은 있으나 DONE이 없다. 이전 latest_state에서 재개해 같은 목표에 도달하면 이미 게시된 step을 덮어쓰지 않는 보호 검사에 걸려 **refusing to replace published step checkpoint**로 중단된다.

실제 작은 트레이너 fixture에서 `mark_done` 실패를 주입하고 auto-resume을 실행해 두 번째 실패까지 재현했다. 체크포인트를 덮어쓰거나 임의로 완료 처리할 문제가 아니다.

대응: 게시된 checkpoint의 run identity·목표 update·파일 hash·필수 표식과 필수 검증 상태를 확인한 뒤 안전하게 완료를 복구하는 경로가 필요하다. 현재는 수동 대조 없이는 자동 복구되지 않는다.

2·3번 재현: [finalization_repro.py](audit_20261009/finalization_repro.py), [독립 재실행 출력](audit_20261009/finalization_repro.log). 모델 산출물은 TemporaryDirectory에만 만들고 정리한다.

## 검증 범위와 남은 불확실성

- 저장소 검사 기존 시험: `python3 -m pytest -q tests/test_b6_storage_preflight.py` → **13 passed**.
- 컴파일 수정 집중 시험: PyTorch2.6 CPU `unittest` → **13 passed**, [출력](audit_20261009/compile_tests.log).
- 정상 파일럿·재개 시험 3개: B6/C2 정확한 목표 종료, 재시작 후 목표 도달, checkpoint byte 동일성. [출력](audit_20261009/pilot_resume_tests.log).
- 종료 오류 재현은 PyTorch2.6 + Accelerate1.10.1의 작은 CPU fixture를 사용했다. production GPU/Inductor 전체 실행을 대체하지 않는다. 추가 dependency는 `/tmp/b6-spec-review-deps`에만 설치했다.
- 정적 DNA 길이 수정에서 추가 신규 버그는 확인하지 못했다. 다만 같은 tier의 세 번째 재컴파일 또는 총 이벤트 상한 초과는 의도적으로 중단한다(`b6_runtime.py:670–678`). 실제 전체 데이터의 이후 label 조합·평가 경로까지 안전하다는 증거는 아직 없다.
- GPU 스모크는 초기247 updates와 저장을 확인했다. 실제 저장본 로드, epoch1 평가/pending-gradient 저장, C2 생산 실행, 4,099-update 최종 게시를 검증한 것은 아니다.
- shared-tensor 저장 경고는 정상 byte 동일성 재개 시험에서도 나타났지만 시험은 작은 모델이다. 이 경고만으로 현재 생산 체크포인트 손상을 판정하지 않았다.

관측값과 SHA: [observations.json](audit_20261009/observations.json). 우선순위는 저장 여유 확보, 종료 처리 보완·검증, 실제 첫 링크와 재개 결과 확인 순서다.
