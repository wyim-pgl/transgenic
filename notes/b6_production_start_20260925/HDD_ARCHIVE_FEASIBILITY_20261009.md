# DeltaAI 완료 체크포인트 HDD 이관 확인 — 2026-10-09

13:18–13:20 CDT에 DeltaAI를 확인했다. **완료된 과거 epoch의 HDD 이관은 가능하다. 단순 mv로는 NVMe 공간이 확보되지 않아, HDD에 새 파일로 복사하고 검증한 뒤 원본을 제거하는 절차가 필요하다.** 이번 요청은 해결 방법·가능성 확인이며, 실제 체크포인트 이동/삭제와 코드·잡 변경은 수행하지 않았다. 1 MiB 임시 파일 시험만 했고 시험 파일은 정리했다.

## 실측

| 항목 | 결과 |
|---|---|
| NVMe quota 여유 | 105.091 GiB (500 GiB soft quota) |
| HDD quota 여유 | 170.714 GiB (1000 GiB soft quota) |
| HDD 부모 경로 | `/work/hdd/bilv/wcyim/transgenic`, 쓰기 가능 |
| 두 경로의 filesystem device | 모두 3290199710 |
| project ID | 양쪽 22169, 상속 P |
| NVMe stripe pool | ddn_ssd |
| HDD stripe pool | ddn_hdd |
| B5 최신 재개 상태 | latest_state global_step30819, epoch14(zero-based) |
| B5 best | epoch_14 |
| B5/B6/C2 Slurm 상태 | 전부 PENDING, GPU 실행 중 아님 |
| 장기 보관 대안 | pronghorn `/data/gpfs` 가용21T(df), archives 부모 쓰기 가능; 사용자/그룹 quota는 별도 확인 필요 |

### 실제 저장 풀 시험

같은 1 MiB 데이터(SHA256 `fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83`)로 확인했다.

1. NVMe에서 생성: inode522420049315627159, pool ddn_ssd.
2. HDD 경로로 rename/mv: **inode 동일, pool 여전히 ddn_ssd**.
3. HDD 디렉터리에 새 파일로 복사: inode216175193956136765, **pool ddn_hdd**, SHA256 동일.
4. 시험 전용 임시 디렉터리와 파일 정리 완료.

따라서 `mv /work/nvme/... /work/hdd/...`만 실행하면 주소만 달라지고 NVMe quota 사용량을 줄이지 못할 수 있다. 복사 대상은 HDD 디렉터리의 저장 풀 상속을 따라 새 inode로 만들어야 한다.

## 권장안: 최신 상태는 NVMe, 완료된 과거 epoch는 HDD

사용자에게 이관 시점(과거 epoch마다 / 학습 전체 종료 후)을 질문했다. 아래는 답변 전의 권장안이며 자동 이관 정책을 활성화하지 않았다.

- **NVMe 유지:** latest_state, latest_state.prev, 최신 완료 epoch 두 개, best가 가리키는 epoch, 작성 중인 .tmp. best가 오래된 epoch여도 보호한다.
- **HDD 후보:** 위 보호 대상에 속하지 않는 완료된 epoch 디렉터리 전체. model만 옮기면 optimizer/RNG/pending gradients가 빠지므로 디렉터리 전체를 보존한다.
- **최종 step:** 모든 검증과 정상 종료가 확인된 후 보관 가능. `PILOT_FINAL`·`PRUNE_EXEMPT`는 보존해야 하며 검증된 HDD 사본도 제거하지 않는다. 아직 해당 run의 완료가 확정되지 않았으면 NVMe를 유지한다.
- 매200업데이트에 생기는 latest_state를 모두 누적 보관하지 않는다. 현재 크기로 두 팔20회씩 복사하면 latest_state만 약213 GiB여서 HDD 여유를 넘는다. latest_state는 회전하는 재개용 파일이며 이관 대상은 완료된 과거 epoch다.
- 기존 B5 보존 결정은 데이터 보존을 계속 요구한다. 실제 NVMe 원본 제거는 검증된 HDD 보관과 사용자 이관 지시의 범위가 확정된 뒤에만 실행한다.

### 당장 검토할 첫 이관 묶음

B5 `epoch_01`–`epoch_08`: **106,360,850,299 bytes = 99.056 GiB**. 모두 model.safetensors, meta.json, eval.json, accelerate_state가 있고 best가 아니다. 이는 메타데이터 검사 결과이며 아직 전체 hash 검증은 하지 않았다. epoch08의 작은 크기는 pending_micro=0과 일치한다.

예정 출발지: `/work/nvme/bilv/wcyim/transgenic/runs/seed123/epoch_01` … `epoch_08`.
예정 도착지: `/work/hdd/bilv/wcyim/transgenic/checkpoint_archive/seed123/epoch_01` … `epoch_08`.

| quota 여유 | 현재 | 이관 후 예상 |
|---|---:|---:|
| NVMe | 105.09 GiB | 약204.15 GiB |
| HDD | 170.71 GiB | 약71.66 GiB |

파일 apparent size로 계산한 예상치이며 실제 quota 반영량은 이관 후 재측정해야 한다. latest_state와 epoch09–14, best(epoch14)는 그대로 둔다. 위 선택은 전체 데이터 삭제가 아니라 보관 위치 변경을 위한 제안이다.

HDD도 한정돼 있다. 8개 이동 뒤 남는71.7 GiB에 모든 미래 B5 epoch를 계속 쌓을 수 없다. 두 WP5 팔의 잔존 checkpoint 추정42.7 GiB와도 공간을 공유한다. HDD 최소 여유/예약량을 검사하고 부족하면 원본을 남긴 채 이관을 멈춰야 한다. 장기 누적분은 pronghorn 등 별도 보관 공간 계획이 필요하다.

## 자동 이관의 안전한 완료 절차

1. 새 epoch가 `.tmp`에서 정식 이름으로 게시되고 모든 rank의 저장이 끝났는지 확인한다. 이전에 완료된 보호 대상 밖 epoch만 선택한다.
2. run별 동시 이관 잠금을 잡고 source 경로·inode·파일 목록·크기와 보호 대상을 기록한다. 겹치는 잡이나 수동 재개가 해당 source를 읽는 경우 보류한다.
3. HDD 저장 풀과 quota를 확인하고 `checkpoint_archive/<run>/<epoch>.partial`에 **새 파일로 복사**한다. 최초부터 HDD에 생성한다.
4. 원본/사본 파일 목록·크기·SHA256 전부 대조하고 source가 복사 동안 바뀌지 않았는지 재확인한다. 복사·검증 실패 시 source는 유지한다.
5. 사본과 manifest를 내구성 있게 저장하고 HDD 내부에서 `.partial`을 정식 이름으로 원자적 게시한다. 완료 영수증에 source identity, commit/run 설정, 전체 hash 목록을 기록한다.
6. 최신/best 보호 대상을 다시 확인한 뒤에만 NVMe 원본을 제거하고 실제 NVMe/HDD quota와 사본 저장 풀을 확인한다. 중단되면 어느 단계에서든 완성 사본 또는 원본이 남아야 한다.

큰 checkpoint 복사와 전체 hash는 계산 자원에서 실행한다. 로그인 노드에서 대용량 hash를 반복하지 않는다. 학습과 분리해 기존 계산 잡의 CPU로 과거 완료 epoch를 처리할 수 있으나 GPU 학습 성공/실패 표식과 이관 성공 표식을 분리해야 한다. 이관 실패가 학습 완료를 위조해서는 안 된다.

활성 run 디렉터리 전체를 HDD로 옮기거나 HDD 절대 경로를 가리키는 symlink로 바꾸지 않는다. 현재 컨테이너는 NVMe `/work`와 evidence 하위 HDD만 bind하므로 새 archive 경로의 symlink는 컨테이너에서 끊길 수 있다. 오래된 epoch를 수동 재개할 때는 검증 후 NVMe로 되가져오거나 명시적 archive bind를 추가한다.

## 나머지 두 오류의 수정 방법

1. **검증 실패를 성공으로 처리하는 오류:** 마지막 업데이트 뒤 필수 컴파일/분산 검증을 먼저 완료하고 최종 checkpoint와 DONE을 게시한다. 런처는 TRAIN_RC가 비정상이면 DONE 존재 여부와 무관하게 실패해야 한다. 기존 DONE만으로 무조건 성공하는 시작 분기도 검증 영수증을 확인해야 한다.
2. **마지막 게시 직후 중단의 재개 오류:** 최종 checkpoint에 run identity·목표 update·모델 hash·필수 검증 완료 기록을 함께 원자적으로 저장한다. 재시작 시 이들을 대조해 이미 정상 게시된 최종 산출물의 완료 표시를 복구한다. 다른 run/다른 hash 또는 검증 기록이 없는 산출물은 덮어쓰거나 임의로 성공 처리하지 않는다.

두 문제 모두 기존 [오류 주입 재현](RISK_AUDIT_20261009.md)을 회귀 시험으로 사용해야 한다. CPU 정상 재개/byte 동일성, 저장 중 중단, archive 복사 실패·hash 불일치·HDD 부족을 확인하고 독립 검토 후 배포한다. 대기 중 Slurm 잡에는 제출 시점의 sbatch가 저장돼 있으므로 로컬 스크립트 수정만으로 이미 제출된 잡의 종료 분기는 바뀌지 않는다. 새 배포본과 제출본의 갱신을 함께 검토해야 한다.
