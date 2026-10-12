# 자동 체크포인트 이관 구현·검증 완료 — 2026-10-09

> **후속 업데이트:** 사용자 "이관 해" 지시로20:15PDT 실제프롱혼이관을시작했다. 현재진행정본은 [PRODUCTION_TRANSFER](PRODUCTION_TRANSFER_20261009.md). 아래 "미활성"은 구현완료시점의기록이다.

**코드·설치·소규모 실제 서버 시험 완료. 운영 체크포인트 대량 이관과 자동 삭제 서비스는 활성화하지 않았다.** 브랜치 `b6-checkpoint-archive`, 최종 커밋 `3b4eb0dfe0e83463775ce8e1ab05fa2f62d1e20b`, worktree `/tmp/b6-checkpoint-archive`를 보존했다. 학습 트레이너·Slurm 잡은 변경하지 않았다.

## 완성한 기능

- DeltaAI NVMe → HDD: 실제 HDD pool `ddn_hdd`·project22169·quota를 확인하고 새 파일로 복사한다.
- DeltaAI → pronghorn: pronghorn에서 SSH/rsync로 가져오고 독립 SHA256 대조·내구성 저장 후 정리한다.
- 검증된 HDD archive → pronghorn의 2차 이관도 지원한다. 원본 archive의 검증 기록을 새 기록에 보존한다.
- 최신 두 **완료된** epoch, best, latest_state, 작성 중 .tmp 및 step/PILOT_FINAL은 자동 정리 대상에서 제외한다.
- 기본은 metadata-only 조회. `--apply`가 복사를, 추가 `--remove-source`가 검증 후 정리를 허용한다.
- `--watch --cycles N --interval S`로 횟수/간격을 제한해 자동 실행한다. 용량·체크포인트 수·최소 여유·속도·통신 시간도 제한한다.
- 중단된 작업은 기록으로 재개하고, 같은 작업을 반복해도 다른 체크포인트를 정리하지 않는다.

## 실제 검증 결과

| 검증 | 최종 결과 | 근거 |
|---|---|---|
| DeltaAI Python3.6 로그인 노드 | 69/69, 9.021초 | [로그](deltaai_final_3b4eb0d.log) |
| pronghorn Python3.11 로그인 노드 | 69/69, 30.541초 | [로그](pronghorn_final_3b4eb0d.log) |
| 실제 NVMe→HDD 작은 fixture 3회 | 전부 PASS, HDD pool·해시·보호 대상 유지 | [로그](deltaai_core_b80c638.log) |
| 실제 HDD/Lustre에서 핵심 오류 주입 | 41/41, 70.860초 | [로그](deltaai_core_8ff381b_hdd.log) |
| 최종 실제 DeltaAI→pronghorn 3개 시나리오 | 직접 전송, 시간 초과 후 재시도, HDD 보관본 재전송 모두 PASS | [로그](pronghorn_final_3b4eb0d.log) |
| 기존 storage preflight 시험 | 13/13, 0.05초 | 로컬 도구 출력; 기존 검사기 무변경 |
| 독립 코드 검토·호스트 수정 재검토 | Critical/Important 0, APPROVED | [최종 검토](reviews/final-review.md) |

69개는 **같은 시험 집합을 두 환경에서 각각 실행**한 수치다. 실제 서버 간 시나리오는 약1.5MiB, 8파일의 시험용 체크포인트를 사용했고, 이관 후 별도 hashlib 계산으로 원래 fixture의 SHA256과 대조했다. 모든 시험용 원격 원본/사본은 정리했다. 실제 B5 모델 파일은 조회만 했다. GPU 잡은 제출하지 않았다.

검토·실험으로 잡아 수정한 문제: 불완전한 새 epoch가 보호 기준을 밀어냄, 완료 기록 초기 작성 중 SIGKILL 후 재시도 차단, 다른 archive의 정리 기록을 잘못 선택, 부분 메시지/쓰기 정체가 timeout을 우회함, 부모 종료 후 남는 자식 프로세스, trailing slash 복구, ARM의 큰 pipe를 고려하지 않은 시험, 실제 rsync 임시 파일의 중단 후 잔존. 마지막 문제는 **공개 전 private stage에서만** inplace 재전송하고 외부 파일과 hardlink된 stage payload를 거부하는 방식으로 해결했다. 알 수 없는 파일을 무조건 지우는 방식은 사용하지 않는다.

## 설치 위치

DeltaAI:
`/work/nvme/bilv/wcyim/transgenic/ops/checkpoint_archive/3b4eb0d`

pronghorn:
`/data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d`

두 경로 안의 실행 파일은 `revision/scripts/b6/checkpoint_archive_cli.py`다. core·CLI·storage helper·시험·문서의 7개 SHA256을 양쪽 설치 시 대조했다(`SOURCE_SHA256.json`, `SOURCE_COMMIT`).

설정 파일은 양쪽에 바이트 동일하게 설치했다.

- [archive.safe.json](archive.safe.json): 기본 예시 제한 1GiB/pass라 현재 큰 B5 epoch를 선택하지 않는다.
- [archive.inventory.json](archive.inventory.json): epoch01–08 조회용, 120GiB/8개 한도. 아래 조회 명령은 복사/정리를 수행하지 않는다.
- [archive.b5-first-batch.json](archive.b5-first-batch.json): B5 첫8개에 한정한 실행 준비안. 한 pass당16GiB/1개, 16MiB/s; HDD 최소50GiB, pronghorn 최소1TiB 유지. SSH 전체 timeout7200초. 아직 실행하지 않았다.

### DeltaAI에서 HDD 대상 조회

```bash
/usr/bin/python3 /work/nvme/bilv/wcyim/transgenic/ops/checkpoint_archive/3b4eb0d/revision/scripts/b6/checkpoint_archive_cli.py \
  --config /work/nvme/bilv/wcyim/transgenic/ops/checkpoint_archive/3b4eb0d/archive.inventory.json \
  --target hdd-b5 --target hdd-b6 --target hdd-c2 --once
```

### pronghorn에서 대상 조회

```bash
/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python /data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d/revision/scripts/b6/checkpoint_archive_cli.py \
  --config /data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d/archive.inventory.json \
  --target pronghorn-b5 --target pronghorn-b6 --target pronghorn-c2 --once
```

두 조회 모두 실제 서버에서 exit0. B5 epoch01–08 **106,360,756,091 bytes(약99.06GiB)** 후보, 보호 epoch13·14(best14), 거부0·미완료 정리0. B6/C2 런 디렉터리는 아직 없어 waiting으로 표시됐다. [DeltaAI 조회](deltaai_production_inventory_3b4eb0d.log), [pronghorn 조회](pronghorn_production_inventory_3b4eb0d.log).

### 실제 첫 배치를 실행할 때 사용할 준비된 명령

아래는 대량 복사와 원본 정리를 실제로 수행하는 명령이며 **이번 작업에서 실행하지 않았다**. HDD와 pronghorn 중 목적지 하나를 선택한다. pronghorn 예:

```bash
/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python /data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d/revision/scripts/b6/checkpoint_archive_cli.py \
  --config /data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d/archive.b5-first-batch.json \
  --target pronghorn-b5 --apply --remove-source --watch --cycles 8 --interval 300
```

HDD는 DeltaAI에서 같은 상대 실행 파일·설정에 `--target hdd-b5`를 선택한다. 복사만 검토하려면 `--remove-source`를 생략한다. 전체99GiB와 여러 번의 해시는 작은 fixture 시험보다 훨씬 큰 I/O다. 실행 위치·예산은 그 규모로 정해야 한다. [DeltaAI 공식 안내](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html)는 로그인 노드에서 small/modest scp·rsync를 사용하고 대용량에는 Globus를 권장한다. 코드가 설치됐다는 것이 운영 watcher가 활성화됐다는 뜻은 아니다.

## 마지막 운영 상태(14:47 CDT)

- HDD: 약829.29/1000GiB 사용, 약170.71GiB 여유. 대부분은 증거 DB 약732.28GiB다.
- NVMe: 약394.91/500GiB 사용, 약105.09GiB 여유. 실제 checkpoint 이관 전이므로 공간 문제 자체는 아직 해소되지 않았다.
- B6 3345917·C2 3345919 Priority 대기, 후속3345918·3345920 Dependency; B5 3309786도 대기. 잔액771GPU-h. [실측](final_live_status.log).
- 이전에 발견한 학습 종료 처리 오류2건은 이관 도구와 별도이며 미수정 상태다. [기존 감사](../b6_production_start_20260925/RISK_AUDIT_20261009.md).

## 적용한 판단과 한계

1. v1 자동 대상은 완료된 과거 epoch다. 최종 step은 기존 종료 검증 오류에 의존하지 않도록 보호했다. 비용: 최종 결과 전체 이관은 별도 명시적 절차가 필요하다.
2. 소규모 시험과 약99GiB 실데이터 정리를 구분해 운영 자동 삭제를 활성화하지 않았다. 비용: 실제 공간 확보는 아직 남아 있다.
3. active run과 명시적 verified archive 입력을 구분했다. 비용: 직접 HDD에서 학습 재개할 디렉터리를 archive로 잘못 지정하면 보호 의미가 달라지므로 종류를 정확히 지정해야 한다.
4. 네트워크 단절 전후를 cleanup 승인 시점으로 구분한다. 승인 전에는 원본 유지, 독립 검증된 사본이 승인된 후에는 원본 정리가 일부/전부 진행됐을 수 있으므로 journal로 확인·재개한다. 비용: '응답을 못 받았으니 원본이 그대로'라고 가정할 수 없다. 검증된 사본은 유지한다.

표준 공개 API의 원격 검증은 같은 사용자의 인증된 SSH 조정자를 신뢰하는 경계다. 암호학적 원격 저장 증명이라고 주장하지 않는다. 동일 사용자에 의한 임의의 파일 수정을 모두 직렬화할 수는 없으므로, 게시된 archive는 정리가 끝날 때까지 불변이어야 한다. 재인증이 필요하면 BatchMode 실패로 중단하고 원본/검증된 사본을 보존한다.
