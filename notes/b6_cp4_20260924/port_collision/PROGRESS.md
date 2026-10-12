# NCCL 사전시험 포트 충돌 수정: 2026-09-24 23:40 PDT

## 관측과 수정

3209815는1분29초후FAILED1:0. 실제NCCL fault시험6사례는 통과했으나 마지막 preserve_step 사례가 TCPStore 초기화의 EADDRINUSE(port49717)로 실패했다. 정식CP4는 실행되지 않았다. 실제 포트 점유자는 미확인이다.

원인은 시험의 free-port 조회 후 socket을 닫고, worker가 나중에 같은 포트를 여는 방식이 포트를 예약하지 않는다는 점이다. 포트를 의도적으로 점유한 회귀시험으로 같은 EADDRINUSE를 재현했다.

- 로컬 커밋 **3511a5a3fe2eec1d3ef318354adbed060de12a9f**, 변경은 `tests/test_b6_checkpoint_io.py` 한 파일뿐.
- worker가 pytest의 고유 temp directory에 있는 file:// rendezvous로 명시적 rank/world/backend process group을 초기화한다. CUDA는 먼저 local rank 장치를 지정한다. Accelerator는 기존 group을 재사용하며 실제 world_size2/rank를 단언한다.
- parent가 MASTER_PORT를 일부러 계속 점유해, 시험이 빈 TCP rendezvous port에 의존하지 않음을 검증한다.
- fault사례 삭제·skip·재시도·판정완화 없음. 모델/트레이너/quota검사/wrapper 변경 없음.

## 검증

- 수정 전 occupied-port 회귀: EADDRINUSE로1failed.
- 수정 후 로컬 실제2-rank Gloo **7passed**.
- 실제26.08 SIF CPU FileStore+Gloo **7passed**.
- 독립검토 agent `c236a6ba-a38d-453`: APPROVE. 실제 GPU/NCCL FileStore 경로는 다음 잡에서 확인한다.

## 새 실행

- 새 HDD repo `/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924`.
- 원격스냅숏 **e2a91b7f4485e5eeb3b23d40fee0cac9a8a6bc10**.
- source/config/test220파일 검증. 매니페스트SHA256 `a52974c239cc25133e4aff6660cd051812cccde40433367c80805152baa4cb0f`.
- 기존검토 wrapper `run_hdd_cp4.sh` 해시불변확인. 활성B6잡0·잔액1023·HDD quota/물리여유100GiB이상 확인, noclobber저널후제출1회.
- **새 잡3210279**, GPU2/1h30m/ghx4-interactive. 23:40PDT PENDING(Priority), 예상시작N/A.
- 실제NCCL fault7사례 통과 후에만 같은 잡에서 정식wp3-fixtures CP4 실행.
- 로그 `/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924/b6-int4-3210279.out`.
- 보고서 `/work/hdd/bilv/wcyim/transgenic/runs/b6_int4/run_3210279/report.json`.

30분 예약의 현재 대상은3210279다. 정식CP4 PASS 뒤synthetic 별도실행. 기존산출물/NVMe147GiB/B5/운영학습/main병합/푸시무변경. NVMe정리는여전히별도승인대기다.
