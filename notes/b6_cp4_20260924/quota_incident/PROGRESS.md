# CP4 quota 사고 및 HDD 재실행: 2026-09-24 23:07 PDT

## 사고와 원인

`3208442`는 22:31 PDT 재조회에서 FAILED1:0, 5분12초였다. 원본 로그 `live3208442.txt`.

- 프로젝트 NVMe 사용548.3GiB, soft500GiB/hard550GiB. filesystem df는 약952GiB 여유를 보여도 프로젝트 pool quota가 먼저 찬다. 앞선 df 중심 사전점검으로 이 한도를 놓쳤다.
- rank0의 resumed epoch01 weights 저장에서 EDQUOT. rank1은 다음 barrier, rank0은 integration 오류 처리 후 다른 gather로 진입하면서 collective 순서가 어긋났다. 이어진 1EB CUDA 할당 오류는 정상 모델 메모리 요구량이나 기존 메모리 게이트 실패의 증거가 아니다.
- `3208442` report.json 없음. straight 완료 및 resumed 일부 진행을 CP4 통과로 취급하지 않는다.
- NVMe B6 결과 점유 실측: `runs/b6_int4`66GiB, `runs/b6_resume_diagnostic`81GiB. **기존147GiB는 삭제/이동하지 않았다.** NVMe 여유 자체는 복구되지 않았으며, 정리는 별도 승인 대상이다.

## 저장소 검증

NVMe/HDD는 동일 Lustre 프로젝트22169지만 다른 pool quota다. 경로 이름만 믿지 않고 실제 stripe layout 및 pool별 quota를 확인했다.

- `/work/nvme/bilv`: ddn_ssd, used574895272KiB/soft524288000/hard576716800.
- `/work/hdd/bilv`: ddn_hdd, 초기used10884KiB/soft1048576000/hard1153433600.
- 새 preflight는 실제 projectID·상속P·stripepool·pool quota·statvfs를 대조하고 최소100GiB를 요구한다. soft/hard 중 작은 유한한 한도로 남은 용량을 계산한다.
- 실제 호스트 Python에서 NVMe는 거부, HDD는 통과했다. source/cache 및 다음 결과 모두 HDD를 사용한다. 기존 NVMe 결과를 정리하지 않고도 다음 시험이 가능하다.

## 수정과 검증

로컬 `/tmp/b6-integration` 커밋 **b4dd15e14541fc7ff765a8fa138980af862dd8a2**.

- `b6_runtime.checkpoint_io`: rank-local I/O 완료/실패를 포착한 뒤 기존 검토된 MVP `_collective_preflight`의 bounded status transport를 재사용한다. MVP 파일 자체는 수정하지 않았다. collectives가 들어가는 sharded backend/automatic naming은 거부한다.
- B6 trainer의 initial identity, latest/epoch/step checkpoint 각 로컬 저장 단계 및 완료 마커에 적용. 이미 발행한 step checkpoint 교체는 거부한다.
- integration sbatch에 `B6_RESULTS_DIR`과 `/b6-results` bind 및 실제 quota 사전검사를 추가했다.
- 두 rank의 weights/state/pending-grad/latest/step/identity 실패와 기존 step 보존을 실제 분산 fault injection으로 검증했다. 수정 전3실패, 수정 후 확장7사례 통과.
- TF4.57.6 관련 스위트 **63passed/91subtests**. TF4.46.3 및 MVP 핀 포함 **57passed/78subtests**.
- 실제 SIF CPU 초기 시험은 자동 MPI singleton 선택으로 실패했다. fixture에 Gloo를 명시하고 실제 world_size=2/rank를 단언해 재실행, **20passed**(fault7+storage13). 실패 로그도 보존했다.
- 독립 검토 `66a97c01-9567-446`: 다음 bounded2GPU CP4 APPROVE. 추가로 CUDA/NCCL fault smoke를 동일 allocation에서 먼저 실행하도록 wrapper와 fixture cuda opt-in을 추가, `b9bd138e-901f-4c4` 재검토 APPROVE. 실제 NCCL7사례는 다음 잡에서 검증되며 아직 통과 주장하지 않는다.

모델·메모리 벤치·config 바이트와 compile_b6 AST가 f1ee718과 같음을 검증했다(`memory_gate_path_unchanged.json`). 변경은 checkpoint I/O와 storage 배치이며, optimize_ddp=False 및 측정된 모델 실행 경로가 유지돼 메모리 벤치3208285를 반복하지 않았다. 정식CP4는 새로 실행한다.

## HDD 재실행

- 원격 source/cache: `/work/hdd/bilv/wcyim/transgenic/repo_b6_io_20260924`.
- 원격 스냅숏 **860bbfc2ba97e2bc659a1eee08321233d1f03e6d**.
- source/config/test220파일 SHA256 일치. 매니페스트 `source-b4dd15e.sha256`, 자체SHA256 `9406725acee1569d575d48304063e063c030963529ee30335b702531cb98910f`.
- wrapper `/work/hdd/bilv/wcyim/transgenic/run_hdd_cp4.sh`, SHA256 `7d035597865c4ebac34f1d5d3258c3127b8c90da57fac93b4e794b1fba82362e`.
- **잡3209815**, GPU2/1h30m/ghx4-interactive. 같은 allocation에서 **실제 NCCL fault7사례 통과 후에만** 정식 DATA=wp3-fixtures CP4를 실행한다. GPU mask를 바꾸지 않고 할당된 두 장치만 warmup한다. smoke의 OMP/MKL 설정은 smoke 명령에만 적용된다.
- 제출 전 활성B6잡0, 잔액1023, HDDquota남음 약999GiB 확인. noclobber 제출시도 저널 후 sbatch1회, rc0.
- 23:07PDT 실측 PENDING(Priority). 예상 시작 **09-25 06:59PDT**(스케줄러 추정, 변동 가능).

### 30분 확인 경로

**NVMe가 아니라 HDD 경로다.**

- 로그: `/work/hdd/bilv/wcyim/transgenic/repo_b6_io_20260924/b6-int4-3209815.out`.
- 보고서: `/work/hdd/bilv/wcyim/transgenic/runs/b6_int4/run_3209815/report.json`.
- 먼저 NCCL pytest7사례 PASS 여부를 확인한다. smoke가 실패하면 CP4는 미실행이며 report가 없는 것이 정상이다.
- 그 뒤 정식 report status/failures/resume_byte_identical/generation/3link 결속과 compile_policy를 확인한다. PASS 후 같은 HDD설정으로 synthetic 별도 제출. 현재synthetic 미제출.

기존 실패/진단 산출물·B5·운영 학습·main 병합·푸시는 보존/무변경이다. NVMe의 공간 회복은 별도 승인 없이는 수행하지 않는다.
