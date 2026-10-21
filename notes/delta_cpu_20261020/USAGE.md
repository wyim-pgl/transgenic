# Delta CPU 사용 준비 — 2026-10-20

NCSA 공식 문서를 확인한 사용 안내다. **10-21 승인 후 계정bilv-delta-cpu/최초잔액1,000core-hours와 실제0GPU CPU잡23104942의 정상 종료를 확인했다.** 기존bilv NVMe/HDD 공유접근과 작은helper무결성도 실제CPU노드에서 통과했다. 신청에 사용한1,000 ACCESS credits와 당시DeltaAI GPU잔액1,641hours는 과거 사용자제공값이며 현재잔액과 구분한다. 현재 상태 정본은 [resume](../../.atl/resume.md)와 [TODO](../../.atl/todo.md)다. 아래 승인전 절차와10-20 관찰은 준비 당시의 기록이다.

## 10-21 CPU 활성화 및 실제 작업 검증

- `accounts`: bilv-delta-cpu, balance/deposited1,000hours. `sbatch --test-only` PASS 후 정확히한번 제출했다.
- 실제잡 **23104942**, cpu/1core/1GiB/10min/no-requeue, cn103/x86_64/Python3.9.21. 7초 후 parent/batch/extern COMPLETED0:0. 실제할당 `cpu=1,mem=1G,node=1,billing=1000`, GPU항목없음.
- 공유NVMe/HDD/기존archivehelper경로 접근과고정helper3개의SHA검사 PASS. 모델·입력의대형해시나이관·삭제는수행하지않았다.
- 실행후 `accounts` 정수잔액999, `jobcharge` 0.00SU. 표시정밀도가 달라1SU가 실제소모됐다고 단정하지않는다.
- 원격정본 `/u/wcyim/transgenic_ops/delta_cpu_preflight_20261021`; [사전검사](STAGING_20261021.json), [제출](SUBMISSION_20261021.json), [실행결과](PREFLIGHT_RESULT_20261021.json), [과금조회](CHARGE_20261021.json).
- 후속 **23105376**도실제CPU1core/1GiB/0GPU/7초/COMPLETED0:0. 기존Pronghorn의gffcompare0.12.6 바이너리SHA `a440e1533611a40a51cf2b5b35a2e15ec6b885660dd35f6b531e0d65601a963a`를보존해DeltaCPU전용경로 `/u/wcyim/transgenic_ops/wp5_cpu_scoring_20261021/bin/gffcompare`에서실행했고,작은정확/변형intron-chain fixture를통과했다. 기본PATH에설치된것은아니며 실제WP5평가통과를뜻하지않는다. [실행증거](../ccloop_current_model_20261021/SCORER_RESULT.json).

Delta는 CPU core-hours를 별도 자원으로 제공한다. 앞으로 파일 복사·해시·아카이브를 CPU 노드에서 처리하는 데 적합하다. 기존 B6/C2의 실제 4GPU 학습·재개 동일성 검증은 DeltaAI에서 수행한다. [NCSA Delta 할당 안내](https://delta.ncsa.illinois.edu/delta-allocations/)

## 1. 승인 후 로그인과 CPU 계정 확인

1. ACCESS에서 해당 자원의 승인·할당 상태와 프로젝트 사용자 연결을 확인한다. ACCESS 할당의 멤버 관리는 ACCESS에서 한다. 기존 NCSA 사용자명은 `wcyim`이며 새 사용자명을 임의로 만들 필요는 없다. [NCSA 프로젝트 관리](https://docs.ncsa.illinois.edu/en/latest/account-mgmt/group-mgmt.html)
2. 아래 Delta 호스트에 접속한다. NCSA 비밀번호와 Duo MFA를 사용하며, 일반 사용자 SSH 공개키 로그인은 비활성화되어 있다. 웹 확인은 [Delta Open OnDemand](https://openondemand.delta.ncsa.illinois.edu/)에서도 가능하다. [NCSA Delta 로그인](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)

```bash
ssh wcyim@login.delta.ncsa.illinois.edu
```

로그인 후 조회:

```bash
hostname
id
accounts
quota
sinfo -p cpu -o '%P %a %l %D %f'
scontrol show partition cpu
```

`accounts`의 **Project 열에 표시된 실제 CPU 과금 계정명과 잔액**을 기록한다. CPU/GPU 과금 이름은 구분된다. 기존 저장소 코드 `bilv`를 보고 새 CPU 계정명을 추측하지 않는다. CPU 항목이 없으면 활성화가 확인되지 않은 상태다. [NCSA 과금 계정](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html#local-account-charging)

`quota`는 저장소 사용량을 보여준다. CPU 자원 추가가 기존 `bilv` 저장소 할당량 증가를 의미하지는 않는다. 저장소 프로젝트 연결과 실제 한도는 출력으로 확인한다. [NCSA 파일 시스템·quota](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)

## 2. 파티션과 과금

| 파티션 | 최대 시간 | 과금 배수 | 이 작업에서의 용도 |
|---|---:|---:|---|
| `cpu` | 48시간 | 1.0 | 첫 확인 및 이후 복사·해시의 기본값 |
| `cpu-interactive` | 1시간 | 2.0 | 짧은 대화형 디버깅 |
| `cpu-preempt` | 48시간 | 0.5 | 중단·재시작이 검증된 작업에 한정 |

`cpu`를 명시하고 GPU 요청 옵션을 넣지 않는다. 기본 파티션은 GPU 계열이므로 생략하지 않는다. `/work/hdd` 사용은 `--constraint=work`로 표시한다. 선점 큐는 중단될 수 있으므로 첫 이관에는 일반 `cpu`를 사용한다. [NCSA 파티션·파일 시스템 constraint](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)

CPU의 1 SU는 **1코어·메모리 2GB 이하·1시간**이다. 실제 사용률보다 예약한 코어/메모리 중 큰 쪽을 기준으로 과금한다. 대략 `max(코어 수, 메모리 GB / 2) × 실행시간 × 파티션 배수`로 예산을 잡되 최종 차감은 `jobcharge`로 확인한다. 여기서 공식 GB는 **10⁹ bytes**다. `--mem=2G`를 2GB와 같다고 가정하지 않고, 첫 확인은 `--mem=1G`로 잡는다. [NCSA SU 기준](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html), [Slurm 메모리 단위](https://slurm.schedmd.com/sbatch.html#OPT_mem)

`cpu`의 1코어·1GiB가 10분 모두 실행되면 계산상 약 0.167 SU다. 동일한 조건의 1코어 작업에는 1,000 core-hours가 총 약 1,000시간이고, 4코어이면 약 250시간이다. 이는 병렬화 효율이나 실제 저장소 처리량을 보장하는 수치가 아니다.

## 3. 기존 데이터와 실행 환경

Delta와 DeltaAI는 `/work/hdd` 및 `/work/nvme`를 공유한다. 현재 확인 대상은 다음 기존 경로다. 새 CPU 계정 이름을 이 경로의 `bilv` 자리에 넣지 않는다. **공유 파일 시스템이라는 사실과 `wcyim`의 실제 경로 접근 권한은 별도로 확인한다.** [NCSA DeltaAI 공유 저장소](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html)

```text
/work/nvme/bilv/wcyim/transgenic
/work/hdd/bilv/wcyim/transgenic/checkpoint_archive
/work/nvme/bilv/wcyim/transgenic/ops/checkpoint_archive/3b4eb0d
```

두 시스템의 `/u/wcyim` 홈은 서로 다르다. Delta CPU는 AMD EPYC 7763 기반의 x86 계열이고 DeltaAI는 Grace ARM 계열이다. `uname -m`에서 각각 `x86_64`, `aarch64`를 확인할 예정이다. 기존 ARM용 NGC 26.08 SIF·가상환경·바이너리의 Delta CPU 실행을 가정하지 않는다. [Delta CPU 구조](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html#cpu-compute-node-specifications), [DeltaAI 구조](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html), [홈·구조 차이](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html#globus)

기존 아카이브 도구 `3b4eb0d`는 Python 3.6 표준 라이브러리 기준으로 준비되어 있어 이식 후보지만, Delta의 Python 버전·필요 모듈·파일 시스템 동작은 새로 확인해야 한다. 초기 확인은 도구를 실행하지 않고 메타데이터와 작은 도구 파일의 무결성만 조회한다. 기존 검증 근거는 [운영 handoff](../RESTART_HANDOFF_20261012.md)와 [quarantine](../../.atl/quarantine.md)에 연결되어 있다.

로그인 노드는 제출·상태 조회에 사용한다. 대용량 복사·전체 데이터 해시·압축은 CPU allocation 안에서 실행한다. 작업 I/O는 `/work`에서 처리하고, 홈은 작은 스크립트/로그에 사용한다. `/work`에는 스냅샷이 없으므로 이를 독립 백업으로 간주하지 않는다. [NCSA 로그인 노드 용도](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html#login-nodes), [NCSA 저장소 용도](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)

## 4. 첫 CPU 작업: 1코어·1GiB·10분

실행 파일: [delta_cpu_preflight.sbatch](delta_cpu_preflight.sbatch). 승인 후 이 작은 파일을 Delta 홈의 별도 작업 디렉터리에 놓는다. 자원 요청은 다음과 같다.

```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --constraint=work
#SBATCH --no-requeue
```

위 블록은 자원 지시문 요약이며, 실제 제출은 링크된 전체 스크립트를 사용한다. 과금 계정은 `#SBATCH` 안에 환경변수로 넣지 않고 CLI 인자로 전달한다. `#SBATCH`는 셸 변수 확장을 하지 않는다. [Slurm sbatch](https://slurm.schedmd.com/sbatch.html)

Delta 로그인 셸에서 `accounts` 출력의 실제 CPU 계정명으로 값을 채우고, 스크립트를 둔 디렉터리에서 실행한다:

```bash
DELTA_CPU_ACCOUNT='REPLACE_WITH_ACTUAL_CPU_PROJECT_FROM_ACCOUNTS'

# 실제 계정명을 채운 뒤 먼저 제출 가능성만 확인한다.
sbatch --test-only --account="$DELTA_CPU_ACCOUNT" \
  --chdir="$PWD" delta_cpu_preflight.sbatch

# test-only가 통과하고 CPU 활성화가 확인된 뒤 한 번 제출한다.
sbatch --parsable --account="$DELTA_CPU_ACCOUNT" \
  --chdir="$PWD" delta_cpu_preflight.sbatch
```

`--test-only`는 실제 잡을 제출하지 않으며, 실행 시 파일 접근·Python 호환성까지 증명하지 않는다. 실제 `sbatch`가 반환한 잡 ID를 기록한다. [Slurm test-only](https://slurm.schedmd.com/sbatch.html#OPT_test-only)

첫 작업의 범위는 Slurm CPU/GPU 요청·할당, 호스트/아키텍처/Python 버전, 위 세 경로의 메타데이터·접근 권한, 작은 helper 파일 해시 확인이다. 데이터 복사·checkpoint 해시·삭제는 포함하지 않는다. 작업 본문은 데이터를 읽기만 하지만 Slurm stdout/stderr 로그는 생성된다. 권한 조회에서 쓰기 가능이 표시되어도 실제 쓰기·quota·원자적 파일 연산의 성공까지 검증한 것은 아니다.

## 5. 결과·잔액 확인과 이후 전환

아래 명령은 **Delta**에서 실행한다. `DELTA_CPU_JOB_ID`에는 방금 제출한 새 CPU 잡 번호를 넣는다.

```bash
DELTA_CPU_JOB_ID='REPLACE_WITH_NEW_CPU_JOB_ID'
squeue -u wcyim
scontrol show job "$DELTA_CPU_JOB_ID"
sacct -j "$DELTA_CPU_JOB_ID" \
  --format=JobID,Account,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,ReqTRES%80,AllocTRES%80,MaxRSS
accounts
jobcharge -a "$DELTA_CPU_ACCOUNT" -d 1 --detail
```

`squeue`는 대기/실행 상태, `sacct`는 종료 결과와 단계별 자원 기록을 확인하는 데 사용한다. parent/batch의 `COMPLETED`, `ExitCode=0:0`, 로그 내용, CPU 계정·`cpu` 파티션·GPU 할당 0을 함께 확인한다. 대기 중인 잡의 `AllocTRES`가 비어 있는 것은 GPU 0 검증으로 보지 않는다. [NCSA 작업 조회](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html), [Slurm sacct](https://slurm.schedmd.com/sacct.html)

`QOSGrpBillingMinutes`가 보이면 요청 자원·시간에 비해 해당 계정 잔액이 부족하거나 같은 프로젝트의 다른 요청이 영향을 주는 상황을 확인한다. 실제 차감은 `jobcharge` 상세와 `accounts` 잔액으로 본다. [NCSA 과금·잔액 조회](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html)

승인 후 남은 확인 순서:

- CPU 계정·잔액과 `wcyim` 연결을 확인한다.
- 위 작은 CPU 잡에서 0GPU 할당과 기존 `bilv` 경로 접근을 확인한다.
- 별도 작은 fixture로 아카이브 도구의 Delta CPU 호환성·쓰기·quota·재시작 동작을 검증한다.
- 이후 새로운 복사/해시 작업부터 일반 `cpu` 파티션으로 옮기고, 실측 처리량·메모리·SU로 자원 요청을 조정한다.

이 문서 작성 시점의 DeltaAI 이관 잡 **3425279**는 별도 진행 작업이다. CPU 신청/승인을 근거로 이를 취소하거나 B5 epoch09–11 이관을 중복 실행하지 않는다. 기존 archive operation의 완료 영수증과 live 상태를 먼저 확인한다. B6의 실제 4GPU 동일성 검증 게이트도 그대로 유지한다. 최신 진행 상황은 [B6 실행 계획](../b6_pilot_resume_20261020/PLAN.md)을 따른다.

## 10-20 실제 접속·공유 확인

- dt-login03.delta.ncsa.illinois.edu/x86_64,UID97434(wcyim),delta_bilv22169. `accounts`는exit60,`No slurm account association found for user wcyim.` 따라서 현재CPU잡제출조건미충족.
- 기존 `/work/nvme/bilv/wcyim/transgenic`와 `/work/hdd/bilv/wcyim/transgenic/checkpoint_archive`에서 DeltaCPU→DeltaAI→DeltaCPU 작은파일R/W·SHA일치·소유시험leaf정리 확인. 기존C2/B5 metadata SHA불변. 대형학습입력/모델복사없음.
- Python3.9.21, lfs/rsync/sbatch/scontrol/sacct 사용가능. 기존archive guard의작은fixture1PASS0.238s.
- 증거: [접속](ACCESS_CHECK.json), [양방향 공유](SHARED_STORAGE_CHECK.json), [도구호환](TOOLCHAIN_CHECK.json).
- pronghorn의 인증된 SSH socket `/data/gpfs/home/wyim/.ssh/cm-delta-cpu`를 재사용할수있다. 크레딧처리후 `accounts`에서실제CPU계정/잔액을확인하고준비한읽기전용preflight부터제출한다.
