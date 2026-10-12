# C2 재개 결정성 복구 — 2026-10-12 재시작 정본

**후속 실행 상태(15:22PDT 이후): [C2_REPLAY_PROGRESS_20261012](C2_REPLAY_PROGRESS_20261012.md).** 아래는재개기준의보존기록이다. 현재구현head188bf3e(전체독립리뷰PASS), **4GPU검증3369338제출(15:48PDT)**. live확인후계속하며중복제출금지. 현재agent/시험/후속상태는위진행기록과resume를우선한다.

## 사용자 지시와 우선순위

- C2 우선. 기존 요청인 원인 수정·검증·자동 이관·학습 재개를 계속한다.
- **2026-10-12 명시 지시: 문제가 해결되고 검증이 통과하면 추가 승인 없이 C2 본학습을 바로 시작한다.** 과학적 검증 조건을 생략하라는 뜻은 아니다. 같은 승인을 다시 묻지 않는다.
- 현재 요청: 전체 작업 푸시, resume/intent(intend)/todo/quarantine 갱신 후 clear 가능한 상태로 보존.
- 구현 브랜치는 main에 병합하지 않는다. main에는 현재 작업 문서/증거/기존 미커밋 내용을 보존하고 구현은 각 브랜치에 남긴다.

## 현재 결론 — 수정 완료 아님

**재개 후 컴파일 이력이 달라지는 것이 관측된 계산 차이의 원인임을 통제 실험으로 확인했다.** 운영용 복구 코드는 아직 구현하지 않았다. 다음은 컴파일 이력 저장·복원 설계/회귀 시험/구현/실제 체크포인트 일치 검증이다.

최근 실측 2026-10-12 14:01 PDT: 학습/진단 RUNNING 없음. C2 본학습 보류, 원본 checkpoint200 보존. 진단 복구 사본의256은 운영 체크포인트로 승격하지 않았다. 후속 제출은 live Slurm 조회 후 진행한다.

| 잡 | 결과 | 의미 |
|---|---|---|
|3345919|FAILED|원본 C2, update280 rank3 compile event3가 기존2한도 초과. latest_state200/epoch0/step4800/pending0 보존. OOM 아님.|
|3360978|COMPLETED|실제190샘플로 기존 guard 실패 재현; Inductor label구간3개 확인.|
|3361110|COMPLETED|수정 정책 실190loss 완전일치+108경계/모드전환 통과, 총12graph, peakallocated56.855GiB.|
|3361386|FAILED1:0|200→256 재개 후 손실 차이 발견하여 의도적USR1. trainer 정상 저장·종료; batch 실패는 시험400최소조건 미달.|
|3361666|FAILED2:0|세학습phase는 정상. old/new200→210 정확일치; 연속/재개210→220은609/770모델텐서·optimizer 불일치. 입력/RNG는 전부일치.|
|3364480|COMPLETED0:0|4GPU/gh150, 39분17초, 10-11 23:26–10-12 00:06PDT. 컴파일 이력 분리 진단12보고서 검증 완료.|

### 3364480 해석 (독립 검토 완료)

동일 checkpoint210·실입력micro5041·모든 parameter/buffer·CPU/CUDA/Python/NumPy RNG, optimizer update0:

1. cold: 기존3361666 C의 첫손실과4rank 모두 정확일치; graph0→1.
2. primed: 역사적 B micro4801–4805를 각rank forward/backward 후 모든 상태 복원. B의 첫손실과4rank 모두 정확일치; graph3→3.
3. primed_reset: 같은 primers/복원 후 Dynamo만 reset. cold와 logits까지4rank 모두일치; graph3→4.

rank2: cold3.4423773288726807 / warm3.4435665607452393; rank3: cold3.4890518188476562 / warm3.48885440826416. rank0/1은 첫forward에서 logits도 동일(이전 긴시험은micro5042부터차이). DDP bucket rebuild는 전부0. 단순 난수 복원 실패·입력순서·optimizer 업데이트가 최초 차이의 원인이 아니다.

실험 성공은 전체 재개 수정 성공이 아니다. 4099updates·epoch평가·최종완료는 미검증이다.

## 코드와 환경

- 현재 복구 구현 worktree `/tmp/b6-c2-recompile`, branch `b6-c2-recompile-20261011`, HEAD **d7b1a37**, clean. 새 컴파일 이력 복구 구현은 **아직 시작하지 않았다**.
- `3f81fb6`: 완료검증 이전DONE/비정상exit성공판정/최종publication 충돌복구 수정. 실제NGC57시험56pass+1skip, 독립리뷰PASS.
- `d7b1a37`: graphs_per_tier3/recompile_limit18 및 정확한 구16/2→신18/3 방향의 identity호환. 실제NGC75시험74pass+1skip. 다른52identity필드는 엄격히 유지.
- 기준 `/tmp/b6-compile-static`, branch `b6-compile-static-len`,233574a.
- 실재생 도구 `/tmp/b6-gpu-diag-static`, branch `b6-gpu-mem-diag-static`,864cde0.
- integration `/tmp/b6-integration`,826e68e; archive `/tmp/b6-checkpoint-archive`,3b4eb0d.
- DeltaAI 접속: 로컬 `ssh pronghorn` → `ssh dtai` (userwcyim). nestedargv는 Python shlex.join/subprocess로 처리.
- WORK `/work/nvme/bilv/wcyim/transgenic`; SIF `/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif`.
- 원격 코드 `/work/nvme/bilv/wcyim/transgenic/repo_b6_d7b1a37`; snapshotgit b254ff4b4b1e640f5b570c00c58578bff16ba9a3, SOURCE_COMMITd7b1a37.
- NGC Python3.12/torch2.14a0; hostPython3.6.15. 로컬 torchPython `/home/wyim/miniconda3/envs/pdb/bin/python`; deps `PYTHONPATH=/tmp/b6-spec-review-deps:src:.`(accelerate1.10.1/duckdb overlay).
- source snapshot·입력·기존 checkpoint는 변경하지 않는다. 원본RUN `runs/b6_wp5_pilot_C2`, 진단256사본 `runs/b6_wp5_pilot_C2_recovery_20261011`.

## 반드시 보호할 운영 상태

- 자동 continuation controller PID3276938 종료됨. `ops/c2_recovery_20261011/continuation_state.json` phase=`blocked_determinism`, jobs=[]. **기존 컨트롤러를 그대로 재가동하지 말 것.** 새 검증/소스/잡ID로 새 계획을 만들고 중복제출 방지 저널을 사용한다.
- 원본3345919실패표식·checkpoint200 그대로. 운영경로 rename/promotion·새 C2본학습 제출은 아직 없었다.
- 구C2후속3345920 JobHeldUser/Nice0. B6 3345917/18 JobHeldUser/Nice1100. 해제 대신 검증된 새 소스 연결이 필요하다.
- B5 3309786 Priority/Nice1100,3309787–90 Dependency.14epoch완료,15번째88%,global30819,best14val0.5550. 최신live재조회 후 보고한다.
- C2 최종4099 검증 후 남은B5/B6 Nice0복구. C2우선 유지.
- B5epoch01–08 99.06GiB pronghorn이관 완료,09–14/latest/best14보호. 재이관 불필요. 목적지 `/data/gpfs/assoc/pgl/data/Transgenic/archives/checkpoint_archive/run-a466a381e8adf981/`.
- NVMe quota 최근진단시 여유167.39GiB(추가진단 사본 이후 다시측정 필요). HDD free170.71GiB는과거치;786GB증거DB삭제금지. pronghorn약21TiB는과거치.

## 다음 구현 방향 — 설계 중, 아직 승인된 완성 설계 아님

직접 얻은 근거를 보존하는 최소 복구가 필요하다. 첫입력 힌트에 따라 생성되는 decoder kernelfusion이 달라진다(예:rank2 L61440의 첫label18 vs312,77 vs75kernels). 임의의 수치허용오차·학습설정 변경으로 우회하지 않는다.

검토 중인 방식: 실제 compile event가 발생한 전역 forward위치의 **각rank 실제 companion 입력**을 descriptor/hash로 기록하고 checkpoint에 결속한다. 재개 시 같은 순서·train/eval view/grad context로 실행한 뒤 parameter/전체buffer registry/모든RNG/미적용gradient를 정확히 복원한다. optimizer/scheduler/학습step은 진행시키지 않는다.

- rank별event목록을 독립적으로 돌리면 DDPcollective 순서가 어긋난다. 전역 event union의 동일 slot을 모든rank가 함께 재생해야 한다. 가짜 padding 예제로 처음특수화를 만들지 않는다.
- 후보 구현은 forward후 any-rank-event collective → 모든rank의 해당batch row/epoch/view/입력hash 저장. train/eval loop가 이미 동일 collective순서를 갖는지 확인. 작은 JSON descriptor를 쓰면 tensor payload폭증을 피할 수 있다.
- 현재 5primer 실험을 event-only로 압축해도 같은 결과인지 별도 증명이 필요하다. fullprefix5와 globalunion이 실제같은지는 확인 가능하나 임의일반화 금지.
- legacycheckpoint200에는 journal이 없다. 원본drawprefix 및checkpointSHA에결속한 **명시적 bootstrap** 필요. rank3 originalrow3915, labels4501 재컴파일은checkpoint200이전이므로 빠뜨리면안된다. drawlog만으로 관측한compileledger라고 주장하지않는다.
- 3364480은진단용으로FX/AOT local/remote cache를모두껐다. **운영cache설정및rank간공유환경에서도복구정확성을검증**해야한다. runtime/compiler fingerprint·autocast/checkpointing·mode·shape/stride/dtype을이력에결속한다. backendcache직렬화만으로guard/dispatch이력이복원된다는증거는없다.
- 저장/복구는 latest_state뿐아니라 epoch checkpoint·pending grads·eval전환까지 포함한다. 최초파일누락/변조/identity/세계크기/입력hash불일치는 거부한다.
- 핵심파일 currentworktree `train/train_HyenaTransgenic_B6.py:114`(run_b6), `src/transgenic/training/b6_runtime.py:621`(compile_b6), `:650`(CompileGuard). graft query/callers 후 수정한다.
- 먼저 실패회귀→최소코드→local/NGC회귀·독립리뷰→실4GPU 연속/재개 model/optimizer/scheduler/RNG/gradient 정확일치. 원래중단전210loss3.2567708492279053와의미해결차이도설명·검증한다.
- 그후200원본의새검증사본에서280이상/400저장·재개 확인→사용자재승인없이C2본학습4099재개. 다음 링크 입력 재검증·저장공간·체크포인트 보존·실패표식 판정을 유지한다.

## 증거·도구 (중복 실행 금지)

- 누적기록 `notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md`.
- 모두 `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/` 아래:
  - `3361666/`: A/B/B210/C trace·comparison·generated B/C wrappers; analysis_summary.json.
  - `3364480/`: cold/primed/primed_reset12report·comparison·source/meta.
  - `probe.py`, `probe.sbatch`, `test_probe.py`(local3+NGC3PASS).
  - `first_forward.py`, `first_forward.sbatch`, `test_first_forward.py`(local2+NGC2PASS).
  - `FIRST_FORWARD_PLAN.md`, `FIRST_FORWARD_ACCEPTANCE.md`, `POST_RUN_ACCEPTANCE.md`.
- 원격 경로 `runs/c2_resume_determinism/3361666` / `runs/c2_first_forward/3364480`。
- immutableops `ops/c2_resume_determinism_20261011/v2` / `ops/c2_first_forward_20261012`。
- first_forward 비교 exit0은 observations 완료일 뿐이다. 이번 추가 검수는 PASS지만 운영 수정 PASS는 아니다.
- reviewer `/root/c2_replay_review`는 read-only 검토자다. 새 세션에서 없으면 review 스킬에 따라 독립 검토자를 새로 배정하며, 저장되지 않은 에이전트 기억에 의존하지 않는다.

## clear 후 첫 행동

`.atl/resume.md`→`.atl/todo.md`→이 문서→`.atl/quarantine.md`를 읽고 git HEAD/worktrees와 live Slurm을 확인한다. 완료된 진단을 재제출하지 말고 C2 컴파일 이력 복구를 구현한다. 사용자는 수정·검증 후 즉시 본학습 시작을 승인했으므로 다시 묻지 않는다.

## Push 기록

- main 전체업무스냅샷 **8ef1871732afffab6ce4adb83a84d3dfd0b49808**,3005files. 과학진단log71개명시포함.
- `git push --atomic --all origin` 성공. 로컬 **103개 branch 모두 원격SHA와일치** 확인. 구현branch를main에병합하지않고각각보존했다.
- [원격검증영수증](PUSH_RECEIPT_20261012.json). 이결과기록을위한후속문서커밋도main에푸시한다. 최종tip은gitlog/origin/main으로확인.
- 기존ignore대상DB/모델/재생성cache는Git대상이아니며로컬/HPC원본보존. 오래된빈index.lock은활성git/FD부재확인후 .git/index.lock.stale-20261012로보존. 불완전nested .git6개는staging중에만이동하고모두원위치복원,source내용은일반파일로보존.
