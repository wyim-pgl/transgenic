# 재개 갱신 — 2026-09-15 15:26 PDT 이후

- 저자 metric 승인 완료: 학습 composite 유지, paired primary 순수 weighted CE + composite 병기; delta=off−on, 0.005 nats 유지. primary는 증거 양성 타일별 차이의 동일 가중 평균.
- DeltaAI 인증 복구 확인(15:26 PDT): seed123 best→epoch_04, eval0.65254; 3131537 Priority대기, 후속 Dependency대기. repo_seeds8f049eb/잡 무변경.
- CPU paired metrics1d3db4 완료(task/b6-mvp-metrics, harness alias pi-agent-3466292f-2e28-4d6): 컨트롤러34 CPU tests PASS, 독립리뷰 spec PASS/quality APPROVE. [검증](../notes/b6_mvp_metrics_controller_20260915.md). main 미병합, trainer 통합/GPU 인증 아님. weights-only init도3a63314 완료(b6-mvp-weights-init-review1): runtime_identity.txt 수정 후 컨트롤러46tests PASS·독립재검토 spec/quality PASS. fixed runtime도a314bf9 완료(b6-mvp-fixed-runtime-review1): 컨트롤러20runtime+46regression PASS·독립재검토 ACCEPT(보고서 metadata검증범위 minor는 controller기록에 한정). paired분산집계/원자적final helper도0a8120b 완료(won/b6-mvp-finalize): controller최신39finalize+46baselinePASS/독립재검토PASS; runtime20은직전3fe33cb에서PASS. exploratory CLI도1c47ae5 완료(pi-agent-49252832-1f04-4d6): controller145 CPUtests PASS/독립재검토PASS. 최신누적코드시작점f847cb8(pi-agent-5b846a12-93fa-4c3), main미병합. CPU구현전체연결리뷰+GPUtopologygate재검토PASS/APPROVE, controller153testsPASS. realcheckpoint/SIF/Hyena/GPU launch-readiness는아직없음; 다음실제환경인증. 기술후보기록 notes/b6_mvp_technical_candidate_20260915.json faab8387…와inventory/receipts companion작성·해시확인완료(생산승인false). 원격17:26 PDT blastx9종DONE/fullDB SUCCESS·overlap0·valid증거1442, 후보DB수용감사 남음. 현재 상태는 todo 맨 위.
- 아래 15:13 기록의 metric 미승인·인증실패는 위로 대체. 나머지 안전 제한 유지.

# 재시작 정본 — 2026-09-15 15:13 PDT

**먼저 [`../notes/RESTART_HANDOFF_20260915.md`](../notes/RESTART_HANDOFF_20260915.md)를 읽는다.** 아래 과거의 “현재/다음 작업/승인 대기”는 이 블록과 상세 인수인계가 우선한다. 상태 근거는 `../notes/restart_status_20260915.txt`; 시간 경과 뒤 다시 조회할 것.

- **원격 실행:** blastx 9종 중 Sitalica·Athaliana·Bdistachyon·Gmax 양쪽 DB 완료. array **6164049**: 3 Osativa·4 Ppatens 실행, 5 Ptrichocarpa·6 Sbicolor·7 Vvinifera 대기(동시2). **중복 제출 금지.**
- **MVP 전체 DB:** **6164118** cpu-51 RUNNING 47분; 입력 해시 통과, builder 진행 중. `qc.json`/`SUCCESS.json` 아직 없음. 부분 DB/WAL은 성공 산출물이 아니다.
- **코드:** main `9a8c8e1`, 미커밋 문서/패널/graft 설정 보존. MVP 최신 **`pi-agent-a92f1af6-926c-40f` / `c7fa759`**(데이터·모델·dataset 포함), blastx 최신 **`pi-agent-cb2fd26a-e84c-456` / `950cf4c`**. 둘 다 main에 병합하지 않았다. 임시 agent worktree는 제거됐고 branch는 남아 있다.
- **검증:** CPU data86, model11, dataset+model20 테스트를 컨트롤러가 각각 재실행해 통과; 각 increment 독립 검토 통과. 실제 Hyena/SIF/GPU/DDP 인증이나 전체 테스트 통과로 해석하지 않는다.
- **저자 승인:** EST blastx, MVP, 별도 worktree, 8종 실행, 독립 작업 병행은 승인됨. 전체 A46 동결은 별개. **학습 손실 vs 검증 weighted CE 정의 질문은 아직 무응답.** 학습기 구현/새 GPU 학습은 미착수.
- **DeltaAI:** 15:13 재조회도 Kerberos/Duo 인증 실패. seed123의 현재 epoch/상태 미확인; 아래 epoch4 기록은 과거값. 원격 학습 checkout/job은 변경하지 않음.
- **주의:** 성공한 payload와 별개로 Slurm `.extern OUT_OF_MEMORY 0:125` 반복. 원인 미해결; 전체 정상 또는 실제 프로그램 OOM으로 단정하지 않는다.
- **Graft:** `graft init --yes` 완료(233files/2,478nodes/5,571edges). 새 AGENTS 지침을 따르되 main 그래프는 미병합 브랜치 기능을 담지 않는다.
- 살아 있는 다음 행동은 [`todo.md`](todo.md) 맨 위, 재발 방지는 [`quarantine.md`](quarantine.md)의 §1m.

---

## 보존된 이전 세션 기록 — 아래는 재시작 정본보다 우선하지 않음

# TransGenic 원고 리비전 — Plant Communications 재투고 (✏️ **2026-09-14 11:5x PDT — B6 증거 모드 결정·워크플랜 v1.2 `c4b9d30`·HEAD `9a8c8e1`; A46 v1.36 초안 + MVP 설계서 미커밋; 시드 123 epoch 4 best 0.6525, 링크 2 대기(11 h 체인)** — 아래 🔄 09-14 블록. 이전: ✏️ **2026-09-11 11:1x PDT — A45(v1.35, `8f049eb`): ~~세 시드 전부 26.08·4 GPU·미리 큐잉 3링크 체인으로 큐에 있음(123 `3131536`… · 456 `3131539`… · 789 `3131542`…)~~ ✏️ 09-12 11:53 PDT: **시드 123만 진행**(`3131536` RUNNING gh129 06:49 PDT~, epoch ≈ 10 h; `3131537`·`3131538` 대기), **456·789는 저자 지시로 취소·연기**(나중에 `submit_chain.sh`로 재제출); 시드 123의 1 GPU 런은 `seed123_1gpu_partial`로 보존·미보고; 리허설 4/4 PASS.** 아래는 09-10 오전 시점 — 2026-09-10 오전 PDT — SEED=123 epoch 2 진행, 첫 48 h 경계 ~~**오늘 21:45 PDT**~~ ✏️ 지남(21:45 PDT COMPLETED, 링크 2 `3128376` PENDING) · 저자 결정 **확인 시드 456/789 = NGC 26.08 compile + 4 GPU DDP**(A44 v1.34, `c646e76`) · 26.08 4 GPU 체인 리허설 PASS · epoch 경계 accumulation 잔여분 보존 수정 · ~~인증 2건 대기 후 제출~~ ✏️ **인증 4건 전부 PASS(3124099·3124229·3124347·3124348) → 코드 준비 완료; 22:00 PDT 경계 뒤 `auto_submit_456.sh`가 시드 456을 자동 제출, 789는 저자 결정**)

---

# 📌 정본 포인터

**최신 문안 상태 — 2026-09-14 15:25 PDT:** A46 3차 리뷰 원문의 남은 8건을 반영하고 MVP v1.2 draft(325행, sha256 `69968c4d…`)와 A46 rule 17을 동기화했다. 패널 TSV·코드는 변경하지 않았으며, 패널 테스트 8건 및 해시·인용문·CLI 재현 검증을 통과했다. 아래 12:5x의 “수정 진행 중”은 이 기록으로 대체한다. **최종 리뷰·저자 승인·동결·커밋은 아직 하지 않았다.** 다음 작업은 수정본 최종 검토 → 명시적 저자 승인이다. 항목별 근거·검증 범위는 [`notes/b6_a46_r3_wording_fixes_20260914.md`](../notes/b6_a46_r3_wording_fixes_20260914.md)를 읽는다. 원격 학습 상태는 재확인되지 않았다(DeltaAI 인증 실패).

- **지금 할 일**: [`todo.md`](todo.md) — 막힌 것·저자 결정 대기·승인 나면 바로 할 것이 거기 있다
- **쓰지 않기로 한 것**: [`quarantine.md`](quarantine.md) — 틀린 계획·실패한 시도·배제된 데이터. 다음 문제의 false positive·추가 피해·위협 분석 근거
- **진행 상태의 정본**: GitHub 이슈 wyim-pgl/transgenic **#4–#71** (열린 것 **48개**, 2026-09-08 실측). A40 사전 검토에서 나온 병합 게이트 강화 3건 **#69·#70·#71** 신규 등록, **#55·#56·#58 종료**. ✏️ 단백질 정렬 **#45·#44는 2026-09-09 종료**(수용 표 ~~`protein_acceptance_v1.tsv`, 6e08e01~~ → **`protein_acceptance_v2.tsv`, `7df570f`, A43** — v1은 mapping_ambiguous 누출, 라벨에 쓰지 말 것; #45 정정 코멘트는 **~~미게시, 저자 승인 대기~~ ✏️ **게시됨(09-09 14:4x PDT, 저자 "진행 해")**: https://github.com/wyim-pgl/transgenic/issues/45#issuecomment-5606950770, 이슈는 닫힌 채**). 열린 이슈 수는 재확인 필요(`gh issue list -R wyim-pgl/transgenic -s open | wc -l`)
- **계획**: `MASTER_PLAN_PC_20260901.md` (v3.7, 9주)
- **프로토콜(고정)**: `transgenic/revision/protocols/PROTOCOL_B1_frozen_v1.md` ✏️ **v1.35**(A45, `8f049eb`, 09-11) → **v1.36 A46 초안 미커밋**(09-14, 951~989행 + 로그 999행; B6 동결 규칙 17항 + 저자 결정 6건, Codex 재검토 대기) · **B6 계획**: `B6_EVIDENCE_MODE_WORKPLAN_v1.md` v1.2(`c4b9d30`, en/ko) · README `B6_WORKPLAN_README.md`(en/ko) · MVP 설계 `B6_MVP_PILOT_v1.md`(미커밋) · 이전: ~~**v1.30**~~ ~~✏️ **v1.33**(2026-09-09 낮, `7df570f`)~~ ✏️ **v1.34**(2026-09-10 오전, `c646e76`) — **A44** = 확인 시드 456/789를 NGC 26.08·compile·4 GPU DDP·`repo_seeds`에서 돌린다는 저자 결정, 그 비용(혼합 런타임 재현, 중립성 주장 없음), 체인·체크포인트·epoch 경계 accumulation 수정 4건, `runtime_identity.txt`. 이전 v1.33 A43 문장은 그대로 유효 — §3–9 고정, 개정은 §A1–**A43**(A43 = 단백질 지지 표 support_version 2, mapping_ambiguous 제외). ⚠️ **A36은 A37이 대체**(≥100 primary, 121은 민감도 arm). **A38**: 수집 전 제외 런은 산문이 아니라 선언 행 → `longread_v1` 재동결(`c2f4c2e8`, 215행). ⚠️ **A39**: §3.3 항목 2의 `-uf` 게이트는 **리드의 방향**(P(R=A))이지 모티프의 방향(P(T=A))이 아니다 — 문자 그대로 읽으면 27/27이 통과해 아무것도 안 걸렀다. 문턱 95 %는 불변
- **B5 데이터셋(§1, A40 동결 2026-09-06)**: `gpu:/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db` · **273,148행** · 23,534,252,032 B · MD5 `dda7e78c880c993435654b57dcd1c498` · 검증기 위반 0 · [새 동결 기록](transgenic/data/freeze/b5_full_a40_v1.freeze.json). 이전 `b5_full_v1`(23,491,260,416 B, 272,224행)과 동결 JSON은 보존됨.
- **저장소**: ✏️ **09-14 HEAD `9a8c8e1`**(phase 경로 B; 오늘 커밋 `6c86169`·`320de01`·`c4b9d30`·`839dab5`·`b1acb65`·`460de14`·`9a8c8e1`는 전부 `docs:` — **코드 무변경**, dtai `repo_seeds`는 `8f049eb` 유지·pull 금지) · **미커밋**: `revision/protocols/PROTOCOL_B1_frozen_v1.md`(A46 v1.36)·`revision/protocols/B6_MVP_PILOT_v1.md`(신규) · 이전: ✏️ HEAD **`df9a415`**(2026-09-10 11:0x PDT, 프로토콜 A44 epoch-ceiling 문단만; **코드는 `c646e76`과 동일**, repo_seeds/인증도 c646e76) · 그 전 `c646e76` (2026-09-10 오전, 푸시 완료; 오전 커밋 4개 — `35b1436` REPO_DIR + CPU 수 계승 + 통합 시험 TERM · `07486a8` preflight가 stdout 마지막 줄만 읽음 · `5433f27` latest_state rename 교체·`runtime_identity.txt`·preflight 엄격화 · `c646e76` epoch 경계 accumulation 잔여분 보존·epoch별 train loss·`KILL_AFTER_EPOCH` 시험·A44). 아래 옛 문장은 09-09 저녁 시점: ~~HEAD **`1a245ce`**~~ (2026-09-09 저녁, 푸시 완료; 저녁 커밋 8개 — `7beb803` 체인 launch-failure 재시도 · `0c3f4bd`/`a486305` 통합 시험 rank 표적·leg-1 재시도·ALLOC 스위치 · `ed10056` GPU 워밍업 + 제자리 재시도 · `4fe9780`/`3ed6413` USR1을 rank에만 · `9161508` **SIGTERM으로 전달** · `840384f` watchdog 강제 선점 · `1a245ce` errexit 수정). dtai `/work/repo`는 여전히 **`08f3e2b`** + `deploy/deltaai/train_b5.slurm` 한 파일만 origin/main(`1a245ce`)과 **바이트 일치**로 교체. ~~HEAD `7df570f`~~ (낮 서술 이하: 이날 커밋 4개 — `5bc4df8` Codex 리뷰 결함 5건 수정 + DDP 통합 시험, `14fe6a4` 통합 시험 compile 스위치, `7f20816` 벤치 nvidia-smi 계측, `7df570f` #45 v2 + A43). 아래 옛 문장은 09-09 아침: HEAD **`2021fd2`** (2026-09-09, 푸시 완료; 09-08 이후 커밋 15개 — 청크 정렬·A41·A42·DeltaAI 발사대 수정 13건·DDP·compile 스위치). dtai `/work/repo`는 **`08f3e2b`로 동결**(시드 체인 코드) — ✏️ **단, `deploy/deltaai/train_b5.slurm` 한 파일만 `5bc4df8`에서 꺼내 넣었다**(`git status` M, diff = wait 루프 rc + `exit 1`; 시드 123 다음 링크부터 적용). dtai `repo_dev`는 **`7f20816`**. 아래 옛 문장은 09-08 시점: HEAD `4ef3dc4` (**푸시 완료**, 2026-09-08). `#55·#56·#58` 종료. ⚠️ 커밋 메시지의 `Closes #55, #56, #58`은 **첫 번째만** 닫았다 — GitHub은 키워드 뒤 쉼표 목록을 파싱하지 않고 번호마다 키워드가 필요하다. 나머지 둘은 손으로 닫았다. A40은 저자 승인 후 **두 커밋으로 나눠** 들어갔다 — `f6e68d4`가 **실행된 그대로**(30파일), `5289077`이 **사후 테스트 강화**. 나눈 이유는 freeze가 `scripts/merge_b5_databases.py`(`dfd26632…`)와 `tests/test_merge_b5_databases.py`(`8ab1d1ba…`)를 SHA256으로 고정하기 때문이다 — `f6e68d4`의 트리가 그 두 해시와 바이트 단위로 일치함을 확인했고, 강화를 같은 커밋에 넣었다면 그 대응이 깨졌다.
- **DeltaAI 체크아웃 3개(2026-09-10)**: `/work/nvme/bilv/wcyim/transgenic/repo` = 시드 123 전용 동결 **`08f3e2b`** + `train_b5.slurm`만 **`ba24b19`**(리허설 통과본, 더 갱신하지 않음) · **`repo_seeds`** = 확인 시드 456/789 전용 동결 **`c646e76`**(HFmodels 복사됨, 학습 중 pull 금지) · `repo_dev` = 실험용(main 추종). SIF: 25.06 `transgenic-ngc.sif` sha256 `3b242b33…` · **26.08 `transgenic-ngc-26.08.sif` sha256 `1a5dee27…`**(torch 2.14.0a0+4fdf77b940.nv26.08). **모든 시간은 PDT로 적는다(저자 지시 09-10)** — DeltaAI 시계는 CDT = PDT+2.
- **📌 이 블록 검증**(인용 전 1분): `cd transgenic && git log --oneline -1` · `gh issue list -R wyim-pgl/transgenic -s all -L1 --json number` · `pytest -q`(**567 passed, 6 skipped**; 두 경로가 한 번에 돈다. skip 6은 torch 필요분으로 pgl-gpu에서 별도로 돈다). ⚠️ **로그인 노드에서도 컴퓨트 노드에서도 `duckdb` import가 실패한다** — `/lib64/libm.so.6: version GLIBC_2.23 not found`. 스위트는 컨테이너 안에서 돌려야 한다: `singularity exec -B /data/gpfs ~/scratch/singularity/micromamba.sif /bin/micromamba run -r ~/micromamba -n base python -m pytest -q` (2026-09-07 실측, 184–204초). ⚠️ **GPU 재빌드 같은 큰 잡이 도는 동안에는 스위트가 55초 → 410초로 느려지고 subprocess 테스트가 타임아웃한다** — 2026-09-05에 4건이 그렇게 실패했고 단독 재실행에서 44/44 통과했다. 실패를 보면 먼저 그 파일만 단독으로 돌려 볼 것. **추출·정리는 이 블록을 갱신해 주지 않는다** — 2026-09-03에 실제로 3커밋 뒤처진 채 남아 있었다
- **증거 드라이버 위치**: 정본은 `transgenic/revision/scripts/evidence/`(저장소 안), `evidence/<이름>`은 **심볼릭 링크**다. 고칠 때는 저장소 쪽을 고친다. 실행 중인 드라이버가 파일을 열고 있으므로 제자리 교체가 필요하면 **임시 파일 → `mv`**로 원자 교체할 것
- **작업 규칙**: ✏️ **저자 지시 2026-09-10: 시간 표기는 전부 PDT.** ✏️ **저자 지시 2026-09-09: 문제가 다시 발생하면 진행하기 전에 Codex(Astra, gpt-5.6-sol)에게 리뷰를 받고 진행한다.** 문제가 생기면 **Codex와 Kimi K3를 함께** 부르고 의견이 갈리면 **실측으로 중재**. 커밋에 Claude 흔적 금지, 저자 Won <wyim@unr.edu> 단일

---

# 🔄 세션 인수인계 (2026-09-14 11:5x PDT 작성) — 여기부터

**한 줄**: B5 시드 123은 epoch 4 완료(best eval 0.6525)·링크 2 대기 중(11 h 링크 체인으로 전환); 456/789는 저자 지시로 연기. **B6(증거 모드)**는 오늘 저자 결정 8건과 Codex 3회·Kimi 1회 검토를 거쳐 워크플랜 v1.2(`c4b9d30`)까지 커밋됐고, **WP0 산출물 2건이 미커밋**이다: A46 v1.36 초안(`PROTOCOL_B1_frozen_v1.md` 951~989행 + 로그 999행, Codex 1차 reject 10건 반영 후 재검토 대기)과 MVP 설계서 `revision/protocols/B6_MVP_PILOT_v1.md`(231행). WP1 착수는 A46 저자 승인·커밋 뒤에만(A46 15항). 상세 이력은 아래 🧊 09-10 블록 안의 ✏️ 09-11~09-14 불릿(60~80행).

**먼저 확인할 것** (⚠️ 이 세션의 WSL 호스트에서는 `dtai` 호스트명이 풀리지 않는다 — pronghorn `login-0`에서 실행하거나 거기서 `ssh -fN dtai` 뒤 사용):
```bash
ssh dtai 'squeue -u $USER -h -o "%i %j %T %M %l %E" | grep -E "3131537|3131538|31493(5[6-9]|6[0-3])"; tail -2 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log; grep "epoch=" /work/nvme/bilv/wcyim/transgenic/runs/seed123/train.err | tail -1'
cd transgenic && git log --oneline -1 && git status --short   # HEAD 9a8c8e1 · M PROTOCOL_B1_frozen_v1.md · ?? B6_MVP_PILOT_v1.md
ls -la revision/protocols/b6_eligible_loci_v1.tsv* revision/protocols/b6_panels_v1.tsv* revision/scripts/7[12]*_*.py 2>/dev/null   # 패널 재실행 산출물(71/72 스크립트); 구 62_ 산출은 폐기 대상
```

| 무엇 | 상태 (09-14 11:5x PDT) |
|---|---|
| **시드 123** (26.08·4 GPU·`repo_seeds` `8f049eb`, A45) | epoch 4 완료 best 0.6525(bad 0); 궤적 0.8707 → 0.7029 → 0.6724 → 0.6525(개선 −0.168/−0.031/−0.020). 링크 1 `3131536` 48 h 경계 PASS(`latest_state` epoch 4 step 36612 global_step 9810, epoch 5 ≈ 74 %). **링크 2 `3131537` PENDING**(TimeLimit 11 h로 축소, Slurm 추정 09-15 09:19 PDT), 링크 3 `3131538`(11 h), 링크 4~11 `3149356`~`3149363`(11 h, CHAIN_MAX=11; `runs/seed123_chain_jobids.txt` 마지막 줄). 링크당 ≤ 200 스텝(≈ 1 h) 재계산 |
| **시드 456/789** | 09-12 저자 지시로 취소·**연기**(생략 아님, 프로토콜 273행). 재제출: `cd $W/repo_dev; SIF=$S8 REPO_DIR=$W/repo_seeds bash $W/repo_seeds/deploy/deltaai/submit_chain.sh 456 3`(789 동일; repo_seeds 커밋은 123과 동일 유지) |
| **B6 커밋됨** | 워크플랜 v1.0 `6c86169` → v1.1 `320de01`(Codex 3차 reject 5) → v1.2 `c4b9d30`(Kimi K3 approve-with-changes) · README en `839dab5` · ko 2종 `b1acb65` · C1' 생략 `460de14` · phase 경로 B `9a8c8e1`(HEAD) |
| **B6 미커밋 (WP0)** | ① **A46 v1.36**(B1 프로토콜 951~989·999행): 저자 결정 6건(대표 모델 = B6 de novo, 옥수수 입력 M-EST+M-HQ18 / 채점 M-FLNC+root-tip ONT, 121 nt arm 채점만, C1' 생략, MVP = Protocol M 표기·50 유전자좌 패널 배제, phase 경로 B) + 규칙 17항; Codex 1차 reject 10건 반영, **재검토 대기** ② **`B6_MVP_PILOT_v1.md`**(mvp-designer): 애기장대 50 유전자좌(핵심 20), C0-lite 5표, K=8 트랙, 0 초기화 투영(인코더 래퍼, 생성 경로는 HF kwargs 전달로 무변경), 250 스텝 × 3 시드 ≈ 18 GPU-h, 채점 = reference-dependent F1(A-ONT1/A-HiFi 정렬본 없음), 검정력 20 유전자좌 ≈ 30 pt·50 ≈ 20 pt — Codex 검토 **approve-with-changes 7건**(11:5x; 코드 인용 전부 실물 일치): STOP을 futility 검정으로(1차 = valid 타일 증거 on/off 짝 L_GSF δ 0.005 nats, 2차 = F1 δ 2 pt), `tile_est`를 valid 타일에도, A-ONT1/A-HiFi 정렬 단계 필수화(5일차까지 없으면 INCONCLUSIVE), 배제 전 분자 중복 제거, donor/acceptor는 `ts:A` 방향, withholding 민감도 1회, SIF 버전 고정 — **설계서에 아직 미반영** ③ **패널 규칙 통일(12:0x)**: `71_b6_panels.py`(적격 held-out 애기장대 유전자좌 전부에 `shuffle_rank`, seed 20260914 → `b6_eligible_loci_v1.tsv`) + `72_mvp_select_loci.py`(+`72a_est_molecules_per_locus.sh`, pronghorn BAM; rank 순 부필터 길이 ≤ 20 kb·A-EST 분자 ≥ 3 통과 첫 50 = MVP, 비MVP 첫 300 = pilot, 나머지 = final → `b6_panels_v1.tsv`); 1차 산출(구 규칙, 적격 2,143, ChrC/ChrM 포함)은 **폐기·핵 염색체만으로 재실행 지시**. 발견: 동결 DB `geneList` 행은 30,720 nt 타일이고 held-out 3,430 중 라벨 타일 보유는 431개(나머지 N-마스킹) — 설계서 §1.2가 FASTA 창을 쓰는 이유 |
| **pronghorn 준비물** | 학습 종 9종 스크리닝 EST BAM/PAF 완료(09-04 배열 `6148523`, 재정렬 불필요) · 검증 종 Zmays/Slycopersicum UniVec 완료(`6163269`·`6163270`, b1_validation_only) · 경로 B diamond DB 완료(`6163304`: Swiss-Prot Viridiplantae 42,106 · OrthoDB v12 12,115,085, `evidence/protein/diamond/`) · **A-ONT1/A-HiFi 정렬본 없음**(`longread_align.sbatch:70-71`이 학습 부적격 런 거부 — quarantine §1l) |

- ✏️ **12:0x PDT — Codex A46 2차: reject 13건**(원문 scratchpad `codex_review_a46_r2_output_raw.txt`; 해소 2·4·5·6·7·8). 남은 것과 조치: ① 동결 문구 — "승인 시 명령·도구 버전(PROVENANCE sha256)·시드·임계값·분모·패널 id/수·N_pilot 고정; 승인 뒤 생기는 산출물(C0·트랙·증거 DB·런 정체성)의 해시는 사용 전에 기록"으로 교체, 'TSV sha256만 미룸' 문장 삭제 ③ WP4 4-rank compile 통합 시험 문장(Dynamo 폴백 실패·0/비0 트랙·캐시 생성 parity·RC·resume) ⑨ **N_pilot = 4,142 update(2 epoch × 2,071)**, 대조군 전부 seed 123, aug_seed 20261037, 최종 스텝 체크포인트 비교 ⑫ **하드 플래그: tile train_weight로는 못 걸러냄(A32는 타일 가중치 1 유지·유전자 단위 마스킹)** → 유전자 단위 A22(GeenuFF)·A30(Swiss-Prot caution) 목록으로 직접 배제, 수 보고 ⑬ 승인 시 실제 풀·final 수 기입(자리표시자 금지) ⑩ rule 17은 MVP 설계서 §5의 futility 문장을 인용 ⑪ MVP 설계서 패널 동기화(진행 중). a46-drafter·panel-builder(하드 플래그 필터 추가·재생성)·mvp-designer에 지시 완료. 다음: 패널 재생성 → pronghorn에서 `72a`+`72` 실행 → 수치 기입 → 최종 Codex 확인 → 승인 요청.

- ✏️ **12:1x PDT — MVP 설계 v1.1 완료**(`B6_MVP_PILOT_v1.md`, 280행, 미커밋): Codex 7건 종결 — futility STOP(1차 = valid 타일 증거 on/off 짝 검증 L_GSF, 한쪽 95 % 상한 < δ 0.005 nats; 2차 F1 δ 2.0 pt; 구조 = 5일차 필수 A-ONT1/A-HiFi 유전자좌 정렬(전체 참조에 §3 명령)의 접합부 P/R; STOP 경로에서 withholding 민감도), GO = 하한 > 0 + 3 시드 양, 그 외 INCONCLUSIVE(프로그램 중단 아님); valid 타일 트랙; 분자 중복 제거; donor/acceptor 전사 방향 + RC 정체성 시험; SIF 버전 기록·캐시 시험 fail-closed; 패널은 71/72/72a 기준(자체 샘플러 없음); 50 유전자좌 전부 디코딩; 예산 ≈ 18(+5.5 STOP 경로) GPU-h. §5.3 문장을 A46 rule 17에 그대로 넣도록 지시. **A46 남은 빈칸: 풀·final 수·TSV sha256(패널 하드 플래그 필터 반영 후 pronghorn에서 72a+72 실행) → 최종 Codex 확인 → 승인 요청.**
- ✏️ **12:15 PDT — pronghorn 72a/72 실행 준비 완료**: 실행 위치 `~/scratch/data/Transgenic/transgenic`(git clone, df9a415; 데이터는 상위 `../evidence/`), samtools = `/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin`(1.3.1, python 3.11.0 동봉; 로그인 노드 PATH에 python3 없음), BAM `../evidence/est_align/Athaliana/Athaliana.est.bam`(+bai) 염색체명 Chr1~5 = 패널 TSV와 일치. `72_mvp_select_loci.py`·`72a_est_molecules_per_locus.sh`는 scp로 복사(미커밋). A46 rule 17 = MVP §5.3 원문 인용 완료(a46-drafter, 960·986·999행). panel-builder 하드 플래그 필터 재생성 대기 중. 시드 123 링크 2 `3131537` PENDING(변동 없음).
- ✏️ **12:3x PDT — 패널 확정**: `71_b6_panels.py`(유전자 단위 A22 4·A30 67 하드 플래그 배제 추가) → 적격 풀 **2,079**(3,430 → −28 ChrC/ChrM −4 A22 −67 A30 −910 단일 exon −5 경계 −337 중첩; 타일 train_weight 탈락 0) sha256 `7ad895fc…`; pronghorn `72a`(BAM, samtools 1.3.1, 30 s) → `b6_est_molecules_v1.tsv` `36de5b68…`(≥3 분자 1,581); `72` → **`b6_panels_v1.tsv` `79367154…`: MVP 50(core 20, rank 62까지 탐색·EST<3 12개 건너뜀) · pilot 300 · final 1,729**. 산출물 로컬로 복사·sha 검증. **72a 계수 규칙을 §1.3 (a)(동일 primary 배치(strand,POS,CIGAR) 병합)로 되돌림**(panel-builder가 read name 기준으로 바꿔 놓았음) + 테스트 기대값·MVP §1.2 문장 동기화(281행, sha `6ab3e0a2…`), 테스트 8건 통과. a46-drafter에 수치·해시 전달(자리표시자 채우기). 다음: A46 완성 → 최종 Codex 확인 → 승인 요청.
- ✏️ **12:5x PDT — Codex A46 3차: REJECT 8건**(`scratchpad/codex_review_a46_r3_output.txt`; 2차 13건 중 9 해소·4 부분; **모든 sha256·행수·패널 수·산술 재계산 일치**). 결함은 전부 문안: ① P984 동결 전제 문장 ② B6/C2 4,142 update 최종 스텝 체크포인트 비교 명시 ③ MVP §1.2 "≥ 2 introns" ↔ 풀은 multi-exon(≥ 1 intron; MVP 50 중 17개가 intron 1) → **풀 재실행 없이 문안을 "≥ 1 intron"으로**, P986 "excluded from rule 12 panels" 정정 ④ 분자 계수 정의 3종 불일치 → **(chromosome, FLAG-0x10 strand, POS, CIGAR)** 로 통일(72 docstring·72a 주석 수정 완료, 테스트 8건 통과, pronghorn 재복사) ⑤ MVP 스키마 14열·실제 명령 ⑥ **Δ_loss 부호 = loss_off − loss_on, STOP 먼저 평가 → GO는 STOP 실패 시만 → 그 외 INCONCLUSIVE**(구간 [0.001,0.004]가 둘 다 만족하던 결함) ⑦ withholding 평가 불가 시 STOP → INCONCLUSIVE ⑧ ONT 명령 `-k14`·조건부 `-uf` 원문. mvp-designer(3~8 → v1.2)·a46-drafter(1·2·3·4) 지시, 이후 rule 17 재동기화 → Codex 4차.

### 다음 세션이 바로 할 일 (순서대로)
0. 규칙(저자 09-09): 새 문제 → Codex(Astra) 리뷰 → 진행; 의견 갈리면 Kimi K3 + 실측 중재.
1. **A46 Codex 재검토 판독** → 남은 지적 반영 → 저자 승인 → 커밋(프로토콜 A46 / MVP 설계서 / 패널 TSV를 **따로** 커밋, `docs:`·`data:`; 메시지에 근거 해시). 그 전에는 WP1 금지.
2. ✏️ 12:2x PDT: **설계서 v1.1 반영 완료(280행, 미커밋; Codex 7건 + 패널 규칙 71/72 + 정지 규칙 원문 리드 전달)** — 남은 것은 A46 rule 17 동기화(a46-drafter)와 71/72 재실행 산출 대조. 원문: **MVP 설계서 수정 → A46 rule 17과 동기화**: (a) Codex 7건(위 표 ②) 반영 — 특히 §5.3 정지 규칙을 futility 검정으로, §2 독립 채점을 "A-ONT1/A-HiFi를 50 유전자좌 ±10 kb 부분 참조에 §3 명령으로 정렬 → `59_evidence_support.py`"로 **필수화**(5일차까지 없으면 INCONCLUSIVE); (b) §1.2를 통일 규칙(위 표 ③)으로 교체 — 패널 id는 `72_mvp_select_loci.py` 산출 `b6_panels_v1.tsv`에서, 자체 샘플러 삭제, 핵 염색체만; (c) A46 rule 12·decision 5·rule 17과 두 스크립트가 **같은 문장**을 인용하는지 대조. ✏️ 12:4x PDT 패널 재실행 완료(pronghorn): **풀 2,079 · MVP 50(핵심 20; rank 62까지 걸어 12개 EST < 3로 제외) · pilot 300 · final 1,729**, `b6_panels_v1.tsv` sha256 `79367154…`; 72a는 (strand, POS, CIGAR) 동일 배치 collapse(§1.3 규칙 (a))로 셈, 설계서 §1.2 문장을 리드가 그에 맞춰 수정(281행, sha256 `6ab3e0a2…`). 승인 시 이 수를 A46에 기입.
3. **시드 123 링크 2 재개 확인**(위 명령): `Resumed from` · step 200 `latest_state saved` · epoch 5 `epoch=` 줄 → `apptainer exec $S8 python revision/scripts/60_epoch_budget.py --log ../runs/seed123/train.err --patience 3`(cwd `repo_seeds`; 로그인 노드 python3는 SyntaxError).
4. A46 승인 뒤 **WP1**: EST×단백질 `diamond blastx` 배열 9종(DB 준비됨) + C0 파이프라인(issue #21, 코드 0줄) — Codex 리뷰 후 실행. **MVP 구현 day 1~2**(C0-lite·트랙 rasteriser, 설계서 §6)는 탐색 arm이라 병행 가능; 4-GPU 스모크(day 4)는 시드 123 큐와 겹치므로 시점 조정.
5. 위키 update 37(시드 123 4 epoch·11 h 링크·B6 결정/워크플랜/A46), `spec.md`.

### 오늘 확정된 것 (09-14; 상세는 아래 ✏️ 09-14 불릿 열 개와 quarantine §1l)
- 저자: EST를 학습에 쓴다 → **B6 = 하나의 모델, 두 모드**(de novo = 트랙 0 / 증거 = 트랙 채움; `E = E_seq + Linear_noBias(K→d)(tracks)`, 다운샘플 전), 대표 모델 **B6 de novo**, B5는 보충자료 비교군. Codex 1차가 "학습에만 증거(보조 헤드)"를 reject → 입력 채널로 전환; Kimi가 early-fusion 채널·blake2b thinning·KV-cache parity·MDE 패널·1주 MVP를 추가. Codex–Kimi 충돌 없음.
- 링크 길이 48 h → **11 h**(저자 "11시간으로"): 대기 잡 TimeLimit은 줄일 수만 있음; `submit_chain.sh`에 APPEND 모드 없음(손으로 8개 제출).
- 학습 종 EST 재정렬은 **불필요**(09-04에 이미 스크리닝 후 정렬; 조사 에이전트 오보 정정).

### 저자 결정 — 완료와 대기
- ✅ B6 진행(de novo + 증거 모드) · ✅ 대표 모델 B6 de novo · ✅ C1' 생략 · ✅ 옥수수 입력/채점 분할 · ✅ 121 nt arm 채점만 · ✅ phase 경로 B · ✅ WP-MVP 착수 · ✅ 11 h 링크
- ⏳ **A46 v1.36 승인**(Codex 재검토 + 패널 재실행 뒤) · ⏳ MVP 정지 규칙 = Codex의 futility 검정(δ_loss 0.005 nats · δ_F1 2 pt) 채택 여부 · ⏳ A-ONT1/A-HiFi 유전자좌 정렬 필수화(리드가 pronghorn에 있는지부터) · ⏳ 456/789 재제출 시점

🧊 **ARCHIVED BELOW** — 아래 09-10 블록의 표·"다음 세션이 바로 할 일"은 **집행됐거나 대체됐다**(인증 4건 PASS, 시드 123 4 GPU 체인 A45, 456/789 연기). 살아 있는 것은 그 블록 끝의 ✏️ 09-11~09-14 불릿뿐이다.

---

# 🧊 🔄 세션 인수인계 (2026-09-10 오전 작성, ~11:00 PDT) — ✏️ 09-14: 표는 낡았고, 블록 끝의 ✏️ 09-11~09-14 불릿이 이후 이력이다

**돌고 있는 것부터 확인할 것.** DeltaAI는 ControlMaster 소켓이 있어야 닿는다(`ssh -O check dtai` → "Master running"; 아니면 사용자가 pronghorn `login-0`에서 `ssh -fN dtai`). 시간은 PDT로 읽는다(DeltaAI 출력은 CDT = PDT+2).

```bash
ssh dtai 'squeue -u $USER -h -o "%i %j %T %M %N"; sacct -j 3124347,3124348 --format=JobID%9,JobName%14,State%11,Elapsed%9 -n -X'
ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo_dev && grep -hE "DDP_INTEGRATION|FAIL: |leg[12] rc|restored [0-9]+ pending|KILL -> " ddp-int-3124347.out ddp-int-3124348.out'
ssh dtai 'tail -4 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log; grep -c "epoch=" /work/nvme/bilv/wcyim/transgenic/runs/seed123/train.err'
# 경계(오늘 21:45 PDT 시그널 / 22:00 PDT 상한) 뒤:
ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo_dev && grep -E "USR1: forwarding|forced preemption|resubmitting chain job|training failed|chain limit" b5-seed-3116844.out; sacct -j 3116844 --format=State%10,ExitCode%8 -n -X; squeue -u $USER -h -o "%i %j %T %E" | grep -E "afterany|b5-seed"'
ssh dtai 'W=/work/nvme/bilv/wcyim/transgenic; git -C $W/repo_seeds log --oneline -1; git -C $W/repo log --oneline -1; md5sum $W/repo/deploy/deltaai/train_b5.slurm | cut -c1-8'   # repo_seeds c646e76 · repo 08f3e2b · 64477d87(=ba24b19)
```

| 무엇 | 상태 (09-10 ~11:00 PDT) |
|---|---|
| **SEED=123** `3116844` · 1 GPU gh015 · 25.06 eager · `/work/repo` 08f3e2b + `train_b5.slurm`(=`ba24b19`, 리허설 통과본) | **epoch 1 완료**: train loss 1.636(ppl 5.13) · **eval loss 0.8617**(ppl 2.37) best. epoch 2 ~~26 %(07:55 PDT, 2.22 it/s) → 종료 ≈ 18:00 PDT~~ ✏️ 11:00 PDT 실측 32 %·2.44 it/s → epoch ≈ 25.5 h → 종료 ≈ ~~**09-11 01:30 PDT**~~ ✏️ 11:39 PDT 실측 42 %·2.03 it/s → **09-11 03:30 PDT**(22:00 PDT 경계는 epoch 2 도중). 링크 1 시작 09-08 22:00:34 PDT → **USR1 21:45 PDT · 상한 22:00 PDT 오늘**. 예산 949/999(07:55) |
| **저자 결정** | ✅ 확인 시드 456/789 = **NGC 26.08 + torch.compile + 4 GPU DDP**(rank당 acc 24, non-reentrant ckpt), `repo_seeds` **`c646e76`**, A44(v1.34) 기록 · ✅ 시간 표기 PDT · ✅ epoch 경계 accumulation 잔여분 수정 지시(반영 `c646e76`) |
| **인증 완료 (26.08·4 GPU)** | `3124099` ddp-int-2608t **PASS**(compile, rank 하나 TERM → 전원 저장·종료·재개·TRAINING_DONE, repo_dev 07486a8) · **`3124229` 체인 리허설 PASS** COMPLETED 0:0 9:54(repo_seeds, SEED=999, 20분, USR1@600: GPU 4개 워밍업·preflight PASS·`runtime identity` 기록·compile·SIGTERM → 4 rank `signal 15 received`·latest_state global_step 7·`exiting cleanly`·chain limit) |
| **인증 ~~대기~~ ✏️ 완료 (c646e76, ~~11:5x·12:1x PDT~~ ✏️ 09:5x·10:0x PDT(sacct 실측 종료 11:54·12:08 CDT; 원문이 CDT를 PDT로 적었다): `3124347` PASS·`3124348` PASS — 아래 ✏️ 불릿)** | `3124347` **ddp-int-epoch**(`KILL_AFTER_EPOCH=1`, `NTRAIN=484`: epoch_01 저장 직후 rank 전원 KILL → leg 2가 epoch 체크포인트에서 재개; PASS 조건 `restored … pending micro-batch` + `latest_state saved at epoch 1 step 23 global_step 6` + `epoch_02/meta.json` `pending_micro: 2`) · `3124348` **ddp-int-2608g**(정상 경로, repo_seeds). 둘 다 PENDING(노드 통째 대기). 취소됨: `3124100`(preflight rc 3, 아래) · `3124112` · `3124228` |
| **4 GPU 벤치(26.08 compile, 09-09)** | 5.89 / 11.04 / 23.23 s/step(30/61/129 kb; 1 GPU 대비 3.62/3.67/3.73×, 효율 91~93 %) → **epoch 6.15 h 벽시계 = 24.6 GPU-h**, 시드 2.0~2.6일·197~246 GPU-h, 두 시드 394~492 GPU-h(잔여로 충분; 안전 여유 ~300 GPU-h 요청 권고 유지) |
| 리허설 잔재 | `runs/seed999` 삭제 완료(09-10 10:5x PDT)**(`rm -rf /work/nvme/bilv/wcyim/transgenic/runs/seed999`); seed997/998은 삭제됨 |

- ✏️ **~~11:5x PDT~~ ✏️ 09:5x PDT(sacct 종료 11:54 CDT) — `3124347` ddp-int-epoch PASS** (c646e76, 26.08 compile, 4 GPU, NTRAIN=484): epoch 1 끝에 rank당 1 micro-batch 잔여·gradient 609 텐서 저장 → epoch_01 저장 직후 4 rank KILL → leg 2가 epoch_01에서 재개, `restored 1 pending micro-batch(es)`, **epoch 2 첫 update가 micro-batch 23**(카운터 유실이면 24), epoch_02 `pending_micro`=2, TRAINING_DONE. accumulation 잔여분 보존 수정이 끝까지 실증됨. 남은 인증은 `3124348`(정상 통합, 큐 대기) 하나.

- ✏️ **~~12:1x PDT~~ ✏️ 10:0x PDT(sacct 종료 12:08 CDT) — `3124348` ddp-int-2608g PASS** (c646e76, 26.08 compile, 4 GPU, gh121 첫 launch 성공, rank 하나 TERM → 4 rank 저장·종료, 재개, TRAINING_DONE). **인증 3건(체인 `3124229`·epoch 재개 `3124347`·통합 `3124348`) 전부 PASS → 코드 쪽 제출 준비 완료.** 남은 관문은 시드 123 경계(21:45/22:00 PDT)와 저자 "시작해".

- ✏️ **10:4x PDT — 저자 결정: 선택지 2 (456 먼저, 789는 시드 123 곡선을 본 뒤), 자동 진행.** dtai 로그인 노드에 `runs/auto_submit_456.sh`가 detached로 돈다(pid 3142608, lock `runs/auto_submit_456.lock`, 로그 `runs/auto_submit_456.log`). 동작: 5분마다 `sacct -j 3116844` → 링크 1이 끝나면(상한 22:00 PDT) 2분 뒤 **관문 검사**(`b5-seed-3116844.out`에 `USR1: forwarding as SIGTERM`·`resubmitting chain job 2`, train.err에 `signal 15 received`, TRAINING_FAILED 없음, 큐에 시드 123 링크 2, 재개 가능한 체크포인트) → 통과 시 `sbatch … SEED=456 SIF=26.08 REPO_DIR=repo_seeds`, `runs/seed456_link1_jobid.txt` 기록, 이후 30분마다 `runs/seed456_monitor.log`. **관문 실패 시 제출하지 않고 `GATE FAILED`만 남긴다** → 사람이 시드 123 경계를 먼저 본다. 시드 789는 저자 결정 대기(프로토콜 273행: 미룰 수 있으나 생략 불가). 확인: `ssh dtai 'cat /work/nvme/bilv/wcyim/transgenic/runs/auto_submit_456.log; cat /work/nvme/bilv/wcyim/transgenic/runs/seed456_link1_jobid.txt'`.

- ✏️ **11:0x PDT — 22 epoch 상한의 근거 기록(`df9a415`, 프로토콜만 변경, repo_seeds는 c646e76 유지)**: 저자 설명 — 22는 공개 체크포인트를 만든 원래 학습(조기 종료 없음)의 길이이고, B5의 실제 epoch 수는 "검증 손실 최소·patience 3"이 만든다. A44에 'Epoch ceiling' 문단 추가. HEAD는 `df9a415`(코드는 c646e76과 동일).

- ✏️ **11:4x PDT — /clear 뒤 재개 실측**: 위 표와 일치(시드 123 RUNNING·epoch 2 42 %·2.03 it/s·예산 943/999; `3124347`·`3124348` COMPLETED+PASS; `auto_submit_456.sh` pid 3142608 가동, 로그 1줄, 456 미제출; repo_seeds c646e76·repo 08f3e2b·md5 64477d87; seed999 삭제됨). 이 WSL 머신에는 `dtai` 호스트가 없다 — **`ssh pronghorn 'ssh dtai …'`로 경유**한다(pronghorn의 ControlMaster가 살아 있어야 함). 세션 안 모니터가 10분마다 경계·자동 제출을 감시한다(세션이 끝나면 모니터도 끝난다; 자동 제출 스크립트는 dtai에서 독립적으로 돈다).

- ⚠️ **19:5x PDT — `auto_submit_456.sh`가 ~14:55 PDT에 죽어 있었다(dtai 로그인 노드 재부팅; gh-login03 부팅 14:27 PDT, ControlMaster 소켓도 15:05 PDT 소멸).** 로그는 10:41 시작 줄 하나뿐, dtai 쪽 `seed123_monitor.log`도 14:55 PDT에서 멈춤, 잠금 디렉터리만 잔존. gh-login01~04 전부 사본 없음·시드 456 미제출(sacct) 확인. 패치본(`runs/auto_submit_456.sh.new`, md5 0ea38238: 잠금에 host+pid 기록·EXIT trap rm -rf·대기 중 매시 heartbeat 로그·sbatch 직전 jobid 파일 재확인)을 Codex 리뷰 뒤 22:00 PDT 전에 재기동. **교훈: 로그인 노드의 detached 헬퍼는 잠금 디렉터리가 아니라 `pgrep` + 로그 mtime으로 생사를 확인한다.** 세션 cron 22:12·22:33 PDT가 경계와 제출을 이중 확인한다.
- ✏️ **19:57 PDT — 재기동 완료(Codex approve)**: gh-login03에서 `bash ./auto_submit_456.sh` **pid 1337021**, 잠금 owner 기록, 로그에 19:57:35 시작 줄, md5 `483c9201`(추가: `--comment=seed456` + sbatch 직전 큐 코멘트 재확인 — Codex 지적 "sbatch 성공 후 jobid 파일 기록 전 사망" 창 대비). 원본은 `auto_submit_456.sh.orig_20260910`. 생사 확인: `pgrep -u $USER -f 'bash ./[a]uto_submit_456.sh'`(자기 명령줄 매칭 회피) + `cat runs/auto_submit_456.lock/owner` + 로그의 매시 heartbeat. dtai 쪽 `seed123_monitor` 루프는 재기동하지 않음(세션 모니터가 대신).
- ⚠️ **21:45~21:50 PDT — 시드 123 경계: 링크 1은 옛 스크립트 텍스트로 돌았다(Slurm은 sbatch 시점에 스크립트를 복사; 09-09의 ba24b19 교체는 링크 2부터).** `.out`: `USR1: forwarding to trainer` → `resubmitting chain job 2` → `Submitted batch job 3128376`; sacct COMPLETED 0:0 21:45:14 PDT. 옛 결함(트랩 뒤 `wait` 즉시 반환)으로 **트레이너는 경계 저장 없이 잡과 함께 죽음** — train.err에 `signal … received`·경계 `latest_state saved` 없음. 마지막 latest_state 19:51 PDT(epoch 1 step 146772 global_step 3600, best 0.8617 bad_epochs 0), 종료 시 81 %(161,300) → **≈14,500행 ≈1.9 h 손실, ≤200 step 허용 경로 안**. TRAINING_FAILED·반쯤 쓰인 파일 없음. 링크 2 `3128376` PENDING(Priority), `scontrol write batch_script`로 캡처한 텍스트 md5 64477d87 = ba24b19, CHAIN_N=2 SEED=123 GPUS=1. **자동 관문은 21:49:35 GATE FAILED**(no-signal-forwarding·trainer-did-not-see-TERM — 새 문구를 요구했는데 링크 1은 그 문구를 찍을 수 없는 스크립트였다; 11:39의 '관문 문자열 검증'은 디스크 파일과 대조한 것이라 이를 놓쳤다) → 456 미제출, 스크립트 종료. **Codex 판정: 링크 2가 `Resumed from …` 로그를 찍은 뒤 456 제출**(재개 경로 실증이 관문의 본뜻; 1 GPU Priority 잡이라 대기 짧음). 세션 모니터 `link2_watch.sh`가 3분마다 감시(성공 `Resumed from`, 위험 `starting fresh`, 실패 시그니처 기준선 2건 초과). 제출 전 점검 5: 456 중복 없음(`squeue %k`=seed456)·seed123 디렉터리 정상·링크 2 md5·로그 이상 없음·sbatch 인자(26.08 SIF·REPO_DIR repo_seeds·exclude gh062,gh093·comment seed456). **교훈: 큐/실행 중 잡이 쓰는 스크립트는 `scontrol write batch_script <id>`로 확인한다; 위키·런 로그에 1.9 h 손실을 편차로 기록.**
- ✏️ **09-11 09:52 PDT — 시드 456 링크 1 수동 제출: 잡 `3131228`** (저자 결정 "지금 제출" — 22:33 PDT에 Slurm이 링크 2 시작을 09-12 02:14 PDT로 추정해 Codex의 '재개 확인 뒤 제출' 전제가 깨졌고, 저자에게 물어 결정; 응답은 09-11 아침). 명령: repo_dev에서 `sbatch --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 --comment=seed456 --export=ALL,SEED=456,SIF=<26.08>,REPO_DIR=$W/repo_seeds $W/repo_seeds/deploy/deltaai/train_b5.slurm`. 확인: NumCPUs=64·gres/gpu=4·ExcNodeList gh[062,093]·Comment=seed456·캡처 스크립트 md5 c326c775 = c646e76. `runs/seed456_link1_jobid.txt` = 3131228, `auto_submit_456.log`에 BY HAND 줄. 시작 추정: 456 N/A(노드 통째), 시드 123 링크 2 `3128376` **09-11 18:48 PDT**(변동). 세션 모니터 `seed456_watch.sh`(5분)·`link2_watch.sh`(3분) 가동 — 456 시작 직후 확인 항목은 위 3번 불릿 그대로(`repo: … @ c646e76`·`image: …26.08`·`runtime identity`·`PREFLIGHT PASS`·`gpu warm-up ok` ×4·`torch.compile scope` ×4·`DDP world size 4: per-rank accumulation 24`·step 200 `latest_state saved`). **시드 789는 저자 결정 대기**(같은 명령을 `SEED=789 --comment=seed789`로; 프로토콜 273행). ⚠️ 밤사이 세션이 질문 대기 상태라 매시 보고가 멈췄다 — 야간 결정은 차단 질문 대신 '기본 행동 + 기한'으로 적을 것.
- ✏️ **09-11 09:56 PDT — 시드 789 링크 1 수동 제출: 잡 `3131255`**(저자 "789 해"). 456과 같은 명령, `SEED=789 --comment=seed789`; NumCPUs=64·gres/gpu=4·ExcNodeList gh[062,093]·캡처 스크립트 md5 c326c775 = c646e76. `runs/seed789_link1_jobid.txt` = 3131255, `auto_submit_456.log`에 BY HAND 줄. 큐(09:56 PDT): 시드 123 링크 2 `3128376` 추정 18:48 PDT · 456 `3131228` 추정 09-12 19:40 PDT · 789 `3131255` N/A. **세 시드 전부 큐에 있음 — 남은 저자 결정은 없다**(c·d·e·pending_grads rank-0은 선택 사항). 세션 모니터 `seed789_watch.sh` 추가.
- ✏️ **09-11 10:0x~10:4x PDT — 저자: "우리 그렇게 시간이 없어" → "예산을 올려두었어 codex가 말한거 고쳐줘" → A45(v1.35) 구현·커밋 `8f049eb`·푸시.** 예산 잔액 1,433/1,499 GPU-h. Codex 계획 리뷰(approve-with-changes: 4 GPU 재시작 승인, 미리 큐잉은 5건 수정 조건) → 구현 → Codex 변경 리뷰(approve-with-changes: 잠금 fail-closed·SEED 검증·명시적 exit 0·A45 문구 2건) → 전부 반영, 테스트 34건 통과. **취소: `3128376`(시드 123 링크 2, 1 GPU)·`3131228`(456)·`3131255`(789). `runs/seed123` → `runs/seed123_1gpu_partial`(보존, 미보고).** dtai `repo_seeds` = **`8f049eb`**(train_b5.slurm md5 a6a16993). **리허설 체인 seed 999: `3131416` → `3131417` → `3131418`**(15분·USR1@600·4 GPU·26.08; `runs/rehearsal_a45.sh`, `SBATCH_TIMELIMIT`/`SBATCH_SIGNAL` 환경변수로 #SBATCH 덮어씀 — TimeLimit 00:15:00 적용 확인; 링크 1은 10:41 PDT gh140에서 즉시 시작). **리허설 절차**: 링크 1 SIGTERM→저장→exit 0 · 링크 2 `Resumed from` → **링크 2 RUNNING 중에 `touch runs/seed999/TRAINING_DONE`**(사람이) → 링크 2 종료 시 `cancelling pre-queued link 3131418` · 그 뒤 4번째 링크 1개 제출 → `training already finished … nothing to do` exit 0. 네 결과 전부 확인돼야 시드 제출(go/no-go, A45 인증 문단). **통과 후**: `cd repo_dev; SIF=…26.08.sif REPO_DIR=$W/repo_seeds bash $W/repo_seeds/deploy/deltaai/submit_chain.sh 123 3`(456·789 동일) → `runs/seed<N>_chain_jobids.txt` → 모니터·매시 보고 갱신 → 위키 update 37.
- ✏️ **09-11 11:11 PDT — 리허설 4/4 PASS → 세 시드 제출 완료(A45 미리 큐잉 체인, 9잡).** 리허설: `3131416` COMPLETED 0:0(USR1 +5분 → 4 rank SIGTERM → latest_state → exiting cleanly) · `3131417` 9분 뒤 시작·`Resumed from`·손으로 만든 TRAINING_DONE → `training finished` → **`cancelling pre-queued link 3131418`**(CANCELLED) · 4번째 링크 `3131519`(1 GPU) → `training already finished … nothing to do` COMPLETED 0:0 8 s. **제출**(repo_dev cwd, `submit_chain.sh <seed> 3`, repo_seeds 8f049eb, 26.08, 4 GPU×64 CPU, exclude gh062,gh093, comment seed<N>): **123 = `3131536`→`3131537`→`3131538` · 456 = `3131539`→`3131540`→`3131541` · 789 = `3131542`→`3131543`→`3131544`**. 기록: `runs/seed<N>_chain_jobids.txt`. 리허설 디렉터리 seed999 삭제. 시작 직후 확인 항목: `.out`의 `repo: … @ 8f049eb`·`image: …26.08`·`runtime identity: … gpus=4 cpus=64 torch=2.14…`·`PREFLIGHT PASS`·`gpu warm-up ok` ×4, train.err `torch.compile scope: model` ×4·`DDP world size 4: per-rank accumulation 24`·step 200 `latest_state saved`, `runs/seed<N>/LINK_ACTIVE.<jobid>`. 링크 경계마다: `.out`에 `USR1: forwarding as SIGTERM`(4 pid) → train.err `latest_state saved`·`exiting cleanly` → `link k ended at its limit; the pre-queued next link continues` → sacct COMPLETED 0:0 → 다음 링크 `Resumed from`. 세션 모니터 `chain_watch.sh <seed>`(5분) ×3, 매시 :37 보고(`check_now.sh`가 jobids 파일을 읽음). **남은 일**: epoch마다 `60_epoch_budget.py --patience 3`, 위키 update 37(A45·리허설·제출), 결과표에 1 GPU 부분 런은 미보고(A45 2항).
- ✏️ **09-11 14:0x PDT — 저자: 22 epoch 상한 "A 유지"(변경 없음).** 근거: `max_epochs`는 LR 스케줄 지평(`train_HyenaTransgenic.py:342-354`, 5 % warmup + 0까지 선형 감쇠)이라 값을 늘리면 레시피가 바뀌고, 레시피 파일은 A40 동결 JSON에 sha256 고정. 22에서 LR이 0이 되므로 patience 3이 먼저 멈춘다. 저자 질문 "3 시드 꼭 필요한가" → 프로토콜 273행(456/789 투고 전 완료, n=3 평균±s.d.); 줄이려면 A46. 권고: 큐에 둔 채 시드 123 초반 epoch 보고 결정(대기 잡은 예산 소모 없음, 취소는 큐 위치 상실). 큐 상태 13:5x PDT: ghx4 idle 0(allocated 52·mixed 99·예약 33노드), 우리 잡 Priority 대기 순위 ≈ 506/1,648(점수 1,367). 기대 결과 정리(사전 등록 정확도 목표 없음; 학습 곡선·GSF 위반·held-out/검증 arm 일치·항등 기준선 초과가 판독 기준)를 저자에게 전달.
- ⚠️ **09-12 06:49~08:52 PDT — 시드 123 링크 1 `3131536` 시작(gh129), 시작 마커 전부 PASS, latest_state 200·400 저장 확인. 실측 속도 rank당 1.4 it/s(2시간 안정) → epoch ≈ 10 h(+검증), 40 GPU-h/epoch — 벤치 기반 기대 2.25 it/s·6.15 h보다 1.6× 느림.** 자원·SIF·acc 24·워커 8은 벤치 3118904와 동일; 차이는 워크로드(단일 티어 30스텝 vs 3티어 혼합 + 실제 라벨 길이 → compile 동적 shape 추정, 미실증). GPU 활용 100 %, 재컴파일 경고 0. 시사점: 시드당 8~10 epoch ≈ 80~100 h(링크 2~3개, 3링크 체인으로 충분), 3 시드 × 10 epoch ≈ 1,200 GPU-h(잔액 1,433; 12 epoch씩이면 초과). 456/789 시작 추정 09-13 07:52 PDT. `LR schedule 182,270 total steps` 표기는 명목값(accelerate가 optimizer 스텝마다 스케줄러를 4번 밟음; 실효 45,567 = 1 GPU와 동일, 버그 아님).
- ✏️ **09-12 11:53 PDT — 저자 지시: 시드 456·789는 나중에 돌리기로 하고 취소, 123만 계속.** `scancel 3131539 3131540 3131541 3131542 3131543 3131544` → 전부 CANCELLED(시작 전이라 예산 소모 0). 시드 123 체인은 손대지 않음(`3131536` RUNNING 5 h, `3131537`·`3131538` PENDING). 기록: `runs/seed456_chain_jobids.cancelled_20260912.txt`·`seed789_…cancelled_20260912.txt`(원본 이름에서 변경), `auto_submit_456.log`에 deferral 줄. 프로토콜 273행상 456/789는 **미룰 수 있으나 생략 불가** — 투고 전에 돌려야 하며, 생략하려면 A46 필요. **나중에 다시 돌릴 때**: `cd $W/repo_dev; SIF=$S8 REPO_DIR=$W/repo_seeds bash $W/repo_seeds/deploy/deltaai/submit_chain.sh 456 3`(789 동일; repo_seeds는 시드 123과 같은 커밋 8f049eb를 유지할 것 — 런타임 정체성 동일 조건). 세션 모니터 456/789 중지, 매시 보고는 123만(jobids 파일 기준). 예산 전망: 시드 123 단독 ≈ 10 epoch × 40 GPU-h ≈ 400 GPU-h(잔액 1,433).
- ✏️ **09-12 16:56 PDT — 시드 123 epoch 1 완료**: `epoch=0: train_ppl 5.2781, train_epoch_loss 1.6636, eval_ppl 2.3886, eval_epoch_loss 0.8707`; best=epoch 1, bad_epochs 0; `epoch_01/`(pending_grads 포함) 저장; epoch 벽시계 **10 h 07 m**(검증 ≈ 13 min 포함) → 링크 1(48 h)이 ≈ 4.7 epoch, 09-14 06:49 PDT경 epoch 5 도중 경계. 1 GPU 부분 런 참고값 0.8617(+0.009; 기술적 비교만, A45 2항). ⚠️ `60_epoch_budget.py`는 dtai 로그인 노드 python3에서 SyntaxError(`from __future__ import annotations`, Python < 3.7) — **컨테이너 안에서 실행**: `apptainer exec $S8 python revision/scripts/60_epoch_budget.py --log ../runs/seed123/train.err --patience 3`(cwd repo_seeds; epoch 2 이후 판정 가능).
- ✏️ **09-13 02:56 PDT — 시드 123 epoch 2**: `epoch=1: train_ppl 2.0482, train_epoch_loss 0.7170, eval_ppl 2.0197, eval_epoch_loss 0.7029`(epoch 1 0.8707 → **−0.168**), best=epoch 2, bad_epochs 0; epoch 2 벽시계 ≈ 10 h 00 m. 학습 손실이 epoch 평균(A44 수정)이라 검증 손실과 근접. 검증 손실 궤적: 0.8707 → 0.7029.
- ✏️ **09-13 12:59 PDT — 시드 123 epoch 3**: `epoch=2: train_ppl 1.9275, train_epoch_loss 0.6562, eval_ppl 1.9589, eval_epoch_loss 0.6724`(epoch 2 0.7029 → **−0.031**), best=epoch 3, bad_epochs 0; epoch 3 벽시계 ≈ 10 h 03 m. 궤적 0.8707 → 0.7029 → 0.6724. 링크 1 경계 09-14 06:49 PDT(epoch 5 도중).
- ✏️ **09-13 23:13 PDT — 시드 123 epoch 4**: `epoch=3: train_ppl 1.8822, train_epoch_loss 0.6324, eval_ppl 1.9204, eval_epoch_loss 0.6525`(epoch 3 0.6724 → **−0.020**), best=epoch 4, bad_epochs 0; epoch 4 벽시계 ≈ 10 h 14 m. 궤적 0.8707 → 0.7029 → 0.6724 → 0.6525(개선 폭 −0.168 → −0.031 → −0.020). **링크 1 경계 09-14 06:49 PDT(epoch 5 ≈ 70 %)** → `USR1: forwarding as SIGTERM`·`latest_state saved`·`exiting cleanly`·`link 1 ended at its limit` 확인 → 링크 2 `3131537` 큐 대기(노드 통째) → `Resumed from` 확인.
- ✏️ **09-14 06:41 PDT — 링크 1 경계 PASS**: 06:34 USR1 → 4 rank SIGTERM → `latest_state saved at epoch 4 step 36612 global_step 9810`(epoch 5 ≈ 74 %) → `exiting cleanly` → `link 1 ended at its limit` → `3131536` COMPLETED 0:0(1-23:47:31), 잠금 마커 제거. meta.json best 0.6525(epoch 4), bad_epochs 0. **링크 2 `3131537` PENDING, Slurm 추정 시작 09-15 09:19 PDT(≈ 27 h 대기)** → 재개 시 ≤ 200 스텝(≈ 1 h) 재계산. 12 h 링크 쪼개기 제안은 저자 결정 대기(대기 잡 TimeLimit은 사용자가 줄일 수만 있고 늘릴 수 없음).
- ✏️ **09-14 07:04 PDT — 저자 "11시간으로 돌려줘" → 11 h 링크 전환 완료.** `scontrol update TimeLimit=11:00:00` for `3131537`·`3131538`(48 h → 11 h, 큐 위치 유지); 11 h 링크 8개 추가 제출(`--time=11:00:00 --dependency=afterany:<prev> --comment=seed123 --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 --export=ALL,SEED=123,GPUS=4,CHAIN_N=k,CHAIN_MAX=11,CHAIN_PRESUBMITTED=1,LAUNCH_RETRY=0`, SIF 26.08, REPO_DIR repo_seeds): **링크 4~11 = `3149356`~`3149363`**. 체인 총 158 h. `runs/seed123_chain_jobids.txt`에 11개 전체 id 줄 추가(모니터·매시 보고는 마지막 줄을 읽음). 전환당 ≤ 200 스텝(≈ 1 h) 재계산; 남는 링크는 완료 시 취소/가드 종료. 직후 추정 시작은 여전히 09-15 09:19 PDT(우선순위 기준; backfill은 발생해야 반영). `submit_chain.sh`에는 APPEND 모드가 없어 손으로 붙임 — 추후 추가 검토.
- ⏳ **09-14 07:1x PDT — 저자: "EST를 학습에 써야 돼(다른 모델과 비교하려면)" → B6 arm 제안, 결정 대기.** 현행: A9가 "EST 가중 B5 학습(J)" 배제(split-only 비교 가능성), A14는 학습 종 EST·ONT·Sequel II+를 학습 측 증거로 허용(A13:586의 '검증 전용'은 A14가 대체). 제안: **B6 = B5 코퍼스 + 증거 지지 isoform(완전 체인, 클론/분자 ≥ 2, 학습 종의 학습 가능 EST·장독만)을 종별 GFF3에 추가 전사체로 주입 → 타일 빌드·동결(A40 파이프라인) → 4 GPU 런**; 시드 123(B5)은 그대로 비교군; A46 개정(애기장대 A-EST는 검증 전용 유지, 본문 모델 사전 지정). **준비 상태**: 11종 EST 원본 있음(`evidence/est/<종>/est.fa.gz`), UniVec 스크립트 있음(실행 미확인), 09-03 EST 정렬(배열 6147674)은 스크리닝 전이라 증거 불가 → 재정렬 필요; 장독 학습 가능 카탈로그 `TRAINING_EVIDENCE_v1.md:21-47`(B. distachyon 없음, S. bicolor ONT 없음, G. max PacBio 불가), 정렬 완료(216행). **C0 증거 표(9개)는 코드 0줄(issue #21)** — B1 검증에도 필요. 라벨 주입점: 종별 GFF3 → `Gene.transcripts`(`gsf_contract.py:184-274`, 유전자당 ≤ 15 전사체) → `geneList`/`window_genes`. 재사용 스펙: `PROTOCOL_M_exploratory_v1.md:20`(EST splice-graph 후보 생성기, ≤ 32 경로, 말단은 poly(A)/5' 증거만, ORF ≥ 90 aa), `:42`(증거 primary 구성 규칙: accession ≥ 2·라이브러리 ≥ 2·UTR 증거 없으면 CDS-only), A23 순위. **계획**: ① UniVec + 9종 EST 재정렬(1일) ② C0 구현 + Codex 리뷰(3~4일) ③ isoform 추출·GFF3 주입·빌드·동결(1~2일) ④ A46 ⑤ B6 런(≈ 4일 + 큐) ≈ 2주. 저자 "시작해" 대기.
- ✏️ **09-14 08:1x PDT — 저자 "univec 사용해서 clean 하고 재정렬해" → 학습 종 9종은 이미 완료(조사 에이전트 오보 정정).** pronghorn 실측: `evidence/est/<SP>/univec/DONE` 9종, 두 arm FASTA(≥100 primary·≥121 sensitivity) 존재, `evidence/est_align/<SP>/`·`est_align_min121/<SP>/`의 DONE이 스크리닝 FASTA md5와 일치, BAM quickcheck ok, PAF 있음 — 09-04 배열 `6148523_0-17`(게이트 3종 포함 `est_align.sbatch`, md5 e0641d9b). 매핑률 92~99 %(Athaliana 93.15 %, Sitalica 92.32 % 최저). 09-03 스크리닝 전 BAM은 `RETIRED_DO_NOT_USE/est_align_prescreen_20260903`. **검증 종 Zmays·Slycopersicum은 스크리닝이 없어 `univec_screen.sh`(MIN_LEN=100) sbatch 2건 제출**(cpu-s1-pgl-0, 로그 `evidence/est/univec_val_logs/`); 이들의 정렬은 B1 §3 검증 패스용 별도 드라이버(est_align.sbatch는 역할 게이트로 거부). B6 계획 ①단계는 사실상 완료 → 다음은 ② C0 파이프라인.
- ❌ **09-14 08:4x PDT — Codex 설계 리뷰: B6(per-base 보조 헤드) reject-as-designed, 13건**(원문 scratchpad `codex_review_output.txt`). CRITICAL: ① 추론 시 증거가 사라져 '증거 사용 주석기'가 아니라 표현 정규화에 그침 → 추론 시 증거 입력 채널 또는 독립 C2 모델이 증거 가치를 분리함 ② 14 클래스는 중첩 multilabel·배경/phase 클래스 없음(`gsf_contract.py:48`) → 14-way CE·positive-only BCE 모두 부적합 ④ A29 마스킹은 유전자 포함 기준이라 타일 가장자리 걸친 유전자(`partial`, `build_b5.py:561`)로 held-out 서열·EST 라벨이 샘 ⑤ paralog·반대 가닥 겹침·공유 접합부 좌표·플랭크 밖 정렬·다중 유전자좌 EST는 raster 전 **관측(분자) 단위 배제** 필요 ⑩ `segmentation_loss`는 최상위 출력 dataclass에 없음(제안서 오류), DDP는 이미 `find_unused_parameters=True`(`train:216`). HIGH: ③ B6 vs B5는 구조·RNG·최적화 혼입 → 같은 시드 λ=0 대조군 + 증거 셔플 대조군 필요 ⑥ 플랭크는 50~150 nt 시드값(A33:471), RC는 donor/acceptor·5'/3' 스왑 ⑦ `--secondary=no`는 분자 유일성 보장 아님(accession 버전·클론 mate·UniVec 분할·supplementary) ⑧ **애기장대 EST 이중 사용은 '유전자좌 분리 in-domain 검증'이지 독립 재현이 아님** — 독립은 검증 전용 장독뿐 ⑨ C0-lite로는 B1 재사용 불가 → 전체 C0 스키마 ⑪ 기울기 간섭·클리핑·Adam, 체크포인트 정책 ⑫ compile: 분기 없는 항상 실행 헤드, blob 7.2 MB/타일, 좌측 패딩(`datasets.py:707`) ⑬ A46 동결 항목 대폭 확대. **저자 결정 필요(갈림길)**: (A) 학습에만 증거(정규화; 손실·누출·대조군 수정) vs (B) 학습+추론 모두 증거를 **입력 채널**로(BRAKER/MAKER식 힌트, 채널 드롭아웃으로 de novo 유지) — '증거 쓰는 도구와 비교' 목적엔 B가 맞음.
- ✏️ **09-14 08:5x PDT — 검증 종 EST UniVec 스크리닝 완료**(pronghorn `6163269` Zmays 17분, `6163270` Slycopersicum 4분, MIN_LEN=100): Zmays 2,019,959 → 1,840,603 레코드(drop 179,445·trim 17,887·split 327), Slycopersicum 301,030 → 298,437(drop 3,208·trim 4,524·split 2,091). 산출: `evidence/est/<SP>/univec/`(DONE·PROVENANCE). ≥121 arm은 미생성(`retrim_a37.sh Zmays Slycopersicum`로 가능). 역할 b1_validation_only — B1 §3 검증 정렬용, 학습 금지.
- ⚠️ **09-14 09:58 PDT — Codex 2차(B6 v2: 증거 입력 채널 + 드롭아웃): "현 명세는 reject, 방향은 교정된 pilot 가치 있음"**(원문 scratchpad `codex_review_v2_output.txt`). 1차 13건 중 해소 6(추론 시 증거·보조 손실 제거·A-EST in-domain 표기·전체 C0·DDP·단일 체크포인트), 부분 7(③ C2 대조군은 **전체 길이**·초기화/데이터 순서 일치·증강 RNG 분리·증거 셔플/presence·증거 전용 재구성 기준선 ④ 타일 가장자리 held-out DNA는 빌더(`build_b5.py:567`)가 여전히 노출 → DNA 마스크도 고치고 서열 전용 비교군 재빌드 ⑤ valid-orthogroup 배제·A33 decoy/component 마스크 추가 ⑥/신규 RC: donor/acceptor 정체성은 RC에서 유지, strand만 뒤집힘(`tests/test_gsf_rc.py:26`) — 1차의 스왑 권고가 틀렸음 ⑦ 기존 정렬은 secondary 배치를 억제(`est_align.sbatch:144`) → 배치 복원 또는 감사 불가 분자 배제 ⑫ 좌측 패딩(`datasets.py:728`)·beam 생성·L×d 활성 메모리·resume 화이트리스트(`b5_runtime.py:224`)·RC 재raster·결정성 ⑬ A46 동결 항목 전부). **신규 CRITICAL: 검증 유전자좌에서 추론 입력으로 넣는 증거와 채점에 쓰는 증거를 분리해야 함**(Protocol M:14). 신규 HIGH: dose-response는 커버리지 층이 아니라 **같은 유전자좌에서 반복 nested thinning**; raster는 관측→타일 소속을 유지해 재raster 가능해야 함. 권고: 인코더 채널(HyenaDNA 뒤·다운샘플 전 bias 없는 투영, `modeling:725`), 드롭아웃 0.5 고정, 분자 보존 비율 {0.01,0.1,0.5,1} 샘플링 + 소스/라이브러리 withholding 대조, 특징 = 소스×strand 커버리지·접합부·unknown-strand·availability 마스크(단백질 phase 보류, log1p 클립, 참조 유래 특징 금지), 정지 규칙 de novo 검증 GSF 손실 patience 3·단일 체크포인트, 게이트 = de novo transcript-F1 하한 ≥ −1 pt(vs 전체 C2)·증거 이득 하한 > 0(vs off·증거 전용 기준선), locus 짝 bootstrap, 누출/RC/메모리 실패 시 중단. M(c) 증거 프롬프트는 싼 비교군. **다음: v3 = 교정 pilot 명세(A46 초안 + 게이트) → Codex 3차 → C0 구현 → pilot.**
- ✏️ **09-14 10:5x PDT — 저자: "codex 의견 반영하고 워크플랜 작성, de novo 모드 + 증거 모드로 진행" → `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md` 작성·커밋 `6c86169`·푸시.** 요지: A46 FREEZE 항목 14개(소스/역할·A9/A18 개정, 전체 C0, 분자 단위, 관측 단위 누출 배제 + 감사 정렬 `--secondary=yes -N 5`(배제용만), 가장자리 held-out DNA 마스크 수리 + 서열 전용 비교 코퍼스 재빌드, 트랙 명세(소스×strand 커버리지·접합부·unknown-strand·단백질·availability; 참조 유래 특징 금지; RC는 donor/acceptor 유지), bias 없는 투영(다운샘플 전), 드롭아웃 0.5 + thinning {0.01,0.1,0.5,1} 별도 RNG, 정지 규칙 de novo 검증 손실 patience 3, runtime identity 확장, **검증 유전자좌의 입력 증거/채점 증거 분리**(DATASET_ROLES `b6_input`/`b6_score`), 대조군 C1(시드 123)·C1'(수리 코퍼스 서열 전용)·C2(트랙 0, 전체 길이)·C3(셔플)·C4(presence)·C5(증거 전용 기준선), 고정 유전자좌 nested thinning dose-response, 게이트 G1(de novo F1 하한 ≥ −1 pt vs C2)·G2(증거 이득 > 0 vs off·C5)·G3(GSF 위반 ≤ B5), 중단 규칙). WP0 A46 초안 → WP1 C0(4~5일) → WP2 배제·마스크 수리·코퍼스 재빌드(3일) → WP3 트랙/thinning(2일) → WP4 모델/트레이너 + 4-rank 시험(2일) → WP5 pilot(≈ 75 GPU-h) → WP6 본 런(B6+C2+C1' ≈ 1,200 GPU-h → **예산 ≈ 600 GPU-h 보충 필요 또는 C1' 생략**) → WP7 평가. 총 5~6주. **저자 결정 대기 4**: 예산 보충 vs C1' 생략 · 본문 모델 · 옥수수 입력/채점 라이브러리 분할(제안: 입력 M-EST+M-HQ18, 채점 M-FLNC+root-tip ONT) · 121 nt arm 범위. Codex 3차(워크플랜) 진행 중.
- ✏️ **09-14 11:1x PDT — Codex 3차(워크플랜): reject 5건 → v1.1 반영·커밋 `320de01`.** 반영: **A46 승인·동결 전 WP1 착수 금지** 문장; 감사 정렬 `--secondary=yes -N 50 -p 0.5` + 직접 구간 검사, 감사 불가 분자 배제, 완전성 ≥ 99.9 %; 단백질 누출 단위 = 쿼리 서열(지지 단위 organism과 분리); C1' = 같은 DuckDB에서 트랙 무시(sha256 하나); thinning rank = hash(aug_seed, 종, 분자)/2^64 nested; aug_seed = 20260914 + seed, withholding 0.1/0.1; Dynamo 폴백 = 실패(`train:317`은 경고만), 메모리 중단 110 GB; 생성 경로가 트랙을 전달해야 함(`modeling:1050`); 역할 전이 규칙(train_track / excluded_heldout / never; excluded_heldout만 입력 가능, 입력 분자는 채점 금지); 옥수수 입력 M-EST + M-HQ18 / 채점 M-FLNC + root-tip ONT, RefGen F1은 reference-dependent 표기; pilot·최종 유전자좌 패널 seed 20260914로 분리 추출; 게이트 = GFFCompare exact intron chain·공통 분모·2,000회 짝 bootstrap·G3에 recall ≥ 0.9×B5 하한; 예산 정정(가용 ≈ 983, 부족 ≈ 292 + 60 GPU-h → 보충 ≈ 350 또는 C1' 생략). **저자 지시 "codex 끝나면 kimi3 리뷰" → Kimi K3 challenge 진행 중**(접근법 자체·최소 pilot·Codex 오류 여부). 그 뒤 WP0(A46 문안 + 저자 결정 4건 확정) → 저자 승인 → WP1.
- ✏️ **09-14 11:4x PDT — Kimi K3 challenge: approve-with-changes → 워크플랜 v1.2 커밋 `c4b9d30`.** Kimi: early-fusion 채널이 최선(별도 모델은 예산 2배·공유 de novo 비교군 상실, M(c) 프롬프트 토큰은 조립 체인이 필요해 단편 EST 90 % 배제, late-fusion 이득 없음); 보완: ① 생성 경로(`modeling:1050-1060` 고정 키, 인코더 1회 캐시 `:713-720`) beam/greedy 트랙 parity 시험 ② pilot 신호 부족 → 패널 크기·MDE(80 % power, 1 pt) A46 동결 ③ 메모리 벤치를 WP4 전에(>100 GB면 K/융합점 재설계) ④ **thinning 해시 미지정 — Python `hash()`는 프로세스별 salt → blake2b 고정 인코딩**(Codex 3회가 놓침); Codex v3 예산 지적은 v1.1에서 이미 해소. **WP-MVP(탐색, 1주)** 제안: 애기장대 1종·held-out 유전자좌 ~20개·입력/채점 수작업 분리·시드 123 best 체크포인트에 채널 얹어 짧은 fine-tune·fractions {0,1}·시드 3개·≈ 20 GPU-h → 효과 0/음이면 WP0 전에 중단, 양이면 WP0~WP7 정당화. Codex–Kimi 충돌 없음(실측 중재 불필요). **저자 결정: WP-MVP 착수 여부(권고: 착수).**
- ✏️ **09-14 10:5x~11:2x PDT — 저자 결정 연속**: ① C1(시드 123)이 유일한 서열 전용 비교군, C1' 생략(`460de14`; 예산 ≈ 935/983 GPU-h) ② 워크플랜 한글판 2종 추가(`b1acb65`) ③ **EST phase 채널 = 경로 B**(diamond blastx `--frameshift 15 --evalue 1e-5 --min-orf 30`, 교차 종 OrthoDB + Swiss-Prot reviewed Viridiplantae, HSP ≥ 30 aa, splice 투영, 경로 A(miniprot) 불일치 시 0, 같은 종 단백질 금지, `est_protein_hsp` 표, C6 phase-off ablation; `9a8c8e1`) ④ "경로 B로 진행" → pronghorn `6163304` `evidence/protein/diamond/build_dbs.sh`(Swiss-Prot 다운로드 + diamond makedb 2종, sha256·PROVENANCE) 제출, 세션 모니터 대기. **WP1(EST×단백질 blastx 등 구축)은 A46 승인 후.** 남은 저자 결정: 대표 모델 · 옥수수 입력/채점 분할 · 121 nt arm 범위 · WP-MVP 착수. GitHub 링크: README(en/ko) `revision/protocols/B6_WORKPLAN_README(.ko).md`, 상세(en/ko) `B6_EVIDENCE_MODE_WORKPLAN_v1(.ko).md`.
- ✅ **09-14 11:1x PDT — 저자 "동의"(권고 4건 채택)**: ① 대표 모델 = **B6 de novo 모드**(증거 모드는 같은 표에 나란히, B5는 보충자료 레거시 비교군; MVP나 G1에서 B6 de novo가 C2보다 유의하게 나쁘면(하한 < −1.0 pt) B5로 복귀하는 조건을 A46에 기록) ② 옥수수 입력 = M-EST + M-HQ18(Wang 2018), 채점 = M-FLNC(Wang 2020) + root-tip ONT(PRJNA822071), Wang 2016 제외 유지 ③ 121 nt arm은 채점/QC 표에만 ④ WP-MVP 착수. **진행 중**: `a46-drafter`(A46 v1.36을 PROTOCOL_B1_frozen_v1.md에 추가, 미커밋) · `mvp-designer`(`revision/protocols/B6_MVP_PILOT_v1.md` 설계, 미커밋) · pronghorn `6163304` diamond DB 빌드(모니터). 다음: A46 Codex 검토 → 저자 승인 → 커밋 → WP1 시작(diamond blastx 배열, C0) + MVP 구현.
- ✏️ **09-14 11:28 PDT — 경로 B diamond DB 완료**(pronghorn `6163304`, 6분): `evidence/protein/diamond/` — Swiss-Prot reviewed Viridiplantae **42,106개**(16.3 M aa, sha256 사이드카), OrthoDB v12 Viridiplantae 필터 **12,115,085개**(5.25 G aa, dmnd 5.75 GB), `DB.sha256`·`PROVENANCE.txt`(diamond 2.1.24, UniProt 쿼리 `(reviewed:true) AND (taxonomy_id:33090)`). 스크립트의 레코드 수 출력은 따옴표 이스케이프 문제로 비었고 `"`라는 잡파일이 생겨 삭제·수기 주석. WP1의 EST×단백질 blastx 입력 준비 완료(A46 승인 후 실행).
- ✏️ **09-14 11:2x~11:4x PDT — WP0 진행**: A46(v1.36) 초안 → Codex 검토 reject 10건(scratchpad `codex_review_a46_output*.txt`) → 수정 지시 반영(패널 결정론적 확정: `62_b6_panels.py` seed 20260914로 mvp 50 / pilot 300 / final = 나머지, **세 패널 서로 소**; 검정력은 WP5에서 기술 보고; 소스별 감사 정렬 명령; WP3 벤치 조건·MVP 체크포인트 정체성; `b6_input` 행의 held-out 추론 입력 carve-out(A13/A18.1 우선); A30 확장(교차 종 Swiss-Prot phase 입력); A33 겹침 성분 플랭크·A18.2 ±3 nt; dose-response 추출별 `aug_seed + d`; B5 복귀 = G1 비열등성 실패만(G2 실패는 증거 모드 주장만 보류); C3/C4/C6 = N_pilot·같은 시드/순서/솎아내기·최종 스텝; MVP 자체 정지 규칙) — A46은 프로토콜 파일 951~989행 + 로그 행 999(미커밋). **병렬**: `panel-builder`(gpu 호스트 B5 DB 읽기 전용 → `revision/protocols/b6_panels_v1.tsv` + sha256), `codex-review-mvp`(MVP 설계서 검토). MVP 설계서 `revision/protocols/B6_MVP_PILOT_v1.md`(미커밋): 애기장대 50 유전자좌(핵심 20), C0-lite 5표, K=8 트랙, 0 초기화 투영(인코더 래퍼, 생성 경로 kwargs 전달), 250 스텝 fine-tune × 3 시드 ≈ 18 GPU-h, **채점은 TAIR10 대비 reference-dependent F1**(A-ONT1/A-HiFi 정렬본 없음 — 장독 드라이버가 학습 부적격 세트 거부), 검정력 20 유전자좌 ≈ 30 pt·50 ≈ 20 pt만 검출. 다음: A46 Codex 재검토 → 저자 승인 → 커밋 → WP1.
- ✏️ **09-14 11:5x PDT — MVP 설계 Codex 검토: approve-with-changes 7건**(코드 인용은 전부 실물 일치). 수정 지시: ① STOP을 futility 검정으로(1차 지표 = valid 타일 증거 on/off 짝 검증 L_GSF, δ_loss 0.005 nats; 2차 = reference-dependent transcript F1, δ_F1 2.0 pt; 구조 지표 = 독립 장독 접합부 P/R; 한쪽 95 % 상한 < δ일 때만 STOP, GO = 하한 > 0 + 3 시드 양, 나머지 INCONCLUSIVE) ② `tile_est`를 valid 타일에도(안 그러면 증거 손실 = de novo 손실) ③ 5일차까지 A-ONT1/A-HiFi 유전자좌 정렬이 없으면 STOP → INCONCLUSIVE, 정렬 단계 필수화 ④ 배제 전 분자 중복 제거(chrom,start0,end0,strand + 클론 접두) ⑤ donor/acceptor는 전사 방향(`ts:A`)으로 정의 + RC 정체성 단위 시험 ⑥ STOP 전 라이브러리 withholding 민감도 1회 ⑦ SIF transformers/torch 버전 고정 기록, 캐시 생성 시험 fail-closed; 50 유전자좌 전부 디코딩, 패널 id는 `62_b6_panels.py` 기준. 수정 후 A46 rule 17(MVP 정지 규칙)과 동기화 필요.
- ✏️ **09-14 12:0x PDT — 패널 규칙 통일**(A46 재검토 에이전트가 A46 rule 12와 MVP 설계서 §1.2의 샘플러 불일치를 발견): `62_b6_panels.py` = 적격 held-out 애기장대 유전자좌 전부에 `shuffle_rank`(seed 20260914) 부여 → `b6_eligible_loci_v1.tsv`(+sha256; 하드 플래그 탈락 수 등 집계); `mvp_select_loci.py`(pronghorn EST BAM 필요) = rank 순으로 MVP 부필터(길이 ≤ 20 kb, 서로 다른 A-EST 분자 ≥ 3) 통과 첫 50개 = MVP, 건너뛴 유전자좌는 잔류, 비MVP 첫 300 = pilot, 나머지 = final → `b6_panels_v1.tsv`(+sha256). A46 rule 12·decision 5·rule 17, MVP 설계서 §1.2, 두 스크립트가 같은 문장을 인용하도록 세 작업자에게 지시. 적격 풀 크기·패널별 수는 승인 시 A46에 기입.
- ✏️ **09-14 12:1x PDT — 패널 1차 산출(구 규칙, 폐기 예정)**: 적격 2,143개(27,416 유전자 → held-out 3,430 → 다중 exon 2,491 → 창 적합 2,485 → 비중첩 2,143 → train_weight>0 2,143). **발견: 동결 DB의 geneList 행은 30,720 nt 타일**(유전자 아님) → 유전자 좌표는 gene_key_map, held-out은 gene_split, exon/intron 수는 build_manifest의 GFF(sha256 검증)에서; held-out 3,430 중 라벨 타일이 있는 유전자는 431개(나머지는 전부 N-마스킹). ChrC/ChrM 5개 포함돼 있어 **핵 염색체만** 필터 추가; `62_` 번호 충돌(62_geenuff_qc.py) → **`71_b6_panels.py`(적격 목록 + shuffle_rank) / `72_mvp_select_loci.py`(+ `72a_est_molecules_per_locus.sh`, pronghorn BAM)**로 개명·재실행 지시(구 `b6_panels_v1.tsv` 삭제). 예상 풀 ≈ 2,138, final = 풀 − 350. a46-drafter·mvp-designer에 이름·필터·타일 구조 문장 반영 지시. 시드 123 링크 2 여전히 PENDING.

### 다음 세션이 바로 할 일 (순서대로)
0. **규칙(저자, 09-09): 새 문제 → Codex/Astra 리뷰 → 진행.** 오늘 오전 리뷰 3회(REPO_DIR approve · 준비 상태 needs-attention → high 3·medium 1 반영 · accumulation 잔여분 수정 needs-attention → 경계 단언 추가) 전부 반영됨.
1. ~~**`3124347`·`3124348` 판독.**~~ ✏️ **둘 다 PASS(~~11:5x·12:1x PDT~~ ✏️ 09:5x·10:0x PDT(sacct 실측 종료 11:54·12:08 CDT; 원문이 CDT를 PDT로 적었다)) — 끝났다.** 원문: 둘 다 `DDP_INTEGRATION PASS`여야 한다. `3124347`이 FAIL이면 `runs/ddp_int/run_3124347/train.err`에서 `restored`·`step 23`·`pending_micro` 중 무엇이 빠졌는지 → Codex 리뷰 → 수정 → `repo_seeds` 갱신 → 재실행(`sbatch --exclude=gh062,gh093 --export=ALL,SIF=$S8,TRANSGENIC_NO_COMPILE=,REPO_DIR=$W/repo_seeds,KILL_AFTER_EPOCH=1,NTRAIN=484 --job-name=ddp-int-epoch deploy/deltaai/ddp_integration.slurm`, cwd `repo_dev`).
2. **시드 123 경계(21:45~22:00 PDT) 검증.** `b5-seed-3116844.out`에 `USR1: forwarding as SIGTERM to trainer ranks: <pid>` → train.err `signal 15 received` → `latest_state saved` → `exiting cleanly` → `resubmitting chain job 2` → `Submitted batch job <id>`; `sacct -j 3116844` COMPLETED; 링크 2가 `afterany` 의존으로 큐에 있고 곧 RUNNING. 대안 경로 `forced preemption`도 정상(마지막 latest_state에서 재개, ≤200 step 손실). `training failed`면 즉시 원인 → 손으로 `sbatch --export=ALL,SEED=123,CHAIN_N=2 /work/nvme/bilv/wcyim/transgenic/repo/deploy/deltaai/train_b5.slurm`(체크포인트는 `runs/seed123/latest_state` 또는 `epoch_02`).
   - epoch 2 `epoch=` 줄(≈ 09-11 01:30 PDT(11:00 PDT 실측 32 %·2.44 it/s → epoch ≈ 25.5 h; 경계 22:00 PDT는 epoch 2 도중) ✏️ 11:39 PDT 실측 **42 %·2.03 it/s → 종료 ≈ 09-11 03:30 PDT**, 예산 943/999) → `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`(epoch 2개부터 판정).
3. ~~**시드 456/789 제출** — 1·2가 끝난 뒤, 저자 "시작해" 한마디로:~~ ❌ **SUPERSEDED (09-10 10:4x PDT, 저자 선택지 2):** **시드 456은 손으로 제출하지 않는다** — `runs/auto_submit_456.sh`가 경계 관문 통과 후 자동 제출한다(`cat runs/auto_submit_456.log`, `runs/seed456_link1_jobid.txt`로 확인; 아래 10:4x 불릿). 손으로 다시 내면 **중복 제출**이 된다. **시드 789만** 저자 결정 뒤 아래 명령을 `SEED=789`로 실행한다:
   ```bash
   W=/work/nvme/bilv/wcyim/transgenic; S8=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
   [ -s $W/runs/seed456_link1_jobid.txt ] && sbatch --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 \
     --export=ALL,SEED=789,SIF=$S8,REPO_DIR=$W/repo_seeds $W/repo_seeds/deploy/deltaai/train_b5.slurm
   ```
   (원문 `for SEED in 456 789` 루프는 폐기.)
   ```
   예상: ≈ 6 s/step(30 kb), epoch 6.15 h, 시드 2.0~2.6일(48 h 링크 2개), 197~246 GPU-h. **시작 직후 확인**: `b5-seed-<id>.out`의 `repo: … @ c646e76` · `image: …26.08.sif` · `runtime identity: … gpus=4 cpus=64 torch=2.14…` · `PREFLIGHT PASS` · `gpu warm-up ok: device 0~3` · train.err `torch.compile scope: model`(4회) · `DDP world size 4: per-rank accumulation 24` · step 200 `latest_state saved`. `/work/repo`·`/work/repo_seeds`는 학습 중 pull 금지, 실험은 `repo_dev`.
4. 시드마다 epoch 뒤 `60_epoch_budget.py --patience 3`; 링크 경계마다 2번과 같은 확인(시드 456/789는 링크 1 시작 +48 h −15 min).
5. 위키 **update 37**(시드 시작 + 오늘 오전: 26.08 결정·A44·인증·accumulation 수정), `spec.md`, 결과표에 시드 123 train loss 보정(아래).

### 오늘 확정된 것 (오전; 상세 quarantine §1k·todo 상단)
- **저자 결정 26.08**: 시드 123과 이미지·torch·컴파일러 커널·4 rank reduction 순서·RNG 스트림이 달라진다 → A44는 이를 **혼합 런타임 재현**으로 명시하고 중립성 주장을 하지 않는다. 비교는 검증 손실 궤적·종료 epoch·하류 평가(A19.4)로.
- **동결 체크아웃 분리**: 시드 123은 `/work/repo`(08f3e2b, 스크립트 ba24b19)에서 그대로; 확인 시드는 `repo_seeds`(c646e76)에서. 체인 스크립트가 `REPO_DIR`를 받아 컨테이너의 `/work/repo`에 바인드하고, 재제출 링크가 같은 체크아웃·SIF·GPU·CPU 수를 물려받는다(Codex approve).
- **첫 4 GPU·26.08 리허설 `3124100`이 preflight에서 rc 3**: 토크나이저 검사가 stderr까지 읽는데 torch 2.14가 import 시 경고 한 줄을 stderr에 찍어 `could not verify`로 launch를 거부했다 → stdout 마지막 줄만(`07486a8`). 그 뒤 `3124229` PASS.
- **Codex 준비 상태 리뷰 → 반영(`5433f27`)**: ① latest_state를 rename만으로 교체(구판→`.prev`, 신판→`latest_state`; `resume_dir`가 `.prev` 인정) — 이전엔 rmtree와 rename 사이에 죽으면 복구점이 없었다 ② 첫 링크가 `runtime_identity.txt`(SIF·sha256·커밋·GPU·CPU·torch·checkpointing·compile 정책)를 쓰고 이후 링크는 다르면 거부(`RUNTIME_OVERRIDE=1`) — `run_config.json`은 레시피 항목만 비교했다 ③ preflight가 소스 검사만으로 통과하던 fail-open 제거(`PREFLIGHT_ALLOW_SOURCE=1`은 진단용).
- **epoch 경계 accumulation 잔여분 보존(`c646e76`, 저자 지시)**: 198,828행은 96의 배수가 아니라 epoch 끝마다 1 GPU 12개·4 GPU rank당 3개 micro-batch의 gradient가 다음 epoch으로 넘어간다. `accelerator.save_state`는 `.grad`를 저장하지 않아 epoch 체크포인트 재개 시 그 12개가 유실되고 이후 모든 update 경계가 밀렸다. 이제 rank마다 `pending_grads/rank<i>.safetensors`를 쓰고 `meta.json`에 `pending_micro`, 재개 시 복원·`micro_done` 이어받기, 파일 없으면 거부. **시드 123(동결)은 미적용** — 정상 체인은 mid-epoch latest_state(optimizer 경계)에서 재개하므로 epoch 체크포인트가 최신 복구점이 되는 경우에만 해당; 발생하면 런 로그에 편차로 기록.
- **시드 123 train loss 과소 보고**: train loss 평균이 epoch을 넘어 누적된 `micro_done`으로 나뉘어 **epoch ≥ 2의 `train_epoch_loss`는 epoch 번호만큼 작게 찍힌다**(epoch 2 값 ×2가 참값). 검증 손실·조기 종료 무관. 새 코드는 `epoch_micro`로 나눔. 결과표에서 보정한다(로그는 수정하지 않음).
- 통합 시험 `KILL_AFTER_EPOCH` 모드: `NTRAIN=484`(rank당 121, 잔여 1)로 epoch 체크포인트 재개 경로를 실측 — 복원이 맞으면 epoch 2 첫 update가 micro-batch 23, 카운터를 잃으면 24.
- 시드 123 경계 시각 정정: ~~21:50 CDT~~ → **21:45 PDT 시그널 / 22:00 PDT 상한**(sacct Start 2026-09-09 00:00:34 CDT).
- Codex 부수 의견(미반영): pending_grads는 DDP에서 4 rank가 동일한 ~1.7 GB 파일을 각각 씀 → rank 0만 쓰는 최적화 가능. 체인 트랩의 `forwarding as SIGTERM … ranks:` echo는 pid를 한 줄에 하나씩 찍어 감시 grep이 첫 pid만 잡는다(표시 문제).

### 저자 결정 — 완료와 대기
- ✅ 확인 시드 이미지 = 26.08(A44) · ✅ 4 GPU · ✅ 예산 보충 · ✅ 시간 PDT · ✅ accumulation 잔여분 수정 · ✅ #45 코멘트 게시
- ⏳ **"시작해"** — 1·2번 뒤 시드 456/789 제출 신호
- ⏳ (c) 트레이너 파일 기반 stop 플래그(dynamo 아래 시그널 의존 제거; 지금은 SIGTERM으로 작동) · (d) NCSA에 첫 launch failure 보고 · (e) medium 발견(`.mpi` 인덱스↔참조 체크섬 결속) · pending_grads rank-0-only 최적화

🧊 **ARCHIVED BELOW** — 아래 "🔄 세션 인수인계 (2026-09-09 저녁 …)"부터는 과거 스냅샷. 저녁 블록의 "다음 세션이 바로 할 일" 1~3번은 **집행됨**(4 GPU 벤치 판독, epoch 1 판독, 26.08 SIGTERM 통합 시험 PASS), 4번의 (a)는 **저자가 26.08로 결정**(권고는 25.06이었다), (b)는 예산상 불필요(여유분만), 5번 위키 36은 게시됨(`599ec17`). 저녁 블록의 "~~21:50 CDT~~ ⚠️ CAUTION (09-10 오전: 계산 오류 — 실제 **USR1 21:45 PDT / 상한 22:00 PDT**)"는 **틀린 시각** — 개별 마커 참조.

# 🧊 🔄 세션 인수인계 (2026-09-09 저녁 작성, 17:2x PDT)

> ❌ **SUPERSEDED (2026-09-10 오전):** 이 블록의 상태 표·할 일·권고는 위 "🔄 세션 인수인계 (2026-09-10 오전)"이 대체한다. 수치 중 경계 시각과 이미지 권고에는 개별 마커가 있다.


**돌고 있는 것부터 확인할 것.** DeltaAI는 ControlMaster 소켓이 있어야 닿는다(`ssh -O check dtai` → "Master running"; 아니면 사용자가 pronghorn `login-0`에서 `ssh -fN dtai`).

```bash
ssh dtai 'squeue -u $USER -h -o "%i %j %T %M %N"; tail -7 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log'
ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo_dev && for f in b5-bench-3118904.out b5-bench-3118906.out b5-bench-3118908.out; do echo "== $f"; grep -hoE "\{\"steps_measured\".*|driver_peak_mem.*|OutOfMemory|invalid, but is output" $f; done'
ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo && git status --short | grep -v "^??"; diff <(git show origin/main:deploy/deltaai/train_b5.slurm) deploy/deltaai/train_b5.slurm && echo FROZEN_SCRIPT_MATCHES_MAIN'
```

| 무엇 | 상태 (09-09 17:20 CDT) |
|---|---|
| **SEED=123** 잡 `3116844` · 1 GPU gh015 · eager + reentrant ckpt · `/work/repo` `08f3e2b` + `train_b5.slurm`(=`1a245ce`) · NGC 25.06 | RUNNING 17 h · epoch 1 진행 중(13:10에 46 %) → 첫 `epoch=` 줄 ≈ **09-09 22:00 CDT**. **첫 48 h 경계 ≈ 09-10 ~~21:50 CDT~~ ⚠️ CAUTION (09-10 오전: 계산 오류 — 실제 **USR1 21:45 PDT / 상한 22:00 PDT**)** — 그때 쓰는 스크립트가 아래 리허설로 실증된 것 |
| **체인 리허설**(동결 트레이너, SEED=999, 15분, `--signal=B:USR1@600`) | `3120089` **COMPLETED 0:0**(5:40): USR1 → SIGTERM 전달 → `signal 15 received` → `latest_state saved … global_step 1` → `exiting cleanly` → `chain limit 1 reached`. `3120124`(강제 선점 경로, `USR1_GRACE=1`, `CHAIN_MAX=2`): watchdog 발동 → `forced preemption (rc=1); resubmitting chain job 2` → 자식 `3120146` 제출 → **손으로 취소**. 리허설 디렉터리 seed997/998/999 삭제 |
| **DDP 통합 시험** 4 GPU (`ddp_integration.slurm`, 480행 30 kb 부분 DB) | **`3119681` PASS**(25.06 eager, 기본 할당기) · **`3119682` PASS**(25.06, `ALLOC=native`) — rank-0 저장, rank 하나 USR1 → 4 rank 모두 `exiting cleanly`, latest_state 재개, 검증 합산, TRAINING_DONE. `3119683`/`3119779`(26.08 compile): 학습·저장·검증·재개·TRAINING_DONE 통과, **USR1 중단만 미발화**(dynamo, 아래) → SIGTERM으로 재실행 필요 |
| **벤치 26.08 compile, 1 GPU** (50 step, warmup 20 폐기, 호스트 nvidia-smi 2 s 샘플링) | `3118903` 30 kb **21.35 s/step**(1.17×) · `3118905` 61 kb **40.54**(1.12×) · `3118907` 129 kb **86.64**(1.12×) → 1 GPU 에폭 ≈ **22.6 h**(eager 25.6 h, 1.13×). 대조 25.06 eager 오늘: non-reentrant `3118909` 31.63 · reentrant `3119174` 29.96(ckpt 형태 ≈ 5 %; 09-08의 24.9와 20~27 % 차이는 노드/일자 변동, 미해명) |
| **벤치 26.08 compile, 4 GPU** `3118904`(30 kb)·`3118906`(61 kb)·`3118908`(129 kb) | **PENDING**(노드 통째 대기) — 시드당 GPU-h·이미지 권고가 이것에 달려 있다 |
| pronghorn 단백질 수용 **v2** `6156132` | 완료 9종 — `protein_acceptance_v2.tsv`(`7df570f`), A43, **#45 정정 코멘트 게시됨** |

### 다음 세션이 바로 할 일
0. **규칙(저자, 09-09): 새 문제가 생기면 Codex/Astra 리뷰 → 진행.** 저녁의 체인 재시도·USR1 변경은 리뷰가 돌 때마다 결함이 나왔고(§1j), 리뷰 2회는 로컬 메모리 부족으로 죽어 **미실행** — `adversarial-review --base 7df570f --scope branch`로 다시 돌릴 것.
1. 4 GPU 벤치 3런 판독 → 스케일링 효율 → **시드당 GPU-h·벽시계** → 저자에게 ACCESS 추가 배정량과 이미지 권고.
2. 시드 123 첫 `epoch=` 줄(≈ 22:00 CDT) → `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`. 09-10 ~~21:50 CDT~~ ⚠️ CAUTION (09-10 오전: 계산 오류 — 실제 **USR1 21:45 PDT / 상한 22:00 PDT**) 경계 뒤 `b5-seed-3116844.out`에 `USR1: forwarding as SIGTERM … / chain job 2` 또는 `forced preemption`이 찍히는지 확인.
3. 26.08 compile 통합 시험을 **SIGTERM 판**으로 재실행(`ddp_integration.slurm`의 `kill -USR1` → `kill -TERM`, 아직 안 고침) — 26.08을 확인 시드 이미지로 쓰려면 이 PASS가 조건.
4. 저자 결정 **(a)** 확인 시드 이미지 25.06 eager(통합 PASS, 시드 123과 동일) vs 26.08 compile(1.12~1.17×, 3번 + A44 필요) **(b)** ACCESS 추가 배정량 **(c)** 트레이너에 파일 기반 stop 플래그(`STOP_REQUESTED`, 수치 무관) 허용 여부 — 체인이 dynamo 아래 시그널 전달에 의존하지 않게 **(d)** 첫 launch 실패 NCSA 보고 **(e)** medium 발견(인덱스↔참조 체크섬) 보류/수정.
5. 위키 update 36(초안은 pgl-wiki에 있음, 미커밋) · `spec.md`.

- ✏️ **17:3x 추가 (Codex 리뷰 반영, `ba24b19`)**: 체인 스크립트 최종판 — GPU마다 워밍업, step-0 launch failure는 **같은 할당 안에서 먼저 재실행**(두 번째도 실패해야 노드 제외 재제출), 트랩은 두 번째 USR1 무시·모든 kill `|| true`, 강제 선점 마커는 `FORCED_PREEMPT.<jobid>`로 잡 단위(상한에서도 소비, launch 시그니처보다 우선). 테스트 18건. 동결 `/work/repo` 스크립트 = `ba24b19`. 리허설 `3120196`(정상 경로, SEED=999, CHAIN_MAX=1) 진행 중 — 통과하면 내일 밤 경계는 이 판으로 간다.

- ✏️ **18:2x 추가 — 4 GPU 벤치 완료 + 최종 리허설 PASS**: `3120196`(ba24b19) COMPLETED 0:0 — 내일 밤 경계용 스크립트 실증 완료. 26.08 compile 4 GPU DDP: 30 kb **5.89** s/step(1 GPU 21.35 → 3.62×, 90.6 %) · 61 kb **11.04**(3.67×, 91.8 %) · 129 kb **23.23**(3.73×, 93.2 %) → **epoch 6.15 h 벽시계 = 24.6 GPU-h**, 시드(8~10 epoch) 49~62 h·197~246 GPU-h, 두 시드 394~492 GPU-h — 잔여 968에서 시드 123 잔여(~190)를 빼도 **추가 배정 없이 들어감**(안전 여유로 ~300 요청 권고). 25.06 eager 4 GPU는 미측정(1 GPU eager 25.6 h/epoch × 같은 효율 ≈ 7.0 h/epoch, 28 GPU-h → 시드 225~280 GPU-h; compile 절감 ≈ 12 %). ~~**권고**: 확인 시드는 25.06 eager 4 GPU~~ ❌ SUPERSEDED (09-10 오전: 저자 결정 **26.08**, A44)(시드 123과 같은 이미지·커널, 3119681/3119682 + 체인 리허설로 완전 인증); 26.08은 12 % 절감 대신 A44 개정 + ddp_integration SIGTERM 재시험 필요. 드라이버 2초 샘플 최대 ≈ 97 GB/GPU(하한).

- ✏️ **09-10 09:55 CDT — 시드 123 epoch 1 완료**: `epoch=0: train_loss 1.6361 (ppl 5.13) · eval_loss 0.8617 (ppl 2.37)`, New best at epoch 1, `epoch_01/`·`best`·`trainer_state.json`(global_step 2071). epoch 2는 26 %(2.22 it/s) → 종료 ≈ 20:00 CDT; 링크 1의 48 h 경계 ≈ **~~21:50 CDT~~ ⚠️ CAUTION (09-10 오전: 계산 오류 — 실제 **USR1 21:45 PDT / 상한 22:00 PDT**)**(epoch 2 도중, latest_state 200 step마다). `60_epoch_budget.py --patience 3` = insufficient_data(2 epoch 이상 필요). 예산 949/999. 동결 스크립트 md5 64477d87 = ba24b19 확인.

### 오늘 확정된 것 (저녁; 상세 quarantine §1j·todo 상단)
- **A28 48 h 체인은 오늘 전까지 한 번도 작동한 적이 없었다.** 체인 스크립트 결함 6건을 각각 15분 리허설로 잡았다: ① `if wait; fi; $?`가 모든 crash를 rc 0으로(`5bc4df8`) ② USR1을 apptainer에 보내 컨테이너 사망 rc 138, 트레이너는 시그널을 못 봄(`3119780`) → rank에만(`4fe9780`); 런처 argv에 스크립트 경로가 있어 런처까지 죽임(`3119819`) → cmdline에 accelerate/bash -lc 없는 것만(`3ed6413`) ③ **dynamo가 켜진 rank는 SIGUSR1 핸들러를 실행하지 않는다**(`3119848` 13분 무반응 TIMEOUT, `3119779`, probe `3119984`: USR1 150 s 무시 → 같은 rank에 **SIGTERM은 즉시** `signal 15 received`; `TRANSGENIC_NO_COMPILE=1`이면 USR1도 됨 `3119954`) → SIGTERM 전달(`9161508`) ④ watchdog: 600 s 안에 안 나가면 rank kill + FORCED_PREEMPT → 마지막 latest_state로 다음 링크(`840384f`) ⑤ 새 4 GPU 할당의 첫 launch `unspecified launch failure`(gh121, gh139; 1 GPU gh062×2, gh093)는 같은 노드 두 번째 launch가 매번 정상 → 첫 접근 경합(Codex: nvidia_uvm/cudaMallocAsync 풀 + Apptainer --nv), NCCL·노드 문제 아님 → SIF 안 `torch.ones(1).cuda()` 워밍업 + 제자리 재시도, 두 번째 실패에서만 노드 제외(`7beb803`/`ed10056`) ⑥ 끝난 watchdog `kill`이 `&&` 뒤에서 실패해 errexit가 재제출 직전에 스크립트를 죽임(`3119991`) → `|| true`(`1a245ce`). 최종 리허설 `3120089` 정상 경로·`3120124` 강제 경로 **둘 다 통과**. `tests/test_train_b5_chain.py` 17건.
- **메모리 해석 정정의 정정**: 낮에 "torch 101 GB는 cudaMallocAsync 풀 계정, 실사용 ≈ 30 GB"라고 적은 것은 **틀렸다.** nvidia-smi 2 s 샘플 최대가 14.8(eager 30 kb)·78.9(eager reentrant 30 kb)·31.5/84.9/97.3 GB(compile 30/61/129 kb)로 같은 torch 101 GB 런들에서 제각각 → 샘플링이 마이크로배치 안 순간 최고점을 놓치는 것이고, torch의 101 GB가 순간 최고점(디코더 지배; 129 kb no-ckpt OOM 100.65 GB 실제 사망과 일치). **no-ckpt 옵션 닫힘**, 여유 ≈ 19 GB, 129 kb compile은 천장 근처.
- **compile 이득은 12~17 %**(1 GPU, 25.06 eager 09-08 대비); 같은 날 같은 노드 대조로는 1.4×. 벽시계 레버는 4 GPU DDP.
- **DDP 경로 인증(25.06)**: `3119681`·`3119682` PASS — 5bc4df8의 DDP 수정 3건(rank-0 소유, 합산 손실, 합산 stop 플래그)이 4 GPU 실전에서 작동. 26.08 compile은 중단 경로만 미인증(SIGTERM 재시험).
- 26.08 통합 시험 첫 판(`3119205`)의 USR1은 **DataLoader 워커**(rank와 같은 cmdline)에 갔었다 → 런처 직계 자식만 표적(`0c3f4bd`).
- 저자 질문 "checkpointing 대신 일정 epoch에서 끝낼 수 있나" → gradient ckpt(메모리)와 잡 체크포인트(48 h 상한)는 별개; `--max-epochs`는 있으나 종료 규칙(patience 3) 변경은 개정 사항.
- bash 잡 알림 `Terminated ( sleep … )`은 watchdog 서브셸이 죽을 때 bash가 찍는 텍스트 — 무해.

(저녁 배너, 기록용) ~~🧊 ARCHIVED BELOW — 아래 "🔄 세션 인수인계 (2026-09-09 낮 …)"부터는 과거 스냅샷~~ → 격리 배너는 위 오전 블록 아래 하나뿐이다; 이 줄부터 아래 전부가 그 배너 아래다. 낮 블록의 "다음 세션이 바로 할 일" 1·2번은 **집행됨**(1 GPU 벤치·계측 대조·통합 시험), 3~5번은 위 1~5번으로 대체. 낮 블록의 메모리 해석("101 GB = 풀 계정")과 no-ckpt 옵션은 **틀린 것으로 판명** — 개별 마커 참조.

# 🧊 🔄 세션 인수인계 (2026-09-09 낮 작성, 11:3x PDT)

> ❌ **SUPERSEDED (2026-09-09 저녁):** 이 블록의 상태 표·할 일은 위 "🔄 세션 인수인계 (2026-09-09 저녁)"이 대체한다. 이 블록의 메모리 해석과 no-ckpt 옵션은 §1j에서 기각됐다.

**돌고 있는 것부터 확인할 것.** DeltaAI는 ControlMaster 소켓이 있어야 닿는다(`ssh -O check dtai` → "Master running"; 아니면 사용자가 pronghorn `login-0`에서 `ssh -fN dtai`).

```bash
ssh dtai 'squeue -u $USER -h -o "%i %j %T %M %N"; tail -7 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log'
ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo_dev && for f in b5-bench-31189*.out ddp-int-*.out; do echo "== $f"; grep -hoE "\{\"steps_measured\".*|driver_peak_mem.*|DDP_INTEGRATION.*|leg[12] rc=.*|OutOfMemory|invalid, but is output" $f; done'
```

| 무엇 | 상태 (09-09 13:10 CDT) |
|---|---|
| **SEED=123** 잡 `3116844` · 1 GPU gh015 · eager + reentrant ckpt · `/work/repo` `08f3e2b`(+ train_b5.slurm 한 파일 패치) · NGC 25.06 | RUNNING 12 h · epoch 1 **46 %**(91,791/198,828, 2.1~3.1 it/s) → **epoch ≈ 26 h**, 첫 `epoch=` 줄 ≈ **09-09 22:00 CDT**. 예산 980/999(저자: 크레딧 필요한 만큼 보충) |
| **벤치 매트릭스, NGC 26.08 compile** (repo_dev `7f20816`, STEPS=50, OUTTAG `c2608`, `--exclude=gh062,gh093`) | `3118903`(30 kb·1 GPU, **COMPLETED 14:0x: 21.35 s/step = 1.17×, 드라이버 peak 31.5 GB, torch peak 101.0 GB**) `3118904`(30 kb·4) `3118905`(61 kb·1) `3118906`(61 kb·4) `3118907`(129 kb·1) `3118908`(129 kb·4) PENDING |
| **계측 대조** `3118909` · 25.06 eager · 30 kb · 1 GPU · nvidia-smi 샘플링 (OUTTAG `e2506`) | ~~PENDING — 101 GB(25.06) vs 11 GB(26.08) 중 어느 쪽이 실제인지~~ ✏️ (09-09 저녁) COMPLETED: 31.63 s/step(non-reentrant), 샘플 최대 14.8 GB, torch 100.9 GB — 판정은 "torch 101 GB가 실제"(§1j) |
| **DDP 통합 시험** `3118861`(25.06 eager) · `3118880`(26.08 compile, `ddp-int-2608`) · 4 GPU · 480행 30 kb 부분 DB | ~~PENDING~~ ✏️ (09-09 저녁) `3118861` leg 1은 step-0 launch failure, `3118880` 취소; 인증은 재제출분 `3119681`·`3119682` **PASS**(위 저녁 블록) |
| pronghorn 단백질 수용 **v2** 배열 `6156132` | **완료** 9종 — `protein_acceptance_v2.tsv`(`7df570f`), A43 |

### 다음 세션이 바로 할 일
0. **규칙(저자, 09-09): 새 문제가 생기면 Codex/Astra 리뷰 → 진행.** 오늘 이 규칙으로 커밋 전 리뷰 3회를 돌렸고, 그때마다 실제 결함이 나왔다(아래 §1i).
1. 벤치 6런 + `3118909` 판독 → 정상 상태 s/step(warmup 20 폐기), `driver_peak_mem_mib_per_gpu`(호스트 nvidia-smi 최대), **4 GPU 스케일링**. 이것으로 시드당 GPU-h → 저자에게 ACCESS 추가 배정량.
2. `3118861`·`3118880` 판독 — 둘 다 PASS여야 확인 시드에 DDP를 낼 수 있다. FAIL이면 `runs/ddp_int/run_<job>/train.err`.
3. 저자 결정: **(a)** #45 정정 코멘트 게시 **(b)** 확인 시드 이미지 — 25.06 eager(시드 123과 동일) vs **26.08 compile**(벤치·통합시험 PASS + 이미지/torch가 시드 123과 다르므로 개정 A44 필요) **(c)** ACCESS 추가 배정량 **(d)** gh062/gh093 NCSA 보고 **(e)** medium 발견(청크 정렬 인덱스↔참조 체크섬 미결속) 수정/보류.
4. 시드 123 첫 `epoch=` 줄(≈ 22:00 CDT) → `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`. 그 뒤 시드 456/789 **4 GPU**(저자 결정 09-09) 제출: `sbatch --gpus-per-node=4 --export=ALL,SEED=456 deploy/deltaai/train_b5.slurm`(이미지 결정 반영; `/work/repo`는 시드 123 체인이 끝날 때까지 건드리지 않고 별도 체크아웃).
5. 위키 update 36, `spec.md`.

### 오늘 확정된 것 (상세는 quarantine §1i·todo 상단)
- ✏️ **13:5x 추가**: `3118903` 정상 상태(warmup 20 폐기) = **21.35 s/step**(25.06 eager 24.9 → **1.17×**, 기대 1.41×보다 작음; 61·129 kb에서 커지는지 `3118905`·`3118907`이 답). torch `max_memory_allocated`는 **101.04 GB**(25.06 eager와 같은 값, 3 step probe의 11 GB는 재현 안 됨)인데 호스트 nvidia-smi 최대(588 샘플)는 **31.5 GB** — ~~101 GB는 cudaMallocAsync 풀 고수위 계정일 가능성~~ ❌ SUPERSEDED (09-09 저녁): 2 s 샘플이 순간 최고점을 놓친 것, 101 GB가 실제 순간 peak(§1j). ~~`3118909`(25.06 eager + nvidia-smi)가 판정. 낮게 나오면 **26.08에서 `TRANSGENIC_NO_GRAD_CKPT=1` 129 kb 벤치 1런**을 돌릴 가치가 있다~~ ❌ SUPERSEDED (09-09 저녁): `3118909`는 14.8 GB(샘플 하한)였고 no-ckpt 옵션은 닫혔다. **#45 정정 코멘트 게시됨**(저자 "진행 해", 14:4x PDT): https://github.com/wyim-pgl/transgenic/issues/45#issuecomment-5606950770 — 이슈는 닫힌 채. 저자 질문 "checkpointing 대신 일정 epoch에서 끝낼 수 있나"에 답함: gradient ckpt(메모리)와 잡 체크포인트(48 h 상한)는 별개, `--max-epochs`는 있으나 종료 규칙(patience 3) 변경은 개정 사항.
- **Codex adversarial review(4ef3dc4..2021fd2)가 high 5건을 찾았고 전부 코드로 검증됐다** → `5bc4df8`: 체인 wait 루프가 트레이너 실패를 rc 0으로 삼켜 최대 8링크 재제출(자식 exit 17 → rc 0 재현; 시드 123의 동결 스크립트에도 있었다 → 한 파일 패치) · reentrant ckpt 기본값 + DDP 충돌 · 모든 rank가 체크포인트 디렉터리를 rmtree/makedirs/rename(50-step 벤치는 save 200에 닿은 적 없음) · rank별 검증 손실 미합산 → early-stop 분기 · `protein_accept.py`가 mapping_ambiguous 판정 전에 지지 행을 써서 1.4~4.1 % 단백질이 locus별 지지(A19.2 위반). 수정안 자체도 Codex 리뷰 2회를 더 거쳐 stop 플래그·train loss 합산, DDP KeyboardInterrupt 저장 금지를 추가. 테스트 22건 신규.
- **#45 v2**: 수용률·mapping_ambiguous 불변, 커버리지 구간 −2~−7 %, ≥2 organism 인트론 −1.3~−4.1 %(Athaliana 345,836→341,479, Gmax 619,329→603,056). v1 산출물은 `accept_v1_ambiguous_leak/`에 보존, 라벨 금지(A43).
- **compile**: 원인은 HF checkpointing + dropout RNG(pytorch#135459/#164186/#161889). per-layer compile(25.06)도, **26.04(torch 2.12.0a0)도 같은 단언**. **26.08(torch 2.14.0a0)에서 통과** — 30 kb·129 kb 모두, OOM 없음(3 step 값이라 속도는 정상 상태가 아님). 보고된 peak 11~24 GB는 26.08 **eager도 11 GB**라 compile 효과가 아니라 ~~torch 2.8→2.14 계측 차이 의심 → 호스트 nvidia-smi가 판정(`7f20816`)~~ ⚠️ CAUTION (09-09 저녁): 50 step에서는 26.08도 torch 101 GB를 보고했다(3 step probe 값은 재현 안 됨); nvidia-smi 2 s 샘플은 하한일 뿐(§1j).
- **저자 결정**: 확인 시드 456/789 = **4 GPU DDP**; 예산은 제약 아님(보충). round 6(`3117671`, 129 kb 2 GPU 55.64 s/step 1.74×, peak 100.5 GB)은 통과했지만 2 GPU 권고는 4 GPU 결정과 DDP 결함으로 **대체됨**.
- 인터랙티브 패턴: `salloc --no-shell` + `srun --jobid --overlap` + detached 스크립트 → 파일. ⚠️ 오케스트레이터의 `scancel`이 먹지 않아 `3118849`가 17분 유휴(≈0.3 GPU-h) — scancel 뒤 `squeue`로 확인.

(낮 배너, 기록용) ~~🧊 ARCHIVED BELOW — 아래 "🔄 세션 인수인계 (2026-09-09 아침 …)"부터는 과거 스냅샷~~ → 격리 배너는 위 저녁 블록 아래 하나뿐이다; 이 줄부터 아래 전부가 그 배너 아래다. 아침 블록의 "다음 세션이 바로 할 일" 1~2번은 집행됨(round 6 판독, 결정표 채움).

# 🧊 🔄 세션 인수인계 (2026-09-09 아침 작성 — `/clear` 직전)

> ❌ **SUPERSEDED (2026-09-09 낮):** 이 블록의 상태 표·할 일은 위 "🔄 세션 인수인계 (2026-09-09 낮)"이 대체한다.


**돌고 있는 것부터 확인할 것.** DeltaAI는 ControlMaster 소켓이 있어야 닿는다(`ssh -O check dtai` → "Master running"; 아니면 사용자가 pronghorn `login-0`에서 `ssh -fN dtai`).

```bash
ssh dtai 'squeue -u $USER -h -o "%i %j %T %M %N"; tail -7 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log'
ssh dtai 'sacct -j 3117671 --format=State%12,Elapsed%9,NodeList%8 -n -X; cd /work/nvme/bilv/wcyim/transgenic/repo_dev && tr "\r" "\n" < b5-bench-3117671.out | grep -oE "\{\"steps_measured\".*\}" | tail -1'
```

| 무엇 | 상태 (09-09 04:35 CDT) |
|---|---|
| **SEED=123** 잡 `3116844` · 1 GPU gh015 · eager + reentrant ckpt · repo `08f3e2b` · NGC 25.06 | RUNNING 4:34 · 17 % · 2.4~2.9 it/s · ~~**첫 `epoch=` 줄 ≈ 09-09 낮**~~ ⚠️ CAUTION (09-09 낮): 13:10에 46 %, **epoch ≈ 26 h → 첫 줄 ≈ 22:00 CDT** · A28 체인 자동(48 h, CHAIN_MAX 8) · 모니터 30분 간격 |
| **round 6** 잡 `3117671` · 129 kb · **2 GPU eager + non-reentrant ckpt** · repo_dev · `--exclude=gh062,gh093` | ~~PENDING~~ ✏️ **COMPLETED (09-09 낮, gh109 47분): 55.64 s/step(1.74×), peak 100.5 GB, acc 48/rank, 오류 0.** ~~통과하면 확인 시드 456/789에 DDP 권고 확정~~ ❌ SUPERSEDED (09-09 낮): 저자가 **4 GPU**로 결정했고 DDP 결함 3건(`5bc4df8`)이 발견돼 권고는 **통합 시험 `3118861`/`3118880` PASS 조건부** |
| 예산 `bilv-dtai-gh` | ~~989/999 GPU-h~~ ✏️ 980/999 (09-09 13:10; 저자: 필요한 만큼 보충) |
| pronghorn 단백질 증거 | **완료** — #45·#44 CLOSED, `protein_acceptance_v1.tsv`(6e08e01) |

### 다음 세션이 바로 할 일
0. **규칙(저자, 09-09): 새 문제가 생기면 Codex/Astra 리뷰 → 진행.** 지난밤은 자율로 진행했고 그중 회귀 1건(`e962864`→`d7c3a6d`)은 스모크가 잡았다 — 리뷰가 먼저였으면 더 쌌다.
1. round 6 판독 → 결정표 마지막 칸(DDP) 채움. 통과 시 `TRANSGENIC_CKPT_NONREENTRANT=1`을 **기본값으로 커밋** 제안(저자 승인). 시드 123 체인엔 적용 금지.
2. 시드 첫 에폭 `epoch=` 줄 → `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`. 링크가 `unspecified launch failure`로 죽으면 **노드(gh062/gh093)** — `--exclude`로 재제출.
3. `spec.md` (intent open question 1 닫힘 — 실측 사이징으로). `intent.md` 갱신됨.
4. 저자 결정: 확인 시드 DDP(2 GPU 확정, 4 GPU 스케일링 미측정) · non-reentrant 기본 · gh062/gh093 NCSA 보고 · 위키 update 35 게시됨(`62420bc`).

### 오늘까지 확정된 결론 (상세는 아래 🌅 블록·quarantine §1h)
- **compile은 현 레시피에 불가**: 원인 HF checkpointing + compile; ckpt OFF면 1.41×/2.4×지만 **129 kb OOM**; 인코더-only는 **오히려 느림**(26.6 vs 24.9). → 시드 123 재시작 **않음(확정)**.
- **DDP 경로 확보**: acc ÷ world_size(`2021fd2`) + non-reentrant ckpt → 2 GPU 17.57 s/step(**1.42×**), 레시피 충실. ⚠️ CAUTION (09-09 낮): 50-step 벤치는 체크포인트 저장·검증에 닿지 않는다 — 그 경로의 결함 3건은 `5bc4df8`에서 고쳤고 통합 시험이 인증 전이다.
- 이미지 **25.06 유지**(25.11은 단언 동일; "죽는다"는 gh062 탓이었음).
- 4-GPU는 **미측정**(2-GPU 스케일링 71 %).

🧊 아래 "🔄 세션 인수인계 (2026-09-08 …)" 블록은 **집행 완료·초과됨** — 그 안의 "다음 세션이 바로 할 일"은 전부 끝났다. 읽기용.

# 🧊 🔄 세션 인수인계 (2026-09-08 작성 — `/clear` 직전)

## 🌅 아침 보고 (2026-09-09 새벽 자율 진행분) — 여기부터 읽을 것

### ⚡ 아침에 할 일 세 가지, 이 순서로
1. **DeltaAI 소켓 재개통** — 새벽에 또 죽었다. pronghorn `login-0`에서 `ssh -fN dtai` (Duo). `ssh -O check dtai` → "Master running".
2. **round 5 결과 읽기** (소켓 없이 못 읽었다; 잡은 DeltaAI에서 스스로 끝났다):
   ```bash
   ssh dtai 'sacct -j 3117589,3117590 --format=JobID%9,State%12,Elapsed%9,NodeList%8 -n -X; cd /work/nvme/bilv/wcyim/transgenic/repo_dev; for j in 3117589 3117590; do echo "--- $j"; tr "\r" "\n" < b5-bench-$j.out | grep -oE "\{\"steps_measured\".*\}" | tail -1; tr "\r" "\n" < b5-bench-$j.out | grep -E "^\[rank[0-9]\]:.*Error|CUDA error|was invalid" | sort | uniq -c | head -3; done'
   ```
   `3117589` = 인코더만 compile + ckpt ON, 1 GPU, 50 step (eager 24.9 s/step와 비교) · `3117590` = eager + non-reentrant ckpt, **2 GPU**, 50 step (성공하면 DDP 경로 확보, `sec_per_step` ≈ 12~13 기대).
3. **시드 확인** — `tail -6 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log`. 04:06 CDT 기준 15 %·2.85 it/s·에폭 ETA ~16.5 h → **첫 `epoch=` 줄은 09-09 낮**. 그 뒤 `60_epoch_budget.py --patience 3`.

### 밤에 확정된 것 (전부 repo_dev·`--exclude` 실험, `/work/repo`는 `08f3e2b` 동결 유지)

**compile 원인**: HF **gradient checkpointing + torch.compile** → AOTAutograd backward 파티셔너 단언(`Node add_316 was invalid, but is output`). 이분법(3117481–84): 디코더만 compile ❌ · 인코더만 ✅ · 전체+ckpt OFF ✅ · mask 수정 무관. **non-reentrant ckpt도 ❌**(3117505). **torch 2.10(25.11)도 ❌**(3117585, gh109). 인코더는 문제 없음.

**compile 이득 (ckpt OFF, 정상상태 30-step)**: 30 kb 17.7 s/step(eager 24.9, **1.41×**) · 61 kb 34.7(45.5, 1.31×) · **2 GPU 30 kb 10.35 s/step**(1-GPU compiled 대비 1.71×, eager 대비 2.4×, 유효배치 96 유지, world_size 2).
**그러나 129 kb는 ckpt 없이 OOM** (3117556: A35 fail-closed, sample 416, `Athaliana:Chr4:15869550-15998574-rc`, 129,024 nt / 3,672 label tokens, allocated 100.65 GB; ckpt ON 벤치는 같은 티어 완주 101.0 GB). → **checkpointing은 레시피 필수 → 전체 compile은 120 GB에서 불가.** 위 배율은 레시피에 쓸 수 없는 숫자. (레이블 캡 축소는 A27 변경 = 저자.)

**DDP**: `2021fd2` per-rank accumulation = 96 ÷ world_size(2 GPU에서 48 확인, 유효배치 보존), `af6cce8` 결함 2건 재발 없음. 새 층: **reentrant ckpt + DDP = `Expected to mark a variable ready only once`**(3117555). non-reentrant + DDP는 gh093에서 launch failure(노드 의심) → 3117590이 깨끗한 노드 재시험.

**노드**: `unspecified launch failure` 3건 = **gh062 ×2, gh093 ×1**, 다른 노드 0. 실험은 `--exclude=gh062,gh093`. 시드 링크가 그 오류로 죽으면 코드 아닌 노드 → exclude로 재제출. NCSA 보고는 저자 승인.
**정정**: "25.11이 GH200에서 죽는다"는 gh062 탓이었다 — 이미지는 정상 실행. 다만 단언은 그대로라 **이미지 25.06 유지**.

### ✏️ round 5 판독 (09-09 아침, 소켓 재개통 후)
- `3117589` **인코더만 compile + ckpt ON, 1 GPU** (gh034): COMPLETED, **26.6 s/step** — eager 24.9보다 **느리다**. 인코더 compile은 이득 없음 → **시드 123 재시작 근거 소멸.**
- `3117590` **eager + non-reentrant ckpt, 2 GPU DDP** (gh072): COMPLETED, per-rank acc 48(유효배치 96 보존), 오류 0, **17.57 s/step**, peak 100.6 GB → 1-GPU eager 대비 **1.42×**(스케일링 71 %). **레시피 충실 다중 GPU 경로 확보.**
- 남은 확인 1건: 129 kb를 DDP + non-reentrant로(round 6, `runs/round6_job.txt`, 2 GPU, exclude). 통과하면 확인 시드 456/789에 DDP 권고. ✏️ PARTIAL (09-09 낮): 통과(55.64 s/step); 권고는 4 GPU 결정 + 통합 시험 결과로 대체.

### 아침 결정표 (저자)
| 질문 | 근거 | 권고 |
|---|---|---|
| 시드 123 재시작? | compile은 레시피에 불가; 인코더-only는 **오히려 느림(26.6 vs 24.9)** | **아니오 (확정)** |
| 확인 시드 456/789에 DDP? | **3117590 성공** — 2 GPU 17.57 s/step(1.42×), ckpt 유지, 파라미터 동일; numerics 미세 차이 | ~~round 6(129 kb) 통과 시 **예** — 2 GPU 확정~~ ❌ SUPERSEDED (09-09 낮): 저자 결정 **4 GPU**; 4 GPU 스케일링은 벤치 `3118904/6/8`이 측정 중 |
| checkpointing을 non-reentrant로 기본 변경? | DDP 필수 조건; 3117590으로 실증 | ~~**예, 커밋 권고**~~ ✏️ **커밋됨 `5bc4df8`**(`resolve_checkpointing`; reentrant는 1 GPU에서 `TRANSGENIC_CKPT_REENTRANT=1`로만), 시드 123 체인에는 적용 금지(그대로) |
| 이미지 | 25.06 | 유지 |
| gh062/gh093 | 2/3, 1/1 실패 | 실험 exclude, NCSA 보고 여부 결정 |

### 커밋 (전부 푸시, dtai `/work/repo`엔 미반영 — 의도적)
`9f95bb7` compile scope/ckpt 스위치·mask device·REPO_DIR · `6f66a38` non-reentrant 스위치 · `2021fd2` acc÷world_size·bench 캐시 게이트. 실험 체크아웃 dtai `transgenic/repo_dev`(`2021fd2`, HFmodels 복사본).

### 오늘 밤 내 실수
`repo_dev`의 HFmodels를 **심볼릭 링크**로 만들어 컨테이너 안에서 안 보임 → 이분법 1차 4런 20초 사망(복사본으로 교체). 원격 감시자가 ssh 소켓에 묶여 있어 소켓이 죽자 round 5 보고를 놓침 — 잡 자체는 무사.

## ✏️ 갱신 (2026-09-08 후속 세션) — 아래 원본 블록은 **집행 완료됐다**

## 🚀 DeltaAI 발사대 (2026-09-08 저녁) — R4 해소, 첫 GPU 지출 직전에서 멈춰 있음

**접속.** `dtai-login.delta.ncsa.illinois.edu` (⚠️ `login.deltaai…`는 존재하지 않는 이름), 사용자 **`wcyim`**, 계정 **`bilv-dtai-gh`**,
파티션 **`ghx4`**(기본, MaxTime **48 h**), GH200 120 GB×4 / 288 CPU / 488 GB. 키 인증은 **NCSA 정책으로 불가**(Duo 필수).
우회: 사용자가 pronghorn **`login-0`**에서 `ssh -fN dtai`로 ControlMaster를 한 번 열어 두면(`~/.ssh/config` `Host dtai`,
소켓 `~/.ssh/cm-dtai`, 12 h 유지) 나머지는 TTY·Duo 없이 소켓에 얹혀 간다. **소켓이 죽었으면 사용자에게 다시 열어 달라고 할 것.**
```bash
ssh -O check dtai            # "Master running" 이어야 함
ssh dtai 'accounts; squeue -u $USER'
```
**예산** 999 GPU-h (초기 배정; 잡이 멈추기 전에 추가 가능 — 저자 확인 2026-09-08). `PriorityFlags=MAX_TRES`,
`TRESBillingWeights CPU=13.89 Mem=9.017/GB GPU=1000`, `DefMemPerGPU=113560 MB` → **`--mem`을 쓰지 말 것**(200g는 1 GPU를 1.8 GPU로 과금).

**DeltaAI 측 상태** (`/work/nvme/bilv/wcyim/`):
- `transgenic/` = 번들 29 GB, **MANIFEST 41/41 md5 일치**, symlink 0. `repo/` = git clone, **`d7c3a6d`**. `repo/HFmodels/` = 인코더 캐시 209 MB(아래).
- `containers/transgenic-ngc.sif` = NGC `pytorch:25.06-py3` 기반, aarch64, sha256 `3b242b33…`. 빌드 로그 `containers/build.log`.
- **preflight PASS (SIF 안, `a4d1ede`)** — db md5/sha256 · 273,148행 · **실제 토크나이즈** `<s> <empty> </s>` · **인코더 캐시 존재** · win_v3.
  ("7 symlink" NOTE는 `repo/HFmodels`의 HF 캐시 내부 상대 링크 — 오탐, 무해.)
- `sbatch --test-only`: train·bench × 1·4 GPU 전부 노드 배정됨.

**시험대 GB10** (`ssh gb10` = spark-d642, aarch64 + NVIDIA GB10 sm_121, Ubuntu 24.04, **apptainer 1.5.3**(PPA; GitHub 릴리스엔 arm64 자산이
없다 — apptainer#3712), docker도 있음, 3.2 TB). 같은 `transgenic.def`로 SIF를 빌드해(`~/gb10_scratch/containers/`) **DeltaAI 잡 명령을 그대로**
돌려 본다. NGC 25.06이 GB10에서 CUDA 동작함(`compute_120`). 번들 복사본 `~/transgenic_data/`(29 GB), repo `~/transgenic`.
⚠️ GB10은 `torch.compile` 불가(SM 공유메모리 101 KB) → 스모크는 `TRANSGENIC_NO_COMPILE=1`. pronghorn→gb10 키 접속 됨.

**GB10이 첫 GPU 잡 전에 잡아낸 것 — 전부 DeltaAI에서 step 0 사망이었을 결함** (커밋 `63c8ee4`·`ec35498`·`e962864`·`a4d1ede`, 전부 푸시):
1. 런타임 `pip install -e . --no-deps`가 SIF 안에서 실패(egg-info를 바인드 repo에 쓰려 함; 동시 잡 경합) → `No module named 'transgenic'`.
   → **`PYTHONPATH=/work/repo/src`**, 설치 없음. `src/transgenic/__init__.py`가 없는 네임스페이스 패키지(`__file__ None` 정상).
2. `import bitsandbytes` 무조건 — def는 의도적으로 미설치(B5는 AdamW). → 선택적 import, 8-bit 분기에서만 오류.
3. **preflight의 토크나이즈 검사가 한 번도 실행된 적 없었다** — 존재하지 않는 `TransgenicTokenizer`를 import, `except: pass`로 숨김.
   지금까지의 "PASS tokenizer"는 전부 소스 스캔. → `GFFTokenizer(vocab_version="v3")`로 실제 토크나이즈; 폴백 시 이유 출력.
4. triton `libcuda.so cannot found!` — `--nv`는 `/.singularity.d/libs`에 묶는데 이미지 ldconfig는 빈 `/usr/local/cuda/compat/lib`을 가리킴.
   → `TRITON_LIBCUDA_PATH=/.singularity.d/libs`.
5. `torch.compile` 폴백이 약속만 — Inductor는 첫 forward에서 죽어 try/except 밖. → `torch._dynamo.config.suppress_errors = True`.
6. 인코더 가중치: `train_HyenaTransgenic.py:14`가 `HF_HOME='./HFmodels'`를 직접 설정(def의 `/work/hf`는 사문). 잡마다 `/work/repo/HFmodels`에
   **다운로드**하게 됨(외부망 의존 + 6잡 경합). → GB10 캐시 209 MB를 gb10→pronghorn→dtai로 이식, `HF_HUB_OFFLINE=1`, preflight 게이트 추가.
   캐시는 두 레이아웃이 다 필요: `HFmodels/hub/models--…`(가중치·원격 코드, HF_HOME 경로) + `HFmodels/models--…`(토크나이저, `cache_dir` 경로).
   DeltaAI에서 오프라인 로드 실측: HyenaDNATokenizer + 6.6 M 파라미터, 18 s.

**Fable 독립 검토가 잡은 것 (커밋 `48e161a`, A42 v1.32)**: `wait`+`set -e`로 USR1에서 체인 사망(A28 무력화) · `--mem`이 MAX_TRES에서 1.8 GPU 과금 ·
`#SBATCH`는 변수 미확장 · `SLURM_GPUS_ON_NODE`(정수) ≠ `SLURM_GPUS_PER_NODE`(입력 문자열) · `$SCRATCH`·apptainer 모듈 없음 · 벤치가 `ctx_v2`(단일 유전자
문법, A40 DB와 불일치)를 재고 있었음 → win_v3 + `--benchmark-tier` · `cpu_tools.slurm`은 DeltaAI 불가(Delta 전용 표기). **A42 = win_v3가 B5 레시피 파일(A25 정오표).**

**다음 (순서대로)**
1. GB10 스모크 결과 확인 — **`~/transgenic_container/smoke5.log`** (5 step · tier 30720 · eager · `EXIT=` 줄로 종료 확인). 통과 시 tokens/s·peak GB가 DeltaAI 벤치 예비값.
   ✏️ smoke4(20 step)는 52분 돌고 **출력을 전부 잃었다** — ssh `timeout`이 끊기며 `tee`의 stdout이 깨진 파이프가 됐고 `tail -40`이 끝에서 흘려보낸 유일한 줄에 tee가 SIGPIPE로 죽었다.
   4.5 s/샘플(GB10 eager)이면 20 step은 2.4 h가 필요하므로 52분 종료는 **어딘가에서 죽은 것**이고 traceback은 사라졌다. 교훈: 긴 원격 잡의 출력은 절대 ssh 세션을 거치지 않게 — detached 스크립트가 파일에 직접 쓰고 파일을 폴링.
   ✏️ **`d7c3a6d`** — 내 `e962864`가 `import torch._dynamo`를 `train()` 안에 넣어 `torch`가 함수 전체의 지역변수가 되었고 168행 `torch.set_float32_matmul_precision`에서 `UnboundLocalError`. DeltaAI 모든 잡을 step 0에서 죽였을 회귀; 스모크가 24초에 잡았다. 모듈 최상위 import로 이동. dtai·gb10 repo `d7c3a6d`.
   스모크 5차 초반 관측: `Using: cuda (NVIDIA GB10, 131GB)` · 인코더 캐시에서 227/339 텐서 로드(exact 80, adapted 147, missing layers 8–11 = A34 부분 사전학습 그대로) · 414,338,689 params · LR 1141 warmup / 22,836 total · **~4.3–5.0 s/샘플**(30 kb, eager, GB10).
2. ✅ **첫 GPU 지출 — 저자 승인 2026-09-08 17:5x PDT, 제출 완료.** `bench_b5.slurm` 6런, **`STEPS=50`**(기본 300은 GB10 실측 272 s/step 기준으로 2 h 한도 초과):
   `3116412`(30720·1GPU) `3116413`(30720·4GPU) `3116414`(61440·1GPU) `3116415`(61440·4GPU) `3116416`(129024·1GPU) `3116417`(129024·4GPU).
   job id 목록: dtai `/work/nvme/bilv/wcyim/transgenic/runs/bench_jobs_*.txt`. 출력: `repo/b5-bench-<jobid>.out` (JSON 요약 한 줄 + 진행바; `tr '\r' '\n'` 후 `grep steps_measured`).
   ```bash
   ssh dtai 'sacct -j 3116412,3116413,3116414,3116415,3116416,3116417 --format=JobID%9,State%14,Elapsed%9,MaxRSS%10 -n | grep batch'
   ```
   GB10 스모크 예비값(30 kb, eager, batch 1): 272 s/step · peak 32.1 GB · 78.6 h/epoch — GH200+compile은 이보다 훨씬 빨라야 하고 그 배율이 벤치의 답이다.
   **✏️ 벤치 결과 (2026-09-08 저녁, GH200 · NGC 25.06 · eager — compile은 실패 후 폴백, 아래 참조)**
   | job | tier | GPU | 상태 | s/step | tok/s | h/epoch | peak GB |
   |---|---|---|---|---|---|---|---|
   | 3116412 | 30720 | 1 | COMPLETED 22:36 | 24.9 | 840 | **7.18** | 101.1 |
   | 3116414 | 61440 | 1 | COMPLETED 39:50 | 45.5 | 873 | **8.33** | 101.1 |
   | 3116416 | 129024 | 1 | COMPLETED 1:22:30 | 96.6 | 808 | **10.09** | 101.0 |
   | 3116413 | 30720 | 4 | **FAILED 0:33** — `run_config.json` 경합(rank 2 JSONDecodeError) | | | | |
   | 3116415 | 61440 | 4 | **FAILED 2:12** — DDP "parameters that were not used in producing loss" (idx 295–320·503–528·609–612) | | | | |
   | 3116417 | 129024 | 4 | PENDING — 시작 시 **수정된 코드(`af6cce8`)로 돈다**(repo 바인드) → DDP 수정의 첫 시험 | | | | |
   - `steps_measured`가 30인 것은 `benchmark_summary(warmup=20)`이 앞 20 step을 버리기 때문(50−20). 정상.
   - **peak 101 GB는 `max_memory_allocated`(실제 할당)이고 30 kb·61 kb에서 동일** → 메모리는 창 길이가 아니라 **8,192-토큰 디코더(A27)**가 정한다. micro-batch 1 × acc 96 고정, 헤드룸 19 GB.
   - **compile은 GH200에서도 실패**(`AssertionError: Node add_316 was invalid, but is output`, torch 2.8.0a0) → `suppress_errors`(e962864)로 eager 폴백. 이 플래그가 없었으면 6런 전부 첫 forward에서 사망. 시드도 같은 코드 → eager. pytorch #164186·#163877·#163572(2025-10 종료)와 같은 계열.
   - **DDP 버그 2건 수정 `af6cce8`**(푸시, dtai·gb10 반영): `find_unused_parameters=True` + `_write_json` 원자화(pid tmp + `os.replace`) + `run_config.json`은 LOCAL_RANK 0만. 죽은 파라미터 제거는 별건.
   - **NGC 25.11 시험**: SIF 빌드 성공(dtai `containers/transgenic-ngc-25.11.sif`, gb10 동일; 25.06 SIF 보존). torch 2.10.0a0 · CUDA 13.0 · transformers 4.57.6 · triton 3.5.0. import·preflight PASS **양쪽**. GB10 GPU 경로는 `CUDA error: unspecified launch failure`(sm_121+CUDA 13 — GB10 사실, GH200 증거 아님). **❌ GH200 판정(잡 3116740, 1:19 FAILED): 같은 `CUDA error: unspecified launch failure`**, inductor 경로 안에서 사망(compile은 걸렸고 컴파일된 커널 launch가 실패). GB10(eager)과 같은 오류 → 공통 인자는 **25.11 이미지(CUDA 13.0·triton 3.5) vs 호스트 드라이버/툴체인**(SIF의 `cuda/compat`가 비어 있어 포워드 호환 shim 없음). **25.11 기각, 25.06 eager 유지.** 시드 체인 전후로 이미지 변경 금지. 25.11 SIF는 디스크에 보존(재시도용). 컴파일은 비차단 후속 — 중간 태그(CUDA 12.9 계열, 25.07/25.08 릴리스노트 확인) 또는 DeltaAI 드라이버 갱신 뒤. `08f3e2b`: `SIF`·`OUTTAG` env 오버라이드 + 잡 출력 첫 줄에 `image:`/`IMAGE_INFO` 기록(#18 인용용).
   - ✏️ **1-GPU eager 사이징 (정정)**: `h/epoch`은 **티어별 행에 대한 값**이다(`--benchmark-tier`가 `fin-start==tier` 행만 고름; 세 티어는 서로 배타적인 train 행 집합 99,562+63,243+36,023). **전체 1 에폭 = 1038×24.9 + 659×45.5 + 376×96.6 = 92,153 s ≈ 25.6 h / GH200 1장.** 앞서 적은 "7~8 h"는 티어 하나의 값을 에폭으로 읽은 오독. patience 3으로 8~10 에폭이면 **시드 1개 ≈ 9~11일 ≈ 210~260 GPU-h**, 48 h MaxTime → A28 체인 5~6링크. 3시드 ≈ 650~800 GPU-h(999 대부분; 저자: 추가 배정 가능). **4-GPU DDP(≈6.5 h/epoch, 시드 ≈2.5일)가 벽시계를 정하는 레버** → `af6cce8` DDP 수정을 3116417이 시험 중. compile 불가(25.06 폴백·25.11 사망)라 이 숫자가 학습이 실제로 얻는 속도. **intent.md open question 1(자원 사이징) 닫힘 → `spec.md` 작성 가능.**
   ⚠️ 129 kb 티어가 GH200 한 장(120 GB)에 들어갈지 미지 — 그 런만 OOM으로 끝나면 그것이 측정값이다. 눈 감고 재시도하지 말고 마이크로배치/티어를 저자와 결정. `--time=02:00:00` 상한, 최악 30 GPU-h.
   ```bash
   ssh dtai 'cd /work/nvme/bilv/wcyim/transgenic/repo && for t in 30720 61440 129024; do sbatch --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; sbatch --gpus-per-node=4 --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; done'
   ```
   결과(tokens/s, s/step, peak GB, hours/epoch)를 **#18에 붙이고**, 그것으로 시드 사이징 → `spec.md`(open question 1 닫힘). 4-GPU tokens/s ÷ 1-GPU = DDP 스케일링 효율.
3. ✅ **`SEED=123` 시작 — 저자 승인 2026-09-08 밤, 1 GPU, 잡 `3116844`(체인 1링크), repo `08f3e2b`, 25.06 SIF, eager.** A28 체인은 48 h 15분 전 USR1로 `latest_state` 저장 후 자동 재제출(CHAIN_MAX 8; `af6cce8`의 `wait` 루프 수정으로 실제로 이어짐). `SYNC_DEST` 미설정(체크포인트는 DeltaAI `/work/nvme/bilv/wcyim/transgenic/runs/seed123/`에만). 예상: 에폭 25.6 h, 시드 9~11일, 5~6링크, ~210~260 GPU-h.
   **감시**: DeltaAI detached `seed123_monitor.sh`가 30분마다 `runs/seed123_monitor.log`에 큐·sacct·epoch 수·latest_state·마커·train.err 마지막 줄·예산을 append. **ssh로 48 h를 기다리지 말고 이 로그를 볼 것.**
   ```bash
   ssh dtai 'tail -12 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log; squeue -u $USER -n b5-seed'
   ```
   에폭마다: `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`.
   **❌ 4-GPU 시험 `3116417` 취소(비용 0), 자동 재제출 해제.** 이유: `accumulation_steps = 96 // batch_size`에 world-size 나눔이 없어 DDP N GPU면 **유효 배치 96×N** — 동결 레시피 위반. 4-GPU는 후속: acc ÷ world_size(96/N), `steps_per_epoch` world-size 인식, **2 GPU**로 시험(4보다 훨씬 빨리 잡힘), 확인 시드 456/789에 DDP 허용 여부는 저자 결정(reduction 순서로 수치 미세 차이). `af6cce8` DDP 수정은 그 경로에 여전히 필요.
   ⚠️ 오늘 세 번째로 `pgrep -f <패턴>`이 **내 자신의 원격 셸을 매치해 kill**했다(감시자 정리 중). 원격 프로세스는 **PID 파일로만** 다룰 것.
4. 미해결: `cpu_tools.slurm` account CHANGE_ME(Delta 전용) · preflight의 HF 캐시 symlink 오탐 정리 · GB10 pytest 22건은 전부 환경(외부 repo 경로·seqkit·root 캐시).


아래 "다음 세션이 바로 할 일" 1·2·4번은 **끝났다.** 3번(수용 필터)만 남았다.
`6150892`는 **죽었다** — 그 잡 번호로 `squeue`를 찍지 말 것.

**지금 돌고 있는 것**
```bash
squeue -j 6153195,6153196 -h -o "%.16i %.9T %.9M"
find ~/scratch/data/Transgenic/evidence/protein_alignments -name "chunk*.DONE" | wc -l   # /54
awk -F'\t' 'NR>1{gsub(/K$/,"",$8); k=$3"["$4"]"; if($8!="NA"&&$8+0>m[k])m[k]=$8+0; if($7+0>p[k])p[k]=$7+0} END{for(k in m) printf "%-18s %6.1f GiB  %5.1f%%\n", k, m[k]/1048576, 100*p[k]/2019181}' \
  ~/scratch/data/Transgenic/evidence/protein_alignments/logs/mem_curve_chunked.tsv | sort -k2,2nr | head
```
- `6153193` 분할 **COMPLETED**(5:33). 6청크 × 2,019,181(마지막 2,019,180), `evidence/protein/chunks_v1/MANIFEST.tsv`.
  검증 2중: 레코드 수 합계 = 12,115,085 **그리고** 청크를 순서대로 이어붙인 압축해제 스트림 md5 = 원본 md5.
- `6153195` 정렬 배열 **RUNNING**, 54태스크 중 24 동시(`%24`), `--mem=120G`, `cpu-s2-core-0` / 계정 `cpu-s2-pgl-0`.
- `6153196` 병합 배열 PENDING(`afterok` 의존).

**비청크 런 `6150892`의 최종 판정** — 취소 시점에 이미 9종 중 3종이 죽어 있었다.
`0 Athaliana` CANCELLED 55.4 GiB(75 % 지점) · `1 Bdistachyon` **OOM** 53.9 · `2 Gmax` **OOM** 55.2 ·
`3 Osativa` **OOM** 63.0 · `4~7` CANCELLED 27–39 GiB. 고아 `.tmp` 5개(17.4 GB) 삭제.

**분할 등가성은 측정됐다** (잡 `6152995`, `evidence/protein_alignments/pilot/split_equiv/REPORT.txt`).
3-arm: A=기존 1×100k, B=**같은 입력 재실행(비결정성 대조군)**, C=2×50k 이어붙임.
GFF 층위 643,907 레코드·**순서까지** 동일, `MP` id만 빼면 md5 동일. `protein_accept.py` 산출물도 전부 동일
(100,851 alignments / 78,847 accepted / 76,183 proteins). **A≡B가 miniprot `-t 32`의 바이트 재현성을 처음으로 확인해 줬다** —
대조군이 없었으면 몰랐을 것이다.

**병합은 재번호가 필수다.** 청크마다 `MP######`가 1부터 다시 시작한다(2분할 파일럿에서 46,833건 충돌).
`MP[0-9]+` 전역 치환 **금지** — `##PAF` 줄이 질의 **이름** 안에 `MP`를 갖고 있다(파일럿 청크에서 578줄).
9번 컬럼의 `ID=`·`Parent=`에만 건다.

**메모리에 대해 알아낸 것 (사이징에 쓸 것)**
- 게놈 크기를 따르지 않는다. 먼저 죽은 둘이 살아 있던 것보다 **작은** 게놈이다.
- **계단식이다.** Gmax가 29.9 GiB에서 몇 시간 평평하다가 0.35M 질의 만에 55.2로 뛰어 죽었다.
  **평지는 안전이 아니다.** 짧은 창에서 잰 기울기는 "평평하다"는 오답을 준다.
- 청크마다 난이도가 다르다. 청크 2는 **벼과에서만** 비싸고(Osativa·Bdistachyon 50–52 GiB, 쌍떡잎은 10),
  청크 4는 **종을 안 가리고** 비싸다. 축은 보존된 정렬 수(`-N 20` × 히트)로 보인다.
- 옛 파일럿이 왜 틀렸는지: 게놈만 1×→8.45×로 바꾸고 **질의 수를 10만에 고정**했다.
  게다가 그 10만은 청크 0 앞머리라 쉬운 구간이었다(11.9 GB). 지금 `Athaliana[4]`는 48.9만 질의에 53.8 GiB다.

**저자 결정 (2026-09-08)**: **`-N 20` 유지.** 메모리 축이 보존 정렬 수로 보여 이걸 낮추는 것이 가장 싼
레버지만, 결과가 바뀌고 A19.2 sweep이 그 값에서 검증된 것이라 **동결 파라미터를 지킨다.** 재론 금지.

**프로토콜**: `A41 / v1.31` 추가 — pronghorn(≠Delta) 기록 + 청크 분할·등가성·재번호 규칙.
§3–9와 A19.2 파라미터·수용 문턱은 불변. 커밋 `fc48700`, **푸시됨**(`origin/main`).

**✏️ 2026-09-08 밤 — ① 정렬 완주 ✅ (54/54, 실패 0) → 병합 9종 ✅ (mRNA 12.4–21.6 M/종, id 연속·고유, PROVENANCE.tsv 9행) → ② `protein_accept.py` 9종 ✅ 배열 `6154025` 완료(23:35 PDT) → ③ ✅ **#45·#44 CLOSED**(2026-09-09 00:4x PDT, 저자 승인 후 코멘트+종료) · **#18에 벤치 코멘트** 게시(1-GPU 3티어 · eager 사유 · peak 디코더 지배 · 36 h/epoch 실측 · 4-GPU 유효배치 문제).**
   **수용 결과** (`revision/results/evidence/protein_acceptance_v1.tsv` + `.PROVENANCE.txt`, 커밋됨): accepted 비율 Ppatens **74.7 %** · Sitalica 78.1 · Bdistachyon 78.4 · Sbicolor 79.0 · Osativa 79.7 · Athaliana 82.4 · Vvinifera 82.4 · Ptrichocarpa 84.1 · Gmax **84.4 %**. 패턴이 생물학적으로 맞다(이끼 최저 · 벼과 78–80 · 쌍떡잎 82–84). 2+ 생물종 지지 인트론 346k(Ath)–619k(Gmax)/종. 기각 1위는 coverage, 2위 frameshift.
**✏️ SEED=123 `3116844` RUNNING (gh015)** — 55분에 6,491/198,828 샘플, **1.52 it/s → 에폭 ≈ 36 h**(tqdm ETA). 벤치 합산 25.6 h보다 길다: 실제 에폭은 `tier_up_prob` 혼합 표본이고 eager 오버헤드가 다르다. → 48 h 링크 ≈ 1.3 에폭, 8~10 에폭 ≈ **12~15일 ≈ 290~360 GPU-h**. 첫 `epoch=` 줄은 ~36 h 뒤. 예산 995/999.
청크 하나가 죽으면 **그 청크만** 재제출하면 된다(더 잘게 쪼개거나 191/256 GB 노드에서 `--mem=180G`).


**돌고 있는 것부터 확인할 것.** 이 문서를 읽는 시점에 이미 끝났거나 더 죽었을 수 있다.

```bash
squeue -u $USER -n protein_align -o "%.12i %.8T %.11M"
sacct -j 6150892 --format=JobID%14,State,Elapsed,MaxRSS -n | grep batch
find ~/scratch/data/Transgenic/evidence/protein_alignments -maxdepth 2 -name DONE | wc -l
```

**SLURM `6150892_[0-8]`** = OrthoDB 단백질 → 9종 miniprot 정렬(#45). 2026-09-08 08:0x 기준
5개 RUNNING · 2개 PENDING · **2개 OUT_OF_MEMORY(task 1 Bdistachyon 53.9 GB, task 3 Osativa 63.0 GB)**,
DONE 마커 0. 종 순서는 `SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor
Sitalica Vvinifera)`, 배열 인덱스가 그대로 이 순서다.

### 다음 세션이 바로 할 일 (순서대로)

1. **분할 등가성 측정** ← 여기서 시작. 파일럿 10만개를 `1×100k` vs `2×50k`로 miniprot에 돌리고
   `protein_accept.py` 산출물이 `alignment_id`(`MP######`)를 빼면 동일한지 대조한다.
   질의가 서로 독립이라 같아야 하지만 **그것이 가정이고, 가정을 사실로 적은 것이 이번 OOM의 원인이다.**
   근거 파일: `evidence/protein_alignments/pilot/Athaliana_100000.gff`
2. 등가 확인되면 죽은 둘(및 추가로 죽는 것)을 **청크 + 상향 메모리**로 재제출.
   ⚠️ `lh3/miniprot` **#68**은 925 GB가 있는 서버에서도 죽었고 수정이 없다 — **메모리 상향만으로는 부족할 수 있다**
3. 정렬 완료 후 **A19.2 수용 필터 9종 실행** — `transgenic/revision/scripts/evidence/protein_accept.py`
   (완성·검증됨, 게놈당 수 분). 그러면 **#45**, 그리고 사실상 끝나 있는 **#44** 정리
4. 🔴 **§12 프로토콜 로그** — 이 정렬이 A19·#45가 적은 **Delta가 아니라 pronghorn**에서 돈다는 것.
   드라이버 주석·PROVENANCE.txt·PROVENANCE.tsv에는 적었으나 **프로토콜 본문에 없다. 첫 결과표 이전에**

### 만들어 둔 것 (재작성 금지)

| 파일 | 상태 |
|---|---|
| `transgenic/deploy/deltaai/preflight_b5.sh` | 커밋 `4ef3dc4`. 첫 시드 전 게이트, **PREFLIGHT PASS** 실측 |
| `transgenic/revision/scripts/evidence/protein_align.sbatch` | ✏️ **커밋됨 `228620e`**(이 표의 "미커밋"은 틀렸다). 9종 배열 드라이버. **비청크판이라 더는 쓰지 않는다** |
| `transgenic/revision/scripts/evidence/protein_accept.py` | ✏️ **커밋됨 `228620e`**. A19.2 수용 필터. 파일럿 검증 완료 |
| `transgenic/revision/scripts/evidence/protein_chunk_split.sbatch` | 커밋 `fc48700`. 6청크 분할 + 2중 검증 |
| `transgenic/revision/scripts/evidence/protein_align_chunked.sbatch` | 커밋 `fc48700`. 청크당 1태스크(종=id/6, 청크=id%6), `--mem=120G` |
| `transgenic/revision/scripts/evidence/protein_merge.sbatch` | 커밋 `fc48700`. id 재번호 + 병합 검증 |
| `intent.md` | 플레이북 Plan 산출물. readiness gate R1–R6 |
| `training_input/` 29 GB | A40 DB + genomes 실체화 완료, 링크 0. **ACCESS 전송 준비 끝** |

✏️ **이 경고는 해소됐다.** 둘 다 `228620e`에 있었고(이 문서가 틀렸다), 청크 드라이버 3개는 `fc48700`으로
커밋·푸시됐다. `origin/main`은 `fc48700`.

### ACCESS 상태 한 줄

**R4(계정)만 남았다.** 나머지는 전부 해소·검증됨. 계정이 나오면 `intent.md`의 "다음 산출물" 절 순서대로:
`build.sh` → 번들 rsync(`-L` 불필요) → `CHANGE_ME` 두 줄 → `bench_b5.slurm` → `sbatch ... SEED=123`.



## ▶️ 현재 상태 스냅샷 (2026-09-08 — ACCESS 준비 완료, 단백질 정렬 중 OOM 2건)

### 🚨 내 사이징이 틀렸다 — 단백질 정렬 OOM 2건

`6150892_[0-8]`에서 **task 1 Bdistachyon(11:10에 53.9 GB)**·**task 3 Osativa(8:51에 63.0 GB)**가
64 GB 한도에서 `OUT_OF_MEMORY`로 죽었다. 원인은 miniprot이 아니라 **내가 사이징한 방식**이다.

드라이버 주석에 이렇게 적었다: *"Memory is dominated by the index and the streamed batch,
**not by the query count**"*. **측정한 적 없는 것을 사실처럼 적었고 틀렸다.**

2점 파일럿은 게놈을 1× → 8.45×로 바꾸면서 **단백질 수를 10만으로 고정**했다. 즉 **실제로 죽인
축(질의 수)을 볼 수 없는 설계**였다. 두 축 중 하나만 재고 나머지가 평평하다고 가정한 것 —
장독 OOM 때 EST 실적을 외삽한 것과 **구조가 같다**(`quarantine.md` §2f). 그 교훈을 인용하면서
같은 형태를 반복했다.

메모리는 게놈 크기를 따르지도 않는다. 죽은 둘(275 MB·382 MB)은 **아직 살아 있는 Gmax(1,024 MB)보다
작은** 게놈이다. `miniprot.err`를 보면 11시간에 걸쳐 **단조 증가**했다.

**알려진 이슈 검색 결과 — 같은 사례 있음, 수정 없음.** `lh3/miniprot` **#68 `[morecore] insufficient
memory`**가 같은 서명이다(동일한 `mapped N sequences` 진행 로그, `--gff`, 대용량 단백질 세트 489만 —
우리는 1,212만). 그쪽은 **925 GB가 가용한 서버에서** 죽었고(miniprot이 실제로 잡은 양은 그 스레드에 없다. `-t 50`이라 스레드 수도 우리와 다르다) 저자 답변은 `"Yeah, probably due to insufficient
memory"`로 끝난다. #25·#71도 같은 결론이고 우회로 제시가 없다. **메모리 상향만으로는 해결이 보장되지
않는다.**

**fail-closed는 설계대로 작동했다.** `PIPESTATUS`가 stage 0의 exit 137을 잡아 거부했고 `.tmp`를 지웠고
**DONE을 쓰지 않았다.** 죽은 두 디렉토리에는 인덱스만 남아 있고 잘린 GFF는 없다 — 재실행이 건너뛰지 않는다.

### 정렬 진행 상황 (2026-09-08 08:0x)

| task | 종 | 상태 | 경과 / 추정 |
|---|---|---|---|
| 0 | Athaliana | RUNNING | 12:00 / ~20 h |
| 2 | Gmax | RUNNING | 12:00 / ~33 h |
| 4 | Ppatens | RUNNING | 12:00 / ~28 h |
| 5 | Ptrichocarpa | RUNNING | 3:09 / ~27 h |
| 6 | Sbicolor | RUNNING | 0:51 / ~31 h |
| 7·8 | Sitalica·Vvinifera | PENDING | 자원 대기 |
| 1 | Bdistachyon | **OOM** | 53.9 GB |
| 3 | Osativa | **OOM** | 63.0 GB |

DONE 마커 **0개**. ⚠️ 살아 있는 것들도 같은 단조 증가를 겪고 있으므로 **완주를 장담할 수 없다.**

### ACCESS readiness gate — R4 빼고 전부 해소

`intent.md`의 R1–R6 중 **R1·R2·R3·R5 완료**, 남은 것은 **R4(계정)·R6(벤치마크)**로 둘 다 계정이 있어야 한다.

- **R1** 번들 DB가 폐기 코퍼스(`b5_full_v1`)였다 → A40 객체로 교체, md5 `dda7e78c…` 대조. 구 DB는
  삭제가 아니라 `RETIRED_DO_NOT_USE/training_input_db_b5_full_v1_20260907/`로 경로 격리
- **R2** genomes 9개가 밖을 가리키는 심볼릭 링크였다 → **번들 안에서 실체화**(링크 0, md5 9/9 일치).
  `rsync -L`을 안 쓴 이유는 그것이 **전송 명령을 정본으로 만들고** 옵션을 빠뜨린 결과가 ACCESS 쪽에서야
  드러나기 때문이다
- **R3** `train_b5.slurm`이 `--db /work/db/b5_v1.duckdb`로 **폐기된 이름을 하드코딩**하고 있었다 → A40 지향
- **R5** `deploy/deltaai/preflight_b5.sh` 신설, **PREFLIGHT PASS**. 기대값을 freeze JSON에서 읽는다

### 오늘 정리한 것

- **todo.md 낡은 줄 3건** — `#59`(실제로는 `1413efd`로 커밋됨)·`#57`(`28cbaa5`로 구현됨)·
  §2c A18.8(`12a89fe`의 `longread_fetch_v2.sh`로 해소, "매핑 이후로 연기" 조건은 이미 충족).
  §4는 "승인 나면 바로" 목록이라 여기가 낡으면 **남은 일이 실제보다 많아 보인다**. 머리말에 규칙 추가:
  **항목을 닫을 때 커밋 해시를 함께 적는다**
- **위키** `b2dca27`(update 33 + README + log + index) · `4fa2fd6`(`guide/pronghorn.md` glibc 절 항목 (i))
- **신규 이슈** #69·#70·#71 (병합 게이트 강화)

### 다음 작업

- **분할이 결과를 바꾸지 않는지 먼저 측정** — #68에 수정이 없으므로 우회는 단백질 FASTA 분할·병합이다.
  질의가 독립이라 정렬은 같아야 하지만 **그게 가정이고, 오늘 가정을 사실로 적어서 이 사달이 났다.**
  파일럿 10만개를 `1×100k` vs `2×50k`로 돌려 수용 테이블이 `ID` 제외 동일한지 대조할 것
- 동일 확인 후 죽은 둘(및 추가로 죽는 것)을 청크+상향 메모리로 재제출
- 🔴 **§12 프로토콜 로그** — 이 정렬이 A19·#45가 적은 Delta가 아니라 pronghorn에서 돈다는 것.
  드라이버·PROVENANCE에는 적었으나 프로토콜 본문에 없다. **첫 결과표 이전에**

## ▶️ 이전 스냅샷 (2026-09-07 — A40 커밋 완료, 매핑 실측 확인, intent.md 신설)

### A40 커밋 — 사전 검토에서 진짜 결함 1건

커밋 전 Codex(Astra)에 독립 검토를 걸고 실측으로 중재했다. Astra는 블로커 5건을 냈고 그중 **확인된 것은 1건**이다.

**진짜 결함**: `test_status_completion_fails_closed`가 8개 fault를 파라미터로 돌리면서 `validate_b5_database`를 **전 케이스에 `ok:False`로 monkeypatch** 하고 있었다. 그러면 검증 이전에 걸려야 하는 7건은 **해당 가드를 지워도 통과한다** — 죽는 주체가 가드가 아니라 주입된 validation이기 때문이다. 변이 실험으로 확증했다: `wrong_db` 가드를 삭제하니 옛 테스트는 그대로 통과했고, 새 테스트는 `FAILED [wrong_db]`를 내며 stdout에 두 종 모두 `completion evidence and fresh validation PASS`가 찍혔다. 고친 뒤 **567 passed, 6 skipped**.

**기각한 것**: Astra의 "freeze 클레임 검토 불가"·"수치 근거 없음" 3건은 **내가 검토용으로 준 파일이 부분집합이었던 탓**이다. `b5_full_a40_v1.freeze.json` 실측 결과 `verification.freeze_time_checks`에 `merged_file_hash_matches_receipt`·`all_table_hashes_match_receipt`·`nine_source_hashes_match_receipt`가 모두 true고, 프로토콜 표의 `8,181`/`978,586`/`8,130`/`978,864`도 이 파일 안에 있다. freeze JSON의 `"adopted": false`는 결함이 아니라 `verification.pre_adoption_results` 안에 **격리 보존된 이력**이고, 채택은 별도 `authorization` 필드에 있다.

**강화 항목 3건은 이슈로 뺐다** — **#69**(`--build-status`가 opt-in이라 생략하면 `.DONE` 신뢰가 조용히 복구됨) · **#70**(수신 JSON 스키마 미검증 → `KeyError`/`FileNotFoundError`로 죽어 무엇이 틀렸는지 안 말함) · **#71**(rejection·manifest를 두 번 해시하면서 동일성 미검사 → freeze가 **검증한 적 없는 해시**를 기록).

### 매핑 — 마커가 아니라 산출물로 확인했다

2026-09-04에 "`.DONE` 122개가 수집 완료를 뜻하지 않았다"를 겪었으므로 이번엔 개수를 셌다.

| 대상 | DONE | 산출물 | PROVENANCE | 빈 파일 | 크기 |
|---|---|---|---|---|---|
| 장독 `longread_align_v1` | 129 | BAM 129 · BAI 129 · PAF 258 | 129 | **0** | 481 GB |
| EST 정본 `est_align` (A37 하한 100) | 9 | BAM 9 | 9 | **0** | 1.7 GB |
| EST 민감도 arm `est_align_min121` | 9 | BAM 9 | 9 | **0** | 1.7 GB |

네 신호(마커·BAM·인덱스·PROVENANCE)가 서로 일치하고 0바이트 산출물이 없다. **매핑은 끝났다.**

⚠️ 단 `evidence/protein_alignments/`는 **비어 있다**(1.0 KB, 0개). A19 단백질 자원은 §1에 등재돼 있으나 정렬 산출물은 없다 — 이게 계획대로인지(자원은 모델 입력이지 정렬 대상이 아님) 누락인지는 확인된 바 없다.

### intent.md 신설

[The AI-Native SDLC Playbook](https://claude.com/blog/the-ai-native-sdlc-playbook)의 Plan 단계 산출물을 도입했다. [`intent.md`](intent.md) — 대상은 **A40 코퍼스로 B5 학습 런 시작**. Problem / Proposed outcome / Affected users and systems / Constraints / Open questions 구조에 이 프로젝트의 마커 규약을 얹었다. 핸드오프 파일은 이제 넷이다 — `intent.md`(왜) · `todo.md`(다음) · `resume.md`(지금) · `quarantine.md`(안 씀).

`spec.md`·`plan.md`는 **일부러 쓰지 않았다**: open question 1(ACCESS 자원)·2(#61 test set 정의)가 열린 채로 spec을 쓰면 자원 가정을 지어내게 된다.

### 다음 작업

- **푸시 여부 저자 결정** — `f6e68d4`·`5289077` 미푸시. 푸시해야 #55·#56·#58이 닫힌다
- open question 1·2가 닫히면 `spec.md` → `plan.md` → 잡 제출
- #69·#70·#71은 다음 병합 전까지. 지금 동결된 객체에는 영향 없음

## ▶️ 이전 스냅샷 (2026-09-06 — A40 / v1.30 동결·§1 채택 완료)

저자가 **2026-09-05 결정한 #55·#56·#58은 모두 해결**: no-CDS는 마스킹·overlap closure에 참여하되 coding label 없음, empty target은 `<s> <empty> </s>` 3토큰, emitter/validator는 canonical emitted-block 순서를 공유한다. 저자 동결 승인과 독립 read-only 대조 후 새 §1 객체로 채택했다.

**[A40 검증 노트](transgenic/notes/a40_verification_20260906.md)**: 273,148 /272,224 (+924), 미설명 변화 0. 334,642 assignments, train 213,702 /valid 29,153 /test 91,787, strict 3,430 모두 불변. 9종 replay·source validation, merged validation, 273,148 token/RC 검사 PASS. 최종 코퍼스에 남은 순서 복구는 **19 forward +19 RC**이며 **97이 아니다**. 97+63은 old-population counterfactual, #56 net +764와 합쳐 +924다.

**다음 작업**: 저자 diff 검토·커밋. B5는 새 §1 DB + 수정 tokenizer + **새 labelled run**으로 시작 가능(A28: 기존 chain 안 패치 금지); ACCESS 실행 전제는 유지. #61 scoring-universe 결정은 별도 미결. 이전 DB·9종 source·기존 freeze·480 GB 정렬 결과는 보존. 이 아래 스냅샷의 #55/#56/#58 결정 대기·빌드 감시 지시는 과거 상태이며 이 기록이 대체한다.

## ▶️ 이전 스냅샷 (2026-09-05 22:10 — 여기서 다시 시작)

### 테스트 경합 확인 — 단독 실행 44/44 통과

전체 스위트에서 4건 실패했던 테스트를 격리해서 단독 실행하니 **44/44 전부 통과**(78초). 실행 시간이 55초 → 410초로 늘어난 것은 GPU 재빌드와의 자원 경합 신호였습니다.

커밋 `aa2623f` 푸시 완료. 빌드 상태: 종 DB 3/9개, 빌더 프로세스 생존.

**교훈**: 실패율만 읽었다면 Astra를 호출해 없는 버그를 찾게 했을 것. 시간 증가가 경합의 지표였고 단독 재실행이 1분 만에 판정을 냈습니다.

### 다음 작업

- 빌드 감시 계속 (완료·실패 어느 쪽이든 통보)
- 실패 시 Astra 재호출

## ▶️ 이전 스냅샷 (2026-09-05 21:55)

### Astra 사전 점검 27건 실패 — 진짜 버그 1건 발견·고침

**문제:** Astra 격리 스냅샷 회귀에서 27건 실패로 멈춤. 분류 결과:
- **5건 진짜 버그**(내 코드): `test_make_training_input_manifest.py` — 스테이징 게이트의 md5 검사가 `/^[[:xdigit:]]{32}$/` 사용(`f5a257f`)
  - pronghorn mawk 1.3.4: `{32}` 반복 지원 → MATCH
  - gpu 구형 awk: 지원 안 함 → NO MATCH, 모든 md5 "invalid" 판정
- **8건 환경**: test_longread_fetch.py — gpu에 seqkit 없음
- **14건 환경**: test_run_polishing_benchmark.py — 형제 트리 부재

**조치:** md5 검사를 `length($5) != 32 || $5 !~ /^[[:xdigit:]]+$/`로 변경(두 awk 호환). 회귀 테스트 추가(`test_awk_programs_avoid_interval_expressions`, 위반 감지 확인). 커밋 `53ccecd`(정정됨).

**재호출:** Astra를 다시 부름. 빌드 관련 테스트만 조건으로, 환경 실패는 사유 기록하고 진행.

### 다음 작업

- Astra 빌드 감시 (사전 점검 통과 대기)
- 통과 확인 시 증거 드라이버 검증 및 다음 단계 진행

## ▶️ 이전 스냅샷 (2026-09-05 18:18)

### 장독 정렬 129/129 완료 — 메모리·방향성 수정

**완료:** 129/129 정렬, 0 실패. 538,393,032 레코드, 89.01% 평균 매핑률(Gmax 97.52%, Osativa 84.52%, 기타 84–93%). 480 GB(BAM+PAF).

정렬 후 미뤄 뒀던 두 수정(`4d89227`):
- `#SBATCH --mem=32G` → 96G (실측 최대 29.5 GB, 32 GB 천장에서 2건 OOM)
- `orientation` 줄을 grep 가능한 단일 토큰으로 통일. 기존 129개 PROVENANCE는 재생성하지 않음(480 GB 재정렬 필요).

**저자 결정 3건 대기** (`transgenic/notes/author_decisions_20260905.md`):
- #58: 공백 3토큰(권장) vs 2토큰 유지 — A: 재빌드 불필요 / B: 필요
- #56: Vvinifera lncRNA 6,205개 마스킹 — 재빌드 필요
- #55: 타일 97건 복구 여부 — 재빌드 필요

### 다음 작업

- 세 가지 저자 결정 논의·확정
- 필요시 재빌드 한 번 실행(3건 함께)

## ▶️ 이전 스냅샷 (2026-09-05 16:18)

### longread_fetch_v2 완료 — 낡은 호출자 4형태 exit=6으로 차단

`longread_fetch_v2` 완료(`12a89fe`). v2는 호출자 이관(`'OXFORD_NANOPORE'` → `PLATFORM_RE=`, `'maize'` → `SUBMITTED_RE=`)을 정확히 처리하고, 파이프 처리 오류를 바로잡아 종료코드를 제대로 반환하도록 수정. 테스트 44개 추가, 전체 **543 passed**.

낡은 4가지 인자 형태(`SRR10611193[[:space:]]`, `'OXFORD_NANOPORE'`, `'PACBIO'`, `'maize'`)를 직접 시험한 결과 모두 `exit=6`으로 정확히 차단됨 확인. 특히 `'maize'`처럼 휴리스틱이 놓치는 형태도 뒤의 빈 선택 가드가 잡아냄.

실수로 다운로드된 201 MB 테스트 잔재 정리, `evidence/ont/` 원래 4개 디렉토리로 복구. 커밋 19개, 모두 푸시됨.

**해석**: Astra 목록의 저자 결정이 불필요한 항목(`longread_fetch_v2`, protocol v1.29, B5 데이터셋)은 이제 완전히 동결됨. A18.8이 "파싱된 컬럼으로 필터하라"는 조항과 호출자 코드 실제 동작이 불일치하는 구조는 가드로 보완되었음.

**남은 것**: `longread_align.sbatch`의 orientation 줄 정리 + `--mem` 상향(A18.8). 배열 119/129 진행 중이므로 배열 완료 후 편집 가능. 저자 결정 대기 항목 #58, #56, #55는 별도 추진 필요.

### 다음 작업

- 배열 117·118·119 모니터링 (117은 33분째 진행, 118·119는 96GB 재실행 중)
- 배열 전체 완료 후 `longread_align.sbatch` 편집
- `author_decisions_20260905.md` 검토하여 #58·#56·#55 결정 진행

## ▶️ 이전 스냅샷 (2026-09-05 16:04)

### 태스크 118·119 OOM 재제출 완료

태스크 118 (Vvinifera `SRR14635690`, 입력 2.71 GB)도 32 GB 한도에서 `OUT_OF_MEMORY`. 119와 동일 원인이므로 96 GB로 재제출(`6149512`) — 대기 중. 119는 이미 96 GB로 재제출된 `6149508` 실행 중.

### 진행 상황

현재 **118/129 완료**. OOM 처리로 진행이 일시 멈춤. 남은 런:
- 태스크 117 (Sitalica 2.28 GB): 33분째 실행 중 — 입력 크기는 안전권이나 진행 모니터링 필요
- 태스크 121–128 (0.62–0.77 GB): 위험 구간 외

### OOM 패턴 확정

두 실패 모두 입력 **≥2.7 GB + Vvinifera T2T 게놈 조합**. 작은 런(0.6–2.3 GB)은 32 GB에서 무탈히 통과.

### 감시 재구성

원 배열 `6148978` + 재제출 `6149508`, `6149512` 세 잡 전체를 감시. 새 실패만 알림(118·119 반복 보고 없음). 도귀 시까지 남은 11개 완료될 것으로 예상.

### 다음 작업

- 117·118·119 모니터링 (117 통과/실패, 118·119 96 GB 재결과)
- 모두 완료 후 저자 결정 대기 항목(#58, #56, #55) 진행

## ▶️ 이전 스냅샷 (2026-09-05 16:01)

### 장독 정렬 119번 OOM — 재제출 완료, 진행 117/129

태스크 119 (Vvinifera `callus_PRJNA732451`, 입력 3.14 GB)가 `.batch` MaxRSS **29.34 GB**로 32 GB 한도에 부딪혀 중단. **이건 `.extern` 함정이 아닙니다** — `.batch` 스텝 자체가 `OUT_OF_MEMORY`. 드라이버의 fail-closed가 정상 작동해 DONE과 BAM 미생성, 잔재 정리 후 96 GB로 재제출(`6149508`), 현재 `cpu-51`에서 실행 중.

### 배열 전체가 32 GB 천장에 붙어 있었다

완료된 태스크들도 위험 상태였습니다 — 태스크 8이 **29.50 GB**, 49가 29.25 GB로 **겨우 살아남았고** 119는 29.34 GB에서 죽었습니다. 장독은 실제로 29.5 GB를 소비하는데, 메모리 산정이 **짧은 리드 EST 실적(10.9 GB + sort 8 GB ≈ 19 GB)을 장독에 외삽한 착오**였습니다. 원래 제 계획 48 GB였으면 무탈했습니다.

### 기록·차후 처리

- `todo.md` §2f에 기록
- 배열 완료 후 SBATCH `--mem` **64 GB 이상**으로 올릴 것 (현재는 A18.8 진행 중이라 편집 불가)
- 진행: **117/129** 완료, 새로운 실패 0건

### 다음 작업

- 배열 나머지 런 대기 (대부분 작은 크기, 무난 예상)
- 119 재제출 모니터링 (96 GB에서 통과 여부)
- 완료 후 저자 결정 대기 항목 3가지(#58, #56, #55) 진행

## ▶️ 이전 스냅샷 (2026-09-05 12:44)

### 장독 정렬 진행 108→109/129

정렬이 한 run 진행했습니다 (108/129 → 109/129). 커밋 `6ebe904..dc3346c` 푸시 완료, origin/main 동기, 미커밋 0. 테스트 **499 passed, 6 skipped**(세션 시작 412 → 499 증가).

드라이버 `longread_fetch_v2`와 orientation 줄 정리(A18.8)가 실행 중이므로 둘 모두 대기 상태. 정렬이 129/129 완료되면 이 두 항목을 Astra에 넘기고 매핑 최종 보고.

### 다음 작업

- 정렬 완료 대기 (현재 109/129)
- 완료 후 저자 결정 대기 항목 세 가지(#58, #56, #55) 진행 가능
- 매핑 최종 보고 예정

## ▶️ 이전 스냅샷 (2026-09-05 12:32)

### #55·#61·#54 완료 — 코퍼스 측정 확정

Astra가 세 이슈를 측정했고 제 전제 하나를 반박했습니다. 97건 타일 드롭이 모두 "동일 span 타이 break"라고 했지만, 실측은 **타이 13건 + 감소 span 84건**입니다. 타이만 복구했다면 84건을 놓쳤을 것입니다. 커밋 5개, 테스트 **412 → 499 passed**.

| 이슈 | 결과 | 재현 |
|---|---|---|
| #24 | 증거 채점기 쌍대 보고 | 프로토콜 애매함 2건 확인 |
| #59 | 재개 순열 버그 | GPU 6 passed |
| #57 | tier margin 빌더 | 3,418/653/86 일치 |
| #55 | 타일 순서 규칙 통일 | 타이 13 + 감소 84 |
| #61·#54 | 분모 확정·매핑 공백 | 15,005/91,787 = 16.3 % |

### 저자 결정 대기 (3항목)

`notes/author_decisions_20260905.md`에 선택지·비용·권장 정리:
- **#58**: `<empty>` 토큰 3(권장) vs 2 유지 — 재빌드 불필요 vs 필요
- **#56**: Vvinifera lncRNA 6,205개 마스킹 — 재빌드 필요
- **#55**: 타일 97건 복구 여부 — 재빌드 필요

셋 모두 정해 재빌드 1회(약 1.5시간). 미결정 시 코퍼스는 쓸 수 있으나 "프로토콜 준수 B5 실행"으로 미흡.

### 다음 작업

- 저자 결정 대기 (세 줄 충분)
- 드라이버 실행 중: `longread_fetch_v2`, orientation 줄 정리 (A18.8)
- 장독 정렬 **103/129** 진행 중 — 끝나면 위 둘 진행 후 완료 보고

## ▶️ 이전 스냅샷 (2026-09-05 11:22)

### #57 완료 — tier_margin_unguaranteed 빌더 구현

프로토콜에는 숫자로 있었지만 빌더 코드에는 없었습니다. 사람이 손으로 계산했기 때문에 테스트는 통과했고 아무도 구현 누락을 발견하지 못했습니다. 커밋 `28cbaa5`, 신규 테스트 2개.

독립 재현 확인: 고정 DB에서 직접 세어 **3,418 / 653 / 86**과 최장 유전자 **196,414 nt**가 프로토콜 수치와 정확히 일치합니다. 로컬 테스트 **489 passed**.

### 결정 대기 항목 명확화

Astra가 `notes/author_decisions_20260905.md`에 정리했습니다.

- **#58**: `<empty>` 토큰 계약 (25,159 타일). 계약 3토큰 vs 트레이너 2토큰 — 토크나이저 수정 vs 스펙 개정 두 선택지
- **#56**: no-CDS 유전자 누출 (Vvinifera 6,205 중 1,605가 valid/test) — 마스킹 우회 문제

아무것도 안 하면 "프로토콜 준수 B5" 기준을 충족하지 못합니다.

### 다음 작업

- #58·#56 저자 결정 대기 (선택지·비용·개정안 모두 notes에 기록)
- 지시 없으면 Astra 권장대로 #55 → #61 → #54 계속 진행
- 장독 정렬 자율 진행 중 (77/129 이상)

## ▶️ 이전 스냅샷 (2026-09-05 10:23)

### #59 완료 — resumed epoch 순열 버그 고정

원인은 두 겹이었습니다: 샘플러가 seedable이 아니라 RNG 상태에 의존 · 새 프로세스는 로더를 에폭 0에서 시작해서 재개 에폭이 epoch 0 순열을 재생한 뒤 엉뚱한 에폭의 N개를 건너뜀. 커밋 `1413efd`, 신규 테스트 6개.

검증 완료: GPU에서 6 passed(8.79초, Astra 보고와 일치). 음성 대조(`seedable=False`)로 원래 버그를 실제 재현(`resumed != original`, 합집합 < 전체) — 테스트가 버그를 잡을 수 있음을 증명합니다.

### Astra 우선순위 결정 · 저자 결정 필요 항목 격리

| 순서 | 항목 | 상태 |
|---|---|---|
| ✅ 59 | resumed epoch | 학습 블로커 완료 |
| ⏸ 58 | `<empty>` 토큰 | **저자 결정 필요** |
| ⏸ 56 | lncRNA 6,205 | **저자 결정 필요** |
| ▶ 57 | `tier_margin_unguaranteed` | 다음 (검증 블로커) |

### 다음 작업

- **#57부터 이어가기**(#58·#56은 결정 대기 항목으로 기록)
- 정렬: 69/129 (약 24시간 내 예상)
- 돌아오실 때 결정 대기 목록 한 번에 검토하면 됩니다

## ▶️ 이전 스냅샷 (2026-09-05 10:12)

### #24 채점기 완료 · 쌍대 해석 결정 적용

Astra 두 라운드 완료. 첫 라운드에서 규칙 두 개(`chain_support` 쌍대 해석·비클리크 진단 기록)를 명시 거절했고, 저자가 프로토콜로 확인해 결정했습니다. 두 라운드에서 그 결정을 구현했습니다.

**검증**: 신규 테스트 43개, 전체 **472 passed**(회귀 없음). 코드는 § 4·6·8 필터·A39 gate 정확 구현, `-uf` 독립 재검증(감사 JSON과 DONE 마커 직접 대조) 추가. 저작 정당성: primary를 합성 픽스처로만 결정, 실제 데이터 전에 고정(`EVIDENCE_SCORER_READINGS.md` 기록).

**커밋**: `aa598f5`, 프로토콜 v1.29 유지, 저자 Won.

장독 정렬 **64/129** 진행(자율).

### 다음 작업

- Astra의 남은 블로커(#61·#59·#58·#57·#56·#55·#54 + `longread_fetch_v2`) 우선순위 결정 대기
- 구현 중 저자 판단 필요 항목은 적어 두고 진행
- 장독 정렬 모니터(예상 24시간 내)

## ▶️ 이전 스냅샷 (2026-09-05 10:01)

### #24 채점기 초안 완료 · 두 결정에서 의도적 정지

Codex Astra가 evidence scorer 초안을 완성했습니다. 커밋 `c998890`, 신규 테스트 32개 통과, 전체 **461 passed**(회귀 없음). 

지시 위반 0: 신규 파일만 3개, 추적 변경 0, `evidence/` 무손상, 프로토콜 무손상. 코드는 §4 필터(0x900·MAPQ 20·read name 그룹핑)·PAF 거부·A39 재검증(sense_fraction 독립 확인)을 정확히 구현했습니다.

원문 검증 완료한 두 애매함:
- **① 쌍대 해석**: primary `chain_support_single_read`(§6·§8 문언: 리드 1개면 지지), 민감도 `chain_support_threshold`(A18.2: ≥3 units). 양쪽이 T1–T6·모든 비율에 라벨과 함께 들어갑니다.
- **② 비클리크 정지 유지**, 진단 패스 추가: 실데이터 전에 정지 빈도를 예측합니다.

저자 결정을 Astra에 넘겼습니다(`astra_24b.md` 준비).

장독 정렬 62/129 진행, 233GB.

### 다음 작업

- Astra의 쌍대 구현·진단 패스 병합 대기
- 병합 후 테스트 검증
- 장독 정렬 모니터 (예상 수 시간)

## ▶️ 이전 스냅샷 (2026-09-05 08:21)

### 정렬 46/129 진행 중 · Athaliana 완료, Gmax 진행

Longread 정렬 자율 실행 중, 실패 없음. 완료분 통계: Athaliana **36/36 완료** (매핑률 93.20%), Gmax 10/27 진행 중 (매핑률 99.83%). 전체 primary 레코드 186,737,147건 중 177,423,645건 매핑 (**95.01 %**), `-uf` 플래그는 7런에 적용. 메모리 사용량 206GB.

보고서 생성 스크립트 작성·검증 완료 (`70_longread_align_report.py`, 커밋 `34345bb`). 미완 런 정확 추적, 완료분 통계 요약. 정렬 대기 중 Codex와의 협업으로 이전 회차 네 가지 결함 최종 수정(awk 따옴표·DONE 마커·staging 게이트·§3.3 `-uf` 의미론, 커밋 13개 추적).

### 다음 작업

- 정렬 완료까지 모니터링 (예상 24시간 내)
- 완료 시 보고서 실행 및 최종 통계 검증
- #24 채점기(§5 접합 호출 전제) 정렬과 병렬 시작 검토

## ▶️ 이전 스냅샷 (2026-09-05 04:12)

### 정렬 15/129 진행 중 · 완료분 16런 검증 완료

Longread 정렬이 자율 실행 중. 완료분 16런을 확인하니 감사 판정(`PASS`/`FAIL`)과 실제 명령(`-uf` 플래그 유무)이 **16/16 일치**. FAIL 런은 모두 `-uf` 없음(예상대로), 아직 PASS 런이 완료되지 않아 그쪽 명령은 미확인.

기록 표기법 발견사항: `PROVENANCE.txt`의 `orientation` 줄이 `-> no -uf`로 끝나서 grep 패턴으로는 켜진 경우와 구분 안 됨. 정보 자체는 `command` 줄과 `DONE`의 `uf=` 필드에 명확히 있어 복구 가능. **A18.8이 실행 중 드라이버 편집을 금지**하므로 지금 수정하지 않음(형식 불일치 악화). 정렬 완료 후 정리 대기(`todo.md` §2e에 기록).

### 일정 및 다음 작업

**예상 완료**: 11.3시간(내일 오전 중). 4동시 진행, 완료분 평균 1,258초 기준.

- 정렬 완료 또는 실패 시 보고
- **#24 채점기**(§5 접합 호출의 전제) — 정렬과 병렬로 시작 권장, 완료와 맞물림

## ▶️ 이전 스냅샷 (2026-09-05 01:19)

### 감사 104/129 진행 중 · A39 핵심 수정 완료

A39 의미론 재정의 완료. §3.3 항목 2는 프래그먼트 센스 비율(`read_sense_fraction`)로 가닥성을 판정하는데, 초기 구현은 주석과의 일치율(`fraction`)을 썼습니다. 통계를 정정하니 129런이 다섯 종으로 명확히 분리됨:

| 종 | 센스 가닥 | 비가닥 | 혼합/불명 |
|---|---|---|---|
| Athaliana | 7 | 26 | 3 |
| Gmax | — | 27 | — |
| Osativa | 28 | — | — |
| Ppatens | 7 | — | — |
| Ptrichocarpa | 3 | — | — |

Athaliana의 혼합 3런(`col0_DRP009401` 계열)을 **안티센스 가닥성**(센스=0.128–0.186)으로 재분류, 밴드 경계 상향(0.10 → 0.20). 라이브러리 프로토콜 차이가 런별로 깔끔하게 드러남 — §3.3 항목 4의 "런 단위 판정" 필요성을 그대로 입증.

보고서 생성기(`69_orientation_report.py`) 동작 확인, 프로토콜 v1.29 고정. 커밋 11개 추적.

### 다음 작업

- 감사 129/129 완료 대기 (현재 104, 오류 0)
- 완료 후 판정 표 최종 출력
- 장독 정렬 제출(`longread_align.sbatch` 32스레드 × 4, ~19시간)
- §5 접합 호출 — **채점기 #24 미구현으로 진행 불가**

## ▶️ 이전 스냅샷 (2026-09-05 01:11)

### 감사 6148698: A39 게이트, 진행 72/129 · 판정 구조 명확화

A39 의미론(읽기 센스 가닥 비율)으로 게이트 재작성. 장독 드라이버가 **감사 키 비교 버그**를 가지고 있었습니다: 4필드(데이터 3개 + 채점기 md5)로 비교하다 보니 감사 마커는 이전 3필드만 가져 **129런 전부 거부**. 테스트 중 검출·고정 (커밋 `6e0da95`).

| 종 | PASS | FAIL | UNRESOLVED |
|---|---|---|---|
| Athaliana | 7 | 26 | 3 |
| Gmax | 0 | **27** | 0 |
| Osativa | **9** | 0 | 0 |

**구조**: 종이 아니라 라이브러리 프로토콜 차이 — Gmax 27은 모두 비가닥(FAIL), Osativa는 센스 가닥성(PASS). 이것이 §3.3 항목 4가 **런 단위 판정**을 요구한 이유입니다.

**UNRESOLVED 3건** 모두 `seed_flcdna_PRJNA1155260`(Athaliana): MAPQ<20이 리드의 48–51%. 중앙값 2.7%인데 비교 데이터셋(`wm82_graft_PRJNA648759`, Gmax도 34–35%)과 이상치. 감사 종료 후 기록.

### 다음 작업

- 감사 6148698 완료 대기 (진행도 확인: 72/129)
- 완료 후 장독 드라이버 정렬 제출 (§5 접합 가닥 태그 의존)
- MAPQ 이상치 데이터셋 정리 기록

## ▶️ 이전 스냅샷 (2026-09-04 15:24)

### 감사 드라이버 버그 발견 및 교정

§3.3 항목 2의 지표 (`fraction` — minimap2 추론 가닥 vs 주석 가닥 일치율)가 **라이브러리 가닥성을 구분하지 못합니다.** 두 지표 모두 같은 전사체 바탕이라 가닥성 유무와 무관하게 ~99% 일치. 27/27 런 전부 95% 문턱을 통과했으므로 게이트가 작동하지 않음.

실제 지표는 `read_sense_fraction` (리드 자체가 센스 가닥인 비율):
- 비가닥 cDNA: 49–53%
- 센스 가닥성: 99%
- 안티센스 가닥성: 13–19%

A39/v1.29로 교정. 지표·문턱 불변. 판정: **27/27 PASS → 4/27 PASS** (더 엄격).

### 배열 재관리

- 낡은 27개 PASS가 현재 판정으로 오인되지 않도록 완료 마커에 채점기 md5 추가
- 데이터 같음 + 채점기만 다름 → 재채점 (재정렬 X, 보관 SAM 재사용)
- 데이터 다름 → 거부

배열 6148698 재제출됨 (A39 게이트, `-t 32` 병렬, 129 런).

### 다음 작업

- 감사 6148698 완료 대기 (현재 진행도 불명, 배열 상태 확인 필요)
- 완료 후 장독 정렬 제출 (§5 접합 가닥 태그는 올바른 감사 판정에 의존)

## ▶️ 이전 스냅샷 (2026-09-04 14:29)

### Codex 검토 — 무결성 구멍 발견·수정·검증 완료

Codex가 장독 드라이버의 실제 버그를 찾았습니다. `DONE` 마커의 `status=` 필드만 읽고 `audit.json`과 대조하지 않아, JSON이 FAIL인데 마커가 PASS면 `-uf`가 켜집니다. 실제 감사 파일로 공격을 시뮬레이션했고 두 케이스 모두 `exit 5`로 차단되었음을 확인. 원본 복원 완료 (md5 검증).

**`-t 32` 프로토콜 준수**: §4가 문자 그대로 `-t 32`를 고정했으나 초안은 16. 개정 대신 **32스레드 × 4동시**로 수정 — 처리량 동일 (2.17 GB/h × 4 ≈ 8.7 GB/h, 약 19시간).

**커밋 8개** (이전 스냅샷 통산): 코드 정리 및 무결성 게이트 재작성 완료.

### 리소스 및 가용성

**디스크**: 추가로 ~423 GB 필요 (EST 실적 배율 2.60×). 정렬 중 임시 공간까지 고려하면 0.6–0.8 TB 여유 필요.

**§3.3 항목 3 (PCR/UMI)**: ENA 메타데이터 부재로 채울 수 없음. 드라이버가 `umi_status unknown`으로 기록, 분자 아닌 리드로만 보고 (중복 제거 미적용). 정렬을 막지는 않음.

### 다음 작업

- 감사 129런 완료 대기 (현재 파티션 45 running / 48 pending)
- 감사 판정 후 정렬 129런 제출
- 커밋 8개 푸시 (정렬 제출 후 일괄 또는 지금)

## ▶️ 이전 스냅샷 (2026-09-04 14:23)

### 주말 매핑 — EST 완료, 감사 제출, 정렬 드라이버 완성

**EST 매핑 18/18 완료** (6148523) — 9종 × 2 arm, 5,068,470 레코드, ≥100: 96.54 %, ≥121: 97.55 %. 민감도 arm이 2026-09-03 A36 실행을 소수점 둘째 자리까지 재현.

**섹션 3.3 감사 129런 제출** (6148654, 12개씩) — 태스크 0 검증: 69초, PASS 99.78 %. FLAG 0x10 방향 변환 결정적(미적용 시 50 % 무작위).

**장독 정렬 드라이버 완성** — 16스레드 병렬 최적화, Codex 검토 완료. 거부 조건: 감사 미완료·다른 입력·audit.json 변경.

**커밋 6개** (`97bf5a1` 신규):
- `67_orientation_audit.py` — 섹션 3.3 항목 2 채점기
- `68_sample_fasta.py` — 결정적 균등 표본
- `orientation_audit.sbatch`, `longread_align.sbatch` — 129런 드라이버

### 다음 작업

- 감사 129런 완료 대기 (15–20분)
- 감사 판정 후 정렬 129런 제출
- 디스크 여유 확인 (수백 GB)

## ▶️ 이전 스냅샷 (2026-09-04 13:57)

### 매핑 잡 진행 · 2/18 완료 · 커밋 637a71b

잡 6148523_1 COMPLETED. 현재 2완료/1실행/15대기, 자원 가용 시 순차 진행 중.

**A37 쌍대 보고 스크립트 완성 및 테스트 추가**:
- `66_est_arm_report.py` — EST 양쪽 arm을 표 형식으로 병렬 보고. flagstat·PROVENANCE.txt·DONE만 읽음(재계산/BAM 불필요).
- DONE이 없는 arm은 pending 표시(exit 3) — 조용히 빼지 않으므로, 2026-09-03 "완전해 보이는 트리" 4시간 손실 사건 재발 방지.
- 테스트 5건: samtools 출력(백분율 같은 줄, primary mapped 행 오독 방지) 파싱 미묘함 고정. 픽스처는 실측(6148523_0).

**이 세션 커밋 합계 5개**:
- `f272bd4`: 재정렬 차단 결함 5건
- `f5a257f`: staging 게이트 강화 
- `033496b`: A38/v1.28 갱신
- `637a71b`: A37 쌍대 보고

### 다음 작업

- 나머지 15 태스크 COMPLETED 대기
- 전 9종 × 2 arm 대조표 완성 후 제시(태스크마다 보고 X)

## ▶️ 이전 스냅샷 (2026-09-04 13:32)

### Item 1 완료 · 커밋 033496b

SRR25203456 선언 행 추가, DATASET_ROLES.longread_v1.tsv v1.28 갱신.
- md5 d644278f → c2f4c2e8 / 214행 → 215행 / 역할 129/31/54 → 129/31/55
- Codex 예측값(129/31/55)과 정확히 일치, 실측으로 basis 확인(`SRR25203456_1.fastq.gz/`는 디렉토리, HEAD 후행 슬래시 301)
- 프로토콜 §1 frozen-objects 테이블 행 업데이트, A38 개정 기록
- 문서(TRAINING_INPUT.md, training_input/README.md) md5값 갱신
- 테스트 18건 통과(include A38 검증)
- todo.md: 낡은 주장(`d644278f` 불변) ✏️ 정정 표시(삭제 X)

### Item 2 · Codex 패치 미적용, 재설계 요청 중

실측으로 Codex 패치 적용 불가 입증:
- 문제: FILT가 run 접근번호 필터가 아님. 호출자들이 'OXFORD_NANOPORE'(행전체 2, run컬럼 0) · 'PACBIO'(1,0) · 'maize'(0,0) · 'SRR25203456'(1,1) 등을 넘김
- 위험: 그대로 적용 시 OXFORD_NANOPORE 데이터셋 전부 0런 선택, 2026-09-03 Col-0 사건 재발(4시간 사라짐)
- 병렬 드라이버(`longread_run_all.sh`, `training_run_all.sh` 등) 5개도 함께 깨짐
- Codex에 재설계 피드백 제출: FILT 범위 좁히기, PLATFORM_RE 추가, 호출자 일괄 갱신, 그리고 **이 작업이 임계경로인지** 확인(수집 122런 완료, 매핑 블로커?)

### 매핑 잡 & 다음 작업

- 잡 6148523: 18 태스크 여전히 PENDING(우선순위 대기)
- Vvinifera 재트림 대기 중
- 다음: Codex 피드백 + 잡 상태 모니터링

## ▶️ 이전 스냅샷 (2026-09-04 13:25)

### 저장소 편입 완료 · 커밋 f5a257f

Codex 권고대로 배치:
- **정본**: `transgenic/revision/scripts/65_make_training_input_manifest.sh` (60–64 다음)
- **문서**: `revision/TRAINING_INPUT.md`
- **매니페스트**: `data/manifests/DATASET_ROLES.{est,longread}_v1.tsv` (md5 `a6375713`·`d644278f` 동결값 일치 확인)
- **테스트**: `tests/test_make_training_input_manifest.py` — 7건 통과, awk `exit` 회귀테스트 포함

Codex 지적 2건 반영: (1) 원자 쓰기(임시→`mv`로 교체), (2) 해시·stat 실패 시 즉시 실패(`2>/dev/null || echo 0` 제거).

### 18 태스크 제출 · 잡 6148523

**9종 × 2 arm** 제출 완료. 현재 PENDING(우선순위). 
- 0–8: primary arm ≥100 → `evidence/est_align/`
- 9–17: min121 arm ≥121 → `evidence/est_align_min121/`

### 재트림 진행

8/9 완료(Ptrichocarpa, Sbicolor 추가). **Vvinifera만 남음**.

### 게이트 재실행 진행 중

28.9 GB 번들로 `make_manifest.sh` 백그라운드 실행. 모든 입력 검증 후 원자로 MANIFEST.tsv 기록.

### 다음 작업

1. Vvinifera 재트림 완료 대기
2. 게이트 재실행 결과 확인 (md5 9종 전부 일치)
3. 매핑 태스크 진행 감시 (잡 6148523)

## ▶️ 이전 스냅샷 (2026-09-04 13:16)

### Codex·중재 규칙 확립 (13:05)

사용자 지시: **논리는 Codex 채택, 코드 실물은 나(Claude) 채택**. 메모리 저장 완료.

### est_align.sbatch 5건 결함 수정 · 커밋 f272bd4

**①DONE 마커**: 빈 파일로 재제출 방어 불가 → task 0이 낡은 DONE을 감지해 `exit 6` 거부.  
**②min_len 하드코딩**: 121 고정 → `OBS_MIN` 변수로 arm별 주장(primary ≥100, secondary 121) 추적.  
**③contig_alias awk**: `$1: unbound variable` 종료(set -u) → 따옴표 문법 수정 후 `Chr1→1`, `ChrM→Mt` 정상.  
**④PROVENANCE 검사 부재**: "checked"라고 쓰면서 코드는 경로만 읽음 → 실제 md5 대조 로직 추가.  
**⑤Codex 제안(awk exit)**: `exit 10`은 awk `END`에 덮어씌워짐 → `err` 플래그로 고침 (실증: `종료코드=13`).

### Codex 실측 검증 3건 — 규칙 적용

1. **A37 양 arm 정렬 요구**: 주말 계획 9→**18 태스크** 수정. 제 계획은 primary만이었음.
2. **contig_alias는 실제로 깨짐**: 0272baf 이후 미실행 → 첫 실제 재트림에서만 노출.
3. **SRR25203456 ENA 경로는 디렉토리**: 301 리다이렉트, `filereport.tsv` md5·read_count·instrument 전부 일치.

### Item 3(staging gate) 완료 · 정상 경로만 통과

9종 md5 전부 일치. 옥수수(`Zmays`)를 애기장대 이름으로 덮는 우회 → `rc=2` 차단 (매니페스트 부재도 fail-closed).

### A36 트리 격리 완료

`evidence/RETIRED_DO_NOT_USE/est_align_a36_min121_20260903/` — 9종·로그 이동. WHY_RETIRED.txt 기록.

### 재트림(retrim_a37) 진행률

**5/9 완료**(Athaliana·Bdistachyon·Gmax·Osativa·Ppatens). 남은 4종(소형) ETA ~13:15. 큐 상황: 32 running + 20 pending(주로 RSEM).

### 다음 작업

1. 재트림 전부 완료 대기 (4종)
2. 양 arm(≥100 + 121) 재정렬 18 태스크 제출
3. `longread_v1` 재동결(SRR25203456 행 추가)

## ▶️ 이전 스냅샷 (2026-09-04 13:01)

### A37 재트림 진행률 (13:02)
3/9 완료 (Athaliana·Bdistachyon·Gmax·Osativa), 순차 실행 ETA 20분. 로그인 노드 고아 없음.

### 새 발견: 재정렬 차단 결함 2건
**①DONE 마커 미정리**: `est_align.sbatch:56`의 조기 exit(`[ -f "$OUT/$SP/DONE" ] && exit 0`)이 재트림 후 즉시 종료. A36 BAM이 덮어씌워지지 않음. A36 세트를 `RETIRED_DO_NOT_USE/`로 사전 격리 필요.

**②PROVENANCE 하드코딩**: `est_align.sbatch` 119행이 `min_len 121 (protocol A36, v1.26)` 고정. 145행 note는 이미 A37 primary(≥100)라고 기록. 지금 재정렬하면 입력값 ≥100을 121이라고 PROVENANCE에 쓸 위험.

### Codex 검토 중
todo.md §2c 3건(SRR25203456 선언·longread_fetch A18.8·staging gate) gpt-5.6-sol이 패치·검증 명령 준비 중. 결과 대기.

### 다음 작업
1. Codex 검토 완료 대기, 패치·검증 통합
2. est_align.sbatch DONE 마커 정리 + A36 BAM 격리
3. PROVENANCE min_len 하드코딩 수정 (A37 로직)
4. 재트림 완료 후 primary arm(≥100) 재정렬 실행

## ▶️ 이전 스냅샷 (2026-09-04 12:50)

### 4번: A37 주석 개정 — 커밋 `34c09db`

`est_align.sbatch` PAF 보존 주석 수정:
- "reference md5 checked against b5_species_v1.tsv, not the section 1 frozen object"로 명확화
- **저자 진술이지 측정이 아님** — 대조는 미연기, 나중에 검증 필요

### 재트림 진행률

- **Bdistachyon**: 완료 ✓
- **나머지 8종**: 백그라운드 실행 중 (Bpentandra, Drotundifolia, Egrandis, Hvulgare, Oeuropaea, Pabies, Slycopersicum, Vvinifera)

### 다음 작업

1. 9종 재트림 완료 대기
2. Primary arm(≥100 nt)으로 재정렬 실행 (BAM + PAF + PROVENANCE 포함)
3. `longread_v1` 재동결 (SRR25203456 제외 행 추가)

## ▶️ 지금 돌고 있는 것

| | 상태 |
|---|---|
| **A37 재트림** | 9종 × 2 arm(≥100 primary / ≥121 민감도). `evidence/retrim_a37.log`, 3/9 완료 시점 기록. blastn 재실행 없이 기존 `hits.tsv` 재사용 |
| **ACCESS** | 아이디 생성 대기 — #51·#19·#20. 유일한 외부 대기 |

재확인: `tail evidence/retrim_a37.log` · `ps -eo cmd | grep [r]etrim_a37`
⚠️ **로그인 노드도 따로 볼 것**: `ssh pronghorn 'ps -eo cmd | grep -c "[l]ongread_fetch"'`
— 2026-09-04에 거기서 2일 4시간 된 고아를 찾았다. 다른 노드 프로세스는 `ps`에 안 보인다.

## ✅ 최근에 끝난 것

**2026-09-04 (Codex 외부 검토 이후)** — **A37(v1.27)**: A36을 대체해 **≥100 primary 복귀**, 121은 민감도 arm,
쌍대 보고·길이 구간별 지표 의무화, 해석 문구 고정(`9d81e7a`) · **`est_align.sbatch` 3중 수정**: PAF 보존(§4)·
**전체 PIPESTATUS 검사**(sort 실패를 놓치던 구멍)·PROVENANCE.tsv·contig alias 표(`0272baf`) ·
**#15 재현성 검사 통과** — 이틀 간격·다른 커밋 재빌드에서 데이터 6개 테이블 **내용 해시 전부 일치**(`8dbc19d`) ·
`training_input/` 스테이징 40파일 28.9 GB + `MANIFEST.tsv` + 테스트 종 차단 게이트.

**2026-09-03** — A21 스크리닝 최초 실행·9종 재정렬 · 장독 7런 회수(ENA md5·리드 수 일치) ·
`longread_fetch.sh` 3중 수정(재개·md5 대조·FAILED 계수) · 드라이버 11개 저장소 편입 ·
로그인 노드 2일 4시간 고아 드라이버 종료 · 장독 범위 v2 동결 · 이슈 12건 종료 · 테스트 155 → **412**.

## ▶️ 다음 작업

**정본은 [`todo.md`](todo.md).** 여기는 요약이고, 지금 순서는 **Codex 검토 지적을 닫는 것**이다.

1. **재트림 완료 대기 → primary(≥100) arm으로 9종 재정렬** (`est_align.sbatch`, 수정본)
2. **Codex 미해결 3건** — ①`SRR25203456` 선언 행 추가 후 `longread_v1` **재동결**(A18.3은 모든 런에 역할을
   요구하며 산문 기록으로는 부족) ②`longread_fetch.sh`의 A18.8 위반(행 전체 정규식·첫 URL만 선택)
   ③staging 게이트가 파일명만 검사(md5·9종 완비·매니페스트 부재 검사 없음)
3. **주말 잡 둘** — **miniprot(#45)**: 준비 완료, 사이징 잡 먼저(2스레드로 20k가 10분 미완) ·
   **장독 정렬**: **§3.3 사전 감사(#25)가 선행**이고 그중 10,000리드 strand 일치도가 `-uf` 사용 여부를 정한다
4. **§5 접합 호출은 아직 착수 불가** — 채점기 **#24 미구현**

⚠️ **저자 판단 대기**: §3.3 항목 1이 "쓰인 FASTA md5 = §1"을 요구하는데 A. thaliana는 다르다
(`513e5ef3` vs `987ca803`). 저자는 **헤더/포맷 차이일 뿐 서열은 같다**고 진술했고 그렇게 기록했다(측정 아님).

## 🖥️ 머신과 데이터 위치

| 머신 | 접속 | 상태 |
|---|---|---|
| pronghorn (여기) | — | 수집·정렬 모두 완료. `evidence/`(training 150 GB). 로그인 노드 CPU 2개, 부하 높음 |
| pgl-gpu | `ssh gpu` | RTX 4090 24 GB, 16 CPU, RAM 62 GB, 디스크 216 GB 여유. python은 `~/micromamba/envs/transgenic/bin/python`(duckdb 1.5.5, torch 2.5.1, accelerate 1.14.0) |
| GB10 | `ssh gb10` | spark-d642, 20 CPU, 128 GB. ⚠️ root의 vLLM 2개가 GPU 약 101 GB 점유, torch 환경 없음 |
| ACCESS Delta | 저자 준비 완료 | 자원명·계정·파티션 미확인 |

- pgl-gpu: `b5/merged/b5_full_v1.db`(23.5 GB, 동결) · `b5/full/<종>.db` 9개(16 GB, 원본 보존) · `splits/final/` · `qc/` · `swissprot_v3/` · `orthodb_filtered_stage2/`(md5 `453cb32b…`) · `proteomes/` · `genomes_src/` · `bench2/`
- pronghorn: `evidence/training` **122런 150 GB(완료)** · `evidence/est/<종>/univec/`(A21 산출) · `evidence/est_align/`(A21+A36 적용) · `evidence/univec/`(UniVec_Core DB) · `evidence/RETIRED_DO_NOT_USE/`(subreads 51런 + 스크리닝 이전 BAM 9종) · `evidence/qc/`·`evidence/splits/`

## 📊 확정된 실측 (재측정 불필요)

**B5 병합 DB (#50)** 종·계층별 행

| 종 | 30,720 | 61,440 | 129,024 | 합계 | rc | train / valid / test |
|---|---|---|---|---|---|---|
| Athaliana | 5,520 | 3,240 | 1,694 | 10,454 | 4,112 | 7,631 / 1,441 / 1,382 |
| Bdistachyon | 11,190 | 6,935 | 3,842 | 21,967 | 7,629 | 16,469 / 1,984 / 3,514 |
| Gmax | 25,633 | 17,198 | 10,416 | 53,247 | 15,475 | 38,954 / 4,842 / 9,451 |
| Osativa | 14,500 | 8,702 | 4,855 | 28,057 | 7,621 | 20,283 / 2,679 / 5,095 |
| Ppatens | 19,258 | 12,720 | 7,031 | 39,009 | 17,317 | 29,110 / 3,666 / 6,233 |
| Ptrichocarpa | 15,227 | 9,523 | 5,326 | 30,076 | 9,774 | 21,750 / 3,332 / 4,994 |
| Sbicolor | 15,097 | 9,965 | 6,161 | 31,223 | 7,842 | 22,730 / 2,777 / 5,716 |
| Sitalica | 12,924 | 8,285 | 4,866 | 26,075 | 6,539 | 18,078 / 2,423 / 5,574 |
| Vvinifera | 15,996 | 10,268 | 5,852 | 32,116 | 7,770 | 23,166 / 2,766 / 6,184 |
| **합계** | **135,345** | **86,836** | **50,043** | **272,224** | **84,079** | **198,171 / 25,910 / 48,143** |

마스킹 카운터: leak 410,386 · hard 27,258 · decoy 31,808 · component 29,367 · dup_collapsed 743 · edge_partial 184,894. `train_weight=0` 행 0, NULL 라벨 0.

거부: mask fraction(A33) 1,543 · transcript cap 333 · token cap 3 · **no CDS 6,205(전부 Vvinifera)** · **canonical order 97**. 뒤 두 개는 결함(#56·#55).

`tier_margin_unguaranteed`(A33.4, 코드에 미구현 → gene_key_map에서 재계산):

| 계층 | 길이 보장선 | 초과 유전자 | 그중 margin 미충족 |
|---|---|---|---|
| 30,720 | 18,480 nt | 3,418 | 2,034 |
| 61,440 | 38,960 nt | 653 | 434 |
| 129,024 | 84,016 nt | 86 | 44 |

가운데 열은 프로토콜의 3,418 / 653 / 86과 정확히 일치(최장 유전자 196,414 nt도 일치) — 좌표 신뢰 가능.

**벤치마크 (RTX 4090 24.56 GB, bf16, batch 1, gradient checkpointing on)**

| 계층 (nt) | 최대 메모리 | 미니배치당 | 라벨 토큰/s | OOM |
|---|---|---|---|---|
| 30,720 | 12.05 GB | 0.70 s | 573 | 0 / 240 |
| 61,440 | 16.94 GB | 1.54 s | 524 | 0 / 240 |
| 129,024 | 24 GB 초과 | (외삽 3.2 s) | — | 1,093 / 1,103 |

비용은 라벨 길이에 선형, 입력 길이에는 거의 무관. 메모리는 고정 6.6 GB + 약 0.16 MB/nt → 129 kb ≈ 27 GB. A100 80 GB·GH200 96 GB에는 여유.

**분할표 (#14)** `data/splits/b5_orthogroup_split_v1.tsv` 334,642행, sha256 `904c6265…`. OrthoFinder 3.1.5 HOG N1. 유전자 기준 train 63.9 % / valid 8.7 % / test 27.4 %(타일 기준 비율은 위 표와 다름 — 타일 split은 블록 추첨).

**단백질 자원 (#44)** OrthoDB v12 Viridiplantae → 1,211만, md5 `453cb32b…`. Swiss-Prot 민감도 세트 42,096 entry.

**GeenuFF QC (#46)** 9종 플래그 192,497건, hard 유전자 8,770.

**EST 정렬 (2026-09-03, minimap2 2.28, `-ax splice:hq -G 200000 --secondary=no --cs=long`) — 9종 전부 완료**

**EST 정렬 — A21 스크리닝 + A36 하한 적용 후 9종 (2026-09-03, SLURM `6147708_[0-8]` 전부 COMPLETED)**

minimap2 2.28, §4 고정 명령, 입력은 `est/<종>/univec/est.univec.fa.gz`. 매핑률은 primary 기준.

| 종 | 원본 | 제외(A36+트림후) | trim | split | 정렬 입력 | **매핑률** | 이전(원본) | 초 |
|---|---|---|---|---|---|---|---|---|
| Gmax | 1,461,724 | 133,659 | 15,137 | 987 | 1,328,299 | **98.94 %** | 94.01 % | 117 |
| Ppatens | 382,587 | 1,883 | 1,732 | 173 | 380,782 | 98.76 % | 98.49 % | 52 |
| Bdistachyon | 206,255 | 1,393 | 5,571 | 189 | 204,868 | 98.42 % | 98.10 % | 32 |
| Ptrichocarpa | 89,943 | 989 | 2,621 | 6 | 88,956 | 97.33 % | 96.68 % | 27 |
| Vvinifera | 446,853 | 4,533 | 3,225 | **11,377** | 447,682 | 97.12 % | **97.61 %** ↓ | 87 |
| Sbicolor | 209,835 | 1,649 | 916 | 67 | 208,212 | 97.05 % | 96.83 % | 48 |
| **Athaliana** | 1,529,700 | **774,531** | 8,546 | 4,974 | 755,367 | **96.96 %** | **73.51 %** | 45 |
| Osativa | 1,255,251 | 16,541 | 43,527 | 8,335 | 1,238,994 | 96.40 % | 95.50 % | 106 |
| Sitalica | 66,027 | 542 | 82 | 44 | 65,513 | 92.80 % | 92.31 % | 25 |
| **합계** | **5,648,175** | **935,720 (16.6 %)** | 81,357 | 26,152 | **4,718,673** | — | — | — |

- **Athaliana 73.51 % → 96.96 %.** 이제 9종이 **92.80–98.94 %** 안에 들어오고 Athaliana는 7위다(이전엔 혼자 20 %p 아래).
- 정렬 입력이 원본에서 제외분을 뺀 것보다 많을 수 있다 — **split이 한 레코드를 여러 조각으로 만들기 때문**(Vvinifera가 그 경우다).
- ⚠️ **Vvinifera만 매핑률이 내려갔다(97.61 → 97.12 %). 원인 확인됨: split 조각.** `_partN` **10,718건이 48.43 %**로만 붙고 나머지 436,964건은 **98.32 %**다. split을 빼면 오히려 97.61 → 98.32 %로 오른다. 즉 A21이 내부 벡터로 판정해 자른 키메라 의심 클론의 절반이 실제로 게놈에 안 붙는다 — 스크리닝이 일을 한 것이지 손해가 아니다.
- ⚠️ **Vvinifera의 split 비율 2.55 %는 다른 종(0.01–0.66 %)의 4–250배다.** 그 EST 세트의 내부 벡터 오염률이 유독 높다. Osativa는 trim이 3.47 %로 가장 높다.

**A21 스크리닝 실측 (UniVec_Core build 10.0, md5 `82f07865`, blastn 2.5.0+)**

전 종 합계 **제외 935,720건(16.6 %)** · trim 81,357 · split 26,152. 제외분의 대부분(82.8 %)이 Athaliana다.

**역할 매니페스트 (#5)** `evidence/DATASET_ROLES.tsv` 187행 — 학습 적격 91 · 검증 전용 33 · 제외 51+.
EST 범위만 `DATASET_ROLES.est_v1.tsv`(11행, md5 `a6375713`)로 **프로토콜 §1에 동결**. 장독은 수집 완료 후 v2.

## ⚠️ 알아야 할 함정

- **`nextval()`은 `ORDER BY`를 따르지 않는다**(위 표). 대량 `INSERT … SELECT`에서 시퀀스로 번호를 매기지 말 것. 리포의 다른 `nextval` 사용처는 전부 한 행짜리 `VALUES (nextval(…), …)`라 안전하다.
- **트레이너는 배치 실패 시 죽습니다**(A35). 24 GB 카드로 129 kb 계층을 돌리면 즉시 실패하며, 그게 정상 동작입니다.
- **인코더는 부분 사전학습**입니다(A34). 339 텐서 중 227개 로드, layers 8–11(112개) 무작위, 147개 타일링 복제. 초기화는 바꾸지 않습니다. **Methods에 "partially pretrained"로 써야 합니다.**
- **`pkill -f <문자열>`은 같은 문자열을 가진 자기 자신의 셸을 죽인다.** 이 세션에서 세 번 당했습니다(exit 143/144). PID로 죽이세요.
- **벤치마크 실행 시 GPU 단독 사용 필수** — `run_bench2.sh`에 `flock`을 걸어 두었습니다.
- **tqdm이 `\r`로 출력**하므로 결과 JSON이 진행률과 같은 줄에 붙습니다. `tr '\r' '\n'`로 분리해야 grep에 걸립니다.
- **compute 노드에서는 로그인 노드의 `setsid nohup` 프로세스가 pgrep에 안 보입니다.** 파일 mtime으로 확인하세요.
- **`importlib.util`로 모듈을 파일 경로에서 불러올 때 `sys.modules`에 먼저 등록**해야 합니다. 안 그러면 그 모듈의 dataclass가 필드 타입을 못 찾고 죽습니다. 리포 관례는 `types.ModuleType` + `sys.modules[name] = mod` + `exec(compile(...))`(`tests/conftest.py`).
- **`gsf_contract.validate_split()`에 TSV 문자열을 그대로 넘기면 위반 242,855건이 나온다 — 전부 거짓이다.**
  `strict_holdout` 컬럼이 literal `true`/`false`인데 검증기는 `r.get("strict_holdout")`을 truthy로 보므로
  `"false"`가 참이 된다. **bool로 캐스팅한 뒤 넘길 것**(캐스팅하면 위반 0). 2026-09-03에 #14 재검증하다 겪음.
- **`sacct`에서 `.extern` 스텝이 `OUT_OF_ME+`로 뜨는 것은 OOM이 아니다.** `6147674`·`6147708` 두 실행 모두
  9개 태스크 전부 `.extern`만 `OUT_OF_ME+ 0:125`이고 `.batch`와 잡 본체는 **전부 COMPLETED**였다.
  `sacct -j <잡>`만 보고 메모리 부족으로 판단하지 말 것 — `.batch` 스텝 상태를 봐야 한다.
- ✅ **`0 runs DONE`이 "받을 게 없었다"로 읽히던 문제는 고쳤다**(2026-09-03 23:5x). 완료 줄이 이제
  `N runs DONE, M skipped, **K FAILED**, U UNRESOLVED`를 적고, 실패가 있으면 **exit 5**로 끝난다
  (미해결은 exit 4). 실패는 `<run>.FAILED` 표식으로도 남는다 — while 루프가 파이프 서브셸이라
  카운터 변수가 살아남지 못하는 것이 원래 원인이었다. 성공하면 표식이 지워져 자정된다.
  ⚠️ 그래도 **계획과 대조하는 습관은 유지할 것** — 이 수정은 스크립트가 스스로 신고하게 만들 뿐,
  애초에 큐에 오르지 않은 런은 여전히 아무도 세지 않는다.
- **종 단위 매핑률은 종 간 비교 지표가 아니다.** EST 길이 구성이 종마다 다르기 때문이다. ≤120 bp EST는
  *모든* 종에서 24.8–71.2 %로만 붙고 >120 bp는 *모든* 종에서 92.8–98.9 %로 붙는다 — 즉 종별 총 매핑률
  차이는 대부분 길이 구성 차이다. Athaliana를 20 %p 이상 낮은 이상치로 본 것이 이 착시였다(2026-09-03).
  **비교하려면 길이를 통제한 값(>120 bp 매핑률)을 쓸 것.**
- **`grep -E "$FILT"`는 filereport의 *행 전체*에 걸린다.** 접근번호로 한 런만 고르려면 `SRR…[[:space:]]`처럼
  뒤에 공백을 붙인다. `^SRR…$`로 앵커를 걸면 **한 줄도 안 맞고 전부 skip된다**(2026-09-03에 겪음).
- **단일 다운로드 스트림은 약 1 MB/s에 묶여 있다** — pronghorn 대역폭 문제가 아니다. 병렬도가 곧 처리량
  (8스트림 3 MB/s, 28스트림 7 MB/s). `evidence/queue_fetch.sh`가 런 단위로 병렬화한다.
- **로그인 노드는 CPU 2개다.** 정렬 같은 건 반드시 SLURM으로. `sbatch`는 PATH에 없으니
  `/cm/shared/apps/slurm/current/bin/`를 쓴다. 파티션 `cpu-s1-pgl-0`(64코어 256 GB × 4노드),
  동시 실행 ~22잡 제한, 대기 원인은 CPU가 아니라 **메모리 과다 예약**.
- ✏️ **(2026-09-04 정정) 학습 장독 드라이버는 멈춘 것이 아니었다** — 로그인 노드에서 **2일 4시간** 살아 O. sativa `nip_pool_PRJNA953663`에서 **17 h 51 m 붙잡혀** #63 금지 ONT WGS를 15.5 GB 받고 있었다. **다른 노드의 프로세스는 `ps`에 안 보인다** — 자라는 `.part` 하나로 추적해 찾았고 PID로 종료했다. 아래는 그때의 서술로 남긴다: ~~9/2 23:29 S. italica에서 멈춰 있었다~~ — O. sativa·P. patens는 시작조차 안 했다(`longread.log`에 "TRAINING LONGREAD ALL FINISHED" 없음). 지금 `training_run_ont_missing.sh`로 ONT 5개만 받는 중. **`rerun_training_after_finish.sh`는 자기가 실행할 스크립트의 완료 문자열을 기다려서 영원히 발동하지 않는다.** Ath PacBio 2건의 탭 붕괴 버그는 `longread_fetch.sh`에서 이미 고쳐졌다(재시도하면 Sequel II 런을 잡는다 — 다만 그게 subreads라 #60).

---

# 🧊 이력은 이 파일에 없습니다 — [`quarantine.md`](quarantine.md)

2026-09-03 01:1x 이전의 스냅샷 **4,690줄**을 [`quarantine.md`](quarantine.md)로 옮겼고, 같은 날 오전분 **229줄**(§2a)과 오후분 **312줄**(§2b)을 더 옮겼다 — **하루에 세 번**이다. 삭제가 아니라 이동이고,
원문은 마커까지 그대로 보존돼 있다. resume.md는 이제 **지금 유효한 것만** 담는다.

| 찾는 것 | 어디에 |
|---|---|
| 지금 상태·다음 작업 | 이 파일 위쪽 📌 정본 블록 |
| 과거 스냅샷, 이미 끝난 작업 | [`quarantine.md`](quarantine.md) §2a·§2b·**§2c**·§2(그 이전) |
| 왜 틀렸는지 / 무엇이 대체했는지 | [`quarantine.md`](quarantine.md) §1·§1d(구성 효과)·§1e(A36 대체)·**§1f(FILT 패치 기각)** |
| 기각된 설계·데이터·코드 경로 | [`quarantine.md`](quarantine.md) §3–§6 |
| 저장소 문서(README·CLAUDE.md)의 낡은 부분 | [`transgenic/quarantine.md`](transgenic/quarantine.md) |

**규칙**: 스냅샷이 정본에서 밀려나면 여기에 링크만 남기고 본문은 `quarantine.md`로 옮긴다.
⚠️ **2026-09-03에 이 규칙을 세 번 써야 했다**(01:1x·13:2x·17:0x). 추출이 잘 도는 것이 아니라 **훅 주기가 세션 속도를 못 따라가는** 것일 수 있다 — 활발한 날은 20–30분마다 20–35줄이 쌓인다.
