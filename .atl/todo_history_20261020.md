# TODO — C2 컴파일 이력 복구 (2026-10-12)

정본: [handoff](../notes/RESTART_HANDOFF_20261012.md). **검증통과후추가승인없이C2본학습시작**. 과학적중단조건은유지한다. 과거todo는[history](todo_history_20261012.md)로분리했다.

## 현재 실행 순서

- [x] 재개 확인: main6e33843, 구현d7b1a37 clean; live RUNNING없음, C2/B6 held·B5pending 유지. 3364480 COMPLETED0:0. NVMe project22169 ddn_ssd349.4/500GiB(soft), 여유약150.6GiB; HDD829.4/1000GiB. 중복제출없음.
- [x] 컴파일이력복구설계·구현188bf3e완료(과학적정확일치는3369338 별도게이트): 전역event위치별모든rank의실제companion descriptor/hash,train/eval view/grad context,checkpoint결속. rank별독립replay/DDP불일치방지. event-only압축은별도검증.
- [x] legacycheckpoint200bootstrap명시화(실제3369694연속/재개PASS): 원본checkpoint/drawprefixSHA,최초tier입력뿐아니라rank3row3915(labels4501)등기존특수화이력포함. 관측하지않은ledger라고정직하게표시하고별도증명.
- [x] 상태복원91c2737·journal789ecf4·trainer180589b·cache경로188bf3e. 실패회귀/구현/독립리뷰PASS; 전체129시험127pass2설명된skip, state별도12pass. latest/epoch/fixed/eval·pendinggrad·RNG복원구현완료. 실제4GPU게이트는별도.
- [x] local검증/오류주입/독립검토PASS(198ce14); NGC회귀3369694 58시험57pass/1skip113.286s. source/main graph갱신. 실제과학게이트는아래별도.
- [x] 실제4GPU전체checkpoint bitwise proof3369694 PASS: 모델차이0,optimizer/scheduler/4rankRNG/입력/shape/loss/context/boundarygradient/RNG/history정확일치. 원래210loss3.2567708492279053복원. parent/batch/externCOMPLETED0:0,1h05m57s. 증거3369694/PARENT_ACCEPTANCE.json.
- [x] 400저장·새프로세스400→410검증 **3370048 PASS**, 10-14 07:28:53PDT 종료, COMPLETED0:0 / 2h01m02s. 모델·optimizer·scheduler·4rank RNG·전체 trace·history 정확 일치, rank3slot6718 관측, 역사적210loss 복원. 10-20 live/accounting/receipt 검수 완료.
- [x] 본학습운영도구 로컬·독립검토·호스트검증: production7266e14(33+NGC CPU33), controller6e5f0f7 in-place v2(52+WorkDir10), 실제연결ea699d5(3). 최종config검토/actualhostPASS 후 PID1545990으로 감시했고, 아래 실제400검증/활성화/본학습까지 완료.
- [x] 검증 통과 후 C2본학습 자동 연결 **3379630**, 검증resumed410 제자리 활성화. 원본200·진단256 보존, 구held3345920 취소. 10-15 09:56:07PDT 시작 → 10-16 06:18:31PDT COMPLETED0:0 / 20h22m24s.
- [x] **4099/4099 완료**: epoch 평가1회, `step_04099/PILOT_FINAL`, `PRUNE_EXEMPT`, `TRAINING_DONE`, allocation postflight 및 parent/batch/extern 정상exit 확인. 10-20 작은 metadata/history/marker 해시·receipt 연결 재검수 PASS. controller complete/revision22/error없음.
- [x] C2 완료 후 우선순위 복구 처리 확인. B6 3345917/18 Nice0·held 유지. B5는 복구 시점 전에 terminal: 3309786 preflight실패, 3309787–90 취소.
- [x] B5 실패 원인 확인·시험 잔여물 정리: C2 controller 호스트 시험용 절대 symlink가 컨테이너에서 dangling. 같은 이미지/bind로 재현 후 정확한 소유 링크만 제거, 대상 receipt 불변 및 컨테이너 경로 부재 확인.
- [ ] **B5 재개 게이트 미통과:** 시험 잔여물 원인은 제거했지만 전체 preflight·checkpoint/재개 조건·현재 자원 확인은 아직 필요. 3309786 실패 후 새 잡은 제출하지 않았다. 학습 source/data/checkpoint 수정 없음.
- [x] B6 후속 검토:3345917/18은학습미시작,3345916은입력검증.198ce14 freshB6 경로 및 frozen4099설정 확인, B6 CPU재개8시험PASS. 기존WP5승인유효; 추가승인공백이 아니라 후속실행누락과 기술게이트였다.
- [ ] [B6실행계획](../notes/b6_pilot_resume_20261020/PLAN.md) Task1: fresh0→410/snapshot400/new-process400→410 evidence-on동일성도구 **19d319d**(구현48d526b),17proof+5hostcontract+38source시험PASS. 독립검토중. 별도branch`b6-pilot-proof-20261020`, trainer198ce14불변.
- [ ] 같은계획Task2:NVMe94.37GiB→100GiB이상 복구. B5과거epoch09–11 39.46GiB HDD검증이관 **3425279 RUNNING/gh027**,16:35PDT시작.12–14/latest/best보존.1GPU는DeltaAI필수최소할당이고CPU파일I/O만실행;완료영수증과quota실측전공간확보완료로판정금지.
- [ ] B6실제GPU증명→검증된후속4099연결. 구3345917/18은현재held유지. C2검증을B6전체검증으로대체하거나구소스held잡만해제하지않는다.
- [ ] Delta CPU1,000core-hours 신청 처리 대기(사용자통지예정). 승인후CPU Slurm계정·bilv공유/work권한·stdlib도구호환을확인해CPU전용작업전환. 현재DeltaAI이관3425279와B6검증준비는유지.
- [x] 기다리는동안DeltaCPU공식사용법조사·[안내](../notes/delta_cpu_20261020/USAGE.md)·zeroGPU읽기전용preflight.sbatch준비. cpu1core/1GiB/10min,구문/실행거절/링크검사PASS. 실제Delta접속/제출은미실행.

진행 기록: [20261012 실행 기록](../notes/C2_REPLAY_PROGRESS_20261012.md).

## 완료

- [x] 자동이관코드3b4eb0d:양서버69시험+실제fault3cases. B5epoch01–08 99.06GiB pronghorn검증이관완료.
- [x] 원본C2실패3345919확인·checkpoint200보호·후속C2/B6hold.
- [x] 종료처리수정3f81fb6(NGC56pass/1skip),compile후보d7b1a37(NGC74pass/1skip).
- [x] 3360978실패원인재현,3361110실190loss일치+108경계PASS.
- [x] 3361386을손실차이발견후의도적정상저장종료:진단256,원본200그대로.
- [x] 3361666: old/new동일,연속/재개609텐서불일치·입력/RNG동일증명.
- [x] 3364480: cold/primed/reset12report검증·독립리뷰,컴파일이력원인분리확인. 운영수정완료는아님.
- [x] 재시작4문서갱신·역사보존. `intend.md`는`intent.md`정본연결.
- [x] 사용자의전체푸시: main스냅샷8ef1871및로컬103개작업브랜치atomic push완료,전체원격SHA일치확인. [영수증](../notes/PUSH_RECEIPT_20261012.json).
