# TODO — C2 컴파일 이력 복구 (2026-10-12)

정본: [handoff](../notes/RESTART_HANDOFF_20261012.md). **검증통과후추가승인없이C2본학습시작**. 과학적중단조건은유지한다. 과거todo는[history](todo_history_20261012.md)로분리했다.

## 현재 실행 순서

- [x] 재개 확인: main6e33843, 구현d7b1a37 clean; live RUNNING없음, C2/B6 held·B5pending 유지. 3364480 COMPLETED0:0. NVMe project22169 ddn_ssd349.4/500GiB(soft), 여유약150.6GiB; HDD829.4/1000GiB. 중복제출없음.
- [x] 컴파일이력복구설계·구현188bf3e완료(과학적정확일치는3369338 별도게이트): 전역event위치별모든rank의실제companion descriptor/hash,train/eval view/grad context,checkpoint결속. rank별독립replay/DDP불일치방지. event-only압축은별도검증.
- [ ] legacycheckpoint200bootstrap명시화: 원본checkpoint/drawprefixSHA,최초tier입력뿐아니라rank3row3915(labels4501)등기존특수화이력포함. 관측하지않은ledger라고정직하게표시하고별도증명.
- [x] 상태복원91c2737·journal789ecf4·trainer180589b·cache경로188bf3e. 실패회귀/구현/독립리뷰PASS; 전체129시험127pass2설명된skip, state별도12pass. latest/epoch/fixed/eval·pendinggrad·RNG복원구현완료. 실제4GPU게이트는별도.
- [x] local검증/오류주입/독립검토PASS(198ce14); NGC회귀3369694 58시험57pass/1skip113.286s. source/main graph갱신. 실제과학게이트는아래별도.
- [ ] 실제4GPU연속/재개전체checkpoint bitwise proof **3369694 RUNNING gh118, 시작17:45:33PDT**. 이전3369338 bootstrap tuple실패는198ce14수정·6시험·독립검토PASS. original210loss3.2567708492279053와의역사적차이도해결. 단일forward일치만으로통과하지않는다.
- [ ] 원본200보존한검증사본에서280이상/400저장·다음재개확인,여유공간재측정.
- [ ] 본학습운영도구 독립검토수정: production1f4bb61 compiled키/CUDA RNG검증; controllerd332dcf atomicno-replace/scriptmanifest결속. 27/38단위·3연결시험통과, 수정후재검증/검토필요.
- [ ] **통과즉시C2본학습4099시작(추가승인불필요)**. 기존controller재사용금지,새소스/검증receipt/중복방지journal. 소유자·상태확인후구held3345920처리,운영경로보존·승격기록.
- [ ] epoch평가·최종step04099/PILOT_FINAL/정상exit/DONE검증.
- [ ] C2완료후B5/B6우선순위복구및검증된소스로B6잡정리. B5진행live확인.

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
