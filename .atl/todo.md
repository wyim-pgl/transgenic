# TODO — C2 컴파일 이력 복구 (2026-10-12)

정본: [handoff](../notes/RESTART_HANDOFF_20261012.md). **검증통과후추가승인없이C2본학습시작**. 과학적중단조건은유지한다. 과거todo는[history](todo_history_20261012.md)로분리했다.

## 현재 실행 순서

- [ ] clear후현재브랜치/HEAD/worktree·liveSlurm·quota확인. 3364480은완료됐으므로중복제출금지.
- [ ] 컴파일이력복구설계완성: 전역event위치별모든rank의실제companion descriptor/hash,train/eval view/grad context,checkpoint결속. rank별독립replay/DDP불일치방지. event-only압축은별도검증.
- [ ] legacycheckpoint200bootstrap명시화: 원본checkpoint/drawprefixSHA,최초tier입력뿐아니라rank3row3915(labels4501)등기존특수화이력포함. 관측하지않은ledger라고정직하게표시하고별도증명.
- [ ] 재현실패시험→최소구현(`/tmp/b6-c2-recompile`,기준d7b1a37). parameter/전체bufferregistry/RNG/pendinggrad정확복원,optimizer/scheduler/step불변. latest/epoch/eval누락금지.
- [ ] local·NGC회귀/오류주입·독립검토. 큰소스변경후선택worktree에서graftbuild.
- [ ] 실제4GPU연속/재개전체checkpoint bitwise proof. original210loss3.2567708492279053와의역사적차이도해결. 단일forward일치만으로통과하지않는다.
- [ ] 원본200보존한검증사본에서280이상/400저장·다음재개확인,여유공간재측정.
- [ ] **통과즉시C2본학습4099시작(추가승인불필요)**. 기존controller재사용금지,새소스/검증receipt/중복방지journal. 소유자·상태확인후구held3345920처리,운영경로보존·승격기록.
- [ ] epoch평가·최종step04099/PILOT_FINAL/정상exit/DONE검증.
- [ ] C2완료후B5/B6우선순위복구및검증된소스로B6잡정리. B5진행live확인.

## 완료

- [x] 자동이관코드3b4eb0d:양서버69시험+실제fault3cases. B5epoch01–08 99.06GiB pronghorn검증이관완료.
- [x] 원본C2실패3345919확인·checkpoint200보호·후속C2/B6hold.
- [x] 종료처리수정3f81fb6(NGC56pass/1skip),compile후보d7b1a37(NGC74pass/1skip).
- [x] 3360978실패원인재현,3361110실190loss일치+108경계PASS.
- [x] 3361386을손실차이발견후의도적정상저장종료:진단256,원본200그대로.
- [x] 3361666: old/new동일,연속/재개609텐서불일치·입력/RNG동일증명.
- [x] 3364480: cold/primed/reset12report검증·독립리뷰,컴파일이력원인분리확인. 운영수정완료는아님.
- [x] 재시작4문서갱신·역사보존. `intend.md`는`intent.md`정본연결.
- [ ] 사용자의전체푸시: main현재작업커밋및모든로컬작업브랜치원격보존,원격SHA검증(진행중;결과는handoff Push기록).
