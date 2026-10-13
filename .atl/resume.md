# Resume — 2026-10-12 C2 재개 결정성 복구

**최신 정본: [RESTART_HANDOFF_20261012](../notes/RESTART_HANDOFF_20261012.md).** 이 파일 → [todo](todo.md) → handoff → [quarantine](quarantine.md) 순서로 읽는다. 과거 handoff는 현재 상태가 아니다.

## 사용자 지시

- C2 우선으로 원인을 수정하고 검증한다.
- **문제 해결·검증 통과 후 추가 승인 없이 C2 본학습을 바로 시작한다(10-12 명시 승인).** 과학적 게이트는 유지한다.
- 전체 작업 푸시 및 이 네 문서 갱신 후 clear 가능한 상태로 보존한다.

현재 재개 작업: [컴파일 이력 복구 실행 기록](../notes/C2_REPLAY_PROGRESS_20261012.md). **4GPU 전체재개동일성3369694 PASS → 400단계검증3370048 제출·Priority대기.** 새controller **PID1545990**이45초간격감시중, phase`waiting_validation`/revision3/error없음. 운영정본`ops/c2_in_place_198ce14_20261012`; 중복초기화·재제출금지. 검증통과후사본제자리활성화→본학습4099자동연결. 원본200·진단256보존.

## 현재 상태

**10-13 14:28 PDT 조회 차단:** DeltaAI SSH가 `Permission denied`로 재인증을 요구했다. 현재 잡/컨트롤러 상태는 미확인이다. 아래 3370048 대기·PID1545990 정상은 **10-12 19:21 PDT 마지막 관측**이다. 사용자 터미널에서 `ssh -t pronghorn 'ssh dtai'` 로그인/Duo 갱신 후 live 조회부터 한다. 중복 제출·초기화 금지.

- **3364480 COMPLETED0:0**, 4GPU/gh150,39m17s. 동일 모델/버퍼/입력/RNG에서 컴파일 이력 재현→기존 손실과 일치; Dynamo만reset→cold logits와일치.12report독립검토PASS. 원인 확인 완료. **컴파일복구198ce14/로컬검증/독립리뷰 완료, NGC58시험57pass/1skip, 4GPU3369694 전체상태동일성PASS**.
- 앞선3361666: old/new 코드200→210 완전일치; 연속/재개210→220은609/770모델텐서불일치. 입력·난수복원실패는 아님.
- 최근3369694 종료18:51:30PDT COMPLETED0:0/gh118/1h05m57s. 400검증3370048은Priority대기(예상시작N/A). 이전3369338 FAILED1:0/32m39/gh063. 본학습보류. 원본200·진단사본256보호. controller3276938종료, journal blocked_determinism/jobs[].
- C2후속3345920 및B6 3345917/18 held. B5 3309786–90 대기. live상태를 다시조회하고 중복제출하지않는다.

## 코드·다음 행동

- `/tmp/b6-c2-recompile`, branch`b6-c2-recompile-20261011`, **198ce14**, clean/pushed. state/journal/trainer통합·캐시경로결속독립검토PASS; d7b1a37은수정기준.
- 구현브랜치는 main과별개. `/tmp`가없으면브랜치에서worktree재생성. 현재main은업무기록/기존미커밋내용보존용이다.
- main운영도구: production7266e14(33시험+NGC CPU33), controller6e5f0f7(52+WorkDir10시험), 연결시험ea699d5(3시험), 최종독립검토·host3.6/sharedFS영수증게시PASS. 원본200은그대로, 검증resumed410을in-place활성화하는v2방식. 실제운영controller PID1545990 동작확인, journalwaiting_validation.
- 다음: 새controller journal/watch.log/PID 및3370048 live확인. 전체400/410동일성·slot6718기록·저장공간등통과시controller가**추가승인없이C2본학습4099**연결. 물리적원본이동없음.
- 상세경로·검증결과·주의점·서버명·원격journal·잡ID는handoff에있다. 지금완료진단을다시돌리거나기존controller를그대로재가동하지않는다.

이전문서: [resume_history_20261012.md](resume_history_20261012.md).

원격보존: main스냅샷`8ef1871`과모든로컬작업브랜치푸시·SHA대조완료. [영수증](../notes/PUSH_RECEIPT_20261012.json). 최종문서tip은`git log -1`로확인.
