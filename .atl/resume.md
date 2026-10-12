# Resume — 2026-10-12 C2 재개 결정성 복구

**최신 정본: [RESTART_HANDOFF_20261012](../notes/RESTART_HANDOFF_20261012.md).** 이 파일 → [todo](todo.md) → handoff → [quarantine](quarantine.md) 순서로 읽는다. 과거 handoff는 현재 상태가 아니다.

## 사용자 지시

- C2 우선으로 원인을 수정하고 검증한다.
- **문제 해결·검증 통과 후 추가 승인 없이 C2 본학습을 바로 시작한다(10-12 명시 승인).** 과학적 게이트는 유지한다.
- 전체 작업 푸시 및 이 네 문서 갱신 후 clear 가능한 상태로 보존한다.

현재 재개 작업: [컴파일 이력 복구 실행 기록](../notes/C2_REPLAY_PROGRESS_20261012.md). **코드188bf3e·검증도구 독립리뷰PASS, 4GPU proof3369338 제출(10-12 15:48PDT), 직후Priority대기**. live상태/로그부터확인하고중복제출금지. legacy후보1..15+3915, 실제과학게이트는아직검증중.

## 현재 상태

- **3364480 COMPLETED0:0**, 4GPU/gh150,39m17s. 동일 모델/버퍼/입력/RNG에서 컴파일 이력 재현→기존 손실과 일치; Dynamo만reset→cold logits와일치.12report독립검토PASS. 원인 확인 완료. **운영복구구현/로컬검증/독립리뷰 완료, NGC/4GPU3369338 검증 중**.
- 앞선3361666: old/new 코드200→210 완전일치; 연속/재개210→220은609/770모델텐서불일치. 입력·난수복원실패는 아님.
- 최근live(10-12 15:49PDT) proof3369338 Priority대기, RUNNING없음. 본학습보류. 원본200·진단사본256보호. controller3276938종료, journal blocked_determinism/jobs[].
- C2후속3345920 및B6 3345917/18 held. B5 3309786–90 대기. live상태를 다시조회하고 중복제출하지않는다.

## 코드·다음 행동

- `/tmp/b6-c2-recompile`, branch`b6-c2-recompile-20261011`, **188bf3e**, clean. state/journal/trainer통합·캐시경로결속독립검토PASS; d7b1a37은수정기준.
- 구현브랜치는 main과별개. `/tmp`가없으면브랜치에서worktree재생성. 현재main은업무기록/기존미커밋내용보존용이다.
- 다음: 3369338 live/log/종료코드부터확인→NGC회귀·4GPU체크포인트정확일치→검증사본400재개→**승인재요청없이C2본학습4099**.
- 상세경로·검증결과·주의점·서버명·원격journal·잡ID는handoff에있다. 지금완료진단을다시돌리거나기존controller를그대로재가동하지않는다.

이전문서: [resume_history_20261012.md](resume_history_20261012.md).

원격보존: main스냅샷`8ef1871`과모든로컬작업브랜치푸시·SHA대조완료. [영수증](../notes/PUSH_RECEIPT_20261012.json). 최종문서tip은`git log -1`로확인.
