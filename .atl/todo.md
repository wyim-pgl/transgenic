# TODO — B6 파일럿 연속 실행 / Delta CPU / 전체 Graphify

정본: [resume](resume.md), [B6 진행 기록](../notes/b6_pilot_resume_20261020/PROGRESS.md). 기존 승인 범위의 다음 실행 가능한 항목을 추가 승인 없이 이어간다. 과학적·소유권·자원 게이트는 유지한다. 과거 상세 완료 체크리스트는 [10-20 기록](todo_history_20261020.md)에 보존했다.

- [x] C2 컴파일 이력 복구198ce14, 실제4GPU3369694/3370048 정확일치 PASS, 본학습3379630의4099/final/epoch평가/DONE/accounting/controller complete 확인. 원본200·진단256 보존.
- [x] B5 preflight 실패3309786의 시험용 dangling symlink 원인 재현·정확한 소유 링크만 정리. 후속3309787–90 취소 확인.
- [ ] B5 전체 preflight·checkpoint/재개 조건 미통과. 새로운 B5 잡은 아직 제출하지 않는다.
- [x] B6 Task1 fresh0→410/snapshot400/newprocess400→410 evidence-on 증명도구 e2bee04. 같은 빈 상태 falseREADY 결함 수정·독립검토 PASS, 실제host3.6·NGC6contract+22proof PASS.
- [x] Task2 fixed B5epoch09–11 archive 설정·독립검토·nativefixture·3425279 제출. epoch09 verified/removed 영수증 및 protected12–14/latest/best 보존 확인. NVMe107.5GiB 실측으로100GiB 및 독립copy-peak 게이트 충족.
- [ ] Task2 나머지epoch10/11 실제 이관 완료 확인.3425279 실행 중,전체완료 아님. 타임아웃/partial이면 재계획 전에 현재manifest/verified/removed 상태를 읽고 기존wrapper blind retry 금지.
- [x] Task3 immutable proof release/staging/actualCPUchecks/소스·환경·liveidentity·storage 확인, 새4GPU 증명 **3425703** 제출.17:54PDT Priority 대기. 구3345918/17은 held 유지.
- [ ] Task3 실제 GPU 전체상태/입력/compilehistory/evidence/3tier 정확일치와 정상 Slurm 종료/READY 확인. C2 PASS를 B6 PASS로 대체하지 않는다.
- [x] Task4 독립 B6 controller/producer 구현8b13234 및 취소경쟁 수정2a48265, report50a7fac. 독립검토 PASS; 실제host3.6와NGC36/36 PASS. production release staging 완료.
- [x] Task4 실제proof3425703·소스/이미지/receipt/heldidentity에 결속된 설정 독립검토/native검사 PASS→init/watch PID1501642,waiting_validation,error없음.
- [ ] Task4 GPU증명통과후 accepted410→4099 자동연결·평가/final/DONE/정상종료 확인. 아직 GPU 본학습 미시작.
- [x] Delta CPU 공식 사용법·cpu1core/1G/10min 읽기 전용preflight 준비. 로그인·공유NVMe/HDD 양방향R/W·파일정리·Python3.9 stdlibfixture1PASS 확인.
- [ ] Delta CPU1,000core-hours 신청 활성화/CPU Slurm association 대기. accounts exit60(no association). 사용자 처리통지 후 실제 계정·잔액 확인→CPU preflight→zeroGPU 작업 전환. 아직CPU잡 미제출.
- [x] `/graphify` 대규모 경고 후 사용자 **전체** 선택, 감지2,696파일. 코드1,688개AST 추출 완료.
- [ ] Graphify 문서/논문/이미지1,008개130묶음 전체 분석. Gemini429 일일한도 중단, Codex host workers 작업 중. 누락·부분읽기·토큰사용량 불명은 정직하게 기록한다.
- [ ] Graphify AST15JSON fallback/2vendorC++ 부분파싱 경고, 병합·무결성 진단·군집명·graph.html/graph.json/GRAPH_REPORT.md·benchmark·manifest/cost 기록.
- [ ] main 운영 기록/재시작4문서와 구현 브랜치 별도 커밋·푸시·원격SHA 대조. 구현은 main에 병합하지 않는다.
