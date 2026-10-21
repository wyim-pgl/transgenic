# TODO — B6 파일럿 연속 실행 / Delta CPU / 전체 Graphify

정본: [resume](resume.md), [B6 진행 기록](../notes/b6_pilot_resume_20261020/PROGRESS.md). 기존 승인 범위의 다음 실행 가능한 항목을 추가 승인 없이 이어간다. 과학적·소유권·자원 게이트는 유지한다. 과거 상세 완료 체크리스트는 [10-20 기록](todo_history_20261020.md)에 보존했다.

- [ ] **10-21 우선순위: ccloop으로 현재 B6/C2 모델 학습·검증 완성.** 현재 모델의 미완료 단계만 교차 검토하고 다음 실행 가능한 작업을 수행한다. 네 종 Helixer 확장 목표는 사용자 일시중지이며 BUSCO/BLAST/OMArk/CDS/mRNA/exon/UTR 요구는 보존한다. [ccloop 진행 기록](../notes/ccloop_current_model_20261021/PROGRESS.md).
- [x] 현재모델ccloop 첫계획교차검토: Claude300초timeout 후Kimi대체검토 수락. 실제모델완료/게이트통과로해석하지않음.
- [x] DeltaCPU scorer23105376: 고정gffcompare0.12.6 바이너리SHA결속·실제x86/0GPU synthetic chain fixture PASS,7초/COMPLETED0:0. 공유C2final4099meta재확인.
- [ ] WP5 readiness: 고정300loci/pairedbootstrap/G1/G2와C2/B6fixed-step·C5·입력/스코어링분리·실행환경을결속한다. 기존MVP50전용평가기그대로사용금지. B6final4099미생성; 준비근거와부족한연결을현재모델진행기록에남긴다.

- [x] C2 컴파일 이력 복구198ce14, 실제4GPU3369694/3370048 정확일치 PASS, 본학습3379630의4099/final/epoch평가/DONE/accounting/controller complete 확인. 원본200·진단256 보존.
- [x] B5 preflight 실패3309786의 시험용 dangling symlink 원인 재현·정확한 소유 링크만 정리. 후속3309787–90 취소 확인.
- [ ] B5 전체 preflight·checkpoint/재개 조건 미통과. 새로운 B5 잡은 아직 제출하지 않는다.
- [x] B6 Task1 fresh0→410/snapshot400/newprocess400→410 evidence-on 증명도구 e2bee04. 같은 빈 상태 falseREADY 결함 수정·독립검토 PASS, 실제host3.6·NGC6contract+22proof PASS.
- [x] Task2 fixed B5epoch09–11 archive 설정·독립검토·nativefixture·3425279 제출. epoch09 verified/removed 영수증 및 protected12–14/latest/best 보존 확인. NVMe107.5GiB 실측으로100GiB 및 독립copy-peak 게이트 충족.
- [x] Task2 epoch09–11 전체 검증 이관/원본 정리 완료.3425279 COMPLETED0:0,1h59m31s,39.46GiB. protected12–14/latest/best 보존, NVMe133.83GiB. 추가 복구 잡 없음.
- [x] Task3 immutable proof release/staging/actualCPUchecks/소스·환경·liveidentity·storage 확인, 새4GPU 증명 **3425703** 제출.
- [x] Task3 실제 B6 GPU 증명3425703의 READY/비교 passed=true·모델/optimizer/scheduler/4rank RNG/compilehistory 정확일치·parent/batch/extern COMPLETED0:0 확인(3h22m26s). 10-21 11:33PDT [실측](../notes/b6_pilot_resume_20261020/LIVE_STATUS_20261021.json). C2 증명으로 대체하지 않았다.
- [x] Task4 독립 B6 controller/producer 구현8b13234 및 취소경쟁 수정2a48265, report50a7fac. 독립검토 PASS; 실제host3.6와NGC36/36 PASS. production release staging 완료.
- [x] Task4 실제proof3425703·소스/이미지/receipt/heldidentity에 결속된 설정 독립검토/native검사 PASS→init/watch PID1501642,waiting_validation,error없음.
- [x] Task4 B6 증명통과 후 구held3345918→3345917 취소·검증된410 in-place 활성화·본학습 **3429921** 자동 제출. Controller waiting_production/error없음.
- [ ] Task4 본학습3429921 accepted410→4099·평가/final/DONE/정상종료 확인. 10-21 11:33PDT 현재 PENDING/Priority·elapsed0으로 본학습은 아직 실행 전. 단일 감시 정상이며 중복제출하지 않는다.
- [x] Delta CPU 공식 사용법·cpu1core/1G/10min 읽기 전용preflight 준비. 로그인·공유NVMe/HDD 양방향R/W·파일정리·Python3.9 stdlibfixture1PASS 확인.
- [x] Delta CPU 사용자 승인통지→bilv-delta-cpu/1,000core-hours 실측→test-only PASS→실제preflight23104942 COMPLETED0:0/7초/1CPU/0GPU. 공유경로와helper해시 PASS. 이후 필요한CPU작업에사용가능; 완료아카이브중복실행없음.
- [x] `/graphify` 대규모 경고 후 사용자 **전체** 선택, 감지2,696파일. 코드1,688개AST 추출 완료.
- [x] Graphify 문서/논문/이미지1,008개130묶음 전체 읽기·SHA/범위 검증 완료. Gemini429 이후 Codex host 분석. 사용량 카운터가 제공되지 않아 실제 토큰 비용은 unknown으로 기록.
- [x] Graphify JSON748개 보완(733 구조 파싱+15 상세 fallback),2vendor C++ 부분 파싱 경고 기록.18,577노드/38,057관계/1,874군집. 잘못된 Markdown 링크·다중 관계 손실 수정과 독립검토 PASS, 브라우저/CLI/최종 연결 대상 검증 PASS. HTML/JSON/report/benchmark/manifest/cost 기록 완료.
- [x] B6 운영 기록 main a777258 및 구현 branch50a7fac 별도 atomic push·원격SHA 대조 완료. 구현은 main에 병합하지 않음.
- [x] Graphify 최종 산출물·아카이브 완료/최신 상태 main9ffb8fd 커밋·푸시·원격SHA 일치 확인. GRAPH_PUSH_RECEIPT.json에 기록. 후속 기록 정리 tip은 git log -1로 확인한다.
