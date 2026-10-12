# Todo — 현재 실행 목록 (2026-09-16 12:51 PDT)

최신 실제 재개 상태/agent/worktree: [복구 기록](../notes/RESTART_RECOVERY_20260916_1251.md). 미커밋 2차 코드 백업 후 3담당 구현·CPU검증·수정범위 독립검토 완료. A/B/C 모두 **준비용 부분구현 수용**, 실제환경·전체트랙 완료 아님.

정본: [resume](resume.md) · [상세 인수인계](../notes/RESTART_HANDOFF_20260916.md) · [재발 방지 §1n](quarantine.md). 과거 체크리스트는 [무손실 archive](archive/todo_before_20260916.md)로 이동했다. 아래만 현재 실행 목록이다.

## 최신 — 대기 중 1–4번 실행, 21:14 PDT

- [x] [4트랙계획](../notes/WAIT_WORK_PLAN_20260916.md)확정·4담당동시착수: 후속GPU/boundbudget, 실제패널입력/Gitauthority, 독립ONT장독평가, 재현·복구/crash.
- [ ] 각담당실행script/시험/readonly실측도착 → controller확인·독립검토 → 조건충족한boundedCPU작업부터실행. GPU후속은P1PASS이전제출금지.
- [ ] 공유referenceauthority/panel계약을2→3담당간결합, 모든실행receipt를4번ledger로연결. 고정b660코드임의변경금지.

## 최신 usage 복구 — 18:03 PDT

- [x] [복구정본](../notes/USAGE_RECOVERY_20260916_1803.md): usage해결후GPU통합/decoder통합/P1배포3담당중단점에서재개. 원본부분코드보존.
- [ ] 저장된partialcode검증완료→독립통합검토→boundedP1실행. Decoderauto-save의.venv/caches배포금지.

## 최우선 — EST8채널 학습 critical path

- [x] **HIGH실제MVPtrain/valid v3라벨연결수정** code981489d/reportd99b1f8, controller92focused시험/독립R1R2재검토수용. 실제MVP학습아직없음.
- [x] probe4ranklocal실패collectivecoordination+TRANSGENIC_RUN_DIR격리수정범위검토수용(d99b1f8).
- [ ] 검토된d99b P1단일프로세스실환경진단배포script검토→실행. 담당7cb8d01d준비중. Resource추가fix/통합검토→6gate→학습은별도.

- [x] 사용자 실제GPU→MVP학습/디코딩평가 연결지시, [우선실행정본](../notes/MVP_TRAINING_PRIORITY_20260916_1545.md). 실제probe·배포실측·평가producer3담당dispatch.
- [x] 단일최종소스 b660bf1(실행362eb899): GPU6mode+train/eval/decoder통합 독립검토수용. Agent599범위시험/controller46연결시험통과,4.57.6검사1skip. 전체repo nativecrash는미해결/전체green아님.
- [x] 최종b660배포패키지독립검토/해시검증, old3163545ownedpending취소·사용0확인, **finalP1 3163986제출20:21PDT**(4GPU30min). B5체인불변.
- [ ] 3163986실제S0/P1결과확인 → 나머지5gate/예산기록 →6gate readiness →3seed250updates. ESTphase완료를선행조건으로추가하지않기. 자동재제출금지.
- [ ] finalizer/참조identity결속 및rawGFFcountproducer → 해당계약으로decoder구현 → 실제checkpoint후300예측/평가.

## 0. 재개·작업 배분

- [x] 최신 MVP `f847cb8`/blastx`950cf4c` branch 존속, main`9a8c8e1` 미병합 확인.
- [x] 저자: 실제환경검증/학습·평가/bootstrap/blastx후속 전체를 서브에이전트로 진행 승인. [범위](../notes/b6_parallel_execution_authorization_20260916.md).
- [x] **1차 발견 워크플로 완료(09-16 오후 PDT):** A/B/C 담당 3명이 읽기전용 현황+설계 수집, 트랙별 독립검토 3건 전부 APPROVE_WITH_FIXES. 종합·must-fix·저자결정 8건은 [synthesis](../notes/b6_three_track_discovery_synthesis_20260916.md). Slurm제출/원격쓰기/worktree생성 없음.
- [x] **2차 구현 복구 dispatch:** 기존 A/B/C worktree의 미커밋 코드 백업 후 담당3명 동시 재개. agent/path는 복구기록 참조.
- [x] 이번준비용부분구현: 각담당CPU시험 → controller diff/재시험 → 독립검토/수정재검토종료. 각코드HEAD와한계는복구기록최상단.
- [ ] 후속로컬구현: 실제GPUprobe프로그램, BrawGFF-count/출처결속/디코딩, Cpilotreceipt·의미론·전체투영. Slurm 제출·checkpoint pinning·A-HiFi·C2/C3 production 범위는 synthesis의 저자결정 8건 대기.
- [ ] 작업별 격리 worktree·불변 snapshot·출력 디렉터리·잡 소유자/예산 기록. DeltaAI 제출자는 A 담당 하나, pronghorn B/C 출력경로 분리. 제출 직전 liveSlurm 확인, 중복금지.

## A. 실제 환경 검증 → launch-readiness → MVP 학습

- [x] Readiness/staging/bounded-wrapper준비도구 및A1/A2수정 `f6b171f`(보고서HEAD `ac8ac2d`); agent219범위시험/controller66도구시험통과, [독립재검토승인](../notes/recovery_20260916_1251/A.fix1_review.md). 실제GPUprobe미구현/실측없음.

- [x] 09-16 10:41PDT DeltaAI접속/잡 읽기전용 확인: seed123 `3131537` RUNNINGgh0864h39m, bestepoch05 eval0.6394507885; repo_seeds8f049eb, 후속Dependency. [실측](../notes/restart_status_20260916.txt).
- [x] 12:51PDT live조회:3131537 RUNNINGgh0866:49:49,bestepoch05,후속Dependency. 원격변경 없음.
- [ ] 실제 제출 전 다시 live조회. 기존seed123 job/checkout/체인 보존;456/789 연기 유지.
- [x] Controller초기화선택:concreteepoch_05,세seed공통,mutablebest추종금지. [선택/pin기록](../notes/MVP_INITIALIZATION_SELECTION_20260916.json), 사용자선택진행위임에따름.
- [ ] 선택checkpoint/sidecar독립staging 및compute에서실제해시재검증. 선택기록만으로ready아님.
- [ ] 실제SIF 경로/바이트SHA/컨테이너 내Torch·Transformers·Accelerate·tokenizer/Hyena정체성 대조. 현재기록 NGC26.08/Torch2.14; 로컬CPU시험은Torch2.6으로 다름.
- [ ] 검토된 사전시험: 실제tokenizer/Hyena, 3tile tiers step0동일성, 트랙off/on·RC/N, 생성hook/KVcache parity, bf16/Inductor, single-node4rank/4visible nonMIGGPU/UUID, 메모리·시간 실측. 코드/Slurm스크립트 검토 후 bounded smoke만 제출.
- [ ] 모든 필수시험PASS와 자원예산·코드/데이터/체크포인트/SIF해시를 묶은 **실제launch-readiness** 작성·검증. 현재기록없음. placeholder/CPU시험만으로 인증하지 않기.
- [ ] readiness통과 뒤 고정250optimizerupdates/12warmup × MVP seeds1,2,3 실행. 기존B5체인과 자원충돌/중복제출 확인. 완료 marker/원자적산출물/해시/actualhardware 확인.

## B. 디코딩·독립 장독 평가·bootstrap

- [x] 준비용scoring/window/junction/bootstrap부분구현 및 수정1차 `2276fcb`; controller95tests+21subtests통과, 독립재검토B1–B3해소. [검토](../notes/recovery_20260916_1251/B.fix1_review.md). 전체B완료아님.
- [x] pair/finalizer/seed/code/init/ref정체성및rawGFFreference-count부분producer `64b6d5b`(reportc024445), controller142tests+21subtestsPASS, 독립R1–R5수정검토수용. 350출력계약유지.
- [x] Panelinput생성기 `0f9f777` 독립검토PASS. 신규11시험별도재검증,controllerarchive25PASS/2환경실패기록유지. Gitauthorityobject배포필수.
- [x] 실제GFFCompare0.12.6 합성16case/function-level진단결과 독립수용. 원job6165521은분석기경로오류로FAILED유지, 기존236파일/34map의읽기전용복구로기대치확인. [검토](../notes/mvp_execution_20260916/gffcompare/local_recovery/independent_review.md).
- [ ] 실제frozenreference호환성·중복isoform정책·독립read증거결속, 최종panel/decoder/runtime통합검토와실제실행. 아직최종판정차단.

- [ ] `B6_MVP_PILOT_v1.md` 승인본과 현행 scoring코드·데이터역할/패널50의 계약 대조; 구현·시험계획 독립검토.
- [ ] A-ONT1/A-HiFi 실제소스·정렬준비 확인, 입력증거와 채점증거 분리. 검증전용 역할을 학습게이트 우회로 사용하지 않기; scoring전용드라이버로 별도 실행.
- [ ] MVP체크포인트 산출 후 승인패널50 전부 디코딩. sequence/evidence on/off 동일조건·체크포인트/생성설정·출력해시 기록.
- [ ] reference-dependent F1와 독립장독 splice구조지지 구분 평가. 독립증거미확보를 음성결과로 간주하지 않기.
- [ ] 승인spec의 paired bootstrap/세seed집계/STOP→GO→INCONCLUSIVE 및 withholding조건 구현·검증·실행. primary=증거양성타일별 pureweightedCE off−on 동일가중평균, composite병기, δ0.005nats 불변. F1/구간/재표본 단위는 원문대로.
- [ ] 결과·한계·사용된고정입력·런타임/통계코드해시 보고. 실제결과 전 유효성/성능주장금지.

## C. blastx 전체 감사 → HSP 수용 → genomic phase 투영

- [x] 15:30PDT 사용자 네항목 ‘지금진행’ 요청. [새 실행정본](../notes/EST_EXECUTION_20260916_1530.md). 감사/OOM·실제HSPfixture·phase계약3담당dispatch. 검토후bounded실행진행; 과거일괄실행승인대기를이작업의중단이유로반복하지않음. A46학습투입승인과는분리.
- [ ] 새담당실측/설계 확인 → controller스크립트검토/livequeue → compute감사/fixture제출 → 결과기반수용·투영구현/검토/실행. 원본blastx재실행금지.

- [x] C1준비용auditor/CPUfixture및수정2차 `5c7a0aa`; controller119시험통과, 독립검토원5건+FASTA회귀해소. [재검토](../notes/recovery_20260916_1251/C.fix2_review.md). 실제raw감사/C2/C3완료아님.
- [ ] Pilot 제출/log-layout receipt복원·검토. 현도구는미확보를명시적UNRESOLVED FAIL로유지; 기존tmp/preflight/execution.lock정합도별도필요. blastx재제출사유아님.

- [x] 9종 양source DONE 관측. remaining8 `6164049_0–7` allocation/batch COMPLETED0:0 재확인. Sitalica6163851은 이전완료. **blastx재제출불필요.**
- [ ] per-source provenance/DONE/raw13/timing, speciesbatch.exitcode, allSlurmsteps 전체대조·해시목록 작성. 큰raw스캔은 검토된compute작업으로 실행, 로그인노드 과부하금지.
- [ ] 승인 routeB HSP≥30aa 수용규칙/frameshift·BTOP 해석 확정·구현·단위/실제fixture검증. rawHSP수를accepted phase evidence로 부르지 않기.
- [x] 정확한screenedquery/BAM/CIGAR→genomic좌표·codonordinal진단라이브러리 `5c46c75`, controller176시험·독립수정검토승인. [검토](../notes/est_execution_20260916_1530/geometry.fix1_review.md). 전체phase투영아님.
- [ ] 실제HSPeventproducer/routeAphaseadapter·충돌정책·분자집계/raster연결 및실제fixture검증. 동종/하위분류군 사전배제·lineage한계·경로A불일치→0 규칙 보존.
- [ ] 검토된수용/투영잡을 별도namespace에 실행, 전종카운트/QC/provenance로 최종수용. 기존raw와실패기록 덮어쓰지 않기.

## 완료 — 반복 구현하지 않기

- [x] 사용자 metric결정: 기존학습 composite 유지, paired primary pureweightedCE+composite병기.
- [x] MVP data/model/dataset + metrics/init/runtime/finalizer/CLI/topologygate 누적 **f847cb8**. 컨트롤러153CPUtests PASS, 전체연결검토·최종재검토PASS/APPROVE. [검토](../notes/b6_mvp_integrated_review_20260915.md). GPU인증 아님.
- [x] fullDB6164118 build+QC+기술수용 감사완료. DBsha`ed2d0eef…`,overlap0,valid1442/1447. [수용](../notes/b6_mvp_technical_candidate_20260915.json)·[inventory](../notes/b6_mvp_candidate_inventories_20260915.json)·[receipts](../notes/b6_mvp_candidate_receipts_20260915.json) 해시/정규형/연결검증완료. 생산승인false.
- [x] 실제 RN목록 train7635/valid1447/test1388 정합, BAM/member/molecule disposition정합, finalDB독립재해시일치.

## 별도 미해결/승인 대기

- [ ] Slurm `.extern OUT_OF_MEMORY0:125` 원인 조사. successfulbatch/payload와별도; 해결됐다고 기록하지 않기.
- [ ] A46 전체 최종승인·동결, fullB6/C0/secondary정렬감사/phase채널본학습 범위는 별도. 탐색MVP승인과혼동금지.
- [ ] main통합/푸시: 미커밋protocol/panel/docs/graft설정보존, 적절한diff검토 뒤 결정. 이번세션미병합/미푸시.
- [ ] B5 seeds456/789 재개시점은저자결정; 현재연기. seed123는실행중읽기전용추적.
- [ ] 논문/위키/spec업데이트, A37두arm보고, #61평가분모/AtRTD3/PacBio등 이전연구결정은 [과거todo](archive/todo_before_20260916.md)에서 최신protocol/issue와대조 후처리. 오래된체크박스만보고재실행금지.

## 운영 유의

- 이전 advisor는한도소진; 이번12:51호출도클라이언트버전오류로실패. 새로운자문없음. 독립검토를별도로진행하며 advisor결과를만들어내지않기.
- 이 파일은 현재작업만 유지한다. 과거실행기록SHA `2f25fe55…24c`는 archive에보존; 상세branch/테스트/실패수정이력은 handoff와controller보고서에있다.
