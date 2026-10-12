# seed123 단일 실행 + 평가 + EST 실자료 연결 — 2026-09-17

## 사용자 착수 지시
> 1. MVP 설정을 seed123으로 정합화  2. 평가 연결 마무리  3. HSP·phase·A11 실제 자료 연결 ###시작

현재seed123만개발/학습/품질확인,추가시드는연기. 이세항목은지금실행승인됨. 이미결정한평가규칙/HSP길이/충돌방향/A11규칙재승인요청없음. A46전체생산/병합/푸시/외부발신은승인범위밖.

## 불변조건
- 기존B5seed123 chain/checkout/checkpoint 보존. MVP초기화는고정epoch05,mutablebestepoch06으로자동변경금지.
- 기존b660P1seed1/옛multi-seedreceipt를123으로재명명하지않음. 불변원격release/패널6165628/DB/실패기록덮어쓰기금지.
- 새development-single실행은기존multi-seed최종통계/350출력실행과구분. baseline B5seed123과finetuned MVPseed123은**seed숫자가같아도서로다른model role/checkpoint**이며키충돌을막아야함.
- seed123단일50패널: paired off/on100 + baseline50=150예측의명시된개발계약을검토. 최종3seed집계/decision을제조하지않음.
- phase완료를MVP학습선행조건으로추가하지않음. 기존실제GPU6gate/예산/출처검증유지.
- EST실자료연결은보존raw재사용,전체blastx/원DB재계산없음. 완전metadata없는A11을인증하지않음;진짜입력표본진단과생산phase를구분. 큰hash/scan/추출은검토된compute잡에서만.
- Advisor전략호출은client/model-version오류,응답/승인없음.

## 실제 시작 상태
- Seed/eval통합worktree `/tmp/mvp-seed123-integration-20260917`, branch `mvp-seed123-integration-20260917`, base **9603e20**. Graftbuild완료. main사용자변경보존.
- EST기검토base **fca911e**, worktree `/tmp/est-approved-contract-20260917`. sourcebinder작업은별도범위로후속배정.
- [11:17PDTlive](seed123_execution_20260917/initial_live_gpu.txt):3163986(seed1옛P1)Priority대기/사용0, B5seed123다음3149356Priority대기/후속Dependency. 새job/기존job변경없음.

## 단계/담당
- [x] **1설정조사:** `e5df57cc-6bf1-44e` — training/config/readiness/probe/finalizer seed전수제약과새개발profile인터페이스. Readonlyplan,보고SEED_SCOPE.md.
- [x] **2평가조사:** `4d8c4d94-f6a3-4ac` — decoder/scorer/panelconsumer·150개출력·baseline/MVP동일seed충돌·이미승인된missingness규칙/ONT연결. Readonlyplan,보고EVAL_SCOPE.md.
- [x] **3실자료조사:** `2b53a5a9-6f0c-43f` — actualraw13/DNA/protein/BAM/A11metadata/routeA의sourcebinder와bounded실자료추출계획. Readonlyplan,보고EST_REAL_SCOPE.md.
- [x] Controller세인터페이스합의/파일소유권분리→4구현담당배정. [인터페이스정본](seed123_execution_20260917/INTERFACE_AND_OWNERSHIP.md). Plan담당은readonly라보고파일미작성,반환내용을controller가이정본으로보존.
- [ ] 구현중: trainingprofile **0f6198b9**, evalroles/scoring **fae82185**, immutablepanelcompat **5d76780c**, actualESTsamplebinder **b0cb4695**. 코드변경대상/타worktree는인터페이스정본참조. 독립commit/원격제출권한없음.
- [ ] Controllerdiff·focusedtests/실자료표본검증 → 독립검토/필요수정.
- [ ] 실제환경실행필요시검토된새immutablepackage/예산/livequeue후controller단일제출. 기존P1교체는소유권/여전히PENDING/사용0확인및새실행안검토후;RUNNING이면임의취소/산출물재라벨없음.
- [ ] 다음TODO/진행·차단·commit/jobID를여기와resume/todo에연결.

## Controller 실제입력 추가 확인
- [Sitalica실경로/크기/raw13prefix](seed123_execution_20260917/est_fixed_path_preflight.txt), [smallprovenance/commands](seed123_execution_20260917/est_source_metadata.txt):선택종양source raw/filteredprotein/query/report/BAM존재. 원자료읽기는boundedprefix/metadata만,큰hash없음.
- [compute runtime](seed123_execution_20260917/compute_runtime_preflight.txt), [samtoolsidentity](seed123_execution_20260917/est_samtools_identity.txt):기존Python3.11.14/samtools1.21-htslib1.23경로/hash확인. 원BAMproducer1.3.1과읽기tool1.21구분해기록. 새remotejob미제출.
- [PacBio6파일prefix](seed123_execution_20260917/pacbio_header_recheck.txt):각16KiB상한읽기에서SRR번호header만있어ccs/subreads제품정체성확정불가. 다운로드완료사실은유지;이관측만으로HiFi수용/배제/재다운로드결정하지않음.
