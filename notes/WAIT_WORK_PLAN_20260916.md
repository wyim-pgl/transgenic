# GPU 대기 중 1–4번 병행 실행 계획

> **최신 정전복구 갱신 — 2026-09-17 01:17 PDT:** [복구정본](POWER_RECOVERY_20260917.md). 아래Task2의중단agent·배포대기는과거. Eval9603배포독립검토수용→새CPU6165628완료→실제50mapping/참조/150artifact·수치재구성검증수용,기존6165559보존. ONT/panel동일authority/reference/panelSHA를[newledger](power_recovery_20260917/runtime_ledger.json)로결합. GPU3163986Priority대기/사용0(00:34관측);B5기존체인보존. externOOM·fullONT/decoder/scientific게이트미해결. 고정GPUb660소스변경없음.

> Agentic workers: 기존 검토된 소스를 사용하고, 각 산출물의 시험·독립검토 후 controller가 원격 실행한다. 작업별 결과는 아래 전용 디렉터리에만 기록한다.

**Goal:** GPU 대기 시간을 이용해 후속 실행안, 실제 평가 입력, 독립 장독 평가, 재현·복구 준비를 실제 실행 가능한 수준으로 완성한다.
**Architecture:** 최종 소스 `b660bf1`은 변경하지 않는다. 네 담당이 서로 다른 notes 경로에서 배포·검증 도구와 결과를 만들고 controller가 결합한다. 원격 쓰기/제출 담당은 controller 한 명이다.
**Tech stack:** Python/stdlib, 기존 DuckDB·PyTorch 도구, Git 객체, SSH/Slurm, minimap2/samtools/seqkit.
**Spec:** `revision/protocols/B6_MVP_PILOT_v1.md` 승인 범위, `notes/MVP_TRAINING_PRIORITY_20260916_1545.md`, `notes/MVP_DECODER_CONNECTION_20260916.md`.

## Global constraints

- source b660bf1: source pin555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44, decoder pinb53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb.
- 기존GPU3163986/B5체인/EST6165525를 수정·중복제출하지 않는다. 3163545는미할당취소된과거진단이다.
- P1실측통과전후속GPU제출금지. 모든진단누적5node-hours상한유지; 계획/CPU시험을실측PASS로표시하지않는다.
- 학습250updates/12warmup/seeds1,2,3, v3/290/8192,8ESTchannels,350decode출력계약불변.
- panel추론은excluded_heldout/rn=None. training membership/filter변경없음. ONT/HiFi는독립채점용이며모델입력에넣지않는다.
- 중복isoform/HiFi정책/withholding/HSPphase정책은임의확정하지않고확정된부품만진행한다.
- 대형hash/DBscan/정렬은compute에서수행. source원본/실패출력보존, 새출력exclusive, 자동retry없음.

## 확인된 시작 상태

21:14PDT `[initial_live_status.txt](wait_work_20260916/initial_live_status.txt)`: GPU3163986PENDINGPriority/사용0, 구3163545CANCELLED/사용0, C1감사6165525RUNNING2h06m. 실제상태는이snapshot 이후바뀔수있으므로제출직전재확인한다.

## Task 1 — P1 이후 GPU 실행안

**Files:** `notes/wait_work_20260916/gpu_followup/`의계획·명령생성기·Slurmtemplate·unittest.
**Consumes:** 실제3163986완료wrapper+DONE와S0prepared/assets/tiles/immutablecode경로, exact b660프로브API, 실제job회계.
**Produces:** 실행순서/동일source를유지하는모드별boundargv/새scratch·receipt경로/예산ledger.
- [x] P1필수산출물검증, step0→generation→bf16→4rank→resource의순서·모드별command생성/조건부runner작성.
- [x] P1없음/FAILED/source/DONE/시간·예산오류거부시험·Slurmtemplate구현. Controller18testsPASS0.309s, syntax/6SHAchecksPASS.
- [x] Seed별binding및별도training/resource예산미선택을명시. 현재도구는seed1P1기준이며seed2/3의별도6receipt필요·5nodeh내가능성미확정; PASS복사금지.
- [x] 독립검토F1 timeout오류수정/재검토수용: 정확int 0<timeout<=outer-120,26testsPASS1.119s/syntax/SHAledgerPASS, `gpu_followup/fix1_review.md` ADDRESSED/no newfixbreakage.
- [ ] G1/G2실행조건대기: 실제P1완료/S0산출물에맞춘mountargv·immutable실행release검토, 최신accounting/no-overlap/resourceauthorization. 구조계획수용이잡승인아님.
- [ ] Controller독립검토후P1성공시만해당bounded후속실행.

## Task 2 — 실제 패널 평가 입력

**Files:** `notes/wait_work_20260916/panel_inputs/`의Git-authorityexporter/소형시험/CPUjobtemplate/inputinventory.
**Consumes:** Git92d8020...의정확한manifest객체, 실제고정FASTA/GFF/DB/panel/candidate, b660panelproducer.
**Produces:** 검증가능한불변Git객체전달, 실제50패널참조호환성/중복구조관측/EST0–6채널support와7availability분리receipt.
- [x] Readonly실제FASTA/GFF/DB경로·크기·runtime METADATA확인. 원본5Git객체만전달하는authoritytransport로고정Git조회보존; CPU용CPython3.10DuckDB휠별도격리/해시기록, 기존환경설치없음.
- [x] 실제referencecompat/50heldoutESTproducer CPUjob4CPU16GB1h작성. Controller9testsPASS22.032s/syntaxPASS. Transport는`panel_inputs/final_release`만사용(이전draft금지).
- [ ] 독립source/object/runtime/deployment검토 `11e5fcbd-5fd0-4b2` 진행. 실제CPU Torch1.13좁은import경로미실행, compute실패시보존/중단. Finaldecoder NumPy/DuckDB정확version/path조건별도.
- [x] 독립검토APPROVE후canonical새release의검토된7payload+transport/원본Gitobjects확인·안전extract·RO source/site, **6165556제출21:50PDT**(4CPU16GB1h). [제출](wait_work_20260916/panel_inputs/submission_receipt.txt).
- [ ] 6165556FAILED127/0초:Singularity3.6.1 `:rw is not a valid bind option`,실제data작업전/outputsempty. 실패보존, 수정담당d1d315b7가전용outputbind의명시:rw3문자만기본RW로바꾼별도`final_release_bindfix1`작성. Controller3narrowtestsPASS1.372s/syntax,6기존payload불변. 새transport1883df3c…18c0a, 원검토자11e5fcbd범위재검토중(`bindfix1_independent_review.md`). 원본/data/gpfs:ro유지. 범위재검토APPROVE후controller별도새namespace/payload해시확인하여 **6165559제출22:01PDT**(4CPU16GB1h), 원6165556실패경로불변. [제출receipt](wait_work_20260916/panel_inputs/bindfix1_submission_receipt.txt). 자동재제출아닌확인된실행오류의검토된수정시도이며실제6165559COMPLETED0:0/12m46s,externOOM별도. **50/50ESTraster게시/zero지원0**,window별선택molecule합3443(전역unique아님). **Referencecompat0/50**,restrictedGFFheader16B만생성/missingpanelgeneAthNNN. ID/rootcause조사완료 `REFERENCE_ID_DIAGNOSIS.md`: panel은builderkeyAthNNN, rawGFF는original*.TAIR10; **기존corpus gene_key_map.gene_id_original을누락**한caller오류. 128KiB실GFFprefix로internal키headeronly/original키정상기제재현,50실map행은compute검증전. Controller71_b6_panels의originalID조인확인후 `87d8e000-d17f-447` eval-onlybranchfromb660에공유mappinghelper+양callers/strictbijection·geometry·sourcehash시험구현dispatch. GPU/P1b660/DB/원본GFF불변, 좌표/접두사heuristic금지. Evalfix구현 **9acd7db**/report **66f130d** 도착, production변경은mvp_produce_counts.py만; src/train/config/panel/probe불변controllergit확인. Sharedmap/helper+수정CPUwrapper tasknotes/reference_mapping/run_panel_cpu.py, 새evalsourcepin44078a.../decoderpinc0e8c1...; 기존wrapperb660hardcodedguard는새패키징시재결속필수. Agent98PASS1SKIP보고, controller첫focused120s명령timeout(결론없음/잔여process없음); 별도검증담당336b15ff가정확66f130d snapshot747trackedblobs대조후원6suite **98PASS1SKIP544.09s/exit0** 재검증완료(`reference_map_test_rerun.txt`). 이는late-publication수정전HEAD의CPU범위결과이며WAL결함을해소한판정아님. 독립검토09d18ba는mapping로직수용/Important1건: 결과직렬화중WAL생성시성공receipt발행가능. 발행수정 **9603e20**/report **1de033a** 도착:WAL/hash전후stat·직렬화후검증·closedpending→atomicreceipt 및CPU두완료경로수정. Controller신규30testsPASS404.32s/소스불변git확인; agentfull128PASS1SKIP보고. Scope재검토8508c4b **원Important late-WALclosed/no newblocking**, independent32tests+원350재현거부확인, pin재계산일치. Actual50data수용아님. 로컬새evalCPUrelease준비 `2bad22d0-34ca-4e8`도병행,원래3b660guards를evalcommit/component/새inventory로명시repin하되sourcefixreview수용전달완료, 새deploymentrepinreview전제출금지. Source/train/GPU원본불변, newevalpackage는수정재검토후. 새eval코드로새panelpublication필요, 원6165559manifest수정금지. Rawref부재/duplicate없음으로해석금지. DB재구축/훈련필터완화/중복정책선택없음.

## Task 3 — 독립 장독 평가

**Files:** `notes/wait_work_20260916/longread/`의역할·출처계약/driver접속계획/소형fixture/jobtemplate.
**Consumes:** A-ONT1 canonicalroles/actualFASTAhash/고정reference/panelwindow, 기존scoring-onlydriver/orientation/junction/pairedscorer.
**Produces:** 원본입력→orientation/정렬→junction→채점receipt연결과좁은실행안.
- [x] CanonicalFLIC ONT3run의b1_validation_only역할/실경로·stat/기록MD5/toolversion확인, HiFi미확보는미확정/음성아님으로표시. Fresh대형hash는compute로유지.
- [x] 출처gate/orientationpilotwrapper/계약·로컬시험작성, controller11testsPASS1.349s/syntaxPASS.
- [x] 독립pilot검토 `longread/independent_review.md` 조건부APPROVE. COMPLETE만성공아님: 종료회계/최종stdout/FAILED부재/receipt+사후compute검증모두필요.
- [x] **21:40PDT6165555제출**, SRR28341943의10000sampleorientation만4CPU8GB30min/no-requeue. 신규release/out canonicalancestor확인·13파일hashPASS·live동명잡없음. [제출receipt](wait_work_20260916/longread/submission_receipt.txt). Fullalignment/평가아님.
- [x] Pilot결과독립검증/Controller로컬재검증완료:16outputs28775072B 재hash·9Gitpins·fullstdout=receipt·COMPLETEdigest·10000FASTA/QNAME·sense2679/5008=.534944/Wilson[.521108,.548726]일치. [결과](wait_work_20260916/longread/RESULT_REVIEW.md).
- [ ] **운영상PILOT_COMPLETE차단:** alloc/batchCOMPLETED0:0 2m28s이나externOOM0:125미해결. OrientationFAIL은use_ufFalse관측이지구조증거음성아님. minimappeak1.732GBvsSlurm51MB를함께보존. 관리자/site증거없이는cleanaccounting으로격상금지; fullalignment/helper/최종bootstrap연결은추가검증필요.
- [ ] Controller검토후작은bounded검증부터실행; 검증전에장시간array/HiFi대량fetch자동제출없음.

## Task 4 — 재현·운영·native crash

**Files:** `notes/wait_work_20260916/repro/`의source/job/artifactledger·복구runbook·crash진단기록.
**Consumes:** 기존실제Git/sourcepins/잡receipt/회계/테스트·segfault로그, immutable최종source.
**Produces:** 출처가있는종합inventory와실패유형별복구절차, nativecrash재현범위·환경/한계.
- [x] immutable파일/문서snapshot구분·jobowner·예산·no-overlap ledger 작성. Controller25capturehash/100Gitfiles/86sourcepin/13decoderpin재검증PASS.
- [x] P1stage전/후timeout·cache/import·OOM·receipt불일치의비파괴복구runbook 및실행가능localpreflight검증완료.
- [x] 기존whole-repo crash로그·알려진환경보존/원인분리. 최소predecessor/core증거없어blind재실행0회, 필요한nativebacktrace/환경기록명시.
- [ ] Nativecrash원인확정은미해결. 실제잡결과가나오면runtimeledger를별도snapshot으로추가.
- [x] ExternOOM site증거요청초안 `repro/SLURM_EXTERN_SUPPORT_DRAFT.md` 작성; 외부전송/티켓생성없음. VictimPID/cgroup/kernel/slurmstepd/epilog증거없이는원인확정안함.

Task4 결과: `notes/wait_work_20260916/repro/{ledger.json,RUNBOOK.md,CONTROLLER_CHECK.md}`. 스냅샷수용이며GPU/scientific승인아님. crash frame의정확b660line141은ids/mask변환,tracks는142라는정정도원본보존상태로기록.

## 최신 결과 관측

22:12PDT [status_after_c1_acceptance.txt](wait_work_20260916/status_after_c1_acceptance.txt): 수정panelCPU6165559RUNNING10m34s/로그오류없음, GPU3163986PENDINGPriority. Panel결과관측담당 `fb7d0aa4-eeb0-443`가현재잡에한정해최대3600초/120초이상간격read-only상태확인 후smalloutputs검증/실패원인기록. 신규job/원격쓰기/무한monitor없음. 보고예정 `panel_inputs/RESULT_REVIEW.md`.

## 담당 배정 — 실제 병렬 착수

| Task | Agent ID | 전용 디렉터리 |
|---|---|---|
|1 후속 GPU 실행|`b4fb7471-a2ef-4b5`|`notes/wait_work_20260916/gpu_followup/`|
|2 실제 패널 입력|`f455c118-a55e-4f1`|`notes/wait_work_20260916/panel_inputs/`|
|3 독립 장독 평가|`5bdc6480-44f1-48f`|`notes/wait_work_20260916/longread/`|
|4 재현·복구/native crash|`69ea074b-3bbd-47d`|`notes/wait_work_20260916/repro/`|

네담당모두같은toolbatch로dispatch완료. 각자단계별계획확정/로컬구현·시험·필요한readonly관측을진행하고, 원격쓰기/잡제출은하지않는다. Agent결과를controller가확인해독립검토후가능한작업부터실행한다.

## 연결과 수용

| 관계 | 검사결과/역할 |
|---|---|
|1→2/3|GPU학습입력과heldout평가/장독증거를섞지않음. 평가코드검증은P1대기중가능|
|2→3|동일고정참조/panelwindow/rootauthority사용, exporter산출물계약을공유|
|1/2/3→4|각job/receipt/sourcehash를공통ledger에기록하되산출물원본을수정하지않음|
|각Task자체|작업별코드/스크립트는한담당만작성, 시험/정확한기록/독립검토로끝남|

네Task모두계획확정과로컬실행준비/readonly실측을즉시시작한다. Controller는결과도착순으로검토·수정·작은실행을이어가며, 계획서만작성하고끝내지않는다. 실제원격제출은해당스크립트검토/예산/livequeue조건충족후한다.
