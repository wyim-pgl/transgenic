# MVP decoder connection — controller handoff

## Verified finding

Plan agent0a8a18f6 inspected B8803afb. Controller independently queried/read `mvp_tracks.rasterise_est`: evaluation explicitly supports `population='excluded_heldout', rn=None`. Builder retains all primary EST block/junction geometry, excludes panel/test only from training tile membership. Suspected missing panel artifact is **not established**; no reason to rebuild training DB or relax training filters. Remoteall50nonzero coverage remains unmeasured.

Call retainedrasterizer with genomicforwardsequence, centered30720window,is_rcFalse,keep_fraction1,aug_seed20260914+seed,omitRN. Channel7availability alone isnotbiologicalevidence; recordnonzero0–6 separately. Zero-supportwindows remainexplicit. ESTinput neverusesindependentONT/HiFiscoringevidence.

## Implementation dispatch (independent native branches from8803afb)

- `68bd8f3a-69d7-415` **mvp-panel-inputs**: newpreparepanelinputsdriver+CPUfixtures. FreshreadonlyDB/ref/panel-boundreceipt andfloat16[L,8]npyontrack per50loci, selectedmolecule/member/disposition/block/junctioncompanion, no trainingchanges. Outputcontract `notes/MVP_PANEL_INPUT_CONTRACT.md`; report `notes/MVP_PANEL_INPUT_REPORT.md` inagentbranch.
- `e188d65e-0125-44b` **mvp-panel-decoder**: finalcheckpointlearnedprojectionpreservingloader/seed123baseline/strictGSF→GFFnormalizer/350outputdriver. Newfilesonly. Report `notes/MVP_PANEL_DECODER_REPORT.md` inagentbranch. Finalinputartifactinterfacewillbealignedbycontrolleroncereviewed; noassumedscienceapproval.

## Binding decoder requirements

- Finalmodelconfig+aliasawaresafetensorsstrictload; **not weights_init0→8initializer** fortrainedfinals, asitrezeroesESTprojection.
- Baselineconcretesourcefrominitreceipt, zeroprojection/zerotracks. Learnedfinalpairusesexactsamecheckpoint/sequence/settingsonandoff.
- Window30720centeredgenomicforward. OfflinepinnedactualHyena/GFFv3tokenizers; onebaseonetoken,stripSEPasintraining.
- Deterministicbeam2,1returned,do_sampleFalse,cache,GSFgrammarv2/v3. A27cap8192TOTALincludingdecoderstart; requireEOS, no truncated-as-empty.
- PreserverawtokenIDs/grammarplan,validatecanonicalGSF,splitgene blocksbeforelegacyconversion; deterministicIDs,1basedinclusivegenomicGFF, explicitexon/CDSparents. Exactemptyvalid; malformed/missingiserror. Noautomaticisoformdedup.
- 50×3×2=300 plus50seed123baseline=350independentfiles matchingBproducertransport. Source/command/input/outputhashesnotproofexecutionbutnecessaryprovenance.

## Panel input 구현 도착 / 검증 범위

Panelinputcode`0f9f777`(base8803afb), agent27panel/rastertests+16candidatetestsPASS보고. Controllerexactarchive재실행 **25PASS/2환경실패**: archive에referenceauthorityGitobject없음,기존BAMsmoke의pysam없음. 전체27통과로보고하지않으며원인확인없이source수정하지않음. Commit-boundreference검증은배포시Gitobject제공/검토된별도고정authority배포가필요함; 파일hash자가진술로대체하지않음.

보고/계약main사본 `notes/mvp_execution_20260916/{panel_input_report.md,panel_input_contract.md}`; neutral `/tmp/mvp-panel-context-20260916/`에도계약/diff보존. 독립검토 `78134bb2-824f-412` dispatch, nativeGitworktree로spec/동작검증. 결과branch `notes/MVP_PANEL_INPUT_INDEPENDENT_REVIEW.md`에서회수예정. Decoderagent에midrun계약전달시도는이미completed라실패; 결과도착후controller가계약정합확인/필요시수정dispatch한다.

## Decoder 부분구현 도착

Code`668299a`(base8803afb). Controllerexactarchive **35testsPASS6.98s**, diffcheckPASS. Agent별도환경35+24subtests보고는controller결과와구분. Learnedprojection엄격복원/seed123baseline/beam2·8192TOTAL/GSF-GFF·350outputs구현. [보고서](mvp_execution_20260916/decoder_report.md), [diff](mvp_execution_20260916/decoder.review.diff).

현재내부panelbatchschema만소비하므로실제producer0f9f777의COMPLETE/manifest계약어댑터 **미통합**. 기존Bproducer8803해시차단만으로최신verifier검토대체안함; 수정64b6d5b/c024445검토후함께연결해야함. 독립검토 `4d7bf90c-bd40-41f` dispatch, neutral실제panel계약도전달. 결과branch `notes/MVP_PANEL_DECODER_INDEPENDENT_REVIEW.md`에서회수예정. 실제GPU/350예측미실행.

Panelinput독립검토9040e69 [panel_input.independent_review.md](mvp_execution_20260916/panel_input.independent_review.md) **PASS/no blocker**. 실제Gitworktree에서신규11tests+outputmutation2probes검증. Controllerarchive의Git실패는이전상태그대로기록하며gitobject존재환경에서ref검사통과경로확인. 저위험배포조건: sourceROOT에서 `git show 92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`가가능하도록 **Git실행파일+불변objectDB**를함께배포해야한다. plainarchive/workingmanifest복사만으로대체불가; 별도검토된authoritytransport없음. 생산/실데이터/decoder통합승인은아님.

## Decoder검토→통합수정 착수

검토3788bcd [decoder.independent_review.md](mvp_execution_20260916/decoder.independent_review.md): D1HIGHjunction3–6에coveragecap64오적용, D2MEDIUMfinalJSON부분쓰기잔재, D3HIGHexternalHyena코드/configidentity미강제. ControllerD1/D2소스확인. 실제panelCOMPLETEschema어댑터도필수.

통합담당`7896c130-8d88-4c4` nativebranch `mvp-decoder-integration-20260916`: reviewedB c024445위에reviewedpanel0f9f777와decoder668299a를격리통합한후세결함수정/실제COMPLETEmanifest소비/350end-to-endCPU검증. **main병합아님**. Producer블랙리스트는actualverifiedgates로대체하고runtimeidentity는actualpreflight/fileclosure로결속, bool자가승인금지. A의변경중코드대신git3b의identity원문/배포기록을neutral `/tmp/mvp-decoder-context-20260916/`에전달. R1/R2A학습수정과별개, 마지막통합시실측receipt호환확인필수. 결과branch `notes/MVP_DECODER_INTEGRATION_REPORT.md`.

## Usage후 decoder 통합 도착

Native통합code **e6950b7**, f4f1690대비정확7taskpaths, accidental.venv미포함/diffcheckPASS. Agentfull107tests427.40s보고, 350비어있지않은2gene출력→counts계약시험포함. 실제ref/GPU/GFFCompare는명시적대체fixture라과학검증아님. [보고서](mvp_execution_20260916/decoder_integration_report.md), [diff](mvp_execution_20260916/decoder_integration.review.diff).

독립전체통합검토 `8df082a6-cf6d-4d9` dispatch. Controller초기subset재실행은150초toolbudget에서8dots후timeout, 성공으로기록하지않음(잔여testprocess없음확인). 보고된427초전체비용에맞춰별도read-only검증담당 `38a14751-1668-481`가정확archive/Gitauthority객체접근/동일deps로full107suite를900초명령한도내실행중; 테스트내timeout변경없음. 실제결과예정 `notes/mvp_execution_20260916/decoder_integration_test_rerun.txt`.

P1job3163545는18:26PDT조회 **PENDING Priority**, GPU측정미시작. [조회](mvp_execution_20260916/p1/status_after_decoder.txt).

통합독립검토7f24616 [decoder_integrated.independent_review.md](mvp_execution_20260916/decoder_integrated.independent_review.md): **D1/D2closed, D3미완료**. I1HIGH현재projectmodelROOT를단순기록하여다른checkoutpreflight와불일치허용; I2MEDIUM실제completedwrapper+DONE대신rawmeasurement만소비; I3MEDIUMHF생성emptyinitializer/scaffold를snapshotimplementation과혼동하여정상환경거부가능. Controllerhelper/실제run_boundedwrapper형식확인.

수정담당 `55fba521-a4f9-46d` nativebranche6950b7 dispatch: 정확한model-relativeidentity/approvedprobehandshakepin, 실제completedreceipt+DONE/status/binding검증, 좁은HF생성scaffold구분/strictsourceclosure, operationalCUDAdevice강제. D1/D2/350panel계약변경없음. reportbranch `MVP_DECODER_RUNTIME_FIX1_REPORT.md` 예정. 107전체재검증담당은별도로원통합HEAD검증중이며수정HEAD승인으로혼동하지않음.

Controller별도검증담당38a14751원통합HEAD **e6950b7전체107testsPASS514.67s(exit0/wall517s)** 확인. [전체로그](mvp_execution_20260916/decoder_integration_test_rerun.txt). 이전150s초과는350orchestration시험단독205.93s등실제CPU시험소요로설명되며testtimeout완화없음. Python3.12.2 pdb -m pytest실행(barepytestbroken shebang은사용안함), archivecwd잔여프로세스없음. 이통과는I1–I3수정전CPU/stub범위이며실제GPU/ref/GFFCompare/현재수정HEAD검증이아님.

GPUqueue대기중병행TODO: **실제GFFCompare0.12.6 소형정답fixture** 준비담당`17c7ce75-b1c0-4d8` dispatch. Readonly원격binaryavailability/실제version확인+localfixture/분석/script, controller검토후2CPU4GB15min내실행계획. 중복isoform과terminalvariant는관측만하고정책임의선택하지않음. 보고예정 `MVP_GFFCOMPARE_SMOKE_PLAN_20260916.md`, code `notes/mvp_execution_20260916/gffcompare/`. 새agent원격쓰기/제출권한없음.

GFFCompare소형진단준비완료 [MVP_GFFCOMPARE_SMOKE_PLAN_20260916.md](MVP_GFFCOMPARE_SMOKE_PLAN_20260916.md). 실제pronghornbinary0.12.6읽기전용확인;16합성case/multi-chain·monoCDS·terminalvariant·strand·empty·duplicate실제비교계획,2CPU4GB15min. Controllerlocalselfcheck/syntax/3SHA재확인PASS, 실제비교미실행. 독립code/실행검토 `f32ce268-6fad-463` dispatch; 결과 `notes/mvp_execution_20260916/gffcompare/independent_review.md`. Author중복정책선택없음, actualtool진단후별도결정.

GFFCompare독립실행검토APPROVE후controller **pronghorn6165521제출18:55PDT**,1node2CPU4GB15min/no-requeue. Freshnamespace `$R/evidence/mvp_gffcompare_smoke_20260916`, 3source/scriptSHA및stagemanifest d269951e…8ff remotePASS, 직전동명잡없음. [제출receipt](mvp_execution_20260916/gffcompare/submission_receipt.txt). 실제16case결과미확인, duplicatepolicy/생물학적평가승인아님.

Decoderruntimefix **6366d8a** 도착, [보고서](mvp_execution_20260916/decoder_runtime_fix1_report.md); controller **28focusedPASS/1SKIP16.34s** (4.57.6미설치skip실제SIF인증아님), agentfull135PASS1SKIP보고. I1relativeprojectidentity+closeddecoderpin/I2실제wrapper+DONE/I3emptyHFscaffold구분/CUDAdevicegate. 독립재검토 `d3cf5c49-3405-439` dispatch, reportbranch `MVP_DECODER_RUNTIME_FIX1_REVIEW.md`. d99probehandshake하드pin은새772receipt호환별도확인필수, 자동같은것으로인정하지않음.

GFFCompare6165521실제job **FAILED1:0,2초**, externOOM0:125. 실제16비교산출물/해시생성후analyze의ordinaryraw tmap경로assert실패. [상태](mvp_execution_20260916/gffcompare/result_status.txt), [실출력](mvp_execution_20260916/gffcompare/real_run/). Rawmap은query옆case루트에생성되는데분석기는case/raw에서탐색. ControllerordinaryproducerTP1FP0FN0확인(전체PASS아님). 재실행없이보존결과전체검증/분석기좁은수정담당 `973fd983-69c2-47a` dispatch, 보고예정 `MVP_GFFCOMPARE_REAL_RESULTS_20260916.md`. 기대값완화/원본덮어쓰기/잡재제출금지.

Decoderfix1독립재검토9c43d4a [decoder_runtime.fix1_review.md](mvp_execution_20260916/decoder_runtime.fix1_review.md): **I1–I3수용/d99범위한정**. GPU최종772의3handshakehash가모두달라현decoder거부확인. 검사완화아닌선택된최종소스정합이필요.

최종격리소스통합담당`cf3f429c-1b79-48f` dispatch: base772위reviewedB eefe/2276/1a925/64b6/panel0f9/decoder668/e695/636순서, accidental.venv미포함. Decoderhandshake핀은reviewed772정확bytes로명시갱신, 실제772runner완료wrapperfixture+negative재검증, 하나의consumedsourceclosure로GPU/eval연결. Main병합/원격job변경없음. 결과branch `MVP_FINAL_SOURCE_INTEGRATION_REPORT.md` 예정. 진행중P1d99결과를최종sourcegate로오인하지않음.

GFFCompare기존결과복구보고 [MVP_GFFCOMPARE_REAL_RESULTS_20260916.md](MVP_GFFCOMPARE_REAL_RESULTS_20260916.md):16기대사례일치/236manifestfiles/34map검증. Controller새회귀10testsPASS33.358s. 재실행/원본수정없음; 원jobFAILED및externOOM기록불변. Actualfunction-levelsemantic수용전독립검토 `46a0f637-e18f-49e` dispatch, 결과 `gffcompare/local_recovery/independent_review.md`. 전체biologicalscoring/GPU/duplicate정책인증으로확대안함.

19:13PDT조회C1reconciled6165525RUNNING5m39s(19:07:28시작,기록시효내), P1GPU3163545PENDINGPriority. [상태](mvp_execution_20260916/p1/status_after_gffcompare_recovery.txt).

GFFCompare복구결과독립검토 [local_recovery/independent_review.md](mvp_execution_20260916/gffcompare/local_recovery/independent_review.md) **합성16case/실제function-levelsemantic검증수용**. 독립236hash/34maps/16handexpectations/좌표/duplicate·empty행태확인. 원6165521jobFAILED 및externOOM불변, 실제생물학reference/fullproducer/decoder/GPU/scientific승인아님. 이작업은재실행없이완료;남은frozenreferencecompatibility/duplicatepolicy분리.

## Runtime linkage

Aprobeowner8cd67cf9 notified to verify nonzero-projection/nonzeroEST beam2generationhook/cache/reorder in actualTransformers4.57.6. Zeroprojectiongreedyparityalonecannotproveevidenceforwarding. Use separateprobeinstance/restoreweights, no actualtrainingchange. ActualSIF/GPUverificationstillpending.

Fullconnectedpipeline waits for independentreview/interfacealignment+realtool/GPUchecks; localCPUtests alone aren'tdecoding/certification. PhaseCnotMVPprerequisite.
