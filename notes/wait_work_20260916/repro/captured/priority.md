# EST 학습 우선 실행 — 2026-09-16 (PDT)

## 사용자 최신 지시

실제GPU검증→MVP3seed×250updates 및 디코딩·평가를 서브에이전트로 연결하여 진행 가능한 만큼 진행하고, **EST training에 가장 필요한 것을 우선**한다.

## 우선순위/범위

1. 주경로는 이미기술수용된EST8채널MVP다. blastxHSP/6phase후처리는별도향후증거확장으로 병행하며MVP선행조건으로추가하지않는다. 기존8채널/학습loss/250updates/12warmup/3seed/판정threshold불변.
2. 당장최대결손은실제Hyena/tokenizer/GPU probe프로그램과실측이다. 준비wrapper와CPU시험만으로GPUreadiness만들지않는다.
3. 데이터와학습을연결하려면같은불변초기화weights/sidecars·실제SIF·offlineassets·DB/panel/code를결속해야한다. 기존seed123B5를변경하지않고선택한완료checkpoint를별도복제한다.
4. 증거효과는step0zero-projection/off동일성, EST입력/RC/N정합, 증거경로finitegradient로먼저확인. 이후학습3seed동일조건on/off평가로검증. 실제연산실패를tolerance/eagerfallback으로숨기지않는다.
5. 평가경로는훈련과병행구현하되생성checkpoint전엔실제결과를만들지않는다. 출처결속/참조검증/rawGFFcountproducer를먼저만들고그계약에decoder를연결한다.

## 연결된담당

| 담당 | agent | 작업/보고서 |
|---|---|---|
| 실제GPUprobe구현 | `8cd67cf9-760d-45e` / mvp-real-probes | Aworktree `/tmp/b6-p2-readiness-1535112` baseac8ac2d. 6실측probe+CPUnegative검증. `notes/MVP_REAL_PROBES_REPORT.md` |
| 배포/초기화실측 | `093b37a5-17a5-482` / mvp-deploy-ops | liveSlurm/B5완료epoch/offlinecache/SIF/DB/전송·quota·예산경로read-only조사. main `notes/MVP_DEPLOY_OPS_20260916_1545.md` |
| 평가producer·출처결속 | `2a90c190-f09a-488` / mvp-score-provenance | Bworktree `/tmp/b6-p2-scoring-1535144` base2276fcb. finalizerreceipt검증·referenceanchor·rawGFFcounts. `notes/MVP_SCORE_PROVENANCE_REPORT.md`; 연결계약 `notes/MVP_EVALUATION_INPUT_CONTRACT.md` |

모두실제dispatch완료. source작성은worktree별한명. remote writer/submitter는controller. 중복스테이징/재제출방지를위해제출직전livequeue확인. 기존EST감사/fixture/geometry담당과자원·출력경로분리.

## 배포실측 도착 / 초기화 선택

[배포보고](MVP_DEPLOY_OPS_20260916_1545.md): 실제SIF buildreceipt는Torch2.14/CUDA13.4/Transformers4.57.6/Accelerate1.15.0. CPU시험환경과다르므로실제probe에서import/launchparser/cachegeneration호환확인이핵심. Probe담당에게전달완료.

Controller는사용자의선택·진행위임하에현재완료best인 **concreteepoch_05**를세seed공통초기화로선택: [선택기록](MVP_INITIALIZATION_SELECTION_20260916.json). 실제복제/compute재hash미실행, readiness아님. 이후B5best변경을추종하지않음. 이선택이틀리면영향은더새checkpoint실험을위해별도재검증/학습이필요하다는점이며자동교체하지않는다.

DeltaAI에는23.5GBcorpus와11.1GBSIF이미있으므로재전송불필요. 추가WAN전송은1.069GBacceptedESTDB+소형companions, 현지복제는checkpoint2.09GB/HF218MB. 남은quota약151GB(관측시), B5성장분을남겨재확인. B5seed1233131537당시RUNNING,원본유지.

Ops제안GPUsmoke최대5nodehours(20GPUhours), 최초0.5nodehour/P4약2nodehours는아직제출되지않은계획이다. 실제프로그램독립검토와제출전운영상한기록이필요. 전체3seed학습비용은P4실측후확정하며이숫자를학습예산승인으로사용하지않는다.

## 평가 producer 부분구현 도착 / decoder 연결점 조사

Bcode`1a9253d02b0202106e29587ff689bb1e2e1f3e72`, reportHEAD`8803afb`, 기존Bworktree. 기존finalizer/candidate검증재사용·commit-boundAthalianaFASTA/GFF·rawGFFreferencecount생성·350prediction(300+50baseline)전송계약구현. Controller **115tests+21subtestsPASS20.32s**, agent별도16candidatetests보고. 실제GFFCompare/독립장독연결미검증,duplicateisoform정책미확정으로차단될수있음. [보고서](../notes/MVP_SCORE_PROVENANCE_REPORT.md)는Bworktree내파일이며main동명파일아님. 독립검토 `9713f4a8-388d-4b4`, 결과main `notes/recovery_20260916_1251/B.producer.independent_review.md`.

Decoder준비계획담당 `0a8a18f6-0641-4cc` dispatch: generator/GFFnormalization/checkpoint/interface연결과 **학습DB의panel제외가추론on증거를비우는지** 우선확인. 학습retainedfilter를풀어패널누수시키지않으며별도EST패널추론증거artifact필요여부조사. 독립ONT/HiFi채점증거는입력에섞지않음. 아직decoder구현/평가실행완료아님.

Decoderplan완료: 제외panelgeometry는기존DB에보존, `rasterise_est(population='excluded_heldout',rn=None)` 경로존재를controller직접확인. trainingfilter변경/DB재구축불필요(실제50coverage는측정전). [연결정본](MVP_DECODER_CONNECTION_20260916.md)에요구사항/dispatch기록. 별도nativebranchesfrom8803afb에 panelinput담당`68bd8f3a-69d7-415`, decoder담당`e188d65e-0125-44b` 실제dispatch. Finallearnedprojection보존·8192TOTALbeam2·350manifest/strictGFF테스트진행. GPU담당에nonzeroprojectiongenerationbeam2연결검증추가지시.

Bproducer독립검토 REQUEST CHANGES: High소스closure/package목록불완전검증·초기화audit실제내용미결속; MediumID공백legacyparsertranscript소실·빈파일생물학적zero허용·finalconfigJSON미검사. Controller소스대조후원담당`2a90c190-f09a-488` 수정1차재개(8803afb기준). 실loaderpositivefixture+5negative요구. 350전송계약유지, decoder담당에빈파일/ID/최종가중치검증함정전달. GPU학습경로변경없음. 결과예정수정보고/재검토.

Bproducer수정1차code`64b6d5bf2494bb5b23e061c6c0917b4bbafcdbb9`, reportHEAD`c024445` 도착. Controller통합 **142tests+21subtestsPASS42.58s**(새27+기존115), diffcheckPASS. 350interface불변. 원reviewerhandle만료로새검토자`2904db09-00b1-41e`에게원5지적/수정diff/보고서전달하여범위재검토중; 결과main `notes/recovery_20260916_1251/B.producer.fix1_review.md`. 아직최종수용전, 실제GFFCompare/independentlongread정책미해결.

## 실제 GPU probe 도착 / 마지막 자원 gate 구현

AprobeHEAD`3b62306`(source1639c7e+b1e5e62): 5실CUDAmode구현, resource_budget미구현을명시. Controller **81도구/probe시험PASS45.58s**. Agent최근전체233PASS/1기존2ranksetupfixturetimeout, 해당시험포함82focused재통과; 전체green으로기록하지않음. 독립검토 `69776642-0ce8-47b`(결과 `notes/mvp_execution_20260916/real_probes.independent_review.md`). 마지막실측자원mode담당 `83dfcd31-783f-41d` 별도nativebranchfrom3b62306:3tiersteady측정/실제pairedvalidation/진실한probe직렬화/보수적250예산, fake250finalizer금지. 실제GPU실행아직없음.

## 실제 데이터 staging — 16:31–16:32 PDT

Controller새DeltaAI namespace `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/` 독점생성. 16:31PDTlive B5seed1233131537RUNNINGgh08610h29m, 후속Dependency유지. Quota349G/500Gsoft확인. [생성receipt](mvp_execution_20260916/staging_create_receipt.txt).

AcceptedESTDB를pronghorn→dtai rsync로 **1,068,773,376B 전송완료**(16:32:05–16:32:32PDT), 목적지 `staging/evidence/fullc0lite.db`; 원본/기존corpus/SIF/B5변경없음, 삭제0. [전송receipt](mvp_execution_20260916/evidence_transfer_receipt.txt). 이는크기/전송완료확인이며 **compute SHA재검증/immutableinputs승격전**. 기대SHAed2d0eef…와실측대조전학습사용금지. Code/companion/checkpoint/cache스테이징과실제SIF검증은다음단계.

Bproducer수정범위독립재검토 `B.producer.fix1_review.md` **R1–R5 ADDRESSED / 새Critical·Important없음**. Code64b6d5b(reportc024445)를부분평가producer로수용, 350계약불변. Decoder통합시base8803의약한verifier대신이수정포함필수. 실제GFFCompare/중복isoform규칙/독립readprovenance/정책/실제GPUdecode미해결유지.

## 최우선 학습 correctness 결함 발견 — GPU 실행 HOLD

`real_probes.independent_review.md`: HIGH **v3config에datasetdefaultv2/2048label**이연결됨. 새probe뿐아니라실제MVP `mvp_cli_runtime.py598–603` train/valid도동일결함. Controller코드확인. 다중gene/empty라벨손상과긴라벨절단가능, CPU153/219기존시험으로못잡았음. 실제MVP학습미실행이므로영향받은MVP결과없음. Fix담당8cd67재개: boundconfig의gff_vocab_version전달+vocab/cap일치검증, realtokenizer다중gene/empty/>2048 및실제trainwiring회귀시험. **이runtime수정은동결학습규칙변경아니라기존config연결오류수정**이며B5원격trainer불변.

MEDIUMranklocalfailure→collective불일치도고침: setup/loader/batchcontract/postcompilecheck를기존phaseguard로조율, CPU분산negative시험. inheritedTRANSGENIC_RUN_DIR로B5경로write가능성도깨끗한env/reject로차단. 모든probe/codebinding재생성필수. Resource담당83dfcd31에수정의존관계전달, 별도브랜치에서중복수정하지않음. R1/R2수정검토전4GPUreadiness실행금지(P1도가능하면고친후진행).

Resource프로브branch `4bd97399` 도착, report [resource_probe_report.md](mvp_execution_20260916/resource_probe_report.md). Controllerexactarchive **126testsPASS58.99s**, agent126+21subtests보고는별도환경. diffcheck에는testEOFblank1건있어clean주장하지않음. 3tier×(cold1+steady2),fullvalidpaired,probe-onlyserialization/전체소요포함보수적250추정/budget고정검증구현. 실제CUDA미실행, 기본R1/R2fix와합치기전실행금지. independentreview `9692a8e0-0589-4e3`, 보고branch `notes/MVP_RESOURCE_PROBE_INDEPENDENT_REVIEW.md`. A원담당에signature/merge순서전달.

실제v3라벨/collective수정code`981489daa2ddbb040ab0056299655899dd2b28b2`, reportHEAD`d99b1f8` 도착. Controller **92focusedtestsPASS130.55s**(실제rank1fault/도구/신규라벨포함), diffclean. Agent245scopedPASS644s보고후마지막offlineimport순서수정뒤92focused재통과; controller전체245재실행아님. 원reviewhandle만료로새검토`7d0d2dad-0758-4b3` dispatch, 결과 `real_probes.fix1_review.md`. Resource검토자에변경signature/guard전달; source통합은아직전. 모든codebinding/preflight재생성필수.

Resource독립검토898f117 [resource_probe.independent_review.md](mvp_execution_20260916/resource_probe.independent_review.md) REQUEST CHANGES: HIGH동일progress경로에fresh-onlywriter재호출로2번째기록항상실패; MEDIUMpairedmetric입력검사rank분기; LOWschemaint검사누락/EOF공백. GPU실행HOLD. 수정담당 `4166b9ce-eeec-4d8` nativebranchfrom4bd97399 dispatch: immutablewriter변경없이순번progresssnapshot, pairedforward전local계약집단검증/실CPUrank1fault, 정확정수형검증. A라벨/one_update소유자fix는별도수용후통합, 중복수정금지.

A실제라벨/collective수정독립재검토 [real_probes.fix1_review.md](mvp_execution_20260916/real_probes.fix1_review.md) **R1/R2 ADDRESSED/새Critical·Important없음**, fixedHEADd99b1f8의P1/P2는정확한staging/binding재생성조건부diagnosticGO. 6gate학습ready아님.

실제P1배포준비담당 `7cb8d01d-d86c-42c` dispatch: 기존전송DB를compute검증·epoch05독립copy·두HFcachelayout·acceptedtrain실RN3tier/RC/N선택·actualSIFcode/bindings생성→boundedreal_tokenizer. 로컬script/plan만작성/readonlyops, remote제출controller만. P1예정상한1node4GPU30min(0.5nodehour), 이후총5nodehours진단운영상한내; 전체3seed학습예산승인아님. 코드patch없이검토된d99b실행, resourcefix완료를P1준비의불필요한의존성으로삼지않음. 보고예정 `MVP_P1_DEPLOYMENT_PLAN_20260916.md`, scripts `notes/mvp_execution_20260916/p1/`.

Resource修正1 `f479172fd64856902f8999a1d74e975f50622a13` 도착. Controllerexactarchive **186testsPASS83.68s**, diffcheckPASS. RP1순번immutableprogress/RP2validate_paired_batchpurehelper추출+forward전집단검사/19int필드검증/EOF수정. 메트릭계산불변이나metrics.py소스변경으로decoder/evaluator와나중동일snapshot통합필수. 재검토`4885ca28-37e8-426` dispatch, reportbranch `MVP_RESOURCE_PROBE_FIX1_REVIEW.md`; 아직Aownerfix와미통합/실행HOLD.

Resourcefix독립재검토bd0f06c [resource_probe.fix1_review.md](mvp_execution_20260916/resource_probe.fix1_review.md) **RP1–4 ADDRESSED/no newblocking**. Reviewerfocused74tests+metricAST동등성확인. 두reviewedbranch통합담당 `c98ab1c9-3313-4e0` dispatch: natived99위resource4bd+f479, sharedfactory/modelcfg/Accelerator단일ownership/collectivecalls결합검증, main미병합. 보고branch `MVP_GPU_INTEGRATION_REPORT.md` 예정. P1배포담당에게final통합codebinding이바뀌면d99receipt재사용불가를통지; 가능하면최종검토snapshot으로한번만실행하여중복GPU비용회피, currentd99실행시진단범위임을명시.

P1배포담당usage복구완료: [MVP_P1_DEPLOYMENT_PLAN_20260916.md](MVP_P1_DEPLOYMENT_PLAN_20260916.md). 정확d99b85filesrelease, accepted3JSON별도pins, freshperjobS0+P1/outer1700s/30min4GPU상한. Controller **9unitPASS7.368s**, shellsyntax/7ledgerentriesPASS, ledgerSHA`34a094c321a59ae4e0613c96d6797a2fdca32284682a2ed07074ca5ee52d203b`. 독립실행검토 `491b3ac0-bee8-4ad`, 결과 `notes/mvp_execution_20260916/p1/deployment.independent_review.md`. 아직원격release복사/제출없음; dynamicqueue/quota직전재조회필수. Combinedsource통합결과미도착으로현재release는d99진단범위고정.

## 실제 GPU S0+P1 제출 — 18:15 PDT

Deployment독립검토 [deployment.independent_review.md](mvp_execution_20260916/p1/deployment.independent_review.md) **APPROVE exactd99boundeddiagnostic**. Controller [실행범위](mvp_execution_20260916/p1/controller_execution_scope.json) 기록후새release소형전송/remoteledger8체크PASS, livequeue동명없음·quota350G/500Gsoft확인. B5후속3131538Priority대기및chain변경없음.

**DeltaAI job3163545 제출**, 1node4GPU64CPU/30min/no-requeue, 최대0.5nodeh/2GPUh. source d99b1f8, ledger34a094c3…203b. [실제제출receipt](mvp_execution_20260916/p1/submission_receipt.txt). Exclude gh062,gh086,gh093. 아직RUNNING/완료/P1PASS조회전. Remote`staging/p1_release/submission.txt`가jobid정본; 자동재제출금지.

S0입력hash/copy/cache/tier선택→real_tokenizer단일진단, TRAINING_DONE/6gate생성없음. 실패시증거DB가perjobinputs로이동했을수있으므로Startedmarker삭제/무조건재실행금지. source가달라질최종통합readiness의대체receipt로사용금지. 전체training예산승인아님.

GPUresource통합완료 **772c38c3a428acedebb0ba3b1bef37f2777a0f35**(544bcb0복구에서3taskpaths추가수정), [보고서](mvp_execution_20260916/gpu_integration_report.md). Agentfull370CPUtestsPASS941.96s보고; Controllerexactarchive **11integrationtestsPASS18.72s**, diffclean/env미포함확인. 실제cfgv3factorycall/양datasetguard/단일Accelerator/집단helper/cleanup통합. 독립최종통합검토 `bdbf9b31-8796-478` dispatch, reportbranch `MVP_GPU_INTEGRATED_REVIEW.md`. 3163545는기존d99P1진단이며새통합snapshotreadiness아님; 새binding필요.

최종GPU통합독립검토617e2984 [gpu_integration.independent_review.md](mvp_execution_20260916/gpu_integration.independent_review.md): **combinedcallsiteACCEPT/no newblocking**, exact772c38c의boundeddiagnosticstaging조건부가능. 실제GPU/6gate/학습승인아님. 상속nonblockingI1 `probe_identity.schema_version`이true/1.0도1과동일시하는엄격형식결함재현; actualproducerint1이며hash우회없음. 후속엄격화TODO로기록, 현재pendingd99job/source를수정하지않음.

18:43:31PDT원격조회 [status_after_gpu_review.txt](mvp_execution_20260916/p1/status_after_gpu_review.txt): C1baseline6165461RUNNING2h56m, DeltaP1 **3163545PENDINGPriority**. 중복잡제출/취소없음. P1d99결과를기다리는동안최종772c38c용새codebinding재생성/후속budget계획가능하지만기존receipt재활용금지.

## 최종 통합 source 도착 — b660bf1

SourceHEAD **b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6**, executable362eb899. ReviewedGPU772+reviewedB/panel/decoder/R1source를하나로결속. Decoder지원handshake772정확pins/누락bootstrap의닫힌audit목록추가. Agentfullscoped **599PASS1SKIP64subtests1105.48s**보고, 초기전체repo시도segfaultexit139는원인미확정/전체green아님. Controller **46핵심연결testsPASS1SKIP36.68s**, diff/env미포함확인. [보고서](mvp_execution_20260916/final_source_report.md). Indepfinalreview`6fb47be4-03e7-43b` 진행, reportbranch `MVP_FINAL_SOURCE_INDEPENDENT_REVIEW.md`.

FinalsourceP1별도패키지준비담당 `d0d51c9a-0239-402` dispatch, local `final_p1/`+`MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md`. 기존d99release/3163545job변경없음; 최종review후에도새job동시/무조건추가금지, oldP1상태/priority손실·중복비용판단후controller만진행.

20:08PDT [조회](mvp_execution_20260916/p1/status_after_final_source.txt): P1 **3163545여전히PENDINGPriority(누적GPU사용0)**, stagingevidence1068773376B존재. C1reconciled6165525RUNNING1h01m. 기존대기잡cancel/재제출아직없음.

최종독립검토0a19b3a [final_source.independent_review.md](mvp_execution_20260916/final_source.independent_review.md): **exactb660boundeddiagnosticexport/freshbindings ACCEPT/no newCriticalImportant**. Ancestry74보존files/86sourcepin/13decoderpin독립확인,35focusedPASS1SKIP+schema1PASS. Whole-reposegfault실로그확인·미해결유지,실행재발시중단. Main병합/학습ready아님. FinalP1패키지담당d0d51에조건부출시수용전달; source파일변경없음/새preflight필요.

최종b660P1패키지준비완료 [MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md](MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md), new `final_p1/release`100files/codearchive1832960B, ledgerSHA **5b36bdf09a8b8eb548147654e23d6ad59fe18542642a05eb720f22be102fdfb8**. Controller11unitPASS15.032s/syntax/7ledgerchecksPASS. Oldd99releasebytes/job불변. 독립배포delta검토 `f501b6f9-398c-47a`, 결과 `final_p1/deployment.independent_review.md`. SourceACCEPT와deploymentACCEPT구분, 아직newrelease원격전송/잡제출없음. old3163545owner/state/freshqueue+remainingbudget/evidencelocation검증후에만단일후속결정; overlap/autorecovery금지.

## 최종 source P1로 대기잡 교체 —20:19–20:21PDT

Source+finaldeployment 독립ACCEPT후controllerfreshowner/state확인:old3163545=wcyim/PENDING/elapsed0/AllocTRES없음/S0marker없음. `scancel --ctld --state=PENDING --user=wcyim --name=mvp8-p1-0916 3163545`로 **본인대기진단만취소**, 최종CANCELLED/0초/할당없음재확인. [전증거](mvp_execution_20260916/final_p1/pre_replacement_state.txt), [취소receipt](mvp_execution_20260916/final_p1/old_pending_cancellation.txt). B5체인미변경. 구대기순위경과손실가능하지만구소스중복GPU비용회피를선택.

검토된100fileb660finalrelease를새 `staging/final_p1_release`에전송, ledger **5b36bdf0…dfb8**와7payloadsha전부PASS. 기존evidence제자리/oldmarker없음, livequeue동명/구job없음, quota350G/500Gsoft확인. 현재runningownednode없어stalegh086제외해제, persistentgh062/gh093제외유지.

**신규DeltaAI3163986제출20:21PDT**, finalsourceb660,1node4GPU64CPU30min/no-requeue,0.5nodeh/2GPUh한도. [범위](mvp_execution_20260916/final_p1/controller_execution_scope.json), [실제제출receipt](mvp_execution_20260916/final_p1/submission_receipt.txt). 이전MVP할당사용0, cumulative5nodeh상한그대로. 아직시작/완료/GPUgatePASS아님. 새perjob `final-p1-job-3163986`, 실제S0/P1만수행/학습없음. 3163545를모니터/재제출대상으로사용하지않는다.

## 다음의존단계

- probecode/배포관측 → 독립검토 → concretecheckpoint선택·실제inputstaging·컨테이너/캐시검증 → bounded4GPUpreflight.
- 6receipt PASS+실측예산/권한결속 → 진짜launchreadiness → seed1/2/3고정250updates. 예상시간만으로무제한/자동재시도승인기록을만들지않음. 실제학습예산은P4결과와대조.
- 평가producer계약확정 → 별도decoder구현·검토 → 실제checkpoint후50panel×3seed×on/off(300predictions) 및baseline → independentlongread정렬/구조평가 → bootstrap/정책검증.
- HiFi확보vsONT-only판정정책은아직미선택. 해당데이터부재가다른로컬코딩을막지않지만, STOP/GO를임의발행하지않는다.

## 현재진행과완료구분

이문서는착수/우선순위기록이다. 실제GPU PASS·학습완료·평가완료주장없음. 별도ESTHSP진단6165460은payload정상완료,externOOM재현; GPUreadiness증거가아니다. ESTphase기존source규칙미확정은독립진행문서참조.
