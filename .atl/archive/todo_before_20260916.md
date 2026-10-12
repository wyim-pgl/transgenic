# todo.md — 지금 해야 할 것

## 재개 — 2026-09-15 15:26 PDT 이후
- [x] 저자 metric 승인: 학습 composite 유지, paired primary 순수 weighted CE, composite도 병기, delta=off−on 및 0.005 nats 유지. primary는 증거 양성 타일별 차이의 동일 가중 평균(토큰 pooled 평균 아님).
- [x] DeltaAI 인증 복구·읽기전용 확인: 15:26 PDT seed123 best→epoch_04, eval0.65254; 3131537 Priority대기, 후속 Dependency대기, repo_seeds8f049eb 유지.
- [x] CPU paired metric increment1d3db4 (task/b6-mvp-metrics; harness alias pi-agent-3466292f-2e28-4d6): 컨트롤러 metrics14+model11+dataset9=34 tests PASS53.713s. 독립리뷰21fc4a07-7b98-425 spec PASS/quality APPROVE, correctness findings 없음. trainer 통합은 후속. [검증 기록](../notes/b6_mvp_metrics_controller_20260915.md).
- [x] weights-only init 계획 조사27d3e8ba-a61e-456 완료, advisor 검토 후 0→8 초기화만 지원(동일8 초기화/재개는 후속), forward semantics·hash·alias 감사 확정.
- [x] weights-only init 3a63314 완료(b6-mvp-weights-init-review1): runtime_identity.txt 누락 수정 후 컨트롤러46 CPU tests PASS49.046s, 독립 재검토f2acec77-a49d-4a3 spec/quality PASS, 신규 문제 없음. [검증](../notes/b6_mvp_weights_init_controller_20260915.md). 0→8 CPU helper만; main 미병합/실제checkpoint·SIF·GPU 인증 아님.
- [x] trainer/runtime 계획조사e12657bf-45c2-416 및 advisor 검토 완료. 별도 helper/CPU DDP 먼저, CLI·paired·atomic publication은 다음 increment.
- [x] fixed runtime a314bf9 완료(b6-mvp-fixed-runtime-review1): 컨트롤러20runtime+46regression PASS(142.24s/82.811s), 독립재검토e8a71519-643f-4e0 prior4건 ADDRESSED/spec PASS/implementation PASS/ACCEPT. 보고서 minor: 불변성 확인은 name/LR/wd/parameter identity 범위이며 모든 metadata byte동일 아님. [검증](../notes/b6_mvp_runtime_controller_20260915.md). 다음 CLI → 실제 GPU 인증. 기존 Slurm/원격checkout 무변경, 전체 A46 동결 승인 아님.


- [x] distributed paired 평가/atomic final 계획bc4d17c5-8fce-4fe + advisor 검토완료: compile평가 유지, DDP-only rank0저장+전rank결과통지, renameat2 nooverwrite, postrenamefsync 실패는 committed불확실내구성으로 보고.
- [x] finalize helper0a8120b 완료(won/b6-mvp-finalize): round3 controller39finalize+46baseline=85 PASS(165.67s/50.153s), 독립재검토55963985-cea4-430 remaining finding ADDRESSED/spec PASS/quality PASS. Runtime20 마지막 재시험은3fe33cb에서PASS; 최신schema-only수정에재실행안함. [전체수정·검증이력](../notes/b6_mvp_finalize_controller_20260915.md).
- [x] CLI 통합 계획45af5d16-ae45-448 완료: 별도entrypoint, fixedrecipe, bound-forwardcompile→DDP, 실측metadata·전체RN검증, 후보DB수용기록 failclosed 필요. 아직 구현 아님.
- [x] fullDB 읽기전용 감사11e30180-f282-426: finalSHA실측일치/count정합/overlap0/valid1442 PASS, 탐색MVP 기술후보로 적격. production/A46승인아님. [판정](../notes/b6_mvp_candidate_audit_20260915.md).
- [x] local technical acceptance faab8387…/inventories9f790196…/receipts73204a4e… 작성완료. 컨트롤러3파일SHA·canonical JSON·companionbytes/hash·RN9082(train7635/valid1447)확인. 원격변경없음.
- [x] exploratory-only CLI1c47ae5 완료(pi-agent-49252832-1f04-4d6): 컨트롤러99CLI/finalize/runtime+46baseline=145 PASS(391.23s/54.743s), 독립재검토8abd7ee2-9540-43c PASS/APPROVE, 남은RNG지적해소. [수정·검증전이력](../notes/b6_mvp_cli_controller_20260915.md). CPU구현범위만; 전체연결최종리뷰/실제launch-readiness는별도, main병합·GPU제출없음.

- [x] 누적전체연결리뷰c5eab5bb-ef9b-43a: CPU계약/품질PASS, publicGPUtopology gate medium1건 남음(실제readiness전blocker).
- [x] 누적CPU구현최종 f847cb8(pi-agent-5b846a12-93fa-4c3): public4visible nonMIGGPU/UUID/bf16 gate·hardwaremetadata, controller153PASS(407.16s/52.661s), scopedreview467d0399-cd17-452 PASS/APPROVE. [최종검토](../notes/b6_mvp_integrated_review_20260915.md).
- [ ] 실제runtime readiness: liveDeltaAI/job/immutablecheckpoint/SIF 재확인 → 실제tokenizer/Hyena step0·3tiers/cache/bf16/Inductor/4rank/resource smoke. readiness기록없음/학습불가. main미병합·새GPU제출없음. 09-16 00:12PDT read-only재조회: DeltaAI접속정상,3131537Priority대기/후속Dependency, bestepoch_04/repo_seeds8f049eb유지. DONE/FAILED마커없음(ls종료2는이두파일부재), 잡/checkout변경없음.

## 원격 완료 갱신 — 2026-09-15 17:26 PDT
- [x] 9종 blastx 양source DONE 관측, 마지막 Vvinifera allocation/batch COMPLETED0:0. full raw/provenance/timing 수용 감사는 남음.
- [x] fullDB6164118 allocation/batch COMPLETED0:0·payload0·SUCCESS PASS, overlap panel/test0, valid evidence1442/1447, retained733812. [실측](../notes/b6_full_completion_snapshot_20260915.md).
- [ ] 최종 DB 해시/전체source count/전후 입력 identity·산출물 검토 뒤 training candidate 수용; production_training_accepted=false 유지. .extern OOM0:125 원인 미해결.

## 재시작 실행 목록 — 2026-09-15 15:13 PDT (원격 실행 상태는 위 갱신 우선)

정본: [상세 인수인계](../notes/RESTART_HANDOFF_20260915.md), [실측 원문](../notes/restart_status_20260915.txt). 아래 역사 체크리스트의 “미제출/구현 중/승인 대기”보다 **이 목록이 우선**한다.

1. [ ] **기존 잡부터 확인, 재제출하지 않기:** array6164049(Osativa·Ppatens 실행, Ptrichocarpa·Sbicolor·Vvinifera 대기), full DB6164118(cpu-51 RUNNING47분). Sitalica·Athaliana·Bdistachyon·Gmax는 양쪽 source DONE/exit0 확인됨.
2. [ ] **완료 게이트 판독:** blastx는 per-source DONE/provenance/raw/timing + species batch.exitcode + sacct all steps; full DB는 `SUCCESS.json`, overlap0, valid evidence>0, full counts/input hash/raster checks. `.extern` 이상 별도 기록. partial DB나 샘플을 학습에 쓰지 않기.
3. [ ] **저자에게 이미 물은 metric 결정 받기:** 현 outputs.loss는 weighted CE+detached ratio penalty. 권고=기존 학습손실 유지, paired 검증은 순수 weighted CE + composite 병기. 아직 승인되지 않았으므로 0.005 nats 기준/primary를 임의 변경하지 않기.
4. [ ] **trainer/runtime 구현:** 결정 후 c7fa759에서 새 isolated worktree. 엄격 weights-only init-from(부모 run_config provenance), 250 optimizer steps/12 warmup, two LRs, partial accumulation, real Accelerate scheduler/DDP tests, paired rn dedup, atomic final publication. [상세 계획 §5](../notes/RESTART_HANDOFF_20260915.md). 아직 구현하지 않음.
5. [ ] **DeltaAI 인증 복구:** pronghorn login-0에서 저자 `ssh -fN dtai`; seed123 현재 상태/고정 초기 checkpoint 및 SIF 정체성 확인. 학습 중 checkout pull/수정 금지,456/789 연기 유지.
6. [ ] **GPU 전 인증:** 실제 tokenizer/Hyena/3tiles step0/생성 hook+cache parity/bf16/compile/4-rank smoke 및 자원 측정. 그 뒤에만 MVP fine-tuning.
7. [ ] **후속 evidence/scoring:** blastx raw HSP≥30-aa 수용·frameshift 해석·genomic phase 투영은 미구현. MVP 독립 A-ONT1/A-HiFi 정렬·디코딩·bootstrap/결정도 미완료. A46 전체 동결/통합/푸시는 별도 작업.

완료·재실행 불필요: data86tests+review, model11tests+review, dataset/model20tests+review; 모두 branch에 보존. **MVP 최신 c7fa759 / blastx 최신950cf4c, main9a8c8e1과 다름.** active agent/반복 모니터 없음; 원격 Slurm만 계속 돈다. Graft 설정과 기존 미커밋 변경을 보존한다.

---

## 이 아래는 과거 실행 기록 — 현재 체크리스트로 집행하지 않기

## 독립 작업 병행 — 2026-09-15
- [x] 저자 “동시에 진행 할수 있는 부분들 진행해줘” 승인. blastx6164049는 유지.
- [x] MVP smoke6164069 FAILED1:0(import duckdb): wrapper PYTHONNOUSERSITE=1이 기존 검증user-site DuckDB를 숨김. sample/build 미실행, builder결함으로 해석하지 말 것. snapshot abca40f/hash 준비됨.
- [x] MVP retry6164081 FAILED1:0 1m26s: imports+inputhash+sample10000 PASS, 컨테이너 내부 `/usr/bin/time` 부재로 builder subprocess 시작전 실패. corecode결함 아님.
- [x] smoke ops inner time→stdlibresource+monotonic 변경, approvedcore/asserts/hash불변 확인.
- [x] MVP smoke6164093 allocation/batchCOMPLETED0:0 4m38s +SUCCESS. builder195.87s/peak149104KiB/DB16,527,360B, sample10000→9466molecules/retained8930, panel/testretainedoverlap0. trainRC/N/offPASS; validwithevidence0(지리적으로편향된firstrecords sample), valid은emptyavailability경로만검증. NOT_FOR_TRAINING. externOOM별도.
- [ ] fullAthalianaMVP C0-lite job6164118 RUNNING cpu-51(2CPU64G12h/norequeue),2m03s시점 builderstdout/stderr+partialDB/WAL생성확인. Approvedabca40f/fullBAM/hash고정; fullQC/valid증거/최종출력성공 아직미확인. output `evidence/mvp_full_20260915`, trainingcandidate untilQC. [실행기록](../notes/b6_mvp_full_build_20260915.md). noGPU/blastx변경.
- [x] MVP model 계획 완료: local pdbPython(torch2.6/transformers4.46.3)에서 tiny actualwrapper/LED/cache test 가능, Hyena factory만 대체. Runtimecheckpointhelper/dataset/trainer는 후속increment로 분리.
- [x] model increment0514b63 branch `pi-agent-1765f0e2-cf6c-4ed` 완료. 컨트롤러별도archive에서11CPU modeltests PASS2.135s. actualwrapper/LED+tinyHyena대체검증, realHyena/SIF/GPU인증 아님.
- [x] model0514b63 independent specPASS/qualityPASS, controller11testsPASS확인(리뷰의rerunpending은이미해소). RealHyena/SIF/GPU/compile/DDP 아직gate.
- [x] dataset/collate계획검토완료: legacy8tuple보존, EST9tuple+선택적10번째rn/has_evidence, DNApadding별도길이, privateworkerRNG, evidence별도readonlyconnection/PID, failclosedrn정체성.
- [x] datasetad07302/reportc7fa759 branch `pi-agent-a92f1af6-926c-40f` 구현완료. 컨트롤러별도archive+localduckdbtarget에서 dataset9+model11=20testsPASS62.368s.
- [x] dataset/collate independent specPASS/qualityAPPROVE, controller20tests이미PASS. Actualfork/realHyenaTokenizer/SIF/DDP경로futuregate. 선택테스트강화: nonzerosuffixpadding/workerexactseed.
- [x] trainer/runtime계획완료(c7fa759), init-from는 checkpointconfig아닌 enclosingrun_config.json recipe로architecture감사필수, safetensorsaliases보존·missingprojection만허용. SchedulerAccelerateworldsize实测·partialaccum/pairedrn중복제거 필요.
- [ ] 저자 metric결정대기: actualmodel outputs.loss=weightedCE+0.05detachedratioPenalty(모델source확인), spec0.005natsfutility와동일시불가. 권고=학습손실그대로, pairedprimary순수weightedCE 별도계산+composite같이기록. 저자에게질문함; cutoff/primary를승인없이바꾸지말것. blastx/fullDB는계속.
- DeltaAI read-only checkpoint/status 확인은 인증 실패(Kerberos/Duo). 원격 학습 상태 미확인; 시드123 잡/체크아웃 변경하지 않음. GPU 단계 전에 사용자 인증 복구 필요.

## 나머지8종 EST blastx — 2026-09-15 13:01 PDT
- [x] 저자 “나머지 8종 blastx 진행 먼저 하고” → array6164049 제출(0–7%2, 각16CPU/64G/12h, Swiss→Ortho 순차). 0Athaliana·1Bdistachyon RUNNING(cpu-50),2–7JobArrayTaskLimit대기 확인. 총5,002,430 screened EST, Sitalica 중복 없음.
- 순서:0Athaliana/1Bdistachyon/2Gmax/3Osativa/4Ppatens/5Ptrichocarpa/6Sbicolor/7Vvinifera.
- 출력 `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx/remaining8_20260915/`, logs `%A_%a.out/err`, 각종batch.exitcode 및 source DONE/provenance/time 확인.
- 입력 A37 min100/c2_training_eligible/summary가 각종 정렬receipt와 일치. 각task compute에서 기록MD5+summary/receiptSHA 검사후 querySHA고정, source/lineage/role/engine 해시검사. `notes/b6_blastx_eight_inputs_20260915.json` 및 실행sbatch에 기록.
- [x] 6164049_1 Bdistachyon 양source DONE, allocation/batchCOMPLETED0:0 33m41s. Swiss137448queries/1363835HSP, Ortho183946queries/4518360HSP. externOOM0:125 별도 기록.
- [ ] 나머지7종 진행: 최근0Athaliana·2GmaxRUNNING,3–7대기. 전종양source완료/QC·메모리·Slurmsteps 판독. MVP 독립작업 병행은 후속 저자승인.


## 실제 EST blastx 시작 — 2026-09-15 09:20 PDT
- [x] 저자 “est blastx 시작” 재지시 → 순차 실데이터 pilot6163851 제출, cpu-50 RUNNING 및 Swiss-Prot `Computing alignments` 확인. Sitalica primary66,040건 전체,16CPU/64G/12h, Swiss→Ortho 순차. 나머지8종 아직 미제출.
- 출력: `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx/sitalica_pilot_20260914/` (경로 suffix는 준비일, 실제 실행일은09-15).
- 실행기950cf4c/script/helper/binary 해시 확인, query는A37 정렬 receipt의MD5 `0b33e9eb0719df2c35a468fd623c5d4d`와 대조 후SHA256고정; summary66,040/100nt arm 사용. 원래 UniVec PROVENANCE의121nt 기록은 superseded, A37 receipt를 정본으로 사용.
- [x] 6163851 allocation/batch COMPLETED0:0 21m54s, batch.exitcode0, 양쪽DONE. Swiss38018/66040 queries aligned,290009HSP(30aa미만1290), 동종4555 단백질3개배제. Ortho56557/66040 queries aligned,1371252HSP(30aa미만3868), 동종4555 단백질22163개배제. BatchMaxRSS7608576K(약7.3GiB). extern OOM0:125 반복, 결과와 별도 기록.
- [ ] 나머지8종 사이징·제출; raw HSP의30aa수용필터·phase투영은 아직 미실행. lineage 완전성 한계와 extern 이상 기록 유지.


## 격리 구현 착수 — 2026-09-14 16:51 PDT
- [x] 저자 “별도 워크트리 사용 진행해” 승인. 네이티브 격리 worktree `/tmp/pi-agent-611e8432-78e4-474-e54d14c0` 생성 확인(기준 9a8c8e1, main 무변경).
- [x] MVP CPU 구현 branch `pi-agent-611e8432-78e4-474` HEAD abca40f (baseline5390864, implementation6e26a70). 컨트롤러가 archive+격리 venv에서 86 tests passed(181.43s) 재확인. 원격 생산 빌드·모델 연결 미실행.
- [x] Agent `f1ca80e4-af43-42e`: MVP CPU increment spec PASS / code quality APPROVE. Production 전 입력 해시/assembly/완전 test-block 메타데이터·불변 snapshot 검증 및 자원 실측 필요.
- [x] MVP 읽기 전용 경로 점검 완료: BAM126,988,517B; corpus23,534,252,032B/noWAL, reference build linkage 미확인.
- [x] CPU identity preflight6163394 allocation/batch COMPLETED0:0 2m10s; corpus/BAM 해시 및 build_manifest fastaSHA와 두 참조 FASTA 바이트 일치 PASS. Athaliana tiles7635train/1447valid/1388test, blocks93/15/13. extern OOM0:125는 별도 기록(스크립트 결과와 구분).
- [x] deps6163395 allocation/batch COMPLETED0:0: 별도 `worktrees/mvp-cpu-deps-20260914/site` pysam0.23.3 설치, container duckdb1.5.5/pysam0.23.3/numpy2.3.1 import PASS. extern OOM0:125 별도 기록.
- [ ] MVP 다음: abca40f CPU 코드·approved2079패널을 새 remote snapshot에 stage/hash 대조 → 학습불가 표기한 작은 real-BAM operational sample 성능 검사 → full build 자원 결정. 모델 연결/전체 MVP 미완료.
- [x] Agent `916abc47-68e4-499` 사전 점검 완료: DIAMOND .dmnd는 --seqidlist 미지원. 동종/하위분류군을 frozen lineage로 제외한 종별 DB를 검색 전에 구축해야 함. BTOP frameshift 출력 지원 확인; 실제 엔진 fixture는 아직 미실행.
- [x] blastx branch `pi-agent-462c26cc-1192-466` HEADcc9735f, 컨트롤러22 tests passed 재확인. pronghorn 새 code snapshot staging 및 로컬/원격 SHA256 일치.
- [x] blastx 독립 검토 완료: fixture getseq의 --no-parse-seqids 미지원 1건. Production은 완전한 lineage/승인 입력 provenance/실측 자원 필요.
- [x] getseq fix0715940 컨트롤러23tests PASS, scoped review GO → pronghorn snapshot `worktrees/b6-est-blastx-0715940`, 로컬/원격 해시 일치.
- [x] fixture6163393 제출(4CPU/8G/30min), FAILED1:0 3초: engine_version이 `diamond version`의2.1.24와 full build2.1.24.178을 혼동. 실제 `diamond help`는2.1.24.178 확인, binarySHA0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d. 정렬 실행 전 중단. extern OOM0:125도 기록(batch/extern RSS 약1.5MB); 이를 정렬 메모리 부족으로 단정하지 않음.
- [x] exact build 검증 수정950cf4c branch `pi-agent-cb2fd26a-e84c-456`; 컨트롤러26tests PASS. remote snapshot `worktrees/b6-est-blastx-950cf4c` 로컬/원격 해시 일치.
- [x] version fix scoped GO → fixture6163396 재실행: allocation/batch COMPLETED0:0, top+두source DONE. 실제 DIAMOND2.1.24.178에서 ordinary/reverse/±1nt/insertion/deletion 각6건의 BTOP/CIGAR 복원 및 동종/descendant/merged 배제 PASS. extern OOM0:125 남음; 정렬 자원 측정 근거로 쓰지 않음. 결과 `evidence/est_blastx_fixture/20260914_review03/provenance.json`.
- [x] lineage preflight6163397 allocation/batch COMPLETED0:0 8m18s, both source DONE+구조 audit PASS. Ortho12,115,085seq/408taxid(전부cache), Swiss42,106seq/2208taxid(신규2015). 미해결taxid0/구조실패0/ancestor conflict0. 원본FASTA canonical SHA 실제바이트 일치. Ortho lineageSHA812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5; Swiss402bca970569cc2c0192ba185c12c8fcaed99ecafd71f46a5e19f4f2789615ce. extern OOM0:125 별도 기록.
- [ ] Production 전 lineage 생물학적 완전성/merged 권위 및 입력receipt 대조 승인, 작은 EST 순차 성능 pilot 필요. 구조 audit만으로 전부 입증됐다고 하지 말 것. 생산 EST blastx 미실행.
- [x] gpu 기존 lineage cache 존재 재확인: `/home/pgl/scratch1/wyim/transgenic_data/orthodb_filtered/lineage_ncbi.tsv` 52,988B, SHA256812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5. 존재/해시만 확인, Swiss 전체 taxid coverage·계보 완전성 승인 아님.
- 모델·트레이너·원격 seed123 체크아웃 변경 없음. MVP/DIAMOND 생산 실행은 코드 검토 및 사전 시험 후.

## 저자 실행 승인 — 2026-09-14 16:41 PDT
- [x] 저자 “EST blastx 실행”, “mvp 승인할께 진행해줘” 수신. [범위·설계 해시 기록](../notes/b6_execution_approval_20260914.md).
- [ ] DIAMOND 입력·DB·동종 배제·실행기·중복 잡 사전 점검 → 실행 코드 검토 → blastx 제출.
- [ ] 승인된 MVP day 1–2 증거 테이블·8채널 트랙 구현 및 테스트 → 모델/학습 연결 → GPU 게이트.
- MVP 승인 대기는 해소됨. A46 전체 승인·동결은 별개이며, seed123 원격 체크아웃·기존 잡은 변경하지 않는다.

## 복구 후 — 2026-09-14 16:35 PDT
- [x] 원격 seed123 링크 2 `3131537` PENDING 및 후속 11시간 체인 확인(16:26 PDT); 중복 제출·원격 변경 없음.
- [x] Astra 최종 검토에서 발견된 MVP 문안 블로커 2건 수정: 트랙 가닥은 A39/scorer의 FLAG+ts 합성값, seed123 동일성은 step 0에 한정(이후 현재 backbone의 sequence-only 경로). 패널 계수 규칙·TSV·실행 코드는 그대로.
- [x] 수정 후 패널 테스트 8 passed, TSV sidecar 3종 OK, tracked diff whitespace 검사 통과.
- [ ] 위 2건 수정본 재검토 및 명시적 저자 승인. 동결·커밋·WP1 실행은 아직 하지 않음.

## 최신 — 2026-09-14 15:25 PDT
- [x] A46 3차 잔여 문안 8건 반영, MVP v1.2 draft·A46 rule 17·영문/한글 워크플랜 동기화. 패널 테스트 8건·해시·인용문·실제 CLI 재현 확인. [항목별 기록](../notes/b6_a46_r3_wording_fixes_20260914.md).
- [ ] 수정본 최종 검토 → 명시적 저자 승인 → 동결·커밋. **WP1은 승인 전 착수하지 않는다.**
- [ ] DeltaAI 인증 복구 후 시드 123 재개 상태 확인. 이번 작업은 문서만 수정했고 잡 제출·원격 변경은 없다.

이하 이전 체크리스트에서 “3차 문안 수정 진행 중”은 위 완료 기록으로 대체하며, 승인 대기는 유지한다.

> 📌 **정본 구분**: 지금 상태는 [`resume.md`](resume.md) 📌 블록 · 쓰지 않기로 한 것은
> [`quarantine.md`](quarantine.md) · 진행 상태의 정본은 GitHub 이슈 wyim-pgl/transgenic ~~#4–#63~~ ✏️ #4–#71(2026-09-08 실측; resume.md 📌와 동일).
> 이 파일은 **다음 행동**만 담는다. 최종 갱신 ✏️ **2026-09-14 11:5x PDT**(HEAD `9a8c8e1`, 미커밋 2건 = A46 v1.36 초안 in `PROTOCOL_B1_frozen_v1.md`·`B6_MVP_PILOT_v1.md`; dtai `repo_seeds` `8f049eb` 시드 123 학습 중, pull 금지 — 아래 09-14 절이 "지금 여기") · 이전: 최종 갱신 **2026-09-10 오전 PDT** · 저장소 HEAD **`c646e76`** (푸시됨) · dtai `/work/repo` 08f3e2b + `train_b5.slurm`(=`ba24b19`, 시드 123 전용, 더 갱신 안 함) · **`repo_seeds` c646e76**(확인 시드 전용 동결) · `repo_dev` c646e76. **시간은 PDT.** ~~최종 갱신 2026-09-09 저녁 · HEAD `1a245ce` · `train_b5.slurm`(=`1a245ce`)~~ ~~최종 갱신 2026-09-09 낮 · HEAD `7df570f`~~ ~~최종 갱신 2026-09-08 · 저장소 HEAD `4ef3dc4`~~
>
> ⚠️ **낡은 줄 점검을 2026-09-08에 했다.** §4의 `#59`(실제로는 `1413efd`로 커밋됨)·`#57`(`28cbaa5`로 구현됨)과
> §2c의 A18.8 항목(`12a89fe`의 `longread_fetch_v2.sh`로 해소됨) **세 줄이 완료된 일을 남은 일로 보이게 하고
> 있었다.** §4는 "승인 나면 바로" 목록이라 여기가 낡으면 **남은 일이 실제보다 많아 보인다** — 승인이 났을 때
> 무엇부터 할지를 이 목록에서 고르기 때문이다. 항목을 닫을 때 **커밋 해시를 함께 적는 것**이 이 점검을 1분으로
> 만든다: 해시가 있으면 `git log`로 대조되고, 없으면 코드를 뒤져야 한다.


**§1 현재 B5 객체 (A40 / v1.30, 2026-09-06)**: `gpu:/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db` — **273,148행**, 23,534,252,032 B, MD5 `dda7e78c880c993435654b57dcd1c498`. [새 freeze](transgenic/data/freeze/b5_full_a40_v1.freeze.json), [A40 검증 노트](transgenic/notes/a40_verification_20260906.md). 이전 `b5_full_v1`(272,224행)은 보존된 superseded 객체다.

- [x] ✅ **저자 A40 diff 검토·커밋 (2026-09-08)** — 저자 승인 후 **두 커밋으로 분리**: `f6e68d4`(실행된 그대로,
      30파일)·`5289077`(사후 테스트 강화). 나눈 이유는 freeze가 `scripts/merge_b5_databases.py`(`dfd26632…`)와
      그 테스트(`8ab1d1ba…`)를 SHA256으로 고정하기 때문 — `f6e68d4`의 트리가 두 해시와 바이트 일치함을 확인했다.
      `4ef3dc4`(실행 스크립트 A40 지향 + preflight)까지 **푸시 완료**, #55·#56·#58 종료.
      ⚠️ 커밋 메시지의 `Closes #55, #56, #58`은 **첫 번째만** 닫았다 — GitHub은 키워드 뒤 쉼표 목록을 파싱하지
      않는다. 번호마다 키워드가 필요하다
- [ ] B5 시작 시 새 §1 DB·수정된 3토큰 tokenizer·**새 labelled run**을 함께 기록. #55/#56/#58 코퍼스 블로커는 해소되어 시작 가능하며, ACCESS 자원 등 기존 실행 전제는 유지. A28에 따라 기존 chain 안에서 바꾸지 않는다. #61 scoring-universe는 별도 결정.

- ✏️ 12:3x PDT: 패널 확정(풀 2,079 · MVP 50 · pilot 300 · final 1,729; `b6_panels_v1.tsv` 79367154…). 72a = §1.3 (a) 배치 병합 계수로 복원. a46-drafter가 A46 빈칸 기입 중 → 최종 Codex 확인 → 승인 요청 → 커밋(A46·MVP v1.1·71/72/72a·테스트·TSV 3종+sha256).

## 진행 중 — 2026-09-14 11:5x PDT: B6 WP0 — A46 재검토 → 저자 승인 → 커밋 → WP1 + MVP 구현 (← 지금 여기)

- ✏️ **12:1x PDT — MVP 설계 v1.1 완료**(`B6_MVP_PILOT_v1.md`, 280행, 미커밋): Codex 7건 종결 — futility STOP(1차 = valid 타일 증거 on/off 짝 검증 L_GSF, 한쪽 95 % 상한 < δ 0.005 nats; 2차 F1 δ 2.0 pt; 구조 = 5일차 필수 A-ONT1/A-HiFi 유전자좌 정렬(전체 참조에 §3 명령)의 접합부 P/R; STOP 경로에서 withholding 민감도), GO = 하한 > 0 + 3 시드 양, 그 외 INCONCLUSIVE(프로그램 중단 아님); valid 타일 트랙; 분자 중복 제거; donor/acceptor 전사 방향 + RC 정체성 시험; SIF 버전 기록·캐시 시험 fail-closed; 패널은 71/72/72a 기준(자체 샘플러 없음); 50 유전자좌 전부 디코딩; 예산 ≈ 18(+5.5 STOP 경로) GPU-h. §5.3 문장을 A46 rule 17에 그대로 넣도록 지시. **A46 남은 빈칸: 풀·final 수·TSV sha256(패널 하드 플래그 필터 반영 후 pronghorn에서 72a+72 실행) → 최종 Codex 확인 → 승인 요청.**

- ✏️ **12:0x PDT — Codex A46 2차: reject 13건**(원문 scratchpad `codex_review_a46_r2_output_raw.txt`; 해소 2·4·5·6·7·8). 남은 것과 조치: ① 동결 문구 — "승인 시 명령·도구 버전(PROVENANCE sha256)·시드·임계값·분모·패널 id/수·N_pilot 고정; 승인 뒤 생기는 산출물(C0·트랙·증거 DB·런 정체성)의 해시는 사용 전에 기록"으로 교체, 'TSV sha256만 미룸' 문장 삭제 ③ WP4 4-rank compile 통합 시험 문장(Dynamo 폴백 실패·0/비0 트랙·캐시 생성 parity·RC·resume) ⑨ **N_pilot = 4,142 update(2 epoch × 2,071)**, 대조군 전부 seed 123, aug_seed 20261037, 최종 스텝 체크포인트 비교 ⑫ **하드 플래그: tile train_weight로는 못 걸러냄(A32는 타일 가중치 1 유지·유전자 단위 마스킹)** → 유전자 단위 A22(GeenuFF)·A30(Swiss-Prot caution) 목록으로 직접 배제, 수 보고 ⑬ 승인 시 실제 풀·final 수 기입(자리표시자 금지) ⑩ rule 17은 MVP 설계서 §5의 futility 문장을 인용 ⑪ MVP 설계서 패널 동기화(진행 중). a46-drafter·panel-builder(하드 플래그 필터 추가·재생성)·mvp-designer에 지시 완료. 다음: 패널 재생성 → pronghorn에서 `72a`+`72` 실행 → 수치 기입 → 최종 Codex 확인 → 승인 요청.

- [ ] ⏳ **A46 v1.36 Codex 재검토 판독**(`PROTOCOL_B1_frozen_v1.md` 951~989·999행, 미커밋; 1차 reject 10건은 반영됨) → 남은 지적 반영 → 저자 승인 → 커밋(`docs: A46 …`, 메시지에 워크플랜 `c4b9d30` 근거). **WP1은 그 뒤에만**(A46 15항).
- [x] ✅ **12:2x PDT 반영 완료(v1.1, 280행, 미커밋; 정지 규칙 원문을 리드에게 전달 → A46 rule 17 동기화는 a46-drafter 몫)** — 원문: 🔴 **MVP 설계서 `revision/protocols/B6_MVP_PILOT_v1.md` 수정(Codex approve-with-changes 7건, 11:5x)**: ① §5.3 STOP → futility 검정(1차 = valid 타일 증거 on/off 짝 L_GSF, δ 0.005 nats; 2차 = reference-dependent F1, δ 2.0 pt; 한쪽 95 % 상한 < δ일 때만 STOP; GO = 하한 > 0 + 3 시드 양; 나머지 INCONCLUSIVE) ② `tile_est`를 valid 타일에도 ③ §2 A-ONT1/A-HiFi 유전자좌 정렬 **필수화**(50 유전자좌 ±10 kb 부분 참조, §3 명령 → `59_evidence_support.py`; 5일차까지 없으면 INCONCLUSIVE) — 정렬본 부재의 이유는 quarantine §1l ④ 배제 전 분자 중복 제거(chrom,start0,end0,strand + 클론 접두) ⑤ donor/acceptor는 `ts:A` 방향 + RC 정체성 단위시험 ⑥ STOP 전 withholding 민감도 1회 ⑦ SIF transformers/torch 버전 기록, 캐시 생성 시험 fail-closed; 50 유전자좌 전부 디코딩 → 수정 뒤 **A46 rule 17과 동기화** → 별도 커밋.
- [x] ✅ **설계서 쪽은 12:2x PDT 반영 완료(§1.1 타일 구조·§1.2 71/72 스크립트·핵 염색체만·50 전부 디코딩); ✏️ 12:4x 재실행 산출도 나옴 — 풀 2,079 · MVP 50(핵심 20, rank 62까지, 12개 EST < 3 제외) · pilot 300 · final 1,729, `b6_panels_v1.tsv` sha256 `79367154…`; 72a = (strand, POS, CIGAR) collapse, 설계서 §1.2 문장 리드가 수정(281행). 남은 것은 A46 문구 대조와 수치 기입** — 원문: 🔴 **패널 규칙 통일 반영(12:0x~12:1x 지시)**: 설계서 §1.2의 자체 샘플러 삭제, 패널 id = `72_mvp_select_loci.py` 산출 `b6_panels_v1.tsv`(`71_b6_panels.py` `shuffle_rank` seed 20260914 → `b6_eligible_loci_v1.tsv`; rank 순 부필터 길이 ≤ 20 kb·서로 다른 A-EST 분자 ≥ 3 통과 첫 50 = MVP, 비MVP 첫 300 = pilot, 나머지 final; **핵 염색체만**). 1차 산출(구 `62_` 규칙, 적격 2,143, ChrC/ChrM 5개 포함)은 폐기 → 71/72로 재실행 결과 확인(`72a_est_molecules_per_locus.sh`는 pronghorn BAM 필요) → A46 rule 12·decision 5·rule 17·설계서 §1.2·두 스크립트가 같은 문장인지 대조 → 풀·패널 수를 A46에 기입.
- [ ] 시드 123 **링크 2 `3131537`** 시작 확인(Slurm 추정 09-15 09:19 PDT): `Resumed from` · step 200 `latest_state saved` · epoch 5 `epoch=` 줄 → `apptainer exec $S8 python revision/scripts/60_epoch_budget.py --log ../runs/seed123/train.err --patience 3`(cwd `repo_seeds`). 이후 링크 3 `3131538`, 4~11 `3149356`~`3149363`(11 h). 완료(`TRAINING_DONE`) 시 남는 링크는 가드가 종료하거나 `scancel`.
- [ ] A46 승인 뒤 **WP1 착수**: ① EST×단백질 `diamond blastx --frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25` 배열 9종(DB `evidence/protein/diamond/`, sha256·PROVENANCE 있음) ② C0 파이프라인(9표 + `est_protein_hsp`, issue #21) — 설계 → Codex 리뷰 → 실행.
- [ ] **MVP 구현 day 1~2**(설계서 §6; 탐색 arm이라 A46과 독립으로 시작 가능): `mvp_build_c0lite.py`(pronghorn, `evidence/est_align/Athaliana/Athaliana.est.bam`, 누출 배제 = 패널 ±150 nt + test 블록, `mvp_qc.retained_overlapping_loci == 0`), `mvp_tracks.py` + 단위시험(RC 스왑·nested thinning·N-zero). day 3 모델/데이터셋/트레이너 edits(설계서 §3–4, step-0 bitwise 동일·KV-cache parity 시험). day 4 4-GPU 스모크는 시드 123 큐와 겹침 → 시점 조정.
- [ ] 위키 update 37 · `spec.md`.

### 저자 결정 대기 (2026-09-14)
- [ ] ⏳ **A46 v1.36 승인**(Codex 재검토 뒤)
- [ ] ⏳ MVP 정지 규칙 = Codex의 futility 검정(δ_loss 0.005 nats · δ_F1 2 pt) 채택 여부(위 🔴 ①; 패널은 50으로 확정, 20 유전자좌 F1은 ≈ 30 pt 효과만 검출)
- [ ] ⏳ MVP 독립 채점용 A-ONT1/A-HiFi 유전자좌 정렬 필수화(위 🔴 ③; 리드가 pronghorn에 있는지부터 확인)
- [ ] ⏳ 456/789 재제출 시점(`submit_chain.sh 456 3`; repo_seeds 커밋은 123과 동일 유지)
- [ ] 🟡 이월: 트레이너 파일 기반 stop 플래그 · NCSA launch-failure 보고 · `.mpi` 체크섬 결속 · pending_grads rank-0-only · `submit_chain.sh` APPEND 모드

## 🧊 진행 중 — 2026-09-10 오전: ~~인증 2건 대기~~ ✏️ 인증 4건 PASS → 시드 123 경계(21:45 PDT) → 시드 456 **자동 제출**(auto_submit_456.sh) → 시드 789 저자 결정 (✏️ 09-14: 집행·대체됨 — 456/789 연기, 시드 123 A45 체인; 살아 있는 것은 이 절 끝의 ✏️ 09-11~09-14 불릿)

- [x] ✅ ~~⏳~~ **인증 (repo_seeds `c646e76`, 26.08, 4 GPU) — `3124347` PASS(~~11:5x~~ ✏️ 09:5x PDT)·`3124348` PASS(~~12:1x~~ ✏️ 10:0x PDT), 아래 ✏️ 불릿** — `3124347` ddp-int-epoch(`KILL_AFTER_EPOCH=1 NTRAIN=484`: PASS = `restored … pending micro-batch` + `latest_state saved at epoch 1 step 23 global_step 6` + `epoch_02/meta.json` `pending_micro: 2`) · `3124348` ddp-int-2608g(정상 경로). 판독: `grep -hE "DDP_INTEGRATION|FAIL: |leg[12] rc|restored" repo_dev/ddp-int-3124347.out repo_dev/ddp-int-3124348.out`. FAIL이면 Codex 리뷰 → 수정 → `git -C repo_seeds checkout <new>` → 재실행(명령은 resume.md 오전 블록 1번).
      - [x] ✅ `3124099` ddp-int-2608t PASS(26.08 compile, TERM) · ✅ `3124229` 4 GPU·26.08·repo_seeds 체인 리허설 PASS(COMPLETED 0:0)
- [>] ⏳ **SEED=123** `3116844` — epoch 1 완료(eval 0.8617 best), epoch 2 26 %(07:55 PDT) → 종료 ≈ 09-11 01:30 PDT(11:00 PDT 실측 32 %·2.44 it/s → epoch ≈ 25.5 h; 경계 22:00 PDT는 epoch 2 도중) ✏️ 11:39 PDT 실측 **42 %·2.03 it/s → 종료 ≈ 09-11 03:30 PDT**, 예산 943/999. **오늘 21:45 PDT USR1 · 22:00 PDT 상한**(링크 1 시작 09-08 22:00:34 PDT).
      - [ ] epoch 2 `epoch=` 줄 → `revision/scripts/60_epoch_budget.py --log runs/seed123/train.err --patience 3`
      - [ ] 22:00 PDT 뒤: `b5-seed-3116844.out`에 `USR1: forwarding as SIGTERM` → `resubmitting chain job 2` → `Submitted batch job`; `sacct -j 3116844` COMPLETED; 링크 2 큐 확인. `forced preemption`이면 정상 폴백. `training failed`면 손으로 `sbatch --export=ALL,SEED=123,CHAIN_N=2 /work/nvme/bilv/wcyim/transgenic/repo/deploy/deltaai/train_b5.slurm`.
- [x] ✅ **시드 456 제출됨(09-11 09:52 PDT, 잡 `3131228`, 아래 ✏️ 불릿)** · ✏️ **09-12 11:53 PDT 저자 지시로 456·789 체인 취소(연기, 생략 아님)** — 재제출 명령은 아래 ✏️ 불릿. ~~✅ 시드 789 제출됨(09-11 09:56 PDT, 잡 `3131255`)~~. 원문: **시드 456/789 제출 — 위 두 항목 뒤, 저자 "시작해"에**:
      ```bash
      W=/work/nvme/bilv/wcyim/transgenic; S8=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
      ❌ SUPERSEDED (09-10 10:4x PDT, 선택지 2): **456은 `auto_submit_456.sh`가 자동 제출 — 손으로 내지 말 것(중복).** 789만, 저자 결정 뒤:
      [ -s $W/runs/seed456_link1_jobid.txt ] && sbatch --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 \
        --export=ALL,SEED=789,SIF=$S8,REPO_DIR=$W/repo_seeds $W/repo_seeds/deploy/deltaai/train_b5.slurm
      (원문 `for SEED in 456 789` 루프 폐기)
      ```
      시작 직후 확인: `repo: … @ c646e76` · `image: …26.08.sif` · `runtime identity: … gpus=4 cpus=64 torch=2.14…` · `PREFLIGHT PASS` · `gpu warm-up ok: device 0~3` · train.err `torch.compile scope: model` ×4 · `per-rank accumulation 24` · step 200 `latest_state saved`. 예상 ≈ 6 s/step, epoch 6.15 h, 시드 2.0~2.6일(링크 2개), 197~246 GPU-h. `/work/repo_seeds` 학습 중 pull 금지.
- [x] ✅ `rm -rf /work/nvme/bilv/wcyim/transgenic/runs/seed999`(`3124229` 리허설 잔재) — 삭제 완료(11:39 PDT 부재 확인)
- [ ] 위키 **update 37**(시드 시작 후: 26.08 결정·A44·인증·accumulation 수정·시드 123 epoch 1) · `spec.md` · 결과표에 **시드 123 train loss 보정**(epoch ≥ 2 값 × epoch 번호; 검증 손실 무관)
- [x] ✅ **저자 결정 (a) 이미지 = NGC 26.08**(09-10) → A44 v1.34(`c646e76`) · (b) 추가 배정은 예산상 불필요(여유분 ~300 GPU-h 요청 권고만)
- [x] ✅ **REPO_DIR 체인 스크립트** `35b1436`(Codex approve) · **preflight stderr** `07486a8` · **Codex 준비 상태 리뷰 반영** `5433f27`(latest_state rename 교체·`runtime_identity.txt`·preflight 엄격화) · **epoch 경계 accumulation 잔여분 보존 + epoch별 train loss + A44** `c646e76`(저자 지시). 테스트 48+ 통과.
- [x] ✅ 4 GPU 벤치(09-09): 5.89/11.04/23.23 s/step, 효율 91~93 %, epoch 6.15 h = 24.6 GPU-h.

- ✏️ **~~11:5x PDT~~ ✏️ 09:5x PDT(sacct 종료 11:54 CDT) — `3124347` ddp-int-epoch PASS** (c646e76, 26.08 compile, 4 GPU, NTRAIN=484): epoch 1 끝에 rank당 1 micro-batch 잔여·gradient 609 텐서 저장 → epoch_01 저장 직후 4 rank KILL → leg 2가 epoch_01에서 재개, `restored 1 pending micro-batch(es)`, **epoch 2 첫 update가 micro-batch 23**(카운터 유실이면 24), epoch_02 `pending_micro`=2, TRAINING_DONE. accumulation 잔여분 보존 수정이 끝까지 실증됨. 남은 인증은 `3124348`(정상 통합, 큐 대기) 하나.

- ✏️ **~~12:1x PDT~~ ✏️ 10:0x PDT(sacct 종료 12:08 CDT) — `3124348` ddp-int-2608g PASS** (c646e76, 26.08 compile, 4 GPU, gh121 첫 launch 성공, rank 하나 TERM → 4 rank 저장·종료, 재개, TRAINING_DONE). **인증 3건(체인 `3124229`·epoch 재개 `3124347`·통합 `3124348`) 전부 PASS → 코드 쪽 제출 준비 완료.** 남은 관문은 시드 123 경계(21:45/22:00 PDT)와 저자 "시작해".

- ✏️ **10:4x PDT — 저자 결정: 선택지 2 (456 먼저, 789는 시드 123 곡선을 본 뒤), 자동 진행.** dtai 로그인 노드에 `runs/auto_submit_456.sh`가 detached로 돈다(pid 3142608, lock `runs/auto_submit_456.lock`, 로그 `runs/auto_submit_456.log`). 동작: 5분마다 `sacct -j 3116844` → 링크 1이 끝나면(상한 22:00 PDT) 2분 뒤 **관문 검사**(`b5-seed-3116844.out`에 `USR1: forwarding as SIGTERM`·`resubmitting chain job 2`, train.err에 `signal 15 received`, TRAINING_FAILED 없음, 큐에 시드 123 링크 2, 재개 가능한 체크포인트) → 통과 시 `sbatch … SEED=456 SIF=26.08 REPO_DIR=repo_seeds`, `runs/seed456_link1_jobid.txt` 기록, 이후 30분마다 `runs/seed456_monitor.log`. **관문 실패 시 제출하지 않고 `GATE FAILED`만 남긴다** → 사람이 시드 123 경계를 먼저 본다. 시드 789는 저자 결정 대기(프로토콜 273행: 미룰 수 있으나 생략 불가). 확인: `ssh dtai 'cat /work/nvme/bilv/wcyim/transgenic/runs/auto_submit_456.log; cat /work/nvme/bilv/wcyim/transgenic/runs/seed456_link1_jobid.txt'`.

- ⚠️ **19:5x PDT — `auto_submit_456.sh`가 ~14:55 PDT에 죽어 있었다(dtai 로그인 노드 재부팅; gh-login03 부팅 14:27 PDT, ControlMaster 소켓도 15:05 PDT 소멸).** 로그는 10:41 시작 줄 하나뿐, dtai 쪽 `seed123_monitor.log`도 14:55 PDT에서 멈춤, 잠금 디렉터리만 잔존. gh-login01~04 전부 사본 없음·시드 456 미제출(sacct) 확인. 패치본(`runs/auto_submit_456.sh.new`, md5 0ea38238: 잠금에 host+pid 기록·EXIT trap rm -rf·대기 중 매시 heartbeat 로그·sbatch 직전 jobid 파일 재확인)을 Codex 리뷰 뒤 22:00 PDT 전에 재기동. **교훈: 로그인 노드의 detached 헬퍼는 잠금 디렉터리가 아니라 `pgrep` + 로그 mtime으로 생사를 확인한다.** 세션 cron 22:12·22:33 PDT가 경계와 제출을 이중 확인한다.
- ✏️ **19:57 PDT — 재기동 완료(Codex approve)**: gh-login03에서 `bash ./auto_submit_456.sh` **pid 1337021**, 잠금 owner 기록, 로그에 19:57:35 시작 줄, md5 `483c9201`(추가: `--comment=seed456` + sbatch 직전 큐 코멘트 재확인 — Codex 지적 "sbatch 성공 후 jobid 파일 기록 전 사망" 창 대비). 원본은 `auto_submit_456.sh.orig_20260910`. 생사 확인: `pgrep -u $USER -f 'bash ./[a]uto_submit_456.sh'`(자기 명령줄 매칭 회피) + `cat runs/auto_submit_456.lock/owner` + 로그의 매시 heartbeat. dtai 쪽 `seed123_monitor` 루프는 재기동하지 않음(세션 모니터가 대신).
- ⚠️ **21:45~21:50 PDT — 시드 123 경계: 링크 1은 옛 스크립트 텍스트로 돌았다(Slurm은 sbatch 시점에 스크립트를 복사; 09-09의 ba24b19 교체는 링크 2부터).** `.out`: `USR1: forwarding to trainer` → `resubmitting chain job 2` → `Submitted batch job 3128376`; sacct COMPLETED 0:0 21:45:14 PDT. 옛 결함(트랩 뒤 `wait` 즉시 반환)으로 **트레이너는 경계 저장 없이 잡과 함께 죽음** — train.err에 `signal … received`·경계 `latest_state saved` 없음. 마지막 latest_state 19:51 PDT(epoch 1 step 146772 global_step 3600, best 0.8617 bad_epochs 0), 종료 시 81 %(161,300) → **≈14,500행 ≈1.9 h 손실, ≤200 step 허용 경로 안**. TRAINING_FAILED·반쯤 쓰인 파일 없음. 링크 2 `3128376` PENDING(Priority), `scontrol write batch_script`로 캡처한 텍스트 md5 64477d87 = ba24b19, CHAIN_N=2 SEED=123 GPUS=1. **자동 관문은 21:49:35 GATE FAILED**(no-signal-forwarding·trainer-did-not-see-TERM — 새 문구를 요구했는데 링크 1은 그 문구를 찍을 수 없는 스크립트였다; 11:39의 '관문 문자열 검증'은 디스크 파일과 대조한 것이라 이를 놓쳤다) → 456 미제출, 스크립트 종료. **Codex 판정: 링크 2가 `Resumed from …` 로그를 찍은 뒤 456 제출**(재개 경로 실증이 관문의 본뜻; 1 GPU Priority 잡이라 대기 짧음). 세션 모니터 `link2_watch.sh`가 3분마다 감시(성공 `Resumed from`, 위험 `starting fresh`, 실패 시그니처 기준선 2건 초과). 제출 전 점검 5: 456 중복 없음(`squeue %k`=seed456)·seed123 디렉터리 정상·링크 2 md5·로그 이상 없음·sbatch 인자(26.08 SIF·REPO_DIR repo_seeds·exclude gh062,gh093·comment seed456). **교훈: 큐/실행 중 잡이 쓰는 스크립트는 `scontrol write batch_script <id>`로 확인한다; 위키·런 로그에 1.9 h 손실을 편차로 기록.**
- ✏️ **09-11 09:52 PDT — 시드 456 링크 1 수동 제출: 잡 `3131228`** (저자 결정 "지금 제출" — 22:33 PDT에 Slurm이 링크 2 시작을 09-12 02:14 PDT로 추정해 Codex의 '재개 확인 뒤 제출' 전제가 깨졌고, 저자에게 물어 결정; 응답은 09-11 아침). 명령: repo_dev에서 `sbatch --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 --comment=seed456 --export=ALL,SEED=456,SIF=<26.08>,REPO_DIR=$W/repo_seeds $W/repo_seeds/deploy/deltaai/train_b5.slurm`. 확인: NumCPUs=64·gres/gpu=4·ExcNodeList gh[062,093]·Comment=seed456·캡처 스크립트 md5 c326c775 = c646e76. `runs/seed456_link1_jobid.txt` = 3131228, `auto_submit_456.log`에 BY HAND 줄. 시작 추정: 456 N/A(노드 통째), 시드 123 링크 2 `3128376` **09-11 18:48 PDT**(변동). 세션 모니터 `seed456_watch.sh`(5분)·`link2_watch.sh`(3분) 가동 — 456 시작 직후 확인 항목은 위 3번 불릿 그대로(`repo: … @ c646e76`·`image: …26.08`·`runtime identity`·`PREFLIGHT PASS`·`gpu warm-up ok` ×4·`torch.compile scope` ×4·`DDP world size 4: per-rank accumulation 24`·step 200 `latest_state saved`). **시드 789는 저자 결정 대기**(같은 명령을 `SEED=789 --comment=seed789`로; 프로토콜 273행). ⚠️ 밤사이 세션이 질문 대기 상태라 매시 보고가 멈췄다 — 야간 결정은 차단 질문 대신 '기본 행동 + 기한'으로 적을 것.
- ✏️ **09-11 09:56 PDT — 시드 789 링크 1 수동 제출: 잡 `3131255`**(저자 "789 해"). 456과 같은 명령, `SEED=789 --comment=seed789`; NumCPUs=64·gres/gpu=4·ExcNodeList gh[062,093]·캡처 스크립트 md5 c326c775 = c646e76. `runs/seed789_link1_jobid.txt` = 3131255, `auto_submit_456.log`에 BY HAND 줄. 큐(09:56 PDT): 시드 123 링크 2 `3128376` 추정 18:48 PDT · 456 `3131228` 추정 09-12 19:40 PDT · 789 `3131255` N/A. **세 시드 전부 큐에 있음 — 남은 저자 결정은 없다**(c·d·e·pending_grads rank-0은 선택 사항). 세션 모니터 `seed789_watch.sh` 추가.
- ✏️ **09-11 10:0x~10:4x PDT — 저자: "우리 그렇게 시간이 없어" → "예산을 올려두었어 codex가 말한거 고쳐줘" → A45(v1.35) 구현·커밋 `8f049eb`·푸시.** 예산 잔액 1,433/1,499 GPU-h. Codex 계획 리뷰(approve-with-changes: 4 GPU 재시작 승인, 미리 큐잉은 5건 수정 조건) → 구현 → Codex 변경 리뷰(approve-with-changes: 잠금 fail-closed·SEED 검증·명시적 exit 0·A45 문구 2건) → 전부 반영, 테스트 34건 통과. **취소: `3128376`(시드 123 링크 2, 1 GPU)·`3131228`(456)·`3131255`(789). `runs/seed123` → `runs/seed123_1gpu_partial`(보존, 미보고).** dtai `repo_seeds` = **`8f049eb`**(train_b5.slurm md5 a6a16993). **리허설 체인 seed 999: `3131416` → `3131417` → `3131418`**(15분·USR1@600·4 GPU·26.08; `runs/rehearsal_a45.sh`, `SBATCH_TIMELIMIT`/`SBATCH_SIGNAL` 환경변수로 #SBATCH 덮어씀 — TimeLimit 00:15:00 적용 확인; 링크 1은 10:41 PDT gh140에서 즉시 시작). **리허설 절차**: 링크 1 SIGTERM→저장→exit 0 · 링크 2 `Resumed from` → **링크 2 RUNNING 중에 `touch runs/seed999/TRAINING_DONE`**(사람이) → 링크 2 종료 시 `cancelling pre-queued link 3131418` · 그 뒤 4번째 링크 1개 제출 → `training already finished … nothing to do` exit 0. 네 결과 전부 확인돼야 시드 제출(go/no-go, A45 인증 문단). **통과 후**: `cd repo_dev; SIF=…26.08.sif REPO_DIR=$W/repo_seeds bash $W/repo_seeds/deploy/deltaai/submit_chain.sh 123 3`(456·789 동일) → `runs/seed<N>_chain_jobids.txt` → 모니터·매시 보고 갱신 → 위키 update 37.
- ✏️ **09-11 11:11 PDT — 리허설 4/4 PASS → 세 시드 제출 완료(A45 미리 큐잉 체인, 9잡).** 리허설: `3131416` COMPLETED 0:0(USR1 +5분 → 4 rank SIGTERM → latest_state → exiting cleanly) · `3131417` 9분 뒤 시작·`Resumed from`·손으로 만든 TRAINING_DONE → `training finished` → **`cancelling pre-queued link 3131418`**(CANCELLED) · 4번째 링크 `3131519`(1 GPU) → `training already finished … nothing to do` COMPLETED 0:0 8 s. **제출**(repo_dev cwd, `submit_chain.sh <seed> 3`, repo_seeds 8f049eb, 26.08, 4 GPU×64 CPU, exclude gh062,gh093, comment seed<N>): **123 = `3131536`→`3131537`→`3131538` · 456 = `3131539`→`3131540`→`3131541` · 789 = `3131542`→`3131543`→`3131544`**. 기록: `runs/seed<N>_chain_jobids.txt`. 리허설 디렉터리 seed999 삭제. 시작 직후 확인 항목: `.out`의 `repo: … @ 8f049eb`·`image: …26.08`·`runtime identity: … gpus=4 cpus=64 torch=2.14…`·`PREFLIGHT PASS`·`gpu warm-up ok` ×4, train.err `torch.compile scope: model` ×4·`DDP world size 4: per-rank accumulation 24`·step 200 `latest_state saved`, `runs/seed<N>/LINK_ACTIVE.<jobid>`. 링크 경계마다: `.out`에 `USR1: forwarding as SIGTERM`(4 pid) → train.err `latest_state saved`·`exiting cleanly` → `link k ended at its limit; the pre-queued next link continues` → sacct COMPLETED 0:0 → 다음 링크 `Resumed from`. 세션 모니터 `chain_watch.sh <seed>`(5분) ×3, 매시 :37 보고(`check_now.sh`가 jobids 파일을 읽음). **남은 일**: epoch마다 `60_epoch_budget.py --patience 3`, 위키 update 37(A45·리허설·제출), 결과표에 1 GPU 부분 런은 미보고(A45 2항).
- ✏️ **09-11 14:0x PDT — 저자: 22 epoch 상한 "A 유지"(변경 없음).** 근거: `max_epochs`는 LR 스케줄 지평(`train_HyenaTransgenic.py:342-354`, 5 % warmup + 0까지 선형 감쇠)이라 값을 늘리면 레시피가 바뀌고, 레시피 파일은 A40 동결 JSON에 sha256 고정. 22에서 LR이 0이 되므로 patience 3이 먼저 멈춘다. 저자 질문 "3 시드 꼭 필요한가" → 프로토콜 273행(456/789 투고 전 완료, n=3 평균±s.d.); 줄이려면 A46. 권고: 큐에 둔 채 시드 123 초반 epoch 보고 결정(대기 잡은 예산 소모 없음, 취소는 큐 위치 상실). 큐 상태 13:5x PDT: ghx4 idle 0(allocated 52·mixed 99·예약 33노드), 우리 잡 Priority 대기 순위 ≈ 506/1,648(점수 1,367). 기대 결과 정리(사전 등록 정확도 목표 없음; 학습 곡선·GSF 위반·held-out/검증 arm 일치·항등 기준선 초과가 판독 기준)를 저자에게 전달.
- ⚠️ **09-12 06:49~08:52 PDT — 시드 123 링크 1 `3131536` 시작(gh129), 시작 마커 전부 PASS, latest_state 200·400 저장 확인. 실측 속도 rank당 1.4 it/s(2시간 안정) → epoch ≈ 10 h(+검증), 40 GPU-h/epoch — 벤치 기반 기대 2.25 it/s·6.15 h보다 1.6× 느림.** 자원·SIF·acc 24·워커 8은 벤치 3118904와 동일; 차이는 워크로드(단일 티어 30스텝 vs 3티어 혼합 + 실제 라벨 길이 → compile 동적 shape 추정, 미실증). GPU 활용 100 %, 재컴파일 경고 0. 시사점: 시드당 8~10 epoch ≈ 80~100 h(링크 2~3개, 3링크 체인으로 충분), 3 시드 × 10 epoch ≈ 1,200 GPU-h(잔액 1,433; 12 epoch씩이면 초과). 456/789 시작 추정 09-13 07:52 PDT. `LR schedule 182,270 total steps` 표기는 명목값(accelerate가 optimizer 스텝마다 스케줄러를 4번 밟음; 실효 45,567 = 1 GPU와 동일, 버그 아님).
- ✏️ **09-12 11:53 PDT — 저자 지시: 시드 456·789는 나중에 돌리기로 하고 취소, 123만 계속.** `scancel 3131539 3131540 3131541 3131542 3131543 3131544` → 전부 CANCELLED(시작 전이라 예산 소모 0). 시드 123 체인은 손대지 않음(`3131536` RUNNING 5 h, `3131537`·`3131538` PENDING). 기록: `runs/seed456_chain_jobids.cancelled_20260912.txt`·`seed789_…cancelled_20260912.txt`(원본 이름에서 변경), `auto_submit_456.log`에 deferral 줄. 프로토콜 273행상 456/789는 **미룰 수 있으나 생략 불가** — 투고 전에 돌려야 하며, 생략하려면 A46 필요. **나중에 다시 돌릴 때**: `cd $W/repo_dev; SIF=$S8 REPO_DIR=$W/repo_seeds bash $W/repo_seeds/deploy/deltaai/submit_chain.sh 456 3`(789 동일; repo_seeds는 시드 123과 같은 커밋 8f049eb를 유지할 것 — 런타임 정체성 동일 조건). 세션 모니터 456/789 중지, 매시 보고는 123만(jobids 파일 기준). 예산 전망: 시드 123 단독 ≈ 10 epoch × 40 GPU-h ≈ 400 GPU-h(잔액 1,433).
- ✏️ **09-12 16:56 PDT — 시드 123 epoch 1 완료**: `epoch=0: train_ppl 5.2781, train_epoch_loss 1.6636, eval_ppl 2.3886, eval_epoch_loss 0.8707`; best=epoch 1, bad_epochs 0; `epoch_01/`(pending_grads 포함) 저장; epoch 벽시계 **10 h 07 m**(검증 ≈ 13 min 포함) → 링크 1(48 h)이 ≈ 4.7 epoch, 09-14 06:49 PDT경 epoch 5 도중 경계. 1 GPU 부분 런 참고값 0.8617(+0.009; 기술적 비교만, A45 2항). ⚠️ `60_epoch_budget.py`는 dtai 로그인 노드 python3에서 SyntaxError(`from __future__ import annotations`, Python < 3.7) — **컨테이너 안에서 실행**: `apptainer exec $S8 python revision/scripts/60_epoch_budget.py --log ../runs/seed123/train.err --patience 3`(cwd repo_seeds; epoch 2 이후 판정 가능).
- ✏️ **09-13 02:56 PDT — 시드 123 epoch 2**: `epoch=1: train_ppl 2.0482, train_epoch_loss 0.7170, eval_ppl 2.0197, eval_epoch_loss 0.7029`(epoch 1 0.8707 → **−0.168**), best=epoch 2, bad_epochs 0; epoch 2 벽시계 ≈ 10 h 00 m. 학습 손실이 epoch 평균(A44 수정)이라 검증 손실과 근접. 검증 손실 궤적: 0.8707 → 0.7029.
- ✏️ **09-13 12:59 PDT — 시드 123 epoch 3**: `epoch=2: train_ppl 1.9275, train_epoch_loss 0.6562, eval_ppl 1.9589, eval_epoch_loss 0.6724`(epoch 2 0.7029 → **−0.031**), best=epoch 3, bad_epochs 0; epoch 3 벽시계 ≈ 10 h 03 m. 궤적 0.8707 → 0.7029 → 0.6724. 링크 1 경계 09-14 06:49 PDT(epoch 5 도중).
- ✏️ **09-13 23:13 PDT — 시드 123 epoch 4**: `epoch=3: train_ppl 1.8822, train_epoch_loss 0.6324, eval_ppl 1.9204, eval_epoch_loss 0.6525`(epoch 3 0.6724 → **−0.020**), best=epoch 4, bad_epochs 0; epoch 4 벽시계 ≈ 10 h 14 m. 궤적 0.8707 → 0.7029 → 0.6724 → 0.6525(개선 폭 −0.168 → −0.031 → −0.020). **링크 1 경계 09-14 06:49 PDT(epoch 5 ≈ 70 %)** → `USR1: forwarding as SIGTERM`·`latest_state saved`·`exiting cleanly`·`link 1 ended at its limit` 확인 → 링크 2 `3131537` 큐 대기(노드 통째) → `Resumed from` 확인.
- ✏️ **09-14 06:41 PDT — 링크 1 경계 PASS**: 06:34 USR1 → 4 rank SIGTERM → `latest_state saved at epoch 4 step 36612 global_step 9810`(epoch 5 ≈ 74 %) → `exiting cleanly` → `link 1 ended at its limit` → `3131536` COMPLETED 0:0(1-23:47:31), 잠금 마커 제거. meta.json best 0.6525(epoch 4), bad_epochs 0. **링크 2 `3131537` PENDING, Slurm 추정 시작 09-15 09:19 PDT(≈ 27 h 대기)** → 재개 시 ≤ 200 스텝(≈ 1 h) 재계산. 12 h 링크 쪼개기 제안은 저자 결정 대기(대기 잡 TimeLimit은 사용자가 줄일 수만 있고 늘릴 수 없음).
- ✏️ **09-14 07:04 PDT — 저자 "11시간으로 돌려줘" → 11 h 링크 전환 완료.** `scontrol update TimeLimit=11:00:00` for `3131537`·`3131538`(48 h → 11 h, 큐 위치 유지); 11 h 링크 8개 추가 제출(`--time=11:00:00 --dependency=afterany:<prev> --comment=seed123 --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 --export=ALL,SEED=123,GPUS=4,CHAIN_N=k,CHAIN_MAX=11,CHAIN_PRESUBMITTED=1,LAUNCH_RETRY=0`, SIF 26.08, REPO_DIR repo_seeds): **링크 4~11 = `3149356`~`3149363`**. 체인 총 158 h. `runs/seed123_chain_jobids.txt`에 11개 전체 id 줄 추가(모니터·매시 보고는 마지막 줄을 읽음). 전환당 ≤ 200 스텝(≈ 1 h) 재계산; 남는 링크는 완료 시 취소/가드 종료. 직후 추정 시작은 여전히 09-15 09:19 PDT(우선순위 기준; backfill은 발생해야 반영). `submit_chain.sh`에는 APPEND 모드가 없어 손으로 붙임 — 추후 추가 검토.
- ⏳ **09-14 07:1x PDT — 저자: "EST를 학습에 써야 돼(다른 모델과 비교하려면)" → B6 arm 제안, 결정 대기.** 현행: A9가 "EST 가중 B5 학습(J)" 배제(split-only 비교 가능성), A14는 학습 종 EST·ONT·Sequel II+를 학습 측 증거로 허용(A13:586의 '검증 전용'은 A14가 대체). 제안: **B6 = B5 코퍼스 + 증거 지지 isoform(완전 체인, 클론/분자 ≥ 2, 학습 종의 학습 가능 EST·장독만)을 종별 GFF3에 추가 전사체로 주입 → 타일 빌드·동결(A40 파이프라인) → 4 GPU 런**; 시드 123(B5)은 그대로 비교군; A46 개정(애기장대 A-EST는 검증 전용 유지, 본문 모델 사전 지정). **준비 상태**: 11종 EST 원본 있음(`evidence/est/<종>/est.fa.gz`), UniVec 스크립트 있음(실행 미확인), 09-03 EST 정렬(배열 6147674)은 스크리닝 전이라 증거 불가 → 재정렬 필요; 장독 학습 가능 카탈로그 `TRAINING_EVIDENCE_v1.md:21-47`(B. distachyon 없음, S. bicolor ONT 없음, G. max PacBio 불가), 정렬 완료(216행). **C0 증거 표(9개)는 코드 0줄(issue #21)** — B1 검증에도 필요. 라벨 주입점: 종별 GFF3 → `Gene.transcripts`(`gsf_contract.py:184-274`, 유전자당 ≤ 15 전사체) → `geneList`/`window_genes`. 재사용 스펙: `PROTOCOL_M_exploratory_v1.md:20`(EST splice-graph 후보 생성기, ≤ 32 경로, 말단은 poly(A)/5' 증거만, ORF ≥ 90 aa), `:42`(증거 primary 구성 규칙: accession ≥ 2·라이브러리 ≥ 2·UTR 증거 없으면 CDS-only), A23 순위. **계획**: ① UniVec + 9종 EST 재정렬(1일) ② C0 구현 + Codex 리뷰(3~4일) ③ isoform 추출·GFF3 주입·빌드·동결(1~2일) ④ A46 ⑤ B6 런(≈ 4일 + 큐) ≈ 2주. 저자 "시작해" 대기.
- ✏️ **09-14 08:1x PDT — 저자 "univec 사용해서 clean 하고 재정렬해" → 학습 종 9종은 이미 완료(조사 에이전트 오보 정정).** pronghorn 실측: `evidence/est/<SP>/univec/DONE` 9종, 두 arm FASTA(≥100 primary·≥121 sensitivity) 존재, `evidence/est_align/<SP>/`·`est_align_min121/<SP>/`의 DONE이 스크리닝 FASTA md5와 일치, BAM quickcheck ok, PAF 있음 — 09-04 배열 `6148523_0-17`(게이트 3종 포함 `est_align.sbatch`, md5 e0641d9b). 매핑률 92~99 %(Athaliana 93.15 %, Sitalica 92.32 % 최저). 09-03 스크리닝 전 BAM은 `RETIRED_DO_NOT_USE/est_align_prescreen_20260903`. **검증 종 Zmays·Slycopersicum은 스크리닝이 없어 `univec_screen.sh`(MIN_LEN=100) sbatch 2건 제출**(cpu-s1-pgl-0, 로그 `evidence/est/univec_val_logs/`); 이들의 정렬은 B1 §3 검증 패스용 별도 드라이버(est_align.sbatch는 역할 게이트로 거부). B6 계획 ①단계는 사실상 완료 → 다음은 ② C0 파이프라인.
- ❌ **09-14 08:4x PDT — Codex 설계 리뷰: B6(per-base 보조 헤드) reject-as-designed, 13건**(원문 scratchpad `codex_review_output.txt`). CRITICAL: ① 추론 시 증거가 사라져 '증거 사용 주석기'가 아니라 표현 정규화에 그침 → 추론 시 증거 입력 채널 또는 독립 C2 모델이 증거 가치를 분리함 ② 14 클래스는 중첩 multilabel·배경/phase 클래스 없음(`gsf_contract.py:48`) → 14-way CE·positive-only BCE 모두 부적합 ④ A29 마스킹은 유전자 포함 기준이라 타일 가장자리 걸친 유전자(`partial`, `build_b5.py:561`)로 held-out 서열·EST 라벨이 샘 ⑤ paralog·반대 가닥 겹침·공유 접합부 좌표·플랭크 밖 정렬·다중 유전자좌 EST는 raster 전 **관측(분자) 단위 배제** 필요 ⑩ `segmentation_loss`는 최상위 출력 dataclass에 없음(제안서 오류), DDP는 이미 `find_unused_parameters=True`(`train:216`). HIGH: ③ B6 vs B5는 구조·RNG·최적화 혼입 → 같은 시드 λ=0 대조군 + 증거 셔플 대조군 필요 ⑥ 플랭크는 50~150 nt 시드값(A33:471), RC는 donor/acceptor·5'/3' 스왑 ⑦ `--secondary=no`는 분자 유일성 보장 아님(accession 버전·클론 mate·UniVec 분할·supplementary) ⑧ **애기장대 EST 이중 사용은 '유전자좌 분리 in-domain 검증'이지 독립 재현이 아님** — 독립은 검증 전용 장독뿐 ⑨ C0-lite로는 B1 재사용 불가 → 전체 C0 스키마 ⑪ 기울기 간섭·클리핑·Adam, 체크포인트 정책 ⑫ compile: 분기 없는 항상 실행 헤드, blob 7.2 MB/타일, 좌측 패딩(`datasets.py:707`) ⑬ A46 동결 항목 대폭 확대. **저자 결정 필요(갈림길)**: (A) 학습에만 증거(정규화; 손실·누출·대조군 수정) vs (B) 학습+추론 모두 증거를 **입력 채널**로(BRAKER/MAKER식 힌트, 채널 드롭아웃으로 de novo 유지) — '증거 쓰는 도구와 비교' 목적엔 B가 맞음.
- ✏️ **09-14 08:5x PDT — 검증 종 EST UniVec 스크리닝 완료**(pronghorn `6163269` Zmays 17분, `6163270` Slycopersicum 4분, MIN_LEN=100): Zmays 2,019,959 → 1,840,603 레코드(drop 179,445·trim 17,887·split 327), Slycopersicum 301,030 → 298,437(drop 3,208·trim 4,524·split 2,091). 산출: `evidence/est/<SP>/univec/`(DONE·PROVENANCE). ≥121 arm은 미생성(`retrim_a37.sh Zmays Slycopersicum`로 가능). 역할 b1_validation_only — B1 §3 검증 정렬용, 학습 금지.
- ⚠️ **09-14 09:58 PDT — Codex 2차(B6 v2: 증거 입력 채널 + 드롭아웃): "현 명세는 reject, 방향은 교정된 pilot 가치 있음"**(원문 scratchpad `codex_review_v2_output.txt`). 1차 13건 중 해소 6(추론 시 증거·보조 손실 제거·A-EST in-domain 표기·전체 C0·DDP·단일 체크포인트), 부분 7(③ C2 대조군은 **전체 길이**·초기화/데이터 순서 일치·증강 RNG 분리·증거 셔플/presence·증거 전용 재구성 기준선 ④ 타일 가장자리 held-out DNA는 빌더(`build_b5.py:567`)가 여전히 노출 → DNA 마스크도 고치고 서열 전용 비교군 재빌드 ⑤ valid-orthogroup 배제·A33 decoy/component 마스크 추가 ⑥/신규 RC: donor/acceptor 정체성은 RC에서 유지, strand만 뒤집힘(`tests/test_gsf_rc.py:26`) — 1차의 스왑 권고가 틀렸음 ⑦ 기존 정렬은 secondary 배치를 억제(`est_align.sbatch:144`) → 배치 복원 또는 감사 불가 분자 배제 ⑫ 좌측 패딩(`datasets.py:728`)·beam 생성·L×d 활성 메모리·resume 화이트리스트(`b5_runtime.py:224`)·RC 재raster·결정성 ⑬ A46 동결 항목 전부). **신규 CRITICAL: 검증 유전자좌에서 추론 입력으로 넣는 증거와 채점에 쓰는 증거를 분리해야 함**(Protocol M:14). 신규 HIGH: dose-response는 커버리지 층이 아니라 **같은 유전자좌에서 반복 nested thinning**; raster는 관측→타일 소속을 유지해 재raster 가능해야 함. 권고: 인코더 채널(HyenaDNA 뒤·다운샘플 전 bias 없는 투영, `modeling:725`), 드롭아웃 0.5 고정, 분자 보존 비율 {0.01,0.1,0.5,1} 샘플링 + 소스/라이브러리 withholding 대조, 특징 = 소스×strand 커버리지·접합부·unknown-strand·availability 마스크(단백질 phase 보류, log1p 클립, 참조 유래 특징 금지), 정지 규칙 de novo 검증 GSF 손실 patience 3·단일 체크포인트, 게이트 = de novo transcript-F1 하한 ≥ −1 pt(vs 전체 C2)·증거 이득 하한 > 0(vs off·증거 전용 기준선), locus 짝 bootstrap, 누출/RC/메모리 실패 시 중단. M(c) 증거 프롬프트는 싼 비교군. **다음: v3 = 교정 pilot 명세(A46 초안 + 게이트) → Codex 3차 → C0 구현 → pilot.**
- ✏️ **09-14 10:5x PDT — 저자: "codex 의견 반영하고 워크플랜 작성, de novo 모드 + 증거 모드로 진행" → `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md` 작성·커밋 `6c86169`·푸시.** 요지: A46 FREEZE 항목 14개(소스/역할·A9/A18 개정, 전체 C0, 분자 단위, 관측 단위 누출 배제 + 감사 정렬 `--secondary=yes -N 5`(배제용만), 가장자리 held-out DNA 마스크 수리 + 서열 전용 비교 코퍼스 재빌드, 트랙 명세(소스×strand 커버리지·접합부·unknown-strand·단백질·availability; 참조 유래 특징 금지; RC는 donor/acceptor 유지), bias 없는 투영(다운샘플 전), 드롭아웃 0.5 + thinning {0.01,0.1,0.5,1} 별도 RNG, 정지 규칙 de novo 검증 손실 patience 3, runtime identity 확장, **검증 유전자좌의 입력 증거/채점 증거 분리**(DATASET_ROLES `b6_input`/`b6_score`), 대조군 C1(시드 123)·C1'(수리 코퍼스 서열 전용)·C2(트랙 0, 전체 길이)·C3(셔플)·C4(presence)·C5(증거 전용 기준선), 고정 유전자좌 nested thinning dose-response, 게이트 G1(de novo F1 하한 ≥ −1 pt vs C2)·G2(증거 이득 > 0 vs off·C5)·G3(GSF 위반 ≤ B5), 중단 규칙). WP0 A46 초안 → WP1 C0(4~5일) → WP2 배제·마스크 수리·코퍼스 재빌드(3일) → WP3 트랙/thinning(2일) → WP4 모델/트레이너 + 4-rank 시험(2일) → WP5 pilot(≈ 75 GPU-h) → WP6 본 런(B6+C2+C1' ≈ 1,200 GPU-h → **예산 ≈ 600 GPU-h 보충 필요 또는 C1' 생략**) → WP7 평가. 총 5~6주. **저자 결정 대기 4**: 예산 보충 vs C1' 생략 · 본문 모델 · 옥수수 입력/채점 라이브러리 분할(제안: 입력 M-EST+M-HQ18, 채점 M-FLNC+root-tip ONT) · 121 nt arm 범위. Codex 3차(워크플랜) 진행 중.
- ✏️ **09-14 11:1x PDT — Codex 3차(워크플랜): reject 5건 → v1.1 반영·커밋 `320de01`.** 반영: **A46 승인·동결 전 WP1 착수 금지** 문장; 감사 정렬 `--secondary=yes -N 50 -p 0.5` + 직접 구간 검사, 감사 불가 분자 배제, 완전성 ≥ 99.9 %; 단백질 누출 단위 = 쿼리 서열(지지 단위 organism과 분리); C1' = 같은 DuckDB에서 트랙 무시(sha256 하나); thinning rank = hash(aug_seed, 종, 분자)/2^64 nested; aug_seed = 20260914 + seed, withholding 0.1/0.1; Dynamo 폴백 = 실패(`train:317`은 경고만), 메모리 중단 110 GB; 생성 경로가 트랙을 전달해야 함(`modeling:1050`); 역할 전이 규칙(train_track / excluded_heldout / never; excluded_heldout만 입력 가능, 입력 분자는 채점 금지); 옥수수 입력 M-EST + M-HQ18 / 채점 M-FLNC + root-tip ONT, RefGen F1은 reference-dependent 표기; pilot·최종 유전자좌 패널 seed 20260914로 분리 추출; 게이트 = GFFCompare exact intron chain·공통 분모·2,000회 짝 bootstrap·G3에 recall ≥ 0.9×B5 하한; 예산 정정(가용 ≈ 983, 부족 ≈ 292 + 60 GPU-h → 보충 ≈ 350 또는 C1' 생략). **저자 지시 "codex 끝나면 kimi3 리뷰" → Kimi K3 challenge 진행 중**(접근법 자체·최소 pilot·Codex 오류 여부). 그 뒤 WP0(A46 문안 + 저자 결정 4건 확정) → 저자 승인 → WP1.
- ✏️ **09-14 11:4x PDT — Kimi K3 challenge: approve-with-changes → 워크플랜 v1.2 커밋 `c4b9d30`.** Kimi: early-fusion 채널이 최선(별도 모델은 예산 2배·공유 de novo 비교군 상실, M(c) 프롬프트 토큰은 조립 체인이 필요해 단편 EST 90 % 배제, late-fusion 이득 없음); 보완: ① 생성 경로(`modeling:1050-1060` 고정 키, 인코더 1회 캐시 `:713-720`) beam/greedy 트랙 parity 시험 ② pilot 신호 부족 → 패널 크기·MDE(80 % power, 1 pt) A46 동결 ③ 메모리 벤치를 WP4 전에(>100 GB면 K/융합점 재설계) ④ **thinning 해시 미지정 — Python `hash()`는 프로세스별 salt → blake2b 고정 인코딩**(Codex 3회가 놓침); Codex v3 예산 지적은 v1.1에서 이미 해소. **WP-MVP(탐색, 1주)** 제안: 애기장대 1종·held-out 유전자좌 ~20개·입력/채점 수작업 분리·시드 123 best 체크포인트에 채널 얹어 짧은 fine-tune·fractions {0,1}·시드 3개·≈ 20 GPU-h → 효과 0/음이면 WP0 전에 중단, 양이면 WP0~WP7 정당화. Codex–Kimi 충돌 없음(실측 중재 불필요). **저자 결정: WP-MVP 착수 여부(권고: 착수).**
- ✏️ **09-14 10:5x~11:2x PDT — 저자 결정 연속**: ① C1(시드 123)이 유일한 서열 전용 비교군, C1' 생략(`460de14`; 예산 ≈ 935/983 GPU-h) ② 워크플랜 한글판 2종 추가(`b1acb65`) ③ **EST phase 채널 = 경로 B**(diamond blastx `--frameshift 15 --evalue 1e-5 --min-orf 30`, 교차 종 OrthoDB + Swiss-Prot reviewed Viridiplantae, HSP ≥ 30 aa, splice 투영, 경로 A(miniprot) 불일치 시 0, 같은 종 단백질 금지, `est_protein_hsp` 표, C6 phase-off ablation; `9a8c8e1`) ④ "경로 B로 진행" → pronghorn `6163304` `evidence/protein/diamond/build_dbs.sh`(Swiss-Prot 다운로드 + diamond makedb 2종, sha256·PROVENANCE) 제출, 세션 모니터 대기. **WP1(EST×단백질 blastx 등 구축)은 A46 승인 후.** 남은 저자 결정: 대표 모델 · 옥수수 입력/채점 분할 · 121 nt arm 범위 · WP-MVP 착수. GitHub 링크: README(en/ko) `revision/protocols/B6_WORKPLAN_README(.ko).md`, 상세(en/ko) `B6_EVIDENCE_MODE_WORKPLAN_v1(.ko).md`.
- ✅ **09-14 11:1x PDT — 저자 "동의"(권고 4건 채택)**: ① 대표 모델 = **B6 de novo 모드**(증거 모드는 같은 표에 나란히, B5는 보충자료 레거시 비교군; MVP나 G1에서 B6 de novo가 C2보다 유의하게 나쁘면(하한 < −1.0 pt) B5로 복귀하는 조건을 A46에 기록) ② 옥수수 입력 = M-EST + M-HQ18(Wang 2018), 채점 = M-FLNC(Wang 2020) + root-tip ONT(PRJNA822071), Wang 2016 제외 유지 ③ 121 nt arm은 채점/QC 표에만 ④ WP-MVP 착수. **진행 중**: `a46-drafter`(A46 v1.36을 PROTOCOL_B1_frozen_v1.md에 추가, 미커밋) · `mvp-designer`(`revision/protocols/B6_MVP_PILOT_v1.md` 설계, 미커밋) · pronghorn `6163304` diamond DB 빌드(모니터). 다음: A46 Codex 검토 → 저자 승인 → 커밋 → WP1 시작(diamond blastx 배열, C0) + MVP 구현.
- ✏️ **09-14 11:28 PDT — 경로 B diamond DB 완료**(pronghorn `6163304`, 6분): `evidence/protein/diamond/` — Swiss-Prot reviewed Viridiplantae **42,106개**(16.3 M aa, sha256 사이드카), OrthoDB v12 Viridiplantae 필터 **12,115,085개**(5.25 G aa, dmnd 5.75 GB), `DB.sha256`·`PROVENANCE.txt`(diamond 2.1.24, UniProt 쿼리 `(reviewed:true) AND (taxonomy_id:33090)`). 스크립트의 레코드 수 출력은 따옴표 이스케이프 문제로 비었고 `"`라는 잡파일이 생겨 삭제·수기 주석. WP1의 EST×단백질 blastx 입력 준비 완료(A46 승인 후 실행).
- ✏️ **09-14 11:2x~11:4x PDT — WP0 진행**: A46(v1.36) 초안 → Codex 검토 reject 10건(scratchpad `codex_review_a46_output*.txt`) → 수정 지시 반영(패널 결정론적 확정: `62_b6_panels.py` seed 20260914로 mvp 50 / pilot 300 / final = 나머지, **세 패널 서로 소**; 검정력은 WP5에서 기술 보고; 소스별 감사 정렬 명령; WP3 벤치 조건·MVP 체크포인트 정체성; `b6_input` 행의 held-out 추론 입력 carve-out(A13/A18.1 우선); A30 확장(교차 종 Swiss-Prot phase 입력); A33 겹침 성분 플랭크·A18.2 ±3 nt; dose-response 추출별 `aug_seed + d`; B5 복귀 = G1 비열등성 실패만(G2 실패는 증거 모드 주장만 보류); C3/C4/C6 = N_pilot·같은 시드/순서/솎아내기·최종 스텝; MVP 자체 정지 규칙) — A46은 프로토콜 파일 951~989행 + 로그 행 999(미커밋). **병렬**: `panel-builder`(gpu 호스트 B5 DB 읽기 전용 → `revision/protocols/b6_panels_v1.tsv` + sha256), `codex-review-mvp`(MVP 설계서 검토). MVP 설계서 `revision/protocols/B6_MVP_PILOT_v1.md`(미커밋): 애기장대 50 유전자좌(핵심 20), C0-lite 5표, K=8 트랙, 0 초기화 투영(인코더 래퍼, 생성 경로 kwargs 전달), 250 스텝 fine-tune × 3 시드 ≈ 18 GPU-h, **채점은 TAIR10 대비 reference-dependent F1**(A-ONT1/A-HiFi 정렬본 없음 — 장독 드라이버가 학습 부적격 세트 거부), 검정력 20 유전자좌 ≈ 30 pt·50 ≈ 20 pt만 검출. 다음: A46 Codex 재검토 → 저자 승인 → 커밋 → WP1.
- ✏️ **09-14 11:5x PDT — MVP 설계 Codex 검토: approve-with-changes 7건**(코드 인용은 전부 실물 일치). 수정 지시: ① STOP을 futility 검정으로(1차 지표 = valid 타일 증거 on/off 짝 검증 L_GSF, δ_loss 0.005 nats; 2차 = reference-dependent transcript F1, δ_F1 2.0 pt; 구조 지표 = 독립 장독 접합부 P/R; 한쪽 95 % 상한 < δ일 때만 STOP, GO = 하한 > 0 + 3 시드 양, 나머지 INCONCLUSIVE) ② `tile_est`를 valid 타일에도(안 그러면 증거 손실 = de novo 손실) ③ 5일차까지 A-ONT1/A-HiFi 유전자좌 정렬이 없으면 STOP → INCONCLUSIVE, 정렬 단계 필수화 ④ 배제 전 분자 중복 제거(chrom,start0,end0,strand + 클론 접두) ⑤ donor/acceptor는 전사 방향(`ts:A`)으로 정의 + RC 정체성 단위 시험 ⑥ STOP 전 라이브러리 withholding 민감도 1회 ⑦ SIF transformers/torch 버전 고정 기록, 캐시 생성 시험 fail-closed; 50 유전자좌 전부 디코딩, 패널 id는 `62_b6_panels.py` 기준. 수정 후 A46 rule 17(MVP 정지 규칙)과 동기화 필요.
- ✏️ **09-14 12:0x PDT — 패널 규칙 통일**(A46 재검토 에이전트가 A46 rule 12와 MVP 설계서 §1.2의 샘플러 불일치를 발견): `62_b6_panels.py` = 적격 held-out 애기장대 유전자좌 전부에 `shuffle_rank`(seed 20260914) 부여 → `b6_eligible_loci_v1.tsv`(+sha256; 하드 플래그 탈락 수 등 집계); `mvp_select_loci.py`(pronghorn EST BAM 필요) = rank 순으로 MVP 부필터(길이 ≤ 20 kb, 서로 다른 A-EST 분자 ≥ 3) 통과 첫 50개 = MVP, 건너뛴 유전자좌는 잔류, 비MVP 첫 300 = pilot, 나머지 = final → `b6_panels_v1.tsv`(+sha256). A46 rule 12·decision 5·rule 17, MVP 설계서 §1.2, 두 스크립트가 같은 문장을 인용하도록 세 작업자에게 지시. 적격 풀 크기·패널별 수는 승인 시 A46에 기입.
- ✏️ **09-14 12:1x PDT — 패널 1차 산출(구 규칙, 폐기 예정)**: 적격 2,143개(27,416 유전자 → held-out 3,430 → 다중 exon 2,491 → 창 적합 2,485 → 비중첩 2,143 → train_weight>0 2,143). **발견: 동결 DB의 geneList 행은 30,720 nt 타일**(유전자 아님) → 유전자 좌표는 gene_key_map, held-out은 gene_split, exon/intron 수는 build_manifest의 GFF(sha256 검증)에서; held-out 3,430 중 라벨 타일이 있는 유전자는 431개(나머지는 전부 N-마스킹). ChrC/ChrM 5개 포함돼 있어 **핵 염색체만** 필터 추가; `62_` 번호 충돌(62_geenuff_qc.py) → **`71_b6_panels.py`(적격 목록 + shuffle_rank) / `72_mvp_select_loci.py`(+ `72a_est_molecules_per_locus.sh`, pronghorn BAM)**로 개명·재실행 지시(구 `b6_panels_v1.tsv` 삭제). 예상 풀 ≈ 2,138, final = 풀 − 350. a46-drafter·mvp-designer에 이름·필터·타일 구조 문장 반영 지시. 시드 123 링크 2 여전히 PENDING.

- ✏️ **11:0x PDT — 22 epoch 상한의 근거 기록(`df9a415`, 프로토콜만 변경, repo_seeds는 c646e76 유지)**: 저자 설명 — 22는 공개 체크포인트를 만든 원래 학습(조기 종료 없음)의 길이이고, B5의 실제 epoch 수는 "검증 손실 최소·patience 3"이 만든다. A44에 'Epoch ceiling' 문단 추가. HEAD는 `df9a415`(코드는 c646e76과 동일).

### 저자 결정 대기 (2026-09-10 오전)
- [ ] ⏳ **"시작해"** — 시드 456/789 제출 신호(위 조건 충족 뒤)
- [ ] 🟡 **(c) 트레이너 파일 기반 stop 플래그** — `$RUN/STOP_REQUESTED`를 optimizer 경계마다 확인. 지금은 SIGTERM이 dynamo 아래서도 작동해 급하지 않음.
- [ ] 🟡 **(d)** 첫 launch `unspecified launch failure` NCSA 보고 여부 — GPU별 워밍업 + 할당 내 재실행으로 우회 중.
- [ ] 🟡 **(e)** medium 발견: `.mpi` 인덱스↔참조 체크섬 결속 — 수정/보류.
- [ ] 🟡 pending_grads를 rank 0만 쓰기(4 × ~1.7 GB → 1) — Codex 부수 의견, 선택.

## 🧊 진행 중 — 2026-09-09 저녁: 4 GPU 벤치 대기·시드 123 첫 epoch·체인 실증 완료 (기록용)

> ❌ **SUPERSEDED (2026-09-10 오전):** 위 오전 절이 대체한다. 4 GPU 벤치·epoch 1·SIGTERM 통합 시험은 끝났고, (a) 이미지는 저자가 26.08로 정했다. 아래 "~~21:50 CDT~~ ⚠️ CAUTION (09-10: 실제 21:45 PDT 시그널 / 22:00 PDT 상한)"는 틀린 시각(21:45/22:00 PDT).

- [>] ⏳ **SEED=123** `3116844` · 1 GPU gh015 · eager + reentrant ckpt · `/work/repo` 08f3e2b + `train_b5.slurm`(=`1a245ce`) · 25.06. 첫 `epoch=` 줄 ≈ **09-09 22:00 CDT**, **첫 48 h 경계 ≈ 09-10 ~~21:50 CDT~~ ⚠️ CAUTION (09-10: 실제 21:45 PDT 시그널 / 22:00 PDT 상한)**.
      - [ ] 첫 `epoch=` 줄 → `60_epoch_budget.py --log runs/seed123/train.err --patience 3`
      - [ ] 경계 뒤 `b5-seed-3116844.out`에서 `USR1: forwarding as SIGTERM to trainer ranks: <pid>` → `chain job 2` 제출 확인(또는 `forced preemption` 경로). `sacct -j 3116844`가 COMPLETED(정상 경로)여야 한다
- [>] ⏳ **벤치 26.08 compile 4 GPU** `3118904`(30 kb)·`3118906`(61 kb)·`3118908`(129 kb) — PENDING(노드 통째). 결과 → 스케일링 효율 → **시드당 GPU-h·벽시계** → 저자 (b)·(c).
- [ ] 🟠 **Codex 리뷰 재실행** — 체인 재시도·USR1/SIGTERM·watchdog 변경(`7beb803`…`1a245ce`)의 리뷰 2회가 로컬 메모리 부족으로 죽었다: `node …/codex-companion.mjs adversarial-review --wait --base 7df570f --scope branch "…"`.
- [ ] 🟠 **26.08 compile 통합 시험 SIGTERM 판** — `ddp_integration.slurm`의 `kill -USR1 "$ONE"`을 `kill -TERM`으로 바꾸고(dynamo가 USR1 핸들러를 막음, §1j) `sbatch --exclude=gh062,gh093 --export=ALL,SIF=…/transgenic-ngc-26.08.sif,TRANSGENIC_NO_COMPILE= --job-name=ddp-int-2608t deploy/deltaai/ddp_integration.slurm`. 26.08을 확인 시드 이미지로 쓰려면 PASS 필요.
- [x] ✅ **A28 체인 실증** — 리허설 `3120089`(정상 경로, COMPLETED 0:0) · `3120124`(강제 선점 경로, 자식 `3120146` 취소). 결함 6건 수정: `5bc4df8` rc · `4fe9780`/`3ed6413` rank 표적 · `9161508` SIGTERM · `840384f` watchdog · `7beb803`/`ed10056` launch-failure 워밍업·제자리 재시도 · `1a245ce` errexit. 테스트 17건. 리허설 디렉터리 seed997/998/999 삭제.
- [x] ✅ **DDP 통합 시험 25.06 PASS** — `3119681`(기본 할당기)·`3119682`(`ALLOC=native`). 26.08 compile(`3119683`/`3119779`)은 중단 경로만 미인증(위 항목).
- [x] ✅ **1 GPU compile 벤치 완료** — 21.35/40.54/86.64 s/step(1.17/1.12/1.12×), 1 GPU 에폭 ≈ 22.6 h. 대조 25.06 eager 오늘 31.63(non-reentrant)·29.96(reentrant).
- [x] ✅ 메모리 판정 — torch 101 GB = 순간 peak(§1j 정정의 정정). ~~(f) no-ckpt 129 kb 벤치~~ ❌ 닫힘.
- [x] ✅ #45 정정 코멘트 게시 — https://github.com/wyim-pgl/transgenic/issues/45#issuecomment-5606950770

- ✏️ **17:3x 추가 (Codex 리뷰 반영, `ba24b19`)**: 체인 스크립트 최종판 — GPU마다 워밍업, step-0 launch failure는 **같은 할당 안에서 먼저 재실행**(두 번째도 실패해야 노드 제외 재제출), 트랩은 두 번째 USR1 무시·모든 kill `|| true`, 강제 선점 마커는 `FORCED_PREEMPT.<jobid>`로 잡 단위(상한에서도 소비, launch 시그니처보다 우선). 테스트 18건. 동결 `/work/repo` 스크립트 = `ba24b19`. 리허설 `3120196`(정상 경로, SEED=999, CHAIN_MAX=1) 진행 중 — 통과하면 내일 밤 경계는 이 판으로 간다.

- ✏️ **18:2x 추가 — 4 GPU 벤치 완료 + 최종 리허설 PASS**: `3120196`(ba24b19) COMPLETED 0:0 — 내일 밤 경계용 스크립트 실증 완료. 26.08 compile 4 GPU DDP: 30 kb **5.89** s/step(1 GPU 21.35 → 3.62×, 90.6 %) · 61 kb **11.04**(3.67×, 91.8 %) · 129 kb **23.23**(3.73×, 93.2 %) → **epoch 6.15 h 벽시계 = 24.6 GPU-h**, 시드(8~10 epoch) 49~62 h·197~246 GPU-h, 두 시드 394~492 GPU-h — 잔여 968에서 시드 123 잔여(~190)를 빼도 **추가 배정 없이 들어감**(안전 여유로 ~300 요청 권고). 25.06 eager 4 GPU는 미측정(1 GPU eager 25.6 h/epoch × 같은 효율 ≈ 7.0 h/epoch, 28 GPU-h → 시드 225~280 GPU-h; compile 절감 ≈ 12 %). **권고**: 확인 시드는 25.06 eager 4 GPU(시드 123과 같은 이미지·커널, 3119681/3119682 + 체인 리허설로 완전 인증); 26.08은 12 % 절감 대신 A44 개정 + ddp_integration SIGTERM 재시험 필요. 드라이버 2초 샘플 최대 ≈ 97 GB/GPU(하한).

- ✏️ **09-10 09:55 CDT — 시드 123 epoch 1 완료**: `epoch=0: train_loss 1.6361 (ppl 5.13) · eval_loss 0.8617 (ppl 2.37)`, New best at epoch 1, `epoch_01/`·`best`·`trainer_state.json`(global_step 2071). epoch 2는 26 %(2.22 it/s) → 종료 ≈ 20:00 CDT; 링크 1의 48 h 경계 ≈ **~~21:50 CDT~~ ⚠️ CAUTION (09-10: 실제 21:45 PDT 시그널 / 22:00 PDT 상한)**(epoch 2 도중, latest_state 200 step마다). `60_epoch_budget.py --patience 3` = insufficient_data(2 epoch 이상 필요). 예산 949/999. 동결 스크립트 md5 64477d87 = ba24b19 확인.

### 저자 결정 대기 (2026-09-09 저녁)
- [x] ✅ ~~🟠~~ **(a) 확인 시드 456/789 이미지** — ✏️ 09-10 저자 결정 **26.08**(A44 `c646e76`). 옛 문장: 25.06 eager(통합 PASS, 시드 123과 동일 환경) vs 26.08 compile(1 GPU 1.12~1.17×; SIGTERM 통합 시험 PASS + 개정 A44 필요). 4 GPU는 결정됨.
- [ ] 🟠 **(b) ACCESS 추가 배정량** — 4 GPU 벤치 뒤 시드당 GPU-h로 계산.
- [ ] 🟡 **(c) 트레이너 파일 기반 stop 플래그** — `$RUN/STOP_REQUESTED`를 optimizer 경계마다 확인(수치 무관). 체인이 dynamo 아래 시그널 전달(지금은 SIGTERM이 됨)에 의존하지 않게. 동결 `/work/repo` 트레이너는 결정 전까지 손대지 않음.
- [ ] 🟡 **(d)** 첫 launch `unspecified launch failure`(gh062·gh093·gh121·gh139) NCSA 보고 여부 — 워밍업으로 우회 중.
- [ ] 🟡 **(e)** medium 발견: 청크 정렬 드라이버가 `.mpi` 인덱스를 참조 체크섬과 결속하지 않음 — 수정/보류.

### 승인 나면 바로 (2026-09-09 저녁)
- [ ] (26.08이면) A44 개정 + `train_b5.slurm`/`preflight_b5.sh` SIF 기본값 정리 + SIGTERM 통합 시험 PASS 확인.
- [ ] 시드 456/789: `sbatch --gpus-per-node=4 --cpus-per-task=64 --exclude=gh062,gh093 --export=ALL,SEED=456[,SIF=…] deploy/deltaai/train_b5.slurm` — **시드 123 epoch 1 판독 뒤**, `/work/repo`가 아닌 별도 체크아웃(새 동결 디렉터리; `train_b5.slurm`은 `/work/repo` 경로를 가정하므로 REPO 경로 파라미터화 필요 — 확인). `--cpus-per-task=64`는 `ddp_integration.slurm`의 값을 따른 것, 제출 전 확인.
- [ ] 위키 update 36 커밋(초안 `pgl-wiki/projects/transgenic/updates/2026-09-09b_*.md`, README·index·log 줄 포함, 미커밋) · `spec.md`.

## 🧊 진행 중 — 2026-09-09 낮: 벤치 매트릭스·DDP 통합 시험·시드 123 (기록용)

> ❌ **SUPERSEDED (2026-09-09 저녁):** 아래 항목의 상태는 위 블록이 대체한다. 계측 대조·통합 시험·1 GPU 벤치는 완료, (f) no-ckpt 옵션은 닫힘, (a) 코멘트는 게시됨.

- [>] ⏳ **SEED=123** `3116844` · 1 GPU gh015 · eager + reentrant ckpt · `/work/repo` 08f3e2b(+`train_b5.slurm` 한 파일 `5bc4df8` 패치) · 25.06. 13:10 CDT epoch 1 **46 %**(91,791/198,828, 2.1~3.1 it/s) → **epoch ≈ 26 h**, 첫 `epoch=` 줄 ≈ **09-09 22:00 CDT**. 예산 980/999(저자: 보충).
      - [ ] 첫 `epoch=` 줄 → `60_epoch_budget.py --log runs/seed123/train.err --patience 3`
      - [ ] 다음 링크(≈ 09-10 ~~21:50 CDT~~ ⚠️ CAUTION (09-10: 실제 21:45 PDT 시그널 / 22:00 PDT 상한))가 **패치된 스크립트로** 재제출되는지 `sacct`·`b5-seed-*.out`에서 확인(`training failed (rc=…)`가 이제 보인다)
- [>] ⏳ **벤치 매트릭스 26.08 compile** (repo_dev `7f20816`, STEPS=50, OUTTAG `c2608`): `3118903` 30 kb·1 GPU(✅ 21.35 s/step 1.17×, 드라이버 peak 31.5 GB / torch 101 GB) · `3118904` 30 kb·4 · `3118905` 61 kb·1 · `3118906` 61 kb·4 · `3118907` 129 kb·1 · `3118908` 129 kb·4. 출력 `repo_dev/b5-bench-<job>.out`(JSON + `driver_peak_mem_mib_per_gpu`), `runs/mem-*.csv`.
      - [ ] 정상 상태 s/step(warmup 20 폐기)·호스트 peak·**4 GPU 스케일링** → 시드당 GPU-h → 저자에게 ACCESS 추가 배정량
- [>] ⏳ **계측 대조** `3118909` · 25.06 eager · 30 kb · 1 GPU · nvidia-smi — 101 GB(25.06 보고) vs 11 GB(26.08 보고) 판정.
- [>] ⏳ **DDP 통합 시험** `3118861`(25.06 eager) · `3118880`(26.08 compile, `ddp-int-2608`) · 4 GPU · 480행 30 kb 부분 DB(`runs/ddp_int/subset_30720.db`). 판정 줄 `DDP_INTEGRATION PASS|FAIL`(`repo_dev/ddp-int-*.out`); FAIL이면 `runs/ddp_int/run_<job>/train.err`.
- [x] ✅ **Codex 리뷰 결함 5건 수정 — `5bc4df8`**(체인 rc·non-reentrant 기본·rank-0 체크포인트 소유·검증/학습 손실·USR1 플래그 합산·DDP KeyboardInterrupt·protein_accept v2·`ddp_integration.slurm`·테스트 22건). `14fe6a4`(통합 시험 compile 스위치) · `7f20816`(벤치 nvidia-smi) · `7df570f`(#45 v2 표 + A43/v1.33).
- [x] ✅ **#45 수용 v2** 배열 `6156132` 9종 완료 — 수용률·mapping_ambiguous 불변, 커버리지 −2~−7 %, ≥2 organism 인트론 −1.3~−4.1 %. v1은 `accept_v1_ambiguous_leak/` 보존.
- [x] ✅ compile probe: per-layer(25.06) ❌ · 26.04 ❌ · **26.08 ✅**(30 kb·129 kb). SIF `containers/transgenic-ngc-26.04.sif`·`-26.08.sif`.

### 저자 결정 대기 (2026-09-09 낮)
- [x] ✅ **(a) #45 정정 코멘트 게시** — 저자 "진행 해"(14:4x PDT) → https://github.com/wyim-pgl/transgenic/issues/45#issuecomment-5606950770 (이슈 닫힌 채 유지)
- [x] ~~🟡 **(f) `3118909` 결과가 드라이버 peak ≪ 101 GB이면**~~ ❌ SUPERSEDED (09-09 저녁): 샘플 최대는 하한일 뿐, torch 101 GB가 실제 peak → 닫힘(§1j): 26.08에서 `TRANSGENIC_NO_GRAD_CKPT=1` 129 kb 1 GPU 벤치 1런(`sbatch --export=ALL,TIER=129024,STEPS=50,REPO_DIR=…/repo_dev,SIF=…26.08.sif,OUTTAG=nockpt2608,TRANSGENIC_NO_GRAD_CKPT=1 --exclude=gh062,gh093 deploy/deltaai/bench_b5.slurm`). 들어가면 1.41× 무개정(활성값 재계산은 수치 불변). OOM이면 그것이 측정값 — 재시도 금지.
- [ ] 🟠 **(b) 확인 시드 456/789 이미지** — 25.06 eager(시드 123과 동일 환경) vs **26.08 compile**(벤치 6런 + `3118880` PASS 필요; 이미지·torch가 시드 123과 달라지므로 개정 **A44** 필요). 4 GPU는 결정됨.
- [ ] 🟠 **(c) ACCESS 추가 배정량** — 4 GPU 스케일링 실측 후 시드당 GPU-h로 계산해 드림.
- [ ] 🟡 **(d)** gh062/gh093 NCSA 보고 여부.
- [ ] 🟡 **(e)** medium 발견: 청크 정렬 드라이버가 `.mpi` 인덱스를 참조 체크섬과 결속하지 않음(`protein_align_chunked.sbatch:97-105`) — 이번 실행엔 오귀속 없음. 수정/보류.

### 승인 나면 바로 (2026-09-09 낮)
- [ ] #45 코멘트 게시(`gh issue comment 45 -R wyim-pgl/transgenic -F <초안>`).
- [ ] (26.08이면) A44 개정 + `train_b5.slurm`/`preflight_b5.sh` SIF 기본값·`TRANSGENIC_NO_COMPILE` 정리, 통합 시험 재확인.
- [ ] 시드 456/789: `sbatch --gpus-per-node=4 --cpus-per-task=64 --export=ALL,SEED=456[,SIF=…] deploy/deltaai/train_b5.slurm` — **시드 123 epoch 1 판독 뒤**, `/work/repo`가 아닌 별도 체크아웃(`repo_dev` 또는 새 동결 디렉터리).
- [ ] 위키 update 36(오늘 분) · `spec.md`(사이징 실측 = 벤치 매트릭스 값으로).

## 🧊 진행 중 — B5 SEED=123 학습 (2026-09-09 아침, 아래는 기록용)

> ❌ **SUPERSEDED (2026-09-09 낮):** 아래 항목의 상태는 위 블록이 대체한다. round 6은 완료, 2 GPU DDP 권고는 4 GPU 결정으로 대체, non-reentrant 기본값은 `5bc4df8`로 커밋됨.

- [>] ⏳ **SEED=123** DeltaAI 잡 `3116844` · 1 GPU gh015 · eager + reentrant ckpt · repo `08f3e2b` · NGC 25.06 · 시작 09-08 밤. 17 % @ 4:34, 2.4~2.9 it/s, ~~**에폭 ≈ 20 h**, 시드 7~9일~~ ⚠️ CAUTION (09-09 낮): 실측 **≈ 26 h**, A28 체인(48 h, CHAIN_MAX 8) 자동.
      모니터: `tail -7 /work/nvme/bilv/wcyim/transgenic/runs/seed123_monitor.log`. **`/work/repo`는 08f3e2b 동결 — 시드 도는 동안 pull 금지**(실험은 `repo_dev`).
      - [ ] ~~첫 `epoch=` 줄(≈ 09-09 낮)~~ → 위 블록(≈ 22:00 CDT)
      - [ ] 링크가 `unspecified launch failure`로 죽으면 노드(gh062/gh093) → `--exclude`로 재제출
- [x] ✅ **round 6** `3117671` — 129 kb · 2 GPU · eager + non-reentrant ckpt: **COMPLETED 55.64 s/step(1.74×), peak 100.5 GB.** ~~통과 시 확인 시드 DDP 권고 확정~~ ❌ SUPERSEDED (09-09 낮): 4 GPU 결정 + 통합 시험 조건부.
- [x] ✅ **#45 OrthoDB→9종 정렬·A19.2 수용** — 종료 2026-09-09(`6e08e01`, 수용 74.7–84.4 %). **#44** 종료. **#18** 벤치 코멘트 게시.
- [x] ✅ DeltaAI 발사대: R1–R6 해소, 결함 13+7건 수정(`48e161a`…`d7c3a6d`, `af6cce8`), A41/A42(v1.32).

### 저자 결정 대기 (2026-09-09 추가)
- [x] ~~🟠 **확인 시드 456/789에 DDP(2 GPU, non-reentrant ckpt)**~~ ❌ SUPERSEDED (09-09 낮): 저자 결정 **4 GPU**("4GPU로 할꺼야"). 근거 3117590(17.57 s/step, 1.42×)은 기록용.
- [x] ✅ **non-reentrant checkpointing을 트레이너 기본값으로** — `5bc4df8` `resolve_checkpointing`(`TRANSGENIC_CKPT_NONREENTRANT`는 no-op, reentrant는 1 GPU `TRANSGENIC_CKPT_REENTRANT=1`). 시드 123 체인엔 적용 안 됨(`/work/repo` 동결).
- [ ] 🟡 gh062/gh093 NCSA 보고 여부.
- [>] 🟡 4-GPU 스케일링 측정(2 GPU 71 %) — ✏️ 벤치 `3118904/3118906/3118908`(26.08 compile)로 측정 중.
- [ ] 🟡 시드 123 재시작 — **권고: 안 함**(compile 불가·인코더-only 무익).

### 승인 나면 바로 (2026-09-09 추가)
- [ ] `spec.md` — 사이징 실측(~~에폭 ≈ 20 h~~ ⚠️ 실측 26 h; 벤치 매트릭스 값으로)·체크포인트 사슬(A28+`af6cce8`+`5bc4df8`)·성공 판정(patience 3). intent open question 1 닫힘.
- [x] ~~non-reentrant 기본값 커밋~~ ✅ `5bc4df8` · [ ] `bench`/`train` env 정리 · `cpu_tools.slurm` account CHANGE_ME(Delta 전용)
- [ ] ~~시드 456/789: `sbatch --gpus-per-node=2 …`~~ ❌ SUPERSEDED (09-09 낮): **4 GPU**, 위 블록의 명령. (원문: DDP 결정 후; `/work/repo`를 새 커밋으로 올리는 것은 **시드 123 체인이 끝난 뒤**이거나 별도 체크아웃)
- [ ] 위키: update 35 게시됨(`62420bc`); 다음은 시드 첫 에폭 결과와 함께.

## 🧊 완료 — 단백질 정렬 (2026-09-08, 아래는 기록용)

- [x] ✅ **#45 OrthoDB → 9종 miniprot 정렬** — ~~SLURM `6150892_[0-8]`~~ → 3종 OOM → 청크 체인 6153193/6153195/6153196 → 수용 6154025 → **종료 2026-09-09**(`6e08e01`). 아래는 09-08 서술.
      2점 파일럿(`6150890` Athaliana, `6150891` Gmax, 각 단백질 10만)으로 사이징: 게놈이 8.45배면 시간은
      1.64배, 메모리는 1.59배 → **비용 ∝ genome^0.23**. 한 점만 봤다면 Gmax를 170 h로 잡았을 것이고
      실측은 33 h다. 합계 **243 node-hours**, `--time=48:00:00`·`--mem=64G`(실측 최대의 3.4배).
      드라이버 `revision/scripts/evidence/protein_align.sbatch`.
      🚨 **A19와 #45는 이 정렬이 Delta에서 돈다고 적는데 pronghorn에서 돌고 있다** — 저자 결정(2026-09-08).
      근거는 입력이 §1 md5로 동결돼 위치가 입력을 바꿀 수 없고, CPU 작업이라 GPU 크레딧을 쓰지 않으며,
      A19.4 3항이 "parallel with B5 training"이라 같은 기계를 전제하지 않는다는 것.
      드라이버 주석·PROVENANCE.txt·PROVENANCE.tsv에 적었다.
- [ ] 🔴 **§12 프로토콜 로그에 위 실행 위치를 기록** — 아직 없다. **첫 결과표 이전에** 넣어야 한다
- [ ] **정렬 완료 후 A19.2 수용 필터 9종 실행** — `revision/scripts/evidence/protein_accept.py` 완성·검증됨
      (파일럿: 100,851 정렬 → 78,847 수용 78.2 %, `mapping_ambiguous` 1,029, ≥2 organism 인트론 65,001).
      organism-depth sweep은 코드를 공유하지 않는 브루트포스로 대조해 **불일치 0**. 게놈당 수 분.
      끝나면 #45, 그리고 사실상 끝나 있는 **#44**도 함께 정리
- [ ] **#69·#70·#71** — A40 사전 검토에서 나온 병합 게이트 강화 3건. 다음 병합 전까지

---

## 표기 규약

| 표기 | 뜻 | 어디로 |
|---|---|---|
| `- [ ]` | 아직 안 함 | 여기 |
| `- [>] ⏳` | 진행 중 (담당·시작 시각을 적는다) | 여기 |
| `- [!] 🚫 (이유)` | 막힘 — **무엇에 막혔는지** 반드시 적는다 | 여기 |
| `- [x] ✅ (날짜)` | 완료 — **지우지 말고** 이렇게 바꾸고 §5로 내린다 | 여기 §5 |
| **기각·제외** | **여기에 상태로 남기지 않는다** | → [`quarantine.md`](quarantine.md) |

끝난 항목을 지우면 "왜 이렇게 됐나"가 사라진다. 기각·제외된 것은 [`quarantine.md`](quarantine.md)로 간다.

**이 파일에는 "기각" 상태가 없다.** 기각된 것은 상태 줄로 남지 않고
[`quarantine.md`](quarantine.md)로 **옮겨진다** — 그래서 헷갈릴 자리가 없다.

⚠️ 그래서 `[~]`도 쓰지 않는다. 랩의 다른 프로젝트(opuntia)가 같은 글자를 "기각"으로 쓰는데,
그건 기각을 todo에 상태로 남기고 있기 때문이다. 여기서는 그럴 일이 없으므로 진행 중은 `[>]`다.

---

## 1. A. thaliana ONT 후보 — 미해결 6항목 (다운로드 추가 전 선결)

Codex 검토(`codex_ath_ont_review_20260903.md` §6)에서 남은 것. **이게 정리돼야 #5 매니페스트에 역할을
적을 수 있고, 그래야 다운로드를 큐에 얹는다.** 617런 52 study 중 즉시 가용 후보는 약 120 GB.

> ✅ **2026-09-03 12:4x — 6항목 전부 해소.** Codex가 1·2·5를, 내가 3·4·6을 맡았다.
> 근거는 `codex_ath_ont_review_20260903.md` §7과 아래 표.

| # | 항목 | 결과 | 상태 |
|---|---|---|---|
| 1 | `PRJDB38182` run-level 매핑 | Table S1의 read_count 12개가 우리 TSV와 **순서까지 일치** → Col-0 `DRR807190/191/192` + `196/197/198`, `atg5-1` `193/194/195` + `199/200/201`. **70.5 GB 전체 적격, 6:6 분리 가능** | `[x]` ✅ |
| 2 | `PRJNA874115` (SPRI2) | 유전형이 **BioSample `alias`/`isolate`에 이미 있었다** — `sample_title`만 봐서 안 보였던 것. Col-0 3런 + spri2 4 + spri2-like 3 + 이중KO 4. 암술머리 특이 TF라 genome-wide RNA 가공이 아님 → **4계층 전부 적격**, 단 SPRI2/SPRI2-like 좌위는 마스킹 권고 | `[x]` ✅ |
| 3 | `PRJNA705300` | 애기장대 3런 중 **WT는 `SRR13799551` 1런(0.9 GB)**. `SRR13799559`=CPSF30 변이체, `SRR13799560`=FIP37 변이체 — 둘 다 RNA 가공 인자라 **배제** | `[x]` ✅ |
| 4 | 유전형 미확인 4건 | `PRJNA765684` 6런 WT(다종 study에서 애기장대만 선별됨) · `PRJNA596358` 1런 · `PRJNA796315` **CK 6런(대조군) / SSG 6런(처리)**로 분리 · **`PRJNA1393863`은 8런 전부 바이러스 감염·agro이고 `At_Buffer` 1런만 대조군** | `[x]` ✅ |
| 5 | AtRTD3 독립성 | **둘 다 AtRTD3 입력이 아니다.** 본문·참고문헌·supplementary 4종 전수 검색, 등장 accession은 `PRJNA755474` 하나뿐 (Zhang et al. 2022 Genome Biol 23:149) | `[x]` ✅ |
| 6 | `fastq_bytes=0` 13건 | fastq가 없는 게 아니라 **fast5/pod5 원시 신호**다(`submitted_format=OXFORDNANOPORE_NATIVE`, 예: 96.8 GB tar.gz). basecalling부터 해야 해 **범위 밖** | `[x]` ✅ |

### 유전형 확정 다운로드 후보 — 약 128 GB (지시 대기)

```
PRJDB38182      12런  70.5 GB   Col-0 6 + atg5-1 6
PRJNA1155260     6런  18.8 GB   종자, 클린
PRJNA874115     14런  15.1 GB   Col-0 3 + 변이체 11 (SPRI2 좌위 마스킹)
PRJNA765684      6런  10.9 GB   WT 추정, 변이체 표지 없음
PRJNA596358      1런   7.1 GB
PRJNA796315      6런   3.0 GB   CK 대조군만
PRJNA1393863     1런   2.1 GB   At_Buffer 대조군만
PRJNA705300      1런   0.9 GB   WT만
```

**중요 — 왜 6항목이 선결인가**: `PRJDB38182`는 70.5 GB인데 절반이 autophagy 변이체다. 매핑이 안 되면
쓸 수 없다. `PRJNA874115`도 같다. 유전형을 모르는 채 받으면 나중에 버리거나, 더 나쁘게는 **변이체
스플라이싱을 정답으로 학습**하게 된다.

### 확정된 것 (재검토 불필요)

- **360.8 GB 스킵**: `PRJDB38444`(238.3 GB — FLEP-seq nascent + 3′-cleavage 인자 변이체, WT 런도 방법론
  때문에 사용 불가) · `PRJNA1179505`(122.5 GB — GEO GSE280623이 **2029년까지 비공개**, 122런 중 약
  110런의 유전형 대조 불가)
- **ENA taxon 필드는 신뢰 가능** — 내가 반대로 보고했던 것은 질의 실수였다(자세히는 `quarantine.md`와
  `ath_ont_candidates_20260903.md`의 ❌ 마커). 나머지 8종도 같은 방식으로 조회하면 된다.
- **분류기 플래그 "—"는 "깨끗함"이 아니라 "분류 불능"** — `PRJNA815899`는 플래그가 하나도 안 붙었지만
  `sample_title`이 숫자 1–15뿐이라 정규식이 걸릴 문자열이 없었을 뿐이고, 실제로는 15런 전부 m6A writer
  knockout이며 WT가 없다.

---

## 2. #5 역할 매니페스트 — 세 항목 전부 해소, EST 게이트 통과

`revision/scripts/make_dataset_roles.py`(fail-closed). 전체 187행 — 학습 적격 91 · 검증 전용 33 · 제외 51+.

- [x] ✅ **Cui 2020 중복** — 저자 결정: **중복분은 쓰지 않는다.** 정본은 프로토콜 A18.3이 이름으로 지목한
      `ont/Athaliana_cui2020_PRJNA594286`, 학습 드라이버가 따로 받은 사본은 `excluded`. 두 사본은 압축을
      풀면 완전히 동일(실측); 파일 md5가 달랐던 건 gzip 헤더 타임스탬프 때문
- [x] ✅ **`source_checksum` 정의** — 저자 결정: **ENA 발행 `fastq_md5`**. 우리가 재압축한 결과물의 md5는
      `local_fa_md5`로 분리. ⚠️ 한계 기록됨: 페치가 fastq를 스트리밍만 하고 저장하지 않아 ENA md5는
      *무엇을 받았는지*는 문서화하나 *가진 것*과 대조 불가. 받는 시점 `tee` 검증이 해법이나 미구현
- [x] ✅ **검증기 구멍** — 데이터셋 가로지르는 런 중복 검사 추가. 넣자마자 Cui 2020을 잡았고,
      정본 미선언 시 매니페스트를 아예 쓰지 않는다
- [x] ✅ **EST 범위 §1 동결** — `DATASET_ROLES.est_v1.tsv`(11행, md5 `a6375713`)를 프로토콜 §1 고정 객체
      표에 등재(`2bf33d8`). 장독은 수집 완료 후 `DATASET_ROLES.longread_v1.tsv`로 v2 동결
- [x] ✅ **EST 정렬 완료 (2026-09-03)** — SLURM `6147674_[0-8]`, **9종 전부 COMPLETED**. 종당 25–117 s.
      잡 자체가 역할 매니페스트로 fail-closed(테스트 종은 배열에 없음). 결과 표는 `resume.md` 📊 절
- [x] ✅ **Athaliana 매핑률 조사 완료 (2026-09-03)** — **결함 아님, 구성 효과.** 원인은 GenBank EST
      디비전의 454 시대 짧은 라이브러리 2개(791,290건 = 전체의 51.7 %, 평균 91–92 bp). 길이를 통제하면
      Athaliana >120 bp 매핑률은 **96.87 %로 9종 중 4위**(범위 92.81–98.91 %). NCBI 원본 대조로 절단이
      아님을 확증했고 **벡터·비Col-0 오염 가설은 기각**했다 → `quarantine.md` §1d

## 2b. ✅ A21 UniVec 스크리닝 미실행 — 해소 (2026-09-03)

A21은 **"only the screened FASTA is aligned"**고 명시한다. 그런데 `evidence/est/<종>/univec/`가
**11종 전부 없고** `evidence/univec/`(UniVec_Core DB)도 없다. 스크립트는 있다
(`revision/scripts/evidence/univec_screen.sh`, `61_univec_trim.py`, 테스트 7건 통과).

→ **`6147674_[0-8]` 9종은 스크리닝 이전 패스여서 프로토콜상 증거가 아니었다(격리 완료).**
   코퍼스 진단에는 유효하지만 접합 호출에는 쓸 수 없다. 아직 접합·지지수를 계산한 적이 없어
   오염된 산출물은 없다(A36에 명시).

- [x] ✅ **A21 스크리닝 + A36 하한 적용 후 9종 재정렬 완료 (2026-09-03 15:23)** — SLURM `6147708_[0-8]`
      **전부 COMPLETED**. 제외 935,720건(16.6 %) · trim 81,357 · split 26,152 · 정렬 입력 4,718,673.
      **Athaliana 73.51 % → 96.96 %**, 9종이 92.80–98.94 % 안에 들어옴. 결과 표는 `resume.md` 📊 절.
      ⚠️ Vvinifera만 하락(97.61 → 97.12 %) — 원인은 split 조각 10,718건이 48.43 %로만 붙는 것이고,
      split을 빼면 98.32 %로 오른다(스크리닝이 키메라를 실제로 잡았다는 뜻)
- [x] ✅ **`est_align.sbatch` fail-closed 게이트 추가** — 이제 게이트가 둘이다: ①A18.3 역할 매니페스트
      ②A21 스크리닝(`univec/DONE` + `est.univec.fa.gz` 없으면 `exit 5`). 입력은 `est.univec.fa.gz`,
      PROVENANCE에 UniVec DB md5·버전·`summary.json`·`min_len=121`을 기록한다
- [x] ✅ **스크리닝 이전 BAM 9종 격리** — `evidence/RETIRED_DO_NOT_USE/est_align_prescreen_20260903/`
      (`WHY_RETIRED.txt` + 실행된 원 스크립트 `est_align.sbatch.as_run` 동봉, README에 절 추가).
      삭제 아님 — `resume.md` 📊 절과 `quarantine.md` §1d의 코퍼스 진단 근거가 이 BAM들이다
- [x] ✅ **게이트 실동작 검증(Sitalica)** — 66,027 → 65,513건. drop 542 = 원래 ≤120 nt **540**
      (내가 BAM에서 잰 값과 일치) + 트림 후 짧아짐 2. 출력 최소 길이 **정확히 121**. trim 82 · split 44

## 2c. 🔴 Codex 외부 검토 미해결 (2026-09-04) — 우선순위 최상

전체 검토는 위키 (32) 이후 항목과 `quarantine.md` §1e. **해소된 것**: A36→A37 전환, `est_align.sbatch`
3중 수정, 핸드오프 불일치 5건. **남은 것**:

- [x] ✅ **`SRR25203456` 선언 행 + `longread_v1` 재동결 — A38(v1.28)로 해소** (2026-09-04).
      빌더에 `DECLARED_RUNS` 기구 추가(자원 선언표와 별개 — 이건 *런*이라 long-read 스코프에 남아야 한다).
      행의 `note`는 `DECLARED_NOT_SCANNED:`로 시작해 선언 행과 스캔 행을 grep으로 가른다. 가짜 `.DONE`이나
      빈 FASTA는 만들지 않았다 — 그러면 수집된 증거와 구분이 안 되고 그게 바로 고치려던 오류다.
      `validate()`는 선언 런이 정확히 1행일 것을 요구한다(나중에 실제로 받으면 스캔이 2행을 만들어 빌드 거부).
      **재동결**: `d644278f`(61,801 B·214행·129/31/54) → **`c2f4c2e8`**(62,511 B·**215행**·129/31/**55**).
      추가된 1행은 `excluded`이고 **적격 판정도 증거 내용도 바뀌지 않았다**. 테스트 18건 통과
- [x] ✅ **`longread_fetch.sh`의 A18.8 위반 3건 — `longread_fetch_v2.sh`로 해소 (`12a89fe`)**
      v2는 파싱된 컬럼으로 필터하고, 낡은 호출자 4가지 인자 형태를 전부 `exit=6`으로 차단한다(실측 확인).
      테스트 44건 추가. ✏️ 이 줄은 2026-09-08까지 "주말 매핑 이후로 연기"로 적혀 있었으나 **매핑은 끝났고
      v2도 이미 들어가 있었다.** 아래 진단 표는 근거로 보존한다 —
      ①`grep -E "$FILT"`가 **파싱 전 행 전체**에 걸린다(A18.8은 파싱된 컬럼 기준을 요구)
      ②`fastq_ftp`의 **첫 URL만** 가져간다 → 다중 파일 런은 일부만 받고도 `.DONE`이 런 전체를 뜻한다
      ③레코드 수를 계산하면서 ENA `read_count`와 **대조는 안 한다**

      ⚠️ **Codex의 첫 패치(`[[ "$run" =~ $FILT ]]`)는 적용하면 안 된다 — Codex도 철회했다.**
      이 저장소에서 `FILT`는 run 접근번호 필터가 **아니다**. 실측(2026-09-04):

      | 필터 | 행 전체 | run 컬럼 | 실제 걸리는 컬럼 |
      |---|---|---|---|
      | `OXFORD_NANOPORE` | 2 | **0** | `instrument_platform` |
      | `PACBIO` | 1 | **0** | `instrument_platform` |
      | `maize` | 0* | **0** | `submitted_ftp` (5행, `RS_Sequel_maize_leaf_hq.fastq.gz`) |
      | `SRR25203456` | 1 | 1 | `run_accession` |

      \* Osativa filereport 기준. Zmays PRJEB22122에서는 `submitted_ftp` 5행에 걸린다.

      run 컬럼에만 걸면 `OXFORD_NANOPORE`/`PACBIO`로 거르는 데이터셋이 **전부 0런 선택**되고
      `0 runs DONE`으로 조용히 끝난다 — 2026-09-03 PRJDB38182에서 Col-0 6런이 4시간 사라진 그 실패 양식이다.
      `queue_fetch.sh:12`·`ont_parallel.sh:10`의 `"${r}[[:space:]]"`도 파싱된 컬럼에는 절대 안 맞는다.

      **확정 설계**(Codex, 실측으로 검증): `FILT`→`RUN_RE`(run 전용) + **`PLATFORM_RE`**(instrument_platform)
      + **`SUBMITTED_RE`**(submitted_ftp), 기존 `MODEL_RE`/`STRAT_ALLOW` 패턴을 따른다.
      필터 미지정 = 제약 없음, 지정했는데 **대상 필드가 비면 `UNRESOLVED`**(never DONE) — 메타데이터
      결측이 선택을 조용히 넓히면 안 되므로. 마이그레이션 가드 둘: 낡은 4번째 인자(`*[[:space:]]` 또는
      `OXFORD_NANOPORE`/`PACBIO`)면 **exit 6**, 그리고 **선택 행이 0이면 exit 6**(후자가 본질적 가드 —
      낡은 표현을 전부 알아볼 수 있는 휴리스틱은 없다). 호출자 **5개를 같은 커밋에** 고친다.

      ⏸️ **왜 지금 안 하나**: 장독 수집은 이미 끝났고(122런 150 GB, 회수 7런은 ENA md5·read_count 대조 완료)
      이 드라이버는 **주말 매핑 임계경로가 아니다**. 드라이버 + 호출자 5개를 건드리는 변경을 마감 앞에
      넣을 이유가 없다. Codex 권고대로 매핑 이후 **`longread_fetch_v2.sh`로 한 번에**, 저장해 둔
      `filereport.tsv`를 픽스처로 검증한 뒤 교체한다.
      ②③은 FILT와 독립이고 호출자 변경이 필요 없어 따로 먼저 적용 가능하다(완료된 런은 `.DONE`에서
      빠져나가므로 122런은 다시 쓰이지 않는다). ⚠️ 그때 함께 고칠 것: **`fastq_md5`는 `fastq_ftp`와
      위치 대응**이므로 지금처럼 `submitted_ftp`에서 고른 URL을 그 md5로 검증하면 **틀린 대조**다
- [x] ✅ **staging 게이트 강화 + 저장소 편입 — 커밋 `f5a257f`** (2026-09-04). md5 대조·9종 완비·매니페스트
      부재 시 실패를 전부 넣었고, 정본은 `transgenic/revision/scripts/65_make_training_input_manifest.sh`다
      (`training_input/` 쪽은 심볼릭 링크가 **아니다** — 번들이 ACCESS로 rsync되므로 밖을 가리키면 깨진다).
      MANIFEST.tsv는 임시 파일에 쓰고 원자 교체하며, stat·md5 실패 시 매니페스트를 쓰지 않는다.
      적대 시험 실증: Zmays 내용을 `Athaliana_167_TAIR10.fa` 이름으로 → `rc=2`(md5 mismatch), 매니페스트 부재 → `rc=2`.
      테스트 7건(`test_make_training_input_manifest.py`), 28.9 GB 번들 없이 돈다
- [ ] 🟡 **A37 쌍대 보고 구현** — EST 유래 수치마다 두 arm(≥100/≥121)을 길이 구간별 callability·접합 지지·
      사슬당 인트론 수·유전자 길이 분포와 함께. 채점기(#24) 설계에 들어가야 한다

## 2d. 🔴 A39 (v1.29) — §3.3 항목 2의 `-uf` 게이트가 공허했다 (2026-09-04)

- [x] ✅ **문자 그대로의 지표는 라이브러리 가닥성에 눈이 멀어 있다.** §3.3 항목 2를 그대로 읽으면
      **P(T = A)** — minimap2가 모티프에서 추론한 전사체 가닥 대 주석 가닥. `T`와 `A`는 같은 전사체를
      두 번 읽은 것이라 **가닥성이든 아니든 일치**한다. 27런 실측 0.9929–0.9987, **전부 PASS**.
      고정 95 % 문턱이 아무것도 거르지 않았다
- [x] ✅ **`-uf`가 주장하는 것은 P(R = A)** — 리드 자체가 센스 가닥인가. 같은 정렬 집합에서 세 부류로 갈린다:
      **0.128–0.186**(안티센스 3런) · **0.492–0.527**(비가닥 20런) · **0.991–0.994**(센스 4런).
      문자 그대로 갔으면 27런 전부 `-uf`를 받고, 센스가 아닌 23런에서 **접합 절반의 가닥이 뒤집혔다**
      (§5가 그 태그를 읽는다). 무익한 게 아니라 **접합 세트를 망가뜨리고 티도 안 났을 것**
- [x] ✅ **A39/v1.29로 게이트 이동, 문턱은 불변.** 기존 지표는 `motif_annotation_agreement`로 남겨
      진단에 쓴다(잘못된 레퍼런스·깨진 FLAG 변환·비정규 모티프). 완료 마커를 **채점기 md5에 묶어**
      낡은 판정이 현재 것으로 안 읽히게 했다. 커밋 `6ebe904`
- [x] ✅ **A36 모양이 아닌 이유**를 개정문에 명시 — 결함이 관측 전에 참인 **추정량의 대수**에서 나왔고,
      대체 지표는 minimap2의 `-uf` 계약이 정하며, **문턱 불변**, 그리고 변경이 **더 엄격**해진다
      (27/27 → 4/27 통과). 느슨하게 만든 개정이 아니다

## 2e. 🟡 장독 정렬 이후 정리 (2026-09-04/05)

- [ ] 🟡 **`longread_align.sbatch`의 `orientation` 줄이 grep으로 모호하다** — `-> no -uf`가 `-uf`로 끝나서
      켜진 경우와 구분이 안 된다. 정보는 `command` 줄과 `DONE`의 `uf=`에 명확히 남아 복구는 되지만,
      요약을 자동으로 뽑을 때 걸린다. **실행 중에는 고치지 않는다**(A18.8: 드라이버는 돌 때 편집 금지 —
      중간에 바꾸면 기록 형식이 런마다 갈려 더 나쁘다). 정렬 완료 후 `uf=-uf` / `uf=none`처럼
      단일 토큰으로 바꾼다
- [ ] 🟡 **MAPQ<20 이상치 두 데이터셋 기록** — 감사 부산물로 나온 실측. 중앙값 2.7 %인데
      `seed_flcdna_PRJNA1155260`(Athaliana) **29–51 %**, `wm82_graft_PRJNA648759`(Gmax) **34–35 %**.
      §4가 MAPQ ≥ 20을 요구하므로 이 데이터셋들은 **리드 수가 시사하는 것보다 훨씬 적게 기여**한다.
      Gmax는 고배수체라 다중 매핑이 많은 것이 부분적 설명이지만 Athaliana의 51 %는 설명되지 않는다.
      완결성 QC 표에 넣을 것

## 2f. ⚠️ 장독 정렬 메모리 산정이 천장에 붙어 있었다 (2026-09-05, 실측)

- [x] ✅ **32 GB는 너무 빠듯했다.** `6148978_119`(Vvinifera `callus_PRJNA732451`, 입력 3.14 GB)가
      `.batch` MaxRSS **29.34 GB**로 OOM. ⚠️ **이건 `.extern` 함정이 아니다** — `.batch` 스텝 자체가
      `OUT_OF_MEMORY`다. 완료된 것들도 아슬아슬했다: 태스크 8이 **29.50 GB**, 49가 29.25 GB로 살아남았다.
      배열 전체가 32 GB 천장에 닿은 채 돌았다
- [x] ✅ **왜 틀렸나** — Codex가 EST 실적(최대 RSS 10.9 GB) + sort 8 GB ≈ 19 GB로 추정해 48 GB를
      "과도하게 보수적"이라 했고 내가 받아들였다. **짧은 리드에서 긴 리드로 외삽한 것이 틀렸다.**
      minimap2는 ONT 장독에서 훨씬 많이 쓴다(실측 29.5 GB). 원래 48 GB였으면 안전했다
- [x] ✅ **드라이버는 편집하지 않고**(A18.8, 배열 실행 중) `sbatch --array=119 --mem=96G`로 재제출(`6149508`)
- [x] ✅ **`--mem` 32G → 96G, 동시성 %4 → %3** (2026-09-05, 배열 완료 후). 실측 최대 29.5 GB에
      32 GB는 천장이었다. 96 GB로 재시도한 2건 모두 성공. 동시성을 3으로 낮춘 것은 96 GB × 4가
      노드 메모리를 넘기 때문
- [x] ✅ **`orientation` 줄을 `uf` 한 토큰으로 교체** — `-> no -uf`가 `-uf`로 끝나 grep이 켜진 경우와
      구분을 못 했다. ⚠️ **기존 129런의 `PROVENANCE.txt`는 낡은 형식 그대로다** — 재생성하지 않았다
      (정렬을 다시 돌려야 하고, 정보는 `command` 줄과 `DONE`의 `uf=`에 이미 정확히 있다).
      그 129개를 읽을 때는 `orientation` 줄이 아니라 **`DONE`의 `uf=`를 볼 것**
- **교훈**: fail-closed가 제 일을 했다 — BAM도 DONE도 안 남겼고 부분 결과가 완성본으로 기록되지 않았다.
  그러나 **자원 산정은 유사 작업의 실적에서 외삽하지 말고 그 작업 자체를 한 번 재보고 정할 것**

## 3. 저자 결정 대기 — 이게 병목

| 항목 | 막고 있는 것 |
|---|---|
| **§3.3 항목 1: 정렬에 쓴 FASTA가 §1 객체가 아니다** | A. thaliana `513e5ef3`(genomes/Athaliana_167_TAIR10.fa) vs §1 `987ca803`(revision/data/TAIR10/TAIR10_genome.fa). 저자 진술: **헤더/포맷 차이일 뿐 서열 동일**(2026-09-04, 측정 아님). A. thaliana는 학습 종이자 B1 복제 종이라 어느 권위가 이 패스를 지배하는지가 갈린다 — 진술을 조항으로 굳힐지, 측정으로 확인할지 |
| **ACCESS 자원명·계정·파티션** (아이디 생성 중) | **#51·#19·#20.** Delta A100 80 GB인지 DeltaAI GH200 96 GB인지에 따라 벤치마크와 크레딧 환산이 달라진다. 4090에서는 129 kb 계층 측정이 물리적으로 불가 |
| **#61** 논문의 "test set" 정의 | 명목 orthogroup split vs 실제 라벨된 16.3 %. Methods 문장이 갈린다 |
| 검증 세트 PacBio 2건 | `Athaliana_zhang2023`는 CCS 변환 미실행(제외가 아님), `Zmays_B73_ccs`는 이름만 ccs |
| `CLAUDE.md`을 git에 넣을지 | 지금 `.git/info/exclude`라 마킹이 이 머신에만 있다 |
| **AtRTD3를 "깨끗한 정답"으로 쓸 수 있는가** | Codex가 심사 이력까지 읽은 결과, **AtRTD3 자체 Iso-Seq에 `upf1`·`upf3`·`xrn2/3/4`(NMD·분해경로 변이체)와 핵분리 RNA가 포함**돼 있고 심사자 둘 다 aberrant transcript 오염을 우려했으나 저자들이 의도적 포함이라 답변했다. **우리가 후보에 적용하는 기준을 우리 검증 기준 자신이 통과하지 못한다** — 검증 설계에 영향 |

---

## 4. 승인 나면 바로 (한 줄~반나절)

- [x] ✅ **#59 seedable sampler — 구현·검증·커밋 완료 (`1413efd`, 2026-09-05 10:22).** ✏️ 이 줄은 2026-09-08까지 "미커밋"이라고 적고 있었다.
      `use_seedable_sampler=True` + 재개 시 명시적 `set_epoch(epoch)`로 순열과 절대 배치 인덱스를 보존한다.
      GPU 실제 체크포인트/새 프로세스 회귀검사 **6 passed** (기존 버그 음성 대조 포함), 전체 로컬
      **473 passed, 6 skipped**. **ACCESS 첫 시드 전 적용; 기존 잡 체인 도중 코드 변경 금지(A28).**
      구현·다음 순서·저자 결정 미해결: [`issue59_resume_20260905.md`](transgenic/notes/issue59_resume_20260905.md).
- [ ] **원고 Methods "partially pretrained"** (A34) — 미반영
- [x] ✅ **#57 `tier_margin_unguaranteed` 빌더 구현 완료 (`28cbaa5`)** — `scripts/report_b5_database.py`가
      쓴다. 독립 재현으로 **3,418 / 653 / 86**과 최장 유전자 **196,414 nt**가 프로토콜 수치와 일치함을 확인했다.
      ✏️ 이 줄은 2026-09-08까지 미구현으로 적혀 있었다
- [x] ✅ **`SRR25203456` 제외 (저자 결정 2026-09-04, #68 종료)** — ENA 경로가 디렉토리라 재시도·재개로
      도달 불가(실측 확인). NCBI S3 우회 경로는 검증됐으나(200 · 재개 · 12.1 MB/s) 드라이버에 SRA 폴백을
      넣는 대신 **런을 버리기로 결정**. `TRAINING_EVIDENCE_v1`와 프로토콜 §1 동결 주석 **양쪽에 기록** —
      트리에 없어 매니페스트 행이 생기지 않으므로, 계획 대조 시 "공백"이 아니라 "이유"가 나오게 하기 위해.
      ✏️ **(2026-09-04 정정)** ~~`longread_v1` md5 `d644278f` **불변**(재동결 불필요)~~ — Codex 지적대로
      **틀렸다**. A18.3은 산문이 아니라 **행**을 요구한다. A38(v1.28)에서 선언 행을 넣고 **`c2f4c2e8`로 재동결**했다.
      "트리에 없어 매니페스트 행이 생기지 않는다"는 것은 빌더의 한계였지 규칙의 면제가 아니었다.
      PRJNA953663의 **증거 기여분은 여전히 0**이다 — 바뀐 것은 그 0이 기록되는 방식뿐이다

---

## 5. 진행 중 / 오늘 끝난 것


- [x] ✅ **A40 / v1.30 동결·§1 채택 (2026-09-06)** — 저자 승인, 새 DB와 `data/freeze/` 미러 완료. HEAD `aa2623f` + 실행 patch + source SHA256, DB/source/table hashes, 실제 argv와 검증 결과 보존. 334,642 assignments, 213,702/29,153/91,787 splits, strict 3,430 불변; +924의 미설명 변화 0.
- [x] ✅ **#55 해결 — 저자 결정 2026-09-05, rebuild 해결 기록 2026-09-06**: emitter/validator canonical order 공유, order reject 0. 최종 직접 복구 **19 forward +19 RC**; 97+63은 counterfactual 전용.
- [x] ✅ **#56 해결 — 저자 결정 2026-09-05, rebuild 해결 기록 2026-09-06**: no-CDS 6,205개 마스킹·overlap closure·rate population 포함, coding/decoy label 없음. 독립 N-coverage 검사 PASS.
- [x] ✅ **#58 해결 — 저자 결정 2026-09-05, rebuild/tokenizer 해결 기록 2026-09-06**: empty target `<s> <empty> </s>` 3토큰; 새 empty 26,057개 및 전체 273,148개 실제/저장 token count 일치.

**진행 중**
- [x] ✅ **A. thaliana ONT 7런 회수 완료 (2026-09-04 00:09)** — 라운드 3회. 전부 ENA md5 일치 + 레코드 수가 발행 `read_count`와 일치. `ath_queue` **30/30**: `PRJDB38182` Col-0 6런
      (DRR807190/191/192/196/197/198, 13:01–13:03 전부 FAILED) + `PRJNA1393863` 1런(SRR36603452, 14:24 FAILED).
      **실패는 일시적이었다** — 두 URL 다 지금 HTTPS 200이고 실제로 받아진다. 큐는 **미수집 목록에서 생성**했다(손으로 적지 않음).
      약 37 GB. ⚠️ **이게 되기 전에는 `DATASET_ROLES.longread_v1.tsv` v2 동결을 하면 안 된다**
- [x] ✅ ~~**ONT 수집**~~ 완료 12 스트림 · `evidence/training` 142 GB. Ppatens 7런 **완료**, Osativa 28런, Athaliana 9런
- [x] ✅ ~~**`SRR25203456`**~~ — **다운로드 실패로 확정, 제외**(2026-09-04, #68 종료). 위 §4 참조. 막힌 항목이 아니다

**완료 (2026-09-03)**
- [x] ✅ **#50** 9종 DB 병합·동결 — 272,224행, 검증기 위반 0, 내용 sha256 동결 → **#16도 종료**
- [x] ✅ **#60** PacBio subreads 51런 43 GB 제외·격리 (저자 결정) — 삭제 아님, 되돌릴 수 있음
- [x] ✅ **#62** 데이터셋이 실패 행을 무작위 대체하던 것 → `strict=True`에서 예외 (테스트 5건)
- [x] ✅ **#63 이슈 종료 (2026-09-03)** `library_strategy` 화이트리스트 — WGS 차단 확인. 본문 2번(`SRR25203456` 재시도)은 **필터가 아니라 다운로드 고장**이라 분리했고 §4에 남아 있다
- [x] ✅ 신규 이슈 **#55–#63** 등록
- [x] ✅ `quarantine.md` 신설 · `resume.md` 4,901줄 → 256줄 추출 (무손실 확인)
- [x] ✅ `README.md`·`CLAUDE.md` 낡은 부분 마커 격리
- [x] ✅ 위키 (24)–(28) + `guide/handoff-hygiene.md` 3개 절(추출·세 파일 체계·기각은 상태가 아니다)
- [x] ✅ **#5 EST 범위 동결** — 프로토콜 §1 등재, EST 정렬 게이트 통과
- [x] ✅ A. thaliana ONT 스카우팅 — 617런 52 study, 360.8 GB 배제, 변이체 제외 후 30런 82.5 GB 수집
- [x] ✅ `ruleout.md` → `quarantine.md` 개명 + 미해결 항목 분리
- [x] ✅ **이슈 18건 서브에이전트 검증 (2026-09-03)** — 닫음 **#11·#12·#13·#14·#46**(+ 앞서 #63).
      열어 둔 13건은 근거를 이슈 코멘트에 남김. **#17은 저자 판단 대기** — 체크포인트 메타에 DB/config
      **해시**가 없고 경로+recipe 전체가 저장되는데, 프로토콜 A28은 해시를 요구하지 않는다(이슈 문구와 A28이 갈림)
- [x] ✅ **검증 중 발견한 결함 4건 등록·수정 (2026-09-03)** `0d92741` — **#64** `ROLES`에 `sensitivity_set`
      추가 + 매니페스트 첫 테스트(3건) · **#65** 빈 `instrument_model`을 skip→`UNRESOLVED`+exit 4 ·
      **#66** OrthoDB md5를 프로토콜 §1에 등재(md5 직접 재계산) · **#67** wm82_graft 4런→**8런** 정정.
      #64·#65·#67 종료, **#66은 절반만**(§1 완료, `DATASET_ROLES.tsv`는 저자 결정 대기 → §3)
- [x] ✅ **저자 결정 2건 반영 (2026-09-03)** — **#17 해시 불필요**(A28이 이미 "path + recipe"만 요구하므로
      프로토콜 개정 없음). 남은 기준은 **동결 B5 DB 272,224행 직접 질의로 충족** — orthogroup 교차 0 ·
      RC split 불일치 0 · forward 없는 RC 0 · strict_holdout 비-test 0 · gene 다중 split 0 · Zmays 0.
      **#66 비-런 자원은 빌더 선언표**(`02868a8`) — OrthoDB + Swiss-Prot 2행, 매니페스트 187→220행
      (자원 2 + 오늘 수집된 ONT 31런), `est_v1` md5 `a6375713` 불변 확인. #17·#66 종료
- [x] ✅ **테스트 스위트 통합 (2026-09-03)** `73397ac` — 같은 이름 파일 둘 때문에 `pytest`가 **수집에서
      중단**되던 것을 해소. 낡은 `revision/scripts/tests/test_geenuff_qc.py`(추측 스키마, `232ebe7`이
      실제 스키마로 대체)를 삭제하되 **리포 전체에서 유일하던 2건**(`loss_mask_decision`=A22 정책,
      `read_flags_tsv`)은 `tests/`로 이관. **409 passed, 1 skipped** — 이전엔 155건만 돌고 있었다
- [x] ✅ **자원 선언표 실측 대조** `7c15866` — `--verify-resources`가 pgl-gpu의 파일에서 md5를 재계산해
      불일치·읽기 실패 둘 다 거부. 음성 대조 3종으로 증명
- [x] ✅ **`longread_fetch.sh` 근본 수정 3건 (2026-09-03)** — ①**재개형 전환**: 원본을 먼저 받고
      (`curl -C - --http1.1`) 무결성·ENA md5 확인 후 변환, 실패해도 원본 보존. kill/restart로 검증
      (8,925,184 → 11,480,383 B 이어받음). ②**ENA `fastq_md5` 대조** — #6이 지적한 "기록만 하고
      비교 안 함"이 이제 실제로 비교된다(완료 4런 전부 일치). ③**완료 줄이 FAILED를 센다** + 실패 시
      exit 5, `<run>.FAILED` 표식. 파이프 서브셸이라 카운터가 안 살아남던 것이 원인이었고, 그래서
      6런 전패 데이터셋이 `0 runs DONE, 0 skipped`로 exit 0 했다
      ✅ **저장소로 편입됨**(`7da946c`) — 아래 항목 참조
- [x] ✅ **증거 드라이버 11개 저장소 편입 (저자 결정 2026-09-03)** `7da946c` —
      `transgenic/revision/scripts/evidence/`가 정본, `evidence/<이름>`은 **심볼릭 링크**라
      기존 호출 경로와 실행 중인 잡이 그대로 동작한다(수집 진행 중에 무중단 전환·검증 완료).
      `quarantine.md` §1b가 지적한 "조항은 안에, 집행 코드는 밖에"가 해소됐다.
      README에 어느 스크립트가 어느 조항을 집행하는지와 오늘 겪은 함정 2건(도구가 쓴 파일은 0644라
      실행 비트를 잃는다 · `pkill -f`는 자기 셸을 죽인다)을 적었다
- [x] ✅ **장독 범위 v2 동결 (2026-09-04)** `1653d6d` — `DATASET_ROLES.longread_v1.tsv` **214행**
      (적격 129·검증전용 31·**제외 54**, 감사 가능하도록 제외분 포함), md5 `d644278f`, 프로토콜 §1 등재.
      `--scope longread` 신설(테스트 3건 선행). `est_v1` md5 `a6375713` 불변 확인
- [x] ✅ **고아 드라이버 종료 (2026-09-04)** — 로그인 노드에서 **2일 4시간** 살아 O. sativa에서 17 h 51 m
      붙잡힌 채 **#63 금지 ONT WGS 15.5 GB** 수집 중이던 `training_run_all.sh` 트리 종료·조각 삭제.
      `quarantine.md`의 "S. italica에서 멈췄다"가 사실이 아님이 드러나 ✏️ 정정
- [x] ✅ **A36 커밋·푸시** `bf207ee` (프로토콜 v1.26 + `61_univec_trim.py` + `univec_screen.sh` + 테스트)
- [x] ✅ **EST 정렬 9종 완료** — `6147674_[0-8]` 전부 COMPLETED, 결과 `resume.md` 📊 절
- [x] ✅ **Athaliana 73.5 % 매핑률 원인 규명** — 454 시대 짧은 라이브러리 2개에 의한 구성 효과로 확정,
      벡터·비Col-0 가설 기각 (`quarantine.md` §1d)
- [x] ✅ **저자 결정 반영: ≤120 nt EST 제외 → 프로토콜 A36 (v1.26)** — A8/A21 하한 100 → 121 nt.
      `61_univec_trim.py` `DEFAULT_MIN_LEN=121` · `univec_screen.sh --min-len 121` · 기본값을 고정하는
      테스트 추가(7건 통과). Athaliana 773,346건 제외 / 756,354건 잔존, Sanger 부수 손실 1.82 %
