# B6 pilot / Delta CPU progress — 2026-10-20

Latest handoff: [.atl/resume.md](../../.atl/resume.md), [TODO](../../.atl/todo.md). All times here use PDT; raw Slurm timestamps in captured JSON use the remote cluster timezone and are not relabeled as PDT.

## Current status — 2026-10-21 11:33 PDT

Live observation at2026-10-21T18:33:57Z confirms proof **3425703 COMPLETED0:0** for parent/batch/extern after3h22m26s. The actual B6 comparison passed: model, optimizer, scheduler, all four rank RNG states, and compiler history agree between continuous and resumed runs. READY identifies global_step410/epoch0/pending_micro0. This is B6 GPU evidence, not an inference from C2.

The persistent controller retired old held jobs3345918 then3345917, activated `runs/b6_compile_history_validation/3425703/resumed` in place, and submitted production **3429921** (ghx4/4GPU/64CPU/48h/no-requeue, target4099). It is **PENDING/Priority, elapsed0**, so production is not training yet. PID1501642 remains alive; journal waiting_production/revision1291/last_error=null, exactly one production link. Keep this job and watcher; do not resubmit proof or production. The next required observations are allocated launch gates, actual training progress, epoch evaluation, final4099/DONE, and normal Slurm completion. [Live receipt and selected activation evidence](LIVE_STATUS_20261021.json).

Later10-21 current-model priority and ccloop review are recorded in [current-model progress](../ccloop_current_model_20261021/PROGRESS.md). At18:58:43UTC the same production remained pending, controllerrevision1321/errornull, NVMe116.54GiB and GPUbalance1624hours. Delta CPU activation and actual0GPU metadata/scorer jobs23104942/23105376 both passed. The broader four-species benchmark goal is paused; completing current training and its frozen WP5 checks is the active scope.

The sections below preserve the10-20 history; later live observations above supersede earlier pending statuses.

## 2026-10-20 staging history

- Source198ce14 remains unchanged. Implementationbranch b6-pilot-proof-20261020 at50a7fac, no mainmerge.
- Task1proof e2bee04: independentreviewPASS after equal-empty-state falseREADY fix; preparedschema8files,AdamW/scheduler/four-deviceRNG invariants. Actualhost3.6+NGC6contract/22proofPASS. [staging](PROOF_STAGING.json),[nativechecks](PROOF_CPU_CHECKS.json),[review](TASK1_REVIEW.md).
- Archive3425279: allocated1GPU(requiredsite minimum)/1CPU/2GiB/2h, CPUfileI/Oonly. epoch09manifestcf528525f4922ffd75349330f8a2d1afd965b62c26e8dc4cebc65940e25d3d0c verifiedHDD22169/ddn_hdd andsourcecleanup;14,123,523,405B.10/11pendingcompletion. protected12–14/latest/bestunchanged. [actualgate](archive/LIVE_GATE.json).
- Storagegate: actual115,452,964,864B(107.5GiB)free; mandatory100GiB and independent42,241,173,949B proof-copy peakpass. Futurearchivebytesarenotcounted. Gatepassesallowproofbeforeall3archiveepochsfinish;allocation/activationwillrecheck.
- Proof **3425703** submitted2026-10-21T00:54:33Z=10-20 17:54:33PDT;ghx4/4GPU/64CPU/6h/no-requeue. Prioritypending. FreshB6evidence-on0→410, snapshot400,newprocess400→410; actualGPUproofstilloutstanding. Immutableintent/result `ops/b6_proof_submit_e2bee04_20261020`;[receipt](PROOF_SUBMISSION.json). Old3345918/17remainheld.
- Task4controller/producer: initial8b13234, cancellationracefix2a48265/report50a7fac; independentreviewPASS. Explicitscancel--ctld state/userfilters, noatomicJobHeldUser-reasonguarantee. Actualhost3.6 andNGC36/36PASS. [review](TASK4_REVIEW.md),[staging](PRODUCTION_STAGING.json),[nativechecks](PRODUCTION_CPU_CHECKS.json).
- Concreteconfig bindsactual3425703,source/manifests,frozenrecipe,expectedSIFsidecar andactualheldidentities18then17. ExpectedruntimeisacceptedC2fingerprintwithonly4B6cachepathchanges;actualB6mustmatch,beforeacceptance. ExpectedpinisnotobservedB6proof. [config](CONTROLLER_CONFIG.json),[derivation](CONTROLLER_CONFIG_DERIVATION.json),[nativecheck](CONTROLLER_CONFIG_NATIVE_CHECK.json). FinalindependentconfigreviewPASS;init/watchPID1501642,45spoll,waiting_validation/revision1/errornull. [startreceipt](CONTROLLER_START.json). InputconfigSHA d36ec064…; controllercopies equivalent parsedJSON into itscanonicalformat, stateconfigSHA afafa877… isrecordedbyjournal.
- DeltaCPU actualdt-login03/x86_64 UID97434;NVMe/HDDbidirectionalR/W+tinyfixturePASS. `accounts`exit60/noSlurmassociation;1000core-hourcreditsactivationpending,noCPUjobsubmitted. [guide](../delta_cpu_20261020/USAGE.md).

Remaining: existingpersistentwatch waitsfor proofscientificPASS/normalexit→exactoldjobretirement→in-place410continuation→4099epoch/final/DONE. Archive10/11mustbeobservedandinterruptedtransactionsinspectedbeforeanyrecovery. WP5panelG1/G2follows;WP6remainsseparate.

## Archive completed

At 2026-10-21T01:34:32Z (10-20 18:34:32 PDT), job3425279 finished after1h59m31s with parent/batch/extern COMPLETED0:0. All three epochs09–11 were durably verified on HDD and removed from NVMe:42,370,570,227bytes (39.46GiB). Protected12–14/latest/best remain unchanged. Parent acceptance and actual NVMe headroom are in [archive/PARENT_ACCEPTANCE.json](archive/PARENT_ACCEPTANCE.json). No recovery job was needed or submitted.

## Latest live state and graph

At2026-10-21T02:42:15Z, B6 proof3425703 remains Priority pending. Controller1501642 is alive, waiting_validation/revision112/errornull; no production link submitted. Old3345918/3345917 remain held. [Live receipt](FINAL_LIVE_STATUS.json). DeltaCPU accounts still returned60/no association at02:52:13Z.

Full detected-repository graph completed:2,696files,18,577nodes,38,057relationrecords,1,874communities. Browser and CLI checks passed; [graph report](../../graphify-out/GRAPH_REPORT.md) and [verification](../../graphify-out/VERIFICATION.json) record raw parser/reference limits and unknown host token usage. Final graph outputs are a captured source snapshot; this completion paragraph is later operational documentation.
