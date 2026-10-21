# Task 4 — automatic B6 continuation after accepted proof

Read this first: it is your requirements. Work in `/tmp/b6-pilot-proof-20261020` on its existing implementation branch after Task1 is committed and reviewed. Preserve source198ce14 and every completed C2 helper/operation. Parent owns all remote mutations and submissions. Do not spawn agents or reviewers.

## Goal and structure

Build an explicit, narrow B6 controller plus an isolated B6 production helper to complete the already authorized WP5 pilot automatically after its fresh-state GPU proof passes. Do not add a generic arm switch to C2 tools, subclass the C2 controller, fabricate a legacy200 checkpoint or monkey-patch imported C2 globals.

Create under `notes/b6_pilot_resume_20261020/`:

- `continue_b6_history.py`: hostPython3.6/stdlib CLI `init`, `status`, `tick`, `reconcile`, `watch`.
- `b6_production_link.py` and `b6_production_link.sbatch`: strict B6 allocation preflight, ordinary training and postflight.
- `test_continue_b6_history.py`, `test_b6_production_link.py`, `test_b6_link_contract.py`.
- `TASK4_REPORT.md`: exact implementation/test/commit/limitations report.

Reuse unchanged C2 infrastructure through explicit manifested dependencies: `Slurm`, submission token/argv/WorkDir normalization, safe JSON/path/pin/source-manifest checks, `Lock`, atomic/exclusive file publication and `terminal_ok`. Existing controller file is `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/continue_c2_history.py`. A separately named B6 production-helper copy with a small reviewed difference is acceptable: preserve the substantial existing model/optimizer/scheduler/RNG/pending-grad/history/final validations and signal supervision in `production_link.py/.sbatch`; change strict B6 profile/arm/kind/paths explicitly. No unrelated refactor.

Task1 supplies `b6_receipt_contract.py`, purePython3.6 predicates `validate_report(report, expected_source, expected_job)` and `validate_ready(receipt, report, expected)`. Read the actual committed API/schema and tests. Publisher and host acceptance must use these shared predicates. A self-consistent rehashed false comparison must still be rejected. GPU tensors/current full-state hashes remain allocated work; host verifies exact small artifact and source/helper pins.

## Frozen values and state

- armB6, seed123, aug_seed20261037, world4, bf16, deterministic, nonreentrant; target4099 and unchanged full-run LR schedule.
- Source198ce14ce53266f53b7bda5d0c4cdfd94deb43b4. Proof and production stable cache paths `/work/compiler_cache/b6_history_v1/{inductor,triton}`; FX/AOT stay enabled and NGC26.08 image environment is preserved.
- Validation candidate is exactly accepted proof root/job/resumed/latest_state atglobal410,epoch0,step9840,pending0. Fresh history has null bootstrap and only observed events. Preserve continuous410 and at400; candidate evolves only through accepted production links.
- New separate operation path/ID and logical B6 claim/lock. Never restart C2 controllers, alter C2 journals, move checkpoint trees or create C2 aliases. Actual Lustre does not support RENAME_NOREPLACE; use in-place activation and exclusively published small receipt.
- Ownerwcyim/UID97434. Obsolete heldB6 jobs3345918 then3345917. Both currently reference source233574a and exact oldwp5_pilot.sbatch; parent provides full live/accounting identity pins. Do not accept basename-only script identity or owner-only cancellation.
- Project22169/ddn_ssd minimum100GiB at activation, every submission and allocated launch. Proof-copy peak is independently checked by proof helper, never added to100GiB as a blanket requirement. Budget/read-only preflight waits are reported honestly, no false scientific PASS.
- Production resources supplied as explicit reviewed config. Allow a bounded regularghx4 walltime up to actual48h limit; parent will choose from measured B6 throughput. Clean positive-progress preemption may link again; failure/timeout/OOM/no receipt/no progress blocks without fallback.

## Controller interfaces and phases

`init --config <json> --state-dir <new path>` adopts an already submitted proof job; it must not submit proof itself. Parent provides proof submission argv/token/owner/time/source/helper/output pins. New directory/claim must be exclusive; existing operation is inspected/reconciled, never silently reset. `status` is read-only. `watch --poll-seconds45` persists progress, retries transient observation waits and exits on complete or explicit scientific/ownership ambiguity blocks. No force/reset/retry/delete command.

Phases:

```
waiting_validation -> validation_accepted
-> retire_intent/waiting -> both_old_jobs_retired
-> activation_intent -> candidate_activated
-> production_submit_intent -> waiting_production
-> link_accepted -> next production submission
-> final_verified -> complete
```

`submission_uncertain` is reconciliation-only. Only the same process invocation that durably wrote a new submission intent may execute its sbatch. A recovered intent never automatically issues another sbatch. Reconcile adopts exactly one fully matching owner/UID/token/normalized fullargv/time job. Zero or multiple matches remains unresolved; loss of SSH output is not permission to retry.

Proof acceptance must verify all of the following before cancellation/activation: successful parent/batch accounting, allowed extern-only exception, exact owned Slurm submission, immutable source/helper/config/image/input provenance, B6 strict comparison, READY digest and current small checkpoint metadata/identity/history/verification bindings. Do not importTorch on host or rehash large model/input files there. Host cannot accept merely passed=true or Slurm0.

Retire obsolete jobs sequentially, follower3345918 first. Each exact cancellation has a durable per-job intent and must observe terminalCANCELLED before the next. Cancel only the pinned same ownedPENDING/JobHeldUser allocation; running/released/requeued/foreign/mismatched jobs refuse. Lost response waits for accounting, never blind retry. Activation requires both retired.

Activation receipt embeds initial410 meta/identity/history bytes and digests, accepted comparison/READY/source/provenance, derived active_run and both retirement observations. Publish it exclusively with fsync under the logicalB6claim. Device/inode observations are audit-only across hosts. Never require live latest_state still equal410 after production has advanced; validate historical410 via embedded receipt and evolving state via accepted predecessor receipt.

Each production request is immutable and binds operation/link/job-token, source/helper/source-manifest/config/inputs/image/runtime, active_run, exact startcheckpoint position/digests, predecessor receipt, target4099 and frozen profile. Link1 predecessor is the B6 READY digest. Later links cite the prior accepted production receipt. Preserve strict same-run/epoch/pending-gradient validation from reviewed producer.

## Allocation and final acceptance

- Fresh input verification on all four ranks in every allocation, using fixed bindings and existing mmap. No bypass on warm data or previous receipt.
- Exact B6 profile/source/manifests/cache fingerprint/identity/startstate before training. Require null bootstrap and observed-only history; postflight history must extend the accepted prefix without rewriting it.
- Ordinary trainer.main; output is accepted active_run. B6 source warning/fallback/failedmarker and nonzero launcher statuses remain failures even if DONE exists.
- Preserve reviewed signal behavior: USR1 translates to SIGTERM for owned training descendants, await actual launcher completion; do not treat a killed leader as successful rank shutdown.
- Verify every completed epoch's committed metadata/evaluation, de_novo and evidence observations, data order and pending gradients. A410 proof does not establish epoch/final behavior.
- Final exactly4099, normal trainer/launcher/rank exits, complete runtime/input receipts, fixed-step model identity/hash and aliases, PILOT_FINAL, PRUNE_EXEMPT and TRAINING_DONE. Periodic latest_state4000 is valid alongside final4099.
- Finish atcomplete without C2's Nice-restoration/followup tail. WP5 pilot-panel evaluation remains the next existing task.

## Required verification

Use RED/GREEN meaningful small real file/checkpoint tests; mock Slurm/GPU boundaries only. Cover:

1. Actual Task1 READY/comparison fixtures consumed by host, activation and producer request; forged/rehashed false proof, wrongarm/cache/source/helper/script and missing rank rejected.
2. Both held-job retirements, partial retirement after crash, lost cancellation response, running/foreign/mismatched job refusal.
3. Durable submission crash boundaries/concurrent watchers and uncertain response; no duplicate sbatch when zero/multiple candidate matches.
4. Complete automated410 -> positive-progress preemption ->4099 chain through actual request/producer/controller schemas.
5. Mutated active candidate before first link refused; legitimate later mutation accepted via current predecessor, immutable proof siblings and embedded410 remain bound.
6. Below-reserve wait/recovery, full source member larger16MiB still streamed while controlJSON cap remains16MiB.
7. Compiled `_orig_mod.` key compatibility, tied aliases, strict4-device CUDA RNG encoding, epoch/pending-gradient resumes, nonzero exit or missing rank despiteDONE refusal.
8. HostPython3.6 pure import and allocation script syntax; focused source-state regressions only where changed behavior warrants them.

Report exact commands/counts and limitations. Independent review follows your commit; do not deploy, merge, push or mark GPU proof successful from local tests. Raise concrete interface uncertainty promptly to parent.
