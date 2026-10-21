# B6 pilot continuation implementation plan

> For agentic workers: use subagent-driven-development task by task. Existing user instructions require continuous execution without repeating authorization for this unchanged pilot.

**Goal:** finish the authorized B6 WP5 4099-update pilot with the repaired compiler-history implementation and an actual evidence-on resume proof.

**Architecture:** preserve completed C2 artifacts and held obsolete B6 jobs. Create a separate B6 proof helper/release using source198ce14. Fresh B6 continuous0→410, immutable snapshot400 and a new-process400→410 must match exactly. Only accepted B6 evidence may authorize continuation; no C2 bootstrap or historical-loss requirement is transplanted. Reuse the reviewed checkpoint archival tool to recover training storage without deleting unverified or protected data.

**Tech stack:** source198ce14, NGC26.08/Python3.12, four production ranks, host orchestration Python3.6 stdlib, Slurm, existing proof and archive tools.

**Spec:** frozen A46/A51 and `notes/a46_approval/A46_RULE11_NPILOT_AMENDMENT_20260927.md`; existing C2 exact-state acceptance; current `.atl/resume.md` and `.atl/todo.md`. Existing WP5 approval10-03 and continuous TODO instruction authorize this work; scientific and resource gates remain binding.

## Global constraints

- seed123; aug_seed20261037; armB6; four GPUs for the production-context proof; bf16; deterministic; nonreentrant checkpointing; target4099 and original full-run LR schedule.
- New proof source is `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`, not the old held233574a script. Trainer worktree stays unchanged unless a separately reproduced defect requires repair.
- Preserve original C2 checkpoint200, diagnostic256, all accepted C2 evidence/final4099, B5 latest/best/latest two completed epochs, and the786GB input DB.
- Existing100GiB project22169/ddn_ssd headroom is strict at submission/activation; proof-copy peak independently fits. Do not require100GiB plus every future byte.
- No model/input hashes on login; heavy proof and transfer hashing in allocations. No compiler-cache traversal.
- No uncertain submission retry, no blind held-job release, no source or input patching inside an existing allocation. New helper releases are immutable and manifested.
- Main holds operations records. New implementation branch `b6-pilot-proof-20261020` in `/tmp/b6-pilot-proof-20261020`; do not merge it into main.

## Current evidence, 2026-10-20

- C2 production3379630 finished4099, controller complete/revision22.
- Replacement B6 training3345917/18 remain JobHeldUser, elapsed0, oldsource233574a. Canonical B6 run absent.3345916 was successful input verification, not training. Original failed3306594 remains historical evidence.
- Eight B6 CPU/aot_eager history lifecycle tests passed141.281s on198ce14; worktree clean. Actual B6 GPU resume equality remains unproved. Smoke3338555 reached247 on oldsource, no restart proof.
- NVMe remaining98,954,840KiB =94.37GiB. HDD remaining178,902,640KiB =170.62GiB. Metadata-only archive tool identifies completed B5epoch09,14,123,523,405bytes; best/latest protection13/14. No transfer has started.

## Task 1 — B6 fresh-state proof helper

Read `TASK1_BRIEF.md` for exact interfaces, constraints and tests.

- [x] Implement separate B6 helper and allocation script with fail-closed acceptance.
- [x] Run focused tests and source/CLI compatibility checks.
- [x] Independent spec and quality review; address material findings and reverify.
- [x] Commit on implementation branch and pin the reviewed release.

## Task 2 — restore execution headroom using existing archive tool

- [x] Metadata-only plan B5 completed older epochs09–11 using reviewed3b4eb0d; preserve12–14, latest, best and all C2 artifacts.
- [x] Pin exact source helper hashes, destination HDD pool/project and reserve50GiB, per-pass16GiB/one checkpoint, rate16MiB/s. Use an allocation for heavy hashing/copying. Actual DeltaAI policy rejects zeroGPU jobs; request its minimum1GPU/1CPU/2GiB/2hours, using onlyCPUfileI/O. Retain durable manifests and verified destination before source cleanup.
- [x] Independently review concrete batch/config before launch; record durable submission intent and actual job ID; no uncertain retry.
- [x] Confirm verified transfer/cleanup receipts and current NVMe reserve before B6 submission. The gate is verified reclaimed headroom, not completion of every planned archival epoch. A timeout/failure never authorizes counting unverified or future bytes; remeasure both gates and preserve interrupted transactions.

## Task 3 — deploy and execute proof

- [x] Stage immutable manifested helper and fresh operation root; verify actual source/config/bindings/image/cache environment.
- [x] Confirm no active B6 training and exact held obsolete job identities; keep them held while proof runs.
- [x] Submit bounded four-GPU proof with fresh allocation input verification and enough wall time for0→410 plus replay. Persist submit intent/result and inspect actual Slurm state.
- [ ] Collect explicit comparison/receipt/test/trace/metadata files, check every scientific condition and parent/batch exit. No completion claim from Slurm alone.

## Task 4 — validated continuation

- [x] Bind the accepted B6 candidate, immutable helper/source/input identity and per-link receipt chain in a separately reviewed continuation wrapper. C2-only production_link/controller cannot be reused unchanged.
- [ ] After acceptance retire exact obsolete held B6 jobs and continue validated B6 toward4099, preserving originals and evidence. Never relocate/replace checkpoint trees on unsupported Lustre rename semantics.
- [ ] Require epoch de_novo/evidence evaluations, exact4099 final artifact, normal exits, DONE and PILOT_FINAL. Frozen WP5 pilot-panel G1/G2 follows training; WP6 remains separate.

## Decisions

- Existing frozen recipe and approvals are unchanged. B6 fresh0→410/snapshot400/resume410 is an implementation choice for validating the known compiler-history defect, not a newly invented protocol threshold.
- The previous C2 controller deliberately restored Nice without releasing old B6 jobs. Root did not follow that with the remaining B6 work; this, not queue scheduling, explains the replacement jobs' lack of progress.


## 2026-10-20 execution ruling

See [PROGRESS.md](PROGRESS.md) and receipts. Verifiedepoch09 HDDcopy+cleanup madeNVMe107.5GiB.100GiB and independent42.24GB proof-copy peak bothpass; Task3 may proceed while10/11archive continues. This changes the implementation task ordering only; no resource/scientific gate is waived. Proof3425703 submitted17:54PDT, pendingPriority. Task1e2bee04 reviewed/native6+22PASS. Task4code2a48265/report50a7fac reviewed/native36PASS; concretecontrollerconfig independentreviewPASS/nativecheckPASS;init/watchPID1501642 waitsforproof3425703.

Final archive acceptance:3425279 completed all09–11 at18:34:32PDT,39.46GiB; native free NVMe133.83GiB. Parent acceptance is archive/PARENT_ACCEPTANCE.json. Proof3425703 remains pending; controller1501642 is waiting_validation/revision112/errornull at19:42PDT.
