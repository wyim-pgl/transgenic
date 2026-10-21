# Task 1 — B6 fresh compiler-history proof

Read this first: it is the task requirements. Parent plan: `PLAN.md` in this directory. Own only new files under `notes/b6_pilot_resume_20261020/` in implementation worktree `/tmp/b6-pilot-proof-20261020`, branch `b6-pilot-proof-20261020`. Base19049ba. Trainer source is separately `/tmp/b6-c2-recompile` at198ce14 and must remain unchanged for this task.

## Deliverables

- `b6_history_proof.py`: separate B6 fresh-run, compare and receipt helper.
- `b6_history_proof.sbatch`: four-rank allocation wrapper, fresh verify, continuous/snapshot/new-process resumed phases and acceptance publication.
- `test_b6_history_proof.py`: meaningful rejection and real small CPU fixture tests for the acceptance/provenance/fresh-start differences.
- `TASK1_REPORT.md`: changed files, commits, exact commands/results, limitations and any unresolved requirements.

You may reuse unchanged C2 modules as bundled dependencies: `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/{history_proof.py,probe.py}` and their required dependencies. Make dependency discovery explicit for local tests and immutable remote staging; never rely on an arbitrary current working directory. Do not modify existing C2 tools or source. Minimize duplication without weakening B6-specific checks.

## Frozen execution

- Source commit198ce14ce53266f53b7bda5d0c4cdfd94deb43b4; runtime NGC26.08; armB6; world_size4; seed123; aug_seed20261037; bf16; deterministic; checkpointingnonreentrant.
- Keep `--stop-after-updates4099` and original full-run LR schedule/optimizer/config. Bound only diagnostic execution using the existing `TrainSpec.stop_at_step` technique.
- Start continuous with NO checkpoint, no C2 bootstrap, and initial observed compile history. Stop at410, snapshot the complete400 checkpoint without interrupting the continuous process. Clone only via reviewed probe semantics. Start a genuinely new process from snapshot400 and stop at410.
- Root layout mirrors accepted proof: `continuous`, `at400`, `resumed`, explicit logs/verified record/comparison/receipt. Root supplied from unique Slurm job ID. Existing destination must refuse ambiguous reuse.
- Stable B6 cache namespace: `/work/compiler_cache/b6_history_v1/inductor` and `/work/compiler_cache/b6_history_v1/triton`. Keep FX/AOT enabled and image TF32/CUTLASS/TRITON environment unchanged; no force-disable or Dynamo reset shortcut.
- Verify all inputs freshly inside this allocation. Bind actual job/source/helper/config/bindings/image/init-weight identity. Existing frozen mmap path must be used. Do not hash model/inputs on login.
- Require project22169/ddn_ssd100GiB headroom via the existing source storage preflight; separately estimate/check proof-copy peak. Default requested regularghx4 wall time6hours is a conservative allowance for fresh0→410 and replay; parent may refine from actual historical timings before submission, not bypass timeout failure.

## Scientific acceptance

- Require meta positions: at400 epoch0/step9600/global400/pending0; both410 epoch0/step9840/global410/pending0. Identity armB6/target4099/frozen seed/world/recipe. Validate embedded CompileHistory through source public loader.
- `bootstrap` is absent/null and every history entry is `observed`; reject reconstructed legacy/C2 history, empty history and cross-run identity/fingerprint mismatch.
- Actual training must observe nonzero evidence tracks and valid chain elements (not only all-zero dropout), all three DNA tiers, and real compiler events. Capture verifiable bounded summaries tied to trace/input hashes; don't infer this solely from arm='B6'. A C2-only expected slot/loss is NOT a B6 requirement.
- Compare complete canonical model/optimizer/scheduler/all four rank RNG states and exact per-micro input/shape/context/loss/RNG traces, accumulation-boundary gradients/RNG, and final compiler history. Require all expected rows and state keys and finite losses. No tolerances, no model-key omission, no missing-rank acceptance.
- Preserve trace separation between priming replay and real training; replay must not consume real step/micro counters. Check warm/snapshot/new-process boundary through the existing reviewed tracing approach.
- No historical C2 step210loss or legacy slots1..15+3915 requirement. Those are C2-only witnesses.
- Emit comparison with explicit kind/profile/arm, source/job pins, observed-evidence summary and detailed mismatch lists. Emit version1 `kind=b6_history_validation_ready` only after complete acceptance and only when comparison bytes, candidate metadata/history/identity/full-state digests agree with current allocated files.
- Ready receipt fields: version, kind, source_commit, validation_job_id, comparison_sha256, and candidate{relative_run:'resumed',relative_checkpoint:'latest_state',epoch,step,global_step,pending_micro,meta_sha256,identity_sha256,history_sha256,canonical_checkpoint_sha256}; include verification/runtime provenance needed for later independent acceptance. Do not invent a C2 predecessor or original checkpoint receipt for a fresh B6 run.
- Job/bash failures propagate nonzero; no READY/DONE on partial work. This proof must not create canonical pilot4099 DONE or replace/activate any existing run.

## Required test cases

1. Fresh run without latest_state is valid only on explicit B6 fresh continuous path; resumed missing state refuses.
2. C2/reconstructed bootstrap or wrong arm/target/world/cache identity refuses.
3. Model mismatch, optimizer/RNG mismatch, history mismatch, missing trace rank/rows, nonfinite loss and missing nonzero evidence/chain/tier observations cannot emit READY.
4. Receipt cannot pass after comparison/candidate/history bytes change or declared hashes are recomputed around a contradictory comparison.
5. A complete small fixture publishes a self-consistent B6 receipt, without C2 legacy/historical-loss assumptions.
6. Snapshot/resume metadata/counters and test expectations use4099 schedule with diagnostic400/410 bounds. CLI/batch syntax and source imports match host/NGC boundaries.

Use real small CPU artifacts for digest/receipt tests; mock only genuinely unavailable GPU/Slurm boundaries. Verify RED before implementation for new acceptance behavior. Use local `/home/wyim/miniconda3/envs/pdb/bin/python` with existing dependency overlay if needed. Parent handles remote deployment and all mutations. Do not submit/cancel/release jobs or create subagents. Commit your scoped changes with author Won Yim <wyim@unr.edu>; do not merge or push unless parent directs. Report DONE/DONE_WITH_CONCERNS/NEEDS_CONTEXT/BLOCKED plus commits and test results.
