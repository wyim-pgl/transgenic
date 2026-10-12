# Resource probe FIX1 — RP1–RP4

Branch `mvp-resource-fix1-20260916`, based on `4bd97399`. Narrow resource fixes
only; owner commits `981489d` / `d99b1f8` were not cherry-picked or duplicated.
**Integration/execution remains HOLD.** No SSH, jobs, CUDA execution, subagents,
merge, push, main-checkout edits, resource PASS or training authorization.

## Changes

- **RP1:** `resource_probe.run.progress` now publishes immutable
  `rankN.000000.profiling.json`, `rankN.000001.profiling.json`, etc. Sequence is
  monotonically increasing per rank after successful collective publication.
  Every earlier snapshot is retained. Consumers should enumerate
  `rankN.*.profiling.json` and select the greatest numeric sequence for that rank,
  not expect an overwritten `rankN.profiling.json`. The existing report was
  updated; graft found no other indexed filename consumer. Snapshots remain
  `INTERMEDIATE_PROFILING_NOT_READINESS`, never receipt/DONE. A fresh-path
  `SystemExit` is converted to `RuntimeError` only inside this progress publisher
  so the existing collective preflight exchanges it. The global immutable
  `_atomic_write` is unchanged. Collision never overwrites the existing file.
- **RP2:** extra source file `src/transgenic/training/mvp_metrics.py` factors the
  existing local input checks into public `validate_paired_batch(batch)`, returning
  `(rn, has_evidence)`. `paired_metric_record` calls the same helper; its off/on
  forwards, CE math, result fields and mode restoration are unchanged. Resource
  retrieval/device transfer calls the helper inside collective preflight before
  any rank enters the first forward. The forward-containing function remains
  OUTSIDE the local-only collective guard. No finalizer orchestration or shared
  `one_update` changes are included.
- **RP3:** exact `type(value) is int` checks now reject bool/float/string schema,
  envelope, rank, update counts and RNs. Also hardened maxlength, input lengths,
  inventory maximum label length and generated-kernel counts. Existing strict
  seed, label length, valid count, capacity, EST count and memory checks are
  covered by negative variants. Projection math is unchanged.
- **RP4:** removed the extra EOF blank in `tests/test_mvp_resource_probe.py`.
  Corrected the previous report's blanket whitespace claim rather than treating
  the independently reproduced original failure as a passing historical check.

## CPU verification

New `tests/test_mvp_resource_fix1.py` adds 60 cases:

- 57 bool/float/string rejection cases across 19 integer fields.
- Two tests execute the actual AST-extracted nested progress function, preserving
  its closure and sequence initialization: two successful publications followed
  by collision or cap failure through the real single-rank collective preflight.
  They check retained bytes, updated stage contents, failure propagation and no
  DONE marker. This is not a simulated successful CUDA `run`.
- A bounded two-process Gloo CPU test calls actual `paired_validation` with a DDP
  model and rank1-only float RN metadata. Both processes exit normally with the
  same ValueError, zero forward calls and no PASS/DONE. Only CUDA synchronize is
  replaced by a CPU no-op; status transport uses real Gloo tensor collectives.
  Process-group timeout is 20 seconds, parent deadline 60 seconds.

Final command (worktree-local `.testdeps` symlink to read-only-use shared
`/tmp/b6-p2-readiness-1535112/.testdeps`; no dependency installation):

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
  tests/test_mvp_resource_probe.py tests/test_mvp_real_probes.py \
  tests/test_mvp_readiness_tools.py tests/test_mvp_readiness_review.py \
  tests/test_mvp_metrics.py tests/test_mvp_resource_fix1.py
```

**186 passed in 48.25s** (the previous 126 plus 60 new cases; no subtests plugin).
The metric regression suite is included. An earlier focused run had 74 passes.
An initial full run using the incomplete runtime-controller dependency directory
had 156 passes / 30 failures, all missing duckdb distribution metadata; switching
only the local dependency symlink to the complete shared directory resolved it.
No fixtures or validators were weakened for those failures.

`git diff --check 3b62306` passed against the FIX1 working tree, covering the
original resource diff plus these fixes, including the removed EOF blank.
This supersedes the incorrect claim for `3b62306..4bd97399`. `git diff --cached --check` and `git diff --cached --check 3b62306`
also passed, as did `py_compile` for both changed source and both test files.
`graft build` refreshed the local graph; generated cache/ignore files are not
part of this commit. Verify the delivered commit with
`git diff --check 3b62306..HEAD`. CPU results are not pinned
SIF/CUDA/NCCL certification or a performance/resource guarantee. Advisor planning
consultation failed with `claude_code_version_too_old`; no advisor review is
claimed.

## Later integration contracts — do not execute this base

Current resource call sites intentionally still use this base's
`p.datasets(args, bindings)` and `p.one_update(accelerator, model, optimizer,
batches)`. The v3/8192 fail-closed check remains. The owner's known v3 factory and
coordinated update fixes are separately reviewed but absent from this branch.
At later integration:

1. Update resource dataset construction to
   `p.datasets(args, bindings, cfg, model_config)` using the loaded model config
   before wrapping/compilation. Do not change only the shared signature and leave
   this two-argument caller. Validate both resource train and separately built
   valid datasets with the owner's `validate_decoder_dataset` contract.
2. Shared `compiled_update(args, bindings, model, ds, accelerator)` receives the
   main-owned Accelerator. Resource dispatch currently occurs before the normal
   main setup and owns its own Accelerator in `run`; reconcile the combined
   early-dispatch/ownership/cleanup paths so exactly one owner exists.
3. Preserve `one_update(accelerator, model, optimizer, batches)` and integrate the
   owner's coordinated implementation. Reconcile `collective_batches` and
   `collective_compiler_check` with the resource preloading/compiler paths without
   nesting forward-containing or collective operations in local-only preflight.
4. Re-review combined callers, regenerate consumed-code/tooling bindings (including
   the changed metric source), and rerun CPU plus bounded real CUDA diagnostics
   only after controller authorization. This commit does not remove that HOLD.

The prior five-node-hour qualification also remains: the child timeout/projection
is not an allocation-wide deadline. Controller accounting must include earlier
smoke, startup, parent validation and teardown, prohibit unaccounted warmups, and
impose an outer job/allocation deadline within the remaining cap.
