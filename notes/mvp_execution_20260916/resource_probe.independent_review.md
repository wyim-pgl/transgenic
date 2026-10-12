# Independent review — resource-budget probe

## Scope and verdict

Reviewed detached `4bd97399` against `3b62306` in isolated native worktree
`/tmp/pi-agent-9692a8e0-0589-4e3-18137c0f`. Consulted graft first; read the complete
neutral diff, committed implementation report, and neutral original
`real_probes.independent_review.md`, then traced the relevant shared helpers.

**Spec/correctness: CHANGES REQUIRED. Integration/execution: HOLD.** There is a
new deterministic progress-publication blocker, independent of the acknowledged
base label/helper fixes. Do not spend GPU allocation on this revision as-is.
No resource PASS, six-gate readiness, training launch, or performance assurance
is established by this review.

No source edits, SSH, jobs, GPU execution, subagents, merge or push. Only this
report is committed. Focused local CPU reproductions below are not CUDA receipts.
Advisor consultation failed with `claude_code_version_too_old`; no advisor
assessment was received.

## Findings

### RP1 — HIGH: second progress publication always fails

**Locations:** `revision/scripts/mvp/resource_probe.py:278-285,299,334,356`;
`revision/scripts/mvp/make_launch_readiness.py:113-138`;
`src/transgenic/training/mvp_finalize.py:409-417`.

`progress()` always writes `rank{rank}.profiling.json` using `_atomic_write`, whose
explicit contract is *fresh path only*. Setup publishes that path first. After
inventory completes, `stage()` calls progress again and `_atomic_write` raises
`SystemExit: FAIL: output path must be fresh`. Thus, after integrating base R1,
a healthy run still stops before its first resource update. Later progress calls
would have the same problem. The retained file contains setup, not the newly
completed inventory or later cold/steady timings claimed in the report.

`_collective_preflight` catches `Exception`, not `SystemExit`, so this failure
also bypasses the intended local-error status exchange. All healthy ranks should
hit the collision, but uneven failures can leave different collective paths.
This is an execution blocker, not a false-PASS route; the launcher must reject it.

**Independent CPU reproduction:** called the actual `_atomic_write` twice on
one fresh temporary path; first succeeded, second raised the message above.
Also AST-extracted the unmodified nested `progress` function, supplied rank0,
a temporary profile directory, a direct local-operation callback and a generous
clock budget; called it with setup and then setup+inventory. Observed the same
second-call failure, and the persisted JSON contained only setup. No training
helper or CUDA operation was mocked into a success receipt.

Minimal repeat of the underlying contract:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from make_launch_readiness import _atomic_write
with TemporaryDirectory(dir='.') as d:
    p = Path(d) / 'rank0.profiling.json'
    _atomic_write(p, b'{}')
    _atomic_write(p, b'{}')  # SystemExit: output path must be fresh
```

**Required change:** use unique sequence/stage snapshot names, or a deliberately
separate replaceable-progress writer. Do not weaken the immutable receipt/DONE
writer globally. Add a CPU test executing at least two actual progress calls,
including failure propagation; arithmetic-only tests cannot detect this.

### RP2 — MEDIUM: new paired-validation guard is too late for local contract failures

**Locations:** `revision/scripts/mvp/resource_probe.py:188-205`;
`src/transgenic/training/mvp_metrics.py:64-92`.

Retrieval/device transfer is coordinated, but `paired_metric_record` performs
local batch/metadata validation before its first DDP forward. On one rank an
invalid metadata dtype, missing key or invalid batch shape raises immediately;
that rank enters the post-metric `_collective_preflight`. Healthy peers have
already entered the DDP forward, which may synchronize buffers. The status
collective is then out of order with the forward collective. A guard after two
forwards cannot coordinate a failure before the first one.

**Source-level reproduction / regression:** after the guarded retrieval, inject
an invalid `batch[9]['rn']` dtype on rank1 only. Trace rank1 to lines202-205 while
peers enter `paired_metric_record`'s off forward at `mvp_metrics.py:91`. This
fault injection was not run distributed here. Actual NCCL outcomes depend on
runtime; no definite hang duration or false-PASS claim is made.

This is a new resource-path coordination gap, not the base `one_update` R2 being
fixed elsewhere. The producer normally supplies valid metadata, so it is a
negative-path robustness issue. Keep unavoidable backend failures bounded by
the launcher, but coordinate ordinary local metric-input checks *before* DDP.
Do not wrap the entire two-forward function in a local-only guard. Add a
rank-specific input-contract failure regression when integrating.

### RP3 — LOW: strict schema claims exceed the implemented integer checks

**Locations:** `revision/scripts/mvp/resource_probe.py:28-33,60-61,86-92,116-118`.

Several identity/count fields use Python equality rather than exact integer
types. Independently reproduced acceptance of `schema_version=True`, envelope
`{'nodes': True, 'ranks': 4.0, 'microbatches_per_rank': 24.0}`, rank0 represented
as `False`, `optimizer_steps=True`, `microbatches=24.0`, and string RN lists.
The modified arithmetic fixture still returned projected run seconds `2460.0`.
Timing, capacity, memory and seed types have stronger checks already.

**Reproduction:** obtain `budget,ranks` from `tests/test_mvp_resource_probe.py`'s
`fixture()`; apply the substitutions above to every update, preserving distinct
RN strings; call `validate_budget(budget, 'a'*64)` and `project_budget(budget,ranks)`.
I loaded only that fixture function via AST because pytest is absent locally.

These substitutions do not change the numeric projection or demonstrate an
unsafe live CUDA PASS: the real producer emits integer counts/RNs. This is a
receipt-schema weakness, not an authorization-authenticity exploit. Require
exact integer types where the schema promises integer IDs/counts and test them.

### RP4 — LOW: committed diff is not whitespace-clean

`git diff --check 3b62306..4bd97399` independently reports:

```
tests/test_mvp_resource_probe.py:150: new blank line at EOF.
```

This contradicts the implementation report's blanket diff-check success claim.
Remove the extra line and report the checked revision/range. No runtime impact.

## Workload and estimate assessment

- **Base integration is correctly disclosed.** `run:318-320` rejects the shared
  factory's v2/2048 dataset before compilation; it does not silently certify it.
  Integrate the owner's v3/8192 train/valid factory fix and coordinated
  `one_update` fix, preserving/updating the actual signatures, then regenerate
  consumed-code/tooling bindings. These acknowledged base defects are not new
  RP findings. The new validation constructor explicitly requests config v3.
  Controller update received during review: owner fix `981489d` (report
  `d99b1f8`),92 focused CPU passes in130.55s; separate fix review `7d0d2dad`
  pending. These are attributed updates, not reviewed or merged here. Exact
  integration requirements: shared `datasets(args, bindings, cfg, model_config)`
  replaces this branch's two-argument call; `compiled_update(args, bindings,
  model, ds, accelerator)` receives the main-owned Accelerator. Reconcile the
  resource early dispatch/Accelerator ownership so there is one owner, preserve
  cleanup, and use the new `validate_decoder_dataset`, `collective_batches` and
  `collective_compiler_check` contracts where applicable. Integrate only after
  both reviews, and review the combined resource call sites rather than assuming
  the owner's shared fix alone repairs this branch.
- **Nine updates are honestly represented.** Three tiers each request one cold
  and two steady updates, each 24 microbatches/rank, with 96 disjoint selected
  RNs per global update. Selection scans the full accepted train inventory and
  chooses the longest labels at each full sequence tier. Reuse across updates
  is explicit; no invented 250-update finalizer call was found.
- **LR zero does not skip AdamW state allocation.** Shared `one_update:477-479`
  resets gradients and warmup index0 on all nine calls. A local Torch2.6 CPU
  AdamW step with LR0 independently created `step`, `exp_avg`, `exp_avg_sq`, left
  parameters unchanged and reset gradients to `None` after `zero_grad()`.
  The helper does backward/reductions/clipping and real optimizer steps. The
  evidence projection in `modeling_HyenaTransgenic.py:531-556` is applied densely
  when tracks are present; there is no zero-weight shortcut there. Thus LR0 is
  not evidence that optimizer memory or dense forward/backward is skipped.
  It does leave the zero-initialized projection unchanged and does not exercise
  nonzero learning, changing parameter values or learning-rate history. Treat
  this as profiling, not a convergence or post-learning numerical certificate.
- **EST and shape representativeness:** training augmentation is restored to
  random thinning/channel dropout; nonzero EST is required per rank/update.
  This is not exhaustive coverage of every evidence pattern. Longest labels
  and uncropped full tiers are sensible expensive cases, but smaller shapes,
  different shape order, allocator fragmentation, other seeds and new guards
  can have distinct costs. Two steady repetitions cannot prove an upper bound.
  Peak allocated/reserved CUDA accounting excludes non-Torch GPU allocations
  and host RAM; full scan/preloaded batches can consume substantial host memory.
- **Compilation policy is fail-closed but operationally fragile.** Cold updates
  allow recompiles; steady updates prohibit them and require zero new kernels.
  Strict cache/recompile-limit support is mandatory. Real label shapes, graph
  breaks, training-state guards or validation mode can cause new compilation or
  limit failures. Whether the pinned Torch2.14 SIF supports these knobs and
  completes the workload is untested. Do not relax fallback policy or assert
  that it must fail/succeed based on this CPU review. Repeated shapes do not
  cover the production sampler's full compile history.
- **Whole validation and duplicates:** the real path checks accepted train and
  valid RN lists, uses full deterministic padded `DistributedSampler`, gathers
  all paired records with byte limits, and uses `aggregate_paired_records`.
  That helper rejects missing/extra RNs and inconsistent duplicate values,
  deduplicating before equal-tile aggregation. I found no new duplicate-count
  bias. Receipt arithmetic checks cross-rank count/hash agreement, not an
  independent replay of validation or external reconciliation of those fields;
  do not describe those hashes alone as proof every sample ran.
- **Publication scope is appropriate.** Calls are low-level payload write,
  manifest hashing/re-read and fsync, with `PROBE_ONLY_CHECKPOINT`, nine steps,
  216 microbatches/rank. No call to `_publish_final`/`_write_done_marker` and no
  `TRAINING_DONE` write was added. Publication filesystem equivalence is only
  an operational prerequisite; recording `st_dev` does not bind a future
  training destination. Wrong-filesystem timing cannot certify publication.

## Arithmetic, identity, and five-node-hour boundary

The recomputed formula uses the maximum individual rank/tier steady update,
adds maximum rank probe elapsed (including all nine updates, setup, scan,
validation and serialization), multiplies by a prospective margin, and adds
launcher allowance. Three-seed hours are three projected runs /3600. Required
stage/tier sets, positive finite times, stage-sum consistency, strict wall fit,
strict three-seed fit, allocated/reserved memory headroom and cross-rank valid
identity checks reject the common missing/undersized cases. Adding all probe
work again is deliberate overcount, not double-counting hidden as a speedup.
No source-proven arithmetic-underestimate route was established for the real
producer; this remains a conditional estimate, not a future-cost guarantee.

**The five-hour arithmetic is not an allocation-level deadline.** Budget input
limits prior smoke + maximum child probe wall to18000 seconds, and resource mode
checks the exported child timeout. However, `run_bounded.py:165-192` still permits
separate warmup attempts and performs identity/asset/authorization validation
after `run_command`'s timeout ends. The child clock also ends before final gather,
measurement publication and cleanup (`resource_probe.py:463-476`). These facts
are inherited runner behavior plus the new probe's measurement boundary, not a
new demonstrated false-DONE path. The report tells operators not to add warmup;
the code does not enforce that prohibition for resource mode.

Consequently do not claim software alone caps total allocation consumption at
five node-hours. The controller must account for all startup, prior attempts,
teardown and parent rehashing, reserve time for them, and impose an outer
allocation/job wall deadline within the remaining cap. In particular, allowing
`max_probe_wall_seconds=18000` with zero prior spend leaves no guaranteed time
for unmeasured parent overhead. A stricter software guarantee needs a shared
outer deadline and rejection/accounting of resource-mode warmups, not an
arithmetic margin on future training. The prospective launcher allowance in the
training projection is not itself such a deadline or measured smoke accounting.

Tooling includes the new module; runtime checks source/config/candidate/SIF and
asset identities, imports, and revalidates inputs/budget. These are byte and
scope bindings, **not authentication of an author/controller**, proof of
execution from hashes alone, or new scientific/production authorization.
The authorization record must be supplied prospectively by the controller;
nonempty intent/reference/rationale strings do not establish provenance.

## Verification and release conditions

- Independent focused CPU evidence: fresh-path/progress failure; type acceptance;
  LR0 AdamW state allocation; committed-range whitespace check. No dependencies
  were installed, no source fixtures changed, no CUDA certificate produced.
- Controller-provided result: **126 tests passed in58.99s**, without the subtests
  plugin in a different CPU environment. Not independently rerun here; local
  interpreter lacks pytest. Writer's126 +21 subtests in52.41s is an attributed
  separate run, not SIF validation. Neither detects RP1 by exercising `run`.
- Fix RP1, harden RP2, and integrate the separately owned base fixes before
  requesting any resource execution. Add actual orchestration/progress and
  distributed negative-path coverage, not only additional arithmetic fixtures.
- Re-review the combined diff/bindings, retain pinned SIF versions, require
  controller-owned budget/provenance and remaining-cap accounting, then run
  bounded real CUDA diagnostics. Successful completion must precede any resource
  PASS; this report supplies no permission to launch training.
