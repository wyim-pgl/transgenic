# Resource-budget CUDA probe implementation — NOT GPU certified

Branch `mvp-resource-probe-20260916`, base `3b62306`, isolated native worktree
`/tmp/pi-agent-83dfcd31-783f-41d-de767c64`. No SSH, jobs, subagents, merge, push,
main edits or A-worktree edits. No controller authorization was invented and no
CUDA receipt was produced. Advisor was attempted but unavailable
(`claude_code_version_too_old`).

## Integration blocker: combine the independent fixes BEFORE real execution

Controller reported R1/R2 from
`notes/mvp_execution_20260916/real_probes.independent_review.md` while this task ran.
That main-worktree file was not opened across the isolation boundary.

- R1: base `real_probes.datasets` defaults to v2/2048 despite the v3/8192 model.
  This branch **explicitly refuses that dataset**, before compilation. Owner's
  shared factory fix must be integrated. New validation-dataset construction
  passes `cfg['gff_vocab_version']` explicitly, checks v3/8192 and vocabulary size.
- R2: owner's coordinated `one_update` fix must also be integrated. This branch
  does **not edit `one_update`, `datasets`, or `mvp_cli_runtime.py`**. It calls the
  shared helper, coordinating its own retrieval and post-kernel checks with
  `_collective_preflight`. DDP/transport collectives are not wrapped in that
  local-work guard. Collective/backend hangs still require bounded-launch teardown.
- Shared-file hunks: `real_probes.py` docstring, MODES, parser's extra resource
  inputs, and early resource dispatch in main only; `make_launch_readiness.py`
  adds `resource_probe.py` to tooling hashes; `run_bounded.py` replaces only the
  resource validator and exports its timeout. Preserve the owner's factory/helper
  signatures when combining; if `datasets(args, bindings)` changes signature,
  update the one call in `resource_probe.run` accordingly.
- `tests/test_mvp_real_probes.py` changes only the obsolete resource-mode refusal
  assertion. `tests/test_mvp_readiness_tools.py::prepared` explicitly mocks the
  resource-certification boundary for unrelated CPU launcher tests. Previously
  those tests silently treated synthetic CUDA booleans as valid certification;
  production now rejects them, independently tested in the new suite.

The actual public trainer is unchanged: 250 updates, 12 warmup, four ranks,
24 microbatches/rank, effective batch96. No public resume/update override exists.
The resource profiler performs nine probe-local updates; it retains the shared
one-update helper's LR-at-warmup-index0 semantics, not a claimed 250-update run.
This is compute/memory profiling, not a scientific convergence experiment.

## Implemented measurement

New `revision/scripts/mvp/resource_probe.py` is dispatched by
`real_probes.py --mode resource_budget`.

1. Explicit SHA-pinned prospective controller budget, CUDA/native bf16, actual
   SIF versions, prepared identities and four distinct physical ranks required.
   `TRANSGENIC_RUN_DIR` is prohibited even when empty. No eager/CPU fallback.
2. Measure setup, model initialization, compile binding and DDP preparation.
   Independently audit accepted corpus/evidence/panel/SIF and candidate companions.
   Full train scan reads actual labels/evidence and verifies the train/valid RN
   inventories against acceptance. Its cost is included, not discounted.
3. For each full 30720/61440/129024 tier, select the **96 longest-label real train
   rows**, shard deterministically into 24 disjoint rows/rank, and perform one
   cold compile/warmup update and two steady updates. The same RNs recur across
   the three updates; each read gets fresh actual training augmentation. This
   repetition is disclosed in the receipt, not fabricated data. Fewer than96
   rows at a tier fails; no crops or synthetic length padding are introduced.
4. Record every update's RNs, input/label lengths, observed nonzero EST count,
   synchronized seconds, actual microbatch/update counts, generated-kernel delta,
   per-rank allocated/reserved CUDA peaks and exact device capacity. Include the
   inventory maximum label length in steady coverage at each tier. Disable
   suppressed compiler errors and cache-limit eager fallback; prohibit recompiles
   in steady updates. Unknown compiler fail-closed support fails explicitly.
5. Complete accepted valid inventory through the actual `paired_metric_record`
   off/on path (fraction0/1), actual padded DistributedSampler, byte-bounded
   record transport, duplicate-aware `aggregate_paired_records`, summary broadcast.
   Record read/transfer, paired metric, transport/reduction and full stage times.
   This is teacher-forced paired validation, not the separate heldout generation
   evaluation. Heldout or final study claims are not authorized by this probe.
6. Fresh `PROBE_ONLY_CHECKPOINT` uses actual low-level `_write_payload_files`,
   `_write_manifest` (serialization, file fsync, hashing/re-read), directory/parent
   fsync. Metadata truthfully says nine updates and 216 microbatches/rank, scope
   resource-probe-only. **Never calls `_publish_final` or `_write_done_marker`;
   never writes TRAINING_DONE.** Keep profiling scratch on the intended training
   publication filesystem; a different filesystem's timings cannot certify it.
7. Independently revalidate inputs, source/sidecars, assets, consumed code and
   budget. Import/cache hashing and all intervening work are included in elapsed.
   Entire probe elapsed is conservatively added as overhead to the projected
   training, including all nine probe updates (intentional overcount).

Formula, recomputed by the receipt validator:

```
worst = max(all ranks, all tiers, both steady update wall times)
projected_run = (250 * worst + max_rank_entire_probe_elapsed) * safety_margin
                + launcher_overhead_seconds
three_seed_node_hours = 3 * projected_run / 3600
```

Require strict wall fit and strict three-seed node-hour fit; all stage allocated
AND reserved peaks below capacity minus the prospective memory-headroom fraction.
All stages, three tiers, four ranks, complete valid identity and two steady
updates are mandatory. Missing/NaN/nonpositive stages, omitted costs, new steady
kernels, duplicate within-update training RNs or absent EST coverage fail.

This is a conservative **estimate**, not a mathematical bound on all future
shape-dependent compilation, cluster contention or seed variation. The explicit
prospective margin/rationale remains controller-owned. Memory outside Torch's
allocator is not directly profiled; require exclusive devices and appropriate
headroom. Model kernels, data-label checks and the orchestration still need real
combined-code testing in the pinned SIF.

## Input budget contract (no authorization supplied here)

Controller creates an immutable JSON and independently supplies its SHA256.
Required fields:

- `schema_version: 1`, `authorized_scope: "exploratory_mvp"`;
- `controller_record_reference`: actual controller's offline authorization record
  reference, not an invented person/signature; `intent`: explicit three-seed intent;
- `seeds: [1,2,3]`;
- `resource_envelope: {"nodes":1,"ranks":4,"microbatches_per_rank":24}`;
- `bindings_sha256`: exact prepared binding digest for this seed/probe;
- `safety_margin > 1`, `margin_rationale`: prospective nonempty rationale;
- `0 < memory_headroom_fraction < 1`;
- positive `max_wall_seconds` per training run and `max_three_seed_node_hours`;
- nonnegative `prior_smoke_node_hours`: count ALL prior smoke/probe allocation
  usage, not zero merely because this executable is new;
- positive `max_probe_wall_seconds`, with
  `prior_smoke_node_hours*3600 + max_probe_wall_seconds <= 18000`;
- positive `launcher_overhead_seconds`: prospective external-launch/startup
  allowance, including any production CUDA-startup retries (up to180 seconds
  under the existing launcher) and controller-side overhead not measured here.

No `approved:true` shortcut, fabricated author identity or asserted digital
signature is accepted in place of that record. The software checks contents,
identity and arithmetic, **not authenticity of a controller's identity**; record
provenance is controller responsibility. A separate seed binding needs its own
prospectively bound record; three-seed intent does not unbind seed-specific data.

The five-node-hour smoke cap is NOT increased. This implementation may require
more than the remaining cap: full label/evidence scanning, long-tier compilation,
864 global microbatches and full paired validation can be expensive. No real
measurement exists here to assert that it fits. The controller must not authorize
an over-cap run. If bounded execution expires, retained per-rank
`rankN.SEQUENCE.profiling.json` (six-digit, increasing sequence per rank;
 retain all snapshots and select the greatest numeric sequence per rank) gives
 useful completed setup/inventory/cold/steady timings,
but **cannot produce PASS or training readiness**. No timing from an incomplete
long tier or partial valid inventory is extrapolated into a success receipt.
`projected_campaign_node_hours` additionally reports probe/prior-smoke plus all
three projected training runs; these costs are not hidden in the training-only
three-seed field.

## Exact CLI shape for later controller review/run

Preserve original pins (NOT the local test environment):
`torch 2.14.0a0+4fdf77b940.nv26.08`, CUDA13.4,
`transformers 4.57.6`, `accelerate 1.15.0`.
Use original offline HF/cache/CUBLAS environment in MVP_REAL_PROBES_REPORT.md,
with a fresh independent compiler cache and scratch, no `TRANSGENIC_RUN_DIR`.
Regenerate prepared bindings/manifests after integration because tooling changed.

Store this expanded argv as a JSON array (absolute Python executable and paths):

```
/ABS/PYTHON -m accelerate.commands.launch --multi_gpu --num_machines 1 \
  --num_processes 4 --mixed_precision bf16 \
  --config_file /work/repo/revision/scripts/mvp/accelerate_mvp.json --max_restarts 0 \
  /work/repo/revision/scripts/mvp/real_probes.py --mode resource_budget \
  --bindings "$BINDINGS" --bindings-file-sha256 "$BINDINGS_SHA" \
  --assets-manifest "$ASSETS" --assets-sha256 "$ASSETS_SHA" \
  --db "$CORPUS" --evidence-db "$EVIDENCE" --config "$CONFIG" \
  --tile-rns "$TILES" --tile-rns-sha256 "$TILES_SHA" --scratch-root "$SCRATCH" \
  --budget-authorization "$BUDGET" --budget-sha256 "$BUDGET_SHA" \
  --profile-dir "$SCRATCH/fresh-resource-profile" \
  --acceptance-record "$ACCEPTANCE" --inventory-record "$INVENTORY" \
  --receipts-record "$RECEIPTS" --panel "$PANEL" --sif "$SIF"
```

The inherited parser still requires tile-rns flags; resource selection instead
comes from the complete accepted training inventory, not those six step0 tiles.

```
/ABS/PYTHON /work/repo/revision/scripts/mvp/run_bounded.py --name resource_budget \
  --bindings "$BINDINGS" --bindings-file-sha256 "$BINDINGS_SHA" \
  --command-file "$ARGV_JSON" --command-sha256 "$ARGV_SHA" \
  --output-dir "$FRESH_RECEIPT_DIR" --timeout-seconds "$AUTHORIZED_REMAINING_BOUND"
```

Runner exports `MVP_PROBE_TIMEOUT_SECONDS`; the resource mode requires it positive
and no greater than the budget's probe bound. Do not add a separate untimed warmup
invocation. Partial profiling uses separate files, never MVP_MEASUREMENT_OUTPUT.
Failed/interrupted probes have no resource DONE. Only a complete resource gate
can subsequently contribute to the six-gate controller readiness workflow.

## CPU verification, not a GPU certificate

Local ignored-by-staging `.testdeps` contains test-only pytest/accelerate/duckdb
and pytest dependencies, installed only in this isolated worktree. Actual local
versions: Torch2.6.0+cu124, Transformers4.46.3, Accelerate1.15.0. They are NOT SIF
certification and must NOT be installed into the SIF.

Test-first: 23 failures, including the existing false acceptance of synthetic
resource flags. Initial interpreter lacked pytest; system Python exercised the
pure arithmetic tests. After implementation, broad tooling run had84 passed /
26 failed because old launcher fixtures relied on synthetic resource acceptance.
Those fixtures now explicitly mock that boundary; the separate resource suite
uses the real validator for rejection and real arithmetic for aggregation.

Final focused command:

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
  tests/test_mvp_resource_probe.py tests/test_mvp_real_probes.py \
  tests/test_mvp_readiness_tools.py tests/test_mvp_readiness_review.py \
  tests/test_mvp_metrics.py
```

Original implementation reported: **126 passed, 21 subtests passed in52.41s**,
including the local-work coordination and label-contract checks. The original
blanket whitespace-success claim was incorrect: independent review found a new
EOF blank line in `tests/test_mvp_resource_probe.py` in `3b62306..4bd97399`.
FIX1 removes it; see `MVP_RESOURCE_PROBE_FIX1_REPORT.md` for the checked range and
fresh CPU results. The original report also recorded `py_compile` and a local
`graft build`. A broader run including
`test_mvp_finalize.py` exceeded the240-second tool bound after substantial test
progress; no full-finalizer-suite success is claimed. No real-CUDA orchestration
or performance test was possible, and none was simulated into a receipt.
