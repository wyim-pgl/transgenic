# B6 MVP fixed-update runtime increment report (2026-09-15)

## Scope and base

- Branch: `b6-mvp-fixed-runtime`
- Required base: `3a633143d48418f22bd084f8d9f41b5065b29c25`
- Native isolated worktree, CPU only. No main-worktree edits, remote actions, GPU actions, merge, or push.
- Added only the runtime helper, its tests/subprocess probe, and this report. There is no CLI, model/data/provenance/publication wiring, checkpointing, or resume support in this increment.
- `graft build`, `graft map`, and focused graph queries preceded source inspection. Generated graph/cache and local test dependencies are ignored and are not committed.

## Implementation

`src/transgenic/training/mvp_runtime.py` provides:

- Frozen `MVPRecipe` production defaults: 250 successful updates, 12 warmup updates, batch size 1, per-rank accumulation 24, required world size 4, backbone LR `2e-5`, projection LR `2e-4`, uniform AdamW weight decay `0.02`, and clipping at `1.0`. Test recipes may explicitly use smaller values. Integer/numeric types, finiteness, and ranges are validated.
- `build_optimizer`, with stable `backbone` and `projection` groups. The only projection parameter is exactly `transgenic.encoder.evidence_proj.weight`; all remaining trainable parameters are assigned once to the backbone group. Both groups use uniform recipe weight decay, with no invented bias/norm exception.
- Canonical post-step LambdaLR-equivalent `lr_scale(k)`, where `k` is the number of successful updates already completed:
  - `k / 12` for `k < 12`
  - `(250 - k) / (250 - 12)` for `12 <= k < 250`
  - zero for `k >= 250`

  Consequently, the LR used by optimizer updates 1, 12, 13, and 250 is respectively zero, `11/12 * peak`, peak, and `1/238 * peak`; after update 250 it is zero. This deliberately does not use the earlier shifted planner table.
- `make_accelerator`, a small factory for `GradientAccumulationPlugin(num_steps=accumulation_steps, sync_with_dataloader=False, adjust_scheduler=False)` and `step_scheduler_with_optimizer=False`.
- `run_fixed_updates`, which validates those settings and the required world size, requires caller-prepared model and optimizer, rejects a prepared dataloader, and under DDP requires one rank-matched `DistributedSampler` on the unprepared loader. Only batch elements 0/1/2/8 (DNA, mask, labels, eight-channel floating evidence tracks) move to the accelerator device; `evidence_tracks=batch[8]` is always forwarded and no transcript count is forwarded.
- Equal nonempty per-rank loader lengths and startup/update/final counter agreement are collective checks. `sampler.set_epoch(epoch)` is called when available. Accumulation and pending gradients continue across epoch boundaries without flushing.
- Loss finiteness is checked collectively before backward. On synchronized boundaries gradients are checked collectively before clipping, followed by clipping, optimizer step, collective skipped-step inspection, zeroing, successful-counter increment, and installation of the next LR. A skipped step aborts rather than shortening the budget. The loop stops after the last required microbatch with no 251st update or extra forward.
- The JSON-compatible summary includes successful updates, each rank's microbatch count, epoch boundaries crossed, arithmetic effective batch size, both group LRs used for every successful update, and final LRs.

## Test-first evidence

### Red

After creating `tests/test_mvp_runtime.py` and before the helper existed:

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_runtime.py
```

Result: collection failed as expected with `ModuleNotFoundError: No module named 'transgenic.training.mvp_runtime'`.

### Isolated dependencies

The configured interpreter initially lacked Accelerate, pytest, psutil, and pysam. Nothing was installed into the shared environment. Worktree-local `.testdeps` received:

```text
/home/wyim/miniconda3/envs/pdb/bin/python -m pip install --target "$PWD/.testdeps" --no-deps accelerate==0.34.2
/home/wyim/miniconda3/envs/pdb/bin/python -m pip install --target "$PWD/.testdeps" --no-deps psutil
/home/wyim/miniconda3/envs/pdb/bin/python -m pip install --target "$PWD/.testdeps" --no-deps pytest==8.3.5
/home/wyim/miniconda3/envs/pdb/bin/python -m pip install --target "$PWD/.testdeps" iniconfig pluggy
/home/wyim/miniconda3/envs/pdb/bin/python -m pip install --target "$PWD/.testdeps" --no-deps pysam
```

Relevant test versions: Python 3.12.2, Torch 2.6.0+cu124 (CPU execution), Accelerate 0.34.2, DuckDB 1.5.5 from `/tmp/b6-dataset-verify.D2Is9a/.testdeps`, pytest 8.3.5, psutil 7.2.2, and pysam 0.24.1.

### New runtime tests

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_runtime.py
```

Result: **17 passed, 0 failed in 141.95 s**; wall time **142.93 s**.

Coverage includes:

- exact schedule points and all 250 pre-step LRs against a real `torch.optim.lr_scheduler.LambdaLR` stepped after AdamW;
- bit-exact final parameters and AdamW `step`, `exp_avg`, and `exp_avg_sq` state;
- exhaustive/disjoint named optimizer groups and exact projection ownership;
- actual Accelerate 0.34.2 CPU execution at 1 rank (`DistributedType.NO`) and torchrun/Gloo at 2 and 4 ranks (`DistributedType.MULTI_CPU`), with actual world-size assertions;
- accumulation 4 with loader length 5/rank for 3 updates, requiring 12 microbatches/rank and crossing two epoch boundaries during update 2, compared with an analytic global averaged-gradient reference;
- synchronized parameters, clipping and optimizer call counts, explicit sampler epochs `[0, 1, 2]`, and a separately seeded DataLoader worker RNG generator (not a claim of full dataset RNG integration);
- a literal 4-rank, 250-update, accumulation-24 run: 6,000 microbatches and forwards per rank, 250 optimizer calls, effective batch 96, no extra forward/update, synchronized final parameters/counters, and final zero LRs;
- coordinated single-rank NaN loss, nonfinite gradient rejection before step, injected skipped-step rejection through the narrow `_step_was_skipped` seam, empty/unequal loaders, invalid settings, malformed batch, prepared-loader rejection, unprepared-model/optimizer rejection, and a one-rank sampler mismatch that all ranks reject without hanging. Subprocesses use bounded timeouts and captured diagnostics; no mocked Accelerator substitutes certify distributed behavior.

### Required 46-test baseline

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Result: **46 passed, 0 failed in 49.348 s**; wall time **54.66 s**.

Required baseline plus new runtime total: **63 passed, 0 failed**.

Syntax check:

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/home/wyim/miniconda3/envs/pdb/bin/python -m py_compile \
  src/transgenic/training/mvp_runtime.py tests/test_mvp_runtime.py tests/mvp_runtime_probe.py
```

Result: exit 0, no output.

## Additional broad-suite observation

A non-required broad repository run was attempted after installing isolated pysam:

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q --ignore=tests/test_mvp_runtime.py
```

Result: **698 passed, 24 failed in 583.01 s**. The failures are outside this increment and include absent external `seqkit`, absent `/tmp/transgenic_comparison/genomes/Athaliana_167_TAIR10.fa.fai`, and existing environment/order-sensitive tests. This broad run is recorded, not represented as green; the required 46-test baseline above is green.

## Limitations and concerns

- These are real CPU/Gloo tests, but not production certification. Production is recorded as Torch 2.14 in NGC 26.08; the local CPU interpreter is Torch 2.6.0+cu124.
- No GPU, mixed-precision overflow, production model, production dataset, or production checkpoint was exercised. The skipped-step path is tested through the narrow flag-inspection seam because CPU has no scaler overflow.
- Sampler epoch transitions and a distinct loader RNG source are tested. Full dataset worker RNG, augmentation, and resume determinism are intentionally not claimed.
- The caller must preserve the prepared-model/prepared-optimizer/unprepared-loader ownership contract when later CLI wiring is added. Preparing the loader would double-shard it and is rejected.
- There is no checkpoint save/resume, CLI, publication record, bootstrap, or paired evaluator integration in this increment.
