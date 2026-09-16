# B6 MVP paired evaluation and atomic publication report (2026-09-15)

## Scope and base

- Branch: `won/b6-mvp-finalize`
- Required base: `a314bf9343846d64d768f14d4ae4909beec08a40`
- Native isolated worktree, CPU only. No main-worktree writes, remote actions, GPU actions, merge, or push.
- Added only `mvp_finalize.py`, its focused tests and subprocess probe, this report, and the requested scoped wording correction. There is no CLI, bootstrap, GO/STOP policy, resume, or general checkpoint framework.
- The exact neutral brief had SHA256 `ef3c585fb08df09ede59c7350cd0492281d0f6998517adc624f3ec7b80968943`. Graph mapping and focused graph queries preceded source edits. Graph files and the local dependency link are not committed.

## API and distributed evaluation

`evaluate_and_publish_final(accelerator, model, eval_loader, expected_rns, final_dir, *, metadata, training_summary)`:

- validates the complete finite unique integer `expected_rns` inventory, metadata, training summary, model, unprepared batch-one finite loader, supported distributed type, and rank/world-matched `DistributedSampler(shuffle=False, drop_last=False)` before a forward;
- compares loader lengths and caller identity payloads across ranks, with `MVP_MAX_RECORDS = 100000` and `MVP_MAX_SERIALIZED_BYTES = 67108864` fail-closed bounds (both deliberately generous relative to 1,447 valid tiles and never truncating);
- moves nested tensor inputs, including `rn` and `has_evidence` metadata, to the accelerator device and calls the reviewed `paired_metric_record` unchanged, preserving its unconditional off-then-on order and model mode restoration;
- evaluates the original prepared/compiled model. It never unwraps to an eager target for evaluation;
- gathers each rank's complete local record list once after local forwards, aggregates only on rank 0 with reviewed `aggregate_paired_records`, broadcasts the result/error, then publishes only on rank 0 and broadcasts a second publication status before the final all-rank barrier;
- returns an identical paired summary on every rank. Padded `DistributedSampler` duplicates are checked and deduplicated by the reviewed metric aggregator.

Replicated DDP and ordinary single-process evaluation are supported. FSDP, DeepSpeed, unknown wrappers, prepared evaluation loaders, and non-DDP multi-rank models are rejected. Only rank 0 unwraps for saving, using the pinned Accelerate 0.34.2 signature `unwrap_model(model, keep_fp32_wrapper=True)`. An exact Torch `OptimizedModule` is reduced once to `_orig_mod`; there is no recursive `.module` stripping. Wrapper-prefixed state keys are rejected.

## Metadata schema

The required top-level metadata keys are:

- `identity_scope`: exactly `production` or `test`;
- `initialization_audit`: `report_sha256` plus `verification`;
- `data_identities`: required `corpus`, `evidence`, and `panel` identity objects;
- `sif_code_identity`: required `sif` and `code` identity objects;
- `runtime_packages`, `normalized_recipe`, and `seed_augmentation`: nonempty JSON objects.

Each identity object has a 64-character lowercase hexadecimal `sha256` and `verification` equal to `recorded` or `independently_verified`. Thus a supplied/recorded digest is not represented as independently certified bytes. Production metadata accepts only `successful_updates=250` and `effective_batch_size=96`; lower positive budgets require explicit `identity_scope: test`. JSON is canonical (sorted compact keys, newline, `allow_nan=False`). The published `metadata.json` adds the validated `training_summary`.

## Publication contract

Rank 0 creates one owned, unique sibling staging directory under a preexisting parent. It writes and fsyncs:

1. `metadata.json`
2. `paired_records.jsonl` (sorted by `rn`)
3. `paired_summary.json`
4. `model.safetensors` via alias-safe `safetensors.torch.save_model(..., force_contiguous=True)`
5. `config.json` from the actual save target's `config`
6. `manifest.json`, containing the actual size and SHA256 of each of the five payload files
7. `TRAINING_DONE`, last

Before publication, JSON/JSONL readability, safetensors readability, manifest sizes, and hashes are checked. The marker and staging directory are fsynced. Publication uses only Linux libc `renameat2(..., RENAME_NOREPLACE)`; unavailable syscall/filesystem support fails rather than falling back to an overwrite-capable rename or an exists/rename check.

Any pre-rename error best-effort removes only the owned staging directory. Existing files, empty/nonempty directories, and symlinks are untouched. A destination race has exactly one winner. After rename, the final tree is never removed. A parent-directory fsync failure raises `PublicationDurabilityError(committed=True, final_dir=...)` on every rank: the complete final tree and marker remain namespace-visible, crash durability is uncertain, and retry refuses overwrite.

## Test-first evidence

The first focused test invocation failed during collection with the expected missing-module error:

```text
ImportError: cannot import name 'mvp_finalize' from 'transgenic.training'
```

The authorized read-only dependency target was linked as worktree-local `.testdeps`; no shared environment was upgraded. Test Python was `/home/wyim/miniconda3/envs/pdb/bin/python` with Accelerate 0.34.2, safetensors 0.5.2, and Torch 2.6.0+cu124 executing on CPU.

### New finalization tests

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_finalize.py
```

Result: **25 passed, 0 failed in 90.52 s**; wall time **91.46 s**.

Coverage includes ordinary one-rank evaluation; actual two-rank CPU/Gloo runs with 3 items/2 ranks and 1 item/2 ranks; exact record fields and hand-derived equal-tile off/on primary and composite means; padded duplicate agreement; identical all-rank summaries; and missing, extra, inconsistent duplicate `rn`, and sampler-rank rejection before publication. The model is an actual lightweight `torch.compile(..., backend=eager)` wrapper prepared through Accelerate; backend invocation and exact forward counts prove evaluation used that wrapper, while rank 1 is configured to fail if saving tries to unwrap there. This is compile-path coverage, not an Inductor certification.

Publication tests cover required metadata and production budgets, nonfinite JSON, record bounds, canonical payloads, manifest sizes/hashes, clean state keys, tied safetensors reload, FSDP/DeepSpeed and unknown-wrapper rejection, all existing destination kinds, fault injection before payload/manifest/marker/rename, a destination appearing at rename, two concurrent publishers with exactly one winner, and collective committed durability errors after rename.

### Existing 20-test runtime suite

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_runtime.py
```

Result: **20 passed, 0 failed in 157.22 s**; wall time **158.25 s**.

### Required 46-test baseline

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -q \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Result: **46 passed, 0 failed in 49.398 s**; wall time **54.60 s**.

Total required evidence: **91 passed, 0 failed**.

Syntax compilation also exited 0 with no output:

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/home/wyim/miniconda3/envs/pdb/bin/python -m py_compile \
  src/transgenic/training/mvp_finalize.py tests/test_mvp_finalize.py tests/mvp_finalize_probe.py
```

## Scoped wording correction

`notes/b6_mvp_fixed_runtime_report_20260915.md` now states the properties actually compared by the malformed-optimizer tests: parameter and gradient values/presence, group name/order/LR/weight decay/parameter identity, membership, and tested optimizer-state entry count. It no longer claims every optimizer metadata byte was unchanged.

## Limitations and concerns

- Distributed tests are real CPU/Gloo and the compile test uses the eager backend. They do not certify GPU, mixed precision, Inductor, production model size, or the production filesystem.
- Coordinated reachable Python validation, aggregation, and publication exceptions are normalized. Rank death, communicator failure, and arbitrary asymmetric DDP-forward exceptions are not recoverable here; launcher/process-group timeouts remain required. Equal padded workloads prevent ordinary forward-count mismatch.
- Strict no-replace publication requires Linux libc/filesystem `renameat2(RENAME_NOREPLACE)`. Unsupported platforms/filesystems intentionally fail closed.
- Parent fsync failure means namespace commit succeeded but crash durability is unknown. The artifact remains and cannot be retried at the same destination.
- DDP synchronization is relied on for replicated parameter values. No all-rank state-dict materialization, all-rank manifests, training-state restoration, CLI, bootstrap, resume, or statistical bootstrap was added.
