# B6 MVP paired evaluation and atomic publication report (2026-09-15)

## Scope and base

- Branch: `won/b6-mvp-finalize`
- Required original base: `a314bf9343846d64d768f14d4ae4909beec08a40`
- Scoped round-1 review base: `428e36edffc1175a6158ae83a43ecdb276ca11a8`
- Scoped round-2 review base: `efee9e52bbaca52260619763b2b4b0b6e51e3e93`
- Scoped round-3 review base: `3fe33cb751c79396cafbc6d9e45c2d32fab0ad3f`
- Native isolated worktree, CPU only. No main-worktree writes, remote actions, GPU actions, merge, or push.
- Added only `mvp_finalize.py`, its focused tests and subprocess probe, this report, and the requested scoped wording correction. There is no CLI, bootstrap, GO/STOP policy, resume, or general checkpoint framework.
- The exact neutral brief had SHA256 `ef3c585fb08df09ede59c7350cd0492281d0f6998517adc624f3ec7b80968943`. Graph mapping and focused graph queries preceded source edits. Graph files and the local dependency link are not committed.

## API and distributed evaluation

`evaluate_and_publish_final(accelerator, model, eval_loader, expected_rns, final_dir, *, metadata, training_summary)`:

- accepts `expected_rns` only when `type(value)` is exactly the built-in `list`, `tuple`, or `range`; type and cheap length bounds are checked collectively before copying or inspecting elements, so arbitrary, subclass-overridden, throwing, or infinite iterators are never consumed;
- validates the complete finite unique integer inventory, metadata, training summary, model, unprepared batch-one finite loader, supported distributed type, and rank/world-matched `DistributedSampler(shuffle=False, drop_last=False)` before a forward;
- compares loader lengths and caller identity payloads across ranks, with `MVP_MAX_RECORDS = 100000` and `MVP_MAX_SERIALIZED_BYTES = 67108864` fail-closed bounds (both deliberately generous relative to 1,447 valid tiles and never truncating);
- moves nested tensor inputs, including `rn` and `has_evidence` metadata, to the accelerator device and calls the reviewed `paired_metric_record` unchanged, preserving its unconditional off-then-on order and model mode restoration;
- evaluates the original prepared/compiled model. It never unwraps to an eager target for evaluation;
- canonicalizes each local record list under a caught error, exchanges only fixed-size error/length controls first, and rejects any local length, aggregate byte sum, or `world_size * max_length` padded transfer above the byte cap before transferring record bytes;
- gathers records as exactly bounded padded `uint8` tensors rather than pickled Python records. Rank 0 decodes and calls reviewed `aggregate_paired_records`; the bounded canonical summary is broadcast as `uint8`;
- carries preflight, aggregation, and publication status through the same scalar-length plus bounded-`uint8` pattern, with no object collective. The fixed status schema caps UTF-8 diagnostics at 2,048 bytes, marks truncation, appends `...[diagnostic truncated]`, and caps final paths at 4,096 bytes. Scientific records are rejected rather than truncated;
- publishes only on rank 0 and broadcasts publication status before the final all-rank barrier. Every rank returns the decoded identical summary; padded sampler duplicates remain checked by the reviewed aggregator.

Replicated DDP and ordinary single-process evaluation are supported. FSDP, DeepSpeed, unknown wrappers, prepared evaluation loaders, and non-DDP multi-rank models are rejected. Only rank 0 unwraps for saving, using the pinned Accelerate 0.34.2 signature `unwrap_model(model, keep_fp32_wrapper=True)`. An exact Torch `OptimizedModule` is reduced once to `_orig_mod`; there is no recursive `.module` stripping. Wrapper-prefixed state keys are rejected.

## Metadata schema

The required top-level metadata keys are:

- `identity_scope`: exactly `production` or `test`;
- `initialization_audit`: `verification`, the complete returned `load_weights_only` `audit`, and `audit_sha256` over the included canonical audit;
- `data_identities`: required `corpus`, `evidence`, and `panel` identity objects;
- `sif_code_identity`: required `sif` and `code` identity objects;
- `runtime_packages`: exactly nonempty version strings for `python`, `torch`, `transformers`, `accelerate`, and `safetensors`;
- `normalized_recipe`: exactly the nine `MVPRecipe` fields (`max_updates`, `warmup_updates`, `batch_size`, `accumulation_steps`, `required_world_size`, both LRs, weight decay, and clipping);
- `seed_augmentation`: exactly integer `seed` and `aug_seed = 20260914 + seed`; production seeds are restricted to 1, 2, or 3.

The embedded initialization audit must match the actual `b6-weights-init-v1` output contract: exact top-level and six-file inventory; complete `sha256/size/device/inode/mtime_ns` snapshots; concrete absolute paths; unchanged source snapshots; exact current semantic field set with `value` and valid source/target provenance; matching source/target semantics except evidence channels 0/8; recomputed manifest hashes; exact missing projection, no unexpected key, unsupported same-8 loading, and a verified-zero projection action. Nested contracts are exact: `recipe` contains only `canonical_sha256`; `compatibility` contains only `transition`, `exception`, and `same8_supported`; `load` contains only `missing_keys`, `unexpected_keys`, and `safetensors_metadata`; implementation is exactly the two reviewed filenames and each record contains only `path` and `sha256`. Safetensors metadata is either `None` or a string-to-string map, matching the pinned API return. A focused integration test runs the real tiny `load_weights_only` path and passes its returned audit into finalization. The wrapper digest binds the included report, but the schema and caller-supplied digest do not prove authenticity or independently attest bytes. Finalization records the successful loader audit and does not rehash those source files.

Each corpus/evidence/panel/SIF/code identity object has a 64-character lowercase hexadecimal `sha256` and `verification` equal to `recorded` or `independently_verified`. Finalization preserves that supplied distinction and does not promote a recorded digest to independent certification. `MVPRecipe` validates every recipe; production requires its exact defaults. `training_summary` must match recipe updates and `batch_size * accumulation_steps * required_world_size`; production therefore requires 250 and 96. Smaller valid recipes require explicit test scope. JSON is canonical (sorted compact keys, newline, `allow_nan=False`). The published `metadata.json` adds the validated training summary.

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

Round-3 result: **39 passed, 0 failed in 184.10 s**; wall time **185.11 s**.

Coverage includes ordinary one-rank evaluation; actual two-rank CPU/Gloo runs with 3 items/2 ranks and 1 item/2 ranks; exact record fields and hand-derived equal-tile off/on primary and composite means; padded duplicate agreement; identical all-rank summaries; and missing, extra, inconsistent duplicate `rn`, and sampler-rank rejection before publication. Four bounded-time two-rank cases make only rank 1 supply a noniterable, throwing iterable, infinite generator, or hostile `list` subclass and verify all ranks receive the same concrete-inventory rejection before forwarding. A huge one-rank exception produces the same capped marked diagnostic on both ranks while patched object collectives remain unused. The model is an actual lightweight `torch.compile(..., backend=eager)` wrapper prepared through Accelerate; backend invocation and exact forward counts prove evaluation used that wrapper, while rank 1 is configured to fail if saving tries to unwrap there. This is compile-path coverage, not an Inductor certification.

Transport tests cover exact sum/padded byte boundaries and a real two-rank case where each canonical local payload is below a patched 1,000,000-byte limit but aggregate/padded transport exceeds it; a patched payload-transfer seam proves the payload collective is not called. Publication tests cover strict audit/package/recipe/seed metadata, production budgets, nonfinite JSON, record bounds, canonical payloads, manifest sizes/hashes, clean state keys, tied safetensors reload, FSDP/DeepSpeed and unknown-wrapper rejection, all existing destination kinds, local fault injection before payload/manifest/marker/rename, collective rank-0 write and rename failures with identical errors and no final artifact, a destination appearing at rename, two concurrent publishers with exactly one winner, and collective committed durability errors after rename. Round 3 mutates a real loader audit by removing every required recipe/compatibility/load/implementation-record field, adding extras to each nested section, changing implementation filenames, and supplying invalid safetensors metadata; every recomputed wrapper is rejected while the unmodified real audit remains accepted.

### Existing 20-test runtime suite

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_runtime.py
```

Round-2 result: **20 passed, 0 failed in 159.01 s**; wall time **160.02 s**.

### Required 46-test baseline

```text
PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps" \
/usr/bin/time -f 'DURATION %e s' \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -q \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Round-3 result: **46 passed, 0 failed in 46.646 s**; wall time **52.23 s**.

Round-3 scoped total: **85 passed, 0 failed**. The unrelated 20-test runtime suite was not rerun because this round changed only nested initialization-audit validation and its tests; its most recent round-2 result above remains recorded.

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
- Coordinated reachable Python validation, bounded transport, aggregation, and publication exceptions are normalized. Once an exception has rendered text, transport bounds it; this cannot stop a malicious or nonterminating exception `__str__` implementation before text exists. Rank death, communicator failure, and arbitrary asymmetric DDP-forward exceptions are not recoverable here; launcher/process-group timeouts remain required. Equal padded workloads prevent ordinary forward-count mismatch.
- Strict no-replace publication requires Linux libc/filesystem `renameat2(RENAME_NOREPLACE)`. Unsupported platforms/filesystems intentionally fail closed.
- Parent fsync failure means namespace commit succeeded but crash durability is unknown. The artifact remains and cannot be retried at the same destination.
- DDP synchronization is relied on for replicated parameter values. No all-rank state-dict materialization, all-rank manifests, training-state restoration, CLI, bootstrap, resume, or statistical bootstrap was added.
