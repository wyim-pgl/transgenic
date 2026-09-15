# B6 MVP audited weights-only initialization report (2026-09-15)

## Scope and base

- Branch: `b6-mvp-weights-init`
- Required base: `1d3db4200d456f04d1d398daca6b8f8f2ec2381c`
- CPU-only isolated increment.
- Added only:
  - `src/transgenic/training/weights_init.py`
  - `tests/test_weights_init.py`
  - this report
- No trainer/main integration, remote commands, GPU work, dependency changes, or changes to existing B5/model/metric behavior.

## Implementation

`load_weights_only(model, init_from)` implements one audited compatibility transition: a normalized B5 checkpoint with `evidence_channels=0` into a separately constructed fresh target with `evidence_channels=8`.

The helper:

1. Resolves the requested checkpoint directory or exact `model.safetensors` strictly, including `best` directory symlinks, and rejects dangling paths, wrong names, non-regular files, and weights symlinks escaping the canonical checkpoint root.
2. Reads only the canonical weights file, parent `run_config.json`, optional run-root `runtime_identity.txt`, and optional checkpoint-root `config.json`, `meta.json`, and `runtime.json`. The stale `run_config["config"]` path is provenance only. Accelerate, pending-gradient, optimizer, and RNG subtrees are not traversed.
3. Requires the embedded recipe and `REQUIRED_CONFIG_KEYS` plus normalized `encoder_n_layer`. It reconstructs source semantics with the existing `b5_runtime.model_kwargs(recipe)` and `HyenaTransgenicConfig`; it does not instantiate a source model or invoke pretrained/network construction.
4. Compares source and target serialized layout/forward semantics, allowing only evidence channels `0 -> 8`. Known semantic recipe extras may attest reconstructed values but cannot override the historical B5 mapping. An optional `config.json` is consistency evidence only; unknown Hugging Face bookkeeping keys are ignored.
5. Records canonical recipe, semantic-manifest, source-file, and current B5/configuration implementation SHA256 hashes. Source file identity and bytes are checked before and after consumption.
6. Uses `safe_open` for key/shape preflight without materializing a second full state dictionary, verifies actual tied target storage, then uses alias-aware `safetensors.torch.load_model(..., strict=False, device="cpu")`.
7. Accepts exactly one missing key, `transgenic.encoder.evidence_proj.weight`, accepts no unexpected key, zeros that projection under `torch.no_grad()`, and verifies it is zero.
8. Preserves model train/eval mode and Torch/Python/NumPy RNG state. It never uses `torch.load`.
9. Marks a passed target poisoned before validation/load. Any failure leaves `_weights_init_poisoned=True`, and retrying that target is refused. Success sets it to `False`. The API documentation requires callers to discard failed targets because safetensors loading can partially mutate them.

The returned audit is JSON-serializable. It includes requested/resolved paths; before/after identity and digest records; explicit nulls for absent optional metadata; canonical recipe and source/target semantic hashes; field provenance (`recipe-explicit`, `mapping-derived`, `config-default`, or `target-config`); safetensors alias metadata; exact missing/unexpected keys; the compatibility exception; and implementation hashes. The audit states that config defaults are reconstructed under the current loader and are not checkpoint-attested historical facts.

## Test-first evidence

### Baseline before implementation

The configured interpreter does not contain pytest:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
  tests/test_mvp_metrics.py tests/test_mvp_model.py tests/test_mvp_dataset.py
```

Result: exit 1, `/home/wyim/miniconda3/envs/pdb/bin/python: No module named pytest`.

The repository tests are `unittest` suites, so the dependency-compatible baseline was run as:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v \
  tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Result: **34 passed, 0 failed, 49.339 s**.

### Red

After writing `tests/test_weights_init.py` and before creating the helper:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v tests.test_weights_init
```

Result: **10 test methods attempted; 15 expected errors**, all caused by `ModuleNotFoundError: No module named 'transgenic.training.weights_init'` (subtests account for the larger error count).

A later added tied-storage test was also observed red before its guard was implemented: it reached alias-aware loading and failed with the wrong missing-key error rather than the required preflight `tied storage` refusal. The poisoned-target retry assertion likewise failed before the one-way poison guard was added.

### Focused green

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v tests.test_weights_init
```

Result: **11 passed, 0 failed, 0.534 s**.

Coverage includes production-like normalized embedded recipes with tiny consistent widths; directory and exact-file input; `best` symlink; stale config-path non-use; aliases and all shared parameters; zero projection; matching and absent optional metadata; semantic drift; malformed normalized recipe; source-8 refusal; target grads; missing/unexpected/projected/wrong-shape/corrupt weights; dangling and escaping links; wrong filename; forbidden-state sentinels; RNG and optimizer-state preservation; independently computed canonical/file/manifest hashes; source identity mismatch; and the failure-poison contract.

### Regression green

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Result: **45 passed, 0 failed, 51.016 s**.

Syntax check:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m py_compile \
  src/transgenic/training/weights_init.py tests/test_weights_init.py
```

Result: exit 0, no output.

## Round 1 scoped review fix

- Review-fix branch: `b6-mvp-weights-init-review1`
- Review base: `c760c26ef8d02ef9521c267dcb45130a9a338e32`

Review found that the initial implementation tracked only checkpoint-root `runtime.json`, while the production B5 layout records runtime provenance in `runtime_identity.txt` beside the parent `run_config.json`. Based on the read-only production inventory in `notes/b6_mvp_runtime_preflight_20260915.md`, this omitted the actual available source-runtime record.

The fix adds optional `run_root/runtime_identity.txt` to the same source-file audit map as weights and `run_config.json`. It is resolved as a regular file at the canonical run root, SHA256/identity snapshotted before and after loading, emitted as explicit `null` when absent, and causes a poisoned-target failure if it changes during loading. Checkpoint-root `runtime.json` remains supported. The text is treated as opaque provenance: it is not JSON-parsed, and hashing the record does not independently certify the SIF bytes named inside it. Its audit record remains under source `files`, separate from hashes of the current loader implementations under `implementation`.

Test-first red command:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v tests.test_weights_init
```

Result before the source fix: **12 methods run; 1 failure and 2 errors**. The real-layout test and absent-value assertion raised `KeyError: 'runtime_identity.txt'`; the injected runtime-file change was not detected.

Focused green with the same command: **12 passed, 0 failed, 0.633 s**.

Round 1 regression command:

```text
PYTHONPATH=/tmp/b6-dataset-verify.D2Is9a/.testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest -v \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
```

Result: **46 passed, 0 failed, 53.174 s** (the original 45 tests plus the runtime-identity change test).

Covering tests:

- `test_success_from_best_is_alias_safe_zeroes_projection_and_audits_hashes`: absent run-root runtime is explicit `null`.
- `test_exact_file_and_matching_optional_metadata_succeed`: production-shaped `run/epoch_04` layout tracks checkpoint `runtime.json` and run-root `runtime_identity.txt` independently, including an independently computed digest.
- `test_run_root_runtime_identity_change_during_load_fails`: a real text-file byte change during alias-aware load trips the common post-consumption identity guard and leaves the target poisoned.

## Limitations and required follow-up

- This is deliberately **0 -> 8 only**. Same-8 learned-evidence initialization/resume remains unsupported because the B5 mapping does not establish evidence-checkpoint provenance.
- This helper is weights initialization, not resume. It does not restore optimizer, scheduler, gradients, counters, pending batches, or RNG state.
- There is no trainer or GPU integration in this increment.
- The tests use real tiny TransGenic wrappers and real safetensors alias save/load while substituting only the external Hyena factories. They do not certify the production checkpoint itself.
- Reconstructed config defaults are current-loader evidence, not historical attestation. Production certification must reconcile recorded source implementation/configuration provenance before a GPU run.
- Sharded checkpoints, alternate weights names, remote model/config loading, and learned 8-channel source projections are intentionally rejected.
- A failed target is not rolled back. It is explicitly poisoned and must be discarded.
