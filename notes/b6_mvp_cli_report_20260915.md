# B6 exploratory MVP CLI integration report (2026-09-15, review round 1)

## Scope

- Review base: `c7b016a7b17d8cfd5dbc576bb86c5a3eace4014e` (`Add exploratory MVP training CLI`).
- Native isolated worktree; CPU only. No main-worktree writes, remote actions, GPU actions, job submission, merge, or push.
- The entrypoint remains exploratory and always emits `EXPLORATORY_MVP_NOT_FOR_PRODUCTION`. No resume, arbitrary budget/LR/world-size, eager fallback, production authorization, or test-only public flag was added.
- Missing real launch readiness still blocks actual public execution before Accelerator/model/dataset construction. `--validate-only` remains available for the technically accepted, production-false candidate.

## Review findings and changes

### 1. Canonical consumption and final identity checks (HIGH)

`execute` now resolves and verifies every consumed path once and passes a frozen internal `_VerifiedInputs` binding into `_train`. `_train` no longer reopens `args.config`, `args.init_from`, `args.db`, `args.evidence_db`, `args.sif`, or the readiness path. Config, initialization, and dataset-open boundaries compare device/inode/size/mtime/ctime snapshots before and after access. The loaded config bytes are rehashed against readiness before optimizer construction.

The actual `load_weights_only` audit is compared before optimizer construction with readiness's concrete resolved checkpoint/weights/run-config paths and both source SHA256 values. Symlink file aliases are rejected at the regular-file gate; a checkpoint directory alias is resolved and the resulting audit records the canonical source.

`evaluate_and_publish_final` has one optional keyword-only `prepublication_check`. The default is unchanged. When supplied, it runs collectively after evaluation aggregation and before staging-directory creation or any publication filesystem work. The CLI uses it to close dataset read handles, repeat canonical stat/WAL checks, rehash/requery candidate DB identities, and revalidate readiness, config, SIF, source checkpoint, and exact code manifest. Evaluation-time mutation therefore rejects all ranks with no final artifact.

This is a canonical-binding and phase-check contract, not proof against an adversary changing and restoring bytes during an opaque read while preserving observable metadata. Preventing that requires immutable/snapshot storage.

### 2. Coordinated setup failure and cleanup (HIGH)

The existing bounded collective status transport now guards fixed noncollective phases: rank identity, config load, model/weights initialization, final readiness/audit, optimizer plus bound-forward compile, and dataset/loader construction. No rank enters `accelerator.prepare` until optimizer/compile setup has succeeded on every rank, and no rank enters runtime collectives until dataset setup has succeeded on every rank.

`_train` now always releases both dataset connection sets when constructed and calls `accelerator.end_training()` in `finally`. A real CPU/Gloo two-rank test injects a dataset setup failure on one rank and verifies all ranks reject, the corpus can be reopened read-write, and no output artifact exists.

This does not claim recovery from arbitrary rank death, communicator failure, or failure during DDP forward/prepare. The external launcher timeout remains the boundary for those cases.

### 3. Production execute/_train E2E (MEDIUM)

`tests/mvp_cli_probe.py` no longer reconstructs the helper chain. It calls production `execute`, which calls production `_train`, using a private `_TestExecution` object that is unavailable from the public parser. The only substitutions are the small test recipe/test identity, CPU mode, a counting eager compile backend, external tiny Hyena factory, and external tokenizer.

The two-rank success path exercises readiness-to-runtime handoff, canonical checkpoint alias resolution, real tiny DuckDB corpus/evidence metadata and dataset/rasterizer/collate, actual TransGenic wrappers, actual `load_weights_only`, optimizer/runtime accumulation across epochs, final evaluation/aggregation, atomic finalization, test-scope metadata, clean safetensors keys, final identity checks, and cleanup. It makes no bf16, Inductor, GPU, or production claim.

Separate real two-rank cases cover one-rank setup failure and post-evaluation/prepublication mutation, both with all-rank rejection and no artifact.

### 4. Closed schemas and deterministic code manifest (MEDIUM)

The technical candidate now requires the exact 12-key `checks` set from the neutral record, with every value exactly `true`. Receipt rows require exact fields, integer byte counts (not booleans), and a string location from `local_git_object`, `local_parent_repo`, or `pronghorn`.

Runtime metadata queries now reconcile the inventory's test count with unique, disjoint `corpus_tile` test RNs. A fabricated `test=999999` count is rejected.

The consumed-code manifest now has exactly two sections. `files` must equal every `src/transgenic/**/*.py` file plus `train/train_HyenaTransgenic_MVP.py` under the actual imported checkout, with every byte SHA256 matching; missing, extra, or substituted paths are rejected. `external_packages` separately binds exact installed versions for Accelerate, DuckDB, safetensors, Torch, and Transformers. A recomputed manifest containing only `pass.py` is rejected.

## Verification

Interpreter/dependency paths:

```text
Python: /home/wyim/miniconda3/envs/pdb/bin/python
.testdeps -> /tmp/b6-runtime-controller-227c6e3/.testdeps
DuckDB dependency path: /tmp/b6-dataset-verify.D2Is9a/.testdeps
```

Commands:

```sh
export PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps"
PY=/home/wyim/miniconda3/envs/pdb/bin/python

$PY -m pytest -q tests/test_mvp_cli.py tests/test_mvp_finalize.py
$PY -m pytest -q tests/test_mvp_runtime.py
$PY -m unittest -q \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset
$PY -m py_compile \
  src/transgenic/training/mvp_cli.py \
  src/transgenic/training/mvp_cli_runtime.py \
  src/transgenic/training/mvp_finalize.py \
  train/train_HyenaTransgenic_MVP.py \
  tests/test_mvp_cli.py tests/mvp_cli_probe.py tests/test_mvp_finalize.py
git diff --check
```

Results:

- CLI suite: **28 passed**.
- Finalizer suite: **41 passed**.
- Fixed-runtime suite: **20 passed**.
- Required reviewed baseline: **46 passed**.
- Total requested tests: **135 passed**.
- Syntax compilation and `git diff --check`: exit 0.

## Remaining boundaries

- No real launch-readiness record exists, so public training remains blocked.
- No production corpus, real tokenizer/Hyena checkpoint, SIF execution, GPU, bf16 numerics, Inductor kernels, four-rank resource behavior, or production filesystem was exercised.
- Trusted readiness evidence is validated, not re-executed. Historical receipt paths remain digest-bound provenance and are not freshly reread.
- Phase collectives coordinate ordinary rank-local Python failures only. Rank death and communicator/DDP failures still rely on launcher timeout/termination.
- Canonical paths, hashes, and before/after snapshots narrow TOCTOU exposure but do not guarantee immutability against adversarial in-place change-and-restore during opaque reads; immutable storage is required for that guarantee.
- `identity_scope="production"` remains only the legacy fixed-250x96 finalizer shape name. Authorization metadata remains `exploratory_mvp`, `production_authorized=false`, and `EXPLORATORY_MVP_NOT_FOR_PRODUCTION`.
