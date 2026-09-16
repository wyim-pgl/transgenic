# B6 exploratory MVP CLI integration report (2026-09-15)

## Scope

- Branch: `feat/b6-mvp-cli-20260915`
- Base: `0a8120b605d67dd46720e823118e36831a1f6a3b`
- Native isolated worktree; CPU only. No main-worktree writes, remote actions, GPU actions, job submission, merge, or push.
- Neutral requirements and the technical candidate/inventory/receipt records were read from `/tmp/b6-mvp-cli-context/`; they were not modified or copied into the commit.
- The entrypoint is exploratory and always emits `EXPLORATORY_MVP_NOT_FOR_PRODUCTION`. It does not expose resume, arbitrary budget/LR/world-size, eager fallback, or production authorization options.

## Implementation

`train/train_HyenaTransgenic_MVP.py` is a lazy entrypoint. Its public arguments are the exact file bindings, seed choice `{1,2,3}`, worker count, and `--validate-only`. Help imports neither Torch nor model/data modules.

`src/transgenic/training/mvp_cli.py` validates the schema-1 `exploratory_mvp_candidate` acceptance record and its byte-pinned inventory and receipt companions. Technical acceptance must be true while production training acceptance, production authorization, author protocol freeze, and scientific approval remain false. QC, count reconciliations, ordered inventory identities, companion digests, and receipt shapes fail closed.

`src/transgenic/training/mvp_cli_runtime.py` then:

1. requires regular files, rejects DuckDB WAL siblings, hashes candidate artifacts with before/after file identity checks, and compares the accepted Athaliana train/valid inventory to read-only `geneList` and `corpus_tile` metadata queries;
2. permits candidate diagnostics without launch readiness only under `--validate-only`;
3. requires real execution to have a schema-1 trusted-operator readiness record for `exploratory_mvp`, with production false, all six named evidence receipts passing, exact candidate/data/SIF/config/source/code-manifest digests and concrete source paths, source seed 123, target seed, and exact `MVPRecipe()`;
4. validates a fresh output name without creating a marker, then constructs the reviewed B5 config/model, runs the audited 0-to-8-channel `load_weights_only`, enables nonreentrant checkpointing, builds the reviewed two-group optimizer, compiles only the bound `forward` with Inductor and `suppress_errors=False`, and prepares only model and optimizer;
5. constructs strict Athaliana K=8 train/valid datasets and unprepared rank-matched samplers/loaders. Worker generators are separate and rank-distinct. For `num_workers=0`, the private dropout/thinning generator is explicitly initialized with `aug_seed + rank` while the documented `aug_seed=20260914+seed` passed to evidence molecule ranking remains unchanged;
6. runs `run_fixed_updates` and `evaluate_and_publish_final` on the same prepared compiled model. Metadata retains finalizer `identity_scope="production"` only for the fixed 250x96 identity shape and separately records `authorization_scope="exploratory_mvp"`, `production_authorized=false`, and the not-for-production output label;
7. rehashes candidate, SIF, config, source, and consumed-code bindings immediately before publication. No resume or optimizer/state restoration exists.

## Verification

Interpreter and dependency versions:

```text
Python 3.12.2
Torch 2.6.0+cu124 (CPU tests only)
Transformers 4.46.3
Accelerate 0.34.2
safetensors 0.5.2
DuckDB 1.5.5
```

Commands:

```sh
export PYTHONPATH="$PWD/src:$PWD/.testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps"
PY=/home/wyim/miniconda3/envs/pdb/bin/python

$PY -m pytest -q tests/test_mvp_cli.py
$PY -m pytest -q tests/test_mvp_finalize.py
$PY -m pytest -q tests/test_mvp_runtime.py
$PY -m unittest -q \
  tests.test_weights_init tests.test_mvp_metrics tests.test_mvp_model tests.test_mvp_dataset

$PY -m py_compile \
  src/transgenic/training/mvp_cli.py \
  src/transgenic/training/mvp_cli_runtime.py \
  train/train_HyenaTransgenic_MVP.py \
  tests/test_mvp_cli.py tests/mvp_cli_probe.py
git diff --check
```

Results:

- New CLI suite: **21 passed**. It includes a strict-timeout actual two-rank CPU/Gloo integration using tiny DuckDB corpus/evidence databases, the real dataset/rasterizer/collate path with only the external tokenizer replaced, actual TransGenic wrappers with only external Hyena factories replaced, real audited weights initialization, nonreentrant checkpointing, cross-epoch accumulation, bound-forward counting eager backend, reviewed runtime/finalizer, rank-distinct augmentation streams, rank-asymmetric identity rejection before artifacts, canonical audit forwarding, expected RN deduplication, and atomic publication with clean safetensors keys.
- Reviewed finalizer suite: **39 passed**.
- Reviewed fixed-runtime suite: **20 passed**.
- Required reviewed baseline: **46 passed** (`weights_init`, metrics, model, dataset).
- Syntax compilation and `git diff --check`: exit 0 with no output.

The eager counting backend in the integration test proves bound-forward wiring and fatal backend errors; it is not an Inductor or GPU claim.

## Boundaries and limitations

- No launch-readiness record was supplied in the neutral context. Therefore real training is deliberately blocked before Accelerator/model/dataset construction. `--validate-only` can accept the technically accepted candidate while preserving all false production flags.
- No production corpus hashes, GPU, bf16 numerics, Inductor kernels, real tokenizer/Hyena checkpoint, SIF execution, shared production filesystem, or four-rank resource behavior was exercised locally. Those remain bound to the absent trusted readiness record; this increment does not fabricate or rerun them.
- No external job creation/submission, resume, bootstrap, GO/STOP policy, public budget override, eager fallback, or production mode was added.
- The receipts companion is digest-bound provenance. The CLI intentionally does not reread or imply fresh hashing of all historical remote receipt paths.
- Coordinated identity disagreement is rejected collectively after Accelerator creation and before model construction. Rank death, communicator failure, and failure before a process joins collectives remain launcher-timeout concerns.
- `identity_scope="production"` remains the legacy finalizer name for fixed recipe/seed shape. It is explicitly not treated as scientific or production authorization.
