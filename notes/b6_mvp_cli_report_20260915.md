# B6 exploratory MVP CLI integration report (2026-09-15, final integrated-review fix)

## Scope

- Final fix base: `1c47ae59deebedca18aaae9cbfa3c77e64a91769` (`Separate MVP runtime RNG domains`).
- Native isolated worktree; CPU only. No main-worktree writes, remote actions, GPU actions, job submission, merge, or push.
- Earlier review findings remain addressed. This final narrow wave adds only the measured public hardware assertion, metadata schema/reporting, and focused tests.
- The public CLI remains fixed to bf16 plus Inductor, exploratory-only, and `EXPLORATORY_MVP_NOT_FOR_PRODUCTION`. No public test, eager, precision, budget, production, or recovery option was added.
- Missing real launch readiness still blocks public execution before Accelerator/model/dataset construction.

## Round-2 findings and changes

### 1. Exact evidence split population

The candidate inventory now requires `counts.evidence_corpus_tile_all_splits` to have exactly `train`, `valid`, and `test`. Every value must be a nonnegative integer with booleans rejected. Train and valid counts still reconcile to the ordered candidate records.

Artifact verification now queries `SELECT split, count(*) FROM corpus_tile GROUP BY split` without filtering to known values. Unknown and `NULL` splits are rejected before count comparison. Missing known groups normalize to zero, and the complete measured map must equal the companion map. Unique/disjoint test RN checks remain. Tests modify DuckDB bytes and update the accepted digest, proving that semantic validation rejects unknown/NULL splits, an extra known-split row, and a fabricated test count after the hash gate passes.

### 2. No nested identity collectives

Identity work is now split into two operations. `_identity_payload` performs local JSON encoding and validation only, under the bounded collective phase-status wrapper. After every rank has reported local success, `require_matching_identity` performs the single identity gather directly, without an outer status collective around it.

All other guarded phase operations were checked and remain rank-local: config load, model/weights initialization, final readiness audit, optimizer/compile setup, and dataset/loader setup. A real two-rank injection fails identity preparation on one rank; both ranks reject within the launcher timeout, clean up, and produce no artifact.

### 3. Safe mutation normalization before final recheck

`evaluate_and_publish_final` now accepts a second optional internal callback, `prepublication_prepare`, immediately before `prepublication_check`. Each callback is independently wrapped by the finalizer's bounded collective preflight after evaluation aggregation and before publication filesystem work.

The private rank-zero mutation hook runs only as local work in `prepublication_prepare`; it contains no barrier or collective. Its success or exception is normalized collectively before any rank enters the final recheck. The normal recheck callback now contains only local close/stat/hash/query/readiness work. Real two-rank tests cover both shared-file mutation and a rank-zero mutation-hook exception; all ranks reject and no staging/final artifact is published.

### 4. Truthful runtime metadata

Runtime metadata is derived from the path actually selected by `_train`:

- Public execution records `execution_mode=public_cli`, `compile_backend=inductor`, and `precision=bf16`.
- Private CPU integration records `execution_mode=private_test_injection`, `compile_backend=eager_counting_test`, and `precision=no`.

Both record nonreentrant checkpointing and the actual `weights_optimizer_compile_prepare` order. The two-rank final metadata assertion verifies the private conditions and therefore makes no bf16 or Inductor certification claim. A focused unit test fixes the unchanged public contract.

### 5. Production-path runtime observations restored

The real two-rank probe still calls production `execute`, which calls production `_train`; it does not reconstruct the helper chain. `_TestExecution` carries only a private in-memory observation dictionary. `_train` records the canonical augmentation seed, sampler seed, exact private augmentation seed, exact train/valid DataLoader generator seeds, and generator object identities used on each rank, then records the actual summary returned by `run_fixed_updates`.

The published final metadata and per-rank observations are asserted to show:

- exactly **2 successful updates** for the test recipe's **2-update budget**;
- **3 crossed epochs**, proving the shared runtime crossed loader epochs;
- effective batch size **4**;
- distinct private, train-worker, and valid-worker streams on both ranks;
- test identity scope, CPU/no precision, eager counting backend, clean saved keys, marker publication, and world size 2.

These assertions are limited to the injected CPU test recipe and external tiny Hyena/tokenizer substitutions. They are not claims about public 250-update, four-rank, bf16, Inductor, GPU, or production behavior.

## Round-3 RNG domain separation

The canonical molecule-ranking seed remains `aug_seed = 20260914 + seed`. The distributed sampler remains independently seeded with the run `seed`. Runtime streams now use these exact formulas:

- private dataset dropout/thinning generator when `num_workers=0`: `aug_seed + rank`;
- train DataLoader worker generator: `aug_seed + 1_000_000 + rank`;
- validation DataLoader worker generator: `aug_seed + 2_000_000 + rank`.

Train and validation each receive a separately constructed `torch.Generator`. The private generator is also a distinct object when present. The nonzero-worker dataset RNG implementation is unchanged. This resolves the prior rank-zero collision between the private and train-worker seeds without changing the canonical molecule-ranking seed, sampler permutation, fixed recipe, or evidence algorithm.

The two-rank production-path probe asserts the complete seed map for each rank, including canonical and sampler seeds, not merely cross-rank inequality. For seed 1 it requires rank 0 `(private, train, valid) = (20260915, 21260915, 22260915)` and rank 1 `(20260916, 21260916, 22260916)`. It also verifies pairwise-distinct stream seeds and generator identities on each rank. A rank-distinct but incorrectly offset implementation therefore fails.

## Final integrated-review hardware assertion

Immediately after `Accelerator` construction, inside the existing cleanup boundary and before model construction, every rank measures its actual Accelerator/Torch topology. Local probing and canonical serialization run through the bounded collective preflight; one fixed 16 KiB frame per rank is then gathered and decoded on every rank.

Public execution now fails closed unless the measured topology is exactly the current single-node plan: four `MULTI_GPU` ranks, rank/local-rank/Accelerator device/current CUDA device mapping `0..3`, all four CUDA devices visible on each rank, native (non-emulated) bf16 support, and four distinct non-MIG device UUIDs. Missing UUID or native-bf16 APIs are errors; CUDA indices are not treated as physical identities. The gate runs before model, optimizer, compile, dataset, or training construction. Missing launch readiness still stops earlier.

Published `runtime_hardware` metadata contains the gathered per-rank observations: topology, CUDA availability/count, device name and UUID, capability, native-bf16 result, Torch version, Torch CUDA build, and cuDNN version. These fields are labelled `runtime_observation`, not readiness or certification. The finalizer requires the schema and reruns the public topology validator for production-scope metadata.

The private integration seam remains forced CPU. It records `test_hardware=true`, `MULTI_CPU`, CPU rank facts, and no GPU identity; it does not expose a public bypass. Synthetic four-GPU facts test only gate logic and are not GPU certification. Focused tests cover CPU, wrong world size, duplicate/missing UUID, unavailable UUID probe, false bf16 support, and bad rank mapping before any model path.

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

- CLI suite: **45 passed**.
- Finalizer suite: **42 passed**.
- Fixed-runtime suite: **20 passed**.
- Required reviewed baseline: **46 passed**.
- Total requested tests: **153 passed**.
- Syntax compilation and `git diff --check`: exit 0.

## Remaining boundaries

- No real launch-readiness record exists, so public training remains blocked.
- No production corpus, real tokenizer/Hyena checkpoint, SIF execution, GPU, bf16 numerics, Inductor kernels, four-rank GPU topology, resource behavior, or production filesystem was exercised. The new gate is a runtime assertion, not a GPU-readiness receipt or permission to launch.
- Trusted readiness evidence is validated, not re-executed. Historical receipt paths remain digest-bound provenance and are not freshly reread.
- Phase collectives coordinate ordinary rank-local Python failures only. Rank death, communicator failure, and DDP prepare/forward failures still rely on launcher timeout/termination.
- Canonical paths, hashes, and before/after snapshots narrow TOCTOU exposure but do not guarantee immutability against adversarial in-place change-and-restore during opaque reads; immutable storage is required for that guarantee.
- `identity_scope="production"` remains only the legacy fixed-250x96 finalizer shape name. Authorization remains exploratory and production false.
