# Real CUDA probe implementation — partial, NOT launch-ready

## Independent-review fix round 1 (supersedes historical wiring claims below)

Fix base: `3b62306acfa989e3bcf2a15b6506068980a9c3fe`.
Implementation: **`981489daa2ddbb040ab0056299655899dd2b28b2`**.
Read the complete controller review
`notes/mvp_execution_20260916/real_probes.independent_review.md` in main read-only.
Only the existing isolated A worktree was modified. No SSH/jobs/subagents,
other-worktree edits, merge, push or main writes.

### R1: actual training and probe decoder inputs

- `src/transgenic/training/mvp_cli_runtime.py`: both train and valid constructors
  now pass the already-bound `cfg['data']['gff_vocab_version']`. Shared
  `validate_decoder_dataset` requires agreement of dataset vocabulary version,
  actual `len(dataset.dt)`, config/model vocabulary size, dataset maxlength,
  config max_label_tokens and config/model decoder positions. Frozen v3 remains
  **290 tokens / 8192 label tokens / 8192 positions**, not v2/2048.
- `real_probes.datasets(args, bindings, cfg, model_config)` receives loaded config
  and actual model config, passes the same decoder version and invokes the same
  guard. It closes the dataset if the guard rejects it.
- Existing real two-rank `execute -> _train -> final` CPU fixture now uses tiny
  compute dimensions but actual v3/290/8192 semantics. An observing constructor
  asserts explicit v3 kwargs and actual tokenizer/cap on **both train and valid
  on every rank**. No production factory is mocked beyond the existing tiny
  external Hyena/DNA tokenizer fixture. This catches future missing-kwarg regressions.
- Real GFFTokenizer/dataset tests cover `<empty>` IDs `[0,289,2]`, two-gene separator
  preservation, 2701-token labels (not truncated at2048), and >8192 labels preserving
  the existing8192/end-token cap. Negative tests reject each version/vocabulary/cap
  inconsistency. No historical dataset defaults or B5 trainer were changed.

### R2: rank-local error boundaries and clean environment

- Compiled-mode main now creates one Accelerator, then uses existing
  `_collective_preflight` around local input setup, real model/loader initialization
  and tier retrieval. Compiler/optimizer configuration and DataLoader construction
  also use local-operation guards. `accelerator.prepare`, optimizer update,
  hardware collection and final gather stay **outside** local-exception wrappers
  because they contain collectives.
- `collective_batches` guards rank-local retrieval of exactly24 batches.
  `one_update` checks `_batch_parts` validity collectively before tensor transfer
  and uses the trainer's collective finite/skip checks. Completion-count agreement
  is also collective. `collective_compiler_check` shares local Inductor-counter/
  suppress-errors failure with every rank **before** `gather_object`.
- Actual two-rank CPU fault programs run under the real bounded runner. Rank1-only
  loader exception and rank1-only zero generated-kernel count both yield the
  original diagnostic on **both ranks**, nonzero launcher exit, FAILED JSON,
  no timeout and **no DONE**. They do not simulate CUDA/NCCL failures; the launcher,
  process-group timeout and outer allocation deadline remain mandatory backstops.
- Both actual MVP runtime and probe reject nonempty `TRANSGENIC_RUN_DIR` before
  model construction, preventing inherited encoder report writes into B5. Explicit
  tests exercise both refusal paths. Probe offline environment is set before
  importing Accelerator/HF modules, including the earlier process-group setup.
- Main owns accelerator cleanup through its outer finally; no collective-containing
  whole function was wrapped as if it were a rank-local operation.

### Exact validation

All commands use `/home/wyim/miniconda3/envs/pdb/bin/python` and
`PYTHONPATH=.testdeps:src` in A.

Initial focused run: **26 passed, 1 failed in73.16s**. The new intended >2048 test
fixture had only1801 real tokens; expanded it from200 to300 gene blocks (2701 tokens).
The implementation and tokenizer/cap policy were not relaxed.

Full scoped command below: **245 passed in644.32s**, no timeout this round:

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
 tests/test_mvp_cli.py tests/test_mvp_finalize.py tests/test_mvp_runtime.py \
 tests/test_weights_init.py tests/test_mvp_metrics.py tests/test_mvp_model.py \
 tests/test_mvp_dataset.py tests/test_mvp_readiness_tools.py \
 tests/test_mvp_readiness_review.py tests/test_mvp_real_probes.py tests/test_mvp_review_fixes.py
```

After the final offline-before-Accelerator ordering adjustment, focused rerun:
**92 passed in112.55s** (all66 existing tooling, all15 earlier probes and11 new fix
cases, including both actual distributed fault tests). Command uses the same
prefix and files `tests/test_mvp_review_fixes.py tests/test_mvp_real_probes.py
 tests/test_mvp_readiness_tools.py tests/test_mvp_readiness_review.py`.
`py_compile` and `git diff --check` passed. Earlier historical fixture timeouts
remain recorded below; no timeout was weakened and no historical root cause is
claimed fixed.

### Controller/resource-track handoff

**Regenerate ALL consumed-code manifests, tooling bindings, prepared records and
probe receipts** after this commit; previous v2-label results cannot certify v3.
Independent controller re-review and real aarch64 SIF P1/P2/P4 remain required.
Torch2.14/CUDA13.4, Transformers4.57.6 generation/cache, Accelerate1.15 parser,
real 400M gradients, Inductor, deterministic GPU kernels, native bf16 and NCCL
termination are still untested. This fix grants no GPU or training authorization.

Resource agent must integrate this fix later (its worktree was not touched).
Internal signatures changed to `datasets(args, bindings, cfg, model_config)` and
`compiled_update(args, bindings, model, ds, accelerator)`; main owns Accelerator.
Reuse `validate_decoder_dataset`, `collective_batches`, and
`collective_compiler_check` as appropriate, never wrap collectives inside a local
phase. Resource mode remains absent in A. No LR,250/12,metrics,RNG,heldout filtering
or training-profile changes; dataset decoder wiring is the explicitly authorized
correctness exception to the original no-trainer-change scope.

---

## Scope and commits

Only writer in `/tmp/b6-p2-readiness-1535112`, base
`ac8ac2d34c3ae403017a4a0d72e7531528926626`.

- `1639c7e6dd716bdf3a9281bd8d9ea1c05c86cd25`: five executable CUDA gate modes,
  actual input/import/cache identity, tooling integration and CPU contracts.
- `b1e5e62992f8618b2aee18e12c6c7f449f8c5121`: nonzero EST generation,
  beam2 expansion/reordering, same-path raw-logit reference and restoration tests.

**Actual GPU/SIF execution is untested. Six-gate readiness is NOT complete.**
`resource_budget` is intentionally not offered as an executable mode and cannot
emit a receipt. No training, evaluation, deployment or budget authorization follows
from this report. No SSH, job, subagent, main write, merge or push occurred.
The pre-existing `.gitignore`, `.ignore`, `.testdeps` state is not committed.
Advisor consultation failed with `claude_code_version_too_old`; no advice received.

## Implemented executable: `revision/scripts/mvp/real_probes.py`

Five modes consume actual immutable inputs and fail before publication on any
exception, CUDA absence, wrong SIF versions, missing asset, unsupported deterministic
operation, nonfinite result, OOM or comparison failure. There is no tiny-model,
CPU, eager, tolerance or random-encoder fallback in the executable.

1. `real_tokenizer`: imports the pinned SIF packages, parses the shipped explicit
   Accelerate configuration using the **installed** launch parser, loads the real
   tokenizer and full 8-channel 400M model, runs a CUDA forward, measures ACGTN
   roundtrip and actual six-tile input alignment/RC/N checks.
2. `real_hyena_step0_three_tiers`: independent evidence0 B5 architecture loaded
   strictly from the selected safetensors; MVP initialized using the existing
   `load_weights_only`, including optional sidecar identity. Full uncropped
   30720/61440/129024-nt forward/RC twins: B5, zero-track MVP, real-EST zero-projection
   MVP and repeat logits must be bitwise identical. Tier fields contain forward
   tensor digests; RC digests are retained separately. Eval, deterministic
   algorithms, TF32 disabled and `:4096:8` workspace are explicit.
   Real CE backprop additionally requires finite **nonzero** projection gradient
   with real EST and exactly zero projection gradient with zero tracks.
3. `generation_cache_parity`: dedicated eager probe model; temporary deterministic
   nonzero diagonal projection on all eight channels; original projection restored
   in `finally`. Real EST must change encoder activations, while zero tracks must
   equal omitted evidence. For evidence off/on × beams1/2 × cache off/on:
   encoder hook must receive exact tracks once; decoder forward hook must observe
   the exact encoder output expanded to the beam batch; beam2/cache must actually
   invoke observable reorder, checked tensor-for-tensor against index-selection.
   Compare sequences and all 16 raw-logit tensors **bitwise against an unwrapped
   eager direct-encoder reference with the same cache/beam setting**. Cached vs
   full-prefix **tokens** must match; no cross-path raw-logit equality/tolerance is
   invented. Token changes due to evidence are not required. This executes actual
   Transformers4.57.6 hooks on deployment; CPU toy tests do not certify that version.
4. `bf16_compile` and 5. `four_rank_smoke`: each performs a real four-rank single-node
   native-bf16/Inductor, nonreentrant-checkpointed optimizer update, with 24 real
   training microbatches per rank. Reuses `make_accelerator`, `build_optimizer`,
   `_batch_parts`, LR, clipping, collective finite/skip checks and bound-forward
   compiler. Uses actual rank-specific augmentation, DistributedSampler and
   batch1 dataset/collator. Requires generated Inductor kernels and disabled
   suppress-errors. Reports each rank's synchronized cold wall time, allocated
   and reserved CUDA peaks and observed microbatch/update counts. This is a
   **probe-local one-update loop**, not a public trainer budget override.

The trainer and approved `MVPRecipe(250 updates, 12 warmup, accumulation24,
effective batch96)` were not edited. No resume API was added.

### Identities and fail-closed boundaries

Prepared bindings are checked against corpus, evidence, config, exact consumed-code
manifest, source weights/run_config/runtime_identity and loader optional sidecars.
The receipt includes the actual probe source and loaded Python/extension import
hashes, immutable offline asset-tree identities, tile-selection digest, versions,
initialization audit digest and prepared binding digest. Runner and later receipt
checks rehash these identities for measurements marked with the real-probe mode.
`tooling_identity()` now includes `real_probes.py`; regenerate old prepared records.

Existing schema-1 trusted-operator measurement compatibility is retained; arbitrary
legacy JSON is not cryptographic proof of execution. A reviewed controller must
require these actual probe receipts rather than handcraft legacy booleans. Import
identity rechecks need the same SIF and container paths; host Python is not equivalent.
No receipt alone authenticates that the pinned SIF is executing; immutable SIF
launch/deployment remains controller responsibility.

## Exact deployment prerequisites and CLI

Read controller's main report `notes/MVP_DEPLOY_OPS_20260916_1545.md` read-only.
Do not install local CPU-test versions into the SIF. Actual executable pins:

```
torch 2.14.0a0+4fdf77b940.nv26.08
CUDA 13.4
transformers 4.57.6
accelerate 1.15.0
```

The actual Accelerate parser must resolve GPU IDs0,1,2,3, MULTI_GPU command,
4 processes, 1 machine, machine-rank0, bf16, DynamoNO, static127.0.0.1:29500,
max-restarts0. Version mismatch fails before expensive DB/cache hashing.

Controller-selected epoch is concrete `seed123/epoch_05`, not `best`:

- weights: `6e5e408681dd5356548a8f4b93d28e561c22d885668a4e0015d3e314e28f8d22`
- parent run_config: `c3ca0b4dfc1c9fbedb0cfba4b42887b22d4a62bc5b0c84be6b711ec630a1ef18`
- parent runtime_identity: `44b6f578ec50d373d01e764d391b2a559ce4d3012c52388fc160f94fef6f887e`
- epoch meta: `844ecbc41f19a9287eb0101f9047b3b0100f88d2bfed1a5057261296d86943d7`
- epoch config.json and runtime.json explicitly absent.

Use existing reviewed `stage_checkpoint.py` and readiness `prepare`; no source
checkpoint selection is embedded in this probe. Existing DeltaAI corpus/SIF remain
read-only; only accepted 1,068,773,376-byte evidence plus small companions need WAN
transfer. Transfer/allocation remains controller-owned.

Run from the immutable code root where `./HFmodels` points to the independent
materialized MVP cache, not B5. Preserve **both** tokenizer `HFmodels/models--...`
and encoder `HFmodels/hub/models--...` layouts, both refs/main exactly
`0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446`.

Set before Python startup, using paths inside the new MVP namespace:

```
HF_HOME=/work/repo/HFmodels
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_MODULES_CACHE=$PROBE_SCRATCH/modules
TORCHINDUCTOR_CACHE_DIR=$PROBE_SCRATCH/inductor
TRITON_CACHE_DIR=$PROBE_SCRATCH/triton
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=/work/repo/src
```

`PROBE_SCRATCH` must already exist, be writable, outside all immutable asset roots
and outside B5. Use a fresh per-mode compiler cache to observe actual compilation;
no compiled-artifact substitution or quiet fallback. Ordinary generation does not
compile; its reference and measured decoder use the same eager path.

Two additional JSON staging inputs, operator-reviewed and SHA256-pinned:

- `assets.json`: `{"roots": {"/work/repo/HFmodels": "<tree sha>",
  "/work/repo/HFmodels/hub": "<tree sha>"}}`. Tree digests are computed **on
  authorized compute** using `real_probes.tree_identity(path)`; this includes every
  file's bytes, relative name and resolved cache-link target. Source roots must be
  immutable. Writable module/compiler trees must NOT be in this manifest; actual
  consumed module files are separately recorded after execution.
- `tiles.json`: `{"30720":[forward_rn,rc_rn],"61440":[forward_rn,rc_rn],
  "129024":[forward_rn,rc_rn]}` with actual integer RNs, selected from accepted
  Athaliana TRAIN inventory. Each pair must have identical genomic interval, explicit
  false/true is_rc, exact DNA reverse-complement relationship, nonzero retained EST
  signal and full stated length. At least one selected tile must contain N. These
  are operator-selected rows, not invented IDs. If such a selection is unavailable,
  STOP: report missing coverage; do not crop/duplicate/fabricate a replacement.

The actual single-process argv for any of the first three modes is:

```
python /work/repo/revision/scripts/mvp/real_probes.py \
  --mode "$MODE" --bindings "$BINDINGS" --bindings-file-sha256 "$BINDINGS_SHA" \
  --assets-manifest "$ASSETS" --assets-sha256 "$ASSETS_SHA" \
  --db "$CORPUS" --evidence-db "$EVIDENCE" --config "$CONFIG" \
  --tile-rns "$TILES" --tile-rns-sha256 "$TILES_SHA" --scratch-root "$PROBE_SCRATCH"
```

For either compiled mode the exact launcher prefix is:

```
python -m accelerate.commands.launch --multi_gpu --num_machines 1 \
  --num_processes 4 --mixed_precision bf16 \
  --config_file /work/repo/revision/scripts/mvp/accelerate_mvp.json --max_restarts 0 \
  /work/repo/revision/scripts/mvp/real_probes.py [same flags as above]
```

Store the fully expanded command as a JSON argv array with an **absolute Python
executable**, not a shell string. Run through the existing bounded runner:

```
python revision/scripts/mvp/run_bounded.py --name "$MODE" \
  --bindings "$BINDINGS" --bindings-file-sha256 "$BINDINGS_SHA" \
  --command-file "$ARGV_JSON" --command-sha256 "$ARGV_SHA" \
  --output-dir "$FRESH_PER_MODE_DIR" --timeout-seconds "$AUTHORIZED_BOUND"
```

Runner supplies MVP_MEASUREMENT_OUTPUT and MVP_BINDINGS_SHA256; no direct manual
success JSON. Each mode is a separate invocation; no automatic retries. The
proposed P1 0.5node-h/P4 2node-h/total5node-h envelope is **not** a training-budget
record and this program does not infer authorization from it.

## Meaningful remaining blocker / smallest next implementation

`resource_budget` cannot honestly be derived from the implemented **cold single
update**. That includes compilation, samples only the shuffled first24 rows per
rank and has no measured paired-validation or publication cost. Extrapolating it
would violate the requested resource contract. The production finalizer also
explicitly requires actual 250-update provenance (`mvp_finalize.py:211-270`);
calling it after one update with a made-up summary would be false certification.

Smallest next change is a **probe-only resource mode**, without modifying the
public trainer/finalizer gates:

1. Execute separate warmup and at least two steady accumulation24 updates per
   each of the three full tiers, using real evidence/labels and actual rank
   sampling; record per-rank wall/peak CUDA and label-length coverage. Reuse the
   existing primitive loop; never project from one short tile.
2. Run real paired fraction0/1 validation over the accepted complete valid
   inventory using `paired_metric_record` and aggregation. Measure read/transfer,
   decoder work and reduction separately.
3. Benchmark actual low-level serialization, fsync and hashing on fresh
   **probe-labelled disposable artifacts** with truthful one-update metadata;
   never invoke production finalization with fabricated250 or publish TRAINING_DONE.
   Measure independent input-revalidation overhead too.
4. Read an explicit controller-signed/offline-authorized budget JSON + pinned
   digest. Conservatively project250 using worst per-rank/tier steady update plus
   measured warmup, full paired eval, publication and setup overhead with a
   prospectively stated safety margin. Require every stage, memory headroom and
   total wall fit. Only then emit resource_budget/authorization_sha256.

The model/compiled gates may themselves fail on actual4.57.6/2.14 (especially cache
API or unsupported determinism); these are intended negative results, not reasons
to relax policy. Controller independent review must precede even bounded execution.
Three-seed training and 50-locus evaluation remain blocked on the missing resource
mode and measured receipts. No phase/C1 EST dependency or training-filter relaxation
was introduced; heldout evaluation remains `population='excluded_heldout', rn=None`.

## CPU verification (not GPU certification)

Interpreter `/home/wyim/miniconda3/envs/pdb/bin/python`, existing isolated
`PYTHONPATH=.testdeps:src`, worktree above.

- Test-first run of new suite: collection failed because real_probes.py did not
  yet exist. No purported GPU fixture was written.
- Initial focused tooling+model/init/runtime/dataset run: **127 passed in271.82s**.
- Full scoped regression before generation expansion: **231 passed in505.20s**.
- Subsequent full scoped regression after generation expansion: **233 passed,
  1 failed in637.13s**. Failure was unchanged two-rank hardware-failure fixture:
  rank1's `tests/mvp_cli_probe.py:179-183` 20-second READY wait expired during
  fixture setup. No runtime assertion failed; root cause beyond that wait is
  unestablished. No timeout/test/trainer was changed to hide it.
- Final focused rerun including that exact failing test, all66 existing tooling
  tests, and all15 new probe contracts: **82 passed in66.85s**.
- `py_compile` and `git diff --check` passed before the generation commit.

Full command used (latest full run had the one fixture timeout just described):

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
 tests/test_mvp_cli.py tests/test_mvp_finalize.py tests/test_mvp_runtime.py \
 tests/test_weights_init.py tests/test_mvp_metrics.py tests/test_mvp_model.py \
 tests/test_mvp_dataset.py tests/test_mvp_readiness_tools.py \
 tests/test_mvp_readiness_review.py tests/test_mvp_real_probes.py
```

Final focused command replaces that file list with:
`tests/test_mvp_cli.py::test_actual_two_rank_single_rank_hardware_probe_failure_rejects_before_model
 tests/test_mvp_real_probes.py tests/test_mvp_readiness_tools.py tests/test_mvp_readiness_review.py`.

The new CPU tests cover CUDA/native-bf16 refusal, wrong actual-version rejection,
bitwise signed-zero/nonfinite negatives, dtype/shape digests, import/asset mutation,
real-receipt mode/identity checks, N masking, and resource-mode refusal. Generation
control-flow tests explicitly substitute TinyHyena under local Transformers: they
check nonzero activation changes, beam2/cache observer logic, same-path raw-logit
identity, missing-evidence failure and projection/hook restoration on failure.
They are NOT real-Hyena, actual4.57.6, CUDA, Inductor or resource-budget certificates.
