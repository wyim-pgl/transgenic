# Independent integrated decoder review — 2026-09-16

## Verdict and scope

Reviewed detached **e6950b7** against **f4f1690**, all seven changed paths, including both new helpers and their callers, the committed integration report, previous D1–D3 review, final panel contract/review, and the actual fixed-A probe and bounded-runner sources obtained with `git show d99b1f8:<path>`. No source changes, SSH, jobs, subagents, merge, push, or full-suite duplication.

- **Specification: substantially implemented, runtime integration incomplete.** COMPLETE-panel consumption, 350-row transport, D1 and D2 are implemented. D3 has substantial enforcement but is not closed.
- **Correctness: changes required before operational acceptance.** The runtime gate does not bind current local model implementation to the implementation actually probed. There are also concrete receipt/remote-module compatibility rejects, distinct from false approval.
- **Scientific status unchanged:** exploratory/MVP only; no real CUDA/SIF, reference-data, GFFCompare, structural, coverage/withholding, or scientific approval is established.

## Findings

### I1 — High: current project model code is exempted from preflight identity matching (D3 remains open)

`revision/scripts/mvp/mvp_decode_runtime.py:33–49`, particularly `:45–48`, compares external imports against `probe_identity.imports`, but every file resolving under the current ROOT is merely recorded. `:82–83` rehashes the receipt's declared paths, not their correspondence to the current local modules. Thus a valid receipt from another still-accessible checkout can bind the old model implementation while the decoder loads a different current `modeling_HyenaTransgenic.py`. Both sets of bytes can remain stable and pass these checks. The loaded remote Hyena checks at `:99–115` do not bind the project-owned generation/cache/encoder wrapper.

The producer does not repair this: `mvp_produce_counts.py:99–108` compares only weights_init, b5_runtime and configuration implementation to historical training code, not the current model implementation. Runner `mvp_decode_panel.py:106–120,162–171` records/rechecks current code, which proves stability, not correspondence to successful probe execution.

**Focused helper reproduction:** the unchanged `verify_loaded_modules` function accepted the real current local model file despite a contradictory receipt hash for that path. This helper-only result is not an end-to-end acceptance claim: the earlier declared-import rehash would reject that same-path contradiction. The practical unblocked case is a receipt referencing a *different checkout*, whose original file still hashes correctly; the current local file is absent from that receipt and is exempted. No source mutation was needed for the reproduction.

**Required:** compare relevant project model/config/tokenizer/generation implementation identities to the actual fixed-A probe's consumed identities, with an explicit relative-path mapping for approved relocation. New decoder-only orchestration files need their own audit, not a requirement that A previously imported them. Also bind the intended approved probe implementation/handshake rather than treating any existing `probe_path` member as sufficient. Keep the operator-pinned receipt digest; it is not a substitute for checking applicability to this code.

### I2 — Medium: actual fixed-A completed receipt is not the object accepted by this gate

`mvp_decode_runtime.py:63–79` reads top-level `mode/device/real_runtime/runtime/probe_identity`. Those are the **raw measurement** fields produced by fixed-A `real_probes.py:660–666` at d99b1f8. The actual bounded-runner publication is different: `run_bounded.py:164–166,187–199` writes `generation_cache_parity.json` containing `schema_version`, `preflight`, `bindings_sha256`, execution status, `pass`, and nested **`measurement`**, then a digest-bearing `.DONE` marker.

Passing that official receipt directly necessarily raises `ValueError: genuine generation preflight required`, before any CUDA/version check. Independently reproduced with a minimal object in the actual wrapper shape using the real decoder verifier. This is a deterministic operational rejection, **not a false PASS** and not a claim that a GPU receipt was generated here.

Pointing the request at `measurement.json` instead is not equivalent to consuming the completed bounded receipt: decoder never checks wrapper `pass`, exit/timeout/signal, DONE digest, or wrapper-to-measurement binding. Fixed-A emits measurement before the runner's completion validation. The current public documentation calls this input a receipt without resolving that distinction.

**Required:** agree on and test the actual completed-receipt interface. Prefer verifying the pinned wrapper and DONE, checking successful bounded execution and binding agreement, then validating its nested measurement. If raw measurement is intentionally the contract, explicitly designate it and require the controller's separately verified completion evidence; do not claim the decoder enforces that completion today. The checked raw field names and canonical asset-tree algorithm *do* match d99b1f8; this is not a speculative assumption about 3b62306.

### I3 — Medium: generated HF package initializers can fail the remote-source allowlist

`mvp_decode_runtime.py:42–44,94–98` requires **every** loaded `transformers_modules.*` file's hash to occur among snapshot `*.py` files. Hugging Face's dynamic loader also creates/imports empty package `__init__.py` files in its module cache. These are scaffolding, not downloaded model source. Unless the immutable snapshots happen to contain a Python file with the empty-file digest, the first post-load runtime check rejects an otherwise authentic module closure, even when the generated initializer is explicitly present in the genuine probe's import inventory.

Focused unchanged-helper reproduction: an empty initializer with its exact digest in receipt imports and a nonempty snapshot-source digest was rejected with `loaded remote-code asset mismatch`. Local installed Transformers 4.46.3 `create_dynamic_module`/`init_hf_modules` explicitly create these empty initializers. The pinned 4.57.6 snapshot/runtime was unavailable, so **the presence/absence of the empty digest in actual deployment assets remains to be checked**; this is a demonstrated predicate incompatibility, not an asserted observed SIF failure.

Additionally the bare `transformers_modules` package (no trailing dot) takes the external exact-path branch, so moving to another module-cache root is not generally supported even if remote class bytes match. All receipt-declared old cache files must remain readable at `:82–83`. Treat same-path scratch retention as an explicit operational prerequisite until relocation is reviewed.

**Required:** distinguish verified HF-generated package scaffolding from actual remote implementation source, preserve strict source/dependency verification, and test a real dynamic import with the pinned loader. Do not blanket-allow arbitrary remote modules or add placeholder files to immutable assets simply to satisfy the hash predicate.

## D1–D3 closure and positive integration assessment

### D1 — closed in this implementation

`mvp_verify_panel_inputs.py:69–75` checks finite/nonnegative values across all eight channels, N/n zero masking and availability, but caps only `array[:,:3]`. Junction channels are not clipped. It compares published bytes to actual producer/rasterizer output. `tests/test_mvp_decoder_integration.py:57–66,90–111` creates 70 real DuckDB molecules, calls the actual rasterizer and adapter, and verifies junction `float16(log1p(70))` versus coverage `float16(log1p(64))`. I independently ran that focused regression successfully. This is stronger than a handcrafted high-valued tensor.

### D2 — closed for atomic visibility/no-replacement, not filesystem crash durability

`mvp_decode_panel.py:25–42` serializes into a unique sibling temporary, disallows NaN, flushes/fsyncs/closes, invokes the supplied check, then publishes with atomic no-replace `os.link`. Existing targets cannot be replaced. `:162–181` rehashes inputs/audits/predictions, reruns run/runtime gates **after final transport serialization and close**, before linking `evaluation.json`. Existing and dangling-symlink output directories are refused.

Independently ran serialization, flush, fsync, close, final-rehash failure injections and no-replace regression: all passed. The implementation does not claim directory-fsync/power-loss durability. A failure *after* successful linking (for example temporary unlink failure) can leave a complete valid final name alongside a raised exception/FAILED diagnostic; therefore the report's broad phrase “failures cannot leave evaluation.json” should be understood as pre-publication failures, not every possible post-link cleanup error. This is not the former partial-final-file defect.

### D3 — partial, not closed

There are real mandatory gates: actual pinned package versions and CUDA availability, offline flags, pinned SIF bytes plus active container path, actual HF cache roots/ref commit, immutable asset-tree identities, current loaded external file checks, remote class source hashes, loaded config commit and config fields with existing architecture overrides. Receipt, imports, caches and SIF are rechecked repeatedly through final publication. There is no public CPU/unverified override. This substantially improves the former offline-only boundary, but I1–I3 prevent signing off the fixed-A integration.

The SIF environment/path comparison and pinned receipt remain trusted operational evidence, not cryptographic execution attestation. Checking CUDA availability does not force the requested model device to CUDA: runner `:127` defaults to CPU. The documented operator requirement to set CUDA should remain explicit; a GPU-bearing host can still request CPU. Genuine 8192-total-token grammar generation is not established by A's 16-new-token generation probe alone.

### Final COMPLETE panel contract — implemented

`mvp_verify_panel_inputs.py:13–84` validates completion digest, manifest policy/types, actual producer preflight authority and ordered input/code identities plus original request, rasterizer/package identity, exact ordered frozen 50 loci, windows and sequence digests. Every generated FASTA/NPY/JSONL is relative, contained, nonsymlink, unique, hashed and byte-size checked. FASTA bytes must match actual genomic-forward slices. It independently reruns excluded-heldout rasterization for all three seeds and reproduces selected provenance and biological/availability summaries. Biological zero is accepted. No training Dataset/RN/tile, retained-flank, ONT, or HiFi substitution is introduced.

`mvp_prepare_panel_inputs.py:30–80` supplies the independently rerun authority, seed-equality and block/junction provenance predicates, not an internal adapter declaration. Final checks retain COMPLETE, manifest and all consumed files, DB-WAL detection, and panel-root identity. Strict NumPy/DuckDB identity and canonical input paths can reject a differently staged genuine publication; those are intentional operational constraints, not malformed approvals. Git-object reference authority remains a deployment prerequisite inherited from the reviewed producer.

### 350 rows, baseline and producer — consistent

`mvp_decode_panel.py:45–56,129–160` reuses actual candidate/final gates, exact seeds, cross-seed consumed-code/source identity, three checkpoint off/on pairs and only baseline off. `mvp_decode.py:162–201` uses strict alias-aware final loading without re-zeroing learned projection; baseline reloads the verified concrete initialization source with the approved weights-only loader and requires zero projection. Per-locus on tracks are reused across seeds and off is zeros_like across eight channels. Generation still enforces encoder evidence forwarding, beam2/cache/no sampling and total max_length8192, persists raw IDs/plans before conversion, and validates GSF/GFF.

Transport remains compatible with `mvp_produce_counts.py:349–471`: exactly 350 distinct prediction files/windows/checkpoint identities, seed123 initialization baseline, and reference counts derived in the producer. The full integration test reaches actual candidate/finalizer/model-load/adapter/normalization/count-producer code and nonempty plus/minus GFFs. Its genuine learned-projection/source-loader coverage must not be conflated with genuine GPU generation. Historical source selection remains externally pinned as already disclosed by the producer; cross-seed consistency alone is not proof of the intended epoch.

## Independent verification and limitations

- Full seven-path diff read; `git diff --check f4f1690..HEAD` passed. No `.venv` path in the diff. Recovery a4e8808 was not restored or cherry-picked.
- Focused command used the existing Python 3.12.2 environment, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, and `PYTHONPATH=src:revision/scripts/mvp:/tmp/mvp-decoder-resumed-testdeps:/tmp/b6-trackb-testdeps`:
  `python -m pytest -q tests/test_mvp_decoder_integration.py::test_real_panel_high_junction_and_biological_zero tests/test_mvp_decoder_integration.py::test_atomic_final_failure_never_publishes tests/test_mvp_decoder_integration.py::test_atomic_no_replace`
  **7 passed in 20.14 s.**
- Additional temporary read-only-source reproductions exercised I1/I3 predicates, I2 wrapper rejection, and equality of d99b1f8/decoder canonical asset-tree algorithms. Initial invocation under default older Python failed because `Path.is_relative_to` is unavailable; rerun under the stated Python 3.12.2 succeeded. This was a review-environment issue, not a decoder regression.
- Implementer's **107 passed in 427.40 s** is supplied evidence, not independently rerun here. No duplicate full suite. CPU fixtures explicitly replace runtime/GPU generation, remote factories/tokenizer, GFFCompare and frozen-reference authority; they do not certify real frozen inputs or a pinned-SIF successful receipt.
- Advisor consultation failed due to configured client/model incompatibility; no advisor conclusion was used. Graph-first context/caller lookups were used (tool estimate approximately 1,004,301 tokens saved).

## Still required before operational/scientific use

Resolve current-code/probe binding and completed fixed-A receipt handling; run actual pinned-loader closure tests, retaining strict fail-closed checks. Then controller-authorized genuine SIF/CUDA/tokenizer/grammar/cache/8192 smoke, accepted real-reference/evidence and completed three-seed runs, actual 350-output decode, and real GFFCompare 0.12.6 transport smoke remain outstanding. Structural scoring, author coverage/withholding policy and scientific decisions remain separate. No review finding authorizes relaxing those gates.

Only this report is intended for commit.
