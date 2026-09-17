# MVP decoder integration — 2026-09-16

## Scope and status

Isolated branch `mvp-decoder-integration-20260916`, based on `c024445` (reviewed producer fixes already present). Cherry-picked `0f9f777` as `fdc5dd9` and `668299a` as `f4f1690`. No main merge, push, SSH, jobs, resource probes, or subagents. Trainer, model, bootstrap thresholds and duplicate-isoform policy are unchanged.

**CPU integration passed; actual GPU/container/tool execution is NOT certified.** The old decoder report describes the pre-integration implementation and is historical. Its internal batch API and producer-SHA blacklist no longer exist. Actual candidate/reference/final publication verification is used instead; a new source hash is not approval.

## Public interface

Run `revision/scripts/mvp/mvp_decode_panel.py --input REQUEST.json --out NEW_DIRECTORY`.
The request retains the producer's `panels`, `fasta`, `reference`, `candidate`, and three `runs` fields and adds:

- `panel_inputs`: directory published by the unchanged reviewed `mvp_prepare_panel_inputs.py`, not an internal JSON batch.
- `encoder_tokenizer`: `{ "path": "/immutable/cache/.../snapshots/0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446", "files": { "relative_filename": "sha256", "...": "..." } }`. The entire snapshot file inventory must match; escapes are rejected.
- `runtime_preflight`: `{ "receipt": { "path": "/operator/fixed-A-generation-receipt.json", "sha256": "independently-supplied-expected-sha256" }, "sif": { "path": "/actual/container.sif", "sha256": "1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6" } }`.
- `device`: actual CUDA device for operational use (CPU fixtures explicitly replace the runtime gate in tests, never via a public override).

The output `evaluation.json` remains the same 350-row transport: 50 loci × (three final seeds × off/on + seed-123 initialization baseline off). `decode_audit.json` binds inputs, COMPLETE/manifest, raw outputs, code and preflight identities. CLI request bytes are captured and rechecked. This is exploratory/MVP provenance, not a scientific approval receipt.

## D1 and final panel adapter

`mvp_verify_panel_inputs.py` requires COMPLETE's exact manifest digest, schema 1, purpose `est_mvp_panel_inputs_not_training`, training-ineligible/genomic-forward/excluded-heldout/EST policies, all three seeds, exact ordered frozen 50 genes and windows. It reruns the actual producer preflight, independently binding candidate/corpus/evidence/reference/panel/code identities and the producer's original request. Every manifest input and every per-locus FASTA/npy/provenance JSONL has SHA256 and size checked. Relative escapes and generated-artifact symlinks are rejected.

FASTA bytes and sequence digests must equal anchored genomic-forward slices. Arrays are float16 `[30720,8]`, finite and nonnegative in **all eight** channels; N/n masks zero all eight, unmasked availability is one. Only channels **0–2** have the `float16(log1p(64))` ceiling. Junction channels 3–6 are not clipped or capped.

The adapter independently rerasterizes actual excluded-heldout evidence through the reviewed producer/rasterizer for all three seeds, checks equality with the published on-array, and reproduces selected molecule/member/block/junction provenance. Biological support summaries and availability are verified separately; biological-zero windows remain valid. Generation reuses each on-array across seeds and supplies `zeros_like` for every off input. No retained-flank, RN/tile, ONT or HiFi fallback exists.

COMPLETE, manifest, all inputs and consumed output artifacts are freshly rehashed before publication, including after final transport serialization. Panel-root retargeting and DB WAL appearance fail closed. Re-rasterization intentionally favors verification over speed; no production timing claim is made.

## D2 publication

JSON writes use a unique sibling temporary file, serialization with `allow_nan=False`, explicit flush/fsync, and successful close before publication. Final input/runtime verification runs **after final serialization and close**, before the publication operation. Publication uses POSIX `link(temp, final)` plus temp unlink: atomic no-replace visibility, unlike a replacing `rename`. Existing outputs and even dangling output-directory symlinks are refused. Failed temporary files are retained for diagnostics; serialization/flush/fsync/close/validation failures cannot create the final filename. Normal failures leave FAILED.json, not evaluation.json. This is a file-content/atomic-visibility guarantee, not a power-loss directory-durability certificate.

## D3 runtime boundary and fixed-A handshake

`mvp_decode_runtime.py` uses the tree identity concept from immutable `real_probes.py` at `3b62306`; no active A files or probe implementation paths were copied. Runtime pins come from the supplied frozen deployment report:

- Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`;
- Transformers `4.57.6`, Accelerate `1.15.0`;
- Hyena commit `0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446`;
- SIF `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`.

Receipt interface currently follows A's `mode=generation_cache_parity`, CUDA/real-runtime fields, cache/off/nonzero-projection/reorder evidence, `runtime.versions`, and `probe_identity={schema_version:1,probe_path,imports,asset_roots}`. The receipt's expected SHA256 is mandatory operator input. The receipt is trusted operator evidence, **not cryptographic proof of execution**, and no approval boolean or allow-unverified route replaces verification.

The gate checks actual imported versions, available CUDA, offline flags, actual SIF bytes and active Apptainer/Singularity container path. It freshly hashes every declared import and immutable asset tree, including link destinations/additions/removals. Both actual `HF_HUB_CACHE` and `./HFmodels` are bound; both Hyena refs must select the pinned revision. The tokenizer must use that revision's actual snapshot. Every actually loaded external Python/extension module must match the receipt closure; relocated dynamic HF files must additionally match Python files in the pinned model/tokenizer snapshots. Missing/unrecorded/changed dependencies fail. Loaded Hyena/model-config/tokenizer classes must be remote classes from those pinned Python bytes; model identity, config commit and config values (with only existing architecture overrides) are checked after every model load. Local loaded implementation files are captured and rehashed too.

All assets/SIF/imports are freshly checked after decoding and immediately before publication; newly appearing imports cannot silently escape the final audit closure. This is deliberately strict and can reject a genuine environment whose preflight omitted a decoder-used dependency. **The fixed-A receipt has not landed or been exercised here. Controller must reconcile its finalized interface and run the genuine preflight; do not loosen closure checks to make a receipt pass.** Actual model/tokenizer/GPU/GSF/8192 compatibility remains a runtime gate, not inferred from CPU fixtures.

## Verification

Local isolated `.venv` uses Python 3.12, Torch 2.6.0, Transformers 4.46.3, Accelerate 0.34.2, DuckDB 1.5.5; these are **not** the pinned SIF versions. Initial extended-suite execution found only missing local `pysam`; installed it inside this worktree's venv, then reran the full suite:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:revision/scripts/mvp \
 .venv/bin/python -m pytest -q \
 tests/test_mvp_decoder_integration.py tests/test_mvp_decode.py \
 revision/scripts/tests/test_mvp_panel_inputs.py \
 tests/test_mvp_score_producer.py tests/test_mvp_score_producer_review.py \
 tests/test_mvp_score_producer_integration.py tests/test_mvp_tracks.py
```

**107 passed in 383.89 s.** Follow-up verification after the final runtime snapshot-closure tightening is recorded in the commit/session handoff.

New tests exercise real schema-1 publication → real candidate/finalizer gates → decoder → producer receipt/count helpers; actual strict safetensors loading, baseline weights-only reconstruction, GFFTokenizer/grammar/GFF conversion and 350 inventory are retained. High-junction regression uses **70 actual DuckDB molecules and the actual rasterizer**, asserting junction `log1p(70)` and coverage `log1p(64)` without clipping. Seed equality, all-eight off zeros, biological-zero acceptance and N/n masks are asserted. Negative tests include missing marker, changed JSONL, escaping symlink, rehashed over-cap coverage, sequence/request/authority mismatch, obsolete internal schema, NaN/negative/N/availability violations, atomic write/flush/fsync/close failure, existing-final refusal, mutation after final serialization, missing/mismatched preflight, unbound/changed/missing loaded dependencies and changed asset trees.

Full CPU integration explicitly substitutes remote Hyena/tokenizer, GPU generation/runtime and GFFCompare. **Additional limitation:** this isolated checkout lacks actual frozen FASTA/GFF bytes, so that integration also uses an explicitly labeled synthetic reference-authority fixture; it does not meet an interpretation of “only remote/GPU/GFFCompare stubs” that requires real frozen reference bytes. Candidate records/DB metadata and final publications are genuinely checked, not bypassed. Separate existing tests exercise the real git-anchored reference rejection gate. No actual SIF success is claimed.

## Remaining gates

1. Fixed-A generation preflight/interface reconciliation, immutable assets and genuine pinned-SIF CUDA execution; actual full model/tokenizer, cached beam-2, EST forwarding, grammar and total-token-limit smoke.
2. Real frozen-reference bytes and accepted production-sized evidence/corpus, three genuinely completed final runs, and actual 350-output decode.
3. **Real GFFCompare 0.12.6**, not the executable CPU stand-in, for the tool transport smoke. No remote action was taken.
4. Independent structural scoring, coverage/withholding policy and author/scientific decisions remain separate. This work cannot establish them.

Advisor planning consultation was attempted but unavailable due to a client/model-version incompatibility; no advisor approval is claimed. Graft was built after integration and refreshed for the final branch.
