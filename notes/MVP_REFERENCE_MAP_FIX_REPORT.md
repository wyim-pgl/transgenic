# Evaluation-only reference mapping fix — 2026-09-16

## Status and exact source

Implementation commit: **`9acd7dbb99304f193704ba6a174340cc540e4c68`**.
Branch: `mvp-eval-reference-map-20260916`, created directly from
`b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6` in an isolated worktree.
This report is a subsequent documentation-only commit; implementation pins below
refer to 9acd7db and remain unchanged by the report.

Local CPU tests pass. **Actual 50-locus compatibility remains unverified.**
Controller independent review and a new immutable evaluation CPU release/allocation
are required. No submission, SSH, job inspection/change, merge, push, production
DB query/hash, reference rebuild, or scientific-policy decision was performed.
Training source b660 and job 3163986 were left untouched; no live job status is
asserted. Job6165559's old failure/publication must remain immutable.

## Root cause and implementation interface

The neutral diagnosis and result review identify an empty restricted reference:
panel keys are builder IDs (`AthNNNNNN`), whereas raw GFF IDs are original IDs.
`71_b6_panels.py:230–260` already joins `gene_key_map` and uses the original IDs
when constructing panel geometry. Panel creation is not changed.

Only production source changed: `revision/scripts/mvp/mvp_produce_counts.py`.
The shared adapter remains inside that already-audited `DECODER_ONLY` member:

- `resolve_reference_mapping(panel, corpus, authority, accepted_hashes, panel_path)`:
  invoked after existing candidate/inventory/byte gates. Rechecks accepted corpus
  and panel SHA256 and frozen reference identities; opens DuckDB read-only with
  one thread and refuses WAL snapshots. Queries only the Athaliana manifest and
  requested 50 species-keyed map rows. Requires total unique keys, non-null/nonempty
  original IDs, one-to-one original IDs, exact map coordinates, and full manifest
  FASTA/GFF SHA256 equality to `frozen_reference`'s actual byte identities.
- `prepare_mapped_reference(panel, reference, restricted, mapping)`:
  restricts unchanged raw GFF by original IDs, checks restriction gene count and
  exact strict-parser inventory, then selects by mapped ID with exact raw
  chromosome/start/end/strand equality. Diagnostic subsets use the same helper,
  not an alias/coordinate/Name/prefix fallback.
- `recheck_reference_mapping(mapping)`: rehashes source identities and rejects a
  newly appeared WAL before publication. Both callers invoke it.
- `select_transcripts(..., reference=True, reference_id=...)` explicitly accepts
  the mapped raw key. The prediction overlap branch is unchanged.

The `reference_mapping` receipt in scorer `receipt.json` and CPU
`reference_compatibility.json` records method `corpus_gene_key_map`, ordered builder
and original IDs, original Name (diagnostic only), species, geometry, panel strand,
corpus/panel/reference/source-code pins and frozen authority. Map strand is not
invented: the raw GFF strand is checked against the panel. Duplicate structures
are neither repaired nor deduplicated; existing author-policy refusal remains.

Patched CPU wrapper: **`tasknotes/reference_mapping/run_panel_cpu.py`**.
Copied from the neutral wrapper; only shared preparation, mapping diagnostics,
per-locus fallback, and identity recheck were changed. The external/main wrapper
was not edited. Candidate checks also run in `--reference-only` mode. Joint
preparation errors remain failed diagnostics; per-locus fallback still calls the
same strict mapped helper. No decoder-ready or scientific PASS is created.

## New pins and deployment boundary

| Pin | SHA256 |
|---|---|
| Producer | `ace42b9233de3b49c40e2b7b930e65c4567ef1cd49ac193bc9472030f0bf806b` |
| Decoder operator `decoder_code_sha256` | `c0e8c141b1963abe1bd5b1b167bada5e3089057da7c4855dd61aa4fb4610b0d7` |
| Evaluation component closure (86 files) | `44078a800aab35c991d6792899659627f76a1867520a40724887c9003bd9133f` |
| Patched CPU wrapper template | `ac727d1caab301eb2c35782512710713ad9ad8c90715653600eb9c4d824bc202` |
| Unchanged panel TSV | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |
| Exact first-gene fixture | `192ae64c5fb3a81067400b4ea8d6600e314e8e4da1747ec7be4b50d4c29b9350` |

Decoder pin uses the unchanged `mvp_decode_runtime.DECODER_ONLY` ordered-independent
relative-path/hash object, sorted compact JSON plus newline, SHA256. No closure
exemption or runtime pin relaxation was added. Component closure uses tracked
`src/transgenic/`, `configs/`, `revision/scripts/mvp/` and
`train/train_HyenaTransgenic_MVP.py` with the same serialization. Recomputing it
at b660 reproduces the old `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`.

**The copied wrapper is not yet a runnable new release.** Per the requested
mapping-only edit boundary it retains the historical hardcoded b660 source,
component and source-inventory guards. Those guards intentionally reject the new
evaluation tree. Controller packaging must create a fresh source inventory, bind
its digest and the new evaluation commit/component in the staged wrapper, and
independently review/pin that final wrapper's new hash. Do not use the template
hash above as the hash of that future repinned entrypoint. No new source-inventory
or release-tar digest is claimed here.

`mvp_prepare_panel_inputs.preflight` hashes the changed producer and the unchanged
verifier requires exact source/input inventory equality. Therefore publish a
**new panel namespace under the new evaluation pin**. Do not rewrite/reuse
6165559's manifest/COMPLETE or bypass its stale-code refusal. Training consumed
code remains separately verified at its retained bound training root by the
existing `verify_training_code` protocol; no training/evaluation HEAD equality
was introduced. Current 772 probe/handshake constants and numerical files are
unchanged. Decoder SIF/package compatibility remains a separate controller gate.

## Verification evidence

Red regression: `test_real_prefix_mapping_and_identity_case` failed with missing
`resolve_reference_mapping` before implementation. Initial system Python lacked
Torch; tests were run with the existing local `/tmp/mvp-final-source-env` instead.
The first combined run exposed a test-fixture window issue: the actual mitochondrial
first gene is too near the chromosome start for an MVP window. The CPU fallback
unit test now explicitly stubs only window eligibility; production window logic
was not changed. Both full350 integrations use the actual frozen panel geometry.

Final command (local CPU):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
/tmp/mvp-final-source-env/bin/python -m pytest -q -rs \
 tests/test_mvp_reference_mapping.py \
 tests/test_mvp_score_producer.py \
 tests/test_mvp_score_producer_integration.py \
 tests/test_mvp_decoder_integration.py \
 tests/test_mvp_decoder_runtime.py \
 tests/test_mvp_final_source_integration.py
```

**98 passed, 1 skipped, 439.67s.** Skip: pinned Transformers 4.57.6 is not installed;
local 4.46.3 is not SIF validation. No GPU/runtime approval is implied.

Coverage includes real-prefix `Ath000001 → ATMG00010.TAIR10`; identity mapping;
long raw ID/short Name and counter negatives; missing/duplicate/wrong-species/null/
empty/many-to-one IDs; wrong coordinates, strand and hashes; duplicate raw genes;
empty reference rejection versus valid empty predictions; CPU shared-helper and
per-locus diagnostic paths; stale panel producer identity refusal; actual full350
producer/decoder integration gates with explicit consistent synthetic
`gene_key_map` and `build_manifest` authority. No actual 50 raw-map query was run.

`git diff --check` passed. The following invariant check returned an empty diff:

```sh
git diff --exit-code b660bf1 -- src train configs revision/protocols \
 revision/scripts/mvp/real_probes.py revision/scripts/mvp/run_bounded.py \
 revision/scripts/mvp/make_launch_readiness.py revision/scripts/mvp/mvp_decode_runtime.py
```

Panel bytes/order/core20, 350 transport keys, evaluation windows, prediction
selection, scoring denominators, training/finalizer/model/probe code remain
unchanged. Only task source/tests/fixture and this report are committed. Graft
was queried before source work and rebuilt locally afterward; graph cache is
untracked. Advisor consultation failed due client/model version incompatibility;
there is no advisor or independent-review approval to claim.
