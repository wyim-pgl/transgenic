# Job 6165628 — independent local actual50 numerical verification

## Decision

**PASS: all 50 copied panel publications, artifact bytes, declared numerical summaries, and array-to-copied-provenance numerical equality.** This closes the previous review's unmeasured array-content item within the explicitly authorized local scope. It is not source-database completeness certification, decoder readiness, whole-genome-reference equality, or scientific approval. **scientific_approval=false. extern OUT_OF_MEMORY 0:125 remains unresolved and separate.**

Prior scope: [panel_result/REVIEW.md](../panel_result/REVIEW.md). Machine evidence: [validation.json](validation.json), [validation.stdout.txt](validation.stdout.txt). Reproduce locally: `python notes/power_recovery_20260917/panel_arrays/validate.py` (stdlib + installed NumPy only).

## Transfer and preservation

Before transfer, summed the existing `panel_result/remote_file_stats.json`: **152 panel files, 29,309,109 bytes**, below the 100,000,000-byte stop threshold. A single successful `scp -r` copied only `pronghorn:/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916/outputs/job-6165628/panel` into the fresh [copied/panel](copied/panel/) directory. Its complete inventory and each size agree with the saved remote inventory: 50 NPY, 50 FASTA (`.fa`), 50 JSONL, manifest and COMPLETE. No remote job, write, hash scan, database query or source/model regeneration. Copied originals remain in place; local hashes before and after validation/tamper testing agree. All intentional writes are under `panel_arrays` (graft context commands reported automatic graph-refresh side effects).

## Independently measured PASS checks

- **150/150 artifact SHA256 and byte sizes** match the original manifest. No duplicate/escaping/symlink artifact paths or unexpected files. All 152 file SHA256 values are recorded in `validation.json`.
- Original manifest hash is **7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e**, equal to the previously collected original. Recomputed original `OBSERVATIONS_COMPLETE` job6165628 → counts → manifest/reference-compatibility links, and exact `COMPLETE` → manifest link. Copied COMPLETE bytes equal the earlier original. All count projections agree with manifest rows.
- Frozen panel TSV digest rechecked; exact ordered 50 distinct locus IDs and 30,720-base windows recomputed from its geometry. Manifest policy, genomic-forward orientation, excluded-heldout population, full-retention policy and training exclusion agree with the existing consumer contract.
- **50/50 NPY** load with `allow_pickle=False`, no trailing bytes; actual float16 dtype and `[30720,8]` shape, all finite and nonnegative. Coverage channels obey `log1p(64)` cap. Every non-N availability value is exactly 1; N positions would require zero across all channels. **There are 0 N positions in these actual windows**, so the zero-at-N branch has no real-data coverage.
- **50/50 FASTA** exact header/window/`genomic_forward` metadata, two-line serialization, 30,720-base ASCII ACGTN alphabet, and sequence-only SHA256 match the manifest. This checks declared orientation metadata, not independent biological orientation relative to the untransferred source FASTA.
- **50/50 JSONL** canonical serialization, counted rows, sorted unique molecule IDs within each window, excluded-heldout/positive-primary selection metadata, nonempty member provenance and member identity agreement. Block molecule/chromosome identity, ordering, containment and window overlap pass; junction identity, intron geometry, strand-dependent donor/acceptor metadata and independent endpoint-overlap predicates pass. Blocks/introns need not lie wholly inside a panel window: clipping and endpoint overlap follow the existing contract.
- **50/50 arrays exactly equal an independent NumPy reconstruction from the copied JSONL and window sequences**, after float32 intermediate/float16 storage conversion: distinct-molecule unioned block coverage by strand, log1p with coverage cap64, distinct-molecule donor/acceptor endpoint counts, unknown-strand junction exclusion, genomic-forward channel order and availability. This is local verification arithmetic only, not regeneration from source inputs. No tolerance was needed, despite local NumPy1.24.3 versus producer receipt1.26.4.

Measured aggregate results:

| Measurement | Result |
|---|---:|
| Selected molecules summed over windows | 3,443 |
| Per-window selected range | 3–398 |
| Distinct molecule IDs across these copied windows | 3,354 |
| Biological nonzero-position sums, channels0–6 | [34789,52639,71107,138,140,193,204] |
| Availability nonzero positions | 1,536,000 |
| Zero-biological-support loci | 0/50 |

The 3,443 sum is not a unique genome-wide count. The 3,354 distinct-ID count applies only to the copied panel provenance.

## Negative checks and execution

**12/12 disposable-copy tamper checks rejected as intended:** artifact-byte corruption; wrong dtype; wrong shape; NaN; availability corruption; orientation-header corruption; sequence-digest corruption; duplicate molecule; invalid block boundary; wrong heldout policy; nonzero numerical value corruption that preserves support counts; wrong channel summary. Except the explicit byte-hash test, modified artifact hashes/sizes were updated in the disposable row records so semantic rejection was exercised rather than merely digest failure. Originals were never edited; temporary copies were removed automatically.

Initial negative-test run rejected the boundary tamper at block-order validation rather than its intended boundary assertion. The fixture was narrowed to a zero-length block without altering order; the subsequent full run exited0 with all50 positive and12 negative checks passing. No real-data check required relaxation.

Local environment: Python3.8.20, NumPy1.24.3. Verifier SHA256: `7ac27f47c615195ca5faac9f275fa37cf017914e2eb94963625d6c4f4d501d4d`. Existing consumer checks were reused as the contract (`notes/wait_work_20260916/gpu_followup/source/revision/scripts/mvp/mvp_verify_panel_inputs.py:13–84`); importing its full verifier would query DBs and require Torch/DuckDB, so it was deliberately not executed. Provenance selection predicates were checked against `mvp_prepare_panel_inputs.py:57–80`; independent arithmetic follows the pinned package `src/transgenic/evidence/mvp_tracks.py`. No production edits or dependencies added. Advisor consultation failed with client/model-version incompatibility and provides no endorsement.

## Explicit unverified residuals

1. **Whole-reference/source identity and freshness:** no independent rehash of the full FASTA/GFF, corpus/evidence DB or remote SIF, no live source/WAL/mutation guard, and no source DB selection/completeness re-query. These remain execution-produced receipt assertions accepted only for consistency by the earlier small-file review. Array-to-copied-JSONL equality cannot prove that JSONL includes every qualifying source molecule or correctly represents the source DB.
2. **Reference sequence/orientation:** per-window FASTA hashes/metadata do not prove equality to slices of the authoritative whole-genome FASTA, nor genomic-forward orientation against it. No full-genome sequence-equality claim is made.
3. **Seeds:** a single shared array per locus and manifest seed policy are available; no independent three-seed source-DB rasterizer execution occurred. No claim of a new runtime seed-independence experiment.
4. **Scheduler and downstream:** externOOM remains unresolved; this does not certify clean scheduler completion, GPU/SIF/tokenizer/decoder consumer compatibility, train checkpoints, inference/scoring, independent long-read evaluation, reference-policy decisions or scientific approval.

Controller handoff: link this report from the recovery/TODO ledger as completion of the authorized local actual50 numerical verification only; retain the above residuals and existing scheduler/downstream blockers. No further remote execution is required or requested by this report.
