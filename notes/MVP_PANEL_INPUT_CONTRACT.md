# MVP panel input contract

Producer: `revision/scripts/mvp/mvp_prepare_panel_inputs.py --input REQUEST.json --out NEW_DIRECTORY`.
Request keys: `panels`, `fasta`, `reference` (GFF), and `candidate` containing
`acceptance`, `acceptance_sha256`, `inventory`, `receipts`, `corpus`, `evidence`.
These are the existing B-producer request/acceptance formats. Paths resolve from cwd.
Reference identity uses the immutable authority in `mvp_produce_counts.frozen_reference`;
acceptance companions and corpus/evidence identities use the existing CPU gates.
No unverified identity override exists in the CLI.

Decoder consumers must require `COMPLETE`, verify its manifest SHA256, then verify
all manifest file identities. `manifest.json` is canonical sorted compact JSON,
schema_version 1, purpose `est_mvp_panel_inputs_not_training`, training_eligible false.
Paths for generated files are relative to the output directory. Each of the 50 rows
has gene_id, window `[chrom,start0,end0]`, genomic-forward sequence SHA256 (ASCII
sequence, no header/newline), sequence FASTA identity, track identity including
bytes/SHA256/dtype/shape, biological_nonzero_per_channel (channels 0..6 only),
availability_nonzero (channel 7), zero_biological_support, selected molecule count,
and per-locus JSONL provenance identity.

Each `.npy` is float16 `[30720,8]`; load with `numpy.load(..., allow_pickle=False)`.
Reuse the single on-array for seeds 1/2/3: producer verifies exact equality at
keep_fraction=1 and aug_seed=20260914+seed. The paired off array is
`numpy.zeros_like(on)` (including availability), with the **same** sequence/window.
Do not use training Dataset, RN, tile_est membership, ONT, or HiFi as panel inputs.
EST selection is solely excluded_heldout with primary_count>0. Retained flank
molecules never substitute for absent heldout support. N/n positions are all zero.
Biological zero windows are reported, not rejected. Junction endpoints count even
when no exon block intersects the window. Availability alone is not EST support.

Provenance JSONL contains one record per selected molecule, complete molecule and
member metadata, and the original (unclipped) blocks/junctions that matched the
window predicates. These records include exclusion cause; no representative-span
shortcut or training membership restriction is applied. Noncontributing unknown-
strand junctions are retained as selected geometry, not called biological support.
Input/request/code hashes are freshly checked before and after manifest publication;
COMPLETE is written last. Existing output directories are never overwritten. Failed
partial directories are not consumable and must not be retried in place.
