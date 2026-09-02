# GSF specification and B5 data contracts — v1.0 (frozen 2026-09-02)

Status: **contract**. Everything below is fixed before any B5 database build, split, or training run. Changes require a new version of this file and are logged in `revision/protocols/PROTOCOL_B1_frozen_v1.md` §12. Source of the defects this contract closes: `revision/protocols/IMPLEMENTATION_ORDER_B5_C0_C2_v1.md` §1.

## 1. Coordinates
- A **window** is the padded genomic interval stored in the database row (`start`, `fin`; 0-based, half-open, chromosome coordinates). Window length is a multiple of 6,144 nt. Padding rule: `pad = (6144 − gene_length mod 6144) mod 6144` split symmetrically (extra base on the right); a gene whose length is already an exact multiple receives **no** extra chunk (closes the "+6,144 at exact multiples" defect).
- GSF feature coordinates are **relative to the window start, 0-based, half-open** `[start, end)`. GFF3 1-based inclusive `[s, e]` converts as `[s − 1, e)` relative to the window: `rel_start = s − 1 − window_start`, `rel_end = e − window_start`.
- Coordinates are always given on the forward genomic strand of the window; the `strand` field carries orientation. Reverse-complement rows (see §5) are re-expressed in the RC window's own forward coordinates.

### Corrected README examples (window start = gene start, no padding shown)
Example 1 (GFF CDS 100–150, 200–280, 350–400, `+`, phases 0/2/1):
```
0|CDS1|51|+|A;100|CDS2|181|+|C;250|CDS3|301|+|B>CDS1|CDS2|CDS3
```
Example 2 (CDS 100–130, 180–220, 280–350; mRNA2 = CDS2+CDS3):
```
0|CDS1|31|+|A;80|CDS2|121|+|B;180|CDS3|251|+|A>CDS1|CDS2|CDS3;CDS2|CDS3
```
Example 3 (5′UTR 500–550, CDS 550–650, 700–800, 3′UTR 800–900; window start 500):
```
0|five_prime_UTR1|51|+|.;50|CDS1|151|+|A;200|CDS2|301|+|B;300|three_prime_UTR1|401|+|.>five_prime_UTR1|CDS1|CDS2|three_prime_UTR1
```
(The README's `0|CDS1|50` form is off by one at every end coordinate and is superseded.)

### 1a. Window policy `tier6144-v2` (variable context; author decision 2026-09-02, protocol A25)
The new-version training database uses three window tiers of 30,720 / 61,440 / 129,024 nt (5, 10, 21 chunks of 6,144). A gene
takes the smallest tier that holds it plus at least 1,000 nt of flank on each side; in training builds the next larger tier is
chosen with probability 0.3 and the gene is placed at a random (seeded) offset inside the tier, so the model sees the same gene
with different amounts of context. Genes that do not fit the largest tier with flanks are rejected. GSF coordinates stay
window-relative; the `window_policy` column records the policy per row and the validator applies the matching cap.
`sym6144-v1` (≤ 49,152, symmetric padding) remains available for the parity control.

### 1b. Window policy `tile6144-v3` and GSF v3 (author decision 2026-09-02, protocol A26)
The new-version database labels **every complete gene inside a window**. Each tier (30,720 / 61,440 / 129,024) tiles every contig
with a seeded random offset (training builds); the label of a tile is the canonical GSF blocks of all genes fully inside it, in
coordinate order, joined by the `<gene>` token; a tile with no complete gene is labelled `<empty>` and only 10 % of such tiles are
kept. Genes crossing a tile edge are excluded from that tile's label and counted (`edge_partial=n` in `qc_flags`). Feature numbering
and strand consistency restart per gene block; gene blocks must not overlap. Caps: 96 genes per window, 8,192 v3 tokens
(`<s>` + blocks + separators + `</s>`; set from the 2026-09-02 tile statistics: A. thaliana 129-kb tiles hold 29 genes on average, p95 42, max 80, ~120 tokens per gene, so 38 % of them exceed 4,096); decoder positions 8,192; vocabulary v3 = v2 + `<gene>` + `<empty>` (290 tokens). A tile's
split comes from 1,032,192-nt genomic blocks (seeded 75/10/15 per species, strict held-out blocks → test; most restrictive among overlapped blocks); genes whose orthogroup split is more restrictive than the tile are N-masked in the sequence and left out of the label (`leak_masked=n`, protocol A29); a tile containing a hard-flagged
gene (A22) has `train_weight 0`. `window_genes` maps tiles to member genes. RC of a tile reverses every block and re-sorts.

### 1c. Tiled inference and stitching (protocol A27)
Whole-genome inference runs every tier over the genome with three offsets (0, ⅓, ⅔ of the tier). A predicted gene is
**accepted from a tile only if it lies at least 1,000 nt inside both tile edges**; identical predictions from different
tiles/tiers (same canonical signature after mapping to genome coordinates) are merged; overlapping non-identical predictions
are resolved by (1) the tile in which the gene is furthest from an edge, (2) the larger tier, (3) grammar-audit violations
(fewer wins), (4) canonical order. The per-locus prompted mode (single-gene window, published behaviour) stays available.

## 2. Feature and transcript grammar
`<features> > <transcripts>` where a feature is `start|TYPEn|end|strand|phase`, TYPE ∈ {CDS, five_prime_UTR, three_prime_UTR}, `n` is the feature number by first use after canonical ordering (§3), phase ∈ {A, B, C} = {0, 1, 2} for CDS and `.` for UTRs. Transcripts are `|`-joined feature names separated by `;`. Features shared by several transcripts appear once in the feature list.

## 3. Canonical ordering `gsf-order-v1`
- Feature list: transcript-oriented features (5′UTR → CDS → 3′UTR along transcription) sorted by genomic coordinate.
- Transcripts sorted by the tuple `(intron_count, oriented_intron_chain, oriented_span, CDS_signature, UTR_signature)`; mono-exonic transcripts have an empty chain; identical signatures are merged; **no ID tie-break** (ordering never depends on transcript IDs).
- Feature numbering (`CDS1`, `CDS2`, …) is assigned by first use after transcript ordering.
- Reverse complement is canonicalised in its own orientation; invariant `canonicalize(RC(RC(x))) == canonicalize(x)`.
- Serialisation is deterministic: the same annotation always yields byte-identical GSF.

## 4. Caps — reject, never truncate
A record is **rejected at build time** (logged with reason) when any of: CDS features > 150, 5′UTR features > 50, 3′UTR features > 50, transcripts > 15, v2 token count > 2,048, window > 49,152 nt. Rejected records are listed in `build_manifest` with counts per species and reason. No mid-structure truncation and no `<unk>` for out-of-range features (closes the tokenizer/dataset truncation defects).

## 5. Reverse-complement augmentation
Single option `--rc {none, all, isoform-only}` (replaces `--add-rc/--add-rc-iso-only`). RC is a **pure transformation** of the canonical GSF: coordinates mirrored in the RC window, feature order reversed, strand flipped, CDS phases recomputed 5′→3′ as `(3 − cumulative_CDS_length mod 3) mod 3`, then re-canonicalised. RC rows carry `is_rc = true` and **inherit the split of their forward row**.

## 6. Database schema additions (`geneList`)
Existing columns are kept. New columns, all NOT NULL unless noted: `species_id`, `gene_id`, `orthogroup_id` (NULL only for singleton groups, which then use `gene_id`), `split` ∈ {train, valid, test}, `strict_holdout` BOOL, `is_rc` BOOL, `ordering_version` = `gsf-order-v1`, `build_version`, `source_fasta_sha256`, `source_gff_sha256`, `split_file_sha256`, `window_policy` = `sym6144-v1`, `gsf_token_count`, `contig_boundary` BOOL (window clipped at a contig end). Prediction/label columns are SQL NULL when absent (never empty strings). All inserts are parameterised. The last gene of every file is flushed (EOF defect closed); chromosome/strand belong to the record being inserted (ownership defect closed).
Tables `gene_split` (mirror of the split TSV) and `build_manifest` (species, input paths and hashes, rejected counts, tool versions, timestamp, git commit) are written in the same build.

## 7. Split table `data/splits/b5_orthogroup_split_v1.tsv`
Columns: `species_id gene_id orthogroup_id split strict_holdout seed source_version`. Orthogroups from OrthoFinder on the nine primary proteomes (§8); singleton genes are their own group. Orthogroup-level 75/10/15 with seed 123; the *A. thaliana* strict held-out loci (3,429; list from the revision results) **and their entire orthogroups** are forced to `test`; RC rows inherit. *Z. mays* is absent by species manifest, not by prefix. The committed TSV's sha256 is stored in every DB row.

## 8. Species manifest
`data/manifests/b5_species_v1.tsv` — the nine training species with assembly, annotation, absolute input paths on pronghorn and md5 of the exact files. Excluded species and reasons are in `data/manifests/b5_excluded_species_v1.tsv`. Builders read only these manifests; a species not listed cannot enter the DB.

## 9. Model configuration for B5
`configs/b5_400m_v1.json` — the released 400M recipe (commit d013418): 12 + 12 layers, d_model 768 (encoder) / 1,536 (decoder), 6 + 6 heads, Longformer window 1,024 per layer, dropout 0.1, AdamW lr 5e-5, weight decay 0.02, effective batch 96, linear warm-up 5 % then linear decay, ≤ 22 epochs. **New rule:** checkpoint = minimum mean validation loss with patience 3 (the release-era script saved every epoch because it compared loss to a stored perplexity). Seeds: 123 primary (reported), 456 and 789 confirmatory (protocol A18.6). The generic and GB10 scripts default to the 1.17B wide config and must not be used for B5 without this file.

## 10. C2 label contract (built only if the week-3 gate passes)
Label tensor `float32[L, 14]` with the 14 classes of `preprocess.py` (`protein_coding_gene, lncRNA, exon, intron, splice_donor, splice_acceptor, 5UTR, 3UTR, CTCF-bound, polyA_signal, enhancer_Tissue_specific, enhancer_Tissue_invariant, promoter_Tissue_specific, promoter_Tissue_invariant`) plus a weight tensor `float32[L, 14]`; the trainer outputs 14 classes (not 9) and uses per-cell weighted loss normalised by the sum of weights per window. Evidence adds positives only, with the weights and class semantics frozen in protocol A18.4; unidentifiable cells have weight 0. Preprocessed labels use the same 0-based conversion as §1 (closes the 1-based shift).

## 11. Tests that guard this contract (`tests/`, written before code)
`test_gff_gsf_roundtrip_tair10`, `test_gff_gsf_roundtrip_refgen_v4` (incl. EOF gene, chromosome transition, 1-bp and 51-bp features), `test_gsf_rc` (involution, phases, UTR order, metadata strand), `test_canonical_isoform_order` (permutation invariance, idempotence), `test_gsf_feature_caps`, `test_gsf_token_cap`, `test_v2_transcript_cap`, `test_rc_augmentation_split_inheritance`, `test_orthogroup_split_integrity` (no cross-split orthogroup, no maize, strict holdouts test-only, no validation-only evidence label in train/valid), `test_segmentation_label_weight_alignment`.
