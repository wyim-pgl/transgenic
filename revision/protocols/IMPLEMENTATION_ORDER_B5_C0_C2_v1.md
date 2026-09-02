# Implementation order — GSF pipeline repair, B5 database/retraining, C0 evidence tables, C2 labels (v1.0, 2026-09-01)

Cross-verified against the repository copy (commit 271e938 tree) by reading the cited lines; Codex audit in `codex_gsf_pipeline_20260901.md`. Status of each Codex claim after verification is given in §1.

## 1. Verified code locations and defects

| Item | Location | Verified | Note |
|---|---|---|---|
| Production GFF→GSF/DB builder | `src/transgenic/datasets/preprocess.py:54` `genome2GSFDataset`; CLI `scripts/create_database.py:53,151` | yes | `scripts/gff2gsf.py` is a separate converter with different coordinate origin (gene start vs padded window) |
| Split decision | `train/train_HyenaTransgenic.py:452–470` (`torch.manual_seed(123)`, `random_split` 75/10/15 on DB rows) | yes | rows are post-RC-augmentation → twin leakage; B5 must replace this with a split-table lookup |
| RC augmentation | `preprocess.py:258–290`; `addRCIsoOnly` is evaluated only inside `if addRC` | yes | CLI declares `--add-rc`/`--add-rc-iso-only` mutually exclusive (`create_database.py:100–107,169–172`) → `--add-rc-iso-only` alone produces no RC rows with current code; the deposited DB was built by an earlier version. Replace by one enum `--rc none|all|isoform-only` |
| Last gene not flushed at EOF | `preprocess.py:210–251` insert-on-next-gene, `:471` `con.close()` without flush | yes | matches the manuscript's "one gene model lost to a file-end flush" |
| Previous row takes next gene's chr/strand | `preprocess.py:182–203` parse before insert | plausible (Codex) | add to tests |
| Padding adds a full extra chunk at exact multiples | `preprocess.py:303–312` | yes | `gene_length % staticSize == 0` → +6,144 |
| Feature caps not enforced at build | tokenizer `tokenization_transgenic.py:81–89` (CDS1–150, UTR1–50); dataset truncates labels at 2,048 (`datasets.py:301–313`) | yes | overlong GSF is cut mid-structure; `<unk>` for CDS>150 |
| >15 transcripts serialized without separators | `tokenization_transgenic.py:175–208` (`tx_count=min(n,15)`, `<iso>` only while `j<tx_count-1`) | yes | reject >15 at build |
| Segmentation class mismatch | labels 14 classes (`preprocess.py:550–581`, `datasets.py:376–399`) vs trainer `numSegClasses=9`, `lab[:,:,0:9]` (`train_HyenaSegment.py:108–124,229–243`) | yes | C2 must fix to 14 and add a weight channel |
| Preprocessed segmentation 1-based shift | `preprocess.py:605–622` `class_tensor[start:end]` from GFF 1-based | yes (code) | on-demand path converts with −1 (`datasets.py:459–476`) |
| Model size defaults | generic `train_HyenaTransgenic.py:163–171` and GB10 script build the **1.17B wide** config; RTX4090 script is parameterized (`encoder_d_model`, `decoder_d_model`) | yes | B5 must run the **400M** config (768/12/6, decoder 1,536) explicitly; the published checkpoint's recipe is the d013418 version |
| README GSF examples off-by-one and origin ambiguity | `README.md:71–83` (`100..150`→`0..50` vs code `0..51`), `:84` vs `:104` | yes | docs fix |
| Sample weights computed but unused | `datasets.py:211–227` vs loader without sampler (`train_HyenaTransgenic.py:149–161`) | yes | leave unchanged for B5 (recipe parity) |

## 2. Frozen contracts (day 1)
- Coordinates: GSF relative to the padded window start, 0-based half-open; GFF 1-based inclusive converts as `[start-1, end)`.
- Canonical ordering `gsf-order-v1`: transcript-oriented features (5′UTR, CDS, 3′UTR in coordinate order); transcripts sorted by `(intron_count, oriented_intron_chain, oriented_span, CDS_signature, UTR_signature)`; monoexonic = empty chain; identical signatures merged; feature numbering by first use after ordering; no ID tie-break; RC canonicalized in its own orientation; `canonicalize(RC(RC(x))) == canonicalize(x)`.
- Split artifact `data/splits/b5_orthogroup_split_v1.tsv` (`species_id gene_id orthogroup_id split strict_holdout seed source_version`), orthogroup-level 75/10/15 seed 123, singletons as their own group, strict held-out A. thaliana loci and their orthogroups forced to test, RC rows inherit the forward split. Z. mays excluded by species manifest (not by prefix).
- Build rejects (not truncates) records with CDS>150, UTR>50, transcripts>15, v2 tokens>2,048; SQL NULL for predict labels; parameterized inserts; EOF flush; contig-boundary windows tagged.
- New DB columns: `species_id, gene_id, orthogroup_id, split, is_rc, ordering_version, build_version, source_fasta_sha256, source_gff_sha256, split_file_sha256, window_policy, gsf_token_count`; `gene_split` table; `build_manifest` table.
- C2: keep the 14-class `float32[L,14]` labels; add `weights_blob float32[L,14]`; evidence adds positives only; `W = 1 + source_weight·log1p(independent_molecules)` on positive cells, capped; trainer changed to 14 outputs with per-cell weighted loss (`train_HyenaSegment.py:233–256`).
- Evidence tables (C0): `evidence_alignment`, `molecule_member`, `junction_observation`, `partial_chain`, `partial_chain_intron`, `aligned_block`, `junction`, `junction_support`, `chain_junction`; exact coordinates only; molecule-unit counting per protocol v1.2.1; `validation_only` flag on every row from test-species or A. thaliana validation-only datasets.

## 3. Order (week 1 → week 2 start)
1. **Day 1** — commit the contracts above (this file + a `docs/gsf_spec_v1.md` with corrected examples) and the nine-species manifest; select the 400M configuration for B5 (RTX4090 script args or restored d013418 config) and record it.
2. **Days 1–2** — tests first (pytest, new `tests/` dir): `test_gff_gsf_roundtrip_tair10/refgen_v4` (incl. EOF gene, chromosome transition, 1-bp and 51-bp features), `test_gsf_rc` (involution, phases, UTR order, metadata strand), `test_canonical_isoform_order` (permutation invariance, idempotence), `test_gsf_feature_caps`, `test_gsf_token_cap`, `test_v2_transcript_cap`, `test_rc_augmentation_split_inheritance`, `test_orthogroup_split_integrity`, `test_segmentation_label_weight_alignment`.
3. **Days 2–3** — one shared serializer/parser module (`src/transgenic/utils/gsf.py` extended) used by both `preprocess.py` and `gff2gsf.py`; fix EOF flush, metadata ownership, exact-multiple padding, NULL labels, parameterized SQL, deterministic numbering; `--rc` enum.
4. **Day 3** — RC as a pure transformation with phase recomputation; all RC tests green before any DB build.
5. **Days 3–4** — OrthoFinder on the nine proteomes → `make_orthogroup_splits.py` → committed split TSV with checksums; manual balance review.
6. **Days 4–5** — smoke DB (`build_b5_database.py` on representative genes of all nine species) + `validate_b5_database.py`; zero maize by manifest and by `Zm*/GRMZM*` search.
7. **End of week 1** — full B5 DB build (immutable, manifest archived); training-script change: split-table selection, 400M config, best-validation-loss checkpoint + patience 3, seeds; throughput benchmark on the chosen GPU (4090 / PRO 6000 / cloud) and GB10 decision.
8. **Week 2 day 1** — start B5 seeds. In parallel (weeks 2–3): C0 ingestion of aligned evidence (validation sets flagged `validation_only`), C2 label/weight generation from training-species evidence only, segmentation trainer fix; neither touches the B5 DB.

## 4. Out of scope until B5 is reproducible
Evidence-derived GSF transcripts (spec in Codex §4), reranker consumption of C0, prompt-free/candidate-pool experiments (Protocol M), README/CLAUDE.md vocabulary-count and split-ratio corrections beyond the GSF spec document.
