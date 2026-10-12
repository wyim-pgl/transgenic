# B6 MVP pilot — does an evidence input channel move the needle at all? (v1.2 draft; v1.1 + A46 round-3 wording corrections)

**Status: exploratory design draft, pending final review and author approval; labelled per Protocol M (`PROTOCOL_M_exploratory_v1.md`), not part of the pre-registered B6 arm.**
Parent: `B6_EVIDENCE_MODE_WORKPLAN_v1.md` v1.2 (§3 WP-MVP; definitions §2.6–2.10),
`B6_WORKPLAN_README.md` §5 and A46 v1.36 draft (`PROTOCOL_B1_frozen_v1.md`, rules 12 and 17). Author decision
2026-09-14: one week (base budget ≈ 18 GPU-h; §5.4 may add ≈ 5.5 GPU-h), *A. thaliana* only, seed-123 `best` checkpoint, evidence on vs off, 3 seeds. The
decision power of the MVP is limited to the futility rule of §5.3; a positive effect justifies WP0–WP7 and replaces none
of them. Every number produced here carries the label `exploratory / MVP`; the MVP panel loci are excluded from the
pilot and final panels by construction (§1.2).

## 1. Data (A. thaliana only)

### 1.1 What exists
- Screened primary-arm EST alignments on pronghorn: `evidence/est_align/Athaliana/Athaliana.est.bam` (+ `.bai`) and
  `Athaliana.est.paf.gz` (est_align.sbatch:50, 58-59), produced by the frozen §4 command
  `minimap2 -ax splice:hq -G 200000 --secondary=no --cs=long` (est_align.sbatch:30). The MVP reads the **BAM** (CIGAR,
  `ts:A` transcript-strand tag, flags, mapq); the `--cs=long` PAF is not needed for coverage/junction tracks.
- Corpus: the tile6144-v3 DuckDB (`/work/db/b5_full_a40_v1.db`, train_b5.slurm:134). **Every `geneList` row is a
  30,720/61,440/129,024-nt tile, not a gene** (build_b5.py:501-504, row at 617; columns at 147-153). Gene coordinates:
  `gene_key_map` (build_b5.py:153-154, 517-518); held-out status: `gene_split` (157-158) or
  `data/splits/b5_orthogroup_split_v1.tsv` (3,430 `Athaliana` rows true); exon/intron counts: the source GFF named and
  sha256-checked in `build_manifest` (159-163); tile→gene: `window_genes` (508, 621, 627); block split: `tile_blocks`
  (548-557, gsf_contract.py:627). Held-out genes in train/valid tiles are N-masked and unlabelled (build_b5.py:553-555,
  565-567, 598-601; flank 50–150 nt, gsf_contract.py:40); only 431 of the 3,430 lie in a labelled test tile, which is
  why evaluation windows are built from the FASTA (§1.2).

### 1.2 Evaluation panel (no sampler of its own; A46 rule 12)
- `revision/scripts/71_b6_panels.py` writes `revision/protocols/b6_eligible_loci_v1.tsv` (+ sha256): every strict
  held-out A. thaliana gene that is multi-exon (≥ 2 exons / ≥ 1 intron in at least one source-GFF transcript),
  non-overlapping on either strand, off contig boundaries, on a nuclear chromosome (ChrC/ChrM excluded), with
  `train_weight > 0` on every carrying tile, and without gene-level A22/A30 hard flags, in one seeded shuffle order
  `shuffle_rank` (seed 20260914). The existing pool is unchanged: 2,079 loci; the 50 MVP loci include 17 with one intron.
- `revision/scripts/72_mvp_select_loci.py` walks that order and applies the MVP sub-filter: gene length ≤ 20,000 nt
  and ≥ 3 **distinct** primary-aligned A-EST molecules after rule (a) of §1.3 (identical primary placements
  collapse; rule (b) is applied in the C0-lite build only), counted by
  `revision/scripts/72a_est_molecules_per_locus.sh` with the key `(chromosome, FLAG-0x10 alignment strand, POS, CIGAR)`
  from the BAM of §1.1 (`POS` is 1-based; alignment strand is `−` iff `FLAG & 0x10`, otherwise `+`, not `ts:A`).
  The helper queries one chromosome at a time, so its `sort -u` key needs only the last three fields. The first 50
  loci that pass are the **MVP panel**, the first 20 of those the **core** (a reported stratum, never the only readout).
  Skipped EST-poor loci stay eligible: the first 300 non-MVP loci by rank are the pilot panel, the remaining 1,729
  are the final panel. Output `revision/protocols/b6_panels_v1.tsv` (+ sha256) has these exact 14 columns, in order:
  `panel, shuffle_rank, species_id, gene_id, chromosome, start, fin, strand, length, orthogroup_id, n_transcripts,
  exon_count, intron_count, est_molecules`. `start`/`fin` are **1-based inclusive**; convert to `[start−1, fin)` for
  half-open interval operations. `intron_count = exon_count−1`, where `exon_count` is the maximum over transcripts.
  Panel labels are `mvp_core` (20), `mvp` (30), `pilot` (300), `final` (1,729); `est_molecules` is filled only for
  `mvp_core`/`mvp` rows, empty elsewhere. A46 cites the same scripts and hashes.
- Evaluation windows: 30,720 nt centred on the gene, sequence from the reference FASTA of
  `data/manifests/b5_species_v1.tsv`, labels = reference transcripts of the panel gene only. **All 50 loci are decoded
  for every readout** (§5.2); the core 20 is a stratum, not a substitute.

### 1.3 C0-lite tables (one DuckDB file `evidence/mvp/athaliana_est_c0lite_v1.db`, built from the BAM in one pass)
| table | columns | source |
|---|---|---|
| `est_molecule` | `molecule_id`, `members` (read names collapsed into it), `strand` ∈ {+,−,?}, genomic transcript strand from FLAG/`ts:A` composition (§1.4; absent/unknown `ts` ⇒ ?), `mapq`, `chrom, start0, end0` of the primary placement, `excluded BOOLEAN`, `cause` ∈ {heldout_locus, test_block, unmapped, supplementary_only} | BAM, flag & 0x904 = 0 for primary |
| `est_block` | `molecule_id, chrom, start0, end0` per aligned block (M/=/X/D consume reference; `N` splits blocks) | BAM CIGAR |
| `est_junction` | `molecule_id, chrom, intron_start0, intron_end0, tx_strand, donor0, acceptor0` per `N` operation; donor/acceptor in **transcription order** (§1.4) | BAM CIGAR + `ts:A` |
| `tile_est` | `rn, molecule_id` for every **train and valid** tile whose `[start, fin)` overlaps a block of a retained molecule (valid tiles are needed for the paired validation readout of §5.2) | join with `geneList` |
| `mvp_qc` | reads in; molecules after dedup (and how many collapsed, by rule); per cause: excluded, retained; `retained_overlapping_panel` (must be 0); `valid_tiles_total`, `valid_tiles_with_evidence` (count written into the result file) | build |
**Dedup before exclusion (MVP stand-in for the A11 molecule unit).** Reads collapse into one molecule when (a) their
primary placements share `(chromosome, FLAG-0x10 alignment strand, POS, CIGAR)` exactly (`POS` is 1-based; alignment
strand is `−` iff `FLAG & 0x10`, otherwise `+`, not the transcript-strand tag), or (b) their names share a
clone/accession prefix where the header carries one (`accession.version` → accession; `<clone>.5`/`.3`, `<clone>_F`/`_R`
→ clone). Rule (a) is exactly the panel helper's definition in §1.2; rule (b) is applied only in the C0-lite build.
Do not substitute equal endpoints or `ts:A` for rule (a). Counts collapsed per rule go to `mvp_qc`.
Full A11 semantics (versions, mates, UniVec parts, audited placements) arrive in WP1.
**Leakage rule** (simplified, workplan §2.4 by hand): a molecule is `excluded` if **any** placement of any member
(primary or supplementary, either strand) overlaps a panel locus ± 150 nt (the maximum A33 flank, gsf_contract.py:40)
or any `tile_blocks` row with `split = 'test'`. No audit alignment (no secondary placements); accepted MVP gap. Excluded
molecules never enter training or validation tracks; the `heldout_locus` ones are exactly the evaluation input (§2).
~1.5 M reads, ~5 M blocks; build ≈ 1 h on one pronghorn CPU node (pysam); DB (< 2 GB) copied to DeltaAI `/work/db/`, sha256 in the run identity.

### 1.4 Track format: float16 `[L, K]`, K = 8, rasterised per tile on the fly
| k | channel | value |
|---|---|---|
| 0/1 | `cov_plus` / `cov_minus` | log1p(#retained molecules with a block covering the base, genomic `tx_strand` +/−), clipped at log1p(64) |
| 2 | `cov_unknown` | same for strand `?` (unspliced, no `ts` tag) |
| 3/4 | `donor_plus` / `acceptor_plus` | log1p(#molecules) on the boundary base, transcripts on + |
| 5/6 | `donor_minus` / `acceptor_minus` | same, transcripts on − |
| 7 | `avail_est` | 1.0 on every base when the EST source is supplied; 0.0 when the channel is dropped |
**Genomic transcript strand follows A39 and `59_evidence_support.py::transcript_strand`.** Let alignment strand
be `−` iff `FLAG & 0x10`, otherwise `+`. For `ts:A:+`, `tx_strand` equals alignment strand; for `ts:A:−`,
it is the opposite strand. Absent or unknown `ts:A` gives `tx_strand = ?` for these non-`-uf` EST alignments.
For example, `FLAG=16, ts:A:+` means genomic `tx_strand = −`, not `+`.
**Donor/acceptor are defined in genomic transcription order.** Intron endpoints below are 0-based inclusive:
for `[intron_start0, intron_end0]` and `tx_strand = +`, donor = `intron_start0`, acceptor = `intron_end0`;
for `tx_strand = −`, donor = `intron_end0`, acceptor = `intron_start0`.
Junctions with unknown `tx_strand` count in unknown-strand coverage only. Rasteriser
`rasterise_est(rn | window, L, offset, is_rc, keep_fraction, aug_seed)`: query `tile_est ⋈ est_block/est_junction` (one
read-only DuckDB connection per worker already exists, datasets.py:259-278), keep molecules with `rank < keep_fraction`,
`rank = int.from_bytes(blake2b(f"{aug_seed}\tAthaliana\t{molecule_id}".encode(), digest_size=8).digest(), "big") / 2**64`
(workplan line 25, nested draws); coverage by difference-array cumsum, junction counts by `np.add.at`. RC tiles (`is_rc`,
build_b5.py:623-627): reverse along L, swap +/− channels, keep donor/acceptor identity (workplan line 33). Tracks are
zeroed on every base where the tile sequence is `N`, so coverage
never reveals a masked gene. **Unit test (day 2, blocks day 3):** a hand-built genomic minus-strand junction (test both `FLAG=0, ts:A:−` and `FLAG=16, ts:A:+`, intron
`[100, 199]`) rasters donor at base 199 and acceptor at base 100 in `donor_minus`/`acceptor_minus`; after RC of a tile
of length L the donor sits at `L−1−199` in `donor_plus`, the acceptor at `L−1−100` in `acceptor_plus` (identity per
base, not channel sums). No phase, protein, ONT, PacBio or genotype channels in the MVP.

## 2. Evaluation-locus evidence: input vs scoring

- **Input (fraction 1)**: the `est_molecule.excluded = true, cause = heldout_locus` molecules whose placement overlaps
  the evaluation window, rasterised with the same function (`keep_fraction = 1`); train-track molecules are never input
  (their contribution in the flanks stays zero). **Fraction 0**: all-zero `[L, 8]`, `avail_est = 0`. This is workplan
  rule 10 applied by hand: input = `excluded_heldout` A-EST only.
- **Secondary readout (reference-dependent, labelled as such)**: transcript-level F1 per GFFCompare v0.12.6 exact
  intron-chain match (monoexonic: exact CDS), reference = the manifest GFF restricted to the panel gene (as
  `revision/scripts/24_restrict_reference_to_test_genes.py` does), predictions restricted to transcripts overlapping the
  panel gene; pooled numerator/denominator over the 50 loci, identical for both fractions (workplan §2.13).
- **Independent structural readout (required, day 5)**: junction-level precision/recall of the decoded transcripts
  against locally aligned A-ONT1 / A-HiFi reads.

  ✏️ **Amended again by PROTOCOL_B1 A47.2 (v1.38, 2026-09-22) — the PacBio arm is A-CCS23.** Read-end measurement
  shows A-CCS23 (PRJNA911826, Col-0 WT SRR22719002-07) is lima-processed full-length Iso-Seq: primers 0.000 in all six
  runs, poly(A) at the 3′ end only (0.988–0.994), no Illumina adapters, internal concatemer signal 0.4–0.9 % (FLNC
  control 0.3 %). By author decision it carries the PacBio structural arm for junctions, intron chains and TSS, never
  for TES. **This readout runs on A-ONT1 (SRR28341943-45) and A-CCS23 (SRR22719002-07)**: the two ONT commands
  apply to A-ONT1 and the `splice:hq` command to A-CCS23, producing `A-CCS23.<RUN>.panel.bam`. A-HiFi stays
  unavailable and is recorded missing per source. The A47 paragraph below is superseded where it says A-ONT1 alone
  and that A-CCS23 is not a replacement; the contract file is `coverage_policy_a47_2.json`.

  ✏️ ~~**Amended by PROTOCOL_B1 A47 (v1.37, 2026-09-21) — the A-HiFi arm of this readout is unsatisfiable and is not
  run.** ERR13994458 (PRJEB77203) exposes exactly one archive product, `Col_isoseq.subreads.bam` (Sequel II,
  9,093,086 subreads, md5 `7e75b3e771529738a1cd8aaabbae76de`); no CCS/FLNC product is downloadable, and by author
  decision FLNC is never produced here (`ccs`/`lima`/`isoseq refine` are not run). A-HiFi is therefore EST-grade
  sequence evidence with no structural authority. A-CCS23 (PRJNA911826) is **not** a drop-in replacement:
  it is CCS, not FLNC, and whether unrefined CCS may carry this arm is an open specification question
  (A47.1), not an implementation choice. **This readout runs on A-ONT1 (SRR28341943-45) alone**: only the
  first two pass-2 commands below apply, the `splice:hq` PacBio command is not used, and no `A-HiFi.panel.bam` is
  produced or expected. Record A-HiFi as *missing per source*, never as zero support.~~ No such alignment exists today: the long-read aligner refuses runs that
  are not `c2_training_eligible` (longread_align.sbatch:70-71) and A-ONT1/A-HiFi are `b1_validation_only`
  (PROTOCOL_B1 lines 100-102, A13 line 584). The MVP therefore runs a **locus-restricted alignment** (not the B1
  validation pass): reads must already be on pronghorn (`longread_fetch_v2.sh` manifests, A-ONT1 SRR28341943-45, A-CCS23
  SRR22719002-07 per A47.2); pass 1 pre-filters reads to the panel windows ± 10 kb; pass 2 aligns only those reads with the frozen B1 §4
  command for the source from `PROTOCOL_B1_frozen_v1.md` **§4**, against the full reference so nothing mis-maps onto
  a sub-reference. Pass 1 is a candidate-read pre-filter only, not the scoring alignment; its sketch below is not
  executable until the interval predicate is implemented:
  ```
  minimap2 -t 32 -x splice REF.fa reads.fq.gz | awk -v W=panel_windows.tsv 'BEGIN{...load W...} $12>=5 && overlaps($6,$8,$9)' | cut -f1 | sort -u > panel_reads.txt
  seqkit grep -f panel_reads.txt reads.fq.gz > panel_reads.fq.gz
  ```
  **Pass-2 commands, verbatim from §4** (substitute only the FASTA/read paths; `reads.fq.gz` denotes the selected
  `panel_reads.fq.gz` for the run):
  ```
  # ONT cDNA (orientation not proven)
  minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
  # ONT cDNA (orientation proven ≥95%)
  minimap2 -t 32 -ax splice -uf -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
  # PacBio FLNC / HQ isoforms / HiFi FLNC
  minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
  ```
  For each ONT run choose exactly one of the first two commands from its §3.3 orientation audit **as corrected by
  A39**: enable `-uf` only at read-orientation agreement ≥ 95%; below threshold or UNRESOLVED means no `-uf`.
  Never drop `-k14` or infer orientation from the panel's outcomes. Pipe the chosen command's SAM output to
  `samtools sort -o A-<SET>.<RUN>.panel.bam`, then `samtools index` it; retain per-run provenance and scores,
  with pooled results in addition. Record the exact expanded command and audit decision.
  **Definition.** A read-supported junction = intron `(chrom, intron_start0, intron_end0, tx_strand)` observed with
  exact coordinates in ≥ 2 A-HiFi or ≥ 3 A-ONT1 reads (the frozen `THRESHOLDS` of `59_evidence_support.py`, PacBio 2 /
  ONT 3) inside a panel window — **under A47.2 the PacBio disjunct is read as `≥ 2 A-CCS23`; counts are pooled
  within a source and never summed across sources, so a junction is supported if either source meets its own
  threshold** (✏️ *superseded A47 reading: only the ONT disjunct, PacBio branch without input*); predicted junctions = introns of the decoded transcripts overlapping the panel gene.
  Precision = supported predicted / predicted; recall = predicted read-supported / read-supported inside the panel
  genes; pooled over the 50 loci, per fraction. Complete-chain support (`chain_support_single_read`, `chain_callable`,
  PROTOCOL_B1 line 162) is added when `59_evidence_support.py` runs on the same BAMs (BAM required, its docstring lines
  4-6). **If the reads are not locally aligned at the panel loci by the end of day 5, any STOP is downgraded to
  INCONCLUSIVE (§5.3).**

## 3. Model change (minimal)

`E = E_seq + Linear_noBias(K → d)(tracks)` on HyenaDNA's per-nucleotide states, before the positional injection
(modeling_HyenaTransgenic.py:725) and the 6× downsampling (line 728). The addition lives **inside `HyenaEncoder`**, not
in `transgenicModel.forward`, because that is the only place both the teacher-forced path and the cached generation
path go through (see the last bullets).
- `HyenaTransgenicConfig`: new field `evidence_channels: int = 0` (config class named at modeling line 221).
- `HyenaEncoder.__init__` (line 408-435): after `self.hyena = AutoModel.from_config(...)` (line 427) add
  `self.evidence_proj = nn.Linear(config.evidence_channels, HyenaConfig.d_model, bias=False)` when
  `evidence_channels > 0`, and `nn.init.zeros_(self.evidence_proj.weight)`.
- `HyenaEncoder.forward(self, input_ids, segLabels=None, evidence_tracks=None, *args, **kwargs)` (line 522): after
  `output = self.hyena(input_ids)` (line 531), `h = output.last_hidden_state; if evidence_tracks is not None:
  h = h + self.evidence_proj(evidence_tracks.to(h.dtype))`; return `h` as `last_hidden_state` (line 533-536).
- `transgenicModel.forward` (line 661-675): add `evidence_tracks=None`; pass it at line 714:
  `self.encoder(input_ids, evidence_tracks=evidence_tracks)`. Lines 716-721 (pre-computed `encoder_outputs`) unchanged.
- `transgenicForConditionalGeneration.forward` (line 877-897): add `evidence_tracks=None`, forward it in the
  `self.transgenic(...)` call at line 929-946.
- **Zero-init must survive HF initialisation and checkpoint loading.** `post_init()` (line 646) and
  `initialize_weights()` (line 865) run `_init_weights` (line 225), and the seed-123 `best` safetensors has no
  `evidence_proj` key, so the loader re-zeroes `encoder.evidence_proj.weight` after loading and asserts
  `weight.abs().sum() == 0` before step 0. A zero weight gives exact zeros, so **step 0 is bitwise the seed-123 model
  for any tracks** under identical evaluation settings. After fine-tuning starts, fraction 0 removes only the
  evidence contribution (`W·0 = 0`): it equals the sequence-only path of the **current** backbone weights, not the
  original seed-123 model, because §5.1 updates the backbone.
- **KV-cache / generation path carries tracks without touching the cache code.** `generate()` runs the encoder once
  through `get_encoder()` (lines 655, 836); Transformers' `_prepare_encoder_decoder_kwargs_for_generation` passes every
  non-decoder `model_kwarg` to an encoder whose `forward` accepts `**kwargs`, which `HyenaEncoder.forward` does (line
  522). So `model.generate(input_ids, evidence_tracks=T, ...)` fuses the tracks into `encoder_outputs.last_hidden_state`,
  which `prepare_inputs_for_generation` (line 1024-1061) re-feeds at line 1052 on every cached step; the stray kwarg is
  absorbed by its `**kwargs` (line 1035).
- **Day-3 test, fail closed.** A forward hook on `HyenaEncoder` records whether `evidence_tracks` arrived non-None
  during `generate()`; with a random (non-zero) projection, cached generation must equal a no-cache teacher-forced argmax
  on the same prefix. If the SIF's `transformers` filters the kwarg (hook sees `None`) the test **fails and the MVP does
  not start**; no fallback injects tracks through `encoder_outputs`. Versions go into the run identity:
  `apptainer exec $SIF python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"`.
- Padding: `hyena_collate_fn` LEFT-pads sequences (datasets.py:726-729); tracks are left-padded with zeros to the same
  `max_len`, so padded bases get `E_seq(pad) + 0`. Decoder, tokenizer, GSF v3 target, grammar-constrained decoding,
  tiling/stitching unchanged. New parameters: 8 × 768 = 6,144.

## 4. Dataset, collate, trainer edits

- `isoformDataHyena.__init__` (datasets.py:175-248): new kwargs `species=None` (adds `AND species_id = ?` to the split
  query at line 214-219), `evidence_db=None`, `evidence_k=8`, `aug_seed=None`, `channel_dropout_p=0.5`,
  `thin_fractions=(0.01, 0.1, 0.5, 1.0)`, `eval_fraction=None` (valid: fixed fraction, no dropout).
- `__getitem__` (line 292-380): after tokenisation (line 358-360), with `region_seq`, `rn`, `region_start`,
  `region_end`, `strand` in hand (line 312), draw `u = torch.rand(2)` from the DataLoader worker RNG (seeded per run by
  `torch.manual_seed(seed)`, train_HyenaTransgenic.py:238: reproducible per seed, varies per epoch, separate from the
  sampler's stream); tile dropped if `u[0] < p` → zeros; else `f = thin_fractions[int(u[1] * 4)]` and
  `tracks = rasterise_est(...)`. Return a 9-tuple `(input_ids, attention_mask, labels, gm, chr, start, end, tx_count,
  tracks)` (line 378-380). Deviation from §2.8: withholding is not in the main runs (one source; §5.4).
- `hyena_collate_fn` (line 707-745): accept 9 elements (the 8-element branch at line 718-724 stays for old callers);
  left-pad each `[L, 8]` with zeros to `max_len` (`F.pad(t, (0, 0, max_len - L, 0))`), stack to `[B, max_len, 8]`
  float16, return it as element 9.
- Trainer loop (train_HyenaTransgenic.py:512-527): `tr = batch[8].to(device, non_blocking=True)`;
  `model(input_ids=ii, attention_mask=am, labels=lab, evidence_tracks=tr, return_dict=True)`.
- **Paired validation** (replaces the eval loop at line 660-670): at the end of the run, every A. thaliana valid tile
  is run twice, with zeros and with its **real** fraction-1 tracks from `tile_est` (§1.3). With `--batch-size 1`
  (b5_runtime.py:263) `outputs.loss` is the per-tile loss; each rank writes `(rn, has_evidence, loss_off, loss_on)` to
  `eval_pairs.rank<i>.jsonl`, the reduced means are logged as `eval_loss_denovo` / `eval_loss_evidence` (line 679-688
  pattern), and §5.2 uses the tiles with `has_evidence = true` (`mvp_qc.valid_tiles_with_evidence` pairs). No early stop.
- `--max-steps N` in `parse_args` (b5_runtime.py:256-272): `t_total = N` (train line 345) so the schedule spans the
  budget; `break` out of both loops when `global_step == N` after line 543; then the paired validation + final save.
- `--init-from <dir>` distinct from `--resume`: load `model.safetensors` from `runs/seed123/best` (symlink to
  `epoch_04`, b5_runtime.py:152-156; written at train line 432-433) with `strict=False`, re-zero the projection,
  **fresh optimizer and scheduler** (epoch 4's `accelerate_state/` holds the old LR schedule and nothing for the new
  parameter; `accelerator.load_state`, line 379, is not used). Run dirs `runs/mvp_seed{1,2,3}`, own `run_config.json`
  (line 768-776).
- Compile: keep `torch.compile` on as in seed 123 and always pass a tracks tensor (zeros when dropped) so the graph
  signature is constant; on a Dynamo fallback to eager (`suppress_errors`, train line 317-318) rerun with
  `TRANSGENIC_NO_COMPILE=1` and re-derive N from a 20-step `--benchmark-steps` run (b5_runtime.py:268).

## 5. Training and measurement

### 5.1 Fine-tune recipe (per seed; seeds 1, 2, 3; `aug_seed = 20260914 + seed` for the blake2b ranks)
Recipe `configs/b5_400m_win_v3.json` (encoder 768 × 12, effective batch 96, `--batch-size 1`): on 4 GPUs the per-rank
accumulation is 24 micro-batches per optimizer step (train line 226-228); at the measured 1.4 it/s per rank (seed 123
link 1; re-measured on day 4) one step ≈ 17 s, so 6 GPU-h ≈ 315 steps gross. **N = 250 optimizer steps** (24,000 tile
samples ≈ 2–3 passes over the A. thaliana train tiles), leaving ≈ 20 % for compile warm-up, paired validation and the
save: ≈ 5.5 GPU-h per seed, ≈ 16.5 for three, plus < 1 GPU-h of decoding ≈ 18 GPU-h; §5.4 adds ≈ 5.5 if needed.
Data: `split = 'train' AND species_id = 'Athaliana'` tiles only (all tiers, RC rows included); valid = the A. thaliana
valid tiles. Optimiser AdamW, weight decay 0.02, grad-clip 1.0, bf16 (train line 219, 329, 545), two parameter groups:
backbone peak LR 2e-5 (seed 123 was at ≈ 4.3e-5 of a 5e-5 linear decay at epoch 4 of 22; halving it protects the
sequence-only behaviour), `evidence_proj` peak LR 2e-4 (a zero-initialised layer needs a larger step to leave zero);
linear warm-up 5 % (12 steps) then linear decay to 0 at step 250 (train line 347-353). Channel dropout p = 0.5,
thinning {0.01, 0.1, 0.5, 1} uniform. Loss = L_GSF only (the existing weighted CE, modeling line 957-970).
Logged per step (wandb offline + `train.err`): loss, both LRs, grad norm, `‖W_proj‖_F` (must leave 0; if it stays
< 1e-3 the channel is being ignored and the result is uninformative), fraction of micro-batches with evidence on, mean
`cov > 0` fraction of the on-tiles, tokens/s; at the end `eval_loss_denovo`, `eval_loss_evidence`, the pair count, and
the run identity (git commit, SIF sha256, **transformers and torch versions**, checkpoint sha256, C0-lite DB sha256,
`b6_panels_v1.tsv` sha256, p, fractions, N, LRs, seed, aug_seed).

### 5.2 Readouts (all three computed for every seed; positive means evidence helps)

For loss, **Δ_loss = loss_off − loss_on** (a reduction is an improvement). For F1, precision and recall,
**Δ_metric = metric_on − metric_off**. Never describe Δ_loss as loss_on − loss_off.
1. **Primary, sensitive**: `Δ_loss = mean(loss_off − loss_on)` over the valid tiles with evidence (§4), in nats.
   Paired **tile** bootstrap: resample tiles with replacement, 2,000 replicates, `numpy.random.default_rng(20260914)`, a
   tile carrying its three seeds' pairs; per seed and pooled; one-sided 95 % bounds = 5th / 95th percentiles of the
   replicate means. Minimum worthwhile effect **δ_loss = 0.005 nats**.
2. **Structural, independent**: junction precision and recall (§2) for fraction 1 and fraction 0 on the 50 loci;
   `Δ_P`, `Δ_R` with a paired **locus** bootstrap (same replicates and seed). No δ; it is the contradiction check and
   the availability condition of §5.3.
3. **Secondary, reference-dependent**: pooled transcript-level F1 on the 50 loci (§2), `Δ_F1 = F1(on) − F1(off)`,
   paired locus bootstrap as above. Minimum worthwhile effect **δ_F1 = 2.0 percentage points**. Core-20 stratum
   reported alongside.
Decoding: each of the 50 windows twice per seed (grammar-constrained, beam as in the B1 benchmark), 300 decodes on one
GPU, under an hour. The seed-123 `best` model on the same windows is a fourth column (fraction 0 should sit near it).

### 5.3 Decision rule (futility rule; verbatim for A46 rule 17)
> Primary improvement is Δ_loss = mean(loss_off − loss_on), pooled over the valid tiles with evidence and three
> seeds, so positive values indicate improvement. Its one-sided 95 % bounds use the paired tile bootstrap
> (2,000 replicates, seed 20260914). Evaluate prospective **STOP** first: the primary upper bound is below
> δ_loss = 0.005 nats, neither the secondary readout (reference-dependent transcript F1 on all 50 loci,
> δ_F1 = 2.0 percentage points) nor the independent structural readout (junction precision/recall against locally
> aligned A-ONT1/A-HiFi reads) contradicts it, and the assessable library-withholding check of §5.4 does not reverse
> the primary bound. A contradiction is a one-sided 95 % lower bound above 0 on ΔF1, or above 0 on ΔR with ΔP not
> below 0; these deltas are evidence-on minus evidence-off. Reversal means the sensitivity check's primary upper
> bound is ≥ δ_loss; in that case STOP fails. **GO** is evaluated only if STOP fails: the original primary lower
> bound must be above 0 and all three seed-level Δ_loss point estimates must be positive. Otherwise the decision is
> **INCONCLUSIVE**. Apply availability safeguards afterward: if the primary futility condition holds, no available
> readout contradicts it, and the withholding check has not demonstrated reversal, missing structural scoring or
> an unassessable withholding check downgrades any prospective STOP to **INCONCLUSIVE**, not GO. In particular,
> unrecoverable library identifiers do not justify STOP, and if A-ONT1/A-HiFi reads are not locally aligned at the
> panel loci by the end of day 5, any prospective STOP is downgraded to INCONCLUSIVE. Thus primary bounds
> [0.001, 0.004] nats with positive seed estimates and satisfied safeguards yield STOP, not GO. A contradiction
> alone never establishes GO; the primary GO criteria must still hold. INCONCLUSIVE does not stop the programme:
> WP0–WP7 proceed subject to their own approval and gates, and the MVP is reported as uninformative at this budget.

**A47.2 application note (2026-09-22; the quoted rule above is unchanged and is not re-quoted here).** Under A47.2
the clause “locally aligned A-ONT1/A-HiFi reads” is read as **A-ONT1 and A-CCS23**, A-CCS23 filling the PacBio arm
(A-HiFi still contributes nothing and is reported missing per source). The availability safeguard binds only if A-ONT1
or A-CCS23 is not locally aligned at the panel loci by the end of day 5; then any prospective STOP is downgraded to
INCONCLUSIVE, not GO. If both are aligned and the structural readout is assessable and does not contradict the
primary bound, STOP is reachable. GO is unaffected. The A47 note below is superseded where it differs.

✏️ *Superseded by the A47.2 note above:* **A47 application note (2026-09-21; the quoted rule above is unchanged and is not re-quoted here).** The decision text
is reproduced verbatim in `PROTOCOL_B1_frozen_v1.md` A46 rule 17 and must stay byte-identical; this note records how
its existing clauses resolve under A47, and adds no new criterion.
- “the independent structural readout (junction precision/recall against locally aligned A-ONT1/A-HiFi reads)” is
  computed **from A-ONT1 only**. A-HiFi contributes nothing and is reported missing per source.
- The availability safeguard is the operative clause: because the A-HiFi structural arm is permanently unavailable,
  **any prospective STOP that depends on it is downgraded to INCONCLUSIVE, not GO.** A STOP remains reachable only if
  the A-ONT1-only structural readout is genuinely computed, is assessable, and does not contradict the primary bound.
  If A-ONT1 is not locally aligned at the panel loci by the end of day 5, the existing day-5 clause already downgrades
  any prospective STOP to INCONCLUSIVE.
- GO is unaffected: it is evaluated only if STOP fails and still requires the primary lower bound above 0 with all
  three seed-level Δ_loss estimates positive. A missing structural arm never creates a GO.
- §5.4 is unaffected.
Power: the per-tile loss difference is continuous and paired, so with a few hundred valid tiles its SE is far below
δ_loss and a null is informative. Transcript F1 at one locus is near-binary (paired SD ≈ 0.5): 50 loci detect only
≈ 20 percentage points at 80 % power, 20 loci ≈ 30; F1 is therefore secondary and can only contradict, never decide.

### 5.4 Library-withholding sensitivity (STOP path only)
Assessed only on the provisional futility path, before any STOP. Seed 1 is re-run with library withholding at rate
0.1 (a kept tile drops all molecules of one library, chosen per tile from the worker RNG) if `est_molecule` carries
an auditable library key; repeat the paired validation and record whether the primary upper bound remains below
δ_loss. A written argument may replace the run only if inspected provenance establishes the equivalence of
library withholding to the implemented augmentation for these data. One EST source, missing library metadata, or
inability to derive keys from accessions does **not** establish that equivalence: library-correlated dropout need
not behave like molecule thinning. Record the inspected metadata and justification; if no run or substantiated
argument is possible, mark the check **unassessable** and the decision **INCONCLUSIVE**, not STOP.

## 6. One-week task list

| day | task | est. | done when |
|---|---|---|---|
| 1 | `71_b6_panels.py` → `b6_eligible_loci_v1.tsv`; `72a_est_molecules_per_locus.sh` + `72_mvp_select_loci.py` → `b6_panels_v1.tsv` (pronghorn BAM); `mvp_build_c0lite.py` (pysam, §1.3: dedup → exclusion → `tile_est` for train **and** valid tiles → `mvp_qc`) | 1 d | both TSVs + sha256; `mvp_qc.retained_overlapping_panel == 0`; dedup and valid-tile counts printed |
| 2 | `src/transgenic/evidence/mvp_tracks.py` (`rasterise_est`, transcription-order donor/acceptor, RC swap, N-zeroing, blake2b thinning); unit tests: minus-strand junction identity after RC (§1.4), nested draws (`keep(0.1) ⊂ keep(0.5)`), zeros for a dropped tile, timing on a 129-kb tile | 1 d | tests pass; DB copied to DeltaAI `/work/db/`, sha256 matches |
| 3 | Model + config + dataset + collate + trainer edits (§3–4); tests: step-0 bitwise equality with seed-123 `best` (3 tiles), cached generation equals no-cache argmax with the encoder hook asserting the tracks kwarg arrived (**fail closed**), 9-tuple collate shapes; record transformers/torch versions | 1 d | all tests pass on GB10 (1 GPU, `TRANSGENIC_NO_COMPILE=1`) |
| 4 | DeltaAI 4-GPU smoke: `--max-steps 20 --benchmark-steps 20` in the seed-123 SIF/checkout; compile does not fall back, memory < 110 GB/rank, it/s re-measured; fix N if it/s differs from 1.4 | 0.5 d + queue | benchmark JSON archived; N confirmed |
| 5 | **Required**: locus-restricted A-ONT1 **and A-CCS23** alignment (§2, pronghorn CPU; reads must be local) → `A-ONT1.panel.bam` + `A-CCS23.panel.bam` + read-supported junction table (**A47: the A-HiFi leg is dropped — no FLNC product exists and none is produced, so `A-HiFi.panel.bam` is not expected; A47.2: the PacBio leg is A-CCS23, `splice:hq`**); in parallel the three fine-tunes (`SEED=1,2,3`), one 4-GPU link each | 1 d + queue | `A-ONT1.panel.bam` and `A-CCS23.panel.bam` indexed, junction table written — else the STOP branch is closed for good; `TRAINING_DONE` ×3 with `eval_pairs` |
| 6 | Fine-tunes finish; decode 300 windows (§5.2); F1 and junction P/R scoring | ≈ 1.5 wall-h each + queue | `eval_pairs.*.jsonl`, `eval_loss_*`, identity file; decoded GFFs ×300 |
| 7 | Bootstraps for the three readouts, decision per §5.3; if the primary bound points to STOP, the §5.4 run or argument; write `B6_MVP_RESULT_v1.md` (exploratory label; first line names the decision and whether the structural readout existed) | 1 d (+ 5.5 GPU-h on the STOP path) | result file with GO/STOP/INCONCLUSIVE and the QC table |

Panel commands (existing scripts, shown for reproducibility; the existing TSVs are not regenerated by this revision).
Run `71` on the lab GPU host holding the frozen corpus and flag files, or use its supported `--remote gpu` mode:
```
python revision/scripts/71_b6_panels.py --remote gpu --seed 20260914 --min-exons 2 --overlap-flank 0 --out revision/protocols/b6_eligible_loci_v1.tsv
```
Its defaults bind the corpus and A22/A30 flag paths and checksums documented in `71_b6_panels.py`. After copying
that TSV to pronghorn, use the actual positional arguments / CLI option names of `72a` and `72`:
```
bash revision/scripts/72a_est_molecules_per_locus.sh revision/protocols/b6_eligible_loci_v1.tsv "$ROOT/evidence/est_align/Athaliana/Athaliana.est.bam" revision/protocols/b6_est_molecules_v1.tsv
python revision/scripts/72_mvp_select_loci.py --eligible revision/protocols/b6_eligible_loci_v1.tsv --est-counts revision/protocols/b6_est_molecules_v1.tsv --max-len 20000 --min-est 3 --mvp 50 --core 20 --pilot 300 --out revision/protocols/b6_panels_v1.tsv
```
Remaining MVP commands are **implementation sketches**, not runnable interfaces yet. Pronghorn CPU:
```
python revision/scripts/mvp/mvp_build_c0lite.py --bam ... --panels revision/protocols/b6_panels_v1.tsv --db b5_full_a40_v1.db --flank 150 --out $ROOT/evidence/mvp/athaliana_est_c0lite_v1.db
bash revision/scripts/mvp/mvp_align_panel_longreads.sh --panels revision/protocols/b6_panels_v1.tsv --ont A-ONT1 --hifi A-CCS23   # day 5, §2 (A47.2: PacBio leg = A-CCS23)
```
DeltaAI (same SIF and a second frozen checkout, as train_b5.slurm:31-35 requires; one link per seed, no chain needed):
```
sbatch --gpus-per-node=4 --export=ALL,SEED=1,REPO_DIR=$WORK/repo_mvp,MVP=1 deploy/deltaai/train_b5.slurm
# inside the link (train_b5.slurm:133-135 extended when MVP=1):
accelerate launch --num_processes=4 --mixed_precision=bf16 train/train_HyenaTransgenic.py \
  --config configs/b5_400m_win_v3.json --db /work/db/b5_full_a40_v1.db --seed $SEED --output-dir /work/runs/mvp_seed$SEED \
  --species Athaliana --evidence-db /work/db/athaliana_est_c0lite_v1.db --evidence-k 8 --channel-dropout 0.5 \
  --thin-fractions 0.01,0.1,0.5,1 --init-from /work/runs/seed123/best --max-steps 250 --lr 2e-5 --proj-lr 2e-4 --save-every-n-steps 0
python revision/scripts/mvp/mvp_decode_panel.py --run /work/runs/mvp_seed$SEED/final --panels ... --evidence-db ... --fractions 0,1
python revision/scripts/mvp/mvp_score.py --pred ... --ref <manifest gff> --panels ... --longread-bams A-ONT1.panel.bam,A-CCS23.panel.bam --coverage-policy revision/scripts/mvp/coverage_policy_a47_2.json --eval-pairs runs/mvp_seed*/eval_pairs.*.jsonl --bootstrap 2000 --seed 20260914
```

## 7. Explicitly out of scope (deferred to WP0–WP7)
Full C0 schema and the audit alignment (§2.2, §2.4b); full A11 molecule semantics (§1.3 dedup is a stand-in); the
≥ 121-nt arm; phase, protein, ONT and PacBio input tracks; withholding outside §5.4; the repaired-mask corpus and the
C2–C6 controls; dose-response beyond {0, 1}; multi-species tracks and QC; compile/DDP hardening and the resume
whitelist; A46 freezing; promotion of MVP numbers to the manuscript. Seed 123 and the frozen B1/B5 protocols are untouched; the §2 panel alignment is a locus-restricted MVP artefact, not the B1 validation pass.
