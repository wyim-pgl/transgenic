# PROTOCOL B1/B4 — Frozen protocol for independent transcript-evidence validation of completion-mode additions (v1.29; v1.0 text unchanged, amendments appended; §A18 effective-rule matrix is the operational reading of §3–§9 + amendments; §A19 protein resource; §A20 class-specific source weights; §A21 vector screening; §A22 annotation-quality loss masking; §A23 evidence-based primary isoform; §A24 grammar-constrained decoding; §A25 variable-context new-version recipe on ACCESS; §A26 whole-window labels, GSF v3; §A27 tiled inference and caps; §A28 job-chain checkpointing; §A29 block splits and leakage masking for tiles; §A30 Swiss-Prot sensitivity set: caution-based masking, phase audit, S5–S7; §A31 held-out loci protected by leak masking, no block forcing; §A32 gene-level masking of hard-flagged genes in tiles; §A33 overlapping blocks, locus-aware stitching, masking parameters from reference statistics; §A34 partially pretrained encoder: audit, gate and record; §A35 a failing batch stops the run, no silent skips; §A36 EST length floor raised to 121 nt, **superseded by §A37**; §A37 the floor returns to 100 nt and 121 nt becomes a labelled sensitivity arm; §A38 a run excluded before collection is declared in the manifest instead of in prose, and the long-read scope is refrozen; §A39 the §3.3 item 2 `-uf` gate is the READ's orientation, not the splice motif's — the literal reading was blind to library strandedness)

**Version 1.0 — frozen 2026-09-01.** Status: FROZEN before any evidence read is downloaded or aligned. Amendments are allowed only (a) before the first evidence alignment is run, or (b) for defects that make a rule inapplicable, and must be logged in §12 with date, reason and the state of the analysis at that moment. Nothing in §§3–9 may be changed after the first result table is produced.

Scope: TransGenic 400M reference-prompted completion additions in training-withheld *Zea mays* (primary) and *Arabidopsis thaliana* (replication), evaluated against independent ONT and PacBio Iso-Seq reads (equal first tier) and EST/full-length cDNA (secondary tier). Tomato is conditional (§4.4). This document covers plan items B1 (frozen-filter out-of-sample test), B4 (orthogonal transcript evidence) and the evidence rules reused by B3/C0. Windowing rules for B7 and the re-serialization control C1 are frozen in separate documents.

Wording rule for the manuscript and response letter (frozen): the results of this protocol are **orthogonal transcript evidence**. They are never described as validation equivalent to isoform-specific RT-PCR. Predictions without support remain computational hypotheses.

---

## 1. Frozen objects (identity recorded at freeze)

| Object | Path (root `/data/gpfs/assoc/pgl/data/Transgenic`) | MD5 | Size (B) |
|---|---|---|---|
| Maize genome (Phytozome RefGen_V4 = AGPv4) | `genomes/Zmays_493_APGv4.fa` | `a8d3069cd9554885670848cc3df185cb` | 2,170,684,968 |
| Maize reference annotation | `genomes/Zmays_493_RefGen_V4.gene_exons.exon.gff3` | `f62364903053e0fb433f6b0b80ed4df9` | 525,831,779 |
| Maize completion prediction (400M, reference-prompted; 3 records/locus, first record kept) | `transgenic_comparison/standardized_results/Z_mays_transgenic400Mprompt.gff3` | `08b45a51e9116f2901b447b9787c7544` | 283,877,445 |
| A. thaliana genome (TAIR10, revision copy) | `transgenic/revision/data/TAIR10/TAIR10_genome.fa` | `987ca803466e79f98b7f06af8ca94557` | 121,662,600 |
| A. thaliana reference | `transgenic/revision/data/TAIR10/TAIR10.gtf` | `c6ccc302afda2a40f729b655288413ef` | 48,386,343 |
| A. thaliana completion prediction (beam1, deduplicated) | `transgenic_comparison/standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3` | `670d419b887271e0a9b8a4d82b9bf700` | 50,403,784 |
| A. thaliana primary transcript IDs | `transgenic/revision/data/TAIR10/primary_transcript_ids.txt` | `8442eed7f20e879ebd1928a615fe0747` | 520,904 |
| **Dataset-role manifest, EST scope (A18.3)** | `evidence/DATASET_ROLES.est_v1.tsv` | `a6375713ca2321c6a0cc55acc8e1d7fe` | 3,252 |
| **Dataset-role manifest, long-read scope (A18.3)** | `evidence/DATASET_ROLES.longread_v1.tsv` | `c2f4c2e8fe68b3e522abb8298e1999ff` | 62,511 |
| **Cross-species protein resource (A19)** — OrthoDB v12 Viridiplantae, leakage-filtered stage 2 | `orthodb_filtered_stage2/odb12_Viridiplantae.filtered.fa.gz` (pgl-gpu `/home/pgl/scratch1/wyim/transgenic_data/`, outside the root above) | `453cb32b02e0799950d7d5f4de5f62ac` | 3,178,195,033 |

Scoring code frozen at repository commit `21e8752` (tag `revision-2026-09`): `28_score_added_isoforms.py` (blob `e98a0fe89158`), `36_filter_additions_structurally.py` (blob `595fc2539905`), `48_score_zmays_additions.py` (blob `e3e09ccce760`). New code written under this protocol (per-structure dump for maize, species-agnostic evidence scorer `59_evidence_support.py`) may add outputs but may not change the definitions in §2.

**Dataset-role manifest, scoping (author decision 2026-09-03).** A18.3 and the precedence matrix of A15
require `evidence/DATASET_ROLES.tsv` to be checksummed here before the first alignment. It is frozen in two
scopes rather than one, because the two bodies of evidence complete at different times. The EST scope is
frozen now: all eleven species finished downloading on 2026-09-01 and cannot change, and the EST alignment
consumes nothing else, so every run it touches carries a role. Long-read collection was still running when
this was written, so a single freeze would have gone stale within the hour; the long-read scope is frozen
separately as `DATASET_ROLES.longread_v1.tsv` before the first long-read alignment, and neither scope may be
used for alignment before its own row appears in the table above.

**Long-read scope frozen 2026-09-04**, once collection actually finished — which took establishing that it
had. The tree looked complete (no process running, 122 `.DONE`, 150 GB) while seven *A. thaliana* runs were
missing: `longread_fetch.sh` counted DONE and skipped but not FAILED, so a dataset whose every run had failed
signed off `0 runs DONE, 0 skipped` and exited 0. All seven were recovered (each verified against its
ENA-published `fastq_md5`, and each converted record count equal to the published `read_count`) after the
fetcher was made resumable; the accounting defect is fixed in the same file. The scope holds **215 rows** —
129 `c2_training_eligible`, 31 `b1_validation_only`, 55 `excluded` — and the excluded rows are kept on
purpose: a freeze that records only what was retained cannot be audited against what was rejected. **One run is absent by decision rather than by scan**: *O. sativa* `SRR25203456` (PRJNA953663, 13,195,758 ONT RNA-Seq reads) was excluded by the author on 2026-09-04 (issue #68) because ENA's published path is a directory and reaching NCBI's mirror would require an SRA fallback in the fetch driver. It carries a **declared** manifest row rather than a scanned one (A38): the builder emits it from `DECLARED_RUNS` and marks it `DECLARED_NOT_SCANNED`, because a tree scan can only tell *here* from *not here* and cannot tell *never considered* from *considered and rejected*. The v1.27 freeze `d644278f6a18a8b2b2162c8378cf61a8` (61,801 B, 214 rows) recorded the reason in this prose only and is superseded. Both are generated by
`revision/scripts/make_dataset_roles.py`, which refuses to write a manifest whose roles are incomplete,
ambiguous, or contain a run present in two dataset paths without a declared canonical copy — the check that
caught the Cui 2020 (PRJNA594286) double download, whose non-canonical copy is `excluded` by author decision
of the same date. The full-scope manifest at the time of the EST freeze held 187 rows
(sha256 `24a85b41f5f8cc5cb1614efdd0cd2bcfbf153ebaf4c739450b16317cf6e8254a`); `source_checksum` is the
ENA-published `fastq_md5` and the md5 of our converted FASTA is carried separately as `local_fa_md5`.

**Protein resource, recorded late (2026-09-03, issue #66).** A19/#44 required the filtered OrthoDB
FASTA's counts and md5 to be recorded here at the database freeze. The freeze happened (#50) and this
row did not, so the protein label resource that fed the frozen B5 build was not identified by checksum
anywhere the protocol could point at. The md5 above was **recomputed from the file** on 2026-09-03, not
copied from the build's own `filter_summary.json`, and matches it. The resource holds **12,115,085**
sequences over 408 taxa, filtered from 12,204,762; an independent check found **zero** sequences from
the three evaluated species (taxids 3702 / 4577 / 4081), which stage 1 had already removed
(`removed_by_taxid: 0`, and `counts_by_taxid.tsv` carries no rows for them). The file lives on pgl-gpu
rather than under the root above, which is why its path column is qualified. ⚠️ The matching row in
`evidence/DATASET_ROLES.tsv` is still missing: that manifest is generated by scanning the evidence tree
per sequencing run, and how a non-run resource is represented there is an open question on #66.

Contig names: maize `1 2 3 4 5 6 7 8 9 10 Pt Mt`; A. thaliana `1 2 3 4 5 Mt Pt`. Every evidence dataset is re-aligned to these exact FASTA files. Published BAM/GFF from other releases (TAIR10.1, "B73_RefGen_v4" from NCBI, v5) are never imported; contig aliases (`Chr1`→`1`, `chr1`→`1`, `ChrC`→`Pt`, `ChrM`→`Mt`) are applied only to *our* alignments' reference names and the alias table is stored with the results.

## 2. Frozen definitions (unchanged from the manuscript)

- **Locus**: one reference gene identifier; evaluation loci are those at which a completion prediction exists (maize 39,174; A. thaliana 27,413).
- **Supplied primary**: the first transcript of the reference gene in file order (script 48 convention for maize; `primary_transcript_ids.txt` for A. thaliana).
- **Addition**: a predicted transcript whose CDS coordinate set differs from the supplied primary's; identical CDS structures within a locus are collapsed to one structure. Maize: 3,363 additions; A. thaliana: 1,103.
- **Reference match**: exact-CDS match to a reference alternative transcript at the same locus (primary removed); intron-chain match reported alongside (script 28 definitions).
- **Frozen structural filter (F)**: an addition passes iff its assembled CDS (a) starts with ATG, (b) ends with a stop codon, (c) has length divisible by three, (d) contains no internal stop, AND (e) every intron of the transcript has GT..AG termini. Script 36 predicates, no parameter changes. Applied to maize **before** any evidence is viewed; the A. thaliana result of this filter (266/1,103 retained; 71.1%/72.2%) is the in-sample reference point.
- **Novelty class of an addition** (relative to the reference at its locus): `junction-novel` = at least one intron interval absent from every reference transcript at the locus; `combination-novel` = every intron present in some reference transcript but the ordered chain matches none; `reference-alt` = exact-CDS or intron-chain match to a reference alternative. Class is computed from the reference only, before evidence.

## 3. Evidence datasets (equal first tier: ONT and PacBio Iso-Seq)

Independence has two separate columns and both are reported: **model-independent** (the model never saw the data: true for all sets below) and **reference-independent** (the data did not contribute to the reference annotation used for scoring).

### 3.1 Zea mays (primary)

| ID | Accession | Type | Tissue/genotype | Ref-independent (vs RefGen_V4) | Use |
|---|---|---|---|---|---|
| M-ONT | PRJNA822071, runs SRR18571905/06/07/11/14/15 (B73 only) | ONT cDNA | root tips, control vs 24 h cold | yes | tier 1 |
| M-FLNC | PRJEB32007; Zenodo 2611319 `F1maize.INTERMEDIATE.flnc.fastq.gz` + `demux_FL_count.txt` (B73 samples only, demultiplexed) | PacBio Sequel FLNC | root, embryo, endosperm | yes (Wang 2020) | tier 1 |
| M-HQ18 | PRJEB22122 (E-MTAB-5957) HQ isoform fastq (5 tissues on ENA) | PacBio HQ isoforms | leaf, shoot, bract, pericarp, seedling | yes vs V4 (used by V5 only) | tier 1 |
| M-EST | NCBI EST division, `Zea mays[Organism]` (B73-attributed records kept; genotype recorded) | Sanger EST | mixed | **no** (historically used) | tier 2 |
| M-FLcDNA | maize full-length cDNA collection (Soderlund et al. 2009) | FL-cDNA | mixed | **no** (used by V4) | tier 2 |
| excluded | PRJNA10769 (Wang 2016) | Iso-Seq | — | no (V4 evidence) | not used |
| excluded | subreads-only Sequel sets without original BAM (PRJNA822292, PRJNA983493) | — | — | — | not used (CCS not regenerable) |

Optional: raw `subreads.bam` from PRJEB32007 (ERR3261692 etc.) → `ccs` → `lima --isoseq` → `isoseq refine --require-polya` → FLNC; if generated it is treated as part of M-FLNC with its own run field.

### 3.2 Arabidopsis thaliana (replication)

| ID | Accession | Type | Tissue | Ref-independent (vs TAIR10 / vs AtRTD3) | Use |
|---|---|---|---|---|---|
| A-ONT1 | PRJNA1087576, SRR28341943/44/45 (FLIC) | ONT PromethION cDNA-PCR | rosette leaf | yes / yes | tier 1 |
| A-ONT2 | PRJNA594286 (Cui 2020), ONT runs | ONT cDNA | whole rosette | yes / yes | tier 1 |
| A-HiFi | PRJEB77203 ERR13994458 `Col_isoseq.subreads.bam` → ccs/lima/refine | PacBio Sequel II FLNC | rosette leaf | yes / yes | tier 1 |
| A-EST | NCBI EST division, `Arabidopsis thaliana[Organism]` | Sanger EST | mixed | **no** / partly | tier 2 |
| A-RAFL | RIKEN RAFL full-length cDNA | FL-cDNA | mixed | **no** / partly | tier 2 |
| excluded | PRJNA755474 (AtRTD3's own Iso-Seq) | — | — | — | not used |

### 3.3 Pre-alignment audit (must be completed and recorded per run before alignment)

1. Reference identity: MD5 of the FASTA used equals §1; contig alias table written.
2. Library orientation: align 10,000 reads without `-uf`; compute the fraction of spliced alignments whose inferred transcript strand agrees with the annotated strand at confidently annotated multi-exon genes (single-strand loci only). `-uf` is enabled for a run only if agreement ≥ 95%; otherwise the run is aligned without `-uf` throughout.
3. PCR/UMI status from metadata (protocol kit, UMI presence). Without UMIs, counts are reported as **reads**, never molecules, and no duplicate removal by identical endpoints is performed.
4. Run structure: every run is aligned and scored separately; pooled results are reported in addition, never instead.
5. Tool versions, command lines, download dates and file MD5s are written to `revision/results/evidence/PROVENANCE.tsv`.

## 4. Alignment (frozen)

```
# ONT cDNA (orientation not proven)
minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# ONT cDNA (orientation proven ≥95%)
minimap2 -t 32 -ax splice -uf -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# PacBio FLNC / HQ isoforms / HiFi FLNC
minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# Sanger EST and full-length cDNA
minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa est.fa
```

- No annotation-derived `--junc-bed`.
- Only primary alignments (flag 0x900 unset), MAPQ ≥ 20. For maize a sensitivity pass at MAPQ ≥ 10 is reported separately and never substitutes the primary result.
- Supplementary/chimeric alignments are discarded; reads with primary blocks on different chromosomes or strands are discarded.
- Mapper choice is frozen as minimap2 2.28 (`transgenic-revision` env). A one-time comparison against GMAP on 20,000 ESTs is recorded in PROVENANCE but does not change the primary mapper.
- Sorted BAM + PAF with `--cs=long` are retained; SAM is never written to disk.

### 4.4 Tomato (conditional)
Run only if, by the end of week 2, (i) the exact ITAG5.0/SL5.0 FASTA used for the tomato predictions is identified and MD5-matched, and (ii) at least one long-read dataset is genotype-verified (ERR4039883 Heinz 1706; PRJNA961334 Moneymaker with CCS status confirmed). Otherwise tomato is dropped from this protocol without amendment.

## 5. Junction calling (frozen)

From `cs` strings, each `N`-type gap in a spliced alignment is a candidate intron (donor = first intronic base, acceptor = last intronic base, 1-based inclusive, strand from the alignment `ts` tag or the `-uf` setting).

| Source | Anchor each side | Mismatches in anchor | Indel exclusion zone | Coordinate correction | Minimum support |
|---|---|---|---|---|---|
| ONT | ≥ 20 nt | ≤ 2 per 20 nt | none within 6 nt of the boundary | up to ±3 nt to a **unique** canonical motif (GT..AG on the alignment strand); raw and corrected coordinates both stored; ambiguous corrections rejected | ≥ 3 reads after filtering; "high confidence" requires ≥ 2 runs |
| PacBio FLNC / HQ / HiFi | ≥ 20 nt | ≤ 2 per 20 nt | none within 6 nt | up to ±1 nt to a unique canonical motif | ≥ 2 reads; "high confidence" ≥ 2 runs or ≥ 3 reads |
| EST / FL-cDNA | ≥ 15 nt | ≤ 1 per 15 nt | none within 8 nt | none | ≥ 2 distinct clone/accession IDs; single-record support stored in a separate column |

A predicted intron is **supported** only when both its donor and its acceptor coincide with a supported junction's corrected coordinates on the same strand. Support of one splice site alone does not count.

Non-canonical predicted introns (not GT..AG) are scored with the same rules but flagged; they are excluded from the frozen filter by construction (§2 F).

## 6. Chain support (frozen)

An evidence read gives **complete intron-chain support** to an addition when: same strand; the read's ordered list of (corrected) introns within the addition's CDS span is identical to the addition's intron list; the read contains no additional intron between the addition's first donor and last acceptor; and the alignment extends ≥ 20 nt upstream of the first donor and ≥ 20 nt downstream of the last acceptor. This is reported as "complete intron-chain support", never as "full-length transcript support". Terminal exons, UTRs and CDS start/stop are not part of the criterion (they are scored by the reference match in §2).

**Combination-novel additions** (§2) count as supported only with complete-chain support from a single read; support of each constituent junction on different reads is recorded as "constituent-junction support" and is not chain support.

A read that lacks upstream introns (5′ truncation) supports only the junctions it spans; it can give complete-chain support only if it spans the whole chain as defined above.

## 7. Callability and denominators (frozen)

Per addition, computed per evidence source and for the union:
- `chain_callable`: ≥ 1 aligned read (any structure) spanning from first donor − 20 nt to last acceptor + 20 nt on the same strand.
- `junction_callable`: for every novel junction of the addition, ≥ 1 aligned read spanning donor − 20 to acceptor + 20 (irrespective of splicing).
- `locus_covered`: ≥ 1 aligned read overlapping the locus on the same strand.
- `uncovered`: none of the above.

Every rate is reported twice: over **all additions** and over **callable additions** (chain rates over `chain_callable`, junction rates over `junction_callable`). Uncovered or non-callable additions are never counted as negative evidence.

## 8. Evidence dimensions and tier (frozen)

Each addition receives, per source and for the union:

| Dimension | Values |
|---|---|
| `chain_support` | complete / partial (≥ 1 supported junction but not complete) / none |
| `novel_junction_support` | all / some / none / N/A (no novel junction: reference-alt or combination-novel) |
| `callability` | chain-callable / junction-callable / locus-covered / uncovered |
| `source` | ONT / PacBio / EST / FL-cDNA |
| `annotation_independence` | independent / historically used by the reference / unknown |

Summary tier for tables (derived, not a substitute for the dimensions): T1 complete-chain support by a tier-1 (ONT/PacBio) read; T2 complete-chain support only by tier-2 (EST/FL-cDNA); T3 all novel junctions supported but no single read spans the chain; T4 some junctions supported; T5 callable, unsupported; T6 not callable; N/A combination-novel without single-read support is reported in T4/T5 with the constituent flag.

## 9. Pre-registered outcomes and success criteria

Primary (maize, tier-1 union; each also per source):
- **P1** Complete-chain support rate of the 3,363 additions, all and chain-callable, Wilson 95% CI; reported for unfiltered additions and for the frozen-filter survivors.
- **P2** Out-of-sample filter test: exact-CDS precision vs RefGen_V4 alternatives before (7.9%, 266/3,363) and after F; matches retained by F; recall change.
- **P3** Novel-junction support rate among junction-novel additions (all / junction-callable).
- **P4** Positive control: the reference's own alternative transcripts at the same loci (n ≥ 500, sampled with a fixed seed 123) scored with identical rules — the expected support rate of true isoforms under this evidence.
- **P5** Negative control: for each addition, one decoy chain built by shifting every intron by a fixed +9 nt (kept canonical only if the shifted motif is canonical; otherwise the decoy is flagged) scored identically — the false-support floor.

Replication (A. thaliana): P1–P5 with A-ONT1/2 and A-HiFi; comparison with the AtRTD3-based 18.5% is descriptive only (AtRTD3 is a catalogue, not reads).

Frozen success criteria for the manuscript:
- The filter is called **validated out of sample** iff, in maize, (a) filtered complete-chain support (callable denominator) ≥ 2 × unfiltered, AND (b) filtered exact-CDS precision vs RefGen_V4 ≥ 3 × unfiltered (i.e., ≥ 23.7%), each with the CI excluding the unfiltered point estimate. Otherwise the filter is reported as **species-specific** and the A. thaliana 71% stays labeled as post hoc.
- A headline change (abstract lead) is made only if maize filtered complete-chain support ≥ 30% of callable survivors AND the positive control P4 ≥ 50% (evidence adequate). Otherwise the abstract keeps the current wording and the results are added as a Results paragraph plus Table S12.

## 10. Analysis order and blinding

1. Write and unit-test the evidence scorer on synthetic data (chains with known support).
2. Freeze this document; commit; record commit hash below.
3. Generate the maize per-structure dump (script 48 + dump flag) and apply F. Record F outcomes (counts only) before any evidence download.
4. Download evidence; run §3.3 audits; align; call junctions.
5. Run P4 and P5 controls first and record them.
6. Score the additions. Produce the frozen tables (§11). No rule changes after this point.
7. Replicate in A. thaliana.

Anyone building the scorer does not look at addition-level evidence overlap before step 6; intermediate BAM inspection is limited to the audit metrics in §3.3.

## 11. Frozen output tables

- `Table S12a` maize: per source and union, all/callable denominators, P1–P5, unfiltered vs filtered, Wilson CIs.
- `Table S12b` A. thaliana replication, same layout.
- `Table S12c` evidence datasets with both independence columns, runs, read counts after filtering, orientation test result, PCR/UMI status.
- Per-addition long table (CSV in `revision/results/evidence/`) with all five dimensions per source.

Manuscript wording (frozen): "We assessed whether predicted intron chains or their constituent junctions were observed in independent transcript-sequencing datasets. Complete-chain support required one alignment carrying the full ordered intron chain; junction support assembled across different reads was reported separately and does not establish the predicted combination. Uncovered loci were excluded from the callable denominator. This provides orthogonal transcript evidence, not experimental validation of transcript expression in the biological material used for prediction."

Response-letter wording (frozen): "We agree that isoform-specific RT-PCR would provide targeted experimental validation. Such experiments were not feasible during this revision. Instead, we added a preregistered analysis of independent ONT, PacBio Iso-Seq, EST and full-length cDNA datasets, with explicit mapping-quality, anchor-length, junction-coordinate, read-support and callability criteria fixed before the data were examined. We describe these results as orthogonal transcript evidence rather than RT-PCR validation, and predictions without such evidence remain labeled as hypotheses."

## 12. Amendment log

## A17. Amendment v1.7 (2026-09-01; before any long-read alignment) — ORF completeness is decided after alignment on genome-spliced sequence; ORF-incomplete chains are chain-training evidence

Author questions (2026-09-01): "how is a complete ORF checked before mapping?" and "can ORF-incomplete reads be used for chain training?". Adopted rules:
- **No ORF verdict before alignment.** Read-level ORF scans are not a judgement of completeness: strand is unknown for ESTs and half of ONT cDNA, ONT indels and single-pass EST errors break frames, and a read cannot show whether an upstream in-frame ATG exists. ORF and CDS assignment (design §4) run only on the **genome sequence spliced through the aligned exon chain**, in the three transcript-oriented frames, never on the read sequence.
- **Optional pre-alignment triage (QC only, never a label or a filter for scoring).** A read may be tagged `cds_candidate` when a DIAMOND blastx hit to a cross-species proteome covers ≥ 70 % of the protein and reaches within 5 aa of both its first and last residue; the hit strand, an untemplated poly(A) and detected 5′ adapter/primer may be recorded. FLNC/"full-length" library tags mean primer presence, not ORF completeness. These tags feed the completeness QC table (A16) and read prioritisation; they never change eligibility, tiers or denominators.
- **ORF completeness and chain completeness are separate axes** (extends A16 orthogonality). Chain evidence requires only that one molecule witnessed the full ordered chain (`IC`, 20-nt anchors), independent-molecule support (§A11/A16) and a training-species, non-held-out, non-`validation_only` source. Non-coding transcripts, NMD-like isoforms and UTR-only full-length reads are legitimate chains.
- **Allowed uses of ORF-incomplete evidence** (adds rows to design §5):

| State | intron-chain training target (Track C1) | C0 junction / partial-chain support | C2 positive mask | GSF label |
|---|---|---|---|---|
| chain complete, ends incomplete (`IC` + 5I/3I) | yes | yes | yes | no |
| chain complete, both ends complete, no complete ORF (incl. `nmd_like`, non-coding) | yes | yes | yes | no — the GSF vocabulary requires CDS features and phases; a non-coding GSF representation is a format change and out of scope for this revision |
| partial chain (`IP`) | no (sub-chain evidence only, tiers T3/T4) | yes (junctions, blocks) | yes | no |
| conflicting / ambiguous (`IX`, `mapping_ambiguous`) | no | family-level only | no | no |

- **Chain-label weighting (Track C1 and C2).** Chain and junction labels carry `W = 1 + source_weight·log1p(independent_molecules)` (capped) and require canonical splice sites; retained-intron-like chains supported by fewer than 3 independent molecules are down-weighted, not removed. Rationale: the benchmark is CDS-based (GFFCompare), so non-coding chains must not dominate the chain objective.
- This revision's scope rule is unchanged: evidence *selects* (B1–B3) and *teaches only the C2 segmentation head*; ORF-incomplete chains as direct training targets belong to Track C1 (`PROTOCOL_M_exploratory_v1.md`, gated) and the follow-up evidence-derived-label work. Nothing in §9 changes.

## A18. Amendment v1.8 (2026-09-02; before any long-read alignment) — corrections to A17, effective-rule matrix, dataset-role manifest, frozen C2 weights and class semantics, genotype rule for all species, seed plan

Source: full-scope Codex re-review (`codex_full_review_20260902.md`, thread 01a060a9), each item cross-checked against the cited lines by the author's assistant before adoption; two Codex claims were rejected (A8 does not prescribe `-k14 -w5`; `est_repair.sh` deduplicates by accession ID, only its header comment was stale).

### A18.1 Corrections to A17 (supersede the corresponding A17 sentences)
- The clause "chain evidence requires … a training-species, non-held-out, non-`validation_only` source" applies **only to training-side use of evidence (C2 masks; any future chain objective)**. For the B1/B4 validation analyses (§6–§9) chain support is taken from the designated validation datasets of the test species and of the *A. thaliana* validation-only sets, exactly as §3 and A14 specify; these rows never enter training, reranking or candidate generation.
- "Track C1" in A17 and design §5a was a misnomer. **C1 is reserved for the reference-label canonical-order control folded into B5** (master plan §C1/v3.4). Evidence-derived chain training targets are renamed **"follow-up chain objective" and are not active in this revision**; the A17 table column is to be read as "representation capability — inactive". The revision-wide rule stands: evidence selects (B1–B3) and teaches only the C2 segmentation head.

### A18.2 Effective-rule matrix (operational precedence; historical §3–§9 text is not rewritten)
| Source | Mapper (authoritative) | Junction correction for **primary** outcomes | Molecule unit | Support threshold (units) | TES/internal priming |
|---|---|---|---|---|---|
| ONT cDNA / direct RNA | §3 `splice -k14 -G 200000` (`-uf` only after the §3.3 strandedness audit ≥ 95 %) | §5: ≤ ±3 nt to a **unique canonical motif**; raw + corrected stored. The design-document rule (correction onto an annotated or ≥ 3-unit junction) is a **labelled sensitivity analysis only** | A16: PCR-equivalence unit (same library, strand, corrected chain, both ends within 10 nt); direct RNA = one read; UMI when present | §5 counts read as **units**: ≥ 3 units; high confidence ≥ 2 runs or ≥ 5 units | A3: reject if downstream 20 nt has ≥ 12 A **or any run ≥ 6 A** |
| PacBio FLNC / HQ / CCS | §3 `splice:hq -G 200000` | §5: ≤ ±1 nt to a unique canonical motif | source FLNC/ZMW molecule, never a polished cluster | ≥ 2 units; high confidence ≥ 2 runs or ≥ 3 units | A3 (same) |
| EST / FL-cDNA | §3 `splice:hq -G 200000` (no clustering, A8/A11) | §5: none | A11: accession stem (versions merged; clone ID when present; same library + identical alignment signature otherwise), 10 per library cap | A11: ≥ 2 independent molecule units (replaces "2 clone/accession IDs" of §5) | A3 (same) |
Precedence: A16 > §3.3 item 3 for ONT counting; A11 > §5 EST support column; A3 (6-A run) > design-document §2 (8-A run) — the design document is amended to 6. The 20-nt-window ≥ 12 A rule is unchanged. This matrix and `evidence/DATASET_ROLES.tsv` are checksummed into §1 before the first alignment.

### A18.3 Dataset-role manifest (fail-closed)
`evidence/DATASET_ROLES.tsv` lists every dataset **and every run** with exactly one role: `b1_validation_only` | `c2_training_eligible` | `excluded` (+ species, genotype stratum, instrument, data type, expected files, expected read count, source checksum). Builders (C0 ingestion, C2 label generation, B1 scorer) refuse any run without a role. Consequences now recorded:
- `ont/Athaliana_cui2020_PRJNA594286` (downloaded by the validation driver) has role `c2_training_eligible`; it is **excluded from the A. thaliana B1 replication**, which uses only FLIC, Zhong 2025 and Zhang 2023 (A14).
- Zhong 2025 (subreads.bam) must be converted (ccs → lima → refine, versions and checksums recorded) **before** the *A. thaliana* replication is scored, or be recorded as unavailable in `DATASET_ROLES.tsv` before alignment; the replication is then FLIC + Zhang 2023 only.
- Wang 2020 FLNC (Zenodo 2611319): the B73-only selection required by §3.1 is implemented from `demux_FL_count.txt` and the FLNC headers before scoring; the all-genotype file may be scored only as the separately labelled pooled stratum (A7-P).

### A18.4 Frozen C2 label weights and class semantics (before any C2 label is built)
- `W = min(4, 1 + source_weight · log1p(independent_molecules))` per positive cell. `source_weight`: protein-homology CDS mask 1.0; PacBio Sequel II+/Revio FLNC/CCS 1.0; ONT 0.8; EST 0.6. Genotype multiplier on `source_weight`: reference genotype 1.0; known non-reference cultivar or hybrid/pooled 0.5; unknown 0.5.
- **Retained-intron-like** (operational): an aligned block that fully spans an intron which has ≥ 3 independent-molecule junction support in the same species, while the spanning block itself has < 3 independent molecules → weight 0.25 on the exon class for those bases. Not removed.
- Class semantics: protein CDS alignments label **CDS-family classes only** (CDS with phase). EST/long-read aligned blocks label the **transcribed-exon family** only; the CDS-vs-UTR distinction inside a block is set only where the block lies in a model with an assigned CDS (design §4), otherwise those cells get weight 0 (neutral) on that distinction. Junction evidence labels the **intron class over the intron interval and the donor/acceptor boundary classes**, nothing else. Every remaining cell is weight 0, never a guessed positive. The 14-class list of `preprocess.py:550–581` is enumerated with its allowed evidence state in the C2 label specification before label construction.
- Loss: per-cell weighted cross-entropy normalised by the sum of weights per window; no additional source balancing. A **unit-weight ablation** (all W = 1) is a mandatory reported control.

### A18.5 Genotype-aware acceptance for every species (generalises A7-P beyond maize)
For all training-species evidence (TRAINING_EVIDENCE_v1.md admits non-reference cultivars and hybrids): primary alignment, MAPQ ≥ 20, aligned query fraction ≥ 0.80, alignment identity over aligned blocks ≥ 95 % (ONT, EST) or ≥ 98 % (CCS/FLNC); equal-best multi-locus placements → `mapping_ambiguous` (family-level counts only, no label); junctions require canonical motif and both anchors (§5). Genotype stratum (reference / known non-reference / hybrid-pooled / unknown) is recorded per library and applied as the A18.4 multiplier; C2 labels from non-reference strata are reported separately in the completeness QC table.

### A18.6 Seed plan and reporting (B5)
Primary seed 123 selects the reported checkpoint (best validation loss, patience 3). Seeds 456 and 789 are confirmatory and **must finish before submission**; main-text numbers come from seed 123, the three-seed mean ± s.d. is reported in the supplement. If only two GPUs are available, seed 789 may run after seed 456 but not be omitted.

### A18.7 Terminology and B7 labelling
- Reference-derived rows carry `assumed_from_reference = true` and are never pooled with read-derived 5C/3C rates.
- Schema names: `read_internal_code` (IC/IP/IM/IX) and `model_internal_code` (MC/MP/MM/MX) everywhere.
- The B7 whole-chromosome comparison is an **analysis-held-out diagnostic**, not a training-held-out test; the true B5 test-locus subset is reported separately and species-level out-of-training evidence comes from maize.

### A18.8 Ingestion hardening required before the §3.3 audit is considered complete
Implementation items (recorded in `IMPLEMENTATION_ORDER_B5_C0_C2_v1.md` §2a): enumerate every ENA file per run (not the first URL), verify source `fastq_md5`, reconcile converted read counts with metadata, filter by parsed columns not whole-row regex, expected-run manifest with `DONE | EXCLUDED(reason)`, strict shell mode and atomic renames for Zenodo/SRA paths, keep sources until validation passes, long-read `PROVENANCE.tsv` with tool versions/commands/dates, completion records carrying script and manifest hashes; EST: per-record sequence validation (non-empty, legal alphabet, terminal newline) and an accession manifest compared as a multiset, not counts alone. Existing downloads are re-audited by a separate checker before use; drivers are not edited while running.

Nothing in §9 changes.

## A19. Amendment v1.9 (2026-09-02; before any protein alignment) — cross-species protein resource and the protein-to-label pipeline

Author decision: the cross-species protein evidence named in A14/A18 is **OrthoDB v12, Viridiplantae partition** (release, file names and md5 recorded in `evidence/DATASET_ROLES.tsv` at download; downloaded on the ACCESS Delta allocation, where alignment runs). No other protein database is used for labels in this revision; Swiss-Prot may be added later only as a separately labelled sensitivity set.

### A19.1 Leakage filter (applied before any alignment, frozen)
- Remove every sequence whose OrthoDB organism is an evaluated species: *A. thaliana* (taxid 3702), *Z. mays* (4577), *S. lycopersicum* (4081), including subspecies/cultivar taxids under them.
- For the nine training species, remove sequences that belong to genes in `test` orthogroups of `data/splits/b5_orthogroup_split_v1.tsv` (mapping by the species' own gene IDs where OrthoDB carries them; unmapped sequences of a training species are kept only if the species' reference proteins were not used to build them, otherwise excluded).
- Record counts per taxid before/after filtering in the provenance table. The filtered FASTA's md5 is frozen in §1.

### A19.2 Alignment and structure calls
- Tool: **miniprot** (protein-to-genome spliced alignment; version and command recorded), one run per training genome, default splice model for plants, `--outs=0.97 -N 20` style parameters fixed at freeze; DIAMOND blastx is used only for the pre-alignment read triage of A17 (`cds_candidate`), never for labels.
- A protein alignment is **accepted** when identity ≥ 30 %, protein coverage ≥ 70 %, no frameshift, all introns canonical (GT-AG, GC-AG) and ≤ 200 kb; equal-best multi-locus placements are `mapping_ambiguous` (family-level only).
- Independent support unit = distinct OrthoDB **organism**; `n` in the A18.4 weight is the number of organisms whose accepted alignments cover the base.

### A19.3 What protein evidence labels (C2) and what it does not
- Labels **CDS-family classes only** (`protein_coding_gene`, `exon` over aligned CDS blocks, CDS phase from the alignment frame) with `source_weight = 1.0` and the A18.4 genotype/cap rules; splice donor/acceptor and intron classes are labelled from protein alignments **only where ≥ 2 organisms place the same intron boundary exactly**, otherwise weight 0.
- Protein evidence never produces UTR labels, GSF transcripts, or B1 support counts; it is not a B1 evidence source and does not touch §6–§9.
- Protocol M candidate pool: accepted miniprot structures (after A19.1) are the protein-derived candidates of Protocol M item N; that remains gated/follow-up.
- Evidence transcript-model phase 2 (design §4): accepted alignments provide the homology tier for ORF assignment (E ≤ 1e-10, identity ≥ 30 %, coverage ≥ 70 %).

### A19.4 Order of operations
1. Downloads finish on pronghorn → Delta account active → OrthoDB v12 Viridiplantae downloaded on Delta, filtered (A19.1), md5 frozen.
2. `data/splits/b5_orthogroup_split_v1.tsv` committed (#14) → training-species held-out orthogroup proteins removed.
3. miniprot per training genome (parallel with B5 training) → accepted-alignment table (`evidence/protein_alignments/<species>.tsv`) with per-base organism counts.
4. At the week-3 C2 gate (#32): if go, protein CDS masks + EST/long-read exon/junction masks → C2 labels/weights (#33) → 14-class trainer (#34).
5. Test-species genomes are never aligned against the resource for training purposes; a maize/tomato alignment is run only inside Protocol M evaluation.

Nothing in §9 changes.

## A20. Amendment v1.10 (2026-09-02; before any C2 label is built) — class-specific source weights: EST leads for introns and splice boundaries

Author decision: intron junction labels weight EST evidence highest. Rationale: Sanger ESTs place splice boundaries at base precision (no homopolymer indel error), whereas ONT cDNA needs ±3 nt correction and protein alignments infer boundaries indirectly; for transcript ends the order is reversed (EST 5′/3′ ends are unreliable), so the A18.4 weights stay for exon/UTR classes.

`source_weight` in `W = min(4, 1 + source_weight · genotype · log1p(n))` now depends on the label class family:

| class family | EST | PacBio Sequel II+ FLNC/CCS | ONT | protein (miniprot) |
|---|---|---|---|---|
| intron, splice_donor, splice_acceptor | **1.0** | 0.9 | 0.7 | 0.5 (only where ≥ 2 organisms agree, A19.3) |
| exon, 5UTR, 3UTR, protein_coding_gene (A18.4 unchanged) | 0.6 | 1.0 | 0.8 | 1.0 (CDS-family only) |

Junction `n` for EST = independent molecule units per A11 (clone-merged, library-capped), never raw accessions. Everything else in A18.4 (cap 4, genotype multiplier, retained-intron 0.25, neutral cells, unit-weight ablation) is unchanged. Implemented in `gsf_contract.evidence_weight(source, n, genotype, retained_intron, family)`.

## A21. Amendment v1.11 (2026-09-02; before any EST alignment) — UniVec vector/adapter screening of EST records

Author decision: every EST set is screened against **NCBI UniVec_Core** before alignment (A8 ingestion filters gain this step; the "aligned portion only" rule of A8 does not protect against internal vector segments that create false junctions in chimeric clones).
- Tool and parameters (frozen, VecScreen): `blastn -task blastn -db UniVec_Core -reward 1 -penalty -5 -gapopen 3 -gapextend 3 -dust yes -soft_masking true -evalue 700 -searchsp 1750000000000`; UniVec_Core build/date and md5 recorded (`evidence/univec/UniVec_Core.version`, `.md5`).
- Categories by raw score and position (terminal = match within 25 nt of either end): strong ≥ 24 terminal / ≥ 30 internal; moderate ≥ 19 / ≥ 25; weak ≥ 16 / ≥ 23.
- Actions: terminal strong or moderate → trim the matched end and everything outboard; internal strong → suspected chimera: the record is **split at the match** and each piece ≥ 100 nt is kept as `<accession>_partN` (pieces remain one molecule unit for A11 counting); internal moderate and all weak → flagged only; records < 100 nt after trimming → dropped.
- Outputs per species: `est/<species>/univec/est.univec.fa.gz` (+ md5), `report.tsv` (accession, length, action, kept ranges, categories), `summary.json`, `PROVENANCE.txt` (input md5, UniVec version/md5, blastn version, parameters, trimmer version). Only the screened FASTA is aligned; counts (kept/trimmed/split/dropped) go to the completeness QC table (A16).
- Implementation: `revision/scripts/evidence/univec_screen.sh` (chunked, resumable) and `revision/scripts/61_univec_trim.py` (tests in `revision/scripts/tests/test_univec_trim.py`).

## A22. Amendment v1.12 (2026-09-02; before the B5 database build) — annotation-quality flags (GeenuFF) and loss masking of partial or erroneous gene models

Author decision: before B5 training, each of the nine reference annotations is imported with **GeenuFF** (Helixer's annotation database; version pinned in the container and recorded) and its error features are used to mask unreliable training labels.
- Flags: every GeenuFF error feature is attributed to its super-locus (gene) and transcript and stored verbatim (`qc/<species>.geenuff_flags.tsv`, `revision/scripts/62_geenuff_qc.py`). **Hard** flags (name matching `missing_start`, `missing_stop`, `wrong_starting_phase`, `mismatched_ending_phase`, `overlapping_exon`, `too_short_intron`, `empty_transcript`, `empty_super_locus`, other phase mismatches) mark partial or internally inconsistent gene models; **soft** flags (missing UTRs) are recorded only.
- Loss-mask policy (implemented in the builder, `build_b5.loss_mask_decision`): a hard flag on the gene, or on every transcript → the row is kept in the database with `train_weight = 0` and is **excluded from the training and validation loaders** (masked from the loss; it stays available for auditing and, unchanged, in the test split); hard flags on some transcripts → those transcripts are removed from the GSF label, the row keeps `train_weight = 1` and records `transcripts_dropped`; soft flags → no change. RC rows inherit the decision. For the C2 segmentation head the same rows receive weight 0 over their whole window.
- Reporting: counts of masked rows and dropped transcripts per species are part of `build_manifest`/the validator output (`rows_loss_masked`) and are reported in the supplement next to the split sizes. The flag file's md5 is frozen with the database; the policy is never changed after a training run has started.
- Rationale: partial gene models teach the decoder to emit truncated CDS and wrong phases; masking them removes that noise without altering the reference annotation used for evaluation (evaluation is unchanged and still uses the full GFF).

## A23. Amendment v1.13 (2026-09-02; before any evidence scoring) — evidence-based primary isoform from EST/long-read chain support

Author decision: chain connectivity in the evidence tables defines an **evidence-primary isoform** per locus, used for evaluation, QC and the B3 reranking channel only (never as a training target in this revision).
- Inputs: C0 `chain_junction`/`partial_chain` tables with independent molecule units (A11/A16 units; A21-screened ESTs; sources weighted per A20 junction table).
- Per transcript *t* of a reference or predicted locus: `full_chain_units(t)` = independent molecules that witness the complete ordered intron chain of *t* (IC reads only); `bottleneck(t)` = minimum over its introns of the junction unit count; mono-exonic transcripts: units of reads fully inside the exon with both ends within it.
- Ranking (frozen): (1) `full_chain_units` descending, (2) `bottleneck` descending, (3) CDS length descending, (4) canonical order (`gsf-order-v1`) as the final tie-break. The top transcript is the evidence-primary; a locus with `full_chain_units = 0` for every transcript has **no** evidence-primary (reported as "uncalled", never defaulted to the annotation's primary).
- Reporting: per species, the fraction of loci with an evidence-primary, agreement with the annotation's primary/representative model, and for TransGenic predictions the fraction whose top-ranked (first canonical) transcript equals the evidence-primary — a secondary outcome (S4), stratified by source (EST-only / long-read / both) and by EST 3′ bias (ESTs alone cannot call the primary for loci whose alternative introns lie outside EST coverage; such loci are "uncalled").
- Caveats recorded with the outcome: EST tissue and 3′ bias, partial chains contribute only to `bottleneck`, and cultivar strata (A18.5) are reported separately.

## A24. Amendment v1.14 (2026-09-02; before any B5-era inference) — grammar-constrained decoding as a pre-registered inference arm

Author decision: TransGenic inference gains a **grammar-constrained decoding** arm that forbids, token by token, the error classes an unconstrained autoregressive decoder can produce: coordinates outside the window or reversed, feature list not coordinate-sorted, more than one strand, wrong phase symbol for the feature type, transcript members that are undefined, repeated or out of transcription order, a transcript without CDS, and a transcript count that disagrees with the `<txN>` plan. Frame consistency (CDS length mod 3), which cannot be forced token by token without deadlocks, stays with the frozen structural filter F and is reported.
- Implementation: `src/transgenic/utils/gsf_grammar.py` (allowed-next-token set and `validate_gsf` audit; tests `tests/test_gsf_grammar.py`), `src/transgenic/model/constrained_decoding.py` (HF `LogitsProcessor`), `src/run_genome_annotation.py --constrained` (deterministic beams; sampling off).
- Evaluation rule (pre-registered): every B5-era benchmark (Fig. 5, additions, B2, B7) is run **both** unconstrained (recipe parity) and constrained; the constrained arm is reported alongside, with the per-class counts of violations that `validate_gsf` finds in the unconstrained output. The headline numbers use the arm named in advance here: **constrained decoding is the primary inference mode for the B5 model**; the unconstrained arm is the parity control. The published-checkpoint results already in the manuscript are not re-decoded.
- Out of scope (follow-up): base-wise probabilities → HMM/CRF constrained decoder → GFF (Helixer/Tiberius-style structured decoding) on top of the C2 head.

## A25. Amendment v1.15 (2026-09-02; before the B5 database build) — B5 becomes the new-version training with variable context; compute on NSF ACCESS

Author decisions: (1) the retrained model is released as a **new version**, so B5 no longer reproduces the published recipe; it uses the same architecture and optimisation (`configs/b5_400m_ctx_v2.json`) with **variable context windows** of 30,720 / 61,440 / 129,024 nt (6,144 × 5 / 10 / 21; the largest exceeds Helixer's 106,920-nt plant context). (2) All B5 seeds train on the **NSF ACCESS allocation** (Delta A100 80 GB or DeltaAI GH200 96 GB); the lab RTX 4090 is used for inference, B7 and development only, because 129,024-nt windows at 400M are not expected to fit 24 GB.
- Window rule (`tier6144-v2`, docs/gsf_spec_v1.md §1a): smallest tier holding the gene plus ≥ 1,000 nt flank on each side; training-build augmentation moves a gene to the next tier with probability 0.3 at a seeded random offset; genes that do not fit 129,024 − 2,000 nt are rejected and counted. Labels remain the single target gene of the window (the whole-window multi-gene labelling is a separate decision, not taken here).
- Consequences stated prospectively: the comparison "published checkpoint vs B5" now confounds leakage removal with the context change; the manuscript reports B5 as the new version and lists the published-recipe result as the legacy comparator. An optional parity control (`sym6144-v1`, published recipe on the leakage-controlled DB, one seed) is run **only if** ACCESS credits remain after the three new-version seeds; it is a supplementary row, not a headline.
- Everything else in A18.6 (seed 123 primary; 456/789 confirmatory; best validation loss, patience 3; ≤ 22 epochs; effective batch 96) is unchanged. #18 benchmarks the three tiers separately and records tokens/s and peak memory per tier before the seeds start.

## A26. Amendment v1.16 (2026-09-02; before the B5 database build) — whole-window labels: every complete gene in a window (GSF v3)

Author decision: the new version is trained on **genome tiles** whose label contains **every gene fully inside the tile** (docs/gsf_spec_v1.md §1b), which makes prompt-free (de novo) whole-genome inference a native mode instead of a per-locus completion. Rules:
- Tiles of 30,720 / 61,440 / 129,024 nt per contig with a seeded offset per tier (training builds); a gene is labelled only in tiles that contain it completely; edge-crossing genes are excluded and counted. Inference uses overlapping tiles and keeps a gene from the tile in which it is complete (stitching rule to be fixed in the inference amendment before B7).
- Label grammar: gene blocks in coordinate order joined by `<gene>`; `<empty>` for tiles without a complete gene (10 % of empty tiles kept, seeded). Vocabulary v3 (290 tokens); label cap 4,096 tokens; 64 genes per window; decoder positions 4,096.
- Leakage: a tile inherits the most restrictive split of its member genes (test > valid > train) and strict held-out membership; `window_genes` records membership so that no test-orthogroup gene sequence is ever labelled in a train tile. Tiles containing a hard-flagged gene (A22) carry `train_weight 0`.
- Grammar-constrained decoding (A24) is extended: `<empty>` only as the first token, `<gene>` only after a complete block, numbering and strand reset per block, blocks non-overlapping and ordered.
- Consequences stated prospectively: this is a further departure from the published recipe (A25); the published-recipe row is a legacy comparator only. The per-locus prompted mode remains available for the additions/B1 analyses by passing single-gene windows.

## A27. Amendment v1.17 (2026-09-02; before any B5-era whole-genome inference) — tiled inference, stitching rule, and v3 caps from tile statistics

- **Caps** (from the 2026-09-02 tile statistics of A. thaliana, O. sativa and G. max reference annotations): 129,024-nt tiles hold up to 80 genes (A. thaliana; p95 42), and 38 % of A. thaliana 129-kb tiles would exceed a 4,096-token label. The v3 caps are therefore **96 genes and 8,192 tokens per window**, decoder positions 8,192. Tiles over either cap are rejected and counted (expected < 0.5 % after the change). Empty tiles (G. max: 45 % at 30 kb) are kept at 10 %.
- **Tiled inference**: every tier over the genome at three offsets (0, ⅓, ⅔ tier). A predicted gene is accepted from a tile only if both its ends are ≥ 1,000 nt from the tile edges (mirrors the training-time exclusion of edge-crossing genes). Identical predictions (same canonical signature in genome coordinates) merge; overlapping non-identical predictions are resolved by (1) distance from the nearest tile edge, (2) larger tier, (3) fewer grammar-audit violations, (4) canonical order. All resolutions are logged per locus for the B7 report.
- **Per-locus prompted mode** (single-gene window, published behaviour) remains a separate, reported mode for the additions/B1 analyses (A26).
- The stitching rule is frozen here before any whole-genome run; changing it after seeing B7 results requires a new labelled arm.

## A28. Amendment v1.18 (2026-09-02; before any B5 seed starts) — job-chain checkpointing on ACCESS (48-hour wall-clock limit)

- Every seed runs as a **chain of SLURM jobs** on the same run directory: `--resume auto` restarts from the newest of the last completed epoch (`epoch_NN/accelerate_state`) and the mid-epoch `latest_state` (saved every `--save-every-n-steps` optimizer steps, default 200, and on the SIGUSR1 that SLURM sends 15 minutes before the limit); the job resubmits itself with `--dependency=afterany` until `TRAINING_DONE` exists, never after `TRAINING_FAILED` or a non-zero exit, and at most `CHAIN_MAX` (default 8) times.
- `latest_state` holds model, optimizer, scheduler and RNG state plus epoch/step/global_step and the early-stopping state, written atomically (`.tmp` → rename); a completed epoch directory supersedes it. Mid-epoch resume skips the already-consumed micro-batches of that epoch, so the effective sample order of an epoch is unchanged by a restart.
- The first job writes `run_config.json` (recipe, database path, seed, batch and accumulation, max epochs, patience); every later job **refuses to resume** if any of these differ. Code changes inside a chain are forbidden; a changed recipe is a new labelled run.
- The per-epoch rsync watcher (deploy/deltaai/sync_watch.sh) restarts with each job and uses `.synced` markers, so the chain never copies an epoch twice.

## A29. Amendment v1.19 (2026-09-02; before the B5 database build) — tile splits by genomic block, leakage masking by N-replacement

Found in the first GPU dry run: with whole-window labels (A26) a tile inherits the most restrictive split of its genes, so a 129-kb tile (about 29 genes in *A. thaliana*) almost always contains a valid or test orthogroup gene and the training set collapses. Replacement rule:
- **Block splits.** Each contig is cut into consecutive blocks of 1,032,192 nt (8 × 129,024); blocks are assigned train/valid/test 75/10/15 by a seeded generator per species; blocks overlapping a strict held-out locus are test. A tile's split is the most restrictive split among the blocks it overlaps. Block assignments are stored (`tile_blocks`) and frozen with the database.
- **Orthogroup leakage masking.** Inside a tile, any gene whose orthogroup split (#14) is more restrictive than the tile split (a test or valid gene in a train tile, a test gene in a valid tile) is **removed from the label and its sequence, with 100 nt of flank, is replaced by N**, so the model never learns "no gene" over a real gene and never sees a held-out gene's sequence in training. Counts are recorded (`leak_masked=n` in `qc_flags`). Genes whose orthogroup split is less restrictive than the tile (a train gene inside a test tile) stay labelled; test-tile evaluation therefore reports metrics both over all labelled genes and over test-orthogroup genes only.
- The per-gene orthogroup split table remains the source of truth for gene-level analyses (B1, additions, strict held-out loci); block splits only decide which tiles may be used for training.

## A30. Amendment v1.20 (2026-09-02; before any protein alignment, before the B5 database build, before any evidence scoring) — Swiss-Prot reviewed Viridiplantae as a separately labelled sensitivity/audit set for start codon, stop codon and phase

Author decision (2026-09-02): the manually curated UniProtKB/Swiss-Prot entries are used to **audit and validate** start codons, stop codons and CDS phase, never as a label source. A19 is unchanged — OrthoDB v12 Viridiplantae remains the sole protein resource for C2 labels; this amendment is the "separately labelled sensitivity set" that A19 reserved.

### A30.1 Resource, parsing and strata
- Resource: `uniprot_sprot_plants.dat.gz` (UniProtKB/Swiss-Prot reviewed entries, taxonomic division "plants"), downloaded on Delta together with OrthoDB (#44); release, date, size and md5 recorded in `evidence/DATASET_ROLES.tsv` with the role `sensitivity_set` (a fourth role value; builders still fail closed on unknown roles).
- Parser: `revision/scripts/63_swissprot_sensitivity.py`. Viridiplantae entries only (OC lineage). Per entry: accession, taxid, protein-existence level (PE), N-terminal experimental evidence (PE 1 and an `INIT_MET` feature or a `CHAIN` feature starting at residue 1 or 2 with an ECO:0000269 evidence code), SEQUENCE CAUTION types with the sequence records they refer to, gene cross-references (TAIR, Araport, EnsemblPlants, Gramene, KEGG, …), sequence sha256. Outputs: `swissprot_viridiplantae.tsv` and `.fa` (the set), per-species flag files (A30.2), `swissprot_summary.json`; md5s frozen at the database build.
- Strata fixed for every report: **(a)** PE 1 with N-terminal experimental evidence; **(b)** all reviewed entries; **(c)** entries with a structural caution — *erroneous initiation*, *erroneous termination*, *frameshift*, *erroneous gene model prediction*. Rationale: most plant Swiss-Prot sequences derive from the same reference gene models (TAIR10, RAP-DB/MSU, …), so agreement with the reference is not independent confirmation; the independent information sits in (a) and (c).

### A30.2 Training side: loss masking only (A22 mechanism, no label change)
- For each training species, an entry with a structural caution whose cross-references map to **exactly one** reference gene of the same taxid (ambiguous or unmapped entries produce nothing and are counted) is compared with the current reference proteome of that gene. If no protein of the gene equals the curated sequence, the gene receives the hard flag `swissprot_caution_<type>` and is masked exactly as an A22 hard flag (`train_weight = 0`, row kept, excluded from the train/valid loaders, weight 0 for C2). If the reference already carries the curated sequence, the caution was fixed in the annotation and only `swissprot_note_caution_resolved_in_reference` is recorded; without a reference proteome the entry gets `swissprot_note_unverified_no_proteome`. Non-structural cautions (*erroneous translation*, *miscellaneous discrepancy*) are soft `swissprot_note_*`. Soft notes change nothing (A22).
- Over-masking is the intended conservative direction: an isoform-level caution masks the whole gene, and any exact-sequence mismatch (isoform choice, proteome version, non-standard residues) counts as hard rather than resolved. A gene without any protein in the reference proteome is `swissprot_note_unverified_no_proteome` (soft), and a species whose reference proteome maps to fewer than 90 % of its records to gene ids aborts the run (fail closed; identifiers are resolved through the GFF3 transcript→gene map after stripping a Phytozome `.p` suffix). Masked counts per caution type are reported so the cost stays visible.
- The flag file is a second A22-schema input of the builder (`scripts/build_b5_database.py --qc-flags qc/<species>.geenuff_flags.tsv qc/<species>.swissprot_flags.tsv`); masked counts per source and per caution type appear in `build_manifest`/validator output (`rows_loss_masked`) and in the supplement next to the GeenuFF counts. Reference annotations are not edited; evaluation uses the full GFF (A22 rationale).
- Leakage: masking touches training rows only. Entries for strict held-out loci or test-orthogroup genes cannot affect training (those genes are N-masked in training tiles, A29) and never alter the reference used for evaluation. No Swiss-Prot sequence enters the OrthoDB resource, the C2 labels, the Protocol M pools or B1 support counts.

### A30.3 Phase audit (OrthoDB alignments; report only in this revision)
- After #45, every reference CDS segment covered by an accepted miniprot alignment (A19.2) has its GFF phase compared with the alignment frame; disagreements supported by ≥ 2 OrthoDB organisms are written to `qc/<species>.phase_audit.tsv` and reported per species in the supplement (counts, fraction of CDS segments, overlap with GeenuFF phase flags).
- The audit does **not** change `train_weight` in this revision: the alignments run in parallel with B5 training (A19.4) and the loss-mask policy is frozen at the database build (A22). Using the audit for masking belongs to a later training run and needs its own amendment.

### A30.4 Validation side: pre-registered secondary outcomes S5–S7 (sensitivity set)
- Alignment: the sensitivity-set FASTA is aligned with miniprot (A19.2 parameters and version) to the evaluated genomes (*A. thaliana* TAIR10, *Z. mays* RefGen_V4, *S. lycopersicum*) **only inside evaluation** (A19.4 item 5). S5/S6 use A19.2-accepted alignments (identity ≥ 30 %, protein coverage ≥ 70 %, no frameshift, canonical introns) with two further conditions: a start call requires residue 1 of the protein aligned at a genomic ATG; a stop call requires the last residue aligned and followed by an in-frame stop codon within one codon. S7 additionally admits alignments containing frameshifts, restricted to frame calls over their aligned CDS segments. Entries are matched to loci by cross-reference and confirmed by alignment overlap; one entry (canonical sequence) per locus.
- Loci: the *A. thaliana* strict held-out loci (3,429) and, for the test species, the reference loci to which the §2 completion-mode additions belong.
- **S5 start-codon agreement**, **S6 stop-codon agreement**: per stratum (a)/(b)/(c), the fraction of loci where the TransGenic call (B5 model, A24 constrained arm, with the unconstrained arm as parity; the original 400M checkpoint's existing unconstrained predictions alongside) equals the Swiss-Prot-aligned position, the fraction where it equals the reference, and the fraction where the reference annotation's own start (stop) equals the Swiss-Prot-aligned position. **S7 phase/frame agreement**: fraction of CDS segments whose predicted phase equals the alignment frame. Denominators: S5/S6 count loci whose matched entry has an accepted alignment; loci whose entry does not align or fails acceptance are reported as a separate count per stratum, never dropped silently; S7 counts CDS segments covered by an alignment.
- In stratum (c) the reference is known to be wrong at the cautioned feature; the report states per locus whether the model follows Swiss-Prot, follows the reference, or neither. This is the only stratum in which "differs from the reference" is a positive finding.
- Alternative starts are not penalised: a locus whose reference or evidence-primary isoform (A23) uses a start different from the Swiss-Prot canonical sequence is reported in a separate "alternative start" column, not as disagreement.
- S5–S7 are descriptive sensitivity outcomes: they enter no success criterion of §9, are reported in the supplement with denominators and per-stratum counts, and are computed after the primary outcomes are scored (analysis order and blinding of §9 unchanged).

### A30.5 Order of operations
1. #44: download the Swiss-Prot file with OrthoDB on Delta; record role `sensitivity_set`; run script 63 with the nine reference proteomes (`--proteome`) → set table, FASTA, per-species flag files — **before the B5 database build**.
2. #46: build with both flag files; report masked counts per source.
3. #45: after the OrthoDB alignments, the phase-audit table (A30.3).
4. #28 / #35 (evaluation): sensitivity-set alignment to the evaluated genomes and S5–S7, after B5 checkpoints exist and after the primary outcomes are scored.

Nothing in §9 changes.

## A31. Amendment v1.21 (2026-09-02; before the B5 database build) — strict held-out loci no longer force genomic blocks to test; *A. thaliana* stays in training

Found in the first real tile build (*A. thaliana*, tile6144-v3, 2026-09-02): the 3,430 strict held-out loci are spread over the whole genome (28.7 per 1,032,192-nt block on average), so the A29 rule "blocks overlapping a strict held-out locus are test" made 120 of 121 blocks test and left 23 training tiles — *A. thaliana* would have dropped out of training. Author decision: *A. thaliana* must be used for training.
- **Block splits are drawn only.** Every block of every training species is train/valid/test by the seeded 75/10/15 draw of A29; no locus forces a block.
- **Held-out loci are protected by the A29 leakage rule.** Every strict held-out locus is `test` in the orthogroup split (spec §7), so in a train or valid tile its sequence (with 100 nt of flank) is replaced by N and it is absent from the label, exactly as for any other test/valid-orthogroup gene. The model never sees a held-out gene's sequence or structure; it does see the neighbouring genes and the intergenic sequence, which the released model also saw through the windows of neighbouring loci. The strict held-out set therefore remains at least as isolated as in the original evaluation.
- Evaluation of held-out loci is unchanged: per-locus windows (B1, additions, S5–S7) and, at inference, tiles in which the masked positions carry the real sequence.
- Reporting: per species, the number of train/valid tiles with leak-masked held-out genes is part of `build_manifest` / validator output (`leak_masked` counts).

Nothing in §9 changes.

## A32. Amendment v1.22 (2026-09-02; before the B5 database build) — hard-flagged genes are masked at gene level inside tiles

Found in the same tile build: the tile-mode implementation of A22 zeroed the loss of every tile containing a hard-flagged gene; with 690 flagged *A. thaliana* genes (GeenuFF hard flags + Swiss-Prot cautions, A30) that removed 13.8 / 26.4 / 45.5 % of the 30,720 / 61,440 / 129,024-nt tiles. Author decision: mask the gene, not the tile.
- In tile builds, a gene with a hard flag (A22 GeenuFF list, A30 `swissprot_caution_*`) is treated like a leaking gene: its sequence with 100 nt of flank is replaced by N and it is absent from the label; the tile keeps `train_weight = 1` and records `hard_masked = n`. Transcript-level hard flags still drop only the flagged transcripts from the gene's label (A22 unchanged).
- Per-locus (gene-centred) rows are unchanged: a hard-flagged gene's own row keeps `train_weight = 0`.
- The masked genes stay in `window_genes` only when labelled; the per-species counts of hard-masked genes are reported with the split sizes (A22 reporting), together with the leak-masked counts of A29/A31.
- Rationale: the purpose of A22 is to keep wrong structures out of the loss; masking the gene achieves that without discarding the correct genes around it.

Nothing in §9 changes.

## A33. Amendment v1.23 (2026-09-02; before the B5 database build) — overlapping gene blocks, locus-aware stitching, and the masking parameters, all fixed from reference-annotation statistics

Two independent reviews of the first tile build (Codex GPT-5 thread `01a064e6`, Kimi K3) agreed that the non-overlap rule of A26 must go and that the masking parameters proposed for it were unfounded. Every threshold below is fixed **before the database build** from the nine reference annotations, in the same way A27 fixed its caps; no B5 result is used.

### A33.1 Overlapping gene blocks are allowed (supersedes the non-overlap rule of A26 and spec §1b)
- A tile label may contain genes that overlap. **Ordering invariant**: the block key `(gene_start, gene_end, canonical_block)` is non-decreasing across the blocks of a label; `canonicalize_v3()` sorts by that key, so equal starts and nested genes order deterministically by end and then by the canonical string. `check_caps_v3()` enforces the key instead of `gene_start >= previous gene_end`.
- **Grammar** (A24, inference): during generation only `new_gene_start >= previous_gene_start` is enforced. Enforcing the full key while decoding would leave a state in which no block-closing token is legal (equal start, shorter end already emitted) and the beam dies silently; instead the decoded label is canonicalised before parsing, which repairs a tie emitted out of order. `validate_gsf` reports a non-canonical tie as a *notice*, not a violation.
- **Annotation duplicates.** In one tile, blocks of the same strand with an identical CDS signature are collapsed to a single block (the one with the longer gene span; ties by canonical order) and counted as `dup_collapsed`. Measured in the nine references: 665 genes (Ppatens 610, Osativa 55; 70 of them already hard-flagged) are duplicate gene models with an identical CDS signature. Labelling both teaches the decoder to emit near-duplicate blocks, and A27 merges only across tiles.
- Evaluation is unchanged: the reference GFF keeps both duplicates and every overlapping gene.

### A33.2 Locus-aware stitching (replaces the overlap-winner rule of A27)
Two predictions are **the same locus** — and only then does the A27 precedence (edge distance → larger tier → fewer grammar violations → canonical order) choose between them — when all of the following hold; otherwise both are kept as distinct genes:
1. same strand (opposite-strand predictions are always distinct);
2. not co-emitted as separate blocks by the same tile (a tile that emits two blocks asserts two genes);
3. reciprocal CDS overlap ≥ **0.90** (each prediction's CDS bases covered by the other);
4. multi-exon pair: at least one **shared intron** (identical donor/acceptor pair); mono-exon pair: both ends within **1,000 nt**.
Identical canonical signatures are merged first, as before, and need none of these tests.
Evidence for the thresholds: across the nine references there are 4,700 genuinely distinct same-strand overlapping gene pairs (annotation duplicates and already hard-flagged genes removed). Rule 3+4 would merge 104 of them (2.2 %); the alternatives considered were worse — reciprocal ≥ 0.50 merges 191 (4.1 %) and "shared intron alone" merges 200 (4.3 %). The 2.2 % is an upper bound because rule 2 does not apply to reference pairs.

### A33.3 Masking parameters (completes A29/A31/A32)
- **Overlap closure.** Masking is applied to whole overlap-connected components: if any gene of a component is leak-, hard- or decoy-masked, every gene of that component is N-masked and left out of the label. Without this a labelled gene keeps a label whose bases were replaced by N.
- **Flank.** The masked interval of each component is padded by a seeded flank drawn uniformly from **[50, 150] nt** (replacing the fixed 100 nt), so the N-run length does not by itself identify a masked gene's extent.
- **Masked-base fraction.** A train or valid tile whose masked bases exceed **0.60** of the tile is dropped and counted (`mask_fraction_dropped`). Measured on *A. thaliana*, the species with the highest mask load (all 3,430 held-out loci, 43 % test): the masked fraction has median 0.245 / 0.262 / 0.280 and p95 0.538 / 0.477 / 0.436 for the 30,720 / 61,440 / 129,024-nt tiers, so 0.60 sits above the p95 of every tier and drops a few per cent of tiles. A threshold of 0.25 would have dropped 49–61 % of tiles.
- **Decoy masks.** In **train tiles only**, each otherwise labelled gene is masked with probability `min(0.05, m/3)`, where `m` is the species' realised leak+hard mask rate, from a seed `seed:species_id:window_id:decoy`; decoy genes are dropped from the label like any masked gene and counted (`decoy_masked`). Decoys must not enter valid tiles: the checkpoint is selected on validation loss (A18.6) and deleting true genes there would corrupt the selection. Purpose: weaken the association "N-run ⇒ an annotated gene was here". This is an artefact-reduction measure, not a leakage control — at inference nothing is masked.
- Random-sequence replacement is rejected: it can create plausible motifs and hides the intervention, whereas N is also what an assembly gap looks like.

### A33.4 Training and inference edge rules
Training labels every gene that lies completely inside the tile (A26 unchanged); the 1,000-nt edge margin of A27 is an inference-side acceptance rule. The two are consistent up to a computed bound: with offsets spaced tier/3, a gene is guaranteed to lie ≥ 1,000 nt inside some tile of that tier only if its length is ≤ **2·tier/3 − 2,000**, i.e. **18,480 / 38,960 / 84,016 nt** for the three tiers (the interval of admissible tile starts has width tier − length − 2,000 and must contain one of three offsets spaced tier/3). A tile edge that coincides with a contig edge satisfies the margin on that side.
- Measured on the nine references (334,642 genes): 3,418 genes (1.02 %) exceed the 30,720-nt guarantee, 653 (0.195 %) the 61,440-nt guarantee and 86 (0.026 %) the 129,024-nt guarantee; the longest gene is 196,414 nt and cannot be labelled in any tier at all (A26 edge exclusion).
- Those 86 genes are recovered only when they happen to fall inside a tile with margin; they are counted in the build manifest as `tier_margin_unguaranteed` rather than silently lost. A fourth offset would raise the bound to 3·tier/4 − 2,000; it is not adopted for 0.026 % of genes.
- The bound and the contig-edge rule are asserted by a test rather than left implicit.

### A33.5 Reporting
`build_manifest` and the validator report, per species and tier: tiles built, tiles rejected by the gene/token caps, `leak_masked`, `hard_masked`, `decoy_masked`, `dup_collapsed`, `mask_fraction_dropped`, and the masked-base fraction distribution. Cap rejection rates are reported before any training run so that a systematic loss of gene-dense tiles is visible; raising the caps would require a further amendment.

Nothing in §9 changes.

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | pre-download | v1.0 frozen (repository commit 5f7b373) | — |

---

# Amendment v1.1 (2026-09-01, analysis state: EST download in progress; no evidence alignment has been run; no addition-level result viewed)

This amendment adds secondary outcomes, QC gates and clarifications. **§§2, 5, 6, 7, 9 primary outcomes and success criteria of v1.0 are unchanged.** Items were selected from a Codex-reviewed brainstorm (`codex_est_brainstorm_20260901.md`, `codex_candidate_pool_20260901.md`); items judged to require new methodology or to create new discovery sets are explicitly excluded (see §A9).

## A1. Correction of the EST source (replaces the EST rows of §3.1–3.2)

NCBI retired dbEST; `db=nucest` now resolves to whole `nuccore` and returns patent, mRNA and genomic records. The frozen EST query is therefore `db=nuccore`, `txid<taxid>[Organism] AND gbdiv_est[PROP]`, fetched by E-utilities history in 10,000-record FASTA batches (`evidence/est/est_fetch.sh`; provenance in `evidence/PROVENANCE_est.tsv`). Counts at download (2026-09-01): Z. mays 2,019,959; A. thaliana 1,529,700; G. max 1,461,724; O. sativa 1,255,251; V. vinifera 446,853; P. patens 382,587; S. lycopersicum 301,030; S. bicolor 209,835; B. distachyon 206,255; P. trichocarpa 89,943; S. italica 66,027. Per-species `TOTAL` after download may fall short of `COUNT` by suppressed records; the difference is recorded, not corrected. GenBank flat files (`rettype=gb`) are fetched additionally for Z. mays and A. thaliana to obtain `/clone`, `/cultivar`, `/tissue_type`, `/dev_stage` and library identifiers; these fields are retained at ingestion (B) even where not analysed in this revision.

## A2. EST partial chains, clone linking and the splice graph (clarifies §6)

- Each accepted spliced EST alignment yields an ordered **partial intron chain**. Clone-linked 5′/3′ reads (same `/clone` and library, consistent strand, non-overlapping or overlapping-consistent placement, outer span ≤ 200 kb) are merged into one partial chain.
- A per-locus **splice graph** is built with nodes = accepted junctions and edges = adjacency observed on one EST or one linked clone pair. **Graph paths define a candidate space only; they are never molecule support.** Combination-novel additions (§2) remain supported only by a single molecule spanning the whole chain (tier T1/T2 of §8).
- An addition's **adjacency support fraction** = observed adjacent-intron pairs ÷ all adjacent pairs in its chain (0–1) is reported as a secondary, continuous statistic; "all novel junctions supported" (T3) is unchanged.

## A3. UTR and terminus support (new secondary outcomes S1–S3)

- **S1 TES support**: ≥ 2 accessions from ≥ 2 libraries whose 3′ ends carry ≥ 10 untemplated terminal A bases end within ±30 nt of the predicted TES on the same strand; rejected as internal priming if the downstream genomic 20-nt window has ≥ 12 A or any run of ≥ 6 A. Single unpaired EST ends without poly(A) are not TES evidence.
- **S2 TSS lower bound**: the predicted TSS lies within ±50 nt of ≥ 2 5′ ends from oriented full-length/RAFL, cap-selected or CAGE evidence; ordinary 5′ EST ends give a lower bound only ("EST start upstream of predicted TSS: yes/no").
- **S3 UTR-intron support**: predicted UTR introns scored by §5 junction rules, reported separately from CDS introns.
- Results are published as a two-axis table (CDS status × UTR status) and never combined with the primary exact-CDS outcomes.

## A4. Mandatory QC gate before any addition is scored (L)

On a prespecified set of 2,000 ordinary annotated multi-exon loci per species (fixed seed 123, excluding the 3,429 strict held-out A. thaliana loci and all addition loci), report the agreement of accepted EST junctions with reference introns, stratified by anchor length, identity, motif and library. Thresholds in §5 are not re-tuned on this set; if agreement < 95% for GT–AG junctions with ≥ 2 accessions, the pipeline is debugged (alignment, alias table, strand) before scoring additions, and the fix is logged.

## A5. Callability map (C) and reference-completeness audit (H) — secondary

- **C**: per locus and per evidence source: aligned EST bases, splice-bearing accessions, independent libraries, callable junction positions, callable TES/TSS flags. Frozen before any addition is scored; used as the denominator source for §7 and for the B7 whole-genome pilot.
- **H**: per locus, accepted EST junctions absent from the reference annotation (TAIR10 / RefGen_V4), stratified by number of independent libraries and by long-read confirmation. Reported as context for "novel" additions; explicitly labeled as not reference-independent (ESTs contributed to both references).

## A6. Independent-source replication score (N) and saturation (O) — secondary

- **N**: for every junction, `S = min(3, libraries) + min(2, BioProjects) + 0.5·log2(1 + accessions)`; "replicated" requires ≥ 2 libraries. Support is counted from **genome-aligned accessions** only (A8); no sequence clustering is performed. Leave-one-library-out support is reported.
- **O**: library rarefaction at 25/50/75/100% (100 fixed-seed draws) with a beta-binomial detection curve; reports the estimated probability that a real junction of given support class remains unobserved. Used only to calibrate the interpretation of "not observed"; no hard-negative labels are derived in this revision.

## A7. Prespecified QC and sensitivity analyses (P, Q, I)

- **P (maize genotype strata)**: libraries classified B73 / known non-B73 / unknown from GenBank metadata; support reported for B73-direct and species-level separately; a > 5-point difference in support precision between all-maize and B73-only is reported as genotype sensitivity. Non-B73 reads: ≤ 8% divergence, unique placement, ≥ 2 allele-discriminating matches within 100 nt where paralogous loci compete.
- **Q (junction placement ambiguity)**: every accepted junction is re-scored at ±1–6 nt alternative placements; rejected if an alternative is within 2 alignment-score units or both anchors are > 50% masked.
- **I (clone-pair span)**: proportion of uniquely mapped consistent 5′/3′ clone pairs whose outer span exceeds 49,152 nt — a check on the input-window assumption.

## A8. EST ingestion filters (complements §5)

**No sequence clustering (v1.2, author decision): every EST is aligned to the genome individually and only the aligned portion of each record is used; unaligned/soft-clipped bases, unaligned records and identical-sequence records are simply not counted beyond their accession identity.** Exact-duplicate sequences within one accession list are collapsed to one accession only for storage; UniVec screening (trim terminal vector, reject internal or > 20% vector); adapter trimming (overlap ≥ 12, error ≤ 0.10); poly(A/T) trimming only for ≥ 10 bases within the terminal 30 nt (tail call saved before trimming); post-trim length ≥ 100; ≤ 5% N; dust masking (64/20) with rejection at > 50% masked and ≥ 12 unmasked bases in every 15-nt anchor; contamination screen against plastid/mitochondrial/rRNA (exclude when ≥ 80% coverage at ≥ 90% identity beats nuclear by ≥ 10); chimera rules (segments ≥ 75 nt on different chromosomes at MAPQ ≥ 20; > 100 kb apart; incompatible strand/order; junction not reproducible after clipping 10 nt each side and realigning); strand inferred from canonical motifs when ≥ 1 intron exists, otherwise from oriented-library metadata or trusted poly(A); strand-ambiguous alignments are retained for coverage only; multi-locus ESTs (no ≥ 10 score margin or MAPQ < 30 where paralogs compete) support a junction family, never a specific gene; per-library molecule contribution capped at 10 per junction; results additionally reported by gene GC decile. Alignment for ESTs: `minimap2 -ax splice --secondary=no -k14 -w5 -G 200000` (no `-uf`), MAPQ ≥ 20 (≥ 30 for maize paralog-sensitive high-confidence tier), aligned fraction ≥ 0.80, identity ≥ 0.90 (high-confidence ≥ 0.95), intron length 20–200,000 nt, ≤ 8% mismatches+indels overall. A one-time minimap2-vs-GMAP comparison on 20,000 ESTs is recorded in PROVENANCE and does not change the primary mapper.

## A9. Explicitly out of scope for this protocol (follow-up work)

Cross-species junction conservation prior (A); hard-negative labels (D); evidence-constrained decoding (E); APA/alternative-last-exon catalogue (F); unannotated-locus discovery (G); EST-weighted B5 training (J, would break the "split-only" comparability of the retraining); evidence-only isoform sets beyond the partial-chain supplement (K); the prompt-free / evidence-prompt / expanded-candidate-pool experiments (M, N) — these have their own exploratory protocol (`PROTOCOL_M_exploratory_v1.md`) and are not secondary outcomes of B1.

## A10. Amendment log

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | pre-download | v1.0 frozen (repository commit 5f7b373) | — |
| 2026-09-01 | EST download in progress; no alignment run; no addition-level result viewed | v1.1: A1–A9 added; primary outcomes, success criteria and evidence-tier hierarchy unchanged | dbEST retirement discovered during download; Codex-reviewed brainstorm on EST use; author decisions (ONT and Iso-Seq equal tier 1; EST/full-length cDNA as orthogonal transcript evidence, never RT-PCR-equivalent) |
| 2026-09-01 | EST download in progress; no alignment run | v1.2: CD-HIT clustering removed from A6/A8; support counted only from genome-aligned accessions and only the aligned portion of each EST is used | author decision: clustering can merge distinct transcripts and corrupt support counts; genome alignment is the only reduction step |

## A11. Amendment v1.2.1 (2026-09-01; entered before EST alignment; Codex-reviewed replacement for the removed clustering step)

CD-HIT-EST clustering is removed from A6 and A8. Every EST accession is aligned independently to the specified genome assembly with the frozen minimap2 command and alignment-quality thresholds. No sequence cluster, representative sequence, or cluster size is used for alignment, filtering, support counting, evidence tiering, or replication scoring.

**Independent molecule units.** The accession identifier is the accession stem without its version suffix (versions are one accession; the newest parsed version is kept and aliases recorded). When a clone identifier is available, all accessions, versions and 5′/3′ reads from the same normalized physical clone (case/whitespace/punctuation and 5′/3′/forward/reverse suffixes normalized) constitute **one molecule** and contribute at most once to a given junction; a clone pair supports only the blocks and junctions each read aligns across, never an imputed junction or adjacency across the unsequenced gap. When clone identity is unavailable, records from the same biological library are collapsed for counting only if they share the same reference assembly, contig, strand and ordered set of CIGAR-derived aligned genomic blocks (the *exact alignment signature*). Records from different libraries, or with different alignment signatures, are never collapsed. All original records and alignments are retained; no consensus transcript is built and no feature is transferred between ESTs.

**Libraries.** A biological library is one independently prepared cDNA library; runs, plates, read directions, accession series and download batches do not create libraries. Aliases known to describe one preparation are merged. A pooled-genotype preparation is one library (labeled pooled; no reference-genotype-specific allele inference). Identical sequences from independently prepared libraries count once per library; unusually extensive cross-library identity is flagged for sensitivity analysis.

**Counting.** Replication score `N = min(3, independent libraries) + min(2, BioProjects) + 0.5·log2(1 + independent molecules)`; each library is capped at 10 independent molecules per junction; "replicated" requires ≥ 2 independent libraries; a high-confidence EST junction requires ≥ 2 independent molecule units after deduplication (this replaces every "≥ 2 distinct clone/accession IDs" wording in §5 and A6). Each accepted EST supports at most one locus (MAPQ ≥ 20; ≥ 30 for the maize paralog-sensitive tier; co-best, ambiguous, split/chimeric alignments rejected; no fractional counting). "Aligned portion only": identity, blocks, junctions and anchors are computed from the accepted aligned portion, while the aligned-fraction threshold (≥ 0.80 of the full EST) remains; soft-clipped sequence supplies no evidence.

**Implementation audit (before any addition is scored).** Code is checked for raw accession counting, `COUNT(*)`, cluster-size weights, two-accession confidence tests, splice-graph edge weights, clone-pair handling, TES/TSS support and representative-only alignment; every evidence table retains accession, molecule, library, BioProject, alignment signature and deduplication-unit identifiers. Storage uses block-compressed FASTA/BAM/PAF and an accession-indexed metadata table; sequence hashes may be stored for integrity only.

**Compute plan.** ~8.0 M ESTs ≈ 4–7 Gb query; minimap2 12–24 h across species on the 64-core node plus 6–24 h for sorting/filtering/junction extraction; 96 GB RAM provisioned; accession-sorted shards of 250,000–500,000 records for restartability, merged before deduplication; no uncompressed SAM.

**Methods sentence (frozen).** "We aligned every EST accession individually to the corresponding genome without sequence clustering and counted junction support from deduplicated molecule units — collapsing accession versions and reads from the same clone and, when clone identifiers were unavailable, only same-library records with identical strand and CIGAR-derived genomic blocks — while retaining distinct libraries and distinct alignment structures as separate evidence."

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | EST download in progress; no alignment run | v1.2.1: A11 replaces the removed clustering step with independent-molecule-unit counting and an implementation audit | Codex review of v1.2: raw accession counts would let re-sequenced accessions and clone pairs inflate support and satisfy the two-ID rule |

## A12. Amendment v1.3 (2026-09-01; before any long-read download or alignment) — PacBio instrument restriction

Author decision: **PacBio evidence is restricted to Sequel II, Sequel IIe and Revio data.** RS II and Sequel (I) datasets are removed from tier 1 and are not used as evidence: Wang 2016 (RS II; already excluded), Wang 2018 HQ isoforms (RS II/Sequel; M-HQ18 removed), Wang 2020 FLNC (Sequel; M-FLNC removed), Cui 2020 PacBio run (Sequel), Han 2023 (Sequel). The ONT rows of §3 are unchanged (ONT and Sequel II+ Iso-Seq remain the equal tier 1).

Consequences at the time of amendment: maize has **no** Sequel II+ Iso-Seq with CCS/FLNC or original BAM identified yet (iFLAS PRJNA983493 is Sequel II but subreads-only without original BAM → excluded by §3); maize tier 1 therefore rests on ONT root tips (M-ONT) until a Sequel II+/Revio B73 set is found (search in progress). A. thaliana: A-HiFi = PRJEB77203 ERR13994458 (Sequel II subreads.bam → `ccs` → FLNC) retained; Kurihara 2022 PRJDB12660 (Sequel II, subreads-only) excluded. Tomato: PRJNA961334 (Sequel II; 1.4–2.8 M reads/run consistent with CCS) usable if CCS status is confirmed from run metadata or read names.

Classification rule for PacBio runs (frozen): instrument model from ENA/SRA metadata; data type from (i) submitted file names/format (`ccs`, `hifi`, `flnc`, `hq` vs `subreads.bam`), (ii) read names when the submitted file is retained (`movie/zmw/ccs` or `transcript/N` vs `movie/zmw/start_end`), (iii) per-cell read count (CCS/FLNC ≈ 0.3–5 M; subreads ≈ tens of millions). Mean read length is not used (Iso-Seq subreads and CCS are both ~1.5–2 kb).

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read download or alignment run | v1.3: PacBio restricted to Sequel II+; M-HQ18 and M-FLNC removed; classification rule added | author decision (chemistry/accuracy consistency across PacBio evidence) |

## A13. Amendment v1.4 (2026-09-01; before any long-read download or alignment) — downloaded long-read data are validation-only

Author decision: **all downloaded ONT and PacBio (Sequel II+) transcript datasets are used only for validation (§§6–9 outcomes and Protocol M final evaluation). They are never used for training any model or head, never for label construction or loss weighting (including the C2 segmentation head masks), and never as a reranking or candidate-generation channel.** Training-side evidence is limited to (i) ESTs of the nine training species (A1, with the locus/orthogroup exclusions of the master plan) and (ii) cross-species protein alignments (Protocol M §5 exclusions), both restricted to training species. Any earlier text that allowed ONT reads in C2 labels "with run separation" is superseded. Evaluated-species long reads therefore never touch the model, the reranker, or the candidate pool.

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read download or alignment run | v1.4: A13 — downloaded long-read data validation-only; C2 masks from ESTs/proteins of training species only | author decision: keep validation evidence strictly separate from anything the model or its selectors learn from |

## A14. Amendment v1.5 (2026-09-01; before any long-read download or alignment) — roles of long-read data by species (supersedes A12 for test species and A13 for training species)

Author decision:
- **Training-side evidence (nine training species only)**: ESTs (A1), **ONT** transcript reads, **PacBio Iso-Seq generated on Sequel II or later**, and cross-species protein alignments. RS II / Sequel I PacBio data are not used for training. Locus/orthogroup exclusions of the master plan apply (strict held-out A. thaliana loci, RC pairs, orthogroups).
- **Test species (Z. mays, S. lycopersicum) validation evidence**: any ONT and **any PacBio instrument generation** (RS II, Sequel, Sequel II, Revio) provided the reference-independence rule of §3 holds (Wang 2016 remains excluded because it fed the RefGen_V4 annotation). Restored to tier 1: M-HQ18 (Wang 2018 HQ isoforms, PRJEB22122) and M-FLNC (Wang 2020 FLNC, Zenodo 2611319). Test-species long reads are never used for training (they are withheld species).
- **A. thaliana (training species and replication-validation species)**: datasets are designated by role before download. **Validation-only, never in training**: A-ONT1 (FLIC, PRJNA1087576) and A-HiFi (Zhong 2025, PRJEB77203 → ccs). **Training-eligible**: A-ONT2 (Cui 2020, PRJNA594286 ONT runs) and other training-species long reads. Validation of A. thaliana additions uses only the validation-only sets.
- The PacBio data-type classification rule of A12 (submitted file names, read names, per-cell read count; not mean length) remains.

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read download or alignment run | v1.5: A14 — training uses training-species ONT + Sequel II+ PacBio + ESTs + proteins; test-species validation accepts any PacBio generation; A. thaliana validation sets designated and excluded from training | author decision, correcting v1.3/v1.4 scope |

## A15. Dataset table update v1.5.1 (2026-09-01; before any long-read download) — Sequel II+/Revio scout results

Scout (ENA/SRA enumeration of all Sequel II/IIe/Revio transcriptomic runs, read type verified from submitted file names, ENA headers/QVs, BioSamples): no Revio/Kinnex pure-B73 or Col-0 dataset exists as of 2026-09-01.

Added to §3.1 (maize, validation, tier 1 PacBio): **M-CCS26** PRJNA1470126 (B73, V3 SAM+leaf, Sequel II CCS, 3 runs, 2.26 M reads; submitted `*.ccs.fastq.gz`, headers `/ccs`; 2025, independent of V4/V5); **M-KNX** PRJNA1290227 (Revio Kinnex FLNC, B73×Mo17 and Mo17×B73 F1 embryo/endosperm 14 DAP, 4 runs, 57 M reads; not on ENA → SRA toolkit; **hybrid genotype: reported under the pooled/non-B73 stratum of A7-P, never as B73-direct**). Excluded: PRJNA983493 (iFLAS) and PRJNA921723 — subreads with ZMW names stripped, no original BAM.
Added to §3.2 (A. thaliana, **validation-only**): **A-CCS23** PRJNA911826 (Zhang et al. 2023 Plant Physiol, doi 10.1093/plphys/kiad193; Col-0 WT runs SRR22719002–07 only, 3 bioreps × 2 runs, ~2.9 M CCS-level reads, 11-day seedlings; mutant runs excluded). A. thaliana validation-only set is now {A-ONT1 FLIC, A-HiFi Zhong 2025, A-CCS23}. Training-eligible A. thaliana long reads: A-ONT2 Cui 2020 and any set listed by the training scout that is not in the validation-only set.
§4.4 tomato: PRJNA961334 confirmed Sequel II HiFi Q20 (cv. Moneymaker, 30 runs, 1.1–3.5 M reads/run); TomatoRTD PRJNA1406150 (M82) raw data are 1 TB/sample subreads.bam — models only from Zenodo if ever used.

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read download or alignment run | v1.5.1: A15 dataset additions (M-CCS26, M-KNX hybrid stratum, A-CCS23 validation-only), exclusions recorded | Sequel II+/Revio scout |

## A16. Amendment v1.6 (2026-09-01; before any long-read alignment) — read completeness codes, molecule-unit counting for non-UMI ONT cDNA, completeness QC output

Adopted from the evidence-first design (`EVIDENCE_TRANSCRIPT_MODEL_DESIGN_v1.md`, Codex design `codex_fulllength_20260901.md`):
- Every accepted read alignment carries three codes stored in the C0 tables: `complete_5p_code` (5C/5I), `complete_3p_code` (3C/3I), `read_internal_code` (IC/IP/IM/IX), with the thresholds of the design document §2. FLNC/"full-length" library labels do not imply 5C; direct-RNA reads default to 5I unless cap/CAGE clustering establishes 5C.
- **Chain support is orthogonal to terminal completeness** (clarifies §6): a 5I or 3I read gives complete intron-chain support when it carries the full ordered chain with ≥ 20 nt aligned beyond the first donor and last acceptor; such a read never supports a TSS/TES, UTR boundary, start codon or GSF label. TES/TSS secondary outcomes (A3) use only 3C/5C reads.
- **Molecule units for ONT cDNA without UMIs** (amends A11 for ONT): reads from the same library with the same strand, the same corrected intron chain and both ends within 10 nt are one PCR-equivalence unit; support thresholds (≥ 3 units, ≥ 2 runs) apply to units; raw read counts are reported alongside. Direct RNA: one read = one unit. PacBio: one source FLNC/ZMW molecule = one unit (never a polished cluster consensus).
- **Completeness QC table** (new secondary output, Table S12d): per species × protocol × library — raw reads, molecule units, mapped/unique/chimeric/multi-mapped %, 5C/5I, 3C/3I, IC/IP/IM/IX counts, tail-positive and internal-priming-rejected fractions, TSS/TES cluster counts, PCR-equivalence compression ratio, median aligned fraction and soft-clipping. Protocols are never pooled into one species completeness rate.
- Evidence-derived transcript models, ORF assignment and GSF labels remain out of scope for this protocol (follow-up); nothing in §9 changes.


## A34. Amendment v1.24 (2026-09-03; before any B5 seed starts) — the encoder is *partially* pretrained: audit, gate and record, initialisation unchanged

Measured while preparing the first training run: the B5 encoder is not fully initialised from the public checkpoint.
The recipe asks for `d_model` 768 and 12 layers; every public HyenaDNA checkpoint is 128 or 256 wide with at most
8 layers. `HyenaEncoder.__init__` therefore builds the target model and copies what it can from
`LongSafari/hyenadna-large-1m-seqlen-hf` (`d_model` 256, `n_layer` 8, `max_seq_len` 1,000,002):

| tensors | count | what happens |
|---|---|---|
| exact key and shape | 80 | copied verbatim |
| key present, shape differs | 147 | tiled along undersized dimensions and cropped from index 0 (`_adapt_tensor_shape`); e.g. `out_proj.weight` (256,256) → (768,768) is the same matrix repeated 3×3, and `pos_emb.z` (1, 1000002, 5) → (1, 129024, 5) is the first 129,024 positions |
| key absent | 112 | **kept at the target model's random initialisation — all of `layers.8`–`layers.11`** |
| total | 339 | 227 loaded (67 %), 112 random |

**The initialisation is not changed.** It is the published 400M recipe (`recipe_source` of `configs/b5_400m_win_v3.json`)
and the reported 92 % F1 was obtained with it; altering it would change the numbers and break the parity claim that the
resubmission rests on. Two independent reviews (Codex thread `01a065f3`, Kimi K3, 2026-09-03) reached the same
conclusion, and both rejected interpolating `pos_emb` instead of cropping: Hyena's positional encoding is a
deterministic function of absolute position, so the prefix is exactly what a model built at `max_seq_len` 129,024
would generate, whereas interpolation would rescale the position axis and change the synthesised filters.
Options considered and refused for this revision: a checkpoint of the exact size (none exists publicly — LongSafari
releases are 128 or 256 wide), changing the architecture to match the checkpoint (A25 parity lost), and initialising
`layers.8`–`layers.11` by stacking the lower layers (scientifically the most defensible, but a numerical change; it
belongs in the follow-up as an ablation arm alongside full-random and function-preserving widening).

What changes is the record and the failure mode (implementation only, no numerical effect):
- `src/transgenic/model/encoder_init.py` builds a per-tensor audit (key, status, source and target shape) and a summary.
- The frozen expectation for this checkpoint at 768×12 — 339 target tensors, 227 loaded, 80 exact, 147 adapted,
  112 missing, missing layers ≤ 11 — is checked at load time. Any deviation, a load of zero tensors, or a tensor
  missing outside the layer stack (a key-naming break rather than a size gap) **stops the run**;
  `TRANSGENIC_ALLOW_ENCODER_DRIFT=1` overrides it only for a deliberate, documented change.
- The loader no longer catches every exception and continues with a randomly initialised encoder; a failed load raises.
- With `TRANSGENIC_RUN_DIR` set, `encoder_init_report.json` (checkpoint, target and source sizes, the 339-row audit,
  violations, timestamp) is written as a run artifact and archived with the checkpoint.

Reporting: the Methods must say *partially* pretrained, not "pretrained HyenaDNA" — the encoder was initialised from
the 256-wide 8-layer checkpoint by width-wise weight replication, with the top four layers randomly initialised. The
supplementary carries the audit table and the statement that reported performance reflects supervised training *and*
this partial tile-and-crop initialisation, and does not isolate the benefit of conventional full-checkpoint transfer.

Nothing in §9 changes.


## A35. Amendment v1.25 (2026-09-03; before any B5 seed starts) — a failing training batch stops the run; no batch is ever skipped under the frozen recipe

Measured on 2026-09-02 while benchmarking the tiers on a 24 GB card: at the 129,024-nt tier **1,093 of 1,103 batches
raised CUDA OOM and were skipped**, and the run continued as if healthy. The loop caught every exception, printed one
stderr line, cleared the gradients and moved to the next batch. In a pre-registered study this silently deletes
training data: the loss curve still descends, the checkpoints still save, and nothing in the artifacts records that
99 % of the largest tier never reached the model. Both independent reviews (Codex thread `01a065f3`, Kimi K3) reached
the same verdict — fail immediately, allowed-skip threshold zero.

Three defects were fixed together (implementation only; the recipe, the data and the optimisation are unchanged):
- **Fail closed.** With a frozen recipe (`--config`), a CUDA OOM raises `RuntimeError` carrying the sample id, the
  input and label shapes and the allocated/reserved/peak memory, and points at resuming from the last checkpoint on a
  GPU that fits the tier. The legacy path (no `--config`) keeps its skip so published behaviour is unchanged.
- **Only OOM is caught.** The handler was `except Exception`, which also hid model and data bugs. Every other
  exception now propagates on both paths.
- **Accumulation and loss accounting.** The optimizer step keyed off the DataLoader index
  (`(step + 1) % accumulation_steps`), so a skipped batch shifted the effective batch size, and the epoch loss divided
  by `len(train_dl)` rather than by the batches actually forwarded. Both now count completed micro-batches.

Consequence for the compute plan: the 129,024-nt tier cannot be trained on a 24 GB card at all, which the previous
behaviour concealed. This is consistent with A25 (all seeds on NSF ACCESS) and is now enforced rather than assumed.

Nothing in §9 changes.

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read alignment run | v1.6: A16 read completeness codes, chain/terminal orthogonality, non-UMI ONT PCR-equivalence units, completeness QC table | author question on non-full-length long reads; Codex design cross-checked |
| 2026-09-01 | no long-read alignment run | v1.7: A17 ORF verdict only after alignment on genome-spliced sequence; pre-alignment homology triage is QC-only; ORF-incomplete complete chains are chain-training (C1/C2) evidence, never GSF labels; chain-label weighting | author questions on pre-mapping ORF checks and on chain training with ORF-incomplete reads |
| 2026-09-02 | no long-read alignment run | v1.8: A18 corrections to A17 (validation-source scope, C1 naming), effective-rule matrix with precedence, dataset-role manifest (Cui → training role, Zhong conversion or unavailable, Wang 2020 B73 selection), frozen C2 weights/class semantics, genotype rule for all species, seed plan, terminology, ingestion hardening | Codex full re-review cross-checked by the author's assistant |
| 2026-09-02 | no protein alignment run | v1.9: A19 OrthoDB v12 Viridiplantae as the sole cross-species protein resource; leakage filter; miniprot acceptance rules; CDS-family-only labelling; order of operations | author decision on the de novo / C2 protein resource |
| 2026-09-02 | no C2 label built | v1.10: A20 class-specific source weights — EST 1.0 / PacBio 0.9 / ONT 0.7 / protein 0.5 for intron and splice-boundary classes; A18.4 weights kept for exon/UTR classes | author decision: intron junctions weight EST |
| 2026-09-02 | no EST alignment run | v1.11: A21 UniVec_Core VecScreen screening, split-at-internal-vector rule, min length 100 nt, provenance | author request to add UniVec processing |
| 2026-09-02 | no B5 database built | v1.12: A22 GeenuFF annotation-quality flags; hard flags mask rows from the training loss (train_weight 0) or drop erroneous transcripts; soft flags recorded | author request to mask partial models and annotation errors before training |
| 2026-09-02 | no evidence scoring run | v1.13: A23 evidence-based primary isoform (full-chain units, bottleneck, CDS length, canonical tie-break; uncalled when no full-chain support) as a secondary outcome | author question on using EST chain connectivity to identify the primary isoform |
| 2026-09-02 | no B5-era inference run | v1.14: A24 grammar-constrained decoding pre-registered as the primary inference arm for the B5 model, unconstrained arm as parity control | author concern about autoregressive coordinate/order/strand/phase hallucinations |
| 2026-09-02 | no B5 database built | v1.15: A25 B5 = new-version training with variable context tiers 30,720/61,440/129,024 (tier6144-v2), all seeds on NSF ACCESS, published-recipe parity control optional/supplementary | author decision: variable context for the new release; use ACCESS |
| 2026-09-02 | no B5 database built | v1.16: A26 whole-window labels (GSF v3: <gene>/<empty>, tiles per tier, most-restrictive tile split, edge exclusion, caps 64 genes / 4,096 tokens, vocab 290) | author decision: label every gene inside the window |
| 2026-09-02 | no whole-genome inference run | v1.17: A27 tiled inference with edge margin 1,000 nt and a fixed stitching precedence; v3 caps 96 genes / 8,192 tokens from tile statistics | author instruction to freeze the stitching rule and to set caps from data |
| 2026-09-02 | no B5 seed started | v1.18: A28 job-chain checkpointing (latest_state every N steps and on SIGUSR1, self-resubmission until TRAINING_DONE, run_config refusal on drift) | 48-hour job limit on ACCESS; no epoch may be lost |
| 2026-09-02 | no B5 database built | v1.19: A29 tile splits by 1,032,192-nt genomic blocks (strict held-out blocks test), orthogroup leakage masking by N-replacement in tiles, dual test reporting | GPU dry run showed the most-restrictive-gene rule leaves no training tiles |
| 2026-09-02 | no protein alignment run; no B5 database built; no evidence scoring run | v1.20: A30 Swiss-Prot reviewed Viridiplantae as a separately labelled sensitivity set — caution-based hard-flag masking through the A22 mechanism only when the reference does not carry the curated sequence; OrthoDB phase audit report-only; secondary outcomes S5–S7 (start/stop/phase agreement per stratum) computed inside evaluation only; A19 label resource unchanged | author question on using curated Swiss-Prot for start/stop codons and phase |
| 2026-09-02 | first tile build done (smoke), no training run | v1.21: A31 strict held-out loci no longer force genomic blocks to test; protection by A29 leak masking (N-replacement + label removal) in train/valid tiles; A. thaliana stays in training | author decision: A. thaliana must be used; the block rule had made 120/121 blocks test |
| 2026-09-02 | first tile build done (smoke), no training run | v1.22: A32 hard-flagged genes are N-masked and unlabelled at gene level inside tiles (tile weight stays 1; per-locus rows unchanged) | author decision: whole-tile masking removed 14–46 % of tiles per tier |
| 2026-09-02 | first tile build done (smoke), no training run | v1.23: A33 overlapping gene blocks allowed with a monotone block key; locus-aware A27 stitching (reciprocal CDS overlap 0.90 + shared intron / 1,000-nt ends); overlap-component masking closure, flank [50,150] nt, masked-fraction cap 0.60, train-only decoy masks; edge-rule invariant; per-tier cap reporting | Codex and Kimi reviews; every threshold fixed from the nine reference annotations before the build |
| 2026-09-03 | no B5 seed started | v1.24: A34 the encoder is partially pretrained (227/339 tensors; layers 8-11 random, 147 tiled/cropped) — initialisation unchanged, but the load is audited, gated against a frozen expectation, written to encoder_init_report.json, and a failed load now raises instead of silently continuing with a random encoder | discovered while preparing the first training run; Codex and Kimi reviews agreed to keep the initialisation and fix the record |
| 2026-09-03 | no B5 seed started | v1.25: A35 a failing training batch stops the run under the frozen recipe (previously every exception was caught and the batch skipped; the 129,024-nt tier skipped 1,093 of 1,103 batches on a 24 GB card while looking healthy); only CUDA OOM is caught, and the optimizer step and epoch loss now count completed micro-batches | measured during the tier benchmark; Codex and Kimi both required an allowed-skip threshold of zero |
## A36. Amendment v1.26 (2026-09-03; before any UniVec screening, before any junction is called, before any evidence scoring) — EST length floor raised to 121 nt: records ≤ 120 nt are excluded

> ❌ **SUPERSEDED by A37 (2026-09-04).** The text below is kept because it is the record of what was
> decided and why, and because the measurements in it stand. What does not stand is its status: it was
> written as a clean amendment and it is not one. The floor returns to 100 nt as primary; 121 nt
> becomes a labelled sensitivity arm. Read A37 before using anything here.

Author decision: **EST records of 120 nt or shorter are excluded from the evidence layer.** This raises the ingestion floor of A8 and A21 from "post-trim length ≥ 100" to "post-trim length ≥ 121". Nothing else in A8 or A21 changes; the §4 alignment command, the §5 junction rules and the §2 definitions are untouched.

- **Exclusion, not down-weighting.** An excluded record contributes to no junction, no chain (§6), no UTR/terminus call (A3), no callability denominator (§7), no independent-molecule count (A11) and no C2 label (A20). It is dropped at ingestion, in the same pass as UniVec trimming, and therefore never reaches the aligner.
- **Measured basis** (2026-09-03; all nine training species aligned to the frozen §4 command). The species-level mapping rate is not comparable across species, because EST length composition differs: records ≤ 120 nt map at 24.8–71.2 % in *every* species, records > 120 nt map at 92.8–98.9 % in *every* species. *A. thaliana*'s apparent 73.5 % is a composition effect — 50.6 % of its GenBank EST records are ≤ 120 nt, against 0.4–9.1 % elsewhere — and its > 120 nt rate is 96.87 %, fourth of nine.
- **What the floor removes in *A. thaliana*** (the species where it bites): 773,346 of 1,529,700 records. Of these, 759,906 come from two 454-era short-read libraries deposited in the GenBank EST division (`8-day Arabidopsis seedlings, aerial tissues`, 541,852 records, mean 91 nt; `Arabidopsis ovule high throughput cDNA library`, 249,438 records, mean 92 nt). The A8 floor of ≥ 100 nt would already have removed 480,880 of them; this amendment removes a further 292,466. Collateral loss of classical Sanger records is 13,440, i.e. **1.82 % of the 738,410 Sanger records**, and the surviving corpus is 756,354 records. The records are not truncated artefacts: `ES211021.1`, `EL034561.1` and `EH803821.1` were re-fetched from NCBI and their lengths match our copies exactly (92 / 104 / 99 nt).
- **Why exclude rather than weight.** The excluded reads are not low quality — 93.8 % of the mapped ones reach MAPQ ≥ 30 — but only 20.6 % of them carry a splice junction, against 55.6 % for Sanger records, while §5 counts them as full units towards "≥ 2 distinct clone/accession IDs". A short read that carries no junction cannot support a chain but can still lift a junction over the support threshold in combination; the floor removes that asymmetry by construction rather than by a weight that would have to be justified per class.
- **Consequence for work already done.** The EST alignment of 2026-09-03 (SLURM `6147674_[0-8]`, nine species, all COMPLETED) was run on the raw `evidence/est/<species>/est.fa.gz` — that is, before the A21 screening that A21 requires ("only the screened FASTA is aligned") and before this floor. **Those nine BAMs are a pre-screening pass and are not evidence.** They must be regenerated from the screened, length-filtered FASTA before any junction is called. No junction, support count or outcome has been computed from them.
- **Implementation.** `revision/scripts/61_univec_trim.py` default `--min-len` becomes 121; `revision/scripts/evidence/univec_screen.sh` passes `--min-len 121` and records `min_len=121` in `PROVENANCE.txt`; a test locks the default so the floor cannot drift silently. Kept/dropped counts by cause go to the completeness QC table (A16) as before.
| 2026-09-03 | nine EST alignments run on raw FASTA (pre-screening); no UniVec screening run; no junction called; no evidence scoring run | v1.26: A36 EST length floor raised from 100 to 121 nt — records ≤ 120 nt are excluded from the evidence layer at ingestion | author decision after the *A. thaliana* 73.5 % mapping-rate investigation: the low rate is a composition effect of two 454-era short-read libraries (51.7 % of that species' EST records), not contamination; short records carry junctions at 20.6 % against 55.6 % for Sanger records yet count as full units under §5 |


## A37. Amendment v1.27 (2026-09-04; before any junction is called, before any evidence scoring) — the EST floor returns to 100 nt as primary; 121 nt becomes a labelled sensitivity arm, and A36 is recorded as a post hoc deviation

Author decision, on external review of A36.

**The primary analysis uses the A8 floor of ≥ 100 nt, unchanged from the frozen text.** The 121-nt floor introduced by A36 is retained, but as a **pre-labelled sensitivity arm** reported beside the primary, never in place of it.

### Why A36 could not stand as an amendment

This amendment exists because A36 failed the protocol's own rule for amendments, and saying so plainly is cheaper than defending it later.

- **The trigger was an outcome, not a defect.** §Amendments admits changes "before the first evidence alignment" or "for defects that make a rule inapplicable". Nothing made the ≥ 100 nt rule inapplicable. The rule ran, and the observation that prompted the change was a *mapping rate* — a result. A floor chosen because it improves the number it is measured by is an ascertainment choice, not a repair.
- **A36 argues against itself.** Its own text records that the excluded reads "are not low quality — 93.8 % of the mapped ones reach MAPQ ≥ 30". A defect argument cannot rest on evidence of adequacy.
- **"§§4–7 untouched" was true of the text and false of the effect.** A36 states the alignment command, junction rules and definitions are unchanged. They are — but excluded records contribute to no junction, chain, callability denominator, molecule count or C2 label, so the frozen quantities move even though the frozen sentences do not.
- **The removal is not uniform across species.** It drops 50.6 % of the *A. thaliana* corpus and 0.4–9.1 % elsewhere, so it changes the between-species comparison it was introduced to explain.

### What is now required

1. **Primary**: ingestion floor ≥ 100 nt (A8/A21 as frozen). All §9 outcomes P1–P5 and the frozen success criteria are computed on this arm.
2. **Sensitivity**: ingestion floor ≥ 121 nt, the A36 arm, computed with identical rules and reported alongside. It is labelled here, before any junction is called, so it is pre-registered rather than selected afterwards.
3. **Paired reporting.** Wherever an EST-derived quantity appears, both arms appear. The report additionally carries, per length bin: callability (§7 denominators), junction support, intron count per supported chain, and gene-length distribution — because the plausible bias of a length floor is structural, not a uniform shift. Short records are a smaller share of long, multi-intron transcripts, so a floor is expected to bias complete-chain support toward short simple structures and to *raise* callable-only rates by removing poorly covered additions from the denominator. Reporting only a mapping rate would hide both.
4. **Interpretation rule (frozen wording).** The *A. thaliana* mapping-rate anomaly is described as **"the gross cross-species mapping-rate anomaly is explained by length composition"**. It is **not** described as rejecting contamination: length is confounded with platform, library, tissue, deposition year and sequence complexity, and re-fetching three accessions establishes only that those three local copies are untruncated. A contamination claim in either direction requires the stratified checks listed in A37.1.

### A37.1 What would settle the contamination question, if it is ever asked

Not required for the primary analysis; recorded so that the open question is not mistaken for a closed one. Any one of these would overturn the composition explanation: an *A. thaliana* deficit remaining inside matched length/platform/library strata; taxonomic assignment of unmapped records to a non-*Arabidopsis* source; UniVec, rRNA or organelle enrichment concentrated in the two 454-era libraries; materially better mapping to a different accession or reference, indicating genotype divergence; or a systematic base-quality or low-complexity difference that survives length matching.

### Implementation

`61_univec_trim.py` returns to `DEFAULT_MIN_LEN = 100`; the 121-nt arm is produced by passing `--min-len 121` explicitly, so the sensitivity arm is always an explicit act and never a default. The screening blastn does not re-run: the per-species `hits.tsv` from the 2026-09-03 pass is the same object either way, and only the trim step is repeated. Both arms are aligned with the §4 command and kept side by side; the primary is `evidence/est_align/`, the sensitivity arm `evidence/est_align_min121/`.
| 2026-09-04 | nine EST alignments run on the 121-nt arm; no junction called; no evidence scoring run | v1.27: A37 restores the ≥ 100 nt primary floor, relabels 121 nt as a pre-registered sensitivity arm, requires paired reporting with per-length-bin callability and junction support, and fixes the wording for the mapping-rate anomaly | external review: A36 was triggered by an outcome rather than a defect, argued against itself by recording that the excluded reads are not low quality, and moved frozen quantities while claiming §§4–7 untouched |

---

## A38. Amendment v1.28 (2026-09-04; before any junction is called, before any evidence scoring) — a run excluded before collection gets a declared manifest row, and the long-read scope is refrozen

**Defect.** A18.3 requires every dataset **and every run** to carry exactly one role, and `excluded` is one of the three. *O. sativa* `SRR25203456` was excluded by author decision on 2026-09-04 (issue #68) and the reason was written into §1 above and into `TRAINING_EVIDENCE_v1.md` — but into prose only. The run had no row. `make_dataset_roles.py` builds the manifest by walking the evidence tree and keying on `.DONE` markers, and a run that was never collected leaves nothing to walk. The v1.27 freeze `d644278f6a18a8b2b2162c8378cf61a8` was therefore an **incomplete object** against the rule it was frozen under.

This is not a bookkeeping nicety. A tree scan distinguishes *here* from *not here*. It cannot distinguish *never considered* from *considered and rejected*, and separating those two is the reason a scope manifest exists at all: a later reconciliation of tree against plan must find an answer, not a gap, and prose is not something a reconciliation reads.

**Correction.** `make_dataset_roles.py` gains `DECLARED_RUNS`, a declaration mechanism for runs known from source metadata but deliberately absent from the tree. It is kept separate from `RESOURCES` (which declares non-run objects such as the OrthoDB partition) because these objects *are* sequencing runs and stay inside the long-read scope. The row is emitted with `note` beginning `DECLARED_NOT_SCANNED:`, so declared and scanned rows can be told apart by grep. No synthetic `.DONE` marker and no placeholder FASTA is created: either would make the run indistinguishable from collected evidence, which is the error being fixed.

`validate()` gains a matching check — a declared run must resolve to exactly one row. If the run is ever downloaded the scan produces a second row for the same `(dataset, run)` and the manifest refuses to build, so the declaration has to be withdrawn deliberately rather than silently colliding.

**The declared row** (14 columns, values taken from the run's ENA `filereport.tsv`, which is retained at `evidence/training/ont/Osativa/nip_pool_PRJNA953663/filereport.tsv`):

| field | value |
|---|---|
| dataset | `training/ont/Osativa/nip_pool_PRJNA953663` |
| run | `SRR25203456` |
| species / stratum | Osativa / reference |
| instrument / data_type | PromethION / RNA-Seq |
| expected_files / expected_reads | 1 / 13,195,758 |
| source_checksum | `a8a59f575ff287efe951305edbedab27` (authority `ENA_fastq_md5`) |
| local_fa_md5 | *(empty — nothing was ever written locally)* |
| role | `excluded` |

**Measurement supporting the basis.** The recorded reason — that ENA publishes the path as a directory — was checked on 2026-09-04 rather than restated: `HEAD` on `https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR252/056/SRR25203456/SRR25203456_1.fastq.gz` returns `301` to the same path with a trailing slash, and the parent listing shows `SRR25203456_1.fastq.gz/` as a directory entry. The 2026-09-03 fetch log records the driver selecting that URL and the run FAILED. This is a measurement, not an author statement.

**Refreeze.** The long-read scope becomes:

| | v1.27 (superseded) | v1.28 |
|---|---|---|
| md5 | `d644278f6a18a8b2b2162c8378cf61a8` | `c2f4c2e8fe68b3e522abb8298e1999ff` |
| bytes | 61,801 | 62,511 |
| data rows | 214 | 215 |
| roles | 129 / 31 / 54 | 129 / 31 / **55** |

The single added row is `excluded`. **No eligibility decision changes and no evidence content changes**: nothing moves into or out of training, no file is added or removed, and every other row is byte-identical. §1 is amended to the new identity and the superseded one is kept here rather than overwritten, so the incomplete freeze remains auditable.

Nothing in §§3–9 changes. §A37 is unaffected: the EST arms and the alignment they feed are a different scope.

---

## A39. Amendment v1.29 (2026-09-04; before any long-read alignment, before any junction is called) — the §3.3 item 2 gate measures the read's orientation, not the splice motif's

**Defect.** §3.3 item 2 reads: *"compute the fraction of spliced alignments whose inferred transcript strand agrees with the annotated strand"*. Implemented literally that is **P(T = A)**, where *T* is minimap2's motif-inferred transcript strand placed on the genome (its `ts:A:` tag composed with SAM FLAG `0x10`) and *A* is the annotated gene strand.

That quantity cannot decide `-uf`, because *T* and *A* are two readings of the same thing. minimap2 derives `ts` from the splice motif (GT–AG versus CT–AC), and the annotation records the strand of the same transcript, so for any correctly placed canonical alignment the two agree **whether or not the library is stranded**. The estimator is invariant to exactly the property being gated.

Measured on the first 27 *A. thaliana* runs: P(T = A) ranged **0.9929–0.9987**, median 0.9958, and **every run passed**. The frozen ≥ 95 % threshold gated nothing.

**What `-uf` actually asserts** is a different proposition: that the **read** is in the forward transcript orientation. minimap2 documents `ts` as the transcript strand *relative to the read*, so the proposition is **P(R = A)** with *R* the read's own genomic strand from FLAG `0x10` — equivalently, among alignments whose motif call agrees with the annotation, the fraction carrying `ts:A:+`.

On the same 27 runs and the same eligible alignments, that statistic separates the libraries completely:

| library | `sense_read_fraction` | runs | verdict |
|---|---|---|---|
| stranded **antisense** (`col0_DRP009401`) | 0.128 – 0.186 | 3 | FAIL |
| **unstranded** cDNA (six datasets) | 0.492 – 0.527 | 20 | FAIL |
| stranded **sense** (`lncrna_survey_PRJNA765684`) | 0.991 – 0.994 | 4 | PASS |

Under the literal reading all 27 would have received `-uf`. On the 23 that are not sense-stranded, `-uf` forces minimap2 to treat the read as the sense strand and therefore **mis-calls the transcript strand of roughly half of every unstranded run's junctions**, and §5 takes junction strand from precisely that tag. The literal reading was not merely uninformative; acting on it would have corrupted the junction set.

**Correction (operative).** The no-`-uf` 10,000-read audit of §3.3 item 2 measures the orientation of each read relative to its source transcript. For every eligible primary spliced alignment at a uniquely assigned, confidently annotated multi-exon gene in a single-strand locus **whose motif-inferred genomic transcript strand agrees with the annotation** (that agreement is the validity condition for the observation, not the statistic), the read's genomic alignment strand from FLAG `0x10` is compared with the annotated gene strand. `-uf` is enabled for a run only if that fraction is **≥ 95 %**, the threshold §3.3 already fixed. Runs below it — including ~50:50 unstranded libraries and predominantly reverse-oriented libraries — are aligned without `-uf` throughout. The minimum-eligible floor is unchanged and still yields `UNRESOLVED`, which never enables `-uf`.

The literal quantity is **retained and reported** as `motif_annotation_agreement`, which is what it is: a consistency check on the alignment and the annotation that would catch a wrong reference, a broken FLAG/`ts` conversion, or systematically noncanonical motifs. It is withdrawn only as the `-uf` decision statistic.

**Why this is not A36-shaped.** A36 was withdrawn (see `quarantine.md` §1e) because a threshold was moved after seeing that moving it improved the number it was measured by. This amendment has the opposite structure, and the distinction matters:

- The defect was identified from the **estimator's algebra**, not from a result being unwelcome: P(T = A) is invariant to read orientation, so it cannot gate a read-orientation assumption. That is true before any run is measured.
- The replacement statistic is **fixed by minimap2's documented contract** for `-uf`, not chosen from the data.
- **The threshold is untouched.** §3.3's 95 % stands exactly as frozen; A36 by contrast moved its own criterion.
- The observed 0.13 / 0.50 / 0.99 clusters **diagnose** the defect. They did not select the statistic, and no run was reclassified to improve any downstream rate.
- The change makes the protocol **more restrictive, not less**: 27 of 27 runs passed before, 4 of 27 pass now. It removes an unearned licence rather than removing inconvenient evidence.
- Every run is treated identically, and the retained no-`-uf` SAMs allow a complete rescore without realignment and without selecting runs by outcome.

**Effect on existing artefacts.** No long-read alignment had been run, so nothing downstream is invalidated. The 27 audits produced under the literal reading are rescored from their retained SAMs; the audit's completion marker binds the scorer's md5 so a verdict produced by the superseded scorer cannot be mistaken for a current one. Nothing in §§3–9 changes, and §3.3 items 1 and 3–5 are unaffected.
