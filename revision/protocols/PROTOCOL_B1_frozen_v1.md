# PROTOCOL B1/B4 — Frozen protocol for independent transcript-evidence validation of completion-mode additions

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

Scoring code frozen at repository commit `21e8752` (tag `revision-2026-09`): `28_score_added_isoforms.py` (blob `e98a0fe89158`), `36_filter_additions_structurally.py` (blob `595fc2539905`), `48_score_zmays_additions.py` (blob `e3e09ccce760`). New code written under this protocol (per-structure dump for maize, species-agnostic evidence scorer `59_evidence_support.py`) may add outputs but may not change the definitions in §2.

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

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | pre-download | v1.0 frozen (repository commit 5f7b373) | — |
