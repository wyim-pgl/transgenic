# PROTOCOL B1/B4 — Frozen protocol for independent transcript-evidence validation of completion-mode additions (v1.8; v1.0 text unchanged, amendments appended; §A18 effective-rule matrix is the operational reading of §3–§9 + amendments)

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

| Date | Analysis state | Change | Reason |
|---|---|---|---|
| 2026-09-01 | no long-read alignment run | v1.6: A16 read completeness codes, chain/terminal orthogonality, non-UMI ONT PCR-equivalence units, completeness QC table | author question on non-full-length long reads; Codex design cross-checked |
| 2026-09-01 | no long-read alignment run | v1.7: A17 ORF verdict only after alignment on genome-spliced sequence; pre-alignment homology triage is QC-only; ORF-incomplete complete chains are chain-training (C1/C2) evidence, never GSF labels; chain-label weighting | author questions on pre-mapping ORF checks and on chain training with ORF-incomplete reads |
| 2026-09-02 | no long-read alignment run | v1.8: A18 corrections to A17 (validation-source scope, C1 naming), effective-rule matrix with precedence, dataset-role manifest (Cui → training role, Zhong conversion or unavailable, Wang 2020 B73 selection), frozen C2 weights/class semantics, genotype rule for all species, seed plan, terminology, ingestion hardening | Codex full re-review cross-checked by the author's assistant |
