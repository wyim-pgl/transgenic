# Evidence-first data layer — design v1.2 (2026-09-02; v1.1 + protocol A18 alignment: internal-priming 6-A run, ONT correction rule = sensitivity only, C1 naming, frozen weights, phase split)

Decision (author, 2026-09-01): a fresh design is acceptable. This document replaces the "patch preprocess.py" approach for the evidence layer with a single-source-of-truth **transcript model** representation from which GSF labels, C2 masks, C0 junction tables and validation metrics are derived. The B5 path (reference annotation → model → GSF → DB) uses the same representation, so the week-1 schedule in `IMPLEMENTATION_ORDER_B5_C0_C2_v1.md` is unchanged; only the module boundaries move. Source: Codex design `codex_fulllength_20260901.md` (tool behaviour verified from IsoQuant/FLAIR/StringTie/Iso-Seq collapse/TAMA documentation), cross-checked against the frozen protocol.

## 1. Core representation
`transcript_model` (one row per observed or annotated transcript structure):
`model_id, locus_id, species, assembly, contig, strand, exons[], introns[] (transcription-ordered, corrected), tss, tes, complete_5p_code {5C,5I}, complete_3p_code {3C,3I}, internal_code {MC,MP,MM,MX}, completeness_code "M.<5>.<3>.<internal>", support_molecules, support_libraries, support_bioprojects, observed_template_molecule_id, orf_status {complete, partial, none}, cds[] with phases, nmd_like, uorf_count, orf_source {reference, homology, longest}, gsf_eligible, validation_only, source {reference, ont, pacbio, est}, provenance`.
Reference annotations are ingested as models with `source=reference`, `5C/3C` assumed and flagged `assumed_from_reference=true` (never pooled with read-derived completeness rates), `orf_source=reference`. Column names: `read_internal_code` (IC/IP/IM/IX) vs `model_internal_code` (MC/MP/MM/MX). Everything downstream (GSF serializer, C2 mask builder, C0 aggregates, validation scorer) consumes this table only.

## 2. Read-level codes (frozen thresholds; proposed by Codex, adopted)
Input filters: primary alignment, MAPQ ≥ 20, aligned query fraction ≥ 0.80 (lower coverage → junction/block evidence only), junction anchors ≥ 20 nt, ONT junction correction for primary outcomes follows protocol §5 (≤ ±3 nt to a unique canonical motif); correction onto an annotated junction or one supported by ≥ 3 molecule units is a labelled sensitivity analysis only (A18.2); chimeric/supplementary/cross-locus excluded; equal-best multi-locus → `mapping_ambiguous` (family-level counts only).
- **5′**: `5C` if (a) cap-trapped/TeloPrime library with 5′ primer detected (≥ 80% identity over ≥ 12 nt), or ONT strand-switch adapter detected AND end within 50 nt of a cap/CAGE TSS; or (b) start within 25 nt of a TSS cluster with ≥ 5 molecule units from ≥ 2 libraries (≥ 2 each in two libraries), ≥ 2 cap/CAGE-associated units, medoid within ±50 nt of a same-strand CAGE peak. Otherwise `5I`. Start pile-ups (≥ 5 starts in a 10-nt bin, ≥ 4× the next five bins) are QC only; gradual decay → `five_prime_decay=true`.
- **3′**: `3C` if untemplated poly(A) ≥ 10 nt passing the internal-priming filter (fail: ≥ 12 A of downstream 20 nt, or ≥ 6 consecutive A; A3/A18.2) or end within 25 nt of a TES cluster with ≥ 2 tail-positive units. Otherwise `3I` (with `internal_priming_suspect` where applicable).
- **Internal**: `IC` exact admitted chain with 20-nt terminal anchors; `IP` proper contiguous sub-chain; `IM` mono-exonic; `IX` conflicting/ambiguous.
- Molecule units: UMI → (library, sample, UMI, locus, strand); ONT direct RNA → one read; ONT cDNA without UMI → **PCR-equivalence unit** = same library, same strand, same corrected chain, both ends within 10 nt (conservative); PacBio → source FLNC/ZMW molecule, never a polished cluster; libraries never merged.

## 3. Collapse policy (custom, observational)
Mandatory invariant: **every emitted exon chain and 5′/3′ endpoint combination is witnessed together by at least one input molecule** — no graph paths, no cross-molecule end pairing. Steps: partition by exact corrected chain → complete-linkage end clustering (diameter ≤ 25 nt) separately for 5′ and 3′ → joint (5′,3′) model only where a molecule belongs to both clusters and observes the whole chain → coordinates taken from the observed molecule closest to both medoids (`observed_template_molecule_id`), never "longest read" → partial reads add `support_compatible` but cannot move ends or add exons → identical chains with different complete terminal clusters are distinct models; differences confined to incomplete ends are support variation → a shorter contiguous sub-chain is truncation unless its missing terminus is independently complete (alternative first exon: `M5C` cluster + ≥ 3 `5C` units from ≥ 2 libraries; alternative last exon: `M3C` + ≥ 2 tail-positive units; both ends: one molecule observing chain + both clusters) → mono-exonic: never merged by overlap; needs both complete ends, ≥ 3 units, ≥ 2 libraries, ≥ 1 cap-supported and ≥ 2 tail-positive.
Tools: IsoQuant/FLAIR may supply corrected alignments but not their assembled models (IsoQuant restores/moves sites from annotation; FLAIR end policies; StringTie is an assembler; Iso-Seq collapse merges extra 5′ exons by default; TAMA 10-nt defaults) — none guarantees the invariant.

## 4. ORF, CDS and GSF eligibility
**Sequence source:** ORF/CDS assignment runs on the genome sequence spliced through the model's aligned exon chain (read errors and unknown strand make read-level ORF scans unreliable; see §4a). ORF search in three transcript-oriented frames; complete ORF = ATG…stop, no internal stop, length % 3 == 0; de novo ≥ 100 aa; homology-supported ≥ 30 aa with E ≤ 1e-10, identity ≥ 30%, coverage ≥ 70%, splice-consistent placement; ranking: exact training-species reference CDS (leakage rules) > cross-species homology > longest valid ORF. Eligibility: `5C/3C` de novo; `5C/3I` CDS only if fully contained, GSF-ineligible; `5I/3C` start only with homology/reference support plus upstream in-frame stop or ≥ 30 nt observed 5′ UTR, GSF-ineligible; `5I/3I` partial ORF only; `MP/MX` none. `nmd_like` (stop ≥ 50 nt upstream of last junction) kept for structure but not a coding GSF label unless reference/homology supports the ORF; uORFs recorded, never chosen as main CDS by position alone. Phases recomputed 5′→3′ (`(3 − cumulative mod 3) mod 3`). `gsf_eligible` requires `M.5C.3C.MC` (or qualifying `MM`), witnessed signature, support thresholds, complete CDS, not unsupported `nmd_like`, window fit, feature/token caps, canonical order + round-trip, and no leakage exclusion.

## 4a. Pre-alignment triage (QC only)
Before alignment a read may be tagged `cds_candidate` (DIAMOND blastx hit to a cross-species proteome covering ≥ 70 % of the protein and reaching within 5 aa of both termini; hit strand, untemplated poly(A) and 5′ adapter/primer recorded). FLNC tags mean primer presence, not ORF completeness. Tags enter the completeness QC table and read prioritisation only; they never set eligibility, tiers or denominators (protocol A17).

## 5. Decision table — state → allowed uses
| State | junction support | chain support | C2 positive mask | UTR endpoint | GSF label |
|---|---|---|---|---|---|
| R.5I.3I.IP / IX (partial, no complete ends) | yes (its junctions) | no (sub-chain only → T3/T4) | yes (aligned blocks) | no | no |
| R.5I.3C.IC | yes | **yes** (chain and terminal completeness are orthogonal) | yes | TES only | no |
| R.5C.3I.IC | yes | yes | yes | TSS only | no |
| R.5C.3C.IC | yes | yes | yes | both | as part of an `M.5C.3C.MC` model |
| R.*.*.IM (mono-exonic) | — | — | yes (block) | if complete | only via qualifying `MM` model |
| mapping_ambiguous | family-level only | no | no | no | no |
| M.5C.3C.MC/MM, complete ORF | — | — | yes | yes | **yes** (training species, non-held-out loci) |
| any model from a test species or an A. thaliana validation-only dataset | validation | validation | no | validation | no (`validation_only`) |

## 5a. ORF-incomplete chains — allowed uses (protocol A17)
ORF completeness and chain completeness are independent axes. A chain fully witnessed by one molecule (`IC`) is chain evidence regardless of its ORF or terminal state; the training-species/non-held-out/non-`validation_only` restriction applies only to training-side use (C2 masks, future chain objective), while B1 validation uses the designated validation datasets (A18.1).

| State | chain-training target (follow-up chain objective — **inactive in this revision**; C1 = canonical-order control in B5) | C0 junction / partial-chain support | C2 positive mask | GSF label |
|---|---|---|---|---|
| `IC` + 5I/3I (chain complete, ends incomplete) | yes | yes | yes | no |
| `M.5C.3C.MC` without complete ORF (incl. `nmd_like`, non-coding) | yes | yes | yes | no (GSF vocabulary requires CDS features/phases; non-coding representation = format change, out of scope) |
| `IP` (partial chain) | no (sub-chain only, T3/T4) | yes | yes | no |
| `IX` / `mapping_ambiguous` | no | family-level only | no | no |

Weights are frozen in protocol A18.4: `W = min(4, 1 + source_weight·log1p(independent_molecules))`, source_weight protein 1.0 / PacBio Sequel II+ 1.0 / ONT 0.8 / EST 0.6, genotype multiplier 1.0 or 0.5, retained-intron-like (block spanning a ≥ 3-unit intron with < 3 molecules) 0.25, unit-weight ablation mandatory; class semantics per A18.4. Current-revision scope is unchanged: evidence selects (B1–B3) and teaches only the C2 head; evidence-derived chain targets are follow-up work.

## 6. Scope
**Two phases (A18, Codex 1.1.11)**: phase 1 (this revision) = reference annotation → transcript_model → GSF/DB for B5, plus read/alignment observations → C0 tables, completeness codes, C2 masks and B1 scoring; phase 2 (follow-up) = observational collapse, ORF/homology/NMD assignment, evidence-derived GSF. B5 does not wait for phase 2.

**Current revision**: per-dataset completeness QC table (species × protocol × library; 5C/5I, 3C/3I, IC/IP/IM/IX, tail-positive and internal-priming-rejected fractions, PCR-equivalence compression, TSS/TES cluster counts) as supplementary output; C0 records carry the three codes; C2 positive masks from observed blocks/junctions (data artifact, evidence-weighted training only under the C2 gate); B1 validation on observed chains unchanged; protocol-stratified limitation statement. **Follow-up**: observational collapse, transcript-model release, ORF/homology/NMD assignment, evidence-derived GSF labels and their ingestion into orthogroup-aware retraining, collapse-tool benchmark and threshold sensitivity.

## 7. Tests (pytest, to be written before the evidence layer is used)
Completeness (14 tests: cap/adapter+CAGE → 5C; strand-switch alone → 5I; TSS cluster ≥ 5 units/2 libraries; single-library pile-up rejected; decay → 5I; poly(A) ≥ 10; internal priming 12/20 and 8-run; TES cluster ≥ 2 tail-positive; IC/IP/IX/IM classification; strand symmetry), molecule counting (4), collapse invariant (13, including idempotence under read-order permutation and "every emitted model has an observed witness"), ORF/CDS/GSF (15, including held-out locus never GSF-eligible). Full names in `codex_fulllength_20260901.md` §6.
