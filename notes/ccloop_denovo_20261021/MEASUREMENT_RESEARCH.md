# Deferred four-species de novo annotation benchmark: measurement research

Research date: 2026-10-21. **Status: deferred at the user's direction.** The active priority is finishing the current B6/C2 model through ccloop. This note preserves findings already obtained; it authorizes no benchmark jobs, retraining, downloads, or changes to that training effort. **BLASTP versus BLASTN remains unresolved.** Recommendations below are proposed measurement rules, not user-approved thresholds or completion gates.

## Scope and the critical training-overlap limitation

The intended comparison is TransGenic versus Helixer with genomic DNA as the inference input, covering rice, Arabidopsis, soybean, and sorghum. Freeze identical assemblies, contig scope, reference releases, preprocessing, translation rules, model/checkpoint hashes, software versions, and postprocessing. Evaluation references may be used for scoring but must not enter inference or test-driven tuning.

The official overview for Helixer's released `land_plant_v0.3_a_0080.h5` explicitly lists all four requested species as **training species**, with these accessions:

| Species | Helixer training annotation | Candidate specific BUSCO lineage |
|---|---|---|
| Oryza sativa | v7.0 | poales_odb12.2 |
| Arabidopsis thaliana | TAIR10 | brassicales_odb12.2 |
| Glycine max | Wm82.a4.v1 | fabales_odb12.2 |
| Sorghum bicolor | v3.1.1 | poales_odb12.2 |

Training membership is directly documented in the [Helixer model overview](https://github.com/usadellab/Helixer/blob/main/docs/model_overview.md). The candidate lineage names were verified in the [official BUSCO dataset manifest](https://busco-data.ezlab.org/v5/data/file_versions.tsv); their `odb12.2` entries were dated 2026-05-13. These are candidates for a future frozen protocol, not a request to upgrade an existing evaluation environment.

Consequently, distinguish **genome-wide descriptive annotation performance** from **leakage-controlled confirmatory generalization**. A chromosome or locus held out only from TransGenic does not establish symmetric holdout from pretrained Helixer. A new reference release does not itself erase training overlap. Confirmatory unseen-species evidence requires exclusion demonstrable for both models, potentially new species or separately trained baselines; this does not block completion of the current B6/C2 training.

Audit TransGenic pretraining, supervised training, validation, checkpoint selection, and prior test inspection separately. Freeze held-out loci/blocks before selection; prevent overlapping windows, duplicate sequences, or homologous groups from crossing splits where the intended claim requires their independence. Previously inspected test results are development evidence unless a defensible untouched evaluation remains. These are protocol recommendations motivated by documented leakage risks, not proof that a particular local run leaked. [Kapoor and Narayanan, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).

## BUSCO

Use **protein mode** on translated predicted annotations for the annotation comparison. Genome mode reannotates the input assembly, so identical input genomes do not measure the differences between the two submitted gene sets. Report `C=S+D`, `S`, `D`, `F`, `M`, and marker count, retaining per-marker outputs. The current official guide is v6.1.0; `odb12.2` requires v6.1.0 or later. Pin the actual executable/container, dependencies, lineage archive/checksum and date; do not mix dataset generations. Choose the most specific applicable lineage consistently within each species; a shared `embryophyta` panel can be additional context. [BUSCO user guide](https://busco.ezlab.org/busco_userguide.html).

Use one predetermined representative protein per gene for the principal comparison, with the same non-outcome-based rule for references and predictions; retain an all-isoform sensitivity report separately. BUSCO warns that unfiltered isoforms inflate duplication. A duplicated complete marker is not inherently an error: biological duplication and assembly redundancy have different interpretations. [BUSCO interpretation guidance](https://busco.ezlab.org/busco_userguide.html#interpreting-the-results). This matters especially for soybean's documented whole-genome duplication history. [Soybean genome paper](https://www.nature.com/articles/nature08670). Compare methods within species before summarizing paired differences across species.

## OMArk

Freeze compatible OMArk/OMAmer versions and the same **broad LUCA database** release/checksum. A plant-only database restricts detection of nonplant contamination. Supply the taxon ID matching the selected assembly, including subspecies where applicable, and record the selected reference lineage and taxonomy database. Prefer the same representative proteomes used above. If evaluating all isoforms, provide the gene-to-protein isoform file to both methods: OMArk chooses the best OMAmer-mapping isoform per gene, which is a different selection rule and must be labeled. [OMArk official repository](https://github.com/DessimozLab/OMArk).

Report completeness as single, duplicated, and missing conserved HOGs; retain expected/unexpected duplication distinctions. Whole-proteome reporting must include consistent, inconsistent, contaminant, and unknown counts/proportions, plus partial mappings and fragments. Completeness and consistency have different denominators: conserved HOGs versus evaluated proteins. Partial mappings have insufficient sequence coverage of the assigned family; fragments are short family matches, with the paper distinguishing them from partial mappings. Unknowns can include novel genes, so they are not automatically false predictions. [OMArk primary paper](https://www.nature.com/articles/s41587-024-02147-w).

Compare matched methods within species and selected lineage; unequal taxonomic sampling limits cross-lineage comparisons. [OMArk interpretation guide](https://omark.omabrowser.org/home/). Database representation can also favor known annotations: document target-species representation and treat these metrics as complementary evidence, not independent ground truth.

## BLAST mapping: preserve both choices

**BLASTP** compares predicted proteins with reference proteins and addresses coding-product recovery. **BLASTN** compares spliced predicted mRNAs or CDS nucleotides with the corresponding reference sequences; specify which. BLASTN against the source genome alone mainly tests sequence placement and cannot establish transcript correctness. BLAST's alignments are local, so a short excellent match does not establish a complete model. [NCBI alignment statistics](https://blast.ncbi.nlm.nih.gov/tutorial/Altschul-1.html).

Proposed common reporting:

- Save sequence lengths, alignment coordinates, identity counts, alignment length, gaps, score and E-value. State identity's denominator; per-HSP identity is identical aligned positions divided by alignment columns, not full sequence length. Do not average HSP percentages as sequence identity.
- Compute query and subject coverage from nonoverlapping unions of accepted aligned intervals divided by their respective full lengths. Preserve per-pair collinearity and strand; do not merge unrelated hits into a fictitious complete match.
- Report supported predictions / **all predictions**, and recovered unique reference genes / **all eligible reference genes**. Repeat the search in the reverse direction for reference-query recovery. Distinguish this bidirectional recovery from reciprocal-best-hit orthology; the latter can lose legitimate paralogs.
- Include unmatched sequences and fragment/fusion/duplicate diagnostics. A high hit rate among predictions can coexist with poor reference recovery or many duplicate predictions.

NCBI defines `qcovs` across HSPs, `qcovhsp` per HSP, and separate identity/length fields. It also documents that `max_target_seqs` truncates retained hits and breaks ties by database order. Retain enough candidates for the declared matching rule. [BLAST+ manual](https://www.ncbi.nlm.nih.gov/sites/books/NBK279684/). Freeze database checksums, task, filtering, identity/coverage/E-value cutoffs and tie handling before evaluation; no numeric cutoff is approved here.

## CDS, mRNA, exon, and UTR structural scoring

Helixer predicts UTR/CDS/intron classes and produces primary gene-model GFF3; its output is **not CDS-only**. Transcript and protein FASTA can be derived from that GFF3 and the genome. [Official Helixer README](https://github.com/usadellab/Helixer).

Recommended scoring contract: normalize contig IDs, strand, coordinates, parent-child relationships, CDS phase and stop-codon conventions. Score protein-coding loci in a frozen assessable region mask. Use one predetermined reference transcript per locus for primary-model comparison, and report recovery against all annotated isoforms separately. Keep duplicate predictions in the precision denominator; use deterministic one-to-one feature/model matching where counts require it.

| Target | Required distinctions |
|---|---|
| CDS | Strand-aware nucleotide precision/recall/F1; exact coding-segment boundaries; whole-CDS-chain match including start/stop and frame; translation validity. |
| mRNA | Exonic-base precision/recall/F1; intron-chain match separately from exact complete spliced-transcript structure, including both terminal boundaries. |
| Exon | Exact strand/start/end interval precision/recall/F1; distinguish full exons from coding-only exon segments. |
| 5′ and 3′ UTR | Separate nucleotide and interval scores; terminal-boundary errors; full-population recovery plus clearly labeled conditional accuracy at matched loci. |

Define precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, and F1 from both. GffCompare provides useful structural summaries, but its ordinary transcript/intron-chain matches allow terminal differences; even strict matching has a configurable terminal tolerance. Its `-R` and `-Q` options remove unmatched references or predictions, changing denominators, so they are unsuitable for an unqualified genome-wide recovery claim. Document deduplication and single-exon handling. [GffCompare official documentation](https://ccb.jhu.edu/software/stringtie/gffcompare.shtml).

Missing explicit UTR rows do not necessarily mean missing reference information: NCBI GFF3 permits derivation from mRNA exons minus CDS, interpreted in transcript orientation. Respect partial-boundary flags. If reliable transcript ends or relevant reference features are unavailable, report **not assessable** with eligible/excluded counts and reasons, not zero or an invented UTR. [NCBI GFF3 specification](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/file-formats/annotation-files/about-ncbi-gff3/).

## Completion and superiority are separate decisions

Measurement completion means both methods have valid outputs and denominators for every requested category/species, or an explicit unsupported/unassessable status. It does not imply superiority. Publish the per-species metric matrix and paired differences; do not use a weighted composite selected after seeing results. Multiple datasets and criteria complicate algorithm comparisons and can yield partial orderings rather than a universal winner. [Jansen et al., 2023](https://www.jmlr.org/papers/v24/22-0902.html).

For later confirmatory work, predeclare primary endpoints, improvement/noninferiority margins, uncertainty method, and multiplicity policy. Proposed uncertainty units are paired held-out loci or genomic blocks, not independently resampled neighboring bases; show species-level results because four species offer limited generalization evidence. Report tradeoffs explicitly. BUSCO duplication and OMArk unknown fractions have no universal lower-is-better interpretation. A strict claim of improvement across all requested dimensions requires support for each declared dimension; unassessable UTRs cannot be silently counted as a win.

Unresolved for the deferred benchmark: BLASTP/BLASTN/both; exact assemblies/references and transcript-selection rules; model/database versions; reliable UTR availability; symmetric holdout feasibility; and the user-approved scientific superiority rule. None is a new gate on finishing B6/C2.
