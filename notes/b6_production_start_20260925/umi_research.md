# ONT UMI source check, training eligible only

Controller check: the table totals 125 runs in 21 study rows, not 22. The 5 `confirmed_no_UMI` entries mean support from the **reported standard kit/workflow**, not direct experimental proof that no unreported UMI step occurred. No author UMI approval or frozen TSV change is implied. Of the 120 unresolved runs, 42 have author-approved direct_RNA protocol; A16 defines these as one read per unit. The remaining 78 are cDNA. Whether direct_RNA metadata can retain `umi_status=unresolved` is being checked against the existing producer and C0 contract rather than relabeling these runs as absent.

Scope covered: all 125 `c2_training_eligible` ONT runs in `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv` sha256 `e21425e1eeaa60ac87648464b684f61fabe06c848ec8d0de4da448d54695b029`, grouped by BioProject and protocol subgroup. Excluded and validation-only ONT rows were not investigated. Public ENA read_run metadata was used by study group, not 125 per-run repeat lookups. No reads, TSV, or protocol files were changed.

Decision rule: `confirmed_no_UMI` only when a public method names a standard non-UMI ONT/SMARTer library workflow or kit. `unresolved` means the source supports cDNA/direct RNA protocol but does not explicitly establish UMI absence. Sample barcodes and library names were not treated as UMIs. Missing UMI mention alone was not used as absence evidence. No `UMI_present` group was found.

| Study group | Runs | Evidence URL and exact method quote | Status | Why |
|---|---:|---|---|---|
| DRP009401 col0 | 3 | https://www.ebi.ac.uk/ena/browser/view/DRP009401, `library_selection: cDNA` | unresolved | cDNA controlled vocabulary only, no UMI design or kit. |
| PRJDB38182 col0_nitrogen | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJDB38182, `library_selection: PCR`; `Then perform library construction and machine sequencing` | unresolved | Generic construction text, no UMI exclusion. |
| PRJNA1044249 nip_sheath | 10 | https://www.ebi.ac.uk/ena/browser/view/PRJNA1044249, `Direct RNA-seq of Oryza sativa` | unresolved | Direct RNA label only, kit and adapter design not specified. |
| PRJNA1097621 ci846_salt | 9 | https://www.ebi.ac.uk/ena/browser/view/PRJNA1097621, `library_selection: cDNA` | unresolved | No kit or UMI statement. |
| PRJNA1155260 seed_flcdna | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJNA1155260, `nanopore full-length cDNA sequencing in dry seed and germinated seed` | unresolved | Full-length cDNA phrase only. |
| PRJNA1291274 indica_flagleaf | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJNA1291274, `Nanopore cDNA of MH63 rep1`; `direct RNAseq of MH63 rep2` | unresolved | Mixed cDNA/direct RNA protocol labels, no UMI method detail. |
| PRJNA416810 wm82_seed | 10 | https://www.ebi.ac.uk/ena/browser/view/PRJNA416810, `library_strategy: FL-cDNA`; `library_selection: Oligo-dT` | unresolved | FL-cDNA/Oligo-dT does not rule out external UMIs. |
| PRJNA517295 sdx_dRNA | 2 | https://www.ebi.ac.uk/ena/browser/view/PRJNA517295, `Nanopore Direct-RNA Sequence` | unresolved | Direct RNA label only. |
| PRJNA594286 cui2020 | 3 | https://www.ebi.ac.uk/ena/browser/view/PRJNA594286, `cDNA from the Arabidopsis RNA samples was synthesized using the SMARTer PCR cDNA Synthesis Kit`; `For ONT sequencing, cDNA-PCR libraries were built using the Ligation Sequencing Kit (SQK-LSK109)` | confirmed_no_UMI | Named SMARTer PCR cDNA plus ONT ligation workflow, no UMI-bearing kit or step. |
| PRJNA596358 inflorescence | 1 | https://www.ebi.ac.uk/ena/browser/view/PRJNA596358, `full-length mRNA of At inlforescence by Nanopore`; `library_strategy: FL-cDNA` | unresolved | No construction method. |
| PRJNA648759 wm82_graft | 8 | https://www.ebi.ac.uk/ena/browser/view/PRJNA648759, `Full length RNAseq`; `library_strategy: FL-cDNA` | unresolved | No kit or UMI statement. |
| PRJNA672182 sdx_drought | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJNA672182, `Nanopore Direct-RNA Sequence of stem-differentiating xylem ... during drought stress` | unresolved | Study title only. |
| PRJNA681088 Ppatens dRNA | 7 | https://www.ebi.ac.uk/ena/browser/view/PRJNA681088, `Physcomitrium patens direct RNA sequencing` | unresolved | No kit or UMI statement. |
| PRJNA705300 wt_only | 1 | https://www.ebi.ac.uk/ena/browser/view/PRJNA705300, `Nanopore direct RNA sequencing of Arabidopsis thaliana - col1` | unresolved | Direct RNA label only. |
| PRJNA732451 Vvinifera callus | 2 | https://www.ebi.ac.uk/ena/browser/view/PRJNA732451, `For long-read cDNA-seq, the libraries were prepared following the protocol of Oxford Nanopore cDNA-PCR kit (SQK-PCS109).` | confirmed_no_UMI | Named ONT cDNA-PCR kit is a no-UMI workflow. |
| PRJNA752930 nip_dRNA_6tissue | 12 | https://www.ebi.ac.uk/ena/browser/view/PRJNA752930, `Direct RNA sequencing of rice tissue` | unresolved | Direct RNA label only. |
| PRJNA765684 lncrna_survey | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJNA765684, `Identification of long non-coding RNAs using long-read cDNA sequencing` | unresolved | cDNA study title only. |
| PRJNA776245 pinotnoir_berry | 9 | https://www.ebi.ac.uk/ena/browser/view/PRJNA776245, `library_selection: RT-PCR` | unresolved | RT-PCR selection alone is insufficient. |
| PRJNA796315 ck_control | 6 | https://www.ebi.ac.uk/ena/browser/view/PRJNA796315, `cDNA transcriptome of Arabidopsis` | unresolved | No kit or UMI statement. |
| PRJNA803218 scn_roots | 9 | https://www.ebi.ac.uk/ena/browser/view/PRJNA803218, `Full length transcriptome: soybean root`; `library_selection: cDNA` | unresolved | No kit or UMI statement. |
| PRJNA874115 col0_stigma | 3 | https://www.ebi.ac.uk/ena/browser/view/PRJNA874115, `library_selection: cDNA`; `RNAseq of Col-0 stigma` | unresolved | No construction method. |

Summary: confirmed_no_UMI = 5 runs in 2 groups, UMI_present = 0 runs, unresolved = 120 runs in 19 listed groups (21 table rows total). The frozen TSV currently says `absent` for all 125 eligible runs, but this investigation supports only 5 as confirmed no-UMI under the stricter source rule above.
