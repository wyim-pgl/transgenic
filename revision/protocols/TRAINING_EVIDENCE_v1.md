# Training-evidence long-read datasets (v1.0, 2026-09-01) — nine training species

Rules (protocol B1 v1.5): ONT of any chemistry; PacBio only if ENA `instrument_model` is Sequel II / Sequel IIe / Revio AND the deposit is CCS/FLNC-level (per-run read count ≤ 5 M; subreads-only sets excluded because CCS cannot be regenerated without original BAMs). Non-reference cultivars are admitted for training labels (junction/UTR masks) with the genotype recorded; A. thaliana validation-only sets (PRJNA1087576 FLIC, PRJEB77203 Zhong 2025, PRJNA911826 Zhang 2023) are excluded here and never enter training. Downloaded as FASTA only (`evidence/training/<ont|pacbio>/<species>/<label>/`), provenance in each `log`.

| Species | Type | Accession | Runs / size (fastq) | Genotype, tissue | Citation |
|---|---|---|---|---|---|
| A. thaliana | ONT | PRJNA594286 | 3 / 28.5 Gb | Col-0 rosette | Cui 2020 Plant Methods 10.1186/s13007-020-00629-x |
| A. thaliana | ONT | PRJDB14952 (DRR424731–33) | 3 / 28 Gb | Col-0 aerial 15 DAS | Sci Rep 2023 10.1038/s41598-023-36618-y |
| A. thaliana | PacBio SqII CCS | PRJNA596358 | 1 / 2.3 M reads | Col-0 inflorescence (poly(A)-preserving library) | unpublished |
| A. thaliana | PacBio SqII FLNC | PRJNA649694 | 2 / 0.7 Gb | seedling ± Pi | unpublished |
| V. vinifera | ONT | PRJNA776245 | 9 / 20 Gb | Pinot Noir berries (PN40024 lineage) | unpublished |
| V. vinifera | ONT | PRJNA732451 | 2 / 18.5 Gb | embryogenic callus | Chen 2022 Mobile DNA 10.1186/s13100-022-00271-5 |
| V. vinifera | PacBio SqII HiFi | PRJNA1185815 | 9 / 6.5 Gb | cv. Zhuosexiang, 9 tissues | genome paper 2024 |
| G. max | ONT | PRJNA648759 | 4 / 17.7 Gb | Williams 82 grafts | unpublished |
| G. max | ONT | PRJNA416810 | 10 / 3 Gb | Williams 82 seed axis | unpublished |
| G. max | ONT | PRJNA803218 | 9 / 55 Gb | genotype 09-138 roots ± SCN | Huang 2022 Front Plant Sci 10.3389/fpls.2022.866322 |
| G. max | PacBio | — | none usable (Sequel II sets are subreads-only) | | |
| P. trichocarpa | ONT dRNA | PRJNA517295 | 5 / 4.4 Gb | SDX | Gao 2021 Genome Biol 10.1186/s13059-020-02241-7 |
| P. trichocarpa | ONT dRNA | PRJNA672182 | 6 / 9 Gb | SDX drought | Gao 2022 Plant Physiol 10.1093/plphys/kiac272 |
| P. trichocarpa | PacBio SqII HiFi | PRJNA709498, PRJNA709499 | 6 / 26 Gb | JGI F1 hybrid (P. trichocarpa × P. deltoides) — **hybrid tag** | unpublished |
| S. bicolor | ONT | — | none exist | | |
| S. bicolor | PacBio SqII/IIe HiFi | PRJNA1275171/229/264/290/324/413/435 | ~33 HiFi-level runs (two 15 M-read subreads runs skipped) | cv. Wray, leaf/sheath/collar/internode | JGI, unpublished |
| S. bicolor | PacBio SqIIe HiFi | PRJNA1034755 | 1 / 0.9 Gb | Baijiu sorghum seedlings | unpublished |
| S. bicolor | excluded | PRJNA13876 BTx623 (955 Gb) | ENA label "Sequel", subreads-only | | |
| B. distachyon | — | none exist (0 ONT, 0 PacBio transcriptomic) | | | |
| S. italica | ONT | PRJNA1097621 | 9 / 56 Gb | cv. Ci846 ± salt | unpublished |
| O. sativa | ONT dRNA | PRJNA752930 | 12 / 68 Gb | Nipponbare 6 tissues | GPB 2023 10.1016/j.gpb.2023.02.002 |
| O. sativa | ONT | PRJNA953663 | 1 / 9.9 Gb | Nipponbare pool | Shang 2023 Mol Plant 10.1016/j.molp.2023.08.003 |
| O. sativa | ONT dRNA | PRJNA1044249 | 10 / 40 Gb | Nipponbare/MH63 sheath | unpublished |
| O. sativa | ONT + PacBio SqII HiFi | PRJNA1291274 | 6 ONT / 26.7 Gb + 3 HiFi / 2.4 Gb | indica SY63/MH63/ZS97 | Zhu 2026 Genome Biol (link unverified) |
| P. patens | ONT dRNA | PRJNA681088 | 7 / 7.2 Gb | gametophore, protonema (Gransden expected) | Fesenko lab, unpublished |

Estimated total ≈ 430 Gb fastq → ≈ 200–250 GB FASTA.gz. Species without long reads for training: B. distachyon (none), S. bicolor ONT (none), G. max / S. italica / P. patens PacBio (none). Reference-annotation feed: Ptrichocarpa v4.1 used JGI Iso-Seq CCS not deposited; Sbicolor v5.1 mentions full-length transcripts (source unverified); all other references predate long reads.
