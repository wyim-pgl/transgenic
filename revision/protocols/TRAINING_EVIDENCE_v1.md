# Training-evidence long-read datasets (v1.0, 2026-09-01) — nine training species

Rules (protocol B1 v1.5): ONT of any chemistry; PacBio only if ENA `instrument_model` is Sequel II / Sequel IIe / Revio AND the deposit is CCS/FLNC-level (per-run read count ≤ 5 M; subreads-only sets excluded because CCS cannot be regenerated without original BAMs). Non-reference cultivars are admitted for training labels (junction/UTR masks) with the genotype recorded; A. thaliana validation-only sets (PRJNA1087576 FLIC, PRJEB77203 Zhong 2025, PRJNA911826 Zhang 2023) are excluded here and never enter training. Downloaded as FASTA only (`evidence/training/<ont|pacbio>/<species>/<label>/`), provenance in each `log`.

> ❌ **SUPERSEDED (2026-09-03) — 아래 표의 PacBio 행 전부.** 저자 결정: **제외**(이슈 #60).
> 이 표는 ENA `instrument_model`(Sequel II / IIe / Revio)과 리드 수로 CCS/FLNC급이라 판정하고
> "SqII CCS" · "SqII FLNC" · "SqII HiFi" · "SqIIe HiFi"로 적었지만, **실제 ENA 기탁물은 51런 전부
> `_subreads.fastq.gz`이고 CCS/FLNC는 0건이다**(2026-09-03 실측, 각 데이터셋 `log`의 `source_url=`).
> `submitted_ftp`가 비어 있어 대체 기탁물이 없고, 원본 BAM 없이는 CCS를 재생성할 수 없다 —
> 바로 위 규칙 문단이 이미 배제한 조건이다. **규칙과 표가 서로 모순이었고, 규칙이 맞다.**
>
> 내려받은 43 GB는 삭제하지 않고
> `evidence/RETIRED_DO_NOT_USE/training_pacbio_subreads_20260903/`로 격리했다(같은 곳의 `README.md`에
> 종·데이터셋·런 수와 되돌리는 법). `longread_fetch.sh`는 이제 `_subreads` URL을 이름으로 거부한다.
>
> **파급**: *S. bicolor*는 ONT가 없어 **장독 증거가 0**이 된다(아래 21–24행). *B. distachyon*은 원래 0.
> 나머지 일곱 종은 ONT로 남는다: A. thaliana 6런 · G. max 27 · O. sativa 12(수집 중) · P. patens
> (수집 대기) · P. trichocarpa 11 · S. italica 9 · V. vinifera 11.
> 상세는 `quarantine.md` D3, 현행 학습 증거 구성은 이 배너를 기준으로 읽을 것.

| Species | Type | Accession | Runs / size (fastq) | Genotype, tissue | Citation |
|---|---|---|---|---|---|
| A. thaliana | ONT | PRJNA594286 | 3 / 28.5 Gb | Col-0 rosette | Cui 2020 Plant Methods 10.1186/s13007-020-00629-x |
| A. thaliana | ONT | PRJDB14952 (DRR424731–33) | 3 / 28 Gb | Col-0 aerial 15 DAS | Sci Rep 2023 10.1038/s41598-023-36618-y |
| A. thaliana | PacBio SqII CCS | PRJNA596358 | 1 / 2.3 M reads | Col-0 inflorescence (poly(A)-preserving library) | unpublished |
| A. thaliana | PacBio SqII FLNC | PRJNA649694 | 2 / 0.7 Gb | seedling ± Pi | unpublished |
| V. vinifera | ONT | PRJNA776245 | 9 / 20 Gb | Pinot Noir berries (PN40024 lineage) | unpublished |
| V. vinifera | ONT | PRJNA732451 | 2 / 18.5 Gb | embryogenic callus | Chen 2022 Mobile DNA 10.1186/s13100-022-00271-5 |
| V. vinifera | PacBio SqII HiFi | PRJNA1185815 | 9 / 6.5 Gb | cv. Zhuosexiang, 9 tissues | genome paper 2024 |
| G. max | ONT | PRJNA648759 | **8** / 28.9 M reads | Williams 82 grafts | unpublished | ✏️ PARTIAL (2026-09-03, #67): the run count read **4**; the BioProject holds exactly **8** Oxford Nanopore FL-cDNA MinION runs (`SRR12327364/365/382/383/384/385/386/387`, 28,923,488 reads, 11 GB converted FASTA, one `.DONE` each) and all 8 were fetched. Counted from the dataset's own ENA `filereport.tsv`, not from the tree. The remaining 18 runs in the project are Illumina (8 miRNA-Seq, 8 RNA-Seq, 2 OTHER) and were correctly not selected. The old **17.7 Gb** figure was a base count for the undercounted 4 and is not carried forward; reads are given instead because that is what was measured. |
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
| O. sativa | ONT | PRJNA953663 | **0 / 0** | Nipponbare pool | Shang 2023 Mol Plant 10.1016/j.molp.2023.08.003 | ❌ EXCLUDED (author decision 2026-09-04, issue #68): the project's single ONT RNA-Seq run `SRR25203456` (13,195,758 reads) **is not collected and will not be**. ENA publishes a path that is a directory — `301` to itself with a trailing slash over HTTPS, `curl 78 file does not exist` over FTP — so no amount of retry or resume reaches it; that is why it survived the 2026-09-04 fetcher rework which recovered every other missing run. NCBI's S3 mirror does hold it (verified: `200`, `Accept-Ranges: bytes`, resume, 12.1 MB/s, `SEQ` 13,195,758 matching ENA's `read_count`), but reaching it needs an SRA fallback in the driver, and the author chose to drop the run instead. The project's other runs were already out of scope: `SRR25241091` is ONT **WGS** (refused by #63), `SRR25241090`/`SRR25241092` are WGS, and the rest are Hi-C/ChIP-Seq/Illumina. **This dataset therefore contributes nothing**, by decision, not by omission. |
| O. sativa | ONT dRNA | PRJNA1044249 | 10 / 40 Gb | Nipponbare/MH63 sheath | unpublished |
| O. sativa | ONT | PRJNA1291274 | 6 ONT / 26.7 Gb | indica SY63/MH63/ZS97 | Zhu 2026 Genome Biol (link unverified) | ❌ SUPERSEDED (2026-09-04): the **"3 HiFi / 2.4 Gb"** half of this row was wrong. The project's three PacBio Sequel II runs (`SRR34539098` 275,533 · `SRR34539106` 203,041 · `SRR34539107` 432,513 reads) are **subreads**, which protocol v1.3/v1.5 admits only at CCS/FLNC level, and the #60 guard refused all three by filename on the first attempt: `skipped: subreads file`. Measured 2026-09-04, the first time this dataset was ever attempted — `training_run_all.sh` had been stuck 17 h upstream of this line and never reached it. **The same defect as #60**: the table records an instrument tier the data does not carry. The ONT half stands (6 runs collected). |
| P. patens | ONT dRNA | PRJNA681088 | 7 / 7.2 Gb | gametophore, protonema (Gransden expected) | Fesenko lab, unpublished |

Estimated total ≈ 430 Gb fastq → ≈ 200–250 GB FASTA.gz. Species without long reads for training: B. distachyon (none), S. bicolor ONT (none), G. max / S. italica / P. patens PacBio (none). Reference-annotation feed: Ptrichocarpa v4.1 used JGI Iso-Seq CCS not deposited; Sbicolor v5.1 mentions full-length transcripts (source unverified); all other references predate long reads.

## Protein resource (added 2026-09-02, protocol A19)
| resource | role | filter | where |
|---|---|---|---|
| OrthoDB v12, Viridiplantae partition | `c2_training_eligible` (CDS-family labels), Protocol M candidate pool, homology tier for ORF assignment | remove evaluated species (3702, 4577, 4081 and sub-taxa) and training-species test-orthogroup proteins (A19.1) | downloaded and aligned (miniprot) on ACCESS Delta |
| UniProtKB/Swiss-Prot reviewed, Viridiplantae (`uniprot_sprot_plants.dat.gz`; added 2026-09-02, protocol A30) | `sensitivity_set` — never a label source: caution-based loss masking of reference models (A22 mechanism) and the S5–S7 start/stop/phase sensitivity outcomes | Viridiplantae only; hard flags only for structural cautions (erroneous initiation/termination, frameshift, erroneous gene model prediction) whose curated sequence is absent from the reference proteome; alignment to evaluated genomes inside evaluation only | downloaded with OrthoDB on ACCESS Delta; parsed by `revision/scripts/63_swissprot_sensitivity.py` |
