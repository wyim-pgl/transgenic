# ONT library-protocol sheet (B6 C0 metadata input) -- author review draft, 2026-09-23

> 2026-09-25 갱신: 학습 대상125런의 **protocol 분류(cDNA83/direct_RNA42)는 저자 승인**됐다. UMI 열은 별도 추가 확인 대상이며 이번 승인에 포함하지 않는다. TSV 바이트/해시는 불변이다. [승인 기록](b6_production_start_20260925/ONT_PROTOCOL_APPROVAL.md).

Sheet: `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv` (143 rows = every ONT row of
`data/manifests/DATASET_ROLES.longread_v1.tsv`, i.e. instrument MinION/PromethION/GridION; 140 unique runs,
SRR10611193-95 appear twice because the manifest lists the cui2020 runs under two dataset paths).
sha256 `e21425e1eeaa60ac87648464b684f61fabe06c848ec8d0de4da448d54695b029` (20 columns after the §8 library-key extension and §8.3 re-key; earlier drafts: 15-col 74988e28…6b96c, 19-col ebca079d…3223). PacBio sheet `data/manifests/PACBIO_LIBRARY.b6_v1.draft.tsv` sha256 `8de4d5830ff1a88587d838c3521a64f2b62803c2f0c4e4d717a401ce102e4dce` (11 columns; 10-col draft was 9462be35…84ee).

목적: `revision/scripts/evidence/b6_c0_metadata.py --source ont` 가 런마다 요구하는 `--protocol cDNA|direct_RNA|unresolved`
와 `--umi-status present|absent|unresolved` 를 저자가 확정하기 위한 근거 표. 이 파일은 초안이며 매니페스트나 역할(role)은
바꾸지 않았다.

## 1. 결과 요약

| 집합 | cDNA | direct_RNA | unresolved |
|---|---|---|---|
| c2_training_eligible (125) | 83 | 42 | 0 |
| b1_validation_only (10) | 10 | 0 | 0 |
| excluded (8) | 4 | 3 | 1 |
| 전체 143행 | 97 | 45 | 1 |

- `umi_status`: 143행 모두 `absent` (ENA의 어떤 필드에도 `UMI` / `unique molecular` 언급 없음, 아래 §4). 신뢰도 medium:
  kit 이름이 적힌 런이 사실상 없으므로 "언급 없음"이지 "없음이 확인됨"은 아니다.
- metadata 판정과 SSP(프라이머) 판정이 **서로 반대인 런은 0개**. 다만 51행은 SSP 쪽이 규칙상 "inconclusive"라 metadata만으로
  정했고(confidence=medium), 1행(SRR25203456)은 리드가 없어 unresolved.
- 이전(미저장) 조사 "129 training runs = direct_RNA 45 / cDNA 84" 와 정합: 그 뒤 4개 런이 excluded 로 바뀌었고
  (SRR36603452 cDNA, sdx_dRNA 중복 3개 direct_RNA), 125 = 84-1 cDNA, 45-3 direct_RNA.

## 2. 방법

### 2.1 ENA metadata
런마다 ENA portal API `filereport` (result=read_run) 로 다음 필드를 새로 받았다 (140 unique runs, 전부 성공):
`study_accession, run_accession, experiment_accession, sample_accession, study_title, experiment_title, library_name,
library_strategy, library_selection, library_source, library_layout, instrument_platform, instrument_model, sample_title,
sample_alias, description, library_construction_protocol, read_count, base_count, fastq_ftp`.
`library_construction_protocol` 이 비어 있지 않은 study 는 PRJDB38182, PRJNA594286, PRJNA732451 셋뿐이고, kit 이름(SQK-PCS109)이
적힌 것은 PRJNA732451 하나다.

### 2.2 리드 서열 검사 (pronghorn login node, 파일당 앞 2,000 레코드만)
리드 파일은 `evidence/longread_align_v1/<species>/<dataset>/<run>/PROVENANCE.txt` 의 `reads` 줄(132런) 또는
`evidence/ont/<dataset>/<run>.fa.gz` (validation 11런) 에서 찾았다. 모두 FASTA(.fa.gz). `ssp_check.pl` 이 `zcat | 앞 2,000 레코드`
에 대해 다음을 센다 (양 가닥, 대문자 정확일치):

| 이름 | 서열 | 용도 |
|---|---|---|
| SSP full / 14-mer | `TTTCTGTTGGTGCTGATATTGC` / `TTTCTGTTGGTGCT` | ONT cDNA strand-switching primer |
| VNP full / 14-mer | `ACTTGCCTGTCGCTCTATCTTC` / `ACTTGCCTGTCGCT` | ONT cDNA VN primer |
| TSO full / 14-mer | `AAGCAGTGGTATCAACGCAGAG` / `AAGCAGTGGTATCA` | SMARTer template-switch oligo (SQK-PCS108 시절 및 Takara/PacBio식 cDNA) -- 첫 파일 두 번째 리드에서 바로 보여 추가 |
| polyA3 | 마지막 50 nt 안에 `A{10,}` | dRNA/ sense 3' 꼬리 |
| polyT5 | 처음 50 nt 안에 `T{10,}` | antisense 가닥 (cDNA 양가닥 신호) |

`ssp_vnp_hits_per_2000` 열은 세 프라이머 14-mer 중 하나라도 있는 리드 수(합집합)이며, `basis` 열에 SSP/VNP/TSO 별 개수와 full 22-mer
합집합(`full=`)을 따로 적었다. 14-mer 를 판정에 쓴 이유: ONT 리드 오류율에서 22-mer 정확일치는 절반 이하로 떨어진다(14-mer 우연일치
확률은 리드당 ~1e-5 로 무시 가능). 보조 근거로 PROVENANCE 의 orientation audit 결과(§3.3 item 2, sense-strand agreement)를
`basis` 에 옮겨 적었다: dRNA 는 가닥특이적이라 PASS@1.00, 양가닥 cDNA 는 FAIL@~0.5 가 기대값이다.

### 2.3 적용한 판정 규칙
1. metadata 판정 -- 우선순위 run-level 자유문(experiment_title, description, library_name, sample_title/alias) > study-level 자유문
   (study_title, library_construction_protocol) > 통제어휘(library_strategy=FL-cDNA, library_selection in {cDNA, RT-PCR, PCR}).
   direct_RNA: `/direct[- ]?RNA|dRNA|drs[_ ]|SQK-RNA|RNA002|RNA004/i`; cDNA: `/cDNA|SQK-PCS|SQK-DCS|SQK-PCB|PCR-cDNA/i`.
2. SSP 판정 -- cDNA: 프라이머 14-mer 리드 ≥ 5 %; direct_RNA: 프라이머 < 0.5 % **and** polyA3 ≥ 30 % **and** polyT5 < 5 %;
   그 외 inconclusive (사유 기록).
3. 최종 protocol: 둘이 반대면 unresolved(해당 없음); metadata 있으면 metadata (SSP 일치 → high, inconclusive → medium);
   metadata 없으면 SSP (medium); 둘 다 없으면 unresolved (low).
4. umi_status: ENA 전체 필드에 `/\bUMIs?\b|unique molecular/i` 가 있으면 present, 없으면 absent. (경계 없는 `/UMI/i` 는
   PRJNA732451 의 "purification kit ... instruction" 등에서 거짓양성을 내므로 단어경계를 썼다.)

## 3. 데이터셋별 요약

| dataset | n | role | protocol | confidence | primer hits /2000 (min-max) | polyA3 (min-max) | deciding metadata text |
|---|---|---|---|---|---|---|---|
| ont/Athaliana_FLIC_PRJNA1087576 | 3 | b1_validation_only | cDNA | medium | 1-5 | 0.48-0.52 | PromethION sequencing: ONT cDNA-seq of Arabidopsis thaliana: mature leaf, bi |
| ont/Slycopersicum_heinz_PRJEB37834 | 1 | b1_validation_only | cDNA | high | 1898-1898 | 0.00-0.00 | inION sequencing / unspecified / Full-length cDNA nanopore sequencing of ripe fruit sample |
| ont/Zmays_roottip_PRJNA822071 | 6 | b1_validation_only | cDNA | high | 1710-1756 | 0.01-0.02 | library_selection=cDNA |
| ont/Athaliana_cui2020_PRJNA594286 | 3 | c2_training_eligible | cDNA | high | 1118-1268 | 0.01-0.01 | Beijing, China). For Pacbio sequencing, the cDNA from the Arabidopsis RNA samples was synt |
| training/ont/Athaliana/buffer_control_PRJNA1393863 | 1 | excluded | cDNA | medium | 1890-1890 | 0.24-0.24 | - |
| training/ont/Athaliana/ck_control_PRJNA796315 | 6 | c2_training_eligible | cDNA | high | 1905-1936 | 0.01-0.02 | MinION sequencing: cDNA transcriptome of Arabidopsis: 24 hours after |
| training/ont/Athaliana/col0_DRP009401 | 3 | c2_training_eligible | cDNA | high | 110-161 | 0.03-0.03 | library_selection=cDNA |
| training/ont/Athaliana/col0_nitrogen_PRJDB38182 | 6 | c2_training_eligible | cDNA | high | 1802-1835 | 0.01-0.03 | library_selection=PCR |
| training/ont/Athaliana/col0_stigma_PRJNA874115 | 3 | c2_training_eligible | cDNA | high/medium | 71-134 | 0.38-0.44 | library_selection=cDNA |
| training/ont/Athaliana/inflorescence_PRJNA596358 | 1 | c2_training_eligible | cDNA | high | 573-573 | 0.01-0.01 | library_strategy=FL-cDNA |
| training/ont/Athaliana/lncrna_survey_PRJNA765684 | 6 | c2_training_eligible | cDNA | high | 1665-1716 | 0.02-0.03 | tion of long non-coding RNAs using long-read cDNA sequencing / |
| training/ont/Athaliana/seed_flcdna_PRJNA1155260 | 6 | c2_training_eligible | cDNA | high | 389-610 | 0.01-0.01 | nanopore full-length cDNA sequencing in dry seed and germinated seed o |
| training/ont/Athaliana/wt_only_PRJNA705300 | 1 | c2_training_eligible | direct_RNA | medium | 0-0 | 0.01-0.01 | MinION sequencing: Nanopore direct RNA sequencing of Arabidopsis thaliana - col1 / |
| training/ont/Gmax/scn_roots_PRJNA803218 | 9 | c2_training_eligible | cDNA | high | 1092-1554 | 0.18-0.44 | library_selection=cDNA |
| training/ont/Gmax/wm82_graft_PRJNA648759 | 8 | c2_training_eligible | cDNA | high | 1411-1554 | 0.38-0.42 | library_strategy=FL-cDNA |
| training/ont/Gmax/wm82_seed_PRJNA416810 | 10 | c2_training_eligible | cDNA | high/medium | 41-102 | 0.00-0.00 | library_strategy=FL-cDNA |
| training/ont/Osativa/indica_flagleaf_PRJNA1291274 | 6 | c2_training_eligible | cDNA/direct_RNA | high/medium | 0-0 | 0.28-0.32 | per-run: "Nanopore cDNA of MH63 rep1/2" (SRR34539092/93) vs "direct RNAseq of MH63/ZS97" ( |
| training/ont/Osativa/nip_dRNA_6tissue_PRJNA752930 | 12 | c2_training_eligible | direct_RNA | high/medium | 0-1 | 0.17-0.36 | PromethION sequencing: Direct RNA sequencing of rice tissue: stem / PromethION |
| training/ont/Osativa/nip_sheath_PRJNA1044249 | 10 | c2_training_eligible | direct_RNA | high/medium | 0-0 | 0.23-0.34 | PromethION sequencing: Direct RNA-seq of Oryza sativa / PromethION sequencing: |
| training/ont/Ppatens/dRNA_gametophore_PRJNA681088 | 7 | c2_training_eligible | direct_RNA | medium | 0-1 | 0.01-0.07 | MinION sequencing: Physcomitrium patens direct RNA gametophores / MinION sequencing: Physc |
| training/ont/Ptrichocarpa/sdx_dRNA_PRJNA517295 | 5 | c2_training_eligible | direct_RNA | medium | 0-0 | 0.01-0.18 | MinION sequencing: Nanopore Direct-RNA Sequence rep2 / MinION sequencing: Nanopore |
| training/ont/Ptrichocarpa/sdx_drought_PRJNA672182 | 6 | c2_training_eligible | direct_RNA | medium | 0-0 | 0.00-0.17 | Nanopore Direct-RNA Sequence of stem-differentiating xylem in Po |
| training/ont/Sitalica/ci846_salt_PRJNA1097621 | 9 | c2_training_eligible | cDNA | high | 1389-1651 | 0.02-0.03 | library_selection=cDNA |
| training/ont/Vvinifera/callus_PRJNA732451 | 2 | c2_training_eligible | cDNA | high | 1349-1399 | 0.03-0.03 | g standard Illumina protocols. For long-read cDNA-seq, the libraries were prepared followi |
| training/ont/Vvinifera/pinotnoir_berry_PRJNA776245 | 9 | c2_training_eligible | cDNA | high | 1694-1768 | 0.43-0.47 | library_selection=RT-PCR |
| training/ont/Athaliana/cui2020_PRJNA594286 | 3 | excluded | cDNA | high | 1118-1268 | 0.01-0.01 | Beijing, China). For Pacbio sequencing, the cDNA from the Arabidopsis RNA samples was synt |
| training/ont/Osativa/nip_pool_PRJNA953663 | 1 | excluded | unresolved | low | NA (no reads) | NA | - |

## 4. metadata 와 SSP 증거가 어긋나거나 SSP 가 규칙을 못 채운 런

규칙상 "반대 판정" 은 없다. 아래는 SSP 가 inconclusive 여서 metadata 만으로 정한 51행 + 검토가 필요한 관찰이다.

### 4.1 direct_RNA 로 적었지만 polyA3 < 30 % 라 서열 규칙을 못 채운 런 (35행)
전부 프라이머 0-1/2000, polyT5 ≤ 0.8 %, orientation audit PASS@1.00 (가닥특이적) 으로 dRNA 와 모순되지 않는다.
polyA3 가 낮은 이유는 dRNA basecall 에서 poly(A) 가 잘리거나 짧게 불리는 것으로 보이며, 특히 2019-2020 MinION 런에서 심하다.
- PRJNA752930 nip_dRNA_6tissue: SRR15371758, SRR15371759, SRR15371761, SRR15558518, SRR15558519, SRR15558520, SRR15558521,
  SRR15558523, SRR15558525 (polyA3 0.17-0.29; 같은 study 의 SRR15558517/22/24 는 0.31-0.36 으로 규칙 충족 → high)
- PRJNA1044249 nip_sheath: SRR26927871, SRR26927872, SRR26927873, SRR26927874, SRR26927875 (0.23-0.29; 나머지 5런 high)
- PRJNA1291274 indica_flagleaf: SRR34539095, SRR34539096 (0.28-0.30; SRR34539094/97 high)
- PRJNA681088 Ppatens dRNA_gametophore: SRR13154249-SRR13154255 (7런, 0.01-0.07)
- PRJNA517295 Ptrichocarpa sdx_dRNA: SRR8491764, SRR12676675 (training) 및 SRR12822922, SRR12822923, SRR12881185 (excluded 중복) (0.01-0.18)
- PRJNA672182 Ptrichocarpa sdx_drought: SRR12904202, SRR12904551, SRR12904624, SRR12904635, SRR13336716, SRR13336717 (0.00-0.17;
  metadata 는 study_title "Nanopore Direct-RNA Sequence ... during drought" 에서만 나옴 -- run-level 텍스트에는 없음)
- PRJNA705300 wt_only: SRR13799551 (0.015)

### 4.2 cDNA 로 적었지만 프라이머가 5 % 미만인 런 (15행)
- **PRJNA1291274 SRR34539092, SRR34539093** ("Nanopore cDNA of MH63 rep1/2", library_selection=cDNA): 프라이머 0/2000, polyT5 0,
  orientation PASS@1.00 -- 서열만 보면 같은 study 의 dRNA 런과 구별이 안 된다. 그러나 FASTA 헤더가 `SRR34539092.1 111:165|<uuid>/1`
  처럼 `start:end|` 좌표를 달고 있고(길이 = end-start), dRNA 런 헤더(`SRR34539095.1 <uuid>/1`)에는 없다. 제출자가 pychopper 류로
  프라이머 제거 + sense 정렬한 cDNA 리드를 올린 것으로 해석했다. 저자 확인 권장 (medium).
- **PRJNA416810 Gmax wm82_seed SRR6244032-SRR6244041** (10런, 2017 MinION, FL-cDNA/Oligo-dT): 14-mer 히트 2.1-3.6 % (VNP 위주),
  polyA3 = 0.000, polyT5 = 0.000, orientation FAIL@0.49-0.51 (양가닥). 리드 양끝에 오류 많은 SSP/VNP 잔재와 ONT adapter 가 보이고
  poly(A) 는 5-6 nt 로 눌려 있다 (구형 R9.4 basecaller 의 homopolymer 압축). cDNA 와 모순 없음, SSP 규칙만 못 채움 (medium).
- **PRJNA874115 col0_stigma SRR21232866, SRR21232867** (3.5-3.9 %; polyA3 0.44 / polyT5 0.41 양쪽 높음 = 프라이머가 잘린 양가닥 cDNA 패턴)
  -- 같은 study 의 SRR21232868 은 6.7 % 로 규칙 충족. cDNA (medium).
- **PRJNA1087576 FLIC SRR28341943-45** (validation): 프라이머 1-5/2000, polyA3 0.48-0.52 / polyT5 0.45-0.48 양쪽 높음, 리드가
  polyT 로 시작하거나 polyA 로 끝남 = 프라이머만 잘린 cDNA. metadata "ONT cDNA-seq" 명시. cDNA (medium).

### 4.3 metadata 가 침묵해서 서열로만 정한 런 (1행)
- **PRJNA1393863 buffer_control SRR36603452** (excluded, 매니페스트 note: "0/2,000 reads carry the ONT strand-switch primer" 로
  unresolvable): SSP/VNP 는 실제로 0 이지만 **SMARTer TSO 14-mer 가 1890/2000 (full 22-mer 1802/2000)**, orientation FAIL@0.52,
  리드 양끝에 ONT sequencing adapter (`…TTCGTTCAGTTACGTATTGCT`) 가 그대로 있다. 즉 template-switch(TSO) 기반 cDNA 라이브러리로
  protocol=cDNA 는 서열로 해결된다. 다만 kit 이 metadata 에 없으므로 umi_status 는 여전히 "언급 없음" 근거의 absent 다. 이 sheet 는
  cDNA/medium 으로 적었고 role 은 손대지 않았다 -- 제외 결정을 유지할지는 저자 판단.

### 4.4 검사하지 못한 런 (1행)
- **PRJNA953663 nip_pool SRR25203456** (excluded): pronghorn 에 리드 파일 없음 (매니페스트대로 DECLARED_NOT_SCANNED, ENA fastq 경로가
  디렉터리). ENA metadata 도 "fulllange rna of NIP", library_selection=RANDOM 뿐이라 protocol=unresolved, confidence=low.

## 5. UMI
140 unique runs 의 ENA 전체 필드(20개)를 `/\bUMIs?\b|unique molecular/i` 로 훑어 0건. kit 이름이 있는 유일한 런 (PRJNA732451,
SQK-PCS109) 도 UMI 없는 kit 이다. 따라서 143행 모두 `umi_status=absent`, 신뢰도 medium ("UMI kit 을 썼다는 기록이 없음").
PROVENANCE.txt 의 `umi_status unknown (§3.3 item 3 ...)` 과 같은 사실관계이며, 이 sheet 는 그것을 producer 가 요구하는 값으로 옮긴 것이다.

## 6. 산출물 / 중간파일
- `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv` -- 143행, 15열 (header: dataset, run, species, role, instrument, protocol, umi_status,
  ena_library_selection, ena_kit_or_protocol_text, ssp_vnp_hits_per_2000, polyA3_frac, polyT5_frac, mean_len, basis, confidence)
- pronghorn `~/ont_protocol_check/` : `runs.txt`, `reads_map.tsv` (run → reads path, orientation, umi 줄), `ssp_check.pl`, `ssp_results.tsv`
- local scratchpad (세션 한정): `ena/<run>.tsv` (140), `ena_all.tsv`, `ssp_results.tsv`, `reads_map.tsv`, `classify.py`

## 7. 실행한 명령 (요지)

```bash
# ONT rows of the manifest
awk -F'\t' 'NR==1 || $5 ~ /MinION|PromethION|GridION/' data/manifests/DATASET_ROLES.longread_v1.tsv > ont_manifest.tsv
cut -f2 ont_manifest.tsv | tail -n +2 | sort > ont_runs.txt        # 143 lines, 140 unique

# ENA metadata, one request per run (xargs -P 6; 3 retries)
F=study_accession,run_accession,experiment_accession,sample_accession,study_title,experiment_title,library_name,library_strategy,library_selection,library_source,library_layout,instrument_platform,instrument_model,sample_title,sample_alias,description,library_construction_protocol,read_count,base_count,fastq_ftp
curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$RUN&result=read_run&fields=$F&format=tsv" -o ena/$RUN.tsv

# pronghorn: locate reads via PROVENANCE, else find under evidence/ and data/
p=$(ls -d $R/evidence/longread_align_v1/*/*/$r/PROVENANCE.txt | head -1); awk '$1=="reads"{print $2}' $p
find $R/evidence $R/data -maxdepth 6 -name "$r.fa.gz" | head -1

# pronghorn login node: first 2,000 records per file (perl script reads zcat output and stops after 2,000 records)
awk -F'\t' '$3!=""{print $1"\t"$3}' reads_map.tsv | xargs -P 4 -n 2 perl ssp_check.pl > ssp_results.tsv
#   ssp_check.pl: counts reads containing SSP/VNP/TSO 22-mer or 14-mer (either strand), A{10,} in last 50 nt,
#   T{10,} in first 50 nt, mean length.

# classification + sheet
python3 classify.py     # rules of section 2.3; writes ONT_PROTOCOL.b6_v1.draft.tsv
```

## 8. Library key (A11) -- ENA experiment accession, added 2026-09-23

규칙: 한 library = 한 독립 제조(preparation). 문서화된 출처는 ENA/SRA **experiment accession** (SRX/ERX/DRX; one experiment = one library preparation), sample accession 은 보조 근거. 런(run) 은 library 를 만들지 않으므로 `library` 열은 experiment accession 그대로이고, ENA 레코드가 없으면 비워 두고 `library_basis` 에 이유를 적었다 (절대 run 으로 채우지 않음). ENA portal filereport(result=read_run) 에서 `experiment_accession, sample_accession, secondary_sample_accession, library_name` 을 받았다.

### 8.1 ONT (`ONT_PROTOCOL.b6_v1.draft.tsv` 에 `library, experiment_accession, sample_accession, library_basis` 4열 추가)
- 143행 모두 library 채움 (140 unique runs → 140 experiments). **여러 런이 한 experiment 를 공유하는 경우: 0.** cui2020 SRR10611193-95 는 같은 런이 두 dataset 경로에 실린 것이라 experiment 도 같다 (두 행 모두 동일 값).
- 보조 근거(sample): 한 BioSample 에 experiment 가 여럿인 경우 5건. 모두 ENA 가 experiment 를 따로 등록했으므로 규칙상 별개 library 다:
  - SAMN07965044 (wm82_seed_PRJNA416810): 5 experiments SRX3351215, SRX3351219, SRX3351220, SRX3351221, SRX3351222 = runs SRR6244034, SRR6244035, SRR6244036, SRR6244037, SRR6244041 (library_name: 2015A, 2015B, 2015C, 2015D, 2015E)
  - SAMN07965045 (wm82_seed_PRJNA416810): 5 experiments SRX3351216, SRX3351217, SRX3351218, SRX3351223, SRX3351224 = runs SRR6244032, SRR6244033, SRR6244038, SRR6244039, SRR6244040 (library_name: 1994A, 1994B, 1994D, 1994E, 1994F)
  - SAMN10824325 (sdx_dRNA_PRJNA517295): 5 experiments SRX5296330, SRX9156840, SRX9290560, SRX9290561, SRX9347134 = runs SRR12676675, SRR12822922, SRR12822923, SRR12881185, SRR8491764 (library_name: Direct-RNA Seq rep1 fast5, Direct-RNA Seq rep1 fastq, Direct-RNA Seq rep2 fast5, Direct-RNA Seq-rep2 fastq)
  - SAMN16560561 (sdx_drought_PRJNA672182): 6 experiments SRX9369085, SRX9369434, SRX9369507, SRX9369518, SRX9763455, SRX9763456 = runs SRR12904202, SRR12904551, SRR12904624, SRR12904635, SRR13336716, SRR13336717 (library_name: D5 rep1, D5 rep2, Day7 rep1, Day7 rep2, WT rep1, WT rep2)
  - SAMN30526306 (col0_stigma_PRJNA874115): 3 experiments SRX17242935, SRX17242936, SRX17242946 = runs SRR21232856, SRR21232866, SRR21232867 (library_name: Col-0_1, Col-0_2, Col-0_4)
  이 중 SAMN10824325 (Ptrichocarpa sdx_dRNA) 는 rep1/rep2 fastq 와 raw-fast5 재제출이 experiment 만 다르고 분자는 같다 -- experiment 키는 이를 구분하지 못하며, 매니페스트의 excluded 처리(2026-09-23)가 중복을 막는다. SAMN07965045/44 (Gmax wm82_seed 1994/2015, 런당 experiment 1개, library_name 1994A…2015E) 는 experiment 키로는 런마다 별개 library 인데, 실제로 flowcell 마다 다시 prep 했는지는 ENA 로 확인 불가 (저자 확인 항목).
- (sample, library_name) 이 같은데 experiment 가 여럿인 ONT 경우: 1건 -- Direct-RNA Seq rep1 fast5: SRX9290560, SRX9347134

### 8.2 PacBio (`data/manifests/PACBIO_LIBRARY.b6_v1.draft.tsv`, 72행 = 매니페스트의 non-ONT 행 전부, 역할 무관)
- 역할: {'b1_validation_only': 21, 'c2_training_eligible': 49, 'excluded': 2}. library 채움 69행, 비움 3행: `F1maize.FINAL.gff`, `F1maize.demux_FL_count.txt`, `F1maize.flnc.fa.gz` (pacbio/Zmays_wang2020_zenodo2611319; Zenodo 파일이라 ENA 레코드 없음). RETIRED_DO_NOT_USE 의 SRR26636578/82 는 ENA 레코드가 있어 library 를 채우고 `note` 에 EXCLUDED(WGS genomic HiFi) 를 적었다 -- 사용 여부는 role 이 막는다.
- **여러 런이 한 experiment 를 공유: 3 experiments** (모두 pacbio/Athaliana_zhang2023_PRJNA911826, validation-only, 각 2 runs = 같은 GSM 라이브러리를 SMRT cell 두 개로 읽음):
  - SRX18680724 (SAMN32207379, GSM6828405, GSM6828405: Col_biorep_1 Arabidopsis tha): SRR22719006, SRR22719007
  - SRX18680725 (SAMN32207378, GSM6828406, GSM6828406: Col_biorep_2 Arabidopsis tha): SRR22719004, SRR22719005
  - SRX18680726 (SAMN32207377, GSM6828408, GSM6828408: Col_biorep_3 Arabidopsis tha): SRR22719002, SRR22719003
- 보조 근거(sample): 한 BioSample 에 experiment 가 여럿인 PacBio 경우 10건 (Sbicolor wray 7, Ptrichocarpa jgi 2, Zmays B73_ccs 1). 규칙상 별개 library.
- **주의 -- (sample, library_name) 이 같은데 experiment 가 여럿인 PacBio 경우 15건** (JGI 제출: 같은 JGI library name 을 SMRT cell 마다 experiment 로 따로 등록한 것으로 보인다). experiment 키를 그대로 쓰면 이들은 서로 다른 library 로 세어지고, 실제로 한 prep 이면 A11 의 독립성이 과대평가된다. 저자가 experiment 키를 유지할지, JGI 런에 한해 library_name 으로 합칠지 정해야 한다:
  - jgi_F1hybrid_PRJNA709499 SAMN18243171 library_name=GZHTA: 3 experiments (SRX30251338, SRX30251339, SRX30251340) = runs SRR35145694, SRR35145695, SRR35145696
  - wray_PRJNA1275171 SAMN49012086 library_name=HOTXU: 2 experiments (SRX29431938, SRX29431939) = runs SRR34257871, SRR34257872
  - wray_PRJNA1275171 SAMN49012086 library_name=HOTXW: 3 experiments (SRX29431875, SRX29431876, SRX29431877) = runs SRR34257933, SRR34257934, SRR34257935
  - wray_PRJNA1275229 SAMN49011411 library_name=HOAGH: 2 experiments (SRX29432161, SRX29432162) = runs SRR34258142, SRR34258143
  - wray_PRJNA1275229 SAMN49011411 library_name=HOAGN: 2 experiments (SRX29432159, SRX29432160) = runs SRR34258144, SRR34258145
  - wray_PRJNA1275264 SAMN49012185 library_name=HOTYB: 2 experiments (SRX29432184, SRX29432185) = runs SRR34258119, SRR34258120
  - wray_PRJNA1275264 SAMN49012185 library_name=HOTYC: 3 experiments (SRX29432143, SRX29432144, SRX29432145) = runs SRR34258159, SRR34258160, SRR34258161
  - wray_PRJNA1275290 SAMN49013182 library_name=HOTXZ: 2 experiments (SRX29431902, SRX29431959) = runs SRR34257851, SRR34257908
  - wray_PRJNA1275290 SAMN49013182 library_name=HOTYA: 3 experiments (SRX29432191, SRX29432192, SRX29432193) = runs SRR34258111, SRR34258112, SRR34258113
  - wray_PRJNA1275324 SAMN49012170 library_name=HOAGC: 2 experiments (SRX29432157, SRX29432158) = runs SRR34258146, SRR34258147
  - wray_PRJNA1275324 SAMN49012170 library_name=HOAGG: 2 experiments (SRX29432154, SRX29432156) = runs SRR34258148, SRR34258150
  - wray_PRJNA1275413 SAMN49012300 library_name=HOZZG: 3 experiments (SRX29431882, SRX29431883, SRX29431884) = runs SRR34257926, SRR34257927, SRR34257928
  - wray_PRJNA1275413 SAMN49012300 library_name=HOZZH: 2 experiments (SRX29431885, SRX29431886) = runs SRR34257924, SRR34257925
  - wray_PRJNA1275435 SAMN49012517 library_name=HOTXX: 2 experiments (SRX29432219, SRX29432220) = runs SRR34258084, SRR34258085
  - wray_PRJNA1275435 SAMN49012517 library_name=HOTXY: 3 experiments (SRX29431905, SRX29431906, SRX29431907) = runs SRR34257903, SRR34257904, SRR34257905

### 8.4 명령
```bash
# PacBio rows of the manifest
awk -F'\t' 'NR==1 || $5 !~ /MinION|PromethION|GridION/' data/manifests/DATASET_ROLES.longread_v1.tsv > pacbio_manifest.tsv   # 72 rows
# ENA per run (ONT: the section-2.1 fetch already carried experiment_accession/sample_accession/library_name; PacBio: 69 SRA runs)
F=study_accession,run_accession,experiment_accession,sample_accession,secondary_sample_accession,library_name,library_strategy,library_selection,instrument_model,experiment_title,description,read_count
curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$RUN&result=read_run&fields=$F&format=tsv" -o ena_pb/$RUN.tsv
# join: library = experiment_accession; empty + reason when no ENA record; multi-run experiments = experiment_accession values with >1 run
```

### 8.3 저자 규칙 (2026-09-23): (sample_accession, library_name) 이 여러 experiment 에 걸치면 그 쌍이 library 키

규칙 (두 sheet 에 기계적으로 적용):
- 두 sheet 에 `library_name` 열 추가 (ENA `library_name` 원문, 비어 있으면 `-`; sample_accession 다음 위치).
- (sample_accession, library_name) 그룹이 **서로 다른 experiment 를 2개 이상** 포함하고 library_name 이 비어 있지 않으면
  `library = <sample_accession>:<library_name>`, `library_basis = "ena:sample+library_name (author 2026-09-23: multi-experiment prep)"`.
- 그 외 런은 `library = experiment_accession`, basis 그대로. **library_name 이 빈 런은 절대 합치지 않는다** (ONT 8런:
  DRR424731-33 col0_DRP009401, SRR10611193-95 cui2020, SRR14635690/91 Vvinifera callus -- 모두 experiment 가 런마다 달라 합칠 대상도 아니다;
  PacBio 는 빈 library_name 없음).
- 검증: 서로 다른 sample_accession 을 가진 두 런이 같은 library 를 갖는 경우 -- ONT 0, PacBio 0.

| sheet | 행 | unique runs | libraries | re-keyed runs | re-keyed groups | basis 분포 |
|---|---|---|---|---|---|---|
| ONT_PROTOCOL.b6_v1.draft.tsv | 143 | 140 | 139 | 2 | 1 | experiment 141 행, sample+library_name 2 행 |
| PACBIO_LIBRARY.b6_v1.draft.tsv | 72 | 72 | 45 | 36 | 15 | experiment 33 행, sample+library_name 36 행, no-ENA 3 행 |

ONT re-key 결과 (library → runs):
- `SAMN10824325:Direct-RNA Seq rep1 fast5` → SRR12822923, SRR12881185 (둘 다 excluded 중복 fast5 재제출; 훈련 런 SRR8491764/SRR12676675 는
  library_name 이 달라 experiment 키 유지)

PacBio re-key 결과 (library → runs), 15 groups / 36 runs, 전부 c2_training_eligible JGI 제출:
- `SAMN18243171:GZHTA` → SRR35145694, SRR35145695, SRR35145696 (jgi_F1hybrid_PRJNA709499)
- `SAMN49012086:HOTXU` → SRR34257871, SRR34257872 (wray_PRJNA1275171)
- `SAMN49012086:HOTXW` → SRR34257933, SRR34257934, SRR34257935 (wray_PRJNA1275171)
- `SAMN49011411:HOAGH` → SRR34258142, SRR34258143 (wray_PRJNA1275229)
- `SAMN49011411:HOAGN` → SRR34258144, SRR34258145 (wray_PRJNA1275229)
- `SAMN49012185:HOTYB` → SRR34258119, SRR34258120 (wray_PRJNA1275264)
- `SAMN49012185:HOTYC` → SRR34258159, SRR34258160, SRR34258161 (wray_PRJNA1275264)
- `SAMN49013182:HOTXZ` → SRR34257851, SRR34257908 (wray_PRJNA1275290)
- `SAMN49013182:HOTYA` → SRR34258111, SRR34258112, SRR34258113 (wray_PRJNA1275290)
- `SAMN49012170:HOAGC` → SRR34258146, SRR34258147 (wray_PRJNA1275324)
- `SAMN49012170:HOAGG` → SRR34258148, SRR34258150 (wray_PRJNA1275324)
- `SAMN49012300:HOZZG` → SRR34257926, SRR34257927, SRR34257928 (wray_PRJNA1275413)
- `SAMN49012300:HOZZH` → SRR34257924, SRR34257925 (wray_PRJNA1275413)
- `SAMN49012517:HOTXX` → SRR34258084, SRR34258085 (wray_PRJNA1275435)
- `SAMN49012517:HOTXY` → SRR34257903, SRR34257904, SRR34257905 (wray_PRJNA1275435)

experiment 키가 그대로 남은 multi-run library (규칙 무관, §8.2 와 동일): SRX18680724/25/26 (zhang2023, 각 2 runs).

sha256 (재키 후): ONT `e21425e1eeaa60ac87648464b684f61fabe06c848ec8d0de4da448d54695b029`,
PacBio `8de4d5830ff1a88587d838c3521a64f2b62803c2f0c4e4d717a401ce102e4dce`.
