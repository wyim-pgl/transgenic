# quarantine.md — 저장소 격리소

> 📌 **여기 있는 것은 전부 "끝난" 것이다.** 계획이 틀렸거나, 시도했는데 안 됐거나, 배제된 것.
> **미해결 항목은 여기 두지 않는다** — 그건 저장소 밖 `Transgenic/todo.md`로 간다.
> 다음 문제 해결 때 **false positive · 추가 피해 · 위협 분석**의 근거로 쓰려고 남긴다.
>
> "지금 무엇을 쓰는가"의 정본은 `revision/protocols/PROTOCOL_B1_frozen_v1.md`(v1.25) ·
> `docs/gsf_spec_v1.md` · `configs/b5_400m_win_v3.json` · `data/freeze/` ·
> GitHub 이슈 wyim-pgl/transgenic #4–#63. 최종 갱신 2026-09-03.
>
> 이 파일은 **저장소 문서·데이터·설계·코드 경로**의 기각 목록이다. 랩 내부 핸드오프 이력에서
> 격리한 것(과거 스냅샷, 낡은 수치, 끝난 "다음 작업")은 저장소 밖 `Transgenic/quarantine.md`에 있다 —
> 공개 저장소가 랩 내부 이력을 담지 않도록 분리했다.

`README.md`는 **공개된 published 모델**의 문서다. 2026-09 리비전(B5)에서 레시피·분할·창 정책·어휘가
전부 바뀌었기 때문에, README를 그대로 따라 하면 B5가 고치려는 결함을 그대로 재현하게 된다.
`guide/handoff-hygiene.md` 규약대로 **원문은 지우지 않고 표시만** 했고, 설명은 전부 이 파일에 모았다.

마커 어휘: `❌ SUPERSEDED` 더 이상 참이 아님(인용 금지) · `⚠️ CAUTION` 조건부로만 참 ·
`✏️ PARTIAL` 일부만 해소. 모든 항목에 날짜와 "현재 진실이 있는 곳"을 붙인다.

전수 목록은 `grep -n 'SUPERSEDED\|CAUTION (\|PARTIAL (' README.md quarantine.md`.

---

## 1. README.md 격리 목록

README의 해당 줄 **바로 위**에 이 표의 ID를 가리키는 마커가 달려 있다.
행 번호는 마커 삽입 **이후** 기준이다(2026-09-03). 마커 자체를 지우면 번호가 다시 어긋난다.

⚠️ **마커 배치 규칙**: "대상 줄 바로 위"를 문자 그대로 지키면 코드 펜스와 마크다운 표를 깨뜨린다.
처음 삽입할 때 README 4곳·CLAUDE.md 4곳이 그렇게 들어갔고, 그중 하나는 `\`로 이어진 명령
한가운데라 **예제를 복사하면 깨지는 상태**였다. 대상이 펜스나 표 안에 있으면 마커는 **그 블록 전체
바로 위**로 올린다. 검사:

```python
# 펜스 안: 앞선 ``` 개수가 홀수 / 표 분리: 앞뒤 비어 있지 않은 줄이 둘 다 '|'로 시작
```

### R1 ❌ SUPERSEDED (2026-09-03) — `random_split` 75/10/15 분할

`README.md:666-672`(`random_split` 호출은 :669), `:1159`. **B5 전체가 이 결함을 고치려고 존재한다.** 두 가지가 겹쳐 있었다.

1. 분할이 **RC 증강 이후 행 단위**로 일어나 정방향 행과 그 역상보 쌍둥이가 train/test로 갈라졌다
   (twin leakage). `train/train_HyenaTransgenic.py:612`가 지금은 이 경로에 경고를 찍는다 —
   *"legacy row-level random_split after RC augmentation (twin leakage); B5 runs must pass --config"*.
2. 상동 유전자(orthogroup)가 분할을 가로질렀다.

**현재 진실**: 동결 분할표 `data/splits/b5_orthogroup_split_v1.tsv`(334,642행, sha256 `904c6265…`,
OrthoFinder 3.1.5 HOG N1, 이슈 #14)와 `src/transgenic/training/b5_runtime.py:192` `split_row_numbers()`.
B5 학습은 `--config`가 필수이며 분할은 DB의 `split` 컬럼에서만 읽는다.

### R2 ❌ SUPERSEDED (2026-09-03) — `train_HyenaTransgenic_RTX4090.py` / `_GB10.py`

`README.md:685-700`이 이 두 스크립트를 권한다. **둘 다 아직 `random_split`을 쓴다**
(`train_HyenaTransgenic_RTX4090.py:804`, `train_HyenaTransgenic_GB10.py:573`) — 즉 R1의 결함이
그대로 살아 있다. B5 학습에 쓰면 안 된다.

**현재 진실**: `train/train_HyenaTransgenic.py --config configs/b5_400m_win_v3.json`. 하드웨어별
분기는 CLI 플래그로 처리한다(`TRANSGENIC_NO_COMPILE=1`이 GB10용 torch.compile 우회).

### R3 ❌ SUPERSEDED (2026-09-03) — "OOM-safe batch skipping" / 과대 배치 자동 건너뛰기

`README.md:685`, `train/train_HyenaTransgenic_RTX4090.py` docstring 9행
("Automatically skips oversized batches (>48k tokens) that would trigger OOMs").

이것이 **프로토콜 A35가 금지한 바로 그 동작**이다. 배치 루프가 예외를 전부 삼키고 건너뛴 결과,
129,024-nt 계층에서 1,103 배치 중 **1,093개가 조용히 사라졌는데 손실 곡선은 정상으로 보였다**
(2026-09-02 실측). 사전등록된 연구에서 데이터를 소리 없이 버리는 것이자, 누적 그룹과 에폭 손실
분모까지 망가뜨린다.

**현재 진실**: 커밋 `b5e4471`, 프로토콜 A35. 이제 CUDA OOM만 잡고, 고정 레시피 경로는 표본·shape·
메모리 상태를 담아 예외를 올린다. 24 GB 카드로 큰 계층을 돌리면 즉시 실패하는 것이 **정상 동작**이다.

### R4 ⚠️ CAUTION (2026-09-03) — "resumed from exactly where it left off — no wasted computation"

`README.md:720`. 에폭 **경계** 재개에는 참이지만 에폭 **중간** 재개에는 거짓이다.
학습 DataLoader가 `shuffle=True`인데 seedable sampler가 꺼져 있어, `accelerator.load_state`가
복원한 중간 RNG로 **다른 순열**이 뽑히고 `step < resume_step` 건너뛰기가 엉뚱한 샘플을 버린다.
실측(torch 2.5.1 / accelerate 1.14.0): 64개 중 20에서 재개 → 15개 중복 관측, 15개 미관측.
`use_seedable_sampler=True`면 0·0.

**현재 진실**: 이슈 **#59**. A28 잡 체인이 재개를 상시로 쓰므로 첫 시드 전에 고쳐야 한다.

### R5 ⚠️ CAUTION (2026-09-03) — 입력 길이 "Multiples of 6,144nt (max 49,152nt)"

`README.md:295`, `:808`. published 체크포인트(`sym6144-v1`)에 대해서는 참이다. B5는 아니다.

**현재 진실**: `tile6144-v3`(프로토콜 A26). 계층 **30,720 / 61,440 / 129,024 nt**,
`max_encoder_seqlen` 129,024, 라벨은 타일 안에 완전히 들어간 모든 유전자의 정규 연접
(없으면 `<empty>`). `docs/gsf_spec_v1.md` §1a·§1b, `configs/b5_400m_win_v3.json`.

### R6 ⚠️ CAUTION (2026-09-03) — 토크나이저 어휘 v1(272) / v2(288)

`README.md:286`. published 체크포인트는 v1 272, 이 저장소 기본값은 v2 288이라는 서술은 맞다.
**B5는 v3 290**이다(v2 + `<gene>` + `<empty>`).

**현재 진실**: `configs/b5_400m_win_v3.json`의 `vocab_size: 290`, `gff_vocab_version: "v3"`.
⚠️ v3 토크나이저에는 미해결 비대칭이 있다 — `<empty>` 라벨만 `<s>`가 없다(토크나이저 2토큰 vs
계약·스펙·테스트 3토큰, 25,159 타일 = 9.24 %). 이슈 **#58**.

### R7 ⚠️ CAUTION (2026-09-03) — "92% base-level F1 on *Arabidopsis thaliana* test data"

`README.md:44`, `:269`. 이 수치는 **R1의 누출 있는 분할**에서 측정됐다 — RC 쌍둥이가 train/test로
갈라졌고 상동 유전자가 분할을 가로질렀다. 재측정 전까지 "일반화 성능"으로 인용하면 안 된다.
리뷰어 R2의 "유용성 미입증" 지적도 이 지점에 걸려 있다.

또한 B5의 새 분할에서는 **"test set"의 정의 자체가 다르다**: 유전자 split(orthogroup 단위)과
타일 split(위치 블록 단위)이 독립이라 명목 test 유전자의 **16.3 %만 실제로 라벨된다**
(valid 25.1 %, train 92.0 %). 이슈 **#61**. 재측정 수치를 쓸 때 어느 정의인지 명시해야 한다.

**현재 진실**: 미정 — B5 재학습(#19–#20) 후 새 test로 전 평가를 재실행한다. 그때까지 이 수치는
published 모델의 자기 분할 위 성적으로만 인용한다.

### R8 ⚠️ CAUTION (2026-09-03) — "Building a DuckDB Database" 절 전체

`README.md:303-547`. `genome2GSFDataset` / `scripts/create_database.py`와 거기 실린 `geneList`
스키마 표(`README.md:318-336`)는 **legacy 12컬럼**이다. B5 스키마에는 그 위에
`species_id`·`gene_id`·`orthogroup_id`·`split`·`strict_holdout`·`is_rc`·`ordering_version`·
`build_version`·`source_fasta_sha256`·`source_gff_sha256`·`split_file_sha256`·`window_policy`·
`gsf_token_count`·`contig_boundary`·`n_transcripts`·`train_weight`·`qc_flags`·`gene_id_original`이
더 있고, `gene_key_map`·`gene_split`·`build_manifest`·`window_genes`·`tile_blocks`·
`rejected_records` 여섯 테이블이 함께 있다.

**현재 진실**: `scripts/build_b5_database.py`(종별 빌드) → `scripts/merge_b5_databases.py`(병합·동결)
→ `scripts/validate_b5_database.py` + `scripts/report_b5_database.py`. 스키마 계약은
`docs/gsf_spec_v1.md` §6, 실제 동결물 기록은 `data/freeze/`.

### R9 ⚠️ CAUTION (2026-09-03) — `--save-every-n-steps` 기본값 5000

`README.md:712`. legacy 스크립트에 대해서는 맞다. B5 경로의 기본값은 **200**이다
(`b5_runtime.parse_args`) — ACCESS 선점 재개를 전제로 하기 때문이다.

### R10 ⚠️ CAUTION (2026-09-03) — "Benchmark Results" 절 (GB10 / RTX 4090)

`README.md:1048-1227`. legacy 레시피(6,144 배수 창)의 처리량이며 **B5 계층 벤치마크가 아니다**.

**현재 진실**(이슈 #18, RTX 4090 24.56 GB, bf16, batch 1, gradient checkpointing on):

| 계층 (nt) | 최대 메모리 | 미니배치당 | 라벨 토큰/s | OOM |
|---|---|---|---|---|
| 30,720 | 12.05 GB | 0.70 s | 573 | 0 / 240 |
| 61,440 | 16.94 GB | 1.54 s | 524 | 0 / 240 |
| 129,024 | 24 GB 초과 | (외삽 3.2 s) | — | 1,093 / 1,103 |

**129,024 계층은 24 GB 카드에서 물리적으로 불가능하다.** 비용은 라벨 길이에 선형, 입력 길이에는
거의 무관. 메모리 = 고정 6.6 GB + 약 0.16 MB/nt.

### R11 ⚠️ CAUTION (2026-09-03) — `CLAUDE.md`도 같은 legacy 서술을 담고 있다

에이전트가 저장소에서 **가장 먼저 읽는 파일**이라 영향이 크다. ✅ **2026-09-03 마킹 완료** —
배너 1개 + 마커 10개(원문 삭제 0). 아래가 그 목록이다.

행 번호는 마커 삽입 **이후** 기준(2026-09-03). CLAUDE.md는 `.git/info/exclude`로 git에서 제외돼 있어
이 마킹은 로컬 디스크에만 있고 커밋되지 않는다 — 새로 클론하면 사라진다.

| CLAUDE.md | 서술 | 현재 진실 |
|---|---|---|
| :29 | `scripts/create_database.py … --add-rc-iso-only` | B5 빌드 경로는 `scripts/build_b5_database.py`. `--add-rc-iso-only` 단독은 무효였고 enum(`--rc none/all/isoform-only`)으로 교체됐다(IMPLEMENTATION_ORDER_B5_C0_C2_v1) |
| :56 | "Run a test script (ad-hoc scripts, **no pytest framework**)" | ❌ 이제 `tests/`에 pytest 스위트 21파일 **155건**이 있고 전부 통과한다(`pytest -q tests`, duckdb 필요) |
| :77, :125 | "Published checkpoint, **92 % F1**" | R7 — 누출 있는 분할에서 측정된 수치다 |
| :79 | "Wide (1.17B) — GB10/RTX 4090 **training target**" | B5 정본은 **400M**(`configs/b5_400m_win_v3.json`). 학습 스크립트 기본이 1.17B였던 것이 B5 설정 명시의 이유다 |
| :184 | `--save-every-n-steps 5000` | R9 — B5 기본값 200 |
| :197 | "Step-level resume: skips already-processed micro-batches within an epoch" | R4 / 이슈 **#59** — 에폭 중간 재개는 셔플이 달라져 **엉뚱한** 마이크로배치를 건너뛴다 |
| :234 | "**49152 bp** — max encoder input (8 × 6144)" | R5 — B5는 `max_encoder_seqlen` **129,024** |
| :237 | "**2048** — max decoder position embeddings" | B5는 **8,192**(`max_decoder_position_embeddings`, 라벨 캡도 8,192) |
| :242 | "GFFTokenizer vocab (**272 tokens**)" | R6 — B5는 v3 **290**(`<gene>`·`<empty>` 추가) |

---

---

## 2. 데이터·데이터셋 제외

### D1 종 제외 — `data/manifests/b5_excluded_species_v1.tsv`

| 종 | 사유 |
|---|---|
| *Z. mays* | held-out 테스트 종(RefGen_V4). B5 DB에 한 행도 들어가면 안 된다(legacy GRMZM 174행 제거 포함) |
| *B. rapa* | 확장 벤치마크 전용 |
| *L. sativa* | 확장 벤치마크 전용 |
| *S. lycopersicum* | 확장 벤치마크 / 조건부 B6 테스트 종 |

`merge_b5_databases.py`가 소스 파일명으로도, 소스 DB 안의 `species_id`로도 거부한다(커밋 `527791c`).
동결물 실측: Zmays 0행, `Zm%`/`GRMZM%` gene model 0행.

### D2 ❌ PacBio RS II / Sequel (I) 세대 제외 — 프로토콜 v1.3 (§A13)

Wang 2016(RS II) · Wang 2018 HQ(RS II/Sequel) · Wang 2020 FLNC(Sequel) · Cui 2020 PacBio(Sequel) ·
Han 2023(Sequel)이 tier 1에서 제거됐다. **✏️ PARTIAL (v1.5, §A14)**: 테스트 종(*Z. mays*,
*S. lycopersicum*) **검증** 증거는 기기 세대 무관으로 복원됐다 — Wang 2018·Wang 2020 복원, Wang 2016은
RefGen_V4 주석을 만드는 데 쓰였으므로 독립성 위반으로 계속 제외.

### D3 ❌ subreads-only PacBio 세트 제외 — 원본 BAM 없이는 CCS 재생성 불가

프로토콜 §3: PRJNA822292 · PRJNA983493(iFLAS, Sequel II지만 subreads-only) · Kurihara 2022
PRJDB12660 · PRJNA921723 — ZMW 이름 소실 또는 BAM 부재로 제외.

⚠️ **CAUTION (2026-09-03)**: 이 규칙이 실제 다운로드에는 적용되지 않았다. `evidence/training/`의
**PacBio 51런 전부(43 GB, 11개 데이터셋)가 `_subreads.fastq.gz`**이고 CCS/FLNC는 0건이다.
`MODEL_RE`는 기기(Sequel II/IIe)를 맞게 통과시키지만 데이터 산물을 보지 않고, subreads 가드로 둔
`MAX_READS=5000000`은 실제 세트가 24.8만–157.8만 리드라 발동하지 않는다. 검증 세트도 3개 중 2개가
같은 상태이며 그중 `pacbio/Zmays_B73_ccs_PRJNA1470126`은 **디렉토리 이름이 ccs인데 내용은 subreads**다.
**✅ 저자 결정 2026-09-03: 제외.** 51런 43 GB를 삭제하지 않고
`evidence/RETIRED_DO_NOT_USE/training_pacbio_subreads_20260903/`로 경로 격리했다(그 디렉토리의
`README.md`에 종·데이터셋별 내역과 되돌리는 법). `longread_fetch.sh`는 이제 `_subreads`가 들어간 URL을
이름으로 거부한다 — `MODEL_RE`(기기)도 `MAX_READS`(리드 수)도 데이터 산물을 보지 않기 때문이다.
`revision/protocols/TRAINING_EVIDENCE_v1.md`의 PacBio 행 전부에 ❌ 배너를 달았다(그 표는 기기와 리드
수만 보고 "SqII CCS/FLNC/HiFi"로 적었는데, 같은 문서의 규칙 문단은 이미 subreads를 배제하고 있었다 —
**규칙과 표가 모순이었고 규칙이 맞다**). 이슈 **#60** 종료.

**파급(확정)**: *S. bicolor*는 ONT가 없어 **장독 증거 0**. *B. distachyon*은 원래 0. 남는 ONT 런 수 —
A. thaliana 6 · G. max 27 · O. sativa 12(수집 중) · P. patens 수집 대기 · P. trichocarpa 11 ·
S. italica 9 · V. vinifera 11.

⚠️ 검증 세트의 PacBio 2건은 **이 결정에 포함되지 않았다**: `pacbio/Athaliana_zhang2023_PRJNA911826`은
프로토콜이 "subreads.bam → ccs → FLNC"로 **변환을 전제**한 세트인데 변환이 아직 실행되지 않았고,
`pacbio/Zmays_B73_ccs_PRJNA1470126`은 이름만 ccs다. 둘 다 별도 판단이 필요하다.

### D4 ❌ AtRTD3 자체 Iso-Seq(PRJNA755474) 제외 — 독립성 위반

AtRTD3 주석이 그 데이터로 만들어졌으므로 검증에 쓸 수 없다.

### D5 ⚠️ *A. thaliana* 검증 전용 세트 — 학습 금지

FLIC(PRJNA1087576) · Zhong 2025(PRJEB77203) · Zhang 2023(PRJNA911826, Col-0 WT SRR22719002–07만;
mutant run 제외). Cui 2020(PRJNA594286)은 학습 가능. 프로토콜 §A14·§A18.3.

### D6 ❌ NCBI `db=nucest`는 EST 전용이 아니다 (2026-09-01)

dbEST 폐지 후 nuccore로 통합되어 특허·mRNA·게놈 레코드(평균 200 KB)가 함께 내려온다. 첫 시도는
배치 하나가 수 GB에 달해 중단·삭제했다. **올바른 질의**: `db=nuccore`,
`txid<N>[Organism] AND gbdiv_est[PROP]`.
❌ 함께 폐기된 수치: "maize EST 489만" 등은 nuccore 전체 건수였다. 실제 EST 건수는
maize 2,019,959 · Ath 1,529,700 · Gmax 1,461,724 · Osa 1,255,251 · Vvi 446,853 · Ppa 382,587 ·
Sly 301,030 · Sbi 209,835 · Bdi 206,255 · Ptr 89,943 · Sit 66,027.

### D7 ⚠️ *V. vinifera* lncRNA 전용 유전자 6,205개 — 타일링 전 탈락

`Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3`에만 `lncRNA` 전사체 7,934건이 있다(다른 8종 GFF3에는
없음). lncRNA 전용 유전자는 정확히 6,205개이고 전부 `no CDS`로 거부된다. **의도된 제외가 아니라
부작용이 있다**: 이 유전자들은 `by_chrom` 진입 *전*에 탈락하므로 A29 누출 마스킹의 대상이 아니고,
그중 valid 634 · test 971이 train 타일에 **평문으로 남는다**. 서열 수준 실측: valid/test split
1,210개가 forward train 타일에 완전히 들어가고(test 737), 표본 3,000쌍 중 2,684(89.5 %)가
유전자 구간 전부 평문 ACGT, N 비율 중앙값 0.0000. → 이슈 **#56**.

---

## 3. 방법·설계 기각

### M1 ❌ CD-HIT 등 서열 클러스터링 (프로토콜 v1.2, 커밋 `6d89d01`)

모든 EST를 개별적으로 게놈에 정렬하고 정렬된 구간만 사용한다. 지지 수는 정렬된 accession 기준.

### M2 ❌ raw accession 단위 지지 계수 (프로토콜 v1.2.1)

독립 **분자** 단위로 센다(클론·버전 병합, 클론 ID가 없으면 같은 라이브러리 + 동일 정렬 서명만 병합,
라이브러리당 10 cap, 고신뢰 접합 = 독립 분자 ≥ 2).

### M3 ❌ masked-fraction 임계값 0.25 (2026-09-02, 실측으로 기각)

Codex 단독 계획의 값. *A. thaliana* 타일의 **49–61 %를 삭제**했다. 실측 분포: 중앙값 0.245/0.262/0.280,
p95 0.538/0.477/0.436(30,720/61,440/129,024). **채택값 0.60** — 모든 계층의 p95 위에 있고 몇 퍼센트만
버린다(동결물 실측: 1,543 타일). 프로토콜 A33.

### M4 ❌ A27 reciprocal overlap 0.50 (2026-09-02, 실측으로 기각)

진짜로 겹치는 유전자 191쌍을 하나로 병합해 버렸다. Kimi의 대안(공유 intron 규칙)도 실측하니 200쌍으로
**더 나빴다**. 9종 참조 주석을 직접 세어 104쌍짜리 최선안을 채택했다. → `[[consult-codex-and-kimi-on-problems]]`

### M5 ❌ whole-tile 마스킹 (프로토콜 A32, 커밋 `f2a330e`)

hard 플래그 유전자를 만나면 타일 전체의 가중치를 0으로 만들던 방식. *A. thaliana* smoke 빌드에서 계층별로 타일의
**14–46 %**를 제거했다. **현재**: hard 플래그는 **유전자 단위**로 N-마스킹되고 라벨에서 빠지며, 타일 가중치는 1로 남는다.
동결물 실측: `train_weight = 0` 행 **0개**.

### M6 ❌ strict held-out 좌위가 자기 블록을 test로 강제 (프로토콜 A31)

*A. thaliana* held-out 3,430 좌위가 121개 블록 중 120개를 건드려 **train 타일이 23개만 남았다**.
**현재**: 블록은 추첨만 하고, held-out 유전자는 orthogroup split상 test이므로 train/valid 타일에서는
A29 규칙으로 N-마스킹·무라벨 처리된다. 재빌드 후 Arabidopsis train 타일 7,631개.

### M7 ❌ 무작위 서열 치환 (프로토콜 A33.3)

마스킹을 N 대신 무작위 염기로 채우자는 안. 그럴듯한 motif를 만들 수 있고 개입을 숨긴다. N은 어셈블리
갭과 같은 모양이라는 점이 오히려 장점.

### M8 ❌ 네 번째 타일 offset (프로토콜 A33.4)

보장선을 `3·tier/4 − 2,000`으로 올리지만 유전자 0.026 %를 위해 채택하지 않았다. 채택된 3-offset의
보장선은 `2·tier/3 − 2,000` = 18,480 / 38,960 / 84,016 nt.

### M9 ⚠️ Helixer **도구** 사용 (v3.1에서 저자 정정)

Helixer를 파이프라인에 넣는 것이 아니라, **Helixer식 분할 헤드를 자체 모델에** 붙인다
(C2: HyenaSegment + EST/ONT 고신뢰 마스크 라벨, 추론은 서열만). Helixer는 6주차 비교 대상일 뿐이다.

### M10 ❌ evigene를 이번 논문 범위에 포함 (v3에서 기각)

후속 논문으로. 이번에는 직접 정렬 접합 DB(C0)를 B3 재순위 채널로만 쓴다.

### M11 ⚠️ `tier_margin_unguaranteed` — 프로토콜에만 있고 코드에 없다

`PROTOCOL_B1_frozen_v1.md:439`는 "build manifest에 계수된다"고 하지만 어떤 `.py`에도 없다.
`gsf_contract.covered_with_margin`(:759)은 구현돼 있으나 빌더가 호출하지 않는다.
**현재**: `scripts/report_b5_database.py`가 `gene_key_map`에서 사후 재계산한다. 두 수치를 구분해서 보고한다 —
길이 보장선 초과 3,418 / 653 / 86(프로토콜 자체 수치와 정확히 일치), 그중 세 offset 모두에서
margin 미충족 2,034 / 434 / 44. contig 길이가 DB에 없어 후자는 **상한**이다. → 이슈 **#57**.

---

## 4. 코드 경로 기각·대체

### C1 ❌ `INSERT … SELECT nextval('row_id') … ORDER BY rn` (2026-09-03)

**SELECT 목록의 시퀀스는 병렬 스캔 중에 평가되고, 정렬은 이미 번호가 붙은 튜플에 적용된다.**
duckdb 1.5.5 실측(원본이 물리적으로 rn 순서인 상태 = 실제 종별 빌드 모양):

| 방식 | 10만 행 | 30만 행(2 스레드) | 30만 행(16 스레드) |
|---|---|---|---|
| 서브쿼리 `ORDER BY rn` + `nextval` | 정확 | 262,144 / 300,000 오류 | 300,000 / 300,000 오류 |
| 최상위 `ORDER BY` + `nextval` | 정확 | 오류 | 오류 |
| `row_number() OVER (ORDER BY rn)` | 정확 | **0 오류** | **0 오류** |

행 수·split·행 내용·기본키·`validate_b5_database`가 전부 정상으로 보이면서 rn만 의미를 잃는다.
학습도 그냥 돌아간다(`split_row_numbers`·`isoformDataHyena`가 실행 시점에 인덱스를 다시 만들고 rn을
저장하지 않으므로). 깨지는 것은 감사 추적뿐 — 그게 동결의 존재 이유다.

**현재**: `scripts/merge_b5_databases.py:99` `row_number() OVER (ORDER BY rn) + {before}`.
회귀 테스트 `test_merged_rn_follows_the_source_rn_order`는 **30만 행**이어야 실패한다
(3만 행에서는 결함 있는 쿼리도 통과한다 — 정렬 블록 경계가 262,144행).
⚠️ 저장소의 다른 `nextval` 사용처는 전부 한 행짜리 `VALUES (nextval(…), …)`라 안전하다.

### C2 ❌ DuckDB 파일 md5를 "재현 가능한 동결"로 쓰는 것

파일 바이트는 쓰기 순서·여유 공간 배치·버전에 의존하므로 같은 아홉 소스로 다시 병합해도 재현되지 않는다.
파일 md5는 **변조 증거**로만 유효하다.
**현재**: 행별 md5를 DuckDB 안에서 만들고 정렬한 다이제스트를 하나의 sha256으로 접는다 —
행의 **집합**에 대해 정의되고 파일 배치·`rn`과 무관하다. 회귀 테스트
`test_content_hash_is_independent_of_the_duckdb_file`. ⚠️ 내용 해시는 duckdb 버전 감응이므로
재계산은 **1.5.5**로 한다.

### C3 ❌ `stream_hash`의 `ORDER BY … LIMIT/OFFSET` 청크 루프 (2026-09-03, 착지 전 폐기)

청크마다 16 GB `sequence` 컬럼을 포함해 테이블 전체를 다시 정렬했다. 첫 병합이 이 단계에서 끝나지
못한 원인. **현재**: 한 번의 스트리밍 패스로 행별 md5를 만들고 32바이트 다이제스트만 정렬한다.

### C4 ❌ `importlib.util`만으로 파일 경로 모듈 로드

`sys.modules`에 먼저 등록하지 않으면 그 모듈의 dataclass가 필드 타입을 해석하지 못하고 죽는다
(`AttributeError: 'NoneType' object has no attribute '__dict__'`).
**현재**: 저장소 관례 — `types.ModuleType(name)` + `sys.modules[name] = mod` +
`exec(compile(...))` (`tests/conftest.py`, `build_b5.py:21`).

### C5 ⚠️ `isoformDataHyena.__getitem__`의 무작위 대체 폴백 — 아직 살아 있다

`datasets.py:306-311`(fetch 예외) · `:329-335`(빈 서열) · `:339-344`(0토큰). 실패한 행을 **무작위 다른
행**으로 바꾸고 stderr 한 줄만 남긴다. A35가 고친 배치 루프의 한 단계 아래에 같은 패턴이 그대로 있다.
동결 DB에서는 현재 **도달 불가**(빈 서열 0 · NULL 라벨 0 · `train_weight ≤ 0` 0 · 서열 길이
30,720–129,024) — 그래서 발동하기 전에 고칠 수 있다. → 이슈 **#62**.

### C6 ⚠️ `validate_b5_database`의 orthogroup 교차 split 검사 — 타일 DB에서 구조적으로 무효

타일 행은 `orthogroup_id`가 전부 NULL이라 `validate_split`의 그룹 키가 `species:window_id`로 떨어지고,
모든 그룹의 크기가 1이 되어 위반이 **발생할 수 없다**. "위반 0"을 누출 없음의 근거로 쓰면 안 된다.
**현재**: `scripts/report_b5_database.py`의 `checks` 블록이 `window_genes` 경유로 직접 단정한다 —
자기 split보다 느슨한 타일에서 라벨된 유전자 0건, strict held-out이 test 아닌 타일에서 라벨된 것 0건,
고아 `window_genes` 0건(전부 통과). 커밋 `c6bc1b1`. → 이슈 **#57**.

### C7 ⚠️ `build_manifest.rejected_reasons` — 요약으로 쓸 수 없다

버킷 키가 `reason.split(":")[0].split(" >")[0]`이라 `masked fraction 0.412 > 0.6`에서 **소수점 값이
키에 남는다**. *V. vinifera*는 6,613건이 266개의 사실상 싱글턴 키로 흩어진다.
**현재**: 거부 계수는 `rejected_records`에서 직접 분류한다(`report_b5_database.py`). → 이슈 **#57**.

### C8 ⚠️ 라벨 방출기와 검증기의 tie-break 불일치 — 타일 97개 폐기

`window_to_gsf_v3`는 `(start0, end0)` 2요소 키로 정렬하고 `check_caps_v3`는
`(gs, ge, canonicalize(block))` 3요소 키로 검사한다. 같은 span을 가진 두 유전자에서 어긋나면
`CapError`가 나고 **타일 전체가 버려진다**(유전자 쌍이 아니라). 거부 메시지가 `key[:2]`만 찍어
`((14739, 17363) after (14739, 17363))` — "자기 자신과 순서가 어긋났다"로 읽혀 오래 눈에 띄지 않았다.
실측: Vvinifera 78 · Ppatens 10 · Athaliana 8 · Ptrichocarpa 1 = **97 타일**. → 이슈 **#55**.

### C9 ❌ `pkill -f <문자열>`

같은 문자열을 명령줄에 가진 **자기 자신의 셸**을 죽인다. 이 세션에서만 세 번 당했다(exit 143/144).
`evidence/est/` 작업에서도 같은 사고가 있었다(`[e]st_fetch` 패턴으로 회피).
**현재**: PID로 죽인다.

### C10 ⚠️ `evidence/rerun_training_after_finish.sh` — 영원히 발동하지 않는다

`until grep -q "TRAINING LONGREAD ALL FINISHED" longread.log; do sleep 600; done` 다음에
`./training_run_all.sh`를 실행한다 — **자기가 실행할 스크립트의 완료 문자열을 기다린다.** 순환이다.
의도는 검증 드라이버의 `LONGREAD ALL FINISHED`를 기다리는 것이었다. → 이슈 **#60**.

### C11 ❌ ENA filereport를 `IFS=$'\t' read`로 파싱

탭은 IFS 공백이라 `read`가 연속 탭을 합쳐, `submitted_ftp` 같은 빈 필드가 있으면 이후 모든 컬럼이
한 칸씩 밀린다. 결과: `instrument_model`이 빈 문자열로 읽혀 Sequel II 런이 `model=`로 건너뛰어졌다.
**현재**: `tr '\t' '\001'` 후 `IFS=$'\001'`(`evidence/longread_fetch.sh:13-15`, 백업
`longread_fetch.sh.bak-20260902`가 옛 버전).

### C12 ⚠️ `named_buffers()`의 기본 중복 제거

`remove_duplicate=True`가 기본이라 12개 층의 Hyena 필터 버퍼 중 하나만 보인다 —
"11개 층이 사전학습 기본값에 머물렀다"처럼 **결함처럼 읽힌다**. 실제로는 12개 층 전부
`pos_emb.z (1, 129024, 5)`·`seq_len` 129024로 정상이다. `remove_duplicate=False`로 확인할 것.

### C13 ⚠️ 플래그 파일 ID를 빌더 키에 직접 조인하는 것

플래그 파일은 GFF 식별자를 쓰고 빌더는 길거나 점이 둘 이상인 id에 자체 키를 생성한다
(`AT1G01010.TAIR10` → `Ath000001`). `gene_key_map.gene_id`에 바로 조인하면 **하나도 안 맞고**,
그것이 "마스킹이 아예 적용 안 됐다"처럼 보인다. `gene_id` / `gene_id_original` / `name_original`로
풀어야 한다(`build_b5.flags_for_gene`, :191-202). 이 저장소는 같은 결함을 한 번 겪었다(#53 항목 7,
커밋 `22308ba`).

---

## 5. 이 세션에서 틀렸다가 정정된 것 (2026-09-03)

기록 가치가 있어 남긴다 — 무엇이 언제 왜 틀렸는지가 세 번째 오진을 막는다.

| # | 틀린 것 | 어떻게 드러났나 | 정정 |
|---|---|---|---|
| S1 | 병합 계획의 `nextval` rn 부여 | Codex가 실측으로 반증(계획 심사 단계) | `row_number() OVER`, 첫 병합 폐기·재실행 |
| S2 | 첫 회귀 테스트를 3만 행으로 작성 | 결함 있는 쿼리도 통과했다 | 30만 행(정렬 블록 경계 너머)으로 확대, 되돌려 실패 확인(190,464/300,000) |
| S3 | 브리핑 표의 합계 `198,141 / 48,173` | Kimi가 지적 | 정본 `198,171 / 48,143`(검증기·동결 매니페스트·독립 재합산 3자 일치). 다른 산출물에는 전파되지 않았다 |
| S4 | "provenance 공백을 메웠다"(커밋 `c6bc1b1`) | wiki 에이전트가 지적 | `c6bc1b1`은 **도구**만 고쳤다. 산출물은 `bd6e4d9`에서 메웠고, 그 사이 log 파일 두 곳이 거짓을 진술하고 있었다 |
| S5 | Swiss-Prot 마스킹 미적용이라는 첫 프로브 결과(0건) | 대조군(GeenuFF)도 같은 0을 낸 것이 이상했다 | ID 매핑 오류(C13). 제대로 풀면 Swiss-Prot-only hard 535개 중 **0개 라벨**(적용됨), GeenuFF-only 8,737개 중 918개 라벨(A22 부분 전사체 케이스) |
| S6 | resume.md의 자동 스냅샷을 중복으로 보고 삭제 | 아카이브 절에 같은 형식이 있었다 | 그게 이 파일의 관례. 이후 것은 유지 |

**교훈(재사용 가능)**: 대조군과 시험군이 **둘 다** 0을 내면 그것은 증거가 아니라 프로브 고장이다.
→ wiki `guide/misc.md`의 "A probe that cannot return a positive is not evidence"와 같은 계열.

---

## 6. 기각의 기각

앞선 마커를 지우지 않고 잇는다.

- **D2 PacBio 세대 제한**: v1.3에서 RS II·Sequel 전면 제외 → **v1.5에서 테스트 종 검증에 한해 복원**
  (Wang 2018·Wang 2020). Wang 2016만 독립성 위반으로 계속 제외.
- **M6 held-out 블록 강제**: A29에서 도입 → **A31에서 폐지**(train 타일 23개 문제). 대신 A29의
  유전자 단위 마스킹으로 처리.
- **M3 masked-fraction**: Codex 0.25 제안 → 실측 기각 → Kimi 대안도 실측 기각 → 0.60 채택.
- **S4 provenance**: "공백 있음"(Kimi) → "메웠다"(잘못된 주장) → "도구만 메웠다"(wiki 에이전트) →
  "산출물도 메웠다"(`bd6e4d9`, 재대조 완료).

---

---

## 7. 원고·제출 측 기각 — ⚠️ 이력에서 인계, 이번 세션 재검증 없음

> ⚠️ **CAUTION (2026-09-03):** 아래는 `resume.md`의 아카이브 구간과 `peer_review_report_20260817.md`에
> 기록된 판정을 옮긴 것이다. **2026-09-03 세션에서 재측정하지 않았다.** 인용 전에 원 기록을 확인할 것.
> §1–§6은 이번 세션에서 코드·데이터로 직접 확인한 것이고, 이 절만 성격이 다르다.

| ID | 기각·정정된 것 | 원 기록 |
|---|---|---|
| P1 | 커버레터의 **"72.5 % 장독 지지"** — 낡은 주장. 구 제목과 함께 트랙 A 블로커로 분류 | `resume.md:265` |
| P2 | published 예측 파일의 **2배 중복**(54,826 레코드 / 27,413 좌위)을 beam export 탓으로 적은 것 — 실제로는 **DuckDB append 아티팩트**다. `manuscript_v2.md` Methods와 `27_rescore_prompted_topbeam.py` docstring이 아직 틀린 이유를 말한다. top-beam 필터 수정과 그 수치는 영향 없고 **서술된 원인만 틀렸다** | 위키 `projects/transgenic/README.md` |
| P3 | self-prompted export 중복 배수 — Fig 5의 93,690은 고유 31,230 × **3**, *P. patens*·*S. lycopersicum*은 2배, **Z. mays는 2회가 아니라 3회**. Methods 중복 문단과 Fig 5 범례 정정 대상 | `resume.md:576`, `:651` |
| P4 | Results의 **"487 predicted transcripts"** — 실제로는 `summary_report.json`의 `novel_genes` 필드로 **좌위 단위**다(`02_gffcompare_analysis.py:417`). 회계로 입증: 8,635 + 18,239 + 52 + 487 = 27,413 locus | `resume.md:779` |
| P5 | Fig 5·S1의 `tiberius` → **`tiberius_softmasked`**로 교체(저자 결정). *Z. mays*는 실패(gffcompare 0.0/0.0, BUSCO 0.2 %), *V. vinifera*는 soft-masked run 없음 → 빈 셀 + × 마커. 182셀 배열 비교로 Tiberius 계열 26셀만 변경 확인 | `resume.md:301` |
| P6 | **"18.1 %가 암기 부풀림"이라는 반론이 데이터로 기각됨** — held-out 분할에서 21.6 %(25/116) vs 학습 노출 17.7 %(175/987), Fisher exact p = 0.31. 방향이 오히려 held-out 쪽이 높다. 별도 held-out prompted 실행도 18.4 %(p = 0.90)로 헤드라인과 사실상 동일 | `resume.md:791` |
| P7 | 제출 메타데이터 미비 — Funding·ORCID·이메일 플레이스홀더, `073026` manifest 낡음, Lead Contact 미지정. GA/KRT/후미 순서는 cell.com 403으로 **UNVERIFIED** | `resume.md:265` |
| P8 | 원고 Methods의 인코더 서술 — 인코더는 **부분** 사전학습이다(339 텐서 중 227 로드, layers 8–11의 112개는 무작위, 147개는 타일링 복제). **"partially pretrained"로 고쳐야 한다.** 초기화 자체는 바꾸지 않는다(공개 레시피 재현) | 프로토콜 A34, 커밋 `b07e85f`·`f02380e` |
