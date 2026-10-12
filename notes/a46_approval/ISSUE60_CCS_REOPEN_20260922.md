# issue #60 재개 — 격리된 51 PacBio 런은 subreads가 아니라 **CCS**다 (2026-09-22)

## 결정적 증거

격리 디렉터리
`evidence/RETIRED_DO_NOT_USE/training_pacbio_subreads_20260903/` 의 **`.fa.gz` 51개 전부**에 대해
첫 리드 헤더를 읽었다.

```
>SRR26636582.1 m64306_220302_061936/4/ccs/1
>SRR26636574.1 m64268e_220629_025953/63/ccs/1
>SRR26636578.1 m64212_220119_203550/2/ccs/1
>SRR34258160.1 m54283Ue_210716_104359/28/ccs/1

파일 51개 중 헤더가 /ccs/ 인 것: 51, 아닌 것: 0
```

PacBio 리드 이름 규약: `{movie}/{zmw}/ccs` = **CCS 리드**. subread라면 `{movie}/{zmw}/{start}_{end}`.
**51/51이 CCS다.**

종별: **S. bicolor 36런 · V. vinifera 9런 · P. trichocarpa 6런.**

## 왜 틀렸나 — 같은 함정에 두 번

issue #60(저자결정 2026-09-03)은 51런 43 GB를 "subreads-only"로 판정해 경로 격리했다.
그 판정의 근거는 **다운로드 URL의 `_subreads` 파일명**이었다(`quarantine.md` D3:
*"`evidence/training/`의 PacBio 51런 전부가 `_subreads.fastq.gz`이고 CCS/FLNC는 0건"*).

**그것은 ENA가 NCBI 경유(brokered) PacBio 런에 붙이는 일반 이름이며 산물을 말해주지 않는다** —
이 세션이 **A47.1에서 이미 A-CCS23·M-CCS26에 대해 정정한 바로 그 아티팩트**다.
같은 함정에 두 번 걸렸고, 이번에는 **학습 데이터** 쪽이다.

당시 기록에 이미 단서가 있었다: `quarantine.md` D3는 subreads 가드 `MAX_READS=5000000`이
"실제 세트가 24.8만–157.8만 리드라 발동하지 않았다"고 적었다. **24.8만–157.8만 리드/셀은 CCS 규모다.**
Sequel II 8M ZMW 셀의 subreads라면 그보다 10–40배여야 한다. 가드가 안 터진 것이 결함이 아니라
**데이터가 subreads가 아니라는 신호였다.**

## 파급

**① D1(PacBio 채널 처분)의 전제가 무너진다.** "학습 가능한 PacBio 0건"은 바로 이 격리의 결과였다.
복원되면 PacBio 채널이 실제 학습 입력을 받고, 죽은 채널 문제도 Z. mays 추론 문제도 사라진다.
**D1은 이 건이 정리될 때까지 결정하지 않는다.**

**② A46 rule 16에 어제 넣은 문장이 틀렸다** — 정정했다. S. bicolor는 CCS 36런을 갖고 있다.
**B. distachyon은 여전히 0이 맞다**(아래 scout).

**③ `DATASET_ROLES.longread_v1.tsv`의 `excluded` 51행 basis가 사실과 다르다** —
"subreads-only, protocol …"이라고 적혀 있다. 재개 결론이 나오면 갱신 대상.

## 같이 돌린 scout — 학습 8종 Sequel II+ Iso-Seq (ENA, 2026-09-22)

`tax_tree(taxid) AND instrument_platform="PACBIO_SMRT" AND library_source="TRANSCRIPTOMIC"`

| 종 | PacBio 전체 | Sequel II+ | 비고 |
|---|---:|---:|---|
| S. bicolor | 64 | 37 | PRJNA1275229/1275324/1275171/1275264 등 — **대부분 이미 격리된 것** |
| V. vinifera | 15 | 9 | PRJNA1185815 — **격리된 것** |
| P. trichocarpa | 12 | 11 | PRJNA709498/709499 — **격리된 것** |
| O. sativa | 139 | 6 | **PRJNA1291274 미사용 후보**(아래) |
| G. max | 32 | 6 | PRJNA1107527 183M·141M 리드 = subreads 규모 / PRJNA1114896 17.4M = subreads 규모 |
| **B. distachyon** | **0** | **0** | — |
| **S. italica** | **0** | **0** | — |
| **P. patens** | **0** | **0** | — |

**B. distachyon · S. italica · P. patens는 PacBio 전사체가 어느 세대도 0건**이다. 그 셋의 공백은 영구적이다.

**O. sativa PRJNA1291274 — 미사용 후보**: Sequel II 3런, `RNA-Seq`/`cDNA`,
SRR34539106 203,041리드 3,176nt / SRR34539098 275,533리드 2,388nt / SRR34539107 432,513리드 2,611nt,
총 911,087리드. experiment_title이 *"IsoSeq of parental RNA: Mixed root"*, *"IsoSeq of Hybrid RNA:
SY63 root"*, *"IsoSeq of MH63: flagLeaf"*. **CCS 규모**(A-CCS23 346k/2,849nt와 같은 서명).
같은 BioProject의 ONT 6런(`training/ont/Osativa/indica_flagleaf_PRJNA1291274`)은 이미 쓰고 있다.

## 다음 단계 (순서 고정)

1. ✅ **51런의 런별 기기 확인** — 완료, 아래 "1단계 결과"(기기 배제 0, WGS 2런 배제). A14는 학습에 **Sequel II 이상**을 요구한다. movie ID가
   `m64306`·`m64268e`·`m64212`는 Sequel II 계열이나 ❌ ~~**`m54283Ue`는 Sequel/Sequel I 계열로 보인다**~~ (ENA: Sequel IIe).
   일부는 A14로 여전히 배제될 수 있다. 기기별로 갈라야 한다.
2. **저자 판단: A47.1의 "CCS는 구조팔 불가"가 학습 트랙에도 적용되는가.** ✏️ 센서스(아래) 후 범위 축소: **V. vinifera 9는 FLNC급이라 대상 아님, 질문은 원시 CCS 40런(Sb 34·Pt 6)에만 남음.**
   A47.1은 **검증 채점**에 대한 결정이었다. 학습 입력 채널은 다른 문제이고, A14는 "Sequel II+
   Iso-Seq"를 요구할 뿐이며 A18.5는 이미 CCS/FLNC에 **≥ 98 % identity** 임계를 따로 두고 있다.
3. 복원 가능한 런 확정 → `DATASET_ROLES` role/basis 갱신 → `quarantine.md` D3 정정
4. **그 다음에 D1** (아마 자동 해소)

## 범위

읽기전용 조사(첫 리드 헤더만, 전체 스캔 없음) + 이 기록 + A46 rule 16 정정.
파일 이동·복원·다운로드·잡 제출·커밋 없음. **격리 디렉터리는 그대로 둔다.**

## 1단계 결과 — 런별 기기·라이브러리 실측 (2026-09-22, ENA portal `read_run`)

스냅샷: [`ISSUE60_ENA51_20260922.tsv`](ISSUE60_ENA51_20260922.tsv) (51행, sha256 `466dd176…`).
매니페스트의 instrument·read_count와 **51/51 일치**, 불일치 0.

**기기 — 51/51 A14 통과.** S. bicolor Sequel IIe 28 + Sequel II 8 · V. vinifera Sequel II 9 ·
P. trichocarpa Sequel II 6. ⚠️ **위 "`m54283Ue`는 Sequel/I 계열" 의심은 틀렸다** — 그 movie의
런 SRR34258160(PRJNA1275264)을 ENA는 **Sequel IIe**로 기록한다(`m54xxx` 일련번호에 붙은
`U`는 Sequel→Sequel II 업그레이드 기기로 읽힌다; 판정 근거는 ENA 기록이지 이 해석이 아니다).
**기기로 배제되는 런은 0이다.**

**🔴 그러나 라이브러리로 2런은 영구 배제 — 전사체가 아니다.**

| run | 데이터셋 | strategy/source | 리드 | 평균 nt | experiment_title |
|---|---|---|---:|---:|---|
| SRR26636578 | Sbicolor/baijiu_PRJNA1034755 | WGS / GENOMIC | 2,076,998 | 13,814 | HiFi seq of sorghum |
| SRR26636582 | Sbicolor/baijiu_PRJNA1034755 | WGS / GENOMIC | 1,986,201 | 15,812 | HiFi seq of sorghum |

**게놈 HiFi다.** `.atl/quarantine.md:137`(이슈 #63)이 이미 "WGS 2런은 baijiu 안에 있다"고 적어 뒀다.
**이 노트 맨 위 "결정적 증거"의 첫 예시 헤더 `SRR26636582 … m64306…/ccs`와 둘째 예시
`SRR26636578 … m64212…/ccs`가 바로 그 게놈 런이었다** — `/ccs`는 HiFi(CCS) 리드라는 뜻일 뿐 전사체라는
뜻이 아니다. 결론(나머지가 subreads가 아니라 CCS)은 유지되지만 **"51/51 CCS"를 "학습 가능 51"로
읽으면 안 된다.** `revision/protocols/TRAINING_EVIDENCE_v1.md:39`가 PRJNA1034755를
"PacBio SqIIe HiFi"로 적은 것도 이 두 런 때문이다.

**전사체 후보 = 49런**: S. bicolor **34**(Wray 7 BioProject 33 + baijiu SRR26636574 1 — `OTHER/RANDOM`,
title "Iso-seq of sorghum") · V. vinifera **9**(`FL-cDNA`/`cDNA`, title `IsoSeq_<tissue>`) ·
P. trichocarpa **6**(93-968 × deltoides F1 hybrid — A18.4 hybrid multiplier 0.5 대상).
Wray·Populus는 `RNA-Seq`/`RT-PCR`. 평균 1,333–3,825 nt, 5.4만–323만 리드/런 — 전부 CCS 규모.

**남은 식별 한계 (2단계 판단에 필요):** 헤더 `{movie}/{zmw}/ccs`는 **lima/refine을 거친 FLNC도
같은 이름을 유지**하므로 헤더만으로 "FLNC 미처리 CCS"와 "FLNC"를 가를 수 없다. 모든 런이
`submitted_ftp` 공란(NCBI 경유)이라 제출 파일명 단서도 없다. 가르려면 리드 말단의 poly(A)·
Iso-Seq 프라이머 잔존 비율을 봐야 한다(원격 격리 디렉터리, 샘플 읽기만). **A47.1("CCS 구조팔 불가")을
학습 트랙에 적용할지 여부가 이 구분에 달려 있다면 먼저 그 측정이 필요하다.**

범위: ENA 메타데이터 조회만. 파일 이동·복원·다운로드·잡 제출·커밋 없음.

## 2단계 전 측정 (c) — 리드 말단 프라이머·poly(A) 잔존 센서스 (2026-09-22, 저자 "C 먼저해")

**방법.** pronghorn `/data/gpfs/assoc/pgl/data/Transgenic/evidence/RETIRED_DO_NOT_USE/training_pacbio_subreads_20260903/`
의 51개 `.fa.gz`마다 **앞 5,000리드**를 읽어, 양 말단 100 nt에 Iso-Seq cDNA 프라이머 k-mer(SMARTer 코어 16-mer 2개,
Iso-Seq Express 5′ 14/16-mer, 각각 역상보 포함, 정확일치)가 있는지와 3′ 50 nt의 A≥15 / 5′ 50 nt의 T≥15를 셌다.
fetch는 `seqkit fq2fa`만 하므로(`longread_fetch_v2.sh:146`) 제출 서열 그대로다. 읽기전용, 잡 제출 없음(login 노드 스트리밍).
스크립트 [`isoseq_end_census_20260922.py`](isoseq_end_census_20260922.py) (sha256 `2310ea83…`),
결과 [`ISSUE60_ENDCENSUS_51_20260922.tsv`](ISSUE60_ENDCENSUS_51_20260922.tsv) (`018b5715…`),
대조군 [`ISSUE60_ENDCENSUS_CONTROLS_20260922.tsv`](ISSUE60_ENDCENSUS_CONTROLS_20260922.tsv) (`abd80d5c…`).

**해석 규칙.** 원시 CCS = 양끝 프라이머 + poly(A)가 **양방향 반반**(미정렬). lima 후(FL) = 프라이머 제거·방향 정렬,
poly(A)는 3′에만. refine 후(FLNC) = 프라이머·poly(A) 모두 없음.

**대조군 (방법 검증).**

| 세트 | 프라이머 | poly(A)3′ / poly(T)5′ | 판독 |
|---|---:|---:|---|
| M-FLNC `F1maize.flnc.fa.gz` (알려진 FLNC) | 0.000 | 0.000 / 0.000 | FLNC 서명 — 음성 기준 |
| M-HQ18 ERR2093901·902 (polished HQ) | ≤0.002 | ≤0.008 / ≤0.006 | 가공 산물 |
| WGS 2런 (게놈 HiFi) | 0.000 | 0.000 / 0.000 | 전사체 아님 |
| Wray·Populus (아래) | **0.99+** | ~0.49 / ~0.49 | 프라이머 k-mer가 실제로 검출됨 — 양성 기준 |

**격리 51런 — 세 부류로 깨끗하게 갈린다.**

| 부류 | 런 | 프라이머 any / 양끝 | poly(A)3′ / poly(T)5′ | 판독 |
|---|---|---:|---:|---|
| **A. FLNC급** | **V. vinifera 9** (PRJNA1185815) | **0.000 / 0.000** | 0.000–0.001 / 0.000–0.001 | **lima+refine 완료** — M-FLNC 대조군과 같은 서명 |
| **B. 원시 CCS** | **S. bicolor 34**(Wray 33 + baijiu SRR26636574) · **P. trichocarpa 6** = **40** | **0.986–0.999 / 0.878–0.955** | 0.37–0.52 / 0.37–0.51 | **lima·refine 미처리** — 프라이머 양끝 잔존, 방향 무작위 |
| C. 비전사체 | S. bicolor baijiu SRR26636578·582 | 0 / 0 | 0 / 0 | 게놈 HiFi (1단계에서 이미 배제) |

B 부류 안의 프라이머 종류는 BioProject별로 갈린다(SMARTer만 / SMARTer+Iso-Seq Express 병존) — 라이브러리 키트 차이이며
판독은 같다. PRJNA1275229·1275324의 poly(A) 합계 0.73–0.87은 나머지(0.87–0.99)보다 낮으나 방향은 여전히 반반이다.

**결론.**
- **V. vinifera 9런은 공개 산물 자체가 FLNC급이다.** A47 규칙 2("공개저장소에서 FLNC를 직접 다운로드 가능할 때만 FLNC
  자격")를 문자대로 충족하는 것으로 보인다 — 우리가 만든 게 아니라 제출자가 올린 서열이 이미 프라이머·poly(A)가 제거돼 있다.
  이 9런은 A47.1 "CCS 구조팔 불가"의 대상이 **아니다**(CCS가 아니라 FLNC급).
- **S. bicolor 34 + P. trichocarpa 6 = 40런은 진짜 원시 CCS다.** FLNC로 만들려면 lima/refine이 필요한데 **A47 규칙 1이 금지**.
  따라서 이 40런에 대해서만 "A47.1이 학습 트랙에도 적용되는가"라는 원래 질문이 남는다.

## ⚠️ 부수 발견 — A47.1의 전제가 대조군에서 흔들린다 (신고, 조치 없음)

대조군으로 넣은 **이미 결정된 세트 둘**이 A47.1의 "CCS without FLNC processing" 전제와 맞지 않는다.

- **A-CCS23** (PRJNA911826, SRR22719002·003): 프라이머 **0.000**, poly(A) **3′에만 0.988–0.994**, 5′ poly(T) ≤0.001
  → **lima 처리된 FL 리드**(프라이머 제거·방향 정렬, refine의 poly(A) 절단만 안 됨). 원시 CCS가 아니다.
- ❌ ~~**M-CCS26** (PRJNA1470126, SRR38818769): 프라이머 0.004, poly(A) 0.001 → **FLNC 서명**.~~ ✏️ **틀림 — 내부 Illumina 어댑터 키메라(57–98 %), 아래 A47.2 절.**
- M-KNX (SRR34503567): 프라이머 0, poly(A) 0 → 가공 산물(Kinnex 분절 후).

**A47.1 저자결정(09-21 "CCS는 구조팔 불가")은 이 둘을 "FLNC 미처리 CCS"로 전제했다.** 파일 1–2개 × 앞 5,000리드만
본 것이라 판정이 아니라 **재검토 신호**다. 이번 작업 범위 밖이므로 A47.1·매니페스트·MVP 구조계약은 **건드리지 않았다.**
확인하려면 두 세트 전 런에 같은 센서스를 돌리면 된다(수 분, 읽기전용).

## A47.1 전제 재확인 — 전 런 센서스 (2026-09-22, 저자 "2번 확인해")

같은 스크립트로 **A-CCS23 6런 · M-CCS26 3런 · M-KNX 4런 전부**, 런당 **앞 20,000리드**.
결과 [`A47_1_ENDCENSUS_20260922.tsv`](A47_1_ENDCENSUS_20260922.tsv) (13런).

| 세트 | 런 | 프라이머 any | poly(A)3′ | poly(T)5′ | 판독 |
|---|---|---:|---:|---:|---|
| **A-CCS23** PRJNA911826 | 6/6 | **0.000** (6런 전부) | **0.988–0.994** | 0.001 | **lima 처리 FL** — 프라이머 제거·방향 정렬, refine의 poly(A) 절단만 없음 |
| **M-CCS26** PRJNA1470126 | 3/3 | 0.000–0.002 | 0.000–0.001 | 0.000–0.001 | ❌ ~~FLNC 서명~~ ✏️ **키메라**(내부 Illumina 어댑터, 아래 A47.2 절) |
| M-KNX PRJNA1290227 | 4/4 | 0.000 | 0.000 | 0.000 | 가공 산물 — 헤더 `…/ccs/{start}_{end}`는 Kinnex 분절(S-read) 좌표, 프라이머·poly(A) 없음 |

**런 간 편차가 거의 없다** — 파일 1–2개로 본 신호가 13런 전부에서 그대로 재현됐다. 표본 제한(앞 2만 리드)은 남지만
세트 단위 판정을 뒤집을 크기는 아니다.

**A47.1이 틀린 곳.** A47.1은 A-CCS23·M-CCS26을 *"CCS without FLNC processing"*으로 보고 저자결정(09-21)
"CCS는 PacBio 구조팔을 맡지 못한다"를 적용했다. 근거는 리드 수·평균 길이 센서스("CCS-level")였고, 그 센서스는
**가공 단계를 볼 수 없다** — 리드 하나당 분자 하나라는 것만 말한다. 말단을 보면:
- ❌ ~~**M-CCS26은 FLNC급 공개 산물이다.**~~ ✏️ 틀림 — 말단만 봐서 내부 Illumina 어댑터 키메라를 놓쳤다. 아래 A47.2 절.
- **A-CCS23은 FLNC가 아니라 FL이다.** lima는 됐고 refine(poly(A) 절단 + 콘카테머 제거)은 안 됐다. 접합·인트론 체인에는
  poly(A) 잔존이 영향이 거의 없지만 TES 경계와 콘카테머 비율은 FLNC와 다르다. 이건 **새 범주**라 A47 규칙 어디에도 없다.

**파급(판정 아님, 저자결정 대상).**
1. A. thaliana MVP 구조계약 "**ONT 단독**" → A-CCS23을 구조팔로 인정하면 PacBio 팔이 되살아나고, §5.3 안전장치가 걸었던
   **STOP→INCONCLUSIVE 하향**의 전제(`≥2 A-HiFi` 미충족)가 달라진다. A-CCS23 WT는 6런이라 개수 조건은 채운다.
2. maize: M-CCS26이 tier-1 구조증거로 복귀할 수 있다(A47 §3.1이 maize 구조증거를 M-HQ18+M-FLNC로 좁혔다).
3. `coverage_policy_a47.json`(`structural_sources:["ONT"]`)과 바이트핀 `eff505f`는 그 결정에 묶여 있다 — 결정이 바뀌면 재핀.
4. issue #60 원시 CCS 40런의 질문(학습 입력 허용 여부)은 이것과 **독립**이다 — 그 40런은 진짜 원시 CCS다.

**미조치:** A47.1·매니페스트 role·MVP §2/§5.3·계약 JSON·핀 전부 무변경. 기록만.

## A47.2 적용 (2026-09-22, 저자 "1. 복귀 2번 인정하되 TES 경계 판정에서만 제외")

**적용 전 추가 측정에서 결정 1의 전제가 무너졌다.** 구조팔 자격의 전제(콘카테머 없음)를 보려고 내부 신호를 쟀더니
M-CCS26 리드의 **57–98 %가 내부에 Illumina P5/P7 어댑터**를 가진 키메라였다(`cDNA–poly(A)–어댑터–poly(T)–cDNA`).
앞선 "FLNC 서명" 판정은 말단·Iso-Seq 프라이머만 본 **내 오판**. → **결정 1(M-CCS26 복귀)은 적용하지 않고 보류**, 재보고.
A-CCS23은 내부 신호도 깨끗(Illumina 0, 내부 A/T≥20 0.4–0.9 %, M-FLNC 0.3 %) → **결정 2 적용.** 측정: [A47_2_INTERNAL_CENSUS](A47_2_INTERNAL_CENSUS_20260922.md).

**문서 변경(커밋 없음):**
- `PROTOCOL_B1_frozen_v1.md`: **A47.2 (v1.38)** 신설(측정표·저자결정 4항·파급·불변사항) + change-log 행; A47.1 저자결정 뒤 ✏️ 포인터,
  A47.1 "ONT-only" 결과 항목 ✏️; A46(DRAFT) **rule 10 채점세트 = A-ONT1 + A-CCS23**(TES 제외), 독립 outcome 정렬·분모에 A-CCS23 추가,
  draft revision 2026-09-22 기록, rule 17 split·A47 적용노트 갱신. **§5.3 인용블록 불변 재검증: 정규화 후 동일(1,958자, `*_\`` 제거+공백정규화)**.
- `B6_MVP_PILOT_v1.md`: §2 A47.2 배너(A-CCS23 = PacBio 팔, `splice:hq`), A47 문단 취소선, 접합정의 PacBio disjunct = `≥ 2 A-CCS23`
  (소스 간 합산 없음), A47.2 적용노트(안전장치는 A-ONT1 또는 A-CCS23 미정렬 시에만 발동), day-5 행·명령 스케치.
- `DATASET_ROLES.longread_v1.tsv`: A-CCS23 6행 basis/note(A47.2), M-CCS26 3행 note(키메라). role 집계 불변 129/55/31.
- 코드(worktree `/tmp/mvp-seed123-integration-20260917`): 계약 v2 `coverage_policy_a47_2.json` + 검증기·결합·정렬·테스트 — writer 에이전트 진행.

## 종결 — A47.3 (v1.39), 저자결정 2026-09-22

저자: "M-CCS26은 복귀 대신 현재대로 구조 권한 없음 / issue #60의 원시 CCS 40런을 학습에 쓴다".

- **M-CCS26**: 구조 권한 없음 유지(이유 = 측정된 키메라). 매니페스트 role 불변, note에 키메라 기록.
- **issue #60 번복**: 적용 전 40런 내부 센서스 — Illumina 0/40(키메라 없음), 콘카테머 신호 대부분 <1 %, 일부 5–10 %
  ([A47_3 센서스](A47_3_TRAINING_CCS_CENSUS_20260922.md)). **49런 복원**(원시 CCS 40 + FLNC급 V. vinifera 9 — 후자는 측정상 쟁점이 없어 함께 복원),
  WGS 2런 격리 유지. pronghorn `mv`, inode·크기 49/49 보존, `.md5` 전수 통과. 정본 매니페스트 동기화(백업 `.bak-20260922-issue60`),
  경로 존재 49/49, role 178/31/6. 유전형층: Populus `hybrid_pooled`, Wray·Zhuosexiang `non_reference`, baijiu `unknown`.
- **프로토콜**: B1 **A47.3** 신설(규칙 6항: 로컬 가공 없음, A18.5 그대로 ≥98 %, 원시 CCS는 `splice:hq` **`-uf` 없이**, A18.4 가중 1.0×0.5,
  콘카테머 신규 필터 없음·런별 비율 기록, A47.1/47.2는 학습에 미적용) + change-log; A46 rule 16 ✏️ 해소; TRAINING_EVIDENCE·quarantine 배너.
- **D1**: 전제("학습 가능한 PacBio 0")가 사라졌다 — 이제 결정 가능.
- **남은 실행**: 복원 49런의 학습용 장독 정렬(`longread_align.sbatch`, 잡 제출) — 저자 지시 대기. O. sativa PRJNA1291274 PacBio 3런은
  `_subreads` 이름 가드에 막혀 미수집(같은 함정의 세 번째 사례일 가능성, 미결).

## 후속 실행 (2026-09-22, 저자 "1. 진행 2. 커밋 재핀 3. 채널 유지 4. _subreads 라면 EST 처럼 사용하는건 어떤가?")

1. **49런 학습 정렬**: 기존 `longread_align.sbatch`는 ONT 전용(`-x splice -k14` + §3.3 방향감사 필수)이라 PacBio용
   `revision/scripts/evidence/longread_align_pacbio.sbatch`를 파생 작성 — 게이트 1(A18.3 역할)·2(참조/GFF md5) 동일, 감사 게이트 없음,
   §4 PacBio 명령 `minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long` (+ 같은 옵션 `-x` PAF), `-uf` 없음(A47.3 규칙3),
   같은 출력 배치·DONE·PROVENANCE. 런목록 `evidence/longread_runs_pacbio_v1.tsv`(49행, md5 `f12af198…`), 원격 스크립트 md5 `cfd53f14…`.
   **파일럿 6170399**(`--array=0`, Pt SRR35145693) 제출 → 게이트 통과·정렬 진행 — 결과 확인 후 나머지 48 제출.
2. **커밋·재핀**(worktree `/tmp/mvp-seed123-integration-20260917`, 푸시 없음): `426c328` 계약 v2 구현, `0d13ece` 바이트핀 `eff505f`→`426c328`
   (핀 테스트 1 passed). author/committer Won <wyim@unr.edu>, 트레일러 없음.
3. **D1 = 채널 유지** → K = 31 A46 rule 6 기입(PROGRESS 참조).
4. **O. sativa PRJNA1291274 PacBio 3런 — subreads가 아니다.** ENA 파일 앞 1.5 MB만 Range로 받아 확인(읽기전용):
   SRR34539098·106 `…/ccs/1`, SRR34539107 `…/{start}_{end}_CCS/1`(RS II 시절 Iso-Seq classify 산물명), 전부 ZMW당 1리드 → **CCS/FL급**.
   **그러나 movie 이름이 RS II 형식**(`m180817_135114_42199_c1014…_s1_p0`, `m150822_…_s1_X0`; 2015·2018년) — ENA의 "Sequel II"는 오기로 보인다.
   → **A14(학습은 Sequel II 이상)로 제외 대상**. "_subreads면 EST처럼"이라는 질문은 이 3런에선 해당 없음.
   교차확인: 복원 49런의 movie 접두어는 전부 `m64…`(Sequel II/IIe) 또는 `m54283U/Ue`(Sequel→II 업그레이드 기기) — RS II 없음.

**파일럿 결과 6170399** (2026-09-22 14:03→14:33 PDT): COMPLETED 0:0, 30m00s, batch MaxRSS 26.7 GB(minimap2 peak 19.0 GB), 1,880,549 레코드,
**매핑 99.76 %**, supplementary 201,255(콘카테머 신호와 부합; A18.5 primary-only가 소비자 단계에서 처리), DONE key
`0f4cdcb7…:20c6f4bb…:splice_hq`, bam md5 `3a493b9b…`. `.extern OUT_OF_MEMORY 0:125`는 09-21 저자결정대로 판정 미사용.
→ **나머지 48런 6170404 제출**(`--array=1-48%4`, 스크립트 `cfd53f14…`, 런목록 `f12af198…`).

**저자결정 (2026-09-22, "아예 제외"):** subreads-only·RS II PacBio 세트(O. sativa PRJNA1291274 3런, A-HiFi ERR13994458)를
**엑손 순서/연결 판단에도 쓰지 않는다** — 연결 채널 신설·타 증거 접합으로의 snapping·사후 탐색분석 모두 없음. A47.3 본문에 기록.
O. sativa 3런은 RS II라 A14로 학습 제외 확정, 다운로드하지 않는다.

**중간 확인 2026-09-22 15:13 PDT (6170404):** 배열 6 COMPLETED · 4 RUNNING · 나머지 PENDING, FAILED/REFUSED 0.
PacBio DONE **7/49** (파일럿 포함): 매핑 97.4–99.8 %, BAM md5 = DONE 기록 7/7, preset `splice:hq` 7/7, 런당 6.5–29분.
P. trichocarpa 709499 3런 매핑 97.4–97.6 %(F1 잡종의 deltoides 대립형질로 예상되는 수준).
**확인 2026-09-22 15:55 PDT:** 배열 14 COMPLETED · 4 RUNNING · 나머지 PENDING, 실패 0. DONE **15/49**, BAM md5·preset 불일치 0. 매핑 S. bicolor 10런 98.95–99.82 %(평균 99.44), P. trichocarpa 5런 97.40–99.76 %.
**확인 16:13 PDT:** 배열 17 COMPLETED · 4 RUNNING, 실패 0. DONE **20/49**, 불일치 0. S. bicolor 15런 98.95–99.82 %(평균 99.53).

## ✅ 학습 정렬 완료 (6170399 + 6170404, 2026-09-22 18:40 PDT 확인)

- 배열 태스크 49/49 COMPLETED 0:0(batch step 포함 96 step 0:0). `.extern OUT_OF_MEMORY 0:125` 48건 — 09-21 저자결정대로 판정 미사용.
- **DONE 49/49**, BAM md5 = DONE 기록 49/49, preset `splice:hq` 49/49, BAM index 49/49, PAF md5 = DONE 기록 49/49,
  런별 PROVENANCE.txt 49/49, 중앙 `revision/results/evidence/PROVENANCE.tsv` 행 49/49.
- 매핑률(raw, A18.5 필터 전): **S. bicolor 34런 98.95–99.88 %(평균 99.57)** · **P. trichocarpa 6런 97.40–99.89 %(98.34)** ·
  **V. vinifera 9런 99.56–99.79 %, 단 SRR31342058(뿌리) 90.66 %**(547,020 중 495,929). 원인 미조사 — 뿌리 조직의 비식물 RNA 혼입 가능성,
  A18.5가 소비자 단계에서 거르므로 차단 사유 아님.
- 출력: `evidence/longread_align_v1/{Sbicolor,Ptrichocarpa,Vvinifera}/<dataset>/<run>/`.

## A-CCS23 MVP 패널 국소정렬 (2026-09-22, 저자 "A-CCS23를 이용하자")

B6_MVP_PILOT §2 day-5의 A-CCS23 다리. 드라이버 `revision/scripts/mvp/mvp_align_panel_longreads.py`(worktree 커밋 `0d13ece`):
1차 `minimap2 -x splice` 전체 → 패널 창(±10 kb) MAPQ≥5 리드 선별 → `seqkit grep` → 2차 §4 `splice:hq`(`-uf` 없음) → sort/index →
`mvp_junction_table.py` 접합표. 출력 네임스페이스 `evidence/mvp_scoring_20260916/<RUN>`(드라이버 고정, 기존 디렉터리 거부).
- 릴리스 `evidence/mvp_scoring_release_ccs23_20260922/`: `git archive 0d13ece src revision/scripts revision/protocols/b6_panels_v1.tsv`
  tar sha256 `1fa200d3…`, sbatch `3e85181e…`. 패널 sha256 `79367154…` (로컬·원격·worktree 일치). 참조 TAIR10 md5 `513e5ef3…`, GFF `0264c349…`.
- 로그인 노드 dry-run(SRR22719004, `--execute` 없음): 역할·리드 md5·참조/GFF md5·패널·창 계산 통과(`PREPARATORY_BYTE_CHECKS_ONLY`).
- **잡 6170576** 제출(`--array=0-5%3`, 16 CPU/48 GB/6 h). 드라이버 경계: `scientific_input_validation=false`, `reference_identity_status=UNVERIFIED`,
  `coverage_policy=PENDING` — 탐색/MVP 전용이며 외부 frozen-reference 게이트는 여전히 필요.
- A-ONT1 다리는 이번 범위 밖(미실행). §5.3 안전장치는 두 다리 모두 day 5까지 정렬돼야 풀린다.

**결과 6170576 (2026-09-22):** 5/6 COMPLETED(각 1–1.5분), **SRR22719003 FAILED 1:0**.
| run | 패널 후보 리드 | 수용 리드 | 창 안 접합 관측 | 고유 접합(표) | 제외(mapq/supp·chim/unmapped) |
|---|---:|---:|---:|---:|---|
| SRR22719002 | 8,171 | 8,147 | 49,587 | 1,751 | 2 / 9 / 13 |
| SRR22719004 | 7,017 | 6,928 | 29,252 | 1,715 | 2 / 74 / 13 |
| SRR22719005 | 9,413 | 9,270 | 39,004 | 1,807 | 5 / 121 / 17 |
| SRR22719006 | 9,710 | 9,683 | 45,783 | 1,866 | 2 / 12 / 13 |
| SRR22719007 | 12,631 | 12,573 | 58,779 | 1,966 | 4 / 31 / 23 |
| SRR22719003 | 12,841 | — | — | — | 접합표 단계에서 중단 |

**SRR22719003 실패 원인:** 리드 `SRR22719003.186798`(Chr2:15,972,926, MAPQ 32)의 CIGAR에 `…1304N681I1201N…` — 두 인트론(N) 사이에
참조 염기 0개(삽입만)인 "빈 엑손". frozen `59_evidence_support.py:parse_sam`(`:132` `require(pos > block_start, "empty exon")`)이
**리드 하나에서 런 전체를 fail-closed로 중단**. 이 리드는 504I·518I·681I 대형 삽입이 있는 병적 정렬(키메라/이질 서열 추정).
panel.bam·panel_reads는 남아 있고 `FAILED.json` 기록. 수정은 frozen 파서 변경이거나 사전 필터(새 규칙)라 **저자결정 대기** —
결정 전에는 SRR22719003을 **missing per source**로 두고 나머지 5런이 A-CCS23 다리를 구성.

## 저자 "같이 진행해" (2026-09-22) — (b) 빈 엑손 리드만 거부 + SRR22719003 재실행 + A-ONT1 패널정렬

- **코드(worktree, 푸시 없음):** `681c5a0` — `59_evidence_support.py`에 `EmptyExonError(EvidenceError)` 추가, 같은 위치·같은 메시지로 raise
  (기존 호출자는 여전히 fail-closed; `revision/scripts/tests/test_evidence_support.py` 변경 전후 43 passed 동일).
  `mvp_junction_table.collect_junctions`만 그 하위클래스를 잡아 **그 리드의 모든 레코드를 버리고 `rejected:empty_exon`으로 계수**(`raw_reads`에는 포함).
  테스트 +2(SRR22719003.186798 모양 리드 / frozen 파서 fail-closed 유지). `729b990` — 바이트핀 `426c328`→`681c5a0`, 핀 테스트 통과.
  scoring 43 passed(기존 `torch` 실패 1), binding 20, driver 10.
- **릴리스 v2** `evidence/mvp_scoring_release_v2_20260922/` (`git archive 729b990`, tar sha256 `305469e2…`, sbatch `1b1e8fef…`).
- SRR22719003 첫 시도는 삭제 없이 `evidence/mvp_scoring_20260916_failed/SRR22719003_20260922/`로 이동(드라이버가 기존 출력 디렉터리를 거부).
- **잡 6170586** 제출(`--array=0-3`, 32 CPU/96 GB/12 h): 0 = A-CCS23 SRR22719003 재실행, 1–3 = **A-ONT1 SRR28341943/44/45**
  (드라이버가 ONT는 1만 리드 표본으로 방향감사(67/68, A39) 후 `-uf` 결정 → `splice -k14` 2차).

**결과 6170586 (2026-09-22, 4/4 COMPLETED 0:0):**
| run | 세트 | 시간 | 패널 후보 리드 | 수용 리드 | 고유 접합 | 방향감사 | 제외(empty_exon/mapq/supp·chim/unmapped) |
|---|---|---:|---:|---:|---:|---|---|
| SRR22719003 | A-CCS23 | 1m25s | 12,841 | 12,785 | 1,973 | — (`splice:hq`) | **1** / 4 / 39 / 12 |
| SRR28341943 | A-ONT1 | 4m57s | 93,848 | 91,323 | 3,420 | sense 0.535 → FAIL, `-uf` 없음 | 0 / 89 / 2,103 / 333 |
| SRR28341944 | A-ONT1 | 7m29s | 141,331 | 131,585 | 5,946 | sense 0.519 → FAIL, `-uf` 없음 | 0 / 184 / 8,783 / 779 |
| SRR28341945 | A-ONT1 | 6m55s | 115,973 | 108,853 | 5,202 | sense 0.504 → FAIL, `-uf` 없음 | 0 / 162 / 6,275 / 683 |

- SRR22719003: 문제의 리드 1개만 `rejected:empty_exon`으로 빠지고 런은 완료 — (b) 수정이 의도대로 동작.
- A-ONT1 방향: 세 런 모두 비배향(sense 0.50–0.53) → A39 규칙대로 `-uf` 없음. 09-16 파일럿(6165555)의 SRR28341943 53.5 %와 일치.
- **A. thaliana MVP 구조 readout 입력 완비: A-ONT1 3/3 + A-CCS23 6/6** (`evidence/mvp_scoring_20260916/<RUN>/`). 모두 드라이버 경계
  `scientific_input_validation=false`, `coverage_policy=PENDING` — 판정 사용 전 외부 frozen-reference 게이트와 계약 v2 결합(`bind`) 단계 필요.
