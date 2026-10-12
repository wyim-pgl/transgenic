# A47.2 내부 센서스 — 콘카테머·키메라 신호 (2026-09-22)

저자결정 "1. 복귀 2. 인정하되 TES 경계 판정에서만 제외" 적용 전, 구조팔 자격의 전제(콘카테머 없음)를 확인한 측정.
pronghorn login 노드, 읽기전용, 런당 앞 20,000리드(poly(A) 위치는 5,000리드). 양 말단 150 nt 밖을 "내부"로 본다.

스크립트: `concat_census_20260922.py`(내부 Iso-Seq 프라이머 k-mer / A≥20·T≥20), `illumina_census_20260922.py`
(Illumina P5 `AATGATACGGCGACCACCGA` / P7 `CAAGCAGAAGACGGCATACG` 코어, 양가닥), `polya_pos_20260922.py`(A/T≥20 위치·예시).

## 내부 프라이머 / 내부 poly(A)

```
file	n	internal_primer	internal_polyA20	either
Athaliana_zhang2023_PRJNA911826/SRR22719002.fa.gz	20000	0.0002	0.0056	0.0056
Athaliana_zhang2023_PRJNA911826/SRR22719003.fa.gz	20000	0.0003	0.0086	0.0086
Athaliana_zhang2023_PRJNA911826/SRR22719004.fa.gz	20000	0.0042	0.0073	0.0073
Athaliana_zhang2023_PRJNA911826/SRR22719005.fa.gz	20000	0.0047	0.0080	0.0083
Athaliana_zhang2023_PRJNA911826/SRR22719006.fa.gz	20000	0.0001	0.0043	0.0043
Athaliana_zhang2023_PRJNA911826/SRR22719007.fa.gz	20000	0.0007	0.0066	0.0066
Zmays_wang2020_zenodo2611319/F1maize.flnc.fa.gz	20000	0.0001	0.0032	0.0032
Zmays_B73_ccs_PRJNA1470126/SRR38818769.fa.gz	20000	0.0000	0.0215	0.0215
Zmays_B73_ccs_PRJNA1470126/SRR38818770.fa.gz	20000	0.0000	0.9695	0.9695
Zmays_B73_ccs_PRJNA1470126/SRR38818771.fa.gz	20000	0.0000	0.5931	0.5931
```

## Illumina 어댑터

```
file	n	illumina_any	illumina_end150	illumina_internal	internal_polyA20
pacbio/Zmays_B73_ccs_PRJNA1470126/SRR38818769.fa.gz	20000	0.6761	0.0102	0.6743	0.0215
pacbio/Zmays_B73_ccs_PRJNA1470126/SRR38818770.fa.gz	20000	0.9853	0.0389	0.9827	0.9695
pacbio/Zmays_B73_ccs_PRJNA1470126/SRR38818771.fa.gz	20000	0.5723	0.1467	0.5666	0.5931
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719002.fa.gz	20000	0.0000	0.0000	0.0000	0.0056
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719003.fa.gz	20000	0.0000	0.0000	0.0000	0.0086
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719004.fa.gz	20000	0.0000	0.0000	0.0000	0.0073
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719005.fa.gz	20000	0.0000	0.0000	0.0000	0.0080
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719006.fa.gz	20000	0.0000	0.0000	0.0000	0.0043
pacbio/Athaliana_zhang2023_PRJNA911826/SRR22719007.fa.gz	20000	0.0000	0.0000	0.0000	0.0066
pacbio/Zmays_wang2020_zenodo2611319/F1maize.flnc.fa.gz	20000	0.0000	0.0000	0.0000	0.0032
pacbio/Zmays_kinnex_hybrid_PRJNA1290227/SRR34503567.fa.gz	20000	0.0000	0.0000	0.0000	0.0001
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342050.fa.gz	20000	0.0000	0.0000	0.0000	0.0025
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342051.fa.gz	20000	0.0000	0.0000	0.0000	0.0021
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342052.fa.gz	20000	0.0000	0.0000	0.0000	0.0032
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342053.fa.gz	20000	0.0000	0.0000	0.0000	0.0029
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342054.fa.gz	20000	0.0000	0.0000	0.0000	0.0024
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342055.fa.gz	20000	0.0000	0.0000	0.0000	0.0044
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342056.fa.gz	20000	0.0000	0.0000	0.0000	0.0043
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342057.fa.gz	20000	0.0000	0.0000	0.0000	0.0061
Vvinifera/zhuosexiang_9tissue_PRJNA1185815/SRR31342058.fa.gz	20000	0.0000	0.0000	0.0000	0.0029
```

## 판독

- **A-CCS23 6/6 깨끗함**: Illumina 0, 내부 프라이머 ≤0.005, 내부 A/T≥20 0.4–0.9 % (M-FLNC 대조군 0.3 %와 같은 수준).
  poly(A) 위치는 3′ 말단 300 nt 안에 몰려 있다(SRR22719002: 4,947/5,000 리드). → lima FL, 콘카테머 문제 없음. **결정 2 적용.**
- **M-CCS26은 FLNC가 아니다 — 키메라.** SRR38818769 67 %, SRR38818770 98 %, SRR38818771 57 % 리드가 내부 Illumina 코어를 가진다.
  전형 구조: `…cDNA…AAAA(20+)` + `CAAGCAGAAGACGGCATACG…`(P7) 또는 `AATGATACGGCGACCACCGA…`(P5) + `…TTTT(20+)…cDNA` —
  꼬리끼리 어댑터로 이어진 두 분자. 9/22 앞선 말단 센서스의 "FLNC 서명" 판정은 **말단과 Iso-Seq 프라이머만 봐서 이걸 놓친 내 오판**이다.
  → 저자결정 1("M-CCS26 FLNC 자격 복귀")은 그 오판에 근거했으므로 **적용하지 않고 보류**, 저자에게 재보고.
- V. vinifera 9 (issue #60): Illumina 0, 내부 A/T≥20 0.2–0.6 % → FLNC급 판정 유지.
- M-FLNC·M-KNX: 깨끗함.
