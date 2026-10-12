# A47.3 — 학습 복원 40런 내부 센서스 (2026-09-22)

저자결정 "이슈 #60의 원시 CCS 40런을 학습에 쓴다" 적용 전, M-CCS26에서 놓쳤던 키메라·콘카테머를 이 40런에서 확인.
pronghorn login 노드, 읽기전용, 런당 앞 20,000리드. 스크립트 `concat_census_20260922.py`, `illumina_census_20260922.py`.
열: 내부(양 말단 150 nt 밖) Iso-Seq 프라이머 k-mer / 내부 A≥20·T≥20 / Illumina P5·P7 코어 any·말단150·내부.

```
file	internal_primer	internal_polyA20	illumina_any	illumina_end150	illumina_internal
Ptrichocarpa/jgi_F1hybrid_PRJNA709498/SRR35145693.fa.gz	0.0654	0.0877	0.0000	0.0000	0.0000
Ptrichocarpa/jgi_F1hybrid_PRJNA709498/SRR35145698.fa.gz	0.0034	0.0464	0.0000	0.0000	0.0000
Ptrichocarpa/jgi_F1hybrid_PRJNA709499/SRR35145694.fa.gz	0.0659	0.0877	0.0000	0.0000	0.0000
Ptrichocarpa/jgi_F1hybrid_PRJNA709499/SRR35145695.fa.gz	0.0605	0.0803	0.0000	0.0000	0.0000
Ptrichocarpa/jgi_F1hybrid_PRJNA709499/SRR35145696.fa.gz	0.0697	0.0910	0.0000	0.0000	0.0000
Ptrichocarpa/jgi_F1hybrid_PRJNA709499/SRR35145697.fa.gz	0.0117	0.0852	0.0000	0.0000	0.0000
Sbicolor/baijiu_PRJNA1034755/SRR26636574.fa.gz	0.0069	0.0073	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275171/SRR34257871.fa.gz	0.0024	0.0044	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275171/SRR34257872.fa.gz	0.0028	0.0052	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275171/SRR34257933.fa.gz	0.0017	0.0044	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275171/SRR34257934.fa.gz	0.0019	0.0060	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275171/SRR34257935.fa.gz	0.0021	0.0049	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275229/SRR34258142.fa.gz	0.1028	0.0790	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275229/SRR34258143.fa.gz	0.1011	0.0775	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275229/SRR34258144.fa.gz	0.0163	0.0172	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275229/SRR34258145.fa.gz	0.0140	0.0169	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275264/SRR34258119.fa.gz	0.0050	0.0035	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275264/SRR34258120.fa.gz	0.0040	0.0037	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275264/SRR34258159.fa.gz	0.0062	0.0049	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275264/SRR34258160.fa.gz	0.0047	0.0037	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275264/SRR34258161.fa.gz	0.0080	0.0071	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275290/SRR34257851.fa.gz	0.0019	0.0016	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275290/SRR34257908.fa.gz	0.0022	0.0019	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275290/SRR34258111.fa.gz	0.0041	0.0032	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275290/SRR34258112.fa.gz	0.0052	0.0044	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275290/SRR34258113.fa.gz	0.0024	0.0019	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275324/SRR34258146.fa.gz	0.0088	0.0189	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275324/SRR34258147.fa.gz	0.0089	0.0187	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275324/SRR34258148.fa.gz	0.0340	0.0211	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275324/SRR34258150.fa.gz	0.0315	0.0204	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275413/SRR34257924.fa.gz	0.0401	0.0604	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275413/SRR34257925.fa.gz	0.0748	0.0868	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275413/SRR34257926.fa.gz	0.0050	0.0338	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275413/SRR34257927.fa.gz	0.0046	0.0347	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275413/SRR34257928.fa.gz	0.0059	0.0346	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275435/SRR34257903.fa.gz	0.0040	0.0043	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275435/SRR34257904.fa.gz	0.0049	0.0049	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275435/SRR34257905.fa.gz	0.0027	0.0031	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275435/SRR34258084.fa.gz	0.0032	0.0036	0.0000	0.0000	0.0000
Sbicolor/wray_PRJNA1275435/SRR34258085.fa.gz	0.0032	0.0032	0.0000	0.0000	0.0000
```

**판독.** Illumina 어댑터 40/40 **0** — M-CCS26 같은 키메라 없음. 콘카테머 신호(내부 프라이머 또는 A20)는 대부분 <1 %이나
일부 런이 5–10 %: *P. trichocarpa* PRJNA709498 SRR35145693, PRJNA709499 SRR35145694/695/696(6–9 %), SRR35145697·698(내부 A20 5–9 %);
Wray PRJNA1275229 SRR34258142/143(~10 %); PRJNA1275413 SRR34257924/925(4–9 %). `refine`을 거치지 않았으니 예상된 값이다.
A47.3 규칙 5: 새 필터 없이 A18.5(primary·query fraction ≥0.80)로 처리하고 런별 비율을 기록해 런별 민감도 제외를 보고할 수 있게 한다.

V. vinifera 9 (FLNC급)의 내부 센서스는 [A47_2_INTERNAL_CENSUS](A47_2_INTERNAL_CENSUS_20260922.md): Illumina 0, 내부 A20 0.2–0.6 %.

## 복원 실행 (pronghorn, 2026-09-22)

- `mv` 3개 디렉터리군(Wray 7, Populus 2, Zhuosexiang 1) + baijiu는 SRR26636574의 `.DONE/.fa.gz/.md5`만 이동, `filereport.tsv`·`log`는 복사.
  목적지 사전부재 확인 후 실행. **inode·크기 49/49 보존**, 복원된 49개 `.md5` **전수 `md5sum -c` 실패 0**.
  전후 목록: pronghorn `~/issue60_restore_{before,after}_20260922.txt`.
- 격리에 남음: baijiu SRR26636578·SRR26636582(WGS 게놈 HiFi) + A. thaliana 빈 디렉터리 2개.
- 정본 매니페스트 `evidence/DATASET_ROLES.longread_v1.tsv` 동기화: 이전 md5 `c2f4c2e8…`(= git HEAD, 09-04) → `e07dc771…`,
  백업 `DATASET_ROLES.longread_v1.tsv.bak-20260922-issue60`. 순차이는 오늘 변경분뿐(49행 복원 + A-CCS23 6·M-CCS26 3·WGS 2행 기록).
  복원 49행 경로 존재 49/49. role 집계 178 / 31 / 6.
- `RETIRED_DO_NOT_USE/README.md`에 번복 절 추가.
