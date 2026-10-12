# FLNC 정책 — 저자 결정 2026-09-21

> ⚠️ **이 문서의 가용성 표(§"가용성 실측")는 A-CCS23·M-CCS26 행이 틀렸다.** ENA fastq 경로의
> `_subreads` 접미사를 산물로 읽은 오판이며, 같은 날 리드 센서스로 정정했다 —
> **정본은 `PROTOCOL_B1_frozen_v1.md` A47.1** 과 [FLNC_AMENDMENTS_20260921.md](FLNC_AMENDMENTS_20260921.md)
> 의 "3차: 정정" 절이다. 유효하게 남는 것: **A-HiFi ERR13994458만 subreads**이고(제출자 파일명·형식 근거),
> 규칙 1~3(로컬 FLNC 미생산 / 다운로드 가능할 때만 FLNC 자격 / 그 외 EST급)은 그대로다.

## 저자 지시 (원문)

> "flnc가 다운 받을때 아니면 만들지 말고 그냥 eST 처럼 쓴다"

앞선 지시 "ccs/lima/refine → FLNC를 돌리지 않는다", "subreads로 판명하고 다음 진행해"의 연장·확정.

## 규칙 (확정)

1. **FLNC를 우리가 생산하지 않는다.** `ccs` / `lima` / `isoseq refine` 실행 금지. subreads→CCS→FLNC 변환 잡을 새로 설계·제출하지 않는다.
2. **FLNC가 공개 저장소에서 직접 다운로드 가능한 경우에만** FLNC(전장 cDNA 분자) 자격으로 쓴다.
3. **그 외에는 EST처럼 쓴다.** 즉 서열수준 증거(A-EST와 동급의 정렬기반 서열 증거)로만 사용하고, tier-1 long-read **구조 검증**(junction/chain 지지) 자격은 부여하지 않는다.

## 가용성 실측 (2026-09-21, ENA portal API, 읽기전용)

| 대상 | 조회 결과 | 판정 |
|---|---|---|
| **A-HiFi ERR13994458** (PRJEB77203, Zhong 2025) | 제출파일 단 1개 `Col_isoseq.subreads.bam` (`submitted_format=BAM`), Sequel II, read_count **9,093,086**, base_count 16,570,241,722, submitted_bytes 32,961,697,765, md5 `7e75b3e771529738a1cd8aaabbae76de`. ENA 파생 fastq `ERR13994458.fastq.gz`는 이 subreads에서 유래. CCS/FLNC 산물 **없음**. 프로젝트 `result=analysis` 산물 **0건**. | **FLNC 다운로드 불가 → EST처럼 사용** |
| **A-CCS23** (PRJNA911826, Zhang 2023, 36런 전체) | 모든 런의 유일한 파일이 `SRR*_subreads.fastq.gz`. `submitted_format`/`submitted_ftp` 공란, bam/sra 산물 없음. `result=analysis` 산물 **0건**. | **FLNC/CCS 다운로드 불가 → EST처럼 사용** (기존 "subreads로 판명" 확정) |
| 같은 PRJEB77203 내 `ERR13987671 Col0_PacBio.fastq.gz` | library_strategy=**WGS** (게놈 PacBio) | IsoSeq 아님, 본 건과 무관 |

조회 범위: ENA `filereport` read_run + analysis. 논문 부록/GEO/Zenodo 등 ENA 밖 경로는 조회하지 않았으므로 "지구상 어디에도 FLNC가 없다"는 주장은 하지 않는다. 다만 **ENA/SRA 공개 경로에는 없다**는 것은 실측이다.

## 결과 (파이프라인에 미치는 영향)

- **A-HiFi(ERR13994458)는 구조 검증 소스로 사용 불가.** B6 MVP §2의 "≥2 A-HiFi **or** ≥3 A-ONT1" 중 A-HiFi 항은 이번 실행에서 충족될 수 없다.
- **A-ONT1(FLIC PRJNA1087576, SRR28341943-45)이 유일한 tier-1 구조 long-read 소스로 남는다.**
- 따라서 MVP §5.3 **missing-structural safeguard가 발동**: 예정 STOP은 그대로 STOP이 아니라 **INCONCLUSIVE로 하향**된다. GO는 자체 1차 기준과 STOP-first 순서를 따로 만족해야 한다. (이전 질의였던 "ONT-only STOP 가능 여부"의 답: 본 결정 아래에서 **ONT 단독으로 STOP을 확정하지 못한다**.)
- 두 PacBio 세트는 버리지 않는다. **서열수준(EST급) 증거**로 편입 가능하며, per-source missingness는 그대로 보존해 구조 검증 슬롯을 채운 것처럼 보이지 않게 한다.

## 후속으로 필요한 문서 개정 (미실행 — frozen 문서 무단수정 안 함)

1. `revision/protocols/PROTOCOL_B1_frozen_v1.md` A14(:591-602)/A15(:605-614): A-HiFi·A-CCS23의 `validation-only` 역할 라벨을 **서열수준(EST급) 증거**로 재지정.
2. `revision/protocols/B6_MVP_PILOT_v1.md` §2(:99-150)·§5.3(:240-285): A-HiFi 구조지지 항의 이번 실행 불충족 명시 + safeguard 경로 고정.
3. `data/manifests/DATASET_ROLES.longread_v1.tsv`: PRJNA911826 6행의 role 재지정, PRJEB77203 행은 여전히 부재.
4. 루트 `quarantine.md:228-230`의 사실오류(PRJNA911826을 "subreads→ccs→FLNC 세트"로 지칭; 실제로는 PRJEB77203/ERR13994458) 정정 — 본 결정으로 두 건 모두 FLNC 미생산이므로 재변환 작업으로 되살아나지 않게 못박을 것.
5. `notes/wait_work_20260916/longread/frozen/revision/scripts/mvp/mvp_score.py:26-29` `require_coverage_policy`의 무조건 raise: 이제 계약이 정해졌으므로 contract-driven 구현으로 교체.

## 범위

본 문서는 기록만 했다. frozen 프로토콜·매니페스트·소스 수정, 다운로드, Slurm 제출·취소, 커밋/푸시 없음. GPU 잡 3187760 및 B5 체인은 무접촉.
