# A46 승인 기록 — 2026-09-22 22:23 PDT

**저자 승인: "승인"** (이 세션, 2026-09-22 22:23 PDT). 대상: `revision/protocols/PROTOCOL_B1_frozen_v1.md`의 **A46 (v1.36)**.
범위는 2026-09-21·09-22 draft revision을 모두 반영한 현재 본문이다.

**이 순간부터 rule 12 동결 조항이 적용된다.** A46의 숫자·명령·규칙은 **후속 개정으로만** 추가하거나 바꿀 수 있다.
**rule 15 충족 → WP1 착수 가능.**

## 승인 전 검토 경로

| 단계 | 결과 |
|---|---|
| 동결 체크리스트 | [A46_FREEZE_CHECKLIST_20260921.md](A46_FREEZE_CHECKLIST_20260921.md) — 09-22 ✏️ 갱신 |
| 체인 메모리 설계 | Kimi(ask) proceed-with-changes → v1, 독립검토 APPROVE_WITH_FIXES → [v2](B6_CHAIN_MEMORY_DESIGN_v2_20260922.md) |
| A46 전체 + SPEC-2c 2차 검토 | 독립검토 APPROVE_WITH_FIXES(CRIT 0 · HIGH 6 · MED 11 · LOW 7) + Kimi approve-with-fixes — 전부 반영 |
| Kimi 최종 확인 | 발췌본만 읽게 한 좁은 ask(job `63a6a056…`): 수정 13항목 확인, 기본값 3건 수용, **yes-after-edits → 1건 반영** |
| rule 17 인용 무결성 | MVP §5.3 인용이 A46 rule 17과 정규화 후 동일(`*_\`` 제거 1,958자 / `**` 제거 1,965자, sha256 `b5b918bf97d0d60003b88b1d2a600e563e702ceb2e5e684ae59b088360b8aa21`) |

## 승인 시점 해시

| artefact | 값 |
|---|---|
| `revision/protocols/PROTOCOL_B1_frozen_v1.md` (승인 표기 후) | 1,141줄, sha256 `e137c01edfa076d293c56d7af2b2f557e111a8389717312be89a97261e3c4284` |
| `revision/protocols/B6_MVP_PILOT_v1.md` (decision 5 재측정) | 373줄, sha256 `067c1e2f900b7461e7ead7f3bd70077351af226f498996f81058c203bceff360` |
| `revision/protocols/b6_panels_v1.tsv` | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` (A46 기록과 일치) |
| `revision/protocols/b6_eligible_loci_v1.tsv` | `7ad895fcc25abf75c8fd682acd448c706c5d7f59c22da7c6aea0c8404219ecd6` (일치) |
| `revision/protocols/b6_est_molecules_v1.tsv` | `36de5b6827cdba2b27c72a663cfb4b2a8c46a567a1d31a6450f438be604e869c` (일치) |
| `revision/scripts/71_b6_panels.py` | `b81ec50baf88e10147c92bba390df5c63daf9fafc644d772f78df6080d86391c` (일치) |
| `revision/scripts/72_mvp_select_loci.py` | `eab68edb76a7bb4397b1bef413fc47d68b4889553715e9efefc6573d9264f0d1` (일치) |
| `revision/scripts/72a_est_molecules_per_locus.sh` | `9e27472eb29798f8008275001293cfbf7dd23e67538127a5f35c146f6311b272` (일치) |
| `data/manifests/DATASET_ROLES.longread_v1.tsv` | md5 `e07dc77124bff2a79a0c2ad09b6545a1` = pronghorn 정본 `evidence/DATASET_ROLES.longread_v1.tsv`(09-22 동기화), sha256 `222dbc8152dc9a3a…` |
| 체인 메모리 설계 v2 | `248947d268cec3c42fb9cf11733dddc291c3b124a9cf5ab67654eefd5322f425` |
| 명령 명세 v2(SPEC-1/1b/2/2c) | `7a06288ce218a49f4bb7c161d1379d49862e79955ff598925e6830667b41a2c6` |
| 도구 PROVENANCE 5파일 | `cpu_tools_v1` `37caa5d9…` · `gpu_host_tools_v1` `92b8babe…` · `gpu_runtime_v1` `c084c76b…` · `univec_screen_v1` `4fd697fe…` · `protein_db_v1` `b356d944…` (체크리스트 §2) |
| 기존 artefact 재검증(09-21) | B5 코퍼스 md5 `dda7e78c…` · SIF `1a5dee27b857…` · OrthoDB `fa0795b6…` · Swiss-Prot `8d66a18e…` · diamond `0fb259d7…` |

## 승인 후로 미룬 것 (rule 12가 허용: 아직 없는 산출물의 해시만)

C0 테이블 · 트랙 · 체인 캐시 · evidence DB · run identity — **첫 사용 전에 기록**한다.

## 주의

- 승인 대상 파일들은 **git에 커밋되지 않았다**(`PROTOCOL_B1_frozen_v1.md` 수정, `B6_MVP_PILOT_v1.md`·패널 TSV는 미추적).
  위 sha256이 동결 기준이다. 커밋은 저자 지시가 있을 때만 한다.
- 이후 A46에 닿는 변경은 모두 **새 amendment**(A48 이후)로 기록한다. 이 파일과 체크리스트는 기록물이며 규칙을 바꾸지 않는다.

## 다음

**WP1 착수.** 체인 메모리 구현은 WP1/WP4 트레이너 브랜치에서 하고, MVP 파일은 건드리지 않는다(`tests/test_mvp_final_source_integration.py`가 681c5a0 핀).
WP1 범위는 C0(SPEC-1), 코퍼스 재빌드(SPEC-1b), 트랙(SPEC-2), 체인(SPEC-2c), route-B phase, 누출 차단이다.

## WP1 착수 기록 (2026-09-22, 저자 "바로진행해")

- **WP0 잔여 해소 — `b6_input`/`b6_score` 할당표:** `data/manifests/DATASET_ROLES.b6_v1.tsv` (28행, sha256 `36aa934d29759e1594f098bd74b4a43225a64a77f9673b373e99a8db5f2f09f7`, 사이드카 `.sha256`).
  A-EST·A-ONT2 입력(트랙+체인), A-ONT1·A-CCS23 채점, M-EST 입력(트랙+체인), **M-HQ18 입력(트랙만)**, M-FLNC·M-ONT 채점. 입력∧채점 동시 행 0.
  ⚠️ **해석:** rule 10은 "written into DATASET_ROLES (`b6_input`/`b6_score` columns, hashed)". 기존 roles 파일에 열을 덧붙이면 EST 정렬 DONE이
  기록한 roles md5와 MVP 드라이버의 엄격한 열 검사가 깨지므로 **같은 DATASET_ROLES 계열의 별도 파일**에 두었다. 저자 확인 대상(필요하면 A48 한 줄).
- **작업 공간:** worktree `/tmp/b6-wp1-20260922`, 브랜치 `b6-wp1-20260922` = `729b990` + `est-approved-contract-20260917` 병합(미커밋, 충돌 0) +
  두 워크트리의 미커밋 route-B 수정(길이 검증 수정·routeA fixture·closure) 이식. 기준선: route-B·evidence·MVP 시험 통과, 예외 = `test_est_real_sample.py` 4건
  (원 워크트리에만 있는 `tasknotes/real_sample/replay_retained.py` 진단 스크립트 의존 — 환경 의존, 회귀 아님), 기존 torch 부재 실패.
- **구현 분담(병렬 writer, 파일 소유권 분리):** A = C0 스키마·빌더·rule 4(a)(b)(c)(d)·병합(`src/transgenic/evidence/`, `b6_build_c0.py`, `b6_merge_evidence_db.py`);
  B = 감사 정렬 드라이버(`b6_audit_align.sbatch`)·route-B 운영 투영(`b6_hsp_project.py`, `est_phase_project` 차단 해제·다중 HSP)·route A/B 합의 QC.
  공유 인터페이스 = `est_protein_hsp` 열 23개(두 지시에 동일 명시).

### WP1 코드 독립검토 (2026-09-23) — REJECT(수정 가능) → 수정 라운드 2 진행
- CRITICAL: route-B HSP 행이 EST 분자 누출상태와 결합되지 않음(A: QNAME 기반 id, B: closure member uid 기반 id → 결합 불가, held-out 인접 HSP가 phase 채널로 샘).
- HIGH: `est_protein_hsp` PK가 `hsp_id`뿐(인트론 넘는 HSP 충돌) / EST 감사 BAM 경로 불일치 / MAPQ<5 규칙이 secondary(MAPQ 항상 0)에 적용 → 다중매핑 분자 대량 제외(저자 판단).
- MEDIUM: edge-partial 분자 전역 적용(긴 유전자 편향, 저자 판단) · split NULL 타일 · ≥50 규칙 분자합 vs 멤버최대(→ 동결 문구대로 분자합) · miniprot 상한 단백질 · 조용한 기본값 3곳 · Swiss-Prot/OrthoDB 합동 충돌해소 · step-7 게이트 비독립.
- 규모: ONT 종 단위 전량 메모리 → OOM 확실 → 스트리밍 재작성.
- 확인된 정상: rule 4(a) 재계산(시드 문자열·decoy·N-mask 바이트 대조), 좌표 변환 1회·donor0/acceptor0 가닥인식, phase 규약, 감사 드라이버 명령 차이 = secondary 옵션뿐.
- 수정 writer A2(C0: 단일 molecule-map 생산기·결합 게이트·PK·경로·메타데이터 생산기·스트리밍 등) · B2(route-B: molecule-map 소비·소스별 충돌해소·N 검사 HSP 구간 한정·A19 함수) 병렬 진행.
- **수정 라운드 2 완료(2026-09-23):** A2(C0) 11건 + B2(route-B) 6건 반영. 단일 EST molecule-map 생산기(`b6_est_molecule_map.py`, member_uid = qname),
  HSP↔EST C0 결합 필수(고아 실패, leakage_excluded·train_tiles_excluded를 EST C0에서), 병합 단계 독립 hsp_gate, PK `(hsp_id, interval_index)`,
  감사 경로 = sbatch 레이아웃, ≥50 규칙 분자 합(동결 문구), miniprot 상한 단백질 제외, 조용한 기본값 제거, step-7 게이트 독립화, 감사 CL 재검사,
  phase 규약 저장, 메타데이터 생산기(`b6_c0_metadata.py`), **스트리밍**(20만 리드 540 MB, 실 A. thaliana ONT 예상 ≈ 2.5 GB), 소스별 HSP 충돌해소,
  A19 함수 단일화. 저자 판단 대기 2건은 이름 붙은 상수(`LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES`, `EDGE_PARTIAL_MOLECULE_WIDE`) + QC 카운터.
  **통합 재실행: WP1 117/117, 기존 492(+4 환경), MVP 74(+3 torch), 바이트 핀 통과.**
- **실데이터 공백(남음):** EST library/clone은 A. thaliana·Z. mays만 GenBank flat 파일로 존재(나머지 7종은 메타데이터 생산기가 fail closed) /
  ONT 런별 protocol·UMI 상태가 저장소 어디에도 없음(A16; 미해소 시 빌드 실패) / PacBio 리드명 SRA 재명명 여부 미확인 / 처리속도 ≈0.45 ms/리드
  (ONT 1억 리드 ≈ 13 h/종) / A19 계보표는 학습 9종만.

### 2026-09-23 저자결정 + 감사정렬 제출
- 저자: "1. 적용 2. 분자 전체 적용 3. 0 4. 실측후 판단 5. 감사 잡 제출" → **PROTOCOL_B1 A48 (v1.40)** 신설: route-B 소스 간(OrthoDB vs Swiss-Prot) phase 불일치 염기 → route-B phase 6채널 0(새 규칙);
  rule 4(b) MAPQ<5를 감사 secondary에도 적용·rule 4(a) edge `partial`을 분자 전체에 적용(문자 독법 기록). N 포함 EST는 실측 후 판단. 프로토콜 sha256 `e6af5b4a…`.
- 코드: `b6_routea_agreement.compare_sources` 추가(출력에 `reason` 열: route_a_b_disagreement / route_b_cross_source_disagreement, QC `cross_source`), 시험 +2;
  `leakage.py` 두 상수 주석을 "AUTHOR RULING 2026-09-23"으로. routea+leakage 시험 24 통과.
- **감사정렬 잡 6170748** 제출(`--array=0-186%4`, 32 CPU/96 GB/2 d): 원격 배포 `transgenic/revision/scripts/evidence/b6_audit_align.sbatch`(sha256 `acacaa33…`)·`b6_audit_runs.py`(`3821459a…`),
  런목록 `evidence/audit_align_v1/runs.tsv`(187 = EST 9 + ONT 129 + PacBio 49, md5 `bf4b771d…`). 사전 점검실행(EST 0 / ONT 9 / PacBio 150): 명령이 1차와 secondary 옵션만 다름, 출력 배치 = C0 빌더 계약.
  저장공간: 1차 장독 527 GB, 파일시스템 여유 26 TB — 첫 완료 태스크 크기로 확대 비율 확인 예정.
- 수정 라운드 2 재검토 진행 중.
- **재검토(라운드 2) APPROVE_WITH_FIXES:** 이전 13건 전부 FIXED, 회귀·데이터 손상 없음, A48 판정대로 동작 확인(시험 b6 119 통과, 규모 시험 20만 리드 521 MB).
  새 지적: HIGH-1 `library`(GenBank /clone_lib) 공백 거부로 EST map 전체 중단 / MEDIUM-1 `--allow-unmapped`가 검사 전체 해제 / **MEDIUM-2 제외된 분자의 HSP가
  학습 염기의 phase 0 처리를 결정(작은 누출 경로)** / LOW 5건. → 수정 라운드 3 writer 진행.
- **감사정렬 6170748 초기 관측:** 16/187 COMPLETED 0:0, REFUSED 0. 감사 BAM/1차 BAM 크기비 1.04–1.10(A. thaliana ONT), 1.51(G. max EST) → 전체 < 1 TB 예상, 여유 26 TB.

### C0 메타데이터 공백 조사 (2026-09-23)
- **EST:** pronghorn에 GenBank flat 파일은 **어느 종에도 없다**(A. thaliana·Z. mays 포함; §3의 flat 파일 수집 계획이 실행된 적 없음). 9종 5,648,175건.
  NCBI 표본 전 레코드에 `/clone_lib` 존재(BioSample ID로 시작 → 안정 키), `/clone`은 대부분. efetch gb는 레코드당 0.6 s라 불가;
  **esearch history + esummary v2.0 `retmax=5000` ≈ 1,134 요청, 1–1.5 h, XML 12 GB → TSV ~0.5 GB** 권고. EST 레코드에 BioProject 없음 → `--bioproject` 저자결정 필요.
- **ONT 129런(22 BioProject):** ENA 메타 + 앞 2k 리드 프라이머(SSP) 검사로 **direct_RNA 45 / cDNA 84** 분류, UMI 언급 전무 → `umi_status=absent`(중간 확신).
  미해결: `buffer_control_PRJNA1393863`(1런, 프로토콜 기술 없음·SSP 0) / **`sdx_dRNA_PRJNA517295` 중복 위험**(fast5·fastq 제출이 같은 분자일 가능성:
  SRR12822923 vs SRR8491764 앞 2만 리드 중 60nt 접두 1,915 동일, SRR12822923·SRR12881185 리드 수 725,158 vs 725,156 — dRNA는 리드=분자라 지지 부풀림).
- **PacBio:** 리드명 두 번째 토큰에 `{movie}/{zmw}/ccs/1` 보존(2만 = 2만 ZMW), 그러나 **메타데이터 생산기가 첫 토큰만 봐서 전 PacBio 리드 실패** → 수정 라운드 3에 추가.
- 표본·ENA 표: scratchpad(`ena_ont.tsv`, esummary 시험 배치).

### 2026-09-23 저자결정: C0 메타데이터 (\"1. BioProject를 직접 지정 2. 제외 3. 한쪽만 남겨\")
- **ONT buffer_control_PRJNA1393863 (SRR36603452) → `excluded`:** 프로토콜 기술 없음, SSP 0/2,000 → A16 상태 설정 불가.
- **sdx_dRNA_PRJNA517295 중복 → fastq 쪽만 유지:** 전체 리드 60-nt 접두 정확일치(잡 6170777, 41 s): fast5 SRR12822923 vs SRR12881185 **98.9 %**(같은 fast5 두 번),
  rep1 fastq SRR8491764 vs fast5 18.4 %/14.8 %, rep2 fastq SRR12676675 vs fast5 SRR12822922 0.50 %/0.45 %, 교차 rep 배경 0.02–0.17 %.
  UUID는 SRR12822923·SRR12676675에만 남아 rep쌍 UUID 대조 불가 → ENA 제목(rep1/rep2 fastq vs raw fast5) + 측정으로 **fast5 3런 제외**, 제출자 fastq 2런 유지.
- 매니페스트 role 174 / 31 / 10, md5 `a12f9f002f97a7c44b60b052d635393c`, **원격 정본 동기화**(백업 `.bak-20260923-exclusions`, 이전 `e07dc771`).
  제외 4런의 1차·감사 정렬은 존재하나 A18.3 role 게이트로 C0에 들어가지 않는다.
- **BioProject 직접 지정:** EST는 종 단위 NCBI EST division 모음이라 **단일 실제 BioProject가 없다**(레코드에도 DBLINK 없음, BioSample만 `/clone_lib`에). 지정할 값 저자 확인 대기.
