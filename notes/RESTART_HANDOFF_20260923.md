# 재시작 인계 — 2026-09-23 10:15 PDT (A46 승인 후 WP1 진행 중)

> 이 문서가 **현재 정본**이다. 이전 인계(09-16, 09-15)와 `.atl/archive/resume_before_20260923.md`는 역사 기록이다.
> 상태는 실측 명령으로만 확인하고, 아래 "진행 중" 항목은 **재시작 후 반드시 다시 확인**한다(에이전트 결과는 새 세션에 전달되지 않는다).

## 1. 한 줄 요약

**A46 v1.36은 2026-09-22 22:23 PDT에 승인됐고, A48(v1.40)이 추가됐다. WP1(C0 증거 DB 파이프라인)의 코드는 워크트리 `/tmp/b6-wp1-20260922`에 있고 커밋되지 않았다.**
코드는 두 번의 독립검토 후 수정 라운드 3이 진행 중이었고, 감사 정렬 잡 6170748(187건)이 pronghorn에서 돌고 있다.

## 2. 승인·개정 (프로토콜 `revision/protocols/PROTOCOL_B1_frozen_v1.md`)

| 항목 | 상태 |
|---|---|
| **A46 v1.36** | **APPROVED** 2026-09-22 22:23 PDT. rule 12 동결이 발효돼, 변경은 새 amendment로만 한다. 승인 기록: [A46_APPROVAL_RECORD_20260922](a46_approval/A46_APPROVAL_RECORD_20260922.md). 이 문서에 승인 해시, WP1 착수 기록, 검토 이력, 저자결정이 모두 있다. |
| A46 주요 내용 | K = 31 트랙(rule 6). **rule 6a 체인 메모리**: 롱리드 인트론 체인을 게이트 교차 어텐션으로 디코더에 입력하며, MVP에서는 쓰지 않는다. C7(파일럿 규모). (ii-c)는 보조 결과로만 본다. A. thaliana 채점 = A-ONT1 + A-CCS23(TES 제외). Z. mays 체인은 M-EST만 쓴다. |
| A47.2 (v1.38) | A-CCS23 = lima FL. 접합·체인·TSS의 PacBio 구조팔로 쓰고 TES는 제외한다. M-CCS26은 키메라여서 구조 권한이 없다. |
| A47.3 (v1.39) | issue #60 번복: 학습 PacBio 49런 복원(원시 CCS 40 + FLNC급 Vv 9). WGS 2런은 제외. subreads와 RS II는 엑손 순서 용도로도 쓰지 않는다. |
| **A48 (v1.40, 2026-09-23)** | route-B 소스 간(OrthoDB vs Swiss-Prot) phase 불일치가 난 염기는 route-B phase 6채널을 0으로 한다(새 규칙). 문자 독법 두 가지를 기록했다: rule 4(b)의 MAPQ<5를 감사 secondary에도 적용하고, rule 4(a)의 edge `partial`을 분자 전체에 적용한다. N 포함 EST는 실측 후 판단한다. |
| 프로토콜 파일 | 커밋되지 않았다. 승인 시 sha256은 `e137c01e…`이고, A48 추가 후 `e6af5b4a…`이다. |

## 3. WP1 코드 (워크트리 `/tmp/b6-wp1-20260922`, 브랜치 `b6-wp1-20260922`, 커밋 안 됨)

- **구성:** base `729b990`(mvp-seed123-integration), 여기에 `est-approved-contract-20260917`를 병합했다(`--no-commit`, 충돌 0). 두 est 워크트리의 미커밋 route-B 수정도 옮겨 왔다.
- **주요 파일**
  - C0: `src/transgenic/evidence/{c0_schema,leakage}.py`, `revision/scripts/evidence/b6_build_c0.py`
  - 병합·보조: `b6_merge_evidence_db.py`, `b6_est_molecule_map.py`, `b6_c0_metadata.py`, `b6_leakage_intervals.py`
  - route-B: `b6_hsp_project.py`, `b6_routea_agreement.py`, `est_phase_project.py`(A46 차단 해제)
  - 감사 정렬: `b6_audit_runs.py`, `b6_audit_align.sbatch`
  - 시험: `tests/test_b6_*.py`
- **검토 이력**
  1. 1차 독립검토: REJECT. CRITICAL은 A(QNAME)와 B(member uid)의 분자 ID 불일치로 held-out 인접 HSP가 새는 것이었다.
  2. 수정 라운드 2 후 WP1 시험 117개 통과.
  3. 재검토: APPROVE_WITH_FIXES. 이전 13건은 모두 FIXED. 새 지적 3건: HIGH-1(`library` 공백 거부), MEDIUM-1(`--allow-unmapped`가 검사 전체를 해제), MEDIUM-2(제외된 분자의 HSP가 학습 염기의 0 처리를 결정하는 누출).
  4. 그 3건과 LOW 5건을 고치는 **수정 라운드 3이 10:15 PDT 현재 진행 중**이었다. 여기에 추가로 넘긴 것:
     - PacBio 리드명 파싱 버그. 메타데이터 생산기가 첫 토큰만 읽어 PacBio 전 리드가 실패한다. movie/zmw는 두 번째 토큰에 있다.
     - EST `--est-meta`(NCBI esummary TSV) 입력 지원.
- **재시작 후 할 일**
  1. 라운드 3의 결과를 확인한다. writer 보고는 새 세션에 오지 않는다. 워크트리 diff를 보고 아래 명령으로 전체 시험을 돌린다.
     ```
     cd /tmp/b6-wp1-20260922 && PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp \
       /tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q tests/test_b6_*.py          # 약 8분
     ... revision/scripts/tests/test_est_*.py tests/test_est_*.py                          # test_est_real_sample.py 4건은 환경 의존 실패(기존)
     ... tests/test_mvp_final_source_integration.py                                         # 바이트 핀 통과 필수, torch 2건 실패는 기존
     ```
     라운드 3 항목(HIGH-1, MEDIUM-1, MEDIUM-2, LOW 1–5, PacBio 파싱, `--est-meta`)이 모두 반영됐는지 하나씩 대조한다.
  2. **Codex 리뷰(저자 요청 "진행하고 있는거 codex로 리뷰").** 라운드 3이 끝난 뒤 실행한다. 프롬프트는
     `/tmp/claude-1000/-mnt-f-scratch-develop-Transgenic-transgenic-local/afbd6aa3-78b2-46f0-84d9-237b610b6c11/scratchpad/codex_wp1_review_prompt.md`에 있다.
     scratchpad가 사라졌으면 이 문서의 §3을 근거로 다시 쓴다. 실행은 `codex exec -s read-only -C /tmp/b6-wp1-20260922 "$(cat <prompt>)"`(codex-cli 0.153.4).
- **알려진 환경 사항**
  - `test_est_real_sample.py`의 4건은 원 워크트리에만 있는 `tasknotes/real_sample/replay_retained.py` 진단 스크립트에 의존한다. 회귀가 아니다.
  - 새 `src/transgenic/evidence` 모듈이 MVP의 `_expected_code_files()`에 잡힌다. 이 브랜치의 MVP 배포 매니페스트는 재생성이 필요하다.

## 4. 원격 (pronghorn) 잡과 데이터

| 잡/데이터 | 상태 (10:15 PDT) |
|---|---|
| **감사 정렬 6170748** | `--array=0-186%4`, 32 CPU/96 GB/2 d. 18/187 COMPLETED, REFUSED 0. 스크립트는 `transgenic/revision/scripts/evidence/b6_audit_align.sbatch`(sha256 `acacaa33…`), 런 목록은 `evidence/audit_align_v1/runs.tsv`(md5 `bf4b771d…`). 감사 BAM은 1차의 1.04–1.5배라 전체 1 TB 미만으로 예상한다(여유 26 TB). |
| 학습 PacBio 1차 정렬 | 49/49 완료·검증(6170399, 6170404). `evidence/longread_align_v1/{Sbicolor,Ptrichocarpa,Vvinifera}/`. |
| MVP 패널 정렬 | A-CCS23 6/6, A-ONT1 3/3(6170576, 6170586). `evidence/mvp_scoring_20260916/<RUN>/`. |
| 매니페스트 정본 | `evidence/DATASET_ROLES.longread_v1.tsv`, md5 `a12f9f00…`, role 174/31/10. 로컬 `data/manifests/`와 같다. 백업 `.bak-20260923-exclusions`와 `.bak-20260922-issue60`이 있다. |
| b6 할당표 | `data/manifests/DATASET_ROLES.b6_v1.tsv`(28행, `36aa934d…`): b6_input/b6_score. rule 10의 "DATASET_ROLES에 열로" 대신 같은 계열의 별도 파일로 둔 해석은 **저자 확인 대기**다. |
| 1차 정렬 | EST 9종, ONT 129런, PacBio 49런, 단백질 9종 모두 존재. blastx 9종 × {orthodb, swissprot}는 raw13 + DONE(`evidence/est_blastx/{remaining8_20260915,sitalica_pilot_20260914}`). |
| MVP 코드 | worktree `/tmp/mvp-seed123-integration-20260917` HEAD `729b990`(커밋 426c328 계약 v2 → 681c5a0 빈 엑손 → 729b990 핀). 푸시·병합하지 않았다. |

## 5. 저자 결정 대기

1. **EST BioProject:** 종 단위 EST에는 실제 단일 BioProject가 없다. (a) `NCBI_EST:txid<taxid>`(권고) 또는 (b) 종마다 대표 BioProject를 지정한다. 저자는 "BioProject를 직접 지정"이라고 했으므로 값 확인이 필요하다.
2. `DATASET_ROLES.b6_v1.tsv`를 별도 파일로 둔 해석(필요하면 A49에 한 줄).
3. N 포함 EST(route-B): 실측 후 판단(A48).
4. NCBI EST 요약 수집(esummary v2.0, `retmax=5000`, 약 1,134 요청, 1–1.5 h): 착수 지시 대기.

## 6. 이번 세션의 저자 결정 (09-23, 모두 반영됨)

- A48 세 항목, 감사 잡 제출.
- ONT `buffer_control_PRJNA1393863`(SRR36603452): 프로토콜을 알 수 없어 **제외**.
- `sdx_dRNA_PRJNA517295`: 중복이므로 **fastq 쪽만 유지**(SRR8491764, SRR12676675)하고 fast5 3런(SRR12822923, SRR12881185, SRR12822922)을 제외. 근거: 전체 리드 60-nt 접두 일치(잡 6170777). fast5 두 런은 98.9%, rep1 fastq↔fast5는 18.4%, 배경은 0.02–0.17%.
- ONT 프로토콜 분류(초안): direct_RNA 45 / cDNA 84. UMI는 언급이 전무해 `absent`로 둔다(중간 확신). 표는 조사 보고(§3 참조 기록)와 scratchpad `ena_ont.tsv`에 있다.

## 7. 다음 순서

1. 라운드 3 결과 검증(§3).
2. Codex 리뷰, 이어서 수정.
3. 감사 정렬 완료 확인: 187 DONE, 크기, REFUSED 0.
4. 메타데이터 준비: NCBI EST 요약 수집, ONT 프로토콜 시트 작성(저자 검토), BioProject 값 결정.
5. C0 빌드(종 × 소스, Slurm CPU) → 병합 → QC 게이트 → rule 4 제외 비율 보고.
6. route-B 투영과 A/B 합의 → `est_protein_hsp`.
7. 이후 WP2(누출 제외, 코퍼스 재빌드) → WP3(트랙·체인, 메모리 게이트) → WP4(트레이너, 체인 메모리 블록).

## 8. 무접촉·주의

- 커밋·푸시·병합은 하지 않았다. 승인 대상 프로토콜과 패널 TSV도 git에 없으므로 sha256이 동결 기준이다.
- B5 seed123 체인과 GPU 잡은 이번 세션에서 확인하지 않았고 건드리지도 않았다.
- Kimi challenge 모드는 `spawn E2BIG`로 실패한다. ask 모드 + 발췌 파일로 우회한다(15분 제한 주의).
