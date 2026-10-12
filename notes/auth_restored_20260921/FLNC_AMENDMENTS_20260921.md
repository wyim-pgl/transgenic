# FLNC 정책 문서개정 5건 — 실행 기록 (2026-09-21)

저자 승인: "① B1 A14/A15 역할 재지정 ② B6 MVP §2/§5.3 ③ DATASET_ROLES.longread_v1.tsv
④ 루트 quarantine.md:228-230의 사실오류 ⑤ mvp_score.py:26-29의 무조건 raise → 계약기반 구현 교체.
### 이거 진행해"

근거 결정/실측: [FLNC_POLICY_20260921.md](FLNC_POLICY_20260921.md).

## ① PROTOCOL_B1_frozen_v1.md — 신규 개정 A47 (v1.37)

`revision/protocols/PROTOCOL_B1_frozen_v1.md` (1001 → 1044행).

frozen 본문을 덮어쓰지 않고 **A1..A46과 같은 방식으로 A47을 추가**했다(이 저장소의 개정 관례).
- **A47 신설**: 규칙 4항(로컬 FLNC 미생산 / 다운로드 가능할 때만 FLNC 자격 / 그 외 EST급 서열증거로
  구조권한 없음 / 학습금지는 불변), ENA 실측표 4행(A-HiFi·A-CCS23·M-FLNC 대조군·M-CCS26),
  변경되는 것(A14·A15·§3.2·§§6-9·A46/B6·Z. mays 불변), 변경되지 않는 것, 기록정정, 인용·해시 무결성,
  자체 date/state/change/reason 표.
- **A14 머리에 ✏️ 포인터 배너**: A-HiFi의 `→ ccs` 지정 무효, A-ONT1이 유일한 A. thaliana tier-1
  구조 검증 세트. 나머지 A14는 유효.
- **A15 머리에 ✏️ 포인터 배너**: A-CCS23 재지정, 검증전용 구조 집합은 이제 {A-ONT1}. M-CCS26의
  `*.ccs.fastq.gz` 기술이 ENA에서 재현 안 됨(측정), role 재지정은 저자 결정 대기.
- 파일 끝 누적 change-log에 v1.37 행 추가.

## ② B6_MVP_PILOT_v1.md — §2 / §5.3 / §6

`revision/protocols/B6_MVP_PILOT_v1.md` (331 → 355행, 이 파일은 아직 untracked).

- **§2 구조 readout 불릿**에 ✏️ 배너: A-HiFi 팔은 충족 불가하며 **실행하지 않는다**. pass-2 명령 중
  앞의 두 ONT 명령만 적용, `splice:hq` PacBio 명령 미사용, `A-HiFi.panel.bam`은 생산·기대하지 않음.
  A-HiFi는 **per-source missing**으로 기록(0 지지로 채우지 않음).
- **§2 Definition**: `≥2 A-HiFi or ≥3 A-ONT1` 문구는 유지하되 이번 실행에서 ONT 항만 충족 가능함을
  명시. `THRESHOLDS` 상수는 변경 없음(PacBio 분기에 입력이 없을 뿐).
- **§5.3**: 결정규칙 인용블록은 **한 글자도 바꾸지 않았다**(A46 rule 17이 verbatim 인용). 대신 블록
  **뒤에** A47 적용 노트를 추가 — 구조 readout은 A-ONT1만으로 계산, 가용성 안전장치가 작동하여
  **A-HiFi에 기대는 예정 STOP은 INCONCLUSIVE로 하향(GO 아님)**, GO 기준과 §5.4는 불변.
- **§6 day 5 행**: A-HiFi 다리 제거, done-when을 `A-ONT1.panel.bam`으로 한정.
- **인용 무결성 검증(실행함)**: §5.3 인용블록과 A46 rule 17의 인용문을 Markdown 강조·줄바꿈 정규화 후
  비교 → **동일**(정규화 1,965자). A46의 "325 lines / sha256 69968c4d…" 핀은 본 개정 이전에 이미
  파일 전체 기준으로 낡아 있었고(개정 전 331행), 승인 시점에 다시 떠야 한다. A47에 그대로 기록했다.

## ③ data/manifests/DATASET_ROLES.longread_v1.tsv

- `pacbio/Athaliana_zhang2023_PRJNA911826` **6행**: `role` `b1_validation_only` →
  **`b1_sequence_evidence_only`**, `basis`/`note`를 A47 실측으로 교체.
- `pacbio/Zmays_B73_ccs_PRJNA1470126` **3행**: `note`에만 측정사실 기록(ENA가 subreads만 제공,
  A15의 `.ccs.fastq.gz` 기술 재현 불가), **`role`은 변경하지 않음 — 저자 결정 대기**
  (maize 검증 계약 §3.1/A15에 걸려 자동 적용 부적절).
- A-HiFi(PRJEB77203)는 애초 이 매니페스트에 행이 없고 신규 취득하지 않는다.
- **안전 확인**: 추적 대상 코드 중 `b1_validation_only` 리터럴을 소비하는 곳은 없다(매니페스트뿐).
  장독 정렬기가 거르는 값은 `c2_training_eligible`이므로, 이 재지정이 PacBio를 학습가능으로
  만들지 않는다. 전 행 14열 유지 확인.

## ④ quarantine.md D3

228행 이하 ⚠️ 문단(2문장)을 ✏️ 정정 블록으로 교체:
① 데이터셋 혼동 정정(변환 전제 세트는 Zhang/PRJNA911826이 아니라 **PRJEB77203/ERR13994458**;
`.atl/archive/todo_before_20260916.md:569`에도 같은 혼동) ② 저자 결정으로 종결 — 어느 세트에서도
변환 미실행 ③ ENA 실측 3세트(PRJNA1470126 포함) ④ 처분(재지정/보류/대조군 M-FLNC).
**"Zhang CCS 변환 미실행"을 근거로 변환작업을 되살리지 말 것**을 명문화.

## ⑤ mvp_score.py `require_coverage_policy` — 계약기반 구현

`mvp_score.py`는 main에 없고 구현 브랜치에만 있다. 작업 위치:
**worktree `/tmp/mvp-seed123-integration-20260917` (branch `mvp-seed123-integration-20260917`, base 9603e20)**.
커밋하지 않음.

- `revision/scripts/mvp/mvp_score.py`
  - 무조건 `raise ValueError('coverage policy pending author approval …')` 제거.
  - 계약 상수 추가: `COVERAGE_POLICY_ID='B1-A47-v1.37/MVP-v1'`,
    `COVERAGE_STRUCTURAL_SOURCES=('ONT',)`, `COVERAGE_UNAVAILABLE_SOURCES=('HiFi',)`,
    `COVERAGE_POOLING='within_source'`.
  - `require_coverage_policy(policy)`가 **검증기**가 됨: 6개 필드 전부 명시 필수(기본값 없음),
    policy_id 일치, 구조소스는 `('ONT',)`뿐, `unavailable_sources`에 HiFi 명시(0으로 채우지 않음),
    임계값은 `mvp_junction_table.SOURCE_THRESHOLDS`(= frozen `59_evidence_support.THRESHOLDS`)와
    일치, `pooling='within_source'`, `per_run_retained=True`. 위반은 전부 fail-closed.
    정규화된 계약을 반환.
  - CLI: `--coverage-policy PATH` 추가. `--final-decision`은 이것 없이는 실패(정책 미지정도 fail-closed),
    `--coverage-policy`만 단독으로 주면 거부. 검증된 계약은 결과 JSON `coverage_policy`에 기록.
- `revision/scripts/mvp/mvp_junction_binding.py`
  - `bind(runs, policy=None)`: 정책 없으면 종전대로 단일런 전용. 정책이 오면 검증 후 **같은 소스
    안에서만** 풀링, 소스 혼합 거부, 구조권한 없는 소스 거부.
  - `pool(runs)` 신설: 접합별 read count 합산 후 frozen 임계값으로 `supported` 재도출.
  - `receipt(runs, rows, policy=None)`: `coverage_policy`가 더 이상 `"PENDING"` 문자열이 아니라
    검증된 계약(또는 `NONE_SUPPLIED_SINGLE_RUN_ONLY`).
  - 모듈 docstring의 "No technology/run coverage policy is author-approved" 문단 교체.
- `revision/scripts/mvp/SCORING.md`: 해당 문단을 A47 계약 기술로 교체.

### 테스트

`tests/test_mvp_scoring.py` +6, `tests/test_mvp_junction_binding.py` +3 (행동 기술 이름, AAA).
계약 수락/정규화, HiFi를 구조소스로 선언 거부, frozen 임계값 모순 거부, unavailable 누락 거부,
교차소스 풀링 거부, 필드 누락 거부; 소스 내 풀링이 단독으로는 임계 미달인 접합(2+2=4≥3)을
지지로 만드는지, 교차소스 풀링 거부, receipt가 검증된 정책을 기록하는지.
기존 두 단언 갱신(`'coverage policy pending'` → `'coverage policy required'`,
receipt `PENDING` → `NONE_SUPPLIED_SINGLE_RUN_ONLY`).

**실행 결과** (`PYTHONPATH=src /home/wyim/miniconda3/envs/helixer/bin/python3 -m pytest …`, Python 3.11 —
기본 `python3`는 3.8이라 이 저장소(>=3.10)를 수집조차 못 한다):

```
tests/test_mvp_scoring.py tests/test_mvp_junction_binding.py
  -> 1 failed, 47 passed
```

유일한 실패 `test_schema_mirrors_real_training_metrics`는 **기존 실패**다:
내 변경을 stash한 상태에서도 같은 `ModuleNotFoundError: No module named 'torch'`로 실패한다(확인함).

전체 수집 기준으로 이 환경에는 `torch`/`duckdb`가 없어 다수 모듈이 수집 실패한다. 전체 실행의
실패 목록 차분은 **stash가 이 worktree의 기존 미커밋 작업까지 되돌려 오염**되므로 그대로 읽으면 안 된다
(그 때문에 `test_mvp_seed123_eval` 블록이 baseline에서만 실패하고, `mvp_junction_binding.py`는
untracked라 stash되지 않아 새 테스트가 baseline에서 실패한 것처럼 보였다). 실제로 확인한 사실:
- 내가 만진 4파일에 의존하는 스위트는 전부 통과(위 47 passed).
- `test_mvp_final_source_integration.py::test_reviewed_a_implementation_bytes_unchanged`는
  `revision/scripts/mvp` 전체를 커밋 `772c38c` 바이트에 고정하는 테스트로, **내가 건드린 적 없는**
  `make_launch_readiness.py`(기존 미커밋 수정)에서 먼저 실패한다 — 즉 내 개정 이전부터 깨져 있었다.
  단 내 `mvp_score.py`/`mvp_junction_binding.py` 수정도 같은 핀을 벗어나므로, 이 핀은 개정 승인 시
  다시 떠야 한다.
- 나머지 실패는 전부 `torch`/`duckdb` 부재.

## 하지 않은 것

커밋/푸시/병합, 원격 쓰기, 다운로드, Slurm 제출·취소 없음. GPU 잡 3187760, B5 체인 3149358-63 무접촉.
M-CCS26 role 재지정과 `--coverage-policy` 계약 JSON 실파일 발행은 저자 결정 대기.

---

# 2차 진행 (같은 날, 저자 "진행해줘")

## ⑥ M-CCS26 (PRJNA1470126) — 같은 규칙 적용

보류했던 결정을 저자 지시로 실행했다. `DATASET_ROLES.longread_v1.tsv` 3행:
`b1_validation_only` → **`b1_sequence_evidence_only`**, basis/note를 A47 실측으로 교체.
A47 표의 M-CCS26 행과 quarantine D3 ④항도 "보류"에서 "재지정 완료"로 갱신했고,
A47 "What this changes"에 **§3.1 maize** 항을 신설했다:

> maize tier-1 PacBio 구조증거는 이제 **M-HQ18 + M-FLNC**가 담당한다(둘 다 표방한 산물을 실제로 공개).
> **M-KNX(PRJNA1290227)**는 A15의 hybrid-stratum 처리 그대로이며 이번에 조사하지 않았다.
> A46 결정2(Z. mays 입력 M-EST+M-HQ18 / 채점 M-FLNC+root-tip ONT)는 M-CCS26을 쓰지 않으므로 영향 없음.

매니페스트 role 집계(216행): `c2_training_eligible` 129 / `excluded` 55 /
`b1_validation_only` 22 / **`b1_sequence_evidence_only` 9**(A-CCS23 6 + M-CCS26 3). 전 행 14열 유지.

## ⑦ coverage policy 계약 JSON 발행 + CLI 실동작 검증

`revision/scripts/mvp/coverage_policy_a47.json` (worktree, untracked):

```json
{"policy_id":"B1-A47-v1.37/MVP-v1","structural_sources":["ONT"],"thresholds":{"ONT":3},
 "unavailable_sources":["HiFi"],"pooling":"within_source","per_run_retained":true}
```

테스트 2건 추가:
- `test_shipped_coverage_policy_file_is_the_approved_contract` — 배포 파일이 검증기를 통과.
- `test_real_score_cli_full_frozen_panel_missing_structural` 확장 — 실제 CLI를 frozen 50패널
  (`revision/protocols/b6_panels_v1.tsv`)로 돌려 `--final-decision --coverage-policy <배포파일>`이
  **rc=0**으로 통과하고 결과 JSON의 `coverage_policy`에 `structural_sources=['ONT']`,
  `unavailable_sources=['HiFi']`가 기록되는지 확인. 정책 없는 `--final-decision`은 여전히 실패하고
  출력 파일을 만들지 않는다(기존 단언 유지).

### 실행 결과

```
tests/test_mvp_scoring.py tests/test_mvp_junction_binding.py   -> 1 failed, 48 passed
+ tests/test_mvp_score_producer.py                             -> 7 failed, 61 passed
```
실패 8건은 **전부 `ModuleNotFoundError: No module named 'torch'`** (score_producer 7건은 모두
`tests/test_mvp_finalize.py:15`에서 유래). 내 변경과 무관한 환경 결손이며, stash 상태에서도 동일하다.

## 남은 것 (실행 안 함, 승인 필요)

- **바이트 재핀**: `test_reviewed_a_implementation_bytes_unchanged`가 `revision/scripts/mvp` 전체를
  커밋 `772c38c`에 고정한다. 이미 `make_launch_readiness.py`(이전 세션 미커밋 수정)에서 깨져 있었고,
  이제 `mvp_score.py`·`mvp_junction_binding.py`·`SCORING.md`·`coverage_policy_a47.json`도 벗어난다.
  재핀은 **리뷰된 구현을 커밋하는 행위**라 저자 지시 없이는 하지 않는다.
- **M-KNX(PRJNA1290227) 산물 확인**: A15가 "Revio Kinnex FLNC"라고 적었으나 ENA 미등재(SRA toolkit)
  세트여서 이번 ENA 조회에 포함되지 않았다. 같은 규칙을 적용하려면 별도 확인이 필요하다.
- main 병합/커밋/푸시, Slurm 제출 없음. GPU 잡 3187760, B5 체인 3149358-63 무접촉.

---

# 3차: 정정 (같은 날) — ENA 파일명은 산물이 아니다

저자 지시 "1번 재핀, 2번 사용하지만"에서 **2번(M-KNX)을 확인하다가 이 세션 자신의 오판을 발견**했다.

## 무엇이 틀렸나

ENA fastq 경로의 `SRR*_subreads.fastq.gz` 접미사를 "아카이브 산물이 subreads"라는 증거로 읽었다.
**틀렸다.** ENA는 NCBI 경유(brokered) PacBio 런을 예치 내용과 무관하게 일반 `_subreads` 이름으로
서비스한다. 바로 그런 런들이 `submitted_ftp`/`submitted_format`이 비어 있어서 파일 이름만 보이는데,
그 이름은 증거가 아니다.

## 실제 판정 근거 — 리드 센서스 (ENA filereport + NCBI SRA runinfo)

| 세트 | 런 | 리드 | 평균길이 | A15 기술 | 판정 |
|---|---:|---:|---:|---|---|
| A-CCS23 WT (SRR22719002-07) | 6 | 2,604,608 | 2,849 nt | ~2.9 M CCS-level | **CCS-level** (분자당 1리드) |
| A-CCS23 전체 | 36 | 15,944,316 | 2,948 nt | — | 동일 |
| M-CCS26 PRJNA1470126 | 3 | 2,260,021 | 2,408 nt | 2.26 M | **CCS**, 정확 일치 |
| M-KNX PRJNA1290227 | 4 | 57,320,590 | 1,723 nt | 57 M | **Revio HiFi(Kinnex)**. Revio는 HiFi만 출력 |
| A-HiFi ERR13994458 | 1 | 9,093,086 | 1,822 nt | — | **subreads** — 제출자가 `Col_isoseq.subreads.bam` / `submitted_format=BAM` / 32.96 GB로 명시 |

한 SMRT 셀에서 subreads라면 리드 수가 분자 수의 10~30배여야 한다. 35만~62만 리드/셀은 CCS 규모다.
M-KNX 런 메타데이터: `library_selection=PolyA`, `TRANSCRIPTOMIC`, 라이브러리명 `*_PacBio_kinnex_RNAseq`,
platform PACBIO_SMRT / Revio. **A15의 "not on ENA → SRA toolkit"도 오기**다(ENA에 있다).

## 되돌린 것

- `DATASET_ROLES.longread_v1.tsv`: A-CCS23 6행 + M-CCS26 3행을 **`b1_validation_only`로 복원**,
  note에 측정값 기록. role 집계: `c2_training_eligible` 129 / `excluded` 55 / `b1_validation_only` 31
  (b1_sequence_evidence_only 0). 전 행 14열.
- `PROTOCOL_B1_frozen_v1.md`: A47 가용성 표의 A-CCS23·M-CCS26 행 교체, **M-KNX 행 신설**,
  §3.1 maize 항 삭제, A15 포인터 배너를 "A15 stands"로 재작성, change-log 문구 수정,
  **`### A47.1` 정오표 신설**(오판 내용·리드 센서스 표·남은 저자 결정).
- `quarantine.md` D3 ③④ 교체 + ⑤ 신설. "이름만 ccs"였던 `Zmays_B73_ccs_*`는 **이름이 맞았다**로 정정.
- `B6_MVP_PILOT_v1.md` §2 배너에 "A-CCS23은 drop-in 대체가 아니다(CCS≠FLNC, A47.1)" 한 줄 추가.
- `FLNC_POLICY_20260921.md` 머리에 ⚠️ 격리 배너(가용성 표 무효, 정본은 A47.1).

## 살아남은 것

- **규칙 1~3은 그대로**: 로컬 FLNC 미생산(`ccs`/`lima`/`refine` 금지), 다운로드 가능할 때만 FLNC 자격,
  그 외 EST급 서열증거.
- **A-HiFi ERR13994458만 재지정** — 근거가 ENA 생성 경로가 아니라 제출자 파일명·형식 + subreads급 리드 수다.
- A-ONT1이 A. thaliana MVP의 유일한 구조 소스라는 결론, 따라서 **A-HiFi 팔에 기대는 예정 STOP은
  INCONCLUSIVE로 하향**. coverage policy `structural_sources:["ONT"]`도 그대로 유효.
- ⑤ 계약기반 `require_coverage_policy` 구현과 ⑦ 계약 JSON·테스트는 영향 없음.

## 남은 저자 결정 (새로 생김)

**FLNC 처리를 거치지 않은 CCS가 PacBio 구조증거 팔을 맡을 수 있는가.**
A-CCS23·M-CCS26은 CCS이지 FLNC가 아니다(lima/refine 산물 미공개, 생산은 규칙 1로 금지).
MVP 구조 정의는 `≥2 A-HiFi`로 A-HiFi를 특정하고 §4 PacBio 명령은 "FLNC / HQ isoforms / HiFi FLNC"다.
A. thaliana에서 잃은 A-HiFi 자리를 A-CCS23이 대신할지, maize는 어떨지는 **스펙 변경**이며 미결이다.
결정 전까지 구조계약은 ONT 단독을 유지한다.

## 바이트 재핀(저자 지시 1번) — 보류한 이유

재핀은 `revision/scripts/mvp` 전체를 새 커밋에 고정하는 행위인데, 그 안에
`coverage_policy_a47.json`(`structural_sources:["ONT"]`)이 들어간다. 위 미결 결정이 ONT 단독을
바꾸면 그 파일이 바로 바뀌므로, **지금 재핀하면 곧 다시 깨진다.** 결정 후에 한 번만 재핀하는 것이 맞다.
현재 핀 `772c38c`가 이미 깨져 있다는 사실 자체는 변함없다(내 변경 이전부터, `make_launch_readiness.py`).

---

# 4차: 저자 결정 2건 반영 + 바이트 재핀 실행

## 결정 1 — CCS는 구조증거 팔을 맡지 못한다 (ONT 단독 확정)

저자 결정(2026-09-21): **FLNC 처리를 거치지 않은 CCS는 PacBio 구조증거 팔을 맡지 못한다.**
A-CCS23·M-CCS26은 CCS이지 FLNC가 아니고(lima/refine 산물 미공개, 생산은 규칙1로 금지) 규칙2를
문자 그대로 적용한다. **A15 검증역할과 `b1_validation_only` 행은 유지**하되 접합·인트론체인·TSS/TES·
UTR 경계·GSF 라벨을 지지하지 않고 `59_evidence_support.py`의 `THRESHOLDS`에 계상되지 않는다.
M-KNX는 A15 hybrid-stratum 유지, 승격 없음.

확정된 귀결:
- A. thaliana MVP 구조계약 **ONT 단독**. `coverage_policy_a47.json`의
  `structural_sources:["ONT"]` / `unavailable_sources:["HiFi"]`가 그대로 정본이며,
  **A-CCS23은 A-HiFi 대체가 아니다**.
- `B6_MVP_PILOT_v1.md` §2의 `≥2 A-HiFi` 항은 계속 미충족이고, CCS를 넣는 스펙 변경은 하지 않는다.
- 안전장치대로 **A-HiFi 팔에 기대는 예정 STOP은 INCONCLUSIVE**(GO 아님).
- maize 실질 무영향(A46 결정2는 M-FLNC+root-tip ONT로 채점, M-CCS26·M-KNX 미사용).

반영 위치: `PROTOCOL_B1_frozen_v1.md` A47.1의 "Open question" 문단을 "Author decision … closing the
question"으로 교체, `quarantine.md` D3 ⑤를 "저자 결정 — 종결"로 교체, TSV 9행 note에 제한 문구 추가
(role은 `b1_validation_only` 유지, 전 행 14열).

## 결정 2 — 바이트 재핀 실행

worktree `/tmp/mvp-seed123-integration-20260917`, 브랜치 `mvp-seed123-integration-20260917`.
**커밋 2개, 푸시·병합 없음.**

- `eff505f` `chore: snapshot the MVP implementation so the byte pin has a commit to hold`
  — `src/transgenic`, `train/train_HyenaTransgenic_MVP.py`, `configs`, `revision/scripts/mvp` +
  이를 검증하는 `tests/`를 커밋. 이전 핀 `772c38c`가 이미 깨져 있던 이유를 본문에 기록:
  `make_launch_readiness.py` 외 18개 파일이 미커밋이었고, 5개 모듈
  (`mvp_junction_binding.py`, `mvp_panel_compatibility.py`, `mvp_verify_pinned_panel.py`,
  `mvp_profiles.py`, `coverage_policy_a47.json`)은 untracked라 핀에 아예 보이지 않았다.
  **본문에 "이 바이트들은 전체로서 독립검토를 거치지 않았다. 핀은 트리 내용을 기록할 뿐 검토를
  증명하지 않는다"를 명시**했다.
- `5febce9` `test: re-pin the reviewed-implementation byte check to eff505f`
  — 테스트의 `772c38c` 2곳을 `eff505f`로 교체하고 같은 취지의 주석 추가.

작성자 확인: 두 커밋 모두 author/committer **Won <wyim@unr.edu>**,
co-author·Claude·session·generated 류 트레일러 **없음**.
`tasknotes/` 7개 디렉터리는 작업메모라 커밋하지 않고 그대로 뒀다.

### 검증

```
tests/test_mvp_final_source_integration.py::test_reviewed_a_implementation_bytes_unchanged -> 1 passed
전체:  164 failed, 794 passed, 12 skipped, 23 errors
```
재핀 직전 실행과 실패목록을 `comm`으로 대조한 결과 **차이는 단 하나 — 바이트핀 테스트가 실패에서
사라진 것뿐이고, 새로 생긴 실패는 0건**이다. 남은 실패는 전부 이 환경의 `torch`/`duckdb`/`accelerate`
부재(import 단계)이며, 그 외 원인을 가진 15건(`test_longread_fetch`, `submit_chain`, gff
standardize, run_tool 파이프라인)도 **내 변경 이전 실행에서 이미 같은 목록에 있던 기존 실패**다.
