# A46 승인 준비 — 진행 정본 (2026-09-21 ~ 09-22)

## 한 줄

**FLNC/ONT 정책 4건이 저자 결정으로 종결됐고, 그 결과를 A46에 반영(개정 5건)한 뒤
rule 15의 동결 체크리스트를 채우는 중이다. PROVENANCE 축은 완료, commands 축은 v2 명세로
진행 중이며 저자 결정 2건(D1/D2)이 K를 막고 있다.**

## 저자 결정 (전부 반영 완료)

| 결정 | 원문 | 기록 |
|---|---|---|
| FLNC 정책 | "flnc가 다운 받을때 아니면 만들지 말고 그냥 eST 처럼 쓴다" | [FLNC_POLICY](../auth_restored_20260921/FLNC_POLICY_20260921.md) |
| CCS 구조팔 | 불가 — ONT 단독 확정 | B1 A47.1 |
| extern OOM | "그냥 무시한다" | [WAIVED](../auth_restored_20260921/EXTERN_OOM_WAIVED_20260921.md) |
| ONT full | "사용하지 않는다" | [DROPPED](../auth_restored_20260921/ONT_FULL_DROPPED_20260921.md) |
| 독립 outcome | 옵션 (a) 최종패널 1,729 loci 국소정렬 | A46 draft revision C2 |
| 재핀 | 실행 — `eff505f` / `5febce9` | worktree, 미푸시 |

## 프로토콜 상태

- **A47 (v1.37) 신설** — 로컬 FLNC 미생산 / 다운로드 가능시에만 FLNC 자격 / 그 외 EST급.
  **A-HiFi ERR13994458만 재지정**(제출자가 `Col_isoseq.subreads.bam`으로 명시).
- **A47.1 정오표** — ENA fastq의 `_subreads` 접미사는 **산물이 아니라 ENA 명명 아티팩트**.
  이 세션이 그걸 근거로 A-CCS23·M-CCS26을 subreads로 판정했다가 **철회**했다. 판정 근거는 리드 센서스.
  A-CCS23 2,604,608리드/2,849nt = CCS-level, M-CCS26 2,260,021/2,408nt = CCS(A15 2.26M 정확일치),
  M-KNX 57,320,590/1,723nt = Revio HiFi. **A15가 맞았다.**
- **A46 draft revision 5건** — rule 10 채점세트 A-ONT1 단독 / 독립 outcome 생산방법 + denominator /
  rule 17 split·적용노트 / rule 16 종별 증거 불균형 선언. **§5.3 인용블록 불변 검증 통과**(1,965자).
  A46는 **여전히 DRAFT**, rule 15가 WP1 차단 유지.

## rule 15 동결 체크리스트

| 범주 | 상태 |
|---|---|
| Seeds · Thresholds · Panels · N_pilot | ✅ |
| **Denominators** | ✅ (결정 (a)로 확정) |
| **Tool PROVENANCE** | ✅ **5파일** |
| **artefact 해시** | ✅ B5 코퍼스 md5 / SIF sha256 / OrthoDB / Swiss-Prot 전부 재검증 일치 |
| Commands | ⚠️ 7건 동결 + v2 명세 3건, **K 대기** |
| A46 본문 정합성 | ✅ |

**PROVENANCE 5파일** (`revision/protocols/provenance/`):
`cpu_tools_v1` `37caa5d9…` / `gpu_host_tools_v1` `92b8babe…` / `gpu_runtime_v1` `c084c76b…` /
`univec_screen_v1` `4fd697fe…` / `protein_db_v1` `b356d944…`

**재검증된 기록 해시**: B5 코퍼스 md5 `dda7e78c…`(47.8s) · SIF sha256 `1a5dee27b857…`(28.4s) ·
OrthoDB `fa0795b6…` · Swiss-Prot `8d66a18e…` · diamond `0fb259d7…`(3중 독립기록) — **전부 일치**.

## 🔴 최우선 — issue #60 재개 (09-22)

격리된 51 PacBio 런이 **전부 CCS**다(리드헤더 `/ccs/`, 51/51). issue #60의 "subreads-only" 판정은
**A47.1이 이미 정정한 ENA 파일명 아티팩트**에 근거했다. S. bicolor 36 · V. vinifera 9 ·
P. trichocarpa 6. **D1의 전제("학습가능 PacBio 0건")가 무너지므로 D1은 이 건 정리 전까지 보류.**
A46 rule 16 문장 정정 완료. 순서: 런별 기기 확인 → A47.1 CCS 규칙의 학습트랙 적용 여부 저자판단
→ role/basis 갱신 → D1. [정본](ISSUE60_CCS_REOPEN_20260922.md)

## ✏️ 2026-09-22 체인 메모리 반영

저자결정으로 롱리드 엑손 체인을 B6 입력(rule 6a, 체인 메모리)으로 추가 — 설계 [v2](B6_CHAIN_MEMORY_DESIGN_v2_20260922.md), A46 DRAFT 조항 적용(rule 6a·C7·(ii-c) 보조결과 등), D-C6 "A". 동결 체크리스트 ✏️ 갱신(K = 31, C7, 체인 상한 64/512 **동결**, PacBio `splice:hq` 사용), 명령 명세 v2에 **SPEC-2c**(체인 구성) 추가. rule 17 인용 불변.

## 저자 결정 D1/D2

**✅ D1 종결 (2026-09-22, 저자 "채널 유지")** — issue #60 번복(A47.3)으로 학습 PacBio 0건 전제가 소멸: S. bicolor 34·P. trichocarpa 6(원시 CCS)·V. vinifera 9(FLNC급)가 PacBio 블록에 학습 입력을 공급한다. **K = 31** (EST·ONT·PacBio × 8 [`cov+,cov−,cov?,donor+,acceptor+,donor−,acceptor−,mask`] + protein CDS 1 + phase 6)을 A46 rule 6(DRAFT)에 리터럴로 기입, §5.3 인용 불변 재확인(1,958자). 아래는 결정 전 기록.

**D1. PacBio 채널 8개 처분.** 학습 가능 PacBio **0건**(`c2_training_eligible` 129행 전부 ONT).
A14 `:597`이 RS II/Sequel I 학습 금지 + 다음 불릿이 테스트종 장독 학습 전면 금지 → M-HQ18 이중 차단.
Z. mays는 학습종 아님. ⇒ 8채널이 학습 내내 0만 보고, 0-init이면 가중치가 영원히 0이며,
**Z. mays 추론에서 PacBio 증거 기여가 문자 그대로 0** → rule 16(ii) 주장이 그 소스에 대해 공허.
(a)유지+추론전용 명시 / (b)제거+Z.mays 증거를 EST 그룹으로(A47 규칙3) / (c)학습종 실소스 확보.

**D2. ✅ 종결 (2026-09-22)** — reading (i) 분리 제약. K 불변, rule 6 개정 불요.
근거: A18.4 `:264` genotype multiplier(손실가중 경로) + A18.5 `:270` "applied as the A18.4 multiplier"
+ `qc_flags`는 `build_b5.py:153`의 `VARCHAR` 자유텍스트(열거 부재가 (i)의 증거) + 문장구조가
"A18.4 weights are not input multipliers"와 나란한 음성 제약.

**D1만 남았고 그것이 K를 결정한다.**

**M-HQ18 census (09-22 실측)**: PRJEB22122 16런 전부 PacBio RS II / FL-cDNA / Oligo-dT /
제출 FASTQ, maize 5런 파일명이 `RS_Sequel_maize_*_hq.fastq.gz`, 7만~14만 리드 / 평균 1.6~2.1 kb,
analysis 산물 0건 → **isoform 규모의 진짜 HQ isoform 산물. EST급이 아니다.**
⇒ A47 규칙 3(구조권한 부정)은 M-HQ18에 **적용되지 않으며**, A14는 이미 tier 1로 복원해 뒀다.
Kimi의 "EST coverage-only 라우팅" 권고는 전제가 무너졌다.
Kimi가 맞힌 것: **M-FLNC는 rule 10이 채점세트로 묶어둬 어떤 입력 채널로도 갈 수 없다**(D1 무관하게 강제),
그리고 **(c)가 살아 있다** — A47 규칙 1은 FLNC 생산 금지이지 다운로드 금지가 아니고,
**A15 scout은 maize·A. thaliana만 훑었고 나머지 7개 학습종은 조사된 적이 없다.**

**권고: (c) scout 먼저.** 있으면 뿌리에서 해결, 없으면 "찾아봤는데 없다"는 기록이 남아
(a)/(b) 어느 쪽이든 승인 문서가 단단해진다.

## 명령 명세

- **v1 폐기** — 독립검토 **REJECT, 변경 17건**.
- **[v2](A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md)** — 15건 반영. 주요 변경:
  `--corpus` 신설(rule 4(a) 타일 기준 제외) · `--flags`에서 A30 제거(새 **규칙** 도입이었음) ·
  rule 4(b) 제외 3종 + rule 4(c) 단계 신설 · `b6_merge_evidence_db.py`(rule 9 단수 sha256) ·
  **SPEC-1b 신설: rule 5 코퍼스 재빌드 + 트레이너** · `rasterise()`에 `libraries` ·
  **중첩 `(source,strand,donor/acceptor)` + RC 순열 `[1,0,2,5,6,3,4,7]` 명시** ·
  **phase RC 규칙**(v1은 자기가 인용한 테스트의 반대를 주장했다) ·
  **mask 의미론 2건**(fraction 0에서 mask 1 유지 / N 염기에서 mask 불변 — 안 고치면
  dose-response 왼쪽 앵커 파괴) · **즉석 DuckDB 래스터 철회**(WP4가 못 잡는 함정) ·
  **WP3 ≤100 GB 게이트 복원**(K 과대를 잡는 유일한 게이트) · step-0 기준 재작성.

## 다음

1. Kimi D1/D2 자문 수령 → 저자 확정
2. K + 채널 순서 전문을 **A46 rule 6 본문에 리터럴로 삽입**(rule 12: 개정 없이는 사후 변경 불가)
3. 트레이너 진입점 문장 확정, route B "HSP ≥ 30 aa" 운영 정의를 RICH17 fixture로 고정
4. 그 뒤 A46 최종 승인 → WP1 착수 가능

## 무접촉 유지

GPU 잡 3187760, B5 체인 3149358-63. 커밋은 worktree 2건(`eff505f`, `5febce9`)뿐이고
푸시·병합 없음. main 미커밋 상태 보존.

## ✏️ 2026-09-22 심야 — Kimi 최종 확인 (저자 "키미 확인해줘")
- 1차 ask는 15분 응답제한 초과(`ASK_KIMI_RESPONSE_TIMEOUT`, 설정파일까지 탐색), 에이전트의 동일범위 재시도는 취소.
  **발췌본(A46 조항 12개, 원 줄번호 보존, 20 KB)만 읽게 한 좁은 ask**(job `63a6a056…`)로 완료.
- 결과: 2차 검토 수정 13항목 **전부 존재·일관** 확인. 기본값 3건(α·블록 = B5 레시피 LR, 블록 dropout 0.1, Z. mays 체인 M-EST만) **모두 수용 가능**
  — α는 0 초기화라 α가 움직이기 전까지 블록 가중치에 그래디언트가 없어 파일럿에서 B6≈C7일 수 있으나 (ii-c)가 보조·서술 전용이고 rule 9가 층별 α 기록을 요구하므로 수용.
- 판정 **yes-after-edits**, 수정 1건 적용: A14 재기술의 `b6_input` 괄호에 "M-HQ18은 트랙만, 체인 메모리 불가(rule 6a)" 명시.
- rule 6a 뒷부분(12층·dropout 0.1·α weight decay 제외·B5 LR) 원문 직접 확인. rule 17 인용 동일.
- **승인 시 할 일:** decision 5의 MVP 파일 해시 재측정(line 977), 승인 기록.
