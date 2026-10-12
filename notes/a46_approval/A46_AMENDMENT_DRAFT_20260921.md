# A46 개정 (draft revision) — 2026-09-21

A46은 `PROTOCOL_B1_frozen_v1.md:957`에서 **DRAFT pending explicit author approval** 상태다.
따라서 승인 전 수정이 정상 절차이고, 이 문서는 **무엇을 어떻게 바꿨는지의 검토 기록**이다.

**왜 지금 고쳤나:** A46 본문이 같은 날 내린 A47 / A47.1 / ONT 결정과 어긋나 있었다. 그대로
승인하면 사실과 다른 명세가 동결되고, 고치려면 rule 12의 *"No other number, command or rule of
A46 may be added or changed later except by a further amendment"* 때문에 **A48이 또 필요**해진다.

**불가침 원칙:** rule 17의 `B6_MVP_PILOT_v1.md` §5.3 **verbatim 인용블록은 한 글자도 건드리지
않았다.** 5개 변경 적용 후 정규화 대조로 **동일함을 재확인**했다.

---

## C1 — rule 10, A. thaliana 채점 세트

**before**
> A. thaliana strict held-out loci: input = `excluded_heldout` A-EST + A-ONT2 (…); **scoring = A-ONT1 + A-HiFi**. Z. mays: decision 2.

**after**
> … scoring = **A-ONT1 alone** (amended by A47/A47.1: A-HiFi ERR13994458 publishes no FLNC product
> and none is produced here, so it carries no structural authority; A-CCS23 is CCS, not FLNC, and by
> author decision does not carry a structural arm either). Z. mays: decision 2.

**근거:** A47(A-HiFi는 subreads만 공개, 제출자가 `Col_isoseq.subreads.bam`으로 명시) +
A47.1 저자결정(FLNC 미처리 CCS는 구조팔 불가).

---

## C2 — rule 10, 독립 outcome의 생산 방법 (저자결정 (a))

**before**
> Any F1 against RefGen_V4 is labelled `reference-dependent`; the independent outcome is
> complete-chain support by the scoring set (§9 P1–P5).

**after** — 위 문장 유지 + 다음을 추가:
> **How that support is produced (A47.1, author decision 2026-09-21, option (a)):** by a
> **locus-restricted A-ONT1 alignment over the frozen final panel (1,729 loci)** — reads are
> pre-filtered to the panel windows ± 10 kb and only those reads are aligned against the full
> reference, exactly the two-pass shape of `B6_MVP_PILOT_v1.md` §2, which names itself
> *"locus-restricted alignment (not the B1 validation pass)"*. The **B1 full-run ONT validation pass
> is not executed** (author decision 2026-09-21); this panel-restricted alignment is a different
> operation and is not blocked by that decision.
> **Denominator for the independent outcome:** read-supported junctions lying inside the final-panel
> gene loci, under the frozen `THRESHOLDS` of `59_evidence_support.py` (ONT 3); the PacBio branch is
> unchanged but has no input. Per-run results are retained alongside the pooled result, and A-HiFi is
> recorded **missing per source**, never as zero support. Cost must be extrapolated from the MVP's
> measured 50-locus run, not estimated in advance.

**이 변경이 한 일 3가지:**
1. "ONT full 안 한다"와 "독립 outcome은 유지한다"가 **모순이 아님을 문서에 박았다** — 둘은 다른
   연산이고, MVP §2가 이미 그 구분을 문장으로 갖고 있다.
2. **denominators 항목을 닫았다.** 체크리스트에서 ⚠️로 남아 있던 유일한 이유가 이 선택 대기였다.
3. 비용을 **실측 외삽으로 강제**했다. 이 프로젝트엔 계획치를 측정치로 오인한 전례가 있다
   (16CPU/32GiB/4h, 352 GPU-h).

---

## C3 — rule 17, 손으로 만든 split 기술

**before** → `(rule 10 applied by hand: A-EST input, A-ONT1/A-HiFi scoring)`
**after** → `(rule 10 applied by hand: A-EST input, **A-ONT1 scoring**; A47 removed A-HiFi's structural authority)`

---

## C4 — rule 17, 인용블록 **뒤에** 적용 노트 추가 (인용문은 불변)

인용문 안의 "locally aligned A-ONT1/A-HiFi reads"는 그대로 두고, 그 뒤에:

> **A47 application note (the quoted rule above is unchanged and is not re-quoted).** Its clause
> "locally aligned A-ONT1/A-HiFi reads" is satisfied by **A-ONT1 alone**; A-HiFi contributes nothing
> and is reported missing per source. Because the A-HiFi arm is permanently unavailable, the quoted
> availability safeguard binds: **a prospective STOP resting on that arm is downgraded to
> INCONCLUSIVE, never to GO.** The GO criteria and §5.4 are unaffected.

`B6_MVP_PILOT_v1.md` §5.3 뒤에 넣은 노트와 같은 취지이며, 두 문서가 같은 말을 하게 된다.

---

## C5 — rule 16 Claims, 종별 증거 불균형 선언

**after** (추가분)
> **Evidence coverage is not uniform across the nine training species and no claim assumes it is:**
> *S. bicolor* and *B. distachyon* carry **no long-read evidence at all** — *S. bicolor*'s PacBio runs
> were path-isolated as subreads-only (issue #60, author decision 2026-09-03) and *B. distachyon*
> never had any — so every cross-species statement is read against the per-species evidence table,
> not against an assumed common baseline.

**왜:** 결함이 아니라 데이터 현실이다. 리뷰어가 짚기 전에 본문에 답이 있어야 한다.

---

## 무결성 검증 (실행함)

- rule 17의 §5.3 인용블록 ↔ `B6_MVP_PILOT_v1.md` §5.3 블록 인용:
  Markdown 강조·줄바꿈 정규화 후 **동일** (1,965자).
- A46 status 블록에 draft revision 5건을 요약 기록. **승인 상태는 변하지 않았다** — 여전히
  explicit author approval 대기이고 rule 15가 WP1을 막는다.
- `PROTOCOL_B1_frozen_v1.md` 1,044 → 1,080행.

---

## 이 개정으로 체크리스트가 어떻게 바뀌나

| 범주 | 전 | 후 |
|---|---|---|
| A46 본문 정합성 | ❌ rule 10이 A47/ONT 결정과 충돌 | ✅ **해소** |
| Denominators | ⚠️ (a)/(b)/(c) 선택 대기 | ✅ **확정** (final-panel gene loci 내 read-supported junction, ONT 3) |
| Seeds · Thresholds · Panels · N_pilot · Tool PROVENANCE | ✅ | ✅ |
| Commands | ⚠️ 절반 | ⚠️ **절반 유지** |
| OrthoDB / Swiss-Prot 해시 | ⚠️ | ⚠️ **유지** |

## 승인까지 남은 2건

1. **Commands** — C0 파이프라인 빌드 명령, float16 트랙 래스터 명령, route B `diamond blastx`
   전체 인자, route A miniprot phase 추출 명령. (패널 생성·§4 정렬·GFFCompare는 이미 동결됨)
2. **OrthoDB v12 Viridiplantae filtered stage2 / Swiss-Prot reviewed Viridiplantae 해시** —
   파일은 지목돼 있으나 sha256 미기록. rule 15의 "승인 시점에 존재하는 artefact 해시 검증" 대상이다.

둘 다 **설계 결정이 아니라 기록 작업**이다. 새 저자 결정을 요구하지 않는다.

---

범위: A46(DRAFT) 본문 수정 + 이 기록. frozen 개정(A1–A45, A47, A47.1) 무수정, 인용블록 무수정,
잡 제출·원격 쓰기·커밋 없음.
