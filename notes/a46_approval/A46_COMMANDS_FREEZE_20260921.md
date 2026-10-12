# A46 commands 동결 현황 — 2026-09-21

rule 15 / rule 12 Freeze clause는 승인 시점에 **commands**가 동결돼 있기를 요구한다.
이 문서는 그 목록이고, **정정 하나를 포함한다.**

## ⚠️ 정정 — 제가 앞서 "둘 다 기록 작업"이라고 한 것은 절반만 맞았다

- OrthoDB/Swiss-Prot 해시: **순수 기록 작업이 맞았고, 아래에서 끝냈다.**
- commands: **절반은 기록 작업이지만 절반은 아니다.** C0 파이프라인과 float16 트랙 래스터는
  **명령을 적어둘 코드가 아직 없다.** 자세한 건 §C.

---

## A. 이미 동결된 명령 (✅ 승인에 그대로 넣을 수 있음)

### A-1 패널 생성 (rule 12)
`71_b6_panels.py` · `72_mvp_select_loci.py` · `72a_est_molecules_per_locus.sh`
**스크립트 sha256이 A46 본문에 이미 박혀 있고 실물과 일치**(09-21 재계산 확인).
`72a`는 `samtools 1.3.1`, `-F 0x904`, 고유 placement `(chromosome, FLAG-0x10 strand, POS, CIGAR)`로 세는 것까지 본문에 문장으로 있다.

### A-2 장독 정렬 (B1 §4, MVP §2 pass 2)
```
# ONT cDNA (orientation not proven)
minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# ONT cDNA (orientation proven >= 95%)
minimap2 -t 32 -ax splice -uf -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# PacBio  -- A47/A47.1로 이번 분석에서는 사용하지 않음
minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
```
`-uf`는 §3.3 orientation audit(A39 보정)에서 **≥ 95%**일 때만. pass 1은 후보 read 사전선별이며
채점 정렬이 아니다.

### A-3 UniVec 스크리닝 (A21)
```
blastn -task blastn -db UniVec_Core -reward 1 -penalty -5 -gapopen 3 -gapextend 3 \
       -dust yes -soft_masking true -evalue 700 -searchsp 1750000000000
```
+ `revision/scripts/61_univec_trim.py min_len=121 terminal_nt=25`.
스크리닝 자신의 receipt가 11개 EST 세트 전부에 이 파라미터와 `blastn: 2.5.0+`를 기록.
→ `PROVENANCE.univec_screen_v1.json`

### A-4 단백질 DB 빌드 (route B 입력)
```
diamond makedb --in swissprot_viridiplantae_reviewed.fa.gz -d swissprot_viridiplantae_reviewed --threads 16
diamond makedb --in odb12_Viridiplantae.filtered.fa.gz     -d odb12_viridiplantae_filtered     --threads 16
```
빌더 `evidence/protein/diamond/build_dbs.sh`, diamond 2.1.24(`evidencemodeler/bin/diamond`).
Swiss-Prot은 UniProt REST `(reviewed:true) AND (taxonomy_id:33090)`.
→ `PROVENANCE.protein_db_v1.json`

### A-5 route B 정렬 (diamond blastx) — 계약 확정
```
diamond blastx --frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25 \
               --max-hsps 1 --no-parse-seqids --threads <N> --tmpdir <private scratch> \
               --outfmt 6 qseqid qlen sseqid slen qstart qend sstart send length evalue \
                          bitscore qframe btop
```
18개 species×source 유닛의 command report가 **전부 같은 pin**(diamond sha256 `0fb259d7…`,
2.1.24, 16 threads, 위 플래그, 13-field BTOP 종료 계약)을 기록하고 있다.
⚠️ **미정 1건**: "HSP ≥ 30 aa"의 운영 정의. `--frameshift` 하에서 raw13 col 9 semantics가
비표준(alignment length vs ungapped aa vs per-frame sub-span)이라 RICH17 fixture로 고정해야 한다.
이건 **임계값 해석이지 명령이 아니다.**

### A-6 route A 정렬 (miniprot, phase 교차검증)
```
miniprot -t 32 -d <index> <REF>                          # 인덱스
miniprot -t 32 --gff --outs=0.97 -N 20 <index> <proteins> # 정렬
```
`--gff`는 필수(acceptance filter가 per-alignment Identity/Frameshift/intron 좌표를 요구).
A41의 6-chunk 분할과 id-renumbering merge 규칙이 적용된다.

### A-7 평가
GFFCompare **v0.12.6** exact intron-chain match(단일엑손은 exact CDS), sha256 `a440e153…`.
참조는 `24_restrict_reference_to_test_genes.py` 방식으로 패널 유전자에 한정.

---

## B. 해시 — ✅ 완료

| artefact | sha256 | 검증 |
|---|---|---|
| `odb12_Viridiplantae.filtered.fa.gz` (3,178,195,033 B, 12,115,085 seq) | `fa0795b6…` | 빌드 자신의 `DB.sha256`과 **일치** |
| `swissprot_viridiplantae_reviewed.fa.gz` (10,266,723 B, 42,106 seq) | `8d66a18e…` | 사이드카와 **일치** |
| `odb12_viridiplantae_filtered.dmnd` (5,751,160,176 B) | `3aac3d44…` | — |
| `swissprot_viridiplantae_reviewed.dmnd` (21,590,888 B) | `b444cb29…` | — |

A19.1 누출 필터 수치도 확보: 12,204,762 → 12,115,085 seq, `exclude_taxid [3702, 4081, 4577]`
(= A. thaliana / S. lycopersicum / Z. mays), 학습 9종의 test-orthogroup 단백질 exact-match 제거
(Gmax 21,119 / Osativa 18,280 / Vvinifera 13,419 / Sitalica 10,805 / Ptrichocarpa 9,227 /
Bdistachyon 7,892 / Sbicolor 6,285 / Ppatens 2,650).

---

## C. ❌ 적을 명령이 아직 없는 것 — 그리고 그게 왜 순환인가

> ⚠️ **정정 2026-09-22 (독립검토 finding 5).** 이 절은 미비 항목을 **C0와 트랙 2건**으로
> 신고했으나 **틀렸다.** rule 15의 WP1 목록은 "C0 pipeline, route-B phase, leakage exclusion,
> **corpus rebuild**, tracks, **trainer**, pilot, full runs, evaluation"이다.
> **rule 5 코퍼스 재빌드와 트레이너 진입점도 어느 문서에도 명세되지 않았다** —
> 즉 마스크 수리 빌드(rule 5의 실체 전부)가 미명세인 채로 승인이 날 뻔했다.
> 인벤토리가 자기 완성도를 과소보고한 것이며, 개별 갭보다 나쁘다.
> 두 항목의 명세는 v2 SPEC-1b에 있다.

**C0 전체 파이프라인 빌드**와 **float16 트랙 래스터**는 명령을 기록하지 못한다.
**코드가 없기 때문이다.**

- main의 `src/transgenic/evidence/`는 **비어 있다.** `mvp_tracks.py`는 worktree와 notes 스냅샷에만 있고 MVP 범위다.
- `mvp_build_c0lite.py`도 MVP 범위의 **C0-lite**이지 A46 rule 2가 말하는 full C0가 아니다.

### 순환 구조

- rule 15는 **C0 파이프라인과 트랙을 WP1에 넣고, WP1은 승인 후에만 시작**하라고 한다.
- 동시에 rule 15/12는 **승인 시점에 commands가 동결**돼 있기를 요구한다.

rule 15 자신이 출구를 갖고 있다: *"C0 tables, tracks, evidence DBs and run identities generated
after approval cannot have hashes at approval; each must instead have its hash recorded and verified
before first use."* → **산출물은 승인 후, 그러나 그것을 만드는 명령은 승인 전에 명세로 적혀야 한다.**

즉 남은 일은 "이미 있는 걸 받아적기"가 아니라 **명세를 쓰는 일**이다:
1. **full C0 빌드 명령/입출력 스키마** — rule 2가 `IMPLEMENTATION_ORDER_B5_C0_C2_v1.md` line 34 +
   `est_protein_hsp` 테이블을 지목하고 있으므로 출발점은 있다.
2. **float16 트랙 래스터 명령** — rule 6(트랙)과 rule 8(blake2b thinning, `aug_seed`) 규칙은 이미
   동결돼 있으니, 남은 건 그 규칙을 실행하는 진입점·인자·출력 경로를 문장으로 고정하는 것.

둘 다 **실행이 아니라 명세 작성**이므로 rule 15 위반이 아니다. 다만 **설계 판단이 들어가므로
순수 기록 작업이 아니고**, 초안 후 독립검토를 받는 편이 맞다.

---

## 요약

| 항목 | 상태 |
|---|---|
| A-1 패널 · A-2 장독정렬 · A-3 UniVec · A-4 DB빌드 · A-5 route B · A-6 route A · A-7 평가 | ✅ 동결됨 |
| B 단백질 DB 해시 | ✅ 완료·교차검증 |
| C-1 full C0 빌드 명령 | ✅ v2 SPEC-1 |
| C-2 float16 트랙 래스터 명령 | ✅ v2 SPEC-2 (K는 D1/D2 대기) |
| C-3 **rule 5 코퍼스 재빌드** | ✅ v2 SPEC-1b — **이 절이 누락했던 항목** |
| C-4 **트레이너 진입점** | ⚠️ v2 SPEC-1b, K 확정 후 문장 확정 |
| A-5의 "HSP ≥ 30 aa" 운영 정의 | ⚠️ RICH17 fixture로 고정 필요 (임계값 해석) |

범위: 읽기전용 조사 + 문서 작성. 프로토콜·코드 수정, 잡 제출, 커밋 없음.
