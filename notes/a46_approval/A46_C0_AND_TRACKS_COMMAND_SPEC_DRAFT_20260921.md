# A46 승인용 명령 명세 초안 — full C0 빌드 / float16 트랙 래스터

> 🧊 **폐기 — 독립검토 REJECT(변경 17건). 정본은 [A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md](A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md).**
> 이 문서는 무엇이 왜 틀렸는지의 기록으로만 보존한다. 여기 적힌 명령·K·RC 규칙을 실행 근거로
> 삼지 말 것. 특히 RC 규칙은 자기가 인용한 테스트의 반대를 주장하고 있었다.

**상태: DRAFT — 독립검토 1차 수령, CRITICAL 1 + HIGH 3 확인. K 확정 철회.** 2026-09-21.

> 검토 보고서가 중간에서 잘려 나머지를 요청했고, Kimi challenge는 진행 중이다.
> 아래 본문은 그 발견들을 반영해 수정 중이며, **§4-1의 K = 31은 무효다.**

## 0. 이 문서의 지위

rule 15는 C0 파이프라인·트랙을 WP1에 넣고 WP1을 승인 후로 미루면서, 동시에 승인 시점에
**commands가 동결**돼 있기를 요구한다. rule 15 자신의 문장이 출구다:

> "C0 tables, tracks, evidence DBs and run identities generated after approval cannot have hashes at
> approval; each must instead have its hash recorded and verified before first use."

**산출물은 승인 후, 명령은 승인 전 명세.** 이 문서는 그 명세이며 **실행이 아니다.**
코드도 작성하지 않았다. 확정되면 A46 rule 2/6 옆에 명령으로 들어간다.

계약 본문은 이미 동결돼 있다(rule 2 스키마 9테이블 + `est_protein_hsp`, rule 3 분자단위,
rule 4 누출제외·감사정렬·≥99.9%, rule 5 마스크·단일 DuckDB, rule 6 채널 정의·clip·RC).
**여기서 새로 정하는 것은 진입점·인자·경로·순서·실패조건뿐이다.**

---

## SPEC-1 — full C0 빌더

### 진입점
```
revision/scripts/evidence/b6_build_c0.py
revision/scripts/evidence/b6_audit_align.sbatch      # rule 4(b) 감사 정렬
```
기존 관례를 따른다: 증거 파이프라인은 `revision/scripts/evidence/`, 번호 스크립트
(`71_`, `72_`)는 패널 전용이므로 번호를 붙이지 않는다.

### CLI
```
b6_build_c0.py
  --roles        data/manifests/DATASET_ROLES.longread_v1.tsv
  --est-roles    data/manifests/DATASET_ROLES.est_v1.tsv
  --species      <species_id>            # 1회 호출 = 1종. 병렬은 배열 잡으로.
  --source       est|ont|pacbio|protein  # 1회 호출 = 1소스.
  --align-root   <evidence/est_align|evidence/longread_align_v1|…>
  --audit-root   <b6_audit_align 산출 디렉터리>
  --splits       data/splits/b5_orthogroup_split_v1.tsv
  --flags        <A22 geenuff_flags.tsv> <A30 swissprot_flags.tsv>
  --overlap      <A33 overlap components + 시드된 flank 값>
  --hsp          <est_protein_hsp 입력, route B 산출>        # --source protein 일 때만
  --out          <evidence_db 경로>                          # 배타적 생성, 덮어쓰기 금지
  --qc           <QC JSON 경로>
```

### 입력 게이트 (fail closed)
1. 모든 런/세트는 `DATASET_ROLES`에 role 행이 있어야 한다. 없으면 **거부**(A18.3 기존 규칙).
2. `role` 값이 `c2_training_eligible`이 아닌 행은 training-side 테이블에 **들어가지 못한다**.
   `b1_validation_only` 행은 `validation_only=true`로만 적재된다.
3. 모든 입력 파일의 sha256/md5를 매니페스트 기록값과 대조. 불일치는 **abort**.
4. 도구 신원은 `revision/protocols/provenance/PROVENANCE.*.json`의 sha256으로 확인.

### 출력
- **evidence DB** (DuckDB) — rule 2의 9개 테이블 + route B의 `est_protein_hsp`.
  모든 행이 source · library · genotype stratum · role · provenance hash를 갖는다.
- **QC JSON** — rule 4(d) 형식: species × source별 molecules in / cause (a)(b)(c)별 제외 / retained,
  그리고 감사정렬 **exhaustiveness**.
- **PROVENANCE.txt** — 날짜, 도구 sha256, 입력 해시, 정확한 확장 명령, 산출물 sha256.

### 순서 (강제)
```
1) b6_audit_align.sbatch   : rule 4(b) 감사 정렬. 각 소스의 frozen 명령에서
                             --secondary=no → --secondary=yes -N 50 -p 0.5 로만 교체.
                             EST: minimap2 -t 32 -ax splice:hq -G 200000 --cs=long
                             ONT/PacBio: A14/§3의 자기 명령 그대로.
2) exhaustiveness 계산     : placements < N(=50)인 분자 비율. 종별 보고.
                             < 99.9 %  →  N을 올리고 1)부터 재실행. 트랙 빌드 금지.
3) b6_build_c0.py          : 누출제외 → 분자단위 병합 → 관측 적재 → QC
4) QC 게이트               : 누출 QC 카운트가 0이 아니면 run abort (rule 13)
                             A. thaliana: held-out loci 겹치는 분자 수 == 제외 수,
                             그 loci의 retained == 0 이어야 한다.
```

### 하지 않는 것
참조 유래 feature 금지(주석 기반 junction snapping, 참조 phase). 라벨 생성 금지 —
감사 정렬은 **오직 제외 판정에만** 쓰고 라벨에 절대 쓰지 않는다(rule 4(b) 명문).
같은 종 단백질 사용 금지(A19).

---

## SPEC-2 — float16 트랙 래스터

### 진입점
```
src/transgenic/evidence/b6_tracks.py          # 라이브러리 (rasterise)
revision/scripts/evidence/b6_build_tracks.py  # 물질화 CLI (dose-response 매니페스트용)
```
MVP의 `mvp_tracks.py`(worktree)를 일반화한 것이며, MVP 버전은 그대로 둔다.

### 왜 라이브러리 + CLI 두 개인가
rule 6은 *"observation→tile membership is kept **so that thinning re-rasterises**"*라고 한다.
thinning fraction이 학습 중 타일마다 바뀌므로(rule 8: p=0.5 dropout, `{0.01,0.1,0.5,1}` 균등추출)
**트랙은 학습 시점에 evidence DB에서 즉시 래스터**되어야 하고 미리 구운 배열로 고정될 수 없다.
CLI는 **dose-response(rule 13)의 고정 fraction 5개 × draw 5개**를 위해서만 물질화한다 —
거기서는 유지 분자 id 매니페스트와 그 sha256을 보관해야 하기 때문이다.

### 라이브러리 계약
```python
rasterise(con, *, species, chrom, offset, L, is_rc, sources, keep_fraction, aug_seed) -> np.ndarray  # float16 [L, K]
```
- **thinning**: `rank = int.from_bytes(blake2b(f"{aug_seed}\t{species}\t{molecule_id}".encode("utf-8"),
  digest_size=8).digest(), "big") / 2**64`, `rank < f`면 유지. `hashlib.blake2b` 고정,
  파이썬 내장 `hash()` **금지**(salted). 중첩성이 구조적으로 보장되고 반올림 없음.
- **채널 K** (rule 6, 순서 동결 필요 — §4 열린 결정 1):
  소스(EST/ONT/PacBio) × strand transcribed-block coverage `log1p(molecules)`, **clip log1p(64)** /
  소스 × strand donor·acceptor support(경계 염기) / unknown-strand coverage(미배향 EST) /
  protein CDS coverage `log1p(organisms)` / **route-B phase 6채널** {0,1,2} × {+,−} `log1p(EST molecules)` /
  소스별 availability mask.
- **RC 타일**: 좌표 역전, donor/acceptor **정체성 유지**, strand 채널 교환
  (`tests/test_gsf_rc.py:26` 규약).
- **정규화 금지**: 타일별 정규화 없음. A18.4 가중치는 입력 배수가 아니다.
- route A/B phase 불일치 염기는 **양쪽 phase 채널 모두 0**, 불일치는 QC에 카운트.

### CLI
```
b6_build_tracks.py
  --evidence-db <SPEC-1 산출>
  --corpus      b5_full_b6_v1.db
  --fractions   1,0.5,0.1,0.01,0
  --draws       0,1,2,3,4          # aug_seed_d = 20260914 + d
  --panel       revision/protocols/b6_panels_v1.tsv
  --out         <매니페스트 디렉터리>
```
출력: (locus, fraction, draw)별 **유지 분자 id 매니페스트 + 그 sha256**(rule 13 요구),
채널 스펙 해시(`track spec hash`, rule 9가 `runtime_identity.txt`에 요구).

### 검증 (승인 전 작성 가능, 실행은 승인 후)
rule 4/6/8이 이미 요구하는 것들: 음성가닥 junction 정체성 RC 후 보존 · 중첩 draw
(`keep(0.1) ⊂ keep(0.5)`) · 드롭된 타일은 전부 0 · 129 kb 타일 타이밍 · step-0에서
seed-123 `best`와 비트 동일(트랙 0일 때).

---

## 4. 열린 설계 결정 — 독립검토가 판단할 것

1. **채널 순서와 K — 🧊 확정 철회. 아래 K = 31 표는 무효이며 재검토 대기.**

   > **이 항목의 저자 결정(2026-09-21 "남겨야 되")은 내가 제시한 틀린 근거 위에서 내려졌다.**
   > 결정 자체를 번복하는 것이 아니라, **결정의 전제가 무너졌으므로 다시 물어야 한다.**

   **내가 틀린 것 (독립검토 발견, 매니페스트로 자체 재확인함):**
   - 근거로 "M-FLNC/M-HQ18이 PacBio를 공급한다"고 했으나 **M-FLNC는 채점 세트**(rule 10 결정 2)이고,
     rule 10은 *"a molecule used as input at a locus never scores that locus"*라고 못박는다.
     입력 채널의 근거가 될 수 있는 것은 **M-HQ18 하나뿐**이다.
   - **학습 가능한 PacBio 런이 0개다.** `DATASET_ROLES.longread_v1.tsv`의
     `c2_training_eligible` **129행 전부 ONT**(PromethION 56 / MinION 66 / GridION 7),
     `pacbio` 데이터셋은 **하나도 없다**. 전체 PacBio 72행은 `b1_validation_only` 21 +
     `excluded` 51(issue #60 subreads 격리)뿐이다.
   - 따라서 PacBio 몫 **8채널**(coverage 2 + donor/acceptor 4 + unknown-strand 1 + mask 1)은
     **학습 내내 0만 본다.** 처음 0이 아닌 값을 받는 시점이 **Z. mays 추론** —
     rule 16(ii)가 재는 바로 그 측정 — 이며, 그때 가중치는 한 번도 학습된 적이 없다.

   **진짜 물어야 했던 결정 (독립검토가 짚음):** PacBio 채널을 남기느냐 빼느냐가 아니라
   > **M-HQ18(`pacbio/Zmays_wang2018_PRJEB22122`, PacBio RS II, FL-cDNA)이
   > PacBio 채널 그룹으로 가는가, EST 그룹으로 가는가.**

   A47 규칙 3은 EST급 PacBio 세트를 *"may be used exactly as an EST source is used"*로 허용하는데,
   **A47은 M-HQ18의 아카이브 산물을 분류한 적이 없다.** EST 그룹으로 가면 PacBio 그룹 자체가
   불필요해지고 죽은 채널 8개가 사라지며, rule 10 결정 2의 Z. mays 입력 배분은 그대로 유지된다.

   **두 번째 결함 — rule 6의 마지막 절을 내가 통째로 빠뜨렸다.**
   rule 6은 *"counts and **genotype/QC flags are separate channels**"*로 끝난다.
   내 표에는 genotype 채널도 QC-flag 채널도 없다. `genotype_stratum`은 실제로 채워진 필드다
   (reference 90 / unknown 83 / non_reference 35 / hybrid_pooled 7).

   **결론: rule 6은 K를 결정하지 못한다.** 29/31/32 중 고르는 문제가 아니었다.
   genotype/QC 채널이 세어지지 않았고, PacBio 그룹의 존재 여부가 아직 내려지지 않은
   M-HQ18 분류 결정에 달려 있다. **A46 rule 6의 채널 목록을 먼저 고쳐야 K를 동결할 수 있다.**

2. **evidence DB를 종×소스별로 쪼갤지, 하나로 합칠지.** 위 CLI는 1호출=1종1소스(배열 잡 친화)로
   두고 병합 단계를 별도로 뒀다. rule 9는 `evidence DB sha256` **단수**를 요구하므로
   최종 병합본 하나가 존재해야 한다. 병합 진입점이 명세에 빠져 있다 — **보완 필요**.
3. **감사 정렬의 비용.** `-N 50 -p 0.5 --secondary=yes`는 frozen 정렬보다 훨씬 무겁다.
   9종 × 3소스 규모의 실측 예산이 없다. MVP에서 한 종 한 소스로 재보고 외삽할 것을 권고한다
   (계획치를 측정치로 쓴 전례: 16CPU/32GiB/4h, 352 GPU-h).
4. **`est_protein_hsp`의 "HSP ≥ 30 aa" 운영 정의.** `--frameshift 15` 하에서 raw13 col 9가
   비표준이다. RICH17 fixture로 고정해야 하며 이는 SPEC-1의 입력 계약에 들어간다.
4-a. **rule 4(b)의 제외 경로 2개를 내가 빠뜨렸다 (독립검토 HIGH).** rule 4(b)는
   *"A molecule with ≥ 50 secondary placements, **or any placement with MAPQ < 5 outside an
   excluded interval, is unauditable and excluded**"* 와 *"**plus, for every retained molecule, a
   direct interval test of its primary/supplementary placements**"* 를 요구한다. 내 SPEC-1은
   N = 50을 **exhaustiveness 보고에만** 썼고, MAPQ < 5 제외도, ≥50 placement **제외**도,
   retained 분자의 직접 구간 검사도 입력 게이트/순서 어디에도 없다. 이대로 만들면 누출 필터가
   rule 4의 제외 경로 2개를 빼먹고, rule 4(d) QC는 원인 (a)–(c)만 세므로 **그 누락을 탐지하지도
   못한다.**

4-b. **`b6_build_c0.py`에 코퍼스 입력이 없다 (독립검토 HIGH).** rule 4(a)의 제외 중 여럿이
   타일 기준으로 정의된다(train/valid 타일, edge-crossing 유전자의 `partial` span). 내 CLI에는
   코퍼스 DB 인자가 없다. — 상세는 독립검토 보고서 발견 4.

5. **트랙을 즉시 래스터한다는 판단이 맞는가.** 학습 처리량이 여기 달려 있다. 틀리면
   WP4 성능 게이트에서 드러난다.

---

## 5. rule 15 관점 자기점검

| 요구 | 이 문서 | |
|---|---|---|
| commands 동결 | 진입점·인자·경로·순서·실패조건 명시 | ⚠️ §4-2 병합 진입점 누락 |
| 실행 금지 | 코드 미작성, 실행 없음 | ✅ |
| 산출물 해시 | 첫 사용 전 기록 조건 명시 | ✅ |
| 새 값 도입 금지 | — | ❌ **K = 31은 A46에 없는 새 값이었고, 게다가 틀렸다**(§4-1 참조). rule 6의 채널 목록 자체를 고치지 않으면 어떤 K도 동결할 수 없다 |

범위: 문서 작성만. 코드·프로토콜 수정, 잡 제출, 커밋 없음.
