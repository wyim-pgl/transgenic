# A46 승인용 명령 명세 — v2 (2026-09-22)

**v1은 독립검토 REJECT(변경 17건).** 이 v2는 저자 결정이 필요한 **2건을 제외한 15건**을 반영한다.
v1: `A46_C0_AND_TRACKS_COMMAND_SPEC_DRAFT_20260921.md` (폐기, 기록용 보존).

## 0. 지위와 미결

> 🧊 **이 §0의 D1/D2·K 서술은 대체됨(2026-09-22):** D1 채널 유지(A47.3로 학습 PacBio 49런 복원), D2 분리 제약, **K = 31**이 A46 rule 6에 기입됐다. "학습 가능한 PacBio 0건"·"K = 31 철회"는 당시 기록. 현행은 문서 끝 "남은 미결" 표와 SPEC-2c.

rule 15는 WP1을 승인 후로 미루면서 승인 시점에 commands 동결을 요구한다. rule 15 자신의 문장이 출구다:
*"C0 tables, tracks, evidence DBs and run identities generated after approval cannot have hashes at
approval; each must instead have its hash recorded and verified before first use."*
→ **산출물은 승인 후, 명령은 승인 전 명세.** 이 문서는 명세이며 실행이 아니다. 코드 미작성.

### 🔴 저자 결정 2건 — 이것이 K를 결정하고 K가 나머지의 전제다

**D1. PacBio 채널 그룹 처분.** 학습 가능한 PacBio가 **0건**이다(`DATASET_ROLES.longread_v1.tsv`의
`c2_training_eligible` 129행 전부 ONT; PacBio 72행은 `b1_validation_only` 21 + `excluded` 51).
A14 `:597`이 RS II/Sequel I의 학습 사용을 금지하고 다음 불릿이 테스트종 장독의 학습 사용을 전면
금지하므로 M-HQ18은 이중 차단. Z. mays는 학습종이 아니다(rule 1 "Never: Z. mays").
⇒ PacBio 8채널은 학습 내내 0만 보고, 0-init이면 가중치가 영원히 0이며, Z. mays 추론에서 증거 기여가
문자 그대로 0이 된다. **어느 선택을 하든 "그 열은 학습되지 않는다"를 A46에 명시해야 한다.**
(a) 유지 + 추론 전용 명시 / (b) 제거 + Z. mays 증거를 EST 그룹으로(A47 규칙 3) / (c) 학습종 실소스 확보.

**D2. ✅ 종결 — reading (i). 아래 D2 절 참조.** ~~rule 6 마지막 절의 독법.~~ *"counts and genotype/QC flags are separate channels"* 가
(i) 분리 **제약**(K 불변)인가 (ii) **일곱째 그룹**(genotype + QC-flag 채널 실재, 개수 확정 필요)인가.
`genotype_stratum`은 4값으로 실제 채워져 있다(reference 90 / unknown 83 / non_reference 35 /
hybrid_pooled 7). QC flags의 열거는 저장소에서 찾지 못했다.

**이 둘이 정해지기 전에는 K를 동결할 수 없다.** v1의 K = 31은 철회됐다.


## D2 — 종결 (reading (i), K 불변, rule 6 개정 불요)

rule 6 마지막 절 *"counts and genotype/QC flags are separate channels"* = **분리 제약**이며
일곱째 채널 그룹이 아니다. 근거 3건, 전부 자체 확인:

1. **문장 구조** — 바로 앞이 *"A18.4 weights are not input multipliers"*로, 이미 다른 곳에 정의된
   양이 트랙에 **어떻게 들어오면 안 되는지**를 말하는 음성 제약이다. 같은 호흡의 세 번째 항목.
2. **genotype_stratum의 확립된 역할은 손실 가중이다** — A18.4(`:264`)가 `source_weight`에
   "reference 1.0 / known non-reference·hybrid-pooled 0.5 / unknown 0.5"를 곱하고,
   A18.5(`:270`)가 "recorded per library and **applied as the A18.4 multiplier**"라고 못박는다.
   C2 라벨의 이산 손실가중 경로이지 B6의 연속 per-nucleotide 트랙이 아니다.
   rule 2의 "every row carries … genotype stratum … and provenance hashes"는 **관측 테이블
   메타데이터**이며, provenance hash를 "싣는다"는 것과 같은 의미다.
3. **`qc_flags`는 채널로 만들 물건이 아니다** — `build_b5.py:153`에서 `qc_flags VARCHAR`,
   `HARD_FLAG_PATTERNS`로 채워지는 자유텍스트 QC 부기 필드다(`edge_partial=n`, `leak_masked=n` 류).
   저장소 어디에도 열거가 없는 것은 **빈틈이 아니라 (i)의 증거**다. rule 6이 phase 6채널의
   값 규약과 clip 상수까지 적는 정밀도로 쓰였는데, 실재하는 채널 그룹을 개수도 없이 흘릴 리 없다.

**조치**: rule 6 개정 불요. 다만 다음 검토자를 위해 각주 한 줄을 넣는다 —
*"genotype/QC-flag information is carried as row/table metadata per rule 2 and A18.4's loss-weight
multiplier; it is not represented among rule 6's tracks."*

## D1 — 재형성됨 (M-HQ18 archive census 결과)

**census 실측 (ENA PRJEB22122, 16런 전부)**: PacBio RS II · `library_strategy=FL-cDNA` ·
`library_selection=Oligo-dT` · **`submitted_format=FASTQ`에 제출자 파일명 명시** ·
`result=analysis` 산물 0건. A46 결정 2가 쓰는 maize 5런:

| run | 파일명 | reads | avg |
|---|---|---:|---:|
| ERR2093901 | `RS_Sequel_maize_bract_hq.fastq.gz` | 87,179 | 2,065 nt |
| ERR2093902 | `RS_Sequel_maize_leaf_hq.fastq.gz` | 138,021 | 1,865 |
| ERR2093903 | `RS_Sequel_maize_pericarp_hq.fastq.gz` | 73,990 | 1,594 |
| ERR2093904 | `RS_Sequel_maize_seedling_hq.fastq.gz` | 77,825 | 2,013 |
| ERR2093905 | `RS_Sequel_maize_shoot_hq.fastq.gz` | 95,430 | 1,932 |

**제출자가 파일명에 `_hq`를 박아뒀고**, 7만~14만 리드 / 평균 1.6~2.1 kb는 **isoform 규모**다.
A15의 "Wang 2018 HQ isoforms"가 맞다.

**⇒ M-HQ18은 EST급이 아니다.** A47 규칙 2("FLNC 자격은 다운로드 가능한 FLNC/CCS 산물을 요구")를
충족하며 HQ isoform은 clustering 후 polished 산물이다. 따라서 **A47 규칙 3(EST급 사용 허용,
구조권한 부정)은 M-HQ18에 적용되지 않는다.** A14는 명시적으로 *"Restored to tier 1: M-HQ18"*로
테스트종 검증 tier 1에 복원해 뒀다.

**Kimi 자문의 정오**:
- ❌ "M-HQ18을 EST coverage 채널만으로 제한" — 전제(M-HQ18이 EST급)가 census로 무너졌다.
  구조 권한이 있는 산물이므로 coverage-only는 junction/donor-acceptor 신호를 **버리는** 것이다.
- ✅ **"M-FLNC는 어떤 입력 채널로도 갈 수 없다"** — rule 10 결정 2가 M-FLNC를 Z. mays **채점**
  세트로 지정했고 rule 10이 "a molecule used as input at a locus never scores that locus"라고
  못박는다. **D1과 무관하게 강제된다.** 내 v1·v2 서술이 M-FLNC를 라우팅 후보로 언급한 것은 오류.
- ✅ **(c)가 살아 있다** — A47 규칙 1은 FLNC **생산**을 금지하지 공개 산물 **다운로드**를 금지하지
  않는다. 그리고 **A15의 2026-09-01 scout은 maize와 A. thaliana만 훑었고 나머지 7개 학습종은
  조사된 적이 없다.**

**선택지**:
| | 내용 | 대가 |
|---|---|---|
| (a) 채널 유지 | Z. mays 추론에서 **진짜 tier-1 구조 증거**가 **미학습 가중치**를 통과 | 임베딩에 통제불가 섭동. 캐비엇으로 못 막는다 |
| (b) EST 그룹으로 | coverage-only면 tier-1 산물의 구조 신호 폐기 / 전부면 EST 통계로 학습된 채널에 분포 불일치 | 정보 손실 또는 불일치 |
| (c) 학습종 PacBio 확보 | 근본 해결. **7개 학습종 미조사** | 신규 scout, 일정 영향 |

**권고**: (c)를 먼저 확인한다. 있으면 죽은 채널 문제가 뿌리에서 사라지고, 없으면 **"찾아봤는데
없다"**는 기록이 남아 (a)/(b) 어느 쪽을 골라도 승인 문서가 단단해진다.

---

## SPEC-1 — full C0 빌더

### 진입점
```
revision/scripts/evidence/b6_build_c0.py
revision/scripts/evidence/b6_audit_align.sbatch     # rule 4(b) 감사 정렬
revision/scripts/evidence/b6_merge_evidence_db.py   # ← 신설 (rule 9는 evidence DB sha256 단수를 요구)
```
배치 근거 정정: 증거 파이프라인은 `revision/scripts/evidence/`에 있다. (v1이 "번호 스크립트는 패널
전용"이라 적은 것은 **틀렸다** — `01_`~`72_`가 전 툴체인에 걸쳐 있고 `59_evidence_support.py`도 그중
하나다. 번호를 안 붙이는 이유는 패널 전용이어서가 아니라 이 스크립트들이 `evidence/` 하위이기 때문.)

### CLI
```
b6_build_c0.py
  --roles        data/manifests/DATASET_ROLES.longread_v1.tsv
  --est-roles    data/manifests/DATASET_ROLES.est_v1.tsv
  --species      <species_id>            # 1호출 = 1종 1소스
  --source       est|ont|pacbio|protein
  --align-root   <frozen 정렬 산출>
  --audit-root   <b6_audit_align 산출>
  --corpus       <코퍼스 DuckDB>          # ★신규. rule 4(a)의 타일 기준 제외에 필수
  --splits       data/splits/b5_orthogroup_split_v1.tsv
  --flags        <A22 geenuff_flags.tsv>  # ★A30 제거됨(아래 참조)
  --overlap      <A33 overlap components + 시드된 flank 값>
  --hsp          <est_protein_hsp 입력>   # --source protein 일 때만
  --out          <evidence_db 경로>       # 배타적 생성
  --qc           <QC JSON 경로>
```

**★ `--corpus` 신규 (v1 결함).** rule 4(a)의 제외 중 **넷이 타일 그리드 기준**이다 —
"test blocks (A29)", "genes of test orthogroups and, for training tiles, of valid orthogroups",
"decoy-masked genes", "the `partial` span of any edge-crossing gene in a tile". split 테이블과 flag
파일만으로는 평가 불가능하다. 선례: `mvp_build_c0lite.py:353`이 `--db`를 받아 `:83`에서
`corpus_tile`을 만든다.

**★ `--flags`에서 A30 제거 (v1은 새 규칙을 도입하고 있었다).** rule 4(a)의 제외 목록은 **A22
hard-flagged genes만** 지목한다. A30은 rule 12의 **패널 적격성**에 쓰이는 별개 목적이다.
A30을 누출 필터에 먹이면 rule 4(a)가 제외하지 않는 분자를 제외하게 되며, 이는 새 값이 아니라
**새 규칙**이다.

### 입력 게이트 (fail closed)
1. 모든 런/세트는 `DATASET_ROLES`에 role 행이 있어야 한다. 없으면 거부(A18.3).
2. `c2_training_eligible`이 아닌 행은 training-side 테이블 진입 불가. `b1_validation_only`는
   `validation_only=true`로만 적재.
3. 모든 입력 파일 해시를 매니페스트 기록값과 대조. 불일치는 abort.
4. 도구 신원은 `revision/protocols/provenance/PROVENANCE.*.json`의 sha256으로 확인.

### 출력
- **evidence DB**(종×소스별) — rule 2의 9테이블 + route B의 `est_protein_hsp`. 모든 행이
  source · library · genotype stratum · role · provenance hash 보유.
- **병합 evidence DB** — `b6_merge_evidence_db.py` 산출. **rule 9가 `evidence DB sha256`을 단수로
  요구하므로 최종 병합본이 정본이다.**
- **QC JSON** — rule 4(d) 형식.
- **PROVENANCE.txt** — 날짜, 도구 sha256, 입력 해시, 확장 명령, 산출물 sha256.

### 순서 (강제)

```
1) b6_audit_align.sbatch
     각 소스의 frozen 명령에서 --secondary=no → --secondary=yes -N 50 -p 0.5 로만 교체.
     EST:       minimap2 -t 32 -ax splice:hq -G 200000 --cs=long
     ONT/PacBio: A14/§3의 자기 명령 그대로.

2) rule 4(b) 제외 판정  ★v1이 통째로 빠뜨린 단계
     (i)  secondary placement ≥ 50 인 분자      → unauditable, 제외
     (ii) 제외구간 **밖**에 MAPQ < 5 placement  → unauditable, 제외
     (iii) retained 분자 전부에 대해 primary/supplementary placement의
           직접 구간 검사(direct interval test)
     N = 50은 (i)의 **제외 기준**이자 exhaustiveness의 분모다. v1은 후자로만 썼다.

3) exhaustiveness  = placement < N 인 분자 비율, 종별 보고
     < 99.9 %  →  N 상향 후 1)부터 재실행. 트랙 빌드 금지.

4) b6_build_c0.py : 누출제외(a)(b) → 분자단위 병합 → 관측 적재

5) rule 4(c) 단계  ★v1에 없던 단계
     excluded gene과 이웃이 공유하는 **좌표**의 junction support 제외.
     좌표 수준 검사라 4)의 분자 병합에 접히지 않는다. QC의 cause (c) 칼럼은
     이 단계만이 채운다 — 없으면 0이 "누출 없음"으로 오독된다.

6) b6_merge_evidence_db.py : 종×소스 DB 병합 → 단일 sha256

7) QC 게이트 : 누출 QC 카운트가 0이 아니면 run abort (rule 13)
     A. thaliana: held-out loci 겹치는 분자 수 == 제외 수, 그 loci retained == 0
```

### 하지 않는 것
참조 유래 feature 금지. 감사 정렬은 **오직 제외 판정에만** 쓰고 라벨에 절대 쓰지 않는다(rule 4(b) 명문).
같은 종 단백질 사용 금지(A19).

---

## SPEC-1b — rule 5 코퍼스 재빌드 ★v1에 아예 없었다

v1의 commands 인벤토리는 미비 항목을 C0와 트랙 **2건**으로 신고했으나, rule 15의 WP1 목록은
"C0 pipeline, route-B phase, leakage exclusion, **corpus rebuild**, tracks, **trainer**, pilot,
full runs, evaluation"이다. **코퍼스 재빌드와 트레이너 진입점이 어느 문서에도 없었다.**

### 진입점 / CLI
```
revision/scripts/evidence/b6_build_corpus.py
  --base      b5_full_a40_v1.db      # A40 동결본, md5 dda7e78c… (09-21 재검증 일치)
  --masks     <A33 overlap components + 시드된 flank>
  --out       b5_full_b6_v1.db       # 배타적 생성
  --freeze    <freeze JSON>          # 단일 sha256 기록처
```
rule 5 요구: 타일 빌더의 gene-containment mask가 edge-crossing held-out DNA를 노출시키므로
(`build_b5.py:561-567`), B6 빌드는 **train/valid 타일에서 그 노출 구간 ± flank를 마스킹**한다.
B6 코퍼스와 sequence-only 비교군은 **한 DuckDB 파일**이며 비교군은 같은 `rn` 순서에 트랙만
무시하는 것 — 따라서 `geneList`, `window_genes`, `tile_blocks`는 **구조적으로 바이트 동일**하고
freeze JSON은 **둘에 대해 sha256 하나**를 기록한다. 트랙 없는 런에 트레이너는
`evidence_channel=off`를 로그한다.

**트레이너 진입점**: B5 레시피 승계(A42 `configs/b5_400m_win_v3.json`, A44 런타임). B6 변경분은
`evidence_channels` 설정값 하나이며 `configuration_transgenic.py:98`이 기본 0,
`modeling_HyenaTransgenic.py:433`이 거기서 `nn.Linear(evidence_channels, d_model, bias=False)`를
만든다. **K는 config 값이다.** 정확한 진입점 문장은 D1/D2로 K가 정해진 뒤 확정한다.

---

## SPEC-2 — float16 트랙 래스터

### 진입점
```
src/transgenic/evidence/b6_tracks.py          # 라이브러리
revision/scripts/evidence/b6_build_tracks.py  # dose-response 물질화 CLI
revision/scripts/evidence/b6_build_obs_cache.py  # ← 신설, 아래 성능 항 참조
```

### 라이브러리 계약
```python
rasterise(con, *, species, chrom, offset, L, is_rc,
          sources, libraries, keep_fraction, aug_seed) -> np.ndarray   # float16 [L, K]
```
**★ `libraries` 신규 (v1 결함).** rule 8은 source-withholding 0.1 **과** library-withholding 0.1
둘 다 요구하고 rule 2는 모든 행에 `library`를 싣는다. v1 시그니처는 source 수준만 있어
library-withholding을 표현할 수 없었다.

### thinning
`rank = int.from_bytes(blake2b(f"{aug_seed}\t{species}\t{molecule_id}".encode("utf-8"),
digest_size=8).digest(), "big") / 2**64`, `rank < f`면 유지. `hashlib.blake2b` 고정, 내장 `hash()` 금지.
**`rank`는 `(aug_seed, species, molecule_id)`의 순수 함수이므로 런 내내 불변이다 — 분자당 1회
계산해 컬럼으로 저장하고, 접근할 때마다 해싱하지 않는다.**

### 채널 — ✏️ **K = 31 확정(2026-09-22, A46 rule 6)**; 아래는 확정 전 기록
rule 6이 정의하는 그룹: 소스×strand transcribed-block coverage(clip `log1p(64)`) /
소스×strand×(donor,acceptor) 경계염기 support / 소스별 unknown-strand coverage /
protein CDS coverage `log1p(organisms)` / route-B phase {0,1,2}×strand / 소스별 availability mask /
**그리고 rule 6 마지막 절의 genotype·QC-flag 채널(D2)**.

확정된 독법 2건(독립검토·Kimi 수렴):
- **unknown-strand는 소스당 1채널**. 프로토타입 `mvp_tracks.py:73`이 `{'+':0,'-':1,'?':2}`로 이미 구현.
  물리적 근거: A39 실측에서 장독 27런 중 **20런이 진짜 비배향**(sense fraction 0.49–0.53) — ONT다.
- **protein CDS는 무strand 1채널**. rule 6은 strand가 붙는 곳마다 명시하고 여기서만 뺐다.

### ★ 중첩과 RC — v1의 가장 위험한 결함

**중첩을 명시한다: `(source, strand, donor/acceptor)`.** v1은 개수 12만 적고 중첩을 안 적었는데,
**RC 정합성이 걸린 건 개수가 아니라 중첩이다.** `(source, donor/acceptor, strand)`로 읽으면 RC가
donor와 acceptor를 뒤바꾼다.

프로토타입이 단일 소스에 대해 고정한 배치와 순열(`tests/test_mvp_tracks.py:61`에 실행 계약으로 고정):
```
0 cov+  1 cov−  2 cov?  3 donor+  4 acceptor+  5 donor−  6 acceptor−  7 mask
RC 순열: [1, 0, 2, 5, 6, 3, 4, 7]
```
읽으면: `+`/`−` coverage 교환 / `?` coverage는 교환할 strand가 없어 불변 /
junction 블록은 `(donor+, acceptor+)` 쌍과 `(donor−, acceptor−)` 쌍을 통째로 교환해
**donor→donor, acceptor→acceptor를 보존**한다. 이것이 rule 6의 "donor/acceptor identity kept,
strand channels swapped"의 구체형이다.

**그룹핑 정합**: 프로토타입은 unknown-strand를 소스 자신의 coverage 블록 안에 둔다(소스당
`{+,−,?}` 연속 → 3소스면 coverage 9채널 연속). v1은 그것을 별도 그룹으로 떼어 junction 뒤에 놓았다.
rule 6이 순서를 핀하지 않으므로 둘 다 조문 위반은 아니지만 **명세와 프로토타입이 조용히 어긋나는
것이 문제다.** v2는 **프로토타입 배치를 따른다**(소스 블록 내 `{+,−,?}` 연속).
`modeling_HyenaTransgenic.py:547`은 텐서 형상만 검증하고 채널 의미는 보지 않으므로 안전망이 없다.

**★ phase와 protein CDS의 RC — v1이 인용한 테스트가 v1의 반대를 증명한다.**
v1은 `tests/test_gsf_rc.py:26`을 근거로 삼았는데 그 줄은 `test_rc_recomputes_phases_five_to_three_prime`
이고 주석은 *"RC flips the strand but **not the transcript** … the 101-bp piece keeps phase
`(3 - 100 mod 3) mod 3 = 2` (C) … a wrong phase (e.g. 1 -> B) must not appear."* 이다.

v2 규칙: **route-B phase 6채널은 RC에서 strand만 교환하고 phase 라벨은 보존한다** — 한 염기는
자기 코돈 내 위치를 유지한 채 `+`/`−` phase 채널 사이를 이동할 뿐이다. GSF 문자열 수준의 phase
재계산(`(3 - len mod 3) mod 3`)은 **feature 목록이 좌표 정렬될 때** 필요한 것이고, 염기별 래스터에는
좌표가 이미 역전되어 적용된다. **이 독법은 추론이며 rule 6의 RC 문장은 phase·protein CDS를
다루지 않는다**(그 문장이 쓰인 GSF 맥락에 두 채널이 없었다) — 그러므로 A46에 명시되어야 한다.
protein CDS는 무strand이므로 좌표 역전만 적용한다.
**요구 시험**: float16 phase 채널 전용으로 `test_rc_recomputes_phases_five_to_three_prime`에
대응하는 단위시험을 둘 것. C6가 phase 채널 기여를 재는 컨트롤이므로, RC 오염은 C6를 조용히 무효화한다.

### ★ mask 의미론 — fraction 0과 N 염기 (v1이 침묵한 새 규칙)

프로토타입은 `keep_fraction == 0`에서 **availability mask 포함 전부 0**을 반환하고
(`mvp_tracks.py:31-33`, `tests/test_mvp_tracks.py:104`에 고정), N 염기에서 **행 전체를 0으로
내린다**(`:107`). 반면 공급됐으나 분자가 없는 소스는 `mask = 1`을 유지한다(`:99`).

**이대로면 fraction 0이 rule 8의 source-withholding과, 그리고 rule 11의 C2(tracks permanently
zero)와 바이트 동일해진다.** A46은 셋을 다른 것으로 취급하는데 구현이 구분하지 못하고,
rule 13 dose-response의 다섯 점 중 **왼쪽 앵커가 파괴되어 G2가 읽을 곡선이 망가진다.**

v2 규칙:
- **`keep_fraction = 0`에서 availability mask는 1을 유지한다.** 의미는 "소스는 있으나 이 draw에서
  유지된 분자가 없다"이며, source-withholding("소스 자체가 없다", mask = 0)과 구분된다.
- **N 염기에서 availability mask는 내리지 않는다.** rule 6이 mask를 *"1 if the species/library has
  that source at all"* — **종/라이브러리 상수**로 정의하므로 개별 염기가 끌 수 있는 값이 아니다.
  N 염기에서는 **카운트 채널만** 0으로 한다.
- 두 규칙 모두 A46 rule 6에 명시되어야 한다(프로토타입 동작을 **명시적으로 대체**한다).

### ★ 성능 — 즉석 DuckDB 래스터를 쓰지 않는다 (v1 판단 철회)

v1은 rule 6의 *"membership is kept so that thinning re-rasterises"*를 근거로 학습 시점 즉석 래스터를
제안하고, 틀리면 "WP4 성능 게이트에서 드러난다"고 적었다. **그 안전망은 없다** — WP4 게이트는
메모리 ≤100 GB, Dynamo fallback 부재, cached-generation parity, RC 타일, resume 결정성뿐이고
**DB 접근 처리량을 보지 않는다.** 잘못되면 WP5 파일럿이 "느리지만 통과"하면서 N_pilot = 4,142
업데이트 예산만 태운다.

v2: **타일별 관측 캐시를 1회 오프라인 물질화**한다(`b6_build_obs_cache.py`) — (molecule_id, interval,
채널별 raw 기여, 선계산된 rank), RAM 또는 로컬 NVMe에 적재. 학습 스텝의 래스터는 이미 적재된
배열에 대한 **벡터화된 filter + scatter**가 되며 DuckDB 조인이 아니다. rule 6의 "membership kept so
thinning re-rasterises"는 그대로 만족된다 — thinning은 여전히 retained 관측에서 래스터를 다시 만든다.
**트레이너 작성 전에 처리량 게이트**(현실적 multi-worker 동시성에서 steps/sec)를 명시적으로 둘 것.

### CLI
```
b6_build_tracks.py
  --evidence-db <병합 evidence DB>
  --obs-cache   <b6_build_obs_cache 산출>
  --corpus      b5_full_b6_v1.db
  --fractions   1,0.5,0.1,0.01,0
  --draws       0,1,2,3,4            # aug_seed_d = 20260914 + d
  --panel       revision/protocols/b6_panels_v1.tsv
  --out         <매니페스트 디렉터리>
```
출력: (locus, fraction, draw)별 유지 분자 id 매니페스트 + sha256(rule 13), 채널 스펙 해시
(`track spec hash`, rule 9가 `runtime_identity.txt`에 요구).

### 검증 (명세는 승인 전, 실행은 승인 후)
- **★ WP3 메모리 게이트 (v1이 타이밍 검사로 바꿔 적었다).** rule 15 원문 조건 그대로:
  compile on, non-reentrant activation checkpointing, **4 ranks, 129,024-nt tier**,
  **peak reserved per rank ≤ 100 GB, 아니면 K 또는 fusion point를 트레이너 코드 작성 전에 재설계**.
  **K가 과대할 때 잡는 A46의 유일한 게이트**이므로 타이밍으로 대체 불가.
- 음성가닥 junction 정체성 RC 후 보존 / 중첩 draw (`keep(0.1) ⊂ keep(0.5)`) /
  드롭된 타일 전부 0 / float16 phase 채널 RC 단위시험(위) / mask 의미론 2건 시험.
- **★ step-0 기준 재작성 (v1은 달성 불가능한 기준을 썼다).** v1은 "seed-123 `best`와 비트 동일"이라
  적었으나 rule 7이 말하는 것은 **B6 자신의 zero-track 경로**가 sequence-only 경로와 정확히
  같다는 것이다. seed 123은 **미수리 코퍼스** 학습본이므로(rule 5, 결정 4) 비트 동일 대상이 아니다.
  v2 기준: **트랙 0일 때 B6 forward가 같은 체크포인트의 sequence-only forward와 비트 동일.**

---

## SPEC-2c — 체인 메모리 구성 (rule 6a) ✏️ 2026-09-22 추가

근거: A46 rule 6a(DRAFT), 설계 [B6_CHAIN_MEMORY_DESIGN_v2](B6_CHAIN_MEMORY_DESIGN_v2_20260922.md). 트랙(SPEC-2)과 **같은 관측 캐시·같은 rank·같은 draw**를 쓴다.

### 진입점
```
src/transgenic/evidence/b6_chains.py              # 라이브러리 (트랙 래스터와 같은 패키지)
revision/scripts/evidence/b6_build_obs_cache.py   # SPEC-2와 공유 — 체인 행을 함께 물질화
```
체인은 **학습 스텝에서 캐시 배열로부터 벡터화 조립**한다(SPEC-2 성능 항과 같은 이유로 즉석 DuckDB 금지).

### 입력 (동결 C0 체인 표, rule 2)
`partial_chain` / `partial_chain_intron` / `chain_junction` — ONT 접합은 ±3 nt 보정 좌표. 새 표를 만들지 않는다.
관측 캐시에 체인 행을 추가: `(species, source, library, molecule_id, placement_id, strand, rank, introns[(donor0, acceptor0)], units_key)`.
좌표: 0-based, 인트론 안쪽 경계 염기(`est_junction`의 `donor0`/`acceptor0` 정의, `mvp_build_c0lite.py:266-268`). 1-based 폐구간(`parse_sam`)은 C0 적재 시 한 번만 변환.

### 라이브러리 계약
```python
build_chains(cache, *, species, chrom, offset, L, is_rc, sources, libraries,
             keep_fraction, aug_seed, population, excluded_junctions, tile_exclusions,
             row, epoch) -> (chain_pos int32[N_tok, 2], chain_feat int16[N_tok, F], chain_mask bool[N_tok], qc)
```
- `N_tok = 512` 고정(0번 슬롯 = 영구 마스크 더미), `N_chain = 64`. `chain_feat` = (order_fwd, order_rev, slot, source, strand, units_bin).
- `population`: 학습·검증 타일 = retained(rule 4 이후); 평가 창 = DATASET_ROLES의 **`b6_input` 행**(rule 10) — **소스 종류가 아니라 데이터셋으로 선택**한다(A. thaliana 학습 ONT 36런 중 입력 가능은 A-ONT2 PRJNA594286 3런뿐). `b6_score` 분자가 섞이면 **오류로 중단**. *Z. mays*는 M-EST만(M-HQ18은 RS II라 트랙만, A47.3).
- 순서(강제): ① `keep_fraction`·`sources`·`libraries` 필터(트랙과 **동일 draw**) → ② **학습·검증 타일에서만** rule 4(a) 타일 제외 구간·rule 4(c) 제외 접합에서 **체인 분할**(평가 창에서는 rule 4 분할 없음), 타일/창 경계를 넘는 인트론에서 분할 → ③ strand `?`·단일엑손 제거 → ④ (strand, 인트론 목록) 병합, `units` = 독립분자 수 →
  ⑤ 2단계 채우기(롱리드 → EST; units ↓, 인트론 수 ↓, blake2b(키) ↑) → ⑥ 상한 절단(소스별 절단 수 → `qc`) →
  ⑦ 슬롯 순열 π = 슬롯 1…63에 대해 `blake2b(f"{aug_seed}\t{row}\t{epoch}\t{slot}".encode("utf-8"), digest_size=8)`의 argsort (aug RNG 비소비, 평가 시 epoch = 0); ⑤의 tie-break 키 = `blake2b(f"{species}\t{strand}\t{intron list}".encode("utf-8"), digest_size=8)` big-endian; 통째로 들어가지 않는 체인은 **버린다**(자르지 않음) → ⑧ `is_rc`면 p → L − 1 − p, 역할 유지, 가닥 교환 → ⑨ 패딩·마스크.
- 모델 쪽 gather 인덱스 `(pad + tok_offset + p) // 6`는 **콜레이트가** 샘플별 `pad`를 더해 만든다(라이브러리는 창 좌표만 낸다).

**이름 주의(L7):** C0 `donor0`/`acceptor0`는 **가닥 인식**(− 가닥에서 donor0 > acceptor0)인 반면 `gsf_contract.add_junction_evidence`는 `donor0 < acceptor0`(좌표 순)를 요구한다 — 같은 이름, 다른 의미. 체인 코드는 C0 정의(인트론 첫·끝 염기, 포함)를 쓰고, gsf_contract에 넘길 때만 좌표 순으로 바꾼다.

### CLI (dose-response 물질화 — SPEC-2 CLI 확장)
`b6_build_tracks.py`가 (locus, fraction, draw)별 매니페스트에 **체인 원소 수·소스별 절단 수**와 **chain spec hash**
(A46 rule 9 목록: d, layers, heads, 투영 차원, 특징 집합, 순서 상한 31, units 인코딩(스칼라 w_count·log1p(min(units,64))), N_chain, N_tok, 채우기 순서·tie-break 인코딩, 슬롯 순열 인코딩, α 초기화, 학습률 그룹, 블록 dropout 0.1, population·분할 규칙, RC `chain-rc-v1`)를 함께 기록한다(rule 9 `runtime_identity.txt`).

### 검증 (명세는 승인 전, 실행은 승인 후)
설계 v2 §7의 12개 시험 + 트랙과의 정합: 같은 (fraction, draw)에서 체인에 쓰인 분자 집합 ⊆ 트랙에 쓰인 분자 집합(단일엑손·`?` 제외분만 차이) ·
중첩(절단 전) · rule 4(c) 분할 · 채점 세트 소스 부재 · RC 두 번 항등. **WP3 메모리 게이트는 체인 블록을 포함한 상태로 측정**한다.

---

## 남은 미결

| | |
|---|---|
| **D1** PacBio 채널 처분 | ✅ 종결 2026-09-22: **채널 유지**(A47.3로 학습 PacBio 49런 복원) |
| **D2** rule 6 마지막 절 독법 | ✅ 종결 2026-09-22: 분리 제약, K 불변 |
| K와 채널 순서 전문 | ✅ **K = 31**, 순서·RC 순열과 함께 A46 rule 6에 리터럴로 기입(2026-09-22) |
| 트레이너 진입점 문장 | ⚠️ K 확정됨 — 체인 메모리 블록(rule 6a) 포함해 작성 필요 |
| ✏️ 체인 메모리 구성 | ✅ 명세 SPEC-2c (2026-09-22) |
| route B "HSP ≥ 30 aa" 운영 정의 | RICH17 fixture로 고정 |
| 채널별 고정 스케일 도입 여부 | Kimi 권고. bias는 rule 7 불변식을 깨지만 대각 스케일은 0과 교환(`diag(s)·0 = 0`) |

범위: 문서 작성만. 코드·프로토콜 수정, 잡 제출, 커밋 없음.
