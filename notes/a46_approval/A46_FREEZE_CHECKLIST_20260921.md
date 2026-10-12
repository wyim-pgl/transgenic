# A46 승인용 사전 명세 동결 체크리스트 — 2026-09-21

근거: `PROTOCOL_B1_frozen_v1.md` **rule 15**(construction rule) + **rule 12 Freeze clause**.
두 조항이 요구하는 것은 동일한 7개 범주다:

> frozen **commands** · **tool versions (by PROVENANCE sha256)** · **seeds** · **thresholds** ·
> **denominators** · **panel identities/counts/hashes** · **N_pilot**
> \+ 승인 시점에 이미 존재하는 artefact의 **해시 검증**
> \+ 승인 후로 미룰 수 있는 것은 **아직 없는 artefact의 해시뿐**

범례: ✅ 동결 완료·검증됨 / ⚠️ 값은 있으나 형식·정합성 미비 / ❌ 없음

> ✏️ **갱신 2026-09-22 (체인 메모리 · A47.2/A47.3 · D1 반영).** 새 항목: rule 6a 체인 메모리 설정, C7, (ii-c). K = 31 확정(D1 채널 유지, D2 분리 제약). PacBio `splice:hq` 줄은 이제 **사용한다**(A-CCS23 패널 채점, 학습 PacBio 49런). 변경 위치는 각 항목의 ✏️ 표시.

---

## PART 0 — 승인 전에 먼저 고쳐야 할 것 (BLOCKER) — ✅ **2026-09-21 해소됨**

> A46 draft revision 5건 적용 완료. B0-1·B0-2·B0-3 전부 닫혔고 denominators도 확정됐다.
> 상세: [A46_AMENDMENT_DRAFT_20260921.md](A46_AMENDMENT_DRAFT_20260921.md). 아래는 당시 진단 기록.

A46 본문이 **오늘 내린 결정들과 어긋난 상태다.** 이대로 승인하면 사실과 다른 명세를 동결하게 된다.

### B0-1 ❌ rule 10의 채점 세트 기술이 무효

현재 본문(`:985`):
> "A. thaliana strict held-out loci: input = `excluded_heldout` A-EST + A-ONT2 …;
> **scoring = A-ONT1 + A-HiFi**."

- **A47**: A-HiFi(ERR13994458)는 FLNC 산물이 없고 생산도 금지 → **구조권한 없음**
- **A47.1**: A-CCS23도 구조팔을 맡지 못함(저자결정)
- ⇒ A. thaliana 채점 구조 소스는 **A-ONT1 단독**

### B0-2 ❌ 독립 outcome의 실현 경로가 사라졌다 — 가장 중요한 항목

rule 10 마지막 문장:
> "Any F1 against RefGen_V4 is labelled `reference-dependent`;
> **the independent outcome is complete-chain support by the scoring set (§9 P1–P5)**."

그런데 **ONT full 실행을 하지 않기로 했다**(09-21 저자결정). 최종 패널 **1,729 loci**에서
complete-chain support를 계산하려면 그 loci에 A-ONT1 정렬이 있어야 한다. 현재 확보된 경로는
MVP의 **50 loci 국소 정렬**뿐이다.

따라서 셋 중 하나를 골라야 하며, **승인 전에 정해야 한다**:

| 선택지 | 의미 | 비용 |
|---|---|---|
| **(a) 최종 패널 1,729 loci 국소 정렬** | rule 10의 독립 outcome 유지. full run이 아니라 패널 한정 — ONT full 결정과 모순 아님 | A-ONT1 3런에 대해 패널 윈도 사전선별 + pass2. MVP(50 loci)와 같은 구조, 규모만 34배 |
| **(b) 독립 outcome 포기** | G1/G2/G3는 전부 transcript-F1이라 **게이트 자체는 그대로 작동**. 다만 모든 결과가 **reference-dependent(TAIR10 대비)**가 되고, B1이 회피하려던 바로 그 순환성이 남는다 | 0. 대신 rule 10 개정 + 주장 축소(rule 16) 필요 |
| **(c) 파일럿 300 loci만 독립 채점** | 중간안. 파일럿 go/no-go에만 독립 증거, 최종 게이트는 reference-dependent | (a)의 약 1/6 |

**✅ 저자 결정 2026-09-21: (a) 최종 패널 1,729 loci 국소 정렬로 간다.** 독립 outcome을 유지한다.
rule 10은 채점 세트를 **A-ONT1 단독**으로, 독립 outcome은 **최종 패널 1,729 loci의 국소 정렬 기반
complete-chain support**로 개정한다. ONT full(전체 run 정렬)은 여전히 하지 않는다.

### B0-3 ⚠️ A47/A47.1/ONT-full 결정을 반영한 **A46 개정안**이 필요

A46은 DRAFT이므로 승인 전 수정이 가능하다. 수정 대상 최소 3곳:
- rule 10 채점 세트 문장 (B0-1)
- rule 10 독립 outcome 문장 (B0-2 선택 결과 반영)
- rule 17 MVP 서술의 `A-ONT1/A-HiFi` 병기 (A47 적용 노트와 정합)

단 **rule 17의 §5.3 인용블록은 건드리지 말 것** — verbatim 인용이며 오늘 정규화 동일성을 검증했다.

---

## PART 1 — 7개 동결 범주

### 1. Commands ⚠️

| 대상 | 현재 | 필요 |
|---|---|---|
| 패널 생성 | ✅ `71_b6_panels.py`, `72_mvp_select_loci.py`, `72a_est_molecules_per_locus.sh` — **스크립트 sha256이 A46 본문에 이미 박혀 있고 실물과 일치** | — |
| 장독 정렬 (§4) | ✅ 명령 3종이 B1 §4에 verbatim 존재 (ONT `-uf` 유/무, PacBio `splice:hq`) | ✏️ ~~A47 반영해 PacBio 줄 사용 안 함 명시~~ → **A47.2/A47.3: PacBio `splice:hq` 사용**(A-CCS23 패널 채점, `-uf` 없음; 학습 PacBio 49런 `longread_align_pacbio.sbatch` 완료) |
| C0 파이프라인 | ❌ 빌드 명령 미동결 (있는 건 MVP용 `mvp_build_c0lite.py`) | 전체 C0 빌드 명령·입력·출력 스키마 |
| route B (diamond blastx) | ⚠️ 규칙은 있음(HSP ≥ 30 aa) | 정확한 `diamond blastx` 인자 전체 |
| route A (miniprot) | ⚠️ A41이 chunk 전략 기록 | phase 추출 명령 |
| 트랙 래스터 | ❌ | float16 트랙 생성 명령 |
| ✏️ **체인 메모리 구성 (rule 6a)** | ⚠️ **명세 작성됨(SPEC-2c, 2026-09-22)** — 규칙은 rule 6a에 동결, 명령 명세는 [명세 v2 SPEC-2c](A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md); 코드 없음(C0·트랙과 같은 처지, 실행은 승인 후) | C0 체인 표(`partial_chain`/`partial_chain_intron`/`chain_junction`) → 타일별 체인 텐서(`chain_pos`, `chain_feat`, `chain_mask`) 생성 명령. 트랙 래스터 명세(SPEC-2)와 같은 문서에 추가 |
| 트레이너 | ⚠️ B5 레시피 승계(A42 `b5_400m_win_v3.json`, A44) | B6 변경분(evidence_channels **= 31**, ✏️ **체인 메모리 블록: d 512, 8×64 헤드, 12층, α 0 초기화·weight decay 제외, 캐시 없음**) 명시 |
| 평가 | ✅ GFFCompare v0.12.6 exact intron-chain / monoexonic exact CDS | — |

### 2. Tool versions (by PROVENANCE sha256) ✅ — **완료 (4파일)**

rule 9: *"Tool versions are recorded in the PROVENANCE files whose sha256 values are listed in the
run log; no version is restated here."* 즉 **A46은 버전을 본문에 적지 않고 PROVENANCE 파일에
위임한다.** 그런데 **프로젝트 수준 PROVENANCE 파일이 없다.** 현재 존재하는 것은 개별 잡 산출물
안의 `provenance.json`뿐(`notes/est_execution_20260916_1530/…`, `notes/approved_execution_20260917/…`).

rule 9가 이름을 든 도구: `minimap2` · `samtools` · `blastn` · `diamond` · `DuckDB` · `torch` ·
`accelerate` + compile policy.

이미 확정적으로 관측된 값(승인 PROVENANCE에 담을 후보):

| 도구 | 값 | 출처 |
|---|---|---|
| minimap2 | 2.28-r1209 | job 6165965 승인 런타임, SHA256 핀 |
| samtools | 1.3.1 | 동일 (패널 EST 집계 helper도 동일 버전 사용) |
| seqkit | 2.13.0 | 동일 |
| python | 3.11.14 | 동일 |
| GFFCompare | v0.12.6 | A46 rule 13 |
| SIF | `transgenic-ngc-26.08.sif` sha256 `1a5dee27…` | A44/A45, rule 17 |
| torch / accelerate / transformers | ❌ 컨테이너 내 실측 미기록 | — |
| diamond / blastn / DuckDB / miniprot | ❌ | — |

**2026-09-21 진행:** `revision/protocols/provenance/` 신설.
`PROVENANCE.cpu_tools_v1.json` sha256 **`0193ea3a8c8ffecddad2bcbeae8045a91dd781a9ea60e73d26657c3669995a65`**
— pronghorn 실측으로 CPU 도구 11개의 **경로 + 바이너리 sha256 + 버전** 기록 완료
(minimap2 ×2, samtools ×2, seqkit, blastn, gffcompare, diamond, miniprot, python3 ×2).

핵심:
- **gffcompare v0.12.6** 존재 확인 — A46 rule 13이 요구하는 바로 그 버전 (`a440e153…`)
- **diamond 2.1.24** `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d` —
  09-16 fixture provenance가 **독립적으로 기록한 값과 일치**. 정본 확정
- **minimap2 두 빌드가 같은 버전 문자열, 다른 sha256** (`8a561f2c…` vs `731e16c5…`) →
  rule 9가 버전이 아니라 sha256을 요구하는 이유의 실물 사례
- **samtools 1.3.1(RNASEQ_bch709) vs 1.21(transgenic-revision)** — rule 12는 **1.3.1**을 핀.
  프로젝트 env의 1.21로 대체 불가

**09-21 추가 완료 3건:**
- `PROVENANCE.gpu_host_tools_v1.json` `92b8babe…` — **DuckDB 1.5.5**(pgl-gpu) + 프로즌 코퍼스 **md5 재검증 일치**
- `PROVENANCE.univec_screen_v1.json` `4fd697fe…` — **blastn 미확인 해소**: 스크리닝 자신의 receipt가
  11개 EST 세트 전부에 `blastn: 2.5.0+`를 기록. A21 파라미터도 B1 본문과 일치. Athaliana
  `input_md5=3d87af2e…`가 `DATASET_ROLES.est_v1.tsv`의 `local_fa_md5`와 일치(교차확인).
  단 receipt가 남긴 건 **버전 문자열이지 sha256이 아니다** — minimap2 사례만큼 강한 식별은 아니다.
- pronghorn의 낡은 duckdb wheel **삭제 완료**(저자 지시). 4개 항목 제거, 사전 sha256 `3bafbe54…` 기록,
  삭제 후 세 env 전부 `ModuleNotFoundError`로 정상화, 무관 패키지 67개 무손상.

**✅ 마지막 1건도 완료 — `PROVENANCE.gpu_runtime_v1.json` sha256 `c084c76b…`** (DeltaAI 2-hop, 인증 살아있음).
- **SIF sha256 재측정 = A44 기록값과 정확히 일치**: `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`
  (11,120,398,336 B, 28.4 s). 컨테이너 크기도 1700s 예산 산정의 "SIF 11.12 GB" 항과 일치.
- 내부 실측: python 3.12.3 / **torch 2.14.0a0+4fdf77b940.nv26.08** / **transformers 4.57.6** /
  **accelerate 1.15.0** / numpy 2.1.0 / safetensors 0.8.0 / huggingface_hub 0.36.2 / datasets 5.0.1 /
  duckdb 1.5.5 / CUDA 13.4 / cuDNN 92500. **앞의 셋은 A44(:923) 기록과 다른 방법·다른 날 측정으로 일치.**
- **compile policy**: `torch._dynamo.config.suppress_errors = false`, cache_size_limit 8, inductor 사용가능.
  Dynamo가 컴파일 실패 시 조용히 eager로 내려가지 않고 raise한다 → rule 9가 요구하는 방향. **단 rule 9의
  나머지 절반(트레이너 자신이 raise)은 미해결** — `train_HyenaTransgenic.py:317`이 아직 warn만 한다.
- **주의 2건**: ① 컨테이너에 **pysam이 없다** → `mvp_build_c0lite.py`(pysam 의존)는 이 SIF 안에서 못 돈다.
  pronghorn `transgenic-revision`(pysam 0.23.3)에서 돌려야 한다. ② apptainer는 `/usr/bin/apptainer`로
  **PATH에 있다** — A42의 "apptainer module 없음"은 모듈이 없다는 뜻이지 바이너리가 없다는 뜻이 아니었다.
- 미측정: seed 123용 `transgenic-ngc.sif`(NGC 25.06, torch 2.8.0a0) 재해시.

**PROVENANCE 축은 이로써 충족.**

**✅ DuckDB blocker 해소 (같은 날, `ssh gpu` 실측):** 의존성이 깨진 게 아니라 **장소 문제**였다.
pronghorn은 `~/.local/lib/python3.11/`의 낡은 user-site wheel을 모든 env가 집어서 실패하지만,
**pgl-gpu에서는 정상 동작** — `/home/wyim/micromamba/envs/transgenic/bin/python3`(3.12.13,
sha256 `13a56117…`), **DuckDB 1.5.5**, 호스트 GLIBC 2.35. 애초에 **프로즌 B5 코퍼스가 pgl-gpu에
있으므로** rule 5의 DuckDB 작업은 거기가 맞는 자리다.
→ `PROVENANCE.gpu_host_tools_v1.json` sha256 **`155bdbf67dfff26e5978e8dba788b60e42b91c9e0b3df7ebcf3dc9af0340898d`**

확인하면서 **A46 기록값 2개가 실측으로 일치**했다(read-only open):
| | 실측 | A46 |
|---|---:|---:|
| 파일 크기 | 23,534,252,032 B | 23,534,252,032 B |
| `geneList` 행 | 273,148 | 273,148 |
그 외 `gene_key_map` 334,642 / `gene_split` 334,642 / `window_genes` 978,864 / `tile_blocks` 4,454 /
`rejected_records` 8,130 / `build_manifest` 9. rule 12이 읽는 4개 테이블 전부 존재.

**조치 권고:** pronghorn의 낡은 wheel을 지우거나 교체할 것. 모든 env에서 `GLIBC_2.23`을 던지는
wheel을 남겨두면 다음 사람이 똑같이 걸린다.

### 3. Seeds ✅

| 용도 | 값 |
|---|---|
| 패널 셔플 | **20260914** |
| paired bootstrap (rule 13, MVP §5.2) | **20260914** |
| thinning `aug_seed` | **20260914 + seed** → seed 123이면 **20261037** |
| dose-response draw d∈{0..4} | `aug_seed_d` = **20260914 + d** |
| 모델 시드 | B6·C2·C3·C4·C6·✏️**C7** 전부 **123** (초기화·샘플러 순서 동일), 확인 시드 456/789 |
| ✏️ 체인 슬롯 순열 | `blake2b(aug_seed, row, epoch)` — 증강 RNG 스트림 비소비(rule 8) |

### 4. Thresholds ✅ (값 존재, 전부 본문에 박힘)

게이트: **G1** (B6−C2) 하한 ≥ **−1.0 pt** · **G2** (evidence−de novo) 하한 **> 0** AND
(evidence−C5) 하한 **> 0** · **G3** `validate_gsf` 위반 ≤ B5 AND recall ≥ **0.9 × B5**
Abort: leakage QC ≠ 0 · replay/RC/resume 결정성 실패 · **Dynamo fallback** · **110 GB/rank**
학습: dropout **p = 0.5** · thinning **{0.01, 0.1, 0.5, 1}** · source/library withholding 각 **0.1**
· patience **3** · ceiling **22** · effective batch **96**
증거: HSP **≥ 30 aa** · EST 길이 하한 **100 nt** (병기 arm **121 nt**)
통계: bootstrap **2,000 replicates**, **95 % percentile**
MVP: δ_loss **0.005 nats** · δ_F1 **2.0 pt**
누출 감사: secondaries audit alignment **≥ 99.9 % exhaustiveness**
✏️ **트랙 채널 K = 31**(3소스 × 8 + protein 1 + phase 6; rule 6)
✏️ **체인 메모리(rule 6a): N_chain = 64 체인, N_tok = 512 원소**(0번 = 더미), units 상한 log1p(min(·, 64)), 순서 임베딩 상한 31, 블록 d = 512 · 8 헤드 × 64 · 12층. **상한은 이 값으로 동결** — WP3 메모리 게이트(100 GB)에 걸리면 후속 개정으로만 바꾼다(저자 2026-09-22 "진행해")

### 5. Denominators ⚠️

- ✅ 게이트 F1: recall 분모 = **패널 loci의 모든 reference transcript**, precision 분모 =
  **모든 predicted transcript**, 전 arm 동일
- ✅ C5 특례: **C5가 아무것도 내지 않는 loci는 제외가 아니라 recall 0**으로 계산
- ✅ G3: **predicted transcript 1,000건당** 위반 수
- ⚠️ 독립 structural outcome(§9 P1–P5)의 분모 — **B0-2 선택에 종속**. (a)를 고르면
  "최종 패널 1,729 loci 안의 read-supported junction"으로 확정해 적어야 한다

### 6. Panel identities / counts / hashes ✅ — **완전히 준비됨**

실물 존재 + sha256이 A46 본문 기록과 **일치함을 이 문서 작성 시 재계산으로 확인**:

| artefact | rows | sha256 (앞 16) |
|---|---:|---|
| `revision/protocols/b6_eligible_loci_v1.tsv` | 2,079 | `7ad895fcc25abf75` |
| `revision/protocols/b6_est_molecules_v1.tsv` | 2,079 | `36de5b6827cdba2b` |
| `revision/protocols/b6_panels_v1.tsv` | 2,079 | `79367154bbf922fd` |
| `revision/scripts/71_b6_panels.py` | — | `b81ec50baf88e101` |
| `revision/scripts/72_mvp_select_loci.py` | — | `eab68edb76a7bb43` |
| `revision/scripts/72a_est_molecules_per_locus.sh` | — | `9e27472eb29798f8` |

`.sha256` 사이드카 3개도 존재.

**카운트**: pool **2,079** → MVP **50**(mvp_core 20 + mvp 30) / pilot **300** / final **1,729**,
attrition **350 of 2,079**.
**Funnel** (3,430 strict held-out에서): −28 ChrC/ChrM, −4 A22 hard flag, −67 A30 caution,
−910 single-exon, −5 contig boundary, −337 overlapping, −0 train_weight ≤ 0 = **2,079**.
**Hard-flag 출처**: A22 `Athaliana.geenuff_flags.tsv` md5 `5ab607c5…`(156 genes) /
A30 `evidence/swissprot_hard_masked_genes_20260902.tsv` sha256 `e191f619…`(567 genes) +
`Athaliana.swissprot_flags.tsv` md5 `1bf39ce9…`. held-out에서 67 = 67 정확 일치.

### 7. N_pilot ✅

**N_pilot = 4,142 optimizer updates** (effective batch 96 × 198,828 training rows × 2 epochs).
B6·C2는 WP5에서 정확히 4,142 update까지 학습하고 **final-step 체크포인트**를 보관(best-validation
아님). C3/C4/C6도 동일 예산의 final-step. 다섯 arm 전부 그 step에서 **pilot 패널에서만** 비교.
✏️ **C7**(체인 메모리 블록 실행·원소 전부 마스크, 트랙 켬)도 **같은 N_pilot = 4,142**, seed 123, 같은 `aug_seed`, final-step, pilot 패널 — **여섯 arm**. C7은 rule 16 **(ii-c) 보고되는 보조 결과**만 뒷받침(D-C6 저자 확정 "A").

### 8. 기존 artefact 해시 검증 (rule 15 추가 요구) ⚠️

| artefact | 상태 |
|---|---|
| 패널 6종 | ✅ 위에서 재계산 일치 |
| B5 코퍼스 `b5_full_a40_v1` | ✅ **3개 값 전부 실측 일치(09-21)**: 크기 23,534,252,032 B · `geneList` 273,148행 · **md5 `dda7e78c880c993435654b57dcd1c498` 재계산 일치**(pgl-gpu, 47.8 s). rule 15 해시 검증 **충족** |
| 게놈 9종 fasta | ✅ md5 `b5_species_v1.tsv`에 기록 |
| EST 9종 | ✅ md5 `DATASET_ROLES.est_v1.tsv`에 기록 |
| 장독 매니페스트 | ✅ ENA md5 + local md5 양쪽 기록 |
| OrthoDB v12 filtered stage2 | ✅ sha256 `fa0795b6…` (3,178,195,033 B, 12,115,085 seq) — 빌드 자신의 `DB.sha256`과 **일치** |
| Swiss-Prot reviewed Viridiplantae | ✅ sha256 `8d66a18e…` (10,266,723 B, 42,106 seq), UniProt query `(reviewed:true) AND (taxonomy_id:33090)` — 사이드카와 **일치** |
| hard-flag 파일 2종 | ✅ 해시 기록 |

---

## PART 2 — 승인 후로 미뤄도 되는 것 (rule 12가 명시적으로 허용)

C0 테이블 · 트랙 · evidence DB · run identity — **아직 존재하지 않는 artefact의 해시**뿐이며,
각각 **첫 사용 전에 기록**해야 한다. 검정력(power)은 WP5에서 서술적으로 보고하며
**승인 전 계산 불요이고 go/no-go를 막지 않는다**(rule 12 명시).

그 외에는 무엇도 미룰 수 없다: *"No other number, command or rule of A46 may be added or changed
later except by a further amendment."*

---

## PART 3 — 추천

### 추천 1 — B0-2는 **(a) 최종 패널 1,729 loci 국소 정렬**로 간다

이유:
1. **"ONT full 안 한다"와 모순되지 않는다.** full은 전체 run 정렬이고, 이건 패널 윈도로
   사전선별한 reads만 정렬하는 것이다. MVP §2가 이미 *"locus-restricted alignment
   (not the B1 validation pass)"*로 둘을 구분해 놨다.
2. **구조가 이미 구현돼 있다.** `mvp_align_panel_longreads.py`가 50 loci로 하는 일을 패널만
   바꿔 돌리면 된다. full wrapper(종료된 트랙)에 의존하지 않는다 — 확인함.
3. **(b)를 고르면 논문의 핵심 방어선이 사라진다.** B6는 TAIR10 주석을 개선한다고 주장하는데
   평가를 전부 TAIR10으로만 하면 순환논증이다. B1 프로토콜이 애초에 독립 증거를 요구한 이유다.
4. 비용이 full 대비 훨씬 작다. reads 사전선별이 먼저 들어가므로 pass2 입력이 패널 윈도 ±10 kb로
   제한된다.

**단, 비용을 추정치로 승인에 넣지 말 것.** MVP 50 loci를 먼저 돌려 실측 wall-time/scratch를
얻고 1,729로 외삽하라. 지금 프로젝트에 "계획치를 측정치로 오인해 생긴 문제"가 이미 여럿 있다
(16CPU/32GiB/4h, 352 GPU-h).

### 추천 2 — 순서: **A46 개정 → MVP 실행 → 승인**, 그 반대가 아니다

지금 A46을 그대로 승인하면 rule 10이 틀린 채로 동결되고, 고치려면 **또 개정**(A48)이 필요하다.
rule 12의 *"No other number… may be changed later except by a further amendment"*가 그걸 비싸게
만든다. 순서는:

1. **A46 개정안 작성** — B0-1/B0-2/B0-3 세 곳 + A47/A47.1 정합 (rule 17 인용블록은 불가침)
2. **PART 1의 ❌ 6건 채우기** — PROVENANCE 도구 파일이 최우선
3. **MVP 파일럿 실행** — 승인 전 허용된 유일한 작업이고, 여기서 나온 실측이 예산·독립검증 경로를
   동시에 증명한다
4. **그 실측으로 A46 최종 승인**

### 추천 3 — PROVENANCE부터 손대라 (❌ 6건 중 유일하게 지금 막힘 없는 것)

`commands` · `tracks` · `C0` 명령은 설계 결정이 얽혀 시간이 걸리지만, **도구 버전 기록은
순수 실측**이고 아무 승인도 필요 없다. 게다가 rule 9가 본문 대신 PROVENANCE에 위임했으므로
**이게 없으면 승인 자체가 형식적으로 불가능**하다. 컨테이너 안의 torch/transformers/accelerate는
SIF를 한 번 열어 기록하면 끝난다.

### 추천 4 — 승인 문서에 "측정 안 된 것"을 명시적으로 적어라

현재 미측정: B5 코퍼스 원격 재해시, OrthoDB/Swiss-Prot 해시, 컨테이너 내부 라이브러리 버전,
패널 정렬 wall-time/scratch. 승인문에 **빈칸으로 남기지 말고 "미측정"이라고 쓰는 편**이 낫다 —
추정치를 적어두면 나중에 측정치로 읽힌다.

### 추천 5 — *S. bicolor* / *B. distachyon* 장독 0을 승인문에 적어라

9종 중 2종이 장독 증거 없이 학습된다. 이건 결함이 아니라 **데이터 현실**이고 issue #60 저자결정의
결과다. rule 16(Claims)에 종별 증거 불균형을 한 줄 넣어두면 나중에 리뷰어가 짚을 때 이미 답이 있다.

---

## 요약

| 범주 | 상태 |
|---|---|
| Seeds · Thresholds · Panels · N_pilot | ✅ **승인 가능 수준** (✏️ 09-22: C7·체인 슬롯 순열·K = 31·체인 상한 추가) |
| Denominators | ✅ **확정 — final-panel gene loci 내 read-supported junction, ONT 3** |
| Commands | ⚠️ **7/11 동결 + v2 명세 3건.** 7건 동결 유지. v1이 미비를 2건으로 과소보고했고 실제로는 **corpus rebuild·trainer**도 빠져 있었다(독립검토 finding 5). v2가 SPEC-1/1b/2로 명세, ✏️ **K = 31 확정(D1·D2 종결)**. ✏️ 체인 메모리 구성 명령은 **SPEC-2c로 명세**(C0·트랙과 같은 '명세 있음·코드 없음' 상태) |
| Tool PROVENANCE | ✅ **완료 (4파일)** |
| 기존 artefact 해시 | ⚠️ 원격·단백질 DB 미검증 |
| **A46 본문 정합성** | ✅ **해소 — draft revision 5건 적용** |

**남은 것은 2건이다 — 그리고 앞서 '둘 다 기록 작업'이라 한 것은 절반만 맞았다:**
①~~OrthoDB/Swiss-Prot 해시~~ **완료** ②**full C0 빌드 명령 + float16 트랙 래스터 명령의 명세 작성** —
이건 받아적기가 아니라 **쓰는 일**이다. 해당 코드가 main에 없다(`src/transgenic/evidence/` 비어 있음,
`mvp_build_c0lite.py`는 MVP 범위의 C0-lite). rule 15가 산출물은 승인 후로 미루되 **명령은 승인 전
명세로 요구**하므로, 실행이 아니라 명세 작성이라 rule 15 위반은 아니다. 다만 설계 판단이 들어가므로
초안 후 독립검토 권장. 상세: [A46_COMMANDS_FREEZE_20260921.md](A46_COMMANDS_FREEZE_20260921.md)

---

범위: 읽기전용 조사 + 이 문서 작성. 프로토콜·매니페스트·코드 수정 없음, 잡 제출·커밋 없음.
