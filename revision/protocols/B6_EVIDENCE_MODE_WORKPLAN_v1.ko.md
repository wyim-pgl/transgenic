# B6 워크플랜 v1.2 (한글판) — 모델 하나, 모드 둘: de novo(서열만)와 증거 모드(서열 + 정렬된 전사체/단백질 증거)

쉬운 설명: [`B6_WORKPLAN_README.ko.md`](B6_WORKPLAN_README.ko.md). 영문 원본(A46 동결 문안의 기준): [`B6_EVIDENCE_MODE_WORKPLAN_v1.md`](B6_EVIDENCE_MODE_WORKPLAN_v1.md). 두 판이 다르면 영문 원본이 우선합니다.

상태: 워크플랜 v1.2(2026-09-14, Codex 1~3차와 Kimi K3 challenge 반영; Kimi 판정 approve-with-changes, 항목 태그 [K-n]). **착수 규칙: WP1 이후의 모든 작업은 A46이 저자 승인을 받고, 아래의 모든 값·해시·명령·시드·유전자좌 패널·임계값이 기록되며, §4의 미결 사항이 하나도 남지 않은 뒤에만 시작한다** [C3-2]. 저자 결정(2026-09-14): "de novo 모드 + 증거 모드로 방향을 정하고 진행". 이 계획은 2026-09-14의 Codex 적대적 리뷰(1차: per-base 보조 헤드 → reject; 2차: 증거 입력 채널 → "현 명세는 reject, 방향은 교정된 pilot 가치 있음")를 반영한다. 리뷰에서 온 요구사항은 [C1-n]/[C2-n](회차, 지적 번호)로 표시. FREEZE 항목은 어떤 B6 학습 런도 시작하기 전에 A46 개정 문안이 된다.

## 1. 목표와 주장
- B5(시드 123, 학습 중): split만 바꾼 재학습 — 참조 GSF 라벨, 서열만 입력. 누출/문맥 변화의 비교군으로 남고 **변경하지 않는다**(A28).
- B6: 같은 레시피 + 염기별 **증거 채널**(선택). 채널 드롭아웃으로 학습하여 가중치 한 벌이 다음을 지원한다.
  - **de novo 모드**: 채널 전부 0 → 서열만으로 주석; Helixer/Tiberius 및 B5와 비교.
  - **증거 모드**: 사용자가 정렬한 EST/장독/단백질로 채널을 채움 → BRAKER/MAKER/PASA류와 비교.
- 할 수 있는 주장: (i) 증거로 학습한 가중치가 de novo 주석을 사전 선언 마진 이상으로 나쁘게 하지 않는다; (ii) 추론 시 증거를 주면 주석이 좋아지며, 고정 유전자좌에서 dose-response를 측정했다; (iii) 완성 모드(참조 프롬프트)가 두 모드에서 작동한다.
- 하지 않을 주장: B5와 B6의 런타임/결과 중립성(구조·RNG가 다름; 대조군 참조), 애기장대 EST의 독립성(in-domain, 유전자좌 분리) [C1-8].

## 2. 동결(FREEZE)할 정의 (A46)
1. 증거 소스와 역할(A14 유지): 학습 측 = 학습 종 9종의 EST(A21 스크리닝, A37 primary arm ≥ 100 nt; ≥ 121 nt arm은 EST 유래 수치를 보고할 때마다 짝으로 보고하는 민감도 arm), ONT, Sequel II+ PacBio, 단백질 정렬(A19/A43). 절대 금지: Z. mays, S. lycopersicum, A-ONT1, A-HiFi. A9(옵션 J가 B6로 범위에 들어옴)와 A18의 범위 문장을 개정 [C2-13].
2. 관측 스키마 = 전체 C0(IMPLEMENTATION_ORDER_B5_C0_C2_v1.md:34): `evidence_alignment`, `molecule_member`, `aligned_block`(분자 내 순서 있음), `junction_observation`(원시 좌표와 ONT ±3 nt 보정 좌표, A20), `partial_chain`, `partial_chain_intron`, `junction`, `junction_support`, `chain_junction`; 모든 행에 소스, 라이브러리, 유전형 층, 역할(`training_eligible`/`validation_only`), 출처 해시 [C1-9, C2-9].
3. 분자 단위: EST = A11 클론 병합 단위(모든 accession 버전, mate 리드, UniVec 분할 조각 포함); 장독 = A16 단위. 단백질: **지지 계수** 단위는 OrthoDB organism(A19), **누출 폐쇄** 단위는 단백질 쿼리 서열(그 모든 정렬, A43 mapping_ambiguous 포함) [C3-3]. 분자는 어디서든 한 단위다 [C1-7].
4. 누출 배제(raster 이전, 관측 단위) [C1-4, C1-5, C2-3, C2-5]:
   a. 종별 배제 구간: strict held-out 유전자좌 ± 시드 플랭크(A33: 50~150 nt, 유전자좌별 정확한 시드값), test 블록(A29), test orthogroup 유전자와 (train 타일에서는) valid orthogroup 유전자, A22 하드 플래그 유전자, A33 겹침 성분·decoy 마스크 유전자, 타일 가장자리에 걸친 유전자의 `partial` 구간.
   b. 분자 단위의 어떤 구성원이든 어떤 배치(primary, supplementary, secondary 상당, `mapping_ambiguous`)라도 배제 구간과 어느 가닥에서든 겹치면 그 분자는 **모든** 학습 측 트랙에서 제거. 동결된 §3 명령은 `--secondary=no`(est_align.sbatch:144)이므로, 배치는 **배제 전용** 감사 정렬로 복원한다(라벨에는 절대 사용 안 함): §3 명령에 `--secondary=yes -N 50 -p 0.5`를 더한 것 + 보존된 모든 분자에 대해 primary/supplementary 배치의 직접 구간 검사; secondary 배치 ≥ 50개인 분자, 또는 배제 구간 밖에 mapq < 5 배치가 있는 분자는 감사 불가로 보고 배제. 감사의 완전성은 배치 수가 N 미만인 분자 비율로 보고(종별 ≥ 99.9 %, 아니면 N을 올려 재실행) [C2-7, C3-3].
   c. 배제된 유전자와 이웃이 좌표를 공유하는 접합부 지지는 배제 [C1-5].
   d. QC 표(A16 완전성): 종/소스별 — 입력 분자, 원인(a~c)별 배제, 보존; 애기장대는 held-out 유전자좌에 겹치는 분자 = 배제 수(보존은 0이어야 함).
5. DNA 마스크: 타일 빌더의 유전자 포함 마스크는 가장자리에 걸친 held-out DNA를 노출한다(build_b5.py:561-567). B6 코퍼스 빌드는 이를 수리하고(train/valid 타일에서 노출 구간 ± 플랭크 마스킹) **같은 수리로 서열 전용 비교 코퍼스도 재빌드**하여 B5-vs-B6 비교가 동일한 DNA 마스킹을 쓰게 한다 [C2-4]. C1' 동일성 검증: B6 코퍼스와 비교 코퍼스는 **하나의** DuckDB 파일; C1'은 같은 `rn` 순서에서 트랙을 무시하므로 geneList/window_genes/tile_blocks는 구성상 바이트 동일; 동결 JSON은 둘에 하나의 sha256을 기록하고 트레이너는 C1'에 `evidence_channel=off`를 로그 [C3-3]. 시드 123(수리 전 코퍼스로 이미 학습 중)은 그대로 보고하고 노출량을 수치화한다.
6. 트랙(염기별, float16, 보존된 관측에서 타일별로 빌드; 관측→타일 소속을 유지해 솎아내기가 재raster 가능) [C2-3]. 솎아내기 규칙(동결): 보존된 각 분자에 `rank = int.from_bytes(blake2b(f"{aug_seed}\t{species}\t{molecule_id}".encode("utf-8"), digest_size=8).digest(), "big") / 2**64`(hashlib.blake2b, 고정 바이트 인코딩 — 프로세스마다 salt가 달라 재현이 깨지는 Python 내장 `hash()`는 절대 사용 안 함 [K-3a]); 비율 f의 추출은 rank < f인 분자를 남기므로 구성상 nested; 동률은 불가능하고 반올림 없음 [C3-3]:
   - 소스(EST, ONT, PacBio) × 가닥별: 전사 블록 커버리지 log1p(분자 수), log1p(64)에서 클립;
   - 소스 × 가닥별: donor/acceptor 지지 log1p(분자 수), 경계 염기에 배치;
   - 소스별 unknown-strand 커버리지(방향 없는 EST);
   - 단백질 CDS 커버리지 log1p(organism 수)(phase 채널은 보류 [C2-Q5]);
   - 소스별 availability 마스크(그 종/라이브러리에 해당 소스가 있으면 1; 드롭아웃 시 소스와 함께 제거).
   참조 유래 특징 금지(주석 기반 접합부 스냅 금지, 참조 phase 금지), 타일 단위 정규화 금지 [C2-Q5]. A18.4 가중치는 입력 배수로 쓰지 않음; 계수와 유전형/QC 플래그는 별도 채널.
   RC 타일: 좌표 반전; donor/acceptor 정체성은 유지, 가닥 채널만 교환(tests/test_gsf_rc.py:26) [C2-new4].
7. 모델: `E = E_seq + Linear_noBias(K → d)(tracks)`를 HyenaDNA의 염기별 상태에 다운샘플링/skip 경로 이전에 적용(modeling_HyenaTransgenic.py ~725) [C2-Q1]. 디코더, 토크나이저, GSF 타깃, 문법 제약 디코딩(A24), 타일링/스티칭(A27) 불변. 트랙 0 ⇒ 정확히 서열 전용 경로.
8. 학습: 손실 = L_GSF만. 채널 드롭아웃 p = 0.5(타일별, 동결). 유지 시 보존 분자 비율을 {0.01, 0.1, 0.5, 1}에서 균등 샘플링, 별도 증강 RNG 스트림(`aug_seed` = 20260914 + seed; 기록); 유지 타일의 0.1 비율로 소스 withholding(소스 하나 전체 제거), 0.1 비율로 라이브러리 withholding [C3-2]. 정지 규칙: de novo 모드 검증 L_GSF 최소, patience 3, 상한 22(A18.6); 그 best 체크포인트 하나를 두 모드에 사용; 증거 모드 검증 L_GSF는 진단용 로그 [C2-Q4].
9. 런타임 정체성 추가: 증거 DB sha256, 트랙 명세 해시, p, 솎아내기 집합, `aug_seed`, withholding 비율, 도구 버전과 정확한 명령(minimap2, samtools, blastn, DuckDB, torch, accelerate), compile 정책; resume 화이트리스트 확장(src/transgenic/training/b5_runtime.py:224). compile 폴백 정책: Dynamo가 eager로 폴백하면 런 **실패**(트레이너는 경고가 아니라 예외를 내야 함 — train_HyenaTransgenic.py:317은 현재 경고만); 메모리 중단 상한: 어떤 rank든 reserved 110 GB 초과 시 링크 중단(마커 `MEMORY_ABORT`) [C3-2, C3-3].
10. 검증 증거 배분 [C2-new1, CRITICAL]: 모든 검증 유전자좌에서 모델 **입력**으로 주는 증거와 **채점**에 쓰는 증거는 서로 소인 집합이며, 추론 전에 선언하고 DATASET_ROLES(`b6_input`/`b6_score` 열, 해시)에 기록. 역할 전이 규칙 [C3-4]: 학습 종 분자는 정확히 세 상태 중 하나 — `train_track`(학습 트랙에 보존), `excluded_heldout`(배제 구간에 닿아 학습에서 제거), `never`; 입력 가능 소스(A-EST, A-ONT2)의 `excluded_heldout` 분자**만** 그것이 닿는 held-out 유전자좌에서 입력으로 줄 수 있고, 어떤 유전자좌에서 입력으로 쓴 분자는 그 유전자좌의 채점에 절대 쓰지 않는다. Z. mays — 입력 = M-EST + M-HQ18(Wang 2018 HQ isoform); 채점 = M-FLNC(Wang 2020) + root-tip ONT(PRJNA822071), tier 1; 입력 집합은 절대 채점하지 않음. M-EST와 RefGen_V4 유래 전사체 참조는 참조 주석과 독립이 아니므로 RefGen_V4 대비 F1은 `reference-dependent`로 표기; 독립 결과는 채점 집합에 의한 완전 체인 지지(§9 P1~P5). 애기장대 strict held-out 유전자좌 — 입력 = `excluded_heldout` A-EST + A-ONT2(in-domain, 유전자좌 분리); 채점 = A-ONT1 + A-HiFi(검증 전용). 유전자좌 패널: pilot 패널(WP5)과 최종 채점 패널은 held-out 유전자좌에서 시드 20260914로 한 번에, 서로 소로 추출하고 id를 WP5 전에 A46에 동결 [C3-4]. 패널 크기는 A46에 기록하는 최소 검출 효과 계산으로 고정(목표: 짝 유전자좌 bootstrap으로 transcript-F1 1.0 pt 차이를 80 % 검정력으로 검출; 유전자좌 수준 분산은 시드 123 epoch 체크포인트에서 측정); 그 검정력에 못 미치는 패널의 pilot은 유효한 no-go가 아님 [K-2b].
11. 대조군(사전 선언, 같은 코퍼스·같은 초기화·같은 데이터 순서·별도 증강 RNG) [C1-3, C2-Q2]:
   - C1: B5 시드 123(기존, 수리 전 마스크 코퍼스) — 역사적 비교군;
   - C1': 수리 코퍼스의 서열 전용 런 — **실행하지 않음(저자 결정 2026-09-14: C1이 유일한 서열 전용 비교군; 수리 전/후 코퍼스의 DNA 마스크 차이는 수치화해 보고)**;
   - C2: B6 구조에 트랙 영구 0(p = 1), **전체 길이**;
   - C3: 증거 셔플(같은 티어의 다른 타일 트랙), 같은 예산;
   - C4: presence-only 트랙(이진), 같은 예산;
   - C5: 증거 전용 재구성 기준선(Protocol M splice-graph + ORF 규칙, 모델 없음), 검증 유전자좌에서.
12. Dose-response [C2-new2]: 고정된 검증 유전자좌 패널에서 증거를 nested 부분집합으로 솎아(비율 1, 0.5, 0.1, 0.01, 0; 각 5회 무작위 추출, 모드 간 같은 추출) 재raster; 비율별 지표와 짝 유전자좌 bootstrap 구간. 커버리지 층 요약은 기술 통계일 뿐.
13. 게이트(사전 등록, A46 동결) [C2-Q7, C3-5]: 지표 = GFFCompare v0.12.6 exact intron-chain 일치(단일 exon: exact CDS)의 전사체 수준 F1, 동결된 최종 패널(§2.10) 전체 pooled, 분모 = recall은 패널 유전자좌의 모든 참조 전사체, precision은 모든 예측 전사체, 모든 arm에 동일; 짝 유전자좌 bootstrap 2,000회, 시드 20260914, percentile 95 % 구간. (G1) de novo 모드: (B6 − C2) 하한 ≥ −1.0 pt. (G2) 증거 모드: (증거 − de novo) 하한 > 0 **그리고** (증거 − C5) 하한 > 0; C5는 같은 패널·같은 입력 증거·같은 분모로 평가(C5가 아무것도 내지 않는 유전자좌는 제외가 아니라 recall 0). (G3) 문법: 예측 전사체 1,000개당 `validate_gsf` 위반 ≤ B5 값 **그리고** 같은 모드에서 전사체 recall ≥ 0.9 × B5 recall — 빈 출력이나 최소 출력으로는 통과 불가. 중단 규칙: 누출 QC 수 ≠ 0, replay/RC/resume 결정성 실패, Dynamo 폴백, 메모리 중단. G1 또는 G2에 실패한 런은 보고하되 본문으로 올리지 않는다. abstract 대표 모델: 런 전에 저자가 A46에서 결정.
14. 보고: EST 유래 수치마다 A37 짝 arm; B1 시대의 모든 표에 두 모드; dose-response 그림; 완전성/누출 QC 표; 학습 커버리지 vs 테스트 커버리지 분포.

## 3. 작업 패키지, 순서, 추정

**WP-MVP(WP0 전, 탐색, 1주, Protocol M 방식으로 표기 — 사전 등록 arm이 아님)** [K-4]: 증거 채널이 효과가 있기는 한가? 스크리닝된 EST BAM/PAF가 있는 학습 종 하나(제안: 애기장대, EST 최다), 손으로 만든 입력/채점 분리(§2.10 규칙을 수작업 적용, 입력 A-EST, 채점 A-ONT1/A-HiFi)의 strict held-out 유전자좌 약 20개, 기존 체크포인트(시드 123 `best`, epoch 4, 4-GPU 26.08)에 채널을 얹어 짧은 fine-tune(고정 스텝 예산, p = 0.5), dose-response는 비율 {0, 1}만, 시드 3개. 전체 C0 없음, 감사 정렬 없음, compile/DDP 하드닝 없음, 다종 QC 없음. 결과: 명확한 0 또는 음의 효과면 WP0/WP1 전에 프로그램 중단; 양의 효과는 WP0~WP7을 정당화하되 대체하지 않음. 예산 ≈ 3 × 6 GPU-h + 평가. 이 유전자좌는 이후 동결 패널에서 제외.

| WP | 내용 | 의존 | 추정 | 검증 |
|---|---|---|---|---|
| WP0 | 모든 FREEZE 값을 채운 A46 문안(§2 + §4 결정 종결), DATASET_ROLES `b6_input`/`b6_score` 열, 유전자좌 패널 추출·해시, 결정 기록; **저자 승인; 그 전엔 아무것도 시작 안 함** | — | 1일 | 개정 문안 Codex 리뷰; 저자 서명 |
| WP1 | 전체 C0 파이프라인: BAM/PAF(+ secondary 포함 감사 정렬) → 종별 DuckDB C0 표; 분자 단위(A11/A16); 접합부 보정(A20); 역할 플래그; 출처; QC 표 | WP0 | 4~5일 | 합성 BAM/PAF fixture로 pytest(§2.3~2.4의 모든 규칙); diff Codex 리뷰; pronghorn CPU 배열로 11종 실행 |
| WP2 | 누출 배제 + DNA 마스크 수리: 종별 배제 구간 빌더(A29/A31/A33/A22, held-out 플랭크, `partial` 구간), 분자 단위 배제, 수리된 타일 빌더; 코퍼스 재빌드(A40 파이프라인) → `b5_full_b6_v1.db` + 짝 맞춘 서열 전용 코퍼스(같은 DB, 트랙 무시) | WP1 | 2일 + 빌드 1일 | 검증기: held-out 유전자좌에 보존 분자 0; 마스크 구간 제외 geneList 바이트 동일; 동결 JSON |
| WP3 | 트랙 + 솎아내기: 관측→타일 소속 표, raster(float16 [L,K]), RC 처리, nested 솎아내기 생성기, 데이터셋/collate(DNA와 함께 좌측 패딩, datasets.py:728), 사이드 DB. **융합 L×d 경로의 129 kb 메모리 벤치(K채널, 투영, compile + non-reentrant 체크포인팅, 4 rank)는 WP3 끝에 실행하며 WP4의 게이트: peak reserved > 100 GB면 트레이너 코드를 쓰기 전에 K 또는 융합 지점을 재설계** [K-2c] | WP2 | 2일 | 단위 테스트(RC 대합, 솎아내기 nested, 패딩), 메모리 벤치 보고 |
| WP4 | 모델/트레이너: bias 없는 투영, 자체 RNG의 드롭아웃/솎아내기 일정, 런타임 정체성 필드, resume 화이트리스트, 두 모드 검증; 생성이 캐시/beam 디코딩에서 트랙을 전달해야 함(`prepare_inputs_for_generation`, modeling_HyenaTransgenic.py:1050-1060은 고정 키 집합 반환; 인코더는 `encoder_outputs is None`일 때 한 번 실행, :713-720) — WP4는 트랙 있는 beam/greedy 출력이 매 스텝 재인코딩하는 참조 구현과 같음을 명시적으로 시험하여 KV 캐시 버그가 증거를 조용히 0으로 만들지 못하게 함 [K-2a]; Dynamo 폴백 시 **실패**하는(train:317) compile/DDP 4-rank 통합 시험: 트랙 0·비0, 캐시 생성 parity, RC 타일, resume 결정성 | WP3 | 2~3일 | 단위 테스트 + DeltaAI 통합 잡 [C3-3] |
| WP5 | Pilot: 4-GPU, 고정 스텝 예산(≈ 2 epoch), 수리 코퍼스에서 B6 vs C2; 동결된 PILOT 유전자좌 패널(최종 패널과 소)에서 분리된 증거 배분으로 G1/G2 측정; go = pilot 패널에서 두 하한이 임계값 초과, 중단 규칙 미발동 | WP4 | ≈ 3 × 25 GPU-h + 큐 | pilot 보고; go/no-go [C3-4] |
| WP6 | 본 런: B6(11 h 미리 큐잉 체인, A45), C2 전체 길이; C3/C4 같은 예산 짧은 런; C5 기준선 | WP5 go | 2 × 400 GPU-h + 짧은 런 2개; 잔액 대비 예산 점검 | A28/A45 체인 규칙 |
| WP7 | 평가: 두 모드 × 모든 B1 시대 벤치마크, dose-response, 게이트, A37 arm; 완성 모드 추가 전사체의 B1 검증(C0 표 재사용) | WP6 | 3~4일 | 사전 등록 표 |

임계 경로 ≈ WP0~WP4 ≈ 2주, pilot ≈ 큐 포함 3~4일, 본 런 ≈ 큐 포함 1~2주, 평가 1주 → ≈ 5~6주. GPU 예산(저자 결정 2026-09-14: C1' 생략): 잔액 1,433 − 시드 123 사용 ≈ 200 − 시드 123 잔여 ≈ 250 = 가용 ≈ 983; pilot ≈ 75; 본 런 B6 + C2 ≈ 800; C3 + C4 ≈ 60 → 필요 ≈ 935, 여유 ≈ 50 GPU-h. 초과(예: B6가 10 epoch을 넘김)가 생기면 WP6 계속 전에 보충이 필요하다.

## 4. 저자가 아직 정해야 할 것
1. ~~예산: C1' + C2 본 런을 위해 보충할지, C1(시드 123)을 유일한 서열 전용 비교군으로 받아들일지.~~ **결정(2026-09-14): C1(시드 123)이 유일한 서열 전용 비교군; C1'은 실행하지 않음.**
2. 대표 모델(B6 de novo 모드 / 증거 모드 / B5) — WP6 전에 A46에 있어야 함.
3. Z. mays 검증 증거 배분: 어느 ONT/PacBio 라이브러리가 입력이고 어느 것이 채점인지(§2.10) — 제안: 입력 = M-EST + Wang 2018 HQ(M-HQ18); 채점 = M-FLNC(Wang 2020) + root-tip ONT(PRJNA822071).
4. ≥ 121 nt EST arm을 입력 트랙 arm으로도 쓸지(트랙 빌드 2배) 아니면 채점 표에만 쓸지(제안: 채점만).

## 5. 바뀌지 않는 것
시드 123 체인(B5)은 계속된다; B1 검증 프로토콜(§3~9)은 증거 배분 열 외에는 불변; GSF 문법, 토크나이저, 타일링, 디코딩, 스티칭 규칙(A24~A27) 불변.
