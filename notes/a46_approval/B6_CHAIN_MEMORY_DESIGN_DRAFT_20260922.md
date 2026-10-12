> 🧊 **대체됨 — [v1](B6_CHAIN_MEMORY_DESIGN_v1_20260922.md)이 정본.** v0는 Kimi 검토 전 원본으로 보존(gather 인덱스가 왼쪽 패딩을 무시하는 등 결함 있음).

# B6 체인 메모리 — 설계 명세 초안 v0 (2026-09-22)

> **상태: DRAFT, 미승인.** 저자 결정 2026-09-22: "롱리드로 엑손 체인 구성을 배우게 하자 → B6에 2번이나 3번을 → (3번 체인 메모리 권고에) 진행하세요".
> 이 문서는 설계 초안이며 코드·프로토콜을 바꾸지 않는다. 독립검토 + Kimi challenge → 저자 승인 → A46 rule 6 개정 → WP1 구현 순서.
> 코드 사실의 출처는 worktree `/tmp/mvp-seed123-integration-20260917` HEAD `729b990` 탐색(아래 §9 인용).

## 1. 목적과 주장

**문제.** rule 6 트랙(K = 31)은 염기별 값이라 두 가지를 잃는다: (a) donor↔acceptor **짝**(엑손 스키핑이 있으면 모호),
(b) 한 분자 안의 **다중 엑손 연결**(A-B-C 동시 관측). 모델은 GSF 라벨에서 "일반적인 체인 문법"은 배우지만,
"이 유전자좌에서 관측된 체인"을 입력으로 받지 못한다. A23은 체인 지지를 평가·QC·B3 재정렬에만 쓴다.

**목표.** 증거 모드에서 디코더가 **관측된 인트론 체인을 참조·조합해 전사체를 구성**하도록 학습시킨다.
주장 (ii)("증거 공급이 주석을 개선한다")의 하위 주장으로, 체인 메모리의 기여를 별도 대조군으로 분리해 잰다.

**버린 대안.** 라벨 경로(롱리드 체인을 학습 목표에 반영)는 증거 조건부가 아니고, 코퍼스 재빌드로 B5 대 B6(C2) 비교에
라벨 차이를 섞는다. 디코더 프리픽스(체인을 GSF 토큰으로 디코더 입력 앞에 붙임)는 좌표가 숫자 한 자리당 토큰 1개라
(`tokenization_transgenic.py:208-212`) 인트론당 ~12 토큰이 들고, v3 디코더 예산 8,192(`configs/b5_400m_win_v3.json`)를 정답과 나눠 쓴다.

## 2. 구조 — 위치 포인터 체인 메모리 + 게이트 교차 어텐션

### 2.1 체인 토큰 (메모리 원소)
체인 = 한 분자(placement)의 **정렬된 인트론 목록**(strand 포함). 메모리 원소는 **(체인, 인트론) 한 쌍당 1개**:

```
v(c, i) = W_d · M[donor_i // 6] + W_a · M[acceptor_i // 6]      # 인코더 메모리에서 위치 gather
        + E_order[min(i, 31)] + E_order_rev[min(n_c-1-i, 31)]    # 체인 내 순서(앞/뒤에서)
        + E_chain_slot[c]                                         # 같은 체인 묶음 식별(슬롯 임베딩, 순서 무의미)
        + E_source[src] + E_strand[s]                             # EST/ONT/PacBio, + / − / ?
        + w_count · log1p(min(units_c, 64))                       # 독립분자 수
```
- `M` = downsample 뒤 메모리 (B, L/6, 1536)(`modeling_HyenaTransgenic.py:749`). **좌표를 숫자로 다시 토큰화하지 않고 서열 표현에서
  위치를 가리킨다** — 메모리에는 downsample 전 사인파 PE가 이미 들어 있어(M:746) 디코더가 좌표 숫자를 내는 데 쓰는 것과 같은 표현이다.
- 단일엑손 분자는 v1에서 제외(인트론이 없으면 연결 정보 없음; 트랙이 이미 coverage로 공급).
- 슬롯 `c`는 체인 순위(아래 2.3)로 부여하되 **슬롯 번호에 의미가 없도록** 학습 중 체인 순서를 매 배치 무작위 순열(aug RNG)로 섞는다.

### 2.2 주입 — 디코더 층마다 게이트 교차 어텐션 (Flamingo식)
기존 LED 디코더 층의 `encoder_attn` 뒤에 새 블록을 둔다:
```
x = x + tanh(α_l) · CrossAttn_l(LN(x), chain_mem, key_padding_mask)      # α_l = 0 초기화
```
- **초기 동작이 B5와 비트 단위로 같다**: `tanh(0) = 0`. 기존 메모리에 붙이는 방식은 0 값 토큰도 softmax 몫을 가져가 B5 동작을 바꾼다(탐색 결과 1).
  `evidence_proj`의 0 초기화(M:431-436, `_is_evidence_projection`)와 같은 원칙.
- 적용 층: **v0 제안 = 12층 전부**(D-C3 저자결정 항목). 파라미터 ≈ 층당 attention 4×1536² ≈ 9.4 M → 12층 ≈ 113 M(400 M 모델의 ~28 %).
  대안: 위쪽 6층만(≈ 57 M), 또는 d=512 저차원 투영 후 어텐션(≈ 12층 × 4×1536×512 ≈ 38 M). **권고: 저차원 512, 12층**.
- 마스크: 체인 없음(소스 dropout, fraction 0, 롱리드 없는 종)이면 모든 원소 마스크 → 블록 출력 0(전부 마스크 시 NaN 방지 처리 필요: 마스크 행이면 출력 강제 0).

### 2.3 체인 집합 구성 (타일 단위)
1. 입력 = 타일에 **유지된** 관측(rule 4 누출 차단 후, rule 7 thinning·source dropout 후). **트랙과 같은 분자 rank(blake2b)** 를 공유해
   트랙과 체인이 항상 같은 분자 부분집합을 본다(`mvp_tracks.py:35-38`의 `f'{aug_seed}\t{species}\t{molecule_id}'`).
2. 분자별 **primary placement**의 인트론 중 **타일 안에 완전히 든 것**만, 좌표 순 정렬. 타일 경계를 넘는 인트론에서 체인을 끊는다(분절 체인은 같은 분자에서 나온 두 체인이 아니라 하나로 유지하되 경계 인트론만 제거).
3. 키 = (strand, 인트론 목록) — 동일 키는 병합, `units` = 독립분자 수(A11/A16 단위).
4. 순위: units 내림차순 → 인트론 수 내림차순 → blake2b(키) 오름차순. **상한 `N_chain` = 64 체인, `N_tok` = 512 (체인,인트론) 원소**;
   넘치면 잘리고 QC에 계수(잘린 체인 수·원소 수).
5. 소스: EST(부분 체인, 대개 1–3 인트론), ONT, PacBio — `E_source`로 구분. **EST 포함 여부는 D-C2.**

### 2.4 RC
창 길이 L, 반개구간 인트론 (s, e) → (L − e, L − s); 체인 내 인트론 순서 역전; strand +↔−, ?는 유지; donor↔acceptor 역할 교환
(트랙 순열 `[1,0,2,5,6,3,4,7]`과 일치, `mvp_tracks.py:103-104`). gather 위치도 교환된 donor/acceptor 기준. **두 번 적용 = 항등**을 시험으로 고정
(`tests/test_gsf_rc.py:12-15` 대응).

## 3. 데이터 경로

- **C0 확장**: 현 C0-lite `est_junction`은 분자별 DISTINCT 합집합이라 placement별 체인이 아니다(`mvp_build_c0lite.py:266-270`).
  새 표 `observation_chain(species, source, molecule_id, placement_id, strand, n_introns, introns_json, units_key)`를
  `member_junction`(`:117`)을 `placement_id`로 묶어 만든다. 롱리드는 A18.5 수용(primary, MAPQ ≥ 20, query fraction ≥ 0.80, identity 기준) 뒤 BAM에서
  `59_evidence_support.parse_sam`의 `introns`(1-based 폐구간, `:117-154`)를 재사용하고 **0-based 반개구간으로 한 번만 변환**.
- **데이터셋/콜레이트**: `isoformDataHyena`(`datasets.py:161-230`) 반환 튜플에 원소 추가(현재 10-튜플 고정: `mvp_runtime._batch_parts` `:212-226`,
  `mvp_metrics.py:67,97-98`, `hyena_collate_fn` `:814-854`). 체인은 **고정 크기 패딩**(`N_tok` = 512) + 마스크 — `torch.compile`(`mvp_cli_runtime.py:292`,
  `suppress_errors=False`)의 정적 형상 유지.
- **생성 경로**: `prepare_inputs_for_generation`(M:1073-1084)이 `last_hidden_state`만 넘기고 M:739-742가 출력 객체를 새로 만든다 →
  체인 메모리를 **`HyenaEncoder.forward` 안에서 만들어 `HyenaModelOutput` 필드로 싣고**(M:204-210) 두 곳을 고친다. "인코더 1회 호출" 계약
  (`mvp_decode.py:136-150` pre-hook) 유지. 빔 확장은 HF가 텐서 필드에 대해 처리.
- **동일성 필드**: `mvp_decode.load_final_model`의 `evidence_channels == 8`, `vocab_size == 290` 검사에 `chain_memory` 설정(차원, 층, 상한) 추가.

## 4. 규칙과의 관계 (A46)

| 규칙 | 영향 |
|---|---|
| rule 6 트랙 | **K = 31 불변.** 체인 메모리는 트랙이 아닌 별도 입력으로 rule 6에 **추가 조항**(정의·상한·RC·게이트) |
| rule 4 누출 차단 | 그대로 — 관측 단위 차단 뒤 남은 분자만 체인이 된다. 분자 전체 폐쇄(molecule-wide)도 동일 |
| rule 7 thinning / dropout | 같은 분자 rank·같은 소스 dropout 공유. fraction 0 → 체인 없음(마스크) |
| rule 8 withholding | 라이브러리 withholding이 트랙과 체인에 동시에 적용 |
| rule 10 입력/채점 분리 | 그대로 — 채점 세트(A-ONT1, A-CCS23, M-FLNC 등) 분자는 입력 체인이 될 수 없다 |
| rule 11 대조군 | **C7 신설: 체인 메모리 끔(트랙은 켬)** — 체인의 순수 기여. 기존 C2(트랙 영구 0)는 체인도 끔 |
| dose-response | 트랙·체인이 같은 fraction을 따르므로 기존 곡선에 체인 효과가 포함. C7이 분리 |
| rule 16 주장 | (ii)에 하위 주장 "(ii-c) 관측 체인 공급이 intron-chain 정확도를 개선" 추가, C7로 검정 |
| WP3 메모리 게이트 | 재측정 필수 — 129 kb 타일에서 기존 메모리 21,504 + 체인 512 원소(별도 블록) |

## 5. 비용 추정 (초안, 측정 전)

- 파라미터: 저차원 512 × 12층 ≈ 38 M + 임베딩 소량 → 400 M 대비 ~10 %.
- 연산: 층당 T_dec × 512 추가 어텐션(기존 교차 어텐션 T_dec × 5,120의 10 %) — 디코더 비용 +수 %.
- 데이터: C0에 placement별 체인 표 1개. 9종 롱리드(ONT + 복원 PacBio 49런) 체인 추출은 CPU 작업.
- 일정: 설계 승인 → A46 rule 6 개정 → 구현(모델·데이터셋·생성경로·시험) → WP3 게이트 재측정. **MVP 포함 여부(D-C1)에 따라 MVP 일정이 달라진다.**

## 6. 시험 (구현 시 필수)

1. 게이트 0에서 B5 체크포인트 출력이 **비트 단위 동일**(체인 유무 무관).
2. RC 두 번 = 항등, 순서 역전·donor/acceptor 교환 정확.
3. thinning 중첩: fraction f₁ < f₂ 이면 체인 집합 ⊆ (units 포함 관계).
4. 소스 dropout·fraction 0 → 전부 마스크, 블록 출력 0, NaN 없음.
5. 누출: 제외 구간에 닿는 분자는 어떤 체인에도 없다.
6. 상한 절단 결정성(동일 입력 → 동일 절단), QC 계수.
7. 생성 경로: 체인 메모리가 빔 전 단계에 동일, 인코더 1회 호출 유지.
8. `torch.compile` 정적 형상 — 체인 수가 달라도 재컴파일 없음.

## 7. 저자 결정 항목

- **D-C1 MVP 포함 여부.** MVP §5.3 결정규칙은 동결 인용문이다. (a) MVP는 트랙만(현행) 유지, 체인은 WP 파일럿부터 / (b) MVP에 체인 팔 추가(탐색 표기).
  **권고: (a)** — MVP의 동결 결정규칙과 50-locus 예산을 건드리지 않는다.
- **D-C2 EST 부분 체인 포함.** 포함하면 체인 원소가 늘고 5′/3′ 편향 부분 체인이 섞인다. 제외하면 롱리드 없는 종(B. distachyon)은 체인 없음.
  **권고: 포함 + `E_source`로 구분**(A. thaliana 입력이 A-EST + A-ONT2라 EST 체인이 실제 입력의 큰 몫).
- **D-C3 층·차원.** 12층 × 512 저차원(권고) / 위 6층 × 1536 / 12층 × 1536.
- **D-C4 체인 끝 정보.** v0는 인트론만. 분자 5′/3′ 끝(TSS/TES 후보)을 넣을지는 보류 — A-CCS23의 TES 제외와 같은 이유로 3′ 끝 신뢰도가 소스마다 다르다.
- **D-C5 상한.** `N_chain` 64 / `N_tok` 512 — WP3 측정 후 확정.

## 8. 위험

- 게이트가 학습 초기에 0 근처에 머물러 체인을 쓰지 않을 수 있다 → α 학습률 별도(예: proj-lr 2e-4와 같은 그룹), C7로 실제 기여를 잰다.
- 롱리드가 풍부한 종(G. max 27런, P. trichocarpa)과 없는 종의 불균형 → rule 16의 종별 증거 불균형 선언에 체인 포함.
- 체인 메모리가 디코더를 "복사기"로 만들어 증거 밖 isoform을 억누를 수 있다 → dose-response와 withholding으로 관찰, 평가에서 신규 isoform 비율 보고.

## 9. 코드 사실 인용 (탐색, HEAD 729b990)

- 메모리: M:746 PE, M:749 downsample → (B, L/6, 1536), M:753 교차 어텐션 마스크 전부 1(패딩 무시), M:764 `encoder_hidden_states`.
- LED 디코더 `LEDDecoderLayer`의 `encoder_attn`은 dense `LEDDecoderAttention`(Longformer 창 무관), 4D 마스크 지원.
- `evidence_proj` = `nn.Linear(8, d_model, bias=False)` 0 초기화(M:431-436), backbone 뒤·PE 전에 더함(M:551).
- 생성: `prepare_inputs_for_generation` M:1073-1084, 재래핑 M:739-742, `_reorder_cache` M:1091-1107.
- 토크나이저 v3 290, 좌표 = 자릿수 토큰, 창 기준 0-based 반개구간(`gsf_contract.py:206-207`); v3 상한 8,192(`gsf_contract.py:33`).
- 트랙: `mvp_tracks.py:14-108`, rank `:35-38`, RC `:103-104`; 데이터셋 `datasets.py:161-230, 336-344, 474-483, 814-854`.
- 창: `gsf_contract.py:16-19` (30,720 / 61,440 / 129,024); MVP 30,720 고정; 배치 1 × 누적 24 × 4 rank.
- 체인 추출 재사용: `59_evidence_support.py` `Alignment`(`:98-114`), `parse_sam`(`:117-154`), `chain_witness`(`:404-415`).
