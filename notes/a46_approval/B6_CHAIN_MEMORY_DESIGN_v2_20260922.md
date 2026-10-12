# B6 체인 메모리 — 설계 명세 v2 (2026-09-22)

> ✏️ **2차 검토(2026-09-22, 독립검토 + Kimi, A46 전체) 반영은 A46 rule 6a 본문이 정본이다** — 평가 창 구성(`b6_input` 행, rule 4 분할 없음), Z. mays 체인 = M-EST만(M-HQ18 RS II는 트랙만), 상수(12층·Linear 1536→512·순서 상한 31·units 스칼라·블록 dropout 0.1·α는 B5 레시피 학습률), 해시 인코딩, 통째로 버리기, (ii-c) 통계. 아래 v2 본문과 다르면 rule 6a를 따른다.
> **정본(1차).** v2 = [v1](B6_CHAIN_MEMORY_DESIGN_v1_20260922.md)(Kimi proceed-with-changes 10건) + **독립검토 APPROVE_WITH_FIXES**(CRITICAL 1 · HIGH 5 · MEDIUM 5 · LOW 5) 반영.
> 저자 결정 2026-09-22: 롱리드 엑손 체인을 B6 입력으로(3번), "수정 후 진행", "계속진행". **DRAFT — 코드 무변경.** 프로토콜 반영안은 §10(A46 개정문 초안).
> 코드 사실: worktree `/tmp/mvp-seed123-integration-20260917` HEAD `729b990`(M = `src/transgenic/model/modeling_HyenaTransgenic.py`).

## 1. 목적

rule 6 트랙(K = 31)은 염기별이라 donor↔acceptor 짝과 한 분자 안의 다중 엑손 연결을 잃는다. 체인 메모리는 **관측된 인트론 체인**을
디코더가 직접 참조하게 해, 증거 모드에서 전사체 구성을 증거에 조건화한다. 라벨 경로·디코더 프리픽스는 v0 §1의 이유로 버렸고,
A23 기반 **추론 재정렬**은 학습된 팔이 실패할 때의 대체 경로로 남긴다(별도 사전등록 필요).

## 2. 구조

### 2.1 원소 = (체인, 인트론) 1개
```
v(c,i) = W_d · M[idx(d_i)] + W_a · M[idx(a_i)]                 # 인트론 안쪽 경계 염기에서 gather
       + E_order[min(i,31)] + E_order_rev[min(n_c-1-i,31)]      # 체인 내 순서(5'→3' 전사 방향 기준)
       + E_slot[π(c)] + E_source[src] + E_strand[s ∈ {+,−}]
       + w_count · log1p(min(units_c, 64))
```
- **gather 위치 = 인트론 안쪽 경계 염기**: `donor0`, `acceptor0` — C0 `est_junction`의 정의 그대로(`mvp_build_c0lite.py:266-268`:
  `+` 가닥 donor0 = intron_start0, acceptor0 = intron_end0; `−`는 반대; intron_end0는 포함 끝). 반개구간 끝(e)으로 gather하면 e = L에서 범위를 벗어난다(검토 #2).
- **`idx(p) = (pad + tok_offset + p) // 6`**: 콜레이트가 왼쪽 패딩(`datasets.py:835-842`). 검토 #13: M[k]의 수용영역은 입력 [6k−2, 6k+6]이라
  p//6이 그 안에 든다. HyenaDNA는 인과적이므로 gather된 상태는 **상류 서열만** 본다 — 그래서 donor와 acceptor 양쪽을 모두 gather한다.
  대응은 수치 시험으로 고정(§7-9).
- **`?` 가닥 체인은 v2에서 제외**(검토 #3): donor/acceptor가 정의되지 않고, 트랙도 `?` 접합을 버린다(`mvp_tracks.py:90`). 비배향 ONT의 체인은
  C0가 판정한 전사 가닥(A39 FLAG × ts:A)이 있을 때만 들어온다.
- 단일엑손 분자 제외(트랙 coverage가 공급).
- **슬롯 순열 π**(검토 #9): `blake2b(aug_seed, rn, epoch)`에서 유도 — 증강 RNG 스트림(`datasets.py:476-478`)을 소비하지 않아 B6/C2/C7의
  dropout·fraction 추첨 순서가 같게 유지된다.
- 치수: 블록 내부 d = 512, **8 헤드 × 64**(디코더 6 헤드로는 512가 나누어지지 않음, 검토 #14). 파라미터: 12층 × (4 × 1536 × 512) ≈ 38 M + W_d/W_a 2 × 1536 × 512 ≈ 1.6 M
  (W_d/W_a도 512로 투영; 1536×1536이면 +4.7 M) + 임베딩.

### 2.2 주입 = 게이트 교차 어텐션, 디코더 층마다
- LED 디코더는 **post-LN**(modeling_led 1054-1069). 새 블록은 각 층의 `encoder_attn_layer_norm` **뒤**에 둔다:
  `x = x + tanh(α_l) · m · Proj_out(CrossAttn(Q(x), K(chain), V(chain), key_padding_mask))`, 이후 새 LayerNorm은 두지 않는다.
- **`m = any(mask)` 곱셈 게이트**(검토 #7): 원소가 하나도 없으면 블록 기여가 **구조적으로 0** — `tanh(α)`가 0이 아니게 된 뒤에도
  증거 없는 모드(de novo)·C2·fraction 0 셀에 학습된 "체인 없음" 편향이 새지 않는다. HF의 finfo.min 가산 마스크는 전부 마스크일 때 NaN이 아니라
  균등 어텐션을 주므로, 곱셈 게이트가 필요하다.
- **α는 원시 `nn.Parameter`**, 0 초기화, **weight decay 제외**, 별도 학습률 그룹. `initialize_weights`가 디코더 내 모든 `nn.Linear`를 Xavier로 재초기화하므로
  (M:886-896) 출력 투영 0 초기화에 기대지 않는다(검토 #16).
- **캐시하지 않는다**(검토 #4): LED 층 캐시는 `past_key_value[:2]`/`[-2:]` 위치 슬라이스로 읽히고 `_reorder_cache`(M:1091-1107)도 위치로 자른다.
  체인 K/V는 매 스텝 재계산(원소 ≤ 512라 비용 작음).
- **DDP**: 고정 크기 패딩의 0번 슬롯에 항상 마스크된 더미 원소 → 분기가 항상 실행(`find_unused_parameters=True`도 이미 강제, `mvp_runtime.py:65-68`).
- **C7 구현 = 블록 실행 + 전부 마스크**(블록 생략 아님) — 드롭아웃 RNG 소비가 B6와 같아진다(검토 #7).

### 2.3 데이터 흐름 — M이 존재하는 곳에서 gather (검토 #4)
- M은 `transgenicModel.forward`에만 있다(PE M:746, downsample M:749). `HyenaEncoder.forward`(M:531-560)는 원시 h(B, L, 768)만 본다.
- 따라서: 원시 **체인 인덱스·특징 텐서**(`chain_pos (B,N,2)`, `chain_feat (B,N,F)`, `chain_mask (B,N)`)를 입력으로 받아
  **M:749 직후 gather**하고, 디코더에 넘긴다. LED `LEDDecoder.forward`와 체크포인트 호출(modeling_led 2117-2128)은 추가 인자를 받지 않으므로
  **디코더 루프를 서브클래스**(각 층 호출 사이에 체인 블록 실행)로 구현한다. NGC 26.08 컨테이너의 transformers 버전(4.57.6)에서 같은 위치를 재확인할 것.
- 생성: 체인 텐서를 `HyenaModelOutput`의 필드로 싣고 `prepare_inputs_for_generation`(M:1073-1084)과 재래핑(M:739-742)에서 **명시 전달**,
  빔 확장도 명시 구현. "인코더 1회 호출" 계약(`mvp_decode.py:137-150`)은 유지.
- `torch.compile` 정적 형상: N_tok = 512 고정 패딩.

### 2.4 체인 집합 구성 (타일 단위)
1. **소스 테이블 = 동결된 C0 체인 표**(rule 2: `partial_chain`/`partial_chain_intron`/`chain_junction`, ONT ±3 nt 보정 좌표; 검토 #10).
   새 `observation_chain` 표를 만들지 않는다. 좌표 규약: `member_junction`은 0-based·포함 끝(`mvp_build_c0lite.py:117,155`),
   `parse_sam`은 1-based 폐구간(`59_evidence_support.py:147`) — C0 적재 시 한 번만 변환하고 시험으로 고정.
2. **체인 단위 = 멤버 placement 1개의 인트론 목록**, `units` = 독립분자(A11/A16) 수로 병합 계수. 분자는 clone 병합 성분이라 placement가 여러 개일 수 있다
   (`mvp_build_c0lite.py:194-226`).
3. **누출 차단**(검토 #5):
   - 학습: rule 4 뒤 유지된 관측만. **rule 4(c)로 제외된 접합(제외 유전자와 이웃이 공유)을 만나면 체인을 그 자리에서 끊는다** — 가운데
     인트론만 빼면 가짜 인접이 생긴다. rule 4(a)의 타일별 제외 구간(가장자리 `partial` 포함)도 타일마다 적용.
   - 평가 입력: rule 10의 **`excluded_heldout` 모집단**에서만(`mvp_tracks.py:19-20,41`의 `population`). A. thaliana = A-EST + A-ONT2만.
     채점 세트(A-ONT1, A-CCS23, M-FLNC, root-tip ONT) 분자는 어떤 체인에도 들어가지 않는다.
   - 타일 경계를 넘는 인트론에서 체인을 끊는다.
4. **thinning·dropout**(rule 8): 트랙과 **같은 분자 rank**(blake2b `(aug_seed, species, molecule_id)`; 코드는 현재 `'Athaliana'` 고정 — `mvp_tracks.py:36`,
   B6 본 구현은 종 변수 사용)·**같은 소스 dropout**·같은 withholding.
5. **2단계 채우기**(Kimi): 롱리드(ONT·PacBio) 체인 먼저 — units ↓, 인트론 수 ↓, blake2b(키) ↑ — 남은 예산에 EST. 상한 **N_chain 64 / N_tok 512**(0번 = 더미).
   **중첩성**(검토 #8): fraction f₁ < f₂ 에서 체인 집합의 포함 관계는 **절단 전**에 성립한다. 절단 후 포함이 깨질 수 있음을 인정하고,
   절단된 원소 수를 소스별로 QC에 기록한다.

### 2.5 RC (검토 #2 — v0/v1 규칙은 틀렸다)
트랙 순열 `[1,0,2,5,6,3,4,7]`은 **가닥을 바꾸고 역할(donor/acceptor)은 유지**한다(+donor 3 → −donor 5). 체인도 같다:
- 위치: 각 경계 염기 p → **L − 1 − p**. donor는 donor로, acceptor는 acceptor로 남는다(역할 유지).
- 가닥: + ↔ −.
- 순서: `E_order`는 **전사 방향 5′→3′** 기준이므로 RC 후 가닥이 바뀌면 같은 인트론이 같은 순서 번호를 유지한다(좌표 순서만 역전).
- 두 번 적용 = 항등. `gsf_contract.reverse_complement`(`:338-347`)의 구간 사상과 정합.

## 3. 비트 동일성 — 범위를 한정한다 (검토 #7)
- **순전파**: 원소 없음 → `m = 0` → 기여 0 (학습된 α와 무관). 원소가 있어도 α = 0이면 `tanh(0) · finite = 0`.
- **시험 조건**: eager, fp32, eval 모드. `torch.compile` 커널 융합과 bf16에서는 비트 동일을 주장하지 않는다(허용오차 명시).
- **학습**: 새 블록의 드롭아웃이 CUDA RNG를 소비하므로 B6와 "블록 없는" 모델은 학습 궤적이 갈린다 → C7은 블록 실행·전부 마스크로 구현해 RNG 소비를 맞춘다.
  디코더 layerdrop 0.1(`configuration_transgenic.py:71`)이 켜져 있으면 층 단위로 블록도 함께 떨어진다(같은 층 객체 안에 둔다).

## 4. 규칙과의 관계 (§10 개정문이 정확한 문구)
체인 메모리는 **rule 6 "Tracks" 제목 밖**의 별도 입력이다 → rule 6의 K = 31 문장과 rule 17 §5.3 인용은 **불변**.
그러나 트랙만 언급하는 조항들이 체인을 덮지 못하므로(검토 #1, CRITICAL) **decision 1, A18 문장, rules 2, 7, 8, 9, 11, 13**에 명시 문구가 필요하다.
de novo 모드·C2·fraction 0 셀은 **정의상 체인도 전부 마스크**.

## 5. C7과 하위 주장 (ii-c) (검토 #6 · Kimi #2)
- **✅ 저자 확정(2026-09-22, "A"): C7 = 체인 메모리 블록을 가진 채 전부 마스크로 학습한 팔, C3/C4/C6과 같은 N_pilot = 4,142 update·파일럿 패널·
  seed 123 초기화·같은 aug 스트림.** 파일럿 규모 팔은 rule 16 주장을 떠받치지 못하므로 **(ii-c)는 "보고되는 보조 결과"로 등록**하고 주장 목록에 넣지 않는다.
  판정 지표는 rule 13 지표를 **독립 outcome**(rule 10 채점 세트의 complete-chain 지지)으로 계산하고 `reference-dependent` F1은 병기만.
- **대안(비용 큼)**: C7을 최종 패널 전체 기간 팔로 등록해 (ii-c)를 주장으로 — rule 13 paired locus bootstrap, G2 옆 다중성 처리, 추가 GPU-h ≈ B6 본 학습 1회분.
  460de14 기준 예산이 "간신히 맞는" 상태라 권고하지 않는다.
- 체인 학습 모델에서 **추론 시 체인만 마스크**한 결과는 "의존도"로 병기(판정 근거 아님). 층별 α 로그, "α ≈ 0 유지"는 사전 선언된 가능한 결과.

## 6. MVP (D-C1(a) 확정 권고) — 검토 #11
MVP 파일(`mvp_runtime`, `mvp_metrics`, `mvp_decode`)과 `src/transgenic`·`configs`는 `tests/test_mvp_final_source_integration.py`가 681c5a0에 바이트 핀한다.
**체인 메모리는 MVP 파일을 건드리지 않고 WP1/WP4 트레이너 브랜치에서 별도 구현**한다. MVP는 트랙만(현행) 유지 — rule 17·MVP 문서와 정합.

## 7. 시험 (구현 필수)
1. 원소 없음 → 블록 기여 0, 학습된 α에서도(eager fp32 eval, 분기 유무 비트 동일).
2. α = 0 초기화 → B5 체크포인트 출력 비트 동일(eager fp32 eval).
3. RC 두 번 = 항등; 역할 유지·가닥 교환·p → L−1−p; `E_order` 전사방향 불변.
4. thinning 중첩(절단 전); 절단 계수 QC.
5. 누출: rule 4(a)(b)(c) 제외 분자·접합 없음, 4(c) 접합에서 체인 분할, 채점 세트 소스 없음, 평가 입력은 `excluded_heldout`만.
6. 2단계 채우기: EST가 많아도 롱리드 먼저.
7. 생성: 체인 텐서가 모든 빔·모든 스텝에서 동일, 인코더 1회, 체인 블록 무캐시로 `_reorder_cache` 무영향.
8. `torch.compile` 재컴파일 없음(고정 N_tok).
9. gather: 패딩이 다른 샘플에서 같은 염기 → 같은 M 인덱스; 수용영역 수치 대응.
10. DDP: 체인 없는 rank 혼재 시 정상.
11. 슬롯 순열이 aug RNG를 소비하지 않음(B6·C2·C7의 fraction 추첨 순서 동일).
12. α가 weight decay 그룹 밖.

## 8. 비용
파라미터 ≈ 40 M(400 M의 ~10 %). 체인 어텐션 층당 ≈ 6 × 8,192 × 512 점수 ≈ 50 MB(bf16) — 129,024 nt 타일의 기존 메모리 21,504 대비 무시할 수준,
WP3 100 GB 게이트 영향 미미(재측정은 필수). "+10 %" 연산 추정은 30,720 nt 타일에서만 성립. 디코더 8,192 토큰 예산 무영향.
C7(기본값) = C3/C4/C6 한 팔과 같은 비용.

## 9. 저자 결정 항목 (적용한 기본값)
- D-C1 MVP 제외 — **적용**(§6).
- D-C2 EST 부분 체인 포함 + 롱리드 우선 — **적용**.
- D-C3 12층 × d 512(8×64) — **적용**.
- D-C4 체인 끝(TSS/TES) — **보류**(v2는 인트론만).
- D-C5 상한 64/512 — ✅ **동결**(2026-09-22 저자 "진행해"); WP3 게이트 실패 시 후속 개정으로만 변경(A46 rule 6a·15).
- **D-C6 (신규) C7 규모 / (ii-c) 지위** — ✅ **저자 확정 2026-09-22 "A"**: 파일럿 규모 C7(N_pilot 4,142) + (ii-c) 보고되는 보조 결과(§5).

## 10. A46 개정문 초안 (DRAFT에 넣을 문구)
**적용 완료(2026-09-22, A46 DRAFT):** `PROTOCOL_B1_frozen_v1.md` — 신규 **rule 6a**(체인 메모리 정의·구성·상한·주입·RC, MVP 미사용, K = 31 불변) +
decision 1(de novo = 트랙 0 **그리고 체인 전부 마스크**) · A18 문장(체인 메모리 포함) · rule 7(게이트 블록 예외, 비트 동일성은 eager fp32로 한정) ·
rule 8(같은 분자 rank·draw, 슬롯 순열은 aug 스트림 비소비) · rule 9(체인 spec 해시·층별 α 기록) · rule 11(C2 체인 마스크, C3 체인 동반 교체, **C7 신설**) ·
rule 13(dose-response에 체인 동반) · rule 16((ii-c) 보고되는 보조 결과, 주장 아님) · draft revision 기록.
검증: rule 17 줄 바이트 동일, §5.3 인용 정규화 동일(1,958자), rule 6 K = 31 문장 존재. 변경 전 사본: scratchpad `PROTOCOL_before_chainmem.md`.

## 11. 검토 기록
- Kimi(ask, job `1dc0c0a3…`; challenge 모드는 `spawn E2BIG`로 실행 불가): proceed-with-changes, 10건 → v1.
- 독립검토: APPROVE_WITH_FIXES. 반영: #1 조항 확장(§4, §10) · #2 RC(§2.5) · #3 `?` 제외 · #4 데이터 흐름(§2.3) · #5 누출(§2.4-3) · #6 C7(§5) ·
  #7 게이트·시험 조건(§2.2, §3) · #8 중첩(§2.4-5) · #9 순열 RNG · #10 C0 체인 표 · #11 MVP 분리 · #12 인용 정정(CAPS_V3 `:34`, rank 코드 `'Athaliana'` 고정,
  thinning = rule 8) · #13 수용영역 · #14 헤드·파라미터 · #15 메모리 · #16 초기화. 미확인 1건: "G. max 27런"은 v2에서 삭제.
