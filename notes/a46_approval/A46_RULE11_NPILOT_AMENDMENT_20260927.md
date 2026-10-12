# A46 rule 11 수치 개정: N_pilot 4,142 → 4,099

기록일: 2026-09-27. 근거: 저자 결정 18(`notes/b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md` §17–18).
사용자 원문: `Vvinifera audit at N=500, and dropping the 1,817 corpus tiles.`

## 개정 내용

- A46 rule 11의 matched budget N_pilot을 **4,099 optimizer updates**로 바꾼다. 모든 arm(B6, C2, C3, C4, C6, C7)이 같은 값을 쓴다.
- 계산식은 그대로다: N_pilot = floor(학습 행 × 2 epoch ÷ effective batch 96).
- 학습 행: 198,828 → **196,774**. WP2 rule 5 repair가 row-set 사건 타일을 RC twin과 함께 삭제했다(plan v4 §2 item 7이 요구한 "further amendment").

## 삭제 규모 (measure v2, pronghorn job 6173258)

- 사건 타일(forward) 1,741개: masked fraction > 0.60 1,481건, RC twin이 마지막 라벨/isoform 조건을 잃는 336건. 76개 타일은 두 사건에 모두 해당한다.
- 삭제 행 2,230개(forward 1,741 + RC 489). 학습 행 2,054개가 줄었다.
- 종별 학습 forward 삭제: Vvinifera 783, Gmax 219, Bdistachyon 144, Sbicolor 130, Athaliana 86, Ptrichocarpa 70, Sitalica 60, Osativa 58, Ppatens 54.

## 코드·config 반영

- b6-integration `c422e5e`: `--apply --drop-row-set-events`. freeze가 `row_set_drop`, `train_rows`, `n_pilot`을 기록하고, 트레이너는 `rt.bind_n_pilot`으로 freeze 값에 결속된다.
- b6-integration `27c34ba`: `N_PILOT`, 두 config의 `n_pilot`, 파일럿 `stop_after_updates`, `wp5_pilot.sbatch`, 시험을 4,099로 바꿨다. GATED gate는 `certified_gated_fields_sha256`로 메모리 벤치 인증 config의 비면제 필드 전체를 고정한다. 면제 필드는 name, notes, gate_status, gate_evidence, arms, wp5_pilot, n_pilot이다.

## 동결 문서에서 이 개정으로 대체되는 문구 (본문은 수정하지 않음)

- `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md:48`, `:64` (N_pilot = 4,142)
- `revision/protocols/PROTOCOL_B1_frozen_v1.md:982`, `:1001` (4,142), `:928` (198,828 행)
- `notes/a46_approval/A46_FREEZE_CHECKLIST_20260921.md:222` (N_pilot = 4,142 계산식)
