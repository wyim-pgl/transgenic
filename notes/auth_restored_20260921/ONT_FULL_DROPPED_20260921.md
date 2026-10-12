# ONT full 실행 — 사용하지 않음 (저자 결정 2026-09-21)

## 결정

> 저자: **"ONT full 사용하지 않는다."**

B1 §§3–9의 **ONT 전체 run 검증 정렬(full ONT / CLEAN / F4)을 수행하지 않는다.**
같은 날 앞선 결정("extern OOM 은 그냥 무시한다")으로 회계 게이트가 열렸지만, 그 뒤에 남아 있던
full 실행 자체를 하지 않기로 했다.

## 무엇이 함께 종료되나 (full 실행 전용 작업)

이 트랙은 **오직 full 실행을 위해서만** 존재했으므로 전부 종료한다. 삭제하지 않고 보존한다.

- `notes/auth_restored_20260921/ont/ont_full_v2/` — round 1·2 wrapper 보강 전체
  (affinity list 기록, `enforce_output_limit()` RLIMIT_FSIZE 이식, 워크로드 전·후 cgroup 관측,
  `check_scratch()`, `fixture_execute.py`, `fixture_substitutions.py`, `binding_identity()`,
  patch 없이 실행되는 `slurm()`). 29 tests GREEN, `sys.settrace` 커버리지 15/15는 **그대로
  유효한 로컬 검증 결과**이지만 **소비처가 사라졌다.**
- **v5 패키지 빌드** — 하지 않는다(애초에 미빌드).
- **원격 bounded fixture 실행안** — 작성하지 않는다.
- **throughput/scratch 예산 측정** — 불요.
- **관리자 kernel·slurmstepd·cgroup 로그, support draft** — 이미 extern 결정으로 종료됨.
- `6165965`의 affinity list 영구 미관측 — 더 이상 결손이 아니다.

## 무엇이 종료되지 않나 — **MVP 패널 정렬은 별개다**

이 둘을 혼동하면 안 된다.

| | ONT full (종료) | MVP 패널 정렬 (별개, 살아 있음) |
|---|---|---|
| 범위 | 전체 run 전부 | 패널 50 loci 윈도 ±10 kb로 **사전 선별된 reads만** |
| 근거 | B1 §§3–9 검증 pass | `B6_MVP_PILOT_v1.md` §2 — 본문이 **"locus-restricted alignment (not the B1 validation pass)"**라고 명시 |
| 코드 | `ont_full_v2/wrapper.py` (오케스트레이션·Slurm·cgroup·scratch) | `revision/scripts/mvp/mvp_align_panel_longreads.py` |
| 의존 | — | **full wrapper에 의존하지 않음**(확인함: stdlib + `mvp_windows`만 import, 자체 `execute()` 보유, `ont_full` 참조 없음) |

즉 **full을 버려도 MVP 구조 readout 경로 자체는 코드상 살아 있다.**

## 남은 결정 하나 — MVP 패널 정렬도 버리는가

A47.1로 **A-ONT1이 A. thaliana MVP의 유일한 구조 소스**다. 따라서:

- **패널 정렬을 한다면**: 구조 readout이 존재하고(`structural_available=True`), §5.3의 모순 검사가
  작동하며, 나머지 조건이 맞으면 **STOP이 도달 가능**하다.
- **패널 정렬도 하지 않는다면**: `structural_available=False`가 확정되고, §5.3 가용성 안전장치에
  따라 **예정 STOP은 무조건 INCONCLUSIVE로 하향**된다(GO 아님). GO는 자체 1차 기준으로만 나올 수
  있고 구조 팔이 GO를 만들지는 못한다. 즉 **MVP의 결론은 사실상 GO 또는 INCONCLUSIVE 둘뿐**이 되고,
  "증거 채널이 도움이 안 된다"는 futility 판정(STOP)은 원리적으로 못 내린다.

이건 MVP 결과의 도달 가능 범위를 바꾸므로 저자 결정이 필요하다. 결정 전까지 패널 정렬 관련
작업은 착수하지 않는다.

## B1 검증 쪽 영향

A. thaliana의 B1 §§6–9 **long-read 구조 검증 pass**는 수행되지 않는다. per-source 가용성 표에
A-ONT1을 **수행 안 함**으로 기록하며, 이는 음성 증거가 아니다(B1 §3의 "non-callable observations
are not negative evidence"). A-HiFi는 A47로 이미 구조권한이 없고, A-CCS23은 A47.1로 구조 팔을
맡지 않는다. 결과적으로 **A. thaliana B1 검증에 long-read 구조 증거는 없다**고 정직하게 보고해야 한다.

## 범위

문서 기록만. 파일 삭제 없음, 잡 제출·취소·원격 쓰기·빌드·커밋·푸시 없음.
GPU 잡 3187760, B5 체인 3149358-63 무접촉.
