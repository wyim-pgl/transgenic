# B6 캐시 API 수정 및 벤치 재실행

## 원인

실패 잡 `3205730`은 Transformers 4.57.6 / torch `2.14.0a0+4fdf77b940.nv26.08`에서 실행됐다. 기존 로컬 시험 환경은 Transformers 4.46.3 / torch 2.6.0이다. 4.46 LED attention은 per-layer KV 튜플을 반환하지만, 4.57은 공유 `EncoderDecoderCache` 객체를 갱신한다. 캐시 없는 학습에서 반환값은 None이므로 기존 B6의 튜플 합산이 `None + None`으로 실패했다. 새 버전으로 로컬 재현 시 생성 경로에서도 빈 캐시와 prefix 처리 오류가 확인됐다.

## 수정 범위

- 로컬 `/tmp/b6-integration`, 브랜치 `b6-integration`, 커밋 `c9e242ceeb7c9f83925985fda01b04861e5fd09c`.
- 변경 파일 2개: `src/transgenic/model/modeling_b6.py`, `tests/test_b6_cache_compat.py`.
- LED API 시그니처로 구·신 캐시 계약 분기. 현대 경로는 공유 캐시, 외부 반환은 기존 4-tensor/layer 튜플 유지.
- chain 없는 B5 위임 이전에도 튜플을 실제 현대 캐시로 변환. B5 파일 자체는 변경하지 않음.
- 빈 생성 캐시를 첫 스텝의 no-past로 취급해 다중 토큰 prefix 보존.
- 명시적 non-reentrant checkpoint는 `layer.forward`를 호출해 새 Transformers 계층의 이중 checkpoint 방지.
- 지원 범위는 기존 튜플 외부 계약 및 DynamicCache 내부 사용이다. StaticCache 기능을 추가하지 않았다.
- 원래 캐시 오류만 없애기 위해 None 합산을 무시하는 우회는 하지 않았다. 가중치 구조·손실·체인 규칙·게이트 조건은 변경하지 않았다.

## 검증

| 환경 | 결과 | 근거 |
|---|---|---|
| 4.57.6 수정 전 기존 시험 | 학습 및 생성 4건 실패 | `baseline457.log` |
| 새 회귀 시험 수정 전 | chain 유/무 chunk 및 greedy/beam prefix 4 subcase 실패 | `red457.log` |
| 4.46.3 수정 후 관련 스위트 | 66 passed, 95 subtests passed | `final446.log` |
| 4.57.6 수정 후 같은 스위트 | 66 passed, 95 subtests passed | `final457.log` |
| MVP 원본 바이트 핀 포함 시험 | 3 passed | `pins.log` |
| 실제 26.08 SIF, CPU tiny 모델 | 8 passed, 4 subtests passed | `sif-cpu-final.log` |

실제 SIF 시험은 cached chunk, 다중 prefix, greedy/beam, non-reentrant gradient parity 및 compile+checkpoint backward를 포함한다. CPU tiny 시험은 GPU 메모리 게이트를 대신하지 않는다.

디스크 remount 뒤 Git 2.34.1의 ownership 검사가 핀 시험 1건을 일시 차단했다. 전역 설정을 바꾸지 않고 `/tmp/b6-resume-safe.gitconfig`에 해당 두 저장소만 명시한 `GIT_CONFIG_GLOBAL`로 재실행해 3건 통과했다.

독립 검토: `openai-codex/gpt-5.5`, agent `cd55b5b0-76c7-406`, **APPROVE: bounded next 2-GPU benchmark**. Critical/High/차단 Medium 없음. 비차단 주의: StaticCache 미지원. 과학적 생산 승인 아님. Advisor는 이번 재시도에서 429 rate limit으로 응답하지 못했다. Advisor 승인으로 표현하지 않는다.

## 배포 및 잡

- 원격 새 디렉터리 `/work/nvme/bilv/wcyim/transgenic/repo_b6_cachefix_20260924`.
- 원격 커밋 `7018e3345c35a984b9b3a7477f0d9fede852180a`는 로컬 `c9e242c` 수정의 스냅숏.
- 216개 추적 소스·설정·시험 파일의 SHA256 대조 전부 통과. 매니페스트 `source-c9e242c.sha256`, 자체 SHA256 `d5e7122449a0e6955bfc5890d1ea5762ed0a6de20bb1bdd70c5a822b2c54d62e`.
- 원래 `repo_b6`와 실패 잡 로그 보존. 원본 모델 해시 `2761d6a2a733710bf1a925505e07b3f153065477bcd4662c7995bcee305fcdec` 유지.
- 제출 전 B6 활성 잡 0 확인, 잔액 1,024, 보수적 여유 기준 304 이상 확인.
- **새 잡 `3207080`**, GPU 2, ghx4-interactive, 시간 상한 1시간. 제출은 1회만 수행, sbatch rc=0.
- 원격 `CACHEFIX_SUBMISSION_ATTEMPT`를 noclobber로 먼저 기록하고 결과/오류 파일 보존. 불확실한 제출을 반복하지 않는다.
- 2026-09-24 17:18 PDT 실측 **PENDING(Priority)**. 예상 시작 17:25:30 PDT, 보장 아님.
- 로그 `/work/nvme/bilv/wcyim/transgenic/repo_b6_cachefix_20260924/b6-mem129k-3207080.out`.
- JSON `/work/nvme/bilv/wcyim/transgenic/runs/b6_bench/mem129k-3207080.json`.

## 다음

30분 확인 대상은 이제 `3207080`이다. 실제 종료 상태·JSON·rank별 측정·compile 조건을 확인한다. 게이트 PASS 뒤에만 같은 수정본에서 CP4(wp3-fixtures, synthetic)를 실행한다. CP4는 아직 제출하지 않았다. B5 잡·운영 학습·main 병합·푸시는 변경하지 않았다.
