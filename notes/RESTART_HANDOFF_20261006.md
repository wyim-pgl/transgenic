# 재시작 인계: 2026-10-06 WP5 파일럿 OOM → ObsCache 메모리 수정 → 재제출 준비

📌 **이 문서가 현재 정본이다.** 이전 인계 [RESTART_HANDOFF_20261004](RESTART_HANDOFF_20261004.md)는 역사 기록이다(산출물·DeltaAI 배치·바인드 설명은 그대로 유효). 세부 로그는 [PROGRESS.md](b6_production_start_20260925/PROGRESS.md) 최상단.

## 1. 한 줄 상태 (2026-10-05 22:40 PDT)

**WP5 파일럿 첫 링크가 두 팔 모두 호스트 메모리 OOM으로 실패했고(학습 약 19분), 원인인 ObsCache 리더를 메모리 매핑 방식으로 고쳐 리뷰·시험까지 마쳤다. 파생 파일 사전 생성 중이며, 이후 DeltaAI 이관 → GPU 스모크 → 파일럿 재제출.** 코드 HEAD `826e68e`(b6-integration, main 미병합·미push).

## 2. 잡 상태

| 위치 | 잡 | 내용 | 상태 |
|---|---|---|---|
| DeltaAI | 3306594 / 3306596 | 파일럿 B6 / C2 link1 | **OUT_OF_MEMORY**(MaxRSS 465/460GB > cgroup 454GB), 두 런 `TRAINING_FAILED`, 체크포인트 없음, 약 4.5 GPU-h |
| DeltaAI | 3306595 / 3306597 | 파일럿 link2 | 취소 |
| DeltaAI | 3309785→3309790 | B5 seed 123 재개 체인 6링크(사용자 승인 10-04) | 우선순위 대기 |
| pronghorn | 6188561 | 파생 mmap 파일 사전 생성(`obs_mmap_prebuild.sbatch`, source_v20) | COMPLETED, 455 shard·653,922,028행·53.7GB |
| pronghorn | 6188565 | 전체 suite source_v21 | COMPLETED, v18과 FAILED/ERROR 동일(193) |
| DeltaAI | 3319603 | **GPU 스모크**(`b6_prep_20260925/dtai_gpu_smoke.sbatch`, 4 GPU 2h, `repo_b6_826e68e`=스냅숏 2ceebc5, 런 `runs/b6_smoke_826e68e_B6`) | 10-05 23:15 PDT 제출 |

## 3. 원인과 수정

- **원인:** `ObsCache.shard()`가 (종, 염색체) shard 전체를 numpy object 배열(행당 약 975B)로 바꿔 프로세스마다 무한 캐시. 4 rank × (main + DataLoader 워커 4) = 20 프로세스가 6.54억 행 캐시를 각자 적재. 메모리 벤치·CP4는 작은 합성 데이터라 못 잡음.
- **수정(브랜치 `b6-obscache-reader-memory`, 결과 불변):** `301e4d0` 메모리 매핑 파생 Arrow 파일(`<cache>.mmap_v1/<shard sha>.arrow`, 캐시 디렉터리 밖) + 문자열 사전 부호화 + 반환 행만 디코드 → `b66b1b6` 지연 디코드·LRU 512·`require_prebuilt`(env `B6_OBS_CACHE_REQUIRE_PREBUILT=1`, 학습 중 생성 금지)·sidecar sha·tmp 정리 → `826e68e` 오류 처리 다듬기. 3회 Codex 리뷰(모두 APPROVE_WITH_FIXES, CRITICAL/HIGH 0, 반영 완료).
- **증명:** 옛 리더(9f003e1) 그대로 복사한 참조와 모든 접근자 값·자료형·순서 동일, 데이터셋 텐서 바이트 동일; reader 시험 26개; 전체 suite v18 대 v19 FAILED/ERROR 동일(165/28), v20·v21 비교 진행.
- **측정(로컬):** 프로세스당 사적 메모리 2.92GB → 0.14GB(4프로세스·20 shard), 노드당 공유 페이지 캐시 약 50GB.
- source 스냅숏: `source_v19`(301e4d0)·`v20`(b66b1b6)·`v21`(826e68e). 파생 파일 형식은 v20=v21 동일.

## 4. 다음 단계

1. 사전 생성 6188561 완료·xjunc/xint 행 수 확인 → `train_v1.mmap_v1`(약 50GB)을 DeltaAI nvme `b6_prep_20260925/obs_cache/`로 전송, DeltaAI에서 사전 생성 재검사(잡 안에서).
2. DeltaAI 코드 `repo_b6_826e68e`(archive 스냅숏 + 로컬 커밋 + HFmodels).
3. **GPU 스모크(사용자 승인 10-05):** `ghx4-interactive` 1노드 4 GPU ≤2h(≈8 GPU-h), `B6_OBS_CACHE_REQUIRE_PREBUILT=1` — 호스트 RSS·GPU 피크(rank2 GPU OOM 재현 여부)·s/step·첫 latest_state.
4. 통과 시 파일럿 런 디렉터리의 `TRAINING_FAILED`를 보존 이동, 검증 게이트 + 두 팔 링크 재제출.

## 5. 이번 기간 사용자 결정·사건

- 10-04: B5 seed 123 재개 승인(내 스테이징이 끊어진 심볼릭 링크 408개를 `/work`에 올려 B5 사전 점검 실패 → 삭제·재개).
- 10-05: "잘 모르겠는데 진행해줘" → ObsCache 수정 방향(메모리 매핑)과 GPU 스모크까지 진행 승인으로 처리.
- DeltaAI 인증 세션이 만료되면 사용자가 별도 터미널에서 `ssh pronghorn` → `ssh dtai`로 재인증하고 열어 둔다.
- 로컬 디스크가 가득 찼다가(여유 2GB) 외부 정리 후 62GB; 그 과정에서 `/tmp/b6-integration` 등 여러 worktree 디렉터리가 사라짐 → `git worktree prune` 후 `/tmp/b6-integration` 재생성(브랜치 커밋은 무손실).
