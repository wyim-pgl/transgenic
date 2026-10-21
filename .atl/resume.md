# Resume — C2 완료 / B6 증명 대기 / Delta CPU 공유 확인

현재 실행 정본: [B6 진행 기록](../notes/b6_pilot_resume_20261020/PROGRESS.md). 이 파일 → [TODO](todo.md) → 정본 진행 기록 → [quarantine](quarantine.md) 순서로 읽는다. C2의 상세 정본은 [기존 handoff](../notes/RESTART_HANDOFF_20261012.md)와 [C2 기록](../notes/C2_REPLAY_PROGRESS_20261012.md)다. 이전 상태는 [10-20 이전 resume](resume_history_20261020.md)에 보존했다.

## 사용자 지시와 범위

이미 승인된 파일럿·체크포인트 검증 이관을 계속하고 검증 통과 후 후속 실행에 승인을 다시 묻지 않는다. 과학적 동일성·소유권·예산·100GiB 게이트는 유지한다. 구현 브랜치는 main에 병합하지 않는다. 모든 변경을 푸시하고 재시작 문서를 갱신하라는 기존 지시가 유효하다. 사용자는 `/graphify` 범위를 **전체**로 선택했다. Delta CPU1,000core-hours는 신청했고 크레딧 처리가 끝나면 알리기로 했다.

## 현재 상태 — 2026-10-20 PDT

- **C2 완료:** 검증3369694/3370048 PASS → 본학습3379630,4099/4099, parent/batch/extern COMPLETED0:0. 최종은 `runs/c2_compile_history_validation/3370048/resumed/step_04099`; periodic latest4000은 정상. controller `ops/c2_in_place_198ce14_20261012` complete/revision22, 감시는 정상 종료했다. 재시작·재제출 금지. 원본200·진단256 보존.
- **B6 증명3425703 제출, Priority 대기:** 10-20 17:54:33PDT 제출,ghx4/4GPU/64CPU/6h/no-requeue. 새 source198ce14와 evidence-on fresh0→410/snapshot400/new-process400→410 전체상태 동일성을 검사한다. 아직 학습 시작·GPU PASS가 아니다. 구3345918/3345917은 JobHeldUser 그대로. 중복제출 금지. 제출 정본 `ops/b6_proof_submit_e2bee04_20261020` 및 [영수증](../notes/b6_pilot_resume_20261020/PROOF_SUBMISSION.json).
- **B6 도구:** 구현 worktree `/tmp/b6-pilot-proof-20261020`, branch `b6-pilot-proof-20261020`,HEAD50a7fac. Proof e2bee04 독립검토/실제host3.6/NGC6contract+22proof PASS. Continuation2a48265/report50a7fac 독립검토와 실제NGC36/36 PASS. 새 immutable releases `ops/b6_history_proof_e2bee04_20261020`, `ops/b6_continuation_50a7fac_20261020`. Controller 설정 독립검토·nativeidentity검사 PASS 후 `ops/b6_in_place_50a7fac_20261020` init/watch 완료(PID1501642,45초poll). journal waiting_validation/revision1/error없음, proof3425703대기. [시작영수증](../notes/b6_pilot_resume_20261020/CONTROLLER_START.json).
- **아카이브3425279 진행:** `ops/b6_headroom_20261020_v2`,10-20 16:35PDT 시작,1GPU/1CPU/2GiB/2h. DeltaAI의 최소1GPU 할당에서 CPU I/O만 수행. B5epoch09는 HDD 검증·원본정리 완료(14,123,523,405B);10 이관 중,11 미완료. 실제NVMe107.5GiB로100GiB와 독립42.24GB proof-copy 피크 조건 통과.12–14/latest/best 보존. 전체3개 완료로 주장하지 않는다. 타임아웃 시9가 이미 없으므로 기존 wrapper를 재실행하지 말고 manifest/receipt/partial 상태부터 확인한다.
- **Delta CPU:** 로그인dt-login03/UID97434와 기존bilv NVMe/HDD 양방향 읽기·쓰기·해시 일치/시험파일정리 확인. Python3.9.21 stdlib archive fixture1PASS. `accounts`는 exit60/No slurm account association found. CPU 크레딧·Slurm association 준비 전이며 CPU잡은 제출하지 않았다. [안내/증거](../notes/delta_cpu_20261020/USAGE.md).
- **B5:**3309786의 preflight 실패 원인인 C2 호스트 시험 dangling symlink만 제거·대상보존 확인.3309787–90 취소. 전체 preflight/재개 조건 미통과, 새 학습잡 없음.
- **Graphify 전체:** 2,696파일 감지. 코드1,688개 구조추출 완료(13,326nodes/33,541edges),문서·논문·이미지1,008개 host130묶음 분석 중. Gemini 일일 무료20회 한도429로 해당 worker 중단, Codex 병렬 분석으로 전환. 현재 상태 `graphify-out/RUN_STATE.json`, `HOST_PLAN.json`, `host_chunks/`; 최종 graph/HTML/report는 아직 미생성. AST15 JSON 실패는 별도 full-scan 보완,2개 vendor C++ 부분파싱 경고는 보존한다.

## 다음 실행

1. 실제 B6 proof/accounting을 확인하고 제출된 잡을 재제출하지 않는다. 단일 감시PID1501642와journal을확인하고중복시작하지않는다. GPU 증명·정상 종료·100GiB가 모두 충족돼야 옛held잡18→17을 정확히 퇴역하고 검증된resumed410을4099까지 이어간다.
2. 아카이브3425279 완료/타임아웃을 검사하고 원본·verified·partial 증거를 보존한다. Delta CPU association이 생기면 준비된 읽기 전용cpu preflight부터 실행한다.
3. Graphify 모든 묶음 검증→AST/semantic병합→진단/군집명/HTML/JSON/report/benchmark. 완료하지 않은 묶음을 캐시 완료로 표시하지 않는다.
4. 진행 기록/네 재시작 문서 갱신, main 운영 기록과 구현 브랜치를 별도로 커밋·푸시한다. main과 source/구현 브랜치를 병합하지 않는다.

SSH: local→pronghorn(wyim)→dtai(wcyim). DeltaCPU는 pronghorn의 ControlPath `/data/gpfs/home/wyim/.ssh/cm-delta-cpu`를 재사용한다. 큰 staging payload는 stdin으로 전달한다. 무거운 모델/실제입력 해시는 allocation 안에서만 실행한다. 소스는 `/work/nvme/bilv/wcyim/transgenic/repo_b6_198ce14`(SOURCE_COMMIT198ce14, snapshot4f12c97), 프로젝트22169/ddn_ssd 기준이다.
