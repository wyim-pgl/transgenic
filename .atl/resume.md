# Resume — 현재 B6/C2 모델 완성 우선 / B6 본학습 대기 / Delta CPU 사용 가능

현재 실행 정본: [B6 진행 기록](../notes/b6_pilot_resume_20261020/PROGRESS.md). 이 파일 → [TODO](todo.md) → 정본 진행 기록 → [quarantine](quarantine.md) 순서로 읽는다. C2의 상세 정본은 [기존 handoff](../notes/RESTART_HANDOFF_20261012.md)와 [C2 기록](../notes/C2_REPLAY_PROGRESS_20261012.md)다. 이전 상태는 [10-20 이전 resume](resume_history_20261020.md)에 보존했다.

## 사용자 지시와 범위

이미 승인된 파일럿·체크포인트 검증 이관을 계속하고 검증 통과 후 후속 실행에 승인을 다시 묻지 않는다. 과학적 동일성·소유권·예산·100GiB 게이트는 유지한다. 구현 브랜치는 main에 병합하지 않는다. 모든 변경을 푸시하고 재시작 문서를 갱신하라는 기존 지시가 유효하다. 사용자는 `/graphify` 범위를 **전체**로 선택했다. Delta CPU1,000core-hours는 신청했고 크레딧 처리가 끝나면 알리기로 했다.

**10-21 최신 지시:** CPU 리소스 승인을 통지했고 실제 CPU 잡 검증까지 완료했다. 사용자는 ccloop으로 현재 모델을 완성하는 데 집중하라고 지시했다. 네 종(rice/Arabidopsis/soybean/sorghum)의 Helixer 우월성 목표는 사용자가 일시중지했으며 새 모델 설계·벤치마크 확장은 보류한다. 그 후속 평가 요구(BUSCO, BLAST mapping, OMArk, CDS/mRNA/exon/UTR)는 보존한다. 현재 B6/C2 학습·재개·WP5 검증과 기존 과학적 게이트를 우선하며 ccloop 기록은 [현재 모델 작업](../notes/ccloop_current_model_20261021/PROGRESS.md)에 연결한다.

## 현재 상태 — 2026-10-21 11:33 PDT (B6 실측; 그 외 항목은 10-20 확인)

- **C2 파일럿 완료:** 검증3369694/3370048 PASS → 본학습3379630,4099/4099, parent/batch/extern COMPLETED0:0. 최종은 `runs/c2_compile_history_validation/3370048/resumed/step_04099`; periodic latest4000은 정상. controller `ops/c2_in_place_198ce14_20261012` complete/revision22, 감시는 정상 종료했다. 재시작·재제출 금지. 원본200·진단256 보존. WP6 전체학습 완료를 뜻하지 않는다.
- **B6 증명3425703 PASS → 본학습3429921 Priority 대기:** 실제4GPU fresh0→410/snapshot400/new-process400→410의 모델·optimizer·scheduler·4rank RNG·compilehistory 동일성 검증 PASS. parent/batch/extern COMPLETED0:0, 실행3h22m26s. 검증된410(epoch0)을 in-place 활성화하고 구3345918→3345917을 순서대로 취소했다. 본학습3429921은4GPU/64CPU/48h/no-requeue로 자동 제출됐으며, 실측 시점에는 PENDING/Priority·elapsed0으로 아직 실행 전이다. 목표4099. 중복제출 금지. [최신 실측/검증 영수증](../notes/b6_pilot_resume_20261020/LIVE_STATUS_20261021.json).
- **B6 도구:** 구현 worktree `/tmp/b6-pilot-proof-20261020`, branch `b6-pilot-proof-20261020`,HEAD50a7fac. Proof e2bee04 독립검토/실제host3.6/NGC6contract+22proof PASS. Continuation2a48265/report50a7fac 독립검토와 실제NGC36/36 PASS. 새 immutable releases `ops/b6_history_proof_e2bee04_20261020`, `ops/b6_continuation_50a7fac_20261020`. Controller `ops/b6_in_place_50a7fac_20261020` watch PID1501642 정상(45초poll). 10-21 11:33PDT journal waiting_production/revision1291/error없음, production link3429921 한 개. [시작영수증](../notes/b6_pilot_resume_20261020/CONTROLLER_START.json).
- **아카이브3425279 완료:** 10-20 18:34:32 PDT, 1h59m31s, parent/batch/extern COMPLETED0:0. B5 epoch09–11 총42,370,570,227B(39.46GiB)를 HDD에 검증 이관하고 원본을 정리했다.12–14/latest/best는 보존했다. NVMe 실제 여유133.83GiB. 후속 복구 잡은 제출하지 않았다. [부모 검수](../notes/b6_pilot_resume_20261020/archive/PARENT_ACCEPTANCE.json).
- **Delta CPU 활성화·실제0GPU PASS:** 계정bilv-delta-cpu, 최초잔액1,000core-hours. 준비된1core/1GiB/10min metadata preflight **23104942**는cn103에서7초 실행 후 parent/batch/extern COMPLETED0:0. 실제AllocTRES cpu1/mem1G/node1, GPU없음; 공유NVMe/HDD 접근·작은helper해시 확인. accounts의후속정수표시는999이며 jobcharge표시는0.00SU(정밀차감량은 반올림으로 확정하지 않음). [안내/증거](../notes/delta_cpu_20261020/USAGE.md).
- **현재모델 ccloop:** 첫계획검토round1 수락. Claude Fable5.1 요청은300초timeout→규정된KimiK3 대체검토 성공(실제backend모델명은응답에서확인불가). 단일B6잡유지·WP5준비를진행하되 모델완료판정은아님. CPU scorer **23105376**도7초/0GPU/COMPLETED0:0; 고정gffcompare0.12.6의정확/변형intron-chain fixture 통과. [진행·검토](../notes/ccloop_current_model_20261021/PROGRESS.md). 11:58PDT B6계속Priority,watchrevision1321/error없음,NVMe116.54GiB/GPU계정1624hours.
- **B5:**3309786의 preflight 실패 원인인 C2 호스트 시험 dangling symlink만 제거·대상보존 확인.3309787–90 취소. 전체 preflight/재개 조건 미통과, 새 학습잡 없음.
- **Graphify 전체 완료:** 감지2,696파일(코드 분류1,688 + 문서·논문·이미지1,008) 처리.18,577노드/38,057관계/1,874군집. [HTML](../graphify-out/graph.html), [보고서](../graphify-out/GRAPH_REPORT.md), [JSON](../graphify-out/graph.json), [검증](../graphify-out/VERIFICATION.json). 브라우저 렌더링/CLI 질의/연결 대상 검증 PASS. Gemini 한도 초과 후 Codex host130묶음으로 처리했다. JSON748개도 구조 파싱으로 보완했다.2개 vendor C++ 부분 파싱·일부 파일 전체 단위 인용·원시 참조 진단은 보고서에 명시했다. 그래프는 생성 시점의 main 작업 폴더 스냅샷이며 이후 완료 기록은 다음 update 대상이다.

## 다음 실행

1. B6 본학습3429921의 실제 Slurm 상태와 단일 감시PID1501642/journal을 확인한다. 검증3425703과 구held잡 퇴역·410 활성화는 완료했으므로 반복하지 않는다. 기존 잡이 할당되면 실행 시 게이트를 확인하고4099까지 이어가며 epoch평가/final/DONE/정상 종료를 검증한다. 중복제출·중복감시 시작 금지.
2. 아카이브3425279와 DeltaCPU preflight23104942는 완료했으므로 반복하지 않는다. 이후 필요한CPU전용 검사·복사·평가에 bilv-delta-cpu/cpu를 사용하되 구체작업의 환경·입력·예산을 먼저 확인한다. ARM NGC학습환경은 x86 CPU에 그대로 옮기지 않는다.
3. Graphify 질의는 완성된 graph.json을 사용한다. 재구축 요청이 없으면 전체 추출을 다시 실행하지 않는다.10개 빈 의미 파일과2개 부분 파싱 코드는 manifest에 완료로 찍지 않았다. 캐시·원본 chunk·로컬 재현 스크립트는 graphify-out의 ignored 경로에 남아 있다.
4. 진행 기록/네 재시작 문서 갱신, main 운영 기록과 구현 브랜치를 별도로 커밋·푸시한다. main과 source/구현 브랜치를 병합하지 않는다.

SSH: local→pronghorn(wyim)→dtai(wcyim). DeltaCPU는 pronghorn의 ControlPath `/data/gpfs/home/wyim/.ssh/cm-delta-cpu`를 재사용한다. 큰 staging payload는 stdin으로 전달한다. 무거운 모델/실제입력 해시는 allocation 안에서만 실행한다. 소스는 `/work/nvme/bilv/wcyim/transgenic/repo_b6_198ce14`(SOURCE_COMMIT198ce14, snapshot4f12c97), 프로젝트22169/ddn_ssd 기준이다.

원격 보존: main 산출물 커밋9ffb8fd 및 구현 branch50a7fac 푸시·원격SHA 대조 완료. [영수증](../notes/b6_pilot_resume_20261020/GRAPH_PUSH_RECEIPT.json). 이후 체크리스트 정리 커밋 tip은 git log -1로 확인한다.
