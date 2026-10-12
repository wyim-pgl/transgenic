# 재시작 정본: 2026-10-06 파일럿 OOM → ObsCache 메모리 수정 → 재제출 준비

> 📌 **먼저 [RESTART_HANDOFF_20261006](../notes/RESTART_HANDOFF_20261006.md)를 읽는다.** 잡 상태, OOM 원인과 수정(`301e4d0`→`b66b1b6`→`826e68e`, 결과 불변 증명), 다음 단계(사전 생성 → DeltaAI 이관 → GPU 스모크 → 파일럿 재제출)가 거기 있다. 산출물·DeltaAI 배치는 [RESTART_HANDOFF_20261004](../notes/RESTART_HANDOFF_20261004.md) 그대로.
>
> **2026-10-12 후속 제출:** 첫forward분리진단 **3364480 RUNNING/gh150**,10-11 23:26PDT시작·입력검증중,4GPU90분,cold/primed/primed_reset3case. 상태복원local2+NGC2PASS·독립검토승인;운영코드d7b1a37무변경. [정본](../notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md). `FIRST_FORWARD_ACCEPTANCE.md` 기준결과해석후수정진행;기존본학습자동연결금지유지.
>
> **2026-10-12 결과 확인:** **3361666 비교FAIL**(세학습phase정상종료). old/new200→210모델·optimizer완전일치;연속/재개210→220은609텐서불일치. 입력·RNG는모두동일,첫forward부터rank2/3차이,DDPbucket재구성없음. **현재학습RUNNING없음**, 본학습보류·원본200/진단256보존. [현재정본](../notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md). 다음은컴파일최초입력의영향을첫forward로분리진단.
>
> **2026-10-11 통제 비교 제출:** 진단 **3361666**,4GPU2h, old200→210/new200→220/new210→220. **PENDINGPriority,예상10-11 14:51PDT(변동),GPU배정대기**. 실제컨테이너진단3시험·독립리뷰PASS. 3361386은의도적USR1로256저장후정상trainer종료(400미달postcheck에따른SlurmFAILED). controller3276938부재·새생산잡없음. [정본](../notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md) 최상단.
>
> **2026-10-11 후속 과학 검증 게이트:** 4GPU3361386은200재개·210학습에성공했으나동일step손실이중단전실행과다름(old3.25677085/new3.25534534). 규약의resume결정성게이트미해결로 **자동본학습연결중단**(controller3276938종료요청, journal blocked_determinism, 새생산잡없음);3361386에는USR1정상저장종료요청. [현재정본](../notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md) 최상단부터읽고 live상태확인. 원본200보존,구C2/B6held유지. 다음은동일노드/입력/RNG/캐시로old·new재개및연속학습비교. 사용자바로진행지시는검증완료후진행이며과학게이트우회가아님.
>
> **2026-10-11 11:35 PDT 긴급 갱신:** C2 `3345919`는 10-10 20:43 PDT **FAILED**. rank3가 update280 입력 `(1,129024)`의 세 번째 compile event에서 guard에 걸림(OOM 아님). `latest_state`는 **200 updates**로 보존, 기본 파일 구조 확인. 후속 `3345920`은 **JobHeldUser**로 보류 확인. 사용자 **원인 확인·수정 지시(10-11)**: 원인 재현 → 수정/회귀·독립리뷰 → GPU 검증 → 체크포인트 재개. [현재 진행](../notes/checkpoint_archive_20261009/C2_RECOVERY_20261011.md). 코드 기준233574a, 수정 worktree `/tmp/b6-c2-recompile`(종료복구3f81fb6+컴파일후보d7b1a37, 운영미배포). 실제재생3360834/3360978완료·원인재현, 후보GPU전체검증3361110제출. B6 3345917/18도동일코드보호목적으로JobHeldUser(Nice1100); B5무변경.
>
> **2026-10-10 19:31 PDT C2 실제 RUNNING:** [C2_RUNNING](../notes/checkpoint_archive_20261009/C2_RUNNING_20261010.md). 3345919는gh069에서18:42PDT시작, GPU4할당/학습4rank확인, **60/4099업데이트**(40→50→60진행), 실패표식/OOM없음. SlurmMaxRSS59.4GiB; 관측가능GPU3개는38.4–38.7GiB/100%(진단step의4번째GPU가시성제한). 첫checkpoint200아직미도달. 다음:200저장·165이후/평가/재개/최종확인. B5/B6대기·C2우선유지.
>
> **2026-10-10 18:21 PDT 최신 확인:** [STATUS](../notes/checkpoint_archive_20261009/STATUS_20261010_1821PDT.md). C2 3345919는 아직 PENDING(Priority). 최신 예상 시작 **오늘10-10 23:16PDT**(변동 가능). C2 우선 유지(1813>B5 1011>B6 713), 실행 디렉터리/학습로그 미생성. B5/B6는10-11 19:58PDT 예상. 조회만 수행, 잡·코드 변경 없음.
>
> **2026-10-10 08:49 PDT 큐 확인:** [STATUS](../notes/checkpoint_archive_20261009/STATUS_20261010_0849PDT.md). **C2아직PENDING(Priority)**, 최신예상10-11 09:52PDT(변동). C2런디렉터리/로그미생성. C2우선유지(1383>B5 779>B6 283),전잡대기. 이관완료유지·재제출/설정변경없음.
>
> **2026-10-09 23:15 PDT 최신:** 사용자 **C2우선** 반영: C2 3345919/20 Nice0, B5 3309786–90·B6 3345917/18 Nice1100(상대순위하향), 의존성/잡번호유지. [C2_PRIORITY](../notes/checkpoint_archive_20261009/C2_PRIORITY_20261009.md). 최신C2예상10-11 00:18PDT(변동). **이관8/8 완료**(23:09PDT,99.06GiB,오류0), NVMe여유204.15GiB; 원본01–08부재/09–14·latest·best14보호확인. 재이관불필요. C2최종완료검증후 남은B5/B6 Nice0복구.
>
> **2026-10-09 22:23 PDT 실측:** [상태 기록](../notes/checkpoint_archive_20261009/STATUS_20261009_2222PDT.md). 이관5/8(epoch01–05) 검증·정리완료/오류0, runner20269계속실행, NVMe여유170.86GiB. **B5는14epoch완료·15번째88%(global_step30819), GPU대기**. 현재예상 B5재개10-10 13:05PDT, B6/C2파일럿10-10 23:02PDT(변동가능). Slurm변경없음.
>
> **2026-10-09 20:15 PDT 실제 이관 실행 중(사용자 지시):** [PRODUCTION_TRANSFER](../notes/checkpoint_archive_20261009/PRODUCTION_TRANSFER_20261009.md). B5 epoch01–08 약99.06GiB → **pronghorn**. login-0 runnerPID20269, 전용tmux socket `codex-archive-20261009`/session `archive-b5-first8-20261009`. 코드3b4eb0d,16MiB/s,8cycles, 검증후원본정리. 완료/실패는 원격run디렉터리 `status.json`에자동기록. **중복실행금지: live status부터확인.** 최신상태·13/14·best보호,Slurm무변경.
>
> **2026-10-09 자동 이관 구현·설치·검증 완료:** [FINAL_HANDOFF](../notes/checkpoint_archive_20261009/FINAL_HANDOFF.md). `b6-checkpoint-archive` HEAD `3b4eb0d`(미병합), DeltaAI/pronghorn 각각69시험 PASS, 실제전송3시나리오 PASS, 독립검토 APPROVED. 두서버 `ops/checkpoint_archive/3b4eb0d` 설치·SHA일치. 실제 B5 epoch01–08 약99.06GiB 후보 조회완료,13·14/best보호. **운영 대량이관·자동삭제서비스는 아직 미활성**(실행준비명령/설정은인계). 14:47 CDT B5/WP5계속대기,잔액771GPU-h. 기존학습종료오류2건은별도미수정.
>
> **자동 이관 구현 진행:** [IMPLEMENTATION_PROGRESS](../notes/checkpoint_archive_20261009/IMPLEMENTATION_PROGRESS.md). core9ecd8a2→b80c638, DeltaAI37시험 및 실제 NVMe→HDD 작은파일3회 PASS. 독립리뷰의 receipt 기록 초기중단 복구 fix2 진행. 프롱혼 pull/CLI/watch는 다음 Task2. 운영잡·실제체크포인트는 무변경.
>
> **2026-10-09 자동 이관 구현 시작(사용자 승인):** HDD 및 pronghorn 전송, 로그인 노드 소규모 반복시험 요청. 작업 브랜치 `b6-checkpoint-archive`(`/tmp/b6-checkpoint-archive`, base233574a, 설계4aac577). 정본 계획 `notes/checkpoint_archive_20261009/PLAN.md`(해당 worktree), 진행 `.superpowers/sdd/PLAN/progress.md`. Task1 core archive worker 진행; 실데이터·대기잡 무변경. 13:43 CDT HDD829.29/1000GiB, 가용170.71GiB; 증거DB733GiB. pronghorn 가용21T·dtai SSH pull 가능.
>
> **2026-10-09 13:19 CDT HDD 이관 확인:** [HDD_ARCHIVE_FEASIBILITY](../notes/b6_production_start_20260925/HDD_ARCHIVE_FEASIBILITY_20261009.md). HDD 여유170.71 GiB·쓰기 가능. 1 MiB 실험: 단순 mv는 ddn_ssd 유지, HDD 새 복사는 ddn_hdd·해시 동일. B5 epoch01–08 약99.06 GiB 이관 후보(완료 메타 확인, 전체 hash는 미실행); 이관 후 여유 예상 NVMe204.15/HDD71.66 GiB. latest_state·최근2epoch·best 유지 권고. 사용자 요청은 해결방법·확인으로 실제 checkpoint 이관·코드/잡 변경은 아직 없음.
>
> **2026-10-09 추가 위험 감사(Codex):** [RISK_AUDIT_20261009](../notes/b6_production_start_20260925/RISK_AUDIT_20261009.md) — NVMe 여유105.09 GiB/시작검사100 GiB, 새 B6 checkpoint5.33 GiB만 생겨도 다음 링크 검사 거부(재현). 완료 표식 전 검증 순서·최종 step 게시 후 재개 결함2건도 CPU 오류 주입과 독립 재실행으로 확인. 집중 시험29개 통과. **후속 우선순위: 저장 여유 확보 → 종료 처리 보완·검증 → 파일럿 진행 확인.** 이번 요청은 위험 점검이므로 원격 삭제·잡 변경·코드 배포 없이 감사 기록만 추가. 현재 잡은 대기 중(12:56 CDT), 소스233574a 유지.
>
> **2026-10-09 12:48 CDT 실측(Codex, Claude 세션 `3b67ed3a-f949-4b63-a149-9170b7fb21e0` 확인):** 스모크 3338555·verify 3345916 COMPLETED 0:0, verify 기록 3306593과 바이트 동일. B6 3345917·C2 3345919 PENDING(Priority, 예상 시작 없음), 후속 3345918·3345920 Dependency. B5 link2 3309786 PENDING(예상 10-10 15:20 CDT, 변동 가능). 잔액 771 GPU-h. 다음은 첫 링크 시작 후 처리량·메모리·체크포인트 확인; 현재 해당 점검은 큐 대기로 진행 불가. 상세 [PROGRESS](../notes/b6_production_start_20260925/PROGRESS.md).
>
> **2026-10-09 01:05 CDT:** 스모크 3338555 PASS(업데이트 240, GPU 41.4 GB/GPU, OOM 없음). **WP5 파일럿 재제출**: verify 3345916 → B6 3345917→3345918, C2 3345919→3345920 (repo_b6_233574a). 다음: verify·첫 링크 확인. 코드는 브랜치 `b6-compile-static-len`(233574a) — b6-integration 반영은 사용자 요청 시.
>
> **2026-10-08 03:15 CDT:** 최종 수정 `233574a`(Codex APPROVE, suite 동일, 컨테이너 시험 통과). **4-GPU 스모크 3338555** 제출, 큐 정체로 예상 시작 **10-09 05:07 CDT**. 통과 시 `cd b6_prep_20260925 && NLINKS=2 ./dtai_submit_pilot_v3.sh`.
>
> **2026-10-08 03:00 CDT:** 수정 GPU 확인 — 재생 3337061에서 OOM 샘플 38.4 GiB(이전 92). torch 시험 3338495 진행 → 4-GPU 스모크 `dtai_gpu_smoke_static.sbatch` → 파일럿 v3.
>
> **2026-10-07 23:57 CDT:** 수정 `09cfc48` 완료(Codex 반영, 전체 suite 동일). DeltaAI 큐 정체 — 재생 **3337061** 예상 시작 10-09 02:25 CDT. 그 뒤 순서: 재생 결과 확인(rn 10431 ≈37–40 GiB) → `cd b6_prep_20260925/logs && sbatch ../dtai_pytest_torch.sbatch` → `sbatch ../dtai_gpu_smoke_static.sbatch`(4 GPU, ≈6 GPU-h, 승인됨) → 파일럿(`NLINKS=2 ./dtai_submit_pilot_v3.sh`, 준비 완료; 링크는 ghx4 큐 — 현재 예상 10-18).
>
> **2026-10-07 17:25 CDT:** GPU OOM 원인 특정 — 같은 샘플(rn 10431)이 단독 37 GiB, 세 DNA 길이 그래프 뒤 4번째 재컴파일 시 92 GiB(유력: automatic dynamic로 L 동적 그래프). **대책 사용자 결정 대기**(DNA 길이 정적 고정 권고). 파일럿 보류.
>
> **2026-10-07 16:15 CDT:** 실데이터 재생 진단 **3334532** 제출(`repo_b6_40311b3`, 예상 16:34 CDT 시작). B5 seed123 link1 정상 완료(epoch 14 best 0.5550), link2 대기. 파일럿 보류.
>
> **2026-10-07 (later):** 실데이터 재생 진단 `40311b3` 준비(Codex 지적 반영), pronghorn source_v23 배포. **DeltaAI 인증 만료 → 사용자 재인증 후** `repo_b6_40311b3` 생성·1 GPU 잡 제출.
>
> **2026-10-07:** 진단 3322147 완료 — 합성 최악 58.5GiB(설정 무관, 증가 0), 실스모크 87GiB → **실데이터 배치가 원인**. 사용자 결정: 실샘플 재생 모드 추가(writer 작업 중) → Codex 리뷰 → 1 GPU ≈1h. 파일럿 보류.
>
> **2026-10-06 진단 제출:** 1-GPU 메모리 진단 **3322147**(`repo_b6_fe24b65`, 예상 시작 10:21 CDT) — 결과로 GPU OOM 대책을 사용자가 선택. 파일럿 재제출 보류.
>
> **2026-10-06 (Codex·Kimi 진단 후):** allocator 스모크 3321974 **취소**. 사용자 결정: **1-GPU 메모리 진단 먼저**(writer 작성 중, worktree `/tmp/b6-gpu-diag` → Codex 리뷰 → ≈1 GPU-h). 원인 미증명: cross-attention 점수 행렬(라벨 길이 비례), 벤치(58.6GiB)와 실데이터(87GiB) 차이 후보 = 동적 형상 재컴파일 / deterministic 알고리즘 / 누적.
>
> **2026-10-06 03:10 CDT:** 사용자 "Allocator first" → 스모크 재실행 **3321974**(expandable_segments, 예상 시작 09:42 CDT). 통과 시 파일럿 재제출(제출 스크립트에 allocator env 추가 필요).
>
> **2026-10-06 02:45 CDT 실측:** GPU 스모크 3319603 **FAILED** — 호스트 메모리 70GB로 해결됐지만 **GPU OOM**(rank3 backward, ~step 165, 87GB+6GB 단편화). 파일럿 재제출 보류, **GPU 메모리 대책 사용자 결정 대기**. 상세 PROGRESS 최상단.
>
> **2026-10-05 23:20 PDT 실측:** 사전 생성 6188561 PASS(455 shard, 653.9M행), suite v21=v18(FAILED/ERROR 동일), mmap·`repo_b6_826e68e`(2ceebc5) DeltaAI 이관 완료. **GPU 스모크 3319603 제출**(ghx4-interactive 4 GPU 2h, `b6_prep_20260925/dtai_gpu_smoke.sbatch`, 런 `runs/b6_smoke_826e68e_B6`) — 결과 확인 후 파일럿 재제출. B5 seed 123 체인 3309785→90 대기. notes/.atl **미커밋**.
>
> **2026-10-05 22:40 PDT 실측:** 파일럿 link1 두 팔 OOM(호스트 465GB), link2 취소. ObsCache 리더 메모리 매핑 수정 완료(Codex 3회, 전체 suite 동일), b6-integration HEAD `826e68e`. 사전 생성 6188561·전체 suite v21 6188565 실행 중. B5 seed 123 체인 3309785→90 대기. GPU 스모크 승인됨(≈8 GPU-h). notes/.atl **미커밋**.

## (역사) 10-04 블록

### 재시작 정본: 2026-10-04 B6 훈련 입력 완성 → WP5 파일럿 DeltaAI 큐 대기

> 📌 **먼저 [RESTART_HANDOFF_20261004](../notes/RESTART_HANDOFF_20261004.md)를 읽는다.** 실행·대기 잡, 산출물과 sha, DeltaAI 배치·바인드, 이번 세션 사용자 결정 1–6, 코드 변경과 증명, 다음 단계가 모두 거기 있다. 세부 로그는 [PROGRESS.md](../notes/b6_production_start_20260925/PROGRESS.md) 최상단. 아래 10-04 이전 블록은 **역사 기록**이다.
>
> **2026-10-04 08:00 PDT 실측:** pronghorn에서 C0 28/28·zero 27/27·병합 DB(786GB, `22af410f…`)·코퍼스 재검증 PASS·obs cache(455 shard, `11b4c757…`)·`verify_inputs` PASS(record `56d4849e…`)를 마쳤다. DeltaAI에서도 같은 검증 3306593이 PASS했고 레코드가 바이트 동일하다. **WP5 파일럿(사용자 승인 10-03)**: B6 3306594→3306595, C2 3306596→3306597, ghx4 Priority 대기(예상 10-05 12:02 CDT). B5 seed 체인 3149363 대기. 코드 HEAD `9f003e1`(b6-integration, 미병합·미push). pronghorn 파티션은 s1-pgl·s2-core만(**s3-sponsored 금지**). 이번 세션 notes/.atl 변경은 **커밋하지 않았다.**

## 10-01~10-03 기록 (역사)

### (역사) 재시작 정본: 2026-09-30 B6 실데이터 증거 DB (C0 마무리 + 병합·obs cache 준비 완료)

> 📌 **먼저 [RESTART_HANDOFF_20260930](../notes/RESTART_HANDOFF_20260930.md)를 읽는다.** 코드 위치(b6-integration HEAD `c8e01f4`)·production 소스 `source_v13`·C0 표·실행 중 잡·zero list·병합/obs cache 시험 결과·사용자 결정·다음 단계가 모두 거기 있다. 세부 로그는 [PROGRESS.md](../notes/b6_production_start_20260925/PROGRESS.md). 아래 09-28 블록부터는 **역사 기록**이다.
>
> **2026-10-04 04:45 PDT 실측:** DeltaAI 입력 검증 3306593 **PASS**(record `56d4849e…`, pronghorn과 바이트 동일). **WP5 파일럿 제출됨, ghx4 큐 대기**: 증거 모델 3306594→3306595, 서열만 대조 3306596→3306597. B5 seed 체인 3149363 대기. ACCESS 보충(3,800 GPU-h) 신청 승인, 잔액 반영 대기.
>
> **2026-10-03 15:50 PDT 실측:** **pronghorn 쪽 훈련 입력 완성·검증 완료**(병합 DB `22af410f`, 재검증 PASS, obs cache 511파일, `verify_inputs` PASS record `56d4849e…`). DeltaAI nvme에 캐시·코퍼스·bindings·source_v18 전송 완료(병합 DB 786GB만 HDD 공간 결정 대기). 다음: DeltaAI repo_b6 갱신·bindings(컨테이너 경로)·verify → **파일럿 제출 전 사용자 확인**. GPU 미제출.
>
> **2026-10-03 15:25 PDT 실측:** **obs cache 완료**(`obs_cache/train_v1`, 455 shard, 653.9M행, 7.9GB, manifest `11b4c757…`; 보험 재시도가 디렉터리를 옮긴 사고 복원됨) · **코퍼스 재검증 PASS**(`corpus/validate_c0.json`, c0_molecules_consistent 37,850,010). `verify_inputs` 독립 실행 6186628 진행 중. 다음: DeltaAI 2차 전송(nvme: cache·corpus·bindings; 병합 DB 786GB는 HDD 공간 결정 대기) → GPU 파일럿 제출 전 사용자 확인. GPU 미제출.
>
> **2026-10-03 03:00 PDT 실측:** obs cache 6185476 OOM(186.6GB) → 빌더에 DuckDB 메모리 한도 옵션 추가(`f06a786`, Codex 반영, shard 바이트 불변; b6-integration HEAD `f06a786`, source_v17) → **obs cache 6186344 RUNNING**(cpu-s2-core-0, 170G, 한도 120GB). **재검증 6185660 RUNNING**(17:32~, Python 스트리밍 단계). 사용자 규칙: cpu-s3-sponsored-0 금지. 결정 대기: DeltaAI HDD 보관본 147G 재배치, `runs/seed123`. GPU 미제출.
>
> **2026-10-02 16:50 PDT 실측:** 코퍼스 재검증 6183698 OOM(209.7GB) → 검증기 수정 `31ddead`(스트리밍 감사 + DuckDB 한도; Codex 반영; 결과 불변 증명) → **source_v15**, b6-integration HEAD `31ddead`. 재검증 쌍둥이 **6185660(s2)·6185661(s3 8h)** 대기. **obs cache 6185476 RUNNING**(s2 cpu-11, 178G; 220G는 노드 포화로 10-14 예상이라 축소). 결정 대기: DeltaAI HDD 보관본 147G 재배치, `runs/seed123`.
>
> **2026-10-02 15:35 PDT 실측:** **본 병합 6183697 COMPLETED**(7h03m, 786GB, sha `22af410f…`, 28 입력 상대경로). zero list 27/27. **검증 6183698 RUNNING, obs cache 6183699 PENDING(Resources)**. DeltaAI: 병합 DB는 nvme 불가·HDD도 소프트 한도 초과 → HDD 보관본 147G 재배치 결정 대기; `runs/seed123` 처리 선택 대기. GPU 미제출.
>
> **2026-10-02 08:25 PDT 실측:** **C0 28/28 완료**(Athaliana ONT 6182077 PASS 225GB, Gmax protein 6181302 PASS 88.7GB). **본 병합 6183697 RUNNING**(08:14~), Gmax zero 6181303–05 RUNNING, 검증 6183698·obs cache 6183699 대기. DeltaAI nvme 정리 1단계 완료(ddp_int·partial 보관·삭제, 291/500G); `runs/seed123`는 B5 체인 학습 중이라 사용자 선택 대기. GPU 미제출.
>
> **2026-10-01 19:10 PDT 실측(재부팅 후):** 6182077·6181301 RUNNING 정상. **후속 체인 선제출:** 병합 6183697 → 검증 6183698 ∥ obs cache 6183699(afterok). DeltaAI 사전 전송 진행 중(`dtai_prestage.sh`). 사용자 결정 대기: DeltaAI 병합 DB 위치(HDD 바인드 vs nvme 정리). 상세 [PROGRESS](../notes/b6_production_start_20260925/PROGRESS.md) 최상단.
>
> **2026-09-30 12:00 PDT 실측:** GPU 훈련 **미제출**. 실행 중: **Athaliana ONT C0 재빌드 6182077**(source_v13, union-find, 160G), **Gmax HSP 6181301**(200G) → protein C0 6181302 → zero 6181303–05(afterok). 나머지 C0는 전부 PASS, zero list 24/27. 병합·obs cache는 1종 시험으로 검증했고 규모 문제는 코드로 해결(병합 DuckDB 한도·PK 미선언, obs cache 염색체 단위, A16 union-find). 모든 수정은 Codex 리뷰를 받았다. **다음:** 두 체인 완료·검증 → 본 병합(source_v13, ~200G, `--memory-limit 170GB`) → 코퍼스 `--c0` 재검증 → obs cache(220G) → bindings → DeltaAI → **파일럿 제출 전 사용자 확인**. Slurm 기본 시간 제한은 전역 13일(pronghorn `~/.bashrc`). 이번 세션 notes/.atl 변경은 **커밋하지 않았다.**

## 09-28 기록 (역사): B6 실데이터 증거 DB 구축 (C0 마무리)

> ⚠️ 역사: 당시 정본은 [RESTART_HANDOFF_20260928](../notes/RESTART_HANDOFF_20260928.md)였다. 코드 위치·C0 표·실행 중 잡·다음 단계·저자 결정 1–18이 모두 거기 있다. 세부 실행 로그는 [PROGRESS.md](../notes/b6_production_start_20260925/PROGRESS.md), 결정 원문은 [AUTHOR_DECISIONS](../notes/b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md). 아래의 09-27 이전 블록은 **역사 기록**이다.
>
> **2026-09-28 14:30 PDT 실측:** GPU 훈련 **미제출**. b6-integration HEAD `27c34ba`(`/tmp/b6-integration`, 미병합·미push). C0 PASS: EST 9종, PacBio 3종, ONT 4종(Ppatens·Ptrichocarpa·Sitalica·Gmax), protein 6종. 종 exhaustiveness 게이트: 전 종 통과 또는 통과 예상(Bdistachyon·Gmax N=100, Vvinifera N=500). **B6 코퍼스 완료·검증 PASS**(`$PREP/corpus/b5_full_b6_v1.db`, train_rows 196,774, **N_pilot 4,099**). 실행 중: HSP Vvinifera 6173197·Gmax 6173591·Osativa 6173182 → protein C0 6173789·6173790·6173791, ONT C0 Athaliana 6173686·Osativa 6173687. 감시 20분 간격. **다음:** C0 완료 → zero list 27 → 병합 → 코퍼스 `--c0` 재검증 → obs cache → bindings → DeltaAI → 파일럿 제출 전 사용자 확인.

## 이전 기록 (2026-09-25 ~ 09-27, 역사)


> **현재 작업 정본: [운영학습 착수 준비](../notes/b6_production_start_20260925/PROGRESS.md).** 사용자 운영학습 시작 승인 후 실제 입력을 준비 중이다. **GPU 훈련은 아직 미제출**. 동결 순서는 real4GPU WP5파일럿4142업데이트 후 go판정, 그다음 WP6다.
>
> **09-27 최신:** 결정 12–18 반영. b6-integration HEAD `27c34ba`(N_pilot 4,099, drop 모드, 종별 flags, 상대경로, 파일럿 런처, HSP/A16 수정). PASS: EST 9종(Bdistachyon·Vvinifera N=100), PacBio 3종(+Vvinifera N=100), ONT dRNA 2종, protein 6종(Sitalica·Ptrichocarpa·Bdistachyon·Sbicolor·Athaliana·Ppatens). 진행: Vvinifera N=500 체인(감사6173193→C0 6173194–96→HSP6173197→protein6173789), Gmax ONT N=100 6172725·HSP 6172729, Osativa HSP 6173182(4d)→protein6173791·ONT 6172995, Sitalica/Athaliana ONT 6172712/13, 코퍼스 apply 6173268. interval manifest 9종 C0 기록과 일치. 다음: C0 완료→zero list 27·병합→obs cache→bindings→DeltaAI.
>
> **09-26 오전:** 결정 9–11 구현(b6-integration `396fcbf` HSP 행정책, `0fc80de` A16 연결성분), source_v3 배포·컨테이너시험 186 pass. 재제출: HSP v3 6172692–6172708(짝수), protein C0 6172693–6172709(홀수, afterok), ONT C0 v2 6172710–13(Vvinifera·Gmax·Sitalica·Athaliana). Osativa ONT 6172409 구코드 실행중. PASS: EST9·PacBio3·ONT dRNA2(Ppatens·Ptrichocarpa, 채택률2–3%는 identity0.95 규칙).
>
> **09-26 새벽:** 전면 병렬화. 실행/대기: PacBio Sbicolor 6172283, ONT C0 Osativa6172409·Sitalica6172410·Gmax6172411·Vvinifera6172567·Ptrichocarpa6172568·Ppatens6172569, HSP 9종 6172570–78(코드6b6dcd3), protein C0 9종 6172579–87(afterok HSP), FLIC 6172527/28→metadata6172588→Athaliana ONT C0 6172589, col0 retry 6172542. 완료 PASS: EST 9종, PacBio Vvinifera·Ptrichocarpa.
>
> **09-25 밤:** 결정 5–8 받음(HSP X 인정(a), protein 역할행 추가, cui2020 재사용, FLIC 별도정렬). protein roles 새파일 `DATASET_ROLES.longread_protein_v1.tsv` 생성(기존 무변경). Athaliana ONT metadata 6172498(35런) 제출. HSP 계약수정·FLIC 드라이버는 에이전트 작업 중. PacBio Vvinifera·Ptrichocarpa C0 PASS, Sbicolor PacBio·ONT 6종 C0 진행중.
>
> **09-25 17:45PDT:** EST C0 **9종 전부 PASS**(Vvinifera 6172296 포함). HSP 파일럿6172318 실패: DIAMOND 기본 masking이 BTOP에 `X`를 남겨 계약 재생(codon/subject truth)과 충돌(약5.5%행) → 의존잡 6172319–26 취소, **해결방식 저자결정 대기**. ONT metadata 6172290 90/90 완료 → **ONT C0 6종 6172409–6172414 제출**(Osativa168M reads 14d 등). PacBio C0 3잡 RUNNING(~5GB). 결정 대기: HSP masking 처리, protein roles 행, Athaliana cui2020 정렬 재사용·FLIC 정렬 방식.
>
> **09-25 17:00PDT:** 저자결정 3건 반영([기록](../notes/b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md)): UMI 125런 absent 유지, Vvinifera 누락175 library=`NCBI-EST`, extern OOM 전역 무시. 주말 무중단 진행 지시. **EST C0 8종 PASS**. 진행중: Vvinifera EST C0 6172296, ONT metadata 배열6172290(6종90런), PacBio C0 6172281–83, HSP투영 6172318(파일럿)→6172319–26. 보류: Athaliana ONT(정렬 경로/검증런 미정렬), **protein C0는 DATASET_ROLES protein행 부재로 저자결정 필요**. 코드 `d929ba2`(b6-integration, --missing-library).
>
> **09-25 16:20PDT:** 사용자 지시로 **C0 10잡 제출**(CPU, GPU아님). EST7종 6172274–6172280(Athaliana/Gmax/Osativa/Bdistachyon/Ppatens/Ptrichocarpa/Sbicolor), PacBio3종 6172281(Vvinifera)/6172282(Ptrichocarpa)/6172283(Sbicolor, 25.1M reads, 7d). 상세 [PROGRESS §전체 C0](../notes/b6_production_start_20260925/PROGRESS.md). 보류: Vvinifera EST(library누락 결정), ONT(UMI 결정), protein C0(EST C0 이후). 기록커밋 브랜치 `b6-production-start-20260925`(main 무변경). 다음: 각 C0 QC PASS·해시·스키마 검증 → A16 후처리·병합.
>
> **09-25 15:45PDT:** **Sitalica EST C0 재시도 6172228 COMPLETED 0:0(3m06s), 생산자 QC status=PASS**. reads_in66,040/accepted31,723/retained19,310, molecules36,510, exhaustiveness0.99984, leakage_nonzero0, 모든gate_*0, tiles_verified19,536/19,536. 출력 `c0/Sitalica_est_v1/out_arrow_retry/Sitalica.est.c0.db` sha256 `02116ca6…1064`(로그·QC일치). ✏️ 15:53 DB해시재계산·스키마(19테이블, 행수=QC)검증 완료. est_protein_hsp0행은 설계상 정상(route-B는 protein C0에서 옴). 남은 확인: extern OUT_OF_MEMORY 0:125(6172167·6172228 공통) 원인·영향, completeness 31,723행 전부 NULL(EST-only 임시정책). **이것은 CPU 증거DB 1종 PASS이며 전체게이트·training input 완성·GPU훈련 승인 아님.** UMI 조사보고 [umi_research.md](../notes/b6_production_start_20260925/umi_research.md) 수령: confirmed_no_UMI 5런(표준kit 근거), UMI_present 0, unresolved 120런(이 중 direct_RNA 42런은 A16상 read=unit). 동결TSV는 125런 모두 `absent`라 근거와 불일치 → **UMI값 저자결정 대기, TSV 무변경**. Vvinifera library누락175records 저자결정 대기 유지. 6172228 완료, 반복조회 금지. 다음 실행: C0 DB 독립검증 후 나머지 종/소스 C0 구축.
>
> **09-25 15:25PDT:** PacBio49 metadata6172167 모두 완료/해시검증, Gmax수정10f2aae·재처리6172217 완료(1,374,461reads), EST총8종준비. Vvinifera175/450180records library누락(0.03887%) 처리필요. **현재 모니터는 pronghorn CPU6172228(✏️ 당시 RUNNING → 15:45 COMPLETED/PASS, 위 참조), Sitalica EST C0 첫빌드**, DeltaAI아님. 최초6172220 PyArrow누락실패보존, 격리DuckDB1.5.5+PyArrow25.0.1/Stage smoke 및원본genome·GFF·split해시검증후새out_arrow_retry로진행. ONT protocol83cDNA/42dRNA 저자승인, UMI는a64e8066-db8d-46f 조사위임·검토대기. 완료된GPU3210585/PB6172167/Gmax6172217반복조회금지. B5체인변경없음.

## 이전 완료 기록: GPU 검증 및 NVMe 정리

> **최신 인계: [RESTART_HANDOFF_20260924_PI](../notes/RESTART_HANDOFF_20260924_PI.md).** 사용자 지정 Claude 세션 `1a00e258-5d2a-4bd8-a5a4-61423fb2e076`을 복구했다. 아래 09-23 내용은 역사 기록이며 최신 실행 지시가 아니다.
>
> **09-25 12:55 PDT 최신:** 사용자 승인으로 NVMe 정리까지 완료. 종료된 B6 4트리를 HDD `/work/hdd/bilv/wcyim/transgenic/archives/b6_nvme_20260925/payload/`에 전수SHA256 검증·fsync 후 NVMe 원본만 제거했다. 파일414개/528항목 보존, **146.262GiB 확보**, NVMe 사용548.266→402.004GiB(soft여유97.996GiB). B5·source repo·메모리와 최신HDD PASS결과 보존. 현재 승인된 검증/정리 TODO 완료, 운영학습 미실행. [정리 정본](../notes/b6_nvme_cleanup_20260925/REPORT.md).
>
> **09-25 01:03 PDT 이전 검증 기록:** **메모리3208285·WP3 CP43210279·synthetic CP43210585 원본검증 모두PASS**, 활성B6잡없음. NCCL7시험·재개byte/hash·generation·compile_policy·메모리연계·실행후source220파일 확인완료. 코드3511a5a/원격e2a91b7. 요청된GPU검증구간완료, 운영학습/생산승인/main병합/푸시없음. 남은현재TODO는NVMe147GiB정리승인대기. 30분예약은유지하되 새잡없음, 완료결과반복검증/재제출금지. [최종 정본](../notes/b6_cp4_20260925/FINAL_VERIFICATION.md).
>
> **09-24 23:07 PDT 실행 기록:** 3208442는 NVMe quota 저장실패. checkpoint collective오류전파·poolquota검사 수정 `b4dd15e`(원격860bbfc), 회귀·독립검토통과. **새 HDD CP4 3209815 PENDING**, 2GPU/1h30m, 예상09-25 06:59PDT(변동가능). 실제NCCL fault7시험 통과후정식CP4. 30분예약대상3209815, **HDD** source/log=`/work/hdd/bilv/wcyim/transgenic/repo_b6_io_20260924`, report=`/work/hdd/bilv/wcyim/transgenic/runs/b6_int4/run_3209815/report.json`. 기존NVMe147GiB는보존, 공간회복별도승인필요. B5무변경. [정본](../notes/b6_cp4_20260924/quota_incident/PROGRESS.md).
>
> 이전 09-24 15:29 PDT 실측: `/tmp/b6-integration` HEAD `2656dab`, DeltaAI 스냅숏 `de15df0`. 2-GPU 메모리 벤치 **3205730 PENDING(Priority)**, `ghx4-interactive`, 예상 시작 17:31 PDT(변동 가능). 결과 JSON/로그 없음. 로컬/원격 벤치 및 CP4 4파일 해시 일치. **벤치 실측 PASS 확인 후에만 2-GPU CP4(wp3-fixtures, synthetic)**. 재제출·운영 학습·B5 변경 없음. 자동 감시는 새로 시작하지 않았다.

## 09-23 보존 기록

> 📌 **먼저 [RESTART_HANDOFF_20260923](../notes/RESTART_HANDOFF_20260923.md)를 읽는다.** 이 파일은 요약이고, 상세는 인계 문서가 정본이다.
> 다음 행동은 [todo.md](todo.md) 맨 위 체크리스트, 다시 하지 말 것은 [quarantine.md §1o](quarantine.md)에 있다.
> 이 날짜 이전의 resume 전문은 [archive/resume_before_20260923.md](archive/resume_before_20260923.md)에 바이트 그대로 보존했다(sha256 `3374fe68…`).
> 그 안의 "최신" 표시는 모두 **과거 기록**이다.

## 지금 상태 (2026-09-23 18:00 PDT 실측)

- **프로토콜:** A46 v1.36 승인, A48 v1.40, **A49 v1.41 + A49.1**(09-23; 완전성 코드는 병합 후 계산 패스, exhaustiveness 종 + 소스별 하한 0.999, 모집단별 HSP 충돌 해소, CAGE 입력 종별 지정, 라이브러리 키 = ENA experiment 또는 `<sample>:<library_name>`). sha256 `687eac42…`, 미커밋.
- **WP1 코드:** worktree `/tmp/b6-wp1-20260922`(미커밋). 수정 라운드 4–8 후 **전체 시험 186 통과, Codex 6차 APPROVE(지적 0)**. 보고서 `tasknotes/WP1_FIX_ROUND{4..7}_REPORT.md`.
- **감사 정렬 6170748:** 51/187 DONE, REFUSED 0, 122 GB. 나머지 대기(`%4`).
- **메타데이터 시트(저자 검토용 초안):** `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv`(143런, 프로토콜·UMI·라이브러리 키), `PACBIO_LIBRARY.b6_v1.draft.tsv`(72행), `CAGE_INPUTS.b6_v1.tsv`(종별 CAGE 입력). 노트 `notes/ONT_PROTOCOL_SHEET_20260923.md`, `notes/CAGE_INVENTORY_20260923.md`.
- **다음(지시 필요):** WP1 브랜치 커밋 / CAGE 세트 pronghorn 다운로드 잡 / NCBI EST esummary 수집 / 감사 완료 후 C0 빌드.

## 저자 결정 대기

1. EST BioProject 값: 종 단위 EST에는 실제 단일 BioProject가 없다. `NCBI_EST:txid<taxid>`(권고)와 종별 대표 프로젝트 중 하나를 고른다.
2. `DATASET_ROLES.b6_v1.tsv`를 별도 파일로 둔 해석.
3. N 포함 EST: 실측 후 판단.
4. NCBI EST 요약 수집(약 1–1.5 h) 착수.

## 무접촉

커밋·푸시·병합은 하지 않았다. 승인 대상 파일(프로토콜, MVP 문서, 패널 TSV)도 git에 없으므로 sha256이 동결 기준이다.
B5 seed123 체인과 GPU 잡은 이번 세션에서 조회도 변경도 하지 않았다.
MVP 코드 worktree `/tmp/mvp-seed123-integration-20260917`의 HEAD는 `729b990`이다.

## 세션 복구 규칙 (AGENTS.md)

`.atl/resume.md` → `.atl/todo.md` 최상단 → `.atl/quarantine.md` 순서로 읽는다. 잡을 다시 제출하기 전에 Slurm 상태를 실측한다.
구현 브랜치는 main에 병합돼 있지 않으므로, 해당 worktree에서 graft를 build/query한다.
