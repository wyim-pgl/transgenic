# 재시작 정본: 2026-09-25 실제 B6 훈련 착수 준비

> **현재 작업 정본: [운영학습 착수 준비](../notes/b6_production_start_20260925/PROGRESS.md).** 사용자 운영학습 시작 승인 후 실제 입력을 준비 중이다. **GPU 훈련은 아직 미제출**. 동결 순서는 real4GPU WP5파일럿4142업데이트 후 go판정, 그다음 WP6다.
>
> **09-25 16:20PDT 최신:** 사용자 지시로 **C0 10잡 제출**(CPU, GPU아님). EST7종 6172274–6172280(Athaliana/Gmax/Osativa/Bdistachyon/Ppatens/Ptrichocarpa/Sbicolor), PacBio3종 6172281(Vvinifera)/6172282(Ptrichocarpa)/6172283(Sbicolor, 25.1M reads, 7d). 상세 [PROGRESS §전체 C0](../notes/b6_production_start_20260925/PROGRESS.md). 보류: Vvinifera EST(library누락 결정), ONT(UMI 결정), protein C0(EST C0 이후). 기록커밋 브랜치 `b6-production-start-20260925`(main 무변경). 다음: 각 C0 QC PASS·해시·스키마 검증 → A16 후처리·병합.
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
