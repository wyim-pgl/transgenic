# Active TODO 교차문서 감사 — 2026-09-17

## 범위와 결론

`.atl/todo.md` **전체를 읽고 현재 미체크 29개를 전수 대조**했다. 아래 번호는 문서 줄번호와 함께 1:1 대응한다. archive TODO 감사는 별도 담당이며 이 보고서는 재감사하지 않는다. 코드/테스트 실행, GPU 재조회, 원격 쓰기/잡 제출, source 수정, merge/push는 하지 않았다. 의도한 유일한 쓰기는 이 보고서다. `NEXT`는 후속 작업의 구체적 시작점이지 이번 턴의 실행/승인 선언이 아니다.

**모두 GPU 때문에 막힌 상태는 아니다.** GPU 실측·학습·실디코딩은 predecessor 차단이지만, 새 eval source 소비계약/decoder repin, ONT full wrapper와 도구 fixture 준비·검토, source/통계 계약 대조, native crash 증거 정리는 로컬에서 가능한 잔여 작업이다. 과학정책을 임의 선택하거나 extern OOM gate를 지우지 않고 진행할 수 있다.

최신 GPU 상태는 controller가 제공한 `notes/power_recovery_20260917/todo_audit_live_gpu.txt`를 사용했다. 원문 `2026-09-17T12:06:07-05:00`은 **10:06:07 PDT**: **3163986 PENDING/Priority/elapsed0**, B5 **3131538 RUNNING 10:28:38/gh024**, 3149356–3149363 Dependency. 본 담당은 중복 조회하지 않았다. 이전 00:34/22:12/agent 진행중 표시는 현재 상태가 아니다.

### 분류

- **완료/대체**: 구현·검토 또는 기존 지시가 이미 대체됨. 실제 runtime 완료와 구별.
- **부분**: 혼합 체크박스 중 완료 부분을 제외한 실제 잔여가 존재.
- **로컬**: 원격 실행/정책 선택 없이 준비·구현·시험·검토 가능한 후속 범위. 이번 감사에서 실행하지 않음.
- **선행 차단**: 실제 GPU/체크포인트/유효한 증거/예산·운영 게이트가 먼저 필요.
- **승인/과학결정**: 기록된 저자·과학적 결정이 필요. 자동 승인하지 않음.

하나의 체크박스에 복수 상태를 병기한다. 29개를 29개의 서로 독립된 실행잡으로 취급하지 않는다.

## 근거 정본 (아래 표의 약칭)

아래 markdown 문서는 발췌 검색만이 아니라 전체를 읽었다. 기계 receipt는 해당 검토의 주장과 구별하며 이번 감사에서 새로 재검증했다고 주장하지 않는다.

| 약칭 | 근거 경로 / 의미 |
|---|---|
| REC | `notes/POWER_RECOVERY_20260917.md` — 마지막 01:17 결과가 이전 블록을 대체 |
| WAIT | `notes/WAIT_WORK_PLAN_20260916.md` — Task1–4와 최신 복구 머리말 |
| DEC | `notes/MVP_DECODER_CONNECTION_20260916.md` — 구현/수정 이력; 뒤의 SRC가 초기 미구현/미통합 상태를 대체 |
| SRC | `notes/mvp_execution_20260916/final_source_report.md`, `final_source.independent_review.md` — 실행362eb899/보고b660bf1, 최종 source 수용 |
| DEP | `notes/mvp_execution_20260916/final_p1/deployment.independent_review.md` — 정확 b660 S0/P1 패키지 한정 검토 |
| PANEL | `notes/power_recovery_20260917/panel_result/REVIEW.md`, `panel_arrays/REVIEW.md` — 새6165628의 실제50mapping/150artifact/수치 수용 |
| ONT | `notes/wait_work_20260916/longread/REPORT.md`, `RESULT_REVIEW.md` — wrapper 경계, 실제6165555의 payload와 운영 차단 |
| REM | `notes/power_recovery_20260917/remaining_tracks.md` — 준비 가능한 범위. 이 문서의 “actual50 pending”은 REC/PANEL로 대체 |
| PHASE | `notes/EST_PHASE_DESIGN_20260916_1530.md`, `notes/power_recovery_20260917/phase_input_inventory.md` — 과학계약/실입력 후보 및 미충족 closure |
| HSP | `notes/EST_HSP_REAL_RESULTS_20260916.md` — 144row/132elicited/12not-elicited/0invalid, INCONCLUSIVE 유지 |
| C1 | `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md` — 6165525 technical PASS, extern/science 별도 |
| REPRO | `notes/wait_work_20260916/repro/RUNBOOK.md` — 로컬 preflight와 native crash 증거 한계 |
| SPEC | `revision/protocols/B6_MVP_PILOT_v1.md` — 현행 문서 전체. draft 머리말/옛 best·명령 sketch와 후속 위임·고정 결정 구별 |
| LIVE | `notes/power_recovery_20260917/todo_audit_live_gpu.txt` — controller 제공 최신 관측 |

`.atl/resume.md`도 전체 확인했다. Source 탐색 이전 graft map/ask를 수행했으며 이 작업은 문서 감사로 한정했다. Advisor 검토 요청은 client/model-version 오류로 실패했으므로 자문 수용/승인으로 계산하지 않는다.

## 현재 미체크 29개 전수 매핑

| # / `.atl/todo.md` | 판정: 완료 부분 / 실제 잔여 | 구체적 NEXT / 근거 |
|---|---|---|
| 01 / L11 후속실행 종합 | **부분·로컬·선행 차단·과학결정**. 실제50 수용/공유 identity는 완료. GPU, ONT full, C phase는 서로 다른 잔여다. | 아래04/14/16/21–25로 분리하여 관리. GPU대기를 전체 로컬 중단으로 해석하지 않는다. REC, WAIT, REM. |
| 02 / L28 partialcode→통합검토→P1 | **완료/대체 + runtime만 선행 차단**. 저장 partialcode 검증·독립통합·배포는 b660에서 완료, 최종 P1도 이미 제출. `.venv` 배제 확인됨. | 새 partial 복구/재통합/중복제출 없음. 3163986의 실제 S0/P1 결과만04에서 회수. SRC, DEP, REC, LIVE. |
| 03 / L34 d99b P1배포/옛 담당 준비중 | **대체됨**. d99 3163545는 owned pending/사용0 확인 후 취소, 최종 b660 3163986으로 교체 완료. resource code도 최종 통합에 포함. | 옛 agent·d99 release 재개 금지. 최종 소스의 후속5gate 계획만04/11에서 계속. SRC, WAIT, REC. |
| 04 / L39 S0/P1→5gate→readiness→3seed | **선행 차단**, 후속 도구는 준비됨. LIVE상 P1 미실행. | controller가 기존3163986 완료 시 wrapper/DONE·S0 staging·accounting 회수; PASS 후 실제 mount/argv/release와 누적예산을 대조하여 step0→generation→bf16→4rank→resource를 검토. seed2/3 별도 binding/6receipt 필요, seed1 PASS 복사 금지. WAIT Task1, DEP, LIVE. |
| 05 / L40 finalizer/countproducer→decoder구현→300평가 | **구현은 완료/대체; 부분·로컬·선행 차단**. finalizer provenance/rawGFF reference-count/350 decoder는 b660에서 통합 수용. reference caller는 eval9603으로 후속 수정. 독립 junction/GFF producer 연결은 아직 잔여. | 새9603 producer/manifest를 소비하는 decoder package/pin/closure를 검토·repin하고 positive/negative 계약시험 준비. 실제 출력은 300 paired+50 baseline=**350**. ONT junction 결속은18, 실디코딩은17. DEC, SRC, PANEL, ONT. |
| 06 / L49 GPUprobe/B디코딩/Cpilotreceipt 후속로컬구현 | **복합 stale/부분**. 실제 GPUprobe 구현·B decoder/counts·C1 pilot receipt reconciliation은 완료. “모든 Slurm/pin이 synthesis8결정 대기”도 후속 위임·선택·제출을 반영하지 못함. C accepted-event/phase는 미완료. | probe/decoder/C1 다시 만들지 말고05/16/21–23의 미완료 계약만 추진. HiFi·C2단위·phase·production 승인은 여전히 분리. SRC, C1, REC, PHASE. |
| 07 / L50 격리/불변snapshot/출력/owner·예산 | **부분·지속 운영 의무**, 일회 미구현 아님. 기존 isolation/release/ledger는 구축됨. | 다음 실제 package마다 source/eval 분리·owner·fresh output·예산을 연결한 새 snapshot 작성; 기존 ledger 덮어쓰기 금지. 제출 직전 controller 단일 live검사. WAIT Task4, REC, REPRO. |
| 08 / L58 제출 전 live조회/B5보존 | **이번 관측 완료; 향후 제출 전 재확인 의무**. LIVE10:06 최신 확인 확보. | 이번 감사 중 추가 query 없음. 미래 제출 직전만 controller가 다시 확인; B5 chain/checkout 변경·456/789 재개 없음. LIVE, REC. |
| 09 / L60 epoch05/sidecar staging·compute hash | **선행 차단**. concrete epoch05 공통 선택은 완료, 실제 최종 S0 staging 완료는 아직 아님. | 3163986 S0의 staged checkpoint/sidecar와 compute 해시를 선택기록에 대조. mutable best 또는 기존 CPU fixture로 대체 금지. DEP, SRC, resume. |
| 10 / L61 SIF·runtime/tokenizer/Hyena identity | **부분·선행 차단**. SIF pin/기록된 해시는 존재하지만 최종 live runtime/source 적용 수용은 미완료. | S0/P1 및 후속 실제 gate의 SIF bytes/versions/offline assets/loaded module closure를 대조. 로컬 Torch2.6·Transformers4.46.3 통과나4.57.6 skip을 실제 SIF 통과로 계산하지 않는다. SRC, DEP. |
| 11 / L62 실제 사전시험 목록 | **구현·소스검토 완료, 실행 선행 차단**. “probe 미구현” 아님. |04의 순서대로 실제6mode receipt 획득·검증. generation의 짧은 parity만으로8192 constrained decode를 인증하지 않는다. WAIT Task1, SRC, LIVE. |
| 12 / L63 실제 launch-readiness 작성 | **선행 차단**. 준비도구는 있으나 실제 모든 gate PASS 기록 없음. |6gate·각 seed binding·input/source/checkpoint/SIF·allocation-wide 사용량·자원허가를 하나의 실제 readiness로 생성하고 독립 검증. CPU/placeholder 인증 금지. SRC, WAIT, REPRO. |
| 13 / L64 3seed×250updates/12warmup | **선행 차단**. 학습 미실행. | readiness와 별도 학습예산 확정 후 seed1/2/3 실행계획 수용; 완료 시 finalizer/paired files/checkpoint/실하드웨어 확인. 5node-hour 진단상한을 학습예산으로 오인하지 않는다. WAIT, SRC, SPEC. |
| 14 / L73 실제50compat·duplicate·독립read·eval배포 | **부분**. 실제50bijective map·좌표/strand·restricted GFF50gene52transcript·duplicate-ambiguous0 및 arrays 검증 완료. **로컬** decoder consumer repin/배포계약 잔여; **과학결정** duplicate/독립read 조건, **선행 차단** 실디코딩. | 기존6165628 receipt를05의 새 eval consumer 시험에 결속. duplicate0은 이번 패널 관측이며 dedup/F1정책 승인으로 확대하지 않음. full-reference/window equality·DBselection 완전성 독립 재검증은 아직 주장하지 말고 실제 downstream 요구에 맞춰 구분. PANEL, REC. |
| 15 / L75 승인spec↔scoring/역할/panel 계약 대조 | **부분·즉시 로컬**. 여러 component 검토는 완료하였으나 최신9603+독립read 전체 release 계약은 별도. SPEC 자체는 draft 머리말과 오래된 sketch를 보유. | 현행 spec 조항→후속 저자 metric/epoch05/250update 결정→SRC/9603/ONT 계약의 작은 차이표 작성·독립검토. 문서의 옛 best/미구현/sketch를 재실행 지시로 취급하거나 후속 명시 승인을 무효화하지 않는다. SPEC, SRC, ONT, resume. |
| 16 / L76 A-ONT1/A-HiFi 소스·정렬 | **부분·로컬·운영/승인 차단**. ONT3 canonical 역할/경로와1run orientation payload 검증 완료. full alignment/helper 실제 fixture 미검증. HiFi canonical 역할/경로 미확보. | 로컬 full wrapper의 executable identity·capture·publication/postflight와 실제 SAM/BAM/junction fixture를 준비·검토. 원격 full은 externOOM 설명/실도구검증/throughput·scratch·예산 충족 후 단일 run부터. HiFi 확보/정체성 또는 ONT-only 정책은 별도 결정, 대량fetch/3runarray 자동 실행 없음. ONT, REM. |
| 17 / L77 checkpoint후 패널50 디코딩 | **선행 차단**. driver는 구현됨; final checkpoints/진짜 runtime/새 eval package 인증 미완료. |05/14 repin 및 실제 generation/runtime gate+3finalizer 수용 후 동일 조건으로350 files 생성, raw tokens/GSF/GFF 및 hashes 보존. EST input과 ONT/HiFi scoring evidence 분리. DEC, SRC, PANEL. |
| 18 / L78 reference F1 vs 독립 splice 평가 | **부분·로컬 + 선행 차단**. 실제GFFCompare16case semantic 검증 완료; 생물학 최종평가 아님. 독립 junction/GFF counts 결속 미완료. | ONT validated table+decoder GFF→기존 junction_counts/scorer로 이어지는 provenance producer를 준비·시험·검토; 실제 평가는 full evidence+350출력 이후. per-run·pooled BAM을 threshold 전에 결합하고 precision window/recall gene 분모 유지. ONT, SRC, DEC. |
| 19 / L79 bootstrap/decision/withholding | **부분**. paired bootstrap/세seed/STOP우선 수학과 CPU시험은 구현됨. 실데이터 실행·최종 availability/withholding 연결 미완료. **로컬** 계약시험, **승인/선행 차단** 최종판정. | 기존 수학을 재구현하지 말고 실제 finalizer pair 및 reference/independent count 계약에 연결. STOP 경로에서 auditable library key 및 승인spec의 run/정당화 가능성을 조사; 불가능은 unassessable→INCONCLUSIVE이며 임의 동등성/STOP 금지. HiFi/duplicate/coverage 미결을 bool로 우회하지 않는다. SPEC§5.2–5.4, SRC, ONT. |
| 20 / L80 결과·한계·고정입력·hash 보고 | **부분·선행 차단**, 보고서 틀/한계 정리는 로컬 가능. 현재는 준비/진단·패널 관측뿐. | 현행 ledger를 결과보고 구조에 연결하고 실제 primary/F1/structural/bootstrap 결과가 있을 때 채움. 탐색MVP label, missing/unassessable, 외부 OOM, source/runtime 범위 명시. 결과 전 성능/유효성 주장 없음. REC, SPEC, ONT. |
| 21 / L85 HSP/phase 의미론·miniprot/A11 | **부분·로컬·과학결정/입력 차단**. 9종 routeA raw와+/−후보 inventory 완료. 실제 replay/전체A11 미완료. | 보존 후보의 DNA/protein/phase/Target·stop/partial 검증 fixture 입력계약과 exact identity manifest 준비; A11 member/library/mate/split/exclusion 누락목록 정리. 후보를 인증 fixture로 부르지 않음. 기존 C1 chain 재사용, legacy121 PROVENANCE로 C1 재개방/스크리닝 재실행 금지. PHASE, C1. |
| 22 / L92 routeB ≥30aa/frameshift·BTOP 확정 | **과학결정 필요 + 구현 잔여**. column9=M+I+D(excludes F)는 actual144row로 뒷받침되지만 biological acceptance unit 승인은 없음. | 저자가 whole-HSP column9≥30인지 query/paired/segment 단위인지 명시 확정하도록 결정표 제공. 그 계약 후 accepted-HSP producer/시험 구현·검토. column9만 확인하려고 engine 재실행 불필요; 필요시만 새이름12query ordered−1 fixture를 사전 고정·별도검토. HSP, REM. |
| 23 / L94 eventproducer/routeAadapter/충돌·raster | **부분·로컬·과학결정**. geometry5c46c75 수용, production projection 미구현. | accepted-event I/O identity/ordered-query codons 계약 및 routeA replay fixture 준비. phase label/order, nonprimary, ts contradiction, insertion, whole-HSP/self/cross-member conflict, pair-local vs final-cell masking 선택 기록 후 adapter/집계/raster 구현·검토. raw coverage TSV로 phase 추정 금지; MVP8channel 변경 없음. PHASE, HSP. |
| 24 / L95 전종 수용/투영 잡→QC | **선행 차단**. scientific contract·eventproducer/adapter·실fixture·A11/input closure·실행리뷰/예산 필요. |21–23을 충족한 뒤 별도 namespace/입출력해시/전종 disposition QC를 포함한 bounded 실행계획 검토. blastx raw 재검색/실패본 덮어쓰기 금지. exploratory 실행을 전체A46승인에 일괄 종속시키지 않되 production 권한과 구별. PHASE, REC. |
| 25 / L106 extern OOM 조사 | **미해결·로컬 증거정리 가능/추가 site증거 필요**. C1/ONT/panel batch 성공과 독립. 지원초안은 작성되었으나 미발신. | 기존 accounting·payload/RSS 대조표에 victimPID/cgroup/kernel/slurmstepd/epilog 미확보를 명시하고 지원요청 초안을 검토. 실제 site자료 획득/외부발신은 controller 권한·승인 경계 준수; 임의 원인 확정·clean 승격·retry 금지. ONT, C1, REPRO, REC. |
| 26 / L107 A46동결/fullB6/C0/secondary/phase본학습 | **별도 승인·범위**. 탐색MVP/진단 승인으로 완료 불가. | 해당 결정권자에게 별도 승인 항목으로 유지; 현재MVP 학습에 phase완료/A46전체동결이라는 새 선행조건 추가 금지. PHASE, SPEC§7, resume. |
| 27 / L108 main통합/푸시 | **로컬 diff 검토 가능; 통합/발신 결정 별도**. 구현브랜치 미병합, 기존 미커밋 변경 보존. | main과 실제 선택된 b660/9603 source 및 문서의 격리 diff·범위/검토 이력 정리 후 결정. 감사 중 merge/push 없음; branch 존재를 main반영으로 보고하지 않음. SRC, REC. |
| 28 / L109 B5 456/789 재개/seed123 추적 | **저자결정 대기**. 현재3131538이 seed123 chain에서 실행중. | LIVE를 최신 관측으로 연결하고 기존체인 보존.456/789는 명시 결정 전 제출/재개 금지. 반복무한 monitor 없음. LIVE, resume. |
| 29 / L110 논문/위키/spec/A37/#61/AtRTD3/PacBio | **archived 항목 담당에 위임된 교차연결**. 이 한 줄을 완료로 간주하지도, 이전 항목들을 여기서 재실행하지도 않음. | 별도 archive 감사 결과를 최신protocol/issue와 결합해 진짜 잔여/결정을 현재 목록에 연결. 이 보고서는 세부 archive 판정을 유보한다. `.atl/todo.md` L110; 별도 담당 범위. |

## WAIT_WORK_PLAN의 남은 체크박스도 현재 상태로 재해석

이 표는 위29개에 중복되지 않는 잔여를 드러내고 과거 체크의 재실행을 막기 위한 보조 대조다.

| 위치 | 현재 판정 / NEXT |
|---|---|
| Task1 G1/G2 실행조건 | **실제 P1·mount·accounting/예산 차단**. generator26tests/수정검토는 완료. 실P1 결과를 받아 exact mount/argv/release 검토. 위04와 동일. |
| Task1 controller후 bounded 실행 | **선행 차단**. 구조계획 수용≠잡 승인. P1 PASS전 후속GPU제출 금지. |
| Task2 독립source/object/runtime/deployment검토 진행중 | **stale/완료·대체**. REC의 새9603 release 검토·6165628 publication·PANEL 수용으로 대체. 이전 agent 활동을 현재로 해석하지 않음. |
| Task2 6165556 실패→bindfix→6165559→9603 긴 미체크 항목 | **실패 원본 보존 + 수정실행/실제50수용 완료**. 6165556 실패127과6165559의0/50을 성공으로 바꾸지 않되 새6165628이 문제를 해소. 새panel 재생성 필요 없음. decoder 소비계약만14에 남김. |
| Task3 PILOT_COMPLETE 차단 | **현재 유효**. orientation payload/산술 검증 완료이나 externOOM 때문에 운영 gate 미충족. FAIL/use_uf=false는 생물학적 음성 아님. 위16/25. |
| Task3 작은 bounded 검증 | **로컬 fullwrapper/실도구 fixture 준비·검토 가능**, 원격 실행은 운영/예산/배포조건 이후. “추가검증”을 orientation job 반복으로 읽지 않는다. |
| Task4 nativecrash 원인/새ledger | **부분**. 새panel/ONT snapshot ledger는 REC에서 완료. **native SIGSEGV 원인 미해결**; GPU와 무관한 로컬 증거조사 가능. 아래 참조. |

## Native crash: 현재 TODO에서 빠뜨리면 안 되는 실제 잔여

근거 SRC/REPRO: whole-repo 시험은 exit139/약22%에서 SIGSEGV. scoped599PASS/1SKIP 또는 의존성 복구는 native 원인 해소가 아니다. 정확 b660 line141은 `ids,mask` device변환이고 tracks는142; 옛 검토의 표현을 그대로 반복하지 않는다.

**NEXT (로컬):** 보존 로그에서 실제 original argv/cwd/PYTHONPATH/plugin/native-library 정보와 ordered test/predecessor 목록의 확보 여부를 목록화하고 기존 core/backtrace 제공 가능 여부를 확인한다. 없는 것을 새 현재환경으로 소급 대체하지 않는다. 최소 predecessor를 특정하기 전 blind whole-suite 재실행 금지. REPRO가 명시한 별도 최소 CPU 재현 범위(조건 충족 후 최대1회/900초)는 이번 감사에서 실행하지 않았다. 성공/비재현도 “fixed”가 아니다. 이 결함은 기존 검토에서 bounded-diagnostic clearance를 막지 않는다고 명시했으므로, 새로운 모든 GPU 작업 차단정책으로 확대하지 않는다; 실제 bounded runtime에서 재발하면 중단/조사한다.

## 실질적 다음 순서 (실행 권한을 추가하지 않음)

1. **로컬 독립 작업:** eval9603↔decoder pin/manifest/SIF-package 소비계약과 regression 계획; ONT fullwrapper/실제도구 fixture/postflight·junction/GFF producer 결속; 최신 spec/source/역할 계약표; native crash 증거 gap 목록.
2. **과학결정:** HSP acceptance unit, phase semantics/conflict 범위, HiFi 확보/ONT-only, duplicate/coverage와 withholding 평가조건을 기존 문서의 결정권자에게 명확히 분리. Withholding 불능은 spec대로 unassessable이며 임의 STOP 불가.
3. **controller 선행조건 후 원격:** 기존3163986 실제 결과→후속5gate·seed별readiness→예산 수용된3seed학습→최종350decode. ONT full은 별도 externOOM/fixture/예산 gate 후. HSP/phase는 계약·실fixture·A11/input closure 후. 서로 독립인 준비작업을 하나의 queue대기로 묶지 않는다.
4. **완료로 유지:** partial 복구/최종 source 통합, d99 교체, 실제 probe·decoder 구현, C1 기술감사, 실제50map/array publication, 공유reference/panel identity 대조. 전체blastx/DB/panel 재계산으로 되돌아가지 않는다.

**요약:** 미체크29개는 미구현29건이 아니다. stale 완료/대체 지시, 지속 운영 의무, 중복된 GPU critical path, 지금 가능한 로컬 작업, 명시 과학/저자결정을 분리해야 한다. 실제MVP 학습·350생물학예측·최종평가/판정은 아직 완료되지 않았다. 전체 프로젝트/whole-suite green 또는 externOOM 해결 주장도 없다.
