# `.extern OOM`과 ONT 전체 실행 조건 재점검 — 2026-09-17

> 🧊 **이 재점검이 다룬 게이트는 2026-09-21 저자 결정으로 해제됐다 — "extern OOM 은 그냥 무시한다."**
> `.extern OUT_OF_MEMORY 0:125`는 더 이상 차단조건이 아니고, 관리자 로그 확보와 support 발송도
> 선행조건이 아니다. 원인은 **미규명 상태로 종료**한다(해결됐다는 뜻이 아니다).
> 이 문서의 사실 서술(§1 step 구조, §2 관측 3층 분리, §3 게이트 출처 추적, §4 독립 진행 가능 범위)은
> 그대로 유효하며, 특히 **"extern만 해결하면 full 실행 준비가 끝난다는 뜻이 아니다"**는 결정 이후에도
> 유효하다. 정본: [EXTERN_OOM_WAIVED_20260921.md](auth_restored_20260921/EXTERN_OOM_WAIVED_20260921.md).

## 결론부터

**“ONT 정렬이 메모리 부족으로 죽었다”는 뜻이 아니다.** 확인된 것은 **실제 계산과 batch는 정상 종료했고, 같은 잡의 별도 Slurm `.extern` step에 OOM 상태가 남았다**는 것이다. 실제 계산의 메모리 고갈을 입증하는 victim PID/커널 기록은 확보되지 않았다. 그렇다고 `.extern` OOM이 무해하거나 단순 표시 오류라고 확정할 수도 없다.

**“모든 step이 깨끗해야 ONT 전체 실행으로 넘어간다”는 조건은 확인한 기록상 독립검토자가 정한 보수적인 운영 수용 조건이다.** 사용자·과학 프로토콜이 직접 명시한 보편적 필수조건으로 제시하면 부정확하다. 기존 승인 조건이므로 이 보고서가 해제하지는 않는다. 다만 이 조건을 **로컬 준비·fixture·이미 검증된 payload 관측까지 전부 차단하는 조건으로 확대할 근거도 없다.**

범위: 기존 로컬 문서/보관 회계만 재검토. SSH, 새 잡, 재실행, 원격 쓰기, 지원요청 발신, 소스 수정 없음. 이 보고서만 작성. 아래 상태는 인용된 관측 시점의 기록이며 새 live 조회가 아니다. HiFi 정책 판단은 별도 담당 범위다.

## 1. `.extern`은 무엇인가

- **allocation 행** (`6165555`): Slurm이 잡에 할당한 자원과 잡 전체 상태를 나타낸다.
- **`.batch` 행**: 제출한 배치 스크립트 실행 step이다. 스크립트 안의 minimap2·샘플러 등 실제 계산을 여기서는 **payload**라고 부른다. payload는 별도 이름의 Slurm step이라는 뜻이 아니다.
- **`.extern` 행**: 해당 allocation에 속하되 일반 batch/`srun` step 밖의 프로세스를 수용·추적하기 위한 Slurm의 특수 step이다. *외부 데이터*, ONT의 추가 분석 단계, 여러 잡을 합친 step이라는 뜻이 아니다. 어떤 프로세스가 실제 들어 있었는지는 이 기록만으로 알 수 없다. 모든 prolog/epilog 프로세스가 여기 있었다고도 단정하지 않는다.

사이트에서 관측한 `PrologFlags=Alloc,Contain`, cgroup 설정은 이 구분과 관련되지만, **OOM 원인을 설명하는 증거는 아니다**. `.extern`이 잡 생애와 함께 존재한 관측과 “각 allocation 소속” 설명은 `notes/EST_AUDIT_OPS_20260916_1530.md:89–101` 참조. `0:125`는 원문 그대로 보존하며 minimap2의 종료 코드나 “signal 125로 정렬이 죽음”으로 번역하지 않는다.

## 2. 실제 관측: 서로 다른 세 층을 분리

| 작업 | allocation / batch | extern | payload·산출물 판단 |
|---|---|---|---|
| ONT orientation `6165555` | 모두 COMPLETED `0:0`, 2분28초 | OUT_OF_MEMORY `0:125` | 최종 stdout=receipt, 16개 출력 해시 검증, 6개 명령 exit 0. 제한된 orientation 결과 일관성 수용 |
| C1 감사 `6165525` | 모두 COMPLETED `0:0`, 2시간49분14초 | OUT_OF_MEMORY `0:125` | 18 units/324 unit checks/117 tree checks 기술 PASS, clean accounting은 false |
| 복구 panel `6165628` | 모두 COMPLETED `0:0`, 18분53초 | OUT_OF_MEMORY `0:125` | 실제50 reference 및 150 artifact·배열 수치검증 수용, OOM은 별도 미해결 |

정확한 근거:
- ONT 원시 회계 `notes/wait_work_20260916/longread/result_review/remote_stat_accounting.txt:1–12`; 해시/종료 검토 `notes/wait_work_20260916/longread/RESULT_REVIEW.md:13–31`.
- C1 `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md:29–36,49–55,97–101`. accounting check PASS는 **예상된 extern 이상 상태를 그대로 확인했다**는 뜻이지 정상 회계라는 뜻이 아니다.
- panel 원시 회계 `notes/power_recovery_20260917/panel_result/terminal_accounting.txt:1–5`; 수용 범위 `notes/POWER_RECOVERY_20260917.md:29–40`.

ONT의 orientation `FAIL`도 계산 실패와 다르다. sense 2679/5008≈53.5%가 95%에 못 미쳐 **`-uf`를 쓰지 말라는 관측**이다. 전체 구조 증거가 없다는 음성 결론이 아니다 (`longread/RESULT_REVIEW.md:51–56`; 이하 `longread/`는 `notes/wait_work_20260916/longread/`).

### 메모리 고갈 증거가 있는가

- ONT minimap2가 직접 보고한 peak RSS는 **1.732 GB**, Slurm batch MaxRSS는 **51000K**, extern은 **1480K**다. 측정값이 서로 다른데 작은 수치만 골라 전체 실행 메모리 예산을 정하면 안 된다 (`longread/RESULT_REVIEW.md:23–31`).
- 실제 ONT 명령들은 0으로 끝났고 최종 검증 후 stdout까지 남았다. **이 payload가 OOM으로 중단되었다는 증거는 없다.** 그러나 이 사실만으로 잡 내 모든 프로세스가 메모리 문제 없이 끝났다고 증명하지는 못한다.
- 기존 EST 9잡에서도 allocation/batch와 18 GNU-time exit가 0이었고, 검색한 36개 작은 로그에 OOM/error/killed/signal 흔적이 없었다. 초소형 fixture와 다른 노드 fullDB에서도 같은 extern 상태가 반복됐다 (`notes/EST_AUDIT_OPS_20260916_1530.md:14–15,89–101`). 이는 **원인 조사 비교군**이지 “Slurm 버그 확정” 근거가 아니다.
- victim PID, 해당 cgroup의 실제 제한/이벤트, kernel/slurmstepd 로그가 없다. 낮은 샘플 RSS와 빈 stderr는 순간 메모리 압박을 배제하지 못한다 (`notes/wait_work_20260916/repro/SLURM_EXTERN_SUPPORT_DRAFT.md:7–21`).

## 3. 누가 이 게이트를 만들었나

확인한 문서에서 추적되는 조건의 흐름은 다음과 같다.

1. `longread/REPORT.md:74–84`: controller가 workload exit와 회계를 확인하도록 하고, full 실행은 별도 wrapper 검토·자원/디스크 추정 뒤 한 run부터 계획했다. 당시 16CPU/32GiB/4h는 **잠정 계획**이지 측정으로 입증한 예산이 아니다.
2. `longread/independent_review.md:11–17`: COMPLETE가 최종 검증보다 먼저 쓰여서, 중간 종료에도 marker만 남을 수 있다는 실제 결함을 지적했다. 그래서 **batch/extern의 timeout/OOM/nonzero 거부 + 최종 stdout + receipt 해시 + FAILED 부재 + compute 검증**을 bounded pilot 승인의 명시적 결합 조건으로 삼았다.
3. `longread/RESULT_REVIEW.md:29–31,66–71`: 실제 최종 stdout과 검증은 확인했지만 extern 조건이 불충족이므로 `PILOT_COMPLETE`를 보류하고, full 실행 전 운영 회계 해명을 우선 조건으로 유지했다.
4. 이 판단이 `notes/POWER_RECOVERY_20260917.md:40`의 “ONTfull 차단” 요약으로 전파됐다.

따라서 **명시적 조건은 존재하지만, 그 직접 출처는 엔지니어링 독립검토**다. 사용자 승인 기록은 독립검토·자원/중복 통제를 우회하지 말라고 하고, extern 미해결을 별도로 적었다. 하지만 “extern 원인 완전 규명 전에는 모든 CPU 준비/검증 금지”라고 하지 않았다 (`notes/b6_parallel_execution_authorization_20260916.md:8,12,16`).

과학 spec의 실제 요구는 전체 run의 후보 선택→선택 reads의 full-reference 정렬, run별 orientation 결정, BAM/index/provenance와 독립 junction 채점이다 (`revision/protocols/B6_MVP_PILOT_v1.md:109–145`). 해당 파일의 `extern/OOM/Slurm` 검색에서는 그 명칭의 조건이 발견되지 않았다. 이는 **이번에 확인한 승인 기록/spec 범위의 판단**이며, 모든 과거 사용자 대화에 그런 지시가 전혀 없었다는 전수조사 주장은 아니다.

C1은 같은 extern 상태에도 좁은 artifact 기술 수용이 가능했고, panel도 별도 검토·예산 아래 후속 실행과 결과 수용이 이뤄졌다. 그러므로 “extern OOM이면 모든 작업/결과가 무효”는 현재 프로젝트 운용과도 맞지 않는다. 다만 다른 작업 수용을 ONT full 실행에 대한 자동 면제로 전용해서는 안 된다.

## 4. 무엇을 독립적으로 진행할 수 있나

| 범위 | 판단 |
|---|---|
| 기존 로그/receipt 대조, full wrapper 설계·코드검토, 실행 파일 pin·stdout/exit/inventory 계약 정리 | extern 원인 규명과 독립적인 로컬 준비 가능 |
| 기존 로컬 자료 또는 작은 합성 입력의 SAM/BAM/junction fixture 준비·검증 | 로컬 도구/자원 범위에서 별도 bounded 작업으로 가능. 원격 데이터 전체 읽기·잡 제출과 구분 |
| 실제 사이트 도구로 새 bounded fixture 실행 | 원격 실행이라면 별도 검토된 좁은 실행안·예산·환경·중복 확인 필요. 이 보고서는 제출 승인하지 않음 |
| ONT 전체 run 정렬 | 기존 운영 수용 조건 처리 **외에도** full wrapper/postflight·실도구 fixture·throughput/scratch 예산이 남음. 지금 승인됐다고 할 수 없음 |
| 최종 구조 채점·STOP/GO | 실제 전체 증거/예측/채점 계약 필요. orientation sample 성공으로 대체 불가; HiFi 정책은 여기서 판단하지 않음 |

이 구분은 최신 `.atl/todo.md:3,15,53`의 로컬 잔여/실행 게이트 분리와도 부합한다. 특히 `longread/independent_review.md:25–27,43–49`와 `longread/RESULT_REVIEW.md:69–74`는 **full wrapper 및 실제 fixture가 아직 미검증**임을 별도 명시한다. extern만 해결하면 full 실행 준비가 끝난다는 뜻도 아니다.

현재 pilot의 marker 순서 결함은 최종 stdout·receipt 검증으로 이 결과에 한해 보완 확인됐다. 일반 full wrapper의 독립 완료 판정은 publication 순서/사후검증을 고쳐 별도 검토해야 한다. OOM 게이트 재검토와 이 코드 결함 해소는 서로 대체할 수 없다.

## 5. 가장 작은 다음 조사 — 새 계산 없이

1. **이미 저장된 자료만으로 비교표를 고정**한다: 1–2초 fixture6165460/6165462, ONT6165555, C1 6165525, panel6165628의 job/node/time, 요청 메모리, allocation/batch/extern, payload exit/최종 stdout·산출물 증거를 나란히 묶는다. 핵심 값과 출처는 위 표 및 지원초안에 이미 있다. 같은 이상을 재현하려고 완료 payload를 다시 돌릴 이유는 없다.
2. **운영 수용 조건과 결과 상태를 별도 판정 항목으로 유지**한다: `payload consistency verified`, `extern cause unresolved`, `clean accounting false`, `full-run readiness incomplete`. “모두 실패”나 “모두 정상”으로 합치지 않는다.
3. **실제 원인을 가르는 최소 추가 증거는 관리자 측 기존 로그**다. ONT6165555의 cpu-50/기록된 시작–종료 시각에 해당하는 kernel/slurmstepd의 victim PID·step/cgroup·event와 effective memory/swap limit, process membership을 요청할 준비를 한다. 필요시 작은 fixture 한 건을 비교한다. 초안은 `notes/wait_work_20260916/repro/SLURM_EXTERN_SUPPORT_DRAFT.md:15–21`에 이미 있으며 **발신하지 않았다**. 로그가 없다면 원인은 미확정으로 남긴다.
4. **controller/독립검토자가 게이트 범위를 명시적으로 재검토**한다. 기존 `.extern` 원인 해명과 full-run 승인 조건을 유지할지, 별도로 어떤 증거가 어떤 제한 실행의 위험 판단에 충분한지 문서화할 문제다. 이 보고서가 조건을 바꾸거나 안전성을 인증하지 않는다. 그 판단을 기다리는 동안 로컬 wrapper/fixture 준비를 멈출 필요는 없다.

사용자에게는 이렇게 설명하는 것이 정확하다: **“정렬 계산은 끝났습니다. Slurm의 별도 관리 step에 OOM 기록이 남아 원인을 모릅니다. 그래서 기존 검토가 전체 실행 승인을 보수적으로 보류했지만, 이것은 정렬의 메모리 부족이 확인됐다는 뜻도, 모든 준비 작업을 멈춰야 한다는 뜻도 아닙니다. 전체 실행용 도구 검증과 자원 계획은 별도로 더 필요합니다.”**

검토 한계: 원시 회계/문서 읽기와 출처 대조이며 모든 산출물 해시를 이번에 재계산하지 않았다. Advisor 검토 시도는 client/model-version 오류로 실패해 추가 자문 승인으로 취급하지 않는다. Graft 조회 후 필요한 문서만 확인했다.
