# 전체 TODO 정합성 확인 — 2026-09-17

> **후속 정정:** [HiFi/ONT 권위 재확인](HIFI_ONT_POLICY_RECHECK_20260917.md), [extern 운영게이트](EXTERN_OOM_GATE_RECHECK_20260917.md). 아래의 포괄적‘HiFi확보/ONT-only정책결정필요’는대체: 평가규칙/MVP실행은이미승인됐고PacBio6run수집도완료다. 실제잔여는A-CCS23의산물정체성과별도ERR13994458산출물위치확인·승인계약구현이다. 대체소스/최종STOP완화같은실제변경만별도결정. extern전체clean조건은독립검토의보수적운영게이트이며명시저자과학정책이아니다. 원인미해결은유지하되전체로컬평가구현을막는근거로확대하지않는다.

## 확인 범위

현재 TODO 미체크29행과 과거 archive 미체크/진행85행을 전수 대조했다. 이는114개의 독립 미구현 작업이라는 뜻이 아니다. Controller가 두 행수를 직접 재집계했다.
- [현재29행 대조·구체적NEXT](TODO_ACTIVE_AUDIT_20260917.md)
- [과거85행 대조·장기계획](TODO_ARCHIVE_AUDIT_20260917.md)
- 현재 감사 당시 줄번호 정본: [.atl/archive/todo_before_full_audit_20260917.md](../.atl/archive/todo_before_full_audit_20260917.md). 감사 후 현재TODO에서 stale지시를 정리했다.
- live GitHub issue closure/외부wiki·실제원고 게시 여부는 확인하지 않았다. 해당 장기항목은 미완료 확정이 아니라 권위/완료근거 재확인으로 분류한다.

## 1. 지금 로컬에서 진행 가능한 실제 잔여

1. **eval9603 → decoder 소비계약/패키지 repin:** 새panel manifest·producer/source closure·reference identity를 최종decoder에 결속, 정상/오염 계약시험 및 독립검토. decoder를 처음부터 다시 구현하는 일이 아니다.
2. **ONT full wrapper/postflight와 실제도구 fixture:** 실행정체성·command capture·원자적publication·SAM/BAM/junction 검증 준비/시험/검토. 원격full정렬은 별도 운영/예산게이트 이후.
3. **독립 junction/GFF → counts/scorer 연결:** 기존통계·bootstrap을 재구현하지 않고 per-run/pooled증거·분모·finalizer를 연결하는producer와 계약시험.
4. **최신spec↔source↔역할 계약 차이표:** 옛draft/best/sketch와 후속승인(공통epoch05,250updates,pureCEprimary)을 구분, withholding/unassessable조건 독립검토.
5. **HSP/phase 정책중립 fixture 준비:** 실제+/−split·partial·stop/Target의DNA/protein replay에 필요한 정확입력identity와 누락A11 closure 정리. acceptance/phase정책을 임의선택하지 않는다.
6. **운영 진단:** externOOM에 필요한site증거 gap/지원초안 보강, nativeSIGSEGV 원argv·환경·predecessor/core 확보 여부 정리. blind전체시험/원인추정확정/자동외부발신 금지.
7. **통합·보고 준비:** 선택된branch와main의격리diff/검토이력,원고·wiki·QC정본/반영상태 대조. merge/push/게시 자체는 별도결정.

**정정:** 모든 작업이GPU대기인 것은 아니다. 위 로컬작업은 남아 있고, 일부 원격실행/최종판정만선행게이트에차단된다. 이번요청은전체확인이므로본감사에서구현/새잡제출을하지않았다.

## 2. 실제 GPU 임계경로

10:06PDT [live](power_recovery_20260917/todo_audit_live_gpu.txt): **3163986 PENDING/Priority/사용0**, B5 **3131538 RUNNING10h28m/gh024**, 후속Dependency. 기존잡변경없음.

기존S0/P1결과수용 → checkpoint/SIF/runtime실identity확인 → step0/generation/bf16/4rank/resource나머지5gate → seed별binding/readiness·누적진단예산/별도학습예산 → seeds1/2/3각250updates → finalizer/checkpoint수용 →300paired+50baseline=350디코딩 →reference F1/독립splice/pairedbootstrap·최종판정.

P1미실행인데PASS복사/학습/후속GPU자동제출금지. ESTphase완료/A46동결을탐색MVP의새선행조건으로추가하지않는다.

## 3. 운영·과학/저자 결정으로 분리할 잔여

- **externOOM:** successfulpayload와별도미해결. cleanpilot/fullONT gate우회불가,sitevictimPID/cgroup/kernel/slurmstepd/epilog증거미확보.
- **ONT full:** 실제도구fixture/배포/throughput·scratch·예산수용후단일run;3runarray자동실행없음. HiFi정체성/확보또는ONT-only정책결정필요.
- **HSP:** 실제column9해석확인과생물학적≥30aa수용단위승인은다름. wholeHSP/query/paired/segment단위명시후accepted-event producer.
- **Phase:** label/order,nonprimary,ts모순,insertion,자기/분자간conflict,A/Bmask범위결정;실routeAadapter검증/전체A11closure후집계·raster·별도실행QC.
- **최종평가:** duplicate/coverage/HiFi/withholding가용성조건. 이번duplicate0관측을일반정책승인으로확대하지않음.
- **B5확인시드456/789**, **A46동결/fullC0·fullB6**, **mainmerge/push/외부발신**은각기별도결정.

## 4. 과거기록에만숨기지말아야할별도연구·보고

- A37≥100primary/≥121sensitivity두-arm지표·callability/QC/결과표;검증종121실산출물존재재확인.
- #61testscoring분모정의와A40후counts/Methods정합.
- AtRTD3독립성·오염/cleantruth취급,검증PacBio accession·산물·역할,A-HiFi와A-CCS23구분.
- B1의두referenceFASTA동등성/authority문제. MVP동일reference결속만으로자동해소되지않음.
- 원고A34partiallypretrained/top4random설명,MAPQ이상치QC표,wiki36/37·spec·실험기록반영확인.
- B5최종종료·3seed보충보고. **옛1GPUtrainloss보정을현재4GPUrun에적용금지**.
- fullC0/B6 WP0–WP7: 전체membership/secondary배제감사·누출mask·multisource/phase·pilot300·controls·final1729/maize·nested-dose/보고. MVP와범위분리,full승인전자동집행금지.
- 선택개선STOP_REQUESTED/chainAPPEND/pending_grads·mpi checksum/env정리,merge gates#69–71,장기issues#7–43은최신코드/issue권위재확인. 일괄미구현/자동재실행으로분류하지않는다.

## 5. 완료/대체 — 반복하지않기

partialcode복구·b660통합/배포검토,옛d99P1교체,실GPUprobe/decoder코드,fullMVPDB,9종blastx+C1technical감사,실제50참조ID수정/새panel/수치검증,ONTorientationpayload·공유reference/panelidentity·ledger는완료범위다. 실제GPU통과·MVP학습·전체ONT/phase·최종과학평가완료라는뜻은아니다.
