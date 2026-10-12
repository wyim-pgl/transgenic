# 과거 TODO 전수 대조 — 2026-09-17

## 범위·판정 원칙

요청은 **전체 남은 일 확인이며 실행 지시가 아니다**. `.atl/archive/todo_before_20260916.md` 전부를 읽었다(Python `splitlines()` 기준668행; read 도구 표시는 말미 개행 포함669행). `[ ]`/`[>]`가 있는 실제 작업 행은 **85개**(표기 설명 345·346·357행 제외). 중복 체크박스와 체크박스 없는 결정표·후속 문장도 아래에 통합했다. 파일 쓰기는 이 보고서뿐이며 잡/원격 파일/외부 API/GitHub/기존 TODO를 변경하지 않았다. GPU 현재 상태·구현 코드 감사는 controller 담당이므로 여기서는 최신 기록의 범위만 인용한다.

- **완료/대체**: 후대 기록이 직접 해소한 옛 행동. 재실행 불필요.
- **잔여**: 최신 기록에서도 별도 의무·차단으로 유지. 구현되지 않았다는 전수 코드 판정은 아님.
- **권위/증거 재확인**: 결론이나 최신 산출물·issue closure를 확인할 근거가 부족함. 열린 결함으로 단정하거나 자동 실행하지 않음.
- `A:Lx` = `.atl/archive/todo_before_20260916.md:x`; `T:Lx` = `.atl/todo.md:x`; `P:Lx` = `revision/protocols/PROTOCOL_B1_frozen_v1.md:x`.
- 최신 운영 정본은 `.atl/resume.md` 및 `notes/POWER_RECOVERY_20260917.md` 01:17 절. 과거의 “아직 미구현/승인 대기/현재 RUNNING”은 그대로 현재화하지 않았다.
- **live GitHub issues는 이 감사에서 조회하지 않았다.** 로컬 문서의 종료 기록과 실제 issue 현재 상태는 구분한다. 외부 wiki/실제 원고 정본의 게시·병합 여부도 미검증이다.

## 1. 별도로 남아 있는 연구·보고 작업

| ID | 묶음 / 남은 일 | 정확한 과거 근거 | 현재 근거와 분류 |
|---|---|---|---|
| R1 | **A46 최종 검토·명시 승인·동결·커밋**. 프로토콜/워크플랜/MVP 인용문·해시·실제 권한 기록 정합성 확인 | A:L111,115,149,158; L215–222 | **잔여**. P:L953은 여전히 DRAFT; T:L107–108. 패널 수·하드플래그·문안 수정을 반복하는 것과 승인/동결은 다르다. `notes/b6_execution_approval_20260914.md:3–16` 및 `notes/b6_parallel_execution_authorization_20260916.md:3–8`은 MVP/blastx 범위만 승인. protocol의 옛 epoch4/초안 hash 및 “MVP 승인 대기” 문구는 최신 epoch05 선택·09-14/16 승인과 조정해야 하지만 이를 A46 승인으로 해석하지 않는다. |
| R2 | **full C0/B6 WP0–WP7**. 아래 세부 목록의 승인·설계·생성·검증·보고 | A:L153,208,215–222 | **잔여/승인 게이트**. T:L107, P:L973–986; `B6_EVIDENCE_MODE_WORKPLAN_v1.md:59–68`. 전체 BAM으로 만든 **full C0-lite**와 전체 C0는 다르다. raw blastx 및 MVP CPU 경로 완료로 이 묶음을 닫을 수 없다. |
| R3 | **B5 confirmatory seeds 456/789 재개 시점 결정 → 완료 → n=3 보충 보고** | A:L161,194–201,227,264,293,318 | **잔여/저자 결정**. T:L109; P:L273의 투고 전 두 seed 완료 의무 유지. 09-11 제출/09-12 취소는 완료가 아니라 연기. `notes/b6_parallel_execution_authorization_20260916.md:7`은 재개를 승인하지 않는다. 옛 자동 제출/2GPU/1GPU 재시작 명령 재사용 금지. |
| R4 | **A37 ≥100 primary / ≥121 sensitivity 두-arm 보고**: EST 유래 수치마다 두 arm; 길이별 callability, junction support, chain intron 수, gene length 분포; #24 scorer 및 QC/결과표에 반영 | A:L509–510; L209,211,218 | **잔여**. P:L749–761 및 L960,985. 9종 두-arm 스크리닝·정렬 완료(A:L209)는 채점·보고 완료가 아니다. 검증종 Zmays/Slycopersicum ≥121은 A:L211에서 미생성으로 기록: 이후 생성/정렬 근거 **재확인**, 없을 때만 승인된 채점 범위에서 준비. 121은 입력 track 아님. A37.1 오염 추가 조사는 primary 필수가 아닌 조건부 연구이며 자동 추가하지 않는다. |
| R5 | **#61 논문 “test performance” scoring universe 확정** 및 Methods/표/필요시 versioned scorer 정합성 | A:L139,568; L561–572 결정표 | **잔여/저자 결정**. `notes/author_decisions_20260905.md:119–129`: accounting은 완료, 정의 결정은 별도. `notes/issues55_61_54_20260905.md:23–31`에는 nominal test 15,005/91,787와 strict431/3430 등이 있으나 이는 **구 corpus** 수치다. A40 후 counts로 갱신 여부 확인; 16.3%를 현행 고정 분모로 복사하거나 full-GFF를 labelled subset으로 몰래 축소하지 않는다. T:L110도 별도 유지. |
| R6 | **AtRTD3를 깨끗한 정답으로 취급할지/독립성·오염 한계·서술 결정** | A:L572,374 | **권위 재확인/잔여 연구 결정**. P:L105는 AtRTD3 자체 Iso-Seq 제외, L192는 AtRTD3 기반18.5% 비교를 descriptive/catalogue로 제한. 이는 AtRTD3의 upf/xrn/핵RNA 포함을 “clean truth”로 정당화하는 결정이 아니다. `.atl/quarantine.md:219–225`의 우려 및 T:L110 유지. 후보 ONT가 AtRTD3 입력이 아니었다는 A:L374의 완료와 혼동 금지. |
| R7 | **검증 PacBio 산물/접근번호/역할 권위 재정리**: 옛 `Athaliana_zhang2023`, `Zmays_B73_ccs` 실제 데이터 종류·변환/제외 상태 | A:L569 | **권위/물증 재확인**. 옛 “Zhang2023 CCS 변환 미실행”을 곧바로 실행하면 안 됨: P:L609는 A-CCS23(PRJNA911826 WT SRR22719002–07)를 이미 CCS-level로 명시. **A-HiFi는 별개** PRJEB77203 ERR13994458 subreads.bam→ccs/lima/refine(P:L102). `notes/RESTART_HANDOFF_20260916.md:6`절 및 resume는 A-HiFi 부재를 기록. maize 명칭만 ccs인 객체와 P:L608 M-CCS26/M-KNX, M-HQ18/M-FLNC는 accession·파일·genotype·역할로 대조해야 함. 51 subreads training set의 제외 완료는 검증 PacBio 두 건의 처리 완료가 아님. |
| R8 | **B1 reference authority 문제**: Athaliana 513e5ef3 vs §1 987ca803의 실제 서열/alias 및 적용 조항 결정 | A:L566 | **권위 재확인**. P:L18/109는 §1 MD5를 요구. 최신 MVP는 `notes/wait_work_20260916/longread/RESULT_REVIEW.md:41–47`의 Git manifest 92d8020/513e5ef3 권위와 panel을 결속했고 `POWER_RECOVERY_20260917.md:41–47`에서 수용. 이것은 **MVP 같은 참조 사용**의 증명이지 두 전체 FASTA 동등성 또는 B1 §1 개정의 증명이 아니다. 재정렬부터 하지 말고 B1용 동등성/authority closure 근거 확인. |
| R9 | **원고 Methods A34 “partially pretrained” + 보충 audit 설명** 반영 | A:L582 | **잔여, 원고 반영 완료 증거 없음**. P:L626–665는 339 tensors 중80 exact/147 adapted/112 random, top4 layers random 및 report 요구. `.atl/intent.md:200`도 미반영. 프로토콜에 문장이 있다는 사실은 원고 수정 완료가 아니다. 초기화 자체를 바꾸는 작업이 아님. |
| R10 | **wiki update36/37·spec·연구 기록 동기화**: A40–45, 검증/실측, 11h chain, 결과/현재 범위, §61·A34, B6/MVP 분리 | A:L155,181,265,294,316,319; L193,197,207 | **권위/게시 근거 재확인 후 잔여 처리**. T:L110 유지. update35는 게시 완료 기록(62420bc, A:L319); 36 초안/37 요청은 최신 게시 증거 없음. 이 checkout에서 `spec.md` 정본 위치를 확인 못함(`docs/gsf_spec_v1.md`와 동일 문서로 간주하지 않음). 외부 wiki를 여기서 조사/게시하지 않았다. |
| R11 | **train-loss/replication 보고 교정의 적용 범위 확인** | A:L181,193,197,202 | **옛 계산지시는 대체, 현행 보고 점검은 잔여**. P:L929의 epoch-index 보정은 옛 1GPU run용. A45 P:L942–943가 그 run을 보존하되 보고 seed에서 제외하고 새4GPU run으로 대체. 새 run은 epoch 평균이므로 현행 loss에 epoch번호를 곱하지 않는다. 혼합-runtime 문구도 A45의 same-runtime으로 대체; 옛1.9h 경계 손실은 run-history 한계로만 남김. |
| R12 | **MAPQ<20 이상치 완결성 QC 표**: Ath seed_flcdna_PRJNA1155260 29–51%, Gmax wm82_graft_PRJNA648759 34–35% | A:L536–540 | **보고 잔여/반영 증거 미확인**. 당시 중앙값2.7%; 낮은 mapping contribution과 원인불확실성 표기. T:L110 일반 보고 의무만 있어 해당 QC row 완료로 판단할 수 없음. 재정렬·생물학 원인 단정 불필요. |

### R2를 한 줄로 숨기지 않기: full C0/B6 세부 잔여

정본: `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md:59–68`, P:L973–986. 전부 **A46/full-production 승인과 그 단계의 선행 게이트**가 필요하다. 아래는 계획상 의무 inventory이며 구현 부재를 새 코드감사로 확정한 목록이 아니다.

1. **WP0**: 최종 A46 승인/동결, 입력·채점 역할 열/해시, panel identities/counts/hash, 명령·도구 출처·임계·분모·N_pilot 기록. 패널2079/50/300/1729 선정은 이미 기록됨; 승인과 별개.
2. **WP1 full C0(#21)**: 9 C0 tables + `est_protein_hsp`, A11 clone/version/mate/UniVec part closure 및 A16 long-read 단위, ordered blocks/chains/junction correction/roles/provenance/QC; train/validation 전체 관련 종 범위. **secondary 감사 정렬**은 배제 전용(`-N50 -p0.5`, exhaustiveness≥99.9%)이며 기존 정렬 전체를 무턱대고 다시 돌리는 명령 아님. blastx raw/C1 기술감사는 완료 부분; HSP 수용·phase는 별도 잔여.
3. **WP2**: held-out/test/valid orthogroup/edge-crossing partial·component/decoy 단위의 관측 누출 배제 및 DNA-mask 수리; A40식 새 corpus 빌드·동결; retained overlap0 및 mask외 bytes 정합; B5 원본과 노출 차이 정량 보고.
4. **WP3–4**: full multisource+phase tracks, observation→tile membership, thinning/source·library withholding, RC/padding, runtime/resume/생성 계약, 129kb memory 및 compiled4rank gate. MVP8채널의 부분 재사용 가능성과 full certification은 구별.
5. **WP5**: 비MVP pilot300, N_pilot4,142, matched seeds/orders/체크포인트, G1/G2, power descriptive 보고.
6. **WP6**: full B6/C2, matched-short C3/C4/C6 및 C5 evidence-only 기준선. **C1'는 저자 결정으로 생략**(A:L218; P:L961), 다시 추가하지 않음. 예전≈935GPU-h 예산은 MVP/C6 누락이라 최신 예산 승인 증거 아님(워크플랜L68).
7. **WP7**: disjoint final1729/maize 패널, 두 모드 × B1 평가, independent chain support와 reference F1 분리, G1–G3, 고정-locus nested dose-response/5draws, paired intervals, A37·누출/완결성·train/test coverage표 및 headline/fallback 규칙 적용. MVP 결과를 본문 모델 승격 근거로 쓰지 않음.

## 2. 운영·개발 선택 사항: 열린 체크를 구현 결함으로 단정하지 않기

| ID | 항목 | 과거 근거 | 판정 / 다음에 필요한 증거 |
|---|---|---|---|
| Q1 | 파일 기반 `STOP_REQUESTED` | A:L162,228,258 | **선택/저자 결정 재확인**. SIGTERM 대체 의무가 아니라 선택 개선. 최신 B5 코드 감사는 controller에게 맡김; 실행 중 checkout 수정 금지. |
| Q2 | NCSA unspecified-launch-failure 보고 | A:L162,229,259,287,311 | **외부발신 승인/발신 여부 재확인**. 우회 성공과 원인 해결/지원 보고 완료는 다름. 자동 메일/issue 생성 금지. |
| Q3 | `.mpi`↔reference checksum 결속 | A:L162,230,260,288 | **현재 코드/issue 증거 재확인**. historical medium; 새로운 MVP reference ledger로 옛 protein driver fix를 입증하지 못함. |
| Q4 | pending_grads rank0-only | A:L162,231 | **선택 최적화**. 저장공간 의견이지 현재 mandatory gate로 새로 만들지 않음. |
| Q5 | submit_chain APPEND 모드 | A:L162,207 | **선택 개선/현행 구현 재확인**. 11h 추가 링크 수동 처리 기록은 feature 구현 완료가 아님. |
| Q6 | bench/train env 정리·cpu_tools CHANGE_ME | A:L317 | **혼합**. P:L907 및L997 A42가 cpu_tools Delta-only 및 launch 수정 기록을 남겼으므로 “DeltaAI에서 cpu_tools 실행”은 대체. 남은 generic env 정리/placeholder 제거 여부는 코드감사 없이 완료/미구현 단정 불가. |
| Q7 | #69·#70·#71 merge gates | A:L337 | **live issue/최신 source 재확인**. 다음 merge 전 gate이나 archive는 세부명을 안 적음. 5289077의 사후 테스트 강화(A:L133–137)를 세 issue 전부 closure로 추정하지 않음. |
| Q8 | CLAUDE.md git 편입 정책 | A:L570 | **저자 정책 재확인**. 현재 AGENTS 존재나 machine-local exclude만으로 완료/폐기 판단 불가. 문서 변경 승인도 이 감사에서 하지 않음. |
| Q9 | main 통합/푸시·기존 protocol/panel/docs 보존 | A:L39,111,115,149 | **별도 잔여**. T:L108 및 최신 resume: implementation branches 미병합, 사용자 변경 보존. 구현 테스트 통과가 merge/push 승인 아님. |
| Q10 | lineage 생물학적 완전성/merged-taxid 권위·수용 한계 | A:L97,78 | **기술 완료와 과학 승인 분리**. 구조/입력 receipt·raw provenance C1 technical PASS(T:L91)는 완료. 생물학적 lineage 완전성을 자동 승인하지 않음; 최신 HSP/phase policy의 수용 근거/명시 limitation으로 다뤄야 함. |

## 3. 현재 트랙에 이미 흡수된 일 (중복 TODO를 만들지 않기)

- **MVP 실제 readiness→학습→50locus decoding/independent long-read/paired bootstrap/STOP-first 판정**: A:L22,38,39,104,154,160 → T:L52–80. CPU data/model/dataset/trainer 등은 완료 범위(T:L100); 원문 “전체 미구현”은 대체. 실환경/최종 결과는 controller가 확인. 원고 승격/효능 주장 불가.
- **HSP≥30aa acceptance/frameshift·BTOP 해석/phase/route-A agreement/A11 closure**: A:L39,78,97,153 → T:L82–95. raw 수를 accepted evidence라 부르지 않음. 준비 geometry 완료는 전체 phase 수용 아님.
- **extern OOM 원인/운영 게이트**: A:L27,34,78,97 → T:L106 및 `POWER_RECOVERY_20260917.md` 마지막 절. successful payload/batch와 별개. blastx 재제출 근거로 삼지 않음.
- **B5 seed123 정상 종료/성능·epochbudget/남은 링크 처리 기록**: A:L152,197,207 → 최신 resume/controller 추적. 과거 특정 시각/잡 ID를 현재 실행 명령으로 쓰지 않음. 오래된 epochbudget 실행 결과가 별도 보존됐는지는 **기록 체크 미확인**이나 옛 잡 모니터를 되살릴 이유는 아님.

## 4. 확실히 완료·대체된 옛 항목

| 묶음 | 닫는 근거 (코드 재실행 없이 기록 대조) |
|---|---|
| full MVP DB 빌드·hash/count/input identity·기술후보 수용; 이전 sample/staging 성능점검 | T:L101–102; A:L52 및L16–18. 생산 fullB6/학습 승인은 별도. |
| blastx pilot/나머지8종 제출·완료 및 whole raw/provenance technical 감사 | T:L89–91, `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md`; A:L76,70보다 후대. HSP/phase/extern는 닫지 않음. |
| weightedCE/composite metric 선택 | A:L4 및 T:L99. pure weightedCE tile-equal off−on .005nats 승인; 반복 승인 요청 불필요. |
| MVP CPU evidence/model/dataset/init/runtime/finalizer/CLI | T:L100; A:L5–21. readiness와 분리. |
| DeltaAI 인증 복구 | A:L5 및 T:L56–57에 성공 관측. 향후 접속 확인은 실행 직전 freshness 체크이지 옛 인증 실패 해결 TODO 아님. |
| MVP 수정안 실행 승인·futility/독립 구조채점 필수화의 승인 대기 | `notes/b6_execution_approval_20260914.md:3–16`, `notes/b6_parallel_execution_authorization_20260916.md:3–8`. 단, A:L111의 당시 “수정 두 건의 별도 추가 리뷰”는 승인노트가 미실시라고 적으므로 그 역사적 reviewer receipt까지 있다고 만들지 않음. 후대 구현 검토/현행 계약과 대조하면 되며 A46 대기는 유지. |
| 1GPU old seed123 상태/첫 epoch/48h boundary/재시작 권고·자동 제출 watcher | A:L193–197, P:L939–948: A45 신규 labelled4GPU run으로 대체, 옛 partial 보존·미보고. 오래된 watcher·hand-sbatch 금지. |
| B5 이미지/4GPU scaling/벤치/통합 시험/체인 Codex review/A44 개정 | A:L165–167,182–187,241–251; 3124099/3124229/3124347/3124348 PASS 및 ba24b19 Codex 반영 기록. 당시 failed trial을 성공으로 바꾸는 것이 아니라 후속 교정으로 의무 해소. |
| 당시 ACCESS 자원명·계정·추가 allocation 결정 | A:L182,196 및 P:L937/997. 후속 fullB6 예산 재산정은 R2, 현재 가용량은 controller 범위. |
| #45 정정 코멘트·A19.2 9종 acceptance·#44 정리 | A:L248,278–284,306; 로컬 기록상 완료. 현재 GitHub closure는 재조회하지 않았음. |
| OrthoDB 실행 위치 pronghorn을 §12 기록 | P:L885–887 및L996. A:L332의 “아직 없다”는 낡음. |
| longread orientation 한 토큰 출력 | A:L555–558가 A:L531–535를 닫음. 기존129 PROVENANCE는 그대로 보존하고 DONE의 uf를 읽는다. |
| B5 A40 corpus/tokenizer/new labelled run 시작 의무 | A:L133–139,196–197 + P:L850–883/942–943의 후대 실행기록. seed456/789 완료 의무까지 닫히지 않음. |
| 오래된 비체크박스 승인 대기(대표모델·maize 입력/채점·121범위·C1') | A:L218–219, P:L958–963: 선택은 기록 완료. A46 전체 동결 승인은 별도. |
| ONT 후보6항목·다운로드 대기/manifest·ENA-md5 미구현 | A:L369–375,606–609,642–658: 후보 조사/수집·md5검증·드라이버 편입 완료. 당시 `source_checksum` “비교 불가” 메모(A:L415–418)를 지금 새 구현 요청으로 되살리지 않음. retroactive 전수 audit 여부는 아래 별도 확인. |
| #17/#66 미결 문구·#55/#56/#58·#57/#59·A18.8 fetch | A:L132–137,462–507,578–595,603–605,630–635: 종료/commit 근거 존재. 잔여로 중복 계산하지 않음. |

## 5. 85개 미체크/진행 행 전수 매핑

다음은 누락 점검용이다. 한 행의 완료 부분과 잔여 부분을 분리했으며, 표의 **원행 숫자들의 합집합은 정확히85개**다.

| A 원행 | 처리 |
|---|---|
| 22,38 | 현재 MVP readiness 트랙(§3); 실GPU 상태 판단은 controller |
| 27,33,34,53,70,78,88,97,103 | DB/rawblastx 완료분 §4; HSP/extern/lineage 잔여 §3/Q10 |
| 35,61 | metric 선택 완료 |
| 36 | CPU runtime 구현 완료, 실제환경 gate 별도 |
| 37,116 | 인증복구 완료; 현재 freshness만 별도 |
| 39 | 현재 scoring/HSP §3 + R1/Q9 |
| 104,154 | MVP CPU 구현 완료부분 + 현재 readiness/평가 §3 |
| 111,115,149,158 | MVP 승인 부분 완료; 전체 A46 R1 및 commit Q9 |
| 139 | A40 새 labelled run 시작 완료; #61 R5 및 confirmatory R3 유지 |
| 152 | 옛 link2 snapshot 대체; seed123 후속 기록 §3 |
| 153 | rawblastx 완료; fullC0 R2/HSP §3 잔여 |
| 155,265,294,316,319 | wiki/spec R10 (update35 완료) |
| 159,160 | MVP 규칙/독립채점 승인 대기 해소; 실제 평가 §3/R7 잔여 |
| 161,227,264,293,318 | R3 (옛 명령·2GPU 계획·자동제출 대체) |
| 162 | Q1–Q5 선택/승인 재확인 |
| 168,169,170,237,238,239,271,272,273,300,302,303,313 | 옛1GPU run/경계/재시작 권고 A45로 대체; 역사 로그에 실패 사실 보존 |
| 181 | wiki/spec R10; 손실 보정 범위 R11 |
| 228,258 | Q1 |
| 229,259,287,311 | Q2 |
| 230,260,288 | Q3 |
| 231 | Q4 |
| 240,274,275,276,312 | 당시 벤치/scaling/메모리 해석 완료 |
| 241 | 후대 ba24b19 Codex 반영/리허설 기록으로 대체 |
| 242,263,277,285,292 | 26.08 결정/A44/DDP SIGTERM 후속 시험 완료 |
| 257,286 | 당시 allocation 선택 해소; fullB6 최신 예산은 R2 |
| 291,333 | #45 코멘트/9종 acceptance 완료 기록 |
| 317 | Q6 (체크된 commit와 줄 안 미체크 env 작업 구분) |
| 332 | A41/§12 실행위치 기록 완료 |
| 337 | Q7 |
| 509 | R4 |
| 531 | orientation 출력 수정 완료(A:L555–558) |
| 536 | R12 |
| 582 | R9 |

## 6. 체크박스 밖 장기 계획: 최신 issue 권위로 재확인할 범위

A:L153의 fullC0 링크가 가리키는 `revision/protocols/IMPLEMENTATION_ORDER_B5_C0_C2_v1.md:6–7,36–46,78–88`의 issue map은 **2026-09-02 스냅샷**이다. 이를 현재 미완료 건수에 더하지 않지만, 전체 연구 TODO 조회라면 다음을 빠뜨리지 않고 live issue/최신 결과표와 대조해야 한다.

- #7–10/#25: Wang B73/Zhong 및 retroactive download integrity·EST integrity/pre-alignment audit. `notes/issue23_addition_dump_20260905.md:28`에 #9→#25→#21 의존성 이력. archive의 md5 fetch 개선 완료는 과거 모든 자료의 소급 감사 완료와 다르다.
- #21–28: fullC0/evidence tests, addition dump, B1 scorer, maize alignment·P4/P5 controls, Athaliana replication. 일부 완료(#23 등)가 있을 수 있어 **일괄 미구현 표시 금지**. MVP50 결과는 B1 전체 validation pass 완료가 아님.
- #29–31: B2/B7 harness·실행 및 비교 headline. 당시 계획의 B7 투고 필수 결정은 별도이며 live closure 미확인.
- #32–34: 구 segmentation **C2 gate/labels/trainer**는 A46의 “tracks zero C2 control”과 이름만 같다. schedule gate/후속논문 유예의 최신 저자 결정을 확인하고, 폐기된 B6 보조head나 C1'를 재활성화하지 않는다.
- #35–40: 결과 재생성·numerical audit·manuscript·response·submission metadata·release audit. 현재 실행 승인/완료 evidence 없이 전부 열린 issue라고 단정하지 않는다.
- #41–43: Protocol M oracle/budget gate, tomato reference/genotype·week2 gate, chain-objective 후속 연구. **조건부/범위밖일 수 있음**; 과거 8주 계획을 자동 실행하지 않는다.

## 결론

과거85행은85개의 현재 할 일이 아니다. 확실한 잔여는 **A46/fullC0·fullB6 승인/후속, B5 확인시드, A37 쌍대보고, #61 분모, 원고/wiki/spec/QC**, 현재 트랙의 **MVP 최종 평가·HSP/phase·extern 원인**이다. **AtRTD3/PacBio/B1 참조 권위, 선택적 운영 개선, merge issue 및 전체 장기 issue map**은 최신 권위/완료 증거를 다시 확인해야 한다. 닫힌 DB/rawblastx/CPU 구현·이미 승인된 MVP 결정을 반복하지 않는다.

검증: archive 전체 완독(`splitlines()`668행), regex85개행 추출(설명행3개제외), 위 전수 매핑 프로그램 대조: expected85/mapped85, 누락0/추가0/중복0. Graft context 선행(`map`, `ask`) 후 문서만 조사; source 구현 감사 없음. Advisor는 client-version 오류로 응답을 받지 못했으며 독립검토/승인으로 취급하지 않았다.
