# 사용자 결정 및 실행 범위 — 2026-09-17

> **후속 시드 결정이 우선:** [seed123 고정](SEED123_ONLY_DECISION_20260917.md). 아래B5‘MVP진단후456→789’운영안은보류/대체됨. 현재는123으로완성·품질확인하고추가시드는그이후. HSP/phase/A11승인과검증결과는유효.

## 이번 사용자 원문

> 1. HSP·phase·A11 — : DIAMOND가 출력한 HSP 전체 정렬 길이 ≥30을 사용#######  Phase: 제가 구현·검증하고, 애매한 선택만 #########가닥·인트론·frameshift를 반영해 위치 계산, 모순되는 증거는 억지로 하나를 고르지 않음
> - 충돌한 증거만 제거하고, 독립적으로 일치하는 증거는 보존 ######A11: 새 정책 선택보다기존 규칙 구현 ####  중복 계산과 데이터 누출을 막는 작업입 진현재 seed123은 그대로 두고, 급한 MVP 진단 자원을 우선 확보한 뒤 456→789를 순차 실행하행

## 해석/정본

앞서 제시한 권고에 대한 승인으로 기록한다. 문장 타이핑 오류를 별도 정책으로 해석하지 않는다.
1. **HSP:** DIAMOND engine-reported 전체 HSP alignment length≥30. 실제2.1.24.178에서column9=M+I+D,frameshift기호제외. evalue/input/trace 유효성 등 기존 조건 유지. query잔기30/paired잔기30/각constant-frame segment30으로바꾸지않음.
2. **Phase:** 가닥·인트론·frameshift를정확반영. 모순증거를임의승자로대체하지않고충돌증거를제외하며독립일치증거보존. 이것은최종cell전체veto가아닌증거단위처리방향승인. ‘충돌증거’의실제HSP/member/molecule/routeA-B 단위를기존문구에맞춰명시/시험/독립검토한다. 승인문에없는MAPQ새값·supplementary수용등새정책을몰래확정하지않음.
3. **A11:** 기존clone/library/accession-version/mate/UniVec part연결및분자단위누출배제구현. 기존규칙을재승인요청하지않음. metadata부재를중복없음으로간주하지않고MVP근사clone_key를전체A11로승격하지않음.
4. **B5:** seed123기존job/checkout보존. 급한MVP진단자원우선확보후456→789순차실행. 각seed자체유효state만resume,없으면동결원초기화에서시작. seed123학습checkpoint복사금지. 실제queue/잔여예산/소유권/현행chain script/무중복검증후controller단일제출. 우선순위방향승인이지무제한GPU예산·즉시두seed제출승인아님.

## 별도 범위/금지 유지

- A46전체동결/fullC0·fullB6생산실행/mainmerge/push/외부발신승인으로확대하지않음.
- 기존rawblastx/DB/panel재계산없음. 최종GPUb660/eval9603/seed123불변.
- 실제MVP학습은기존S0/P1및후속6gate/readiness/예산검증후. Phase완료를MVP선행조건으로추가하지않음.
- HiFi/ONT은이미승인된평가규칙에따른다. 별도산물정체성대조는기술잔여이지새포괄정책승인대기가아님.
- Advisor는최근client/model-version오류로응답불가;자문/승인으로기록하지않음.

## 실행 체크리스트

- [x] 이번저자결정기록,옛포괄승인대기와분리.
- [x] 최신HSP/phase코드/API·실제fixture·durablebase확인→격리수용단위/충돌처리시험우선구현.
- [x] Controllerdiff/실fixture144/CPU278시험→HSP·phase·A11수정재검토·모듈연결독립검토수용.
- [x] 기존A11규칙/재사용코드/입력완결성대조→명시normalized입력의연결/누출진단primitive구현·검증. 실제9종metadata/audit완결성은아직미확보.
- [ ] 실sourceID/FASTA/member/placement/assembly/arm을결속하는consumer,routeA실traceadapter와완전HSP·A11입력확보후실제투영/집계/raster검증. 현재primitive를생산phase/fullC0완료로부르지않음.
- [ ] B5실상태/예산/자체state/현행script확인→MVP우선조건을보존한순차실행안검토→조건충족시controller실행.
- [x] 실제결과·커밋·시험·잡ID및차단사유를이문서와TODO/resume에연결.

## 구현 진행 — 근거 확인 후

- 격리worktree `/tmp/est-approved-contract-20260917`, branch `est-approved-contract-20260917`. NativeAgentworktree는특정base지정이없어controller gitworktree생성. geometry5c46c75위에기검토a98f5c0의HSPdiagnostic만cherry-pick한base **c9a3514**. main/최종GPU/eval/B5소스불변.
- Graftbuild/query후controlleractualfunctions확인. 기존focused **70pytest PASS0.25s**. unittest discovery는HSP13통과/phase0수집이어서이를총시험으로쓰지않고pytest70으로대체.
- **HSP담당74d142a9-ad91-403:** 새est_hsp_contract.py/tests. 실제DNA/protein으로raw13→rich17검증후reportedMID길이≥30판정,원querycodonevent/identity산출. 원144실fixture기반정확replay검증;옛12notelicited상태보존. wholefile/production배포아님.
- **Phase담당7a0c103c-052b-48d:** 새est_phase_conflicts.py/tests. 이미label된완전HSP관측에대한충돌원자단위·pairlocalA/B·분자dedup/미해결처리. 기존geometry/productiongate불변. genome전수materialization/실routeAadapter완료주장없음.
- **A11담당35ff64cc-da98-41f:** 새est_molecule_closure.py/tests. 명시normalizedmember/library/version/clone/mate/split/audit입력의결정적연결과누출전파,미확인identity입력전체차단. 실제9종metadata완결성은별도. MVP근사rules재사용금지.
- 세담당은서로다른새files만작성,동일격리worktree에서독자commit금지;controller시험·독립검토후통합commit. TDDred/green기록필수.
- [A11 조사](approved_execution_20260917/A11_IMPLEMENTATION_BOUNDARY.md): 기존규칙/재사용mechanics와실제metadata누락분리.
- [B5 준비](approved_execution_20260917/B5_SEQUENTIAL_READINESS.md):10:34PDT balance1154GPUh,456/789run없음,seed1233131538완료→3149356Priority/나머지Dependency,P1Priority. seed123전체학습완료아님/bestepoch06도MVP고정epoch05를바꾸지않음. **seed123완료를456의새필수선행조건으로추가한agent제안은controller기각**;사용자승인은기존123보존/MVP우선/456→789이다. 현재P1대기때문미제출. 미래11h직접sbatch안은별도검토전실행금지.

## 최종 수용 — 11:01 PDT

- **코드8b646efbcebf275e9b5229f8a6a61bab9891ccdb**, 검토/승인스냅샷보존 **fca911e58fde44124b9359f9adc03a9d20059599**. Branch/worktree위경로보존,main미병합/미푸시. Graftrefresh완료. 격리worktree의`.gitignore`/`.ignore`는graft자동housekeeping만남고코드미커밋없음.
- HSP와phase개별검토APPROVE. A11의‘동일accession다른library로version을분리해oldplacement누출놓침’Important재현→preunioncanonical-library일관성guard+8회귀추가→재검토APPROVE. [HSP](approved_execution_20260917/HSP_REVIEW.md) · [Phase](approved_execution_20260917/PHASE_REVIEW.md) · [A11fix](approved_execution_20260917/A11_FIX1_REVIEW.md) · [연결검토](approved_execution_20260917/INTEGRATION_REVIEW.md).
- Controller **278focusedtests PASS1.23s**. Python3.8기존59타입annotationcollection오류는원본로그보존후기존Python3.12환경으로실행;소스우회/전체suite성공주장없음. HSP실144raw13→rich17정확재현,126수용/18길이미달,옛12NOT_ELICITED유지. 세새모듈과4case합성연결시험만수용,실9종생산데이터연결인증아님.
- 원본RED로그의pytesttrailingwhitespace는바이트그대로보존하여해당로그만diff-whitespace검사에서제외. 코드/문서검사는통과,로그내용정규화/실패은폐없음.
- [11:01live](approved_execution_20260917/final_live_gpu.txt): **3163986PENDINGPriority/사용0**, seed123 **3131538COMPLETED0:0→3149356PENDINGPriority**, 나머지3149357–63Dependency. seed123전체종료아님. 456/789미제출/MVP우선유지. 이번세션신규원격job/기존job변경0.
- 남은실행제약: productionconsumer/fullsourceidentity·routeA실trace/A11실metadata·auditclosure미확보,기존geometry의−1self-overlap미해결은명시유지(길이수용≠투영수용). B5실행은MVP우선해소와그시점queue·예산/실행안검토후.
