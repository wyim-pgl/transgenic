# Todo: 현재 실행 목록 (2026-09-25, 실제 B6 훈련 착수)

## 현재: 실제 데이터 준비 후 WP5 실제 훈련

정본: [운영학습 착수 준비](../notes/b6_production_start_20260925/PROGRESS.md). 사용자의 운영학습 시작 승인을 받았다. GPU검증 완료와 실데이터 구축 완료는 별개다.

- [x] 실제조건·자원·중복잡확인: balance1022GPUh, B5대기5잡 reserve220GPUh. B6 GPU잡없음. 첫훈련은4GPU/seed123 WP5파일럿4142업데이트, WP6는go판정후.
- [x] 실제원천자료 확인: audit184DONE(없는3런은승인된excluded), EST metadata11종, longread primary178DONE, CAGE88파일/88marker. 실제B6 C0/corpus/cache/bindings는미발견.
- [x] EST9종 CPUarray6172149 실행. 7종메타데이터완료, Gmax/Vvinifera 실패보존. 원본정렬·FASTA·B5무변경.
- [x] PacBio49 metadata CPUarray6172167 전부COMPLETED0:0, 49개METADATA_DONE/출력SHA256대조통과.
- [x] Gmax 표기비교 수정10f2aae(원본clone유지), 25시험PASS·독립검토APPROVE. 재처리6172217 완료, 1,374,461reads/출력해시 및rawclone보존확인. 기존7종미재실행.
- [ ] Vvinifera library누락 처리: screened450,180중175records(0.03887%), metadata접근누락0. GenBank 예시에도clone_lib없음. 임의library생성/무단제외금지. 출처복구 또는 저자처리결정필요.
- [x] **ONT protocol 승인(09-25 14:44PDT):** 사용자 ‘실험 방식 으로 채택’. 학습125런=cDNA83/direct_RNA42 채택, TSV해시불변. [승인 기록](../notes/b6_production_start_20260925/ONT_PROTOCOL_APPROVAL.md).
- [ ] **UMI 추가 확인:** absent125런은 이번 protocol승인에 포함되지 않음. 논문·키트·원자료 근거를 확인하고 미확인값을 확정하지 않는다.
- [x] Sitalica C0입력준비: 격리container DuckDB1.5.5, genome/GFF/split DBmanifest해시일치, 새symlink view, currentroles/toolprovenance, 독립검토APPROVE. 최초6172220 PyArrow누락실패보존.
- [x] Sitalica EST C0 재시도 CPU6172228 COMPLETED0:0(3m06s), 생산자QC status=PASS: accepted31,723/retained19,310, exhaustiveness0.99984, leakage_nonzero0, gate_*전부0, sha256 `02116ca6…1064` 로그·QC일치. 반복조회금지.
- [ ] **다음:** (해시·스키마 검증 15:53 완료, batch MaxRSS3.98G/32G) est_protein_hsp0행=설계상 정상(route-B는 protein C0, b6_hsp_project.py). 다음: 나머지 종/소스 C0 구축. UMI는 보고서상 no_UMI5/unresolved120 vs TSV absent125 불일치 → 저자결정 대기(TSV 무변경).
- [ ] 전체C0·A16후처리·병합/QC → B6corpus/freeze/cache → realbindings검증 → gatedconfig/4142에정지하는운영런처/예산확인 → 실제4GPU파일럿제출. 시험입력/시험체크포인트로대체금지.

## 이전 완료 기록 (GPU 검증 및 NVMe 정리)

상세: [RESTART_HANDOFF_20260924_PI](../notes/RESTART_HANDOFF_20260924_PI.md). 아래 09-23 체크리스트는 당시 기록이며 현재 실행 지시는 이 절이 우선한다.

- [x] 지정 세션 `1a00e258-5d2a-4bd8-a5a4-61423fb2e076`의 최신 지시 복구: A51, 테스트 GPU 1~2개, 벤치 다음 CP4.
- [x] 로컬 `2656dab` / 원격 스냅숏 `de15df0` 확인, 벤치·CP4 4파일 SHA256 일치, 원격 런처 shell 문법 확인.
- [x] 09-24 15:29 PDT Slurm 실측: `3205730` PENDING(Priority), GPU 2, ghx4-interactive, 예상 시작 17:31 PDT(변동 가능). B6 잡 중복 없음, 결과 JSON/로그 없음.
- [x] **09-24 15:59 PDT 종료 확인:** `3205730` FAILED 1:0, 81초. 양 rank `modeling_b6.py:208`에서 `None + None` TypeError. JSON 없음, 메모리 게이트 판정 불가.
- [x] 30분 확인 예약 `4GzTTiZ04g` 등록(첫 실행 16:29 PDT). Advisor 재시도는 클라이언트 버전 오류로 차단.
- [x] **캐시 API 수정·검증·독립검토:** Transformers 4.46 튜플 vs 4.57 공유 캐시 차이 재현. 커밋 `c9e242c`(B6 2파일), 두 버전 각66시험·95subtests, 실제 SIF CPU8시험·4subtests, MVP 핀 포함3시험 통과. Codex APPROVE(벤치 한정). [보고서](../notes/b6_cachefix_20260924/REPORT.md).
- [x] **수정본 벤치 1회 제출:** 새 경로 `repo_b6_cachefix_20260924`, 원격 `7018e33`, 216파일 해시 일치. 잡 **3207080**, 2GPU/1h/ghx4-interactive. 17:18 PDT PENDING, 예상 시작 17:25 PDT(변동 가능). 실패본3205730 보존.
- [x] **09-24 17:30 PDT 결과 확인:** `3207080` COMPLETED 0:0, 00:03:15. 로그·JSON PASS, 양 rank reserved 33.154 GB < 100 GB, measured_updates=8, compile_verified=true. [최신 관측](../notes/RESTART_HANDOFF_20260924_PI.md). Advisor 1회 요청은 클라이언트 버전 오류로 검토 불가, 자동 업데이트 없음. 동일 terminal 재검토 금지.
- [x] **본 세션 선행조건 검증:** 실제 원본·실행 소스 해시 대조, 매 step CompileGuard.observe와 예외 처리 확인. 현재 guard 기준 warmup 이후 graph-count 증가 거부 조건 통과. 독립 재컴파일 계수0 주장 아님. Advisor 도구 장애는 기록 유지, 새 승인 조건으로 만들지 않음.
- [x] **CP4 wp3-fixtures 제출:** 잡 **3207149**, 같은 수정본/216파일 재검증, GPU2/1h30m/ghx4-interactive. 17:32 PDT PENDING(Priority), 예상 시작 N/A. [진행 기록](../notes/b6_cp4_20260924/PROGRESS.md).
- [x] CP4 `3207149` FAILED(67초) 검증 및 원인 수정: 공용 검증기의 CUDA→NumPy 변환3곳 제거. 실제 CUDA7실패 재현 후 통과, 회귀78passed/91subtests, 실제 SIF CPU24passed/7CUDA-skipped, Codex APPROVE. 커밋 `a48579d`, [보고서](../notes/b6_cp4_20260924/cuda_numpy/REPORT.md).
- [x] 수정본 새 CP4 **3207331** 1회 제출: `repo_b6_devicefix_20260924`, 원격 `6446692`, 217파일 검증. 18:08 PDT PENDING(Priority), GPU2/1h30m.
- [x] CP4 `3207331` 종료 검증: FAILED(6분4초), 재개 byte identity 실패. epoch01 weights770개/RNG/drawlogs 같지만 pendinggrad각610개 및 step3weights622개 다름. 실제 SIF CPU tiny Inductor는 일치해 GPU/DDP 원인 미확정.
- [x] 제한 진단 **3207863** 제출: GPU2/30분, 같은 불변 소스로 eager 및 unsplit Inductor 순차 비교. readonly repo·별도HFcache·전후해시, 독립검토APPROVE. 19:47 PDT PENDING. [정본](../notes/b6_cp4_20260924/resume_mismatch/PROGRESS.md).
- [x] **09-24 20:30 PDT 진단 결과 확인:** 3207863 COMPLETED0:0(6분7초). eager/unsplit 각각 exit0, report PASS, resume_byte_identical=true, case후 소스217항목 모두OK. Advisor1회 요청은 클라이언트 버전 오류로 검토 불가, 자동 업데이트 없음. 정식CP4·synthetic 차단 유지. [관측 정본](../notes/RESTART_HANDOFF_20260924_PI.md). 동일 terminal 재검토 금지.
- [x] 진단 원본확인 후 unsplit compile 수정 `f1ee718` / 원격 `9ad97de`: optimize_ddp=False, DDP/Inductor 유지, compile_policy identity결속·구정책재개거부. 회귀68시험/95subtests, 구버전·핀37시험/78subtests, SIFCPU2시험, Codex APPROVE. [정본](../notes/b6_cp4_20260924/unsplit_fix/PROGRESS.md).
- [x] **새 메모리 벤치3208285** 1회 제출: GPU2/1h, repo_b6_unsplit_20260924, source217파일검증. 20:43PDT PENDING. 분할변경에 따른 메모리 재측정, 기존PASS 재사용안함.
- [x] 새 벤치3208285 원본 검증PASS: 각rank62.959GB≤100GB, optimize_ddp=false, measured8/compile_verified/오류없음.
- [x] 정식CP4 wp3-fixtures **3208442** 1회 제출. 같은f1ee718/원격9ad97de, source217파일재검증, GPU2/1h30m. 21:02PDT PENDING. [정본](../notes/b6_cp4_20260924/unsplit_fix/PROGRESS.md).
- [x] CP43208442는 NVMe quota(548.3/550GiB) 저장실패 후 collective불일치로 FAILED(5분12초). 원본보존. df여유와 allocation poolquota가 다름을 확인.
- [x] checkpoint collective오류전파·poolquota검사·HDDbind 수정 `b4dd15e`, 원격860bbfc. 회귀63/91subtests, 구버전·핀57/78subtests, SIFGloo20시험, 독립검토APPROVE. [정본](../notes/b6_cp4_20260924/quota_incident/PROGRESS.md).
- [x] **HDD CP43209815** 1회제출: GPU2/1h30m, source220파일검증, HDDquota/물리여유100GiB이상. 23:07PDT PENDING, 예상09-25 06:59PDT(변동가능).
- [x] **09-24 23:30 PDT 확인:** 3209815 FAILED1:0(1분29초), NCCL smoke6통과/1실패. preserve_step case의 TCPStore 초기화 port49717 EADDRINUSE. 정식CP4 미실행/report없음. Advisor1회 요청은 버전 오류로 검토 불가, 자동업데이트/재제출 없음. [관측 정본](../notes/RESTART_HANDOFF_20260924_PI.md).
- [x] 사전시험 포트 경쟁 제거: `3511a5a`/원격e2a91b7, test파일만변경. 고유FileStore, 명시적rank/world/backend; MASTER_PORT점유회귀 RED후 로컬·SIFCPU 각7PASS, 독립검토APPROVE. [정본](../notes/b6_cp4_20260924/port_collision/PROGRESS.md).
- [x] **새 HDD 잡3210279** 1회제출, GPU2/1h30m, source220파일·quota·활성잡0검증. 23:40PDT PENDING, 예상N/A.
- [x] **WP3 fixtures CP4 3210279 검증 완료:** 실제NCCL7PASS, 정식 report PASS, 재개step3해시일치·rank별generation·3link결속·compile_policy·메모리게이트 연결 확인. [09-25 정본](../notes/b6_cp4_20260925/PROGRESS.md).
- [x] **NVMe 정리 완료(09-25 12:55 PDT, 사용자 승인):** B6 종료산출물4트리, 파일414개/총528항목을 HDD archive에 SHA256 검증·fsync 후 NVMe 원본만 제거. 사용548.266→402.004GiB, **146.262GiB 확보**, soft여유97.996GiB. B5·source repo·메모리 및 최신HDD PASS결과 보존. [정리 정본](../notes/b6_nvme_cleanup_20260925/REPORT.md).
- [x] **synthetic CP4 3210585 제출:** 같은소스3511a5a/원격e2a91b7, HDD·source220파일·quota확인, GPU2/1h30m. 09-25 00:04PDT PENDING(Priority).
- [x] **synthetic CP4 3210585 검증 완료:** COMPLETED0:0(7분22초), NCCL7PASS·reportPASS·재개step5해시일치·양rank generation시퀀스일치·compile_policy확인.
- [x] **요청된 GPU검증 구간 종결(09-25 01:03PDT):** 메모리 및 두정식CP4 PASS, 실행후source220파일OK, 활성B6잡0. [최종 정본](../notes/b6_cp4_20260925/FINAL_VERIFICATION.md). 기존실패산출물보존, B5/운영학습/main병합/푸시없음. 당시 남은 NVMe정리도 09-25 12:55PDT 승인 후 완료(위 기록).
- 30분예약4GzTTiZ04g는유지. 새잡/새지시전까지 완료결과의반복해시·Advisor재요청·재제출없이 완료상태유지. 생산학습허가로해석하지않는다.

## 09-23 보존 기록

> 📌 상세 인계는 [RESTART_HANDOFF_20260923](../notes/RESTART_HANDOFF_20260923.md). **아래 "## 현재 (09-23)" 절이 정본**이다.
> 그 아래는 이전 기록이며, 그 안의 "다음 행동"과 "최신" 표시는 과거 기록이다.

## 현재 (09-23)

- [x] **WP1 수정 라운드 3 결과 확인(09-23 10:30)**: HIGH-1·MEDIUM-1/2·LOW 5·PacBio 2번째 토큰·`--est-meta` 전부 코드에 반영 확인. 전체 시험 509 통과/7 실패(기존 환경 6 + 시험 helper 버그 1 → 수정, `test_b6_c0_metadata` 12/12). 바이트 핀 통과.
- [x] **Codex 리뷰 실행(09-23 10:35)** → **REJECT, HIGH 5**: (1) 거부된 롱리드 멤버가 per-read 분자 ID를 받아 분자 단위 누출 폐쇄를 벗어남 (2) 단백질 쿼리 제외가 HSP 자격·route-A 0 목록에 전파 안 됨 (3) route B가 C0가 거부한 EST 배치를 받음(`accepted` 미검사) (4) 병합이 단백질 DB의 est_c0 sha·제외 provenance를 대조 안 함 (5) 매니페스트 `non_reference` vs 코드 `known_nonreference`로 학습 74행 차단. 결과: 세션 scratchpad `codex_wp1_review_result.md`.
- [x] **수정 라운드 4 완료(09-23 11:25)**: Codex HIGH 5건 전부 프로토콜로 확인(rule 3 L989 / 4(b) L990 / 6 L992 / 9 L997 / A16 L625 / A18.5 L270) 후 수정. (1) `rejected_member_unit`로 거부 멤버도 A16 단위 (2) `load_hsp` 단백질 쿼리 상태 JOIN + `b6_routea_agreement --protein-c0` 필수 (3) 분자 맵 `accepted` 열 + `verify_hsp_placements` (4) `provenance_gate`(est_c0 sha·제외 identity) (5) `non_reference`→`known_nonreference` 별칭. writer 보고 169 통과·바이트 핀 통과. 보고서 `/tmp/b6-wp1-20260922/tasknotes/WP1_FIX_ROUND4_REPORT.md`. **저자 결정 4건**(보고서 끝): route-B protein_id의 rule 4(b) 상속 / 거부된 ONT non-UMI 리드 per-read 단위 잔여 / roles 매니페스트를 병합 identity에서 제외 / 별칭 위치.
- [x] **라운드 4 Codex 재검토(09-23 11:45)** → **REJECT(HIGH 3, MEDIUM 3)**: 3(거부 EST 배치)·5(genotype 별칭) FIXED, 1·2·4 부분. 잔여/신규: (1) 가닥 미상 UMI 형제의 분자 키 불일치(None vs "?") (2) `b6_hsp_project` 충돌 해소가 제외 단백질 P와 함께 적격 Q까지 삭제 (3) `--protein-c0` 게이트가 실제 EST C0·GFF·HSP·사이드카 해시에 결속 안 됨 (4) 병합 build_flags 키 비교가 한쪽 키만 순회 (5) rule 4(b) exhaustiveness가 소스별 강제(종 단위여야 함) (6) 승인 정렬의 A16 완전성 코드 3열 NULL. 결과: scratchpad `codex_wp1_rereview_result.md`.
- [x] **수정 라운드 5 코드 완료(09-23 12:41)**: 6건 확인·수정(5번은 부분: 임계는 99.9 %). (1) UMI 단위 키를 승인 경로와 동일하게 None (2) 다중 HSP 규칙을 `b6_hsp_project`에서 `b6_build_c0.load_hsp`로 이동(자격 판정 후), 선해소 투영 거부, step-7·병합 재검사, never 분자 HSP는 leakage_excluded (3) `b6_routea_agreement` 재작성: 입력 4개를 단백질 C0 기록 해시에 결속, accept 표 내용 결속, route B는 단백질 C0에서 읽음 (4) identity 키 합집합 비교 (5) 소스별 exhaustiveness는 기록만, 병합에서 종 단위 ≥ 99.9 % 강제 (6) A16 완전성 코드: 생산자·입력 부재 → NULL 정책 선언·계수·게이트(저자 결정). 보고서 `/tmp/b6-wp1-20260922/tasknotes/WP1_FIX_ROUND5_REPORT.md`(시험 결과 칸은 내 재실행으로 채움). **계약 변경 4건**: route-B TSV 미해소(단백질 C0가 해소본) / agreement `--protein-c0` 필수·해시 결속 / 소스별 빌드는 exhaustiveness로 중단 안 함 / 모든 C0가 completeness 정책 선언.
- [x] **라운드 5 Codex 3차 검토(09-23 13:50)** → **REJECT(HIGH 1, MEDIUM 1)**: 1·3·4·5 FIXED, 2·6 부분. 신규 HIGH: 롱리드 `library = run`(b6_c0_metadata:281)이 A11 라이브러리 정의(독립 제조) 위반 → 같은 제조의 두 런에 걸친 ZMW/PCR 단위가 분자 2개로 갈라져 held-out secondary가 형제를 제외 못 함. MEDIUM: 라운드 5 이후 leakage_excluded(held-out 평가) HSP는 충돌 해소를 안 거쳐 모순 HSP가 그대로 기록됨. 결과: scratchpad `codex_wp1_review3_result.md`. 라운드 5 전체 시험 내 재실행: **176 통과**(175 B6 + 바이트 핀, 1,255 s, `wp1_round5_tests.log`).
- [x] **수정 라운드 6 코드 완료(09-23 15:25)**: `--library` 필수(ENA experiment `[SED]RX<n>` 또는 `<BioSample>:<library_name>`만 허용; writer가 세션 한도로 중단돼 두 형식 검사는 내가 추가), 모집단별(학습/제외) HSP 충돌 해소, 소스별 exhaustiveness 하한 0.999(빌드 중단 + 병합 재검사). 보고서 `/tmp/b6-wp1-20260922/tasknotes/WP1_FIX_ROUND6_REPORT.md`. 전체 시험(`wp1_round6_tests.log`)·Codex 4차 검토(`codex_wp1_review4_result.md`) 진행 중.
- [x] **Codex 4차 검토 → REJECT(HIGH 1·MEDIUM 1·LOW 1) → 라운드 7(내가 직접 수정)**: 빌더 수집 단계에서도 A49.1 라이브러리 키 강제(공용 `c0_schema.is_longread_library`), HSP의 library/est_accession을 분자 멤버와 결속(빌더+병합), 소스별 abort 시 exhaustiveness 카운터 보존. 보고서 `tasknotes/WP1_FIX_ROUND7_REPORT.md`. 전체 시험 진행 중(`wp1_round7_tests.log`) → 다음 Codex 5차.
- [x] **WP1 코드 리뷰 종결(09-23 18:00)**: 라운드 7 전체 시험 185 통과 → Codex 5차 REJECT(거부된 clone 없는 EST의 A11 서명 누락) → 라운드 8(`rejected_est_signature`) → 전체 시험 **186 통과**(바이트 핀 포함) → **Codex 6차 APPROVE(지적 0)**. 보고서 `tasknotes/WP1_FIX_ROUND{4..7}_REPORT.md`(7번에 라운드 8 포함). 코드는 worktree `/tmp/b6-wp1-20260922`에 **미커밋**.
- [x] **WP1 커밋(09-23 17:10, 저자 "시작해줘")**: worktree 브랜치 `b6-wp1-20260922`에 `e84cd6a`(route-B 병합) + `c26eac9`(WP1 파이프라인). Won <wyim@unr.edu>, 트레일러 없음, 푸시 안 함.
- [ ] **CAGE 다운로드 잡 6171128**(88파일·216 GB, `%8`, `cpu-s2-core-0`/`cpu-s2-pgl-0`): `$ROOT/evidence/cage_v1/<종>/<BP>/<런>/` + 파일별 ENA md5·크기 검증 DONE. 스크립트 `revision/scripts/evidence/b6_cage_fetch.sbatch`(미커밋). 매니페스트 수정: PRJNA438108 = RAMPAGE 10런만, PRJNA562289 = 야생형 3반복만(`CAGE_INPUTS.b6_v1.tsv` sha256 `8c3318df…`, A49.1 해시 갱신).
- [ ] **NCBI EST esummary 수집(진행 중, 로컬)**: `revision/scripts/evidence/b6_est_esummary.py`(미커밋; nuccore `txid<t>[Organism] AND gbdiv_est[PROP]`, esummary v2.0, 요청당 45 s 하드 제한). 출력 `/mnt/f/scratch/develop/Transgenic/est_esummary_v1/<종>/est_esummary.tsv`(+sha256, PROVENANCE). 3 레인 + Sitalica, 약 15 h 예상. 끝나면 pronghorn `evidence/est/<종>/`로 복사. EST `--bioproject` = `NCBI_EST:txid<taxid>`(권고안 채택).
- [x] **메타데이터 보강 완료(09-23 14:20, 저자 검토 대상)**: ONT 시트 19열(`library`=ENA experiment accession, 143/143, 다중 런 experiment 0) / `data/manifests/PACBIO_LIBRARY.b6_v1.draft.tsv` 72행(학습 49·검증 21·제외 2; 69 키 있음, Zenodo F1maize 3파일은 ENA 없음; 다중 런 experiment 3 = zhang2023 검증용). **저자 결정**: JGI 제출(Sbicolor wray HOTXU/HOTXW…, Ptrichocarpa jgi_F1hybrid GZHTA)은 SMRT cell마다 experiment를 따로 등록해 같은 library_name이 2–3 experiment에 걸침(15그룹, 학습 PacBio) → experiment 키 유지 vs (sample, library_name)으로 합침(권고: 합침이 rule 4(b) 폐쇄에 fail-closed). 목록은 `notes/ONT_PROTOCOL_SHEET_20260923.md` §8.
- [x] **저자 결정(라운드 5 신규) 반영 → A49 v1.41(09-23 14:05, sha256 `60b9a979…`)**: A16 완전성 코드는 병합 후 계산 패스(NULL 정책은 그때까지 유지; cap/CAGE 입력 종별 확인 필요) / exhaustiveness 종 단위 + 소스별 하한(값 99.9 % 가정, 확인 필요) / 충돌 해소에 train_tiles_excluded 포함 확정 / never 분자 HSP leakage_excluded(내 판단). 구현 독법 4건(단백질 쿼리 상속, 거부 멤버 A16 단위, 라이브러리 키 = ENA experiment accession[저자 확인 대기], 병합 identity)도 기록.
- [ ] **후속: A16 완전성 코드 계산 패스** (병합 DB 위: TSS/TES 클러스터·admitted chain·cap/CAGE). ✅ cap/CAGE 인벤토리(`notes/CAGE_INVENTORY_20260923.md`) ✅ **A49.1(09-23 15:10)**: 저자 결정 반영 — TeloPrime SRR15462255 사용 안 함 / CAGE 입력 종별 지정 완료 `data/manifests/CAGE_INPUTS.b6_v1.tsv`(sha256 `4cbdc854…`; 도달 가능 6종, 불가 5종 → 5I) / 유전형·품종 확인 생략(`genotype_as_stated`). **다음**: 지정 CAGE 세트 pronghorn 수집(ENA fastq; 9 BioProject) → 피크 호출 방법 확정(설계 §2) → 계산 패스 구현.
- [x] **라이브러리 키 재키잉 완료**: ONT 139 라이브러리/143행(병합 1그룹 = 제외된 fast5), PacBio 45/72(JGI 36런 → 15그룹), 샘플 교차 0. sha256 ONT `e21425e1…`, PacBio `8de4d583…`. §8.3 기록.
- [x] **ONT 프로토콜 시트 초안 완료(09-23 10:50, 저자 검토 대상)**: `data/manifests/ONT_PROTOCOL.b6_v1.draft.tsv`(143행) + `notes/ONT_PROTOCOL_SHEET_20260923.md`. 학습 125 = cDNA 83 / direct_RNA 42 / unresolved 0; 검증 10 전부 cDNA; 제외 8 = cDNA 4 / dRNA 3 / unresolved 1(SRR25203456, 리드 미수집). umi_status 전부 absent(ENA 20필드 무언급). 메타 vs 프라이머 불일치 0, medium 확신 52행. 저자 확인 2건: SRR36603452(buffer_control)는 SMARTer TSO 1,890/2,000으로 cDNA 판정 가능(제외 유지) / SRR34539092·93은 pychopper식 트리밍 리드로 cDNA(medium).
- [ ] **감사 정렬 6170748 완료 확인.** 187 DONE, REFUSED 0, 크기 확인. 18:00 PDT 기준 51 DONE / REFUSED 0 / 122 GB (52–186 대기).
- [ ] **저자 결정**
  - EST BioProject 값: `NCBI_EST:txid<taxid>`(권고) 또는 종별 대표 프로젝트
  - `DATASET_ROLES.b6_v1.tsv`를 별도 파일로 둔 해석
  - N 포함 EST(실측 후)
- [ ] **메타데이터 준비**
  - NCBI EST 요약 수집(esummary v2.0 `retmax=5000`, 약 1,134 요청, 1–1.5 h; 착수 지시 대기)
  - ONT 프로토콜 시트 작성(direct_RNA 45 / cDNA 84, UMI absent; 저자 검토)
  - 제외된 4런은 C0 role 게이트가 거른다.
- [ ] **C0 빌드**(종 × 소스, Slurm CPU) → 병합 → QC 게이트 → rule 4 제외 비율 보고. 보고 항목: `excluded_b_cause_secondary_mapq`, `excluded_a_cause_edge_partial`.
- [ ] **route-B 투영과 A/B 합의** → `est_protein_hsp`. N 포함 EST 손실(`drop_noncanonical_dna_outside_hsp_span_only`)을 저자에게 보고한다.
- [ ] 이후 WP2(누출 제외·코퍼스 재빌드) → WP3(트랙·체인, 129 kb 메모리 게이트 ≤ 100 GB) → WP4(트레이너, rule 6a 체인 메모리 블록) → WP5 파일럿.
- [x] 09-23: A48 개정, 감사 잡 제출, ONT 제외 4런(buffer_control 1, sdx fast5 3), 원격 정본 매니페스트 동기화(`a12f9f00…`).
- [x] 09-22: A46 승인, WP1 착수, 1차 검토·수정 라운드 2, b6 할당표, 체인 메모리 설계 v2, MVP 패널 정렬(A-CCS23 6 + A-ONT1 3), 학습 PacBio 49런 정렬.

---

# (이전) Todo — 현재 실행 목록 (2026-09-22 A46 승인준비)

**현재 임계경로 (09-22 A46 승인 후): WP1.** (이전) **현재 임계경로:** **issue #60 재개 처리**(격리 51런이 CCS로 판명) → 그 결과로 D1 해소 여부 결정 → K 확정 → A46 rule 6 본문에 리터럴 삽입
→ A46 최종승인 → WP1 착수. **D2는 종결**(rule 6 마지막 절 = 분리 제약, K 불변, 개정 불요).
**다음 행동(09-22 심야):** ⏳ **B6 체인 메모리 설계**(저자결정: 롱리드 엑손 체인을 B6 입력으로, 3번 방식) — 초안 v0 → **v1** `notes/a46_approval/B6_CHAIN_MEMORY_DESIGN_v1_20260922.md`(Kimi proceed-with-changes 10건 반영, 저자 "수정 후 진행"; Kimi challenge 모드는 E2BIG로 실패해 ask 모드로 우회), 독립검토 APPROVE_WITH_FIXES(CRITICAL 1·HIGH 5) 반영 **v2** `B6_CHAIN_MEMORY_DESIGN_v2_20260922.md` → **A46 DRAFT에 rule 6a + 관련 조항 적용**(rule 17·K=31 불변 확인). 기본값 적용: D-C1 MVP 제외, D-C2 EST 포함·롱리드 우선, D-C3 12층×512(8×64), D-C4 보류, D-C5 WP3 후, **D-C6 ✅ 저자 확정 "A": C7 = 파일럿 규모 팔 + (ii-c) 보조 결과**. ✅ 동결 체크리스트 갱신 + rule 6a 상한 64/512 동결 + 명령 명세 SPEC-2c. ✅ 2차 검토(독립 APPROVE_WITH_FIXES HIGH 6·MED 11·LOW 7 + Kimi approve-with-fixes) 전부 반영 — A-CCS23을 rule 1 Never·A14에, rule 6a 평가 구성·상수·해시 인코딩, rule 10 파일럿 패널 정렬, rule 11 여섯 arm, rule 15 개정 경유, rule 16 (ii-c) 통계, 오래된 문장 ✏️. rule 17 바이트 동일·§5.3 인용 sha256 `b5b918bf…`. ✅ Kimi 최종확인(좁은 ask): 수정 13항목 확인·기본값 3건 수용·수정 1건(A14 괄호에 M-HQ18 트랙 전용) 적용. ✅ **A46 v1.36 승인(저자 "승인", 2026-09-22 22:23 PDT)** — 상태 표기 변경, MVP 파일 해시 재측정(373줄 `067c1e2f…`), 승인 기록 `notes/a46_approval/A46_APPROVAL_RECORD_20260922.md`(프로토콜 sha256 `e137c01e…`). rule 12 동결 발효. **WP1 착수(09-22 저자 "바로진행해")** — ✅ 인벤토리 완료 ✅ worktree `/tmp/b6-wp1-20260922`(브랜치 b6-wp1-20260922 = 729b990 + route-B 병합, 미커밋) ✅ `DATASET_ROLES.b6_v1.tsv`(b6_input/score 할당, 별도 파일 해석은 저자 확인 대상) ✅ writer A(C0 스키마 c0-v1·빌더·rule 4(a)-(d)·병합, 시험 44) ✅ writer B(감사정렬 드라이버 187건·route-B 운영투영·A/B 합의 QC, 시험 35) — 통합 재실행: WP1 79 통과, 기존 492 통과/4 환경실패, MVP 74/3 torch. ✅ 코드 독립검토 REJECT → 수정 라운드 2(A2·B2) 완료, WP1 시험 117+2 통과 ✅ 저자결정 → **A48**(소스간 phase 불일치 0, rule 4 문자 독법 2건) ✅ **감사정렬 잡 6170748**(187건, %4) 제출 ⏳ 재검토 · 감사정렬 진행 · 실데이터 공백 조사(EST 7종 library/clone, ONT 129런 protocol/UMI).  **실데이터 전 필요:** per-read `c0_metadata.tsv` 생산기(없음), protein DATASET_ROLES 행(없음), A/B molecule_id 결합(member uid = QNAME?) 확인, N 포함 EST 손실 측정(저자결정), MVP 코드 매니페스트가 새 src 모듈을 포함하는 부작용. **다음: WP1 착수**(C0·코퍼스 재빌드·트랙·체인·route-B·누출 차단; 체인 구현은 트레이너 브랜치, MVP 무접촉). ⚠️ 승인 대상 파일 미커밋 → 수정 → 저자결정 D-C1~C5 → A46 rule 6 개정 → WP1. (이전) **다음 행동(09-22 밤):** ✅ A-CCS23 MVP 패널정렬 6170576 — 5/6 완료(고유 접합 1,715–1,966/런), SRR22719003 빈 엑손 → 저자 "같이 진행해": (b) 적용(`681c5a0`, 핀 `729b990`) ✅ **6170586** 4/4 완료: SRR22719003 1,973 접합(빈 엑손 1리드만 제외), A-ONT1 3런 3,420–5,946 접합(비배향 → `-uf` 없음). **MVP 구조 readout 입력 완비(A-ONT1 3 + A-CCS23 6)** — 남음: 외부 frozen-reference 게이트, 계약 v2 결합, 판정코드의 구조 가용성 입력 확인. (이전) **다음 행동(09-22 오후):** ✅ 파일럿 6170399(99.76 % 매핑) ✅ **PacBio 학습 정렬 49/49 완료·검증**(6170399+6170404; DONE·BAM/PAF md5·index·PROVENANCE 49/49, 매핑 Sb 99.57 %·Pt 98.34 %·Vv 99.6 % 단 SRR31342058 90.66 % 미조사) / ✅ 커밋 426c328·재핀 0d13ece / ✅ D1 채널유지·K=31 / ✅ O. sativa PacBio 3런 = RS II CCS → A14 제외 확정, subreads/RS II의 엑손순서 용도도 저자결정으로 **아예 제외**(A47.3 기록) / 남음: 판정코드의 A-CCS23 부재 시 STOP→INCONCLUSIVE 확인, A46 최종승인 준비. (이전) **다음 행동(09-22 최신):** ✅ issue #60 종결(A47.3, 49런 복원·정본 매니페스트 동기화) ✅ M-CCS26 구조권한 없음 확정 ✅ 코드 v2 구현·재검증(미커밋) / ⏳ 저자 지시 대기: 49런 학습 정렬 잡 · 코드 독립검토→커밋·재핀(판정코드가 A-CCS23 부재 시 STOP→INCONCLUSIVE를 거는지 확인 포함) · **D1 결정** · O. sativa PacBio 3런. (이전) **다음 행동(09-22 갱신):** ✅ A47.2 적용(A-CCS23 구조팔, TES 제외; 문서 반영 완료) / ⏳ 코드 계약 v2 writer 진행 → 결과 검증·독립검토 → 커밋·재핀은 지시 대기 / ❓ **M-CCS26 결정 보류 — 키메라(내부 Illumina 57–98 %)라 FLNC 복귀 불가, 저자 재확인** / ❓ issue #60 원시 CCS 40런 학습 허용 여부. (이전) **저자 판단 대기 2건** — ①원시 CCS 40런(Sb 34·Pt 6)에 A47.1 적용 여부 (V. vinifera 9는 센서스상 FLNC급이라 대상 아님) ②✅전 런 재확인 완료(13런, 앞 2만 리드): A-CCS23 6/6 lima FL, M-CCS26 3/3 FLNC 서명 → **A47.1 전제 붕괴. 저자결정: M-CCS26 FLNC 자격 복귀? A-CCS23(FL) 구조팔 인정? (인정 시 ONT단독·INCONCLUSIVE 전제·계약JSON·핀 eff505f 재검토)** 런별 기기 확인은 ✅ 완료 — 기기 배제 0, WGS 2런 배제, 전사체 후보 49런.
scout은 완료 — B. distachyon·S. italica·P. patens는 PacBio 전사체 0건(영구 공백), O. sativa PRJNA1291274가 미사용 후보. 그 외 rule 15 동결항목은 전부 충족됐다. [진행정본](../notes/a46_approval/PROGRESS.md)

## 09-21 인증복구 실측 후 착수
[진행정본](../notes/auth_restored_20260921/PROGRESS.md).
- [x] **FLNC 정책 저자확정(09-21):** "flnc가 다운받을때 아니면 만들지 말고 그냥 EST처럼 쓴다". ENA 실측 — ERR13994458은 제출파일이 `Col_isoseq.subreads.bam` 단일(Sequel II, 9,093,086 subreads, md5 `7e75b3e7…`), PRJNA911826 36런 전부 `_subreads.fastq.gz`, 두 프로젝트 `result=analysis` 산물 0건 → **FLNC 다운로드 경로 없음**. 따라서 ccs/lima/refine 금지 유지, 두 PacBio 세트는 **EST급 서열증거로만** 사용하고 tier-1 구조검증 자격 미부여. 귀결: A-ONT1이 유일 구조 long-read → MVP §5.3 missing-structural safeguard 발동, **예정 STOP은 INCONCLUSIVE로 하향(ONT 단독 STOP 확정 불가)**. 후속 문서개정 5건은 미실행. [기록](../notes/auth_restored_20260921/FLNC_POLICY_20260921.md)
- [x] **FLNC 문서개정 5건 실행완료(09-21, 저자 승인 "이거 진행해"):** ①B1에 **A47(v1.37)** 신설(frozen 본문 무단수정 없이 개정 추가) + A14·A15 머리 ✏️포인터 ②B6_MVP_PILOT §2 A-HiFi 팔 제거·§6 day5 ONT 단독·§5.3 **인용블록 불변**(A46 rule17과 정규화 후 동일 확인, 1,965자) + 적용노트 ③DATASET_ROLES PRJNA911826 6행 `b1_validation_only`→**`b1_sequence_evidence_only`**, M-CCS26 3행도 저자 "진행해줘"로 **같은 규칙 적용·재지정 완료** ④quarantine D3의 PRJNA911826↔PRJEB77203 혼동 정정 + 변환 되살리기 금지 명문화 ⑤worktree `/tmp/mvp-seed123-integration-20260917`에서 `require_coverage_policy`를 무조건 raise→**A47 계약 검증기**로 교체(+`--coverage-policy` CLI, `bind()` 소스 내 풀링, `pool()`, receipt 실계약 기록), 테스트 +9. **검증:** 관련 스위트 47 passed / 1 failed, 그 1건은 내 변경 stash 상태에서도 동일 실패하는 `torch` 부재 기존실패. 커밋·제출 없음. [실행기록](../notes/auth_restored_20260921/FLNC_AMENDMENTS_20260921.md)
- [x] **FLNC 2차 진행(09-21, 저자 "진행해줘"):** ⑥보류했던 **M-CCS26(PRJNA1470126) 3행 재지정**(`b1_sequence_evidence_only`) — A47에 §3.1 maize 항 신설, maize tier-1 PacBio 구조증거는 **M-HQ18+M-FLNC**로 축소, A46 결정2 무영향. 매니페스트 role 집계 129/55/22/**9**. ⑦**계약 JSON 발행** `revision/scripts/mvp/coverage_policy_a47.json` + 테스트 2건(배포파일 검증 통과 / **실제 CLI가 frozen 50패널로 `--final-decision --coverage-policy` rc=0**, 결과에 계약 기록). **검증:** 61 passed, 실패 8건은 전부 `torch` 부재(기존). **미실행(승인필요):** 바이트 재핀(`772c38c` 핀은 이미 `make_launch_readiness.py`로 깨져 있었음), **M-KNX(PRJNA1290227)는 ENA 미등재라 이번 조회 미포함 — 같은 규칙 적용하려면 별도 확인 필요**. [실행기록](../notes/auth_restored_20260921/FLNC_AMENDMENTS_20260921.md)
- [x] **⚠️정정 3차(09-21): ENA fastq 파일명은 산물이 아니다 — 본세션 오판 철회.** M-KNX 확인 중 발견. ENA는 NCBI경유 PacBio런을 예치내용과 무관하게 `_subreads` 일반이름으로 서비스한다(그런 런은 submitted_* 공란). 리드 센서스가 정본: **A-CCS23 WT 6런 2,604,608리드/평균2,849nt=CCS-level**(A15 "~2.9M" 일치), **M-CCS26 2,260,021/2,408nt=CCS**(A15 2.26M 정확일치), **M-KNX 4런 57,320,590/1,723nt=Revio HiFi Kinnex**(ENA에 **있다** — A15의 'not on ENA'가 오기), **A-HiFi ERR13994458만 진짜 subreads**(제출자가 `Col_isoseq.subreads.bam`/BAM/32.96GB로 명시). **되돌림:** TSV 9행 `b1_validation_only` 복원(집계 129/55/**31**), A47 표·§3.1항·A15배너·change-log 수정, **`### A47.1` 정오표 신설**, quarantine D3 재작성(`Zmays_B73_ccs_*`는 **이름이 맞았다**), MVP §2 한 줄, FLNC_POLICY 머리에 격리배너. **살아남음:** 규칙1~3, A-HiFi만 재지정, ONT 단독 구조소스, STOP→INCONCLUSIVE, ⑤⑦ 구현 전부 무영향. **신규 저자결정 필요:** FLNC 미처리 **CCS가 PacBio 구조팔을 맡을 수 있는가**(A-CCS23이 A-HiFi 자리 대체? maize는?) — 스펙변경이라 미결, 결정 전 ONT 단독 유지. **재핀(지시1번) 보류:** 재핀 대상에 `coverage_policy_a47.json`(`structural_sources:["ONT"]`)이 포함되는데 위 결정이 그걸 바꾸면 즉시 다시 깨진다 → 결정 후 1회 재핀. [기록](../notes/auth_restored_20260921/FLNC_AMENDMENTS_20260921.md)
- [x] **저자결정 2건 반영 + 재핀 완료(09-21):** ①**CCS 구조팔 불가 확정** — A-CCS23·M-CCS26은 A15 검증역할/`b1_validation_only` 유지하되 접합·체인·TSS/TES·UTR·GSF 미지지, `59_evidence_support` THRESHOLDS 미계상. **A. thaliana MVP 구조계약 ONT 단독 확정**(A-CCS23은 A-HiFi 대체 아님), `≥2 A-HiFi` 계속 미충족 → 안전장치대로 **STOP→INCONCLUSIVE**. maize 무영향. A47.1·quarantine⑤·TSS note 반영. ②**바이트 재핀 실행** — worktree `/tmp/mvp-seed123-integration-20260917`에 커밋 2개(푸시·병합 없음): `eff505f`(구현 스냅샷, 이전 핀이 깨져 있던 이유와 **'독립검토 미이행, 핀은 트리 기록일 뿐'**을 본문 명시) + `5febce9`(테스트 핀 `772c38c`→`eff505f`). author/committer **Won <wyim@unr.edu>**, 트레일러 없음. `tasknotes/`는 미커밋 유지. **검증:** 바이트핀 테스트 1 passed; 전체 794 passed/164 failed인데 재핀 직전 실행과 `comm` 대조 시 **차이는 바이트핀 테스트 1건 해소뿐, 신규 실패 0건**. 잔여 실패는 전부 `torch`/`duckdb`/`accelerate` 부재 + 기존 15건. [기록](../notes/auth_restored_20260921/FLNC_AMENDMENTS_20260921.md)
- [x] **extern OOM 게이트 해제(09-21 저자결정 "그냥 무시한다"):** `.extern OUT_OF_MEMORY 0:125`는 더 이상 차단조건 아님. ONT CLEAN/F4/full의 **회계게이트 해제**, 관리자 kernel·slurmstepd·cgroup 로그 선행조건 아님, support draft **미발송 종료**, 원인조사 종료. 게이트 출처가 저자가 아니라 독립검토였고(`longread/independent_review.md:11-17`, pilot의 COMPLETE marker 선기록 결함 대응) `B6_MVP_PILOT_v1.md`엔 extern/OOM 조건이 없어 해제 가능. **해제되지 않는 것:** 실제 승인도구 fixture 미검증(6165965는 wrapper 배포만·미실행), round1·2 수정본은 로컬 합성 검증뿐, v5 미빌드, throughput/scratch 예산 미측정, 자원·중복 통제/독립검토 유효. `.extern` 관측은 계속 receipt에 기록하되 판정에 쓰지 않는다. **A47.1로 A-ONT1이 유일 구조소스가 됐으므로 이 트랙이 구조 readout 전체를 떠받친다.** [결정기록](../notes/auth_restored_20260921/EXTERN_OOM_WAIVED_20260921.md)
- [x] **✅ issue #60 종결(09-22, A47.3): 49런 학습 복원, WGS 2런 제외, M-CCS26 구조권한 없음.** (아래는 재개 시점 기록) **🔴 issue #60 재개(09-22, 최우선):** 격리 51런 `.fa.gz` **전부** 리드헤더가 `{movie}/{zmw}/ccs` — **51/51 CCS, subreads 아님**. S. bicolor 36 · V. vinifera 9 · P. trichocarpa 6. 격리 근거였던 URL `_subreads`는 **A47.1이 이미 정정한 ENA 아티팩트**(같은 함정 두 번째, 이번엔 학습데이터). 당시 `MAX_READS` 가드 미발동 이유였던 "24.8만–157.8만 리드"가 **CCS 규모라는 신호**였다. **D1은 이 건 전까지 결정 보류.** A46 rule 16 문장 정정 완료, quarantine D3 ✏️재개 표시. 격리 디렉터리 무이동. **순서: (1)✅런별 기기 완료(09-22 ENA: 51/51 Sequel II/IIe, `m54283Ue`=Sequel IIe; SRR26636578·582는 WGS 게놈 HiFi로 영구배제 → **전사체 후보 49**: Sb34·Vv9·Pt6; 헤더로 CCS/FLNC 미식별) (c)✅말단센서스: Vv9=FLNC급, Sb34+Pt6=원시CCS 40, A-CCS23/M-CCS26 대조군이 A47.1 전제와 불일치(신고만) (2)A47.1 CCS 규칙이 학습 트랙에 적용되는지 저자판단 (3)role/basis 갱신 (4)D1**. [정본](../notes/a46_approval/ISSUE60_CCS_REOPEN_20260922.md)
- [x] **A46 승인준비 09-22:** **D2 종결**(rule 6 마지막 절 = 분리 제약, K 불변, 개정 불요 — A18.4 `:264` 손실가중 경로 / A18.5 `:270` / `qc_flags`는 `build_b5.py:153` VARCHAR 자유텍스트 / 문장구조). **명령명세 v2** 작성(독립검토 REJECT 17건 중 15건 반영, v1 폐기배너). **M-HQ18 census 실측**: PRJEB22122 16런 전부 RS II/FL-cDNA/Oligo-dT/제출FASTQ, maize 5런 `RS_Sequel_maize_*_hq.fastq.gz`, 7.4만~13.8만 리드/평균 1,594~2,065nt, analysis 0건 → **진짜 HQ isoform 산물, EST급 아님** ⇒ A47 규칙3 미적용, A14 tier 1 유지. **Kimi 정오**: EST coverage-only 권고는 전제 붕괴 / **M-FLNC는 rule 10이 채점세트로 묶어 어떤 입력채널도 불가**(D1 무관 강제) / **(c) 살아있음 — 학습 7종 미조사**. **D1 미결.** [진행정본](../notes/a46_approval/PROGRESS.md)
- [x] **ONT full 종료(09-21 저자결정 "ONT full 사용하지 않는다"):** B1 §§3-9 ONT 전체 run 검증정렬을 수행하지 않는다. **함께 종료:** `ont_full_v2` round1·2 전부(보존, 착수 안 함), v5 빌드, 원격 fixture 실행안, throughput/scratch 예산, affinity list 결손. **별개로 살아있음:** MVP 패널 정렬 `mvp_align_panel_longreads.py` — §2가 **"locus-restricted alignment (not the B1 validation pass)"**로 명시하고 full wrapper 비의존(stdlib+mvp_windows만, 자체 execute). **⚠️남은 저자결정 1건:** 패널 정렬도 버리는가 — 버리면 `structural_available=False` 확정 → §5.3 안전장치로 **예정 STOP이 무조건 INCONCLUSIVE**, 즉 MVP 결론은 사실상 GO/INCONCLUSIVE 둘뿐이고 futility(STOP)는 원리적으로 못 냄. 결정 전 패널정렬 작업 미착수. **B1 영향:** A. thaliana §§6-9에 long-read 구조증거 없음으로 정직 보고(미수행은 음성증거 아님, B1 §3). [결정기록](../notes/auth_restored_20260921/ONT_FULL_DROPPED_20260921.md)
- [x] Duo인증복구/원격실측: pronghorn경유2단홉 확인. 기존잡 변경·신규제출 없음.
- [x] B5 seed123 체인 실측: 3149356/3149357 COMPLETED0:0 추가완료, 3149358Priority/3149359-63Dependency 대기. 전체학습 완료아님. 3163986 terminalFAILED 불변.
- [x] staging I/O차단 해소 확인: PID1944304 종료/다른스테이저 없음/ls·stat 즉시응답. blocker 안전조건2개 충족.
- [x] **GPU staging 명시적 정합화 완료(읽기전용):** ledger13중12·transport14중12 byte-exact 자체해시 검증, 내용오류/손상/symlink/잉여 없음. release.sha256은 0B·모드0660 stub(잘린ledger아님), source-manifest.json은 inode자체 부재. inode연번497495-497508로 복사루프가 release.sha256쓰기에서 정확히 정지 확인, 09-17의 fd4 build_release.py는 복사프런티어가아닌 2차패스 재읽기로 **증거해석 정정**. 승인 미소진(submitted:false)이나 자체 14파일/13항목 게이트 fail-closed로 이 트리에서는 행사불가. [보고서](../notes/auth_restored_20260921/gpu/STAGING_RECONCILIATION.md)
- [x] **B5 실소비 반영:** 3149356/3149357 COMPLETED로 21.5958 node-h / **86.3833 GPU-h 실소비**. forward reserve는 v6기록 352가 아니라 **264 GPU-h**. 352 유래 산식은 재계산 필요. 3163986 terminal·과금0.0566666667GPU-h 유지.
- [x] **v7 staging 완료/검증:** `staging/dev123_p1_release_v7_20260921` 생성전 부재확인(ABSENT_OK)→원격 기존transport(26c1dc0a… 사용직전 재계산/불일치시abort)에서 전개. 쓰기는 mkdir/tar -xpf/chmod 4회뿐. **14파일·13ledger항목 측정확인**(tar tf 14, find -type f 14, `sha256sum -c release.sha256` **13 OK 0 FAILED**, 전파일0444, owner-writable 0, symlink 0). v6 gap 2건(release.sha256 1181B/74a740ea…, source-manifest.json 12174B/4eb7476a…) 해소. payload mtime epoch0은 불변transport 속성(결함아님). [검증](../notes/auth_restored_20260921/gpu/V7_STAGING_VERIFICATION.md)
- [x] **v6 무접촉 확인:** 쓰기후 전수 재해시 결과 파일집합·13해시·크기·모드·**inode**·mtime 동일, source-manifest.json 여전히 부재, release.sha256 여전히 0B/0660. incident증거 온전. 3163986·B5 3149358-63 무변경, queue 동일 6 PENDING.
- [x] **v7 실행결정 draft 작성:** [draft](../notes/auth_restored_20260921/gpu/EXECUTION_DECISION_V7_DRAFT.md). 재핀 86.3833GPU-h소비/264reserve, v6의 392→**304**(264+20+20) 재계산. caps·3163986과금·3163545 0·attempt상한 승계. host 3건(python3.6.15 / 288CPU vs 64 / epoch05 accelerate_state·pending_grads weights-only skip) 처리명시.
- [x] **범위 대조 완료(사용자 질의):** v7은 S0+P1 진단 범위를 바꾸지 않음. p1.sbatch `no TRAINING_DONE`, deploy.py는 `real_probes.py --mode real_tokenizer` 900s 단일실행. `mvp_finetune`은 prepared receipt의 run_role 바인딩 검증값(deploy.py:241-243)이지 학습지시 아님. 입력은 EST 증거DB fullc0lite.db(1068773376B/ed2d0eef…) 읽기전용 사본.
- [x] **게이트3 잔액 PASS(실측):** `accounts` 08:58PDT `bilv-dtai-gh Balance 1068 / Deposited 1499`, 요구 264+20+20=**304** 대비 충분. 교차검증: 1154−1068=**86**이 두 B5 elapsed 유도 86.3833GPU-h와 독립일치 → draft 소비산수 확증.
- [x] **제출 직전 STOP — payload가 v6에 namespace-bound(본세션 transport에서 자체검증):** `admission.py:22 NAMESPACE`=v6, `p1.sbatch:13 --chdir`=v6, `:14,15 --output/--error`=dev123-p1-**v6**-%j, `controller_execution_scope.json:56/57`, `authorization.example.json:44` 전부 v6. **제출시 --chdir이 보존된 v6 증거트리로 보내고 `p1.sbatch:38`의 `> S0_P1_STARTED`가 거기에 파일을 씀**; deploy.py 실패는 그 다음. admission NAMESPACE 상수가 v7을 거부하므로 애초 성공불가. 동일바이트 재배치는 리바인딩이 아니다. 담당이 Option A 권고를 '완결성은 맞고 충분성은 틀림'으로 자체정정. 노드제외는 gh062,gh093 **2개**(본세션이 옛문서에서 3개로 오인용, 정정).
- [x] **run_role 확인:** `deploy.py:241-243`의 `run_role=='mvp_finetune'`은 `validate_prepared()` 내 require guard일뿐 아무것도 선택/시작하지 않음. 실제경로 `deploy.py host`→`real_probes.py --mode real_tokenizer`(:513), `p1.sbatch:43` 'No separate warmup and no TRAINING_DONE'. 학습 아님 확정.
- [x] **재바인딩 표면은 4곳이 아니라 9곳/6파일(독립검토 발견):** 추가 3곳 — `deploy.py:290` job디렉터리 `dev123-p1-v6-job-`, `build_release.py:98` 출력 `release_v6`(**재빌드 도구 자체가 v6바인딩**), `deploy.py:264` authority **파일명** 하드코딩. 정정: `controller_execution_scope.json` v6문자열 **3개**(1개 아님); `authorization.example.json`은 **손편집 금지·`admission.example()`로 재생성**(admission.py:76-78이 집합+필드 동일성 강제, 편집본은 런타임 fail closed); `gh062,gh093`은 **4곳 강제**이며 admission.py:319가 정확리터럴 요구라 p1.sbatch:12만 고치면 admission 실패.
- [ ] **최고위험 `admission.py:23 ACCOUNTS_SHA`:** 사이트소유 `/sw/user/scripts/accounts`를 해시핀. 저자통제 밖이며 **1068 수치를 만든 바로 그 게이트**. 재빌드 전 재측정 필요.
- [ ] **미해소 MEDIUM(현 최상위 위험) 1700s 외부예산:** 1.07GB evidence DB + 최대 2GB HF cache + 2.2GB checkpoint를 각각 전후 해싱하고 나서 900s probe 시작인데, 이 파일시스템에서 **아무도 측정한 적 없음**. 외부 타임아웃이 물면 시도는 과금되고 아무것도 입증 못함. 측정된 I/O 하한을 먼저 잡을 것.
- [x] **본세션 `build_release.py` 전문독해(아무도 끝까지 안 읽은 파일):** ①`if out.exists() or out.is_symlink(): raise FileExistsError` + 배타적 `out.mkdir()` → **덮어쓰기 기계적 불가, 삭제 없이는 같은 경로 재빌드 안 됨** ②빌드가 6파일을 candidate가 아닌 **`HERE`(도구 디렉터리)에서 복사하고 `build_release.py` 자신도 릴리스에 포함** → **재빌드 전 도구 디렉터리에서 리바인딩 선행 필수**, 산출물 후보정은 ledger와 어긋나 등가아님 ③`release.json`에 `scope:S0_P1_DIAGNOSTIC_ONLY`/`node_hours_max:0.5`/`gpu_hours_max:2`/`training_authorized:False`/`review_status:PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW`가 **기계강제로 박힘** — 산문아님. `run_role:mvp_finetune`이 같은 dict의 scope 옆에 있어 역할라벨 확정(세번째 경로).
- [x] **잔액 교차검증 정정:** `1154−1068=86`은 양끝이 `math.floor` 출력(admission.py:459)이라 **±1에서만 일치**. 09-17 잔액이 [1154.3833,1155)이어야 성립하며 이 구간은 비어있지않아 정합적이고 대형오류는 배제하나, **소수점은 잔액이 아니라 sacct elapsed가 확립**. 게이트3은 여유통과: admission.py:461 `max(0,bal-1)`=**1067 ≥ 304**(마진 763). B5완료는 게이트 안깸(admission.py:334-336 terminal수용, spent루프가 b5-seed 건너뜀 → 5/20 cap에 불산입).
- [x] **§2·§3이 게이트와 우연일치가 아니라 코드가 강제하는 공식:** admission.py:198-202가 `balance>=sum(b5)+20+20`(b5는 TimeLimit×GPUs/3600으로 live구성), :190-195가 committed floor를 5/20 대비 계산.
- [x] **v8 빌드/스테이징/검증 완료(09:25), 미제출.** `staging/dev123_p1_release_v8_20260921` 부재확인 후 생성(inode …131249). 14파일, symlink·hidden·owner-writable 0, 0444아닌파일 0, ledger **13 OK/0 FAILED**, 원격14해시=로컬빌드산출물 전부일치. **47/47 OK**(11.772s), 단언완화 없음(전 테스트모듈을 v4원본과 토큰정규화 diff → 차이없음; 초기41은 test_recovery.py 미복사+test_field_alias.py fixture로 41+6=47 정산). `authorization.example.json`은 `admission.example()` 재생성 **IDENTICAL**. 새 핀: transport `90d90482…`, release.json `f716d552…`, ledger `f56f3a29…`. 불변: code.candidate.tar `da161088…`, source-manifest.json `4eb7476a…`, operator-pins.json `45bd5764…`.
- [x] **C1/C3은 패치가 아니라 구조적 해소:** `deploy.py:282`(release=Path(admission.NAMESPACE))·`admission.py:54`·`admission.py:305`가 전부 `admission.py:22` 단일 NAMESPACE 상수에서 파생 → :22 리바인딩이 셋을 동시해결. **`deploy.py:282`가 `--chdir`만 고쳐선 절대 안 됐을 이유.**
- [x] **1700s 예산 정산(미룸없음):** 코드열거 **51.53GB** sha256(SIF 11.12+corpus 23.53 지배), 호스트실측 sha256 **400.1MB/s**·Lustre read 5.3GB/s → 병목은 FS가 아니라 **sha256 CPU**. 1700−900=800s 가용, 필요 128.8s+복사12s, 넉넉히 ≤300s. 손익분기 64.4MB/s = **6.2배 마진**, 1/4속도(100MB/s)에서도 515s로 닫힘. **예산·범위 변경 불요.** caveat 자인: login노드≠compute, `sha256sum`vs`hashlib`, **쓰기 처리량은 추정이지 측정 아님**(승인 namespace 밖 쓰기 불가). 권고: 실제 시도가 자기 staging 경과시간을 기록해 추론→측정 전환.
- [x] **ACCOUNTS_SHA 두번 재측정 일치**(09:18·09:23) `44302f2e…be35f5` = admission.py:23 핀. 1068 잔액게이트 유효. **여전히 drift에 가장 노출 → 제출시점 재확인 대상.**
- [x] **제외노드 정직한 결론:** live `sinfo`에 ghx4 down/drained/failing/unresponsive **0개**, gh062·gh093 정상, B5 배치없음 → **변경근거 없음**. 근거 미기록 guard를 임의로 떨구지 않고 쌍 유지(가드 테스트명이 `test_new_namespace_and_preserved_exclusions`). 제외는 **5곳** 강제(admission.py:66/318/319, p1.sbatch:12, controller_execution_scope.json:58) — 담당이 자기 '2곳' 진술 정정.
- [x] **빌더 자기검증:** `release_v6.tar`를 자기 빌더로 **byte-exact 재현**(`26c1dc0a…27628a`) 후 v8 transport 생성 → 방법이 controller 원본과 일치함을 입증. 지시하지 않은 자발적 검증.
- [x] **⚠️정정: v7 내용은 교체됨. '`v6·v7` 둘 다 보존'은 오류.** 본세션 원격 직접확인: v6=11항목·release.sha256 **0B/660**·release.json `64a547b8…`(원본 그대로), v7=12항목·1181B/444·release.json **`d65684a6…`**(v7-bound 재빌드), v8=12항목·`f716d552…`. 담당 inode 확인: v7 **디렉터리**는 원본(…131208)이나 **파일 inode 전부 신규**(…131234–131248); 원본13파일+notes/는 **09:12 PDT에 `find "$V7" -mindepth 1 ! -name '.transport.tar' -delete`로 삭제 후 교체**. **이 삭제는 당시 승인된 것**(본세션이 저자 승인을 'You are authorized to replace dev123_p1_release_v7_20260921'로 전달), 철회는 그 뒤 도착. 담당이 자기 V8_REBUILD.md·v8_rebuild_evidence.txt의 동일오류도 수정·주석.
- [x] **바인딩결함 증거는 미손실 — 폐기할 것은 '`v7`=증거'라는 주장뿐:** ①원격 v6가 여전히 v6-bound(admission.py NAMESPACE·p1.sbatch --chdir 모두 v6, inode …497495–497508, ledger 0B/0660) ②로컬 `deployment_v4/release_v6/` 무손상, transport `26c1dc0a…` ③`V7_SUBMISSION_BLOCKER.md`의 file:line 리터럴·실패기전.
- [x] **⚠️정정: 담당 공적 과장.** 본세션이 '담당이 본세션·검토자 어느쪽에도 없던 5곳을 찾았다'고 했으나 **검토자가 그중 4곳을 독립적으로 보유**(N1 deploy.py:264, N2 deploy.py:290, N3 build_release.py:98, K5 내 controller_execution_scope.json:34). 담당 고유는 **transport 밖의 테스트스위트 6곳뿐**. 실제 가치는 특종이 아니라 **두 독립 sweep이 동일한 9개 transport 사이트로 수렴**한 것.
- [x] **재사용할 교훈(sweep 설계):** 공유 추가 4곳 전부가 절대경로 리터럴이 아니라 **root에 fragment를 join해 구성되는 경로** — `ROOT/'staging/…v6.json'`, `ROOT/('dev123-p1-v6-job-'+job_id)`, `HERE/'release_v6'`, 상대문자열 `staging/…`. 절대 namespace 경로 검색은 `admission.py:22`와 `p1.sbatch`만 잡는다. **sweep은 root-joined fragment를 표적해야 한다.**
- [ ] **v8 독립검토 진행중(기계요구 선행조건):** release.json이 `review_status: PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW`를 자체 각인. 앞선 v7검토는 '재빌드에 판정 미이월' 명시. 검토항목: 단일NAMESPACE 파생주장 코드검증, 재빌드중 완화된 단언 유무, 예산열거 누락, v7 교체경위, **§8.6은 login노드 결과로 해소불가(실제 실행호스트 필요)**.
- [ ] **~~저자결정 변경: v8 신규 namespace로 재빌드(삭제 없음)~~ (완료, 위 항목들로 대체):** 앞선 'v7 덮어쓰기' 지시는 빌드도구 제약으로 **철회**. v6=I/O정지 증거, v7=바인딩결함 증거로 **둘 다 보존**. 9곳 전부 v8 바인딩, 도구 디렉터리 선행 리바인딩, 개정 결정문은 v8 문서로 신규작성(v7 draft는 편집않고 병존).
- [ ] **HIGH 문서수정:** `EXECUTION_DECISION_V7_DRAFT.md:89-90` 'payload 불변, namespace만 변경'은 **거짓이며 load-bearing**(바인딩을 가린 문장) → 삭제. §5·§8.5를 '재빌드 후 승인'으로 재기술. admission.py:370-376의 3163986 start/end·gh077 핀은 복구시도용 **의도된 것으로 명시**.
- [ ] **~~저자결정: v7 경로 덮어쓰기로 재빌드~~ (철회, 위 v8 항목으로 대체):** 재빌드는 4파일 해시→ledger 74a740ea…→release.json 64a547b8…→transport 26c1dc0a…→**authorization 전 identity핀** 연쇄변경. 함께수정: --job-name v6-0917, stale --exclude. 바인딩결함 증거는 트리가 아닌 notes(V7_SUBMISSION_BLOCKER.md+본세션 인용 리터럴)에 보존. v6 무접촉 유지.
- [ ] **재빌드 후 필수:** 자체 exhaustive 리터럴 전수조사(deploy.py/build_release.py/operator-pins.json/release.json/notes 멤버 포함)+독립검토 목록과 대조, 47test 재실행, 결정문 개정(account사실·caps·1068/86 교차검증 승계), **새 독립검토**(진행중 검토는 제출불가 패키지 대상이라 판정 미이월).
- [x] **v8 독립검토 APPROVE_WITH_FIXES(CRITICAL·HIGH 없음)** 및 6개 fix 적용완료. M1: `review_status`/`runtime_ready`/`measured`/`scope`는 **선언적일 뿐 fail closed 안 함**(두 실행파일 grep 결과 소비 코드경로 없음), release.json이 해시핀이라 제자리 수정불가 → **v9 재빌드 부정당**, `review_reference`가 정본. M2: 누락 ~14 sha()/pinned() 사이트 접어넣음(9.6MB/0.02s, 예산 무영향). M3: **p1.sbatch:38이 deploy.py host 이전에 S0_P1_STARTED를 쓰므로 이후 어떤 실패든(통제밖 노드장애 포함) 네임스페이스 영구소진→v9 재빌드 강제** §8b로 명시. L3: **3163986 근본원인은 exact-inventory 검사이지 I/O정지 아님** → 예산분석은 다른 가설의 마진분석이며 선행실패를 닫지 않음.
- [x] **§8.6 재기술(in-job, fail closed·fail fast):** pre-flight 불가·불필요. `p1.sbatch:44`가 `/usr/bin/python3 deploy.py host`를 돌리므로 **검사할 두번째 인터프리터가 없음**; `CPUs/Task` 파서는 live scontrol 응답 파싱이라 **애초에 pre-flight 불가**(`deploy.py:295` admission.live, `:296` verify_source_manifest·`:326` 해싱보다 앞). 비용0 login pre-flight(정확한 staged 바이트, in-memory compile ALL_OK, `text=True` REJECTED/`universal_newlines` rc=0, __pycache__ 0): **방증이지 해소 아님**으로 기록. `test_host_python36_api`도 `gh-login03` 관측도 §8.6을 해소하지 않음.
- [x] **저자 명료화로 위험 축소:** 계산은 전부 **해시핀 SIF**(`transgenic-ngc-26.08.sif`, 11,120,398,336B, deploy.py:324) 안에서 `--cleanenv --containall --no-mount hostfs`로 실행 — stage_checkpoint(:378)/make_launch_readiness(:396)/real_probes(:513)/run_bounded(:520) 모두 SIF Python 3.12.3. 호스트 3.6.15는 apptainer 명령줄 조립만. **호스트 인터프리터 발산은 결과를 오염시킬 수 없고 래퍼를 일찍 죽여 네임스페이스를 소진시킬 뿐.** 잔여위험 근거: 3163986 보존 stderr의 `/usr/lib64/python3.6/subprocess.py`가 **compute노드 gh077**에서 생성됐고 deploy.py host가 **완주 후** 컨테이너 내 epoch05 guard에서 실패; `text=`는 3.7 추가라 모든 3.6.x가 동일거부.
- [x] **Gate5 발급:** authority `dev123_p1_controller_authorization_v8.json`(2916B, sha `8ac56646…cb20d`), transport `release_v8.tar`(2887680B, `90d90482…`). 둘 다 **생성전 부재확인**, `admission.example()`에서 생성(템플릿 동일성 구조적 보장), 실제 `admission.validate_authorization` + canonical-JSON 왕복 **PASS**(스테이징 전), `review_reference`=V8_INDEPENDENT_REVIEW.md sha `fd946236…c699e9e9`. **v6 authority 재사용 안함·무접촉.**
- [x] **Gate8 환경변수 3종 검증:** GATE `8ac56646…`, TRANSPORT `90d90482…`, LEDGER `f56f3a29…`. 각각 자기가 핀하는 산출물 대비 검증(노트 복사 아님). 전부 64 소문자hex로 `p1.sbatch:33-35` 정규식 통과. `DEPLOY_*`가 `p1.sbatch:20-26` 접두사 스크럽 클래스와 무충돌 확인(담당 자발).
- [x] **🚀 Gate4 통과 후 제출 1회 — job `3187760` (2026-09-21 09:58:02 PDT submit).** Gate4 단일패스·무재시도: 잔액 1068→게이트 `max(0,1068-1)`=**1067≥304**, 우리 RUNNING 잡 없음, 3163986 terminal, B5 6건 PENDING, ghx4 down/drain/fail **0**, gh062·gh093 정상(정책상 제외 유지), **S0_P1_STARTED v6·v7·v8 전부 부재**, gate5 산출물 정확. 담당이 숨기지 않고 보고한 변화: 신규 reservation `sup-26517`(타 계정, 18:00 CDT 시작) — 정지기준 불충족·1노드 잡 차단 불가로 진행, 스냅샷에 기록.
- [x] **journal/snapshot을 sbatch 이전에 원자적 생성**(`set -o noclobber`): `submission-attempt.json`(877B), `reservation.json`(534B, gate4 캡처 포함), `submission-raw-output.txt`(13B), `submission-result.json`(169B, `ambiguous:false`/`resubmission:forbidden`). **sbatch rc=0·출력 `3187760` 명확 → 불확실 아님, 정합화 불요, 재제출 절대금지.**
- [x] **본세션 독립확인:** `squeue -j 3187760` → **dev123-p1-v8-0921 PENDING(Priority), 1 node, 30:00**. `sacct` Submit `2026-09-21T11:58:02`(CDT). 사용자 큐 총 7건(=신규1+B5 6). 제출후 보존 재확인: v6 13파일·ledger 0B/0660/inode …497508, v7 14파일, 3163986 FAILED 00:00:51 유지.
- [ ] **⏳ 결과 대기 — 큐잉은 측정이 아님.** 학습·후속잡·재시도·추가시드·full ONT·merge/push·과학적 수용 전부 **여전히 미승인**. 실패시 전 artifact 보존 후 **별도 검토된 다음 결정** 필요. 반복 모니터 없음 — 사용자 요청시에만 관측.
- [x] **~~제출 전 7조건~~ (전부 충족 또는 재기술 완료):** ①저자승인 ②draft·staging 독립검토 ③**live 잔액조회(미수행)**로 304 헤드룸 확인 ④제출시점 live 재조회 ⑤v7 control namespace/정본 authorization 생성(부재확인·저널1회) ⑥host 3종 재실행 통과 ⑦v6·3163986·B5 무접촉. 완비된 staging은 실행허가가 아니다.
- [x] **eval decoder repin 완료 — 실제 stale은 다른 것:** [기록](../notes/auth_restored_20260921/eval_decoder/REPIN.md). runtime handshake repin은 **이미 최신**(PROBE_CODE 핀 3개 live바이트 재계산 전부일치, 무접촉) → tasknotes/seed123_eval/REPORT.md 잔여항목2 대체. 진짜 stale은 **decoder consumption contract** 2곳.
- [x] **(a) producer에 closure 요구 부재:** `mvp_produce_counts.produce()`가 `payload['decoder']['files']`를 비어있지않은 매핑이면 수용하고 나열된 바이트만 재확인 → `mvp_panel_compatibility.py`/9603 reference-mapping producer 이전의 legacy·d99-era transport가 그대로 통과. 기존 fixture(`test_mvp_score_producer_integration.py:61`)가 **테스트파일 자신을 decoder패키지 전체로 선언**했는데 수용된 것이 그 구멍의 실증.
- [x] **(b) decoder가 자기 핀된 closure보다 적게 기록:** `decode_panel()`의 수동목록이 DECODER_ONLY 14중 **13**만 커버, 누락은 `src/transgenic/evidence/mvp_bootstrap.py`(`mvp_produce_counts`가 모듈스코프에서 validate_paired_record import하므로 실제 로드됨; operator decoder_code_sha256 핀은 이미 포함).
- [x] **수정 4파일(미커밋):** decoder는 inventory가 DECODER_ONLY 밑으로 drift시 **모델로드 전 fail closed**, producer `verify_decoder_package()`가 DECODER_ONLY 전경로 존재 요구(per-file check_file 바이트동일성 불변), fixture를 실제closure 공급으로 정정(계약완화 아님), 신규 계약테스트. RED→GREEN 경과: baseline47 → RED 6failed → seam만·gate없음 3failed/3passed(거부2건 DID NOT RAISE) → **GREEN 6passed** → producer스위트에서 **gate가물어 1failed**('decoder package omits reviewed decoder closure') → fixture정정 후 53passed. 넓은범위 202passed/1skip, decoder integration 24passed, 최종 161passed/1skip. **본세션 재현: 6 passed·변경4파일 일치.**
- [x] **핀 구분 확정(흐리지 말것):** `b53c3175…`(MVP_FINAL_SOURCE_INTEGRATION_REPORT.md)와 `c0e8c141…`(MVP_REFERENCE_MAP_FIX_REPORT.md)는 **각자의 검토된 source state에서 기록된 operator 핀=승인값**. `cec4611d…`는 담당이 현재 바이트로 계산한 **측정값이며 핀아님·미승인·미검토**. 셋은 불일치하며 **담당 수정 이전부터 이미 불일치**(선행 미커밋 handshake repin이 DECODER_ONLY 튜플에 mvp_panel_compatibility.py를 추가, 두 핀은 그 이전). **caveat: 담당이 pristine pre-edit 트리 digest를 계산하지 않아 이 선행 staleness는 튜플변경으로부터의 추론이지 측정이 아님.**
- [x] **핀 검사 pass→fail 전환 없음:** decoder_code_sha256은 소스에 컴파일되지않고 런타임에 operator가 공급, 이 소스라인에서 두 기록핀 대비 production decode 실행된적 없음, 시험은 자체계산 핀 공급.
- [ ] **새 operator 핀 필요(controller 작업, 담당·본세션 모두 미발행):** 운영 decode 전 `runtime_preflight.decoder_code_sha256` **정확히 1개 값**(DECODER_ONLY 위 digest)을 **모든 소스작성자 종료 후 source freeze 시점에** controller가 동결+독립검토. 담당 §5 값 복사 금지(추가 DECODER_ONLY 편집시 또 이동). **현재 ONT·GPU 담당이 아직 소스 작성중이므로 freeze는 그 이후.**
- [x] **decoder 독립검토 APPROVE(CRITICAL·HIGH 없음):** [검토](../notes/auth_restored_20260921/eval_decoder/REPIN_INDEPENDENT_REVIEW.md).
- [ ] **MEDIUM-1 landing-order 제약(결함아님, 운영주의):** `decoder_code_paths()`가 `mvp_panel_compatibility.py`를 `strict=True`로 resolve하고 `set(DECODER_ONLY) <= recorded`를 단언. 그런데 그 파일은 **untracked**이고 14번째 튜플 항목도 **미커밋**이며 **둘 다 다른 소유자 것**. → **이 4파일만 깨끗한 `9603e20`에 올리면 모든 decode가 `FileNotFoundError`.** 랜딩 순서를 반드시 조율할 것.
- [ ] **MEDIUM-2 기록의 과장:** `verify_decoder_package()`는 존재를 DECODER_ONLY 대비, 바이트를 **이 checkout 대비** 확인할 뿐 기록 digest를 operator `decoder_code_sha256` 핀에 **결속하지 않음**(그 검사는 decoder측 `mvp_decode_runtime`에 있음). 따라서 **미검토 로컬편집이 있어도 `produce()`는 통과**. 범위분리 자체는 정당하나 기록의 '검토된 closure에 결속했다'는 표현은 과장 → 문구 수정 필요.
- [x] **⚠️정정 '13 of 14'→'12 of 14':** 구현자 보고의 '14중 13 커버'는 `9603e20` 기준으로는 **12 of 14**이며 더이상 검증 불가(본세션도 구현자 수치를 그대로 전달했음).
- [x] **LOW:** `tests/test_mvp_eval_decoder_repin.py:26-29` inventory 테스트는 **공허**(`decoder_code_paths()`가 assert 평가 전에 raise), positive 경로 2개가 `decoder_code_paths()`에서 패키지를 유도해 구성상 자기정합, §6 caveat가 측정가능한 범위를 과소진술.
- [x] **검토자 자체실행 검증(구현자보다 강함):** HEAD producer gate와 HEAD fixture가 테스트파일을 패키지 전체로 선언하는 것 확인; `mvp_produce_counts.py:19`의 모듈스코프 `mvp_bootstrap` import와 HEAD decoder 목록에서의 부재 확인; drift gate raise 확인; `decoder_code_paths`가 model load 이전 실행 확인; **14개 closure 항목을 하나씩 차례로 제거해 전부 개별강제 확인**(구현자는 단일항목 테스트만); PROBE_CODE 3/3 일치; `check_file` 무수정. 6/53/202+1skip/24 전부 재현. skip은 기존것(`tests/test_mvp_decoder_runtime.py:203`, transformers 버전조건, HEAD:188과 동일 pytest.skip).
- [x] **⚠️구현자의 '측정불가' caveat를 검토자가 반증·측정:** 두 기록핀이 **자기 커밋의 13항목 트리에서 정확히 재현**(`b660bf1`→`b53c3175…`, `66f130d`→`c0e8c141…`) → 각자의 검토된 state에서의 진짜 핀임 확인. base `9603e20`의 digest는 **`b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088`로 둘 중 어느것도 아님** → **두 핀 모두 미커밋 편집이 존재하기 이전에 이미 커밋된 base 대비 stale이었음**, git객체 위에서 추론없이 측정. 튜플논증도 추론이 아닌 증명(핀은 13키로 계산됐는데 pre-edit 트리는 이미 14키). **영구 측정불가인 것은 pre-edit digest의 수치값뿐**(바이트 미커밋, git stash 비어있음, .orig/.bak/swap 없음). 검토자는 HEAD-blob 재구성을 두 핀과 다름을 확인하는 용도로만 계산하고 **측정으로서의 지위를 명시적으로 부정**. `cec4611d…`도 독립 재현. **operator 핀 발행·제안·계산 없음.**
- [x] **얽힘 분리가능:** diff가 seed123 profile 소유자 작업과 섞여 있으나 mtime이 profile/panel 작업=09-17, repin=09-21로 분리하며 검토된 4개 추가는 profile 코드경로를 건드리지 않음.
- [ ] **~~decoder 변경 독립검토 진행중~~ (완료):** 선례가 독립검토로 끝나는데 미획득이라 착수. cross-checkout 소비는 설계상 미지원 유지(`verify_decoder_package`가 producer ROOT 하 절대경로로 대조, 기존 exact same-checkout readiness gate와 정합) — 재배치·교차버전 adapter는 별도 검토결정.
- [x] **ONT wrapper 보강/커버리지 발견:** [기록](../notes/auth_restored_20260921/ont/WRAPPER_AND_FIXTURE.md). **정정: 6165965는 full wrapper를 배포했을뿐 실행하지 않았다.** 두 wrapper.py byte-identical(sha256 314d6027…, 본세션 자체검증 일치). fixture가 오케스트레이션을 자체재구현하여 `execute`·`slurm`·`check_boundary`·`fresh`·`pinned`·`stop_group` **6개 미도달**(담당 보고 '11중3사용', 본세션 AST로는 귀속방식따라 3~5 — 결론동일). `slurm()`은 어떤 실제allocation에서도 미실행. **커버리지 문제이지 wrapper 결함아님.** 기존 '기술결과수용/controller14PASS'를 'full wrapper가 실제할당에서 검증됨'으로 읽지 않는다.
- [x] **실제 실행된 것:** runtime_observed.json 기준 minimap2 2.28-r1209/samtools 1.3.1/seqkit 2.13.0/python 3.11.14 전부 SHA256핀·9record nested ledger. 승인런타임 도구는 실제 실행됨.
- [x] **wrapper 4갭 보강(TDD, ont_full_v2/):** CPU affinity **리스트** 영속화(명명된 미해결갭), RLIMIT_FSIZE 상한 이식(request선언), cgroup **pre/post** 관측(6165965는 pre만; 기록전용·판정안함), scratch예산. RED 21tests failures=4 errors=21 → GREEN **21 OK** (python3.10.12·3.12 양쪽 exit0). 3.8 guard ValueError 유지. 신규2건은 실제 child프로세스 RLIMIT_FSIZE 행사, 나머지는 선언된 synthetic 치환으로 compute containment 아님. F1 closure를 **실제 배포된** frozen tree 양쪽 대비 재검증 PASS(종전엔 synthetic 사본 대비만).
- [x] **잔여 synthetic 명시:** receipt 자인 4seam + 7read/64kb 입력 + **신규식별 오케스트레이션 우회**. 로컬에 승인런타임 부재(minimap2 없음, samtools 1.13≠1.3.1, seqkit 2.0.0≠2.13.0)로 입력규모 한계는 로컬해소 불가.
- [x] **게이트 불변 재확인(읽기전용 1회):** 6165965 COMPLETED0:0/.batch0:0/**.extern OUT_OF_MEMORY 0:125**/.0 0:0. CLEAN/F4/full ONT 차단유지, 관리자 kernel·slurmstepd·cgroup 로그가 차단증거, support draft 미발송. 6165965의 affinity list·reaping은 **영구 미관측**이며 이번 수정은 향후 실행에만 유효.
- [x] **저자결정: execute() 우회 수정 승인·완료. 4개추가/request-contract 수용. v5패키지는 게이트 해제까지 미빌드.**
- [x] **ONT execute() 우회 수정 완료:** `fixture_execute.py`가 v4 driver의 중복 오케스트레이션을 대체하고 `wrapper.execute()`에 제어를 넘김. `fixture_substitutions.py`는 gate binding만 담은 단일 핀·해시 파일(guard 완화 없음). **커버리지 숫자 확정(담당 자체정정):** 1차 AST가 `fixture.py:77`의 `w.check_source_closure`를 놓침(verify_release 안에서 wrapper가 `w`로 바인딩; panel_seam의 `w`는 driver.w라 scope-aware 필요) → **직접호출 4/11, 런타임실행 6/11(require·stop_group 전이), 미실행 5/11(check_boundary·execute·fresh·pinned·slurm)**. 본세션 3~5 범위가 이를 감쌈. 담당이 `stop_group`을 미도달로 잘못분류한 것도 정정(child의 finally에서 실행).
- [x] **치환이 자기신고가 아니라 탐지되게 구현(요구 이상):** `receipt['bindings']`가 `binding_identity()`로 각 치환가능 callable의 실제 해결파일을 기록 → patch된 `slurm`은 소스파일이 없어 request 주장과 무관하게 `matches_expected:false`. `request['seams']` 필수, `request['substitutions']`는 SHA256 핀·기록(production은 null). **`slurm()`은 더이상 patch-out 아님** — `execute()` 안에서 선언된 stand-in `scontrol`(실제 scontrol 필드문법 방출) 상대로 실제 실행되어 parser·ownership·TRES·memory·time-remaining·affinity 전부 동작. 치환으로 남는것은 '실제 allocation이 아님' 뿐이며 선언됨.
- [x] **RED→GREEN:** RED `Ran 29 tests, FAILED(failures=2,errors=5)` — 7개 ExecuteDrivenFixture 전부 재구성 round-1 wrapper에 실패. GREEN `29 tests OK`, python3.10.12·3.12.2 양쪽 exit0. **커버리지는 측정:** wrapper 파일 필터 `sys.settrace`(test_wrapper.py:518)가 entered==defined 요구, `execute()` 경유 **15/15 도달**(원본11+추가4). 향후 변경이 함수를 고립시키면 해당 테스트가 실패하며 이름 지목. **본세션 재현: 29 OK, wrapper 함수 15개(baseline11 + binding_identity·cgroup_observation·check_scratch·enforce_output_limit) 확인.**
- [x] **담당 자인 caveat 2건:** ①8번째 신규테스트(test_driver_rejects_a_release_pin_mismatch_before_import)는 red/green 양 트리에 있는 새 driver를 시험하므로 **구조상 red 불가** ②7개중 2개가 맨 assertRaises(ValueError)로 옛 wrapper에서도 통과했었고(무관한 이유), **red 로그 전에 특정 메시지 단언으로 조임**.
- [x] **게이트 불변/원본 무수정:** v5 미빌드, externOOM0:125 미해결, CLEAN/F4/full ONT 차단, 관리자로그 차단증거, 6165965 affinity list 영구미관측. `ont_fixture_v4/fixture.py` 포함 원본 전부 무수정, 제출·커밋·푸시 없음.
- [x] **A11 4항목 완료:** [기록](../notes/auth_restored_20260921/a11/AUDIT_AND_JUNCTION.md). completeness boolean 3개 **전부 false 유지**. 미강제 규칙2건 수정(est_molecule_closure.py): ①`library==dataset`(alias canonical 경유 포함) 통과→**다운로드 전체를 한 molecule component로 병합할 수 있던 버그**, 이제 raise ②문자열 `unknown` BioProject가 조용히 통과해 diagnostic_ready=True 가능→`unresolved_identity:<uid>` **blocker**로(error도 부재주장도 아님; timeout된 BioSample에 맞는 처리). unknown **genotype**은 계속 허용.
- [x] **routeA 실trace fixture:** 인증 route-A 관측은 로컬자료로 **구성불가**(reference SHA256 미기록/MD5만, normalized protein SHA256 미기록/전체FASTA해시만, c0는 replay필요—GFF col8은 miniprot phase이지 codon ordinal 아님, MP000002 acceptance/ambiguity 미조사). 대신 `est_routea_fixture.py`가 기록된 Sitalica MP000002 trace를 싣고 **5갭 표시+MD5·전체FASTA해시·raw GFF phase 대체 거부**. 추가producer 참조SHA는 **차단유지**(est_real_sample.py:31 핀 a2843aa3…은 caller제공 registry JSON 해싱이라 repo파일 대조불가; resolver가 producer 정확히2개 요구; Ptrichocarpa는 새 컴파일·검토핀+존재하지않는 reference SHA256 필요, MD5 20c6f4bb…만 있음).
- [x] **junction-counts 독립결속 구현:** 갭 확인—`mvp_align_panel_longreads.py:198-200`이 junctions.tsv 생성, `mvp_score.py:53 junction_counts()`는 완성된 supported집합 소비, **둘을 잇는 코드 부재**. 새 `mvp_junction_binding.py`: 이름+MD5로 alignment.json 결속, frozen PANEL_SHA256 확인, run-id/source를 run inventory 대조, 중복 run identity 거부, frozen threshold로 read count→support 재유도, **증거부재시 0이 아니라 None**, multi-run pooling 거부(receipt에 `coverage_policy: PENDING` 방출로 하류가 승인된 pooled결과로 오독 불가).
- [x] **spec계약/nativecrash:** 소스가 SPEC보다 엄격한 지점 확인(stage_checkpoint.py:94-96이 SPEC 명령줄의 mutable `best` symlink 거부). 실제갭 1건: **N==250 강제 소스측 guard 없음**. SIGSEGV는 UNRESOLVED 보존—로그SHA 142745e3… 재계산일치, frame `mvp_decode_panel.py:141`(이전검토의 142→141 의역 정정), exit139, collected item **274번째**에서 폴트(총1200~1225은 유도추정/receipt아님), argv·cwd·core 복구불가. **좁은스코프 통과(145.20s/133.26s)는 수정입증 아님**—폴트가 전체repo세션 누적 in-process 상태 의존. 비파괴 다음수는 `pytest --collect-only -q`로 274까지 node ID 복구(미수행).
- [x] **시험 재현 확인(본세션 자체실행):** junction `10 passed`(담당보고 일치), closure+routeA `59 passed`. 앞선 ModuleNotFoundError는 본세션 호출오류(PYTHONPATH=src 누락, repo기본 python3.8은 패키지 import 불가). 정본 인터프리터 `/tmp/mvp-final-source-env/bin/python3.12`.
- [x] **coverage policy는 저자결정 아님으로 정정:** 담당이 자기 항목을 철회. 평가규칙은 이미 승인(b6_execution_approval_20260914 / MVP §5.3)이며 `require_coverage_policy` 무조건 raise는 **소프트웨어 차단이지 보류된 승인 아님**. §5.3/§5.4 계약을 hardcoded PENDING reject 대신 구현하는 **엔지니어링 과제**로 재분류.
- [ ] **저자결정 2건(좁은 범위):** ①명명된 A-HiFi ERR13994458이 bounded canonical inventory에 부재한 상태에서 **ONT 단독이 complete MVP structural availability를 충족해 STOP을 낼 수 있는가**(미결정시 ONT-only STOP 불가; 진실한 INCONCLUSIVE 경로는 이미 허용/결정불요) ②**A-CCS23(PRJNA911826, `_subreads.fastq.gz`에서 6run 다운로드)이 명명된 A-HiFi 산물을 대체할 수 있는가**(CCS/FLNC 산물정체 미해소; 미결정시 대체결속 불가, 기존 artifact 정합화가 선행, 재다운로드 미승인).
- [ ] **엔지니어링 차단(증거필요/결정불요):** 정본 library·preparation identity와 alias표+mate/parent 관계표, 전member A46 audit(`--secondary=yes -N 50 -p 0.5`)와 global+train-context exclusion manifest, `Ptrichocarpa_533_v4.0.fa` reference SHA256 receipt+새 검토된 producer-registry 핀, routeA closure 4종, SIGSEGV 선행 node ID, N==250 guard.
- [x] **caveat 기록:** MVP worktree는 **공유중이며 타 소유자 미커밋 편집 포함**(mvp_score.py, mvp_produce_counts.py, 테스트 일부). 95-pass는 깨끗한 9603e20이 아니라 그 공유 작업상태 기준 — 인용시 전제 명시.
- [ ] 4담당 결과 수령 후 정합화 → 필요한 수정·독립검토 → 실행조건 충족분만 후속. 커밋/병합/푸시는 저자 지시 시에만.

## 16:01 PDT 인증복구 후 후속 진행
[진행정본](../notes/auth_resume_20260917/PROGRESS.md).
- [x] 인증복구/livequeue확인:기존P1/B5Priority대기,잡변경없음.
- [x] 새seed123GPU복구패키지release_v6/47시험/독립승인/실패51초정산·durable승인고정.
- [ ] GPU배포검증·6gate실측:원격작은파일I/O정지로미제출. 소유PID1944304미종료/부분release보존. I/O·종료확인→명시정합화/전체해시검증→livegate후단일제출.
- [ ] 실제50panel호환성 CPU검증패키지·독립requestpin·런타임일치 확인후실행.
- [x] ONT실제도구fixture준비/독립검토/실행6165965및기술결과검증. 기대BAM/junction/런타임/4GiB검증통과. 앞선6165962시작guard실패별도보존.
- [x] ~~ONT CLEAN/F4최종/전체실행~~ **종료 — 저자결정 09-21 "ONT full 사용하지 않는다".** 회계게이트는 이미 해제됐으나 full 실행 자체를 하지 않는다. 함께 종료: `ont_full_v2` round1·2 wrapper 보강(29 tests GREEN·커버리지 15/15는 유효하나 **소비처 소멸**), v5 빌드, 원격 bounded fixture 실행안, throughput/scratch 예산, 6165965 affinity list 결손. 보존하되 착수 안 함. **혼동 금지:** MVP 패널 정렬(`mvp_align_panel_longreads.py`, §2 "locus-restricted alignment (not the B1 validation pass)")은 **별개이고 full wrapper에 의존하지 않는다**(확인: stdlib+`mvp_windows`만 import, 자체 execute). [결정기록](../notes/auth_restored_20260921/ONT_FULL_DROPPED_20260921.md)
- [x] EST16개실GenBankrecord/서열일치/clone_lib준비법확인,routeA경로/2추가producerreceipt위치확보.
- [ ] A11전체audit/exclusion/alias정합화·routeA실tracefixture·추가producer참조SHA연결. BioSample3추가조회SSLtimeout,미확인을부재로판정하지않음.

## 14:00 PDT 중단 복구
[복구 기록](../notes/seed123_recovery_20260917_1400/RECOVERY.md).
- [x] 실제 worktree 존속 및 미커밋 MVP31/EST14파일 백업·검증.
- [x] 기존 검토 지적과 최신 코드 대조, 원격 잡 중복 여부 확인. 14:01 P1/B5 대기, EST sample 제출 흔적 없음.
- [x] scorer 명시 run_id 모순 재현8FAIL → 최소수정 → 관련158PASS/1skip 및 publication2PASS. 독립검토10PASS/수정범위 수용 완료.
- [x] runtime reviewed-source repin35PASS/1skip, panel 독립 request-pin caller22PASS, scoring 계약 대조(기존pooledF1유지) 수용.
- [x] EST배포 D1/D2/권한guard 수정·독립수용/controller26PASS,원격패키지검증/직전중복확인→6165875 단일제출14:40PDT(2CPU8GiB30min).
- [x] ONTwrapper F1-F3수정/controllerdiff·14PASS확인. 준비코드 범위만 수용.
- [x] 6165875 실제결과/회계/진단·출처 검증:allocation/batch0,해시/결속독립수용+controller재확인. bounded진단만 수용,externOOM미해결.
- [x] 실표본77replay실패·5geometry실패 원인분석,producer-path길이검증수정/독립수용/controller200PASS. 원본보존,새로컬재생414HSP길이통과48truth실패/geometry380성공34충돌. [최종체크포인트](../notes/seed123_recovery_20260917_1400/FINAL_CHECKPOINT.md).
- [x] GPU인증차단해소/livequeue16:01재확인. 이후실행은최상단후속목록.
- [ ] ~~ONT실제승인런타임fixture/운영게이트~~(09-21 저자결정 종료)·A11실metadata/routeA/전체projection잔여. [복구 기록](../notes/seed123_recovery_20260917_1400/RECOVERY.md).
- [ ] 남은 수정·focused 검증·독립검토 후 가능한 후속 실행. 기존 seed123 체인/불변 산출물 보존.

## 현재 착수 — 사용자 1·2·3 시작 지시
[실행정본](../notes/SEED123_THREE_TRACK_EXECUTION_20260917.md): seed123설정/평가연결/EST실자료3담당실source조사착수. 조사→controller인터페이스/파일소유권정리→TDD구현→검증/독립검토→실행조건충족시후속. 기존B5/옛receipt/실데이터보존. 계획보고만하고중단하지않음.

## 최우선 최신 결정 — seed123 고정

[저자결정 정본](../notes/SEED123_ONLY_DECISION_20260917.md): 현재는seed123으로개발·학습·품질확인. 다른시드는완성/품질확인후로연기. 아래의‘MVP3seed’/‘MVP진단후456→789’는현재실행지시가아니며후속반복계획으로보류한다.
- [ ] 기존MVP seed binding/config/readiness를123단일실행과대조,필요수정·시험·독립검토. 기존receipt재명명/불변제출release수정금지.
- [ ] seed123결과의예측/평가·품질/오류확인에집중. 단일시드를다중시드최종집계로보고하지않음.
- [ ] 추가시드재개는완성·품질확인이후. 현재456/789및다른시드자동제출없음.

## 최신 승인/착수 — HSP·phase·A11·B5

[저자결정/실행정본](../notes/APPROVED_HSP_PHASE_A11_B5_20260917.md): 전체HSP reportedlength≥30,정확phase/충돌증거만제외·독립일치보존,기존A11구현승인. seed123보존/MVP진단우선후456→789순차실행승인. 아래동일항목의옛‘저자결정대기’는이범위에서대체됨;예산/실환경검증/A46전체/mainpush는별개.
- [x] 격리HSP수용/phase충돌primitive구현·실144fixture재검증/전체focused278PASS·독립검토수용. 코드8b646ef/검토보존fca911e,main미병합.
- [x] A11normalized입력의연결·누출진단구현/검증,version-library모순누출Important수정재검토수용.
- [ ] 후속consumer/sourceidentity·routeA실trace/A11실9종metadata·auditclosure연결→실제투영/집계/raster. Primitive수용을전체phase/fullC0완료로확대하지않음.
- [ ] B5추가시드456/789는최신seed123단일진행결정으로보류. 기존실queue/예산조사기록보존,품질확인전제출없음.

**09-17 재확인 정정:** [HiFi/ONT](../notes/HIFI_ONT_POLICY_RECHECK_20260917.md), [externOOM](../notes/EXTERN_OOM_GATE_RECHECK_20260917.md). 평가정책/MVP진행은이미승인됨;소프트웨어의PENDING을새저자승인필요로해석하지않는다. PacBio6run다운로드는완료,검증산물CCS/FLNC정체성및별도A-HiFi ERR13994458의위치/산출물대조가실제잔여. 무조건재다운로드/ONT-only재승인요청금지. externclean조건은보수적운영검토게이트이지새과학정책아님.

**전체 확인 정본:** [중복제거·우선순위·차단구분](../notes/TODO_FULL_RECONCILIATION_20260917.md). 감사시현재29미체크/과거85행전수대조. 감사줄번호는 [원본snapshot](archive/todo_before_full_audit_20260917.md) 기준. GPU대기와무관하게 **evaldecoder repin/ONTfullwrapper·fixture/junction-counts연결/spec계약/nativecrash증거** 로컬작업은남아있다. 이들이모두차단됐다고해석하지않는다.

마지막11:01PDT실조회:3163986Priority대기/사용0,seed123의3131538COMPLETED→3149356Priority대기/후속Dependency. seed123전체종료아님. [live](../notes/approved_execution_20260917/final_live_gpu.txt).

## 최우선 정전 복구 — 00:34 PDT

정본: [복구 진행](../notes/POWER_RECOVERY_20260917.md). 아래 오래된 미완료 체크는 최신 Task2/실제 상태와 대조하며 반복 실행하지 않는다.
- [x] 최신 기록/원본 Git 객체/미커밋 파일 보존 및 pronghorn 기존잡 종료 상태 확인.
- [x] GPU3163986/B5 실제 상태 확인: P1 Priority대기/사용0, B5 3131538RUNNING; 원격학습보존. GPU후속은 P1미실행으로 차단.
- [x] 정전 직전 reference_map_release 독립 배포검토 APPROVE/새12testsPASS, canonical 새namespace7payload해시 확인 → **6165628 단일CPU4/16GB/1h제출00:41PDT**.
- [x] 6165628 실제50ID/참조/새panel산출물·회계 검증. 50/50참조호환,150artifact해시/배열독립재구성PASS,12tamper거부/controller재실행exit0. 기존 실패본/externOOM 별도 보존.
- [x] 공유ONT/panel referenceauthority·파일identity·panelSHA대조PASS, [새ledger](../notes/power_recovery_20260917/runtime_ledger.json) 및복구정본에결과/차단연결.
- [ ] 후속실행: P1실측대기,**~~ONTfullwrapper/실toolfixture·운영/예산검증~~(09-21 저자결정으로 종료)**,HSP/phase승인규칙의실source/routeA/A11입력연결잔여. 이미승인된정책재요청금지. Decoder/최종평가수용아님.

최신 실제 재개 상태/agent/worktree: [복구 기록](../notes/RESTART_RECOVERY_20260916_1251.md). 미커밋 2차 코드 백업 후 3담당 구현·CPU검증·수정범위 독립검토 완료. A/B/C 모두 **준비용 부분구현 수용**, 실제환경·전체트랙 완료 아님.

정본: [resume](resume.md) · [상세 인수인계](../notes/RESTART_HANDOFF_20260916.md) · [재발 방지 §1n](quarantine.md). 과거 체크리스트는 [무손실 archive](archive/todo_before_20260916.md)로 이동했다. 아래만 현재 실행 목록이다.

## 최신 — 대기 중 1–4번 실행, 21:14 PDT

- [x] [4트랙계획](../notes/WAIT_WORK_PLAN_20260916.md)확정·4담당동시착수: 후속GPU/boundbudget, 실제패널입력/Gitauthority, 독립ONT장독평가, 재현·복구/crash.
- [x] Task4재현ledger/runbook/25capture검증완료(원인미확정nativecrash보존). Task1도구26시험+수정검토수용,실P1/mount/budget대기.
- [x] Task3orientationpilot검토/13filehash확인후 **6165555제출21:40PDT**(1ONT/10000sample/4CPU8GB30min); marker단독완료수용금지.
- [x] Task2배포검토→CPU6165628실행→실제50참조/산출물수치검증완료. 6165555실결과검증은기완료/externOOM운영게이트별도차단. GPU후속은P1PASS이전제출금지.
- [x] 공유referenceauthority/panel계약2→3identity대조및새runtimeledger연결. GPU고정b660불변,eval-only9603분리유지.

## 최신 usage 복구 — 18:03 PDT

- [x] [복구정본](../notes/USAGE_RECOVERY_20260916_1803.md): usage해결후GPU통합/decoder통합/P1배포3담당중단점에서재개. 원본부분코드보존.
- [x] 저장partialcode검증·b660독립통합·배포검토·3163986제출완료. 실제P1실측은아래GPU경로대기;재통합/중복제출금지. .venv/caches배포배제유지.

## 최우선 — EST8채널 학습 critical path

- [x] **HIGH실제MVPtrain/valid v3라벨연결수정** code981489d/reportd99b1f8, controller92focused시험/독립R1R2재검토수용. 실제MVP학습아직없음.
- [x] probe4ranklocal실패collectivecoordination+TRANSGENIC_RUN_DIR격리수정범위검토수용(d99b1f8).
- [x] 옛d99b배포지시대체:최종b660통합/resourcefix/배포검토완료,구3163545사용0취소→3163986제출. 옛담당/옛release재개금지. 실제6gate/학습은미완료.

- [x] 사용자 실제GPU→MVP학습/디코딩평가 연결지시, [우선실행정본](../notes/MVP_TRAINING_PRIORITY_20260916_1545.md). 실제probe·배포실측·평가producer3담당dispatch.
- [x] 단일최종소스 b660bf1(실행362eb899): GPU6mode+train/eval/decoder통합 독립검토수용. Agent599범위시험/controller46연결시험통과,4.57.6검사1skip. 전체repo nativecrash는미해결/전체green아님.
- [x] 최종b660배포패키지독립검토/해시검증, old3163545ownedpending취소·사용0확인, **finalP1 3163986제출20:21PDT**(4GPU30min). B5체인불변.
- [ ] 3163986실제S0/P1결과확인 → 나머지5gate/예산기록 →6gate readiness →3seed250updates. ESTphase완료를선행조건으로추가하지않기. 자동재제출금지.
- [ ] **로컬후속:** 구현완료finalizer/rawGFFproducer/decoder에새eval9603·panelmanifest소비계약/closure repin·정상/오염시험·독립검토. 독립junction/GFF→counts결속은잔여. 실제checkpoint후300paired+50baseline=350디코딩/평가.

## 0. 재개·작업 배분

- [x] 최신 MVP `f847cb8`/blastx`950cf4c` branch 존속, main`9a8c8e1` 미병합 확인.
- [x] 저자: 실제환경검증/학습·평가/bootstrap/blastx후속 전체를 서브에이전트로 진행 승인. [범위](../notes/b6_parallel_execution_authorization_20260916.md).
- [x] **1차 발견 워크플로 완료(09-16 오후 PDT):** A/B/C 담당 3명이 읽기전용 현황+설계 수집, 트랙별 독립검토 3건 전부 APPROVE_WITH_FIXES. 종합·must-fix·저자결정 8건은 [synthesis](../notes/b6_three_track_discovery_synthesis_20260916.md). Slurm제출/원격쓰기/worktree생성 없음.
- [x] **2차 구현 복구 dispatch:** 기존 A/B/C worktree의 미커밋 코드 백업 후 담당3명 동시 재개. agent/path는 복구기록 참조.
- [x] 이번준비용부분구현: 각담당CPU시험 → controller diff/재시험 → 독립검토/수정재검토종료. 각코드HEAD와한계는복구기록최상단.
- [ ] 후속로컬구현은새evalconsumer/~~ONTfullwrapper·실fixture~~(09-21 종료)/junction-counts/scoring계약/acceptedHSP·phase경계로한정. GPUprobe/Bdecoder·counts/C1pilotreceipt는구현완료로중복개발금지. 제출·epoch05·HSP/phase방향승인은반영완료;HiFi는산물정체성대조잔여,실제정책변경/fullproduction만별도.
- [ ] 작업별 격리 worktree·불변 snapshot·출력 디렉터리·잡 소유자/예산 기록. DeltaAI 제출자는 A 담당 하나, pronghorn B/C 출력경로 분리. 제출 직전 liveSlurm 확인, 중복금지.

## A. 실제 환경 검증 → launch-readiness → MVP 학습

- [x] Readiness/staging/bounded-wrapper준비도구 및A1/A2수정 `f6b171f`(보고서HEAD `ac8ac2d`); agent219범위시험/controller66도구시험통과, [독립재검토승인](../notes/recovery_20260916_1251/A.fix1_review.md). 당시준비범위이며후속b660에서실제probe구현완료;실제GPU실측은아직없음.

- [x] 09-16 10:41PDT DeltaAI접속/잡 읽기전용 확인: seed123 `3131537` RUNNINGgh0864h39m, bestepoch05 eval0.6394507885; repo_seeds8f049eb, 후속Dependency. [실측](../notes/restart_status_20260916.txt).
- [x] 12:51PDT live조회:3131537 RUNNINGgh0866:49:49,bestepoch05,후속Dependency. 원격변경 없음.
- [ ] 실제 제출 전 다시 live조회. 기존seed123 job/checkout/체인 보존;456/789는승인된MVP진단우선→456→789순서와자원검증유지.
- [x] Controller초기화선택:concreteepoch_05,세seed공통,mutablebest추종금지. [선택/pin기록](../notes/MVP_INITIALIZATION_SELECTION_20260916.json), 사용자선택진행위임에따름.
- [ ] 선택checkpoint/sidecar독립staging 및compute에서실제해시재검증. 선택기록만으로ready아님.
- [ ] 실제SIF 경로/바이트SHA/컨테이너 내Torch·Transformers·Accelerate·tokenizer/Hyena정체성 대조. 현재기록 NGC26.08/Torch2.14; 로컬CPU시험은Torch2.6으로 다름.
- [ ] 검토된 사전시험: 실제tokenizer/Hyena, 3tile tiers step0동일성, 트랙off/on·RC/N, 생성hook/KVcache parity, bf16/Inductor, single-node4rank/4visible nonMIGGPU/UUID, 메모리·시간 실측. 코드/Slurm스크립트 검토 후 bounded smoke만 제출.
- [ ] 모든 필수시험PASS와 자원예산·코드/데이터/체크포인트/SIF해시를 묶은 **실제launch-readiness** 작성·검증. 현재기록없음. placeholder/CPU시험만으로 인증하지 않기.
- [ ] readiness통과 뒤 고정250optimizerupdates/12warmup × MVP seeds1,2,3 실행. 기존B5체인과 자원충돌/중복제출 확인. 완료 marker/원자적산출물/해시/actualhardware 확인.

## B. 디코딩·독립 장독 평가·bootstrap

- [x] 준비용scoring/window/junction/bootstrap부분구현 및 수정1차 `2276fcb`; controller95tests+21subtests통과, 독립재검토B1–B3해소. [검토](../notes/recovery_20260916_1251/B.fix1_review.md). 전체B완료아님.
- [x] pair/finalizer/seed/code/init/ref정체성및rawGFFreference-count부분producer `64b6d5b`(reportc024445), controller142tests+21subtestsPASS, 독립R1–R5수정검토수용. 350출력계약유지.
- [x] Panelinput생성기 `0f9f777` 독립검토PASS. 신규11시험별도재검증,controllerarchive25PASS/2환경실패기록유지. Gitauthorityobject배포필수.
- [x] 실제GFFCompare0.12.6 합성16case/function-level진단결과 독립수용. 원job6165521은분석기경로오류로FAILED유지, 기존236파일/34map의읽기전용복구로기대치확인. [검토](../notes/mvp_execution_20260916/gffcompare/local_recovery/independent_review.md).
- [x] **실제referenceID연결오류수정:** 9603e20 수정/8508c4b재검토 및 새6165628 실행완료. 실제50key↔50rawID/좌표일치, GFF50genes52transcripts, duplicateambiguous0 독립검증수용. 원DB/훈련P1소스불변. [결과](../notes/power_recovery_20260917/panel_result/REVIEW.md). 배열내용후속검증은복구최상단참조.
- [ ] 실제50map/reference/array수용은완료. 잔여는evalconsumer배포/독립read증거결속·일반duplicate/coverage정책·최종실디코딩/판정. localrepin과remote실행게이트를분리.

- [ ] `B6_MVP_PILOT_v1.md` 승인본과 현행 scoring코드·데이터역할/패널50의 계약 대조; 구현·시험계획 독립검토.
- [ ] 기확보A-CCS23 6run의accession/실제산물CCS·FLNC/정렬출처대조,별도A-HiFi ERR13994458 기존산출물위치확인. 승인된ONT/PacBio역할·OR지원기준·결측STOP→INCONCLUSIVE를평가코드에연결. 새승인은실제dataset대체/STOP조건완화시만별도검토;단순PENDING가드는저자권위아님. 입력/채점분리유지.
- [ ] MVP체크포인트 산출 후 승인패널50 전부 디코딩. sequence/evidence on/off 동일조건·체크포인트/생성설정·출력해시 기록.
- [ ] reference-dependent F1와 독립장독 splice구조지지 구분 평가. 독립증거미확보를 음성결과로 간주하지 않기.
- [ ] 승인spec의 paired bootstrap/세seed집계/STOP→GO→INCONCLUSIVE 및 withholding조건 구현·검증·실행. primary=증거양성타일별 pureweightedCE off−on 동일가중평균, composite병기, δ0.005nats 불변. F1/구간/재표본 단위는 원문대로.
- [ ] 결과·한계·사용된고정입력·런타임/통계코드해시 보고. 실제결과 전 유효성/성능주장금지.

## C. blastx 전체 감사 → HSP 수용 → genomic phase 투영

- [x] 15:30PDT 사용자 네항목 ‘지금진행’ 요청. [새 실행정본](../notes/EST_EXECUTION_20260916_1530.md). 감사/OOM·실제HSPfixture·phase계약3담당dispatch. 검토후bounded실행진행; 과거일괄실행승인대기를이작업의중단이유로반복하지않음. A46학습투입승인과는분리.
- [ ] 잔여 HSP/phase: 수용단위/충돌처리방향승인·핵심모듈검증완료,실source연결·miniprot replay/전체A11실입력closure 미충족. 원본blastx재실행금지. 09-17 [입력inventory](../notes/power_recovery_20260917/phase_input_inventory.md)에9종raw보존/+/−후보및정확한차단기록; legacy121provenance로기수용C1재개방금지.

- [x] C1준비용auditor/CPUfixture및수정2차 `5c7a0aa`; controller119시험통과, 독립검토원5건+FASTA회귀해소. [재검토](../notes/recovery_20260916_1251/C.fix2_review.md). 실제raw감사/C2/C3완료아님.
- [x] Pilot 제출/log-layout원본receipt복원·독립검토 및19역사경로정합완료. 새reconciliation반영감사6165525PASS;원본/실패기록보존,blastx재제출없음.

- [x] 9종 양source DONE 관측. remaining8 `6164049_0–7` allocation/batch COMPLETED0:0 재확인. Sitalica6163851은 이전완료. **blastx재제출불필요.**
- [x] C1전체산출물/출처technicalaudit수용:6165525COMPLETED0:0,18units/324unitchecks/117treechecksPASS 및독립최종검토. [수용](../notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md). ExternOOM/C2phase/scientific/production승인은별도false.
- [x] 저자승인wholeDIAMOND HSP length≥30/MID(F제외) per-row수용·raw13실DNA/protein replay구현/검토/144fixture검증. rawHSP개수나HSP길이수용을accepted genomicphase로부르지않음;실stream/sourcebinder는최상단후속.
- [x] 정확한screenedquery/BAM/CIGAR→genomic좌표·codonordinal진단라이브러리 `5c46c75`, controller176시험·독립수정검토승인. [검토](../notes/est_execution_20260916_1530/geometry.fix1_review.md). 전체phase투영아님.
- [ ] HSPcodonevent와충돌/분자dedupprimitive는fca911e수용. 잔여=실source/HSP전체정체성결속·routeAphaseadapter·완전A11입력·집계/raster및실fixture. 동종/하위분류군배제·lineage한계·A/B증거단위불일치제거/독립일치보존.
- [ ] 검토된수용/투영잡을 별도namespace에 실행, 전종카운트/QC/provenance로 최종수용. 기존raw와실패기록 덮어쓰지 않기.

## 완료 — 반복 구현하지 않기

- [x] 사용자 metric결정: 기존학습 composite 유지, paired primary pureweightedCE+composite병기.
- [x] MVP data/model/dataset + metrics/init/runtime/finalizer/CLI/topologygate 누적 **f847cb8**. 컨트롤러153CPUtests PASS, 전체연결검토·최종재검토PASS/APPROVE. [검토](../notes/b6_mvp_integrated_review_20260915.md). GPU인증 아님.
- [x] fullDB6164118 build+QC+기술수용 감사완료. DBsha`ed2d0eef…`,overlap0,valid1442/1447. [수용](../notes/b6_mvp_technical_candidate_20260915.json)·[inventory](../notes/b6_mvp_candidate_inventories_20260915.json)·[receipts](../notes/b6_mvp_candidate_receipts_20260915.json) 해시/정규형/연결검증완료. 생산승인false.
- [x] 실제 RN목록 train7635/valid1447/test1388 정합, BAM/member/molecule disposition정합, finalDB독립재해시일치.

## 별도 미해결/승인 대기

- [x] ~~Slurm `.extern OUT_OF_MEMORY0:125` 원인 조사~~ **종료 — 저자결정 09-21 "그냥 무시한다".** 원인 미규명 그대로 종료하며 해결됐다고 기록하지 않는다. `.extern` 상태는 계속 receipt에 기록하되 **그것으로 판정하지 않는다.** support draft 미발송 종료. 게이트 출처가 사용자가 아니라 독립검토(`longread/independent_review.md:11-17`)였고 과학 spec엔 해당 조건이 없어 저자가 해제 가능.
- [ ] A46 전체 최종승인·동결, fullB6/C0/secondary정렬감사/phase채널본학습 범위는 별도. 탐색MVP승인과혼동금지.
- [ ] main통합/푸시: 미커밋protocol/panel/docs/graft설정보존, 적절한diff검토 뒤 결정. 이번세션미병합/미푸시.
- [ ] B5 seeds456/789: 최신저자결정으로완성·seed123품질확인후재개. 이전MVP진단후순차실행안대체. seed123기존체인보존.
- [ ] 논문/위키/spec업데이트, A37두arm보고, #61평가분모/AtRTD3/PacBio등 이전연구결정은 [과거todo](archive/todo_before_20260916.md)에서 최신protocol/issue와대조 후처리. 오래된체크박스만보고재실행금지.

## GPU와 무관한 추가 잔여

- [ ] NativeSIGSEGV 원argv/cwd/PYTHONPATH/native환경·orderedpredecessor/core 증거목록 정리. blindwhole-suite재실행금지;최소재현조건수용후bounded진단,비재현을fixed로보고하지않음.
- [ ] [과거전수감사](../notes/TODO_ARCHIVE_AUDIT_20260917.md)의R1–R12/Q1–Q10 및장기issue범위에서현재권위/반영상태확인: A37두arm/#61/A34Methods/AtRTD3·PacBio/B1참조동등성/wiki·spec/MAPQQC. issueclosure/원고게시미확인을미완료확정으로바꾸지않기.

## 운영 유의

- 이전 advisor는한도소진; 이번12:51호출도클라이언트버전오류로실패. 새로운자문없음. 독립검토를별도로진행하며 advisor결과를만들어내지않기.
- 이 파일은 현재작업만 유지한다. 과거실행기록SHA `2f25fe55…24c`는 archive에보존; 상세branch/테스트/실패수정이력은 handoff와controller보고서에있다.
