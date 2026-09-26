# 실제 B6 훈련 착수 준비: 2026-09-25

## HSP v3 첫 성공과 protein C0 A19 필터 (2026-09-26)

- **Sitalica HSP v3 6172692 COMPLETED**(36m, 0.45GB): orthodb raw 1,371,252 → projected 560,303, dropped(query codon truth) 17,767, length M+I+D 529,002 / M+I+D+F만 95,186, masked-X HSP 41,355; swissprot raw 290,009 → projected 123,872, dropped 1,213.
- **protein C0 6172693 실패:** `miniprot query 4555_0:000000: same-species protein (A19)`. miniprot 입력 단백질(faa md5 453cb32b…, OrthoDB 필터본)은 평가종(3702/4577/4081)만 뺐고 학습종 자신의 단백질은 남아 있다. 프로토콜은 "Same-species proteins are never used (A19 cross-species rule)"(WORKPLAN L32)이고 builder는 이를 fail-closed로 강제한다. `accept/` 단계도 A19를 적용하지 않았다.
- **조치(프로토콜 규칙 적용):** [a19_filter/](a19_filter/) — 종별 miniprot GFF에서 query가 그 종 또는 하위 taxon인 `##PAF` 블록을 제거한 사본 `evidence/protein_alignments_a19/<Sp>/`(DONE: 새 gff_md5, source_gff_md5, lineage sha, 제거 블록 수). 판정은 builder의 `B.A19`(`b6_hsp_project.is_same_species_or_descendant`, 동결 lineage)를 그대로 쓰고 미상 taxon은 중단. miniprot은 query별 독립 정렬이라 사후 제거는 해당 query를 넣지 않은 것과 같다. 필터 배열 **6172756**, protein C0 v4 **6172757–6172765**(`--align-root protein_alignments_a19`, 각 필터 task+HSP afterok). 이전 대기 protein C0(6172695·6172701·03·05·07·6172720·30·32)는 취소.

## C0 이후 runbook 요약 (에이전트 조사, 2026-09-26)

순서: A interval manifest 9종(`b6_leakage_intervals.py`, 지금 가능) → B 코퍼스 repair/freeze(`b6_build_corpus.py --measure-only` 후 `--apply`, `b6_validate_corpus.py`) → C zero list 27개(`b6_routea_agreement.py`, 종×{train_track,valid_track,eval_input}) → D 병합(`b6_merge_evidence_db.py`, DuckDB 한도 미설정이라 Slurm 메모리 넉넉히) → E observation cache(`b6_build_obs_cache.py --family train`) → F bindings JSON(`train_HyenaTransgenic_B6.py:563-572`). **A16 completeness NULL 정책은 파일럿 비차단**(`b6_bindings.py:24,432-434`). DeltaAI: 새 repo 스냅숏 필요(현재 e2a91b7=3511a5a), SIF transgenic-ngc-26.08, 컨테이너 내부 경로 기준.
**결정 필요:** (1) freeze가 interval manifest 절대경로를 저장해(`b6_corpus_repair.py:119`) pronghorn 경로가 DeltaAI에서 풀리지 않음 → A/B를 DeltaAI 최종 경로에서 만들지, 코드를 바꿀지. (2) `configs/b6_400m_win_v3.json:64` UNGATED → GATED 전환(메모리 벤치는 2GPU였고 CP2는 4rank). (3) D1 flank geometry "pending author confirmation"(plan v4.1 §11). 확인 필요: A-ONT2 dataset 키와 Athaliana ONT C0 정본 dataset 일치.

## Gmax·Vvinifera N=100 (결정 13, 2026-09-26)

- 감사 배열 **6172723**(49 task: Gmax EST1+ONT27, Vvinifera EST1+PacBio9+ONT11, %6, [b6_audit_align_n100_gv.sbatch](audit_n100/b6_audit_align_n100_gv.sbatch)) → C0 N=100([c0_n100.sbatch](audit_n100/c0_n100.sbatch)가 [c0_view_n100.py](audit_n100/c0_view_n100.py)로 뷰 생성 후 빌드, source_v3): Gmax EST **6172724**·ONT **6172725**, Vvinifera EST **6172726**·PacBio **6172727**·ONT **6172728** → HSP Gmax **6172729**·Vvinifera **6172731** → protein C0 **6172730**·**6172732**.
- 대체되어 취소: Gmax/Vvinifera ONT v2(6172710·11), HSP v3(6172698·6172708, 부분출력 `hsp_v3/<sp>.cancelled_n50`), protein(6172699·6172709). 이미 PASS한 N=50 Vvinifera EST·PacBio C0는 보존하되 병합에는 N=100본을 쓴다.

## 종 exhaustiveness 예측과 Bdistachyon N=100 (결정 12, 2026-09-26)

- 완료 C0 계수 합산 예측(병합 게이트 ≥0.999, 종 내 전 소스 동일 N): Athaliana 0.99991·Osativa 0.99936·Ppatens 0.99979·Ptrichocarpa 0.99972·Sbicolor 0.99959·Sitalica 0.99984 OK. **Bdistachyon 0.99897(EST 단독) 병합 거부 확실**, Gmax 0.99872·Vvinifera 0.99708은 ONT 결과 대기.
- 결정 12(N=100): [audit_n100/](audit_n100/) — `b6_audit_align.sbatch`와 차이는 `-N 100`, OUTROOT `audit_align_n100`, task 1(Bdistachyon EST)뿐. 감사 **6172717** → EST C0 N=100 **6172718**(`c0/Bdistachyon_est_n100`, `--audit-n 100`) → HSP **6172719** → protein C0 **6172720**(protein C0는 병합되는 EST C0 파일과 같은 sha여야 하므로 재연결). 이전 Bdistachyon HSP 6172696(부분출력은 `hsp_v3/Bdistachyon.cancelled_n50`로 이동)·protein 6172697 취소.
- 감시 주기 20분.

## 결정 9–11 구현·재제출 (2026-09-26 오전)

- **HSP 행 정책 `396fcbf`**(cherry-pick of 3b85e15): 행마다 M+I+D → M+I+D+F 순으로 길이 계약 확인, QC `length_contract_mid`/`length_contract_midf_only`. contract replay의 ValueError만 행 단위로 제외(`dropped_nonreplaying_hsps`, `dropped_nonreplaying:<class>`). A19·EST FASTA 누락·qlen/slen 불일치·sidecar 등 구조 오류는 그대로 중단.
- **A16 연결성분 `0fc80de`**(cherry-pick of ab793b4): `59_evidence_support.assign_molecules`가 비-clique 성분을 한 분자로 합침. unit id = sha256(key, 성분의 최소 (start,end)); 기존 clique 분자 id는 불변. QC `pcr_nonclique_components`/`pcr_nonclique_reads`, provenance `a16_pcr_linkage`.
- 병합(`b6_merge_evidence_db.provenance_gate`)은 코드가 아니라 intervals·seed·corpus/splits/flags/reference/gff만 대조하므로(`IDENTITY_INPUTS`), 이미 PASS한 EST·PacBio·direct RNA ONT C0는 재빌드하지 않는다. roles도 대조 대상이 아니라 protein 전용 roles 파일과 충돌하지 않는다.
- 소스 `source_v3`(tar sha `0dd676a4…0add`). pronghorn 컨테이너(pyarrow 포함) 관련 시험 **186 passed**([source_v3_deploy_tests.txt](source_v3_deploy_tests.txt)).
- 제출([v3_submission.txt](v3_submission.txt)): HSP v3 9종 6172692·94·96·98·6172700·02·04·06·08, protein C0 9종 6172693·95·97·99·6172701·03·05·07·09(각 HSP afterok), ONT C0 v2 Vvinifera 6172710·Gmax 6172711·Sitalica 6172712·Athaliana 6172713(48G/DuckDB36GB, `_ont_v2`). Osativa 6172409(구코드)는 계속 실행, 실패시 v2로 재제출.

## ONT cDNA C0 실패: A16 비추이적 PCR 동치 (2026-09-26)

- Vvinifera ONT 6172567(cDNA 11런)이 8h52m 후 `A16 molecule units: A16 non-transitive 10-nt PCR equivalence`로 실패(batch MaxRSS 24.1GB, 메모리 문제 아님).
- 규칙(`revision/scripts/59_evidence_support.py:478-531`): UMI absent cDNA read는 (library, contig, strand, intron chain)으로 묶고 양 끝 10nt 이내를 PCR 동치로 본다. 연결성분이 clique가 아니면 linkage를 고르지 않고 거부한다. 동결 프로토콜이 linkage를 정하지 않은 부분이며, 실데이터 깊이에서는 비-clique 성분이 사실상 필연이다.
- 영향: cDNA ONT가 있는 모든 종(Vvinifera11·Gmax27·Sitalica9·Athaliana34·Osativa2런). direct_RNA는 read=단위라 무관(Ppatens·Ptrichocarpa PASS).
- Gmax 6172411(12h36m)·Sitalica 6172410(15h34m)·Athaliana 6172589(8h32m)는 같은 실패가 확실해 취소했다. Osativa 6172409는 cDNA 2런뿐이라 계속 실행한다.
- **저자 결정 필요:** 비-clique 성분 처리 규칙.

## HSP v2 실패: 길이 계약·잔여 codon 불일치 (2026-09-26)

- 6172570(Sitalica, 코드6b6dcd3)이 46초 만에 `reported length disagrees with length contract: DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_GE30`로 종료. 같은 실패가 예상되어 HSP 6172571–78과 protein C0 6172579–87을 취소했다.
- masked X 코드로 Sitalica swissprot 290,009행 재진단([hsp_length_diag.py](hsp_length_diag.py)): ok 286,079(X 포함 13,015), **query codon truth 3,052**(X 포함2,932 / X 없음120), noncanonical protein 876. swissprot에서는 길이 실패가 없어 길이 실패는 orthodb 행에서 난 것이다. `b6_hsp_project`는 행 하나라도 실패하면 잡 전체를 중단하므로, 남은 불일치의 원인·처리 방침이 필요하다. 전량 진단 잡 **6172597**(swissprot+orthodb, 토큰 수준 원인) 실행 중.
- col0 retry 6172542 3/3 완료, FLIC 정렬·metadata 6172588 3/3 완료 → Athaliana ONT C0 6172589 의존 해제.

## 전면 병렬 제출: 2026-09-26 새벽

사용자 지시 "지금 전부 패러럴로 돌려줘". 의존성이 있는 단계는 Slurm `afterok`로 미리 걸어 전 단계가 끝나는 즉시 시작하게 했다.

- **ONT C0 재제출(메모리 축소):** 대기 중이던 Vvinifera·Ptrichocarpa·Ppatens(6172412–14)를 취소하고 48G/DuckDB36GB로 재제출 **6172567–6172569**. 근거: PacBio에서 peak RSS가 DuckDB 한도+약3GB였다. 노드당 할당 가능 메모리는 MemSpecLimit 24GB를 뺀 232GB다.
- **HSP 계약 수정 `6b6dcd3`**(masked X, opt-in; 파일별 전체시험 1089→1111 pass, 실패 ID 전후 동일). 소스 `source_masked_x`(tar sha `2fab3690…77bb`). **HSP 9종 동시 제출 6172570–6172578**(파일럿 없음, 32G, `hsp_v2/`, [hsp_project.sbatch](hsp_project.sbatch) 원격 sha `866f5201…1efe`).
- **protein C0 9종 6172579–6172587**(각 HSP afterok, 64G/DuckDB40GB, [protein_c0.sbatch](protein_c0.sbatch) sha `cad2ecd1…ce0f`). tool provenance: miniprot 0.18-r281, DIAMOND 2.1.24.178([protein_c0_provenance.py](protein_c0_provenance.py)).
- **Athaliana 체인:** FLIC metadata **6172588**(afterok 6172528), Athaliana ONT C0 **6172589**(afterok 6172588+6172542). C0 잡이 [athaliana_ont_view.py](athaliana_ont_view.py)로 뷰(cui2020 정본 이름, audit_view, col0 retry metadata, FLIC)를 만든 뒤 빌드한다.

## 결정 5–8 적용 시작: 2026-09-25 밤

결정 원문·범위는 [AUTHOR_DECISIONS_20260925_evening.md](AUTHOR_DECISIONS_20260925_evening.md) §5–8.

- **5 HSP masked X (a):** 계약·시험 변경을 writer 1명에 위임했다(b6-integration, 시험 선작성). 완료 후 소스 배포 → HSP 8종 재제출.
- **6 protein 역할행:** 기존 동결 roles는 무변경. 새 `inputs/DATASET_ROLES.longread_protein_v1.tsv`(sha256 `b190e4e8…454e`) = 기존 longread roles + `protein/<Sp>` 9행(run `miniprot`, role c2_training_eligible, genotype_stratum은 각 종 EST 역할행 값). protein C0에서만 `--roles`로 쓴다([protein_roles.txt](protein_roles.txt)).
- **7 cui2020 재사용:** Athaliana ONT 학습35런(cDNA34/direct_RNA1) manifest를 정본 dataset 기준으로 동결하고 cui2020은 물리 디렉터리 `cui2020_PRJNA594286`·제외 사본 reads 경로로 대응([prepare_ont_manifest_athaliana.py](prepare_ont_manifest_athaliana.py)). metadata 배열 **6172498**(0-34%4, [ont_metadata_athaliana.sbatch](ont_metadata_athaliana.sbatch) sha `5ec6b13f…38fe`). C0 뷰는 primary·audit 모두 정본 이름으로 symlink한다(audit도 뷰 root 사용).
- **8 FLIC 별도 드라이버:** production 스크립트를 복제하되 역할검사만 3런 검증전용으로 바꾼 드라이버 작성을 에이전트에 위임했다(제출은 검토 후 직접).
- Athaliana ONT C0는 FLIC 정렬·metadata 완료 후 제출한다.
- Athaliana ONT metadata 6172498: 32/35 완료, `training/ont/Athaliana/col0_DRP009401` 3런(DRR424731–33) 실패. 원인: dataset 경로에 PRJ* 없음(`b6_c0_metadata.py` longread_rows). ENA API 조회로 DRP009401 = **PRJDB14952** 확인(실험 DRX409928–30이 시트와 일치). 기존 `--bioproject` 옵션으로 새 경로 `metadata/ont_retry/`에 재처리 **6172542**. 실패 출력은 보존.
- FLIC 별도 드라이버 검토 후 제출: 방향성 감사 **6172527**, 정렬 **6172528**(aftercorr). production 대비 차이는 역할검사·런 목록·driver 표기·덮어쓰기 방지뿐이다([flic_align/](flic_align/)). 중앙 PROVENANCE.tsv scope는 `longread_validation_evidence`.

## ONT C0 제출: 2026-09-25 18:00 PDT

- ONT metadata 6172290 **90/90 COMPLETED 0:0**. 뷰 생성 시 각 run의 metadata sha·protocol·UMI(absent)를 다시 대조했다([ont_c0_views.txt](ont_c0_views.txt)).
- [ont_c0.sbatch](ont_c0.sbatch) sha256 `e18b742c…0670`, 16CPU/128G/DuckDB64GB: Osativa **6172409**(28런, 168.4M reads, 14d), Sitalica **6172410**(9런, 85.2M, 10d), Gmax **6172411**(27런, 80.8M, 10d), Vvinifera **6172412**(11런, 43.5M, 6d), Ptrichocarpa **6172413**(8런, 13.8M, 3d), Ppatens **6172414**(7런, 8.1M, 3d). 시간은 추정이며 PacBio 완료 후 처리량으로 재평가한다. GPFS 여유 25TB, 사용자 quota 없음.
- PacBio C0 3잡은 RUNNING, batch MaxRSS 약4.9–5.1GB(35분 시점). Vvinifera PacBio는 1h06m·MaxRSS24.4GB로 끝났다(약37k reads/min). 이 속도면 Osativa ONT 168M reads는 약3일이다. 메모리가 read 수에 비례하는지는 아직 모른다.

## HSP 투영 실패와 원인: 2026-09-25 17:40 PDT

- 파일럿 6172318(Sitalica): 분자맵 성공(66,040행, rejected34,317, sha `757d7781…`), `b6_hsp_project.py`가 `ValueError: query codon truth`로 종료(`est_hsp_contract.py:104`).
- **원인 확인([hsp_codon_diag.py](hsp_codon_diag.py)):** Sitalica swissprot raw13 290,009행 중 ok273,064, query codon truth7,944, subject truth8,123, noncanonical protein876. 실패 토큰은 모두 BTOP 문자가 `X`이고 실제 서열은 정상 잔기다(예: query `X` ↔ codon `CAC→H`, subject `X` ↔ 실제 `D`). production blastx(`commands.json`)는 DIAMOND 2.1.24 기본 masking(tantan)으로 실행됐고 `--masking` 지정이 없다. 승인된 계약은 BTOP를 실제 서열로 재생하며 잔기 일치를 요구한다.
- 의존잡 6172319–6172326은 DependencyNeverSatisfied라 취소했다. **해결 방식은 과학적 계약 변경이라 저자 결정 대기**: (a) BTOP `X`를 masking으로 인정하고 실제 codon/잔기 존재만 검증, (b) `--masking 0`으로 blastx 재실행, (c) 해당 행 제외(약5.5%, 저복잡도 편향).
- **Athaliana ONT 조사**: cui2020 두 사본은 gzip 시각 2바이트만 다르고 내용 동일(quarantine.md:1319와 일치). builder는 reads 경로를 대조하지 않아 기존 정렬을 정본 dataset 이름의 뷰로 재사용 가능하나, PROVENANCE·BAM header에 제외 사본 경로가 남는다(저자 승인 필요). FLIC 검증3런은 `longread_align.sbatch:70`·`orientation_audit.sbatch:50`이 비학습 런을 거부해 정렬 불가(코드 변경 또는 별도 드라이버 결정 필요). `audit_status=FAIL`은 §3.3 방향성 감사 결과(비가닥)이며 C0는 `uf=`만 읽는다(`b6_build_c0.py:677`).

## 저자 결정 반영과 후속 제출: 2026-09-25 17:00 PDT

결정 원문과 적용 범위는 [AUTHOR_DECISIONS_20260925_evening.md](AUTHOR_DECISIONS_20260925_evening.md).

- **EST C0 8종 PASS**(아래 표). Sbicolor·Gmax·Bdistachyon EST 단독 exhaustiveness <0.999는 병합 단계 위험으로 유지.
- **Vvinifera library:** `b6_c0_metadata.py --missing-library` 추가(b6-integration `d929ba2`, 시험 2건 선작성·실패 확인 후 구현, 로컬 21 pass(나머지 실패는 pyarrow 부재로 기준선과 동일), pronghorn 컨테이너 27/27 pass). 소스 `source_missing_library`(tar sha256 `cbdf2d5d…3725`). 재처리 6172286 COMPLETED: reads450,180, `NCBI-EST` 175 records, 출력 sha 일치. **Vvinifera EST C0 6172296 제출.**
- **ONT 메타데이터:** UMI absent 확정에 따라 6종 90런(cDNA49/direct_RNA41) manifest 동결([prepare_ont_manifest.py](prepare_ont_manifest.py), [ont_manifest.txt](ont_manifest.txt)), 배열 **6172290**(0-89%4, [ont_metadata.sbatch](ont_metadata.sbatch) sha256 `7a31e28f…a934`).
- **Athaliana ONT 보류(기술 문제):** 정본 행 `ont/Athaliana_cui2020_PRJNA594286` 3런의 정렬은 제외된 사본 이름 `cui2020_PRJNA594286` 아래에 있고 audit도 없다. FLIC 검증 3런은 정렬 자체가 없다. C0는 비제외 역할행을 모두 요구한다. 정렬 경로 대응과 검증런 정렬이 필요하다([ont_c0_readiness.txt](ont_c0_readiness.txt)). Slycopersicum·Zmays는 검증 전용이라 학습 C0 대상 아님.
- **route-B 입력:** lineage는 swissprot `lineage.tsv`(sha `402bca97…`)를 쓴다. orthodb 413 taxid가 모두 swissprot 2,428에 동일 행으로 포함됨을 확인했다. 분자맵→HSP 투영 [hsp_project.sbatch](hsp_project.sbatch)(sha `daef8ece…4a63`, 128G/24h): Sitalica 파일럿 **6172318**, 나머지 7종 6172319–6172325(afterok 파일럿), Vvinifera 6172326(afterok 파일럿+6172296). 기존 production blastx(6163851, 6164049)를 재사용하며 blastx 재실행 없음.
- **protein C0 차단(저자 결정 필요):** 동결 manifest `DATASET_ROLES.*`에 `protein/<Sp>` 행이 없어 `--source protein`이 거부된다. protein 도구 provenance 파일도 아직 없다.

## C0 결과 검증 누적 (원문 [c0_verify_log.txt](c0_verify_log.txt))

| 종·소스 | 잡 | status | reads_in | accepted | molecules | exhaustiveness | leak/gates | sha 재계산 | batch MaxRSS |
|---|---|---|---|---|---|---|---|---|---|
| Sitalica EST | 6172228 | PASS | 66,040 | 31,723 | 36,510 | 0.99984 | 0/0 | 일치 | 3.98G |
| Bdistachyon EST | 6172277 | PASS | 206,123 | 196,940 | 119,422 | **0.99897** | 0/0 | 일치 | 3.49G |
| Ptrichocarpa EST | 6172279 | PASS | 89,450 | 79,129 | 69,974 | 0.99913 | 0/0 | 일치 | 4.06G |
| Ppatens EST | 6172278 | PASS | 381,809 | 347,763 | 236,483 | 0.99946 | 0/0 | 일치 | – |
| Sbicolor EST | 6172280 | PASS | 209,441 | 195,394 | 158,271 | **0.99873** | 0/0 | 일치 | – |
| Athaliana EST | 6172274 | PASS | 1,046,825 | 891,439 | 918,316 | 0.99991 | 0/0* | 일치 | 3.31G |
| Osativa EST | 6172276 | PASS | 1,244,141 | 1,081,943 | 1,189,706 | 0.99936 | 0/0 | 일치 | 4.51G |
| Gmax EST | 6172275 | PASS | 1,374,461 | 1,273,169 | 1,033,891 | **0.99872** | 0/0 | 일치 | 7.57G |
| Vvinifera EST | 6172296 | PASS | 450,180 | 391,527 | 422,773 | **0.99421** | 0/0 | 일치 | – |
| Vvinifera PacBio | 6172281 | PASS | 2,456,610 | 2,030,546 | 2,456,610 | 0.99758 | 0/0 | 일치 | 24.42G (1h06m) |
| Ptrichocarpa PacBio | 6172282 | PASS | 9,847,562 | 7,218,072 | 9,847,562 | 0.99939 | 0/0 | 일치 | 66.57G (4h01m) |
| Sbicolor PacBio | 6172283 | PASS | 25,091,958 | 20,977,140 | 25,091,958 | 0.99960 | 0/0 | 일치 | – |
| Ppatens ONT | 6172569 | PASS | 8,125,618 | **158,810 (2.0%)** | 8,125,618 | 0.99980 | 0/0 | 일치 | 15.63G (1h47m) |
| Ptrichocarpa ONT | 6172568 | PASS | 13,752,619 | **388,829 (2.8%)** | 13,752,619 | 0.99995 | 0/0 | 일치 | 26.45G (3h03m) |

Ppatens ONT(전부 direct_RNA) 채택률 2.0%: rejected identity 5,891,310, unmapped 1,023,288, mapq 497,095, aligned_fraction 422,518. 학습 역할 ONT의 identity 하한 0.95(`59_evidence_support.py:733-735`, PacBio 0.98)가 동결 규칙이며 direct RNA의 전형적 identity(약85–92%)보다 높아 대부분 제외된다. 버그가 아니라 규칙의 결과다. 메모리는 DuckDB36GB 한도로 15.6GB였다.

메모리 관찰: Ptrichocarpa PacBio MaxRSS 66.6GB는 DuckDB 한도64GB+약2.6GB로, read 수 비례가 아니라 DuckDB 한도에 묶인 것으로 보인다(Vvinifera 2.5M reads는 24.4GB로 한도 미만). 확정은 아니며 ONT 대형 잡(128G)의 batch MaxRSS를 계속 본다. 처리량은 약41k reads/min.

\* Athaliana는 held-out 유전자좌가 있어 `gate_heldout_overlapping`=`gate_heldout_excluded`=104,170(겹친 분자를 모두 제외했다는 계수)이고 위반 지표 `gate_heldout_retained`=0이다.

- exhaustiveness 규칙: C0는 소스별 하한 0.99(A49.2)만 강제하고, 종 단위 0.999(rule 4(b))는 병합(`b6_merge_evidence_db.py`)에서 전 소스 합산으로 강제한다(`src/transgenic/evidence/leakage.py:90-97`). **Bdistachyon EST 단독 0.99897 <0.999**라 다른 소스가 합산값을 올리지 못하면 병합이 거부된다. 병합 전 위험으로 기록한다.

## 전체 C0 구축 제출: 2026-09-25 16:20 PDT

사용자 지시("커밋 하고 전체 진행해 수정 잡제출")에 따라 준비된 모든 EST·PacBio C0를 제출했다. CPU 증거DB 구축이며 GPU 훈련이 아니다. 런타임·입력 규칙은 검증된 Sitalica 6172228과 같다(source3511a5a, micromamba.sif+DuckDB1.5.5/PyArrow25.0.1, seed123, A22+A30 flags, current PREP roles).

- **입력 인벤토리** [all_species_inventory.txt](all_species_inventory.txt): 9종 모두 genome/GFF sha256이 B5 build_manifest와 일치, geenuff/swissprot flags·EST primary/audit 존재.
- **EST C0 7종** ([est_c0.sbatch](est_c0.sbatch) sha256 `801d3f04…98e4`, 뷰 [est_c0_views.txt](est_c0_views.txt)): Athaliana6172274·Gmax6172275·Osativa6172276(180G/24h/DuckDB48GB), Bdistachyon6172277·Ppatens6172278·Ptrichocarpa6172279·Sbicolor6172280(64G/12h/24GB), 8CPU. Gmax 메타데이터는 `metadata_est_clone_display_fix/Gmax`. 출력 `c0/<sp>_est_v1/out/`.
- **Vvinifera EST C0 보류:** library 누락175records 저자결정 전에는 메타데이터 없음.
- **PacBio C0 3종** ([pacbio_c0.sbatch](pacbio_c0.sbatch) sha256 `8f3868bb…34f0`, 뷰 [pacbio_c0_views.txt](pacbio_c0_views.txt)): training49런 전부 metadata/primary/audit 준비([pacbio_c0_readiness.txt](pacbio_c0_readiness.txt)). Vvinifera6172281(9런, 2.46M reads, 2d)·Ptrichocarpa6172282(6런, 9.85M, 4d)·Sbicolor6172283(34런, 25.09M, 7d), 16CPU/200G/DuckDB64GB. 시간은 Sitalica 처리량으로 추정했고 실측 전이다.
- **범위 밖:** Athaliana6·Zmays15 PacBio는 b1_validation_only이고 정렬·메타데이터가 없다. training C0에 넣지 않았다.
- **ONT C0 보류:** UMI 값 저자결정 대기(보고서 no_UMI5/unresolved120 vs TSV absent125). ONT 메타데이터 미생성.
- **protein C0(route-B) 대기:** 종별 EST C0 완료 후 `--hsp`(b6_hsp_project.py)·`--est-c0`·`--lineage`가 필요하다.

## Sitalica C0 DB 독립검증: 2026-09-25 15:53 PDT

- 읽기 전용 확인만 수행했다. 원문 출력은 [sitalica_c0_db_verify.txt](sitalica_c0_db_verify.txt)에 있다.
- **sha256 재계산 일치:** `02116ca6e92a628965b638a418dea3c52e6d0af83ef8c47114d386b7c9481064`(128,724,992 bytes), 로그·QC 값과 같다.
- **스키마 직접확인(컨테이너 DuckDB1.5.5, read_only):** 테이블19개. molecule_member66,040=reads_in, molecule_exclusion36,510=molecules, partial_chain31,723=accepted, excluded_junction_coordinate131=coordinates_4c, heldout_locus0=heldout_loci0, c0_qc68행.
- **extern OOM:** 6172228 batch MaxRSS3,983,096K/요청32G, extern MaxRSS1,480K. batch 단계 산출물 해시·행수는 정상이다. extern 표시의 근본원인은 미확정이며 추정으로 닫지 않는다.
- **`est_protein_hsp` 0행은 설계상 정상:** `--hsp`는 `--source protein`에만 필수(`revision/scripts/evidence/b6_build_c0.py:1884`)이고, route-B는 **protein C0**의 `est_protein_hsp`를 읽는다(`b6_routea_agreement.py:158`, 입력은 `b6_hsp_project.py`). 따라서 EST C0에서는 비어 있는 게 맞다. route-B 채널에는 종별 protein C0 구축이 별도로 필요하다.
- 호스트 python에서 duckdb_runtime 직접 import는 GLIBC_2.23 부재로 실패한다. DuckDB 확인은 반드시 micromamba.sif 컨테이너 안에서 한다.
- 제출·재제출·코드수정·커밋·푸시 없음.

## 예약 관측: 2026-09-25 15:30 PDT

- 최신 인계에 따라 pronghorn CPU **6172228**만 BatchMode=yes/ConnectTimeout=20, 전체 timeout150초로 읽기 전용 확인했다. 완료된 DeltaAI 잡·B5·GPU 운영학습은 조회하지 않았다.
- **15:30:29 PDT** squeue 활성 항목 없음. sacct job/batch **COMPLETED 0:0, elapsed00:03:06**, cpu-48. 그러나 **extern OUT_OF_MEMORY 0:125**이며 원인·산출물 영향은 미확정이다.
- 지정 로그 전체와 `c0/Sitalica_est_v1/out_arrow_retry/Sitalica.est.qc.json`을 확인했다. 로그 및 QC **status=PASS**, schema_version=c0-v1, reads_in66,040, accepted31,723, retained19,310, molecules36,510. exhaustiveness=0.9998356614626129(36,504/36,510), leakage_nonzero=0, 모든 gate_* 값0, tiles_verified19,536/19,536이다.
- 로그/QC의 output_sha256은 `02116ca6e92a628965b638a418dea3c52e6d0af83ef8c47114d386b7c9481064`로 일치한다. **DB 해시 재계산·스키마 직접검증은 미수행**. A16 completeness는 accepted31,723개 모두 NULL(coded0, policy_violation0)로 기록된 initial EST-only 임시정책과 구분한다. **생산자 QC 보고서 PASS 확인이며 전체 게이트·최종 training input·운영훈련 PASS가 아니다.** 후속에는 DB 직접검증과 extern OOM 원인·영향 확인이 남는다.
- 새 terminal 결과에 Advisor 검토1회 요청했으나 클라이언트 버전 오류(현재2.1.160, 요구2.1.251 이상)로 **검토 불가**. 자동 업데이트 없음. 동일 terminal 결과에 Advisor를 반복 호출하지 않는다.
- 제출·재제출·취소·파티션변경·코드수정·커밋·푸시 없음. graft map은15초 제한 종료됐으며 소스 탐색 없이 지정 기록과 원격 산출물만 확인했다. 아래 RUNNING은 이전 관측이다.

## 예약 관측: 2026-09-25 15:00 PDT

- 최신 인계의 대상 변경을 따라 pronghorn CPU array **6172167**만 읽기 전용 확인했다. BatchMode=yes/ConnectTimeout=20, 각 원격 조회 timeout150초. 완료된 DeltaAI 검증잡은 재조회하지 않았다.
- **15:00:24 PDT** squeue는 Invalid job id로 활성 항목 없음. sacct task0..48과 batch 모두 **COMPLETED 0:0**(2~115초), 그러나 **extern49개 모두 OUT_OF_MEMORY 0:125**다. extern 원인과 산출물 영향은 미확정이며 정상 종료로 뭉뚱그리지 않는다.
- 지정 로그49개 전체에 output/reads/sha256 JSON이 있고 오류 문구는 관측하지 않았다. METADATA_DONE49개 및 대응 nonempty c0_metadata.tsv49개 존재. marker는 `metadata producer succeeded; C0/corpus/cache still required`다. **출력 SHA256 재계산·내용 전수검증은 미수행**이므로 생산 종료·산출물 존재 확인으로 한정하고 전체 게이트 PASS를 선언하지 않는다.
- 새 terminal 결과에 Advisor 검토1회 요청했으나 클라이언트 버전 오류(현재2.1.160, 요구2.1.251 이상)로 **검토 불가**. 자동 업데이트 없음. 동일 결과의 반복 Advisor 요청 금지.
- 후속 범위에는 출력 해시/내용 검증과 extern OOM 원인·영향 확인이 남는다. 이번 예약은 확인·검토만 수행했으며 B5·GPU 운영학습 조회/변경, 제출·재제출·취소·코드수정·커밋·푸시 없음. 아래 진행 중 표기는 이전 관측이다.

## 최신 실행 상태: 2026-09-25 15:25 PDT

- ONT protocol 분류(cDNA83/direct_RNA42)는 저자 승인 기록 완료. UMI는 별도 조사 중이며 확정하지 않았다.
- **PacBio49런 metadata 전부 완료 및 SHA256 검증**, Gmax 제목비교 수정 후 **6172217 완료/1,374,461 reads/해시검증**. EST는 이제8종 완료다.
- Vvinifera는 실제 screened450,180 records 중 **175 records(175 accessions, 0.03887%)**가 library 미상이다. 원본 GenBank 예시에도 clone_lib가 없으며 임의 보충·제외하지 않았다.
- **현재 모니터 대상: pronghorn CPU6172228 (b6_si_est_c0a), RUNNING**. 실제Sitalica EST C0 첫빌드다. 앞선6172220은PyArrow누락으로중단됐고, 격리runtime에PyArrow25.0.1을추가한후실제Stage/Arrow/DuckDBsmoke를통과해 새출력경로로재시도했다. 기존실패QC는보존했다.
- source joblog `logs/b6_si_est_c0a_6172228.out`, QC `c0/Sitalica_est_v1/out_arrow_retry/Sitalica.est.qc.json`, DB `.../Sitalica.est.c0.db`. 최종QC PASS/스키마/누출 및exhaustiveness게이트확인전완료로취급하지않는다.
- UMI 근거 조사는agent a64e8066-db8d-46f에위임했으며 결과검토대기. GPU 실제훈련은아직미제출이다.

## 승인과 현재 상태

사용자가 운영학습 시작을 지시했다. 이는 fixture/synthetic 시험을 다시 돌리라는 지시가 아니라 실제 생물학 데이터의 B6 학습을 시작하라는 승인으로 기록한다. 다만 현재 실데이터 C0, B6 코퍼스·freeze, observation cache, training bindings가 아직 준비되지 않아 **GPU 훈련은 제출하지 않았다**. 검증된 시험 체크포인트나 B5 원본 DB를 임의 대체 입력으로 쓰지 않는다.

동결 계획상 순서: 실데이터 구축·검증 → **WP5 실제 4-GPU 파일럿 4,142 optimizer updates** → WP5 go 판정 → WP6 전체훈련. seed123, aug_seed20261037. 테스트의 2GPU 승인과 운영의 4GPU를 구분한다. 현재 CLI에서 B6/C2는 full budget이므로 파일럿만 실행할 운영 런처/명시적 정지 연결도 준비해야 한다.

## 실측 자원·자료

- DeltaAI 잔액1,022 GPU-h. 기존 B5 대기5잡은 각4GPU×11h, forward reserve220GPU-h. 취소·변경하지 않았다. B6 GPU 활성잡 없음.
- DeltaAI 기존 db에는 B5 a40 DB만 확인, evidence 디렉터리는 비어 있음. HDD는 지금까지 시험/아카이브와 코드다.
- pronghorn ROOT `/data/gpfs/assoc/pgl/data/Transgenic`. host의 python3 명령은 없으며 실제 CPU Python은 `/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python`(3.11.14).
- audit184 DONE. 원래187목록에서 없는3런 SRR12822922/SRR12822923/SRR12881185는 승인된 excluded fast5 중복런이다. 재정렬하지 않는다. 초기 probe의 header/열순서 오독 로그는 무효, `audit_final.txt`가 정정본이다.
- 원래 longread primary178 DONE, EST primary9 DONE. CAGE fastq88개 및 done형태marker88개 존재(전수 md5 재검증은 아직 아님).
- EST esummary11종 TSV/sidecar/provenance 존재. training9종 count=records 확인. Vvinifera with_clone_lib 카운터가 records보다 커 원본파일의 실제 필드 검사와 구분한다.
- 지정 root/depth 범위에서 실제 B6 C0/corpus/cache/bindings 산출물은 발견하지 못했다. 발견된 MVP·fixture 산출물을 실제 B6 입력으로 간주하지 않는다.

## 실행한 실제 입력 준비

공통 새 prefix:
`/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925`

원본 정렬/FASTA/B5는 읽기 전용으로 사용한다. source는3511a5a snapshot, readonly. source archive SHA256 `9ee8d99560814afa07169900d645469975cb13edfd49206c3232a65ef36783b2`.

### EST 메타데이터: CPU array6172149

9종, 1CPU/8GB/2h, `0-8%2`. 독립검토APPROVE, 기존 producer 단위시험14PASS. roles·esummary provenance/hash·A21screened FASTA md5 확인 후 새 출력만 생성한다. BioProject namespace는 이전 저자 채택값 NCBI_EST:txid<taxid>다.

- 완료7종: Athaliana1,046,825 / Bdistachyon206,123 / Osativa1,244,141 / Ppatens381,809 / Ptrichocarpa89,450 / Sbicolor209,441 / Sitalica66,040 reads.
- Gmax 실패: DN591027.1의 clone 표기. NCBI GenBank와 FASTA를 직접 조회해 동일 accessionversion에서 원본 `AGTT>Gm-r1030-2842`가 FASTA 제목에는 `_`로 표시됨을 확인했다. **비교용 표시만 정규화하고 원본clone/A11identity는 그대로 두는 수정10f2aae 완료**. 신규RED1건 재현, metadata/moleculemap25시험PASS, 독립검토APPROVE. Gmax만 새출력metadata_est_clone_display_fix/Gmax로재처리한6172217 COMPLETED0:0, 1,374,461reads, SHA256 `4cd56f6a730db0dc9c2361f1f24b3d4a01ae9b9df3c5aa83283794ca1057f29f`. DN591027.1 rawclone보존직접확인. 기존7종은재실행하지않았다.
- Vvinifera 실패: JZ984178.1의 /clone_lib 누락. NCBI 원본 GenBank에도 clone만 있고 clone_lib/BioSample 정보가 없다. 임의 library 생성·제외·검사완화는 하지 않았다. 실제 영향 범위 조사 및 처리 결정 필요.

출력 `metadata/est/<species>/c0_metadata.tsv`, 입력영수증 및 METADATA_DONE. 실패출력은 보존하며 재시도는 새 경로를 쓴다.

### PacBio 메타데이터: CPU array6172167

A49.1의 승인된 keyed library sheet(sha2568de4d583...)와 current roles, audit run list를 결합해 **49개 training-eligible 런**을 유일하게 결속했다. ONT protocol/UMI 열은 사용하지 않는다.

- 1CPU/8GB/2h, `0-48%2`. 독립검토APPROVE. 원래 primary PROVENANCE의 reads path/md5를 확인한 뒤 새 metadata만 생성한다.
- 15:13PDT 관측: 49/49 COMPLETED0:0. 모든METADATA_DONE과출력SHA256을본세션이대조해통과했다.
- 원격 `pacbio_metadata.sbatch`, tasks JSON sha256 `cf9ee1fdd5b95ecc0b4a8f8e229fb6046a01262eccb9b7b8d85287f3c98ff7dc`.
- 로그 `logs/b6_pb_meta_6172167_<task>.out`, 출력 `metadata/pacbio/<species>/<dataset>/<run>/c0_metadata.tsv`와 METADATA_DONE.

**이 두 array는 실제 입력 준비이며, 모델의 가중치를 학습하는 잡이 아니다.**

## 후속 및 차단 조건

1. PacBio49 출력·해시 검증.
2. Gmax 제목비교 수정·재처리·검증 완료. 기존7종 및실패출력은보존.
3. Vvinifera의 실제 사용 EST 중 library 누락 규모 확인과 출처 복구. 불가능하면 저자 결정이 필요하다.
4. **ONT protocol 분류는09-25 14:44PDT 저자 승인**: cDNA83/direct_RNA42, 대상sheet해시불변. [승인 기록](ONT_PROTOCOL_APPROVAL.md). **UMI absent 일괄판정은 승인에 포함하지 않으며 추가 확인 대상**이다. keyed library의 기존승인과도 구분한다.
5. 첫 real EST C0(Sitalica) 실행중6172228. 실제B5headerDuckDB1.5.5확인, 기존micromamba.sif(Python3.11/glibc2.41)+격리DuckDB1.5.5/PyArrow25.0.1사용. genome/GFF/split SHA가 실제B5build_manifest와모두일치. 새로운primary symlink view에검증metadata를결합했고원본정렬은수정하지않았다. source3511a5a, CPU4/32GB/4h, DuckDB8GB/4threads, seed123, A22+A30flags, currentPREP roles사용. 최초실행독립검토APPROVE, 런타임의존성수정후Stage smoke통과. initial EST-only C0의A16 completeness NULL은기록된임시정책이며최종training input완료를뜻하지않는다.
6. 전체 C0·A16후처리·병합/QC → B6코퍼스/freeze·cache → realbindings 검증 → 새 gated config와4GPU파일럿 런처/예산검토 → 제출. `--allow-ungated`를 운영 우회로 사용하지 않는다.

## 모니터링 범위 변경

완료된 DeltaAI3210585를 반복 확인할 필요가 없다. 현재 활성 대상은 **pronghorn CPU6172228**이다. PacBio6172167과Gmax6172217은완료·출력검증돼반복조회하지않는다. 접속은 `ssh pronghorn`만 사용한다(DeltaAI2-hop 아님). 30분 예약은 위 로그/metadata marker를 조회해 실패·완료의 새 상태만 보고한다. 기존GPU검증PASS가 실제훈련진행이라는 뜻은 아니다. 새 GPU훈련ID가 생기기 전까지 GPU훈련 미제출 상태를 유지한다.

Advisor/초기자료조회agent는 rate limit 또는 클라이언트 버전 오류로 사용할 수 없었고 본 세션이 직접 확인을 이어갔다. 코드·자료를 근거 없이 승인한 것으로 간주하지 않는다.
