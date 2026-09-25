# 실제 B6 훈련 착수 준비: 2026-09-25

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
