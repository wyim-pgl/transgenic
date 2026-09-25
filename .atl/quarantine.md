# quarantine.md — 격리소

> **전역 규칙(저자 결정 2026-09-25):** Slurm `.extern` 단계의 `OUT_OF_MEMORY 0:125` 표시는 모든 잡에서 무시한다. 판정은 job/batch 상태·ExitCode·산출물 QC·해시로 한다(batch OOM은 해당 없음). [결정 기록](../notes/b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md).

## §1o — 2026-09-21~23: issue #60 번복 · A46 승인 · WP1 구현에서 폐기한 경로

현재 상태는 [resume](resume.md) / [todo](todo.md) / [RESTART_HANDOFF_20260923](../notes/RESTART_HANDOFF_20260923.md)에 있다. 아래는 **다시 하지 않을 판단과 검증된 대체**다.

| 폐기한 경로/오해 | 검증된 대체·주의 |
|---|---|
| ENA fastq 파일명 `_subreads`로 산물을 판정했다. 같은 함정에 세 번 걸렸다: A-CCS23·M-CCS26, 학습 51런(issue #60), O. sativa 3런. | **리드 헤더**를 본다. `{movie}/{zmw}/ccs`면 CCS, `{movie}/{zmw}/{s}_{e}`면 subreads. ZMW당 리드 수와 리드 수·평균 길이 규모도 함께 본다. ENA는 NCBI 경유 PacBio 런에 일반명 `_subreads`를 붙인다. |
| ENA `instrument_model`을 그대로 믿었다(O. sativa 3런은 "Sequel II"였지만 실제로는 RS II). | movie 이름으로 교차 확인한다. RS II는 `mYYMMDD_…_c…_s1_p0`, Sequel은 `m54xxx`, Sequel II는 `m64xxx`, `U` 접미사는 업그레이드 기기다. |
| "51/51 CCS"를 "학습 가능 51"로 읽었다. | 라이브러리 유형(ENA `library_strategy`/`source`)을 확인한다. baijiu 2런은 WGS 게놈 HiFi였다. `/ccs`는 HiFi라는 뜻일 뿐 전사체라는 뜻이 아니다. |
| 말단 센서스(프라이머·poly(A))만 보고 M-CCS26을 "FLNC 서명"으로 판정했다. | **내부 신호도 본다.** 내부 Illumina P5/P7 코어와 내부 A/T≥20을 확인한다. M-CCS26은 리드의 57–98%가 내부 어댑터 키메라였다. |
| 리드 수·길이 센서스로 "FLNC 미처리 CCS"라고 판정했다(A47.1 전제). | 가공 단계는 말단 센서스로 판정한다. 원시 CCS는 양끝 프라이머와 poly(A) 방향 반반, lima FL은 프라이머 0과 3′ poly(A)만, FLNC는 둘 다 0이다. |
| 동결 파서(`59_evidence_support`)가 리드 하나 때문에 런 전체를 중단했다. | `EmptyExonError` 하위 클래스를 두고 MVP 접합표만 그 리드를 제외한 뒤 계수한다. 다른 호출자는 여전히 fail-closed다. |
| 롱리드 드라이버 하나로 ONT와 PacBio를 모두 정렬하려 했다. | `longread_align.sbatch`는 ONT 전용이다(`-k14`, 방향 감사 필수). PacBio는 `longread_align_pacbio.sbatch`(`splice:hq`, `-uf` 없음)로 한다. |
| 체인 메모리 설계에서 RC 때 donor↔acceptor 역할을 교환했다(v0/v1). | 트랙 순열 `[1,0,2,5,6,3,4,7]`은 **가닥만 바꾸고 역할은 유지**한다. 위치는 p → L−1−p로 옮긴다. gather는 왼쪽 패딩을 반영해 `(pad+p)//6`로 한다. |
| 기존 메모리에 0 값 체인 토큰을 붙였다. | 0 값 토큰도 softmax 몫을 가져가 B5 동작이 바뀐다. tanh 게이트(α=0)와 곱셈 마스크 `m = any(mask)`를 쓴다. |
| 규칙 하나를 추가하면서 rule 6만 고쳤다. | 트랙만 언급하는 조항(decision 1, A18, rules 7/8/9/11/13/16)도 모두 함께 고친다. 검토자가 CRITICAL로 지적했다. |
| 기존 DATASET_ROLES 파일에 `b6_input`/`b6_score` 열을 덧붙였다. | EST 정렬 DONE이 roles md5를 기록하고 MVP 드라이버가 열을 엄격히 검사해서 둘 다 깨진다. 같은 계열의 별도 파일(`DATASET_ROLES.b6_v1.tsv`)을 쓴다. |
| 분자 ID를 두 곳에서 만들었다(C0는 QNAME, route-B는 closure member uid). | **생산자는 하나**(`b6_est_molecule_map.py`)만 둔다. HSP↔EST C0 결합이 실패하면 전체를 실패시킨다. 병합 단계에서 독립 게이트로 한 번 더 검사한다. |
| writer 에이전트의 사실 주장을 그대로 인용했다("운영 blastx 출력 없음"). | 원격 `ls`로 직접 확인한다. 9종 × 2 DB 모두 raw13과 DONE이 있었다. |
| Kimi challenge 모드를 썼다. | `spawn E2BIG`(인자 하나가 128 KB를 넘음)로 실패한다. **ask 모드 + 발췌 파일만 읽게** 한다. 넓은 범위의 ask는 15분 응답 제한에 걸린다. |
| MAPQ<5 규칙을 minimap2 secondary에 적용할 때의 효과를 모른 채 두었다. | secondary의 MAPQ는 항상 0이다. 저자 판정(A48)으로 적용하되, 효과를 QC `excluded_b_cause_secondary_mapq`로 실측 보고한다. |
| sdx_dRNA fast5와 fastq 제출을 독립 런으로 취급했다. | 전체 리드 60-nt 접두 일치로 중복을 판정한다. direct RNA는 리드 하나를 분자 하나로 세므로 중복이 지지 수를 부풀린다. |

## §1n — 2026-09-16 재시작: MVP CPU 구현에서 해소한 실패·오해

현재 상태는 [resume](resume.md) / [todo](todo.md) / [최신 handoff](../notes/RESTART_HANDOFF_20260916.md). 아래는 **다시 시도하지 않을 경로와 검증된 대체**이다. 실제 GPU readiness, 장독평가, phase투영, `.extern` 원인 등 미해결 작업은 todo에 남긴다.

| 폐기한 경로/오해 | 검증된 대체·주의 |
|---|---|
| 현재 main에서 MVP소스를 찾아 미구현판정 / 옛 임시worktree재사용 | 누적정본 `f847cb8` branch `pi-agent-5b846a12-93fa-4c3`. 새격리worktree+그곳의graft. 일부옛worktree가남아도최신은아님 |
| 사용자 “전체서브에이전트로” 승인만으로 실제작업이시작됐다고보고 | 승인노트만작성, 새3stream workflow/agent는아직dispatch안함. `/advisor`는5/5한도실패; 결과/승인생성금지 |
| mutable `best`를MVP여러seed가각자따라감 | 동일concretecheckpoint/hash고정. epoch04는옛후보,09-16실측bestepoch05; 실제사용선택은새ready기록에결속 |
| 초기화출처를checkpoint/runtime.json에서만찾음 | 실제runroot/runtime_identity.txt도전후hash/identity추적. 기록SIFhash는실제이미지재검증과별개 |
| 임의dict/축약fixture를완전한weights_init감사로인정 | 실제loader반환값통합시험, snapshot/semantics/provenance/중첩exactkeys+aliasmetadata검증. 이것은schema검사이지문서진위증명아님 |
| pairedprimary를global토큰pooledCE로교체 | 각타일target-weight분모의pureCE계산후, evidencepositive타일 off−on 동일가중평균. 기존composite별도기록 |
| update1에peak/12·update250에0인shifted스케줄 | canonicalpost-stepLambdaLR동등계산. update1LR0,13peak,250peak/238,완료후0; exacttrace/AdamWstate시험 |
| toy모델DDP통과면실제unused파라미터도안전 | factory/runtime에서find_unused_parameters=True와실제wrapper검사. unusedtrainable포함2/4rank실증 |
| 준비된optimizer면그룹순서/파라미터가맞다고가정 | exactbackbone/projection이름·순서·identity·중복/누락/wd를변경전집단검증 |
| skippedstep표시만주고실제로optimizer를업데이트하는시험 | truenoopstep seam +가중치/state/LR진행불변·시도횟수검증. CPU시험은GPUoverflow인증아님 |
| 오류검사wrapper안에이미collective인함수/테스트barrier넣기 | localfallible준비→집단오류통지→집단연산 분리. prepublish callback단계도local만수행; rank비대칭실패시험 |
| 예상RN목록을상한검사전list변환 / isinstance로무한iterator재정의subclass허용 | exactbuiltinlist/tuple/range,길이/형식검사집단화후복사 |
| localJSON64MiB검사후무제한pickle objectgather | 실제총량/padding전송크기상한을payload전검사, uint8tensor전송. 오류status도제한,UTF8diagnostic2048B명시적truncation |
| 평가만eager로풀어compile실패를피함 | 기존prepared/compiledforward유지, unwrap은rank0저장에만. Accelerate0.34.2에없는keep_torch_compile인자금지 |
| exists체크+rename으로경합중덮어쓰기방지됐다고가정 | Linuxrenameat2NOREPLACE,같은FS staging/markerlast/fsync. rename후parentfsync실패는committedTrue/내구성미확인,완료본삭제금지 |
| 검증한canonical파일대신args의원래symlink를사용 / init감사와readiness를대조안함 | canonical소비·실제initweights/config해시대조·단계identity검사·평가후저장전재검사. 악의적inplace변경후복원까지막는불변저장소주장은하지않음 |
| knownsplit만조회해unknown/NULL추가행무시 / 임의pass.py를consumedcode로인정 | 정확한train/valid/test키/전체groupedcount검증,실행소스트리+entrypoint의정확한manifest |
| helper수동재조립을CLI종단간시험이라고부름 | 실제execute→_train호출,readinesshandoff/cleanup/metadata/RNG/epoch/update/final산출물검사 |
| CPUtestmetadata에bf16/Inductor를하드코딩 | 실제precision/backend기록,privatetestscope/testhardware명시. 공개CLI는GPU4/UUID/nativebf16집단게이트 |
| rank간seed차이만보며stream간동일seed를놓침 | canonicalaug20260914+seed고정; privateaug+rank/trainworkeraug+1000000+rank/validworkeraug+2000000+rank. 별도Generator및정확맵검증 |
| subprocess가부모PYTHONPATH의격리deps를그대로읽는다고가정 | 일부probe는checkout/.testdeps로재설정. controller아카이브에명시적link설치후동일소스통과; 초기missingaccelerate실패기록보존 |
| main의미커밋brief를격리agent가반드시읽을수있다고가정 | neutral/tmp로복사·해시대조·경로전달. 최초finalizeragent는이유로코드0줄중단후정상재개 |
| candidate기술수용과생산학습승인을같은bool로취급 | technicalcandidate / exploratorylaunchreadiness / scientificproduction승인 분리. 현재기술수용true·생산false·actualreadiness없음 |
| 153CPUtests/합성GPUfacts를실제SIF/Hyena/GPU인증으로확대 | 실제runtimeprobe별도. public는4visible nonMIGsingle-node만지원. 전체repo optional698pass24fail도전체통과로쓰지않음 |

독립검토지적과수정별실측은 `notes/b6_mvp_*_controller_20260915.md` 및 [전체검토](../notes/b6_mvp_integrated_review_20260915.md)에 보존돼 있다. 미설명외부환경실패는확정원인처럼쓰지않고 untriaged로남긴다.

---

## §1m — 2026-09-15 재시작 전 확정된 실패·오해와 대체 경로

현재 상태는 [resume.md](resume.md) / [상세 인수인계](../notes/RESTART_HANDOFF_20260915.md). 아래는 **기각·해소된 경로**이며 재실행하지 않는다. `.extern` 원인 및 검증 손실 정의처럼 미해결인 문제는 [todo.md](todo.md)에 남긴다.

| 격리한 것 | 왜 틀렸나 / 맞아 보였나 | 검증된 대체 |
|---|---|---|
| `.dmnd`에 `--seqidlist`로 동종 제외 | help에 옵션은 있으나 DIAMOND2.1.24 dmnd backend는 accession filter 미지원. top25 뒤 필터도 유효한 교차종 hit를 잃음 | frozen lineage로 **검색 전** 종별 FASTA/DB 필터링;950cf4c, 실제 fixture6163396 및 Sitalica 실데이터 통과 |
| `getseq --no-parse-seqids` | makedb/blastx에서는 유효하나 getseq workflow에는 허용되지 않음 |0715940에서 getseq에만 제거; 전체 ID 검증과 build/search 플래그 유지 |
| `diamond version`이 반드시2.1.24.178을 출력한다 | 실제 version은2.1.24, help banner는2.1.24.178. fixture6163393은 정렬 전 잘못된 guard로 실패 |950cf4c는 release와 help의 exact build 둘 다 검증;26tests+real fixture6163396 통과 |
| smoke6164069의 `PYTHONNOUSERSITE=1` | 환경 격리를 강화하려다 검증된 user-site DuckDB까지 숨김. builder 시작 전 import실패 | 해당 플래그 unset, 별도 pysam PYTHONPATH 유지;6164093 import/build/raster 통과 |
| smoke6164081 내부 `/usr/bin/time` | 호스트 실행 파일 존재를 컨테이너 내부에도 있다고 가정. hash+sample 뒤 builder launch 실패 | 내부는 sys.executable+stdlibresource/monotonic, 외부 host time 유지;6164093 통과 |
| 첫10,000 BAM sample을 전체 학습 증거로 취급 | 타일/분자 다른 배치를 놓칠 수 있고 지리적 편향으로 valid evidence0. overlap0은 샘플 범위만 의미 |sample NOT_FOR_TRAINING 보존, 전체 BAM builder6164118 별도 제출. full QC 결과는 아직 대기 |
| `ts:A`를 게놈 가닥으로 직접 사용 | reverse SAM에서 ts는 read-relative. FLAG16/ts+는 genome− |A39/scorer의 FLAG+ts 합성, MVP 설계·CPU구현/시험에 반영. 패널 dedup 키는 FLAG가닥 그대로 |
| evidence off면 모든 학습 시점에 원본seed123과 동일 | backbone도 fine-tune되므로 W·0=0은 증거 기여만 제거 |원본과 bitwise 동일성은 step0/shared weights/settings에서만; 이후 current sequence-only 경로와 동일 |
| 현재 main에서 새 MVP 코드를 찾고 “미구현” 판단 | 구현은 격리 branch에 있고 main은9a8c8e1. temporary worktree가 제거돼도 git branch는 존속 |MVP c7fa759, blastx950cf4c에서 git show/newworktree; main graph도 이 브랜치들을 자동 포함하지 않음 |
| sample 성공/단위시험을 realHyena·GPU·DDP 인증으로 확대 | 모델 시험은 tiny Hyena factories/CPU, 전체 suite 환경도 분리됨 |data86/model11/dataset+model20 범위를 각각 기록; production SIF/real tokenizer/GPU 게이트 별도 |
| 오래된 UniVec121nt provenance를 현재100nt 입력 정본으로 읽기 | A37에서 primary가100nt로 바뀌고 일부 원래 파일 문안은 남음 |종별 est_align/PROVENANCE의 min_len100·정제MD5·동일summary/roles와 실제 bytes 결속 |
| Python3.8 site-packages를3.12 앞에 두고pytest 공유 | 바이너리 NumPy가 다른 ABI로 먼저 잡힘 |네이티브 runtime+worktree-local deps 사용; 공유 환경 변경 없이 실제 테스트 |

기존 failed output/로그/lock은 진단용으로 보존한다. 성공 재시험은 새 directory를 썼으며, 실패본을 덮어써 성공처럼 만들지 않았다. payload/batch 성공과 `.extern OOM0:125`는 별개 관측이다. **원인이 해소됐다고 기록하지 않는다.**

---

## 이전 격리 기록

> 📌 **여기 있는 것은 전부 "끝난" 것이다.** 계획이 틀렸거나, 시도했는데 안 됐거나, 아예 배제된 것.
> **미해결 항목은 여기 두지 않는다** — 그건 [`todo.md`](todo.md)로 간다.
> 지금 유효한 상태는 [`resume.md`](resume.md)의 📌 블록. 최종 갱신 ~~2026-09-07~~ ~~2026-09-09 낮~~ ~~2026-09-09 저녁~~ ~~2026-09-10 오전 PDT~~ **2026-09-14 낮 PDT**(§1l 추가).

## 왜 지우지 않고 두는가

다음에 문제를 풀 때 쓰려고 둔다. 구체적으로 세 가지다.

| 용도 | 이 파일에서 찾을 것 |
|---|---|
| **False positive 분석** | 어떤 검사가 "정상"이라고 말했는데 실제로는 아니었나. 예: 분류기 플래그 "—"가 깨끗함이 아니라 **분류 불능**이었던 `PRJNA815899`(15런 전부 knockout), 대조군과 시험군이 **둘 다 0**을 내서 프로브 고장이었던 Swiss-Prot 조회 |
| **추가 피해 범위** | 이미 배제한 것이 다른 곳까지 오염시켰나. 예: subreads 51런을 격리했더니 그중 2런은 **WGS이기도** 했다(이중으로 틀림), Cui 2020이 두 경로로 중복 다운로드돼 있었다 |
| **위협 분석** | 같은 구멍이 다른 데도 있나. 예: `MODEL_RE`(기기)와 `MAX_READS`(리드 수)가 **둘 다 데이터 산물을 안 본다**는 사실은 PacBio에서 드러났지만 ONT에도 같은 모양으로 있었다(#63) |

즉 이 파일은 사후 기록이 아니라 **다음 실수를 막는 입력**이다. 항목마다 "무엇이 틀렸나"만이 아니라
**"왜 그때는 맞아 보였나"**를 적는다 — 그게 재발을 막는 부분이다.

## 구성

- **§1** `resume.md`에서 격리한 주장 — 무엇이 왜 틀렸고 무엇이 대체했는지
- **§1b** 증거 수집에서 배제된 것 — 두 필터 구멍과 각각이 실제로 통과시킨 것
- **§1c** A. thaliana ONT 스카우팅에서 배제된 것 — 약 460 GB와 그 근거
- **§1d** 기각된 Athaliana EST 오염 가설 — 지표 자체가 종 간 비교 불가였던 사례
- **§1g** ACCESS 준비 점검 — 저장소가 둘이라 생긴 착시, grep으로 문서를 판정해 생긴 오독, 검토자에게 파일을 덜 준 결함
- **§1h** 2026-09-09 새벽 — DeltaAI 발사·컴파일·다중 GPU에서 틀렸거나 막힌 것
- **§1i** 2026-09-09 낮 — Codex 리뷰가 잡은 결함 5건이 왜 이전 검토를 통과했나, per-layer compile·26.04 기각, 이미지 간 메모리 수치 비교 금지, 2 GPU 권고 대체
- **§1j** 2026-09-09 저녁 — A28 체인이 한 번도 작동한 적 없었다(결함 6건), dynamo가 SIGUSR1 핸들러를 막음, "101 GB = 풀 계정" 기각(정정의 정정), "노드 제외로 launch failure 해결" 기각, bash·Slurm 함정 4건
- **§1k** 2026-09-10 오전 — 25.06 권고는 저자가 26.08로 대체, 경계 시각 계산 오류(21:50 CDT → 21:45/22:00 PDT), 26.08 preflight의 stderr 오독, latest_state 교체 창, epoch 경계 accumulation 잔여분(두 런타임 모두; 시드 123 미수정), 시드 123 train loss 과소 보고, 교훈
- **§1l** 2026-09-14 — B6 설계 기각 3건(라벨 주입 v0·보조 헤드 v1·C1'), Codex 1차의 RC 스왑 오류, "학습 종 EST 재정렬 필요" 오보, "MVP를 A-ONT1/A-HiFi로 채점" 전제(정렬본 없음), 1.4 it/s의 근거 위치, Python `hash()` thinning, 스크립트 번호 충돌, diamond 빌드 잡파일
- **§2** 옮겨 온 이력 원문 (2026-08-04 ~ 2026-09-03 01:1x, 시간 역순)
- 저장소 문서·설계·코드 경로의 기각 → [`transgenic/quarantine.md`](transgenic/quarantine.md)

마커: `❌ SUPERSEDED` 더 이상 참이 아님 · `⚠️ CAUTION` 조건부로만 참 · `✏️ PARTIAL` 일부만 해소.
전수: `grep -n 'SUPERSEDED\|CAUTION (\|PARTIAL (' quarantine.md`

---

## 1. resume.md에서 격리한 주장 — 무엇이 틀렸나

§2 본문의 해당 위치에 같은 내용의 마커가 붙어 있다.

| 원래 주장 (resume.md) | 지금 | 근거 |
|---|---|---|
| 이슈 범위 **`#4–#54`** | **#4–#63**. #16·#50·#60·#62 종료 | GitHub |
| 커밋 **`b5e4471`**, 테스트 **145건** | tip **`a9e208b`**, 테스트 **155건**(+ torch 필요분 5건은 pgl-gpu) | `git log`, `pytest -q tests` |
| **"다음 작업: 1. #50 병합 …"** (4항목) | #50 **완료**(272,224행 동결) · #51은 ACCESS 자원 미확인으로 **차단** · #19–#20은 #51·#59 뒤 | 이슈 #50 종료 |
| **"다음 작업 (순서대로)"** (6항목) | 대체됨 — 현재 계획은 resume.md 📌 블록 | |
| pgl-gpu 디스크 **239 GB 여유** | 약 **216 GB** (동결 DB 23 GB) | `df -h` |
| **"장독 107런 111 GB (진행 중, 완료 표식 없음)"** | 두 가지가 틀렸다 — 드라이버는 진행 중이 아니라 **2026-09-02 23:29 S. italica에서 멈춰** 있었고(O. sativa·P. patens 미시작), 그 안의 **PacBio 51런 43 GB는 전부 subreads라 제외·격리** <br>✏️ PARTIAL (2026-09-04): **"S. italica에서 멈췄다"는 부분이 사실이 아니다.** 그 드라이버(`training_run_all.sh`)는 멈춘 적이 없다 — 로그인 노드에서 **2일 4시간** 살아 있었고, O. sativa `nip_pool_PRJNA953663`에서 **17시간 51분째 붙잡혀** 있었다. 받고 있던 것은 `SRR25241091`, **#63이 금지한 ONT WGS**이며 15.5 GB를 썼다(화이트리스트는 11:41에 들어갔으나 07:06에 시작된 프로세스엔 적용되지 않았다). 이 세션이 컴퓨트 노드에 있어 `ps`에 보이지 않았고, 자라는 `.part` 하나를 추적해서야 드러났다. 2026-09-04에 PID로 종료. **PacBio 51런 subreads 격리 부분은 그대로 참이다.** 경위는 위키 (32). | 이슈 #60, 위키 (32) |
| 프로토콜 **v1.19 → v1.25** | v1.25는 유효하나 §A13/§A14의 PacBio 조항이 실제 데이터와 어긋나 있었다 | 이슈 #60, `TRAINING_EVIDENCE_v1.md` ❌ 배너 |
| **"여기서 다시 시작"** × 3 (2026-08-10·08-11) | **여기서 시작하지 말 것.** 정본은 resume.md 📌 블록 | |
| **"Vitis 거부 6,613건 원인 미확인"** | 해소 — lncRNA 전용 유전자 6,205 + 마스킹 비율 330 + canonical order 78 | 이슈 #56 |
| **"maize EST 489만"** 등 | nuccore 전체 건수였다. 실제 EST는 maize 2,019,959 · Ath 1,529,700 … | `db=nuccore … gbdiv_est[PROP]` |
| **"Athaliana EST 매핑률 73.58 %는 이상 — 벡터·비Col-0 오염 가능성"** | ❌ **기각.** 오염이 아니라 **길이 구성 효과**. 길이를 통제하면 96.87 %로 9종 중 4위 | 아래 §1d |

⚠️ **이 표 자체가 같은 방식으로 낡는다.** 2026-09-03 추출 직후, resume.md의 📌 정본 블록은 커밋을
3개 뒤처진 `c6bc1b1`으로, 이슈 범위를 `#4–#62`로 적고 있었고 이 표도 tip을 한 커밋 뒤로 적었다 —
**추출은 머리 포인터를 갱신해 주지 않는다.** resume.md 📌 블록에 검증 명령 세 줄을 넣어 두었다.

**일반 규칙**: §2 안의 "현재 상태" · "다음 작업" · "진행 중" · "미커밋"은 **전부 그 스냅샷 시점의
서술**이다. §2에는 그런 블록이 121개 있고 개별 마커는 위 표의 것들에만 달았다 — 나머지는 이 문단이
일괄 격리한다. §2에서 무언가를 인용하려면 먼저 지금도 참인지 확인할 것.

## 1b. 증거 수집에서 제외된 것 (2026-09-03 결정·구현)

`evidence/longread_fetch.sh`의 필터는 **기기**(`MODEL_RE`)와 **리드 수**(`MAX_READS`)만 봤고,
그 둘 다 **데이터 산물**을 판별하지 못한다. 두 개의 구멍이 각각 실제로 뚫렸다.

| 제외 | 무엇이 통과했나 | 왜 필터가 못 막았나 | 지금 |
|---|---|---|---|
| ❌ ~~**PacBio subreads** (이슈 #60, 저자 결정: 제외)~~ ✏️ **2026-09-22 번복(A47.3): subreads가 아니라 CCS였다 — 49런 학습 복원, WGS 2런만 제외** | 학습용 51런 43 GB **전부** `_subreads.fastq.gz`, CCS/FLNC 0건 | `MODEL_RE`는 Sequel II/IIe를 맞게 통과시킨다(기기는 맞다). `MAX_READS=5000000`은 실제 subread 세트가 24.8만–157.8만 리드라 발동 안 함 | URL 이름에 `_subreads`가 있으면 거부. 43 GB는 `evidence/RETIRED_DO_NOT_USE/`로 격리(삭제 아님) |
| **비전사체 라이브러리** (이슈 #63) | `PRJNA953663`의 `SRR25241091` = **ONT WGS**(게놈 DNA). 같은 BioProject에 ChIP-Seq·WGS·RNA-Seq가 섞여 있다 | 접근번호 필터가 `OXFORD_NANOPORE`, 즉 **플랫폼**만 본다. `library_strategy`는 파싱만 하고 쓰지 않았다 | `STRAT_ALLOW='^(RNA-Seq\|FL-cDNA\|ncRNA-Seq\|OTHER)$'` 화이트리스트 |

**라이브 증거에 오염은 없다**(실측): 받아 둔 153런의 library_strategy는 RNA-Seq 108 · FL-cDNA 32 ·
OTHER 13 · **WGS 2**인데, WGS 2런은 둘 다 이미 #60으로 격리된 `Sbicolor/baijiu_PRJNA1034755` 안에 있다
— 그 데이터셋은 **subreads이자 WGS로 이중으로 틀렸다**. 화이트리스트는 나머지 151런을 전부 통과시키므로
소급 적용해도 재다운로드나 공백이 생기지 않는다.

⚠️ **남는 구멍**: `OTHER`를 허용해야 한다. ENA가 direct-RNA 런을 그렇게 라벨하기 때문이다(여기 13런이
전부 dRNA). 즉 `OTHER`는 여전히 아무거나 들어올 수 있는 통로이고, 그래서 채택된 strategy를 로그에
남긴다.

⚠️ **이 두 가드는 버전 관리되지 않는다.** `evidence/longread_fetch.sh`는 git 저장소 밖(pronghorn
파일시스템)에만 있다. 프로토콜 조항(§3, v1.3/v1.5)은 저장소 안에 있는데 **그 조항을 집행하는 코드는
밖에** 있다 — 저장소에는 `revision/scripts/evidence/univec_screen.sh` 선례가 있다.
(옮길지는 미해결이므로 [`todo.md`](todo.md) §3에 있다. 여기에는 **사실만** 둔다.)

---

---

## 1c. A. thaliana ONT 스카우팅에서 배제된 것 (2026-09-03)

ENA에 애기장대 ONT 전사체가 **617런 / 52 study** 있는데 우리는 6런만 갖고 있었다. 이유는 데이터 부족이
아니라 **대부분이 학습에 쓸 수 없는 종류**였기 때문이다. 배제 근거와 함께 남긴다.

| 배제 | 규모 | 왜 |
|---|---|---|
| `PRJDB38444` | 238.3 GB (최대) | FLEP-seq **nascent RNA** + 3′-cleavage 인자 변이체. 방법론 자체가 미성숙 전사체라 **WT 런도 못 쓴다** |
| `PRJNA1179505` | 122.5 GB (2위) | GEO GSE280623이 **2029년까지 비공개** — 122런 중 약 110런의 유전형을 대조할 방법이 없다 |
| `PRJNA788163` FLEP-seq2 | 50.9 GB | nascent (FLEP-seq) |
| RNA-processing 변이체 15+ study | — | m6A(mta/mtb/vir/fip37/hakai) · cleavage factor I · U1 snRNP knockdown 등. **잘못된 스플라이싱을 정답으로 가르친다** |
| `fastq_bytes=0` 13 study | — | fastq가 없는 게 아니라 **fast5/pod5 원시 신호**(`OXFORDNANOPORE_NATIVE`, 예 96.8 GB tar.gz). basecalling부터 해야 해 범위 밖 |
| 변이체 서브셋 (저자 결정) | 45.8 GB | `PRJDB38182`의 `atg5-1` 6런 · `PRJNA874115`의 spri2 계열 11런 · `PRJNA705300`의 CPSF30·FIP37 2런 · `PRJNA1393863`의 감염 7런 · `PRJNA796315`의 SSG 처리 6런 |

**합계 약 460 GB를 받지 않기로 했다.** 받아서 버리는 것보다 낫다.

### ⚠️ False positive — 이 스카우팅에서 내 검사가 틀린 방식

세 번 같은 모양이었다: **"조회한 필드에 없다"를 "존재하지 않는다"로 읽었다.**

| 사례 | 겉보기 | 실제 |
|---|---|---|
| `PRJNA815899` | 자동 분류 플래그 **"—"(깨끗)** | `sample_title`이 숫자 1–15뿐이라 **정규식이 걸릴 문자열이 없었다.** 실제로는 15런 전부 m6A writer knockout, WT 0 |
| `PRJNA874115` | 유전형 불명 | **BioSample `alias`/`isolate`에 처음부터 있었다.** `sample_title`만 봤다 |
| `fastq_bytes=0` | 데이터 없음 | **`submitted_ftp`에 fast5로 있었다.** 정규화 fastq만 없었다 |

→ **분류기 플래그가 안 붙은 것은 "안전"이 아니라 "분류 불능"이다.** 제목에 정보가 없는 study는
다른 필드를 열어야 한다.

### ❌ 내가 틀렸고 Codex가 반증한 것 — ENA taxon 필드

> 원문: *"ENA에는 617런 전부 A. thaliana로 등록돼 있는데 `PRJNA788163`의 sample_title은 rice/soybean/maize다.
> taxon 필드와 표본명이 모순이며, taxon만 믿으면 다른 종이 학습 증거로 들어온다."*

**틀렸다.** 617런 중 그 study의 런은 15개이고 **전부 애기장대 조직**이다. rice/soybean/maize 런은
`tax_id` 4530 / 3847 / 4577로 **정확히 등록돼 있고 애초에 목록에 들어오지 않았다.**
원인은 내 질의 실수 — **study 수준 질의**(`accession=PRJNA788163`, taxon 무관 전체 반환)의 앞 4줄을
taxon 필터 결과와 혼동했다. 같은 패턴을 `PRJNA705300`(Toxoplasma 혼재)·`PRJNA765684`(Eutrema·Camelina
혼재)·`PRJNA1393863`(Nicotiana 혼재)에서 재확인했고, **taxon 필터는 매번 정확히 애기장대만 골라냈다.**

→ **ENA taxon 필드는 신뢰할 수 있다. 나머지 8종도 같은 방식으로 조회하면 된다.**

### ⚠️ 위협 분석 — 우리 기준이 우리 참조에 적용되지 않았다

Codex가 AtRTD3 논문(Zhang et al. 2022 Genome Biol 23:149)의 심사 이력까지 읽은 결과,
**AtRTD3 자체 Iso-Seq에 `upf1`·`upf3`·`xrn2/3/4`(NMD·분해경로 변이체)와 핵분리 RNA가 포함**돼 있고
심사자 둘 다 "aberrant transcript 오염"을 우려했으나 저자들이 의도적 포함이라 답변했다.

**우리가 후보 데이터셋을 460 GB 배제한 바로 그 기준을, 우리가 검증 기준으로 삼는 참조 주석 자신이
통과하지 못한다.** 판단은 저자 몫이며 [`todo.md`](todo.md) §3에 있다.

---

## 1d. ❌ 기각 — "Athaliana EST 매핑률 73.58 %는 오염 때문" (2026-09-03)

> 원문(resume.md·todo.md): *"Athaliana 73.58 %는 다른 종보다 20 %p 이상 낮다. GenBank EST가 오래된 데다
> **벡터·비Col-0가 섞였을** 가능성이 있다 — B1 채점 전에 확인할 것."*

**틀렸다. 오염이 아니라 길이 구성 효과다.** 9종 전부 정렬이 끝난 뒤 길이를 통제해 보면 Athaliana는
이상치가 아니다.

| 종 | 전체 매핑률 | ≤120 bp 비율 | **>120 bp 매핑률** |
|---|---|---|---|
| Athaliana | 73.51 % | **50.6 %** | **96.87 %** (9종 중 4위) |
| Gmax | 94.01 % | 9.1 % | 98.91 % (1위) |
| Sitalica | 92.31 % | 0.8 % | **92.81 %** (9위) |
| 나머지 6종 | 95.5–98.5 % | 0.4–1.0 % | 96.2–98.7 % |

길이를 통제하면 9종이 **92.81–98.91 %** 안에 다 들어온다. ≤120 bp EST는 *어느 종에서나* 24.8–71.2 %로만
붙는다 — 즉 종별 총 매핑률 차이는 대부분 **EST 길이 구성 차이**다.

### 진짜 원인

GenBank EST 디비전의 Athaliana 항목 중 **51.7 %가 454 시대 짧은 라이브러리 두 개**다.

| 라이브러리 | 건수 | 평균 길이 | 매핑률 |
|---|---|---|---|
| `8-day Arabidopsis seedlings, aerial tissues` | 541,852 | 91 bp | 55.98 % |
| `Arabidopsis ovule high throughput cDNA library` | 249,438 | 92 bp | 41.61 % |
| **합계** | **791,290 (51.7 %)** | — | **51.45 %** |
| 그 외 Sanger 라이브러리 전부 (RAFL·CERES·Lambda-PRL2 …) | 738,410 | 380–590 bp | **97.15 %** |

Sanger 라이브러리는 개별로도 전부 **93–99.8 %**다. 0.517 × 51.45 % + 0.483 × 97.15 % = 73.5 %로 총계가
정확히 재현된다.

### 절단 의혹도 함께 기각

파일이 손상돼 서열이 잘린 것 아닌가를 먼저 의심했다 — 수집 로그에 `repair: refetched truncated batch`가
두 번 있었고, 레코드당 서열 줄 수가 **2줄(723,452건)에 몰린 이봉분포**였기 때문이다.
`ES211021.1`·`EL034561.1`·`EH803821.1`을 NCBI efetch로 직접 받아 대조한 결과 길이가 **92 / 104 / 99 bp로
정확히 일치**했다. 우리 사본은 원본 그대로이고, 이봉분포는 두 라이브러리의 존재 자체가 만든 것이다.

### 매핑된 454 리드는 품질이 나쁘지 않다

| 집단 | 매핑 수 | MAPQ = 0 | MAPQ ≥ 30 | 접합 보유 |
|---|---|---|---|---|
| 454형 (짧음) | 407,092 | 3.9 % | 93.8 % | 20.6 % |
| Sanger | 717,344 | 0.6 % | 98.9 % | 55.6 % |

다좌위 모호성이 문제가 될 줄 알았으나 MAPQ ≥ 30이 93.8 %다. **버릴 근거가 없다** — 다만 접합 기여가
작으면서 §3의 "독립 분자 ≥2" 카운트에는 Sanger와 동등하게 들어가므로, 가중치 문제는 저자 결정으로
[`todo.md`](todo.md) §3에 있다.

### ⚠️ 왜 그때는 맞아 보였나 — 이 파일이 기록해야 하는 부분

**세 종만 보고 종 간 비교를 했다.** 그 시점에 끝난 것은 Athaliana 73.58 · Bdistachyon 98.11 ·
Gmax 94.04뿐이었고, 셋 중 하나만 20 %p 낮으면 그 하나를 의심하는 게 자연스러워 보였다.
게다가 "GenBank EST는 오래됐다"는 사실 자체는 참이라 오염 가설에 그럴듯한 서사가 붙었다.

**놓친 것은 지표의 성질이다.** 종 단위 매핑률은 **분모의 구성이 종마다 다르므로 애초에 종 간 비교가
불가능한 수치**다. 낮은 값을 보고 "이 종의 데이터가 나쁘다"로 읽었지만, 실제로는 "이 종의 EST 코퍼스가
다른 종과 다른 것으로 이루어져 있다"였다.

→ **일반화**: 종·데이터셋을 가로질러 비율을 비교하기 전에 **분모가 같은 것으로 이루어져 있는지 먼저
확인할 것.** §1c의 "조회한 필드에 없다를 존재하지 않는다로 읽었다"와 같은 계열이다 — 그때는 *필드*를,
이번에는 *분모*를 확인하지 않았다.

⚠️ **추가 피해 범위**: 이 가설을 믿고 움직였다면 Athaliana EST 40만 건(매핑된 454 리드)을 오염으로
간주해 버렸을 것이다. B1은 Athaliana가 주력 종이므로 증거가 그만큼 줄었다.

---

---

## 1g. ACCESS 준비 점검 — 세 항목 모두 문서가 아니라 **내 확인 방식**이 문제였다 (2026-09-07 실측)

"바로 학습 가능하게 하라"는 지시를 받고 준비 상태를 **문서가 아니라 파일로** 확인하다 나온 것들이다.
처음에는 셋 다 "문서가 틀렸다"로 적었고, 실측을 더 하니 **셋 다 내 쪽 결함**이었다:

| 내가 처음 적은 것 | 실제 |
|---|---|
| todo.md가 "번들에 심볼릭 링크 없다"고 잘못 적었다 | **내 오독.** 그 문장은 게이트 스크립트 얘기고 참이다 |
| #59 구현이 저장소에 없다 | **내가 다른 저장소를 봤다.** `1413efd`로 커밋돼 있다 |
| Astra가 freeze 근거 부족을 지적했다 | **내가 검토용 파일을 덜 줬다.** 근거는 다 있었다 |

그래서 이 절의 값은 격리된 주장이 아니라 **판정 절차**에 있다 — grep 한 줄로 문서를 판정하지 말 것,
저장소를 먼저 찍을 것, 검토자에게 근거를 다 줄 것. 실제로 남은 미해결(전송 시 링크 실체화 방식)은
하나뿐이고 그것은 [`intent.md`](intent.md) R2에 있다.

### ✏️ PARTIAL (2026-09-07) — 번들의 심볼릭 링크: 내가 문서를 잘못 읽었다

**처음에 이렇게 적었다가 같은 날 정정한다.** `todo.md` 174행의 "(`training_input/` 쪽은 심볼릭 링크가
**아니다** — 번들이 ACCESS로 rsync되므로 밖을 가리키면 깨진다)"를 보고 "번들 전체에 링크가 없다는
주장"으로 읽었고, `find training_input -type l`이 **9개**를 내놓자 문서가 틀렸다고 결론 냈다.

**틀린 것은 문서가 아니라 나였다.** 앞뒤 15줄을 읽으면 그 항목은 **staging 게이트 스크립트의 위치**에
관한 것이다. 증거 드라이버는 "정본은 저장소 안, `evidence/<이름>`은 심볼릭 링크" 방식인데, 그 괄호는
**`training_input/` 쪽 스크립트는 그렇게 하지 않았다**는 대조다. 실측하면 그 진술은 **참이다** —
`training_input/make_manifest.sh`는 실파일이고, 링크 9개는 전부 `genomes/*.fa`다.

그리고 그 9개는 **번들 README가 의도된 것으로 명시**하고 있다: "`genomes/` — 학습 9종만.
`Transgenic/genomes/`로의 심볼릭 링크이며, 표에는 실경로와 md5가 들어간다." `MANIFEST.tsv`가 실경로와
md5를 담으므로 링크여도 내용 정체성은 보존된다.

**왜 그때는 맞아 보였나.** 괄호 안 문장이 **선행사 없이** 시작한다 — "`training_input/` 쪽은"이
가리키는 것이 스크립트인지 번들 전체인지가 그 문장만으로는 안 정해진다. 항목 제목이 "staging 게이트
강화 + 저장소 편입"이라는 것을 봐야 정해진다. **grep 한 줄로 문서를 판정하면 선행사를 잃는다.**

**그래서 남는 진짜 미해결은 무엇인가**: 문서 오류가 아니라 **전송 방식이 안 정해진 것**이다. 링크는
`/data/gpfs/assoc/pgl/data/Transgenic/genomes/`를 가리키고 ACCESS에는 그 경로가 없다. `rsync -L`로
실체화하든 번들 안에서 실체화하든 정해서 적어야 한다 → `intent.md` R2 / open question 3. 미해결이므로
격리소가 아니라 거기에 있다.

### ✏️ PARTIAL (2026-09-07) — "#59 미커밋": 낡은 줄 하나 + 내가 잘못된 저장소를 본 것

`todo.md` §4는 "#59 seedable sampler — 구현·검증 완료, **미커밋** (2026-09-05)"이라고 적고 있다.
**커밋돼 있다** — `1413efd` "#59: a resumed epoch was replaying epoch zero's permutation"
(2026-09-05 10:22:17 -0700), `train/train_HyenaTransgenic.py` + `tests/test_b5_resume.py`, 트리 깨끗.
그 줄은 커밋 **이전에** 쓰였고 이후 갱신되지 않은 낡은 줄이다. 규모가 작다.

**더 중요한 것은 내가 그것을 확인한 방식이다.** 나는 `gpu:/home/wyim/transgenic`에서
`grep -rn use_seedable_sampler train/ src/`를 돌려 **0건**을 받고 "구현이 아예 없다"고 판단했고,
노트가 인용한 base 커밋 `aa598f5`를 그 저장소에서 찾다 `not a valid object name`을 받고 "노트가
존재하지 않는 커밋을 인용한다"고까지 결론 냈다. **둘 다 틀렸다.** 리비전 저장소에서는
`aa598f5`가 정상 커밋이고(`#24: both chain readings...`), 구현도 거기 있다.

**저장소가 둘이고 겉보기로 구분되지 않는다.**

| 저장소 | 위치 | 역할 | 2026-09-07 HEAD |
|---|---|---|---|
| **리비전 저장소** | `pronghorn:~/scratch/data/Transgenic/transgenic` | **학습 코드·프로토콜·증거의 정본** | `5289077` |
| 모델 저장소 | `gpu:/home/wyim/transgenic` | 낡은 클론 | `e36f051` |

둘 다 `deploy/deltaai/train_b5.slurm`·`revision/`·`train/`을 가지고 있어서 **디렉터리 이름만으로는
어느 쪽인지 알 수 없다.** 🚨 **"코드가 없다"고 판정하기 전에 `git log --oneline -1`으로 어느
저장소인지 먼저 찍는다.** 이것이 이 항목이 기록할 값이다 — 낡은 todo 줄 하나보다 이쪽이 재발한다.

### ⚠️ CAUTION (2026-09-07) — Codex 검토 결과를 그대로 받으면 허위 블로커가 섞인다

A40 커밋 전 Astra(Codex)에 독립 검토를 걸었더니 블로커 5건이 왔고 **확인된 것은 1건**이었다.
기각된 3건(`freeze 클레임 검토 불가`·`window_genes 수치 근거 없음`·`272,224/26,057/6,205이 감사
JSON에 없음`)은 전부 **내가 검토용 디렉터리에 파일 부분집합만 넣었기 때문**이다 —
`b5_full_a40_v1.freeze.json`·`ledger.json`·`population-v2.json`을 빼놓고 "근거가 없다"는 답을 받았다.

**교훈**: 외부 검토자에게 "근거가 충분한가"를 물으려면 **근거 파일을 다 줘야 한다.** 안 그러면
돌아오는 것은 코드 판단이 아니라 **내가 만든 결핍의 반사**다. 확인된 1건(테스트가 가드가 아니라
주입된 validation 때문에 통과하던 것)은 파일 부분집합과 무관한 코드 자체의 결함이었고, 그것만이
살아남았다는 점이 구분선이다.


---

## 1h. 2026-09-09 격리 — DeltaAI 발사·컴파일·다중 GPU에서 틀렸거나 막힌 것

### ❌ 기각 — "torch.compile로 시드를 빠르게" (현 레시피에서)
원인은 **HF gradient checkpointing + torch.compile**(AOTAutograd backward 파티셔너 `Node add_316 was invalid, but is output`). 이분법(잡 3117481–84·3117505·3117585): 디코더 compile ❌ · 인코더만 ✅ · 전체+ckpt OFF ✅ · mask device 수정 무관 · **non-reentrant ckpt ❌** · **torch 2.10(NGC 25.11) ❌**. ckpt OFF면 30 kb 1.41×(17.7 vs 24.9 s/step)·61 kb 1.31×·2 GPU 2.4×까지 나오지만 **129 kb가 ckpt 없이 OOM**(3117556, A35 fail-closed sample 416, `Athaliana:Chr4:15869550-15998574-rc`, allocated 100.65 GB; ckpt ON 벤치는 완주 101.0 GB). 인코더만 compile은 **26.6 s/step로 오히려 느림**(3117589). → checkpointing은 레시피 필수, 전체 compile 불가, 인코더-only 무익. **시드 123 재시작 근거 없음.** 다시 열려면 A27 레이블 캡(8,192) 축소 = 레시피 변경 = 저자.

### ❌ 기각 — NGC 25.11 이미지
torch 2.10.0a0·CUDA 13.0·triton 3.5. import·preflight는 양쪽 통과했으나 **단언 동일**(위). GB10에선 `unspecified launch failure`(sm_121, 별개). **이미지 25.06 유지.** SIF는 디스크에 보존.

### ✏️ 정정 — "25.11이 GH200에서 죽는다"
그 판정(3116740)은 **노드 gh062**에서 났고 gh062는 25.06 잡(3117507)도 같은 오류로 죽였다. gh109 재시험(3117585)은 정상 실행. `unspecified launch failure` 3건 = gh062 ×2 · gh093 ×1 · 다른 노드 0. **노드 문제로 본다.** 실험은 `--exclude=gh062,gh093`; 시드 링크가 그 오류로 죽으면 코드 아닌 노드. NCSA 보고는 저자 승인.

### ❌ 기각 — "4 GPU로 그냥 돌리면 4배" (코드 그대로)
`accumulation_steps = 96 // batch_size`에 world-size 나눔이 없어 N GPU면 **유효 배치 96×N** = 동결 레시피 위반(첫 4-GPU 벤치 3116413/15에서 발견; 사용자의 "1 GPU면 안 되나?"가 드러냄). 수정 `2021fd2`(per-rank 96/N, 2 GPU에서 48 확인). 추가로 **reentrant ckpt + DDP = `Expected to mark a variable ready only once`**(3117555) → **non-reentrant ckpt**가 DDP 조건(3117590 성공, 17.57 s/step). 4-GPU 스케일링은 **미측정**(2-GPU 71 %).

### ⚠️ CAUTION — `PREFLIGHT PASS`가 재지 않던 것
토크나이즈 검사는 존재하지 않는 클래스를 import해 **한 번도 실행된 적 없었고**(`except: pass`), 인코더 캐시·런타임 pip·bitsandbytes·triton libcuda·compile 폴백은 preflight 범위 밖이었다. 전부 **GB10에서 같은 SIF·같은 잡 명령을 실행**해서만 잡혔다(`63c8ee4`…`a4d1ede`). "PASS" 줄은 무엇을 실제로 재는지 읽을 것.

### ⚠️ 내 실수 (재발 방지)
- **`pgrep -f <패턴>`이 내 원격 셸을 매치해 kill** — 세 번째. 원격 프로세스는 기록된 PID로만.
- 긴 원격 잡 출력을 ssh 세션에 물림(`| tail | tee`) → 소켓/timeout이 끊기며 **52분치 출력 소실**(SIGPIPE). detached 스크립트 → 파일 → 폴링.
- 원격 명령 전체를 `&`로 백그라운드 → heredoc `cat`의 stdin이 `/dev/null` → **0바이트 스크립트**.
- `repo_dev/HFmodels`를 **심볼릭 링크**로 → 컨테이너 안에서 호스트 경로가 안 보여 4런 20초 사망 → 복사본.
- 함수 안 `import torch._dynamo` → `torch`가 지역변수 → `UnboundLocalError`(`e962864` 회귀, `d7c3a6d` 수정). 스모크가 24초에 잡음.
- "Athaliana만 확실히 사망"(평지였음) · "에폭 7~8 h"(티어 하나를 에폭으로 오독; 실측 ~20 h) · apptainer arm64 `.deb` URL(존재 안 함, apptainer#3712).

## 1i. 2026-09-09 낮 격리 — Codex 리뷰가 잡은 결함 5건, 기각된 compile 우회 2건, 이미지 간 메모리 수치, 2 GPU 권고

### ❌ 기각 — "어젯밤 15커밋(4ef3dc4..2021fd2)은 검토됐다"
`/codex:review`는 **작업 트리만** 본다. 로컬 클론이 83커밋 뒤처져 있어 그 리뷰는 `.atl/` 파일 2개를 보고 "결함 없음"을 냈다 — **아무것도 검토하지 않은 것**이다. `adversarial-review --base 4ef3dc4 --scope branch`로 다시 돌리자 high 5·medium 1이 나왔고 전부 코드로 확인됐다(`5bc4df8`).
**왜 그때는 맞아 보였나.** 각 결함은 그것을 볼 수 없는 시험만 통과했다:
1. `train_b5.slurm` wait 루프 — `if wait; then …; fi` 다음 줄의 `$?`는 **if 문의 종료 코드(0)**다. 자식 exit 17 → rc 0 재현. 스모크·벤치는 실패한 적이 없어 이 분기가 실행된 적이 없었다. 시드 123의 동결 스크립트에도 있었다(한 파일 패치로 다음 링크부터 적용).
2. reentrant ckpt 기본값 + DDP `find_unused_parameters` — 3117555에서 이미 죽었지만 "스위치로 non-reentrant를 켜면 된다"로 끝냈다. 런처는 스위치를 켜지 않는다.
3. 모든 rank가 `_save_latest_b5`의 rmtree/makedirs/rename을 실행 — **50-step 벤치는 save 200에 닿지 않는다.** "2 GPU 경로 확보"는 저장 없는 경로였다.
4. 검증 손실이 rank별 로컬 합/로컬 길이 — 1 GPU에서는 정의상 맞다. 다중 GPU 검증은 실행된 적이 없다.
5. `protein_accept.py`가 수용 시점에 블록·인트론을 쓰고 mapping_ambiguous는 사후 판정 — 파일럿(100,851 정렬)의 "브루트포스 대조 불일치 0"은 **같은 누출을 양쪽이 공유**한 대조였다. 수치: 종별 1.4~4.1 % 단백질, 수용 정렬의 2.4~6.0 %, 커버리지 −2~−7 %, ≥2 organism 인트론 −1.3~−4.1 %. v1 산출물은 `accept_v1_ambiguous_leak/`, 표는 `protein_acceptance_v2.tsv`(A43).
교훈: **"통과"는 어떤 경로를 지났는지와 함께 적는다.** 수정안 자체도 리뷰 2회를 더 거쳐 rank별 stop 플래그(USR1이 rank마다 다른 시점에 오면 한 rank만 저장·종료)·미합산 train loss 분기·DDP KeyboardInterrupt의 collective 저장이 추가로 나왔다.

### ❌ 기각 — Codex 1b "per-layer compile(checkpoint 바깥, compile 안쪽)이면 25.06에서도 된다"
probe(`deploy/deltaai/probe_compile.py`, PROBE_MODE=layers, 30 kb, 할당 3118807 gh072): **같은 단언** `Node add_316 was invalid, but is output`. dropout RNG 기능화 버그(pytorch#135459)는 레이어 그래프 하나에서도 난다. 가설은 실측 전 "reasoned hypothesis"로 표시돼 있었다 — 그 표시가 맞았다.

### ❌ 기각 — NGC 26.04 (torch 2.12.0a0+0291f960b6)
같은 단언. 유지보수자의 "nightly ≥ 2.12.0a0에서 재현 안 됨"은 우리 변형(add 노드)을 덮지 못한다. **26.08(torch 2.14.0a0)에서만 통과.** SIF는 둘 다 `containers/`에 보존.

### ⚠️ CAUTION — "`max_memory_allocated`로 이미지 간 peak를 비교한다"
> ❌ **SUPERSEDED (2026-09-09 저녁, §1j):** 이 CAUTION과 아래 PARTIAL의 결론("이 카운터는 절대값으로도 믿을 수 없다, 풀 고수위 의심")은 뒤집혔다 — torch 101 GB가 실제 순간 peak이고 nvidia-smi 2 s 샘플이 하한이다. 마커는 남긴다(규칙 7).

25.06 eager 101 GB vs 26.08 **eager도 10.9 GB**, compile 11.3/24.4 GB. compile 효과가 아니다. Codex rescue(codex-memcheck): 이 카운터는 allocator backend와 무관하게 같은 훅이므로 torch 2.8→2.14 계측 변화이거나 실제 차이 — **드라이버 수준(nvidia-smi)만이 판정**한다. `7f20816`부터 벤치가 호스트 nvidia-smi를 2초마다 샘플링해 per-GPU 최대를 찍는다. 그 전 수치(101 GB 포함)는 같은 이미지 안에서만 비교할 것.

### ❌ SUPERSEDED — "round 6 통과 시 확인 시드 456/789에 2 GPU DDP 권고 확정"
round 6(`3117671`)은 통과했다(55.64 s/step, 1.74×, peak 100.5 GB). 그러나 (1) 저자가 **4 GPU**로 결정했고(2026-09-09), (2) 위 결함 3·4가 저장·검증 경로에 있어 벤치 통과가 인증이 아니었다. 인증은 `ddp_integration.slurm`(3118861·3118880)의 `DDP_INTEGRATION PASS`다.

### ⚠️ CAUTION — `salloc --no-shell` 할당은 `scancel` 뒤 `squeue`로 확인
오케스트레이터가 `scancel 3118849`를 찍고 "released"를 기록했지만 잡은 **17분 더 RUNNING**이었다(≈0.3 GPU-h 유휴). 손으로 다시 취소해 CANCELLED 확인. 에이전트용 인터랙티브 패턴(`salloc --no-shell` → `srun --jobid --overlap` → detached 스크립트 → 파일)은 유효하다; 반납 확인만 빠져 있었다.

✏️ **PARTIAL (2026-09-09 14:0x)**: 50 step `3118903`(26.08 compile 30 kb)은 torch peak **101.04 GB**를 보고했다 — 3 step probe의 11 GB는 재현되지 않았고, 25.06 eager와 같은 값이다. 같은 런의 호스트 nvidia-smi 최대는 **31.5 GB**. 즉 이 카운터는 이미지 간뿐 아니라 **절대값으로도** 믿을 수 없다(풀 고수위 의심). 판정은 `3118909`(25.06 eager + nvidia-smi). 그때까지 101 GB를 "실사용"으로 인용하지 말 것; 129 kb no-ckpt OOM(100.65 GB allocated)은 실제로 죽었으므로 그 사실만 유효.

### 🧊 3 step probe 수치는 사이징에 쓰지 않는다
26.08 compile 30 kb 45.2 s/step · 129 kb 110.0 s/step · eager 30 kb 29.5 s/step은 warmup 폐기 없는 3 step 값(compile 시간 포함). 정상 상태는 벤치 매트릭스(50 step, warmup 20 폐기)로.

## 1j. 2026-09-09 저녁 격리 — A28 체인 결함 6건, dynamo와 SIGUSR1, 메모리 해석 정정의 정정, launch failure의 정체

> 이 절의 정정은 §1i의 정정을 뒤집는 것이 둘 있다. 앞선 마커를 지우지 않고 잇는다(규칙 7) — 두 번 뒤집힌 이유가 세 번째 오진을 막는다.

### ❌ 기각 — "A28 48 h 체인은 작동한다(`af6cce8`의 wait 루프 수정으로 실제로 이어짐)"
**한 번도 작동한 적이 없었다.** 동결 트레이너로 15분 리허설(`--time 15 min`, `--signal=B:USR1@600`, SEED=999)을 돌리기 전까지 아무도 48 h 경계를 지난 적이 없다. 결함 6건이 겹쳐 있었고, 하나를 고칠 때마다 다음 것이 드러났다:
1. `if wait "$PID"; then …; fi; TRAIN_RC=$?` — if 문의 종료 코드 0을 기록(`5bc4df8`). 그 전에는 **어떤 죽음도 rc 0**이라 마지막 latest_state에서 조용히 재제출됐을 것(최대 200 step 손실, 매 링크).
2. 트랩이 `pkill -USR1 -P $TRAIN_PID; kill -USR1 $TRAIN_PID` — apptainer와 그 자식(`bash -lc`)에 USR1 → **컨테이너가 rc 138로 죽고 트레이너는 시그널을 본 적이 없다**(`3119780`). rank에만 보내도록(`4fe9780`) 고쳤더니 rank 탐지가 **런처(`accelerate launch … train/train_HyenaTransgenic.py …`, argv에 스크립트 경로)**까지 골라 런처가 죽고 rank도 따라 죽음(`3119819`). 지금: 컨테이너 자손 중 cmdline이 트레이너 스크립트를 담고, `accelerate`·`bash -lc`를 담지 않으며, 부모 cmdline이 `accelerate`를 담는 것 = rank(`3ed6413`). DataLoader 워커는 cmdline이 같지만 부모가 rank라 제외.
3. **dynamo가 켜진 rank는 SIGUSR1 파이썬 핸들러를 실행하지 않는다** — 아래 CAUTION.
4. 트레이너가 시그널을 무시하면 링크는 48 h에서 그냥 죽고 다음 링크도 없다 → watchdog(600 s 뒤 rank kill + `FORCED_PREEMPT` → 마지막 latest_state로 다음 링크, `840384f`).
5. 새 할당의 첫 launch 실패 — 아래 ❌.
6. `[ -n "$WATCHDOG" ] && kill "$WATCHDOG"`: watchdog이 이미 끝났으면 `kill`이 실패하고, **`&&` 뒤 마지막 명령의 실패는 errexit를 발동**한다 → 재제출 직전에 스크립트 사망(`3119991` FAILED 1:0). `[ -z … ] || kill … || true`(`1a245ce`).
**왜 그때는 맞아 보였나.** "체인이 이어진다"는 `af6cce8`이 wait 루프의 errexit 문제를 고쳤다는 사실에서 **추론**됐고, 경계를 실제로 지난 잡이 없었다. 리허설 한 번(15분, 0.25 GPU-h)이면 09-08에 알 수 있었다. 교훈: **시간 상한 경로는 상한을 짧게 잡은 리허설로만 검증된다.** 최종: `3120089`(정상 경로 COMPLETED 0:0) · `3120124`(강제 경로 → 자식 `3120146` 제출 → 취소).

### ⚠️ CAUTION — dynamo(torch.compile)가 켜진 프로세스는 SIGUSR1 핸들러를 실행하지 않는다; SIGTERM은 된다
실측: 같은 동결 트레이너, 같은 rank 탐지. `TRANSGENIC_NO_COMPILE=1`이면 USR1 → `signal 10 received` → 저장 → 종료(probe `3119954`). compile 시도+폴백(시드 123 조건, probe `3119984`)이면 USR1 150 s 무반응, **같은 rank에 SIGTERM → 즉시 `signal 15 received`** → 저장 → 종료 rc 0. 26.08 진짜 compile(`3119779`, 대상이 `python3 -u train/…`임을 cmdline으로 확인)도 USR1 무반응. 트레이너는 두 시그널을 같은 핸들러에 등록한다. **왜 USR1만 막히는지는 모른다**(torch 내부의 USR1 사용 의심, 미확인). 조치: 체인은 SIGTERM을 보낸다(`9161508`); 통합 시험 스크립트는 아직 USR1이라 26.08 판이 중단 경로에서 FAIL로 보인다. 근본 조치 후보는 파일 기반 stop 플래그(저자 결정 (c)).

### ❌ 기각 — "torch `max_memory_allocated` 101 GB는 cudaMallocAsync 풀 고수위 계정이고 실사용은 ≈ 30 GB" (§1i CAUTION·PARTIAL의 정정의 정정)
낮에 3 step probe가 11 GB를 보고하고 50 step `3118903`의 nvidia-smi 최대가 31.5 GB로 나와 그렇게 적었다. 저녁 데이터: 같은 torch 101 GB 런들의 nvidia-smi 2 s 샘플 최대가 **14.8**(eager non-reentrant 30 kb `3118909`)·**78.9**(eager reentrant 30 kb `3119174`)·**31.5/84.9/97.3**(compile 30/61/129 kb)로 제각각 — 같은 설정에서 5배 차이는 실사용 차이가 아니라 **샘플링이 마이크로배치 안의 순간 최고점을 놓치는 것**이다. torch의 101 GB(모든 티어·이미지에서 일정, 디코더 8,192 토큰 지배)가 순간 peak이고, 129 kb no-ckpt OOM(allocated 100.65 GB)이 실제로 죽은 사실과 일치한다. **왜 그때는 맞아 보였나.** 3 step probe의 11 GB는 50 step에서 재현되지 않았고(원인 미상), nvidia-smi 값을 "실측"으로 격상했다 — 계측 주기가 현상의 시간 척도(수십 ms)보다 두 자릿수 길었다. 결과: **no-ckpt 옵션 닫힘**, 여유 ≈ 19 GB, 129 kb compile은 천장 근처(샘플만으로 97 GB). 벤치의 nvidia-smi 줄은 **하한**으로만 읽을 것.

### ❌ 기각 — "`unspecified launch failure`는 gh062/gh093(그리고 gh121·gh139) 노드 문제 → 제외하면 된다" (§1h 정정의 정정)
새 4 GPU 할당의 첫 launch가 step 0에서 죽고(gh121 `3118861`, gh139 `3119204`), **같은 노드에서 30 s 뒤의 두 번째 launch는 매번 정상**이었다. 1 GPU(gh062×2, gh093)도 같은 증상이라 NCCL이 아니다. Codex 진단: 첫 CUDA 접근에서 cudaMallocAsync 풀 생성/nvidia_uvm 초기화가 Apptainer `--nv` 아래 경합(singularity#1441 계열). `ALLOC=native`는 처음 쓰는 노드(gh045)에서 첫 launch 성공(약한 증거). 조치: SIF 안 `torch.ones(1).cuda()` 워밍업 + 제자리 재시도(두 번째 실패에서만 노드 제외). **왜 그때는 맞아 보였나.** 실패 3건이 두 노드에 몰려 있었고, 같은 노드 재시도를 해 본 적이 없었다. 노드 제외는 멀쩡한 하드웨어를 버리는 조치였다.

### ⚠️ CAUTION — "nvidia-smi 2 s 샘플 최대 = peak"
위 ❌의 계측 측면. 순간 peak를 재려면 `torch.cuda.max_memory_allocated`(프로세스 내) 또는 10 Hz 이상 샘플링이 필요하다. `bench_b5.slurm`의 `driver_peak_mem_mib_per_gpu` 줄은 남겨 두되 "샘플 하한"으로 읽는다.

### ⚠️ CAUTION — bash 잡 알림 `Terminated ( sleep "${USR1_GRACE:-600}"; … )`은 무해
watchdog 서브셸을 `kill`하면 bash가 그 명령 텍스트를 여러 줄로 찍는다(`3120089` .out). 오류가 아니다.

### 교훈 — 오늘 잡힌 셸·Slurm 함정 4건
- **`A && B`에서 B의 실패는 errexit를 발동한다**(마지막 `&&` 뒤). 정리용 `kill`은 `|| true`.
- **`--export=NAME=a,b`는 쉼표에서 갈라져 두 변수가 된다.** 목록은 프로세스 환경(`export`)으로 넘기고 `--export=ALL`로 상속.
- **런처의 argv에 스크립트 경로가 들어간다.** "cmdline에 스크립트 이름이 있으면 rank"는 런처·`bash -lc` 래퍼까지 잡는다.
- **`grep -q` + `pipefail`**: 일치 즉시 종료하면 `tail`이 SIGPIPE로 죽어 조건이 거짓이 된다. 전체를 소비하는 `grep -F … > /dev/null`.
- (재확인) **`pgrep -f <패턴>`은 자기 셸을 매치한다** — 테스트 정리에서 harness 자신을 죽여 30 s 타임아웃.

## 1k. 2026-09-10 오전 격리 — 이미지 권고의 대체, 경계 시각 오류, 26.08 preflight, 체크포인트 교체 창, epoch 경계 accumulation, 시드 123 train loss

> 시간은 전부 PDT(저자 지시 09-10). DeltaAI 출력은 CDT = PDT+2.

### ❌ SUPERSEDED (2026-09-10) — "확인 시드 456/789는 25.06 eager 4 GPU로 (시드 123과 같은 이미지·커널)"
09-09 저녁·09-10 아침의 **권고**였고 틀린 것은 아니다 — 저자가 **26.08**(compile, 12~17 % 절감, 4 GPU 6.15 h/epoch)을 골랐다. 대체한 이유와 비용은 A44(v1.34, `c646e76`)에 있다: 세 시드는 혼합 런타임 재현이 되고 중립성은 주장하지 않는다. 이 권고 문장을 인용하지 말 것; 인용할 것은 A44와 `runtime_identity.txt`.

### ❌ 기각 — "시드 123의 첫 48 h 경계 ≈ 09-10 21:50 CDT"
계산 오류. `sacct -j 3116844 Start` = 2026-09-09 00:00:34 CDT → 상한 2026-09-11 00:00:34 CDT = **2026-09-10 22:00 PDT**, `--signal=B:USR1@900` → **21:45 PDT**. "21:50 CDT"는 22:00 PDT를 CDT로 옮기다 두 시간을 빠뜨린 것. **왜 그때는 맞아 보였나**: 잡의 elapsed(1-09:xx)만 보고 시작 시각을 역산했고 시간대를 적지 않았다. 교훈: 경계 시각은 `sacct Start`에서 계산하고 항상 시간대를 붙인다 — 저자가 "모든 시간은 PDT"로 못 박은 계기.

### ⚠️ CAUTION — 26.08(torch 2.14)에서 preflight의 토크나이저 검사가 stderr를 읽는다
`3124100`(첫 4 GPU·26.08 리허설)이 학습 전에 rc 3: torch 2.14가 import 시 `KernelPreference is an Enum subclass … torch.compile` 경고를 stderr에 찍고, 검사가 `2>&1`로 잡아 `could not verify`로 launch를 거부했다. 25.06에서는 경고가 없어 드러나지 않았다. 수정 `07486a8`(stdout 마지막 줄만). **왜 그때는 맞아 보였나**: 검사 결과를 문자열 그대로 case로 분기했고 "확인 불가"는 안전한 실패로 보였다 — 그러나 정상 환경에서 launch를 막는 것도 결함이다. 같은 리뷰에서 반대 방향의 결함도 나왔다: 토크나이저 예외 시 소스 검사만으로 통과하던 fail-open(`5433f27`에서 제거, `PREFLIGHT_ALLOW_SOURCE=1`은 진단용).

### ⚠️ CAUTION — latest_state를 "지우고 rename"하던 교체 창 (`5433f27` 전)
rank 0이 `rmtree(latest_state)` 뒤 `rename(tmp → latest_state)`를 했다. 그 사이(수 GB rmtree는 초 단위)에 watchdog KILL이나 노드 장애가 오면 `latest_state.tmp`만 남고 `resume_dir`는 tmp를 무시 → epoch 체크포인트나 처음부터. Codex 준비 상태 리뷰가 fault injection으로 재현. 이제 rename만(구판→`.prev`, 신판→`latest_state`, 그 뒤 `.prev` 삭제)이고 재개는 `.prev`도 인정한다. **시드 123의 동결 트레이너는 옛 방식** — 그 창은 200 step마다 몇 초, watchdog KILL은 시그널 무시 시에만.

### ⚠️ CAUTION — epoch 경계의 accumulation 잔여분은 두 런타임 모두에 있고, 시드 123은 보존하지 않는다
198,828행 = 96 × 2071 + 12. epoch 끝마다 12개 micro-batch(4 GPU: rank당 3개)의 gradient가 다음 epoch으로 넘어가며, `micro_done`이 epoch을 넘어 누적되므로 epoch 2의 첫 update는 84번째 micro-batch에서 난다(중단 없이 돌면). `accelerator.save_state`는 `.grad`를 저장하지 않으므로 epoch 체크포인트에서 재개하면 12개가 유실되고 이후 경계가 12씩 밀린다. `c646e76`(확인 시드)은 rank별 `pending_grads/rank<i>.safetensors` + `meta.json` `pending_micro`로 보존·복원한다. **시드 123(동결 08f3e2b)은 미적용** — 정상 체인은 optimizer 경계에서 저장된 latest_state로 재개하므로 이 경로는 epoch 체크포인트가 최신 복구점일 때만 밟힌다(A44 3항: 발생 시 런 로그에 편차 기록). 인증은 `3124347`(`NTRAIN=484`, rank당 121, 잔여 1: 복원이 맞으면 epoch 2 첫 update가 micro-batch 23, 잃으면 24).

### ⚠️ CAUTION — 시드 123 로그의 `train_epoch_loss`는 epoch ≥ 2에서 epoch 번호만큼 작다
train loss 평균이 epoch을 넘어 누적된 `micro_done`으로 나뉘었다(같은 카운터를 accumulation과 평균에 겸용). epoch 1 값(1.636)은 맞고, epoch 2 값은 참값의 1/2, epoch k 값은 1/k. **검증 손실·best·조기 종료는 별도 계산이라 무관.** 새 코드는 `epoch_micro`로 나눈다(`c646e76`). 시드 123은 로그를 고치지 않고 결과표에서 보정한다(A44 4항).

### 교훈 (2026-09-10 오전)
- 새 이미지는 코드가 아니라 **검사 스크립트**부터 깨뜨린다 — 정상 환경에서 launch를 막는 검사는 결함이다(preflight stderr).
- 체크포인트 교체는 **rename만** — 삭제가 rename보다 앞서면 창이 생긴다.
- 런 식별자는 레시피만이 아니라 **런타임**(이미지 sha256·커밋·GPU·CPU·torch·정책)까지 — `run_config.json`은 이미지가 바뀌어도 통과했다.
- 한 카운터를 두 용도(accumulation 경계·평균 분모)로 쓰지 않는다.
- 경계 시각은 `sacct Start`로, 시간대를 붙여서.
- 인증 잡을 낸 뒤 코드를 또 바꿨으면 **그 잡은 참고용**이다 — `3124112`·`3124228`은 그래서 취소했다.

## 1l. 2026-09-14 격리 — B6 설계에서 기각된 것, 검토가 놓친 것, 실행 불가로 드러난 전제

원문 로그: `resume.md`의 ✏️ 09-14 07:1x~11:4x 불릿(10개). 리뷰 원문: scratchpad `codex_review_output.txt`·`codex_review_v2_output.txt`·`codex_review_a46_output*.txt`. 커밋: 워크플랜 `6c86169`→`320de01`→`c4b9d30`, 결정 `460de14`·`9a8c8e1`.

### ❌ 기각 — B6 v0 "증거 지지 isoform을 종별 GFF3에 전사체로 주입해 라벨로 학습" (07:1x 제안)
EST의 90 % 이상이 단편이라 완전 체인이 필요한 라벨 주입은 대부분의 증거를 버린다(Kimi: 프롬프트 토큰도 조립 체인이 필요해 같은 이유로 배제). 대체: per-nucleotide 트랙(단편이 덮는 염기만큼 정확히 기여, 조립 불필요). **왜 맞아 보였나**: `Gene.transcripts`(`gsf_contract.py:184-274`)에 주입하면 코퍼스 빌더·트레이너를 손대지 않아도 됐다 — 구현 비용이 설계 판단을 앞섰다.

### ❌ 기각 — B6 v1 "per-base 보조 세그멘테이션 헤드(학습에만 증거)" (Codex 1차, reject-as-designed 13건)
추론 시 증거가 사라져 "증거 쓰는 도구(BRAKER/MAKER/PASA)와 비교"라는 저자 목적을 채우지 못한다(표현 정규화에 그침). 14클래스 라벨은 중첩 multilabel·배경/phase 클래스 없음(`gsf_contract.py:48`)이라 14-way CE도 positive-only BCE도 부적합; `segmentation_loss`는 최상위 출력 dataclass에 없음(제안서 오류). 대체: 증거 **입력 채널** + 채널 드롭아웃 0.5(한 모델 두 모드). **왜 맞아 보였나**: C2 세그멘테이션 계획(`IMPLEMENTATION_ORDER_B5_C0_C2_v1.md:33`)이 이미 있어 "있는 헤드에 증거를 붙이면 된다"로 읽었다.

### ❌ 정정 — Codex 1차 ⑥ "RC 타일에서 donor/acceptor·5'/3' 채널을 스왑"
RC에서 donor/acceptor **정체성은 유지**되고 strand 채널만 바뀐다(`tests/test_gsf_rc.py:26`; Codex 2차가 1차를 스스로 정정). 외부 검토의 지적도 코드로 대조한 뒤 반영한다.

### ❌ SUPERSEDED (저자 결정 09-14, `460de14`) — C1'(마스크 수리 코퍼스의 서열 전용 재빌드 런)
예산(≈ 600 GPU-h 부족)으로 생략; 시드 123(C1)이 유일한 서열 전용 비교군, 수리 전/후 DNA 마스크 차이는 정량 보고. **주의**: B5-vs-B6에 "동일 DNA 마스크" 주장은 더 이상 할 수 없다 — 워크플랜 v1.1의 "C1' = 같은 DuckDB에서 트랙 무시" 문장은 살아 있지만 실행되지 않는다.

### ❌ 오보 정정 — "학습 종 9종 EST는 스크리닝 전 정렬이라 재정렬 필요" (07:1x 조사 에이전트)
pronghorn 실측(08:1x): `evidence/est/<SP>/univec/DONE` 9종, `est_align/<SP>/DONE`의 md5가 스크리닝 FASTA와 일치, 09-04 배열 `6148523_0-17`(게이트 3종 포함 `est_align.sbatch`, md5 e0641d9b), 매핑률 92~99 %. **왜 맞아 보였나**: 조사가 `RETIRED_DO_NOT_USE/est_align_prescreen_20260903`(09-03 스크리닝 전 BAM)만 보고 판단했다 — 디렉터리 이름의 `RETIRED`를 읽지 않았다. 스크리닝이 없던 것은 검증 종 2종뿐이고 그날 완료했다(`6163269`·`6163270`).

### ⚠️ CAUTION — "MVP 채점 = A-ONT1/A-HiFi 완전 체인 지지" (워크플랜 52행 · A46 17항)
정렬본이 없다: `longread_align.sbatch:70-71`은 `c2_training_eligible`이 아닌 런을 거부하고 A-ONT1/A-HiFi는 `b1_validation_only`(A13:584)라 `evidence/longread_align_v1/`에 없다. **왜 맞아 보였나**: A46 로그 행의 "long-read alignments complete"는 학습 적격 런(216행)만 뜻하는데 검증 전용까지 포함한 것으로 읽혔다. 대체: MVP 설계서 §2 — reference-dependent F1을 1차 지표로 명시하고, 독립 채점은 50 유전자좌 부분 참조에 별도 정렬 후 `59_evidence_support.py`(BAM 필요; PAF만으로는 supplementary/chimera 배제 불가, 그 파일 docstring 4~6행). A46 17항 문구는 커밋 전에 맞춘다.

### ⚠️ CAUTION — "rank당 1.4 it/s"는 저장소 어디에도 없다
근거는 `resume.md` 09-12 06:49 불릿(시드 123 링크 1, 2시간 실측)뿐이고, 벤치 값 2.25 it/s보다 1.6× 느린 이유는 미실증(3티어 혼합·실제 라벨 길이 → compile 동적 shape 추정). MVP 스텝 예산(250 = 24 micro-batch/스텝 × 17 s)은 이 값에서 나왔으므로 day-4 `--benchmark-steps`로 재산정한다.

### ⚠️ CAUTION — thinning rank에 Python `hash()` (Codex 3회가 놓치고 Kimi가 잡음)
프로세스별 salt로 replay가 깨진다 → `hashlib.blake2b(f"{aug_seed}\t{species}\t{molecule_id}", digest_size=8)` 고정(워크플랜 25행). "결정론"을 주장하는 규칙은 함수 이름과 바이트 인코딩까지 명세한다.

### ⚠️ CAUTION — 스크립트 번호 `62_b6_panels.py`(A46 12항)는 `62_geenuff_qc.py`와 충돌
`revision/scripts/`는 `69_orientation_report.py`까지 쓰였다. ✏️ 12:1x 해소: **`71_b6_panels.py` / `72_mvp_select_loci.py` / `72a_est_molecules_per_locus.sh`**로 개명·재실행 지시, A46·설계서 문구 동기화 지시. 같은 산출의 1차 실행(구 규칙, 적격 2,143)은 **폐기**: ChrC/ChrM 5개가 섞였고(핵 염색체 필터 없음) MVP 부필터가 A46 rule 12와 달랐다. **왜 맞아 보였나**: 동결 DB의 `geneList` 행이 유전자가 아니라 30,720 nt 타일이라는 것을 패널 빌더가 실행 뒤에야 알았다(held-out 3,430 중 라벨 타일 보유 431개, 나머지 N-마스킹) — 유전자 좌표는 `gene_key_map`, held-out은 `gene_split`, exon 수는 `build_manifest`의 GFF에서 읽어야 한다.

### ⚠️ CAUTION — diamond `build_dbs.sh`의 레코드 수 출력이 비고 `"` 잡파일이 생김 (pronghorn `6163304`)
따옴표 이스케이프 문제. DB 자체(`DB.sha256`·`PROVENANCE.txt`)는 정상, 레코드 수는 수기 주석(Swiss-Prot 42,106 / OrthoDB 12,115,085). 잡파일은 삭제됨.

### 교훈 (2026-09-14)
- 목적("증거 쓰는 도구와 비교")을 먼저 적고 설계를 고른다 — v0·v1은 둘 다 구현 비용에서 출발해 목적을 놓쳤다.
- 외부 검토는 **연쇄**로 쓴다: Codex가 세 번 놓친 `hash()`를 Kimi가 잡았고, Codex 2차가 1차의 RC 스왑을 정정했다. 충돌이 없어도 두 검토자를 쓴 값이 있다.
- "정렬 완료"는 **어느 역할의** 런인지까지 적는다 — 드라이버의 역할 게이트가 검증 전용 세트를 조용히 제외한다.
- 실측값은 쓰는 자리마다 출처 행을 붙인다; 출처가 인수인계 문서뿐이면 저장소 기록으로 옮긴다.
- 조사 에이전트의 "없다"는 `RETIRED`·`DONE` 마커를 읽었는지 확인한 뒤 믿는다.

## 2b. 2026-09-03 오후 스냅샷 (14:00 ~ 16:44)

> ⚠️ **같은 날 세 번째 추출이다.** §2a가 "하루 만에 10개가 쌓여 정본 블록을 밀어냈다"고 적었는데,
> 그 뒤 오후에만 **13개가 더 쌓여 495줄이 됐다.** 오전에는 훅 주기가 문제라고 보지 않았지만, 하루에
> 세 번 같은 규칙을 적용해야 한다면 그것은 추출 규칙의 성공이 아니라 **훅 주기의 문제**다.
> 세션이 활발한 날에는 스냅샷이 20~30분마다 쌓이고, 각각이 20~35줄이다.
>
> 이 절의 내용은 `resume.md`에서 **원문 그대로** 옮긴 것이고 삭제가 아니다. 여기서 시작하지 말 것 —
> 정본은 `resume.md`의 📌 블록이다. 이 스냅샷들의 "현재 상태"·"다음 작업"·"진행 중"은 전부 **그
> 시점의 서술**이며, §1의 일반 규칙이 일괄 격리한다.

<!-- 이하 resume.md 41–352줄 원문, 2026-09-03 17:0x 추출 -->

## ▶️ 이전 스냅샷 (2026-09-03 16:44)

### pgl-gpu 실측 검증 — #66 완료, 선언표 체크섬 자동 대조 부착

pgl-gpu에 ssh로 접근해 OrthoDB 단백질(`md5:453cb32b…`), Swiss-Prot 민감도(`md5:552089d0…`) 파일의 md5를 재계산. `make_dataset_roles.py --verify-resources` 플래그 추가: 빌더 실행 시 선언값과 실제 파일의 체크섬을 대조하고 불일치 또는 읽기 실패 시 매니페스트 쓰기 거부. 음성 테스트(훼손된 선언값·없는 경로) 3건으로 검사 동작 증명. 테스트 신규 4건 추가(10건 통과), 기존 155건 유지.

**상태**: 커밋 `7c15866`(리포)·`4895f0a`(위키) 푸시. #66 이슈에 follow-up 코멘트. 선언표 MD5 `a6375713` 불변, 매니페스트 221줄.

### 다음 작업

- **#51·#19·#20**: ACCESS 자원명 결정 (저자)  
- **#58·#55·#56·#61·AtRTD3**: 판단 대기

## ▶️ 이전 스냅샷 (2026-09-03 16:32)

### 위키 갱신 완료 — 커밋 `343b668`

신규 파일 **(31)** 추가, 파일 **(30)**의 열린 질문 2건을 ✅ 해소 표시. `projects/transgenic/README.md`, `index.md`, `log/2026.md`, `log.md` 갱신. 링크 0 broken.

### 논리 정리

두 결정의 공통점: **프로토콜 고정 조항이 이슈 텍스트보다 우선**. #17은 이슈가 A28보다 낡았고, #66은 이슈 #44의 "TSV에 md5 기록"이 스키마상 불가능했음. 동결 DB 직접 질의로 누출 0건 증명(리포트의 checks 블록엔 누출 검사가 없음을 명시).

### 상태

이슈 11건 종료(#11·#12·#13·#14·#17·#46·#63·#64·#65·#66·#67), 신규 4건 등록, 열린 것 48개. 저장소 tip **`02868a8`**, 프로토콜 **v1.26**(A36), 위키 tip **`343b668`**. EST 증거 정렬이 프로토콜상 유효. §5 접합 호출 앞의 장애 없음.

### 다음 작업

- **#51·#19·#20**: ACCESS 자원명 (저자 결정)
- **#58·#55·#56·#61·AtRTD3**: 판단 대기

## ▶️ 이전 스냅샷 (2026-09-03 16:30)

### #17·#66 저자 결정 반영·종료

**#17 종료** — A28이 이미 "recipe, path, seed, batch and accumulation, max epochs, patience"만 요구하고 해시를 언급하지 않으므로 프로토콜 개정 불필요. 동결 DB(272,224행)에 직접 질의한 결과: orthogroup split 누출 0건, RC split 불일치 0건, forward 없는 RC 0건, strict_holdout→test 불일치 0건, gene 다중 split 0건, Zmays 행 0건.

**#66 종료** — 빌더에 `RESOURCES` 선언표 추가해 비-런 자원 해결. OrthoDB 단백질 `md5:453cb32b…`, Swiss-Prot 민감도 세트 `md5:552089d0…` 포함. TDD 3건 실패 후 구현하고 6건 통과. 매니페스트 187행 → 220행(신규: 비-런 자원 2 + 오늘 수집된 ONT 31런). `est_align` 오염 0건, `DATASET_ROLES.est_v1.tsv` 여전히 `a6375713`.

### 상태

커밋 `02868a8` 푸시. 이슈 11건 종료(#63·#11·#12·#13·#14·#17·#46·#64·#65·#66·#67), 신규 4건 등록. 열린 이슈 48개. 테스트 155건 통과.

### 다음 작업

1. **#51·#19·#20**: ACCESS 자원명 결정 (저자)
2. **#58·#55·#56·#61·AtRTD3**: 판단 대기
3. 위키 갱신 — #17·#66 반영 여부

## ▶️ 이전 스냅샷 (2026-09-03 16:23)

### 18개 이슈 검증 완료 — 6개 종료, 새 결함 4건 #64–#67 즉시 수정

4개 서브에이전트가 18개 이슈의 acceptance criteria를 검증했습니다. #11 #12 #13(결정적 불변식·token cap·segmentation) #14(split 코드경로) #46(B5 데이터셋) #63(문서 오기)이 종료되었습니다.

### 4건 신규 결함 등록·커밋 — 빌더·필드 처리·자원 표·문서

**#64 종료**: ROLES에 4번째 값 `sensitivity_set` 추가, 기존 187행 매니페스트 재검증 무위반, 신규 3건 테스트 통과.

**#65 종료**: `longread_fetch.sh:18` 빈 `instrument_model` 처리 변경(continue → .UNRESOLVED 마커 + 로그 + exit 4), 세 경우 직접 테스트.

**#66 절반**: OrthoDB 단백질 FASTA md5 `453cb32b…` 파일에서 재계산해 Protocol §1 고정객체에 등재 완료. DATASET_ROLES.tsv는 빌더가 런 단위로만 스캔하므로 설계 결정 대기(todo.md §3).

**#67 종료**: BioProject 직접 세어 ONT FL-cDNA MinION 런 정확히 8개 확인(문서 4→8 정정, ✏️).

### 상태

커밋 `0d92741` 푸시, 테스트 155건 통과, 미커밋 0, 열린 이슈 50개(-9+4). 프로토콜 v1.26.

### 다음 작업

1. **#66 설계**: 비-런 자원을 DATASET_ROLES.tsv에 표현할지, 아니면 매니페스트는 런 전용·자원 체크섬은 Protocol 정본만 유지할지
2. **#17**: A28이 "경로·recipe"만 요구하는데 이슈는 해시를 요구 — A28이 대체했는지 확인
3. **#14·#44·#46** acceptance 마무리

## ▶️ 이전 스냅샷 (2026-09-03 16:05)

### 4건 이슈 검증·수정 완료 (2026-09-03 16:xx)

**#64 종료** — ROLES에 4번째 값 `sensitivity_set` 추가 (A30 고정 텍스트 반영). `revision/scripts/tests/test_dataset_roles.py` 신설 3건 통과. 기존 187행 매니페스트 재-validate 무위반 확인. 회귀: `tests/` 155건 유지.

**#65 종료** — `evidence/longread_fetch.sh:18` 빈 `instrument_model` 처리 변경. `continue` → 이제 `.UNRESOLVED` 마커 작성 + 로그 1행 + `exit 4`. 세 경우(빈 model, 정상 model, 불일치) 직접 테스트. 수집 프로세스 0개 상태에서 안전하게 적용.

**#66 절반** — OrthoDB 단백질 FASTA md5 `453cb32b02e0799950d7d5f4de5f62ac` (3,178,195,033 B) 파일로부터 직접 재계산해 Protocol §1 고정객체 표에 등재 완료. `DATASET_ROLES.tsv`는 여전히 열림 — 빌더가 증거 트리를 **런 단위로만 스캔**하므로 비-런 자원(단백질, Swiss-Prot) 삽입이 설계 문제. `todo.md` §3에 옮김.

**#67 종료** — BioProject 직접 세어 ONT FL-cDNA MinION 런 **정확히 8개** 확인 (문서 4 → 8로 정정, ✏️ 마커). 트리가 과다 수집이 아니라 문서 오기.

**커밋 `0d92741` 푸시**. 열린 이슈 50개 (오늘 -9 +4). 테스트 155건 통과, 미커밋 0.

### 다음 작업

1. #66 설계 결정: 비-런 자원 선언 표를 빌더에 줄지, 아니면 DATASET_ROLES.tsv는 런 전용·자원 체크섬은 Protocol 정본만 유지할지
2. #14·#44·#46 acceptance criteria 검증 계속

## ▶️ 이전 스냅샷 (2026-09-03 15:47)

### #11·#12·#13 검증 후 종료 (2026-09-03 15:50)

**#11 종료** — RED→GREEN 순서 재현: `2c68baa`(테스트)가 `c79a9a2`(RC 구현)의 조상 ✓. `docs/gsf_spec_v1.md` §11 지목 테스트 10개 전부 실재 ✓. 결정적 불변식 64건 통과(절단 금지, SQL NULL, RC 이중적용).

**#12 종료** — acceptance 전부 구현·테스트: `test_gsf_feature_caps`, `test_token_cap`, `test_v2_transcript_cap`, roundtrip 2종, build_b5 계 **46건 통과**. `rejected_records` 코드 확인 ✓.

**#13 종료** — segmentation 가중 정렬·RC 증강·orthogroup split 무결성 테스트 각각 통과.

### #17 열어 둠 — 저자 판단 필요

**불일치**: 이슈 본문 "DB 해시, config 해시" ↔ 프로토콜 A28 "경로, recipe, seed, epochs, patience" (해시 미언급). 현 구현은 A28 정확히 따르되, **같은 경로·다른 내용 DB**는 recipe 비교로만 감지 — A28이 충분한가?

**미확인**: 실제 B5 DB에 train/valid/test 멤버십·RC/orthogroup 누출 없음의 단일 dry-run 리포트 필요 여부.

### 다음 작업

1. #17: A28 의도 재확인 및 split/누출 검증 dry-run 실행 판단
2. 데이터 산물 #14·#44·#46·#15 검증 진행 중

## ▶️ 이전 스냅샷 (2026-09-03 15:32)

### A36 커밋·#63 종료 (2026-09-03 15:28~15:45)

**커밋 `bf207ee`**: 프로토콜 v1.26(A36 ≤120 nt 제외) · 스크리닝 스크립트 3개, 저자 Won <wyim@unr.edu> 단일, origin/main 푸시됨.

**#63 종료**: `library_strategy` 필터 결함은 화이트리스트로 고쳐졌고, 153런 소급 감사로 WGS 0/153 확인. 본문 2번(SRR25203456)은 필터가 아니라 다운로드 고장이라 분리, todo.md §4 남김.

**#10**: 코멘트만(4개 scope 미완). 기록: 스크리닝 9/11종, EST 무결성 확인(절단 아님), A36 적용.

**#14·#44·#46**: 수치는 resume.md에 있으나(34만 행·OrthoDB 1천만·GeenuFF 19만 건) acceptance criteria 검증 필수. 미확인으로 종료 불가.

상태: 열린 이슈 54개 · 미커밋 0 · 테스트 155건 통과.

### 다음 작업

1. #14·#44·#46 acceptance 검증 후 종료/개선
2. §5 접합 호출 (EST 정렬 완성 후)
3. ACCESS 자원명 #51·#19·#20 미해소

## ▶️ 이전 스냅샷 (2026-09-03 15:28)

### EST 재정렬 `6147708_[0-8]` 전부 COMPLETED (15:23)

**9종 모두 완료**. 원본 5,648,175 → 정렬 입력 4,718,673 (제외 935,720건, 16.6 %). **Athaliana 73.51 % → 96.96 %**, 9종이 92.80–98.94 % 안에 들어왔다. 제외분의 82.8 %가 Athaliana이다.

**Vvinifera 하락은 정상**. 매핑률 97.61 % → 97.12 %(2.55 % split)인데, split 조각만 48.43 %, 나머지 98.32 %. A21이 내부 벡터로 판정해 자른 클론의 절반이 게놈에 안 붙는다는 뜻 — 스크리닝이 일을 한 결과다. 종별 벡터 상태 편차(Vvinifera split 2.55 % vs 다른 종 0.01–0.66 %)는 A21을 건너뛰면 안 됐던 실측 근거다.

`.batch` 스텝 전부 COMPLETED. `.extern`만 `OUT_OF_ME` — 이미 `guide/pronghorn.md §412`에 성문화된 클러스터 버그(무작위 표본 100 % extern-OOM)라 중복 기록하지 않고 앵커로 연결했다.

**갱신 완료**: resume.md 📊 절 교체, todo.md §2b 해소(`🚨` → `✅`), 위키 커밋 **`b68ad3b`** 푸시 (update·README·index·log, broken link 0).

### 다음 작업

1. **§5 접합 호출** — 프로토콜상 유효한 EST 정렬 처음 생김
2. ACCESS 자원명 문제 (#51·#19·#20) 해소 대기
3. 저장소 A36 관련 4파일 미커밋 상태

## ▶️ 이전 스냅샷 (2026-09-03 14:58)

### 진행상황 타임스탬프 적용 + 위키 커밋 `e8d456d` 푸시

타임스탬프 형식 `[YYYY-MM-DD HH:MM:SS TZ] 완료 N/9 — 신규: 종명 종명 ...` 적용. 현재 **3/9 진행** (Athaliana · Bdistachyon · Gmax 완료, 2026-09-03 14:53:16 PDT).

**위키 갱신 완료:**
- `projects/transgenic/updates/2026-09-03g_...md` (신규 update, 131줄)
- `projects/transgenic/README.md`: 프로토콜 **v1.25 → v1.26**, 주제별 찾기 2줄
- `guide/installs.md`: **UniVec_Core 절 신설** (build 10.0 md5, 처리량, 함정)
- `index.md`, `log.md`, `log/2026.md`, `.gitignore` 갱신
- 검증: 0 broken links

**판단:**
- 위키 CLAUDE.md 규약대로 사실/해석/열린 질문 분리, 기각 가설 ❌, 오류 주장 ✏️ 표시
- `projects/sylvan-egapx/` 파생 변경은 되돌림 (무관 프로젝트)
- `guide/pronghorn.md` SLURM PATH 기존 기록 있어 중복 제외

### 다음 작업

1. 배열 6147708_[0-8] 전 9종 완료 감시
2. 결과 테이블 생성 후 `resume.md` 📊 절에 반영
3. 위키 update에 최종 수치 덧붙이기

## ▶️ 이전 스냅샷 (2026-09-03 14:25)

### UniVec DB + est_align.sbatch 재작성 — 게이트 2개, 둘 다 fail-closed

UniVec_Core build 10.0 (3,155 서열, md5 `82f07865`)을 `evidence/univec/`에 구축. 9개 배열 작업이 동시에 `makeblastdb`를 치면 경합하므로 로그인 노드에서 미리 한 번만 만들었습니다.

**est_align.sbatch 재작성** — A18.3 역할 매니페스트(`exit 2`) + **A21 스크리닝 + A36 하한**(`exit 5`). 입력이 `est.univec.fa.gz`로 바뀌고, PROVENANCE에 UniVec DB md5·버전·`min_len=121` 기록. 스크리닝 누락이 조용히 통과한 게 원래 사고 원인이므로 그 구멍을 막는 게 핵심입니다.

### 스크리닝 이전 BAM 9종 격리 + 원본 스크립트 보관

`evidence/RETIRED_DO_NOT_USE/est_align_prescreen_20260903/`에 옮김 (삭제 아님). 실행됐던 원 스크립트(`est_align.sbatch.as_run`)·WHY_RETIRED.txt·기존 README 보존 — 이 BAM들은 어제 보고한 길이별 매핑률과 quarantine §1d의 근거입니다.

### 게이트 실동작 검증 — Sitalica, 가장 작은 종

66,027 → 65,513 (keep 65,359 · trim 82 · split 44 · **drop 542**). Drop 542 = **원래 ≤120 nt 540** + 트림 후 짧아짐 2. 540은 제가 BAM에서 독립적으로 잰 값과 정확히 일치합니다. 출력 최소 길이 **정확히 121** — A36 하한이 실제로 걸렸습니다.

### SLURM 제출 — 6147708_[0-8]

`sbatch est_align.sbatch` → **`6147708_[0-8]`**. Athaliana(`_0`)가 실행 중, 나머지 8종은 Priority 대기(다른 프로젝트 잡 큐 많음).

### 다음 작업

1. **9종 배열 완료 감시** — SLURM 잡 6147708_[0-8] 결과 대기
2. **저장소 밖 스크립트 이동** — est_align.sbatch를 transgenic/ 하위로 이관 (todo §3, 저자 결정 미해결)
3. **A36 커밋** — 저자 Won <wyim@unr.edu> 단일 (DOCX/위키 검증 후)

## ▶️ 이전 스냅샷 (2026-09-03 14:12)

### 저자 결정 A36(v1.26): ≤120 nt EST 제외 — 프로토콜·코드·테스트 반영

프로토콜 A36 신설, v1.25→**v1.26**. 결정은 **제외** — 가중치 아님. 접합·사슬·UTR·callability 분모·독립분자 카운트·C2 라벨 어디에도 기여하지 않음.

**Athaliana 분해:**
- ≤120 nt 제외 773,346건: A8 기존 (≥100) 이미 걸렀을 480,880건 + 이번 결정 (100–120 구간) **292,466건**
- >120 nt 잔존 **756,354건**
- Sanger 부수 손실 13,440건 = 1.82 %

**반영 완료 (미커밋):**
- `PROTOCOL_B1_frozen_v1.md`: A36 신설, 제목 v1.26
- `61_univec_trim.py`: `DEFAULT_MIN_LEN = 121` (docstring·argparse 기본값)
- `univec_screen.sh`: `--min-len 121` (PROVENANCE 기록)
- `test_univec_trim.py`: 기본값 고정 테스트, 7건 통과
- `resume.md`·`todo.md`·`quarantine.md` 갱신

### 🚨 발견: A21 UniVec 스크리닝이 실행된 적 없음

A21 "only the screened FASTA is aligned" 명시이나 `evidence/est/*/univec/` 11종 전부 없음. 스크립트 있으나 미실행 → 어제 정렬 9종은 원본 `est.fa.gz` 사용, 프로토콜 위반. 접합 계산 전 A21+A36 재정렬 필요 (아직 접합·지지수 산출 없어 오염물 없음). `est_align.sbatch`가 `est/<종>/est.fa.gz` 직독하는 것이 원인. 입력을 `univec/` 경로로 바꾸고 없으면 fail-closed 처리 필요.

### 테스트·상태

- `tests/` **155건 통과** (📌 블록 기준값 동일, 회귀 없음)
- `revision/scripts/tests/test_geenuff_qc.py` 1건 실패 — 제 변경 무관, 낡은 사본 문제
- 미커밋: 4파일

### 다음 작업

1. **A21 스크리닝 + A36 재정렬** (SLURM, blastn 병목)
2. **`est_align.sbatch` fail-closed 수정** (입력 경로 검증)
3. **A36 커밋** (저자 Won 단일)
4. 낡은 test_geenuff_qc.py 정리 (선택)

## ▶️ 이전 스냅샷 (2026-09-03 14:00)

### Athaliana 매핑률 73.5% 근본원인: 구성 효과 확정

GenBank EST의 454 시대 짧은 라이브러리 두 개(91–92 bp)가 전체의 **51.7%(791,290건)**을 차지.
→ 길이 통제 시 96.87%(9종 중 4위). 참조결손·절단·벡터오염 가설 배제. 결함 아님.

### 9종 EST 정렬 완료 및 3파일 갱신

Athaliana–Vvinifera 모두 COMPLETED(2026-09-03 13:4x).
**resume.md**: 9종 최종표+라이브러리 분해, "종 단위 매핑률은 종 간 비교지표 아님" 함정 추가.
**todo.md**: §2 EST항 체크, 진행중 제거.
**quarantine.md**: §1d 신설(구성효과·착각원인 기록).

### ✅ 저자 결정: ≤120 nt EST 제외 (2026-09-03) → 프로토콜 **A36 (v1.26)**

가중치가 아니라 **제외**다. A8/A21의 수집 하한을 100 → **121 nt**로 올렸다. 제외된 레코드는 접합·사슬·
UTR·callability 분모·독립분자 카운트·C2 라벨 어디에도 기여하지 않는다.

- Athaliana에서 **773,346건 제외 / 756,354건 잔존.** 그중 A8 기존 하한(≥100)이 이미 걸렀을 것이 480,880,
  **이번 결정으로 새로 빠지는 것이 292,466**(100–120 nt 구간)
- **Sanger 부수 손실 13,440건 = Sanger 전체의 1.82 %** — 454형만 정확히 겨냥할 수는 없다
- 근거: 짧은 리드는 품질은 괜찮으나(MAPQ≥30 93.8 %) **접합 보유 20.6 %** 대 Sanger 55.6 %인데,
  §5의 "서로 다른 clone/accession ID ≥2"에는 Sanger와 동등하게 계수된다
- 반영: `61_univec_trim.py` `DEFAULT_MIN_LEN=121` · `univec_screen.sh --min-len 121` · 테스트로 고정

### 🚨 A21 UniVec 스크리닝이 한 번도 실행된 적 없었다 → 조치 완료, 재정렬 돌고 있음

A21은 **"screened FASTA만 정렬한다"**고 못박는데 `evidence/est/*/univec/`가 11종 전부 없었다.
즉 2026-09-03 EST 정렬 9종은 **원본 `est.fa.gz`로 돌린 스크리닝 이전 패스**였다. 접합·지지수를
계산한 적이 없어 오염된 하위 산출물은 없다.

조치한 것:
- **`evidence/univec/` DB 구축** — UniVec_Core build 10.0, md5 `82f07865`, 3,155서열
- **`est_align.sbatch`에 게이트 2개** — ①A18.3 역할 매니페스트 ②A21 스크리닝(`univec/DONE`과
  `est.univec.fa.gz` 없으면 `exit 5`). **스크리닝을 건너뛴 게 조용히 통과한 것이 원인**이었다
- **이전 BAM 9종 격리** — `evidence/RETIRED_DO_NOT_USE/est_align_prescreen_20260903/`
  (삭제 아님; 코퍼스 진단 근거라 진단 용도로만 읽는다)
- **재정렬 제출** — SLURM `6147708_[0-8]`

⚠️ **blastn은 병목이 아니다**(제가 처음에 틀리게 말했다). UniVec_Core는 922 KB 소형 DB라 2스레드에서
2,230 레코드/s — 11종 796만 건이 2스레드 약 1시간. Sitalica 66,027건 실측 66 s(스플릿·트리밍 포함).

### 다음 작업

1. ACCESS 자원명 확정(#51·#19·#20 차단) — 승인 나면 바로
2. **#59 seedable sampler**(한 줄, 첫 시드 전 필수)

## ▶️ 이전 스냅샷 (2026-09-03 13:41)

### 세 파일 정리 및 📌 블록 검증 강화

하루에 스냅샷 10개가 쌓여 resume.md의 정본 블록을 밀어낸 문제를 해결했다. resume.md를 **155줄로 정리**, todo.md와 quarantine.md도 정책에 맞춰 갱신했다.

📌 정본 블록에 **검증 명령 3줄 추가**: `git log --oneline -1`, `gh issue list`, `pytest -q`. 오늘 이 블록이 3커밋 뒤처진 채 남아 있었던 사건을 재발 방지하기 위해.

quarantine.md에 **하루치 스냅샷 10개를 §2a로 옮김** (229줄 추가). 같은 날 두 번 추출 규칙을 써야 했다는 것 자체가 기록할 가치가 있어 섹션 머리에 명기.

**새로운 함정 3개 기록**: 
- `grep -E "$FILT"`가 행 전체에 걸림 (앵커 쓰면 전부 skip)
- 단일 ONT 스트림 1 MB/s 제한
- 로그인 노드 CPU 2개라 SLURM 필수 (sbatch PATH에 없음)

**✏️ Athaliana 매핑률 73.5 % — 조사 완료, 결함 아님.** GenBank EST 디비전의 454 시대 짧은 라이브러리
2개(791,290건, 전체의 51.7 %, 평균 91–92 bp)가 원인이다. 길이를 통제하면 >120 bp 매핑률 96.87 %로
9종 중 4위. 벡터·비Col-0 오염 가설은 기각(`quarantine.md` §1d).

### 다음 작업

1. ACCESS 자원명 확정 (#51·#19·#20 차단) — 승인 나면 바로
2. **#59 seedable sampler** (한 줄, 첫 시드 전 필수) — todo.md §3 참조
3. ~~Athaliana EST 재검증~~ ✅ 완료 — 구성 효과로 확정. 남은 것은 **≤120 bp EST를 증거 계층에서 어떻게 다룰지**(저자 결정, `todo.md` §3)


---

## 1e. ❌ A36 — 결과를 보고 고른 기준이었다 (2026-09-04 A37이 대체)

> 원문(A36, v1.26): *"EST records of 120 nt or shorter are excluded from the evidence layer."*
> A8/A21의 수집 하한을 100 → 121로 올린 개정.

**개정으로 성립하지 않았다.** 외부 검토(Codex)가 지적했고 저자가 받아들여 **A37(v1.27)이 ≥100을 primary로
되돌리고 121은 사전 표기된 민감도 arm**으로 재정의했다.

| 왜 성립하지 않나 | 근거 |
|---|---|
| **계기가 결함이 아니라 결과** | 프로토콜은 "규칙을 적용 불가능하게 만드는 결함"에만 개정을 허용한다. ≥100은 잘 돌았고, 바꾸게 만든 것은 **매핑률**이다. 자기가 측정되는 수치를 개선한다는 이유로 고른 하한은 **수리가 아니라 ascertainment 선택**이다 |
| **A36이 스스로를 반박** | 본문에 *"제외되는 리드는 품질이 나쁘지 않다(MAPQ≥30 93.8 %)"*라고 적혀 있다. 결함 논거가 적합성의 증거 위에 설 수 없다 |
| **"§4–7 불변"은 문장에만 참** | 제외된 레코드는 접합·사슬·callability 분모·분자 수·C2 라벨 어디에도 기여하지 않는다. **고정 문장은 그대로여도 고정 수치는 움직인다** |
| **종별로 불균일** | Athaliana 50.6 % 제거 대 나머지 0.4–9.1 %. **설명하려던 종 간 비교 자체를 바꾼다** |

⚠️ **A36의 측정값은 유효하다.** 두 454 라이브러리(791,290건, 평균 91–92 nt), 길이별 매핑률, NCBI 재대조
결과는 그대로 참이다. 무효가 된 것은 **그 측정에서 기준을 도출한 방식**이다.

⚠️ **표현도 함께 정정됐다**: "벡터·비Col-0 오염 가설 기각"은 과잉이다(§1d 참조). 길이가 플랫폼·라이브러리·
조직·등록 연도와 교락돼 있고, 접근번호 3개 재대조는 그 셋이 안 잘렸다는 것만 보인다. 허용 표현은
**"교차 종 매핑률 이상은 길이 구성으로 설명된다"**. 오염 질문을 실제로 결판낼 검사는 A37.1에 있다.

→ **교훈**: 고정 프로토콜에서 관측을 보고 기준을 바꾸면, 그 기준은 관측의 함수가 된다. 바꿔야 한다면
**primary를 유지하고 새 기준을 사전 표기된 민감도 arm으로** 두는 것이 유일하게 깨끗한 경로다.

---

## 1f. ❌ 기각 — `longread_fetch.sh`의 FILT를 run 컬럼에만 거는 패치 (2026-09-04)

> 제안(Codex 외부 검토 item 2): `tail -n +2 filereport.tsv | grep -E "$FILT"` 를
> `[[ "$run" =~ $FILT ]] || continue` 로 바꾼다. A18.8이 *"filter by parsed columns not whole-row
> regex"* 를 요구하므로.

**취지는 맞고 구현은 틀렸다. Codex 본인이 실측을 보고 철회했다.**

이 저장소에서 `FILT`(4번째 인자)는 run 접근번호 필터가 **아니다**. 최소 세 컬럼에 걸쳐 쓰이고 있었다:

| 호출자가 넘기는 값 | 실제로 걸리는 컬럼 | run 컬럼 매치 |
|---|---|---|
| `OXFORD_NANOPORE` | `instrument_platform` | **0** |
| `PACBIO` | `instrument_platform` | **0** |
| `maize` | `submitted_ftp` (`RS_Sequel_maize_leaf_hq.fastq.gz`, PRJEB22122에서 5행) | **0** |
| `SRR185719(05\|06\|…)` · `ERR4039883` · `DRR42473(1\|2\|3)` | `run_accession` | 매치 |
| `"${r}[[:space:]]"` (`queue_fetch.sh:12`·`ont_parallel.sh:10`) | 행 전체 — 접근번호 뒤 공백으로 고정 | **0** |

**왜 위험한가.** run 컬럼에만 걸면 `OXFORD_NANOPORE`/`PACBIO`로 거르는 데이터셋이 **한 런도 선택되지
않는다**. 그리고 그 실패는 조용하다 — 스크립트는 `0 runs DONE`을 적고 정상 종료한다. 이건 이 프로젝트가
**이미 당한** 실패 양식이다: 2026-09-03 PRJDB38182에서 Col-0 6런이 4시간 동안 사라져 있었고, 트리는
완전해 보였다. 그때의 교훈으로 완료 줄에 FAILED 계수와 exit 5를 넣었지만, **애초에 선택되지 않은 런은
여전히 아무도 세지 않는다** — 그 구멍에 정확히 떨어지는 변경이었다.

**남은 것 — 무효가 된 것은 구현이지 지적이 아니다.** A18.8 위반 세 건은 전부 유효하다. 확정 설계와
연기 사유는 `todo.md` §2c에 있다: `RUN_RE` + `PLATFORM_RE` + `SUBMITTED_RE`로 컬럼을 나누고,
지정했는데 대상 필드가 비면 `UNRESOLVED`, 낡은 4번째 인자와 **선택 행 0건**은 exit 6으로 죽인다.
호출자 5개를 같은 커밋에 고치고, 저장해 둔 `filereport.tsv`를 픽스처로 검증한 뒤 교체한다.

**교훈.** *"조항이 무엇을 요구하는가"를 읽는 것과 "이 코드에서 그 인자가 실제로 무엇을 하는가"를 아는
것은 다른 일이다.* 조항 해석만으로 패치를 쓰면, 그 패치는 조항에는 맞고 코드에는 틀릴 수 있다.
여기서는 필터 네 개를 실제 `filereport.tsv`에 걸어 컬럼별 매치 수를 세는 것으로 1분 만에 갈렸다.


## 2c. 2026-09-03 심야 ~ 2026-09-04 낮 스냅샷 (16:55 ~ 12:48)

> ⚠️ **하루 만에 또 12개.** §2b가 "하루 세 번째 추출"이라 적었고, 이것이 네 번째다.
> [guide/handoff-hygiene.md](https://github.com/wyim-pgl/wiki)에 관측을 남겼다 — 추출 규칙이 잘 도는 것이
> 아니라 **훅 주기가 세션 속도를 못 따라간다**는 신호로 읽는 편이 맞다.
>
> 원문 그대로이며 삭제가 아니다. 여기서 시작하지 말 것 — 정본은 `resume.md`의 📌 블록이다.

<!-- resume.md 37–356줄 원문, 2026-09-04 12:5x 추출 -->

## ▶️ 이전 스냅샷 (2026-09-04 12:48)

### 3번: `est_align.sbatch` 수정 완료 — 커밋 `0272baf`

§4·§3.3 요구사항 세 가지 반영:

1. **PAF 보존** — BAM만 남고 `--cs=long` PAF가 없었음. 두 번째 패스(`-x`) 추가
2. **fail-closed 검사** — `PIPESTATUS[0]`만 봐서 samtools sort 실패를 놓쳤음. 이제 모든 단계 상태 확인 + `samtools quickcheck` 통과 필수
3. **PROVENANCE.tsv** — 종마다 `revision/results/evidence/PROVENANCE.tsv`에 도구 버전·명령·날짜·입력/출력 md5 기록

**contig alias 표는 코드에만** — §3.3 항목 1이 `1…Mt/Pt` 명명 표를 요구하지만, 완성된 BAM 이름 변경은 별도의 감사 가능한 단계이므로 아직 미루었습니다.

### 발견된 이슈: A. thaliana FASTA md5 불일치

| | md5 | 파일 |
|---|---|---|
| §1 동결 객체 | `987ca803…` | `revision/data/TAIR10/TAIR10_genome.fa` |
| 정렬에 쓴 것 | `513e5ef3…` | `genomes/Athaliana_167_TAIR10.fa` |

A. thaliana는 학습 종이면서 B1 복제 종이라, 권위 경로가 갈립니다. PROVENANCE 주석에 사실을 적어 두었습니다.

### A. thaliana 재트림 완료

- Primary (≥100 nt): keep **1,032,698** · drop 483,107
- 민감도 arm (121 nt): keep 741,649 · drop 774,531
- 차이 29만 레코드 — 예상 일치

### 다음 작업

1. 나머지 8종 재트림 완료 대기
2. Primary arm으로 9종 재정렬 (BAM + PAF + PROVENANCE 포함)
3. `longread_v1` 재동결 — SRR25203456 제외 행 추가

## ▶️ 이전 스냅샷 (2026-09-04 12:29)

### 1번 완료 — A37 (v1.27) 프로토콜 개정

A36(v1.26)을 제거하지 않고 **A37(v1.27)로 supersede**:
- ≥100 nt가 primary로 복귀 (A8/A21 규칙)
- 121 nt는 민감도 arm (명시적 플래그 필수)
- A36에 ❌ SUPERSEDED 배너 추가

**근거**: A36 자체가 "제외 리드는 품질 나쁘지 않다(MAPQ≥30 93.8%)"고 적혀, 결함이 아니라 매핑률로만 선택됨. 프로토콜 개정 규칙 위반.

**A37 요구사항**: 쌍대 보고(EST 수치마다 두 arm), 길이 구간별 callability·접합·인트론, "길이로 설명됨"(오염 기각 금지), A37.1에 오염 결판 조건 다섯 가지.

**코드**: `DEFAULT_MIN_LEN=100` 복귀, 121은 명시 플래그만. `univec_screen.sh`는 `MIN_LEN` 환경변수로 arm 선택.

**커밋**: `9d81e7a` · 테스트 412 통과

### 다음 작업

1. 2번: 9종 재트림 `--min-len 100` → primary, `.min121.fa.gz` → 민감도 arm
2. 3번: 9종 재정렬 — primary로 §4, 민감도 병렬
3. 4번: `longread_v1` 재동결 (SRR25203456 제외 행 추가)
4. 나머지: `longread_fetch.sh` A18.8, `est_align.sbatch` PAF·fail-closed, 게이트 md5

## ▶️ 이전 스냅샷 (2026-09-04 12:26)

### 핸드오프 불일치 5건 정정 완료

resume.md·todo.md의 모순 해소:
- `412/409` 동시 기재 → **412** 통일 (실측)
- 저장소 커밋 해시 → **`8dbc19d`** 갱신
- **`SRR25203456`** 항목 → "다운로드 실패로 확정, 제외" (todo.md #156)
- §5 즉시 착수 vs #24 미구현 → **"§5는 아직 착수할 수 없다"** 명시 (⚠️ 저자 결정 대기는 아님)
- `4090` 오탐·과거 스냅샷 3건 포함 → 📌 블록 실제 모순만 정정

### univec 재사용 가능 확인

9종 `hits.tsv` 전부 존재(1.2~211 MB). blastn 재실행 불필요 — 재트림만으로 ≥100 primary 확인:
```
Athaliana, Oryza, Solanum, Zea, Vitis, Fragaria, Malus, Citrus, Manihot
```

### 다음 작업

1. 프로토콜 A36 개정 — ≥100을 primary로, >120은 민감도 분석 재정의(문안 검토)
2. 9종 재트림 `--min-len 100` → `est.univec.fa.gz`(primary) + `.min121.fa.gz`(민감도)
3. 9종 재정렬 — primary로 §4 실행, 민감도 arm 병렬
4. `longread_v1` 재동결 (SRR25203456 제외 선언 행 추가)
5. 나머지: `longread_fetch.sh` A18.8, `est_align.sbatch` PAF·fail-closed, staging 게이트 md5 검증

## ▶️ 이전 스냅샷 (2026-09-04 11:47)

### #15 재현성 검사 완료 — 세 번째 기준 충족 ✅

다른 날짜·다른 커밋(`0aa5fab` → `e36f051`)으로 재빌드 완료. **데이터 테이블 6개 내용 해시 전부 일치**:

| 테이블 | 행 | 내용 해시 |
|---|---|---|
| geneList | 10,454 | cc0d2ed8f309b12b ✅ |
| gene_key_map | 27,416 | 388922bfeb587d37 ✅ |
| gene_split | 334,642 | 61766859a1d712b4 ✅ |
| rejected_records | 148 | d579489599c14dce ✅ |
| tile_blocks | 121 | 7f31e5f9506855ec ✅ |
| window_genes | 73,901 | e857bc9160575a5a ✅ |

`build_manifest` 차이는 provenance 필드뿐(`git_commit`, `built_at`). 빌더 파일 4개는 세 커밋(`0aa5fab`, `fea0d1d`, `e36f051`)에서 byte-identical 확인됨. DuckDB 파일 md5는 재현 안 됨(페이지 레이아웃·삽입 스레딩).

### 발견: 동결 DB의 빌드 출처

동결 기록에 따르면 B5는 두 커밋으로 빌드됨 — `0aa5fab`(6종) + `fea0d1d`(3종). 빌더 파일이 세 커밋에서 동일해 결과에 영향 없으나, **"동일 입력" 정의에 코드 버전이 포함된다는 걸 모르면** 검사가 허위 통과할 뻔했음.

### 개선: 검사 스크립트 수정

- `build_manifest` 해시에서 provenance 필드(`git_commit`, `build_driver`) 제외
- VOLATILE(`built_at`, `file_md5`, `wall_seconds` 등)과 provenance 구분
- git_commit 다를 때 경고+빌더 확인 명령 출력

### 다음 작업

- Codex 리뷰(`codex_review_out.txt`, 6,092줄) 정리 및 요약
- #15 커밋 및 이슈 종료 반영
- 위키 갱신

## ▶️ 이전 스냅샷 (2026-09-04 11:25)

### C2 트랙(#45 miniprot) 입력 스테이징 완료

OrthoDB 전송 완료(3.0 GB, md5 `453cb32b` 검증), training_input/ 폴더 구성 — **40개 파일, 28.9 GB**:
- `db/` — 동결 B5 (272,224행)
- `genomes/` — 학습 9종 (심볼릭 링크)
- `protein/` — OrthoDB 필터본 (C2용)
- `splits`, `configs`, `freeze` — 매니페스트·레시피·동결 기록

MANIFEST.tsv (재생성 가능, `bash make_manifest.sh`)로 적재 내용 기록, 파일별 bytes·md5·원본 경로·용도.

### 발견·해결: 게놈 폴더 혼입

첫 적재에 게놈 13개가 모두 들어갔으나, **Zmays·Slycopersicum (held-out 테스트 종)과 BrapaO·Lsativa (프로젝트 외)가 포함** — `*.fa` 와일드카드 전부 가져온 것. 제거하고, 스크립트에 fail-closed 게이트 추가(`b5_species_v1.tsv` 대조, 어긋나면 거부+종료). 일부러 Zmays 삽입해 게이트 작동 확인 완료.

### 장독 정렬 상태 확인

BAM 0개, §3.3 감사 산출물 0개 — 정렬 미실행. #25(선행 감사)가 필요하므로 별도 정리. 감사 도구(`content_sha256` 함수, `merge_b5_databases.py`)와 재빌드 입력사본(pgl-gpu)은 #15 재현성 검사 준비됨.

### 다음 작업

- **#45 miniprot 돌리기** — 입력 준비 완료
- **#15 재현성 검사** — 동결 시 기록한 내용 해시로 재빌드 비교 (방법 확인)
- **#25 정렬 전 감사** — 장독 정렬 전 필수

## ▶️ 이전 스냅샷 (2026-09-04 00:36)

### 위키·기록 갱신 및 `SRR25203456` 우회 경로 규명

위키: `guide/pronghorn.md` 섹션 추가, README·index·log 갱신 (`d527cc6`).
`quarantine.md` 정정: ✏️ S. italica 항목이 사실이 아님. 드라이버는 2일 4시간 살아 있었고 O. sativa에서 17시간 51분 붙잡혀 있었음(정지 요청 전).

### `SRR25203456` (O. sativa RNA-Seq 13M 리드) 우회 경로 검증 — #68

ENA 경로(`ftp.sra.ebi.ac.uk/...fastq.gz`) 실측 확인 → **디렉토리**:
- HTTPS: `301` + 같은 경로`/`, 받으면 802 B 인덱스 HTML
- FTP: `curl 78` 파일 없음

AWS S3 경로 발견·검증:
- `https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR25203456/SRR25203456`
- `200` · `Accept-Ranges: bytes` · 재개 검증 · 12.1 MB/s (9.39 GB ≈ 13분) · SEQ 일치

### 구현은 아침 판단 — 손 수신 파일의 검증 문제

받을 수 있지만 저는 받지 않았습니다. 손으로 받으면 기기·subreads·library_strategy·모델·ENA md5·**실패 계수** 검증을 전혀 안 거칩니다. `.DONE` 파일이 검증된 증거와 구분되지 않습니다. 드라이버에 SRA 폴백(`vdb-dump` 해석) 구현 필요. 근거·방향·재동결 조건은 #68에 기록.

### 현황

- 저장소: `1653d6d` · 미커밋 0 · 테스트 412 passed
- 프로세스: 0개 · 프로토콜 v1.26 · 이슈 열린 것 49개

### 다음 작업

- **바로**: `todo.md` §5 접합 호출 (#24)
- **판단**: #68 (SRA 폴백), ACCESS 자원명 (#51·#19·#20)

## ▶️ 이전 스냅샷 (2026-09-03 23:53)

### 드라이버 11개 버전 관리 편입

`evidence/` 디렉토리의 11개 스크립트(`longread_fetch.sh`, `queue_fetch.sh`, `ath_retry_loop.sh`, `est_align.sbatch`, `training_run_all.sh`, `training_run_ont_missing.sh`, `longread_run_all.sh`, `longread_after_est.sh`, `ont_parallel.sh`, `update_provenance.sh`, `rerun_training_after_finish.sh`)를 `transgenic/revision/scripts/evidence/`로 이동하고 원래 경로를 심볼릭 링크로 변경. 커밋 `7da946c`.

### 무중단 전환 검증

실행 중인 다운로드 루프(`ath_retry_loop.sh` → `queue_fetch.sh` → `longread_fetch.sh`)가 심볼릭 링크 교체(원자 `mv -T`) 후에도 정상 진행 중(4/7 완료, 프로세스 6개 활성).

### README 기록

**함정 2건**: 도구가 쓴 파일은 0644로 나오므로 위에 `mv`하면 실행 비트 손실; `pkill -f <패턴>`은 자신도 죽임. **현행 안 됨**: `rerun_training_after_finish.sh`는 자기가 실행할 스크립트 완료를 기다려 발동 불가(보관용만).

### 미해결 항목 상태 갱신

`todo.md` §3 미해결 항목 1건(`evidence/` 버전 관리) 해소. 남은 병목: ACCESS 자원명(#51·#19·#20) 및 #58·#55·#56·#61·AtRTD3.

### 다음 작업

- 라운드 3 모니터링(진행 중, 남은 전송 ~1.1 GB)
- 루프 완료 후 `DATASET_ROLES.longread_v1.tsv` v2 동결

## ▶️ 이전 스냅샷 (2026-09-03 23:48)

### 근본 해법 배포 — 재개형 페처 실행

원본 `.fastq.gz` 먼저 수신(curl -C -), 변환은 별도 진행. 중도 실패해도 byte 범위 재개 가능.

**검증 완료**: kill/restart로 재개 테스트, ENA md5 대조 4건 전부 일치. 현황: 4/7 완료, 라운드 3 진행 중, 남은 전송 약 1.1 GB.

### 보고 결함 발견 및 고침 — 실패를 세지 않던 문제

완료 줄이 `DONE`·`skipped`만 세고 **`FAILED`는 계수 안 함**. 파이프로 먹히는 while 루프 서브셸에서 카운터 변수가 밖으로 살아남지 못하는 구조 결함.

**조치**: `<run>.FAILED` 표식 파일 + 계수 로직 개선. 이제 `finished: N runs DONE, M skipped, K FAILED, U UNRESOLVED`로 출력, exit 5(FAILED 있음)/4(UNRESOLVED)/0(완료).

계수 로직을 5개 케이스로 격리 검증 — 낡은 `.FAILED`가 남아도 그 런이 나중에 성공하면 0으로 자정됨.

### ⚠️ 미결 항목

이 수정 3건이 모두 **`evidence/longread_fetch.sh`(저장소 밖)**이고 버전 관리를 받지 않습니다. `todo.md` §3의 저자 결정 항목입니다.

### 다음 작업

- 라운드 3 모니터링 (진행 중)
- 루프 완료 후 `DATASET_ROLES.longread_v1.tsv` v2 동결
- 저자 결정: 파일을 저장소 내 옮길지, 스크립트 버전 관리 체계 개선할지

## ▶️ 이전 스냅샷 (2026-09-03 18:10)

### 근본 해법 배포 — 원본 선수신 + 재개 구조

2단계: ① `curl -C -`로 원본 `.fastq.gz` 수신 ② 변환. 이전과 달리 중도 실패해도 원본 조각이 남아서 byte 범위 재개 가능.

**재개 검증**: SRR36603452, 90초 받고 강제 종료(8.9 MB) → 재시작 → 11.5 MB로 이어받음.

부수 개선: `gzip -t` 무결성 확인, `fastq_md5` 대조 추가 (#6), 로그 `FAILED` 대신 진행 상황 기록.

### 진단 반전 — ENA 스로틀링, HTTP/2 문제 아님

초기 가설: `--http1.1`이 HTTP/2 다중화 문제 해결(측정: h2는 71 KB에서 실패, http1.1은 53.9 MB 성공).

30분 뒤: 모든 프로토콜 동시 6–11 KB/s 무너짐. 오전에 같은 호스트가 150 GB를 내줬으므로 우리 연결 스로틀링으로 추정. 스크립트 주석에서 단정 제거.

### 현재 상태

12라운드 · P=2 · 30분 대기로 재기동 중. 중도 실패해도 조각이 보존되므로 진행분이 버려지지 않습니다.

### 다음 작업

- 루프 진행 모니터링 (5분 간격)
- 완료 후 `DATASET_ROLES.longread_v1.tsv` v2 동결, 위키 업데이트

## ▶️ 이전 스냅샷 (2026-09-03 17:25)

### 7런 재시도 실패 — 병렬 속도 무너짐 원인 확정

17:16 DRR807190/196, SRR36603452 재시도 모두 FAILED. 각 런 5.63 GB.
- longread_fetch.sh: `.part`는 변환 결과물이라 재개 불가, 실패 시 삭제(:56)
- 7스트림 병렬에서 .part 정지 (55초 간격 측정 무변화), 단일 스트림 2 MB/s 정상

### 조치: 병렬도 7→2, 재시도 루프 배포

ath_retry_loop.sh 작성: 30분 대기, 8라운드 자동 재시도. 병렬도 감소 이유 주석 명기(대용량에선 단일 빠름이 다중 느림보다 성공률 높음). 감시 개선: 중복 줄 도배 → 신규 이벤트만.

### 근본 개선 (미결)

fetcher 2단계 구조: ① `.fastq.gz` 수신(curl -C - 재개) ② 변환. 디스크 29 TB 여유. 루프 8라운드 소진 시 제안.

### 다음 작업

- 라운드 1 모니터링 (17:23:28 시작, 2런씩)
- 7런 완료 후 §5 접합 호출
- 루프 최대값 초과 시 fetcher 구조 개선

## ▶️ 이전 스냅샷 (2026-09-03 17:04)

### 다운로드 상태: 멈춤 확인, 7런 미완료 발견

프로세스 0개, 최근 30분 갱신 0건, 마지막 활동 15:08. **명확히 멈춰 있다.** 그런데 "멈춤" ≠ "완료" — A. thaliana 두 데이터셋이 빠졌다.

| 데이터셋 | 계획 | 받음 | 로그 상태 |
|---|---|---|---|
| `col0_nitrogen_PRJDB38182` | 6런 | 0 | DRR807190/191/192/196/197/198 전부 FAILED (13:01–13:03) |
| `buffer_control_PRJNA1393863` | 1런 | 0 | SRR36603452 FAILED (14:24) |

### 원인: "0 runs DONE, 0 skipped" 로그가 실패를 숨김

로그는 `finished: 0 runs DONE, 0 skipped`라고 기록했는데, 이는 "받을 게 없었다"와 "다 받았다"를 구분하지 못한다. 프로세스·`.DONE` 수만 보면 완료로 오독된다. 계획과 대조할 때만 보인다. 오늘 반복된 패턴.

### 실패는 일시적: URL 정상 응답, 재시도하면 수집 가능

두 데이터셋 URL 모두 HTTPS 200 응답, 실제 데이터 흐름 확인됨. 구조적 오류(경로 디렉토리 같은 것)는 아니다. **재시도하면 `.DONE` 표식이 중복을 막아 7런만 수집된다.**

⚠️ **그 전에 `DATASET_ROLES.longread_v1.tsv` v2 동결 금지** — 불완전한 세트를 완전한 것으로 잠그게 된다.

### 다음 작업

- 7런 재시도: `col0_nitrogen_PRJDB38182` Col-0 6런(~35 GB) + `buffer_control_PRJNA1393863` 1런
- 재시도 전 `DATASET_ROLES.longread_v1.tsv` v2 동결 미연기
- 수집 완료 후 §5 접합 호출 진행

## ▶️ 이전 스냅샷 (2026-09-03 16:55)

### 테스트 수집 오류 해결 — 파일명 충돌로 중단되던 pytest 복구

`tests/test_geenuff_qc.py`와 `revision/scripts/tests/test_geenuff_qc.py`가 같은 베이스명을 가져 pytest 수집이 중단되던 문제 원인 규명. 낡은 파일 4건 검토: 2건(`test_loss_mask_policy_a22`, `test_read_flags_tsv`)은 리포 전체에서 유일한 A22 정책 및 플래그 읽기 커버리지이므로 `tests/`로 이관(출처·미이관 사유 주석 명기), 2건은 추측 스키마 기반·폐기.

### 테스트 전수 실행 확인

`pytest -q` 한 번에 **409 passed, 1 skipped** — 수집 오류·실패 0건. 이전 상황(155건만 실행)과 달리 모든 테스트 루트 활성화. 커밋 `73397ac` 푸시.

### #46 팔로우업 — A22 단위 테스트 복구 기록

이슈 종료 근거의 한 줄이 부정확했음: A22 정책 함수의 유일한 단위 테스트가 미실행 파일에 있었음을 코멘트로 명기. 현재 통과 중.

### 상태

저장소 **`73397ac`**, 테스트 **409건**(기존 155 → 완전한 수집). 📌 블록의 테스트 기준값·검증 명령 갱신.

### 다음 작업

- resume.md 크기 재증가(495줄, 스냅샷 55개) — 오전 규칙대로 오후 스냅샷 추출 시점 검토
- **#51·#19·#20**: ACCESS 자원명 (저자 결정)
- **#58·#55·#56·#61·AtRTD3**: 판단 대기


---

## 2a. 2026-09-03 하루치 스냅샷 (10:53 ~ 13:28)

> ⚠️ 훅이 자동으로 쌓은 당일 스냅샷이다. 하루 만에 10개가 쌓여 정본 블록을 밀어냈다 —
> 추출 규칙(`guide/handoff-hygiene.md`)을 같은 날 다시 적용한 사례.
> **지금 상태는 [`resume.md`](resume.md) 📌 블록.**

## ▶️ 현재 상태 스냅샷 (2026-09-03 13:28 — 여기서 다시 시작)

### EST 정렬 게이트 열음 — DATASET_ROLES.est_v1.tsv §1 등재

`DATASET_ROLES.est_v1.tsv`(11행, md5 `a6375713`)를 프로토콜 **§1 고정 객체 표에 등재**, 커밋 `2bf33d8`. EST는 9/1 수집 완료(안 변함), 장독은 지금도 들어오는 중이라 분리 고정. 각 범위는 §1에 행이 없으면 정렬 불가.

### EST 정렬 9종 배열 잡 제출

프로토콜 §3 명령 그대로: `minimap2 -t 32 -ax splice:hq` minimap2 2.28-r1209, 종당 1잡. **잡 `6147674_[0-8]`**. Fail-closed 역할 검사 사전 확인: `est/Athaliana` ✅ `c2_training_eligible`, `est/Zmays` ⛔ held-out 테스트 종(배열 제외). 각 종 `PROVENANCE.txt`에 게놈·EST md5·건수·도구·명령·벽시계·BAM md5·역할 매니페스트 md5 기록.

### 큐 상태 — 다른 프로젝트 잡들 대기 중

**QOS 동시 실행 ~22잡 제한**, 메모리 과다 예약 압박(32 GB만 요청). 다운로드는 계속 진행 중 **28 스트림, 134 GB**.

### 다음 작업

① 정렬 완료 대기 → #45(miniprot) · #24(B1 채점기) · ACCESS 아이디 · todo.md §3(#58 · #55/#56 · #61 · AtRTD3)

## ▶️ 이전 스냅샷 (2026-09-03 12:45)

### quarantine.md 구조 재정의 — 미해결 항목 격리

`ruleout.md` → `quarantine.md` 이름 변경(최상위 + 저장소 양쪽). 머리말을 사용자 설명 세 용도로 재작성: false positive 분석 근거 · 추가 피해 범위 추적 · 위협 분석(기기/리드 수 조건 맹점). **미해결은 격리소에 두지 않음** — §7(미결 항목) 전부 `todo.md`로 이관, §8(원고·제출 측 기각)은 격리소 유지(§7로 번호 당김).

### Cui 2020 중복 판정 완료

내용 완전 동일(크기·압축 해제 후 md5 같음). `.md5` 파일은 원본이 아니라 **우리 재압축 결과물의 해시** — gzip 헤더 타임스탐프만 상이. 저장소 quarantine.md에 기록.

### 저장소 동기화 완료

README·CLAUDE.md 링크 갱신, 커밋 8b665d6, 푸시. 클라우드 보드: https://claude.ai/code/artifact/9e31511d-261a-4b46-8de3-7d6a7fb45331

### 다음 작업

① 중복 런 처리(PRJNA594286 어느 쪽 정본) ② source_checksum 정의(재압축본 vs ENA md5) ③ 검증기에 런 중복 검사 추가 ④ EST 정렬 진행.

## ▶️ 이전 스냅샷 (2026-09-03 12:34)

### #5 매니페스트 생성 완료 — 175행

`revision/scripts/make_dataset_roles.py` 작성, 저장소에 저장. 스키마 A18.3 구현: `c2_training_eligible` 91런 · `b1_validation_only` 33런 · `excluded` 51런. 검증 실패 시 파일 미작성(fail-closed). Zenodo 세트 구조(flnc.DONE/gff.DONE)도 처리.

### PRJNA594286 중복 발견 — 내용은 동일, md5 다름

`SRR10611193/94/95` 세 런이 `training/ont/Athaliana/`와 `ont/Athaliana_cui2020_PRJNA594286/`에 각각 존재. 압축 풀어서 비교: **내용 동일**(gzip 헤더 타임스탐프 차이로 md5만 상이). A18.3 수용 기준 위반: `(dataset, run)` 중복 검사만 있었고 **데이터셋 간 런 중복**은 미포함.

### source_checksum 구조 문제

현재 파일명.md5는 우리가 재압축한 결과물 해시. ENA filereport의 `fastq_md5`는 한 번도 사용 안 됨. #60에서 눈에 띤 지점. 원본 대조 불가능한 상태.

### 다음 작업

① 중복 런 처리 방침 결정(어느 쪽 정본) · ② `source_checksum` 정의 명확화(재압축본 vs ENA md5) · ③ 검증기에 런 중복 검사 추가 · ④ EST 정렬 진행.

## ▶️ 이전 스냅샷 (2026-09-03 12:15)

### 617런 분류표 생성 · Codex 검토 진행 중

`ath_ont_candidates_20260903.md` 작성: ENA 전수 617런을 52개 study로 재분류. 프로토콜 §A14 · §A18.3 · 성숙 mRNA 조건 적용. **4개 주요 의심점** 명시—taxon/표본명 모순(PRJNA788163 전부 *A. thaliana*로 등록되었으나 sample_title = `rice_rep1`·`soybean_rep2`), study 제목 불일치(col-0 표기 vs 변이체 혼입), 비공개 설명(GEO, PRJDB38182), 유전형 코드화(PRJCA018751). 특히 **가장 큰 두 study(238 GB + 122 GB)가 쓸모 없을 가능성** 명시 후 Codex(gpt-5.6-sol)에 전달.

### 병렬 다운로드 계속 진행

16 프로세스, 2/21런 완료, 누적 116 GB(예상 21시간 남음).

### 다음 작업

Codex 검토 완료 대기. 리뷰 결과 확정된 런만 다운로드 큐에 추가. ACCESS ID 정보(Delta vs DeltaAI, allocation ID, 로그인 호스트·사용자명, MFA) 수집 후 #51 계층 메모리 재측정.

## ▶️ 이전 스냅샷 (2026-09-03 11:58)

### 병렬 드라이버 완성 — 대역폭 3배 향상, 21시간 → 7시간

`ont_parallel.sh` 구현 완료: 8병렬 스트림으로 `longread_fetch.sh` 호출. 버그 두 개 해결—① xargs 중첩 따옴표(인자 비움) → while 루프 ② `grep -E "^SRR…$"`(행 전체 매칭) → 탭 경계 고정. 단일 스트림이 약 1 MB/s 제한(ENA 또는 파이프라인)이라 병렬화로 **3 MB/s** 달성. 남은 76 GB 21런 약 21시간 → **7시간**으로 단축. 백그라운드 진행 중.

### 부재도 증거 — 기록 방침 변경

#60 "손실" 표현 제거, **의도된 대조군**으로 기술. *S. bicolor*(ONT 없음, subreads 제외)와 *B. distachyon*(원 없음)은 서열만 학습하는 기준선, C2 라벨 효과 측정 근거.

### A. thaliana 부족 — ENA 전수 탐색 필요

현재 6런 18 GB(Cui 2020, DRP009401만 2세트). 검증 전용 제외 후 Col-0 ONT 전사체 매우 적음. ENA 조회로 학습용 추가 세트 탐색(현 다운로드 영향 없음).

### 다음 작업

병렬 다운로드 진행(7시간 예상). ENA Col-0 ONT 조회. 76 GB 완료 후 A. thaliana 추가 세트 통합.

## ▶️ 이전 스냅샷 (2026-09-03 11:45)

### #63 library_strategy 화이트리스트 추가 — 완료

`longread_fetch.sh`에 정규식 필터 추가: `STRAT_ALLOW='(RNA-Seq|FL-cDNA|ncRNA-Seq|OTHER)$'`. WGS 런들이 이제 사유와 함께 건너뜀. 라이브 153런 전부 통과 확인.

### SRR25203456 재시도 실패 — ENA 구조적 문제

문제 파일이 `301` 리다이렉트로 **자신의 디렉토리 목록(HTML)으로** 도는 중. ENA API는 9.4 GB로 광고하지만 실제 경로에는 디렉토리가 있음. `curl -sL`이 HTML을 받고 `zcat`이 실패해 20바이트만 남음. 네트워크 재시도로 해결 불가(구조적 문제).

### 대체 경로 — SRA 툴킷

`/data/gpfs/assoc/pgl/bin/conda/conda_envs/sra/bin/prefetch` 설치돼 있음. NCBI에서 직접 받기 가능. 지시 있으면 전환.

### ACCESS 정보 필요

#51/19/20 풀기 위해: ① 자원명(Delta vs DeltaAI) ② allocation ID ③ 로그인 호스트·사용자명·파티션 ④ MFA 대응(직접 로그인 vs 키 설정). 계층 메모리 재측정 → 크레딧 요청량 확정 필요.

### 다음 작업

ACCESS 정보 확인 후 #51 계층 재측정 · 크레딧 계산. 그 전에 **#59(재개 셔플) 한 줄 처리** — ACCESS 첫 시드 전 필수.

## ▶️ 이전 스냅샷 (2026-09-03 11:39)

### resume.md 4,901줄 → 216줄로 압축 — quarantine.md 격리 완료

4,690줄의 이전 스냅샷(2026-08-10~08-11)을 `/data/gpfs/assoc/pgl/data/Transgenic/quarantine.md`로 옮기고 링크만 남겼습니다. 무손실 확인: 원본 3,355개 비어있지 않은 줄 전부 보존, 누락 0. 백업 `resume.md.bak-20260903-preruleout` 기록.

### 낡은 주장 마커 삽입 — 9개 어긋남 표시

마커 추가: 이슈 범위 `#4–#54` → `#4–#63`, 커밋 `b5e4471` → `e36f051`, 테스트 145건 → 155건, 디스크 239GB → ~216GB, "여기서 다시 시작" 반복(3주 묵은 상태 유도). quarantine.md §1에 비교표 제시, 나머지 121개 블록은 일괄 격리.

### 위키 · transgenic/quarantine.md 상호링크

`guide/handoff-hygiene.md`에 새 절 추가: 마킹만으로는 수천 줄 대응 불가, **추출은 삭제 아님 · 목적지가 원문·마커 보존 · 이동 무손실 검증** 원칙. 저장소 quarantine(데이터·설계·코드)과 랩 내부 이력을 분리해 둠(커밋 `a9e208b`).

### 다음 작업

#51·#19 (ACCESS 자원), #58 (`<empty>` 의도 확인), #55·#56 (재빌드 여부), #61 (Methods test set).

## ▶️ 이전 스냅샷 (2026-09-03 11:06)

### #60 subreads 제외 — 완료 (이슈 종료)

43 GB를 **삭제 없이** `evidence/RETIRED_DO_NOT_USE/training_pacbio_subreads_20260903/`로 경로 격리. 그 디렉토리 README에 종·데이터셋별 내역과 되돌리는 명령 기록. `longread_fetch.sh`는 이제 `_subreads`를 이름으로 거부. 파급: *S. bicolor* 장독 증거 0, *B. distachyon*은 원래 0.

### #62 데이터셋 폴백 — 완료 (이슈 종료, `e36f051`)

`isoformDataHyena`에 `strict=` 인자 추가. 트레이너의 고정 레시피가 `strict=True`를 넘김. strict 모드에서 세 경로 각각 행 번호·인덱스·split·gene model·원인을 담아 예외 발동. strict 아니면 대체 동작 그대로(published 동작 보존). 테스트 5건 추가, torch 없는 곳 skip — pgl-gpu 전부 통과. **전체 160건 통과**(`pytest -q tests`).

### CLAUDE.md 마킹 — 완료, git exclude 보고

12곳 마커 추가. 3개가 코드 펜스 안에 들어가 명령을 끊었으므로 펜스 밖으로 이동. 오타 2개 수정(어깋나는→어긋나는, 엉뚜한→엉뚱한). 펜스 정합성 확인 완료. **다만 `.git/info/exclude`로 git 추적 제외돼 있음** — 이 머신 디스크만 적용, 커밋 안 됨, 새 클론 시 사라짐. 의도된 설정 같아 강제 추가 안 함.

### TRAINING_EVIDENCE_v1.md — PacBio 표 표시

표의 PacBio 행 전부(Sbicolor, Vvinifera CCS)에 ❌ SUPERSEDED 배너 추가. 규칙 문단(L3–5)이 이미 subreads 배제를 명시했는데 표와 모순됨. 규칙이 맞았음.

### 다음 작업

ACCESS 자원명·계정 — #51·#19 블락. #58(`<empty>` 의도 여부) · #55/#56 재빌드 여부(1.5h) · #61 Methods test set 정의. 검증 세트 PacBio 2건. ONT 수집 계속 중(세 번째 데이터셋).

## ▶️ 이전 스냅샷 (2026-09-03 10:53)

### quarantine.md 8절 생성 완료 — README 12곳 격리

`guide/handoff-hygiene.md` 규약대로 원문은 유지하고 마커만 추가. `❌ SUPERSEDED` / `⚠️ CAUTION (2026-09-03)` 12곳을 README에 달고 `quarantine.md` 링크. 절 구성: §1–§3 요약(R1–R11, D1–D7, M1–M11), §4–§7 상세(C1–C13, 이번 세션 6개 오류, 기각의 기각, 미결). 커밋 `61e21f3`.

### README 최대 함정: random_split 75/10/15 (이미 사용 중)

§3 **M5**: RC 증강 후 정방향·역상보 쌍둥이가 train/test로 갈라져 상동도 분할. B5가 이 문제를 해결하는데, README가 권하는 `_RTX4090.py` / `_GB10.py` (:669, :682)는 **아직도 이걸 씀**. 같은 줄의 "OOM-safe batch skipping"은 A35가 금지한 동작.

### CLAUDE.md 미건드 — 9곳 어긋남을 R11에 표 정리

에이전트 첫 읽기 파일이라 영향 큼. §1–§7은 오늘 코드로 검증했으나 CLAUDE.md는 **지시 파일**이라 승인 필요. 92% F1 / 49152bp / pytest 없음 등 9개 어긋남 명시. 말씀하시면 같은 마킹 방식 적용.

### §8 성격 다름 — 원 기록만 옮김

72.5% 장독, 2배 중복 원인, 487 transcripts 좌위 단위 등은 `resume.md` / `peer_review_report`에서 가져온 것. **이번 세션 재측정 없음**을 절 머리에 명시. 테스트 155건 통과.

### 다음 작업

- CLAUDE.md R11 마킹 여부 결정 (승인 대기)
- #59 (에폭 재개 셔플 불일치) 한 줄 수정
- #55/#56 B5 재빌드 (1.5h, rc 제약 영향 측정)

## ▶️ 이전 스냅샷 (2026-09-03 01:34)

### #50 B5 동결·신규 이슈 8건

동결 완료: `b5_full_v1.db` 272,224행, 검증기 위반 0. **nextval() ORDER BY 함정** 발견·폐기: 병렬 스캔 중 시퀀스 평가가 ORDER BY 전에 진행되어 30만 행에서 rn 262,144–300,000개 오류. `row_number() OVER`로 재실행. Codex·Kimi K3 독립 검증(내용 일치, 제 합계 오타 지적).

### 신규 이슈 #55–#62

#55 tie-break 97타일 · #56 V. vinifera lncRNA 6,205개 탈락(train 유입 1,210개, 서열 89.5% ACGT) · #57 tier_margin·orthogroup · #58 `<empty>`에 `<s>` 결여 25,159타일 · **#59 에폭 중간 재개 셔플 불일치** · **#60 PacBio 51런 전부 subreads 43 GB** · **#61 명목 test 유전자 16.3% 만 라벨** · #62 실패 행 무작위 대체. #53 체크박스 4건 완료(Hyena 129,024).

### 부수

장독 다운로드 멈춤: S. italica 9/2 23:29. ONT 5개 수신 중(1.1 GB), PacBio 줄 #60 대기.

### 다음 작업

#59 한 줄 수정 · #58 의도 여부 · #55/#56 재빌드(1.5h) · **#60 subreads 처리** · #61 Methods 기술 · ACCESS 자원 확보.

## ✅ 이 세션에서 끝난 것 (2026-09-03 00:30–01:1x)

### #50 9종 DB 병합·동결 — **완료, 이슈 닫음** (#16도 함께 종료)
`b5_full_v1.db` 272,224행. rn 1..272224 조밀·중복 0·종별 연속 블록. train 198,171 / valid 25,910 / test 48,143. rc 84,079. `gene_split` 334,642행 1회만 복사. `validate_b5_database` **위반 0**, Zmays 0행, Zm/GRMZM 0행, NULL split 0. 파일 md5 `ebf4fb51…`, geneList 내용 sha256 `bf6cde9c…`. 동결 기록 `transgenic/data/freeze/b5_full_v1.{freeze,validate,report}.json` + README.
- 9종 빌드가 **같은 커밋이 아님**: 6종 `0aa5fab`, 3종 `fea0d1d`. 빌더(build_b5.py·gsf_contract.py) 두 커밋 사이 diff는 **비어 있음** — 동결 매니페스트에 둘 다 기록.
- 파일 md5는 재현되지 않는다(DuckDB 레이아웃). 재현되는 것은 내용 해시(행별 md5를 DuckDB 안에서 만들고 정렬해 sha256으로 접음).

### ⚠️ 이 세션 최대의 함정: `nextval()`은 `ORDER BY`를 따르지 않는다
첫 병합은 `INSERT INTO geneList (rn, …) SELECT nextval('row_id'), … FROM (SELECT * FROM src.geneList ORDER BY rn)`였다. **이건 원본 순서로 번호를 다시 매기지 않는다.** SELECT 목록의 시퀀스는 스캔 중에 평가되고, 정렬은 이미 번호가 붙은 튜플에 적용된다. duckdb 1.5.5 실측(원본이 물리적으로 rn 순서인 상태 = 실제 종별 빌드 모양):

| 방식 | 10만 행 | 30만 행(2 스레드) | 30만 행(16 스레드) |
|---|---|---|---|
| 서브쿼리 `ORDER BY rn` + `nextval` | 정확 | 262,144 / 300,000 오류 | 300,000 / 300,000 오류 |
| `row_number() OVER (ORDER BY rn)` | 정확 | 0 오류 | 0 오류 |

행 수·split·행 내용·기본키·`validate_b5_database` 전부 정상으로 보이면서 rn만 의미를 잃는다. 학습도 그냥 돌아간다(`b5_runtime.split_row_numbers`·`isoformDataHyena`가 실행 시점에 인덱스를 다시 만들고 rn을 저장하지 않으므로). 깨지는 건 감사 추적뿐 — 그게 동결의 존재 이유다. 첫 병합은 폐기하고 다시 돌렸다. 회귀 테스트는 **30만 행**이어야 실패한다(3만 행에서는 nextval도 통과한다).

### 새로 만든 이슈 5건
- **#55** 라벨 방출기와 검증기의 tie-break 불일치로 타일 97개 폐기(`window_to_gsf_v3`는 2요소 키로 정렬, `check_caps_v3`는 3요소 키로 검사)
- **#56** V. vinifera lncRNA 전용 유전자 6,205개가 타일링 전에 탈락 — 그중 **valid 634 · test 971**이 A29 leak 마스킹을 우회해 train 타일에 마스킹 없이 남는다. GFF 실측으로 원인 확정(lncRNA 전용 유전자 정확히 6,205개), 20개 수작업 확인 — 주석 오류 아님
- **#57** `tier_margin_unguaranteed` 미구현 · `build_manifest.rejected_reasons` 집계 불가(키에 소수점이 남음) · 검증기의 orthogroup 교차 split 검사가 타일 DB에서 구조적으로 무효(orthogroup_id가 전부 NULL)
- **#58** `<empty>` 라벨만 `<s>`가 없다 — 토크나이저는 2토큰, 계약·스펙·테스트는 3토큰. **25,159 타일(9.24 %; train 22,311)**
- **#59** **에폭 중간 재개 시 셔플이 달라진다** — `use_seedable_sampler`가 꺼져 있어 `accelerator.load_state`가 복원한 중간 RNG로 다른 순열이 뽑히고, `step < resume_step` 건너뛰기가 엉뚱한 샘플을 버린다. 실측: 64개 중 20에서 재개 → 15개 중복·15개 미관측. `use_seedable_sampler=True`면 0·0. **A28 잡 체인이 재개를 상시로 쓰므로 첫 시드 전에 고쳐야 한다**

### 이슈 3건 추가 (01:2x–01:5x)

- **#60 다운로드된 학습용 PacBio 51런 전부가 subreads** — 11개 데이터셋 43 GB, CCS/FLNC 0건. `MODEL_RE`는 기기(Sequel II/IIe)를 맞게 통과시키지만 데이터 산물을 안 보고, subreads 가드 `MAX_READS=5000000`은 실제 세트가 25만–158만 리드라 발동 안 함. ENA에 이 런들의 CCS는 없음. 검증 세트도 3개 중 2개 동일, 그중 `pacbio/Zmays_B73_ccs_PRJNA1470126`은 **이름이 ccs인데 내용은 subreads**. 제외하면 **S. bicolor는 장독 증거가 전부 사라짐**(ONT 없음).
- **#61 명목 test 유전자의 16.3 %만 실제로 라벨된다** — test 15,005/91,787(종별 13.1–19.3 %) · valid 7,316/29,153 = 25.1 % · train 196,592/213,702 = 92.0 %. 유전자 split(orthogroup)과 타일 split(위치 블록)이 독립이라 test 유전자는 test 블록에 들어갈 때만 라벨된다. 누수가 아니라 설계의 곱이지만 **논문의 test set 기술이 이 정의를 따라야 한다.** A29 불변식 3건은 0 위반으로 확인해 report checks에 편입(`c6bc1b1`).
- **#62 데이터셋이 실패한 행을 무작위 다른 행으로 대체** — `datasets.py:306-311`·`:329-335`·`:339-344`. A35가 고친 배치 루프 한 단계 아래에 같은 패턴. 동결 DB에서는 현재 도달 불가(빈 서열 0·NULL 라벨 0·weight 0 행 0) — 발동 전에 고칠 수 있다.

### ✏️ #56 서열 수준 확인 완료

`gene_key_map` 좌표로 dropped 유전자를 forward train 타일과 조인해 타일 서열을 직접 읽었다. valid/test split인 no-CDS 유전자 **1,210개**(3,149 쌍)가 train 타일에 완전히 들어가고 **그중 737개가 test**. 표본 3,000쌍 중 **2,684(89.5 %)가 유전자 구간 전부 평문 ACGT**, N 비율 중앙값 0.0000. 예: `Vvi000879`(test)는 train 타일 세 개 안에서 613 bp 전부 N=0. 코드 추론이 아니라 실측으로 확정.

### #53 체크박스 2건 해소
- 라벨 캡: 병합 DB 최대 라벨 5,920토큰(캡 8,192), 캡 도달 0행. 트레이너의 `GFFTokenizer`로도 최장 4,198. **조용한 절단 없음.** (부산물이 #58)
- Hyena 필터 길이: 12개 층 전부 `pos_emb.z (1, 129024, 5)`·`seq_len` 129024, 공유 인스턴스 없음. CPU forward가 세 계층 전부 통과(129,024 → `(1, 129024, 768)`). A34 게이트 `227/339, missing 112 = layers 8-11`. **backward는 ACCESS 필요.** ⚠️ `named_buffers()`는 기본 `remove_duplicate=True`라 층 하나만 보인다 — `remove_duplicate=False` 써야 한다.
- 마지막 불완전 누적 그룹: 버려지지 않는다(`micro_done`이 에폭 간 누적). 다만 `steps_per_epoch`가 에폭마다 새 그룹을 가정해 `t_total`이 45,430 vs 실제 45,414로 과다 계산(LR이 0 대신 최고치의 0.035 %에서 끝남).

---

## 2. resume.md에서 옮겨 온 이력 (2026-08-04 ~ 2026-09-03 01:1x)

> ℹ️ 이 절 안의 `🧊 ARCHIVED BELOW` 배너 2개는 원본 resume.md 스택의 중첩 경계를 **원문 그대로** 옮긴 것이다(추출 규칙: 마커 포함 보존). 새 위반이 아니므로 다시 지적하지 말 것 (Codex 문서 리뷰 2026-09-10).

> ⚠️ **이 절 전체가 과거 스냅샷이다.** 시간 역순이며, 각 스냅샷은 그 시점에는 정확했다.
> 지금 유효한 상태는 [`resume.md`](resume.md)의 📌 블록에 있다.

# 🧊 ARCHIVED BELOW — 이하는 2026-09-03 01:1x 이전 스냅샷(시간 역순). 정본은 위 📌 블록.

## ▶️ 이전 스냅샷 (2026-09-03 00:23, 세션 clear 직전)

### 🧊 (보관) TransGenic 원고 리비전 — Plant Communications 투고 준비 (2026-09-03 컨텍스트 리셋 스냅샷)

---

### 🧊 (보관, 이미 대체됨) 정본 포인터 (2026-09-03 00:2x)

> ❌ **SUPERSEDED (2026-09-03):** 이슈 범위 `#4–#54`는 낡았다 — 현재 **#4–#63**이고 #16·#50·#60·#62는 닫혔다. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

- **계획**: `MASTER_PLAN_PC_20260901.md` (v3.7, 9주) · **진행 상태의 정본은 GitHub 이슈** wyim-pgl/transgenic #4–#54
- **프로토콜(고정)**: `transgenic/revision/protocols/PROTOCOL_B1_frozen_v1.md` **v1.25** — §3–9 고정, 개정은 §A1–A35. 오늘 추가: **A30** Swiss-Prot 민감도 세트 · **A31** held-out 좌위가 블록을 test로 강제하지 않음 · **A32** hard flag는 유전자 단위 마스킹 · **A33** 겹침 블록 허용·locus-aware stitching·마스킹 파라미터 · **A34** 인코더는 부분 사전학습(감사·게이트·기록) · **A35** 배치 실패 시 즉시 중단
- **코드 계약**: `docs/gsf_spec_v1.md`(§1a 창 정책, §1b 타일 라벨) · `src/transgenic/utils/gsf_contract.py` · `src/transgenic/datasets/build_b5.py` · `src/transgenic/model/encoder_init.py`(A34) · 설정 `configs/b5_400m_win_v3.json`(정본 학습 설정)
> ❌ **SUPERSEDED (2026-09-03):** 커밋 `b5e4471`·테스트 **145건**은 낡았다 — 현재 tip은 `a9e208b`, 테스트 **155건**(+ torch 필요한 5건은 pgl-gpu에서). 전수 목록은 [`quarantine.md`](quarantine.md) §1.

- **저장소**: 최신 커밋 `b5e4471`, origin/main 동기. 로컬 클론 `Transgenic/transgenic`, GPU 클론 `gpu:~/transgenic`. 테스트 **145건 통과**(`pytest -q tests`, duckdb 필요)
- **원고**: `manuscript_v3_plant_communications.md`(v2→v3 재생성 금지, 직접 수정) · 태그 `revision-2026-09` = `21e8752`
- **작업 규칙**: 문제가 생기면 **Codex와 Kimi K3를 함께** 부르고 의견이 갈리면 **실측으로 중재**(오늘 임계값 2건이 실측으로 뒤집힘). 커밋에 Claude 흔적 금지, 저자 Won <wyim@unr.edu> 단일.

## ▶️ 이전 스냅샷 (2026-09-03 00:23)

### 이슈 현황화
#50–#54 총 5개 생성: #50 9종 DB 병합·md5 동결, #51 ACCESS 129 kb 재측정, #52 계층 추첨, #53 트레이너·QC 결함 9건, #54 Swiss-Prot 매핑 공백.

### resume.md 재작성
74줄 새 스냅샷 추가 (정본 포인터·머신 세 대·데이터 위치·다음 작업·벤치마크·함정 5가지). 백업: `resume.md.bak-20260903-preclear`. 기존 이력 보존.

### 데이터 준비
9종 타일 DB 준비 완료 (`/home/pgl/scratch1/wyim/transgenic_data/b5/full/`, 16 GB, .DONE 표식). GPU RTX 4090 비어 있음.

> ❌ **SUPERSEDED (2026-09-03):** 이 4항목은 전부 소화됐거나 막혔다: #50 **완료**(272,224행 동결, 이슈 종료) · #51은 ACCESS 자원 미확인으로 **차단** · #19–#20은 #51·#59 뒤 · 병렬 항목은 아래 정본 블록 참조. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

### 다음 작업
1. #50 9종 DB 병합 및 md5 동결
2. #51 ACCESS 129 kb 재측정 (크레딧 요청량 확정)
3. #19–#20 seed 학습 시작 (ACCESS)
4. 병렬: #45 miniprot·#25–#27 maize·#5 manifest·#9 장독 재감사

## 🖥️ 머신과 데이터 위치

| 머신 | 접속 | 상태 |
|---|---|---|
| pronghorn (여기) | — | 증거 다운로드 진행 중. `evidence/` |
> ⚠️ **CAUTION (2026-09-03):** pgl-gpu 디스크 "239 GB 여유"는 병합 전 값이다 — 동결 DB 23 GB가 올라가 현재 약 **216 GB**. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

| pgl-gpu | `ssh gpu` | RTX 4090 24 GB, 16 CPU, 디스크 239 GB 여유. 준비 작업 전부 여기서. 데이터 `/home/pgl/scratch1/wyim/transgenic_data/` |
| GB10 | `ssh gb10` | spark-d642, 20 CPU, 128 GB. ⚠️ root의 vLLM 2개가 GPU 약 101 GB 점유 중, torch 환경 없음 |
| ACCESS Delta | 저자 준비 완료 | 자원명·계정·파티션 미확인 |

- pgl-gpu 데이터: `b5/full/<종>.db` 9개(16 GB, 전부 DONE) · `b5/smoke2_Athaliana.db` · `splits/final/` · `qc/`(GeenuFF 플래그) · `swissprot_v3/` · `orthodb_filtered_stage2/`(최종 단백질 자원, md5 `453cb32b02e0799950d7d5f4de5f62ac`) · `proteomes/` · `genomes_src/` · `bench2/`(벤치 원본 로그)
> ✏️ **PARTIAL (2026-09-04):** 아래 2026-09-03 정정 자체가 한 군데 틀렸다 — **드라이버는 "멈춘" 것이 아니라 O. sativa에서 17시간 51분째 붙잡힌 채 2일 4시간을 살아 있었고, #63이 금지한 ONT WGS를 받고 있었다**(§1 표 참조). PacBio subreads 격리 부분은 그대로 참이다.
>
> ❌ **SUPERSEDED (2026-09-03):** "107런 111 GB(진행 중, 완료 표식 없음)"는 두 가지가 틀렸다. 드라이버는 **진행 중이 아니라 2026-09-02 23:29 S. italica에서 멈춰** 있었고(O. sativa·P. patens 미시작), 그 안의 **PacBio 51런 43 GB는 전부 subreads라 제외·격리**됐다(이슈 #60). 전수 목록은 [`quarantine.md`](quarantine.md) §1.

- pronghorn: `evidence/training` 107런 111 GB(진행 중, 완료 표식 없음) · `evidence/qc/`·`evidence/splits/`·`evidence/INPUT_INVENTORY_<날짜>_all.tsv`(마스터, pronghorn+pgl-gpu 통합)

> ❌ **SUPERSEDED (2026-09-03):** 이 6항목 계획은 대체됐다. 현재 계획은 문서 맨 위 📌 정본 블록의 「다음 작업」이다. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

## ▶️ 다음 작업 (순서대로)

1. **#50 9종 DB 병합** → 검증기 0 위반, maize 0행, 계층·종별 카운트 보고, md5 동결. 여기부터가 #16의 완성.
2. **#51 ACCESS에서 129 kb 계층 재측정** → 크레딧 요청량 확정. 4090에서는 불가(아래 벤치마크 표).
3. **#19 seed 123 학습 시작**(ACCESS), 이어서 #20 seed 456·789. 잡 체인은 A28대로 자동 재제출.
4. 병렬: **#45 miniprot 정렬**(입력 준비 완료) → A30.3 phase 감사 · **#25–#27 maize 증거 정렬**(pronghorn 다운로드 완료 후) · **#5 역할 manifest**(`sensitivity_set` 역할값 필요) · **#9 장독 재감사**
5. **#52 계층 추첨**은 프로토콜 개정 후에만. **#53** 남은 확인 항목(다중 GPU 유효 배치, 라벨 캡 절단, Hyena 필터 길이, gradient norm 로깅).
6. 저자 항목: ACCESS 자원명·계정·파티션 확인, GB10의 vLLM 정리, pgl-gpu 공유 계정 비밀번호 교체, #39 행정.

## 📊 오늘 확정된 실측 (재측정 불필요)

**벤치마크 (RTX 4090 24.56 GB, bf16, batch 1, gradient checkpointing on, expandable_segments)**

| 계층 (nt) | 최대 메모리 | 미니배치당 | 라벨 토큰/s | OOM |
|---|---|---|---|---|
| 30,720 | 12.05 GB | 0.70 s | 573 | 0 / 240 |
| 61,440 | 16.94 GB | 1.54 s | 524 | 0 / 240 |
| 129,024 | 24 GB 초과 | (외삽 3.2 s) | — | 1,093 / 1,103 |

- 비용은 **라벨 길이에 선형, 입력 길이에는 거의 무관**(토큰/s가 계층 무관). 메모리는 고정 6.6 GB + 약 0.16 MB/nt → 129 kb ≈ 27 GB. A100 80 GB·GH200 96 GB에는 여유.
- 순수 forward+backward는 30 kb에서 7.21 GB, checkpointing을 끄면 30 kb도 OOM. Hyena 블록 12개가 각각 2회 forward되어 **인코더 checkpointing은 실제로 작동**.
- 9종 학습 타일 약 19만 개 → 4090 기준 epoch당 약 103시간, seed당 2,266 GPU-시간(22 epoch), 3 seed 약 6,800 GPU-시간. ACCESS 환산은 #51에서.

**분할표 (#14 완료, 커밋 4630768)** `data/splits/b5_orthogroup_split_v1.tsv` 334,642행, sha256 `904c6265…`. OrthoFinder 3.1.5 HOG N1 기준(저자 결정 B). train 63.9 % / valid 8.7 % / test 27.4 %. held-out 3,430 좌위와 그 2,891 HOG가 test. 위반 0.

**단백질 자원 (#44 완료)** OrthoDB v12 Viridiplantae 1,232만 → A19.1 1단계(평가종 taxid·서열 일치) → 1,220만 → 2단계(학습종 test orthogroup 단백질) → **1,211만, md5 `453cb32b…`**. Swiss-Prot 민감도 세트 42,096 entry.

**GeenuFF QC (#46 실행 완료)** 9종 플래그 192,497건, hard 유전자 8,770. `qc/<종>.geenuff_flags.tsv` + md5.

**smoke 빌드 2차 (A31·A32·A33 적용 후)** Arabidopsis 10,454행, 거부 148(전부 마스킹 비율 0.60 초과 = 1.4 %), 검증기 0 위반, train 73 %. 1차(6,019행·거부 2,905·검증 실패·train 23행) 대비 정상화.

## ⚠️ 알아야 할 함정

- **트레이너는 이제 배치 실패 시 죽습니다**(A35). 이전에는 조용히 건너뛰어 129 kb에서 데이터의 99 %가 사라졌습니다. 24 GB 카드로 큰 계층을 돌리면 즉시 실패합니다. 정상입니다.
- **인코더는 부분 사전학습**입니다(A34). 339 텐서 중 227개 로드, layers 8–11 (112개)은 무작위, 147개는 타일링 복제. **초기화는 바꾸지 않습니다**(공개 레시피 재현). 게이트가 이 수치와 다르면 실행이 중단됩니다. `TRANSGENIC_RUN_DIR` 설정 시 `encoder_init_report.json`이 남습니다. **Methods에 "partially pretrained"로 써야 합니다.**
- **벤치마크 실행 시 GPU 단독 사용 필수.** 드라이버 스크립트가 `for` 루프라 자식만 죽이면 다음 계층을 새로 띄웁니다. `run_bench2.sh`에 `flock`을 걸어 두었습니다.
- **tqdm이 `\r`로 출력**하므로 결과 JSON이 진행률과 같은 줄에 붙습니다. `tr '\r' '\n'`로 분리해야 grep에 걸립니다.
- **compute 노드에서는 로그인 노드의 `setsid nohup` 프로세스가 pgrep에 안 보입니다.** 파일 mtime으로 확인하세요.
- pronghorn 학습 장독 다운로드는 **아직 완료 표식이 없습니다**. Ath PacBio 2건은 탭 붕괴 버그로 skipped 상태이고 `rerun_training_after_finish.sh`가 대기 중입니다.

## 🗂️ 오늘 만든 것

> ⚠️ **CAUTION (2026-09-03):** "프로토콜 v1.19 → v1.25"는 그 시점 기록이다. v1.25는 지금도 유효하지만 §A13/§A14의 PacBio 조항은 실제 데이터와 어긋나 있었다(이슈 #60, `TRAINING_EVIDENCE_v1.md`의 ❌ 배너 참조). 전수 목록은 [`quarantine.md`](quarantine.md) §1.

- 커밋 18건(`c79a9a2`…`b5e4471`), 프로토콜 v1.19 → **v1.25**(A30–A35)
- 이슈 신설: **#50** 9종 DB 병합 · **#51** ACCESS 129 kb 재측정과 크레딧 · **#52** 계층 추첨(개정 선행) · **#53** 트레이너·QC 결함 기록과 남은 확인 · **#54** Swiss-Prot 매핑 공백 보고
- 스크립트: `scripts/make_orthogroup_splits.py` · `revision/scripts/63_swissprot_sensitivity.py` · `revision/scripts/64_orthodb_filter.py` · `src/transgenic/model/encoder_init.py` · `evidence/make_input_inventory.py`
- 검토 원문: `kimi_a30_review_20260902.md`, scratchpad의 `tile_review/`·`bench_review/`·`enc_review/`

---

# 🧊 ARCHIVED BELOW — 이하는 2026-09-01 이전 스냅샷(시간 역순). 정본은 위 📌 블록.

# 🔄 세션 클리어 후 여기부터 (2026-08-17 20:2x 최종 — **P2 전체 완료, 커밋 d41ae76 푸시**)

## ▶️ 이전 스냅샷 (2026-09-01 밤)

📌 **정본: `MASTER_PLAN_PC_20260901.md`** (PC 재제출 9주 계획; 공유 페이지 https://claude.ai/code/artifact/e3d83bcd-42e1-4ef1-86e9-ca4188a1c4be). 같은 디렉토리에 `codex_pc_audit_20260901.md`(제출요건 감사), `codex_methods_20260901.md`(모델·방법 개선안 11개 순위), `codex_review_20260901.md` + `_verified.md`(오후 전문 리뷰·재검증).

### 🔄 PC 재제출 마스터 플랜 수립 (2026-09-01 밤)

- 저자 판단: PC에 낸다. 개선 방향 개방(새 모델·더 나은 방법·intron-chain 발전). 통과 확률은 현재 반반 이하로 평가(R2 "유용성 미입증"이 리비전으로 반박이 아니라 확인된 형태).
- 조사: 공개 장독 데이터셋(ENA/Zenodo 검증) — maize 1순위 Wang 2020 PRJEB32007/Zenodo 2611319 FLNC(v4 좌표, v4·v5 주석과 독립); Wang 2016은 v4 주석에 쓰여 비독립; SRA PacBio subreads는 BAM 부재로 CCS 불가 → FLNC/ONT 사용. 토마토는 Heinz 독립 Iso-Seq 없음(Moneymaker PRJNA961334 + Heinz ONT ERR4039883). Ath는 AtRTD3가 자체 Iso-Seq만 써 FLIC PRJNA1087576 등 독립.
- 인프라: maize 추가분은 요약 JSON뿐(per-structure 재생성 필요, 스크립트 48 덤프), 스크립트 36 판정함수 재사용 가능·58은 Ath 하드코딩. 장독 데이터 없음. isoseq/minimap2/SQANTI3 v5.0 있음, IsoQuant/FLAIR 없음. GPU 파티션 권한 없음; pgl-gpu 4090·GB10 모두 유휴.
- 계획 골자: B1 사전 고정 프로토콜 + maize 독립 장독 검증 + 고정 필터(1순위, GPU 불필요) → B2 검토예산 매칭 유용성 벤치마크 → B3 다중 beam 오라클 게이트 → 증거 재순위 → B4 블라인드 RT-PCR(저자 결정) → B5 orthogroup 재분할 재학습(선택, W2 시작) → B6 토마토(조건부). 1.17B·intron-chain 재정식화·fine-tune 본실험은 별도 논문.
- 트랙 A 블로커(Codex 감사): Funding·ORCID·이메일 플레이스홀더, 커버레터(구제목·"72.5% 장독 지지" 낡은 주장), 073026 manifest 낡음, Lead Contact 미지정. cell.com 403이라 GA/KRT/후미 순서는 UNVERIFIED.
- 🔐 resume.md 4422행 pgl-gpu 평문 비밀번호(계정명과 동일) 마스킹, 위키 스냅샷도 마스킹(514c6c0). 위키 이력엔 잔존 → 비밀번호 교체 권고.

### ✏️ v3.1: 저자 정정 — Helixer 도구가 아니라 Helixer식 분할 헤드를 자체 모델에(C2, HyenaSegment + EST/ONT 고신뢰 마스크 라벨, 추론은 서열만, 6주차 Helixer 대비 게이트).
### ✏️ v3 (같은 밤, Codex 2차 검토 2건): EST·ONT는 직교 전사 증거(RT-PCR 대체 아님)·증거 차원 5개 고정; evigene는 후속 논문, 이번엔 직접 정렬 접합 DB(C0)를 B3 재순위 채널로만; 포인터 파일럿 → 정규 순서 재직렬화 대조(C1); Helixer 앞단은 공정 전장 벤치마크(B7, 새 기능 아님). 상세 MASTER_PLAN v3 절 + codex_review_v2_items·v3_proposals.
### ✏️ v2 (같은 밤, 저자 지시): RT-PCR → EST·cDNA 증거 계층(B4), ONT 우선(maize PRJNA822071 1순위), 트랙 C 신설(intron-chain 재정식화 파일럿: 인코더 동결 + 포인터식 사슬 디코더, W3–W6), 인코더는 HyenaDNA 400M으로 확정(저자 결정; 교체 후보는 보류 기록). 상세는 MASTER_PLAN v2 절.

### 🔄 증거 계층 새 설계 채택 (revision/protocols/EVIDENCE_TRANSCRIPT_MODEL_DESIGN_v1.md; Codex codex_fulllength_20260901.md 교차 검증): transcript_model 단일 정본(참조 주석도 모델로 흡수) → GSF·C2·C0·검증 파생. 리드 완전성 코드 5C/5I·3C/3I·IC/IP/IM/IX, 관찰 기반 collapse 불변식(모든 사슬+말단 조합은 한 분자가 목격), ORF/GSF 자격(M.5C.3C.MC + 완전 ORF), 상태→허용 용도 결정표. 프로토콜 v1.6: C0에 완전성 코드, 사슬 지지는 말단 완전성과 직교(5I 리드도 완전 사슬 지지 가능), non-UMI ONT cDNA는 PCR-동등 단위(같은 라이브러리·사슬·말단 ±10 nt), 완전성 QC 표(S12d). 증거 유래 GSF 라벨은 후속.
### 🔄 구현 순서 확정 (revision/protocols/IMPLEMENTATION_ORDER_B5_C0_C2_v1.md; Codex 감사 codex_gsf_pipeline_20260901.md를 코드로 교차 검증) — 핵심 확인: 분할은 train_HyenaTransgenic.py:452–470 random_split(RC 증강 후 행 단위) → 분할표 조회로 교체; RC 증강 preprocess.py:258–290(--add-rc-iso-only 단독은 무효, enum으로 교체); EOF 마지막 유전자 미저장(:471); 정확 배수 패딩 +6,144; 토크나이저 CDS150/UTR50/tx15 한도 미강제·2,048 절단; 분할 헤드 14클래스 vs 트레이너 9출력 불일치; 현재 학습 스크립트 기본이 1.17B → B5는 400M 설정 명시. 순서: 계약 고정 → 테스트 먼저 → 공용 직렬화기 → RC 순수 변환 → orthogroup 분할표 → smoke DB → 전체 DB·학습 스크립트 수정·벤치마크 → W2 B5 시작, C0/C2는 병렬.
### 🔄 학습용 장독 목록 확정 (revision/protocols/TRAINING_EVIDENCE_v1.md): ONT 전부 + PacBio Sequel II+ CCS/FLNC급만, 9종 23항목 ≈430 Gb fastq(FASTA ≈200–250 GB). 없음: Bdistachyon 전무, Sbicolor ONT, Gmax·Sitalica·Ppatens PacBio. 드라이버 evidence/training_run_all.sh — 검증용 longread_run_all.sh 끝에 자동 연결(EST → 검증 장독 → 학습 장독 순).
### ✏️ v1.5.1: Sequel II+/Revio 조사 — 순수 B73·Col-0 Revio/Kinnex 없음. 추가: maize PRJNA1470126(B73 Sequel II CCS 2.26M), PRJNA1290227(Revio Kinnex FLNC 57M, B73×Mo17 잡종 → pooled 층), Ath PRJNA911826(Zhang 2023 Col-0 CCS급, 검증 전용); tomato PRJNA961334 HiFi Q20 확인; iFLAS·PRJNA921723·Kurihara 2022는 ZMW 이름 소실로 제외. 드라이버에 반영(SRA toolkit 경로 포함).
### ✏️ 프로토콜 v1.5 (저자 정정): 학습 = 학습 9종 ONT + Sequel II+ PacBio + EST + 교차종 단백질; 테스트종(maize·tomato) 검증은 ONT·PacBio 기기 무관(Wang 2016만 독립성으로 제외) → Wang 2018·2020 복원; Ath 검증 전용 = FLIC·Zhong 2025(학습 금지), Cui 2020은 학습 가능.
### ✏️ 프로토콜 v1.4 (저자 지시, v1.5로 범위 정정): 내려받은 ONT·PacBio는 검증 전용 — 학습·라벨(C2 마스크 포함)·재순위·후보 생성 어디에도 사용 금지; 학습 측 증거는 학습 9종 EST + 교차종 단백질뿐.
### ✏️ 프로토콜 v1.3 (저자 지시, 커밋 c50d47b): PacBio는 Sequel II 이후만 — Wang 2018·2020·Cui PacBio·Han 2023 제외; maize Sequel II+ Iso-Seq은 현재 후보 없음(Revio/Kinnex 탐색 중), Ath는 Zhong 2025 subreads.bam→ccs, tomato Bold 2026은 CCS 확인 후. 장독 드라이버에서 Wang 2018·2020 제거.
### ✏️ 프로토콜 v1.2.1 (Codex 검토): 지지 수는 raw accession이 아니라 독립 분자 단위(클론·버전 병합, 클론 ID 없으면 같은 라이브러리+동일 정렬 서명만 병합, 라이브러리당 10 cap, 고신뢰 접합 = 독립 분자 ≥2) — 구현 감사 항목 포함.
### ✏️ 프로토콜 v1.2 (저자 지시, 커밋 6d89d01): CD-HIT 등 서열 클러스터링 없음 — 모든 EST 개별 게놈 정렬, 정렬된 구간만 사용, 지지 수는 정렬된 accession 기준.
### ✏️ v3.5: 프로토콜 v1.1(EST 원천 정정·부분 사슬·UTR 2차 결과·QC 게이트·필터) + 탐색 프로토콜 M(프롬프트 없는 모드·증거 프롬프트·단백질 후보 풀, 오라클 게이트, 기본 후속 논문) 커밋; C2 = 단백질 CDS 라벨 + EST UTR·접합 라벨; EST 우선순위 L→C→N→M→H→I. Codex 결과: codex_est_brainstorm_20260901.md, codex_candidate_pool_20260901.md.
### ✏️ v3.4 (저자 결정): **B5 orthogroup 재학습 필수** — 학습 9종 명시(Table S1 동일), Z. mays 종 전체 제외, OrthoFinder 기반 orthogroup 분할 75/10/15 seed 123·RC 동일 분할·분할표 사전 커밋, C1 정규 순서 타깃, 400M 22 epoch 재학습(검증 손실 최소 체크포인트), 새 test로 전 평가 재실행·원 체크포인트는 보충. W1 준비·W2 학습.
### ✏️ v3.3: EST 학습 종 = 기존 9종 그대로(Z. mays·벤치마크 3종·Ath strict held-out loci 제외; 학습 종 EST는 검증 불가).
### ✏️ v3.2 (저자 지시): ONT·Iso-Seq **동등 1순위**(maize: ONT PRJNA822071 + FLNC Wang 2020 + HQ Wang 2018; Ath: FLIC + Cui 2020 + Zhong 2025 ccs). **프로토콜 고정 문서 v1.0 작성·커밋** → `transgenic/revision/protocols/PROTOCOL_B1_frozen_v1.md` (고정 객체 MD5 7건, 스크립트 blob 3건, 데이터셋 독립성 2열, 정렬 전 감사 5항, minimap2 파라미터, 소스별 접합 규칙, 완전 사슬 정의, callability 4단계·분모 2종, 증거 차원 5개, 사전 등록 결과 P1–P5(양성·음성 대조 포함), 성공 기준(필터 out-of-sample: 지지 ≥2×, 정밀도 ≥3× → ≥23.7%; 헤드라인 교체: 필터 후 지지 ≥30% & P4 ≥50%), 분석 순서·블라인딩, 고정 문구, 수정 로그). **이후 §3–9 변경 금지.**

### 🔄 EST 다운로드 진행 중 (2026-09-01 17:07 시작, `evidence/est/`, 두 레인 setsid nohup)
- ⚠️ **NCBI `db=nucest`는 더 이상 EST 전용이 아니다** — dbEST 폐지 후 nuccore로 통합돼 특허·mRNA·게놈 레코드(평균 200 KB)가 내려옴. 첫 시도는 배치 하나가 수 GB → 중단·삭제. 올바른 질의: `db=nuccore`, `txid<N>[Organism] AND gbdiv_est[PROP]`. 앞서 보고한 "maize EST 489만" 등은 nuccore 전체 건수였고 EST 실제 건수는 maize 2,019,959 · Ath 1,529,700 · Gmax 1,461,724 · Osa 1,255,251 · Vvi 446,853 · Ppa 382,587 · Sly 301,030 · Sbi 209,835 · Bdi 206,255 · Ptr 89,943 · Sit 66,027.
- 스크립트 `evidence/est/est_fetch.sh`(재개 가능: `<label>/retstart`), 드라이버 `est_run_all.sh`(레인 A: Zmays Athaliana Gmax Ptrichocarpa Bdistachyon / 레인 B: Sitalica Osativa Sbicolor Vvinifera Ppatens Slycopersicum). 완료 표식 `<label>/DONE`, 총건수 `TOTAL`, md5. 출처 기록 `evidence/PROVENANCE_est.tsv`.
- 교훈: `pkill -f <스크립트명>`은 같은 문자열을 가진 ssh 셸 자신을 죽인다 → `[e]st_fetch` 패턴 + 별도 ssh 호출. curl URL의 `[Organism]`은 `-g` 필요.

### 다음 작업 (W1)
1. ✅ 프로토콜 고정 문서 v1.0 커밋 완료. EST 다운로드 완료 확인(`lanes.log`의 ALL LANES FINISHED, 각 TOTAL≈COUNT). 다음: 스크립트 48 per-structure 덤프 플래그 + 59 증거 채점기(합성 데이터 단위 테스트) → F 적용 결과(카운트만) 기록 → 그 다음에야 데이터 다운로드.
2. Zenodo 2611319 + PRJNA822071 + PRJNA1087576 다운로드(`isoseq`/`sra` env).
3. 스크립트 48 per-structure 덤프, 59 종 비종속 채점기.
4. 저자 항목·편집자 기한 확인·RT-PCR 협력 가능 여부 문의.

## ▶️ 이전 스냅샷 (2026-09-01 저녁)

### 🔄 Figure 5·S1 soft-masked Tiberius 전환 (저자 결정, 커밋 21e8752, 태그 `revision-2026-09` 이동)

- TOOLS `tiberius` → `tiberius_softmasked`(Fig 5·S1 공유). Z. mays(실패: gffcompare 0.0/0.0, BUSCO 0.2%)·V. vinifera(soft-masked run 없음)는 빈 셀 + × 마커. 182셀 배열 비교로 Tiberius 계열 26셀만 변경 확인. pdffonts Liberation만. 073026 Figure5/FigureS1 교체(원본 `.bak-20260901`).
- 범례 3곳(Fig 5 L~293·Fig S1 script 20·Results L42) soft-masked 기준으로 재작성, DOCX 2종 재생성·검증. 19_verify 262/262.
- ⚠️ `revision/figures`도 gitignore + tracked → `git add -f` 필요(두 번째 무산 사례).
- 태그 `revision-2026-09`는 2c83920 → **21e8752**로 재생성·force-push(생성 20분 내, 외부 참조 전).

### 🔄 Codex 잔여 비GPU 편집 8건 착지 (커밋 2c83920 + 태그 `revision-2026-09` 푸시)

#### 이번 세션에서 끝낸 것

1. **#8 AUGUSTUS 공통 loci 재채점** — 신규 `57_score_additions_shared_loci.py`(28·47 함수 import, 자체 sanity 실패 시 출력 거부). 공통 25,594 loci·공통 분모 5,554: TransGenic 1,094 추가 → 18.3%/18.6%(recall 3.6%), AUGUSTUS 43,433 → 1.3%/10.2%(10.3%), AUGUSTUS top-1,094 → 0.6%/24.1%. **결론 불변**. Results L52·Methods L130·응답서 R2-13에 반영. 부수: "25,641 AUGUSTUS loci"는 gffcompare 클러스터 수, gene id 기준은 25,597(Methods에 명시).
2. **#7 1,103 추가분 구조통계** — 신규 `58_headline_additions_structural.py`(36 기준). 완전 ORF 441(40.0%), 전 인트론 canonical 752(68.2%), 둘 다 266(24.1%); 필터 후 71.1%/72.2%, recall 3.6→3.4%. 정답 200개 전부 완전 ORF. Results L60·Discussion L86 반영. ✏️ 내 브리프의 "35.7% = 1,014 재실행분"은 오류 — 35.7%는 Helixer-boundary arm(4,512/12,628), 원고 문장은 원래 맞음(1,014 재실행분은 39.2%).
3. **#16 Figure 5 Tiberius** — 🔴 **전제가 틀렸음**: Fig 5는 unmasked Tiberius 계열만 그린다(soft-masked Z. mays 막대는 애초에 없음). 그림 유지, 범례 3곳 정정(Fig 5 범례·Fig S1 범례(script 20)·Results L42에 "Fig 5는 unmasked 계열" 명시). 15_revision_figures.py에 failed-cell 메커니즘(FAILED_RUNS + × 마커) 추가 — 현 TOOLS로는 inert, PNG 바이트 동일 검증. 미리보기 `~/fig5_check/preview_softmasked/`(soft-masked로 바꾸면 Z. mays·V. vinifera 빈 셀).
4. **#12 용어 분리** — "intron-chain match"(GFFCompare '=', 전체 주석) vs "exact-CDS match"(추가분) 명명, Results L46/L48·Methods L130·초록(241단어) 반영.
5. **#20 결론 축약** — L98 수치 8→2개(18.1 vs 1.3).
6. **#14 가용성** — "complete evaluation pipeline" → "evaluation scripts behind every analysis … except the Tiberius run"; 태그 `revision-2026-09`(annotated, 2c83920) 생성·푸시, 문장에 명시. **Zenodo DOI는 저자 계정 필요 → 미착수.**
7. **#17 체크포인트 규칙** — 릴리스 당시 `train/train_HyenaTransgenic.py`(d013418): eval loss < best 저장인데 best를 **perplexity**로 저장 → 매 epoch 저장 → 릴리스 가중치 = 최종 epoch(22), early stopping 없음. Methods L118·응답서 R2-2에 명문화. ⚠️ HF 업로드본이 최종 epoch 파일인지는 **J. Lomas 확인 필요**.
8. **#18 후미 순서** — PC 저자 가이드(검색 요약: methods → funding → author contributions → acknowledgments → declaration of interests → references → figure legends → supplemental information): Data and code availability를 Methods 하위(###)로, Ack↔DoI 순서 교환, Supplemental information을 맨 뒤로. cell.com 직접 접근은 403이라 요약 기반 — 투고 시스템 템플릿과 최종 대조 필요.

검증: 19_verify 262/262, 22 통과, pytest 228, DOCX 3종 재생성(이탤릭 잔존 0)·XLSX 15시트·073026 교체, DOCX 텍스트 추출로 신규 문구 12종 확인. 응답서 "27,213 loci"는 gffcompare query loci(stats 파일 확인) — "GFFCompare loci"로 명시만.

#### 남은 것

- **저자**: Funding/ORCID/이메일/기여문/제목/AT3G61390/Tiberius S10, Task 6, Zenodo DOI 발급(태그 `revision-2026-09` 기준), HF 체크포인트=최종 epoch 확인, PC 템플릿과 후미 순서 대조, 공동작업자 re-clone 안내, GitHub Support(PR 보관 커밋 2건).
- **GPU 재실험군(저자 결정)**: RC-twin 없는 재분할, held-out/homology 벤치마크, seed 반복, 순서 민감도.

## ▶️ 이전 스냅샷 (2026-09-01 후반)

### 🔄 Codex 전문 리뷰 → 재검증 → 5건 수정 착지 (2026-09-01 오후, 커밋 e99f01d 푸시)

#### 이번 세션에서 끝낸 것

1. **Codex 리뷰**(session 01a05e6d…, 7분): 리뷰어 38요청 커버리지 ADDRESSED 33/PARTIAL 4/CLAIMED-BUT-ABSENT 1(R2-14 Table S10), 20항목 우선순위 목록. **Claude 재검증**: 사실주장 대부분 확인, Major 절반은 원고가 이미 공개한 한계(RC twin 30%·held-out 5/32·45 loci training·seed)라 우선순위 하향, #19(참고문헌 57≠56)는 내부 카운트 문제일 뿐(v3=refs 파일=57, 인용 교차 OK), #18(후미 순서)은 템플릿 없이 판정 불가.
2. **수정 5건**(모두 GPU 불필요):
   - 초록 **395 → 237단어**(한도 250). v3 L14 직접 교체.
   - 응답서 L119 `0.9-1.2%` → `1.0–1.8%` + strand 정정 재실행 설명; 응답서 L3의 "Fisher p=0.82를 Discussion에 보고"가 거짓이었으므로 원고 L88에 `(Fisher's exact test, p = 0.82)` 추가(5/32 vs 195/1063 직접 계산 p=0.8197).
   - TSS 범위(L64)에 "400M checkpoint (12 species; L. sativa 400M de novo 부재)" 명시 — Table S6 계산으로 본문 범위가 400M 전용임을 확인(160M은 정확 TSS 1.49–9.27%).
   - Table S10에 BRAKER3 v3.0.8 / GeMoMa 1.9 / **EGAPx 0.3.2-alpha**(컨테이너 `ncbi/egapx:0.3.2-alpha`, Nextflow 24.10.6 — Sylvan Ath run의 slurm-5337757.out 실측; 위키의 "v0.5.0"은 다른 캠페인) 3행 추가(스크립트 20). 명령은 braker.log / protocol_GeMoMaPipeline.txt / input.yaml에서 옮김.
   - Fig 6D 범례(L313)·Fig S2 범례(v2 L165→script 20) "13 species" → de novo 계열 12종 명시.
3. **재생성·검증**: script 20 → 보충자료(S10 17행), 19_verify **262/262**, 22_check_citations 통과, pytest **228**(sylvan env `pytest -q revision`; 루트 수집은 torch 없어 실패). pandoc → DOCX 3종 + 종명 이탤릭(잔존 0), XLSX 15시트(S10 17행·수치형 확인). DOCX 본문 XML 추출로 핵심 수치 17종 전수 확인. 073026에 v3 md/docx 복사. CSV → repo `revision/supplemental_tables/`.
4. **커밋 e99f01d 푸시**: script 20 + `build_supp_xlsx.py`·`italicize_species.py`(스크래치패드에서 소실됐던 보조 스크립트를 repo에 재작성·수록) + S10 CSV.

#### ⚠️ CAUTION (2026-09-01) — v2→v3 재생성 규칙은 더 이상 유효하지 않다

- `manuscript_v3_plant_communications.md`는 8-17 R8/R11/R13/R14 사이클에서 **직접 편집**돼 v2 재빌드와 **45개 문단(93행) 불일치**(v2 초록 248단어 vs v3 395단어, v2엔 15.6%·strand 정정 등 전무). **script 23을 다시 돌리면 8-17 작업이 전부 사라진다.** 이제 **v3가 정본**이며 직접 수정한다. 단 **보충 Fig S1–S3 범례만은 script 20이 v2에서 읽으므로** v2의 해당 범례 3줄은 계속 v2에서 고친다(이번에 v2 L62·L161도 v3와 같은 문구로 맞춰 둠).
- 이번 세션 임시 재빌드가 `supplementary/supplemental_table_legends_superseded.md`를 덮어썼다가 백업으로 복원함.

#### 알려진 제한

- **MarkItDown 깨짐**(pronghorn 로그인 노드): numpy `_multiarray_umath` 로드 시 `/lib64/libm.so.6: version GLIBC_2.27 not found`. 8-17엔 동작했음. 대체: clean_env python으로 `word/document.xml` 태그 제거 텍스트 추출(이번 검증 방식). 위키 installs.md에 기록.
- pronghorn 로그인 셸의 `git`은 1.8.3.1(`-C` 없음); `cd` 후 실행. push 시 `credential-` 경고는 무해.
- 참고문헌 카운트: 내부 기록의 "56"은 낡음, 실제 **57**(v3 = refs 파일, 22번 교차 OK).

#### 남은 것 (Codex 리뷰 잔여 — 저자 결정)

- GPU 재실험군: RC-twin 없는 재분할(#4), held-out/homology 벤치마크(#5), seed 반복(#9), 순서 민감도 재학습(#13).
- 편집·비GPU: AUGUSTUS 공통 loci 재채점(#8), 1,103 추가분 구조통계 한 문장(#7, L86 35.7%와 연결), "transcript-level" 두 정의 명명 분리(#12), 결론 수치 축약(#20), 릴리스 태그+Zenodo DOI(#14), 체크포인트 선택 규칙(#17, 저자 정보), Tiberius Z. mays bar 처리(#16), 후미 순서 템플릿 대조(#18).
- 기존 저자 항목(Funding/ORCID/이메일/기여문/제목/AT3G61390/Tiberius S10) + Task 6 + re-clone 안내 + GitHub Support(PR 보관 커밋 2건).

## ▶️ 이전 스냅샷 (2026-09-01 오전)

### 🔄 PR 본문 AI 흔적 정리 (2026-09-01, 위키 정책 확장 대응)

#### 이번 세션에서 끝낸 것

1. **재감사** (9-01 확장 정책: 이슈·리뷰·코드 주석까지): 추적 파일 grep 0건(CLAUDE.md는 `.git/info/exclude` 비추적), 로컬=origin/main `250fd44`(0/0, clean), 이슈 #3 본문·코멘트 0건.
2. **PR #1·#2 본문 정리**: `🤖 Generated with Claude Code` footer(+PR #1의 `Updated CLAUDE.md …` bullet)를 `gh pr edit --body-file`로 제거, 재조회 0건.
3. 위키 기록: projects/transgenic/updates/2026-09-01_pr-body-ai-trace-cleanup-….md + log.

#### 알려진 제한

- ✏️ **PARTIAL (2026-09-01)**: 8-20의 "PR #1·#2 페이지 원본 trailer 남음 (Support 요청 필요)" → 본문은 해소. Support 대상은 PR 페이지가 보관하는 **재작성 전 커밋 2건**(`e633c38`·`eaa7de1`)의 `Co-Authored-By` 메시지뿐.
- upstream 포크(JohnnyLomas/transgenic) — 타인 소유.

### 다음 작업 (변동 없음)

1. 저자 항목(Funding/ORCID/기여문, 제목, AT3G61390 그림, Tiberius S10) 확정.
2. Task 6 — polishing-benchmark 결과 원고 편입 여부(저자 결정).
3. 공동작업자 re-clone 또는 `git fetch && git reset --hard origin/main` 안내.
4. GitHub Support: PR #1·#2 보관 커밋 메시지 2건.

## ▶️ 이전 스냅샷 (2026-08-20 12:13)

### 🔄 Claude 흔적 제거·설정 완료 (2026-08-20)

#### 이번 세션에서 끝낸 것

1. **transgenic 공개 저장소 히스토리 정제**: 176커밋 재작성 후 force-push — 메시지의 Co-Authored-By 트레일러(Claude Fable 5/Opus 4.5·4.6·5, Claude-Session URL) 전부 제거(164줄), 저자 통일(Won <wyim@unr.edu>), CLAUDE.md 히스토리에서 완전 제거, 파일 내용의 `tmp/claude-…/scratchpad` 경로 → `tmp/scratch`, 관련 원격 브랜치 4개 삭제.
2. **검증 완료**: 19_verify 262/262, pytest 228 통과. GitHub API 확인: 원격 커밋 clean.
3. **재발 방지 설정**: git 전역 신원 `Won <wyim@unr.edu>` 설정, `~/.claude/settings.json`에 `attribution: {commit:"", pr:"", sessionUrl:false}` 추가(기존 includeCoAuthoredBy:false가 못 막던 세션 URL도 차단).
4. **wiki 조사**: trailer 0건 확인(기존 251건 보고는 grep 오탐). 커밋 메시지의 claude 언급 39건은 위키 자체 주제(claude-setup 등).

#### 알려진 제한 (git으로 제거 불가)

- GitHub PR #1·#2 페이지 원본 trailer 남음 (Support 요청 필요). ✏️ PARTIAL (2026-09-01): 본문은 제거 완료, 보관 커밋 메시지 2건만 잔존.
- upstream 포크(JohnnyLomas/transgenic) — 타인 소유 저장소.

### 다음 작업

1. 공동작업자들에게 re-clone 또는 `git fetch && git reset --hard origin/main` 안내.
2. 원고 저자 항목(Funding/ORCID/기여문) 확정.

## ▶️ 이전 스냅샷 (2026-08-17 20:17)

### 🔄 DOCX·XLSX·위키 최종 검증 완료 (2026-08-17 20:4x)

#### 이번 세션에서 끝낸 것 (전부 **커밋 완료**)

1. **DOCX 종명 이탤릭 후처리**: docx-scientific-formatting 스킬 패턴 적용, italicize_species.py 작성 및 실행 — manuscript_v3_plant_communications.docx + supplementary/supplemental_information.docx 모두 비이탤릭 종명(*Arabidopsis thaliana*, *Zea mays* 등) 잔존 0 확인 (XML rPr-i 레벨).

2. **XLSX 재생성·검증**: build_supp_xlsx.py 실행 (clean_env python3) — 14 시트 생성, **S7 신규 토큰 컬럼 수치형 저장 확인**, **S11 신규 17행 수치형 저장 확인** (openpyxl로 검증, Zea mays 사례 확인).

3. **MarkItDown 전수 검증**: md 추출본에서 이번 사이클 핵심 수치 20종 grep 전수 일치 확인 (71.4/0.1%·970/7.9/23.9/90.2·89.3/92.0/74.0/3.1/21.6/1.0·1.8/0.976/98.89/99.02/14,609/76.0·76.2).

#### 진행 중 (백그라운드)

- **랩위키 2026-08-17d 업데이트**: 에이전트 작업 중 (projects/transgenic-polishing-benchmark_20260817.md).

#### 다음 작업

1. 위키 에이전트 완료 → 3자 대조 (위키 파일 직접 Read + `git -C /data/gpfs/assoc/pgl/claude log origin/main -1` 푸시 확인).
2. 저자 항목(Funding/ORCID/기여문/제목) 확정.

## ▶️ 현재 상태 — Roadmap P2(R8~R14) 전부 완료·커밋·푸시. 남은 것은 저자 항목뿐

- **커밋 d41ae76 푸시 완료**(origin/main 확인): 스크립트 51~56·run_arm 2종, 19번(+62핀,
  **262/262**)·20번·35번(strand 수정)·55번, Fig3 스크립트+pdf/png, 결과 기탁 65파일.
  pytest 228 통과.
- **R8 완결**: 누락 셀 3.1%(221/7,092; 필터 21.6%) — no-UTR 4.6%보다 낮음(틀린 UTR이 없는
  것보다 나쁨). strand 버그 수정 후 _fixed 재실행 1.0%/1.8% — **정답 수 두 빌드 동일(3·4)**,
  결론 불변. L84·L88·S11(17행)·결론 반영, 버그 공개.
- **R11 완결**: Fig 3A = nr_ 정본(앵커 92.0/74.0, 각주 †V.vin ‡Z.mays), **Panel B 재작도로
  저자 플래그 2 해소**(alt-only AtRTD3: 72.9/74.4 vs 76.0/76.2). 073026 투고 패키지 교체.
- **R13 완결**: S7 +5 토큰 컬럼, L68 확장, Methods 99.02%(토큰)+98.89%(합집합) 분리,
  Fig S3 범례 vocabulary limits(원천 v2 — 보충 범례만은 여전히 manuscript_v2.md가 원천),
  201-CDS 유전자 창 배제 보정.
- **R14 완결**: 사용 시나리오 Discussion(L88 말미)+결론 명문화.
- **DOCX 2종 재생성·검증 완료**: pandoc 후 MarkItDown 추출 grep 전수 일치(71.4/0.1%·970/
  7.9/23.9/90.2·89.3/92.0/74.0/3.1/21.6/1.0·1.8/0.976/98.89/99.02/14,609/76.0·76.2),
  종명 이탤릭 후처리(scratchpad/italicize_species.py)로 비이탤릭 잔존 0 확인.
- **XLSX 재생성**: 14시트, S7 토큰 컬럼·S11 17행 수치형 확인. build_supp_xlsx.py는
  9e265f6a…/scratchpad에 있음(openpyxl은 clean_env 파이썬).
- 랩위키 2026-08-17d 업데이트: 에이전트 실행 중 — **완료 후 3자 대조 필수**
  (update 파일 직접 Read + `git -C /data/gpfs/assoc/pgl/claude log origin/main -1`).

### ⚠️ 남은 것 = 저자 항목만

Funding/ORCID/이메일/기여문, 제목 확정, AT3G61390 그림 교체 여부, Tiberius 명령
(S10 "미보존" 공개 유지 or 저자 확인). 원고 md는 git 밖 — 위키 스냅샷이 백업.

# 🔄 이전 스냅샷 (2026-08-17 15:5x 갱신 — 인계 세션, R11·R13 반영 완료)

## ▶️ 이전 스냅샷 (2026-08-17 17:24)

### 이번 세션에서 끝낸 것 (전부 **미커밋**)

1. **R8 완주·반영 완료**: TAIR10 CDS + Helixer UTR = 3.1% (221/7,092; 필터 후 21.6%), 부정확한 UTR이 UTR 부재(4.6%)보다 더 해롭다는 발견 — Discussion L84에 새 셀 문장 추가. Table S11에 무필터·필터 2행 추가(총 17행).
2. **검증·테스트 통과**: 19_verify 244/244, pytest 228. 기탁 디렉토리(`revision/results/prompt_transfer/`)에 결과 JSON + `tair10selfutr_filter*.json` 복사 완료.

### 진행 중 (백그라운드, 자동 채점)

- **_fixed 기증 arm**: 17:20 자동 시작(helixertairutr_fixed → annevotairutr_fixed, 각 ~1.3시간), 완주 알림 대기.

### 다음 작업

1. _fixed arm 완주 대기 → rerun 값(0.9%/1.2%) vs strand-split 추정(2.2%/3.5%) 교차 검증.
2. L84 기증 문장 교체, 버그 공개 서술.
3. R14 사용 시나리오 작성 → 커밋 + DOCX + 위키 갱신 + 마감.

## ▶️ 이전 스냅샷 (2026-08-17 16:01)

### 이번 세션에서 끝낸 것 (전부 **미커밋**)

1. **r13 정정 반영**: Methods L106 오류 수정 — 99.02%는 토큰 예산만, "fully representable"(두 한계 모두) 기준은 98.89% (Z. mays 437/39,257). 문구를 "…99.02% fit token budget, and 98.89% satisfy both bounds…" 형태로 분리. 19_verify 핀 재정리 (238/238).

2. **Fig S3 용어 통일**: manuscript_v2.md L167 "GSF token limits" → "vocabulary limits". 20번 스크립트 재실행 → supplementary 파일(TableS7, supplemental_information.md) 재생성. 리뷰어 M7 지적(S3/S7 용어 충돌) 동시 해소.

3. **r8 인계 반영**: 35번 minus_strand_utr_fix 버그 수정·검증 완료(revision/results/prompt_transfer/minus_strand_utr_fix.diff 제출). _fixed arms GPU 작업 스테이징(PID 15503) — 4927 종료 시 자동 시작. 모니터링 루프 장착.

- **pytest 238/238**, **19_verify 238/238** 통과

### 다음 작업

1. r8 GPU 작업 완주 대기(약 2.6시간 예상)
2. strand-split 추정치(plus 2.2%/3.5%) 해석 → L84 factorial 문단 수정 판단
3. 마감: 커밋 + DOCX + 위키 갱신

## ▶️ 이전 스냅샷 (2026-08-17 15:56)

### 이번 세션에서 끝낸 것 (전부 **미커밋**)

1. **R11 완료**: Fig 3A 최종 재작도(panel B 제목 이탤릭 수정), 원고 초록/L36/L128/L307 반영, Figure3 투고 패키지(073026) 갱신.
2. **R13 완료**: Table S7 토큰 길이 컬럼 추가(20번 스크립트 수정→보충자료 재생성), Methods L106 디코더 2,048 토큰 문단 추가, L68 Z.mays 어휘 초과 사례(383/0.976%, p99 2,026, max 14,609@366tx) 및 201-CDS 유전자 보정(73,728→49,152 창) 기술.
3. **검증 완료**: 19번 236/236 통과(R11 20개 + R13 16개 핀), pytest 228 통과.

### 진행 중 (백그라운드, 로컬 생존)

- **R8 GPU 작업**: ssh gpu 로그 확인 — 원격 추론 진행 중(Chr1 완료 15:54, 계속 실행), PID 4927+4934 정상. 완주 시 34→36→34 자동 채점.

### 다음 작업

1. R8 완주 대기(1시간 내), 결과 해석(1% vs 5% 임계) → L84 factorial 문단 수정.
2. R8 후속: 35번 minus-strand UTR 기증 버그 검토 → G/H_fixed 재실행 → 재채점.
3. 마감: 커밋 + DOCX + 위키 갱신 + resume 최종 스냅샷.

## ▶️ 현재 상태 — R8만 남음 (러너 생존·자동 채점), R11·R13은 원고 반영·검증 완료

### 이번 세션에서 끝낸 것 (전부 **미커밋**)

1. **R11 반영 완료**: Fig 3A를 정본(nr_*.stats)으로 재작도 — panel A = 수리 재실행+폭주 제외
   (앵커 A.tha 92.0 / Z.mays 74.0), **panel B도 범례에 맞게 재작도**(alt-only AtRTD3, de novo
   72.9/74.4/73.6 vs prompted 76.0/76.2/76.1 → **저자 플래그 2(Fig 3B) 기계적 해소**, 범례가
   이미 그 설계였음). 원고: 초록 92.2→92.0, Results L36 앵커 2건(92.0 + Z.mays 74.0/36,352 loci,
   71.1=7,458 loci 구판 병기), Methods L128 폭주 제외 규칙 문단(G.max 2/P.patens 1/P.tri 1/S.bic 3;
   P.patens 건은 gffcompare가 파스 단계서 이미 거부), 범례 L307 각주 †V.vin ‡Z.mays.
   073026/Figure3.{pdf,png} 교체, FIGURE_HANDOFF·073026/README 갱신.
2. **R13 반영 완료**(r13-tokencap 에이전트가 타 세션에서 생존·보고, 산출물 3중 검증됨):
   Table S7 +5 토큰 컬럼(20번 수정, supplementary/+기탁 CSV 재생성), Results L68 디코더 2,048
   문단(Z.mays 383/0.976%·p99 2,026·max 14,609@366tx·92건 어휘합법 초과, AtRTD3 99, B.rapa 2,
   나머지 0) + 201-CDS 유전자 보정(region 73,728nt > 49,152 창이라 모델 미도달), Methods L106
   ≥99.02% 병기. Fig S3은 어휘 전용이라 유지.
3. **19_verify 236/236** (R11 20개 + R13 16개 핀 추가), **pytest 228 통과**.
   contingency 키는 over_token_cap_only/over_vocab_limit_only.

### 추가 갱신 (2026-08-17 16:0x) — r8/r13 에이전트 타 세션 생존, 인계 완료

- **r13 정정 반영**: Methods L106의 "99.02% fully representable"은 토큰 예산 단독 값이라 오류
  — 두 한계 합집합(Z. mays 437/39,257)은 **98.89%**. 두 값 분리 서술로 수정, 19_verify 핀 2개
  추가(합집합 437, within-both 98.89) → **238/238**. Fig S3 범례 "token limits"→"vocabulary
  limits"(원천은 manuscript_v2.md — 20번이 v2에서 범례 인용, v2 수정 후 20번 재실행).
  ⚠️ 참고: v3는 23번이 v2에서 생성하는 구조였으나 8/6 이후 v3 직접 편집이 관행(23번은 assert로
  실패할 뿐 조용히 덮지 않음). 단 **보충 범례만은 여전히 v2가 원천**.
- **r8 인계 보고 접수 — 35번 버그 수정·검증 이미 완료**(place_donated_utr 헬퍼; 격리 검증:
  비기증 5개 arm md5 불변; diff: revision/results/prompt_transfer/minus_strand_utr_fix.diff).
  _fixed 입력 2종 생성·GPU 스테이징·md5 대조 완료(56번). **run_fixed_donation_arms.sh
  PID 15503이 arm lock 대기 — 4927 종료 시 자동 시작**(helixertairutr_fixed→annevotairutr_fixed,
  각 ~1.3h, 채점까지 자동, 로그 logs/fixed_donation_arms_20260817_153914.log). 감시자 armed
  (작업 bqprqc7w3). strand-split 추정치: helixertairutr plus 2.2%/annevotairutr plus 3.5%
  (minus 0.0%), 대조군 18.1/18.4 중립. 설계 제약 문서는 53·55번 docstring.
- **새 셀 완주·반영 완료(17:19 채점, 17:3x 반영)**: tair10cdshelixerutr = **3.1%**(221/7,092,
  AtRTD3 4.6%, recall 4.2%), 필터 후 **21.6%**(212/983, AtRTD3 31.5%). 해석: no-UTR 4.6%보다
  낮음 = **틀린 UTR이 없는 UTR보다 나쁨**; 최소 확장 boundary 잔여 교란은 원고 문장에 공개.
  반영: L84 새 셀 문장, S11 +2행(17행), 19_verify +6핀 → **244/244**, JSON 기탁 복사 완료
  (prompt_transfer/, tair10selfutr_filter* 포함).
- 남은 순서: _fixed 2 arm 완주(helixertairutr_fixed 17:20 시작, 각 ~1.3h; 감시 bqprqc7w3)
  → rerun 값으로 L84 기증 문장(0.9%/1.2%) 교체 + 버그 공개(55번이 전체 표) → R14 사용 시나리오
  → 마감 절차(커밋+DOCX+XLSX+위키 3자 대조+resume 갱신).

# 🔄 이전 스냅샷 (2026-08-17 밤 갱신 — P2 진행 중 스냅샷, clear 대비)

## ▶️ 현재 상태 (2026-08-17 밤) — Roadmap P2 진행 중, **백그라운드 2건이 세션과 함께 죽음**

### ⚠️ 새 세션 첫 확인 (순서대로)

1. **R8 GPU 작업 생존 확인**: `ssh gpu 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -af prompt_mode | head -3'`
   — R8(=factorial 누락 셀 TAIR10 CDS+Helixer UTR) 에이전트가 clear로 죽으면 원격 추론이
   **고아로 살아남을 수 있음**(과거 전례). 살아 있으면: 완주 대기 후 채점만 하면 됨(아래 3).
   고아가 이상 상태면 PID kill 후 재발사.
2. R8 재발사 방법(죽었을 때): 입력은 이미 생성됨 —
   `polishing_benchmark/inputs/tair10cdshelixerutr_Athaliana.gff3` (22.6MB).
   러너 초안: `transgenic/revision/scripts/run_arm_tair10cdshelixerutr.sh` + `54_run_extra_arm.py`
   (에이전트가 만들다 중단됐을 수 있음 — 내용 검토 후 사용). 패턴은 33번 스크립트의 원격 실행
   (ssh gpu, micromamba transgenic, prompt_mode.py --batch-size 1 --num-beams 2 --max-length 2048).
3. R8 완주 후: 34번으로 채점 → `polishing_benchmark/results/tair10cdshelixerutr_as_additions.json`,
   36+34로 filtered 병산. 해석: ≈1%면 UTR 정확성이 결정, ≥5%면 포맷/OOD 성분 실재.
4. **R13(디코더 토큰 상한) 상태 확인**: `transgenic/revision/results/decoder_token_cap/` 존재 여부.
   없으면 미완 — `revision/scripts/52_decoder_token_cap_audit.py`(에이전트 작성분, 검토 필요)로
   13종 GSF 토큰 길이 분포(2,048 초과 n/%) 산출. 기대값(관점석): Z. mays p99 1,739–2,223, 0.7–1.2% 초과.

### ✅ P2 완료·반영분 (원고에 이미 적용, **미커밋** — 아래 커밋 계획 참조)

| 항목 | 결과 | 원고 반영 위치 |
|---|---|---|
| R10 S11 1행 필터 | **18.2→71.4%** (245개, 175/185 유지, recall 3.4%) | Discussion 필터 문단 + S11 15행 |
| R12 인용 3건 | Enformer(활동 트랙)·Korf/Majoros(위치 이동)·Stanke(ab initio, evidence-based=Yandell&Ence) | Intro·Discussion |
| R9 self-prompted | **1/970=0.1%** (AtRTD3 0.6%) — 3중 통제 검증, "18.1%는 prompt 귀속" 확정 | Results L52 말미 |
| R9 부수 | self-prompted export 3중 중복(원출력 단계; Fig5 93,690=고유 31,230×3; P.patens·S.lyco 2×) | Methods 중복 문단 + Fig5 범례 |
| R11 진단 | 발산 원인=**폭주 전사체**(수 Mb 가짜 exon이 base Pr만 오염). 정본=폭주제거 재실행: G.max 90.2, P.tri 89.3, S.bic 90.4. 원본 G.max 66.0·재실행 S.bic 75.2는 인공물 | **미반영** — 아래 5 |

### 🆕 R8 중간보고의 신규 버그 (2026-08-17 밤 — 원고 수치 정정 필요)

**arm G/H(UTR 기증)의 minus-strand 기증 실패 버그**: `35_build_frame_shift_prompts.py`가
strand 라벨된 UTR을 좌표 기준으로 필터해 minus에서 기증이 거의 전부 실패
(helixertairutr 35/11,989, annevotairutr 2/10,412). 발표된 0.9%/1.2%는
"기증 성공(plus) 2.2% + 기증 실패(minus) 0.0%"의 혼합. 대조군(tair10selfutr)은
strand 중립(18.1/18.4%) 확인. **r8 에이전트에 지시 완료**: 새 셀 완주 후 35번 수정 →
G/H `_fixed` arm 재생성·GPU 재실행·재채점. 완료 시 원고 L84(0.9%/1.2%)와 S11 두 행의
처리(재실행값으로 교체 + 버그 공개) 필요. 새 셀 설계 제약("TAIR10 boundary+기증 UTR"은
구성 불가 — gene row가 참조 UTR extent를 인코딩; 최소 확장 방식 사용, 잔여 교란은
해석 시 반영)도 보고에 포함될 예정. **세션 clear 시 이 에이전트도 죽음** — 원격 GPU
고아 확인 절차는 위 1번과 동일, 재개 시 이 블록의 지시 내용을 새 에이전트에 전달.

### ⏳ 남은 적용 작업 (순서)

5. **R11 반영**: Methods에 폭주 제외 규칙(예측 mRNA span > 동일 종 레퍼런스 최대 span; 제외 수
   G.max 2/P.tri 1/S.bic 3) + Fig 3A 값 교체 서술. **Fig 3A 재작도**(정본 = `revision/results/
   fig3a_divergence/gffcompare_noRunaway/nr_*.stats`; fig3_regen에 fig3A_metrics.csv·작도 스크립트
   단서). 주의: V. vinifera Sn −4.4 미규명, Z. mays 레퍼런스 상이(+299%, 원본과 비교 불가) —
   그림 각주로 처리. 사용자 승인 완료(그림 작업 포함 전체 진행).
6. R13 결과 반영: S7 관련 본문/보충에 디코더 2,048 상한 분석 추가.
7. R8 결과 반영: Discussion factorial 문단(84행)에 새 셀 + S11 행 추가(generator ARMS).
8. **R14 사용 시나리오**: Journal-Fit W1 대응 — "참조급 primary 보유·isoform 층 부재 유전체"
   시나리오를 Discussion/결론에 명문화(R8·R9 결과 반영해 작성).

### 📋 마감 절차 (사용자 지시 — 반드시 수행)

- 커밋: 미커밋 = 원고 3파일(md) + generator(20번) + 신규 스크립트 51~54·run_arm_*.sh +
  결과(tair10selfutr_filter*.json→repo 복사 필요, selfprompted_additions/ 8 JSON,
  fig3a_divergence/ 전체, decoder_token_cap/, R8 산출물). 대형 파일 제외 원칙 유지.
- **DOCX 검증(지시)**: pandoc(RNASeq_postanalysis env) 재생성 후 MarkItDown으로 텍스트 추출,
  핵심 신규 수치 grep: 71.4 / 0.1% (970) / 7.9 / 23.9 / 90.2·89.3·90.4 / R8 값. 종명 이탤릭
  보존 확인(docx-scientific-formatting 스킬 서식 기준).
- **XLSX**: build_supp_xlsx.py(scratchpad — 새 세션이면 위키 기록의 원본 위치 참조, 숫자는 수치형).
- **랩위키 검증(지시)**: 위키 에이전트 보고 후 README·update 파일 직접 Read로 3자 대조 +
  `git -C /data/gpfs/assoc/pgl/claude log origin/main -1` 푸시 확인.
- resume.md 갱신 + 위키 update 파일(2026-08-17d 예정) 백그라운드 에이전트.

### 참고 — 이 사이클의 정본 수치 모음

reference-prompted 18.1%(200/1,103) / filtered-reference **71.4%**(175/245) / AUGUSTUS 1.3%
(예산 1,103 매칭 시 AtRTD3 23.9%) / self-prompted **0.1%**(1/970) / Z. mays **7.9%**(266/3,363) /
held-out 15.6%(5/32)·validation 27.7%(13/47) / 항등 baseline F1 67.8·AtRTD3 Pr 74.7 /
splice 참조 4,399/5,957(가닥 수정 후) / AtRTD3 조성 78.0·14.7·7.4(additions 일치의 92.6% Iso-Seq).
직전 커밋 17342a9(P1). 모의심사·integrity 산출물: peer_review_report_20260817.md ·
integrity_report_20260817.md. 위키 최신: 386e694.

# 🔄 이전 스냅샷 (2026-08-17 저녁 갱신)

## ▶️ 이전 스냅샷 (2026-08-17 15:26)

### ✅ 마감 절차 갱신 — DOCX·랩위키 실물 검증 추가

위키/DOCX 재생성 후 mtime·보고만 믿지 말고 실물 대조:
- **DOCX**: MarkItDown으로 텍스트 추출, grep으로 P1 신규 수치(71.4%, 7.9%, 23.9%, R1–R7 결과 등) 재확인
- **랩위키**: 위키 에이전트 완료 후 README·update 파일 직접 읽어 내용 일치·원격 반영 대조(git log origin/main)

절차 메모리 저장: verify-docx-and-wiki-at-wrapup.md (MEMORY.md에 항목화).

### 다음 작업
P1 실행 마무리 시 DOCX·위키 검증 체크리스트 적용.

## ▶️ 이전 스냅샷 (2026-08-17 저녁 — Roadmap P1 전체 완료, 커밋 17342a9)

### ✅ P1 7건 + Tiberius 플래그 전부 반영 (매 단계 원천 재검증, 228 테스트 + 19_verify 200/200)

- R1 항등 baseline: echo가 완성 모드와 동급 이상(F1 67.8 vs 67.6; AtRTD3 Pr 74.7 vs 72.5)
  → 초록·Results·결론에 "전체-주석 수치는 입력 복사를 넘지 못함" 명시. 46번 스크립트 기탁.
- R2 AUGUSTUS 예산 스윕: 1,103 매칭 시 AtRTD3 23.9%(동점 규칙 무관 23.2–25.0) > 18.5%,
  TAIR10-alt는 전 예산 0.6~1.6% vs 18.1% → 초록·결론이 축별로 조건 명시. 47번 기탁.
- R3 Z. mays 대조군: additions 7.9%(266/3,363, alt 밀도 6.7배 풍부한 레퍼런스) →
  held-out 문단 재작성(p=0.82 은퇴, validation 27.7% 보고, 18.1%=상한/7.9%=무누출 앵커).
  Z. mays export는 locus당 3회 중복(2회 아님) — Methods 정정. 48번 기탁.
- R4 A5SS/A3SS 가닥 반전 버그 수정(03번): 참조 4,399/5,957(Araport11 비 1.334 재현),
  per-class 정밀도 대소 역전(A5SS>A3SS), 8개 실행 재생성+백업, Fig 6C 재작도,
  19_verify pin 갱신. Methods에 정정 공개.
- R5 AtRTD3 출처: 공식 BED에서 복원 — additions 일치 204 중 189(92.6%) Iso-Seq 지지,
  쇼케이스 45 중 39. "independent long-read" → 조성 정확 서술(78.0/14.7/7.4).
  AT3G61390 등 6 loci는 short-read 단독 지지 명시. 49·50번 기탁.
- R6 쇼케이스 45 loci 학습 노출 라벨, R7 Methods 중복 3종(+Z. mays 3회) 확장.
- Tiberius: 저자 확인 "플랜트 디폴트" → S10·응답서 반영. **플래그 ③ 해소.**

모의 심사 산출물: peer_review_report_20260817.md / integrity_report_20260817.md (+docx).
DOCX 3종·XLSX(14표) 재생성 완료. 스크립트 46~50 신설(46·47=baseline, 48=zmays,
49·50=AtRTD3 provenance — r5가 만든 46·47은 49·50으로 개명됨).

### 남은 것 (P2 이하 + 저자 항목)
1. P2 7건: R8 factorial 누락 셀(TAIR10 CDS+Helixer UTR), R9 self-prompted additions,
   R10 S11 1행 필터, R11 10종 재추론 발산(=Fig 3A 플래그) 규명, R12 인용 문맥 HIGH 4건
   (Enformer·Korf/Majoros·Stanke), R13 디코더 2,048 상한 분석, R14 사용 시나리오.
2. 저자: Fig 3B 재작도, Funding/ORCID/이메일, 제목 확정, AT3G61390 그림 교체 여부.

## ▶️ 이전 스냅샷 (2026-08-17 갱신)

## ▶️ 이전 스냅샷 (2026-08-17 — 모의 동료심사 + integrity 완료)

### ✅ 92.2% 재현 격차 해소 (커밋 9a099a3) — 상세는 아래 08-11 스냅샷 플래그 ① 참조

### ✅ academic-paper-reviewer 스킬로 모의 심사 실행 (5석 패널 + integrity)

산출물: `peer_review_report_20260817.md`(+docx), `integrity_report_20260817.md`(+docx).
**판정: Major Revision(4석 만장일치), Integrity PASS**(참고문헌 46/46 정상 — 기존 미검증
3건 전부 적법 판명, 데이터 표면 100% 재현, MAJOR_DISTORTION 0).

**DA CRITICAL 3건(전부 계산 검증됨, Accept 차단)**:
1. 항등 baseline 부재 — 프롬프트 echo만으로 F1 67.8 vs 완성 67.6 (전체-주석 수치 무력화)
2. AUGUSTUS 예산 매칭(top-1,103) 시 AtRTD3 정밀도 23.9 > 18.5로 역전 (TAIR10-alt 우위는 생존)
3. held-out 층이 다중 isoform 6.3배 고갈 — p=0.82 무검정력, validation arm 27.7% 미보고

**신규 입증 결함 2건**: AtRTD3 "independent long-read" 거짓(Iso-Seq 78%+AtRTD2 14.7%+
Araport11 7.4%), **A5SS/A3SS 라벨이 마이너스 가닥에서 반전**(03_splice_event_detection.py
가닥 미참조 — Fig 6C·S4c 클래스별 값 무효). 쇼케이스 45 loci 전부 train/val.

**Revision Roadmap**: P1 7건(전부 계산만, 재학습 불필요 — R1 항등 baseline, R2 AUGUSTUS
예산매칭, R3 Z. mays additions 대조군, R4 가닥 버그 수정, R5 AtRTD3 재서술, R6 쇼케이스
라벨, R7 Methods 중복 3종 확장), P2 7건, P3 편집. 상세는 peer_review_report의 roadmap.

### 다음 작업 후보
1. Roadmap P1 실행 (전부 기존 산출물 위 계산)
2. 저자 플래그 잔여: Fig 3A 종별 precision 불일치, Fig 3B 재작도, Tiberius 명령,
   Funding/ORCID, 제목 (참고문헌 3건은 integrity가 적법 판명 — 해소)

# 🔄 세션 클리어 후 여기부터 (2026-08-11 01:3x 갱신)

> ❌ **SUPERSEDED (2026-09-03):** "여기서 다시 시작"은 2026-08-11 기준이다. **여기서 시작하지 말 것** — 정본은 문서 맨 위 📌 블록. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

## ▶️ 현재 상태 스냅샷 (2026-08-11 04:34 — 여기서 다시 시작)

### 상태 — 벤치마크 1회 재실행 완주, 저자 플래그 ⑧ 해소 ✅

9개 arm 전부 기준 시그니처 재현 성공(13시간 26분, FRESH 판정, 경보 0건). 전 arm 교차 실행 동일성 확보 → Methods "per-arm structural signatures were identical across runs" 문장이 **전 측정으로 확정**됨.

기탁: `revision/results/repeat_determinism/` (signatures.tsv, freshness.tsv, 측정 기록) — 커밋 **`2ed9b36`** 푸시 완료. 러너/감시자 정상 종료.

### ⚠️ 저자 플래그 7건 남음 (투고 전 해소 필요)

| # | 항목 | 비고 |
|---|---|---|
| 1 | ~~92.2% 재현 격차~~ **✅ 해소(08-17)** | RC 좌표 버그 확정·수리, 92.0 재현 (`9a099a3`) |
| 2 | Fig 3B 재작도 (76.0/76.2) | |
| 3 | Tiberius 실제 명령 | F. Ramazan 확인 |
| 4 | Funding·이메일·ORCID·기여문 | 데스크 반송 항목 |
| 5 | 제목 확정 | |
| 6 | held-out F1·GRMZM 영향 재산출 | 권장 |
| 7 | PASA 버전·참고문헌 3건 | 외부 확인 |

### 다음 작업

- 플래그 1 (92.2% 격차) RC 좌표 처리 검증
- 플래그 2–7 순서대로 해소
- 원고 최종 갱신 및 투고 준비

## ▶️ 이전 스냅샷 (2026-08-11 01:30 — critic 루프 10라운드 완주)

### 상태 — Blocker 자동 수정 루프 10/10 완료, 총 35건 해결

저자 지시 "critic 10회 돌리며 Blocker 자동 해결" 완료. 라운드별: R1 3건(487=loci,
어휘 272, 미기탁 4수치→실측 2건 정정) / R2 2건(UTR 최우선 폐기, 응답서 ordering 충돌) /
R3 2건(**Table S4d 신설**, 59,844 통일) / R4 4건(**alt-only AtRTD3 레퍼런스 재구성** —
빌더가 TAIR10 primary ID로 오삭제, 14,065 유전자 잔존·2,454 오삭제; 76.0/76.2,
11.1@45.9로 수치 교체; BUSCO v6 재검증) / R5 5건(89.9%가 공급 primary 캐비엇,
precision→recall ranking) / R6 3건(**BUSCO "bounded below" 수학적 반증** — Z. mays
78.1%가 반례; L. sativa 공개문 분리; Fig S1 경고) / R7 4건(S10 completion 명령,
**held-out 대조군 전면 교체** — 기존 2,781은 제3 split 산물로 75.8% 학습 노출,
실측: test 3,429에서 15.6% vs 18.3%, p=0.82, "opposite to memorization" 삭제;
Tiberius 명령 미보존 공개) / R8 5건(**옥수수 GRMZM 174모델/176행 유출 공개**,
분할이 행 단위+RC 쌍둥이 1,469/4,899 공개, held-out→test-split 강등, 86/88 분해) /
R9 5건(문서 간 동기화, 응답서 (f)(g) 추가=seven changes) / R10 2건(저장소 낡은 CSV
스냅샷 교체, 174/176 단위 통일).

**커밋 체인**: da5cc0f→b51f3db(전체 수정)→c8de7e6→2afb87f→917ebb2→3ae2fae→35b6a1d→
2f9f496→4054e42→ad7cb7d→e0ed6f7. 매 라운드 228 테스트 통과, DOCX/XLSX 재생성.
신규 스크립트 36~45번, 기탁 JSON 30여 건.

### ⚠️ 저자 플래그 (투고 전 해소 필요, 우선순위순)
1. ~~92.2% 재현 격차~~ **✅ 해소(08-17, 커밋 9a099a3)** — 가설 A 확정: fig3_infer.py
   flip_rc가 절대좌표를 로컬좌표처럼 뒤집어 -rc 예측 전부가 region_start만큼 부족했음.
   수리 후 92.0 재현(Sn 90.6 동일), 순수 held-out 92.3으로 부풀림 가설 기각(노출 효과 −0.9).
   버그 수정 적용, 진단 기탁(fig3_gap_diagnosis/), Methods에 재현 문장 추가.
   ⚠️ **파생 신규 플래그**: 원본 Fig 3A의 일부 종 precision이 수리 재실행과 크게 다름
   (G. max 66.0 vs 87.9, S. bicolor 90.6 vs 75.2) — RC 버그와 무관한 별건, 원인 미규명.
   Fig 3A를 수리값으로 전면 교체하려면 이것부터 규명 필요.
2. **Fig 3B 재작도** — 새 alt-only AtRTD3 값(76.0/76.2)으로. 본문 인용은 복원돼 있음.
3. **Tiberius 실제 명령** — F. Ramazan(framazan) 확인. S10은 "미보존" 공개 상태.
4. Funding 절 + [INSTITUTIONAL EMAIL] + [ORCID]×5 + 기여문 "acquired funding" 상충.
5. 제목 확정(현행: "...completing gene annotations with alternatively spliced isoforms").
6. 순수 held-out 3,430 유전자 base F1 재산출(권장), GRMZM→Zm F1 영향(권장).
7. PASA v2.5.2/2.5.3 표기, 참고문헌 3건(Stiehler 연도/권, Gabriel·Zhang 2026 DOI).
8. ~~벤치마크 1회 재실행 완주 확인~~ **✅ 해소(08-11 04:32)** — 9개 arm 전부 기준
   시그니처 재현(전 arm FRESH, 경보 0건). Methods 재현성 문장 측정으로 확정.
   기록은 `revision/results/repeat_determinism/` 커밋 `2ed9b36`. 러너·감시자 모두 종료됨.

### 다음 세션 시작 시
- 벤치마크 재실행은 완주·종결 — 더 확인할 것 없음. 남은 것은 위 저자 플래그 1~7뿐.
- critic 통합 보고서: `critic_report_20260810.md` (병렬 10렌즈) — 루프 결과는 이 스냅샷과 커밋 로그가 정본.

## ▶️ 이전 스냅샷 (2026-08-10 16:0x 갱신)

> ❌ **SUPERSEDED (2026-09-03):** "여기서 다시 시작"은 2026-08-10 기준이다. **여기서 시작하지 말 것** — 정본은 문서 맨 위 📌 블록. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

## ▶️ 현재 상태 스냅샷 (2026-08-10 21:55 — 여기서 다시 시작)

### 상태 — Critic-loop r1 Blocker 3건 검증 시작

Critic-loop-r1 에이전트가 전체 원고 대조(JSON 소스, 보충정보, 응답서, 검증 스크립트)로 Blocker 3건 적시. **Blocker 1**: 행 50 Results 절 — "487 predicted transcripts"는 사실 `novel_genes` 필드(locus 단위)인데 transcript로 기술됨. 파일별 검증: `summary_report.json` → `novel_genes: 487`, 생성 스크립트 `02_gffcompare_analysis.py:417` → `len(isoform_dist['category'=="novel_gene"])` 확인하고 있음. 회계 확인(8,635 + 18,239 + 52 + 487 = 27,413 locus)로 단위 오류 입증. **Blocker 2, 3**: 메시지 절단으로 세부 미확인.

### 다음 작업

1. Blocker 1 수정: "487 predicted transcripts" → "487 predicted loci"로 원고 정정
2. Blocker 2, 3 전문 확인 및 각각 검증
3. 3개 Blocker 모두 수정 후 원고 재생성 및 커밋

## ▶️ 이전 스냅샷 (2026-08-10 15:51)

### 상태 — held-out 재채점 완료, 암기 반론 기각, 저장소 커밋

대조군 완전 재현 성공(27,413 / 1,103 / 200 / 18.1%). **held-out 분할 결과**: 2,781 loci 테스트 셋에서 21.6%(25/116), 학습 노출 loci 17.7%(175/987) — Fisher exact p=0.31으로 유의한 차이 없음. 방향은 오히려 held-out이 높아 "18.1%가 암기 부풀림"이란 반론이 데이터로 기각됨. 별도 held-out prompted 실행(18.4%, p=0.90)도 헤드라인과 사실상 동일.

seed 123 분할 명령을 스크립트 28~34에서 grep 확인 완료. 결과 JSON 5개(제어, held-out 제한, 포함/제외 분할, 정밀도 정리)를 revision/results/heldout_additions/ 보관 후 스크립트 38번과 함께 커밋(`b51f3db` 대기).

⚠️ **오류 수정**: fig4_forensics/raw_TAIR10_hyenaTest_prediction_noPost.gff3는 3,328 loci 파일이 아님 (같은 디렉토리 FIG4_LOCUS_FORENSICS.md에 기록된 대로, 진짜 3,328 loci 파일은 fig3_original/prompted/ 경로).

### 다음 작업

1. 원고 Methods: "재실행 시그니처 동일" 문장에 seed(123) 명시 + 18.1% vs 21.6%·17.7% 보충 데이터 병기
2. Discussion caveat: held-out 21.6% 실측값 삽입
3. DOCX 재생성 및 푸시

## ▶️ 이전 스냅샷 (2026-08-10 16:00 — critic 10회 + 전체 수정 반영)

### 상태 — paper-critic 10회 완료, 지적사항 전면 수정, 커밋·푸시 완료

저자 지시 정정: "10번"은 벤치마크 반복이 아니라 **paper-critic 10회**였다.
반복 실행은 1회로 축소(`run_repeat.sh` 라운드 인자화, 기본 1), 15:06 재시작 —
**완주 예상 내일(08-11) 새벽 4:30**, `ALERTS.txt`만 보면 된다.

**critic 10회**: 렌즈 10개(초록/서론/Methods/수치/Discussion/교차문서/통계/표그림/논리/적대적
리뷰어) 병렬. 원시 Blocker 53·Major 78·Minor 55 → 통합 보고서 `critic_report_20260810.md`.
9/10 에이전트가 수렴한 최대 이슈: **18.1%(1,103, 스크립트 28, 27,413 loci)와
18.2%(1,014, 스크립트 34, 24,202 loci)는 서로 다른 실행**인데 혼용됨.

**전체 수정 반영 완료** (방침: 재실행 없이 18.1% 헤드라인 유지 + 18.2%를 독립 재실행으로 명시 구분):
- 원고: 제목(de novo 함의 제거), 초록(조건·recall 역전·95.4%·학습종 명시), 서론(동기 재정의,
  AUGUSTUS 샘플링 선행연구, 인용 방향 정정), Results(Fig5B 비교 철회, 43,433=25,597 loci 기준
  서술 정정, beam→duplicated export, 분모 5,580 명시), Discussion(18.2% re-run 명시+31.5%
  intron-chain 병기, 필터 사후선택·6.1% ORF단독·95.5% 상한 공개, 185→275 정정, 학습중첩 caveat,
  1.0% vs 95.4% 분리, novelty 축소, 72.5% 방향 오류 수정), 결론(단서 부기), Methods(batch-size 1
  함정, 표준화 단계, 스크립트 28/34 계보, helixer_reframed, seed 부재 공개, recall 분모 정의),
  가용성(완성 모드 명령 예시 추가).
- 보충: `20_build_supplementary_tables.py` 수정 → S11 캡션 전면 정정+Loci scored 열+
  helixer_reframed 행(13행), S5/S6 무효 L. sativa 행 제외(원고 인용 범위 8.9–27.4, 2.9–22.1
  유지 확인), S2/S3 범례 beam·스코프 정정. md/CSV/XLSX/DOCX 전부 재생성.
- 응답서: three→four changes, §101 beam·27,415 정정, §13 loci 관계 명시.
- 저장소: transfer arm JSON 15개+드라이버 스크립트 5개 저장소 내 보관(가용성 진술 실화).
  **커밋 `da5cc0f` 푸시, 228 테스트 통과.**

### ✅ held-out 재채점 완료 (16:4x) — 암기 반론 데이터로 기각

대조군 완전 재현(1,103/200/18.1% 필드 일치) 후: **held-out 2,781 loci에서 21.6%(25/116),
학습 노출 loci 17.7%(175/987), Fisher p=0.31** — 차이 없음, 방향은 오히려 held-out이 높음.
별도 held-out 추론 실행은 18.4%(21/114). Discussion caveat에 실측값 병기, Methods에 분할
seed(123) 명시. 산출물 `revision/results/heldout_additions/` + 스크립트 38번으로 커밋
`b51f3db` 푸시. docx 재생성 완료.

⚠️ **중요 발견**: `fig4_forensics/raw_TAIR10_hyenaTest_prediction_noPost.gff3`는 held-out
파일이 **아니다** — 4,875 loci, 출처 미해결(FIG4_LOCUS_FORENSICS.md 기록대로). 진짜 3,328
loci 파일은 `fig3_original/prompted/TAIR10_hyenaTest_prediction_noPost.gff3`.

### ⚠️ 미결
2. **Methods의 "재실행 시그니처 동일" 문장** — round 2 실측은 helixer·annevo 2개 arm뿐.
   진행 중인 1회 재실행이 완주(내일 새벽)하면 9개 arm 전부 교차실행 시그니처 비교가 완성됨.
   ALERTS.txt에 DIVERGED 뜨면 해당 문장 수정 필요.
3. 원고 md/docx는 git 밖(프로젝트 루트) — 위키 스냅샷만 존재.
4. critic 미검증 항목: AtRTD3 구축의 기존 주석 의존성(Zhang 2022 원문 확인),
   9.1% Helixer ORF율의 결과 파일 부재(resume 기록만).

## ▶️ 이전 스냅샷 (2026-08-10 14:2x 갱신)

> ❌ **SUPERSEDED (2026-09-03):** "여기서 다시 시작"은 2026-08-10 기준이다. **여기서 시작하지 말 것** — 정본은 문서 맨 위 📌 블록. 전수 목록은 [`quarantine.md`](quarantine.md) §1.

## ▶️ 현재 상태 스냅샷 (2026-08-10 15:23 — 여기서 다시 시작)

### 상태 — 비평 10회 완료, Blocker 5개 식별

비평 10회(critic-01~critic-09)가 모두 완료되었습니다. 논리 정합성 분석(critic-09)에서 제출 전 반드시 수정해야 할 Blocker 5개가 식별됨.

### 발견 — Blocker 5개 (B1–B5)

| # | 위치 | 문제 | 현황 |
|---|------|------|------|
| B1 | L96 | precision 72.5% "상승"이라 했으나 실제 하락(74.4%→72.5%) | L48 TAIR10 vs AtRTD3 수치 방향 반전 |
| B2 | L42 vs L307 | Results는 "17–48 F1 우위", 그림 범례는 "비교 불가" | completion에 공급 전사체 포함된 것 미표기 |
| B3 | 초록 L14, L76 vs L86 | "85–98% complete ORF" 무주석 | 추가 구조만 35.7% (공급 포함하면 그 이상) |
| B4 | L92 vs L48 | "재생산 1.0%(저빈도)" → 실제 "95.4% 항상 재생산" | 통계량 오역 + 방향 반전 |
| B5 | L122 vs L52·L130·L307 | 파일 2배 중복 원인 상반 서술 | Results 내 일관성 결손 |

### 다음 작업

- 10건 비평 통합 보고서 파일 생성 (병합·중복 제거)
- Blocker B1–B5 각각 코드/원고 수정 (원고 revision v4 신설)
- 수정 후 재비평 (critical path 확인)

## ▶️ 이전 스냅샷 (2026-08-10 14:43)

### 상태 — resume.md 비공개 위키에 스냅샷 보관 완료

2026-08-10 14:2x 러너 재시작 상태까지 반영한 resume.md를 github.com/wyim-pgl/wiki(비공개)의 `projects/transgenic-polishing-benchmark/resume.md`에 보관했습니다. 원본(`/data/gpfs/assoc/pgl/data/Transgenic/resume.md`)은 git 추적 밖이므로 원본 그대로 두고, 위키 스냅샷에만 저장합니다. 커밋 hash `9b3a1ad`, 커밋 전 시크릿 스캔 완료(거짓양성만).

### 산물

| 항목 | 상태 |
|---|---|
| 위키 스냅샷 | ✓ 저장 (`projects/transgenic-polishing-benchmark/resume.md`) |
| 커밋 | ✓ 완료 (`9b3a1ad`) |
| 시크릿 스캔 | ✓ 통과 (거짓양성만) |

### 다음 작업

- resume.md가 갱신될 때 다음 위키 업데이트 시 함께 스냅샷 갱신
- 재실행 모니터링 계속 (위키와 무관)

## ▶️ 이전 스냅샷 (2026-08-10 14:20 — 러너 재시작됨)

### 상태 — 러너·감시자 세션 종료로 사망 → 14:13 재시작 완료

이전 세션 종료와 함께 러너가 죽었다(13:06:13, round 2 tair10self 시작 직후).
남긴 기록은 전부 유효: round 1 완주 9 arm 전부 FRESH·기존 확정값 일치,
round 2 helixer·annevo 동일(`876d32079f7aa968`, `13a0f7e0a7fea1a2`). **경보 0건.**

**14:13:01 재시작** — 러너 PID 24205, 감시자 PID 24256, 새 로그
`results/repeat/repeat_20260810_141301.log`. 재시작이므로 **round 넘버링은 1부터 다시**
시작하고 기존 tsv 행 뒤에 append된다(기존 round 1–2 행과 새 round 1– 행이 공존 —
시그니처로 대조하면 되므로 문제없음).

**보충자료 DOCX 재생성은 이미 완료돼 있었다** — DOCX 3종 mtime(13:14)이 md(13:13)보다
뒤이고, `supplemental_information.docx`에 Table S11 절(1곳)·`(deduplicated)` 라벨(14곳)이
md와 정확히 일치. "다음 작업"의 재생성 항목은 종료.

### 다음 작업

- 재실행 모니터링만 남음 (`ALERTS.txt` 확인, 회당 ~13.4시간, 완주 ~2026-08-16)
- 러너가 또 죽으면 §1 절차대로 재시작

## ▶️ 이전 스냅샷 (2026-08-10 13:17)

### 상태 — Task 1·2 완료, 원고·스크립트 수정 반영

Table S11 생성을 정식 스크립트에 통합하고, 2배 중복 표현을 정정했습니다. 94.0% ORF 검증 수치는 손으로 넣은 값이 아니라 `--orf-audit` 모드로 재현 가능하게 만들었습니다.

### 발견·수정

**ORF 라벨 정정**: resume의 "94.0%가 완전 ORF"는 틀렸습니다. TAIR10 데이터:
| 기준 | 비율 |
|---|---|
| 완전 ORF만 | 99.9% |
| ORF + canonical intron 둘 다 | 95.5% (≈94%) |

94% 대는 **두 조건 모두 통과**의 정확도입니다(보충자료에 이미 명시됨).

**2배 중복 원인 정정**: 25,252 locus 모두 구조가 **완전 동일**합니다(서로 다른 beam의 산물이 아님). GM=으로 분석하면 같은 gene ID 쌍이 아니라 같은 locus가 GM= 태그로 묶인 2개 전사체. 원고 "두 beam에서 나온 서로 다른 예측" 표현 → "같은 구조의 2개 전사체"로 수정.

응답서 29,713 vs 원고 29,922: 세포기관 유전자 제외 여부(응답서가 맥락상 정확함).

### 산물

| 파일 | 변경 |
|---|---|
| `revision/scripts/36_filter_additions_structurally.py` | `--orf-audit` 모드 추가 |
| `revision/scripts/20_build_supplementary_tables.py` | Table S11 생성 로직 추가 |
| `manuscript_v3_plant_communications.md` | beam 표현 2곳 정정 |
| `supplementary/supplemental_information.md` | S11 포함 |
| 테스트 | 17개 전부 통과 |

git commit 완료.

### 다음 작업

- 보충자료 문서 최종 재생성 (`supplementary/supplemental_information.docx`)
- 재실행 모니터링 계속 (약 7일 남음)

## ▶️ 이전 스냅샷 (2026-08-09 22:18)

### 상태 — DOCX·XLSX 업데이트 완료, 재실행 계속 진행 중

원고 및 보충자료 문서를 모두 재생성했습니다. DOCX 3종(`manuscript_v3_plant_communications.docx`, `response_to_reviewers.docx`, `supplementary/supplemental_information.docx`)은 새 내용(Discussion 추가 문단, Methods, Table S11)을 반영했고, XLSX(`supplementary/TransGenic_supplementary_tables.xlsx`) 11개 시트(S1–S11)는 모든 표를 포함합니다.

### 발견·수정

**Table S11 배치**: `supplementary_tables.md`는 superseded 문서였습니다. 정식 투고용은 `supplementary/supplemental_information.md` + `supplementary/TableS11_prompt_transfer.csv`입니다. Table S11을 여기에 정식으로 추가하고, 기존 파일은 출처 매핑용 주석으로 줄였습니다.

**DOCX 재생성**: pandoc으로 3개 모두 신규 생성. `supplementary/supplemental_information.docx`도 table S11과 함께 포함.

### 산물

| 파일 | 상태 |
|---|---|
| `manuscript_v3_plant_communications.docx` | ✓ 재생성 |
| `response_to_reviewers.docx` | ✓ 재생성 |
| `supplementary/supplemental_information.docx` | ✓ 재생성 (Table S11 포함) |
| `supplementary/TransGenic_supplementary_tables.xlsx` | ✓ 생성 (11 sheets) |

백업: scratchpad/docx-backup/ 에 DOCX 3개.

### 실행 상태

러너(bash run_repeat.sh) 및 감시자(bash watch_repeat.sh) 모두 생존. 재실행 진행 중.

### 다음 작업

- XLSX 열람 확인 (각 시트 헤더·데이터 행 정상 여부)
- 최종본 투고 준비 (DOCX 레이아웃 최후 확인)
- 재실행 10회 모니터링 계속

## ▶️ 이전 스냅샷 (2026-08-09 21:52)

### 상태 — 원고 반영 완료, 재실행 진행 중

저자 승인으로 초안을 원고에 적용했습니다. Discussion 4문단(새로 3개 추가), Methods 문단, Table S11(새로, 보충자료), 응답서 §119 교체. 단어 수 10,829→11,601. 모순 검사: "다루지 않는다" 류 문장 모두 제거됨. 별개로 남은 "separate question"은 조직 특이적 발현에 관한 것으로 무관.

### 발견·수정

**숫자**: `tair10helixer_as_additions.json`에서 arm F 실측값 275. resume.md 8곳, 위키 2곳 고침.  
**테이블**: S10은 이미 "software versions and command lines" 사용 중. 새 표는 S11로 배치.  
**외부 도구 5종**: BRAKER3·GeMoMa·EGAPx까지 0.0–0.1% (신규 발견, 정확도는 Helixer·ANNEVO와 같은 자리).

### 산물

| 파일 | 변경 |
|---|---|
| `manuscript_v3_plant_communications.md` | Discussion + Methods + 보충자료 목록 |
| `supplementary_tables.md` | Table S11 신규 (12개 조건) |
| `response_to_reviewers.md` | §119 교체 |
| 위키 | 커밋 2개 (7917288 → be9cddd 정정) |

백업: scratchpad의 `.bak-before-transfer-insert` 2개.

### 실행 상태

러너(39667)·감시자(47494) 생존. Chr3 완료(21:50), 경보 없음.

### 다음 작업

- 재실행 10회 전부 완주까지 monitoring (약 8일 남음)
- 구조 시그니처의 첫 이상(DIVERGED) 감지 여부 확인
- 원고 최종본 `.docx` 재생성 필요 시 pandoc 이용

## ▶️ 이전 스냅샷 (2026-08-09 21:28)

### 상태 — 재실행 살아남, 진척 확인됨

클리어를 넘기고 러너(39667)·helixer arm(39677)·원격 추론 모두 생존. round 1 진행 중:

| 시각 | 부위 | 소요시간 |
|---|---|---|
| 21:03 | ChrM 완료 | 3분 |
| 21:24 | Chr1 완료 | 19분 15초 |
| 진행 중 | Chr2 (GPU 94%) | — |

`signatures.tsv`는 아직 헤더뿐. 첫 데이터 행은 helixer arm 완주 시 생성.

### 발견 — 러너가 낡은 파일을 결과로 적을 수 있음

`run_repeat.sh`가 `[ -s ${arm}_completed.gff3 ]`로만 검사하므로 새 파일과 직전 회차 파일을 구별하지 못함. 러너의 1515행(모든 chunk 끝난 뒤)에서만 `_completed.gff3`를 삭제함. **chunk 단계에서 죽으면(ssh 끊김, CUDA OOM, walltime) 직전 회차 파일이 남아 이번 회차 시그니처로 기록됨.** "10회가 동일했다"는 주장 하나인 실행에서 조용히 통과되면 안 될 유일한 실패 모드.

### 대응 — `watch_repeat.sh` 감시자 배포

kill 권한이 차단되어 정공법(재시작) 대신 비침습적 감시 선택. `watch_repeat.sh`(PID 47494)가 실시간으로:
- 각 행의 mtime과 러너 로그의 그 arm 시작 시각 대조 → STALE/FRESH 판정
- 첫 시그니처와 비교 → DIVERGED 감지
- 결과를 `results/repeat/freshness.tsv`와 `ALERTS.txt`에 기록

다음부터 먼저 볼 파일은 `signatures.tsv`가 아니라 **`freshness.tsv`**.

### 다음 작업

- 감시자가 200시간 동안 정상 작동하는지 모니터 (차단되지 않는지, 메모리 누수 없는지)
- freshness.tsv에 FRESH 행만 쌓이는지 확인
- round 1 완주까지 기다려 구조 다양성 첫 확인 (현재 22분, 약 20시간 남음)

## 1. 먼저 실행 상태부터 확인할 것

10회 재실행이 돌고 있다. harness 백그라운드라 **세션 종료와 함께 죽었을 수 있다.**
새 세션에서 가장 먼저:

```bash
pgrep -af "^bash .*run_repeat.sh"      # 러너
pgrep -af "^bash .*watch_repeat.sh"    # 감시자 (둘 다 있어야 정상)
cd /data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark/results/repeat
cat ALERTS.txt          # ← 제일 먼저 볼 것. 비어 있으면 모든 회차가 동일하다는 뜻
cat freshness.tsv       # 행별 FRESH/STALE 판정
cat signatures.tsv      # 원본 시그니처
```

**죽었으면 재시작** (Bash 도구의 `run_in_background: true`로. `nohup setsid`는 이 환경에서
호출이 타임아웃되면 프로세스 그룹째 죽는다):

```bash
bash /data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark/run_repeat.sh
bash /data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark/watch_repeat.sh   # 감시자도 같이
```

재시작하면 **round 1부터 다시** 시작한다(이어받기 없음). 기록된 행은 보존되고 뒤에 붙는다.
lock 파일이 남아 있어도 스크립트가 PID 생존을 확인하므로 stale lock은 문제없다.

**2026-08-10 13:1x 시점**: round 1 완주, round 2 진행 중(helixer·annevo 동일 확인).
경보 0건.

## 2. 이 실행이 무엇이고, 무엇을 기대해야 하나

저자 지시: **"전체 다시 10번 돌려줘"**. 모델은 `do_sample=False, num_beams=2`로 결정론적이라
**10회가 전부 같을 것으로 예상**된다. 이 실행의 산출물은 "같더라"를 가정이 아니라
**측정으로** 남기는 것이다. 어긋나는 회차가 나오면 그것이 유일한 새 정보다.

- 9 arm × 10 라운드, 매 회 `--force`로 chunk 새로 생성
- **round 1 실측 13시간 23분** → 10회 전부 약 134시간(5.6일), 완료 예상 **2026-08-15 무렵**
  (초기 추정 "회당 20시간/전체 200시간"은 과대였다)
- 패널 9 arm은 전부 TAIR10/Helixer/ANNEVO 기반이다. locus가 많은 **egapx·gemoma·braker3는
  이 패널에 없다** — 그래서 arm별 소요가 고르다
- round 1의 9개 정밀도가 전부 기존 확정값과 일치했다(0.0/0.0/4.6/18.2/3.2/2.2/0.0/0.9/1.2).
  round 2 helixer·annevo도 기준 시그니처와 동일

### 러너의 알려진 결함과 감시자

`run_repeat.sh`는 arm 성공 여부를 `[ -s ${arm}_completed.gff3 ]`로만 본다. 이 검사는 새 파일과
**직전 회차가 남긴 파일을 구별하지 못한다**(`33_run_polishing_benchmark.py`는 모든 chunk가
끝난 뒤에야 그 파일을 지운다). chunk 단계에서 죽으면 낡은 파일을 해시해 "동일"로 적는다 —
동일성을 재는 실행에서 가장 나쁜 실패다.

`watch_repeat.sh`가 **탐지**한다(예방은 아니다). 행마다 해시된 파일의 mtime을 arm 시작 시각과
비교해 `FRESH`/`STALE`/`DIVERGED`/`NO_OUTPUT`을 `freshness.tsv`에 적는다.
`signatures.tsv`보다 **`freshness.tsv`가 먼저**다. 폴링은 120초 주기라 한 행 뒤처지는 것은
정상이고, 4분 넘게 벌어지면 의심할 것.

## 3. 원고 작업은 끝났다

**저자가 "전체 반영"을 지시했고 적용 완료.** 프롬프트 전이 실험이 본문에 들어갔다.

| 파일 | 상태 |
|---|---|
| `manuscript_v3_plant_communications.md` | Discussion 4문단(답·원인·필터·범위), Methods 문단, Table S11 참조. 10,829 → 11,601 단어 |
| `supplementary/supplemental_information.md` | Table S11 절 (스크립트가 생성) |
| `supplementary/TableS11_prompt_transfer.csv` | 12개 조건 |
| `supplementary/TransGenic_supplementary_tables.xlsx` | 13시트 + Index |
| `response_to_reviewers.md` | §119 교체, 서두에 (d) 추가 |
| DOCX 4종 | 전부 재생성 |

**코드도 전부 푸시됨** — transgenic `b427943`, 위키 `303d422`. **228 테스트 통과.**
Task 6은 (a)(b) 모두 종료.

⚠️ `supplementary_tables.md`(프로젝트 루트)는 **SUPERSEDED 문서**다. 투고용은
`supplementary/` 쪽이고 `20_build_supplementary_tables.py`가 생성한다. 표를 고칠 일이 있으면
루트 파일이 아니라 스크립트를 고칠 것.

## 4. 남은 것

1. 재실행 감시 — `ALERTS.txt`만 보면 된다. 이탈이 나오면 그때가 처음 나오는 새 정보다
2. 초록의 18.1%에 조건을 달지 여부(현재 미변경 — 본문 Discussion 마지막 문단이 조건을 진술한다)
3. 지표를 정밀도 대신 recall로 보면 결론이 뒤집힌다는 점(arm F가 recall 5.3%로 최고, 정답 275개로
   최다). 원고는 정밀도 기준으로 서술돼 있다 — 저자가 다르게 보길 원하면 재서술 필요

## 5. 이 환경에서 반복해 겪은 함정 (다시 밟지 말 것)

1. **`nohup setsid`로 장시간 작업을 띄우지 말 것.** 호출이 타임아웃되면 프로세스 그룹째 죽는다.
   harness `run_in_background: true`를 쓸 것.
2. **`pgrep -f`/`grep -F`의 자기 매치.** `pgrep -f "bash .../run_repeat.sh"`는 **자기 명령줄까지
   매치**해 생존 검사가 항상 참이 된다 → `^`로 앵커할 것. `grep -F 'name[i]pattern'`도
   대괄호를 리터럴로 처리해 항상 빈 결과다.
3. **`pkill -f <패턴>`은 자기 자신을 죽인다.** PID로 kill 할 것.
   (kill이 권한 분류기에 차단될 수 있다 — 그때는 죽이는 대신 탐지로 우회할 것)
4. **0이 나오면 대조군으로 먼저 검증할 것.** "ORF 완전 0개"는 게놈 FASTA가 Ensembl 이름
   (`1..5,Mt,Pt`), 어노테이션이 TAIR 이름(`Chr1..ChrM`)이라 조회가 전부 실패한 것이었다.
   이 대조군은 이제 `36_filter_additions_structurally.py --orf-audit`로 상시 실행 가능하다.
5. **실제 실행되는 파일을 확인할 것.** `standardize_gff.py`는 저장소 안팎에 두 벌 있고,
   벤치마크는 `transgenic_comparison/`(git 밖) 쪽을 쓴다.
6. **위키·resume의 숫자를 그대로 믿지 말 것.** 이번에만 세 건이 틀렸다 —
   arm F 정답 수 278(실제 **275**), "94.0%가 완전 ORF"(실제 ORF 단독 **99.9%**, 94~95%는
   **두 기준 모두**), 2배 중복의 원인 "beam export"(실제 **DB append 중복**).
   원고에 넣기 전에 원천 파일에서 재계산할 것.
7. **`Table S10`은 이미 "software versions"다.** 새 표는 S11부터.
8. **문체 기준은 STYLE_PROFILE이 아니라 이 원고 자신이다.** 프로파일은 평균 23–27어절이나
   이 원고 Discussion은 31.0어절이다. 삽입문은 원고를 기준으로 맞출 것.

---

## ▶️ 현재 상태 (2026-08-09 21:26) — 재실행 생존 + 감시자 투입

### ✅ 클리어를 넘기고 살아 있음

러너 39667, helixer arm 39677, 원격 GPU 추론 모두 생존. round 1 진행 중:
ChrM 21:03 → Chr1 21:24 완료(7,122 서열, 19분 15초) → Chr2 실행 중. GPU 94% 점유.
`signatures.tsv`는 아직 헤더뿐 — 첫 행은 helixer arm 완주 시 생긴다.

### ✅ round 1 완주 (08-09 21:04:56 → 08-10 10:27:57, 13시간 23분)

**9개 arm 전부 FRESH, 9개 정밀도 전부 기존 확정값과 일치.** 경보 0건.

| # | arm | 소요 | precision | 기존 확정값 | 기준 시그니처 |
|---|---|---|---|---|---|
| 1 | helixer | 1h14m | 0.0 | 0.0 (1/2,462) | `876d32079f7aa968` |
| 2 | annevo | 1h21m | 0.0 | 0.0 (0/348) | `13a0f7e0a7fea1a2` |
| 3 | tair10self | 1h32m | 4.6 | 4.6 (33/722) | `de4d269f46425ad0` |
| 4 | tair10selfutr | 1h32m | **18.2** | 18.2 (185/1,014) | `ab1645ef8cd51623` |
| 5 | tair10helixerframe | 1h37m | 3.2 | 3.2 (33/1,040) | `b10140c366f03d4c` |
| 6 | tair10helixerframeutr | 2h00m | 2.2 | 2.2 (275/12,628) | `874429c9481ea884` |
| 7 | helixer_reframed | 1h26m | 0.0 | 0.0 (0/528) | `c3f25e09ba0e9240` |
| 8 | helixertairutr | 1h23m | 0.9 | 0.9 (3/350) | `f2a7489208b5f087` |
| 9 | annevotairutr | 1h13m | 1.2 | 1.2 (4/345) | `d29bcaa6faa04435` |

**round 1은 아직 결정론의 증거가 아니다.** 9개 값이 기존과 같다는 것은 파이프라인이
이전과 동일하게 동작한다는 확인이고, 기준선을 만든 것이다. 반복 재현성은 2회차부터 잰다.

### ✅ 저자 지시 "차례대로 진행해" 완료 (2026-08-10) — 두 항목 모두 처리

#### 1. Table S11을 정식 스크립트가 생성하게 함 (commit `14f53d2`)

- `20_build_supplementary_tables.py`에 S11 생성부 추가. 손으로 만든 CSV는 대체됐다.
- `36_filter_additions_structurally.py`에 **`--orf-audit`** 추가 — 어노테이션 하나를 두 기준으로
  채점하고 끝낸다. 필터가 늘 필요했던 대조군이다(좌표 조회가 전부 실패해도 예외 없이
  False만 나오는 실패 모드를 구별할 방법이 없었다). 테스트 3개 추가, **228개 통과**.

⚠️ **정정: "TAIR10 primary의 94.0%가 완전 ORF"는 라벨이 틀렸다.**

| 기준 | 프롬프트 세트 | 전체 어노테이션 |
|---|---|---|
| 완전 ORF만 | **99.9%** | 99.8% |
| ORF + canonical intron 둘 다 | **95.5%** | 94.8% |

94~95%대는 ORF 단독이 아니라 **두 기준 모두**의 통과율이다. Table S11 노트는 이제
`polishing_benchmark/results/tair10_primary_orf_audit.json`에서 읽어 두 값을 다 적는다.

#### 2. Task 6b — 2배 중복의 기술된 원인 정정 (commit `b427943`)

**직접 검증했다**(위키 기록을 믿지 않고 파일에서 재계산):

| 확인 항목 | 값 |
|---|---|
| GM locus / gene 레코드 / 전사체 | 27,413 / 54,826 / 59,844 |
| 정확히 2개인 locus | 25,252 |
| 그 쌍의 구조가 완전 동일 | **25,252 전부** (다른 것 0) |
| 홀수 전사체 locus | **0개** |
| `prompt_mode.py` | `num_return_sequences=1` |

서로 다른 beam이라면 구조가 달라야 한다. → **예측 DB 기록 시점의 중복**이지 두 번째
가설이 아니다. 고친 곳: 원고 Methods, `27_rescore_prompted_topbeam.py` docstring,
응답서 §71, 그리고 보충자료 라벨 `(top beam)` → `(deduplicated)` 5곳.

**파생 수치는 하나도 안 바뀐다** — 중복은 GFFCompare에서 두 번째 beam과 똑같이
false positive로 채점되므로 필터와 그 결과는 그대로다. 틀린 것은 원인 설명뿐이었다.

⚠️ 확인 중 알게 된 것: 응답서의 29,713 / 59,426과 원고의 29,922 / 59,844 차이는
**세포기관(ChrC/ChrM) 제외 여부**다. 둘 다 맥락상 옳으므로 건드리지 않았다.

재생성: DOCX 4종, `TransGenic_supplementary_tables.xlsx`(13시트).

### 🔍 round 2 진행 중 (10:27:57 시작) — 여기부터가 실제 판정

round 2 이후 각 arm의 시그니처를 위 기준과 대조한다. 다르면 감시자가 `ALERTS.txt`에
`DIVERGED`를 적는다. **볼 것은 `ALERTS.txt` 한 파일이다** — 비어 있으면 모든 회차가 동일하다.

새 행이 생기면 위 표가 아니라 여기에 이탈만 기록할 것. 동일한 회차는 적을 것이 없다.

| round | 이탈 |
|---|---|
| 2 | helixer 동일 (`876d32079f7aa968`, loci·structures·precision 전부 일치). 나머지 arm 진행 중 |

**첫 재현 판정 결과: 동일하다.** round 2 helixer가 round 1과 시그니처·locus 수·구조 수·정밀도
네 가지 모두 같다. 완전히 새로 생성한 chunk에서 나온 값이다(`--force`, FRESH 판정).
`do_sample=False, num_beams=2` 설정의 결정론이 **가정이 아니라 측정으로** 확인된 첫 사례다.

### ⏱️ 소요 시간 — 회당 ~20시간이 아니라 13시간 23분 (round 1 실측)

**중요**: 반복 패널 9개 arm은 전부 TAIR10/Helixer/ANNEVO 기반이다
(`helixer annevo tair10self tair10selfutr tair10helixerframe tair10helixerframeutr
helixer_reframed helixertairutr annevotairutr`). locus가 훨씬 많은 **egapx·gemoma·braker3는
이 패널에 없다** — 그래서 arm별 소요가 고르다.

**round 1 실측 13시간 23분.** 10회 전부면 **약 134시간(5.6일)**, 완료 예상 **2026-08-15 무렵**.
실행 시작 당시 적었던 "회당 ~20시간 / 전체 ~200시간(8일)"은 과대 추정이었다(아래 21:04 스냅샷).

**결정론이 처음 판가름나는 지점은 round 2의 helixer**다. 그 전까지는 새 정보가 없다.

### 🐛 발견: 러너가 "낡은 파일"을 이번 회차 결과로 기록할 수 있다

`run_repeat.sh`는 arm이 결과를 냈는지를 `[ -s ${arm}_completed.gff3 ]`로만 판단한다.
이 검사는 **새로 만든 파일과 직전 회차가 남긴 파일을 구별하지 못한다.**

`33_run_polishing_benchmark.py`가 canonical `_completed.gff3`를 지우는 시점은
**모든 chunk가 끝난 뒤**(스크립트 1515행 근처)다. 그래서 몇 시간짜리 chunk 단계에서
죽으면 — GPU ssh 끊김, CUDA OOM, walltime — 직전 회차 파일이 그대로 남는다. 러너는
그 파일을 해시해 이번 회차 시그니처로 적는다. **"같다"고 적히지만 같은 이유는
아무것도 새로 쓰이지 않았기 때문이다.** 산출물이 "10회가 동일했다"는 주장 하나인
실행에서, 이건 조용히 통과되면 안 되는 유일한 실패다.

실행이 끝난 뒤 `--force`로 인한 chunk 재생성 90회(9 arm × 10 라운드) 중 한 번도 안
죽을 것으로 볼 근거는 없다.

### ✅ 대응: `polishing_benchmark/watch_repeat.sh` (비침습 감시자)

러너를 죽이고 고치는 게 정공법이지만(파일 선삭제 + exit code 검사), 그러려면 재시작이
필요하고 kill이 권한 분류기에 차단됐다. 대신 **읽기 전용 감시자**를 붙였다:

- `signatures.tsv`에 행이 생길 때마다, 해시된 `${arm}_completed.gff3`의 mtime을
  러너 로그에서 읽은 **그 arm의 시작 시각**과 비교한다.
- 시작 시각보다 오래됐으면 → `STALE`. 그 행은 아무것도 증명하지 않으므로 버린다.
- 새것이면 → `FRESH`. 그리고 그 arm의 **첫 FRESH 시그니처와 비교**해 다르면 `DIVERGED`.
- 러너가 죽으면 그것도 기록한다 (tsv가 안 자라는 것과 침묵은 구별되지 않으므로).

출력: `results/repeat/freshness.tsv` (행별 판정), `results/repeat/ALERTS.txt` (이상만).
**예방이 아니라 탐지다** — 잘못된 행은 여전히 쓰이되, 사후에 걸러낼 수 있게 된다.

주의: 감시자 안의 `pgrep -f`는 **앵커(`^bash ...`)를 반드시 붙일 것.** 앵커가 없으면
pgrep이 자기 명령줄까지 매치해 항상 참이 되고, 러너의 죽음을 영원히 못 잡는다
(§5 함정 2·3과 같은 계열).

### ✅ 원고 반영 완료 — 프롬프트 전이 실험이 본문에 들어갔다 (21:5x)

저자가 "전체 반영"을 선택했다. 초안(`prompt_transfer_manuscript_insert.md`)을 실제로 적용했다.

| 파일 | 변경 |
|---|---|
| `manuscript_v3_plant_communications.md` | Discussion 82행의 "these experiments do not address" 문장을 답으로 교체 + 3개 문단 신규(원인·필터·범위). Methods `Inference and prompt-based completion`에 실험 설계 문단. 보충자료 목록에 Table S11. 10,829 → 11,601 단어 |
| `supplementary_tables.md` | Table S11 신규 (12개 조건 전체) |
| `response_to_reviewers.md` | §119 교체(측정 결과로), 서두 주요 변경에 (d) 추가 |

**주의**: `Table S10`은 이미 "software versions"로 쓰이고 있어 새 표는 **S11**이다.

### ✅ DOCX 재생성 + 보충표 XLSX (22:1x) — 그리고 Table S11 위치 정정

**⚠️ 먼저 알아야 할 것: `supplementary_tables.md`는 SUPERSEDED 문서다.**
투고용 보충자료는 `supplementary/supplemental_information.md` + `supplementary/TableS*.csv`이고
`20_build_supplementary_tables.py`가 생성한다. 처음에 Table S11을 superseded 쪽에 넣었다가
정식 위치로 옮겼다. superseded 문서에는 출처 매핑 항목만 남겼다(그 문서의 원래 역할).

| 산출물 | 내용 |
|---|---|
| `supplementary/TableS11_prompt_transfer.csv` | 12개 조건, 손으로 작성 |
| `supplementary/supplemental_information.md` | Table S11 절 추가(캡션·출처·필터 설명 포함) |
| `supplementary/TransGenic_supplementary_tables.xlsx` | **13개 시트**(S1–S11, S4a/b/c 별도) + Index 시트. 숫자는 문자열이 아니라 수치로 기록 |
| `manuscript_v3_plant_communications.docx` | 재생성 (8/4 → 새 본문 반영) |
| `response_to_reviewers.docx` | 재생성 |
| `supplementary/supplemental_information.docx` | 재생성 |
| `supplementary_tables.docx` | 재생성 |

pandoc: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASeq_postanalysis/bin/pandoc`
openpyxl: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/clean_env/bin/python3` (3.1.5)
XLSX 생성 스크립트는 스크래치패드의 `build_supp_xlsx.py`.

**⚠️ 남은 재현성 부채**: `20_build_supplementary_tables.py`는 Table S11을 생성하지 않는다.
CSV가 손으로 만든 것이라 원천 JSON이 바뀌면 자동으로 따라가지 않는다.

docx 백업은 스크래치패드 `docx-backup/`에 있다.

백업: 스크래치패드의 `manuscript_v3.bak-before-transfer-insert`,
`response_to_reviewers.bak-before-transfer-insert`.

초안 파일(`prompt_transfer_manuscript_insert.md`)은 근거·문체 검증 기록으로 남겨 뒀다.

들어 있는 것: Discussion 4개 문단(¶A 답 제시 / ¶B 원인 / ¶C 필터 처방 / ¶D 범위·recall),
Table S10, Methods 문단, `response_to_reviewers.md` §119 교체문, 예상 질문 대비표.

문체는 STYLE_PROFILE이 아니라 **현 원고 Discussion에 맞췄다**(프로파일은 평균 23–27어절이나
이 원고 Discussion은 31.0이다). 초안 평균 31.6, p90 52, max 58, em dash 0으로 정렬됨.

**새로 드러난 사실**: Helixer·ANNEVO만이 아니라 **BRAKER3(0/145)·GeMoMa(16/25,259)·
EGAPx(13/23,141)까지 외부 도구 5종 전부 0.0–0.1%**다. 특정 도구 결함이 아니라 조건 문제다.

### ⚠️ 정정: arm F 정답 수는 278이 아니라 275

`tair10helixerframeutr_as_additions.json`의 `added_matching_TAIR10_alternative_exact_CDS`
실측값은 **275**다. 이전 스냅샷들의 2×2 표에 278로 적혀 있던 것을 전부 275로 고쳤다
(8곳). 필터 절의 "262/275 유지"가 처음부터 옳았다. 백분율(2.2%, recall 5.3%)은 영향 없다.

### 다음 작업

1. `freshness.tsv` / `ALERTS.txt` 확인 — `signatures.tsv`보다 이쪽이 먼저다
2. helixer arm 완주 시 첫 행 확인, 이후 회차와 대조
3. ~~원고 반영 여부 저자 결정~~ **완료 — "전체 반영" 선택, 적용됐다.**
4. 남은 것: docx 재생성 여부, Table S11 번호가 저널 양식과 맞는지 확인,
   초록의 18.1%에 조건을 달지 여부(현재는 손대지 않았다 — 본문 ¶D가 조건을 진술한다)

## ▶️ 이전 상태 스냅샷 (2026-08-09 21:06)

### ✅ 결정론적 벤치마크 실증 — ChrM 재생성으로 구조 동일성 검증

ChrM chunk 강제 재생성 후 비교:
- **파일 md5**: `160e5c85…` vs `e3a18056…` (다름)
- **좌표·구조만**: 동일한 md5
- **GM별 CDS 구조 집합**: 완전 동일 (31 locus, 37 구조)

파일 달라진 건 transcript ID가 UUID라서. **채점에 쓰이는 모든 단위는 비트 단위 동일**. `do_sample=False, num_beams=2` 설정 의도와 일치.

### ⚠️ 10회 반복 실행 시작 — round 1 진행 중

`polishing_benchmark/run_repeat.sh` 생성, 9개 arm × 10 라운드 시작. 회차별 구조 시그니처(`polishing_benchmark/results/repeat/signatures.tsv`)를 기록해 동일성을 측정 결과로 남김. 어긋나는 회차 있으면 모니터가 `DIVERGED` 알림.

**비용**: 회당 ~20시간, 전체 ~200시간(8일).

### 다음 작업

1. 10회 진행 모니터링 (라운드별 시그니처 확인)
2. 1라운드 완료 시 2-10라운드와 대조 — 구조 동일성 측정 결과 기록
3. 완료된 라운드까지 원고·위키 갱신

## ▶️ 이전 스냅샷 (2026-08-07 17:26)

### ✅ 자동 진행 완료 — 코드 3건·테스트 225개 통과

`36_filter_additions_structurally.py`(신규), `37_analyse_prompt_requirements.py`(신규), `standardize_gff.py`(고정) 세 건 커밋 푸시(origin/main). 테스트 225개 통과.

필터 재검증: 정식 스크립트가 스크래치와 동일 결과. 12,628 → 1,547(87.7% 제거), 정답 262/275 유지, **TAIR10 2.2% → 16.9%**, **AtRTD3 3.6% → 28.4%** — 원고 18.1%를 AtRTD3 기준으로 초과.

위키 갱신 `f760a5c` 푸시 완료.

### ⚠️ "criptic 10번" 대상 불명

저장소·PATH·공유 tools·문서에 `criptic` 없음. 실행하지 못했습니다. 벤치마크는 `do_sample=False, num_beams=2`로 결정론적 — 반복 측정하려면 샘플링 켜기나 입력 변형 필요.

### 다음 작업

1. "criptic 10번"의 대상 확인 (도구명? 샘플링 반복? 입력 변형?)
2. 원고 고찰에 필터 효과 + 9-arm 결론 추가

## ▶️ 이전 스냅샷 (2026-08-07 12:40)

### ✅ ANNEVO 프로필 분석 — gene 경계는 좋은데 UTR 전무

ANNEVO 베이스라인: CDS 68.6%, **gene 경계 일치 2,656/25,181 (10.8%)** — Helixer 0.004%의 2,700배. 하지만 exon = CDS (138,151 좌표 동일), UTR 행 0개. 의도적으로 CDS만 예측하는 도구입니다.

프롬프트 정확 locus 16,701개에서 추가 구조 2개 (Helixer 1,296개). UTR 부재로 생성이 심하게 억제됨.

### ⚠️ arm H 시작 — ANNEVO CDS + TAIR10 UTR

ANNEVO는 경계가 좋으니 "UTR만 채우면"이 arm F/G(Helixer)보다 더 공정한 시험. gene 경계 오류로 인한 과다생성이 적어서, Helixer보다 높은 정밀도가 나올 가능성 있음.

arm H 프롬프트 생성 후 실행 중(~04:00 완료 예상).

### 다음 작업

1. arm H 완료 후 결과 확인 (정밀도, 재현율)
2. arm F/G와 비교 — ANNEVO+UTR 보정이 Helixer보다 유리한지 판단
3. 미커밋 코드 정리 후 최종 논문 고찰 수정

## ▶️ 이전 스냅샷 (2026-08-07 09:57)

### ✅ arm F + arm G 완료 — 18% 미달, 불가능 확정 (새벽)

arm F (TAIR10 + Helixer 프레임): 12,628개 생성, 275개 맞힘 (2.2%). 원고 18%와 88배 차이.

### ✅ ORF 검증 버그 발견 및 수정 — 게놈 헤더 불일치

검증기가 0개 완전 ORF를 반환했으나, 게놈 FASTA는 `1,2,...,Mt,Pt`, 어노테이션은 `Chr1,...,ChrM,ChrC`라 모든 조회가 실패. 별칭 매핑 후 검증기 정상 확인. ⚠️ **당시 적은 "94.0%가 완전 ORF"는 라벨이 틀렸다** — 2026-08-10 `--orf-audit`로 재측정하니 완전 ORF는 **99.9%**이고, 94~95%대는 **ORF+canonical intron 두 기준 모두**의 통과율(95.5%)이다.

### ✅ 후처리 필터가 실제로 통합 — 의미 있는 소득

arm F에 두 가지 필터 적용:
- **ORF 완전만**: 4,512개 (64.3% 제거), 275개 모두 유지, 정밀도 6.1%
- **ORF + canonical splice site**: 1,547개 (87.8% 제거), 262개 유지 (95%), **정밀도 16.9%**

원고 18.1%과 사실상 동등. 두 필터 모두 정답을 보지 않으므로 데이터 스누핑 아님.

### ✅ Helixer 실제 프롬프트는 필터 효과 없음

같은 필터 적용 결과: ORF 완전 비율 9.1% (arm F 35.7%), splice 필터 후 0.0%. 생성물이 구조적으로 훨씬 나음.

### ⚠️ 최종 결론: 18% 달성 불가능 (일곱 조건 모두 실패)

Helixer 어노테이션에서는 최대 0.9%. 18.2%는 프롬프트 CDS+UTR이 정확할 때만 달성 가능. 부분 보정 (UTR만, 경계만, CDS 범위만) 전부 실패.

### 소득: 원인 완전 규명 + 후처리 전략 수립

여섯 개 가설 배제 (채점기, CDS 품질, 프롬프트 파괴, 생성량, 경계 오차, UTR 근접도). 생성량 과다는 후처리로 관리 가능 (정답 유지율 95%).

### 다음 작업

1. 미커밋 변경 5건 정리 (테스트 196개 통과)
2. 원고 고찰 수정: 필터 효과 + 원인 규명 추가
3. 랩 위키 갱신

## ▶️ 이전 스냅샷 (2026-08-07 08:25)

### ✅ tair10helixerframe 완료 — Helixer CDS만 3.2% (08:14)

Helixer 재프레이밍 arm F: added=1,040, any 3.2%, recall 0.6%. 원고 조건에서 정답 CDS 범위를 프롬프트로 줬을 때도 3.2% — Helixer가 만드는 양의 문제가 아니라 생성된 모든 예측이 TAIR10과 불일치합니다.

### ⚠️ arm G 실행 중 — "UTR만 고쳐도 18%가 될까" 테스트 (11시경 완료)

Helixer CDS + 정답 TAIR10 UTR 입력. 23,961개 중 13,925개(58%)는 Helixer가 TAIR10 UTR 영역까지 CDS로 예측해서 UTR을 받지 못함 — "UTR 보정 최대한 적용 시" 현실적 천장입니다. 결과로 "CDS 개선 vs UTR 보정" 중 어디에 노력을 써야 할지 결정됩니다.

### 다음 작업

1. arm G 완료 후 결과 확인 — 18% 도달 여부로 최종 경로 결정
2. 논문 고찰 수정: 원고 18.1%는 학습 데이터와 일치할 때만 성립
3. 미커밋 코드 정리 (gates 33·34·35, standardize_gff.py)

## ▶️ 이전 스냅샷 (2026-08-07 03:54)

### ✅ tair10selfutr 완료 — UTR이 재현의 전부입니다

`tair10selfutr`: added=1,014, TAIR10_alt=18.2%, AtRTD3=18.3%, any=18.2%, recall=3.6% — 원고 18.1%와 사실상 동일. UTR이 하나 추가되자 4.6% → 18.2%로 도약했습니다. 채점기, CDS 품질, 프롬프트 파괴, 생성량 모두 배제되고 **UTR만이 결정 요인**입니다.

### ✅ Helixer의 UTR 정확도 — 0/17,830, all-or-nothing

Helixer가 CDS를 76.9% 맞추면서도 TAIR10 UTR과 일치하는 locus는 **단 0개**입니다. UTR 차이가 1–10 bp일 때 0.0%, 51–200 bp일 때 0.2%, 어느 거리든 무너집니다. 5 bp와 500 bp 차이가 동일합니다 — 점진적 개선이 불가능합니다.

### ⚠️ 목표 달성 불가능 — 데이터로 확정

18% 달성 경로가 모두 막혔습니다. 원고 18.1%는 **TAIR10 어노테이션 자신**을 프롬프트로 줄 때만 성립합니다. Helixer CDS가 정확해도 UTR이 0%이고, UTR을 빼면 상한이 4.6%입니다.

### ✅ 원인 규명 — 암기 가설 확정

모델은 프롬프트가 **학습 데이터와 정확히 동일**할 때만 올바른 isoform을 재생합니다. A. thaliana는 학습 종이고, 다른 도구의 어노테이션 좌표는 학습 데이터와 겹치지 않습니다. 18.1%의 적용 범위(TAIR10 재사용 시에만)가 이제 설명됩니다.

### 다음 작업

1. `tair10helixerframeutr`(~05:35) 완료 대기 — 원고 조건에서 프레임 효과만 측정
2. 아침에 `frame_experiment_status.txt` 확인 후 논문 고찰 수정
3. 미커밋 코드 정리 (gates 33·34·35, standardize_gff.py)

## ▶️ 이전 스냅샷 (2026-08-07 01:10)

### ✅ tair10self 4.6% — 원고 18.6% 미재현, 원인: UTR 부재

tair10self 첫 결과 (added=722): TAIR10_alt 4.6%, any 4.6%. 원고 예측 재채점 18.6%과 4배 차이 — CDS만 넣고 UTR을 뺀 arm 설계 결함입니다. 원고는 TAIR10 exon을 그대로 줬으니 UTR이 포함됐습니다.

### ✅ arm E/F 생성 및 우선순위 변경

UTR 포함 arm 두 개 생성:
- arm E: `tair10selfutr` (TAIR10, 5'3'UTR 포함, TAIR10 경계)
- arm F: `tair10helixerframeutr` (Helixer, 5'3'UTR 포함, CDS범위 경계)

입력 파일 테스트 통과, pytest 147개 통과, 원격 푸시 md5 일치 완료.

### ✅ 실행 순서 재구성 — 원고 재현 확인 최우선 (01:09 시작)

1. `tair10selfutr` (~03:20) — **~18% 회복 여부가 결정적**
2. `tair10helixerframeutr` (~05:35) — 원고와 같은 조건의 순수 프레임 효과  
3. `tair10helixerframe` (~07:45) — UTR 없는 버전 (청크 보존, 중단분 이어서)

orchestrator 24853, runner 10668 종료 후 `run_utr_arms.sh` 재실행. lock 파일 제거 완료.

### 다음 작업

1. `polishing_benchmark/results/frame_experiment_status.txt` 모니터 (arm E ~03:20 완료)
2. arm E가 ~18%면 UTR이 재현의 열쇠, arm F로 원고와 동일 조건의 정확한 프레임 효과 측정
3. arm E도 낮으면 원고 입력 조건부터 재규명 필요

## ▶️ 이전 스냅샷 (2026-08-06 22:43)

### ✅ helixer_reframed 완료 — 목표 18%에 미달 (22:26)

Helixer 재프레이밍 arm 결과: 추가 528개, TAIR10_alt 정밀도 0.0%, AtRTD3 0.6%. CDS 범위를 TAIR10 기준으로 좁혔으나 정확도는 그대로입니다.

### ✅ 채점기 정의 가설 기각

원고 예측 파일을 우리 채점기(34번)로 재채점: 1,035 추가, 18.6% — 원고 18.1%과 사실상 동일 (locus 필터만 차이). 정의가 달라서 낮게 나온 게 아닙니다.

### ✅ CDS 동일해도 Helixer 실패 확인

원고가 맞힌 200개 locus 중, CDS가 정확히 동일한 169개 locus에서 Helixer 프롬프트로는: **추가 0개, 정확도 0개**. 

### ⚠️ 프레임 어긋남 크기는 무관 (부분 반증)

경계 어긋남 1–100bp, 101–1,000bp, 1,001–10,000bp 모두 생성률 4.5~8.5% 유사. Helixer 생성률 4.7%는 TAIR10(3.9%)과 비슷 — **만드는 양이 아니라 만든 것이 전부 틀립니다.**

### 현재 가설: 암기

학습 때 본 좌표와 정확히 일치할 때만 재생. Helixer–TAIR10 경계 완전 일치는 25,657개 중 1개뿐 → 발동하지 않음. 18% 목표는 정답 좌표를 알아야만 재현되는 수치.

### 다음 작업

1. tair10self (~01:00), tair10helixerframe (~03:15) 완료 대기
2. tair10helixerframe 결과로 암기 가설 확정: 0% ≈ 암기 확정, ≥1% ≈ CDS 내용이 관건

## ▶️ 이전 스냅샷 (2026-08-06 20:19)

### ⚠️ tair10self 1차 실패 원인 규명 — 내 중복 실행

grep -F 오독으로 tair10self 1차를 "죽었다"고 판단하고 동일 arm을 재시작 → 두 러너가 원격 ~/polishing_benchmark/predictions에 동시 append → genome2GSFDataset이 중복 gene 행 생성. 결과 정확히 두 배 (6,350 locus → 12,362 gene 행). 33번 무결성 게이트가 잡아냈음.

### ✅ 정리 완료 — 클린 스타트로 재개

- 원격 tair10self_* 산출물 제거
- 로컬 _chunks/tair10self_* 제거  
- pkill -f 실수로 자신까지 죽인 부분 기록

### ✅ 현재 진행

- **PID 23869 helixer_reframed** 실행 중 (~22:45 예상)
- **PID 24853 오케스트레이터** lock 파일로 중복 원천 차단, helixer_reframed 완료 후 tair10self → tair10helixerframe 순차 실행
- 예상 완료: tair10self ~01:00, tair10helixerframe ~03:15

### 다음 작업

1. helixer_reframed 완료 (~22:45) 대기
2. `frame_experiment_status.txt` 확인 → helixer_reframed ≥18% 달성 여부 판정
3. 야간 자동 실행 모니터링

## ▶️ 이전 스냅샷 (2026-08-06 19:14)

### ✅ I-3 고쳤음 — synthetic gene 파일 오류, pytest 196개 통과

`transgenic/scripts/` 사본이 아니라 실제 `transgenic_comparison/standardize_gff.py`를 수정해야 했음. orphan mRNA 대리 gene의 `GM=` 기본값 추가 → 정밀도 검증 완료.

### ⚠️ 게이트 40 재진입 — 65개 locus 설명 없이 소실

락 arm이 I-3 후 max_length 초과로 추정되는 손실 발생. Chr1에서 이미 프롬프트 영향 없음 확인했고 목표와 무관함 → **락 arm 중단**, frame arm 우선.

### ✅ 프로세스 관리 완전 정리

타임아웃으로 프로세스 그룹 동시 사망 → harness 백그라운드로 전환. grep -F 오독 문제 해결. tair10self 중복 실행 (PID 11151, 11839) 및 annevo_locked 제거.

### ✅ 입력 파일 GPU 호스트 동기화 완료

세 arm 입력 파일 (tair10self, helixer_reframed, tair10helixerframe) scp + md5 검증 완료.

### 자동 진행 중

frame 세 arm 순차 실행: `tair10self` → `helixer_reframed` → `tair10helixerframe`. 예상 완료 **02:00경**.

### 다음 작업

1. 02:00~03:00 완료 대기
2. `frame_experiment_status.txt` 읽기 → 세 arm 결과 비교
3. helixer_reframed ≥18% 달성 여부 판정

## ▶️ 이전 스냅샷 (2026-08-06 18:45)

### ✅ 프레임 실험 자동화 배포 완료 — 3개 arm, 밤새 진행 예정

3개 arm 데이터 생성 완료 (각 24,210 locus):
- `tair10self` — 기준선 (18.1% 재현 확인)
- `helixer_reframed` — 목표 후보 (Helixer CDS만 + CDS범위 경계 재설정)
- `tair10helixerframe` — 기전 규명용

pytest 147개 통과 후 데이터 생성 완료.

### ✅ "gene만 넣으면?" 해결 — prompt mode 제약 규명

`preprocess.py` 219행에서 CDS 없으면 GSF 라벨 생성 안 됨 → `prompt_mode.py`가 샘플 스킵. 사실상 de novo 모드이며, 원고 파일럿(Chr4)에서 정밀도 4.4%.

### ✅ 자동화 스크립트 실행 중

`run_frame_experiment.sh` (PID 5864) 배포, ANNEVO locked 완료 후 arm 순차 실행 시작. 채점까지 자동화, **02:00~03:00경 완료 예상**.

결과: `polishing_benchmark/results/frame_experiment_status.txt` (실시간 업데이트)

### 다음 작업

1. 02:00~03:00 완료 대기
2. `frame_experiment_status.txt` 읽기 → 세 arm 결과 비교
3. `helixer_reframed` ≥18% 달성 여부 판정 → 목표 달성 여부 확인

## ▶️ 이전 스냅샷 (2026-08-06 18:02)

### ✅ 프롬프트 정확도 고정 검증 — 무관성 확인됨

프롬프트가 이미 TAIR10과 정확한 locus만 선별해서 재채점해도 정밀도 동일 (Helixer Chr1: 0.3%, 전체: 0.1%). 프롬프트 파괴나 품질이 원인이 아님. **패턴**: Helixer/ANNEVO 같은 모델도, GeMoMa/BRAKER3 같은 다른 도구도 모두 0.1% 선에서 일치.

### ⚠️ 학습 윈도우 좌표계 불일치 의심

Helixer/ANNEVO는 TAIR10 gene 경계 기준의 6,144bp 윈도우로 학습됨(원고 Methods). 그 결과 GSF 좌표는 TAIR10 gene 경계 기준. 하지만 평가 시 GeMoMa/BRAKER3 어노테이션의 gene 경계로 "추가 구조" 좌표를 정의 → **학습/평가 기준 프레임 다름**. 같은 구조도 좌표계가 다르면 일치 판정 불가.

### 검증 대기

gene 경계 오프셋 분석(Helixer 학습 윈도우와 GeMoMa gene 경계의 거리)으로 최종 확인.

### 다음 작업

1. TAIR10 vs 각 도구의 gene 경계 오프셋 통계
2. 좌표계 정렬 후 재채점 → 정밀도 회복 확인
3. 원인 규명 시 연락

## ▶️ 이전 스냅샷 (2026-08-06 17:44)

### ✅ pytest 30개 통과 — 테스트 케이스 수정 완료

one-to-one 매칭에서 두 locus가 같은 참조 유전자에 겹치는 경우 하나만 살아남는 이슈 해결. 단일 locus 케이스로 분리해서 모두 통과.

### ✅ 채점 필드 추가 — primary + alternative 통합 지표

채점기에 "primary 일치율" 필드 추가 (기존 "대체전사체 일치율"과 분리). 18.1% 임계값은 TAIR10 **대체전사체** 기준이므로, primary/alternative 두 지표 모두 확인 필요.

### 진행 중

Chr1 stock/locked + ANNEVO stock 채점 중. 백업 파일 생성 완료. 예상 완료 18~19시경.

### 다음 작업

1. 채점 완료 대기
2. Helixer 전체 실행 후 primary/alternative 일치율 확인
3. 18.1% 이상 달성 여부 판정

## ▶️ 이전 스냅샷 (2026-08-06 16:17)

### ✅ 코드 검증 완료 — primary 동일 제외 로직 확인됨

34번 376행이 `additions = {s for s in output_all if s != supplied}` 로 구현 — 프롬프트 supplied 구조와 정확히 일치하는 것은 이미 제외. 구조는 CDS 좌표 튜플이므로 좌표 다르면 다른 isoform.

### ⚠️ 데이터 검증 스크립트 실행, 결과 미처리

check_lock2.py 실행했으나 출력 결과 미확인. GeMoMa/BRAKER3/EGAPx 각 도구별로 primary 완전 일치 제외가 실제로 적용되는지 데이터상 확인 필요.

### 다음 작업

1. 검증 스크립트 결과 재확인 (각 도구별 added structure 현황)
2. Primary 제외 로직이 모든 도구에 일관되게 적용됨 확인
3. 결과 정리

## ▶️ 이전 스냅샷 (2026-08-06 15:25)

### ✅ Primary transcript 락 구현 완료 — ChrM 스모크 테스트 성공

Primary transcript를 강제 보존하는 constrained decoding 구현 완료. 두 구현:
- `primary_lock.py` — 토큰 단위 락 로직 (단위 테스트 21개 통과)
- `prompt_mode_primary_locked.py` — 원격 `prompt_mode.py` 복사본에 `prefix_allowed_tokens_fn` 전달

ChrM 스모크 (32개 유전자):
| | 프롬프트 보존 | 추가 구조 | 추가 있는 locus |
|---|---|---|---|
| **락** | **31/31 (100%)** | **21** | 19 |
| stock | 27/31 (87.1%) | 10 | 6 |

원본과 달리 추가 구조 2배 증가 — 모델이 primary 재생성에 쓰던 자리를 alternative에 할당.

### ⚠️ 원인: `get_single_decoder_input_ids`가 features만 주입

프롬프트의 features 섹션만 디코더에 넣고 transcript 리스트(`>` 직후)는 주입하지 않아 모델이 primary를 새로 생성. 이 때문에 프롬프트 파괴 및 정확도 저하.

### 실행 중

PID 6955 (helixer_locked, annevo_locked), 15:23 시작. 18:00~19:00 완료 예상.

### 다음 작업

1. 전체 실행 완료 대기 (18:00~19:00)
2. 34번으로 채점 → stock 대비 정확도 비교
3. 결과 검토 후 의사결정

## ▶️ 이전 스냅샷 (2026-08-06 13:10)

### ✅ docstring 정정 + 25개 커밋 푸시 완료

`34_score_as_additions.py` docstring의 "GeMoMa 21,739를 정확히 재현" 주장을 정정했음.
올바른 기대값은 21,744 (GeMoMa의 게인 매칭 버그로 인한 편차 규명 완료).
pytest 24개 전부 통과, 커밋 수행, 25개 미푸시 커밋 전부 `origin/main`으로 푸시 완료.

### GPU 벤치마크 실행 중

PID 1149 계속 실행 (helixer Chr2 진행 중). 18:00 전후 완료 예상.
완료 후 Helixer/ANNEVO 결과에 34번 적용 → AS 추가 정확도 표 완성.

### 다음 작업

1. GPU 벤치마크 완료 대기
2. Helixer/ANNEVO 34번 적용 → 표 완성
3. 위키 갱신 (`/data/gpfs/assoc/pgl/claude`)

## ▶️ 이전 스냅샷 (2026-08-06 13:06)

### ✅ 34번 검증 완료 — GeMoMa 5개 불일치 규명됨

32번 docstring(119~144행)이 이미 문서화: GeMoMa 페어링 버그로 21,739는 낡은 값, 올바른 값은 21,744.
34번이 독립 구현으로 정확히 21,744를 냈으므로 **두 구현의 일치 = 검증 통과**.

- `34_score_as_additions.py` — 존재, docstring "reproduces exactly" 주장 수정 완료
- `test_as_addition_scoring.py` — pytest 24개 전부 통과
- 미완료 표 행: Helixer / ANNEVO 아직 측정 중

### GPU 벤치마크 진행 중

PID 1149 (12:27 시작) 계속 실행 중. Helixer Chr1 완료, Chr2 진행.
완료된 chunk 3개(helixer·annevo raw.gff3) 확인됨. 18:00 전후 종료 예상.

### 다음 작업

1. 34번 파일명/docstring 최종 승인 → 커밋 (저자 대기 중)
2. GPU 벤치마크 완료 (18:00 전후) → 34번 적용 → 표 완성

## ▶️ 이전 스냅샷 (2026-08-06 12:57)

### ⚠️ 12:43 스냅샷의 "AS 채점기 파일 미생성"은 사실과 다름 — 정정

이전 에이전트가 세션 클리어 직전에 파일을 실제로 작성했다. 확인된 상태:

- `revision/scripts/34_score_as_additions.py` (30,650 B, 12:52) — 존재. **파일명이 지시와 다름**
  (지시는 `34_score_added_isoforms_external.py`)
- `revision/scripts/tests/test_as_addition_scoring.py` — 존재, **pytest 24개 전부 통과**
- 둘 다 아직 git untracked
- 3도구 검증까지 실행됨: `polishing_benchmark/results/{gemoma,braker3,egapx}_as_additions.json`

### ✅ 검증 통과 — GeMoMa 5개 차이는 규명 완료, 34번이 옳음

| 도구 | 34번 산출 | 판정 |
|---|---|---|
| GeMoMa | **21,744** | ✅ 올바른 값 (21,739은 32번의 알려진 버그가 낳은 낡은 값) |
| BRAKER3 | 128 | ✅ |
| EGAPx | 22,711 | ✅ |

**근거 — `32_score_polishing.py` docstring 119~144행이 이 문제를 이미 문서화해 뒀다:**

- 32는 `GM=`을 입력 gene의 `ID=`와 매칭한다. 그러나 실제 규칙(`preprocess.py:314`)은
  **첫 속성 값**이고, GeMoMa gene 행은 `Name=`으로 시작한다. → 32는 GeMoMa를 하나도
  식별자로 짝짓지 못하고(`gm_paired=0`) 29,561개 전부 좌표겹침 폴백으로 넘어간다.
- 32는 그 대가를 **실제로 측정해 기록해 뒀다**: 올바른 키로 재채점 시
  `added_structures 21,739 -> 21,744` (131행). 그 5개는 `loci_without_output`에 잘못
  분류된 34개 중 5개로, **페어링 실패지 생성 실패가 아니다.**
- 32는 **일부러 고치지 않았다** — "이미 발표된 숫자가 바뀌므로 별도 리뷰 라운드가 필요".
- BRAKER3·EGAPx는 gene 행이 `ID=`로 시작 → 영향 없음.

즉 32가 "올바른 페어링이면 21,744"라고 예측했고, 34가 독립 구현으로 정확히 21,744를 냈다.
**두 독립 구현의 일치이므로 검증 실패가 아니라 통과다.** GeMoMa에서 21,739가 나왔다면
오히려 34가 32의 버그를 물려받았다는 뜻이 된다.

**조치 완료** — 34의 docstring에서 "reproduces `32` exactly — GeMoMa 21,739" 주장을
사실에 맞게 수정했다(GeMoMa 예외와 그 이유를 명시). 코드 로직은 건드리지 않았고
테스트 24개 통과 유지.

⚠️ 이전 스냅샷들에 남아 있는 "GeMoMa 21,739와 맞아야 함"이라는 검증 조건은 **잘못된 기대**였다.
올바른 기대값은 21,744다.

### GPU 벤치마크

PID 1149 실행 중 (12:27 시작, 18:00 전후 완료 예상). 현재 helixer Chr1 완료, Chr2 진행 중.
산출물 `predictions/{helixer,annevo}_completed.gff3` 아직 없음.

### 다음 작업

1. ~~5개 불일치 규명~~ 완료 (34번이 옳음)
2. ~~커밋 + 푸시~~ 완료 (2026-08-06 13:10)
   - `673d084` — as-scorer가 만든 34번 스크립트 + 테스트 (에이전트가 직접 커밋)
   - `50adc81` — docstring의 "reproduces exactly" 주장 정정 (로직 무변경, 24 테스트 통과)
   - `origin/main`으로 **미푸시 25개 전부** 푸시 완료 (`44bc8ad..50adc81`). 데이터 파일 혼입 없음
   - 파일명은 `34_score_as_additions.py`로 확정 (원래 지시의
     `34_score_added_isoforms_external.py`는 채택하지 않음)
3. ~~GPU 벤치마크 → Helixer/ANNEVO 채점~~ **완료 (15:03 종료, 예상보다 3시간 빠름)**

### 🔴 AS 추가 정확도 표 — 완성. 결과는 "AS 추가 실패" 분기다

| 프롬프트 | 추가 구조 | TAIR10 대체전사체 일치 (exact CDS) | AtRTD3 일치 |
|---|---|---|---|
| TAIR10 자신 (원고 기존) | 1,103 | **18.1%** | 18.5% |
| AUGUSTUS 샘플링 (대조) | 43,433 | 1.3% | 10.2% |
| **Helixer** | 2,462 | **0.0%** (1건 / 2,462) | 0.3% (8건) |
| **ANNEVO** | 348 | **0.0%** (0건 / 348) | 0.3% (1건) |

인트론 체인 기준(더 느슨)으로도 Helixer 0.6%, ANNEVO 1.7%에 그친다.
두 도구 모두 **AUGUSTUS 무작위 대조군보다 낮다.**

**decomposition이 더 결정적이다 — 추가분의 대부분이 프롬프트를 파괴한 locus에서 나왔다:**

| | 추가 총계 | 프롬프트 생존 locus에서 | 프롬프트 파괴 locus에서 |
|---|---|---|---|
| Helixer | 2,462 | 1,019 | **1,443 (59%)** |
| ANNEVO | 348 | **5** | **343 (99%)** |

ANNEVO는 프롬프트가 살아남은 24,997개 locus에서 추가한 구조가 **5개**뿐이다.
즉 "추가"로 집계된 것이 사실상 전부 파괴의 부산물이다.

**채점 품질은 문제없다** — 두 도구 모두 `gm_paired`가 전량(27,031 / 25,172),
`overlap_fallback=0`, `output_split=0`, `output_merged=0`. GeMoMa 같은 페어링 실패 없음.
벤치마크 산출물도 정상(미설명 손실 Helixer 1건 / ANNEVO 2건, 게이트 40 한참 아래).

### ⚖️ 저자 판단 필요 — 분기는 "실패" 쪽

resume.md에 미리 정해둔 분기대로라면 **현재 상태 유지**(고찰의 Helixer/ANNEVO 문장은
제거된 채로 둠)가 맞다. 되살릴 근거가 확보되지 않았고, 오히려 반대 방향 증거가 나왔다.

## ✅ 밤샘 실험 완료 (2026-08-07 08:14) — 여기부터 볼 것

### 최종 결과 — CDS를 TAIR10으로 고정한 2×2 설계

| | **TAIR10 경계** | **Helixer 경계** |
|---|---|---|
| **UTR 있음** | **18.2%** (추가 1,014 / 맞음 185) | 2.2% (추가 **12,628** / 맞음 **275**) |
| **UTR 없음** | 4.6% (추가 722 / 맞음 33) | 3.2% (추가 1,040 / 맞음 33) |

참고 — 실제 도구 프롬프트: Helixer stock 0.0%(추가 2,462/맞음 1),
`helixer_reframed` 0.0%(추가 528/맞음 0), ANNEVO stock 0.0%.

**두 요인의 역할이 완전히 분리된다:**

- **UTR = 정답을 만들 수 있는가.** 맞은 수가 33 → 185/275로 갈린다.
  경계와 무관하게, UTR이 없으면 33개에서 멈춘다. 1–10 bp만 틀려도 0%(all-or-nothing).
- **gene 경계 = 얼마나 쏟아내는가.** UTR이 있을 때만 폭발한다(1,014 → 12,628).
  UTR이 없으면 경계를 바꿔도 생성량이 거의 안 변한다(722 → 1,040).
  → **UTR+틀린 경계** 조합이 과다생성을 유발하고 정밀도를 희석한다.

### 🔬 arm G (09:46) — UTR만 고쳐서는 안 된다. CDS도 함께여야 한다

`helixertairutr` (Helixer CDS + TAIR10 UTR): added=350, **TAIR10_alt=0.9%**, recall=0.1%

전체 조건표 (맞은 수 = 추가 × 정밀도):

| CDS | UTR | 경계 | 추가 | 정밀도 | 맞은 수 |
|---|---|---|---|---|---|
| TAIR10 | TAIR10 | TAIR10 | 1,014 | **18.2%** | 185 |
| TAIR10 | TAIR10 | Helixer | 12,628 | 2.2% | **275** |
| TAIR10 | 없음 | TAIR10 | 722 | 4.6% | 33 |
| TAIR10 | 없음 | Helixer | 1,040 | 3.2% | 33 |
| **Helixer** | **TAIR10** | CDS+UTR 범위 | 350 | **0.9%** | **3** |
| Helixer | Helixer | Helixer | 2,462 | 0.0% | 1 |
| Helixer | 없음 | CDS 범위 | 528 | 0.0% | 0 |

**부분 보정은 통하지 않는다.** UTR만 정답으로 바꿔도 0.9%에 머문다. TAIR10 UTR은 TAIR10
CDS에 맞춰 잘린 것이라, Helixer CDS에 붙이면 UTR–CDS 경계가 어긋난 구조가 된다.
프롬프트는 **CDS와 UTR이 한 벌로 정확**해야 의미가 있다.

주의: arm G의 23,961 locus 중 13,925(58%)는 Helixer CDS가 TAIR10 UTR 영역을 침범해
UTR을 받지 못했다. 그만큼 희석됐지만, 나머지 42%가 전부 18%였다면 전체가 ~7.5%로
나왔어야 한다. 실측 0.9%는 UTR을 받은 쪽도 낮다는 뜻이다.

### 💡 실용적 발견 (10:1x) — answer-blind 필터가 프레임 문제를 거의 해소한다

`scratchpad/orf_splice_filter.py`. 게놈과 예측 좌표만 보는 두 필터(완전한 ORF,
모든 인트론이 GT..AG)를 적용했다. 참조를 보지 않으므로 데이터 스누핑이 아니다.

| | arm F (TAIR10 어노테이션 + Helixer 프레임) | Helixer 실제 프롬프트 |
|---|---|---|
| 필터 전 | 12,628 / 275 / 2.2% | 2,462 / 0 / 0.0% |
| ORF 완전 | 4,512 (35.7%) / 275 / 6.1% | **224 (9.1%)** / 0 / 0.0% |
| ORF + canonical splice | **1,547 / 262 / 16.9%** | 178 / 0 / 0.0% |

**arm F는 필터만으로 2.2% → 16.9%가 된다 — 원고 18.1%와 사실상 같은 수준.**
87.8%를 버리면서 정답은 95%(275→262) 유지했다.

→ **gene 경계(과다생성) 문제는 후처리로 해결 가능하다.** 원고에 쓸 수 있는 결과다.
→ 반면 Helixer 실제 프롬프트는 필터를 거쳐도 0%다. 생성물이 구조적으로도 나쁘다
   (ORF 완전 비율 35.7% vs **9.1%**). 프롬프트 품질이 생성물의 구조 타당성까지 좌우한다.

⚠️ 검증기 버그 기록: 처음에 "ORF 완전 0개"가 나왔는데, 게놈 FASTA가 Ensembl 이름
(`1,2,…,Mt,Pt`)이고 어노테이션은 TAIR 이름(`Chr1,…,ChrM,ChrC`)이라 **모든 조회가 실패**한
것이었다. 별칭 매핑 후 검증기가 정상임을 확인했다(⚠️ 당시의 "94.0%가 ORF 완전"은 라벨 오류 —
완전 ORF는 99.9%, 94~95%대는 두 기준 모두의 통과율이다. 2026-08-10 재측정)(원고 기술 85–98%와
일치) 검증기가 정상임을 확인했다. **0을 결과로 보고하기 전에 대조군으로 검증할 것.**

### 🔬 arm H (13:53) — ANNEVO에서도 UTR 기증만으로는 안 된다

`annevotairutr` (ANNEVO CDS + TAIR10 UTR): added=345, **TAIR10_alt=1.2%**, recall=0.1%

ANNEVO는 gene 경계가 **10.8%** 일치해(Helixer 0.004%) 과다생성 교란이 거의 없는,
UTR 기증에 가장 유리한 조건이었다. 그런데도 1.2%다(Helixer 버전 0.9%).

| 프롬프트 | 정밀도 |
|---|---|
| TAIR10 CDS + TAIR10 UTR + TAIR10 경계 | **18.2%** |
| TAIR10 CDS + TAIR10 UTR + Helixer 경계 | 2.2% → **필터 후 16.9%** |
| **ANNEVO CDS + TAIR10 UTR** | **1.2%** |
| **Helixer CDS + TAIR10 UTR** | **0.9%** |
| Helixer CDS + Helixer UTR | 0.0% |
| ANNEVO CDS + UTR 없음 | 0.0% |

**→ CDS 정확도가 진짜 벽이다.** 도구 CDS(68~77% 정확)에 정답 UTR을 붙여도 1% 언저리다.
TAIR10 수준의 CDS가 있어야 UTR이 의미를 갖는다.

### 📌 ANNEVO 프로필 (Helixer와 다른 실패 방식)

| | Helixer | ANNEVO |
|---|---|---|
| CDS 정확도 | 76.9% | 68.6% |
| UTR | 있음, **정확한 것 0/17,830** | **아예 없음** (exon 좌표 = CDS 좌표, 138,151개 완전 동일) |
| gene 경계 일치 | 1 / 25,657 (0.004%) | **2,656 / 24,600 (10.8%)** |
| 추가 구조 | 2,462 | **348** |
| 프롬프트 정확 locus에서 추가 | 1,296 | **2** |
| 생성물 ORF 완전 비율 | 9.1% | — |

Helixer = "틀린 UTR로 과다생성", ANNEVO = "UTR이 없어 생성 억제". 결과는 둘 다 0.0%.

### 🔧 모델로 UTR을 붙일 수 있는가 → 붙지만 3.0%만 정확

TAIR10 CDS만 주고 UTR을 생략했을 때(`tair10self`), 프롬프트 구조를 유지한 23,861개 중:
- UTR을 붙임 **19,735 (82.7%)**, 안 붙임 4,126
- 붙인 UTR이 TAIR10과 **정확 일치 594 (3.0%)**, 10bp 이내 근접 2,179 (all-or-nothing이라 무의미)

**2단계 파이프라인(CDS→UTR생성→재프롬프트)이 안 되는 이유**: 1단계가 CDS 품질에 의존한다.
모델이 만든 UTR 행 수 — TAIR10 CDS **44,337** vs Helixer CDS **175** vs ANNEVO CDS **138**.
CDS가 부정확하면 UTR을 아예 안 만든다.

### ⛔ 저자 목표(Helixer에서 18% 이상) — 미달, 도달 불가

18% 정밀도에는 **UTR 정확 + 경계 정확**이 동시에 필요하다. Helixer는
UTR **0/17,830**, 경계 **1/25,657**로 둘 다 충족하지 못한다.
UTR을 빼면 상한이 4.6%로 떨어지고, 경계만 고쳐도(`helixer_reframed`) 0.0%다.

**단, 지표가 recall이면 결론이 뒤집힌다.** arm F(경계 틀림)가 recall 5.3%로 가장 높고
맞은 개수도 275개로 최다다. "TAIR10 대체전사체를 얼마나 회수하는가"가 목적이면
과다생성이 유리하다. 저자 판단 필요.

### 원고 함의

고찰의 Helixer/ANNEVO 문장을 뺀 결정이 옳았음이 실험으로 뒷받침된다. 18.1%는
**그 어노테이션 자신을 프롬프트로 줄 때**의 수치이고, 다른 도구의 어노테이션에는
전이되지 않는다. 이유도 특정됐다: 프롬프트 UTR의 정확도.

### ✅ 커밋·푸시·위키 완료 (2026-08-07)

**transgenic repo** (`origin/main`, 전부 푸시 완료, **225 테스트 통과**)
- `210f060` 9-arm 실험 + 34번 필드 추가 + `36_filter_additions_structurally.py` 신규
- `b139f6f` synthetic gene의 `GM=` 상속 수정 (I-3 게이트가 잡은 결함)
- `c90d6ff` `37_analyse_prompt_requirements.py` 신규 — 결론의 근거 3종 측정
  (utr-accuracy / utr-closeness / generated-utr). 스크래치 수치를 재현 확인함

**랩 위키** (`f760a5c`, 푸시 완료, 링크 0 broken)
- `projects/transgenic-polishing-benchmark/updates/2026-08-07_what-completion-mode-needs-from-its-prompt.md`
- README Date/Status/Task 6, `index.md`, `log/2026.md`, `log.md` 갱신

**36번 필터 실제 검증** (정식 스크립트가 스크래치 결과 재현):
추가 12,628 → 1,547 (87.7% 제거), 정답 262/275 유지,
TAIR10 alt **2.2% → 16.9%**, AtRTD3 **3.6% → 28.4%**

## 🔁 전체 10회 재실행 중 (2026-08-09 21:04 시작)

저자 지시: "criptic"은 오타였고, **전체를 다시 10번** 돌릴 것.

**`polishing_benchmark/run_repeat.sh`** (harness 백그라운드 `bxham85v9`, lock으로 중복 방지).
9개 arm × 10 라운드, 매번 `--force`로 chunk를 새로 생성한다.

- 로그: `polishing_benchmark/results/repeat/repeat_*.log`
- **회차별 결과: `polishing_benchmark/results/repeat/signatures.tsv`** (round / arm / 구조
  시그니처 / locus 수 / 구조 수 / 정밀도)
- 회차별 채점 JSON: `results/repeat/run{N}_{arm}.json`

### ⚠️ 먼저 실증한 것 — 모델 출력은 결정론적이다

ChrM chunk를 `--force`로 재생성해 원본과 비교:

| 비교 단위 | 1회차 | 2회차 |
|---|---|---|
| 파일 md5 | `160e5c85…` | `e3a18056…` ← **다름** |
| 좌표·구조만 (ID 제외) | `3609f1ef5d9b2b2c` | **동일** |
| GM별 CDS 구조 집합 | `c7d6b1cb3bf1b718` (31 locus, 37 구조) | **동일** |

파일 md5가 달랐던 이유는 transcript ID가 UUID이기 때문. **채점이 읽는 모든 단위는
비트 단위로 같다.** `do_sample=False, num_beams=2`와 일치하는 결과다.

→ 따라서 10회 결과는 전부 동일할 것으로 예상된다. 그래도 지시대로 진행하되,
   **회차마다 시그니처를 남겨 "동일하다"를 가정이 아니라 측정으로 만든다.**
   어느 회차가 어긋나면 모니터가 `DIVERGED`로 알린다.

### ⏱️ 비용

**회당 약 20시간(9 arm), 10회 전부면 약 200시간(8일).**
라운드는 끝나는 대로 기록되므로 중간에 멈춰도 완료된 회차는 그대로 남고 비교 가능하다.
세션이 8일을 넘기지 못할 가능성이 크므로, 완료 회차까지만 남을 수 있다.

### (해소) "criptic 10번" — 오타였음

2026-08-07 지시 중 **"criptic 10번 돌리고"**의 대상을 특정하지 못했다.
저장소·PATH·`/data/gpfs/assoc/pgl/tools/`·위키 문서 어디에도 그런 이름이 없다.
오타로 짐작되나 추측 실행은 하지 않았다. 나머지 자동 진행은 전부 수행했다.

참고: 이 벤치마크는 `do_sample=False, num_beams=2`로 **결정론적**이라
같은 입력을 10번 돌려도 같은 결과가 나온다. 반복이 의미를 가지려면
샘플링을 켜거나(`--do-sample`) 입력을 바꿔야 한다.

### (이전) 미커밋 목록 — 위에서 전부 해소됨

- `revision/scripts/33_run_polishing_benchmark.py` — TOOLS에 실험 arm 7개 추가
- `revision/scripts/34_score_as_additions.py` — primary/any/prompt-correct 필드 + 테스트
- `revision/scripts/35_build_frame_shift_prompts.py` — 신규 (arm 생성)
- `transgenic_comparison/standardize_gff.py` — synthetic gene에 `GM=` 상속 (I-3 수정)
- 원격 GPU 호스트: `examples/{primary_lock.py,prompt_mode_primary_locked.py}` (untracked)
- 테스트 전체 196개 통과

---

## 🌙 밤샘 실행 경과 (2026-08-06 18:41 시작)

### 저자 목표 (2026-08-06 18:35)

> "helixer에 as 구현하고 18% 이상을 맞춘다. 전체를 진행해. 지금 밤이라 확인할 수 없으니 최대한 자동으로 진행해"

### ⚠️ 19:00~19:10 사이에 겪은 문제 세 가지 (전부 해결, 기록용)

1. **`nohup setsid`로 띄운 스크립트가 계속 죽었다.** 원인은 확인 명령이 타임아웃될 때
   프로세스 그룹째 종료된 것. → harness 백그라운드(`run_in_background`)로 전환해 해결.
   **교훈: 장시간 작업은 `nohup setsid`로 띄우지 말 것.**
2. **`ps aux | grep -F '33_run_pol[i]shing'`이 항상 빈 결과였다.** `-F`가 대괄호를
   리터럴로 처리한다. 이걸 "프로세스가 죽었다"로 오독해서 중복 실행을 만들었다.
   → 확인은 `pgrep -af` 사용.
3. **새 입력 파일이 원격 GPU 호스트에 없어 세 arm이 45초 만에 전멸.**
   → `scp`로 직접 푸시하고 md5 대조 완료.

### 자동 실행 (01:09 기준 — 우선순위 재정렬)

**`polishing_benchmark/run_utr_arms.sh`** (PID 13109, harness 백그라운드 `bpvr62qke`).

`tair10self`가 4.6%로 원고 조건 재현에 실패했으므로, **재현이 최우선 과제**가 됐다.
UTR 없는 프레임 비교는 원고가 한 번도 돌린 적 없는 영역에서 재는 것이라 의미가 약하다.

| 순서 | arm | 묻는 것 | 예상 |
|---|---|---|---|
| 1 | `tair10selfutr` | **~18% 회복되나?** 안 되면 재현 실패 원인이 UTR도 아니다 | ~03:20 |
| 2 | `tair10helixerframeutr` | 원고와 같은 조건에서의 순수 프레임 효과 | ~05:35 |
| 3 | `tair10helixerframe` | UTR 없는 버전(중단분 이어서). 1·2가 답하면 생략 가능 | ~07:45 |

`tair10helixerframe`은 GPU를 위 둘에 넘기려고 중단했다. chunk는 보존돼 이어서 실행된다.

### 이전 구성 (20:20 — 참고용)

**`polishing_benchmark/run_remaining_arms.sh`** (PID 24853, harness 백그라운드 `bk32yjfsd`).
lock 파일(`.arms_running.lock`)로 중복 실행을 막는다.

- **PID 23869 `helixer_reframed` 실행 중** (20:12 시작) — 목표 arm. 부모 스크립트가 죽어
  고아가 됐지만 정상 동작 중이라 그대로 두고, 오케스트레이터가 완료를 기다렸다 채점한다.
- 이후 순서: `tair10self` → `tair10helixerframe`
- 예상: helixer_reframed ~22:45, tair10self ~01:00, tair10helixerframe ~03:15

### ⚠️ 19:00~20:20에 겪은 문제 (전부 해결, 재발 방지용 기록)

1. **`nohup setsid`로 띄운 스크립트가 계속 죽었다.** 확인 명령이 타임아웃될 때
   프로세스 그룹째 종료됐다. → harness 백그라운드(`run_in_background`)로 전환.
   **교훈: 장시간 작업을 `nohup setsid`로 띄우지 말 것.**
2. **`ps aux | grep -F '33_run_pol[i]shing'`이 항상 빈 결과.** `-F`가 대괄호를 리터럴로
   처리한다. 이를 "죽었다"로 오독해 중복 실행을 만들었다. → `pgrep -af` 사용.
3. **`pkill -f "패턴"`이 자기 자신을 죽였다** (내 명령줄에 그 패턴이 있어서).
   → 정리는 정확한 PID로 `kill`.
4. **새 입력 파일이 원격 호스트에 없어 세 arm이 45초 만에 전멸.**
   → `scp` 푸시 + md5 대조 완료.
5. **🔴 tair10self 1차 실행 폐기** — 위 2번이 낳은 중복 실행으로 두 러너가 같은 원격
   경로에서 경합, `genome2GSFDataset`이 공유 DuckDB에 append해서 gene 행이 정확히 2배가
   됐다(6,350 locus에 12,362 행). 33번의 무결성 게이트가 잡았다.
   → 원격·로컬 산출물 전부 삭제 후 새로 시작. **재개가 아니라 클린 스타트다.**

- 로그: `polishing_benchmark/logs/frame_experiment_*.log`
- **결과 요약: `polishing_benchmark/results/frame_experiment_status.txt`** ← 아침에 이거부터
- 실행 순서와 이유:
  1. `tair10self` — **기준선**. 원고 18.1%가 *이 파이프라인에서* 재현되는지.
     원고 수치는 다른 스크립트(28번)·다른 실행이라 대조군이 될 수 없다
  2. `helixer_reframed` — **목표 달성 후보**. Helixer 자신의 CDS만 쓰고 gene 경계를
     CDS 범위로 재설정. 참조 지식을 안 쓰므로 실제 배포 가능한 형태
  3. `tair10helixerframe` — 기전 확정용. 1·2가 이미 답하면 결정에 영향 없음

예상 완료: 각 arm 약 2~2.5시간 → **02:00~03:00경**. ANNEVO locked 완료 후 시작.

### 왜 이 실험인가 — 원인 규명 결과 (18:00)

배제된 설명 (전부 데이터로 확인):
- 프롬프트 품질 아님 — 프롬프트가 TAIR10 transcript와 정확히 같은 locus에서도 0.1~0.3%
- 프롬프트 파괴 아님 — 락으로 184→1로 없애도 0.2%
- UTR 유무 아님 — UTR 주는 Helixer(0.1%) = 안 주는 GeMoMa(0.1%)
- 양 부족 아님 — 10.9배 늘려도 정밀도 불변

**유력 원인 — 좌표 프레임.** 원고 Methods: 학습 윈도우는 gene 행 기준으로 6,144 배수 확장,
GSF 좌표는 그 윈도우 상대 좌표. `preprocess.py` 174·211행이 확인.
A. thaliana는 학습 종(held-out은 maize뿐).

| | 참조와 겹치는 gene | gene 경계 완전 일치 |
|---|---|---|
| Helixer | 25,657 | **1개 (0.0%)** |
| ANNEVO | 24,600 | 2,656 (10.8%) |

→ CDS가 같아도 모델이 보는 윈도우와 좌표 원점이 학습 때와 다르다.

원고 고찰 78행이 같은 실패를 이미 기록: Chr4 파일럿, 제약 없는 스캔 → locus당 ~12 transcript,
정밀도 4.4%. 락 실험이 이를 재현(locus당 2.87, 0.2%).

### 실험 데이터 (생성 완료)

`35_build_frame_shift_prompts.py`가 만든 것. **CDS는 완전 동일, gene 경계만 다름:**

| arm | 입력 파일 | locus | 내용 |
|---|---|---|---|
| A | `tair10self_Athaliana.gff3` | 24,210 | TAIR10 primary CDS + TAIR10 경계 |
| B | `tair10helixerframe_Athaliana.gff3` | 24,210 | 같은 CDS + Helixer 경계 |
| D | `helixer_reframed_Athaliana.gff3` | 27,039 | Helixer CDS + CDS범위 경계 |

A/B는 프레임 동일 locus가 **1개뿐** → 대조가 깨끗하다.
Helixer 경계가 TAIR10 CDS를 포함하는 locus만 써서 CDS가 잘리는 일은 없다.

### 🔴 helixer_reframed 결과 (22:26 완료) — 목표 미달

| arm | 추가 구조 | TAIR10 alt | AtRTD3 | any | recall |
|---|---|---|---|---|---|
| Helixer stock | 2,462 | 0.0% | 0.3% | 0.2% | 0.0% |
| **Helixer reframed** | **528** | **0.0%** | **0.6%** | 0.6% | 0.0% |

gene 경계를 CDS 범위로 좁히자 **추가가 1/5로 줄었고**(2,462 → 528) 정확도는 그대로다.
18% 목표에 한참 못 미친다. 프레임을 CDS 범위로 바꾸는 것은 해법이 아니다.

⚠️ 단, `tair10self` 기준선 없이는 이 결과를 최종 해석할 수 없다. 실행 중.

### ✅ 채점기 검증 완료 (22:35) — 우리 34번은 원고 18.1%를 재현한다

GPU 없이 확인 가능했다. 원고 수치가 나온 예측 파일:
`transgenic_comparison/standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3`

| 채점기 | 추가 | TAIR10 alt | AtRTD3 | recall |
|---|---|---|---|---|
| 원고의 28번 (그대로 재실행) | 1,103 | **18.1%** | 18.5% | 3.6% |
| **우리 34번, 같은 예측 파일** | 1,035 | **18.6%** | 18.8% | 3.7% |

차이는 locus 필터뿐(24,209 vs 27,413 — `tair10self` 입력이 Helixer 경계 포함 조건으로
걸러졌기 때문). `gm_paired=24,209`, `overlap_fallback=0`으로 페어링도 완벽.

**→ "채점기 정의가 달라서 낮게 나온다"는 가설 기각. Helixer의 0.0~0.6%는 실제 현상이다.**

이로써 `tair10self` arm의 역할이 바뀐다: 채점기 검증은 이미 끝났으므로, 그 arm은
"**같은 프롬프트를 새로 생성했을 때도 18%가 재현되는가**"(실행 조건 재현성)를 잰다.

### 🔬 결정적 대조 (22:45) — 같은 CDS인데 Helixer 프롬프트에서는 "시도조차 안 한다"

원고 실행이 **맞힌** 200개 locus를 뽑아, 같은 locus에서 Helixer 프롬프트 실행이 무엇을
했는지 봤다. (`scratchpad/compare_hits.py`, CDS 구조 동일성으로 locus 대응 — 좌표
페어링 불필요)

| | |
|---|---|
| 원고 실행이 올바른 alternative를 추가한 locus | **200** |
| 그중 Helixer가 **동일한 CDS**를 프롬프트로 준 locus | **169** |
| 그 169개에서 Helixer 실행이 추가한 구조 | **0** |
| 그 169개에서 맞은 것 | **0** |

**정확도가 낮은 게 아니라 생성 자체를 안 한다.** Helixer 전체 평균이 locus당 0.09개
(2,462/27,039)이므로 169개면 15개쯤 기대되는데 하나도 없다.

프롬프트 CDS가 글자 그대로 동일하므로 남은 차이는 **좌표 프레임**뿐이다. GSF 좌표는
윈도우 상대값이고 윈도우는 gene 행 기준이라, 같은 CDS도 gene 경계가 다르면 다른 숫자로
디코더에 들어간다. → `tair10helixerframe` arm이 이를 결정적으로 검증한다.

### 🔬 프레임 가설 정교화 (22:55) — shift "크기"는 무관하다

`scratchpad/shift_vs_output.py`. 공급 CDS가 실제 TAIR10 transcript와 동일한 locus만
대상으로, gene 경계가 TAIR10에서 얼마나 떨어졌는지에 따라 생성률을 봤다.

| 경계 어긋남 | locus | 추가 있음 | 비율 |
|---|---|---|---|
| 1–100 bp | 9,387 | 427 | 4.5% |
| 101–1,000 bp | 9,554 | 448 | 4.7% |
| 1,001–10,000 bp | 293 | 25 | 8.5% |
| 전체 | 19,237 | 900 | **4.7%** |

**평평하다.** 10 bp와 1,000 bp가 같다. 그리고 Helixer 생성률 4.7%는 TAIR10 프롬프트의
3.9%(1,076/27,413)와 비슷하거나 오히려 높다.

→ **"프레임이 생성을 억제한다"는 그림은 틀렸다.** 모델은 Helixer에서도 비슷한 빈도로
추가를 만든다. 문제는 양이 아니라 **만든 것이 전부 틀린다**는 것이다.

→ 프레임이 원인이라면 "얼마나 어긋났나"가 아니라 "**정확히 일치하는가**"여야 한다.
이는 **암기 가설**과 부합한다 — 학습 때 본 좌표와 정확히 같을 때만 정답이 재생된다.
경계 완전 일치는 25,657개 중 1개뿐이므로 Helixer에서는 사실상 절대 발동하지 않는다.

**`tair10helixerframe`이 이를 결정적으로 가른다** (CDS는 TAIR10, 경계만 Helixer):
- 0%에 가까움 → 암기 확정. **18% 목표는 원리적으로 도달 불가**(정답 좌표를 알아야 하므로)
- 18% 근처 → 좌표가 아니라 CDS 내용이 관건. 다른 접근 여지 있음

### 🔴 tair10self 결과 (01:0x) — 원고 조건 재현 실패, 원인은 arm 설계 결함(UTR 누락)

`tair10self`: added=722, **TAIR10_alt=4.6%**, AtRTD3=5.1%, recall=0.6%

원고 예측을 같은 채점기로 잰 값은 18.6%인데 4.6%가 나왔다. **재현 실패다.**

**원인: 내가 만든 arm에 UTR이 없다.** `35_build_frame_shift_prompts.py`는 CDS 행만
쓴다. 원고 실험은 TAIR10을 그대로 프롬프트로 줬으므로 exon−CDS에서 도출되는 UTR이
GSF에 들어갔다. GSF vocab에 `five_prime_UTR1-50`/`three_prime_UTR1-50` 토큰이 있으므로
프롬프트 내용 자체가 다르다.

| 조건 | CDS | UTR | gene 경계 | TAIR10 alt |
|---|---|---|---|---|
| 원고 예측 (34번으로 재채점) | TAIR10 | **있음** | TAIR10 | **18.6%** |
| `tair10self` | TAIR10 | **없음** | TAIR10 | **4.6%** |
| Helixer stock | Helixer | 있음 | Helixer | 0.0% |
| `helixer_reframed` | Helixer | 없음 | CDS 범위 | 0.0% |

**UTR 유무만으로 4배(18.6→4.6).** 앞서 "UTR 유무는 원인이 아니다"라고 적었던 것은
도구 간 비교(Helixer UTR있음 0.1% vs GeMoMa UTR없음 0.1%)에 근거한 것이었고,
그 도구들은 다른 이유로도 낮았다. **통제된 비교에서는 UTR이 큰 요인이다. 정정한다.**

**해석 주의:** `tair10helixerframe`은 UTR이 없으므로 18.6%가 아니라 **4.6%와 비교**해야
한다. 프레임 효과는 그 대비로 읽을 것.

### ✅✅ 원고 조건 재현 성공 (01:1x) — UTR이 열쇠였다

`tair10selfutr`: added=1,014, **TAIR10_alt=18.2%**, AtRTD3=18.3%, recall=3.6%

| 조건 | CDS | UTR | 경계 | TAIR10 alt |
|---|---|---|---|---|
| 원고 예측 (34번 재채점) | TAIR10 | 있음 | TAIR10 | 18.6% |
| **`tair10selfutr`** | TAIR10 | **있음** | TAIR10 | **18.2%** |
| `tair10self` | TAIR10 | **없음** | TAIR10 | 4.6% |
| Helixer stock | Helixer | 있음 | Helixer | 0.0% |

**UTR 유무가 4.6% ↔ 18.2%를 가른다.** 우리 파이프라인이 원고 조건을 재현하므로
이후 모든 비교가 유효해졌다.

**다음 질문: Helixer도 UTR을 주는데 왜 0%인가?**
→ Helixer의 UTR이 TAIR10과 얼마나 다른지가 관건. GPU 없이 계산 가능하므로 먼저 확인한다.
→ `tair10helixerframeutr`(~05:35)가 프레임 효과를 원고 조건에서 처음으로 정확히 잰다.

### 🎯 원인 규명 완료 (01:2x) — 병목은 **프롬프트의 UTR 정확도**다

`scratchpad/utr_accuracy.py`. Helixer 입력에서 CDS가 실제 TAIR10 transcript와 동일한
locus만 골라, UTR도 일치하는지와 그에 따른 정확도를 봤다.

| Helixer locus (CDS 정확) | locus | 추가 | 맞음 | 정밀도 |
|---|---|---|---|---|
| CDS 정확 + **UTR 정확** | **0** | — | — | — |
| CDS 정확 + UTR 부정확 | 17,830 | 1,174 | 1 | **0.1%** |

**Helixer는 CDS를 76.9% 맞추면서 UTR은 17,830개 중 0개를 맞춘다.**

이것이 전체 그림을 완성한다:

| CDS | UTR | 정밀도 | |
|---|---|---|---|
| TAIR10 | TAIR10 (정확) | **18.2%** | `tair10selfutr` |
| TAIR10 | 없음 | 4.6% | `tair10self` |
| Helixer (76.9% 정확) | Helixer (0% 정확) | 0.0% | stock |
| Helixer | 없음 | 0.0% | `helixer_reframed` |

**결론: 모델은 프롬프트의 UTR이 정확할 때만 올바른 alternative를 만든다.**
CDS만 맞고 UTR이 틀리면 0.1%로, UTR을 아예 안 주는 것(4.6%)보다도 나쁘다.

### 🔬 효과는 all-or-nothing이다 (01:3x) — 점진적 개선 여지 없음

`scratchpad/utr_closeness.py`. CDS가 정확한 locus에서, UTR이 TAIR10과 몇 bp 다른지에
따라 정확도를 봤다.

| UTR 차이 | locus | 추가 | 맞음 | 정밀도 |
|---|---|---|---|---|
| **1–10 bp** | 172 | 7 | **0** | **0.0%** |
| 11–50 bp | 3,351 | 179 | 0 | 0.0% |
| 51–200 bp | 9,498 | 598 | 1 | 0.2% |
| 201–1,000 bp | 4,763 | 387 | 0 | 0.0% |
| >1,000 bp | 46 | 3 | 0 | 0.0% |

**5 bp 틀린 것과 500 bp 틀린 것이 같다.** UTR을 "더 정확하게" 만들어도 정확히 맞기
전까지는 아무 이득이 없다. 이는 암기 가설과 정확히 부합한다 — 학습 때 본 좌표 조합과
**완전히 동일**할 때만 정답이 재생된다.

### 🔄 arm F 결과 (03:5x) — 앞선 "완전 일치해야만 정답 재생"은 **틀렸다. 정정한다**

`tair10helixerframeutr`: added=**12,628**, TAIR10_alt=**2.2%**, AtRTD3=3.6%, recall=**5.3%**

| CDS | UTR | 경계 | 추가 | 정밀도 | **맞은 수** | recall |
|---|---|---|---|---|---|---|
| TAIR10 | TAIR10 | TAIR10 | 1,014 | **18.2%** | 185 | 3.6% |
| TAIR10 | TAIR10 | **Helixer** | **12,628** | 2.2% | **275** | **5.3%** |
| TAIR10 | 없음 | TAIR10 | 722 | 4.6% | 33 | 0.6% |
| Helixer | Helixer | Helixer | 2,462 | 0.0% | 1 | 0.0% |

**gene 경계만 바꿔도 정답을 만든다 — 오히려 더 많이(275 vs 185), recall도 높다(5.3%).**
바뀌는 것은 생성량(12배 폭증)과 그에 따른 정밀도 희석이다.

따라서 두 요인의 역할이 다르다:

- **UTR 정확도 = 정답을 만들 수 있는가** (없거나 틀리면 33개·1개, 있으면 185개·275개).
  1–10 bp만 틀려도 0%인 all-or-nothing.
- **gene 경계 = 얼마나 많이 쏟아내는가**. 틀리면 12배 과다생성 → 정밀도 희석.
  원고 고찰 78행의 Chr4 파일럿(무제약 스캔 시 locus당 ~12개, 정밀도 4.4%)과 같은 현상.

⚠️ 01:45에 적었던 "모델은 프롬프트가 학습 데이터와 **완전히 동일**할 때만 올바른
alternative를 재생한다"는 서술은 **arm F가 반증했다.** 정확한 서술은 위와 같다.

### ⛔ 저자 목표(18% 이상)에 대한 판정

**현 조건에서는 도달 불가능하다.** 근거:

- 18.2%는 UTR이 **정확할 때**만 나온다
- Helixer의 UTR 정확도는 **0/17,830 = 0.0%**
- UTR을 빼면 상한이 4.6%로 떨어진다(그것도 CDS가 TAIR10일 때. Helixer CDS면 0.0%)
- "Helixer CDS + TAIR10 UTR" arm은 만들 의미가 없다 — Helixer CDS가 TAIR10과
  일치하는 locus에 TAIR10 UTR을 붙이면 그건 곧 TAIR10 primary이고, 이미 18.2%로 측정됐다

→ 18%를 넘기려면 **프롬프트의 UTR을 정확히 만들어야 하고**, 그건 TransGenic 쪽이 아니라
   입력 어노테이션(또는 long-read 기반 UTR 보정)의 문제다. 모델 사용법을 바꿔서 얻을 수
   있는 것이 아니다.

**arm F를 반영한 보강 (03:5x):** 정밀도 18%가 되려면 두 조건이 **동시에** 필요하다.
1. UTR이 정확해야 정답을 만들 수 있고 (Helixer: 0/17,830)
2. gene 경계가 정확해야 과다생성으로 희석되지 않는다 (Helixer: 1/25,657)

Helixer는 둘 다 충족하지 못한다. 다만 arm F는 **정밀도가 아니라 recall이 목표라면**
이야기가 달라짐을 보여준다 — 경계가 틀려도 정답 개수와 recall은 오히려 늘었다.
저자가 정한 기준이 "18% 정밀도"(원고와 같은 지표)이므로 판정은 미달 그대로다.

### 판정 기준

- A가 18% 근처 → 프레임 가설 지지, 원고 수치는 이 파이프라인에서 재현됨
- A도 낮음 → 원고 18.1%와 우리 채점 정의가 다르다는 뜻. **재현 실패부터 규명해야 함**
- D가 18% 이상 → **저자 목표 달성**
- D도 낮음 → 프레임 재설정만으로는 부족. 다른 접근 필요

### 코드 변경 (미커밋)

- `33_...py` TOOLS에 arm 3개 추가 (기존 arm 동작 무변경, 테스트 113개 통과)
- `34_...py` 필드 추가: `added_matching_TAIR10_primary`, `..._any_transcript`,
  `loci_where_prompt_equals_a_reference_transcript` 및 그 부분집합 정밀도 (테스트 34개 통과)
- `35_build_frame_shift_prompts.py` 신규
- 전체 147개 테스트 통과

---

## 🔒 primary-lock 실험 — 전체 실행은 중단, Chr1 결과로 결론 (19:05)

**락 arm은 전체 파이프라인을 통과하지 못했고, 통과시킬 이유도 없다고 판단해 중단했다.**

- `helixer_locked`가 두 개의 무결성 게이트에 연속으로 걸렸다:
  1. **I-3** (gene 행 수 ≠ distinct GM= 수). 원인은 33번 주석의 추정(synthetic
     `gene_for_*`)이 **맞았지만 파일이 달랐다** — 실제 사용 파일은
     `transgenic_comparison/standardize_gff.py`이고 `transgenic/scripts/standardize_gff.py`가
     아니다. 처음에 후자를 고쳐서 효과가 없었다. **수정 완료**(orphan mRNA의 `GM=`을
     synthetic 부모에 상속). 196개 테스트 통과. ⚠️ 미커밋
  2. 수정 후 다음 게이트: **65개 locus가 설명되지 않은 이유로 소실**(임계 40).
     락이 긴 유전자에서 `max_length=2048`을 넘기게 만든 것으로 추정. 미규명.
- **중단 근거**: Chr1(7,123 locus)에서 이미 답이 나왔다 — 프롬프트 파괴를 184→1로
  없애도 정밀도는 0.2%로 불변. 게이트를 올려 강행하면 이미 답이 난 질문에 GPU를 쓴다.
- `annevo_locked`도 같은 이유로 중단(PID 6955 종료).

## 🔒 primary-lock 실험 상세 (2026-08-06 15:23 시작) — 저자 지시로 진행

### 저자 질문

"메인 로커스를 건들지 않고 isoform만 추가하면? primary를 고정하고 alternative만 추가하게 해서
그것들이 TAIR10/AtRTD3와 같은 것을 만들어내는지"

### 사후 관찰로는 이미 답이 나와 있었다 (실험 전 확인)

프롬프트가 **자발적으로 보존된** locus에 한정해도 정확도가 회복되지 않는다:

| 도구 | 프롬프트 생존 locus | 거기서 추가 | TAIR10 일치 | 정밀도 |
|---|---|---|---|---|
| Helixer | 26,288 | 1,019 | 1 | 0.1% |
| ANNEVO | 24,997 | 5 | 0 | 0.0% |
| GeMoMa | 21,641 | 3,778 | 1 | 0.0% |
| EGAPx | 16,112 | 2,972 | 5 | 0.2% |

파괴된 locus 쪽도 Helixer 1,443개 중 0건, GeMoMa 21,481개 중 15건(0.07%).
→ 저자는 그럼에도 **강제 제약 실험을 지시**했고 진행 중이다.

### 근본 원인 (코드 확인)

`get_single_decoder_input_ids`는 label을 `>`(토큰 17)에서 자르고 **features 앞부분만** 디코더에
주입한다. transcripts 섹션은 주입하지 않으므로 **모델이 primary transcript까지 새로 생성**한다.
그래서 프롬프트가 파괴된다.

### 구현 (원격 GPU 호스트에 배포 완료)

- `examples/primary_lock.py` — 순수 토큰 로직, torch 비의존. **단위 테스트 21개 통과**
- `examples/prompt_mode_primary_locked.py` — 핀된 `prompt_mode.py`의 복사본 +
  `prefix_allowed_tokens_fn` 한 줄. `>` 직후 프롬프트의 feature 체인을 토큰 단위로 강제하고,
  그 뒤는 전부 자유(alternative 강제 생성 안 함)
- **기존 `examples/prompt_mode.py`는 건드리지 않았다.** 핀 커밋 `5d9929e` 무변경, 새 파일은 untracked
- `33_run_polishing_benchmark.py`에 `helixer_locked` / `annevo_locked` arm 추가.
  도구 이름으로 스크립트가 결정되므로 라벨과 실제 실행이 어긋날 수 없다.
  입력은 base tool 파일 공용, 출력만 `_locked_*`로 분리. **기존 테스트 113개 통과**

### ✅ ChrM 스모크 검증 (락이 실제로 작동함)

| | 프롬프트 보존 | 추가 구조 | 추가 있는 locus |
|---|---|---|---|
| **락 적용** | **31/31 (100%)** | **21** | 19 |
| stock (기존) | 27/31 (87.1%) | 10 | 6 |

primary 100% 보존 확인. **추가 구조가 오히려 2배로 늘었다** — 모델이 primary 재작성에 쓰던
자리를 실제 alternative에 쓰는 것으로 보인다. 단 ChrM은 TAIR10 대체전사체가 거의 없어
정확도 판단 불가.

### 실행 중

```
PID 6955
로그: polishing_benchmark/logs/run_20260806_152308_locked_arms.log
명령: python3 revision/scripts/33_run_polishing_benchmark.py \
        --tool helixer_locked annevo_locked \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold
```
15:23 시작. stock 실행이 2.6시간이었으므로 **18:00~19:00 완료 예상**(제약 오버헤드로 더 걸릴 수 있음).
산출: `predictions/{helixer,annevo}_locked_completed.gff3`. 종료 감지 모니터 걸어 둠.

### 완료 후 할 일

1. `34_score_as_additions.py`로 두 락 arm 채점 (입력은 base tool의 `_Athaliana.gff3`)
2. stock vs locked 비교표 작성 → 저자 판단
3. 33번 수정 + 락 스크립트 커밋 여부 판단 (현재 미커밋)

### 남은 것 (기존)

- AS 추가 결과를 `polishing_benchmark/RESULTS.md`에 기록할지
- 랩 위키(`/data/gpfs/assoc/pgl/claude`) 항목 갱신 — 아직 안 함
- 결과 JSON은 저장소 밖, 커밋 대상 아님

**참고** — 두 에이전트(`as-scorer`, `gemoma-diff`)는 질문에 답하지 않고 idle 알림만 반복했다.
규명은 직접 수행했다. 에이전트 응답에 의존하지 말 것.

## ▶️ 이전 스냅샷 (2026-08-06 12:41)

### 현재 실행 중 (두 개)

**1. GPU 벤치마크 — Helixer / ANNEVO**
- PID 1149, 12:27 시작, 약 18:00 전후 완료 예상 (5-6시간)
- 중단 시 같은 명령 재실행 → 완료된 염색체는 자동 건너뜀
- 산출: `polishing_benchmark/predictions/{helixer,annevo}_completed.gff3` + 메타데이터

**2. AS 추가 채점기 구현 — 에이전트 `as-scorer` (⚠️ 세션 클리어 시 소멸 — 아래 지시로 재시작할 것)**
- 생성 대상: `revision/scripts/34_score_added_isoforms_external.py`
- 검증 조건: 기존 3도구(GeMoMa 21,739 / BRAKER3 128 / EGAPx 22,711) 숫자 일치 후 진행

### 원고 상태

폴리싱·refine 주장 5곳 제거, 응답서 2곳 정합성 회복 완료.
- v3 재생성, 200/200 검증 통과
- 초록 248단어 (제한 250), 제3자 주장 0건
- 커밋 전 파일 상태

### 완성될 표

| 프롬프트 | 추가 구조 | TAIR10 대체전사체 일치 | AtRTD3 일치 |
|---|---|---|---|
| TAIR10 자신 (기존) | 1,103 | 18.1% | 18.5% |
| AUGUSTUS 샘플링 | 43,433 | 1.3% | 10.2% |
| Helixer | 측정 중 | — | — |
| ANNEVO | 측정 중 | — | — |

### 결과 분기

- **AS 추가 성공**: 고찰의 지운 Helixer/ANNEVO 문장 되살릴 근거 확보
- **AS 추가 실패**: 현재 상태 유지

### 다음 작업

1. GPU 벤치마크 완료 모니터링 (18:00 전후)
2. AS 채점기 검증 결과 확인
3. 분기 지점에 따라: 원고 손질 또는 커밋 준비

## ▶️ 이전 스냅샷 (2026-08-06 12:45)

### 저자 결정 (2026-08-06): 폴리싱은 원고에서 배제, AS 추가 정확도로 전환

**결정 내용**
- **폴리싱 주장은 원고에서 완전히 제거한다.** 벤치마크로 성립하지 않음이 확인됨.
- **측정 대상은 "Helixer/ANNEVO 출력에 올바른 alternative splicing을 얼마나 추가하는가"**로 전환.
  이것이 도구의 실제 용도이고 원고 고찰이 이름을 대며 주장했던 바로 그것.
- 폴리싱 분석 자체는 `polishing_benchmark/RESULTS.md`에 기록으로 남기되 원고에는 넣지 않음.

### 🔄 지금 돌고 있는 것 (두 개)

**1. GPU 벤치마크 — Helixer / ANNEVO**
```
PID 1149
로그: polishing_benchmark/logs/run_20260806_122653_helixer_annevo.log
명령: python3 revision/scripts/33_run_polishing_benchmark.py --tool helixer annevo \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold
```
- 12:27 시작, 약 5~6시간 예상 → **18:00 전후 완료**
- 산출: `polishing_benchmark/predictions/{helixer,annevo}_completed.gff3` + `_provenance.json` + `_missing_loci.txt`
- 종료 감지 모니터 걸려 있음 (세션 유지 시)
- **중단되면 같은 명령 재실행하면 됨** — 완료된 염색체는 자동으로 건너뜀

**2. AS 추가 채점기 구현 — 에이전트 `as-scorer`**
- 만드는 것: `revision/scripts/34_score_added_isoforms_external.py` (+ 테스트)
- 하는 일: 외부 어노테이션을 프롬프트로 준 예측에서 **추가된 isoform만** 뽑아
  TAIR10 대체전사체 / AtRTD3와 대조
- `28_score_added_isoforms.py`(TAIR10 프롬프트 전용)의 일반화. 정의를 그대로 맞춰야 숫자가 비교 가능
- **검증 조건**: 기존 세 도구에서 `added_structures`가 GeMoMa 21,739(90 일치) /
  BRAKER3 128(0) / EGAPx 22,711(31)과 맞아야 함. 어긋나면 원인부터 규명

**재시작용 지시 (새 세션에서 그대로 사용)**
> `revision/scripts/34_score_added_isoforms_external.py` + 테스트를 만든다.
> 외부 어노테이션을 프롬프트로 준 예측에서 **추가된 isoform만** 뽑아 TAIR10 대체전사체 / AtRTD3와 대조.
> - `28_score_added_isoforms.py`(TAIR10 프롬프트 전용)의 **일반화**. 정의를 그대로 맞춰야 원고의 18.1%/18.5%와 비교 가능
> - `32_score_polishing.py`의 locus 페어링 machinery를 **재사용**(두 번째 matcher를 만들지 말 것).
>   특히 `_first_attribute_value` — 모델의 `GM=`은 입력 gene 행의 **첫 속성 값**(키가 무엇이든)
> - 인자: `--input` `--output` `--tool` `--json`. 참조 기본값 TAIR10 = `transgenic/revision/data/TAIR10/`,
>   AtRTD3 = `transgenic/revision/data/AtRTD3/atRTD3_TS_21Feb22_transfix.gtf`
> - 산출: `added_structures`(동일 emission 병합), TAIR10 대체전사체 정확 CDS 일치 / 인트론 체인 일치,
>   AtRTD3 일치, 각각의 비율, `recall_of_TAIR10_alternatives_pct`, `loci_with_at_least_one_addition`, provenance
> - **검증**: 기존 세 도구로 먼저 돌려 `added_structures`가 GeMoMa 21,739(90) / BRAKER3 128(0) /
>   EGAPx 22,711(31)과 맞는지 확인. 어긋나면 진행 전에 원인 규명
> - Helixer/ANNEVO는 locus당 전사체 1개이므로 공급 구조가 명확하고 나머지는 전부 순수 추가
> - 유전자 ID는 `standardize_gff`가 정렬 순서로 재명명하므로 **도구 간 비교 불가** — 도구 내에서는 `GM=`,
>   참조와는 좌표 겹침으로 페어링
> - TDD, 4-space, 타입 힌트, argparse. 결과는 `polishing_benchmark/results/`(저장소 밖), 데이터 커밋 금지

### 목표 표 — 이게 원고에 들어갈 것

| 프롬프트 | 추가 구조 | TAIR10 대체전사체 일치 | AtRTD3 일치 |
|---|---|---|---|
| TAIR10 자신 (원고 기존 수치) | 1,103 | 18.1% | 18.5% |
| AUGUSTUS 샘플링 (대조) | 43,433 | 1.3% | 10.2% |
| **Helixer** | 측정 중 | — | — |
| **ANNEVO** | 측정 중 | — | — |

### 다섯 도구 베이스라인 (모델 이전, TAIR10과의 거리)

| 도구 | 매칭 locus | 처음부터 맞음 | 틀림 | 비율 |
|---|---|---|---|---|
| BRAKER3 | 23,761 | 20,106 | 3,655 | 15.38% |
| EGAPx | 23,068 | 19,363 | 3,705 | 16.06% |
| GeMoMa | 25,551 | 20,427 | 5,124 | 20.05% |
| **Helixer** | 24,994 | 19,227 | 5,767 | **23.07%** |
| **ANNEVO** | 24,342 | 16,704 | 7,638 | **31.38%** |

원고가 지목한 두 도구가 TAIR10에서 **가장 멀리** 출발 — 추가할 여지가 가장 크고, 시험대로 적절.

### ✅ 원고 수정 완료 (커밋 전, 파일 상태로 존재)

`manuscript_v2.md` 5곳 수정 — "refine"/제3자 폴리싱 주장 전부 제거:

| 위치 | 변경 |
|---|---|
| 초록 | "refinement layer over existing annotation pipelines" → "extends an annotation it is given rather than replacing the pipeline that produced it" |
| 서론 | "annotation and gene model refinement를 위한 다목적 도구" → 삭제 |
| 결과 34행 | "refine existing annotation" → "extend an existing annotation" |
| 결과 46행 | "the refined annotation" → "the returned annotation" |
| **고찰 80행** | **"primary-transcript annotations such as those produced by Helixer or ANNEVO"** 문장 제거 + 적용범위 명시 문장 추가 |

`response_to_reviewers.md` 2곳 수정 — 위 삭제로 응답서가 존재하지 않는 문장을 설명하게 된 것을 정정:
- 3행: "refinement layer" 표현 + `refine` 표현을 쓰지 않는 이유 명시
- **119행**: 저자들이 리뷰어 2에게 "Helixer/ANNEVO 문장은 예외로 남긴다, 그 조합은 벤치마크하지 않았다"고 밝힌 대목 →
  "가정법으로 남기는 대신 **삭제했다**"로 변경. 리뷰어 요구("미검증 능력엔 가정법")에 대해 더 강한 답이 됨

**검증 상태**: `23_build_pc_manuscript.py`로 v3 재생성 완료, `19_verify_manuscript_numbers.py` **200/200 통과**, 초록 248단어(제한 250), v3에 제3자 주장 0건.

### ⚠️ 결과가 나오면 판단할 것

- **Helixer/ANNEVO에서 AS 추가가 잘 되면** → 고찰에서 지운 문장을 **되살릴 근거**가 생김.
  단, 이번엔 측정치를 동반한 주장으로 ("Helixer/ANNEVO 출력에 추가한 isoform의 X%가 실제 대체전사체와 일치")
- **잘 안 되면** → 현재 상태(주장 제거) 유지가 정답. 추가 수정 불필요

### 완료된 것 (Task 1~5, 커밋 `4706ec1`…`860dcbc`)

폴리싱 벤치마크 5개 태스크 완료, 리뷰 5라운드, 테스트 162개 통과.
전체 결론과 수치는 `polishing_benchmark/RESULTS.md` (521행, 5개 독립 스코어러가 재현 검증).
실행 기록·결함 목록은 `transgenic/.superpowers/sdd/2026-08-05-polishing-benchmark/progress.md`.
운영 제약(GPU 호스트, 배치 크기, 커밋 고정 등)은 같은 디렉토리 `gpu-environment.md`.
랩 위키 항목: `projects/transgenic-polishing-benchmark/` (커밋 `0789263`, 푸시 완료).

### 사후 분석에서 추가로 나온 것 (원고 반영 여부 미정)

**1. pre→post 방향성** — 공급된 primary가 TAIR10에 가까워지는가 멀어지는가, 세 축 모두 "멀어짐":

| 기준 | GeMoMa | BRAKER3 | EGAPx |
|---|---|---|---|
| CDS 정확일치 (수리:손상) | 84:6,218 (74배) | 1:13 | 25:7,274 (291배) |
| CDS 염기겹침 연속거리 | 2.95%:27.91% (9.5배) | 0.33%:0.16% | 1.91%:34.38% (18배) |
| 단백질 서열 정확일치 | 77.7%→54.2% | 82.8%→82.7% | 83.5%→52.1% |
| 내부 종결코돈 새로 생김 : 없어짐 | 6,873 : 0 | 93 : 0 | 7,479 : 8 |

단백질 축이 손상의 성격을 말해줌 — **리딩 프레임이 깨짐**. 반대 방향(고쳐짐)은 사실상 0.
번역기는 참조 대조 99.87% 일치로 검증함 (마이너스 가닥 phase 버그를 한 번 잡고 재검증).

**2. `.fai` 인덱스 결함 (프로젝트 데이터 디렉토리)**
`transgenic_comparison/genomes/Athaliana_167_TAIR10.fa.fai`가 **다른 FASTA의 인덱스**.
Chr1 오프셋을 74로 기록하나 이 파일 헤더는 `>Chr1`(5자)이라 실제는 6. 74는 `genomes/` 쪽
긴 헤더 버전 값. 두 `.fai`가 md5까지 같은 게 증거(헤더가 다른데 같은 인덱스가 맞을 수 없음).
→ 이 인덱스로 서열을 뽑으면 68염기 어긋남. gffread가 TAIR10조차 깨진 단백질을 냄.
**벤치마크 결과에는 영향 없음** (GPU 실행은 `loadGenome()`으로 FASTA 직접 읽음,
`standardize_gff`는 염색체 **길이만** 사용하며 길이 값은 정확).
재생성하면 정상: `samtools faidx` 후 AT1G01010.1이 `MEDQVGFGFRPNDEELVGHY...`로 올바르게 번역.

**3. gffread 0.12.7이 GeMoMa/EGAPx의 TransGenic 출력에서 힙 손상으로 죽음**
모든 핵 염색체에서, exon 행 제거해도, 올바른 인덱스로도 동일. 세 도구 **입력**과 TAIR10 참조,
BRAKER3 **출력**은 정상 처리. 크래시 지점의 유전자는 평범함(441bp 단일 exon)이라 원인은 더 앞.
→ 단백질 비교는 직접 번역으로 수행함.

### 원고 관련 미결 (저자 판단)

**발행된 예측 파일의 2배 중복은 빔 출력이 아님.**
`A_thaliana_transgenic400Mprompt.gff3`의 54,826 gene 레코드 / 27,413 locus에서, 전사체가 2개인
25,252 locus **전부**가 CDS·exon·양쪽 UTR까지 동일하고 locus당 전사체 수 분포가 전부 짝수.
고정된 `prompt_mode.py`는 `num_return_sequences=1`. → DuckDB append 아티팩트.
`manuscript_v2.md` Methods("both beam hypotheses were exported per locus")와
`27_rescore_prompted_topbeam.py` docstring의 **설명이 틀림**. 톱빔 필터와 그 수치는 영향 없음
(중복이든 경쟁 빔이든 첫 레코드만 남기므로 결과 파일 동일, 54,826→27,413).

### 배포된 소프트웨어에서 찾은 결함 (상류 보고 가치)

1. **`prompt_mode.py` 배치 크기 >1에서 출력 손상, CLI 기본값이 32** — 그 값에서 전 locus 손실하면서
   `Output written to:` 출력하고 exit 0. 측정: bs1 18/18, bs2 17/18, bs4 15/18, bs32 0/18
2. **`datasets.py:189` `cache_dir="./HFmodels"` 상대 경로 + 강제 오프라인** — `HFmodels/`가 있는
   디렉토리에서만 동작, 아니면 "huggingface.co 연결 불가"라는 오해 유발 오류로 죽음(네트워크는 정상)
3. **`genome2GSFDataset`이 모든 GFF3 파일의 마지막 유전자를 잃음** — 다음 gene 행을 만날 때
   이전 유전자를 넣는데 루프 종료 후 flush 없음. 추가로 `maxLen=49152` 초과 유전자를 조용히
   건너뛰고, 기존 DuckDB에 **append**하므로 재실행이 출력을 2배로 만듦


---

## 📋 이전 스냅샷 (2026-08-06 12:27)

### 🟢 Helixer/ANNEVO 조합 벤치마크 시작 — 응답서 주장 직접 검증

**설정**: 원고 고찰에서 지목했던 **정확히 그 조합**을 직접 시험하기 위해 13종 벤치마크의 표준화 결과에서 입력 추출.

**입력 확인** (표준화된 결과에서):
| | 유전자 | mRNA | UTR | 조건 |
|---|---|---|---|---|
| Helixer | 27,039 | 27,039 | 있음 | ✓ locus당 정확히 하나 |
| ANNEVO | 25,181 | 25,181 | 없음 | ✓ locus당 정확히 하나 |

- seqid 이미 TAIR10과 동일 (정규화 불필요)
- 프롬프트되지 않은 isoform 없음 → 96% 폐기 현상 개입 불가
- 모델에 **가장 유리한 조건**

**작업 완료**:
1. 입력 파일 `polishing_benchmark/inputs/`에 스테이징 ✓
2. 스크립트 TOOLS 상수에 두 도구 추가 ✓
3. 테스트에서 도구 목록을 하드코딩한 부분 수정 (상수 참조로) ✓
4. GPU 호스트에 입력 전송 ✓
5. 베이스라인 실행 시작 ⏳

**예상 시간**: 베이스라인(~1시간) + Helixer/ANNEVO(~5시간) = 약 6시간

### 다음 작업

1. 베이스라인 완료 대기
2. Helixer/ANNEVO completion mode 실행 및 결과 수집
3. GeMoMa/EGAPx와 비교 분석 — 손상/수리율, isoform 폐기율 확인

## ▶️ 이전 스냅샷 (2026-08-06 12:21)

### 🟢 리뷰 응답 정합성 복구 — 응답서 2곳 수정, 검증 통과

**문제**: 제3자 주장 제거 후 응답서를 확인하지 않음 → 응답서 119행에서 "그 조합을 벤치마크하지 않았다"는 저자 선언이 존재하지 않는 문장을 설명 중.

**정정** (response_to_reviewers.md):
| 섹션 | 변경 |
|---|---|
| 리뷰어 2 응답 (119행 근처) | 예외 선언 제거 → "그 문장을 가정법으로 남기는 대신 **삭제했다**. 모든 completion 모드 결과는 참조 어노테이션의 primary를 프롬프트로 얻은 것이며 고찰이 이제 그렇게 명시한다" |
| 요약부 (초반) | "refinement layer" 설명 확장: "primary를 개선하는 것이 아니라 primary에 전사체를 **추가**하는 것" |

**리뷰어 항목 전수 확인**: 리뷰어 1·2·3 총 22개 항목 → 모두 처리 완료 (새 항목 없음).

**검증**: v3 재생성, 200/200 통과, 제3자 주장 0건, 응답서 일관성 복구.

### 다음 작업

1. 벤치마크 결과를 원고 Methods에 **추가**할지 여부 저자 판단 (현재는 고찰에서만 언급)
2. 응답서·원고 최종 패스 후 git 커밋

## ▶️ 이전 스냅샷 (2026-08-06 12:15)

### 🟢 제3자 어노테이션 폴리싱 주장 제거 완료 — 5곳 수정 및 검증 통과

**변경 사항** (manuscript_v2.md):
| 섹션 | 변경 |
|---|---|
| 초록 | "refinement layer" → 삭제, "받은 어노테이션을 **확장**하며, 그것을 만든 파이프라인을 대체하지 않는다"로 대체 |
| 서론 | "annotation 및 유전자 모델 **refinement**" → 삭제 |
| 결과 (2곳) | "refined annotation" → "returned annotation"; "refine하기 위한" → "확장하기 위한" |
| 고찰 | "Helixer·ANNEVO가 만든 primary-transcript에 추가하도록 설계" → 삭제하고 범위 제한 문장 추가: "보고된 결과는 참조 어노테이션 자신의 primary를 프롬프트로 얻은 것이며, 다른 파이프라인 primary에서의 동작은 별개 문제" |

**검증**: v3 재생성 완료 (스크립트 `23_build_pc_manuscript.py` 성공), 검증기 200/200 통과, 초록 248/250 단어.

### 다음 작업

1. 벤치마크 결과를 원고에 **추가**할지 여부 확인 (현재는 주장만 제거, 데이터 미포함)
2. 이 변경 사항을 git에 커밋할 준비

## ▶️ 이전 스냅샷 (2026-08-06 12:05)

### 🟢 **데이터 인덱스 오류 진단 완료** — `.fai` 불일치로 인한 gffread 크래시 해결

**문제**: gffread가 GeMoMa·EGAPx 출력 처리 중 메모리 손상으로 크래시 (BRAKER3는 정상) → 참조 단백질도 내부 종결코돈 99.7%로 깨져 나옴.

**근본 원인**: 게놈 `.fai` 인덱스가 Chr1 시작을 바이트 **74**로 기록했으나, 실제 FASTA 헤더 `>Chr1`(5자)이므로 시작은 바이트 **6**. 
68염기 어긋난 좌표로 gffread가 오염된 서열을 읽음. 인덱스 md5가 다른 파일(`genomes/` 버전)과 동일 → 헤더 길이 다른 두 FASTA에 같은 인덱스를 사용한 오류.

**정정**: `samtools faidx`로 올바른 인덱스 재생성.

**검증**: 
- 참조: `MEDQVGFGFRPNDEELVGHY...` (정상 서열)
- 내부 종결코돈: 0.12% (예상 수준, 99.7%는 인덱스 오류)
- BRAKER3·GeMoMa·EGAPx 입출력 모두 추출 성공

### 다음 작업

1. 올바른 인덱스로 단백질 pre/post 비교 실행 (completion mode 손상률 재측정)
2. RESULTS.md에 `.fai` 인덱스 오류 및 해결 기술
3. 랩 위키에 "TransGenic v2.0 데이터셋 인덱스 오류" 기록

## ▶️ 이전 스냅샷 (2026-08-06)

### 🟢 Task 5 **완료** — 리뷰 5라운드 종료, `polishing_benchmark/RESULTS.md` 확정

**결론**: 원고의 completion mode 주장은 외부 주석으로 이전되지 않는다. 수리보다 훼손이
압도적이며(GeMoMa 30.46%, EGAPx 37.61% 훼손 vs 수리 1.65%/0.68%), 5개 채점 기준 전부에서
결론이 유지된다. BRAKER3만 예외이나 이는 **모델이 생성을 포기한 것**이지 안전한 폴리싱이
아니다(순수 추가 14개 @ 9 loci).

**커밋**: `3026913` → `0a91250` → `1c3f07d` → `eb5e3e8` → `88d8c27` → `1d90c4c` → (마감 패스)
모두 docstring/문서 전용, 동작 변경 없음. 152/152 테스트 통과.
`polishing_benchmark/`는 저장소 밖이며 git에 들어간 적 없음(확인: 0건).

**핵심 부수 결과 (원고에 없음)**: 프롬프트되지 않은 isoform을 **모든 입력이 최소 93.75%
버린다** (정확 CDS 기준 96~100%, splice-junction 기준 93.75~99.8%, 대조군이 93.75%로 최저).
분포 내 대조군도 마찬가지이므로 외부 입력 특유의 문제가 아니라 completion mode 일반의
성질이다. 폐기율은 세 행이 비슷해 보이나 기전은 셋 다 다르다 — 출력의 non-prompted 구조
수는 BRAKER3 15개(소멸), GeMoMa 15,130·EGAPx 15,540(2~5배 대체), 대조군 1,089(80% 축소).

**BRAKER3 = 무동작**: 순수 추가 14개 @ 9 loci(0.04%). 대조군 의존 없이 런 내부에서도
동일 — GeMoMa 10.65% · EGAPx 10.44% vs BRAKER3 0.04%, **276~281배**.

**5라운드에 걸쳐 같은 결함이 반복됨 (기록 가치 있음)**: 두 기전을 뭉뚱그린 단일 집계를
한 방향으로만 보고하는 패턴. **3건은 BRAKER3에 유리한 방향, 2건**(1-in-200 precision,
EGAPx 4.6배)**은 반대 방향** — 이 반대 부호가 "관심 방향의 편향"이 아니라 **집계 자체가
원인**이라는 증거. 매번 리뷰와 자체 검증을 통과한 이유는 집계값 자체는 항상 옳았기 때문.
현재 문서의 모든 도구 간 비교는 분해 후에만 제시된다.

**세부 정정 이력**: 아래 "라운드 1~5에서 정정된 주장" 참조 (항목 1~19).

### 다음 작업

1. **Task 6 (저자 판단 필요)**: 발표 파일의 2배 중복이 beam export가 아니라 DuckDB append
   아티팩트 → `manuscript_v2.md` Methods 및 `27_rescore_prompted_topbeam.py` docstring의
   *사유* 정정 (수치는 영향 없음)
2. 원고 병합 — completion mode를 "신뢰 가능한 primary를 요구하는 isoform 생성 단계"로
   문서화, 초록의 "polishing" 표현 재검토
3. 랩 위키 항목 작성 (5차 수정 라운드 + 릴리스 소프트웨어 결함 3건 + DuckDB append 발견)

## ▶️ 이전 스냅샷 (2026-08-06 05:50)

### 🟢 Task 5 + 리뷰 완료 — 결론 재검증 완료 (커밋 3026913)

**작업 흐름**: 구현자가 스코어러 3곳을 수정 → 리뷰어가 스코어러를 **완전 재구현**해서 24개 핵심 수치 전부 재현.

**리뷰가 공격한 6개 축** (모두 측정으로 반박됨):
| 해석 방식 | GeMoMa 손상 | EGAPx 손상 | 수리율 |
|---|---|---|---|
| 발표된 기준 | 30.46% | 37.61% | ≤1.65% |
| 정확한 신원 매칭 | 30.46% | — | 1.64% |
| 원시 출력 기준 | 30.47% | 37.61% | 1.64% |
| 어떤 전사체든 인정 | 23.86% | 29.54% | — |
| 스플라이스 접합 기준 | 22.99% | 30.29% | 2.07% |
| 접합 + 전사체 혼합 | **21.17%** | 26.57% | 2.42% |

~~**핵심 발견**: TAIR10 대조군(완벽한 primary)도 16.35% 손실 → 외부 입력 특유 문제가 아니라 **completion 모드 전반의 특성**.~~
> **철회 (라운드 2)**: 16.35% vs 11.2% 비교는 두 입력의 non-prompted isoform 보유량
> (16.9% vs 10.8%)에 교란되어 있어 모델 행동을 재지 않는다. 동일 조건 축에서는
> BRAKER3 0.48% vs 대조군 0.05%로 **방향이 반대**(약 9배 나쁨). "completion 모드 전반의
> 특성"이라는 결론 자체는 **다른 표로 더 강하게 성립** — 프롬프트되지 않은 isoform을
> 모든 입력이 96~100% 버린다(대조군 200/5,580 = 3.6% 반환).

**정정**: GeMoMa의 21.6%는 순서 강등(부분 보존), 78.4%는 완전 소실 — 다만 소실만 세어도 결론은 불변.

**발견된 오류** (RESULTS.md에 기술):
- TAIR10 "파싱 실패 3건" = 실은 마지막 유전자 버그
- 거짓 검증 주장 2개
- 원고 주장을 과다 양보하는 문장들

### 다음 작업

1. RESULTS.md 사실관계 정정 (숫자 조정 금지, 대체 내용 명시)
2. 원고에 "완벽한 primary도 손실" 발견을 Methods/토론에 추가할지 판단
3. 랩 위키 `projects/TransGenic_*.md` 업데이트

## ▶️ 이전 스냅샷 (2026-08-06)

### 🔴 Task 5 완료 — 폴리싱 벤치마크 결론: **모델이 외부 입력을 훼손한다**

`polishing_benchmark/RESULTS.md` 작성 완료. 원고에 인용 가능한 형태.

**질문**: 원고의 모든 completion-mode 결과는 TAIR10 자체 primary로 프롬프트한 뒤
TAIR10에 대해 채점했다. 프롬프트가 곧 정답이므로 "polishing"은 측정된 적이 없다.
외부 주석 3종(TAIR10과 실제로 불일치)으로 처음 측정.

**답**: 계획서의 두 번째 결과 — 수리보다 훼손이 압도적.

| 도구 | 원래 틀린 loci 중 수리 | 원래 맞던 loci 중 훼손 |
|---|---|---|
| GeMoMa | 84 / 5,105 (1.65%) | 6,218 / 20,412 (**30.46%**) |
| BRAKER3 | 1 / 3,653 (0.03%) | 13 / 20,103 (**0.06%**) |
| EGAPx | 25 / 3,684 (0.68%) | 7,274 / 19,340 (**37.61%**) |

추가 구조: GeMoMa 21,739개 중 TAIR10 일치 90개(0.41%), EGAPx 22,711개 중 31개(0.14%),
BRAKER3 128개 중 0개.

**메커니즘** — 공급된 모델이 출력에 살아남는 비율:

| 도구 | 출력 첫 transcript | 출력 어딘가 | 순서만 밀림 | **완전 소실** |
|---|---|---|---|---|
| GeMoMa | 19,616 (66.4%) | 21,641 (73.3%) | 2,025 | **7,884 (26.7%)** |
| BRAKER3 | 26,499 (99.5%) | 26,502 (99.5%) | 3 | 126 (0.5%) |
| EGAPx | 14,253 (60.7%) | 16,112 (68.7%) | 1,859 | **7,355 (31.3%)** |
| *TAIR10 대조군* | *27,399 (99.95%)* | *27,399 (99.95%)* | *0* | *14 (0.05%)* |

**BRAKER3는 양방향 모두 예외** — 유일하게 모델이 **프롬프트된 transcript를** 거의
손대지 않는 입력이며, 따라서 유일하게 수리도 하지 않는다. 단 locus 수준에서는 아님
(아래 소실 채널 참조). **원인은 규명하지 못했다**:
UTR 유무도(BRAKER3·GeMoMa 둘 다 UTR 없으나 정반대), TAIR10과의 거리도
(BRAKER3 15.38% vs EGAPx 16.06%로 거의 같으나 정반대) 설명하지 못함. 관측만 기술.

**결론이 채점 방식의 산물이 아님을 확인**: 훼손의 약 21%는 단순 순서 밀림이지만,
5개 기준(CDS/representative, CDS/primary, CDS/any, splice-junction/representative,
splice-junction/any) 전부에서 훼손율이 GeMoMa 21% · EGAPx 26% 아래로 내려가지 않고,
수리율은 **최대 2.42%**를 넘지 않으며 BRAKER3는 0.05~0.12% 유지.

**또 하나의 손실 채널 (라운드 1 추가 → 라운드 2에서 분해)**: 입력에 있었으나 출력에
없는 구조. 총량은 GeMoMa 13,855 / BRAKER3 2,980 / EGAPx 10,375 / 대조군 5,394이나,
**총량끼리 도구 간 비교는 하면 안 된다** — 각 주석이 우연히 가진 isoform 수
(non-prompted 비중: 22.1% / 10.8% / 12.2% / 16.9%)를 주로 반영할 뿐이기 때문.
`prompt_mode.py`는 locus당 **첫 transcript만** 공급하므로 두 축으로 분해해야 한다:

| 행 | 프롬프트 안 된 isoform | 그중 반환됨 | 프롬프트된 구조 소실률 |
|---|---|---|---|
| GeMoMa | 7,246 | 33 (0.5%) | 26.03% |
| BRAKER3 | 2,867 | 1 (0.03%) | **0.48%** |
| EGAPx | 3,204 | 36 (1.1%) | 31.30% |
| *TAIR10 대조군* | *5,580* | *200 (3.6%)* | ***0.05%*** |

→ **결과 1**: 분포 내·외를 막론하고 **프롬프트되지 않은 isoform의 최소 93.75%를 버린다**
(정확 CDS 일치 기준 96~100%, splice-junction 기준 93.8~99.8%로 대조군이 93.75%로 최저 —
기준을 반드시 명시할 것). 원고에 없는 새 결과.
→ **결과 1의 핵심**: 폐기율은 세 행이 같아 보이지만 **기전이 셋 다 다르다**. 출력의
non-prompted 구조 수는 BRAKER3 **15개**(소멸), GeMoMa 15,130개·EGAPx 15,540개(입력 대비
2~5배 **대체**), 대조군 1,089개(80% **축소**). 유지율 표만으로는 이 차이가 안 보인다.
→ **결과 2**: 동일 조건 축(모델이 **실제로 본** 구조의 파괴)에서는 BRAKER3 0.48% vs
대조군 0.05%로 **약 9배 나쁘다**.
→ **결과 3 (라운드 3 추가)**: gain 축도 같은 분해가 필요하고 같은 방향으로 뒤집힌다.
추가 구조의 84~89%는 **프롬프트된 구조가 파괴된 locus에서 나온 대체물**이다
(GeMoMa 18,336/21,739, BRAKER3 **114/128**, EGAPx 19,799/22,711, 대조군 22/903).
순수 추가만 세면 대조군 **881개(864 loci)** vs BRAKER3 **14개(9 loci)** — 구조 63배,
locus 96배 차이. 즉 BRAKER3는 생성 측면에서 **사실상 무동작**이다.

**커버리지 한계 (RESULTS.md에 명시)**: oversized 15개 loci(GeMoMa 4, EGAPx 11)는
`preprocess.py`의 maxLen=49,152 초과로 제외 — 두 도구가 내놓는 가장 크고 복잡한
융합 loci다. TAIR10에는 하나도 없어 기존 발표 런에서는 드러나지 않았음.

**검증**: 모든 수치를 독립 재구현으로 재계산. BRAKER3·EGAPx는 8개 필드 전부 정확히 일치.
baseline↔polishing 잔차가 `loci_without_output`과 정확히 일치(34/5/44), 21개 청크 전부
`distinct_loci_out == genes_out` 확인.

**부수 발견 (스코어러 결함, 영향 미미)**: `32_score_polishing.py`는 GeMoMa를 GM= 매칭하지
못한다(출력 `GM=`를 입력 `ID=`와 대조하나 `preprocess.py`는 첫 속성값을 쓰고 GeMoMa는
`Name=`가 먼저). `gm_paired: 0`으로 전량 위치 기반 폴백. 올바른 매칭으로 재채점 시
훼손 6,218 → 6,219, **수리율 1.65% → 1.64%** (한 셀이 아니라 표 전체가 움직임).
Task 4 러너는 고쳤으나 스코어러는 안 고쳐진 상태.

### 라운드 1 리뷰에서 정정된 주장 (이전 스냅샷·초안의 오류)

1. **"TAIR10 대조군 파싱 실패 3건(0.011%), 외부 입력이 6~12배 더 실패"** → **철회.**
   누락된 3개(`AT1G80990`·`AT3G63540`·`AT5G67640`)는 전부 각 염색체의 **마지막 유전자**로,
   바로 위에서 보정하는 `preprocess.py` off-by-one이다. 발표 런의 생성 파싱 실패는
   **식별 가능한 것이 0건**. 비율은 근거 없었음. (단 단일 파일 런이면 1건, 7분할이면 7건이
   예상되는데 3건인 이유는 미해명 — RESULTS.md에 명시.)
2. **"TAIR10에서 공급된 구조의 99.95%가 그대로 돌아온다"** → 99.95%는 **프롬프트된
   representative**의 생존율. 전체 CDS 구조 기준으로는 **32,993개 중 27,599개(83.65%)**.
   원고에 인용될 가능성이 가장 높은 문장이라 정정 필수.
3. **"수리율은 모든 기준에서 2% 미만"** → splice-junction 기준에서 **2.42%**로 초과.
   결론은 불변이나 문장은 틀렸음.
4. **"EGAPx 1,616개는 childless gene shell"** → 틀림. 자식이 아예 없는 유전자는 **0개**,
   1,616개는 비coding 자식만 가진 유전자. 2,503개 비coding feature와 더하면 안 됨
   (coding 유전자 43개도 비coding 자식을 가짐).
5. **"`loci_without_output`은 세 도구 모두 manifest의 부분집합"** → GeMoMa는 34개 중 29개만.
   나머지 5개는 출력이 존재하는 pairing 실패. (보증 자체는 불변 — 어느 쪽이든 분모에서 제외.)

### 라운드 2에서 정정된 주장

6. **"BRAKER3는 입력을 거의 손대지 않고 반환한다"** → **프롬프트된 transcript**에 한정.
   locus 수준에서는 구조의 11.2%(TAIR10 일치 1,531개 포함)를 잃고, 프롬프트된 구조 자체도
   23,756개 중 114개에서 소실.
7. **"TAIR10 대조군이 16.35%로 BRAKER3(11.2%)보다 더 많이 잃으므로 외부 입력 특유가 아니다"**
   → **철회.** 이 비교는 각 주석의 isoform 보유량에 교란되어 있다. 동일 조건 축(모델이 실제로
   본 구조의 파괴)에서는 BRAKER3 0.48% vs 대조군 0.05%로 **방향이 반대** — 약 9배 나쁘다.
   일반 결론은 다른 표(위의 non-prompted isoform 반환율)로 더 강하게 성립.
8. **"소실 채널이 gain보다 크다"** → ~~BRAKER3만 참~~. **라운드 4에서 재정정**: raw gain
   기준으로는 BRAKER3만 참이지만, raw gain의 84~89%가 대체물이므로 **순수 gain 기준으로는
   세 도구 모두 소실이 더 크다** — GeMoMa 13,855 vs 3,403(4.1배), EGAPx 10,375 vs
   2,912(3.6배), BRAKER3 2,980 vs 14(213배).

### 라운드 3에서 정정된 주장

9. **"대조군도 gain은 있다 — 903개(878 loci) vs BRAKER3 128개(123 loci), 어느 쪽도 무동작이
   아니다"** → **철회.** 이것은 철회된 손실 비교의 **거울상**이며 같은 방향으로 BRAKER3를
   유리하게 보이게 했다. 분해하면 BRAKER3의 128개 중 **114개(89%)가 위 문단에서 손실로 센
   바로 그 114건**을 gain으로 재계상한 것. 순수 추가는 대조군 881개(864 loci) vs BRAKER3
   14개(9 loci) — **63배/96배**이며, 23,756 loci 중 9곳에서 14개 추가는 **무동작이 맞다**.
10. **"프롬프트되지 않은 isoform의 96~100%를 버린다"** → 기준 명시 필요. 정확 CDS 일치
   기준에서만 참(99.54/99.97/98.88/96.42). splice-junction 기준에서는 93.75~99.8%이고
   **대조군이 93.75%로 범위 밖**. 정직한 하한은 96%가 아니라 **93.75%**.

### 라운드 4에서 정정된 주장 (해석 섹션 §6이 인용되는 부분)

11. **§6의 "BRAKER3는 128개 구조를 123 loci에 추가"** → **순수 추가는 14개, 9 loci.**
    §2에서 이미 철회한 표현이 §6에 남아 있었음. 구조 9배·locus 13.7배 과대.
12. **"TAIR10 일치가 200개 중 1개 미만"** → **거짓** (이 건은 모델에 *불리한* 방향 오류).
    순수 추가 기준 precision은 GeMoMa 22/3,403 = 0.65%, EGAPx 19/2,912 = 0.65% —
    즉 **155개 중 1개**. raw 총계는 GeMoMa 242개 중 1개, EGAPx 733개 중 1개로 서로 다르며
    (부풀려진 분모), 과소평가 배율도 각각 1.6배·4.78배로 다르다.
13. **"추가의 84~89%는 대체물이다"** → *위치*는 정확하나 *동일성*은 BRAKER3에서만 성립
    (파괴된 locus당 정확히 1.00개). GeMoMa 2.76 / EGAPx 2.75개이므로 그중 최대 1개만 대체물.
14. **§2·§3(as-run 위치 기반 페어링)과 §4(교정된 Name 기반 GM 페어링)가 다른 페어링을
    쓴다는 사실이 문서에 없었음** → 양쪽에 명시. §4의 "GM= 1:1 완전 해소"와 §7의
    "`gm_paired: 0`"은 모순이 아니라 각각 러너/스코어러를 기술한 것.
15. **대조군의 순수 추가 881개에는 정확도 지표가 없고 구조적으로 불가능**하다는 점을 명시
    (대조군의 입력이 곧 레퍼런스이므로 `added ∩ reference = ∅`, 실측 0/881). 차선책으로
    176개만 TAIR10 intron chain 공유, **705개는 구조적으로 미검증**. 이 비대칭은 이 문서에서
    유일하게 BRAKER3에 **불리한** 방향이므로 명시해야 head-to-head가 공정해진다.

### 라운드 5에서 정정된 주장 (최종)

16. **"순수 추가의 155개 중 1개가 TAIR10과 일치"에 기준 라벨이 없었음** — 정확 CDS 일치
    기준 전용 수치이고, intron chain 기준에서는 **31.4%(GeMoMa)·33.0%(EGAPx)로 48배 뛴다**.
    그대로 두면 "모델이 그럴듯한 isoform을 만든다"로 오독될 수 있음. **방어는 측정 가능**:
    chain 일치 1,069개 중 **1,050개**, 961개 중 **942개**가 **프롬프트된 전사체 자신의
    intron chain**을 재사용한 말단 변이일 뿐. 프롬프트와 다른 chain만 세면 19개(0.56%)·
    19개(0.65%)로 정확 CDS 수치와 사실상 동일. 대조군은 반대로 176개 중 30개만 프롬프트
    chain 재사용 → **146개(881개 중 16.6%)가 진짜 새 splice 구조**, BRAKER3는 **2개**.
    즉 생성 격차는 느슨한 기준에서 오히려 더 벌어진다.
17. **대조군 손실 열의 "5,394개 전부 TAIR10 일치"는 구조적 100%**이며 결과가 아님
    (소실 구조 ⊆ 입력 = 레퍼런스이므로 `dropped ∩ reference = dropped` 항등). 위 15번의
    `added ∩ reference = ∅`와 정확히 거울상. "대조군이 TAIR10 정답 구조 5,394개를 잃고
    BRAKER3는 1,531개"라는 비교는 무의미 — 앞은 정의, 뒤는 측정.
18. **"정직한 하한 ~94%"** → **93.75%** (헤드라인은 이미 93.7%로 정확했음).
19. **"EGAPx 과소평가 4.6배"** → **4.78배** (0.65%를 반올림된 0.14%로 나눈 오류. 실제
    분모는 0.1365%). 이 오류도 모델에 *불리한* 방향 — 15번과 같은 부호.

### 다음 작업

1. **Task 6 (저자 판단 필요)**: 발표된 파일의 2배 중복이 beam export가 아니라 DuckDB
   append 아티팩트라는 발견 → `manuscript_v2.md` Methods와
   `27_rescore_prompted_topbeam.py` docstring의 *사유* 정정 (수치는 영향 없음)
2. 원고에 이 결과를 넣을지 결정 — completion mode를 "신뢰 가능한 primary를 요구하는
   isoform 생성 단계"로 문서화하고 초록의 "polishing" 표현 재검토
3. 랩 위키 항목 작성 (5차 수정 라운드 + 릴리스 소프트웨어 결함 3건)

## ▶️ 이전 스냅샷 (2026-08-06 04:51)

### 🟢 세 도구 완료 (04:49) → 손실 전부 설명됨, 스코어링 진행 중

**완료** (재시작 후 8시간 6분):

| 도구 | 입력 → 최종 | 구조적 | 파싱 실패 | 좌표 탈락 |
|---|---|---|---|---|
| GeMoMa | 29,561 → 29,525 | 11 | 21 | 4 |
| BRAKER3 | 26,635 → 26,628 | 7 | 0 | 0 |
| EGAPx | 23,515 → 23,467 | 18 | 30 | 0 |

**핵심**: 설명되지 않는 손실 = 0. 모든 손실이 모델 자체 보고와 정확히 일치.

**파싱 실패율 (입력 형식별)**:
- BRAKER3: 0% (exon+CDS)
- GeMoMa: 0.071% (CDS만)
- EGAPx: 0.128% (exon+CDS+UTR)

→ 입력이 학습 분포에서 멀수록 모델 파싱 실패 증가 (논문 결과).

**현재**: 스코어링 3개 도구 진행 중.

### 다음 작업

1. 스코어링 완료 대기
2. RESULTS.md 작성
3. Task 5 전환: 폴리싱 효과 분석

## ▶️ 이전 스냅샷 (2026-08-06 01:25)

### 🟡 BRAKER3 완료 → EGAPx 진행 중, 손실률 차이 명확

**BRAKER3 완료** (1h 26m, 계획대로):
- 최종: 26,628 유전자 (입력 26,635)
- 손실 7 = 구조적 7 (염색체당 마지막 유전자 1개)
- 파싱 실패 0, 표준화 탈락 0, 설명되지 않는 손실 0

**GeMoMa 대비**:
- GeMoMa: 파싱 실패 21 + 좌표 탈락 4 (입력 구조 차이 — exon 행 없음)
- BRAKER3: 파싱 실패 0 (exon+CDS 모두 포함 → 모델 구문 적중)
- 해석: 입력 형식이 훈련 분포에서 멀수록 파싱 불가능 출력 증가 (결과 논의용)

**현재 진행**:

| 도구 | 상태 | 소요 | 입력→최종 | 손실 내역 |
|---|---|---|---|---|
| GeMoMa | ✅ | 3h15m | 29,561→29,525 | 구조적 11, 파싱 실패 21, 좌표 탈락 4 |
| BRAKER3 | ✅ | 1h26m | 26,635→26,628 | 구조적 7 |
| EGAPx | 🔄 | — | — | — |

**모든 손실 설명됨** (unexplained = 0, 두 도구 모두)

### 다음 작업

1. EGAPx 완료 대기 (새벽 3~4시경 예상)
2. 세 도구 손실 통합 비교표 작성
3. Task 5 전환: 스코어링 및 결과 원고 작성

## ▶️ 이전 스냅샷 (2026-08-06 00:00)

### 🟡 GeMoMa 완료 → 손실 경로 전부 귀속, 나머지 도구 진행 중

**GeMoMa 완료** (3h 15m, 예상보다 빠름):
- 최종: 29,525 유전자 (입력 29,561)
- 손실 경로 전부 설명됨:
  - 11개: 구조적 도달 불가 (마지막 유전자 7 + 초과 길이 4)
  - 21개: 생성 단계 파싱 실패 (모델 자체 보고와 정확히 일치)
  - 4개: 표준화 좌표 검증 탈락
- 설명되지 않는 손실 0개 (I1 수정 덕분에 단계별 기록 가능)

**모델 품질 신호**:
- 파싱 실패율 21/29,550 = 0.071% ~~(논문 TAIR10 0.011%보다 6배 높음)~~
  > **정정 (Task 5 라운드 1)**: 괄호 안 비교는 **철회**. 발표 런의 누락 3건은 파싱 실패가
  > 아니라 각 염색체 마지막 유전자(`preprocess.py` off-by-one)다. 발표 런의 생성 파싱
  > 실패는 식별 가능한 것이 0건이므로 "6배" 배수는 성립하지 않는다. 0.071% 자체는 유효.
- 4개 locus 좌표는 모델이 생성했으나 기하학적 유효성 탈락 (파이프라인 아님)

**톱빔 필터**: 무동작 확인 (num_return_sequences=1)

**현재**: EGAPX, SNAP2 도구 계속 진행 중 (새벽 6~7시경 완료 예상)

### 다음 작업

1. 나머지 두 도구 완료 대기
2. 손실 데이터 통합 (세 도구 비교표)
3. Task 4 전체 통계 확정 후 결과 원고 작성

## ▶️ 이전 스냅샷 (2026-08-05 18:52)

### 🟡 Gate R4-3 교체 완료 → t4-fixer 진행 중, 동시 편집 회피

**교체 내용**:
- `check_loss` 게이트: 설명되지 않는 손실(`structurally_unreachable_locus`) 0% 임계값 → 5% 변경
- 설계: 발행 실행 실측(27,416 중 2개 손실, 0.007%)에서 유추
- 상수 정의 및 근거 주석 포함 (tree a72c964에 반영됨)

**t4-fixer 충돌 해결**:
- 동시 편집 감지 → 즉시 중단, 파일 오너십 t4-fixer에 양보
- t4-fixer 메시지: "tree clean at a72c964, 대기 중"

**현재 상태**:
- Tree: a72c964 (clean)
- Gate R4-3 수정 반영됨
- t4-fixer: 추가 작업 가능 상태 대기

### 다음 작업

1. t4-fixer 완료 대기 (남은 3건 Critical 이슈)
2. 통과 후 Task 4 full-scale 실행 개시

## ▶️ 이전 스냅샷 (2026-08-05 18:44)

### 🟡 Fix round 4 (t4-fixer) 진행 중 — Task 4 실행 준비 완료

**Fix round 4 수정 상태**:
- 3가지 Critical 이슈 수정 중:
  1. EGAPx 필터링 전/후 provenance 기록 간 불일치
  2. 재개 시 크래시 리스크 (인터럽트 핸들링)
  3. 21~99 유전자 게이트 오탐지 구간 (매니페스트 미검사)
- 예상: 새 커밋 들어오면 테스트 통과 → 즉시 Task 4 실행

**준비 사항 완료**:
- GPU 호스트 정리: 스모크 캐시 제거, 입력 3개 배치만 남음
- GPU 유휴: 사용률 0%, 361G 여유 (도구당 ~6시간 예약 가능)
- Task 5 브리프: 사전 준비 완료

### 다음 작업

1. Fix round 4 커밋 대기 (520c64f 이후)
2. 테스트 통과 확인 (143/143 예상)
3. Task 4 full-scale 실행 개시 (도구당 ~6시간, 합계 ~19시간)
4. 완료 후 Task 5 집계로 전환

## ▶️ 이전 스냅샷 (2026-08-05 18:41)

### 🟢 Fix round 3 완료 (commit 520c64f) → 남은 3건으로 새 구현자 투입

**Fix round 3 확인**:
- 테스트: 143/143 통과
- `structurally_unreachable_locus` 구현 확인 (build_local_subset, chunk provenance 기록)
- `write_missing_loci_manifest` 조건부 기록 동작 확인
- `check_loss` 게이트 동작 확인
- Per-chunk 도달 불가능 유전자: 7/도구 (21 총, 7/27,416 = 0.0255%)
- EGAPx ChrM gene-egapxtmp_025071 GPU 재실행과 일치

**남은 3건** (새 구현자 상위 모델로 투입):
1. EGAPx 재개: 필터링 전 유전자 수로 완료 판정 vs 필터링 후 값 provenance 기록 → 6시간 도구에서 매번 처음부터 재실행
2. 재개 시 크래시 리스크 (재개 인터럽트 핸들링)
3. 21~99 유전자 염색체에서 게이트 오탐지 가능 구간 (매니페스트 기록되나 미검사)

### 다음 작업

1. 세 건 수정 완료 대기
2. 통과 후 바로 Task 4 full-scale 실행 (GPU 예약)

## ▶️ 이전 스냅샷 (2026-08-05 18:00)

### 🟡 Task 4 Fix 라운드 1 완료 → 재리뷰에서 새로운 Critical 2건 발견

**Fix 라운드 1 완료** (commit d9a8b7a):
- 테스트: 131/131 통과 (새로운 케이스 78개 추가)
- C1-C3 (atomic 파일 핸드오프, 염색체 격리, 손실 임계값) 모두 해결
- I1-I10 모든 Important/Minor 이슈 해결 완료

**재리뷰 발견**:
- 새로운 Critical 2건 발견 (GPU 시간 투입 전 마지막 관문에서 포착)
- 두 이슈 모두 fix round 1 자체에서 도입됨 (예: unfalsifiable 테스트 교체 시)
- Fix round 2 dispatched, 상세 내용은 task-4-findings-round2.md

**ChrM 손실 설명**:
- 논문 Methods: TAIR10 오르가넬 유전자 210개 중 209개만 예측 통과 (발행 실행도 1개 손실)
- ChrM 손실 1/13은 정상 범위 내

### 다음 작업

1. Fix round 2 완료 대기 (새로운 Critical 2건)
2. 통합 smoke 테스트 (egapx 추가 검증)
3. Task 4 full-scale 실행 (모든 수정 후 GPU 예약)

## ▶️ 이전 스냅샷 (2026-08-05 16:37)

### 🟡 배치 크기 효과 확정 — 배치 1–2 측정 중

**배치 크기 통제 재현 완료**:
- 배치 4: 정확히 15/19 재현 (되돌린 코드)
- 배치 32: 0/19 (전부 손실)
- **배치 크기가 유일한 실제 변수 확정**

**DB 구조 재확인**:
- 유전자당 한 행 (TAIR10 18행, GeMoMa 19행)
- 윈도우 구성 가설 검증 불가능

**패딩 문제 추측**:
- 배치 내 서로 다른 길이 시퀀스 패딩이 생성 품질 악화
- 단조 관계: 배치 클수록 손실률 높음

**진행 중**: 배치 1, 2 측정 → 결과에 따라 Task 4 최적 배치 크기 결정

### 다음 작업

1. 배치 1, 2 측정 완료 대기
2. Task 4 실행 파라미터 확정 (배치 1이 최적인 경우 도구당 ~6시간 예상)
3. Task 4 full-scale 실행 (밤새 GPU 예약)

## ▶️ 이전 스냅샷 (2026-08-05 16:36)

### 🟡 배치 크기 효과 vs. 코드 버그 구분 중 — Task 3 baseline 재정정

**배치 크기 검증 결과 불일치**:
- 배치 32: 출력 0 (전체 손실)
- 배치 4: 15/19 (손실률 79%)
- 20유전자 서브셋 vs. 논문 전체 실행(27,416 중 3개 손실) → 급격한 차이
- 가설: 배치 크기가 모델 출력 품질에 영향 vs. 코드 revert가 완전하지 않음
- 검증 중: 코드 되돌린 버전에서 배치 4 재현(GPU 통제 재현 진행 중)

**Task 3 baseline 재정정**:
- 이전 문제: baseline() 함수가 "ANY isoform 일치" 카운트, score()는 대표 전사체만 사용
- 정정값 (대표 기준): gemoma 5,124 (20.06%), braker3 3,655 (15.38%), egapx 3,705 (16.06%)
- 검증: TAIR10에서 repaired=0, still_wrong=0 확인됨

**Task 3 fix 상태**:
- 코드 변경 중 (커밋 미작성), Task 4 통제 재현과 병렬 진행

### 다음 작업

1. GPU 통제 재현 완료 대기 — 배치 크기가 진짜 변수인지 재현 검증
2. Task 3 수정 완료 대기 — 코드 커밋, 테스트 통과 재확인
3. Task 4 전체 규모 실행은 배치 크기 문제 해결 후에 (20% 손실 재현 시에만)

## ▶️ 이전 스냅샷 (2026-08-05 16:35)

### 🟡 Task 3 완료, Task 4 배치 크기 검증 진행 중

**Task 3 구현 완료** (commit 862a815):
- 테스트: 48/48 통과 (test_polishing_scoring.py 38 + test_stage_annotations.py 10)
- 베이스라인 wrong-locus counts (CDS, 천 단위):
  - gemoma: 4945/25551 (19.35%), correct=20606
  - braker3: 2632/23761 (11.08%), correct=21129
  - egapx: 2921/23068 (12.66%), correct=20147
- stop condition NOT tripped — 세 도구 모두 Task 5 정보성 있음
- 결과 파일: polishing_benchmark/results/{gemoma,braker3,egapx}_baseline.json

**Task 4 진행**:
- 배치 크기=32 손실률 검증 중 (prompt_mode.py, GPU)
- 논문 실행과의 유일한 파라미터 차이 확인 필요
- 이전 라운드 패턴: 세 태스크 모두 "조용한" 숫자 버그 (spec 불일치)

### 다음 작업

1. Task 4 배치 크기 검증 완료 대기
2. Task 2 fix 라운드 2 (docstring 정정, I3/I5)
3. Task 3 spec 재검증 (이전 라운드 패턴 재현 여부)

## ▶️ 이전 스냅샷 (2026-08-05 16:16)

### 🟡 GPU 호스트 코드 버전 불일치 발견 — 동기화 중

**발견**:
- 발행 예측 (27,413/27,416 loci, 99.99%, 좌표 역전 0건)과 스모크 입력 구성 불일치 원인 파악
- GPU 호스트: 커밋 5d9929e (296행, `_normalize_gff_lines` 부재)
- GPFS 현재 코드: 491행, 좌표 뒤집힘·malformed 행 처리 포함

**영향**: 
- Task 4 스모크를 GPU 낡은 코드로 실행하면 손상된 출력 가능성
- 발행 예측이 정상인 이유: GPFS 코드 사용했기 때문
- 논문 Methods 확인: `prompt_mode.py` `num_beams=2, do_sample=False` (내 호출과 일치)

**진행 중**: GPU 호스트 동기화 (prompt_mode.py, src/transgenic/), 스모크 재실행 예약

### 다음 작업

1. 스모크 테스트 완료 대기 (GPU 동기화 코드)
2. Task 2 재리뷰 수정 (docstring 정정, I3/I5)
3. Task 1 재검증 (C2 수정 후 seqid 매칭)

## ▶️ 이전 스냅샷 (2026-08-05 15:55)

### 🟡 Task 2 Fix 라운드 1 완료 → 재리뷰에서 추가 이슈 발견

**구현 완료** (commit 346fcd4):
- 테스트: 35/35 통과
- Sanity: repaired=0, still_wrong=0, loci_compared 27,413, damaged 14

**재리뷰 발견**:
- C2 리뷰 근거 오류 ("2,496개 불일치" 측정 버그)
  - TAIR10 파일 순서 = primary 100% 일치
  - 두 테이블 구분은 유효하나 **근거가 docstring에 검증 없이 박혀있음**
- I3, I5 부분 해결, GM= convention 미검증

### 다음 작업

1. **Fix 라운드 2** — docstring 정정, I3/I5 완료
2. **Task 1 재검증** — seqid matching 재확인
3. **Task 3–4 병렬 진행**

## ▶️ 이전 스냅샷 (2026-08-05 15:19)

### 🟡 Task 2 구현 완료 → 코드 리뷰에서 Critical 2건 발견

**구현 결과** (commit 6a9450d):
- 테스트: 11/11 + 디렉토리 전체 21/21 통과
- Sanity check (TAIR10): repaired=0, still_wrong=0, preserved_correct=27,391 ✓

**코드 리뷰 결과**:
- Spec 불일치, Critical 2건
- **C2 (중대)**: "첫 mRNA" 규칙이 입력/예측만 아니라 참조에도 잘못 적용됨
  - TAIR10 파일 순서 첫 전사체가 primary 아닌 2,496개 locus에서
  - 도구당 ~1,500개가 오분류 (repaired에 위양성 포함 가능)
- 수정 지시 파일 생성

**damaged=15 vs 22 불일치 결론**:
- 내 기록(99.95%)과 구현자 결과(99.945%)는 같은 통계
- 22는 AtRTD3 기준, 15는 TAIR10 기준 — 다른 데이터셋

### 다음 작업

1. **C2 수정** — reference ruling 제거 후 재테스트
2. **Task 1 재검증** — C2 수정 후 seqid 매칭 assert 재확인
3. **Task 3 진행 가능** — baseline loci_wrong 분포 분석 (Task 2 블로킹 해제)

## ▶️ 이전 스냅샷 (2026-08-05 14:41)

### 🟡 Task 4 스모크 — GeMoMa 입력 데이터 품질 추적

**실행 결과**:
- 파이프라인 end-to-end 완료 (exit 0)
- **문제**: 3/5 예측 파싱 실패(60% skipped), 출력 GFF 손상 (CDS end<start, three_prime_UTR 유전자 범위 밖, feature type `-`)

**진단**:
- 가설 1: GeMoMa 입력 GFF 품질 불량 → 필터링/정규화 필요
- 가설 2: 스모크 설정 오류 (좌표 정규화, 서브셋 추출)
- 대조군: 동일 Chr1 범위 TAIR10 프롬프트 스모크 병렬 실행 중 (GPU)

**확인 필요**: GeMoMa input _smoke_gemoma_Chr1_20genes.gff3 feature 유효성

### 다음 작업

1. **TAIR10 대조군 완료 대기** — 정상 실행 확인 (parsing failures 0)
2. **GeMoMa 입력 GFF 검사** — feature 타입, 좌표 유효성, seqid 일관성
3. **Task 2 병렬 진행** — baseline 추출 시작

## ▶️ 이전 스냅샷 (2026-08-05 14:40)

### 🟢 Task 4 스모크 테스트 — 모델 캐시 경로 문제 해결

**진단과 해결**:
- `prompt_mode.py`는 오프라인 환경변수 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 동시 설정
- 상대경로 `./HFmodels`에서 LongSafari 토크나이저 로딩 필요
- GPU 호스트 `~/lsativa_rerun/HFmodels/` 기존 캐시 발견 (필요한 토크나이저 포함)
- 심볼릭 링크: `~/transgenic/HFmodels` ← `~/lsativa_rerun/HFmodels` 연결 완료
- 스모크 테스트 재실행 중

### 다음 작업

1. **Task 4 스모크 완료 대기** — 모델 로딩 + 점수 계산 성공 확인
2. **Task 1 커밋** — seqid 정확 매칭 검증 테스트 6개 추가
3. **Task 2–3 로컬 실행** — baseline 추출, loci_wrong 분포 분석

## ▶️ 이전 스냅샷 (2026-08-05 14:36)

### 🟢 Task 1 검증 완료, Task 4 스모크 테스트 버그 해결

**Task 1** (commit 4706ec1):
- seqid 정확 매칭 + dropped_seqids 검증 추가, 6개 테스트 추가 (모두 통과)
- 스모크 테스트: gemoma/braker3/egapx seqid exact match, dropped=0, exit 0 ✓

**Task 4 스모크 테스트 중 발견·해결**:
- 게놈 파일 seqid 이름 불일치 (Ensembl vs TAIR) → `transgenic_comparison/genomes/Athaliana_167_TAIR10.fa`로 교체 (동일 어셈블리, 7개 길이 일치)
- huggingface_hub 0.36.2 비호환 (transformers 4.44.2) → 0.24.7로 수정

### 다음 작업

1. **Task 4 스모크 재실행** — 모델 로딩 검증 후 GPU 시간 예약
2. **Task 2–3 진행** — baseline 추출 및 loci_wrong 분석 (로컬)

## ▶️ 이전 스냅샷 (2026-08-05 14:26)

### 🟢 GPU 환경 패키지 설치 완료 — Task 4 실행 준비 완료

**실행**: ssh gpu에서 micromamba activate transgenic 후 pip install
- transformers==4.44.2 설치 확인
- scipy, biopython 동시 설치
- exit code 0 (성공)

**상태**: 외부 GPU 환경이 Task 4(참조-프롬프트 실행 9시간)를 수행할 준비 완료. TAIR10 primary baseline 채점 완료 후 실행 가능.

### 다음 작업

1. **Task 1–2 실행**: baseline 추출 및 좌표계 정규화 검증 (GPU 불필요)
2. **Task 3 검토**: loci_wrong 분포 분석 후 도구별 측정 가능성 판정
3. **Task 4 예약**: GPU에서 외부 annotation(GeMoMa·BRAKER3·EGAPx) 데이터셋 다운로드 및 참조-프롬프트 실행

## ▶️ 이전 스냅샷 (2026-08-05 14:05)

### 🟢 writing-plans로 polishing-benchmark 계획 작성 완료

**파일**: `docs/superpowers/plans/2026-08-05-polishing-benchmark.md` 저장

**설계**: 2×2 전이표로 측정
| | 출력 맞음 | 출력 틀림 |
|---|---|---|
| **입력 맞음** | 보존 | damaged |
| **입력 틀림** | **repaired** | 여전히 틀림 |

TAIR10 primary는 왼쪽 위만 99.4%로 아무것도 측정 불가. 외부 annotation(GeMoMa·BRAKER3·EGAPx)은 수천 locus 차이 → repaired 칸 처음 채워짐.

**Task 순서**: 3 < 4 — GPU 시간 쓰기 전에 baseline에서 `loci_wrong` 분포를 보고 도구별 측정 가능성 판정. Task 2는 GPU 없이 채점 로직 검증(TAIR10 입력 시 repaired=0 확인).

**확인된 제약**:
- **GPU 없음** — Task 4는 외부 GPU 환경 필요 (참조-프롬프트 실행 9시간)
- **좌표계 3가지** — GeMoMa/BRAKER3는 `Ath_Chr1`, EGAPx는 `Chr1`. GFFCompare는 이름 불일치 시 0.0. Task 1에서 assert 필수
- **모델 캐시** — 로컬 없음, HF 토큰 확인 필요

**Placeholder 스캔**: 통과(0건)

### 다음 작업

1. **실행 방식 선택**: Subagent-driven (태스크별 검토) vs inline (제약점 두고 배치)
2. **Task 1–3 먼저 진행 가능** — baseline 추출로 polishing 실험 성립도 판단 가능

## ▶️ 이전 스냅샷 (2026-08-04 밤)

### 🟢 paper critic 14회 완주 — 검증 149 → 200, 원고·응답서 모두 정합

**방식**: `paper-scientific-critic` 에이전트 하나를 14회 이어서 실행(각 회차가 이전 수정을 검증 + 새 문제 탐색). 후반 10회는 `/loop` dynamic mode. 레퍼런스는 `paper-ref-hunter`로 2회 Crossref 전수 대조.

**성과**: Blocker 9건 + Major 30건 이상 해소. **상당수가 직전 회차에 내가 만든 것** — 반복 검토가 값을 했다.

### 재분석 5건 (문장 수정으로 안 되는 것들)

1. **AUGUSTUS 비교가 비대칭** → `28_score_added_isoforms.py --augustus`로 동일 기준 채점. **18.1% vs 1.3%(14배)**, AtRTD3 18.5% vs 10.2%. recall은 AUGUSTUS가 높다(10.3% vs 3.6%, 40배 많이 제안)
2. **Table S2/Figure 5가 v0.12.10** → `29_rescore_benchmark_v0126.py`로 128조합 전부 v0.12.6 재채점. Figure 5와 6이 같은 척도(67.2 vs 67.6). base는 불변
3. **400M prompted가 2-beam으로 채점** → top-beam 재채점. A. thaliana precision 46.8 → 93.7
4. **BUSCO beam 문제** → 파일럿으로 completeness 동일(97.6%) 입증, 재실행 불필요 확인
5. **S. lycopersicum 0.0 행** → 염색체 이름 불일치(Chr0 vs 0). seqid 정규화 + 불일치 시 즉시 실패

### 발견한 결함 (원고 수치가 바뀐 것)

- **off-by-one**: `summary_report.json`의 `total_predicted`가 tmap 헤더를 레코드로 계수 → 29,923이 아니라 **29,922**. Methods + Table S4b 6칸
- **Spearman 공식 오류**: d² 축약식은 동점이 없을 때만 유효. 순위 Pearson으로 재계산하니 CDS ρ = **−0.88**(내가 쓴 −0.81 아님). 체크리스트의 −0.880이 처음부터 옳았음
- **35,386을 "nuclear"로 오표기** — 실제로는 organellar 210 포함 전체
- **Helixer가 transcript-level 1위**라는 서술 → v0.12.6에서 Tiberius soft-masked 9종 + unmasked 1종(V. vinifera)
- **레퍼런스 3건**: Stiehler 2020→2021, BUSCO Manni→Tegenfeldt(v6), SegmentNT 제목

### 응답서 전면 재작성

원고가 4라운드 바뀌는 동안 응답서는 그대로여서 **제출 blocker**로 격상. 부분 수정을 3회 했더니 매번 새 불일치가 생겨(그중 다수가 내 대체 텍스트), **전체 읽기**로 전환 → 10건 발견(7건 신규). GFFCompare 버전 프로토콜 모순, 근거 없는 22,601, "auxiliary segmentation supervision"(Methods에 없는 학습 요소), "리뷰어 코멘트 전문 수록"(사실 아님) 등.

### 상태

검증 **200/200**, 인용 58=58, abstract 247/250, 본문 7,233단어, 패키지 무결성 20/20. 커밋 `d3ab629..44bc8ad` 푸시.

### 다음 작업 — 저자 결정 4건

1. **Z. mays Tiberius soft-masked 재실행 여부** (실패 런: base F1 0.0, BUSCO 0.2%. 3곳에 공시됨. 리뷰어 #2가 해당 도구 그룹)
2. **abstract에 AUGUSTUS recall 트레이드오프 추가 여부** (247/250단어, 넣으려면 잘라야 함)
3. **figure callout 하우스 스타일 변환** (`(Fig. 5)` → `(Figure 5)`, 원고·보충·응답서 동시)
4. **Funding · ORCID 5명 · 교신 이메일**

### 방어 가능하나 리뷰어가 물을 수 있는 것

헤드라인 수치가 95.4% 공급분 annotation 채점(6곳 공시) · 구조 일관성 대비가 디코딩으로 교란(deterministic de novo 1회면 해소) · Figure 4가 보고 지표를 내지 않는 실행에서 왔고 저장소에 없음 · split seed 미기록 · 평가 설정 caveat 7회 반복

## ▶️ 이전 스냅샷 (2026-08-04 22:58)

### 🟢 루프 10회차(최종) 완료 — Majors 3건 발견·수정, CDS ρ 오류 규명

**M1 — Table S4b 캡션 orphaned fragment**: Supplement l.329의 중복 절 "recovered only on an exact intron-chain match (GFFCompare class code '=')." 삭제. 제 캡션 편집에서 문장 머리는 삭제했으나 꼬리가 남은 것 — 세 번째 발생 사례.

**M3(a) — CDS 상관계수 계산 오류 발견**: Spearman ρ 계산이 동점을 무시하는 공식 사용. 올바른 tie-safe 계산으로 **CDS ρ = −0.88** (제 논문본의 −0.81은 오류). length ρ = −0.11은 정확. 원고 수정 완료(`manuscript_v2.md`).

**M3(b) — CDS bin 15개 F1=0 검증**: 실제 성능 결과, 데이터 오류 아님. 예측은 1~7개, 레퍼런스는 수십~116개 → exposure bias 논증 지지. 퇴화 bin 없음.

**상태**: 검증 **196/196**, `manuscript_v3_plant_communications.md` 생성, 수정 커밋 완료.

### 다음 작업

1. 저자 정보 3건 — Funding, ORCID 5명, 교신 이메일
2. 최종 제출 준비

## ▶️ 이전 스냅샷 (2026-08-04 22:51)

### 🟢 루프 9회차 완료 — 검증기 구조적 빈틈 발견·수정, Major 4건 해결

**검증기 빈틈 규명**: 181개 체크는 표·JSON에서 직접 읽은 값만 고정했고, (i) 표 전체를 가로질러 집계한 통계와 (ii) `revision/results/` 밖의 값은 미검사. Results의 벤치마크 주장 4개(Results l.40 Table S2 128행 집계) 중 하나는 2회차 동안 틀린 채 방치됨.

**M1 — Table S4b 캡션**: "Source: `transgenic/revision/results/*/summary_report.json`"은 이제 부정확. `Predicted transcripts` 컬럼은 예측 mRNA 파일에서 직접 계산되는데, 캡션이 요약 보고서에서 생성된다고 표기. 스크립트 고정.

**M3·M4 해결**: Figure 3 Abstract 헤드라인 92.2%, 71.1% 재계산. Z. mays Tiberius 71.18 → 71.2로 반올림(0.1 오차는 `.stats` 소수 1자리 정밀도 내). 네 벤치마크 주장(Results 순위·카운트)을 이제 코드로 직접 재도출.

**상태**: 커밋 `02fcd8f`, 검증 **196/196**(181에서 15개 증가). 루프 10회차 개시.

### 다음 작업

1. 루프 10회차(마지막) 실행 — 생성 표·JSON 인용 수치 전수 재검증
2. 루프 종료 후 **저자 정보 3건** — Funding·ORCID 5명·교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:46)

### 🟢 루프 8회차 완료 — Major 2건 발견·수정, off-by-one이 Table S4b 6셀까지 퍼짐 확인

**Major 1 — Table S4b의 off-by-one**: Methods l.120에서 분모 고쳤으나 Table S4b는 같은 JSON 필드에서 생성되며 수정 안 됨. `Predicted transcripts` 여섯 칸이 전부 +1(27,634 / 29,923 / 69,050). 표 생성 스크립트가 이제 예측 파일의 mRNA 행을 직접 세도록 수정. 바로 옆 Table S4a, Figure 5 범례, Results, Methods 모두 정확한 값 사용 중이라 **보충자료 자기모순** 해결.

**Major 2 — Figure 5 범례 93,690 vs Table S5 93,063**: Figure 5는 전체, Table S5는 nuclear만(organellar 627건 제외). 범례에 "(Table S5 counts are lower because that table excludes organellar transcripts)" 단서 추가, 두 값 모두 예측 파일로 재검증.

**무효 이슈 추적**: `exact_matches`·`duplicate_exact_matches`·`distinct_ref_matched`는 class code `=` 행만 세므로 헤더 영향 없음. isoform-count 합산은 27,413으로 정확.

**상태**: 커밋 `ef88ac5`, 검증 **181/181**, manuscript_v3 docx·웹페이지 갱신. 루프 9회차 진행 중(검증 스크립트 빈틈 탐지).

### 다음 작업

1. 루프 9회차 결과 수집(생성 표·JSON 인용 수치 전수 검토)
2. 루프 종료 후 저자 정보 3건 — Funding, ORCID 5명, 교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:41)

### 🔴 루프 7회차 완료 — Blocker 1개(제 실수), Major 패턴 1개 확정, off-by-one 버그 발견

**Blocker 1개 확정·수정**: Methods l.120에 지난 회차에서 삽입한 "two more than the 27,413…" 절이 동사와 AUGUSTUS 플래그 사이에 들어가, **TransGenic이 `--sample=100 --alternatives-from-sampling=true`로 돌았다**고 거짓 서술. 문법 재정렬로 AUGUSTUS 명령줄 명확 분리.

**Major 패턴 1개 확정**: "92.2/92.4·71.1/70.6" 체크리스트 금지 쌍이 **Methods에서 나란히 놓임**(l.49·50 base-level 비교). 응답서·Results 패턴과 동일. Supplement 검색 결과 더 이상 인스턴스 없음.

**off-by-one 버그 규명**: 29,922 vs 29,923 불일치 근인 — tmap 결과 파일의 실제 mRNA 레코드는 29,922개이나, 헤더 줄 포함 행수 29,923을 파이프라인이 분모로 사용(반올림 22,260/29,922 = 74.4%는 변화 무). 원고 분모 명시 오류, 검증 스크립트 고정점 추가.

**패턴 전수 조사 완료**: organellar 배제·평가 설정 caveat·top-beam 필터링·GFFCompare 버전·디코딩 caveat·반환본/추가분 구분 **모두 일관성 있음**.

**상태**: 커밋 `f26fd32`, 검증 **179/179**, docx·웹페이지 갱신. 루프 8회차 진행 중.

### 다음 작업

1. 루프 8회차 Major 적발 결과 수집
2. 루프 종료 후 저자 정보 3건 — Funding·ORCID 5명·교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:35)

### 🟢 루프 6회차 완료 — Major 2건 발견, l.44 locus count 불일치

**pass-9 수정 전부 확인**: Maj-2(l.89 "stored as auxiliary records" 중복 제거됨), min-1·2·3·4 모두 저장, Methods l.120 organellar 제외 명시 확인.

**Maj-1 — locus count 모순**: l.44 "27,413 loci에서 생성" vs l.50 "27,213 loci 커버". altonly 통계 파일 확인 결과 29,922 mRNA in **27,213 loci**(1.1 per locus) — l.50이 정확. l.44 텍스트 교체 필요(에이전트 제공 텍스트에서 유래).

**Maj-2 상세 미정**: 아직 report에서 불완전. l.50 이후 전체 크리틱 대기 중.

**상태**: 저장소 외 파일(원고·응답서 docx) 갱신 완료, 커밋 대상 없음. 루프 7회차 진행 중.

### 다음 작업

1. Maj-2 report 전문 수집 후 수정(l.50 이후 내용)
2. l.44 locus count 수정("27,413" → 평가셋 크기 명시로 변경)
3. 루프 종료 후 **저자 정보 3건** — Funding·ORCID·교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:27)

### 🟢 루프 4회차 완료 — 응답서 전체 읽기로 신규 7건 발견, 원고 오류 1건 수정

**전체 읽기 전환 성공**: 타겟 라인 수정 방식에서 놓친 응답서 항목 10건 발견(신규 7건 미지적). 두 가지 패턴 — ① 제 대체 텍스트가 원문을 안 지운 기계적 손상(Blocker L1, Major M1·M4), ② 타겟 지정으로 건드리지 않은 문단들.

**원고 오류 발견·수정**: m5 추적 중 Table S8(27,206/35,176 nuclear)과 Results(35,386 전체) 불일치 원인 규명. Results가 35,386을 "nuclear transcripts"로 표기했으나 실제는 organellar 210건 포함한 전체. 라벨 정정(35,386 → 35,386 transcripts, of which 35,176 are nuclear) + Table S8 캡션 nuclear-only 명시 + 검증 스크립트에 고정점 추가.

**응답서 주요 내용**: Blocker 3건(문장 중복, training signal 허위 주장, 리뷰어 인용 누락), Major 7건. 리뷰어 #2 "junction 추가 방식" 답변에서 Methods와 불일치(auxiliary segmentation supervision 언급했으나 실제는 없음).

**상태**: 검증 **178/178**, 커밋 `4c38212` 푸시. 루프 5회차 진행 중.

### 다음 작업

1. 루프 5회차 결과 수집(Blocker·Major 패턴 추적)
2. 루프 종료 후 저자 정보 3건 — Funding, ORCID 5명, 교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:21)

### 🟢 루프 3회차 완료 — 블로커·주요 불일치 5건 해결

**Blocker 1개 확정·수정**: 응답서 l.61이 BUSCO를 "gene-boundary resolution에 의존하지 않는다"며 리뷰어 #2 핵심 반론의 **해결책으로 제시**. Discussion l.80은 "completion mode에서 공급된 annotation이 하한이라 그것만으로는 결론 낼 수 없다" 명시 → 리뷰어 #1 BUSCO 답변 쪽만 단서 붙음, 정작 중요한 쪽엔 없음. 문단 전체 정정.

**Major 4건 해결**:
1. **Z. mays 71.1 비교값 문제** — 응답서가 70.6 자리 비교값과 함께 인용 → 체크리스트 금지 항목. 교체 제거.
2. **splice-event 방향성 왜곡** — 제 문장이 이긴 축(우리 14,643)만 인용 → 양쪽 모두 기술, 원고·검증 수정.
3. **Gabriel 참조·Tiberius 버전** — 응답서 l.32·38 저자 표기 + v1 모호성 정리. 공식 지침 추가.
4. **M6 평가 설정 명확화** — Table S3 캡션에 "Table S2처럼 TransGenic은..." 평가-설정 문구 추가.

**상태**: 검증 **175/175**, 커밋 `4dfeb12` 푸시. Docx/웹페이지 갱신.

**핵심 패턴**: 응답서 부분 수정마다 새 불일치 발생(3회 중 2회는 내 대체 텍스트 원인) → 루프 4회차부터 **전체 읽기 방식** 전환.

### 다음 작업

1. 루프 4회차 진행 중(응답서 전체 크리틱)
2. 루프 종료 후 저자 정보 3건 — Funding·ORCID·교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:15)

### 🟢 루프 2회차 완료 — 응답서·Results 모순 3건 해결

**Blocker 1개 확정·수정**: 응답서 l.107이 같은 문단에서 모순 — "v0.12.6 전체 분석"으로 개시하되 결론은 "v0.12.10 벤치마크"라 기술. 뒤 문장(이미 폐기된 텍스트)을 "chromosome-4 pilot만 v0.12.10 보유"로 정정.

**Major 3건 해결**: 
1. **622/311 교체 근거 문제** — 모든 컬럼이 정확히 2배(59,426 = 2×29,713)라 중복 입증 불가. 응답서를 실제 근거(27,413 locus, 54,826 gene 레코드)로 재서술.
2. **참조 희소성 주장 재정리** — Discussion은 이미 철회했는데 응답서에 세 곳 남음. Results/Figure·Table S2 범례에 reviewer #2가 요구한 "TransGenic 유리" 명시 추가(whole-annotation 72.5% 기준).
3. **splice-event 비교 누락** — Table S4c·응답서에만 있던 3,033건 37.5% vs AUGUSTUS 41,917건 4.4%(8.5배 우위)를 Results에 추가, 검증 고정(**173/173**).

**상태**: 응답서·원고 수정, 검증 **173/173**, 커밋 `84de8e5` 푸시.

### 다음 작업

1. 루프 3회차 대기(응답서 신규 크리틱 중)
2. 루프 종료 후 **저자 정보 3건** — Funding, ORCID 5명, 교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 22:10)

### 🟢 루프 1회차 완료 — 5차 크리틱 + 응답서 blocker 해결

**5차 크리틱 적용**: S. lycopersicum de novo 좌표계 버그 재확인(71.4/64.1로 정정), Tiberius soft-masked 1위 9종 + V. vinifera unmasked(unmasked만 완료), Z. mays Tiberius soft-masked 행 공시(base 0.0%, BUSCO 0.2% — 경쟁 도구 0.0 설명 없음 플래그).

**응답서 blocker 격상 이유**: GFFCompare 버전 프로토콜이 v0.12.10으로 서술되는데 보충자료는 v0.12.6 → 리뷰어가 정확 버전 요구 리비전에서 편집자 나란히 확인 시 모순 노출.

**응답서 6건 수정**: (1) 버전 프로토콜 v0.12.10 → v0.12.6, (2) "three times more likely" → 실제 수치 기반 재서술, (3) 47.3%/16.3% 제거, (4) 추상 샘플링 비대칭 비교 정리, (5) 22,601은 Table S5와 불일치 → 622/311로 교체(자동 검증 불가 데이터 제거), (6) Manni 2021 → Tegenfeldt 2025.

**상태**: 검증 **169/169**, 인용 **58=58**, abstract 247/250, 커밋 `7e005d9` 푸시.

### 다음 작업

1. 루프 2회차 대기(응답서 신규 크리틱 중, Blocker 없으면 루프 종료)
2. **저자 결정 필요** — Z. mays Tiberius 재실행 여부, AUGUSTUS recall 추가(abstract 단어 수), Funding·ORCID·교신 이메일 3건

## ▶️ 이전 스냅샷 (2026-08-04 21:53)

### 🟢 4차 크리틱 + 레퍼런스 재검증 완료 — 염색체 이름 버그 발견·수정

**크리틱 적용**: Abstract에서 2.5% → 1.3% 정정, Discussion의 비대칭 비교(47.3/16.3) 제거, 결론에서 AtRTD3 기준 1.8배 추가, precision 기반 서술 정직화.

**주요 발견**: v0.12.6 재채점 중 S. lycopersicum de novo 두 행이 0.0이 된 것은 **예측 좌표계(Chr0) vs 레퍼런스(0) 불일치**였음. GFFCompare는 이름 미매칭 시 모든 점수를 0.0으로 조용히 채점. 스크립트 수정 → 복구된 값 71.4/64.1(v0.12.10과 정확히 일치) → Table 검증 0.0 셀 없음 확인.

**주장 수정**: *"Helixer가 1위"* → 실제로 v0.12.6에서 Tiberius(soft-masked)가 13종 중 10종 1위, Helixer 2종. Gabriel et al. 저자 13명 전부에서 10명+et al. 규칙 적용. AUGUSTUS --sample 모드 기초논문 Stanke et al. (2006) 추가.

**레퍼런스 구조 문제**: Tegenfeldt 알파벳 정렬 오류 + 빈 줄 손실로 파일이 56개로 파싱됨 → 이동·재구조화. 이중 빈 줄 제거.

**상태**: 검증 **169/169**, 인용 **58=58**, abstract 233/250. 커밋 `35f675a` 푸시, 패키지/웹페이지 갱신.

### 다음 작업

1. `response_to_reviewers.md` 갱신 — 4라운드 원고 변경 반영
2. 저자 정보 3건: Funding · ORCID 5명 · 교신 이메일

## ▶️ 이전 스냅샷 (2026-08-04 밤)

### 🟢 paper 스킬 3차 검토 + 레퍼런스 전수 검증 — 재분석 3건 수행

**검토 방식**: `paper-scientific-critic`을 3회 이어서 실행(각 회차마다 이전 수정 검증 + 새 문제), `paper-ref-hunter`로 레퍼런스 56건 Crossref 대조.

**재분석 3건** (문장 수정으로 안 되는 것들):

1. **AUGUSTUS 비교가 비대칭이었다** — 결론의 "7배"는 양변이 다른 측정(우리는 추가분만·중복병합·exact CDS, AUGUSTUS는 전체·중복유지·exon chain). `28_score_added_isoforms.py --augustus`로 동일 기준 채점 → **18.1% vs 1.3% (14배)**, AtRTD3 기준 18.5% vs 10.2%. 정직한 반쪽도 기재: AUGUSTUS는 recall이 높다(10.3% vs 3.6%) — 40배 많이 제안하므로.
   ⚠️ 첫 시도에서 레퍼런스를 AUGUSTUS 자체 primary로 걸러 8.0%가 나왔음. 양쪽 모두 TAIR10 primary 제거로 수정.

2. **Table S2와 Figure 6A 불일치의 원인은 GFFCompare 버전** — 같은 파일·같은 레퍼런스로 v0.12.10은 79.3/93.7, **v0.12.6은 62.0/73.3**(Figure 6A의 61.9/74.4와 일치). `29_rescore_benchmark_v0126.py`로 **128개 조합 전부 v0.12.6 재채점** → 두 그림이 같은 척도. base level은 불변이라 Figure 5A·"8 of 13" 유지, completion 마진은 **24–53 → 17–48**.

3. **BUSCO beam 문제는 측정으로 종결** — 같은 BUSCO 버전으로 top/both 비교: **completeness 97.6%로 동일**, F·M도 동일, S/D 분리만 이동(0.0/97.6 → 88.5/9.2). 원래 캡션 주장이 옳았고, 내가 "beam-invariant로 읽지 말라"로 바꾼 것이 근거 없는 과교정이었음. 수치와 함께 진술하고 검증에 고정.

**레퍼런스 3건 수정** (56건 전수 Crossref 대조, 53건 clean):
- **Stiehler 2020 → 2021** (print issue). Holst·Guo·Vogel 연도 정정은 했는데 이것만 누락돼 있었음
- **BUSCO Manni 2021 → Tegenfeldt 2025** (Methods가 v6.0.0을 쓰는데 Manni는 v4/v5용)
- **SegmentNT** 제목 접두 제거(저자들이 2025-03에 개제)
- HyenaDNA DOI 추가, **Tiberius clade 논문(Gabriel 2026) 추가** → 57건
- ANNEVO 서술을 논문 주장에 맞게 축소("state-of-the-art" → "기존 ab initio를 능가하고 evidence-based에 필적")

**원고 구조 변경**: `de novo 용이 아님`을 축으로 abstract 전면 재작성 + Introduction/Discussion/결론까지 일관화. 프롬프트 구성 명시(29,922 중 28,536이 공급분, 1,103 distinct 추가, 1,076 locus).

**파이프라인 결함 3건**: 빌드 스크립트가 abstract·back matter·availability를 하드코딩해 v2 수정이 버려지고 있었음 → 전부 v2에서 읽도록. 보충자료 그림 범례 누락을 빌드가 assert로 막음.

**상태**: 검증 **167/167**, 인용 57=57, abstract 246/250, 패키지 무결성 20/20. 커밋 `abdbc6d..d3ab629` 푸시.

### 다음 작업

1. **4차 검토 진행 중** (크리틱 + 레퍼런스 재확인) — 결과 나오면 Blocker/Major 자동 수정
2. `response_to_reviewers.md` 갱신 — 원고가 4라운드 바뀌었는데 응답서는 그대로 (리뷰어가 나란히 읽음)
3. 저자 입력 3건: Funding(v3:161) · ORCID 5명(v3:10) · 교신 이메일(v3:8)

## ▶️ 이전 스냅샷 (2026-08-04 21:28)

### 🟢 1·2·3 완료 — GFFCompare 버전 일치, BUSCO beam-invariant 검증

**1. Tiberius 인용 추가**: 식물 모델 벤치마킹 후 Gabriel et al. 2026 (Plant Communications 권장) Methods에 추가. **인용 57 = 57** 확인.

**2. Figure 5B — 버전 차이 해소**: 같은 파일·레퍼런스를 두 GFFCompare 버전으로 재실행:
- v0.12.10: 79.3/93.7 (현행)
- v0.12.6: **62.0/73.3** ← Figure 6A(61.9/74.4)와 일치
128개 조합 전수를 v0.12.6으로 재채점. **Base level은 불변** (Figure 5A, "8 of 13 species" 유지), transcript level만 전 도구 하락. A. thaliana completion F1이 67.6과 일치. 마진 **24–53 → 17–48 정정**. Table S10에 버전 기록.

**3. BUSCO beam-invariant 검증**: A. thaliana에서 top-beam vs both-beam을 동일 BUSCO 버전(5.8.3)으로 측정:
- Completeness: **둘 다 97.6%** (F, M도 동일)
- 차이: S/D 분리만
원 캡션 주장("S/D 영향, completeness 무영향") 재확인. 검증 스크립트 네 값 고정.

**상태**: 167/167 검증, 인용 57, abstract 246/250. 커밋 e995979, d3ab629 푸시, 패키지·웹페이지 갱신.

### 다음 작업

1. `response_to_reviewers.md` v3 변경사항 반영
2. 저자 정보 3건 확정 (Funding, ORCID, 교신 이메일)

## ▶️ 이전 스냅샷 (2026-08-04 19:44)

### 🟢 레퍼런스 검증 및 3건 오류 수정 완료 — 53/56 clean

**검증 결과**: 56개 참고문헌 전수 확인(Crossref·PubMed·bioRxiv API). 53개 정합 통과, 3건 오류 발견 및 수정.

**오류 3건**:
- #7 de Almeida et al. 2024: bioRxiv 제목 v3/v4(2025-03-25/26) 재명명 반영. "SegmentNT:" 프리픽스 제거 → 원고 수정 완료.
- #27 Manni et al. 2021: 방법론에 BUSCO **v6.0.0** 명시되어 있으나 인용은 v4/v5용. → v6 정식 인용(Manni et al. 2024 Nature Biotechnology)으로 변경·Methods 버전 일관성 확인.
- 알파벳 순서: Tegenfeldt et al. 2021 M구간에 잘못 배치 → **T 구간으로 이동**.

**빌드·검증 완료**: 
- v3 재생성(23_build_pc_manuscript.py) ✓
- 인용 검증(22_check_citations.py) ✓  
- 원고 수치 검증(19_verify_manuscript_numbers.py) ✓
- DOCX 생성 ✓
- 커밋 완료(revision/scripts).

### 다음 작업

1. `response_to_reviewers.md` vs v3 불일치 확인 — #16–19 지점 대조
2. 투고 준비물(cover letter, author affiliation) 최종 검수

## ▶️ 이전 스냅샷 (2026-08-04 19:38)

### 🟢 B2·B1 비대칭 비교 수정 및 원고 v3 갱신 완료 — 163/163 검증 통과

**AUGUSTUS 비교 재계산**: reference 필터링 오류(AUGUSTUS 자체를 primary로 걸러 TAIR10 primary가 alternative에 섞임)를 양쪽 모두 TAIR10 primary만 사용하도록 수정. 결과 **정확한 대칭**: TransGenic precision 18.1%, AUGUSTUS 1.3%(**14배**, 이전 7배 오류). recall은 AUGUSTUS 높음(10.3% vs 3.6%).

**원고 일괄 수정**:
- 28,536개(95.4%) → 1,386개(non-supplied) → 1,103개(distinct structures) 분할 명시
- "forty times"(7배 오류) → "nearly forty times"(14배 정정)
- Methods: CDS 기준·beam 필터링·non-pooling 경고 추가
- Table S3: beam-search 중복으로 S/D 비율 영향 명시

**검증 결과**: 163/163 통과. v2 에서 v3으로 빌드, DOCX 생성 완료.

### 다음 작업

1. M3(l.40과 l.74 모순)·M4(l.74 수정 미흡)·M6(수치 설명) 해소
2. `response_to_reviewers.md` 검증 — 응답서와 v3의 불일치 확인

## ▶️ 이전 스냅샷 (2026-08-04 19:25)

### 🟢 2-beam 중복 수정·추가분 재채점 완료 — Table S2·v3 갱신

**비크리틱이 찾은 400M prompted 채점 오류**: 4개 종(A. thaliana, P. trichocarpa, S. lycopersicum, Z. mays)의 2-beam 파일(유전자 2배)이 실수로 사용됨. A. thaliana만 54,826개(정상의 2배). 재채점 후 **400M prompted TxF1: 58.9% → 85.9%** (160M의 84.2%를 넘음). de novo 최고 도구 대비 마진도 **24–53점**(이전 기록 2–55점은 오류).

**추가분 단독 채점 결과**: TAIR10 대체형식 기준 TransGenic 18.1% vs AUGUSTUS 2.5%. Figure 5B·Table S2 갱신됨. 검증 157/157 통과.

**원고 일괄 수정**: 
- "refinement layer over existing annotation pipelines" 추가
- 추가분 정확도 18.1% 명시  
- 구성 기준 명확화 (CDS vs GFFCompare =)
- Discussion/Results 모순 제거

**파이프라인 결함 해결**: Figure S4 범례·supplemental_information.md 평가 설정 문구 추가. Data and code availability 하드코딩 제거. v3가 v2 변경사항 모두 반영 확인. 패키지 갱신·커밋 완료.

### 다음 작업

1. `response_to_reviewers.md` 검증 — 원고와의 불일치 해소 (#16–19 지점)
2. 다중 isoform 층화·BUSCO 순환성 등 재분석 필요 여부 결정

## ▶️ 이전 스냅샷 (2026-08-04 19:02)

### 🟢 종합 크리틱 완료 — 빌드 스크립트·abstract·프롬프트 구성 수정

**크리틱 에이전트의 지적 중 검증 결과**:
- **Blocker 1/5**: 빌드 스크립트가 abstract·back matter를 하드코딩하고 있었음. v2 수정이 투고본에 반영 안 됨. **수정 완료** — 두 섹션 모두 v2에서 읽도록 변경, v2에 Funding 섹션 신설.
- **Blocker 2/8**: 채점된 29,922개 중 91.6%(27,399개)가 입력 프롬프트. **재분석 결과 설계 의도임** — "기존 annotation 폴리싱"이 목표라면 정상. **명시 완료** — 95.4% 재현·1,386개 추가·1,064개 locus 분포로 정확 수치 기록, 155/155 검증 통과.
- **Blocker 3**: de novo 샘플링(True) vs completion 비임(False) 미명시. Methods·Discussion 모순 있음. **수정 완료** — 디코딩 모드 문서화, "beam search deterministic" 제거.
- **Blocker 6**: "inside the range" 논리 오류(85.4 < 92.9 하한). **수정** — prompt가 "제거"가 아니라 "감소"됨 명확히.
- **Blocker 9**: 저장소 비공개라고 함. **잘못된 판단** — 2026-08-05 기준 공개 상태 확인.

**Abstract 재작성** — "de novo 용이 아님"을 축으로 순서 변경: 이제 prompt-completion이 먼저, de novo는 "학습 능력 입증용"으로 후퇴.

**남은 문제**: response_to_reviewers.md와 v3의 어긋남(#16–19) — 응답서가 "모든 그림에 명시됨", "per-base는 부차 맥락"이라 주장하지만 원고가 그 뒤 여러 번 바뀜.

### 다음 작업

1. `response_to_reviewers.md` 검증 및 원고와의 불일치 해소
2. 재분석 필요한 지적들(다중 isoform 층화·BUSCO 순환성·Table S2/Fig 6A 불일치) 검토 여부 결정

## ▶️ 이전 스냅샷 (2026-08-04 18:29)

### 🟢 Back matter 전부 작성 — 원고에서 `[TODO]` 소멸

**작성 완료**:
- Author contributions — CRediT 형식 (프로젝트 기록에서 구성, **읽고 확인 필요** — J.C.C./H.T. 역할 정정 가능성 높음)
- Declaration of competing interests — 표준 문구
- Acknowledgments — Pronghorn HPC·GPU, Phytozome/JGI, AtRTD3/AGAT/GFFCompare/BUSCO 개발자
- ORCID 필드 **신설** (원고에 자리 자체가 없었음) + 교신저자 줄 추가

교신 이메일은 placeholder만 (`wyim@unr.edu` 추정 불가 — 원 제출본에 교신 줄 자체 없음).

### 🟢 웹 페이지 스크립트화 — 어긋남 구조적으로 제거

`26_build_submission_page.py` 신설. 패키지에서 페이지 재생성하므로 폰트·그림 변경이 자동 반영됨.

**검증**: 임베드 10장 원본과 바이트 단위 재현, 폐기된 locus 흔적 0.

### 현재 상태

검증 **149/149**, 인용 56=56, 패널 9/9, 패키지 24파일 4.1 MB, 그림 20/20, 폰트 10/10 Liberation. 커밋 `4461410` 푸시.

Johny 의존성 이미 제거 (Figure S4 재구성 완료), 실제 막힘 없음.

### 다음 작업

1. **저자**: Funding 기관·과제번호, ORCID 5명, 교신저자 이메일 → `manuscript_v2.md` 수정 후 `23_build_pc_manuscript.py` 재실행
2. Author contributions 기여 분담 확인 (J.C.C./H.T. 역할)
3. (선택) Johny 연락되면 Figure S4 출처 이메일(`email_jlomas_figS4_provenance.md`) 보내기

## ▶️ 이전 스냅샷 (2026-08-04 밤)

### 🟢 Johny 없이 마감 — 남은 건 저자만 아는 정보 3개뿐

**back matter 전부 작성** (원고에서 `[TODO]` 소멸):
- Author contributions — CRediT 형식으로 서술 (기여 분담은 **읽고 확인 필요**, 특히 J.C.C./H.T. 역할)
- Declaration of competing interests — "The authors declare no competing interests."
- Acknowledgments — Pronghorn HPC·학과 GPU, Phytozome/JGI, AtRTD3 저자, AGAT/GFFCompare/BUSCO 개발자
- ORCID 필드 **신설** (이전엔 자리 자체가 없었음) + 교신저자 줄 추가

**남은 3개 — 저자만 채울 수 있음** (v3 기준 위치):
| 항목 | 위치 |
|---|---|
| Funding 기관·과제번호 | l. 161 `[TO BE COMPLETED BY THE AUTHORS]` |
| ORCID 5명 | l. 10 `[ORCID]` |
| 교신저자 기관 이메일 | l. 8 `[INSTITUTIONAL EMAIL]` |

⚠️ 교신 이메일은 원 제출본에도 없어서 **추정하지 않고 자리만** 뒀음 (`wyim@unr.edu`로 쓸 뻔했다가 되돌림).

### 🟢 웹 페이지를 스크립트로 전환 — 세 번 조용히 낡았던 문제 해결

`073026/index.html`은 07-30에 수동 작성 후 **폰트 교정·Figure 4 복원·Figure S4 재구성** 세 번을 놓쳤다. 유일한 단서가 "PNG보다 오래된 타임스탬프"였음.

**`revision/scripts/26_build_submission_page.py`** 신설 (파이프라인 01–26) — 패키지 내용에서 페이지를 재생성하므로 더 이상 어긋날 수 없다. 그림은 900px·96색으로 다운샘플 후 임베드(0.61 MB), 원본 해상도는 PDF로 동봉.

**검증**: 임베드 10장이 현재 패키지 PNG에서 동일 파이프라인으로 바이트 단위 재현, 새 Figure S4 locus 반영, 폐기된 AT1G44575·AT3G56730 흔적 0.

**상태**: 검증 149/149, 인용 56=56, 패널 9/9, 패키지 24파일 4.1 MB, 그림 무결성 20/20, 폰트 10/10 Liberation. 커밋 `01df393..4461410` 푸시.

### 다음 작업

1. **저자**: Funding·ORCID·교신 이메일 3개 입력 → `manuscript_v2.md` 수정 후 `23_build_pc_manuscript.py` 재실행 (v3 직접 편집 금지)
2. Author contributions 분담 확인
3. (보류) Johny 연락되면 `raw_TAIR10_...` 정체 — 아무것도 의존하지 않음. 초안 `email_jlomas_figS4_provenance.md`

## ▶️ 이전 스냅샷 (2026-08-04 18:09)

### 🟢 Figure S4를 재현 가능한 소스로 이전 — 출처 불명 파일 의존 제거

**계기**: Johny가 `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)를 기억하지 못함. 코드·데이터 가용성이 리뷰어 2명의 Major Issue인 원고에서, 인용한 파일로 재도출 못 하는 보충 그림은 방어 가치가 없다고 판단.

**해결**: 방어할 필요가 없었다. **원고가 이미 보고하는** prompted 예측(`A_thaliana_transgenic400Mprompt_beam1.gff3`, 전체 27,413 locus)으로 같은 스캔을 돌리니
- **45개** locus 조건 충족 (chain이 AtRTD3와 정확 일치 + TAIR10과 불일치 + intron 2개↑)
- 그중 **13개**는 TAIR10에 없는 splice junction 사용, **32개**는 조합만 새로움

**새 Figure S4 6개**: AT4G30510, AT4G12610, AT2G37450, AT1G52500, AT4G22540, AT3G61390 → 전부 novel feature 직접 보임. 기존 6개는 3개가 약한 증거였음.

### 🔴 재구성 중 검증이 잡아낸 결함 2건

1. **GTF/GFF3 혼동**: TAIR10을 GTF에서 절취하니 exon 행 처리로 highlight 판정이 깨짐 → `Athaliana_167_TAIR10.gene.clean.gff3`에서 재추출
2. **AT4G30510 지표 혼동**: "9개 전부"는 틀림. **7개**가 novel junction 사용, **2개**만 chain 전체 일치 → `25_scan_panelC_prompted.py`에 `atrtd_sharing_chain`/`atrtd_carrying_novel_junction` 분리 추가

**검증**: 149/149 통과. 커밋 `abdbc6d..01df393` (42파일) 푸시.

### 다음 작업

1. 저자: Funding, ORCID 5명 추가 (원고 필드 신설)
2. `073026/index.html` 재생성 (폰트·Figure 4·Figure S4·원고 톤 전부 미반영)
3. raw_TAIR10 정체 — 이제 비의존. 초안 `email_jlomas_figS4_provenance.md` 검토

## ▶️ 이전 스냅샷 (2026-08-04 밤)

### 🟢 Figure S4를 재현 가능한 소스로 이전 — 출처 불명 파일 의존 제거

**계기**: Johny가 `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)를 기억하지 못함. 코드·데이터 가용성이 리뷰어 2명의 Major Issue인 원고에서, 인용한 파일로 재도출 못 하는 보충 그림은 방어 가치가 없다고 판단.

**해결**: 방어할 필요가 없었다. **원고가 이미 보고하는** prompted 예측(`A_thaliana_transgenic400Mprompt_beam1.gff3`, 전체 27,413 locus — Results transcript 수치의 출처)으로 같은 스캔을 돌리니

- **45개** locus 조건 충족 (chain이 AtRTD3와 정확 일치 + TAIR10과 불일치 + intron 2개↑)
- 그중 **13개**는 TAIR10에 없는 splice junction 사용, **32개**는 조합만 새로움
- **Figure 4C의 AT1G19650이 이 45개에 포함** → Figure 4와 S4가 별개 실험이 아니라 한 스캔의 두 단면

**새 Figure S4 6개**: AT4G30510, AT4G12610, AT2G37450, AT1G52500, AT4G22540, AT3G61390
→ 전부 novel feature가 **직접 보임**(junction 5 + cassette exon 1). 기존 6개는 3개가 "설명해야만 보이는" 약한 증거였음. AT4G30510은 novel junction을 AtRTD3 9개 중 7개가 사용.

**새 스크립트**: `revision/scripts/25_scan_panelC_prompted.py` (파이프라인 01–25)

### 🔴 재구성 중 검증이 잡아낸 결함 2건 — 둘 다 눈으로는 못 봤다

1. **GTF `exon`은 GFF3의 CDS+UTR**. TAIR10을 GTF에서 잘라 쓰니 highlight 판정이 전 패널에서 조용히 깨짐 → `Athaliana_167_TAIR10.gene.clean.gff3`에서 exon 행 제외하고 재추출
2. **AT4G30510 "AtRTD3 9개 전부"는 틀림**. 9는 locus의 transcript 총수, junction 사용은 **7**, chain 전체 일치는 **2**. 원고 수정 + 스캔 스크립트에 두 지표를 별도 컬럼으로 분리(`atrtd_sharing_chain` / `atrtd_carrying_novel_junction`)

**검증**: 140/140 → **149/149**, 인용 56=56, 패널 9/9. 커밋 `abdbc6d..01df393`(42파일) 푸시.

### 다음 작업

1. 저자: Funding, ORCID 5명 (원고에 ORCID 필드 없음 → 자리 추가 먼저)
2. `073026/index.html` 재생성 (폰트·Figure 4·Figure S4·원고 톤 전부 미반영)
3. (선택) `raw_TAIR10_...` 정체 — 이제 아무것도 의존하지 않음. 이메일 초안 `email_jlomas_figS4_provenance.md`

## ▶️ 이전 스냅샷 (2026-08-04 17:49)

### 🟢 raw_TAIR10_hyenaTest_prediction_noPost.gff3 정체 판명 — 테스트 split 차이

**판정**: 두 파일은 같은 모델·같은 레퍼런스·같은 prompted 모드에서 나온 것. 차이는 test split만.

- `raw_TAIR10_hyenaTest_prediction_noPost.gff3` (4,875 locus) = prompted 모드, split A
- `TAIR10_hyenaTest_prediction_noPost.gff3` (3,328 locus) = prompted 모드, split B  
- 겹치는 657 locus에서 예측 100% 동일 (같은 모델 증거)
- 657 = 4,875 × 3,328 / 24,700 (독립 추첨 기대치와 정확 부합)

**결론**: Figure S4 범례의 "raw_TAIR10_..." 파일은 전체 평가 세트의 한 test split. 다른 split 데이터도 존재 가능.

### 다음 작업

1. Johny에게 이메일: 파일 출처(test split 명시), Figure S4 범례에 쓰인 이유, 다른 split 유무 확인
2. 회신 대기

## ▶️ 이전 스냅샷 (2026-08-04 14:53)

### 🟢 원고 톤 교정 — 사전 방어(pre-emptive defense) 제거

**진단**: 같은 caveat이 반복되고 있었다.
- "단일 locus 평가라 whole-genome과 다르다" → Abstract·Results 2곳·Discussion·Figure 5 범례 **5회**
- "de novo 구조 일관성이 낮다" → Abstract·Results·Discussion·결론 **4회**

결과를 보고하는 대신 미리 변명하는 구조. 리뷰어 #20의 "soften overclaimed statements" 요청에 대응하면서 **자기 측정치까지 과교정**된 상태였다.

**원칙**: caveat을 삭제하지 않고 **제자리에 한 번만**. 수치는 손대지 않고 프레이밍만 변경.

**수정 13곳**: Abstract — 결과 우선, 마무리를 기여로. Results — caveat 1문장 + Discussion 참조. **과잉예측 0.1%를 강점으로 재진술**. 구조 일관성 — prompted 94.1%(레퍼런스 범위 안)를 먼저. Discussion/결론 — 한계 나열에서 기여 명시로. Figure 5 범례 — "inherently favors"를 사실 진술로(리뷰어 #10 요구 유지). 응답서·체크리스트 전수 검색 — 깨끗.

**검증**: 140/140, 인용 56=56, Abstract 230/250, 방어 표현 **0건**, hedge 정상 수준.

### 다음 작업

1. Johny 확인: `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)의 정체 — Figure S4 근거
2. 저자: Funding, ORCID 5명
3. `073026/index.html` 재생성 (폰트·Figure·톤 전부 미반영)

## ▶️ 이전 스냅샷 (2026-08-04 후반)

### 🟢 원고 톤 교정 — 사전 방어(pre-emptive defense) 제거

**진단**: 같은 caveat이 반복되고 있었다.
- "단일 locus 평가라 whole-genome과 다르다" → Abstract·Results 2곳·Discussion·Figure 5 범례 **5회**
- "de novo 구조 일관성이 낮다" → Abstract·Results·Discussion·결론 **4회**

결과를 보고하는 대신 미리 변명하는 구조. 리뷰어 #20의 "soften overclaimed statements" 요청에 대응하면서 **자기 결과까지 과교정**된 상태였다.

**원칙**: caveat을 삭제하지 않고 **제자리에 한 번만**. 수치는 손대지 않고 프레이밍만 변경.

**수정 13곳**:
- Abstract — caveat 삽입구 제거, AUGUSTUS 대비 3배 결과 추가, 마무리를 기여 진술로 (230단어, 제한 250)
- Results §Benchmarking — 문단이 `With this caveat,`로 시작하던 것을 결과 우선으로. caveat은 1문장 + Discussion 참조
- Results — `reflecting the difficulty`, `conservatively under-predicts`, `exposing a genuine limitation` 제거
- **over-prediction 0.1%를 강점으로 재진술** (리뷰어 #12의 과잉예측 의심에 대한 정면 반박)
- 구조 일관성 — prompted 94.1%(레퍼런스 92.9–100% 범위 안)를 먼저, de novo는 대비로
- Discussion — `may partly reflect`→사실, `We emphasize…hypotheses`→결과 우선, repeat-rich는 Z. mays 결과 먼저
- 결론 — 한계 나열로 끝나던 것을 기여 명시로
- Figure 5 범례 — `inherently favors` → 사실 진술 (리뷰어 #10이 요구한 설정 차이 명시는 **유지**)
- `submission_checklist.md` §2 Figure 4 항목 → RESOLVED로 교체 (AT1G43770/AT4G17760 결론 철회 기록)

**검증**: 140/140, 인용 56=56, Abstract 230/250, 본문 5,828단어. 방어 표현 전수 검색 0건, hedge(however/although/may) 각 1회로 정상 수준.

**저장소 변경 없음** — 원고·체크리스트는 git 밖. 패키지 `073026/` md·docx 갱신 완료.

### 다음 작업

1. **Johny 확인 필요**: `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)의 정체 — Figure S4 근거
2. 저자: Funding, ORCID 5명 (원고에 ORCID 필드 없음 → 자리 추가 먼저)
3. `073026/index.html` 재생성 (폰트·Figure 4·원고 톤 전부 미반영)

## ▶️ 이전 스냅샷 (2026-08-04 14:13)

### 🟢 원본 Figure 4 locus 확정 — 저자(J. Lomas) 제공, 데이터로 검증 완료

**받은 정보** (2026-08-04, Johny):

| 패널 | TAIR10 | Transgenic | AtRTD3 |
|---|---|---|---|
| A | AT4G10840 | `614ed78268b7` | AT4G10840 |
| B | AT3G50550 | `884f69210f1f` | AT3G50550 |
| C | AT1G19650 | `11e52f9e2918` | AT1G19650 |

**검증**: hex는 예측 GFF UUID의 **마지막 블록**. 셋 다 `fig3_original/prompted/TAIR10_hyenaTest_prediction_noPost.gff3`(07-29에 jlomas 계정에서 회수한 원본)에만 존재하고 행 수까지 1:1 일치. 원본 파일로 `scan_panels.py` 재실행 → 원 제출본 범례와 정확히 일치 (A/B는 TAIR10 2 chain 전부, C는 novel chain이면서 AtRTD3와 정확 일치).

### 🔴 함께 드러난 문제 — 07-30 5패널은 다른 inference에서 뽑혔다

원본(3,328 locus) ↔ 재구성본 교집합 657개. **AT1G43770은 원본 파일에 아예 없다**. **AT1G44575는 원본에서 chain이 2개**(재구성본에선 4개). 원고의 *"AT1G44575가 chain 4개를 전부 회수, 테스트셋에서 유일"* 주장은 거짓이었음 — Results·범례·Methods 3곳에서 삭제.

### 처리

- Figure 4 → **`figure4_example_loci_original`** (3패널). 5패널본은 `_deprecated/`로 격리 (GitHub **404** 확인)
- 두 예측 디렉토리 분리 (`panelC_examples/` vs `original/`), `source_dir()` 패널별 선택
- Figure S4 유지, 범례·Methods에 "다른 inference"임을 명시
- 검증 **123/123 → 140/140** (Figure 4 체크 17개 추가), 56=56, 9/9, 무결성 20/20
- 커밋 `abdbc6d` 푸시, 위키 `9a8f610` 푸시

### 다음 작업

1. **Johny 확인**: `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)가 무엇인지 — Figure S4 기준
2. 저자: Funding, ORCID 5명
3. `073026/index.html` 재생성 필요

## ▶️ 이전 스냅샷 (2026-08-04)

### 🟢 원본 Figure 4 locus 확정 — 저자(J. Lomas) 제공, 데이터로 검증 완료

**받은 정보** (2026-08-04, Johny):

| 패널 | TAIR10 | Transgenic | AtRTD3 |
|---|---|---|---|
| A | AT4G10840 | `614ed78268b7` | AT4G10840 |
| B | AT3G50550 | `884f69210f1f` | AT3G50550 |
| C | AT1G19650 | `11e52f9e2918` | AT1G19650 |

**검증**: hex는 예측 GFF UUID의 **마지막 블록**. 셋 다 `fig3_original/prompted/TAIR10_hyenaTest_prediction_noPost.gff3`(07-29에 jlomas 계정에서 회수한 원본)에만 존재하고 행 수까지 1:1 일치. 예) AT4G10840 = `9f6af336-6811-473b-81bc-614ed78268b7`.

원본 파일로 `scan_panels.py` 재실행 → 원 제출본 범례와 정확히 일치:
- A(AT4G10840)·B(AT3G50550): TAIR10 2 chain 전부 재현, 예측이 그 밖으로 안 나감
- C(AT1G19650): novel chain, AtRTD3 `AT1G19650.13`과 정확 일치 — **원본 평가에서 유일한 C 후보**

### 🔴 함께 드러난 문제 — 07-30 5패널은 다른 inference에서 뽑혔다

`fig4_forensics/raw_...`(4,875 locus) ↔ 원본(3,328 locus) **교집합 657뿐**.

- **AT1G43770은 원본 파일에 아예 없다** → "원본 그림의 locus"라는 07-29 포렌식 결론 반증
- **AT1G44575는 원본에서 chain이 2개** (5패널에선 4개) → 원고의 "all four … the only locus in the test set" 주장은 **원고가 보고하지 않는 파일** 기준이었음. Results·범례·Methods 3곳에서 삭제

### 처리

- Figure 4 → **`figure4_example_loci_original`** (3패널). 5패널본은 `_deprecated/`로 격리(저장소 404 확인)
- 두 예측을 디렉토리로 분리 (`panelC_examples/` vs `panelC_examples/original/`), `source_dir()`가 패널별로 선택 — **절대 섞지 말 것**
- 범례 오류 1건 추가 수정: 07-31에 `✓`→`=`로 바꿨는데 원고 범례는 "tick"이라고 쓰고 있었음
- Figure S4는 유지하되 "다른 inference"임을 범례·Methods에 명시
- 검증 **123/123 → 140/140** (Figure 4 체크 17개 추가), 인용 56=56, panelC 9/9
- 패키지 `073026/` 갱신: 그림 20개 무결성 20/20, PDF 폰트 10/10 Liberation only
- 커밋 `b6ab7cb..abdbc6d` (33파일) 푸시 완료

### 다음 작업

1. **Johny 확인 필요**: `raw_TAIR10_hyenaTest_prediction_noPost.gff3`(4,875 locus)가 무엇인지 — Figure S4가 이 파일에 근거함. 원본과도 `fig3_regen`(4,896)과도 다름
2. 저자: Funding, ORCID 5명 (원고에 ORCID 필드 자체가 없음 → 자리 추가 먼저)
3. `073026/index.html` 재생성 — 폰트 수정과 새 Figure 4 모두 미반영

## ▶️ 이전 스냅샷 (2026-08-03 18:18)

### 🟢 위키 갱신 완료 · 07-30/31 스냅샷 4개 추가 및 검증

**갱신 내용**:
- Projects transgenic_20260724.md에 Figure 4 개선(✓마크, 축 정수화, 겹침 정리), FIGURE_HANDOFF.md 작성, `073026/` 로컬 유지 결정 스냅샷 4개 추가
- index.md, guide/installs.md, log.md 병행 갱신

**검증된 사실들**:
- 폰트: Liberation Sans 2.1.5 설치, `073026/` 전체 PDF에서 `/BaseFont /LiberationSans*` 임베드만 확인
- 그림: 073026의 10개 그림(Figure 1–6, S1–S4) ↔ 저장소 원본(Figures/, revision/figures/) 바이트 동일 검증 완료
- 검증 스크립트: 19_verify_manuscript_numbers.py (수치 123/123), 22_check_citations.py 모두 통과
- Git: 커밋 "update(transgenic): blocker 0 closed..." (a1e5690..b6ab7cb) 위키에 push 완료

### 다음 작업

1. 저자: Funding 정보, ORCID 5명 입력 (v3 `[TO BE COMPLETED]`)
2. 투고 후 원고 텍스트를 저장소에 추가

## ▶️ 이전 스냅샷 (2026-07-31 00:51)

### 🟢 폰트 정책 통일 · 범례 겹침 3곳 해소

**발견**: 클러스터에 Arial/Helvetica 미설치 → 모든 그림이 무음으로 DejaVu Sans 폴백. 인쇄 후에야 드러나는 메트릭 차이 문제.

**처리**:
- Liberation Sans 설치 (Arial 메트릭 호환, 오픈 라이선스)
- `Figures/figstyle.py`: `font.sans-serif` 앞에 Arial 추가, `pdf.fonttype = 42` (Type 3 → TrueType 임베드)
- 범례 겹침: Figure 6D, 6C, Figure S2 → 축 위로 이동, `loc="best"` 제거
- `✓`(U+2713) → `=`: 양쪽 관계 대칭 표현, 글리프 폴백 방지

**검증**:
- `073026/` 패키지 그림 10개 재생성, Figure 4 검증 8/8 통과
- PDF 스캔: 모든 투고본에 Liberation만 임베드 (`/BaseFont` 확인)
- 커밋: `c65179f..b6ab7cb`

### 다음 작업

1. 저자: Funding 정보, ORCID 5명 입력 (v3 `[TO BE COMPLETED]`)
2. 투고 후 원고 텍스트를 저장소에 추가

## ▶️ 이전 스냅샷 (2026-07-30 19:23)

### 🟢 핸드오프 문서 작성 · 모든 그림 설계 규칙 기록

**작성 내용**:
- `Figures/FIGURE_HANDOFF.md` — 10개 전수(Figure 1-6, S1-S4)
- 각 그림당: 소스 스크립트, 입력 데이터 경로, 색상 팔레트, 폰트 규칙(rcParams), 저장 옵션(bbox_inches) 기록
- 재현·재설계 시 참고할 모든 상수와 설정값 한곳에 모음

**커밋**:  
`Figures: add a handoff document covering all ten manuscript figures`

**검증**:  
Git 추적됨 (Figures/ 내 정상 커밋). `revision/figures/` 는 .gitignore로 신규 파일만 무시하되, 이미 추적 중인 그림은 계속 갱신됨을 확인.

### 다음 작업

1. Funding, ORCID 5명 입력 (v3 `[TO BE COMPLETED]`) — 저자 담당
2. 투고 후 원고 텍스트를 저장소에 추가

## ▶️ 이전 스냅샷 (2026-07-30 15:59)

### 🟢 Figure 4 가독성 4가지 개선 + 범례 재작성

**완료 항목**:
- TransGenic 예측 ↔ AtRTD3 확인 **명시적 표시** (✓ 마크, 상호 참조) — 이전은 색으로만 암시
- TAIR10 라벨 충돌 해소: `.1 .2` → `t1, t2, t3, t4` / AtRTD3는 실제 ID 유지
- X축 정수 bp 표시 (소수점 제거) — 절대 좌표는 축 라벨에 이동
- 겹침 정리: 음영 최대 3개 제한, 범례 최소화, 합본 패널 제목 단순화

**검증**:
- 수치 123/123, 인용 56=56, 패널 8/8 통과
- 원고 Figure 4 범례 전체 재작성 (✓의 의미, t 표기, x축 정의 명시)
- Git 커밋: `d6e1e52..4d4b0da` (Figures/)
- 웹 페이지·`073026/` 패키지 동시 갱신 (같은 URL)

### 다음 작업

1. 저자: Funding 정보, ORCID 5명 입력 (v3 `[TO BE COMPLETED]`)
2. 투고 후 원고 저장소 추가

## ▶️ 이전 스냅샷 (2026-07-30 15:48)

### 🟢 `073026/` 로컬 유지 결정 · 의도 문서화

**결정**: `073026/`은 저장소에 올리지 않고 로컬 투고 패키지로 유지. 원고 텍스트는 투고 후 공개 예정.

**근거**: 
- 그림 10개(Figure 1–6, S1–S4) → 이미 `Figures/`·`revision/figures/`에서 공개
- 코드·평가 파이프라인(30 스크립트) → 이미 공개
- 보충 데이터·CSV·모델 체크포인트 → 이미 공개
- **원고만** 투고 후 공개 예정

**문서화**:
- `073026/README.md` 상단에 "로컬 전용, 의도적으로 저장소 미포함"과 사유 명시
- `resume.md`에 결정 근거 기록 → 나중에 헷갈림 방지

### 다음 작업

1. Funding 정보(v3 `[TO BE COMPLETED]`), ORCID 5명 입력 — 저자 담당
2. 투고 후 원고 텍스트를 저장소에 추가

## ▶️ 이전 스냅샷 (2026-07-30 15:50)

### 📌 결정: `073026/`은 저장소에 올리지 않는다 (로컬 유지)

`073026/`은 `transgenic/`보다 **상위** 디렉토리라 git이 보지 못한다. **이 상태를 유지하기로 결정.**

| 항목 | GitHub | 비고 |
|---|---|---|
| 그림 10개 | **공개됨** | `Figures/`, `revision/figures/`에 원래 이름으로 |
| 원고 md/docx | 없음 | 투고 후 공개 |
| `index.html` (웹 페이지) | 없음 | 원고 전문 포함 |

→ 리뷰어가 확인할 코드·그림·데이터는 전부 공개돼 있고, 원고 텍스트만 보류. `073026/README.md` 상단에 이 의도를 명시했다.

### 웹 페이지 (Artifact, 비공개)

https://claude.ai/code/artifact/19ce3d64-d416-4b6d-a5ef-cce45cb6fd49 — 그림 10개 + 원고 전문 + 보충자료 + Table S1–S10 미리보기. **기본 비공개**, 공유는 페이지 메뉴에서 수동 결정.

### 원고 md가 정본

`073026/manuscript_v3_plant_communications.md` = 루트 정본과 바이트 동일. docx는 pandoc 렌더 결과.

> ⚠️ **docx → md 역변환은 정본을 복원하지 못한다** (markitdown 기준 145줄 차이 — 소속 줄바꿈 병합, `viridiplantae\_odb10` 이스케이프). 편집은 `manuscript_v2.md` → `23_build_pc_manuscript.py` → pandoc 순서. markitdown 테스트 산출물 `073026/from_docx/`는 열화본이라 제거했다.

### 다음 작업

1. **Funding** (v3 `[TO BE COMPLETED]`), **ORCID 5명** — 저자만 가능. 남은 유일한 블로커
2. 투고 후: 원고를 저장소에 추가


## 📋 이전 스냅샷 (2026-07-30 15:43)

### 🟢 원고 md 정본 확정 · 역변환본 중복 제거

**상황**: 원고 4종(manuscript_v3, response_to_reviewers, cover_letter, submission_checklist)이 여러 벌씩 존재 — 루트 정본과 `073026/from_docx/` 역변환본(docx → md)이 섞여 있었음.

**발견**: 역변환본이 **열화** 상태 — 소속 줄바꿈 합쳐짐, `viridiplantae_odb10` 같은 심볼이 이스케이프됨. 루트 md는 정본으로 이미 존재하므로 역변환본 4개는 순수 중복.

**정리**: `073026/from_docx/` 디렉토리 전체 제거  
**검증**: `073026/manuscript_v3_plant_communications.md` == 루트 정본 (바이트 단위)

**072306/ 패키지 최종 구성**
- `manuscript_v3_plant_communications.md` (정본, 루트와 동일)
- `manuscript_v3_plant_communications.docx` (md에서 pandoc 렌더)
- 그림 10개 (Figure 1–6, S1–S4)
- `README.md` (md = 소스 / docx = 렌더본 명시)

### 다음 작업

1. Funding, ORCID 5명 입력 (저자 담당)
2. 위키 `projects/transgenic_*.md` 최종 갱신

## ▶️ 이전 스냅샷 (2026-07-30 15:41)

### 🟢 웹 페이지 완성·발행 + MarkItDown 0.1.5 설치

**웹 페이지**: https://claude.ai/code/artifact/19ce3d64-d416-4b6d-a5ef-cce45cb6fd49 (0.77 MB)
- 그림 10개 (Figure 1–6, S1–S4), 원고 전문, 보충자료 전문, Table S1–S10 미리보기, 저장소·모델 링크
- 그림 다운샘플(900px, 96색) 및 임베드: 2.5 MB → 0.42 MB
- 기본 비공개. 공유는 페이지 메뉴에서 사용자 결정

**MarkItDown 설치**: `/data/gpfs/assoc/pgl/tools/markitdown/` (공유 venv, uv, Python 3.11.14)
- 워드 파일 4개 변환 완료: `073026/from_docx/` (제목·강조·표·인용 모두 보존)
- **한계**: Markdown → DOCX 미지원 (pandoc 계속 사용 필요)

**위키 + CLAUDE.md**: `guide/installs.md` 추가, `index.md`·`log.md` 갱신, `~/.claude/CLAUDE.md`에 "Document conversion" 섹션 추가
- 커밋: `ec39bb4..830732a` (위키)
- gotcha: 절대경로 호출 필수, 단방향 전용, pydub ffmpeg 경고는 무해

### 다음 작업

1. Funding, ORCID 5명 입력 (원고 최종화)
2. 위키 `projects/transgenic_20260724.md` 갱신

## ▶️ 이전 스냅샷 (2026-07-30 15:31)

### 🟢 Figure 5·6·S1–S3 공개 — gitignore 누락 해결

`revision/.gitignore`가 `figures/` 제외해 원고 그림 5개(PDF/PNG 쌍 10파일)가 저장소에 없었음. `git add -f`로 추가·푸시: `dc4303a..d6e1e52`.

**검증**: 원고 그림 10개 모두 HTTP 200
- Figure 1–4, S4 (`Figures/`) — 기존 공개
- Figure 5·6, S1–S3 (`revision/figures/`) — 새 공개

생성 스크립트도 저장소에 있음.

### 다음 작업

1. Funding, ORCID 5명 입력
2. 위키(`projects/transgenic_20260724.md`) 갱신

## ▶️ 이전 스냅샷 (2026-07-30 15:40)

### 🟢 투고 패키지 `073026/` 생성 · 그림 10개 전부 공개 완료

**투고 패키지**: `/data/gpfs/assoc/pgl/data/Transgenic/073026/` — 23파일, 3.9 MB
- 메인 원고 `manuscript_v3_plant_communications.{md,docx}`
- `Figure1–6.{pdf,png}` + `FigureS1–S4.{pdf,png}` (원고 번호로 리네임)
- `README.md` — 출처·생성 스크립트·생성일·검증 기록

**Figure 번호 ↔ 산출물 매핑** (혼동 방지):
| 번호 | 산출물 | 위치 |
|---|---|---|
| Figure 3 | `figure3_performance_original` (재구성본 아님!) | `Figures/` |
| Figure 4 | `figure4_example_loci_rebuilt` (추측 locus본 아님!) | `Figures/` |
| Figure 5·6, S1–S3 | `figure5_gffcompare_f1` 등 | `revision/figures/` |
| Figure S4 | `figure4_panelC_all` | `Figures/` |

### 잡아낸 누락 — `revision/.gitignore`가 그림 절반을 가리고 있었다

`revision/figures/`가 gitignore 대상이라 **Figure 5·6·S1·S2·S3이 저장소에 없었다**. Figure 1–4와 S4는 `Figures/`에 있어 공개돼 있었으므로 그림 세트가 절반만 보이는 상태였다. `-f`로 10파일(1.2 MB) 추가 후 푸시.

**푸시**: `dc4303a..d6e1e52` (main)

**GitHub 검증 — 원고 그림 10개 전부 HTTP 200**: Figure 1·2·3·4·5·6 + S1·S2·S3·S4

### 원고 공개는 여전히 "투고 후"

`073026/`은 저장소 밖(`/data/.../Transgenic/`)이고 원고 사본을 포함하므로 커밋하지 않았다. 그림·코드·데이터만 공개된 상태로 Data availability 3문장은 충족.

### 다음 작업

1. **Funding** (v3 `[TO BE COMPLETED]`), **ORCID 5명** — 저자만 가능. 남은 유일한 블로커
2. 투고 후: 원고를 저장소에 추가
3. 위키(`projects/transgenic_20260724.md`) 갱신 미실행


## 📋 이전 스냅샷 (2026-07-30 15:28)

### 🟢 투고 패키지 완성 · 무결성 검증 완료

**최종 산출물**: `/data/gpfs/assoc/pgl/data/Transgenic/073026/` — 23파일, 3.9 MB

| 구성 | 내용 |
|---|---|
| 원고 | `manuscript_v3_plant_communications.{md,docx}` |
| 본문 그림 | Figure 1–6 (PDF/PNG) |
| 보충 그림 | FigureS1–S4 (PDF/PNG) |
| 메타 | `README.md` (출처·생성 스크립트·검증 기록) |

**파일명 리네임**: `figure3_performance_original` → `Figure3`, `figure4_example_loci_rebuilt` → `Figure4` 등 — 전부 복사본, 원본은 그대로.

**무결성 검증 완료**: 모든 복사본이 원본과 동일 확인. Figure 5·6·S1–S3은 `revision/figures/` 위치에서 발견, 누락 없이 10개 그림 전수 확인.

**그림 수정 불필요**: Figure 6C는 "prompted만" 표시되므로 de novo 수치 정정(2.9–7.7% → 0.0–0.1%)과 무관 — 정정 대상은 원고 텍스트였음.

### 📋 보충 파일 위치 대기

보충 텍스트(`supplemental_information.docx`), 표(`TableS1–S10.csv`), 커버레터, 응답서는 저장소 루트에 그대로. 패키지 폴더로 이동할지 선택 대기.

### 다음 작업

1. Funding (`[TO BE COMPLETED]`), ORCID 5명 입력
2. 보충 파일 정리 여부 결정
3. 위키(`projects/transgenic_20260724.md`) 갱신

## ▶️ 이전 스냅샷 (2026-07-30 13:19)

### 🟢 Figure 4 재구성본 푸시 완료 · 구버전 격리 완료

**푸시**: `a1e5690..dc4303a` (main), 39파일. 자격증명 스캔 clean.

**GitHub 검증** (`gh api`, ref=main):

| 경로 | 상태 |
|---|---|
| `Figures/figure4_example_loci_rebuilt.pdf` | **200** (새 Figure 4) |
| `Figures/figure4_example_loci.pdf` | **404** (구버전 격리 성공) |
| `revision/results/fig4_forensics/panelC_examples/panels.tsv` | 200 |
| `revision/results/fig4_forensics/FIG4_LOCUS_FORENSICS.md` | 200 |

**구버전 격리**: 추측 locus 3개(AT1G02630/AT1G19350/AT1G01080)로 그린 `figure4_example_loci.{pdf,png}` → `Figures/_deprecated/`(gitignore). 저장소에서 제거, 로컬 보존. `_deprecated/README.md`에 폐기 사유 기록.

**원고 결정**: 미투고 원고를 공개 GitHub에 올리면 preprint 공개가 되고 되돌리기 어렵다. 투고 후 추가하기로 결정. 현재 코드·그림·데이터 공개로 Data availability 3문장 이미 충족.

### 다음 작업

1. 위키 갱신(`projects/transgenic_20260724.md`)
2. 투고 후: 원고 파일 저장소 추가
3. Funding, ORCID 5명 — 저자 정보 완성만 남음

## ▶️ 이전 스냅샷 (2026-07-30)

### 🟢 Figure 4 재구성본 푸시 완료 · 구버전 격리 완료

**푸시**: `a1e5690..dc4303a` (main), 39파일. 자격증명 스캔 clean.

**GitHub 검증** (`gh api`, ref=main):
| 경로 | 상태 |
|---|---|
| `Figures/figure4_example_loci_rebuilt.pdf` | **200** (새 Figure 4) |
| `Figures/figure4_example_loci.pdf` | **404** (구버전 격리 성공) |
| `revision/results/.../panels.tsv` | 200 |
| `revision/results/fig4_forensics/FIG4_LOCUS_FORENSICS.md` | 200 |

**구버전 격리**: 추측 locus 3개(AT1G02630/AT1G19350/AT1G01080)로 그린 `figure4_example_loci.{pdf,png}` → `Figures/_deprecated/`(gitignore). 저장소에서 제거, 로컬 보존. `_deprecated/README.md`에 폐기 사유 기록.

### 📌 원고 공개는 "투고 후"로 결정 (2026-07-30)

원고 파일(`manuscript_v3_plant_communications.md/.docx`, 커버레터, 응답서)은 `/data/gpfs/assoc/pgl/data/Transgenic/` 루트에 있고 **이 디렉토리는 git 저장소가 아니다**. 저장소는 `transgenic/` 하위뿐.

**결정: 지금은 공개하지 않고, 투고·수락 후 저장소에 추가한다.** 미투고 원고를 공개 GitHub에 올리면 사실상 preprint 공개가 되고 git 히스토리라 되돌리기 어렵기 때문. 현재 코드·그림·데이터 공개만으로 원고의 Data availability 문장 3개는 이미 충족된다.

### 다음 작업

1. **Funding** (v3 `[TO BE COMPLETED]`), **ORCID 5명** — 저자만 가능. 남은 유일한 투고 블로커
2. 투고 후: 원고를 저장소에 추가
3. 위키(`projects/transgenic_20260724.md`) 갱신 미실행


## 📋 이전 스냅샷 (2026-07-30 13:07)

### 🟢 Figure 4 5패널 완성 · 원고 반영 · 검증 통과

**최종 산출물**: `figure4_example_loci_rebuilt.{pdf,png}` — 5패널, 동일 색 규칙·글리프·행 높이, 패널 아래 지지 수치 표기

| Panel | Locus | 유형 | 근거 |
|---|---|---|---|
| **A** | AT1G43770 | 2/2 chain 재현 | 발표 모델 유일 plot-prep 파일 |
| **B** | AT1G44575 | **4/4 chain 재현** | 테스트셋 전체에서 TAIR10 chain 전 회수 유일 locus |
| **C** | AT2G37450 | novel cassette 63 bp | TAIR10 0, AtRTD3 정확 일치 |
| **D** | AT4G22540 | novel junction | AtRTD3 5개 정확 일치 |
| **E** | AT3G56730 | novel junction | AtRTD3 1 + splice-site 공유 5개 |

**원고 v3 재생성** — Figure 4 범례·Results(AT1G44575 4/4 기술)·Methods(패널 선정 절차) 개정. **수치 123/123, 인용 56=56 통과**, 5,721단어.

**검증 완료**: `verify_panelC_figures.py` **8/8 locus 통과** — 그려진 모든 트랙 색이 원천과 일치.

**미실행**: 새 산출물(Figure 4 재구성본, `scan_panels.py`, `panels.tsv`, 원고 v3) 커밋·푸시 미실행. 구버전 `figure4_example_loci.{pdf,png}`(추측 locus 3개) 정리 미실행.

### 다음 작업

1. 새 산출물 커밋·푸시
2. 구버전 파일 `_deprecated/`로 격리
3. 위키 갱신

## ▶️ 이전 스냅샷 (2026-07-30)

### 🟢 Figure 4 전면 재구성 완료 — 5패널, 전부 검증됨

**최종 그림**: `transgenic/Figures/figure4_example_loci_rebuilt.{pdf,png}` (300 dpi, 5패널 2열)

| Panel | Locus | 유형 | 근거 |
|---|---|---|---|
| **A** | AT1G43770 | 2/2 chain 재현 | **원본 그림의 실제 locus** — 발표 모델 디렉토리의 유일한 plot-prep 파일 + 스캔 독립 확인 |
| **B** | AT1G44575 | **4/4 chain 재현** | 테스트셋 전체에서 TAIR10 chain을 전부 회수한 유일 locus |
| **C** | AT2G37450 | novel cassette exon 63 bp | TAIR10 0, AtRTD3 `.7` 정확 일치 |
| **D** | AT4G22540 | novel junction | AtRTD3 5개 정확 일치 |
| **E** | AT3G56730 | novel junction | AtRTD3 1 + site-shared 5 |

**전수 스캔** (`panelC_examples/scan_panels.py` → `panels.tsv`): 4,875 locus 분류 → Panel A 29 / **B 3** / C 10.
Panel B 3개 전부 작도 비교 → AT5G01230(음영 14개)·AT5G62890(23개)은 판독 불가, **AT1G44575만 적합**.

**원고 반영** (v3 재생성, 수치 123/123 · 인용 56=56 통과, 본문 5,721단어):
- Figure 4 범례를 5패널·locus 명시로 전면 개정
- Results에 AT1G44575 4/4 회수 기술
- Methods에 패널 선정 절차(4,875 locus 분류 기준) 기술

**자동 검증**: `verify_panelC_figures.py` **8/8 locus 통과**

### 2차 포렌식 — 이전 결론 2건 정정 (`FIG4_LOCUS_FORENSICS.md`)

1. **AT1G43770은 Panel C가 아니라 Panel A**. 이전엔 CDS+UTR 매칭으로 C에 배정했으나, CDS intron chain 기준으로 novel isoform이 **0개**(예측이 TAIR10 2 chain을 정확히 재현, 그 외 없음)
2. **AT4G17760 신뢰도 하향**. grep 세션이 폐기된 구모델(`SinusoidalDownsample_E15…`) 디렉토리였음. 발표 모델 디렉토리의 단일-locus 추출물은 `temp.gff3`(AT1G43770) **하나뿐**

**새로 판 소스 6곳 전부 음성**: IPython history(AT1G58150은 160M 데모, 더미좌표), `.viminfo`(4월 기록 소실), `.vscode-server`/`.cursor-server`(state.vscdb 없음), 계정 전역 이미지(temp.png는 attention 히트맵), Desktop/Documents/Downloads(전부 빔), `grep …> temp.gff3` 전수(1회뿐)

→ **원본 B·C locus는 복원 불가 확정**. J. Lomas 로컬 머신 외 어디에도 없음.

### 다음 작업

1. 새 산출물 커밋·푸시 (미실행) — Figure 4 재구성본 + scan_panels.py + panels.tsv + 원고 v3
2. **Funding** (v3 `[TO BE COMPLETED]`), **ORCID 5명** — 저자만 가능
3. 구버전 `figure4_example_loci.{pdf,png}`(추측 locus 3개) 폐기 처리 검토
4. 위키 갱신 미실행


## 📋 이전 스냅샷 (2026-07-29 20:19)

### 🟢 블로커 0 해소 — 리비전 자료 `main`에 푸시 완료

`isoform_token_version` → `main` **전체 머지** 후 푸시. `e3cffe0..a1e5690`, 7 커밋.

**GitHub 검증 완료** (`gh api`, ref=main):
- `revision/scripts/` — **01–24 + fig3_* = 30개** (이전 01–06)
- `examples/run_self_prompted.sh` — **HTTP 200** (이전 404)
- `revision/supplemental_tables/`, `panelC_examples/`, `figure4_panelC_strong3.pdf` — **모두 200**
- `pushed_at = 2026-07-30T03:18:00Z` (2026-04-22 → 현재)

**원고 3문장 실제 참**: "complete evaluation pipeline … publicly available", "All evaluation scripts are provided in the repository's revision directory", "summary tables … in the repository"

**충돌 2건 처리**: `.gitignore` 양쪽 합집합. `examples/prompt_mode.py` `prefetch_factor`는 main 값(2) 유지—원고와 무관한 성능 파라미터.

⚠️ **작업 중 사고·복구**: 머지 시뮬레이션에서 `git checkout _tmp_mergetest`가 미커밋 PDF로 실패했는데, `git reset --hard origin/main`이 **`isoform_token_version`에서 실행돼 브랜치를 밀었습니다.** reflog로 완전 복구, 원격 영향 없음. 이후는 worktree로 격리 수행.

### 다음 작업 (저자 확인 필요 — 코드 완료)

1. **Funding** — v3 154행 `[TO BE COMPLETED BY THE AUTHORS]`
2. **ORCID 5명**
3. **J. Lomas** — Figure 3 noPost 변종 / Figure 4 (A)(B) locus 확정

## ▶️ 이전 스냅샷 (2026-07-30 03:20)

### 🟢 블로커 0 해소 — 리비전 자료 `main`에 푸시 완료

`isoform_token_version` → `main` **전체 머지** 후 푸시. `e3cffe0..a1e5690`, 7 커밋.

**GitHub 검증 완료** (`gh api`, ref=main):
- `revision/scripts/` — **01–24 전부 + fig3_* 5개 = 30개** (이전엔 01–06만)
- `examples/run_self_prompted.sh` — **HTTP 200** (이전 404)
- `revision/supplemental_tables/` — 200
- `revision/results/fig4_forensics/panelC_examples/` — 200 (Panel C 재현 자료)
- `Figures/figure4_panelC_strong3.pdf` — 200
- `pushed_at = 2026-07-30T03:18:00Z` (이전 2026-04-22)

**원고 3문장이 이제 참**: "complete evaluation pipeline … publicly available" / "All evaluation scripts are provided in the repository's revision directory" / "summary tables … in the repository"

**충돌 2건 해결 방식**:
- `.gitignore` — 양쪽 합집합
- `examples/prompt_mode.py` — `prefetch_factor`는 **main 값(2) 유지**. isoform 브랜치는 8이었으나 원고와 무관한 성능 파라미터라 main의 최신 작업(`prompt mode refine`)을 존중

> ⚠️ **작업 중 사고와 복구**: 머지 시뮬레이션 중 `git checkout _tmp_mergetest`가 미커밋 PDF 때문에 조용히 실패했는데 후속 `git reset --hard origin/main`이 그대로 실행돼 **`isoform_token_version`이 origin/main으로 밀렸다**. reflog의 `db543fa`로 완전 복구(102파일·산출물 전부 확인), **원격에는 아무것도 나가지 않았다**. 이후 테스트는 `git worktree`로 격리해 수행. 교훈: 브랜치 전환 후 `git rev-parse --abbrev-ref HEAD`로 검증하기 전엔 파괴적 명령을 잇지 말 것.

### 다음 작업 (저자 확인 필요 — 코드 쪽은 모두 완료)

1. **Funding** — v3 154행 `[TO BE COMPLETED BY THE AUTHORS]`. PC 필수 항목
2. **ORCID 5명**
3. **J. Lomas**: Figure 3 noPost 변종 확정 / Figure 4 (A)(B) locus 확정 — 확정되면 `make_figure4_panelC.py`에 locus ID만 넣어 A·B도 재작도, Figure 4 다섯 패널 스타일 완전 통일 가능
4. 위키(`projects/transgenic_20260724.md`) 갱신 미실행



## 📋 이전 스냅샷 (2026-07-29 20:04)

### Figure 4 재구성 · Methods 보강 · Figures 정리 · 저장소 스테이징 완료

**Figure 4 최종 구성**:
- 본문 3패널: AT2G37450(C) · AT4G22540(D) · AT3G56730(E) — Figure 4 범례 (A)(B) + (C–E)로 개정
- 보충 Figure S4: 약한 3개 포함 6개 전부 + 범례 신규
- Methods: pooled 채점 명시(종별 단일 GFFCompare, F1 = 2·Sn·Pr/(Sn+Pr)) · Panel C 판정절차(4단계 우선순위) · AtRTD3 지지 카테고리 정의 추가
- 수치 123/123, 인용 56=56 유지 통과

**Figures 정리**:
- Gemini 워크플로우 이미지: README에서 실사용 중 → 삭제 대신 용량 감소(2816×1536 RGBA → 1600×873 RGB, 7.75MB → 1.83MB, 76% 감소)
- figure3_performance 재구성본 → `Figures/_deprecated/` 격리 + .gitignore 처리

**저장소 스테이징** (102파일, 커밋·푸시 없음):
- Panel C 재현 자료 강제 추가(스크립트 4개 + locus별 GFF3/GTF 18개, 901KB, `-f` 옵션)
- 스크립트 절대경로 → 저장소 상대경로 변경 + 검증 6/6 통과
- .bak 파일 제외, 자격증명 스캔 clean

### 다음 작업

1. 브랜치 전략 결정 후 커밋·푸시 (`isoform_token_version` → `main` 머지/cherry-pick/태그 전략 확정)
2. **J. Lomas**: Figure 4 (A)(B) locus 확정 — 확정 시 같은 스크립트로 A·B 재작도하면 Figure 4 전체 스타일 통일
3. Funding · ORCID 5명 · 위키 갱신

## ▶️ 이전 스냅샷 (2026-07-29 21:05)

### Methods 보강 · Figures 정리 · 저장소 스테이징 완료

**Figure 4 재구성 (강한 3개 본문 + 6개 보충)**:
- 본문 `figure4_panelC_strong3.{pdf,png}` — (C) AT2G37450 카세트 엑손 / (D) AT4G22540 / (E) AT3G56730 junction. 패널 라벨 포함
- 보충 `figure4_panelC_all.{pdf,png}` — 6개 전부 (약한 3개 포함)
- Figure 4 범례를 (A)(B) + (C–E)로 개정, Figure S4 범례 신규

**Methods 보강** (`manuscript_v2.md` → v3 재생성):
- pooled 채점 명시: 종별 단일 GFFCompare 실행, F1 = 2·Sn·Pr/(Sn+Pr), per-gene 평균 아님
- Panel C 스캔·판정 절차 기술 (4단계 우선순위) + AtRTD3 지지 카테고리 정의
- Results에 AT2G37450 63 bp 카세트 엑손 구체 기술
- **검증 재통과**: 수치 123/123, 인용 56=56. 본문 5,598단어 (기존 5,202)

> ℹ️ **정정**: resume에 "Methods 2문장 추가 필요"로 남아 있던 base-level F1 정의는 **이미 반영돼 있었다**(114행: harmonic mean / raw noPost / test-set gene models). pooled 여부만 추가했다. d_model 768/1536도 98행에 이미 있었다.

**Figures 정리**:
- `Figures/_deprecated/` 신설(gitignore) — 재구성본 `figure3_performance.*` 격리. 투고본은 `_original`
- `Gemini_Generated_Image_*.png`: **README 워크플로우 이미지로 실사용 중이라 삭제 불가** → 2816×1536 RGBA → 1600×873 RGB, **7.75 MB → 1.83 MB (76% 감소)**. 원본은 `_deprecated/`에 백업

**저장소 스테이징 (커밋·푸시 안 함)**:
- `./publish_revision_to_github.sh` 실행 → **102 파일**, 자격증명 스캔 clean
- Methods가 기술한 Panel C 재현 자료를 `-f`로 추가 (revision/results/가 gitignore 대상이라 빠져 있었음): 스크립트 4개 + locus별 GFF3/GTF 18개, 901 KB
- 스크립트 절대경로 → 저장소 상대경로로 변경 (fresh clone에서 실행 가능하게)
- `.bak-*` 파일 스테이징 제외

### 다음 작업 (사용자·저자 결정 필요)

1. **브랜치 전략 결정 후 커밋·푸시** — `isoform_token_version` → `main` 머지 / cherry-pick / 태그 릴리스. 푸시해야 원고 3문장이 참이 됨
2. **Funding** (v3 154행 `[TO BE COMPLETED]`), **ORCID 5명**
3. **J. Lomas**: Figure 3 noPost 변종 확정 / Figure 4 (A)(B) locus 확정 — 확정되면 같은 스크립트로 A·B도 재작도해 Figure 4 전체 스타일 통일 가능
4. 위키(`projects/transgenic_20260724.md`) 갱신 미실행

## 📋 이전 스냅샷 (2026-07-29 19:52)

### 패널 C 6개 산출물 최종 확정

요청: 통일된 규칙으로 생성된 6개 패널 결과 확인  
결과: **산출물 최종 확정. 검증 완료 (6/6 통과).**

**확정된 산출물**:
- `figure4_panelC_all.{pdf,png}` (300 dpi, 합본 및 개별 6장)
- AT2G37450·AT4G22540·AT3G56730·AT3G13740·AT1G78940·AT3G29185 전사체 트랙
- verify_panelC_figures.py로 색상↔데이터 주장 일치 확인 완료

**문서 갱신**:
- README.md: 6개 locus 유형·근거도 명기 (novel cassette exon / novel junction / retained intron 분류)
- 스크립트: 기존 2개 예제 → 영구 6개 locus 데이터 복사 완료

**다음 작업**:
- Panel C 최종 구성 결정 (단독 AT2G37450 vs 원본 AT1G43770과 병렬)
- 원고 Methods에 Panel C 정의·검증 방식 기술
- 보충자료에 합본 시트 수록 검토

## ▶️ 이전 스냅샷 (2026-07-29 20:10)

### Panel C 6개 예제 전체 재작도 + 자동 검증 도입

요청: 6개 예제를 전부 다시 그려 통일성 확보  
결과: **6개 전부 단일 규칙으로 재작도. 렌더링 버그 1건·데이터 주장 오류 1건 발견·수정.**

**통일 규칙** (`transgenic/Figures/make_figure4_panelC.py`, 백업 `.bak-20260729`):
- 트랙 순서·색 고정: TAIR10(회색) → TransGenic novel(주황)/재현(파랑) → AtRTD3 정확일치(진초록)/feature 지지(초록)/기타(연초록)
- 진초록은 **CDS chain 전체 일치 전사체 전용** (feature만 공유하면 초록)
- 행 높이·그림 폭 고정, 전사체 ID를 축 우측에 표기, x축 kb·offset 표기 제거
- AtRTD3 선정 결정적(카테고리별 번호순), **동일 전사체 중복 배치 방지**

**하이라이트 유형 5종으로 분류** (약한 근거를 강한 것처럼 그리지 않기 위함):

| Locus | 유형 | 근거 |
|---|---|---|
| **AT2G37450** | novel cassette exon | **강함** — 권장 |
| **AT4G22540** | novel junction | **강함** — AtRTD3 5개 정확일치 |
| **AT3G56730** | novel junction | **강함** |
| AT3G13740 | UTR 내 unspliced intron | 약함 — CDS 조기 종결 |
| AT1G78940 / AT3G29185 | chain 조합만 새로움 | 약함 |

**잡아낸 오류 2건**:
1. **렌더링 버그** — `parse()`가 mRNA 행의 `Parent=<gene>`까지 전사체로 잡아 **locus마다 빈 트랙 1개 삽입**. 기존 그림들의 불규칙한 여백 원인
2. **데이터 주장 오류** — AT3G29185를 "retained intron, TAIR10 0개"로 그릴 뻔했으나, 해당 인트론을 **TAIR10 `.2`가 이미 unspliced로 보유**. 판정에 "TAIR10 어느 isoform도 unspliced로 갖지 않을 것" 조건 추가 → combination으로 강등

**자동 검증 도입**: `panelC_examples/verify_panelC_figures.py` — 그려진 각 트랙이 색이 주장하는 바와 일치하는지 원천 파일로 재확인, 불일치 시 non-zero exit. **현재 6/6 통과**

**산출물**:
- 개별 6장 + 합본 `figure4_panelC_all.{pdf,png}` (300 dpi, `transgenic/Figures/`)
- 6개 locus 데이터 전부 `panelC_examples/`로 영구 복사 (기존 2개 → 6개)
- `panelC_examples/README.md` 전면 개정 (유형·등급·지지 카테고리 정의)
- `panelC_examples/audit_panelC.py` 신규

### 다음 작업

- Panel C 최종 확정: AT2G37450 단독 vs AT1G43770(원본 locus)과 병렬 구성
- 원고 Methods에 Panel C 정의·검증 방식 기술
- 보충자료에 합본 시트(`figure4_panelC_all`) 수록 검토

## ▶️ 이전 스냅샷 (2026-07-29 19:19)

### AT2G37450 조건 검증 및 근거 정정

요청: TAIR10 부재 · TransGenic 예측 · AtRTD3 지원 조건 충족 확인  
결과: **그림 완성했고, 조건 모두 충족. 근거 수치 1건 정정 완료.**

**검증**:
- TAIR10: isoform 2개 모두 Chr2:15,724,032–15,724,094 지점을 인트론으로 통과 → **엑손 없음** ✅
- TransGenic: `…a7c.t2` CDS6 = 정확히 15,724,032–15,724,094 → **카세트 엑손 예측** ✅
- AtRTD3: **동일 엑손 보유는 `AT2G37450.7` 1개뿐** (이전: "8개" 과장)
  - 나머지 7개(.1 .3 .4 .5 .8 .10 .11)는 이 구간을 더 긴 엑손으로 덮음
  - 이전 문구에서 단순 좌표 overlap 카운트를 카세트 엑손 개수로 오독

**정정**:
- `README.md`: 11개 전사체별 카세트 엑손 상태 분해표 추가
- `resume.md`: "8개→1개" 정정 노트 기록
- 그림(`figure4_panelC_AT2G37450.png`): 수정 불필요 (전사체 수 표기 없음, 5개 트랙 정확)

**미완성**: AT3G56730 그림 1개 미생성 (표는 6개, 그림은 5개)

### 다음 작업

- AtRTD3 추가 트랙 포함 검토: `.3`/`.8` 같은 스플라이스 부위 공유 전사체로 반복 관측 강조 여부
- 원고 Methods에 Panel C 검증 방식 기술
- 필요시 AT3G56730 그림 재생성

## ▶️ 이전 스냅샷 (2026-07-29 17:35)

### Figure 4 Panel C 최종 예제: AT2G37450 선정 및 그림 완성

요청: Panel C에 넣을 예제 1개를 결정해서 그리기  
결정: **AT2G37450** — 카세트 엑손 (가장 명확한 TAIR10 부재)

**선택 근거**:
- 6개 후보의 splice 이벤트를 분류한 결과, AT2G37450만 **진정한 카세트 엑손** (novel exon)
- 나머지는 junction 경계 미세 이동 → 그림으로 구별 어려움
- 엑손 검증: Chr2:15,724,032–15,724,094 (63 bp)
  - TAIR10: 2개 isoform 모두 이 자리 **인트론으로 건너뜀** (엑손 0개)
  - TransGenic novel: 이 자리에 엑손 예측 (예측 t2의 CDS6)
  - AtRTD3: **동일 카세트 엑손 보유는 AT2G37450.7 1개** (11개 중). 5-인트론 chain도 .7과 CDS·엑손 양쪽 정확 일치. 스플라이스 부위별로는 acceptor 3개(.3/.7/.8)·donor 2개(.5/.7) 독립 지지

> ⚠️ **정정(2026-07-29 재검증)**: 앞서 "AtRTD3 8개 전사체가 같은 엑손 보유"라 적었으나 **8은 단순 좌표 overlap 카운트**였다. 실제로는 동일 엑손 1개(.7), 더 긴 엑손으로 덮는 것 7개(.1 .3 .4 .5 .8 .10 .11 — retained intron/다른 5′ 말단이라 카세트 엑손 아님), 인트론 3개(.2 .6 .9). **그림에는 전사체 수 표기가 없어 투고 그림은 수정 불필요**; README와 이 노트의 텍스트만 정정했다.

**산출물**:
- 최종 그림: `transgenic/Figures/figure4_panelC_AT2G37450.{pdf,png}` (300 dpi)
- 차점(AT4G22540) + AT1G78940 · AT3G13740 · AT3G29185 그림 생성 (총 **5개**). 표의 6번째 **AT3G56730은 그림 미생성** — 필요하면 `find_panelC.py`로 데이터 재추출 후 생성
- 스크립트: `make_figure4_panelC.py`
- 선택 이유 및 근거: `transgenic/revision/results/fig4_forensics/panelC_examples/README.md`

### 다음 작업

- AT2G37450를 Panel C 최종 사용할 것 vs AT1G43770(원본 locus)과 다중 패널 구성할 것 중 선택
- 원고 Methods 반영: Panel C 정의 및 "TAIR10 부재, AtRTD3 확인" 검증 방식 기술
- 보충자료에 모든 6개 예제 그림 + 선택 기준 추가 기록

## ▶️ 이전 스냅샷 (2026-07-29 17:30)

### Panel C 예제 6개 발굴 및 검증

요청: TAIR10엔 없지만 TransGenic 예측·AtRTD3 지원 예제  
결과: jlomas 원본 prompted 전체 스캔 → **Panel-C 타입 6개** 발굴

**선택 예제: AT4G22540** (Chr4, −strand, 9 intron)
- TransGenic: primary(파랑) + novel(주황) 2개 예측
- novel의 CDS intron chain이 **AtRTD3 5개 전사체**와 정확히 일치 (read 반복 관측)
- **TAIR10 4개 전사체 어디에도 없음** — 구별 junction **11,864,127–11,865,712** 음영 강조
- AT1G78940, AT3G13740 백업 예제도 그림 생성(더 간결하지만 차이 미묘)

**검증**: 모든 intron chain 3-way 비교·저장, 구별 junction 계산 완료

**위치**:
- 그림: `transgenic/Figures/figure4_panelC_{AT4G22540,AT1G78940,AT3G13740}.{pdf,png}` (300 dpi)
- 스크립트: `transgenic/Figures/make_figure4_panelC.py`
- 원천 데이터·검증: `transgenic/revision/results/fig4_forensics/panelC_examples/`

### 다음 작업

- 최종 Figure 4 Panel C로 어느 예제를 쓸지 결정 (AT4G22540 권장, 또는 여러 패널 병렬 표시)
- 원고 Methods에 Panel C 정의 및 TAIR10과 겹치는 vs 신규 예제 구분 방식 기술
- 원본 Figure 4 데이터(locus AT1G43770)와 새 Panel-C 예제 중 최종 선택 확정

## ▶️ 이전 스냅샷 (2026-07-29 11:00)

**목표 불변**: `transgenic/` + `transgenic_comparison/` 분석으로 원고를 완성해 *Plant Communications*에 투고.

**한 줄**: 투고 패키지(원고·보충자료·응답서·커버레터, .md + .docx)는 완성했고 수치 123/123·인용 56=56으로 검증했다. 그 과정에서 원고 오류 4건과 데이터 결함 1건을 잡았다. **Figure 3은 J. Lomas 계정(jlomas@pgl-gpu)에서 원본 산출물을 찾아 발표 그림을 그대로 재생성했고(앵커 92.2/71.2 정확 재현), 앞서 "데이터와 모순"이라던 내 보고는 재구성 오염 탓 오류로 철회한다.** 남은 최대 블로커는 **공개 저장소에 리비전이 하나도 올라가 있지 않다**는 것.

---

## 🔴 블로커 0 — 공개 저장소에 리비전이 없다 (담당: W.C. Yim / J. Lomas)

**리뷰어가 30초 만에 반증할 수 있는 유일한 항목이고, 3명 중 2명이 Major Issue로 지적했다.**

GitHub API 확인 (2026-07-29):
- `github.com/wyim-pgl/transgenic` 공개 맞음, HF 체크포인트 2개도 공개 맞음
- 그러나 **`pushed_at = 2026-04-22`** — 리비전 작업 시작 3개월 전
- `main`의 `revision/scripts/`에는 **01–06번만**. 07–23번과 `fig3_*` 전부 없음
- `examples/run_self_prompted.sh`는 **404** (issue1은 "MET"으로 기록되어 있었음 → 정정함)
- 로컬 `isoform_token_version`은 origin과 동일(0/0), **`origin/main` 대비 5 ahead / 6 behind**, 리비전 자료는 미커밋(untracked ~40, modified 10)

원고의 문장 3개가 현재 거짓:
| 문장 | 위치 |
|---|---|
| "the complete evaluation pipeline … publicly available" | Data and code availability |
| "All evaluation scripts are provided in the repository's revision directory" | Methods |
| "summary tables … as machine-readable CSV files in the repository" | Data availability (`.gitignore`가 `revision/results/` 제외) |

**준비해 둔 것**: `./publish_revision_to_github.sh` — 57파일 스테이징 + 자격증명 스캔, **커밋·푸시 안 함**. `--reset`으로 원복.
**먼저 정할 것**: 브랜치 전략 (`isoform_token_version` → `main` 머지 / cherry-pick / 태그 릴리스로 URL 변경).

> ⚠️ `revision/scripts/fig3_evaluate.sh`에 **pgl-gpu 평문 비밀번호**가 있었다. 환경변수 방식으로 교체 완료. `git log --all -S`로 **커밋된 적 없음** 확인(히스토리 재작성 불필요). 공유 계정이면 비밀번호는 교체 권장.

---

## ✅ Figure 3 — 원본 산출물로 해결 (2026-07-29)

**J. Lomas 계정(`jlomas@pgl-gpu`, pw [redacted 2026-09-01])에 발표 그림의 원본 산출물이 전부 보존되어 있었다.** `/home/pgl/scratch1/jlomas/transgenic/TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_768L12_E22-*`:
- 종별 gffcompare `.stats` (de novo post/noPost) → 패널 A
- TAIR10 length/CDS bin `.stats` → 패널 C/D
- completion(prompted) 예측 GFF3 → 패널 B (GPFS에서 채점만)

GPFS로 복사: `transgenic/revision/results/fig3_original/`. 재생성: `transgenic/Figures/make_figure3_original.py` → `figure3_performance_original.{pdf,png}` (**투고본은 이것**).

**앵커 정의 확정**: 발표 "base-level F1"은 **noPost 예측을 test 라벨과 비교한 pooled gffcompare F1** (2·Sn·Pr/(Sn+Pr)). A. thaliana 90.6/93.8 → **92.2**(발표 92.2), Z. mays 63.8/80.5 → **71.2**(발표 71.1). 미상 아니었음.

**길이 주장 검증됨**: 원본 TAIR10 bin ρ = 길이 **−0.109**(데이터 많은 구간 평탄), CDS **−0.880**(강한 단조 하락). 원고 서술이 정확. **본문 수정 불필요.**

> ⚠️ **내 어제 보고 철회**: "데이터와 모순(ρ=−0.332)"은 틀렸다. 원인 3가지가 모두 같은 방향으로 작용 — (1) `-rc` 오염(24,709행 좌표 미매핑 F1=0, 긴 유전자 집중, 위키 리뷰가 독립 발견), (2) per-gene 집계(원본은 bin별 gffcompare), (3) 재구성 split. 원본 데이터가 셋 다 무효화.

**남은 것(작음)**: J. Lomas가 noPost(post 아님) 변종 확인 + Methods에 base-level F1 정의·noPost/test-label 설정 2문장.

**패널 B GPU chain 중단**: prompted 원본이 보존돼 있어 12–18h GPU 재실행 불필요 → kill 완료. de novo 재구성(pid 2494590)은 독립 replication으로만 완주 중(~99%).

---

## 🟡 (구) 블로커 1 — Figure 3 재구성 경위 (참고용, 그림 소스로는 폐기)

### 살려낸 경위

7/27 13:01 시작된 `fig3_infer.py`가 **17:53에 크래시**했고 27시간 동안 아무도 몰랐다.

```
File "fig3_infer.py", line 125, in write_pred
    out_fh[sp].write(...)          KeyError: 'unknown'
```

`species_of()`가 분류 못한 유전자에 `"unknown"`을 반환하는데 `out_fh`에 그 키가 없다 → **유전자 하나가 전체 실행을 죽였다**. DB 조회 결과 해당 유전자는 **정확히 361개**:

| 패턴 | 개수 | 종 | 이유 |
|---|---|---|---|
| `ChrSy.*` | 89 | O. sativa | MSU v7 synteny pseudo-molecule |
| `ChrUn.*` | 96 | O. sativa | MSU v7 미배치 |
| `GRMZM*` | 176 | Z. mays | RefGen_V4 레거시 v3 ID |

셋 다 평가용 참조 주석에 실제로 존재(O. sativa 2,306행, maize 865행)하므로 버리면 안 된다. 크래시 지점 rn 355,002 = `ChrSy.fgenesh.gene.1.MSUv7.0` = V. vinifera 직후 O. sativa 첫 행.

**수정**: 3접두사 매핑 + `None` 반환 + 미분류는 `_unassigned` 파일로 보내고 **중단하지 않음**. 재개 후 `_unassigned`는 0바이트, 경고 0건 → 361개가 전부 정확히 분류됨.

**2차 장애**: HF 401(`~/.cache/huggingface/token` 만료 + 400M이 `pytorch_model.bin`이라 없는 safetensors를 허브 조회). `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`로 우회. **토큰 갱신은 별도로 필요.**

### 진행

> 이 재구성은 **그림 소스가 아니다**(위 ✅ 섹션의 원본 데이터가 대체). pid 2494590이 ~99%로 완주 중이나 독립 replication 목적일 뿐이고, `-rc` 좌표 버그(flip_rc가 window start `st` 미전달)로 ~22.9% 행이 F1=0이라 원본과 직접 비교하려면 그 행을 제외/재플립해야 함(GPU 불필요).

> 재구성 과정에서 발견·수정한 평가 버그(그림 소스로는 무의미하나 재구성 재실행 시 필요): staged `fig3_evaluate.sh`가 test split 예측을 **전체 유전체 주석**과 비교(A. thaliana base Sn 14.0 vs Pr 92.5) → `24_restrict_reference_to_test_genes.py`로 참조 제한(10종 ID 매칭 100%). `fig3_infer_prompted.py`도 동일 `KeyError` 버그가 있어 패치해 뒀으나, 원본 prompted가 있으므로 재실행 불필요.

---

## 🟡 블로커 2–3 (저자 정보)

- **Figure 4 원본 locus — 역연산 성공(2026-07-29)**: jlomas `.bash_history`에서 실제 후보 복원. **AT1G43770 확정**(유일하게 AGAT-clean 플롯 준비 파일 `temp.clean.gff3` 존재, 2025-04-13, prompted 디렉토리; TAIR10 2 / AtRTD3 5 / 예측 3). 나머지 2개는 AT4G17760·AT1G53690·AT1G05770 중. **재선정본 AT1G02630/AT1G19350/AT1G01080은 히스토리에 전무 → 오류**. 상세: `transgenic/revision/results/fig4_forensics/FIG4_LOCUS_FORENSICS.md`. J. Lomas에 좁은 질문으로 확인
- **Funding** (원 투고본에 없었음), **저자기여**(초안 있음), **Acknowledgments**(초안 있음), **ORCID 5명**

---

## ✅ 완료된 투고 패키지

| 파일 | 내용 |
|---|---|
| `manuscript_v3_plant_communications.md/.docx` | 초록 **230단어**(한도 250) ✏️ PARTIAL (2026-09-01): 8-17 편집으로 395단어까지 불어났다가 9-01에 237단어로 재축약, Key words, PC 후미 순서, Cell Press 참고문헌, 본문 5,202단어, Figure 1–6 범례 포함 |
| `supplementary/supplemental_information.md/.docx` + `TableS*.csv` ×12 | Figure S1–S3 범례 + **Table S1–S10 실데이터**. 기존 `supplementary_tables.md`는 한국어 매핑 메모였을 뿐 표가 아니었음 |
| `references_plant_communications.md` | 56건, CrossRef 48건 자동 + 7건 수동 검증. 저자 6명 절단을 실제 10명으로 복원(임의 생성 없음, 2건 PubMed 재확인) |
| `response_to_reviewers.md/.docx` | 3명 point-by-point |
| `cover_letter.md/.docx` | 초안 — 편집자명·원고번호 필요 |
| `submission_checklist.md/.docx` | 블로커 + 담당자 |
| `FIG3_FINDINGS.docx` | Figure 3 상세 |

**검증**: 수치 **123/123**, 인용 **56 = 56**(누락 0/미인용 0), 응답서 수치 전부 원고에 존재.

### 잡아낸 오류

1. **de novo splice-event recall "2.9–7.7%" → 실제 0.0–0.1%** (2자릿수). de novo는 locus당 ~1전사체라 AS 이벤트를 342개만 예측, 14,643개 중 7개 일치. 정정하니 prompting 효과가 "2배"가 아니라 **2자릿수 개선**으로 드러남
2. **max CDS "146 (Z. mays), 한도 내" → 실제 201** (S. lycopersicum, 유일한 CDS 한도 초과)
3. **AUGUSTUS "2.7/locus"의 분모 오해 유발** → 69,049/**25,641** loci 명시. 부수적으로 AUGUSTUS가 locus의 6.5%에서 아무것도 예측 못 한다는 사실이 드러남
4. **문헌 인용**: AtRTD3 "over 169,000 / 1,500 novel genes" → 원문 **169,503 / 1,541 (vs Araport)**. 분석에 쓴 `_transfix` GTF는 **169,499**(핵 168,951 + 세포기관 548)로 4개 차이 → Methods에 명시
5. **데이터 결함**: `busco_summary_final.csv` 128행 중 **37행**에서 종/도구 컬럼 오분할(마지막 언더스코어 기준). BUSCO 수치는 불변(라벨링만). Figure 5/S1은 주요 7개 도구만 읽어 영향 없음 확인

---

## 🧰 파일·명령 지도

```bash
cd /data/gpfs/assoc/pgl/data/Transgenic

# ⚠️ CAUTION (2026-09-01): 아래 규칙은 무효 — v3가 정본, script 23 실행 금지(상단 스냅샷 참조)
# (구) 원고는 v2를 고치고 v3를 재생성 (v3 직접 수정 금지)
python3 transgenic/revision/scripts/18_normalize_busco_summary.py
python3 transgenic/revision/scripts/19_verify_manuscript_numbers.py   # 123/123
python3 transgenic/revision/scripts/20_build_supplementary_tables.py
python3 transgenic/revision/scripts/21_restyle_references.py          # 네트워크 필요
python3 transgenic/revision/scripts/23_build_pc_manuscript.py
python3 transgenic/revision/scripts/22_check_citations.py \
    --manuscript manuscript_v3_plant_communications.md                # v2 아님!

# .docx 재생성
PANDOC=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASeq_postanalysis/bin/pandoc

# Figure 3 (추론 완주 후)
python3 transgenic/revision/scripts/24_restrict_reference_to_test_genes.py
export PGLGPU_HOST=… PGLGPU_USER=… PGLGPU_PASS=…
export SSH_PY=/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python
bash transgenic/revision/scripts/fig3_evaluate.sh --force-pull   # 앵커 자동 대조 출력
python3 transgenic/revision/scripts/fig3_per_gene_metrics.py
python3 transgenic/Figures/make_figure3.py
```

- **pgl-gpu 접속**: `.venv-ssh/bin/python` (paramiko). 원래 `fig3_evaluate.sh`가 쓰던 `/tmp/ssh_env`는 소멸했었음
- **투고 금지**: `manuscript_v2.docx`, `supplementary_tables.docx` (둘 다 7/27, 정정 이전)
- **이슈 추적**: `issues/issue1`~`issue6` — issue1·issue6의 "acceptance criteria MET"은 로컬 기준이었어서 정정함

---

## 다음 작업

1. **저장소 공개** (블로커 0) — 브랜치 결정(`isoform_token_version`는 origin과 동일, `origin/main` 대비 5 ahead/6 behind) → `publish_revision_to_github.sh` → 커밋 → 푸시 → `main/revision/scripts`에 07–24·fig3_* 보이는지 확인
2. **J. Lomas 확인**: Figure 3 noPost 변종 확정 / Figure 4 원본 3 locus
3. **Methods 2문장 추가**: base-level F1 = pooled gffcompare 2·Sn·Pr/(Sn+Pr), Figure 3A는 noPost를 test 라벨과 비교. + config.json의 `d_model` 768은 encoder 폭·decoder는 2배(1,536) 한 문장
4. **투고본 그림 교체**: `figure3_performance_original.{pdf,png}`를 Figure 3로 (재구성본 `figure3_performance.*` 폐기)
5. **저자 정보**: Funding, ORCID
6. 비필수: Helixer·Tiberius 정확 버전, Singularity digest, `Figures/`의 7.4 MB Gemini 플레이스홀더 정리

