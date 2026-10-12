# 재시작 인계: 2026-10-04 B6 훈련 입력 완성 → WP5 파일럿 큐 대기

📌 **이 문서가 현재 정본이다.** 이전 인계 [RESTART_HANDOFF_20260930](RESTART_HANDOFF_20260930.md)는 역사 기록이다. 세부 시간순 로그는 [PROGRESS.md](b6_production_start_20260925/PROGRESS.md) 최상단(10-01~10-04 항목).

## 1. 한 줄 상태 (2026-10-04 08:00 PDT 실측)

**pronghorn에서 B6 훈련 입력을 모두 만들고 검증했고, DeltaAI에서도 같은 검증을 통과했다. WP5 파일럿(사용자 승인 10-03)은 DeltaAI ghx4 큐에서 대기 중이다.** 코드 HEAD `9f003e1`(b6-integration, main 미병합·미push).

## 2. 실행 중·대기 중인 잡

| 위치 | 잡 | 내용 | 상태 |
|---|---|---|---|
| DeltaAI | 3306594 → 3306595 | WP5 파일럿 B6(증거 모델), link1 → link2(afterany) | Priority 대기, 예상 10-05 12:02 CDT |
| DeltaAI | 3306596 → 3306597 | WP5 파일럿 C2(서열만 대조), link1 → link2 | 같음 |
| DeltaAI | 3149363 | B5 seed 123 체인 다음 링크 | Priority 대기, 예상 10-04 19:52 CDT |
| pronghorn | 없음 | — | — |

각 파일럿 링크: 4 GPU, 24h, `--stop-after-updates 4099`, 링크마다 `verify_inputs` 재실행(약 10분). 완료 표시는 `runs/b6_wp5_pilot_<ARM>/TRAINING_DONE`과 `step_04099/`(PILOT_FINAL, PRUNE_EXEMPT).

## 3. 산출물 (pronghorn `$PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925`)

| 산출물 | 경로 | 핵심 값 |
|---|---|---|
| C0 28개 | `c0/<dir>/out/` | 전부 QC PASS, `c0v.py` sha 일치 (Athaliana ONT 225GB `4f556eec`, Gmax protein 88.7GB `a1df7a87`) |
| zero list 27개 | `zero/<Sp>/` | Gmax 1.15 / 1.36 / 1.53% |
| 병합 증거 DB | `merged/b6_v1/evidence_c0.db` | 786,283,638,784 B, sha `22af410f…f98e`, manifest 28 상대경로, PK 미선언 기록 |
| 코퍼스 재검증 | `corpus/validate_c0.json` | PASS, c0_molecules_consistent 37,850,010, counts = noc0 |
| obs cache | `obs_cache/train_v1` | 455 shard, 653,922,028행, 7.9GB, manifest `11b4c757…` (`RESTORE_NOTE_20261003.txt` 참고) |
| bindings·검증 레코드 | `bindings/train_v1.pronghorn.json`, `…verify_record.json` | record_sha256 `56d4849e…`, 레코드 파일 sha `a977d04e…`, 종 exhaustiveness 전부 ≥0.9995 |
| 소스 스냅숏 | `source_v15`(31ddead) · `source_v17`(f06a786) · `source_v18`(9f003e1) | 운영 obs cache는 v17, 최신은 v18 |

실패·대체 산출물(지우지 않음): `obs_cache/train_v1.oom_6185476`, `.oom_6186344`, 스필 `b6_build_obs_cache.spill-w66q0qcy.oom_6186344`.

## 4. DeltaAI 배치 (`/work/nvme/bilv/wcyim/transgenic` = 컨테이너 `/work`)

- `b6_prep_20260925/` (nvme): obs_cache·corpus·freeze·intervals·zero·c0 사이드카·bindings·source_v18. obs cache 511파일 sha 전수 일치.
- `/work/hdd/bilv/wcyim/transgenic/b6_prep_20260925/` (HDD, 컨테이너 `/hdd_prep` 읽기 전용): `merged/b6_v1/evidence_c0.db` + c0 사이드카.
- `repo_b6_9f003e1/`: 9f003e1 archive 스냅숏 + 로컬 커밋 2f822e1 + HFmodels. (기존 `repo_b6`는 2656dab 스냅숏, `repo_dev`는 B5 — 건드리지 않음.)
- bindings `b6_prep_20260925/bindings/train_v1.deltaai.json` (컨테이너 경로). 검증 3306593 PASS, 레코드 pronghorn과 바이트 동일.
- 제출 스크립트 `b6_prep_20260925/dtai_submit_pilot.sh`: `APPTAINER_BIND`로 HDD를 `/hdd_prep:ro`, nvme intervals를 freeze가 기록한 pronghorn 절대 경로에 `:ro` 바인드.
- 저장소: nvme 약 322/500G(파일럿 체크포인트 팔당 30–40GB), HDD 약 1,1xx/1,000G 소프트(병합 DB 786G; 09-25 보관본은 pronghorn `archives/deltaai_hdd_b6_nvme_20260925`로 이동·HDD 삭제, MATCH 422).
- GPU 잔액 848 h / 1,499 (10-04). ACCESS 보충 3,800 GPU-h 신청 승인, 잔액 반영 대기. 문안 `notes/access_supplement_AGR260032/`.

## 5. 이번 세션 사용자 결정 (10-01~10-04)

1. "속도를 내줘" → 후속 단계는 afterok 체인으로 선제출.
2. DeltaAI nvme `runs/` 정리: `ddp_int`·`seed123_1gpu_partial` 166G 보관(pronghorn `archives/deltaai_nvme_runs_20261001`, MATCH 266) 후 삭제. `runs/seed123`은 **B5 체인 종료까지 유지**.
3. 훈련 입력은 nvme, 병합 DB는 HDD(보관본 147G를 pronghorn으로 옮겨 공간 확보).
4. pronghorn은 `cpu-s1-pgl-0` 또는 `cpu-s2-core-0`(계정 `cpu-s2-pgl-0`)만. **`cpu-s3-sponsored-0` 절대 금지.**
5. **WP5 파일럿 제출 승인**(10-03).
6. ACCESS 보충 신청(AGR260032) — 신청량 3,800 GPU-h(B5 seed 456·789 포함), 승인됨.

## 6. 코드 변경 (모두 Codex 리뷰, 결과 불변 증명)

| 커밋 | 내용 | 증명 |
|---|---|---|
| `cab83cb`·`175c008`·`31ddead` | `b6_validate_corpus.py`: 분자 감사를 SQL JOIN 스트리밍, `--memory-limit/--threads/--temp-dir`, DuckDB 오류를 FAILED 1줄로 | repair 시험 33→38 passed, 전체 suite FAILED/ERROR 차이 = flaky 1건 |
| `5685a60`·`d134b1d`·`f06a786`·`9f003e1` | `b6_build_obs_cache.py`: 같은 옵션(읽기 전용 evidence 연결), insertion order 기본값 유지, temp-dir 선검사, 1GB 하한 | obs cache 시험 63 passed, 전체 suite v15=v16=v17 동일 |

## 7. 다음 단계

1. 파일럿 link1 시작 시: `storage_preflight` 통과, 첫 `latest_state`(200 step), 처리량(s/step), GPU 메모리, `train.err` 확인.
2. link2 → `step_04099/` PILOT_FINAL·`TRAINING_DONE` 두 팔 확인 → G1/G2 파일럿 패널 평가(WP5 go 판정).
3. ACCESS 보충 잔액 반영 확인 → WP6 전체 학습·대조 실험·B5 seed 456/789 계획.
4. 기록 커밋은 사용자 요청 시(기록 브랜치 `b6-production-start-20260925`).
