# 재시작 인계: 2026-09-30 12:00 PDT — B6 실데이터 증거 DB (C0 마무리 + 병합·obs cache 준비 완료)

📌 **이 문서가 현재 정본이다.** 이전 인계 [RESTART_HANDOFF_20260928](RESTART_HANDOFF_20260928.md)는 역사 기록이다(§2 C0 파일 표와 §5 단계 설명은 아래에서 갱신). 세부 실행 로그는 [PROGRESS.md](b6_production_start_20260925/PROGRESS.md)(시간 역순), 저자 결정 원문은 [AUTHOR_DECISIONS_20260925_evening.md](b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md)(결정 1–18). 이번 세션의 사용자 결정은 §6.

## 0. 한 줄 요약

**GPU 훈련은 아직 제출하지 않았다.** 남은 C0는 Athaliana ONT(재빌드 중)와 Gmax protein(HSP 대기) 두 개다. zero list는 24/27 완료(Gmax 3개는 체인으로 대기). 병합·obs cache는 1종 시험으로 명령·자원을 검증했고, 규모 문제 4건을 코드로 고쳐 **production 소스 `source_v13`(`c8e01f4`)** 에 배포했다. 모든 C0가 끝나면 병합 → 코퍼스 재검증 → obs cache → bindings → DeltaAI → 파일럿 제출 전 사용자 확인.

## 1. 코드 위치

- **통합 브랜치:** `/tmp/b6-integration` (branch `b6-integration`) **HEAD `c8e01f4`**. main에 병합하지 않는다. **push하지 않았다.**
  - 이번 세션 커밋(오래된 순, 모두 author/committer `Won Yim <wyim@unr.edu>`, 트레일러 없음):
    - `4e6b9d8` 병합 `--memory-limit/--threads/--temp-dir`(DuckDB가 cgroup보다 큰 기본 한도를 쓰던 문제)
    - `51037f6` obs cache 종 단위 생성(CacheWriter, 결과 바이트 동일)
    - `85293de` 병합 DB 데이터 테이블에 PRIMARY KEY 미선언(ART 인덱스가 커밋 때 메모리에만 존재, duckdb#5529). 키는 기존 중복 검사로 강제, `c0_meta.primary_keys`에 기록
    - `b2a3f56` Codex 리뷰 수정: EST 분자 대표 (dataset, run, role) 모호시 거부(any_value 제거), `check_species_coverage`, 시험 보강
    - `5edf9f8` 빈 coverage 거부
    - `3dc93fb` obs cache 염색체 단위 생성(drops는 분자 id 집합 합집합 후 계수, est_mol은 종 수준)
    - `3a360d3` 추출 대상 없는 scaffold 건너뜀(성능, 결과 동일)
    - `c8e01f4` **A16 PCR 성분 union-find**(`59_evidence_support.pcr_components`, 결과 동일, 메모리 선형)
  - 기능별 worktree(모두 fast-forward/cherry-pick 완료): `/tmp/b6-merge-memlimit`, `/tmp/b6-obs-cache-stream`, `/tmp/b6-merge-nopk`, `/tmp/b6-review-fixes`, `/tmp/b6-obs-cache-chrom`, `/tmp/b6-obs-chrom-discovery`, `/tmp/b6-pcr-unionfind`.
- **git:** 소유권 검사로 거부되므로 `GIT_CONFIG_GLOBAL=<scratchpad>/ro.gitconfig`(`[include] path=/home/wyim/.gitconfig`, `[safe] directory = *`)로 실행한다. 커밋은 `git -c user.name="Won Yim" -c user.email="wyim@unr.edu" commit`.
- **기록:** 주 저장소 `/mnt/f/scratch/develop/Transgenic/transgenic_local`의 notes/.atl는 **이번 세션에서 커밋하지 않았다**(작업 트리만 갱신). 이전 세션은 `b6-production-start-20260925` 브랜치에 plumbing으로 커밋했다.
- **pronghorn 배포 소스(읽기 전용):** `$PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925`
  - `source_v3`(0fc80de): 기존 EST·PacBio·ONT·protein C0 빌드. `source_v5`: 코퍼스·zero list.
  - `source_v6`–`v12`: 중간 시험용. **`source_v13`(`c8e01f4`): 본 병합·obs cache·Athaliana ONT 재빌드에 쓴다.** `b6_build_c0.py` sha `63d52619…`는 source_v3와 같다(v3 이후 C0 관련 변경은 `c0_schema.py` 기본값 불변뿐).
  - 실행: `singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python`, `PYTHONPATH=$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence`. 호스트 python은 sylvan env(`python3` 없음).
  - 단위 시험: 같은 컨테이너, `PYTHONPATH=$PREP/duckdb_runtime:src:revision/scripts:revision/scripts/evidence:tests`, `pytest -q -p no:cacheprovider --continue-on-collection-errors -rfE tests revision/scripts/tests`. 컨테이너에 torch가 없어 `test_mvp_*` 등 약 165 failed/28 errors는 기존 환경 실패다. base 대비 **실패 집합 diff**로 판정한다.

## 2. C0 현황 (병합에 쓸 파일)

`$PREP/c0/<dir>/out/<Sp>.<src>.c0.db` (Sitalica EST만 `out_arrow_retry/`). 표시 없는 칸은 PASS·sha 재계산 일치·leakage 0·위반 gate 0.

| 종 | EST | PacBio | ONT | protein | 종 exhaustiveness |
|---|---|---|---|---|---|
| Athaliana | `Athaliana_est_v1` | – | **재빌드 중 6182077** (`Athaliana_ont_v2`) | `Athaliana_protein_v1` | EST 0.99991 (ONT 합산 대기) |
| Bdistachyon | `Bdistachyon_est_n100` | – | – | `Bdistachyon_protein_v1` | 0.99991 |
| Gmax | `Gmax_est_n100` | – | `Gmax_ont_n100` | **대기 6181302** (HSP 6181301 후) | 0.99981 |
| Osativa | `Osativa_est_v1` | – | `Osativa_ont_v2` (09-29 PASS, sha `617618ec`) | `Osativa_protein_v1` (09-29 PASS, sha `c650909b`) | **0.99968** |
| Ppatens | `Ppatens_est_v1` | – | `Ppatens_ont_v1` | `Ppatens_protein_v1` | 0.99979 |
| Ptrichocarpa | `Ptrichocarpa_est_v1` | `Ptrichocarpa_pacbio_v1` | `Ptrichocarpa_ont_v1` | `Ptrichocarpa_protein_v1` | 0.99972 |
| Sbicolor | `Sbicolor_est_v1` | `Sbicolor_pacbio_v1` | – | `Sbicolor_protein_v1` | 0.99959 |
| Sitalica | `Sitalica_est_v1`(out_arrow_retry) | – | `Sitalica_ont_v2` | `Sitalica_protein_v1` | 0.99984+ |
| Vvinifera | `Vvinifera_est_n500` | `Vvinifera_pacbio_n500` | `Vvinifera_ont_n500` | `Vvinifera_protein_v1` (09-28 PASS, sha `043d270d`) | 0.999997 |

- N 규칙·protein↔EST sha 결속·산출물 보존 규칙은 09-28 인계 §2와 같다.
- 대체·실패 산출물: `hsp_v3/Gmax.oom_6173591/`, `c0/Athaliana_ont_v2/out_failed_6173686/`.

## 3. 실행 중 잡 (2026-09-30 12:00 PDT)

| 잡 | 내용 | 경과/제한 | 다음 |
|---|---|---|---|
| 6181301 | HSP Gmax (200G, cpu-51) | 13h10m / 13d. 이전 6173591은 2d03h52m에 출력 단계 OOM(32G) | → protein C0 **6181302** → zero **6181303**(train)·**04**(valid)·**05**(eval), 64G, afterok 체인 |
| 6182077 | ONT C0 Athaliana 재빌드 (source_v13, 160G, DuckDB 64GB) | 59m / 13d. 이전 실행은 1d17h48m | 완료 후 `~/c0v.py Athaliana_ont_v2:Athaliana:ont` + 종 exhaustiveness(EST+ONT) |

감시: 백그라운드 bash until-loop(15분 간격)가 이 잡들의 종료를 알린다. 새 세션에서는 `sacct -j 6181301,6181302,6181303,6181304,6181305,6182077 -X`로 실측한다.

## 4. zero list (24/27 완료)

`$PREP/zero/<Sp>/<Sp>.<pop>.zero.{tsv,qc.json}`, 스크립트 `$PREP/zero_list.sbatch`(source_v5). 입력 accept 표는 A19 필터본 `evidence/protein_alignments_a19/<Sp>/accept/`(`a19_filter_accept.py`, 9종 kept/dropped가 GFF 필터 DONE과 일치).

| 종 | train/valid/eval 불일치율 | 비고 |
|---|---|---|
| Athaliana | 3.37 / 3.34 / 3.15% | 다른 종의 5–10배, 원인 미조사(비차단) |
| Bdistachyon | 0.69 / 0.71 / 0.74% | |
| Osativa | 1.96 / 1.93 / 1.84% | |
| Ppatens | 0.61 / 0.62 / 0.66% | |
| Ptrichocarpa | 0.44 / 0.44 / 0.42% | |
| Sbicolor | 0.30 / 0.31 / 0.29% | |
| Sitalica | 0.32 / 0.36 / 0.32% | |
| Vvinifera | 1.20 / 1.15 / 1.07% | |
| Gmax | 대기 6181303–05 | |

메모리: Athaliana 25–31GB, 나머지 12–26GB → 64G로 낸다.

## 5. 병합·obs cache 사전 시험 결과와 본 실행 설정

| 시험 | 결과 |
|---|---|
| Bdistachyon 병합 6174332(무제한) | PASS 33m47s, RSS 54.3GB |
| Bdistachyon 병합 6174402(`--memory-limit 24GB`) | PASS 34m37s, RSS 25.7GB → DuckDB 스필 정상 |
| Bdistachyon obs cache 6174333 | PASS 28m, RSS 59.9GB, 27.16M행 |
| Sbicolor 병합 6181312(PK 선언) | FAILED 커밋 시 `37.2/37.2 GiB` → `85293de`로 해결 |
| Sbicolor 병합 6181753(PK 미선언, 40GB 한도) | PASS 47m38s, RSS 44.0GB, 입력 150GB → 85.7GB |
| Sbicolor obs cache 6181754(종 단위, 220G) | FAILED signal 9, RSS 214.9GB → 염색체 단위 `3dc93fb`로 해결 |
| Sbicolor obs cache 6181768(염색체 단위, 128G) | **PASS 1h50m, RSS 101.9GB**, 165.7M행 55 shard 3.4GB, verify_cache 통과 |

- 전체 병합 입력: 24 C0, 6.40B행(PK 테이블 4.53B, aligned_block 712M). 시험 스크립트: `notes/b6_production_start_20260925/dryrun_20260929/`(`merge_test*.sbatch`, `obs_cache_test*.sbatch`, `c0_rowcounts.py`).
- **본 병합 설정(제안):** `source_v13`, 한 노드 약 200G 할당, `--memory-limit 170GB --threads 16`. 입력 크기 비례면 약 5h.
- **본 obs cache 설정(제안):** `source_v13`, 220G(최대 염색체가 Athaliana ONT Chr1일 수 있음). roles `source_v5/data/manifests/DATASET_ROLES.b6_v1.tsv`, `--reference SP=FASTA`×9(C0 provenance의 reference와 같아야 함), `--corpus $PREP/corpus/b5_full_b6_v1.db --freeze $PREP/corpus/freeze.json`, `--zero pop:sp:tsv:qc`×27.

## 6. 이번 세션 사용자 결정

1. **Slurm 시간 제한 전역 13일:** pronghorn `~/.bashrc` 16행 `SBATCH_TIMELIMIT="13-00:00:00"`(비대화형 가드 앞). `~/.claude/CLAUDE.md`에도 규칙. 파티션 MaxTime이 짧은 곳(cpu-s6-test 15분, cpu-s3-sponsored 8h)은 명시 `--time`. DeltaAI GPU는 해당 없음.
2. protein C0 대기 잡 3개를 13일로 재제출.
3. zero list는 protein C0 afterok로 미리 제출.
4. 병합·obs cache를 작은 입력으로 사전 시험.
5. 병합 DuckDB 자원 옵션 추가(option 1).
6. obs cache 종 단위 → 한 종도 안 맞으면 염색체 단위(option 1).
7. 병합 DB 데이터 테이블 PK 미선언(option 1).
8. 문제는 Codex에게 리뷰받는다(모든 수정 커밋에 Codex 리뷰를 받았다).
9. Athaliana ONT OOM은 union-find로 해결(option 1).

## 7. 다음 단계 (순서대로)

1. Athaliana ONT C0 6182077 완료 → `c0v.py` 검증, 종 exhaustiveness ≥0.999(EST+ONT).
2. Gmax HSP 6181301 → protein C0 6181302 검증(EST `Gmax_est_n100` sha 결속) → zero 6181303–05 확인 → **27/27**.
3. **본 병합**(§5 설정): §2의 24 C0 + Athaliana ONT + Gmax protein = 26 입력. 게이트: 종 exhaustiveness ≥0.999, 소스별 ≥0.99, leakage·HSP·MULTI_HSP, protein↔EST sha, 중복 키.
4. 코퍼스 재검증 `b6_validate_corpus.py --c0 <merged>`(200G).
5. obs cache `--family train`(§5 설정).
6. bindings JSON `{"evidence","roles","cache","freeze","corpus"[,"references"]}` 작성, `verify_inputs(accept=("training",))`.
7. DeltaAI 전송: 병합 DB와 C0·`.PROVENANCE.txt`를 같은 상대 배치로(병합 DB의 merge_manifest 경로는 상대 `../../c0/...` 형식), freeze·intervals도 같은 상대 배치, 새 repo 스냅숏(`c8e01f4` 이상), 해시 대조.
8. **GPU 제출 전 사용자 확인:** WP5 파일럿 B6·C2(`revision/scripts/b6/wp5_pilot.sbatch`, 4 GPU, `--stop-after-updates 4099`, ≈98 GPU-h). 잔액 1,022 − B5 예비 220. WP6(800+ GPU-h)는 부족할 수 있음.

## 8. 알려진 제약·주의

- 이전 인계 §7(ONT identity 0.95 등)은 그대로다.
- Athaliana zero list 불일치율이 3%대로 높은 원인은 조사하지 않았다(규칙대로 해당 구간 route-B phase 0).
- `tests/test_est_real_sample.py::test_abort_kills_descendant_after_leader_exit`는 시간 의존 flaky 시험이다(양쪽 트리에서 간헐 실패).
- 도중 실패한 per-chrom obs cache는 manifest 없는 shard를 남긴다(ObsCache가 열기 거부) → 재실행 전 그 디렉터리를 비운다.
- `check_species_coverage`는 표마다 `DISTINCT species` 스캔을 한다(대형 표 비용 미측정).
