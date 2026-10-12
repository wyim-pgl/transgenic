# 재시작 인계: 2026-09-28 14:30 PDT — B6 실데이터 증거 DB 구축 (C0 마무리 단계)

📌 **이 문서가 현재 정본이다.** 이전 인계(`RESTART_HANDOFF_20260924_PI.md` 등)는 역사 기록이다. 세부 실행 기록은 [PROGRESS.md](b6_production_start_20260925/PROGRESS.md), 저자 결정 원문은 [AUTHOR_DECISIONS_20260925_evening.md](b6_production_start_20260925/AUTHOR_DECISIONS_20260925_evening.md)(결정 1–18).

## 0. 한 줄 요약

실제 생물학 데이터로 B6 학습 입력을 만드는 중이다. **GPU 훈련은 아직 제출하지 않았다.** C0 증거 DB는 대부분 PASS, B6 코퍼스 생성·검증 PASS(N_pilot 4,099). 남은 C0(ONT 2종, protein 3종)가 끝나면 zero list → 병합 → obs cache → bindings → DeltaAI → 파일럿 제출 전 사용자 확인.

## 1. 코드 위치

- **통합 브랜치:** `/tmp/b6-integration` (branch `b6-integration`) **HEAD `27c34ba`**. main에 병합하지 않는다. push하지 않았다.
  - 이 세션 커밋(오래된 순): `d929ba2` --missing-library · `6b6dcd3` masked X · `396fcbf` HSP 행 정책 · `0fc80de` A16 연결성분 · `b85b901` WP5 파일럿 런처+GATED · `83616e4` freeze 상대경로 · `4d1b752` 종별 flags · `b009dcc` 병합 입력 상대경로 · `c422e5e` 사건 타일 drop + N_pilot freeze 결속 · `27c34ba` N_pilot 4,099 + gate 강화.
  - 보조 worktree(각 기능 원본, 모두 cherry-pick 완료): `/tmp/b6-hsp-rowpolicy`, `/tmp/b6-a16-components`, `/tmp/b6-wp5-pilot-launcher`, `/tmp/b6-freeze-relocatable`, `/tmp/b6-repair-merge-paths`, `/tmp/b6-corpus-drop-events`, `/tmp/b6-npilot-4099`.
- **git 주의:** 이 저장소들은 소유권 검사로 거부된다. `GIT_CONFIG_GLOBAL=<scratchpad>/ro.gitconfig`(내용: `[include] path=/home/wyim/.gitconfig` + `[safe] directory = *`) 또는 `/tmp/b6-resume-safe.gitconfig`를 쓴다. 커밋 author/committer는 `Won Yim <wyim@unr.edu>`.
- **기록 브랜치:** 주 저장소 `/mnt/f/scratch/develop/Transgenic/transgenic_local`의 `b6-production-start-20260925`(main worktree는 main 그대로). notes/.atl 기록을 plumbing(read-tree + commit-tree + update-ref, 임시 GIT_INDEX_FILE)으로 이 브랜치에 커밋해 왔다.
- **pronghorn 배포 소스(읽기 전용):** `$PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925`
  - `source`(3511a5a, EST·PacBio C0), `source_v3`(0fc80de: HSP v3·ONT v2·N=100 C0), `source_v4`(b009dcc), `source_v5`(c422e5e: 코퍼스 measure v2·apply·validate). protein C0는 `source_v3`.
  - 실행 환경: `singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python`, `PYTHONPATH=$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence`. 호스트 python은 `/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python`(DuckDB는 컨테이너 안에서만).

## 2. C0 현황 (병합에 쓸 파일)

`$PREP/c0/<dir>/out/<Sp>.<src>.c0.db` (Sitalica EST만 `out_arrow_retry/`). 모두 status PASS·sha 재계산 일치·leakage 0·위반 gate 0.

| 종 | EST | PacBio | ONT | protein | 종 exhaustiveness |
|---|---|---|---|---|---|
| Athaliana | `Athaliana_est_v1` | – | **재실행 중** 6173686 (`Athaliana_ont_v2`) | `Athaliana_protein_v1` | EST 0.99991 (ONT 합산 대기) |
| Bdistachyon | **`Bdistachyon_est_n100`** | – | – | `Bdistachyon_protein_v1` | 0.99991 |
| Gmax | **`Gmax_est_n100`** | – | **`Gmax_ont_n100`** | 대기 6173790 | 0.99981 |
| Osativa | `Osativa_est_v1` | – | **재실행 중** 6173687 (`Osativa_ont_v2`) | 대기 6173791 | EST 0.99936 |
| Ppatens | `Ppatens_est_v1` | – | `Ppatens_ont_v1` | `Ppatens_protein_v1` | 0.99979 |
| Ptrichocarpa | `Ptrichocarpa_est_v1` | `Ptrichocarpa_pacbio_v1` | `Ptrichocarpa_ont_v1` | `Ptrichocarpa_protein_v1` | 0.99972 |
| Sbicolor | `Sbicolor_est_v1` | `Sbicolor_pacbio_v1` | – | `Sbicolor_protein_v1` | 0.99959 |
| Sitalica | `Sitalica_est_v1`(out_arrow_retry) | – | `Sitalica_ont_v2` | `Sitalica_protein_v1` | 0.99984+ |
| Vvinifera | **`Vvinifera_est_n500`** | **`Vvinifera_pacbio_n500`** | **`Vvinifera_ont_n500`** | 대기 6173789 | 0.999997 |

- **N 규칙:** 한 종의 모든 소스는 같은 audit N. Bdistachyon·Gmax는 N=100, Vvinifera는 N=500, 나머지 N=50. 대체된 N=50/100 C0(`*_est_v1` for Bdistachyon/Gmax/Vvinifera, `Vvinifera_*_n100` 등)는 병합에 쓰지 않는다.
- protein C0는 병합되는 EST C0 파일과 같은 sha여야 한다(Bdistachyon·Gmax·Vvinifera protein은 N=100/500 EST에서 다시 만든 것).
- 대체·실패 산출물은 지우지 않고 `*.superseded_*`, `*.timeout_*`, `out_failed_*`, `*.oom_*`로 옮겨 두었다.

## 3. 실행 중 잡 (2026-09-28 14:26 PDT)

| 잡 | 내용 | 경과/제한 | 다음 |
|---|---|---|---|
| 6173197 | HSP Vvinifera (EST N=500) | 16h/24h (N=100 때 17h30m) | → protein C0 6173789 |
| 6173591 | HSP Gmax (EST N=100) | 19h44m/4d | → protein C0 6173790 |
| 6173182 | HSP Osativa | 28h/4d | → protein C0 6173791 |
| 6173686 | ONT C0 Athaliana (200G/DuckDB 64GB) | 3h48m/10d | 이전 6172713 batch OOM |
| 6173687 | ONT C0 Osativa (128G/DuckDB 96GB) | 3h41m/4d | 이전 6172995 DuckDB 한도 초과 |

감시는 20분 간격(사용자 지시). 완료되면 `~/c0v.py <dir>:<Sp>:<src>`(pronghorn 홈, 검증 스크립트)로 status·sha·leak·gates·exhaustiveness를 확인하고 PROGRESS 표에 추가한다.

## 4. B6 코퍼스 (완료)

- `$PREP/corpus/b5_full_b6_v1.db` (23.8GB, sha256 `6583b1b202c4c5b93b88f893e6c15f267a5ec54f0c2db675b29466071f15111c`), `$PREP/corpus/freeze.json` (train_rows 196,774, n_pilot 4,099, interval manifest `../intervals/<Sp>.intervals_v2.jsonl` 상대경로).
- 사건 타일 1,741 forward + 489 RC = 2,230행 삭제(결정 18). 독립 검증 `validate_noc0.json` PASS(`--c0` 없이). **병합 후 `--c0 <merged>`로 재검증 필요.**
- interval manifest 9종(`$PREP/intervals/`)은 모든 C0의 `interval_manifest_sha256`과 일치.
- 스크립트: [corpus_apply.sbatch](b6_production_start_20260925/corpus_apply.sbatch), [corpus_validate.sbatch](b6_production_start_20260925/corpus_validate.sbatch)(200G 필요).

## 5. 다음 단계 (C0 완료 후, 순서대로)

1. **zero list 27개:** `revision/scripts/evidence/b6_routea_agreement.py`(argparse ~L488-504), 종 × {train_track, valid_track, eval_input}. 입력: 병합할 EST·protein C0, `$PREP/hsp_v3/<Sp>/<Sp>.est_protein_hsp.tsv`, `evidence/protein_alignments_a19/<Sp>/<Sp>.protein.gff.gz`(A19 필터본), accept/mapping_ambiguous. 상류 sha는 병합 manifest와 같아야 한다.
2. **병합:** `b6_merge_evidence_db.py --inputs <모든 C0> --out evidence_c0.db`(DuckDB 한도 미설정 → Slurm 메모리 넉넉히). 게이트: 종 exhaustiveness ≥0.999(동일 N), 소스별 ≥0.99, leakage·HSP·MULTI_HSP, protein↔EST sha.
3. **코퍼스 재검증** `--c0 <merged>`.
4. **obs cache:** `b6_build_obs_cache.py --family train`(roles `data/manifests/DATASET_ROLES.b6_v1.tsv` + sha, `--reference SP=FASTA`×9, `--corpus`, `--freeze`, `--zero pop:sp:tsv:qc`×27).
5. **bindings JSON:** `{"evidence","roles","cache","freeze","corpus"[,"references"]}`(`train/train_HyenaTransgenic_B6.py`), `verify_inputs(accept=("training",))`.
6. **DeltaAI 전송:** 병합 DB·C0 DB·`.PROVENANCE.txt`를 같은 상대 배치로, freeze·intervals도 같은 상대 배치로. 새 repo 스냅숏(`27c34ba` 이상). 컨테이너 내부 경로 기준. 해시 대조.
7. **파일럿:** `revision/scripts/b6/wp5_pilot.sbatch`(ARM=B6/C2, BINDINGS=…, `--stop-after-updates 4099`, GATED config `configs/b6_400m_win_v3_wp5_pilot.json`). **GPU 제출 전 사용자 확인**(예산: B6+C2 ≈ 98 GPU-h, 잔액 1,022 − B5 예비 220; WP6 800+는 부족 가능).

## 6. 저자 결정 요약 (원문은 AUTHOR_DECISIONS 파일)

1 UMI 125런 absent 유지 · 2 Vvinifera 누락 175 records library=`NCBI-EST` · 3 extern OOM 전역 무시 · 4 주말 무중단 진행 · 5 HSP BTOP masked X 인정 · 6 protein 역할행 추가(새 roles 파일 `inputs/DATASET_ROLES.longread_protein_v1.tsv`) · 7 Athaliana cui2020 기존 정렬 재사용 · 8 FLIC 별도 드라이버 정렬 · 9 A16 비-clique PCR 성분 = 한 분자 · 10 HSP 길이 계약 M+I+D 또는 M+I+D+F · 11 재생 불가 HSP 행 제외+QC · 12 Bdistachyon N=100 · 13 Gmax·Vvinifera N=100 · 14 freeze 경로 코드 변경(상대경로) · 15 config GATED · 16 D1 flank geometry (A) 기본안 · 17 Vvinifera N=500 · 18 코퍼스 사건 타일 제외, N_pilot 재계산(4,099; 개정 기록 `notes/a46_approval/A46_RULE11_NPILOT_AMENDMENT_20260927.md`).

## 7. 알려진 제약

- 학습 ONT identity 하한 0.95(PacBio 0.98, `59_evidence_support.py:733-735`) 때문에 direct RNA 채택률이 낮다(Ppatens 2.0%, Ptrichocarpa 2.8%, Sitalica cDNA 1.2%). 동결 규칙이라 바꾸지 않았다.
- 파일럿 LR: 전체 스케줄을 4,099에서 자름. 4 GPU에서 accelerate가 업데이트당 scheduler 4스텝 → warm-up 이후, LR ≈ 0.96×peak.
- Athaliana·Zmays PacBio와 Slycopersicum·Zmays ONT는 검증 전용이라 학습 C0 대상이 아니다.
