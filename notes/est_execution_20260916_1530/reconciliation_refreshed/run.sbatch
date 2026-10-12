#!/bin/bash
#SBATCH --job-name=est_c1_reconciled_20260916
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled/logs/%j.out
#SBATCH --error=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled/logs/%j.err
# LOCAL PROPOSAL ONLY: controller review and separate staging/submission required.
# Fixed captured queue is NOT refreshed or filtered here; >1h at audit startup fails.
set -euo pipefail
umask 077
R=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$(realpath -e "$R")" == "$R" ]]
cd "$R"
# Inputs and code must have been exclusively staged from the exact reviewed bundle.
sha256sum --check --strict <<'PINS'
74d2337199fe0880e01a38cd136c03f28de4383b833393657ce8420ce3c57415  /data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11
b4392729aaff73feb61e4ffef8df7b6ddc66235ec6aa1fd988babb0da46bebfb  code/revision/scripts/evidence/est_blastx_audit.py
5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  code/revision/scripts/evidence/est_blastx.py
32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  code/revision/scripts/64_orthodb_filter.py
5d122878c84e75d4b536a6c98413f03bb8ea5ec9f5a7892274d7a626449e8ad4  code/revision/scripts/evidence/est_blastx_audit.sbatch
c8b03a35bcf527b70d69fa4c11eb0ab2d90da1da3dd0b487d4dd85c6bfe00a2e  inputs/manifest.proposed.json
61120e921d3c8897e46608e02a8c46d9233812f71791c024a2626150bfbbb237  inputs/accounting.psv
fc30d8cb7e0f85d96416ba7587c4981853a30a54e4ad2f83081b5797a3722a6e  inputs/queue.json
f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23  inputs/pilot_submission_recovered.json
PINS
for p in code inputs logs; do
  [[ -d "$R/$p" && "$(realpath -e "$R/$p")" == "$R/$p" ]]
done
for p in execution.guard audit scratch batch.exitcode; do
  [[ ! -e "$R/$p" && ! -L "$R/$p" ]] || exit 2
done
mkdir "$R/execution.guard"
# Guard persists on failure/success: this wrapper is one-shot, never restartable.
set -o noclobber
trap 'rc=$?; printf "%s\n" "$rc" > "$R/batch.exitcode"; exit "$rc"' EXIT
mkdir "$R/scratch"
export TMPDIR="$R/scratch" TMP="$R/scratch" TEMP="$R/scratch"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
"$PY" -I -B "$R/code/revision/scripts/evidence/est_blastx_audit.py" \
  --blastx-root /data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx \
  --out-dir "$R/audit" \
  --sacct-file "$R/inputs/accounting.psv" \
  --pilot-submission-receipt "$R/inputs/pilot_submission_recovered.json" \
  --reconciliation-manifest "$R/inputs/manifest.proposed.json" \
  --reconciliation-sha256 c8b03a35bcf527b70d69fa4c11eb0ab2d90da1da3dd0b487d4dd85c6bfe00a2e
