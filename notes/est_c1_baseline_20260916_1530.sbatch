#!/bin/bash
#SBATCH --job-name=est_c1_20260916_1530
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1/logs/%j.out
#SBATCH --error=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1/logs/%j.err
set -euo pipefail
umask 077
R=/data/gpfs/assoc/pgl/data/Transgenic
N=$R/evidence/est_followup_20260916_1530/c1
W=$N/code
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3
export PATH=/usr/bin:/bin LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 ]]
[[ "$(readlink -f "$N")" == "$N" && "$(readlink -f "$W")" == "$W" ]]
[[ ! -e "$N/c1_baseline" && ! -L "$N/c1_baseline" ]]
mkdir "$N/execution.lock"
trap 'rc=$?; printf "%s\n" "$rc" > "$N/batch.exitcode"' EXIT
mkdir "$N/scratch"
export TMPDIR=$N/scratch
cd "$W"
printf '%s\n' \
 'd4955d6277aa540399bd8c32def14e352fa131e761ff79e51c522c7ee353f6f8  revision/scripts/evidence/est_blastx_audit.py' \
 '5d122878c84e75d4b536a6c98413f03bb8ea5ec9f5a7892274d7a626449e8ad4  revision/scripts/evidence/est_blastx_audit.sbatch' \
 '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  revision/scripts/evidence/est_blastx.py' \
 '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  revision/scripts/64_orthodb_filter.py' \
 "decb225348454b2df31e4f5715c6fb1eb74b7b08958c67e3e82ca3ad24133284  $N/sacct_c1.psv" \
 "74d2337199fe0880e01a38cd136c03f28de4383b833393657ce8420ce3c57415  $PY" \
 | sha256sum -c - > "$N/hash-check.txt"
"$PY" -I -B -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; print("filesystem_free_bytes=",free); assert free>=32*1024**3, "32GiB scratch headroom required"' "$TMPDIR" > "$N/headroom.txt"
{
 date -u; hostname; printf 'job=%s\ncommit=5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35\n' "$SLURM_JOB_ID"
 "$PY" -I -B --version
 printf 'Baseline diagnostic: known unreconciled historical directories/pilot receipt gate may produce FAIL; not a blastx rerun.\n'
} > "$N/runtime.txt" 2>&1
/usr/bin/time -v -o "$N/time.txt" "$PY" -I -B "$W/revision/scripts/evidence/est_blastx_audit.py" \
 --blastx-root "$R/evidence/est_blastx" --out-dir "$N/c1_baseline" --sacct-file "$N/sacct_c1.psv"
