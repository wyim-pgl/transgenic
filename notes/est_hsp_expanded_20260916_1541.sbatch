#!/bin/bash
#SBATCH --job-name=est_hsp_expand_1541
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1541/expanded/logs/slurm-%j.out
# TEMPLATE ONLY. Controller reviews, checks live jobs, stages archive, submits.
# Stage code from a98f5c0cc08e6f6a5ac27905d71a4478e9a0e789, not a mutable worktree.
# No acceptance or projection, even if payload exit is zero.
set -euo pipefail
umask 077
N=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1541/expanded
W=$N/code
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
DIAMOND=/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond
export PATH=/usr/bin:/bin LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 2 ]]
[[ -x "$PY" && -x "$DIAMOND" && -d "$W" && ! -L "$W" ]]
[[ "$(readlink -f "$N")" == "$N" && "$(readlink -f "$W")" == "$W" ]]
[[ ! -e "$N/result" && ! -L "$N/result" ]]
mkdir "$N/execution.lock"
trap 'rc=$?; printf "%s\n" "$rc" > "$N/batch.exitcode"' EXIT
mkdir "$N/tmp"
export TMPDIR=$N/tmp
cd "$W"
# Complete non-stdlib runtime import closure, measured AFTER source commit.
printf '%s\n' \
 'aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4  revision/scripts/evidence/est_hsp_expanded.py' \
 '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  revision/scripts/evidence/est_blastx.py' \
 '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  revision/scripts/64_orthodb_filter.py' \
 "0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d  $DIAMOND" \
 | sha256sum -c - > "$N/hash-check.txt"
{
 date -u; hostname; printf 'job=%s\n' "$SLURM_JOB_ID"
 printf 'source_commit=a98f5c0cc08e6f6a5ac27905d71a4478e9a0e789\n'
 printf 'python=%s\ndiamond=%s\nthreads=2\n' "$PY" "$DIAMOND"
 "$PY" -I -B --version
 "$DIAMOND" version
 "$DIAMOND" help
} > "$N/runtime.txt" 2>&1
/usr/bin/time -v -o "$N/time.txt" \
 "$PY" -I -B "$W/revision/scripts/evidence/est_hsp_expanded.py" \
 --diamond "$DIAMOND" --out-dir "$N/result" --tmpdir "$N/tmp"
