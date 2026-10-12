#!/bin/bash
#SBATCH --job-name=est_hsp_diag_1530
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/fixture/logs/slurm-%j.out
set -euo pipefail
umask 077
R=/data/gpfs/assoc/pgl/data/Transgenic
N=$R/evidence/est_followup_20260916_1530/fixture
W=$N/code
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
DIAMOND=/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond
export PATH=/usr/bin:/bin
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
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
# Import closure, not just the top-level extension. Hash on compute only.
printf '%s\n' \
 '8da86ae1bc4323da4efb24dffd177d060479f958e431b7e79722a88b57d80f36  revision/scripts/evidence/est_phase_accept_fixture.py' \
 '0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3  revision/scripts/evidence/est_phase_accept.py' \
 '18da9abd2a51a870e327226c8a1f9d225711a7c4551de40fd620c59470dbbad2  revision/scripts/evidence/est_blastx_fixture.py' \
 '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  revision/scripts/evidence/est_blastx.py' \
 '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  revision/scripts/64_orthodb_filter.py' \
 "0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d  $DIAMOND" \
 | sha256sum -c - > "$N/hash-check.txt"
{
 date -u; hostname; printf 'job=%s\n' "$SLURM_JOB_ID"
 printf 'commit=5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35\n'
 printf 'python=%s\ndiamond=%s\nthreads=2\n' "$PY" "$DIAMOND"
 "$PY" -I -B --version
 "$DIAMOND" version
 "$DIAMOND" help
} > "$N/runtime.txt" 2>&1
/usr/bin/time -v -o "$N/time.txt" \
 "$PY" -I -B "$W/revision/scripts/evidence/est_phase_accept_fixture.py" \
 --diamond "$DIAMOND" --out-dir "$N/result" --tmpdir "$N/tmp" --threads 2
