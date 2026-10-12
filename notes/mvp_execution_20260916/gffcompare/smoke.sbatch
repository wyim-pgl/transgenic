#!/bin/bash
#SBATCH --job-name=mvp-gffcompare-smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Controller supplies approved account/CPU partition and absolute log paths.
# No submission is performed by this file outside sbatch.
set -euo pipefail
: "${SLURM_JOB_ID:?must run as reviewed Slurm job}"
: "${R:?controller must export absolute revision root}"
[[ "$R" = /* ]] || exit 2
BASE="$R/evidence/mvp_gffcompare_smoke_20260916"
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
cd "$BASE"
# Controller stages only these reviewed files plus the exact producer source.
sha256sum -c stage.sha256
"$PY" -I "$BASE/smoke.py" selfcheck --producer "$BASE/mvp_produce_counts.py"
# New immutable run directory; second execution refuses overwrite.
"$PY" -I "$BASE/smoke.py" run --producer "$BASE/mvp_produce_counts.py" --root "$BASE/run"
# stdout goes to Slurm log; analyzer never writes into run/.
"$PY" -I "$BASE/smoke.py" analyze --root "$BASE/run"
sha256sum "$BASE/run/hashes.json"
