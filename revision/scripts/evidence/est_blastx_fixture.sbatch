#!/usr/bin/env bash
#SBATCH --job-name=b6_blastx_fixture
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Submit only after controller review. No array and no production input paths.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    printf 'Usage: sbatch %s REVIEWED_CHECKOUT NEW_OUTPUT_DIR EXISTING_GPFS_TMPDIR\n' "$0" >&2
    exit 2
fi
CHECKOUT=$1
OUT=$2
TMP=$3
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
DIAMOND=/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond
[[ "$CHECKOUT" = /* && "$OUT" = /data/gpfs/* && "$TMP" = /data/gpfs/* ]] || { printf 'Absolute checkout and GPFS output/tmp paths required\n' >&2; exit 2; }
[[ -d "$CHECKOUT" && -d "$TMP" && ! -e "$OUT" && ! -L "$OUT" ]] || { printf 'Invalid checkout/tmp or existing output\n' >&2; exit 2; }
[[ -x "$PY" && -x "$DIAMOND" ]] || { printf 'Required Python/DIAMOND executable missing\n' >&2; exit 2; }
cd "$CHECKOUT"
# exec propagates interpreter failures; Python records FAILED on TERM/INT after
# workdir creation. SIGKILL/OOM may leave no FAILED but can never publish DONE.
exec "$PY" revision/scripts/evidence/est_blastx_fixture.py \
    --diamond "$DIAMOND" --out-dir "$OUT" --tmpdir "$TMP" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
