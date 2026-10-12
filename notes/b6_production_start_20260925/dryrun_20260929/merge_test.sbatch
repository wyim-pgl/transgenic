#!/bin/bash
#SBATCH --job-name=b6_merge_test
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Dry run of the evidence merge on one species (Bdistachyon: EST N=100 + protein C0). Not the production merge.
# Output under dryrun_20260929/ only; production C0 files are read-only inputs.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v5
OUT=$PREP/dryrun_20260929/bdistachyon
mkdir -p "$OUT"
test ! -e "$OUT/evidence_c0.db"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_merge_evidence_db.py" \
  --inputs "$PREP/c0/Bdistachyon_est_n100/out/Bdistachyon.est.c0.db" "$PREP/c0/Bdistachyon_protein_v1/out/Bdistachyon.protein.c0.db" \
  --out "$OUT/evidence_c0.db"
ls -la "$OUT"; cat "$OUT/evidence_c0.db.sha256"
