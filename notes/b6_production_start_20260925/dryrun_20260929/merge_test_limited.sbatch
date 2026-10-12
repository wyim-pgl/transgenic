#!/bin/bash
#SBATCH --job-name=b6_merge_test_limited
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Dry run of the evidence merge under a DuckDB memory limit below the unlimited peak (54GB): source_v6 (4e6b9d8).
# Output under dryrun_20260929/ only; production C0 files are read-only inputs.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v6
OUT=$PREP/dryrun_20260929/bdistachyon_limited
mkdir -p "$OUT"
test ! -e "$OUT/evidence_c0.db"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_merge_evidence_db.py" \
  --inputs "$PREP/c0/Bdistachyon_est_n100/out/Bdistachyon.est.c0.db" "$PREP/c0/Bdistachyon_protein_v1/out/Bdistachyon.protein.c0.db" \
  --out "$OUT/evidence_c0.db" --memory-limit 24GB --threads 8
ls -la "$OUT"; cat "$OUT/evidence_c0.db.sha256"
