#!/bin/bash
#SBATCH --job-name=b6_merge_test_sbicolor_v8
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Dry run of the evidence merge for the largest species (Sbicolor: EST + PacBio + protein, ~150GB) under a DuckDB
# memory limit, source_v8 (85293de, no declared data-table PKs). Feeds obs_cache_test_sbicolor.sbatch. Not the production merge.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v8; C=$PREP/c0
OUT=$PREP/dryrun_20260929/sbicolor_v8
mkdir -p "$OUT"
test ! -e "$OUT/evidence_c0.db"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_merge_evidence_db.py" \
  --inputs "$C/Sbicolor_est_v1/out/Sbicolor.est.c0.db" "$C/Sbicolor_pacbio_v1/out/Sbicolor.pacbio.c0.db" "$C/Sbicolor_protein_v1/out/Sbicolor.protein.c0.db" \
  --out "$OUT/evidence_c0.db" --memory-limit 40GB --threads 8
ls -la "$OUT"; cat "$OUT/evidence_c0.db.sha256"
