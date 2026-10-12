#!/bin/bash
#SBATCH --job-name=b6_obs_cache_test_sbicolor_v11
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Dry run of the per-species train observation cache on the largest species (Sbicolor), source_v11 (3dc93fb, per chromosome). Measures peak memory. Not production.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v11
SP=Sbicolor; OUT=$PREP/dryrun_20260929/sbicolor_v8; Z=$PREP/zero/$SP
test -s "$OUT/evidence_c0.db.sha256"; test ! -e "$OUT/cache_train_v11"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_build_obs_cache.py" \
  --evidence "$OUT/evidence_c0.db" --family train --roles "$SOURCE/data/manifests/DATASET_ROLES.b6_v1.tsv" \
  --reference "$SP=$ROOT/training_input/genomes/Sbicolor_730_v5.0.fa" \
  --corpus "$PREP/corpus/b5_full_b6_v1.db" --freeze "$PREP/corpus/freeze.json" \
  --zero "train_track:$SP:$Z/$SP.train_track.zero.tsv:$Z/$SP.train_track.zero.qc.json" \
  --zero "valid_track:$SP:$Z/$SP.valid_track.zero.tsv:$Z/$SP.valid_track.zero.qc.json" \
  --zero "eval_input:$SP:$Z/$SP.eval_input.zero.tsv:$Z/$SP.eval_input.zero.qc.json" \
  --out "$OUT/cache_train_v11"
du -sh "$OUT/cache_train_v11"; ls "$OUT/cache_train_v11" | head
