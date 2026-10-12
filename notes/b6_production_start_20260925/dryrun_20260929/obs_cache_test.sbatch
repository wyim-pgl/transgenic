#!/bin/bash
#SBATCH --job-name=b6_obs_cache_test
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Dry run of the train observation cache on the one-species merged DB from merge_test.sbatch. Not production.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v5
SP=Bdistachyon; OUT=$PREP/dryrun_20260929/bdistachyon; Z=$PREP/zero/$SP
test -s "$OUT/evidence_c0.db.sha256"; test ! -e "$OUT/cache_train"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_build_obs_cache.py" \
  --evidence "$OUT/evidence_c0.db" --family train --roles "$SOURCE/data/manifests/DATASET_ROLES.b6_v1.tsv" \
  --reference "$SP=$ROOT/training_input/genomes/Bdistachyon_314_v3.0.fa" \
  --corpus "$PREP/corpus/b5_full_b6_v1.db" --freeze "$PREP/corpus/freeze.json" \
  --zero "train_track:$SP:$Z/$SP.train_track.zero.tsv:$Z/$SP.train_track.zero.qc.json" \
  --zero "valid_track:$SP:$Z/$SP.valid_track.zero.tsv:$Z/$SP.valid_track.zero.qc.json" \
  --zero "eval_input:$SP:$Z/$SP.eval_input.zero.tsv:$Z/$SP.eval_input.zero.qc.json" \
  --out "$OUT/cache_train"
du -sh "$OUT/cache_train"; ls "$OUT/cache_train" | head
