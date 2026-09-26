#!/bin/bash
#SBATCH --job-name=b6_a19_filter
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%A_%a.out
set -euo pipefail
SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v3
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  $PREP/a19_filter_miniprot.py "$SP" "$ROOT/evidence/protein_alignments/$SP/$SP.protein.gff.gz" \
  "$ROOT/evidence/est_blastx/preflight_20260914/swissprot/lineage.tsv" "$ROOT/evidence/protein_alignments_a19/$SP"
