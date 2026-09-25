#!/bin/bash
#SBATCH --job-name=b6_si_est_c0
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
RUN=$PREP/c0/Sitalica_est_v1
SOURCE=$PREP/source
SIF=/data/gpfs/home/wyim/scratch/singularity/micromamba.sif
PY=/data/gpfs/home/wyim/micromamba/bin/python
SAMTOOLS=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/samtools
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONNOUSERSITE
singularity exec -B /data/gpfs "$SIF" "$PY" -c 'import duckdb,sys; sys.exit(0 if duckdb.__version__ == "1.5.5" else "REFUSED DuckDB runtime version")'
singularity exec -B /data/gpfs "$SIF" "$PY" "$SOURCE/revision/scripts/evidence/b6_build_c0.py" \
  --roles "$PREP/inputs/DATASET_ROLES.longread_v1.tsv" \
  --est-roles "$PREP/inputs/DATASET_ROLES.est_v1.tsv" \
  --species Sitalica --source est --align-root "$RUN/align" \
  --audit-root "$ROOT/evidence/audit_align_v1" \
  --corpus "$ROOT/training_input/db/b5_full_a40_v1.db" \
  --splits "$ROOT/training_input/splits/b5_orthogroup_split_v1.tsv" \
  --flags "$ROOT/training_input/qc/Sitalica.geenuff_flags.tsv" \
  --build-flags "$ROOT/training_input/qc/Sitalica.geenuff_flags.tsv" "$ROOT/training_input/qc/Sitalica.swissprot_flags.tsv" \
  --reference "$ROOT/training_input/genomes/Sitalica_312_v2.fa" \
  --gff "$ROOT/genomes/Sitalica_312_v2.2.gene_exons.clean.gff3" \
  --provenance "$RUN/tool_provenance.json" --seed 123 --samtools "$SAMTOOLS" \
  --memory-limit 8GB --threads 4 --out "$RUN/out/Sitalica.est.c0.db" --qc "$RUN/out/Sitalica.est.qc.json"
