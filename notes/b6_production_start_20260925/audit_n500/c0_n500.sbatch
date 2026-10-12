#!/bin/bash
#SBATCH --job-name=b6_c0_n500
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Real C0 rebuild with audit N=500 (author decision 17) for one species x source; code source_v3 (0fc80de).
# Usage: sbatch [--mem/--time] c0_n100.sbatch <species> <est|pacbio|ont> <fasta> <gff> <duckdb_memory_limit>
set -euo pipefail
SP=$1 SRC=$2 FASTA=$3 GFF=$4 DBMEM=$5
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
RUN=$PREP/c0/${SP}_${SRC}_n500
SOURCE=$PREP/source_v3
SIF=/data/gpfs/home/wyim/scratch/singularity/micromamba.sif
PY=/data/gpfs/home/wyim/micromamba/bin/python
SAMTOOLS=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/samtools
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONNOUSERSITE
/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python "$PREP/c0_view_n500.py" "$SP" "$SRC"
test ! -e "$RUN/out/$SP.$SRC.c0.db"
singularity exec -B /data/gpfs "$SIF" "$PY" -c 'import duckdb,pyarrow,sys; sys.exit(0 if (duckdb.__version__,pyarrow.__version__) == ("1.5.5","25.0.1") else "REFUSED C0 runtime versions")'
singularity exec -B /data/gpfs "$SIF" "$PY" "$SOURCE/revision/scripts/evidence/b6_build_c0.py" \
  --roles "$PREP/inputs/DATASET_ROLES.longread_v1.tsv" \
  --est-roles "$PREP/inputs/DATASET_ROLES.est_v1.tsv" \
  --species "$SP" --source "$SRC" --align-root "$RUN/align" \
  --audit-root "$ROOT/evidence/audit_align_n500" --audit-n 500 \
  --corpus "$ROOT/training_input/db/b5_full_a40_v1.db" \
  --splits "$ROOT/training_input/splits/b5_orthogroup_split_v1.tsv" \
  --flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" \
  --build-flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" "$ROOT/training_input/qc/$SP.swissprot_flags.tsv" \
  --reference "$FASTA" --gff "$GFF" \
  --provenance "$RUN/tool_provenance.json" --seed 123 --samtools "$SAMTOOLS" \
  --memory-limit "$DBMEM" --threads 16 --out "$RUN/out/$SP.$SRC.c0.db" --qc "$RUN/out/$SP.$SRC.qc.json"
