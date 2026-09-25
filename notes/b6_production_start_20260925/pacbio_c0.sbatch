#!/bin/bash
#SBATCH --job-name=b6_pb_c0
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Real PacBio C0 build for one species; same runtime/inputs as the verified Sitalica run 6172228.
# Usage: sbatch [--mem/--time overrides] pacbio_c0.sbatch <species> <fasta> <gff> <duckdb_memory_limit>
set -euo pipefail
SP=$1 FASTA=$2 GFF=$3 DBMEM=$4
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
RUN=$PREP/c0/${SP}_pacbio_v1
SOURCE=$PREP/source
SIF=/data/gpfs/home/wyim/scratch/singularity/micromamba.sif
PY=/data/gpfs/home/wyim/micromamba/bin/python
SAMTOOLS=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/samtools
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONNOUSERSITE
test -d "$RUN/align/$SP" && test -s "$RUN/tool_provenance.json"
test ! -e "$RUN/out/$SP.pacbio.c0.db"
singularity exec -B /data/gpfs "$SIF" "$PY" -c 'import duckdb,pyarrow,sys; sys.exit(0 if (duckdb.__version__,pyarrow.__version__) == ("1.5.5","25.0.1") else "REFUSED C0 runtime versions")'
singularity exec -B /data/gpfs "$SIF" "$PY" "$SOURCE/revision/scripts/evidence/b6_build_c0.py" \
  --roles "$PREP/inputs/DATASET_ROLES.longread_v1.tsv" \
  --est-roles "$PREP/inputs/DATASET_ROLES.est_v1.tsv" \
  --species "$SP" --source pacbio --align-root "$RUN/align" \
  --audit-root "$ROOT/evidence/audit_align_v1" \
  --corpus "$ROOT/training_input/db/b5_full_a40_v1.db" \
  --splits "$ROOT/training_input/splits/b5_orthogroup_split_v1.tsv" \
  --flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" \
  --build-flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" "$ROOT/training_input/qc/$SP.swissprot_flags.tsv" \
  --reference "$FASTA" --gff "$GFF" \
  --provenance "$RUN/tool_provenance.json" --seed 123 --samtools "$SAMTOOLS" \
  --memory-limit "$DBMEM" --threads 16 --out "$RUN/out/$SP.pacbio.c0.db" --qc "$RUN/out/$SP.pacbio.qc.json"
