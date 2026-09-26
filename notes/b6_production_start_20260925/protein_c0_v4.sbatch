#!/bin/bash
#SBATCH --job-name=b6_prot_c0
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Real protein C0 (miniprot + route-B est_protein_hsp) for one species. Roles: longread roles + protein rows
# (author decision 2026-09-25, decision 6). HSP from hsp_project_v3 (code 0fc80de). miniprot GFF A19-filtered (same-species queries dropped).
# Usage: sbatch protein_c0.sbatch <species> <fasta> <gff> <est_c0_db> <duckdb_memory_limit>
set -euo pipefail
SP=$1 FASTA=$2 GFF=$3 ESTC0=$4 DBMEM=$5
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
RUN=$PREP/c0/${SP}_protein_v1
SOURCE=$PREP/source_v3
SIF=/data/gpfs/home/wyim/scratch/singularity/micromamba.sif
PY=/data/gpfs/home/wyim/micromamba/bin/python
SAMTOOLS=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/samtools
HSP=$PREP/hsp_v3/$SP/$SP.est_protein_hsp.tsv
LINEAGE=$ROOT/evidence/est_blastx/preflight_20260914/swissprot/lineage.tsv
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONNOUSERSITE
test -s "$HSP" && test -s "$RUN/tool_provenance.json"
test ! -e "$RUN/out/$SP.protein.c0.db"
sha256sum "$LINEAGE" | grep -q '^402bca970569cc2c'
singularity exec -B /data/gpfs "$SIF" "$PY" -c 'import duckdb,pyarrow,sys; sys.exit(0 if (duckdb.__version__,pyarrow.__version__) == ("1.5.5","25.0.1") else "REFUSED C0 runtime versions")'
singularity exec -B /data/gpfs "$SIF" "$PY" "$SOURCE/revision/scripts/evidence/b6_build_c0.py" \
  --roles "$PREP/inputs/DATASET_ROLES.longread_protein_v1.tsv" \
  --est-roles "$PREP/inputs/DATASET_ROLES.est_v1.tsv" \
  --species "$SP" --source protein --align-root "$ROOT/evidence/protein_alignments_a19" \
  --audit-root "$ROOT/evidence/audit_align_v1" \
  --corpus "$ROOT/training_input/db/b5_full_a40_v1.db" \
  --splits "$ROOT/training_input/splits/b5_orthogroup_split_v1.tsv" \
  --flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" \
  --build-flags "$ROOT/training_input/qc/$SP.geenuff_flags.tsv" "$ROOT/training_input/qc/$SP.swissprot_flags.tsv" \
  --reference "$FASTA" --gff "$GFF" \
  --hsp "$HSP" --est-c0 "$ESTC0" --lineage "$LINEAGE" \
  --provenance "$RUN/tool_provenance.json" --seed 123 --samtools "$SAMTOOLS" \
  --memory-limit "$DBMEM" --threads 8 --out "$RUN/out/$SP.protein.c0.db" --qc "$RUN/out/$SP.protein.qc.json"
