#!/bin/bash
#SBATCH --job-name=b6_hsp
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Route-B inputs for one species: EST molecule map from the verified EST C0, then the est_protein_hsp projection of
# the production blastx outputs (orthodb + swissprot). Lineage: swissprot lineage.tsv (orthodb rows are a verified
# identical subset). No new blastx is run. Code 0fc80de: masked X, per-row length contract, non-replaying rows dropped (author decisions 5, 10, 11).
# Usage: sbatch hsp_project.sbatch <species> <blastx_dir> <est_c0_db>
set -euo pipefail
SP=$1 BX=$2 ESTC0=$3
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
OUT=$PREP/hsp_v3/$SP
SOURCE=$PREP/source_v3
SIF=/data/gpfs/home/wyim/scratch/singularity/micromamba.sif
PY=/data/gpfs/home/wyim/micromamba/bin/python
SAMTOOLS=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/samtools
LINEAGE=$ROOT/evidence/est_blastx/preflight_20260914/swissprot/lineage.tsv
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONNOUSERSITE
sha256sum "$LINEAGE" | grep -q '^402bca970569cc2c'
for s in orthodb swissprot; do test -f "$BX/$s/DONE" && test -s "$BX/$s/$s.raw13.tsv"; done
mkdir -p "$OUT"
test ! -e "$OUT/$SP.molecule_map.tsv" && test ! -e "$OUT/$SP.est_protein_hsp.tsv"
X() { singularity exec -B /data/gpfs "$SIF" "$PY" "$@"; }
X "$SOURCE/revision/scripts/evidence/b6_est_molecule_map.py" --est-c0 "$ESTC0" --species "$SP" --out "$OUT/$SP.molecule_map.tsv"
X "$SOURCE/revision/scripts/evidence/b6_hsp_project.py" --species "$SP" \
  --raw13 "orthodb=$BX/orthodb/orthodb.raw13.tsv" --raw13 "swissprot=$BX/swissprot/swissprot.raw13.tsv" \
  --protein-fasta "orthodb=$BX/orthodb/orthodb.filtered.fa" --protein-fasta "swissprot=$BX/swissprot/swissprot.filtered.fa" \
  --blastx-command "orthodb=$BX/orthodb/commands.json" --blastx-command "swissprot=$BX/swissprot/commands.json" \
  --diamond-version 2.1.24.178 \
  --est-fasta "$ROOT/evidence/est/$SP/univec/est.univec.fa.gz" \
  --alignment "$ROOT/evidence/est_align/$SP/$SP.est.bam" --samtools "$SAMTOOLS" \
  --molecule-map "$OUT/$SP.molecule_map.tsv" --lineage "$LINEAGE" \
  --out "$OUT/$SP.est_protein_hsp.tsv" --qc "$OUT/$SP.est_protein_hsp.qc.json"
