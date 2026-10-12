#!/bin/bash
#SBATCH --job-name=b6_zero
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Route A vs B zero list (b6_routea_agreement.py) for one species and population.
# accept tables: A19-filtered (a19_filter_accept.py). HSP: hsp_v3. GFF: protein_alignments_a19.
# Usage: sbatch zero_list.sbatch <species> <est_c0_db> <protein_c0_db> <population>
set -euo pipefail
SP=$1 ESTC0=$2 PROTC0=$3 POP=$4
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v5
A19=$ROOT/evidence/protein_alignments_a19/$SP
OUT=$PREP/zero/$SP
mkdir -p "$OUT"
test -s "$A19/accept/ACCEPT_DONE"
test ! -e "$OUT/$SP.$POP.zero.tsv" && test ! -e "$OUT/$SP.$POP.zero.qc.json"
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  "$SOURCE/revision/scripts/evidence/b6_routea_agreement.py" --species "$SP" \
  --route-b "$PREP/hsp_v3/$SP/$SP.est_protein_hsp.tsv" --miniprot-gff "$A19/$SP.protein.gff.gz" \
  --accepted "$A19/accept/$SP.alignments.tsv" --mapping-ambiguous "$A19/accept/$SP.mapping_ambiguous.txt" \
  --est-c0 "$ESTC0" --protein-c0 "$PROTC0" --population "$POP" \
  --out-intervals "$OUT/$SP.$POP.zero.tsv" --qc "$OUT/$SP.$POP.zero.qc.json"
sha256sum "$OUT/$SP.$POP.zero.tsv" "$OUT/$SP.$POP.zero.qc.json"
