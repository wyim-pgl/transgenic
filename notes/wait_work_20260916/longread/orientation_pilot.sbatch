#!/usr/bin/env bash
#SBATCH --job-name=mvp_ont_orientation
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Controller supplies --output/--error in an already-created NEW release directory.
# NO ARRAY; no retry. Review before submission. This job does not align a full run.
set -euo pipefail
: "${RELEASE:?controller-reviewed release directory required}"
: "${OUT:?fresh exclusive output directory required}"
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
"$PY" "$RELEASE/gate.py" \
  --root "$ROOT" --run SRR28341943 \
  --reads "$ROOT/evidence/ont/Athaliana_FLIC_PRJNA1087576/SRR28341943.fa.gz" \
  --fasta "$ROOT/genomes/Athaliana_167_TAIR10.fa" \
  --gff "$ROOT/genomes/Athaliana_167_TAIR10.gene.clean.gff3" \
  --authority "$RELEASE/authority.tsv" \
  --source "$RELEASE/frozen" \
  --panels "$RELEASE/frozen/revision/protocols/b6_panels_v1.tsv" \
  --bin "$BIN" --out "$OUT"
