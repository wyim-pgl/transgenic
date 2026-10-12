#!/bin/bash
#SBATCH --job-name=b6_pcr_groups
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
set -euo pipefail
PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925
export PYTHONPATH="$PREP/duckdb_runtime"; export PYTHONDONTWRITEBYTECODE=1
X(){ singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python "$@"; }
echo "== Athaliana staging (failed run 6173686)"; X $PREP/pcr_groups.py $PREP/c0/Athaliana_ont_v2/out/Athaliana.ont.c0.db.partial-fh429_et.stg
