#!/bin/bash
#SBATCH --job-name=b6_est_multirun
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
set -euo pipefail
PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925; C=$PREP/c0
export PYTHONPATH="$PREP/duckdb_runtime"; export PYTHONDONTWRITEBYTECODE=1
singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python $PREP/est_multirun.py \
  $C/Athaliana_est_v1/out/Athaliana.est.c0.db $C/Bdistachyon_est_n100/out/Bdistachyon.est.c0.db $C/Gmax_est_n100/out/Gmax.est.c0.db \
  $C/Osativa_est_v1/out/Osativa.est.c0.db $C/Ppatens_est_v1/out/Ppatens.est.c0.db $C/Ptrichocarpa_est_v1/out/Ptrichocarpa.est.c0.db \
  $C/Sbicolor_est_v1/out/Sbicolor.est.c0.db $C/Sitalica_est_v1/out_arrow_retry/Sitalica.est.c0.db $C/Vvinifera_est_n500/out/Vvinifera.est.c0.db
