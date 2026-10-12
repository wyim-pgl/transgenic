#!/bin/bash
#SBATCH --job-name=b6_c0_rowcounts
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
set -euo pipefail
PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v5; C=$PREP/c0
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src"; export PYTHONDONTWRITEBYTECODE=1
DBS=$(ls $C/{Athaliana_est_v1,Athaliana_protein_v1,Bdistachyon_est_n100,Bdistachyon_protein_v1,Gmax_est_n100,Gmax_ont_n100,Osativa_est_v1,Osativa_protein_v1,Ppatens_est_v1,Ppatens_ont_v1,Ppatens_protein_v1,Ptrichocarpa_est_v1,Ptrichocarpa_pacbio_v1,Ptrichocarpa_ont_v1,Ptrichocarpa_protein_v1,Sbicolor_est_v1,Sbicolor_pacbio_v1,Sbicolor_protein_v1,Sitalica_ont_v2,Sitalica_protein_v1,Vvinifera_est_n500,Vvinifera_pacbio_n500,Vvinifera_ont_n500,Vvinifera_protein_v1}/out/*.c0.db $C/Sitalica_est_v1/out_arrow_retry/Sitalica.est.c0.db)
singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python $PREP/c0_rowcounts.py $DBS
