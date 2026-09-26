#!/bin/bash
#SBATCH --job-name=b6_intervals
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%A_%a.out
# Runbook step A: b6-intervals-v2 manifest per species (D1 geometry A, author decision 16); must equal the
# interval_manifest_sha256 each C0 recorded. Code source_v3 (0fc80de).
set -euo pipefail
SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
FA=(Athaliana_167_TAIR10.fa Bdistachyon_314_v3.0.fa Gmax_880_v6.0.fa Osativa_323_v7.0.fa Ppatens_318_v3.fa Ptrichocarpa_533_v4.0.fa Sbicolor_730_v5.0.fa Sitalica_312_v2.fa Vvinifera_T2T_ref.fa)
GF=(Athaliana_167_TAIR10.gene.clean.gff3 Bdistachyon_314_v3.1.gene_exons.clean.gff3 Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3 Osativa_323_v7.0.gene_exons.exon.gff3 Ppatens_318_v3.3.gene_exons.clean.gff3 Ptrichocarpa_533_v4.1.gene_exons.clean.gff3 Sbicolor_730_v5.1.gene_exons.clean.gff3 Sitalica_312_v2.2.gene_exons.clean.gff3 Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3)
i=$SLURM_ARRAY_TASK_ID; SP=${SPECIES[$i]}
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v3; OUT=$PREP/intervals
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT; test ! -e $OUT/$SP.intervals.tsv
X() { singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python "$@"; }
X $SOURCE/revision/scripts/evidence/b6_leakage_intervals.py --corpus $ROOT/training_input/db/b5_full_a40_v1.db --species $SP \
  --splits $ROOT/training_input/splits/b5_orthogroup_split_v1.tsv --flags $ROOT/training_input/qc/$SP.geenuff_flags.tsv \
  --build-flags $ROOT/training_input/qc/$SP.geenuff_flags.tsv $ROOT/training_input/qc/$SP.swissprot_flags.tsv \
  --reference $ROOT/training_input/genomes/${FA[$i]} --gff $ROOT/genomes/${GF[$i]} --seed 123 \
  --out $OUT/$SP.intervals.tsv --manifest-out $OUT/$SP.intervals_v2.jsonl
