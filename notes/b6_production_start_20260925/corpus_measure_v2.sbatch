#!/bin/bash
#SBATCH --job-name=b6_corpus_measure2
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# Runbook step B (measure only, writes nothing but the report): WP2 rule-5 repair of the A40 corpus, D1 geometry (A),
# per-species A22/A30 flags (code c422e5e = source_v5; projects the decision-18 drop, train rows and N_pilot). --apply follows only if both row-set events are zero.
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic; PREP=$ROOT/evidence/b6_prepare_20260925; SOURCE=$PREP/source_v5; Q=$ROOT/training_input/qc
export PYTHONPATH="$PREP/duckdb_runtime:$SOURCE/src:$SOURCE/revision/scripts:$SOURCE/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1
SP=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
declare -A FA=([Athaliana]=Athaliana_167_TAIR10.fa [Bdistachyon]=Bdistachyon_314_v3.0.fa [Gmax]=Gmax_880_v6.0.fa [Osativa]=Osativa_323_v7.0.fa [Ppatens]=Ppatens_318_v3.fa [Ptrichocarpa]=Ptrichocarpa_533_v4.0.fa [Sbicolor]=Sbicolor_730_v5.0.fa [Sitalica]=Sitalica_312_v2.fa [Vvinifera]=Vvinifera_T2T_ref.fa)
declare -A GF=([Athaliana]=Athaliana_167_TAIR10.gene.clean.gff3 [Bdistachyon]=Bdistachyon_314_v3.1.gene_exons.clean.gff3 [Gmax]=Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3 [Osativa]=Osativa_323_v7.0.gene_exons.exon.gff3 [Ppatens]=Ppatens_318_v3.3.gene_exons.clean.gff3 [Ptrichocarpa]=Ptrichocarpa_533_v4.1.gene_exons.clean.gff3 [Sbicolor]=Sbicolor_730_v5.1.gene_exons.clean.gff3 [Sitalica]=Sitalica_312_v2.2.gene_exons.clean.gff3 [Vvinifera]=Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3)
A=(); M=()
for s in "${SP[@]}"; do A+=(--flags "$s=$Q/$s.geenuff_flags.tsv" --build-flags "$s=$Q/$s.geenuff_flags.tsv" "$s=$Q/$s.swissprot_flags.tsv"); M+=("$PREP/intervals/$s.intervals_v2.jsonl"); A+=(--reference "$s=$ROOT/training_input/genomes/${FA[$s]}" --gff "$s=$ROOT/genomes/${GF[$s]}"); done
mkdir -p $PREP/corpus; test ! -e $PREP/corpus/measure_v2.json  # first attempt 6172912 failed on the build_manifest GFF path (no report written)
singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python \
  $SOURCE/revision/scripts/evidence/b6_build_corpus.py --base $ROOT/training_input/db/b5_full_a40_v1.db \
  --splits $ROOT/training_input/splits/b5_orthogroup_split_v1.tsv "${A[@]}" --seed 123 \
  --interval-manifest "${M[@]}" --measure-only --report $PREP/corpus/measure_v2.json
