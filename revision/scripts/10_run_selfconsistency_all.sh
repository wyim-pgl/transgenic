#!/bin/bash
# Run 07_selfconsistency_stats.py for 13 species x 6 TransGenic variants + references.
set -uo pipefail

BASE=/data/gpfs/assoc/pgl/data/Transgenic
CMP=$BASE/transgenic_comparison
GEN=$BASE/genomes
REV=$BASE/transgenic/revision
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
OUT=$REV/results/selfconsistency
mkdir -p "$OUT"

declare -A FA=(
  [A_thaliana]=Athaliana_167_TAIR10.fa
  [B_distachyon]=Bdistachyon_314_v3.0.fa
  [B_rapa]=BrapaO_302V_711_v1.0.fa
  [G_max]=Gmax_880_v6.0.fa
  [L_sativa]=Lsativa_467_v8.fa
  [O_sativa]=Osativa_323_v7.0.fa
  [P_patens]=Ppatens_318_v3.fa
  [P_trichocarpa]=Ptrichocarpa_533_v4.0.fa
  [S_bicolor]=Sbicolor_730_v5.0.fa
  [S_italica]=Sitalica_312_v2.fa
  [S_lycopersicum]=Slycopersicum_796_ITAG5.0.fa
  [V_vinifera]=Vvinifera_T2T_ref.fa
  [Z_mays]=Zmays_493_APGv4.fa
)
declare -A REF=(
  [A_thaliana]=Athaliana_167_TAIR10.gene.clean.gff3
  [B_distachyon]=Bdistachyon_314_v3.1.gene_exons.clean.gff3
  [B_rapa]=BrapaO_302V_711_v1.1.gene.gff3
  [G_max]=Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3
  [L_sativa]=Lsativa_467_v5.gene_exons.gff3
  [O_sativa]=Osativa_323_v7.0.gene_exons.exon.gff3
  [P_patens]=Ppatens_318_v3.3.gene_exons.clean.gff3
  [P_trichocarpa]=Ptrichocarpa_533_v4.1.gene_exons.clean.gff3
  [S_bicolor]=Sbicolor_730_v5.1.gene_exons.clean.gff3
  [S_italica]=Sitalica_312_v2.2.gene_exons.clean.gff3
  [S_lycopersicum]=Slycopersicum_796_ITAG5.0.gene.gff3
  [V_vinifera]=Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3
  [Z_mays]=Zmays_493_RefGen_V4.gene_exons.exon.gff3
)

SPECIES="A_thaliana B_distachyon B_rapa G_max L_sativa O_sativa P_patens P_trichocarpa S_bicolor S_italica S_lycopersicum V_vinifera Z_mays"
VARIANTS="transgenic160M transgenic160M_prompt_denovo transgenic160Mprompt transgenic400M transgenic400M_prompt_denovo transgenic400Mprompt"

CMDS=$OUT/commands.txt
> "$CMDS"
for sp in $SPECIES; do
  echo "$PY $REV/scripts/07_selfconsistency_stats.py --pred $CMP/reference_annotations/${REF[$sp]} --fasta $GEN/${FA[$sp]} --out $OUT/${sp}_REF > $OUT/${sp}_REF.log 2>&1" >> "$CMDS"
  for v in $VARIANTS; do
    pred=$CMP/standardized_results/${sp}_${v}.gff3
    [ -f "$pred" ] || { echo "MISSING $pred"; continue; }
    echo "$PY $REV/scripts/07_selfconsistency_stats.py --pred $pred --fasta $GEN/${FA[$sp]} --ref $CMP/reference_annotations/${REF[$sp]} --out $OUT/${sp}_${v} > $OUT/${sp}_${v}.log 2>&1" >> "$CMDS"
  done
done

xargs -P 8 -I CMD bash -c CMD < "$CMDS"
echo "SELFCONSISTENCY ALL DONE"
ls "$OUT"/*.json 2>/dev/null | wc -l