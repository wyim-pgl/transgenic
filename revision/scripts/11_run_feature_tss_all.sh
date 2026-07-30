#!/bin/bash
# Run 08_feature_tss_stats.py: feature counts + AS stats on all references,
# TSS/TES accuracy for all TransGenic variants.
set -uo pipefail

BASE=/data/gpfs/assoc/pgl/data/Transgenic
CMP=$BASE/transgenic_comparison
REV=$BASE/transgenic/revision
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
OUT=$REV/results/feature_tss
mkdir -p "$OUT"

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
  ref=$CMP/reference_annotations/${REF[$sp]}
  echo "$PY $REV/scripts/08_feature_tss_stats.py --mode features --gff $ref --out $OUT/features_${sp} > $OUT/features_${sp}.log 2>&1" >> "$CMDS"
  echo "$PY $REV/scripts/08_feature_tss_stats.py --mode asstats --gff $ref --out $OUT/asstats_${sp} > $OUT/asstats_${sp}.log 2>&1" >> "$CMDS"
  for v in $VARIANTS; do
    pred=$CMP/standardized_results/${sp}_${v}.gff3
    [ -f "$pred" ] || continue
    echo "$PY $REV/scripts/08_feature_tss_stats.py --mode tss_tes --pred $pred --ref $ref --out $OUT/tsstes_${sp}_${v} > $OUT/tsstes_${sp}_${v}.log 2>&1" >> "$CMDS"
  done
done
# AtRTD3 reference stats (Arabidopsis comprehensive AS)
echo "$PY $REV/scripts/08_feature_tss_stats.py --mode features --gff $REV/data/AtRTD3/AtRTD3.gtf --out $OUT/features_AtRTD3 > $OUT/features_AtRTD3.log 2>&1" >> "$CMDS"
echo "$PY $REV/scripts/08_feature_tss_stats.py --mode asstats --gff $REV/data/AtRTD3/AtRTD3.gtf --out $OUT/asstats_AtRTD3 > $OUT/asstats_AtRTD3.log 2>&1" >> "$CMDS"

xargs -P 8 -I CMD bash -c CMD < "$CMDS"
echo "FEATURE/TSS ALL DONE"
ls "$OUT"/*.json 2>/dev/null | wc -l