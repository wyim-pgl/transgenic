#!/bin/bash
GENOMES_DIR=~/data/genomes
OUTPUT_DIR=./prompt_mode_output

mkdir -p $OUTPUT_DIR

# Arabidopsis thaliana
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Athaliana_167_TAIR10.fa \
    --gff $GENOMES_DIR/Athaliana_167_TAIR10.gene.clean.gff3 \
    --output $OUTPUT_DIR/Athaliana.gff

# Brachypodium distachyon
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Bdistachyon_314_v3.0.fa \
    --gff $GENOMES_DIR/Bdistachyon_314_v3.1.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Bdistachyon.gff

# Brassica rapa
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/BrapaO_302V_711_v1.0.fa \
    --gff $GENOMES_DIR/BrapaO_302V_711_v1.1.gene.gff3 \
    --output $OUTPUT_DIR/BrapaO.gff

# Glycine max (Soybean)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Gmax_880_v6.0.fa \
    --gff $GENOMES_DIR/Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Gmax.gff

# Lactuca sativa (Lettuce)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Lsativa_467_v8.fa \
    --gff $GENOMES_DIR/Lsativa_467_v5.gene_exons.gff3 \
    --output $OUTPUT_DIR/Lsativa.gff

# Oryza sativa (Rice)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Osativa_323_v7.0.fa \
    --gff $GENOMES_DIR/Osativa_323_v7.0.gene_exons.exon.gff3 \
    --output $OUTPUT_DIR/Osativa.gff

# Physcomitrella patens (Moss)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Ppatens_318_v3.fa \
    --gff $GENOMES_DIR/Ppatens_318_v3.3.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Ppatens.gff

# Populus trichocarpa (Poplar)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Ptrichocarpa_533_v4.0.fa \
    --gff $GENOMES_DIR/Ptrichocarpa_533_v4.1.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Ptrichocarpa.gff

# Sorghum bicolor
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Sbicolor_730_v5.0.fa \
    --gff $GENOMES_DIR/Sbicolor_730_v5.1.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Sbicolor.gff

# Setaria italica (Foxtail millet)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Sitalica_312_v2.fa \
    --gff $GENOMES_DIR/Sitalica_312_v2.2.gene_exons.clean.gff3 \
    --output $OUTPUT_DIR/Sitalica.gff

# Solanum lycopersicum (Tomato)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Slycopersicum_796_ITAG5.0.fa \
    --gff $GENOMES_DIR/Slycopersicum_796_ITAG5.0.gene.gff3 \
    --output $OUTPUT_DIR/Slycopersicum.gff

# Vitis vinifera (Grape)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Vvinifera_T2T_ref.fa \
    --gff $GENOMES_DIR/Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3 \
    --output $OUTPUT_DIR/Vvinifera.gff

# Zea mays (Maize)
python examples/prompt_mode.py \
    --genome $GENOMES_DIR/Zmays_493_APGv4.fa \
    --gff $GENOMES_DIR/Zmays_493_RefGen_V4.gene_exons.exon.gff3 \
    --output $OUTPUT_DIR/Zmays.gff
