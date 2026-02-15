#!/bin/bash
# run_all_genomes.sh
# Process all genome files with TransGenic prompt mode

DATA_DIR=~/data/genomes
BATCH_SIZE=96

# Define genome pairs: "fasta:gff3"
declare -a PAIRS=(
    "Athaliana_167_TAIR10.fa:Athaliana_167_TAIR10.gene.clean.gff3"
    "Bdistachyon_314_v3.0.fa:Bdistachyon_314_v3.1.gene_exons.clean.gff3"
    "BrapaO_302V_711_v1.0.fa:BrapaO_302V_711_v1.1.gene.gff3"
    "Gmax_880_v6.0.fa:Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3"
    "Lsativa_467_v8.fa:Lsativa_467_v5.gene_exons.gff3"
    "Osativa_323_v7.0.fa:Osativa_323_v7.0.gene_exons.exon.gff3"
    "Ppatens_318_v3.fa:Ppatens_318_v3.3.gene_exons.clean.gff3"
    "Ptrichocarpa_533_v4.0.fa:Ptrichocarpa_533_v4.1.gene_exons.clean.gff3"
    "Sbicolor_730_v5.0.fa:Sbicolor_730_v5.1.gene_exons.clean.gff3"
    "Sitalica_312_v2.fa:Sitalica_312_v2.2.gene_exons.clean.gff3"
    "Slycopersicum_796_ITAG5.0.fa:Slycopersicum_796_ITAG5.0.gene.gff3"
    "Vvinifera_T2T_ref.fa:Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3"
    "Zmays_493_APGv4.fa:Zmays_493_RefGen_V4.gene_exons.exon.gff3"
)

for pair in "${PAIRS[@]}"; do
    FASTA="${pair%%:*}"
    GFF="${pair##*:}"
    OUTPUT="${GFF%.gff3}_prompt.gff3"

    echo "=========================================="
    echo "Processing: $GFF"
    echo "=========================================="

    python examples/prompt_mode.py \
        --genome "$DATA_DIR/$FASTA" \
        --gff "$DATA_DIR/$GFF" \
        --output "$DATA_DIR/$OUTPUT" \
        --batch-size $BATCH_SIZE

    echo "Done: $OUTPUT"
    echo ""
done

echo "All genomes processed!"
