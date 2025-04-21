#!/bin/bash

# This script is used to prepare a GFF3 file for segmentation. It will first sort the gff and add both
# exons and introns. Then splice junctions are added. The output will be a GFF3 file with exons, introns,
# and splice junctions.

# Usage: bash prepGffForSegmentation.sh <gff3_file>

gff=${1%.*}

agat_sp_add_introns.pl --gff $gff.gff3 --out $gff.exon.gff3
python hisat2_extract_splice_sites_gff3.py $gff.exon.gff3 -f exon -a > splice_sites.txt

#rm $gff.exon.gff3 $gff.exon.splice.gff3

printf "\nSegmentation GFF3 file created: $gff.segmentation.gff3 \nSplice sites file created: splice_sites.txt\n"