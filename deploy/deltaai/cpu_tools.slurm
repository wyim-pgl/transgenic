#!/bin/bash
#SBATCH --job-name=b1-cpu
#SBATCH --account=CHANGE_ME            # Delta CPU allocation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240g
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# CPU evidence pipeline via Biocontainers (issues #14, #25/#26, #45). Pull once:
#   apptainer pull $SCRATCH/containers/minimap2.sif   docker://quay.io/biocontainers/minimap2:2.28--h577a1d6_4
#   apptainer pull $SCRATCH/containers/miniprot.sif   docker://quay.io/biocontainers/miniprot:0.13--h577a1d6_2
#   apptainer pull $SCRATCH/containers/orthofinder.sif docker://quay.io/biocontainers/orthofinder:3.0.1b1--hdfd78af_0
# Record the exact tags and `--version` output in the corresponding issue (protocol §3.3 item 5).
set -euo pipefail
module load apptainer
WORK=$SCRATCH/transgenic; C=$SCRATCH/containers
TOOL=${TOOL:?minimap2|miniprot|orthofinder}
case "$TOOL" in
  minimap2)   apptainer exec --bind "$WORK":/work $C/minimap2.sif minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long \
                /work/genomes/${REF:?}.fa /work/evidence/${READS:?} > /work/aln/${OUT:?}.sam ;;
  miniprot)   apptainer exec --bind "$WORK":/work $C/miniprot.sif miniprot -t 32 --gff -N 20 \
                /work/genomes/${REF:?}.fa /work/proteins/orthodb_v12_viridiplantae.filtered.fa > /work/protein_alignments/${REF}.gff ;;
  orthofinder) apptainer exec --bind "$WORK":/work $C/orthofinder.sif orthofinder -t 32 -a 8 -f /work/proteomes ;;
esac
