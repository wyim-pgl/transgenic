#!/bin/bash
# run_self_prompted.sh
# Two-stage self-prompted TransGenic inference:
#   1) de novo annotation to obtain a primary transcript
#   2) prompt completion using the de novo output as the prompt
#
# Usage:
#   bash examples/run_self_prompted.sh GENOME.fa GENES.gff3 OUTPUT_PREFIX [MODEL]
#
# Example:
#   bash examples/run_self_prompted.sh ATH_Chr4.fas ATH_Chr4.sorted.gff3 chr4_self

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 GENOME.fa GENES.gff3 OUTPUT_PREFIX [MODEL]"
    exit 1
fi

GENOME="$1"
GFF="$2"
PREFIX="$3"
MODEL="${4:-jlomas/HyenaTransgenic-768L12A6-400M}"

# Resolve repository root (assumes this script is in examples/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DENOVO_GFF="${PREFIX}_denovo.gff3"
SELF_GFF="${PREFIX}_self_prompted.gff3"

echo "=== Stage 1: de novo annotation ==="
python "$ROOT_DIR/src/run_genome_annotation.py" \
    "$GENOME" "$GFF" \
    -o "$DENOVO_GFF" \
    --device cuda

echo "=== Stage 2: self-prompted isoform completion ==="
python "$SCRIPT_DIR/prompt_mode.py" \
    --genome "$GENOME" \
    --gff "$DENOVO_GFF" \
    --output "$SELF_GFF" \
    --model "$MODEL" \
    --batch-size 96

echo "Done: $SELF_GFF"
