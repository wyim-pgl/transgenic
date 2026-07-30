#!/bin/bash
# Driver: run AS evaluation (02 gffcompare + 03 splice events) for
# TransGenic 400M predictions (prompted + de novo) against TAIR10 and AtRTD3.
set -uo pipefail

REV=/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision
CMP=/data/gpfs/assoc/pgl/data/Transgenic/transgenic_comparison
ENV=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin
export PATH="$ENV:$PATH"

TAIR10=$REV/data/TAIR10/TAIR10.gtf
ATRTD3=$REV/data/AtRTD3/AtRTD3.gtf

declare -A PREDS=(
  [prompted400M]=$CMP/standardized_results/A_thaliana_transgenic400Mprompt.gff3
  [denovo400M]=$CMP/standardized_results/A_thaliana_transgenic400M.gff3
)

for pname in "${!PREDS[@]}"; do
  pred=${PREDS[$pname]}
  for refname in TAIR10 AtRTD3; do
    ref=$TAIR10; [ "$refname" = "AtRTD3" ] && ref=$ATRTD3
    out=$REV/results/${pname}_vs_${refname}
    mkdir -p "$out"
    echo "=== $pname vs $refname : gffcompare ==="
    python $REV/scripts/02_gffcompare_analysis.py -r "$ref" -p "$pred" -o "$out" --prefix "${pname}_vs_${refname}" \
      > "$out/gffcompare_analysis.log" 2>&1 || echo "FAILED 02 $pname $refname"
    echo "=== $pname vs $refname : splice events ==="
    python $REV/scripts/03_splice_event_detection.py -r "$ref" -p "$pred" -o "$out" --prefix "splice_events" \
      > "$out/splice_events.log" 2>&1 || echo "FAILED 03 $pname $refname"
  done
done
echo "ALL DONE"