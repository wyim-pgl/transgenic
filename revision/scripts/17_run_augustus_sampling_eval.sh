#!/bin/bash
# AUGUSTUS posterior-sampling baseline evaluation (issue4, Reviewer 2).
#
# Mirrors the TransGenic prompted evaluation exactly:
#   1) gffcompare v0.12.6 against the alt-only references (primary transcript
#      removed) — same commands as results/altonly/*
#   2) full-reference transcript-level + splice-event analysis via
#      02_gffcompare_analysis.py / 03_splice_event_detection.py — same as
#      09_run_as_eval_driver.sh
#
# Input: standardized_results/A_thaliana_augustusSampling.gff3
# (per-locus `augustus --species=arabidopsis --sample=100
#  --alternatives-from-sampling=true --noInFrameStop=true`, AUGUSTUS v3.5.0,
#  primary + sampled transcripts kept, mirroring the prompted TransGenic
#  prediction files which also contain the primary transcript)
set -uo pipefail

REV=/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision
CMP=/data/gpfs/assoc/pgl/data/Transgenic/transgenic_comparison
ENV=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin
export PATH="$ENV:$PATH"

PRED=$CMP/standardized_results/A_thaliana_augustusSampling.gff3
TAIR10=$REV/data/TAIR10/TAIR10.gtf
ATRTD3=$REV/data/AtRTD3/AtRTD3.gtf
TAIR10_ALT=$REV/data/TAIR10/TAIR10.altonly.gtf
ATRTD3_ALT=$REV/data/AtRTD3/AtRTD3.altonly.gtf

[ -s "$PRED" ] || { echo "missing prediction: $PRED"; exit 1; }

# 1) alt-only gffcompare (same flags as results/altonly/*)
mkdir -p "$REV/results/altonly"
for refname in TAIR10 AtRTD3; do
    ref=$TAIR10_ALT; [ "$refname" = "AtRTD3" ] && ref=$ATRTD3_ALT
    out=$REV/results/altonly/A_thaliana_augustusSampling_vs_${refname}
    echo "=== altonly gffcompare vs $refname ==="
    gffcompare -r "$ref" -o "$out" "$PRED" || echo "FAILED altonly $refname"
done

# 2) full-reference 02/03 analyses (same as 09_run_as_eval_driver.sh)
for refname in TAIR10 AtRTD3; do
    ref=$TAIR10; [ "$refname" = "AtRTD3" ] && ref=$ATRTD3
    out=$REV/results/augustusSampling_vs_${refname}
    mkdir -p "$out"
    echo "=== augustusSampling vs $refname : gffcompare ==="
    python $REV/scripts/02_gffcompare_analysis.py -r "$ref" -p "$PRED" -o "$out" --prefix "augustusSampling_vs_${refname}" \
      > "$out/gffcompare_analysis.log" 2>&1 || echo "FAILED 02 $refname"
    echo "=== augustusSampling vs $refname : splice events ==="
    python $REV/scripts/03_splice_event_detection.py -r "$ref" -p "$PRED" -o "$out" --prefix "splice_events" \
      > "$out/splice_events.log" 2>&1 || echo "FAILED 03 $refname"
done
echo "ALL DONE"
