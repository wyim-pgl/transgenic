#!/bin/bash
# The missing factorial cell: TAIR10 CDS carrying Helixer's UTR (arm built by
# 53_build_tair10cds_helixerutr_arm.py). Registered with 33's pipeline by 54_run_extra_arm.py
# rather than by editing 33's TOOLS list.
#
# Writes only paths containing `tair10cdshelixerutr`; no existing prediction or result file
# of any other arm is touched.
set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1

TOOL=tair10cdshelixerutr
BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
LOG=$BENCH/logs/arm_${TOOL}_$(date +%Y%m%d_%H%M%S).log
STATUS=$BENCH/results/frame_experiment_status.txt
LOCK=$BENCH/.arms_running.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "another run holds the lock (pid $(cat "$LOCK")) — refusing to start" | tee -a "$LOG"
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
say "=== $TOOL started (pid $$) ==="

python3 revision/scripts/54_run_extra_arm.py --tool "$TOOL" \
    --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
say "<<< pipeline returned $?"

OUT=$BENCH/predictions/${TOOL}_completed.gff3
if [ ! -s "$OUT" ]; then
    say "NO OUTPUT — stopping before scoring"
    echo "$TOOL: NO OUTPUT" >> "$STATUS"
    exit 1
fi

say ">>> scoring unfiltered"
python3 revision/scripts/34_score_as_additions.py \
    --input "$BENCH/inputs/${TOOL}_Athaliana.gff3" --output "$OUT" \
    --tool "$TOOL" --json "$BENCH/results/${TOOL}_as_additions.json" >> "$LOG" 2>&1

# The structural filter (36) then the same scorer (34) on what survives — the "filtered"
# column every other arm reports.
say ">>> structural filter"
python3 revision/scripts/36_filter_additions_structurally.py \
    --input "$BENCH/inputs/${TOOL}_Athaliana.gff3" --output "$OUT" \
    --filtered "$BENCH/predictions/${TOOL}_filtered.gff3" \
    --json "$BENCH/results/${TOOL}_filter.json" >> "$LOG" 2>&1

if [ -s "$BENCH/predictions/${TOOL}_filtered.gff3" ]; then
    say ">>> scoring filtered"
    python3 revision/scripts/34_score_as_additions.py \
        --input "$BENCH/inputs/${TOOL}_Athaliana.gff3" \
        --output "$BENCH/predictions/${TOOL}_filtered.gff3" \
        --tool "${TOOL}_filtered" \
        --json "$BENCH/results/${TOOL}_filtered_as_additions.json" >> "$LOG" 2>&1
fi

line=$(python3 -c "
import json
d = json.load(open('$BENCH/results/${TOOL}_as_additions.json'))
print(f\"added={d['added_structures']:,} TAIR10_alt={d['precision_vs_TAIR10_alternatives_pct']}% \"
      f\"AtRTD3={d['precision_vs_AtRTD3_pct']}% any={d['precision_vs_TAIR10_any_transcript_pct']}% \"
      f\"recall={d['recall_of_TAIR10_alternatives_pct']}%\")
" 2>>"$LOG")
say "SCORED $TOOL: $line"
echo "$TOOL: $line" >> "$STATUS"
say "=== $TOOL finished ==="
