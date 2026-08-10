#!/bin/bash
# arm G: Helixer's CDS with TAIR10's UTR — decides where UTR-correction effort would go.
set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1
BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
LOG=$BENCH/logs/arm_g_$(date +%Y%m%d_%H%M%S).log
STATUS=$BENCH/results/frame_experiment_status.txt
LOCK=$BENCH/.arms_running.lock

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "another run holds the lock" >> "$LOG"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

echo "[$(date +%H:%M:%S)] running helixertairutr" >> "$LOG"
python3 revision/scripts/33_run_polishing_benchmark.py --tool helixertairutr \
    --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
out=$BENCH/predictions/helixertairutr_completed.gff3
if [ -s "$out" ]; then
    python3 revision/scripts/34_score_as_additions.py \
        --input $BENCH/inputs/helixertairutr_Athaliana.gff3 --output "$out" \
        --tool helixertairutr --json $BENCH/results/helixertairutr_as_additions.json >> "$LOG" 2>&1
    line=$(python3 -c "
import json
d = json.load(open('$BENCH/results/helixertairutr_as_additions.json'))
print(f\"added={d['added_structures']:,} TAIR10_alt={d['precision_vs_TAIR10_alternatives_pct']}% \"
      f\"AtRTD3={d['precision_vs_AtRTD3_pct']}% recall={d['recall_of_TAIR10_alternatives_pct']}%\")
")
    echo "helixertairutr: $line" >> "$STATUS"
    echo "[$(date +%H:%M:%S)] SCORED: $line" >> "$LOG"
else
    echo "helixertairutr: NO OUTPUT" >> "$STATUS"
fi
echo "arm G finished $(date +%H:%M)" >> "$STATUS"
