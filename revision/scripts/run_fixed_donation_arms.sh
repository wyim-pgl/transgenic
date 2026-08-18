#!/bin/bash
# The two UTR-donation arms rebuilt with the minus-strand fix (56_rebuild_donation_arms_fixed.py),
# run back to back. Waits for whatever holds the arm lock rather than refusing, so this can be
# queued behind a run already in progress.
#
# Writes only paths containing `helixertairutr_fixed` / `annevotairutr_fixed`. The original
# arms' inputs, predictions and results are never touched — the predictions already made from
# them remain valid measurements of what those files actually contained.
set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1

BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
LOG=$BENCH/logs/fixed_donation_arms_$(date +%Y%m%d_%H%M%S).log
STATUS=$BENCH/results/frame_experiment_status.txt
LOCK=$BENCH/.arms_running.lock

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

# Wait for the lock instead of exiting, but give up rather than wait forever.
waited=0
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do
    if [ "$waited" -ge 21600 ]; then
        say "gave up waiting for the arm lock after 6h (held by pid $(cat "$LOCK"))"
        exit 1
    fi
    [ "$((waited % 600))" -eq 0 ] && say "waiting for arm lock (pid $(cat "$LOCK")), ${waited}s"
    sleep 60
    waited=$((waited + 60))
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
say "=== fixed donation arms started (pid $$, waited ${waited}s) ==="

run_arm() {
    local tool=$1
    say ">>> $tool"
    python3 revision/scripts/54_run_extra_arm.py --tool "$tool" \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
    say "<<< $tool pipeline returned $?"

    local out=$BENCH/predictions/${tool}_completed.gff3
    if [ ! -s "$out" ]; then
        say "$tool: NO OUTPUT"
        echo "$tool: NO OUTPUT" >> "$STATUS"
        return 1
    fi
    python3 revision/scripts/34_score_as_additions.py \
        --input "$BENCH/inputs/${tool}_Athaliana.gff3" --output "$out" \
        --tool "$tool" --json "$BENCH/results/${tool}_as_additions.json" >> "$LOG" 2>&1

    python3 revision/scripts/36_filter_additions_structurally.py \
        --input "$BENCH/inputs/${tool}_Athaliana.gff3" --output "$out" \
        --filtered "$BENCH/predictions/${tool}_filtered.gff3" \
        --json "$BENCH/results/${tool}_filter.json" >> "$LOG" 2>&1
    if [ -s "$BENCH/predictions/${tool}_filtered.gff3" ]; then
        python3 revision/scripts/34_score_as_additions.py \
            --input "$BENCH/inputs/${tool}_Athaliana.gff3" \
            --output "$BENCH/predictions/${tool}_filtered.gff3" \
            --tool "${tool}_filtered" \
            --json "$BENCH/results/${tool}_filtered_as_additions.json" >> "$LOG" 2>&1
    fi

    local line
    line=$(python3 -c "
import json
d = json.load(open('$BENCH/results/${tool}_as_additions.json'))
print(f\"added={d['added_structures']:,} TAIR10_alt={d['precision_vs_TAIR10_alternatives_pct']}% \"
      f\"AtRTD3={d['precision_vs_AtRTD3_pct']}% recall={d['recall_of_TAIR10_alternatives_pct']}%\")
" 2>>"$LOG")
    say "SCORED $tool: $line"
    echo "$tool: $line" >> "$STATUS"
}

run_arm helixertairutr_fixed
run_arm annevotairutr_fixed
say "=== fixed donation arms finished ==="
