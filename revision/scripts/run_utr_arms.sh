#!/bin/bash
# Priority-reordered continuation, 23:25.
#
# tair10self returned 4.6% where the manuscript's own prediction scores 18.6% under the
# same scorer. The arms were built from CDS rows only, and GSF carries dedicated UTR
# tokens, so those arms were never running the manuscript's condition. Reproducing it is
# now the blocking question: until an arm recovers ~18%, a frame effect measured against
# 4.6% is measured in a regime the manuscript never ran in, and no result here can be
# compared to the figure in the paper.
#
#   tair10selfutr          TAIR10 CDS + UTR + TAIR10 frame — must recover ~18% or the
#                          reproduction has failed for a reason still unidentified
#   tair10helixerframeutr  the same prompt, Helixer's gene boundary — the frame effect,
#                          finally measured in the regime the manuscript ran in
#   tair10helixerframe     the CDS-only frame arm, resumed last; its chunks are kept, so
#                          this picks up where it stopped rather than restarting
#
# tair10helixerframe was stopped mid-run to free the GPU for the two above. That is a
# deliberate trade: it answers a version of the frame question asked at 4.6%, which the
# UTR arms supersede.

set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1

BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
LOG=$BENCH/logs/utr_arms_$(date +%Y%m%d_%H%M%S).log
STATUS=$BENCH/results/frame_experiment_status.txt
LOCK=$BENCH/.arms_running.lock
RUNNER=revision/scripts/33_run_polishing_benchmark.py
SCORER=revision/scripts/34_score_as_additions.py

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "another copy is running (pid $(cat "$LOCK")) — refusing to start" >> "$LOG"
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
say "=== UTR arms started (pid $$) ==="

score() {
    local tool=$1
    local out=$BENCH/predictions/${tool}_completed.gff3
    if [ ! -s "$out" ]; then
        say "SKIP scoring $tool — no completed output"
        echo "$tool: NO OUTPUT" >> "$STATUS"
        return 1
    fi
    python3 "$SCORER" --input "$BENCH/inputs/${tool}_Athaliana.gff3" --output "$out" \
        --tool "$tool" --json "$BENCH/results/${tool}_as_additions.json" >> "$LOG" 2>&1
    local line
    line=$(python3 -c "
import json
d = json.load(open('$BENCH/results/${tool}_as_additions.json'))
print(f\"added={d['added_structures']:,} TAIR10_alt={d['precision_vs_TAIR10_alternatives_pct']}% \"
      f\"AtRTD3={d['precision_vs_AtRTD3_pct']}% any={d['precision_vs_TAIR10_any_transcript_pct']}% \"
      f\"recall={d['recall_of_TAIR10_alternatives_pct']}%\")
" 2>>"$LOG")
    say "SCORED $tool: $line"
    echo "$tool: $line" >> "$STATUS"
}

run_arm() {
    local tool=$1
    say ">>> running $tool"
    python3 "$RUNNER" --tool "$tool" \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
    say "<<< $tool returned"
    score "$tool"
}

run_arm tair10selfutr
run_arm tair10helixerframeutr
run_arm tair10helixerframe

say "=== UTR arms finished ==="
echo "finished $(date +%H:%M)" >> "$STATUS"
