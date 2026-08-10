#!/bin/bash
# Overnight frame experiment, run unattended.
#
# Order is deliberate: arm A first because nothing downstream is interpretable without a
# baseline measured in THIS pipeline (the manuscript's 18.1% came from a different script
# and a different run, so it cannot serve as the control). Then helixer_reframed, which is
# the arm that could actually reach the goal. Then tair10helixerframe, which confirms the
# mechanism but changes no decision if the first two already answer it.
#
# The locked arms are NOT here. helixer_locked fails its own loss gate (65 unexplained
# missing loci against a threshold of 40 — pinning the prompt makes long loci overrun
# max_length), and Chr1 already answered what it was built to ask: precision does not move.
# Forcing it through with a raised gate would spend GPU hours on a settled question.
#
# Each tool is invoked separately so one failure does not take the rest down with it, and
# every stage logs, so a dead run can be diagnosed from the log alone.

set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1

BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
LOG=$BENCH/logs/frame_experiment_$(date +%Y%m%d_%H%M%S).log
STATUS=$BENCH/results/frame_experiment_status.txt
RUNNER=revision/scripts/33_run_polishing_benchmark.py
SCORER=revision/scripts/34_score_as_additions.py

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

say "=== frame experiment started (pid $$) ==="
echo "started $(date +%H:%M)" > "$STATUS"

score() {  # score <tool>
    local tool=$1
    local out=$BENCH/predictions/${tool}_completed.gff3
    local inp=$BENCH/inputs/${tool}_Athaliana.gff3
    if [ ! -s "$out" ]; then
        say "SKIP scoring $tool — no completed output"
        echo "$tool: NO OUTPUT (runner did not finish)" >> "$STATUS"
        return 1
    fi
    say "scoring $tool"
    python3 "$SCORER" --input "$inp" --output "$out" \
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

run_arm() {  # run_arm <tool>
    local tool=$1
    say ">>> running $tool"
    python3 "$RUNNER" --tool "$tool" \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
    say "<<< $tool runner returned"
    score "$tool"
}

run_arm tair10self
run_arm helixer_reframed
run_arm tair10helixerframe

say "=== frame experiment finished ==="
echo "finished $(date +%H:%M)" >> "$STATUS"
