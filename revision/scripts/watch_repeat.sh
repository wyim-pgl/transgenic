#!/bin/bash
# Watchdog for run_repeat.sh. Non-invasive: reads only, writes only its own files.
#
# Why this exists. run_repeat.sh decides an arm produced a result by testing
# `[ -s ${arm}_completed.gff3 ]`. That test cannot tell a fresh file from the previous
# round's leftover. 33_run_polishing_benchmark.py only unlinks the canonical
# `_completed.gff3` after every chunk has finished (script line ~1515), so a crash during
# the multi-hour chunk phase — ssh to the GPU box dropping, CUDA OOM, walltime — leaves
# last round's file untouched. run_repeat.sh would then hash that file and record it as
# this round's signature: a row reading "identical" that is identical only because nothing
# new was written. In a run whose entire product is the claim "ten rounds came out the
# same", that is the one failure that cannot be allowed to pass silently.
#
# So: for every row run_repeat.sh appends, compare the mtime of the file it hashed against
# the moment that arm's run started (taken from the runner's own log). Older than the
# start means the hash came from a leftover; the row is evidence of nothing and is marked
# STALE. Newer means the row is real. Signatures are also compared against the first
# round that recorded each arm, so a genuine divergence is called out as loudly.
#
# Detection, not prevention — the row still gets written. Fixing run_repeat.sh to delete
# the file first and check the exit code would prevent it, but that means restarting the
# run. This costs nothing and makes every row auditable after the fact.

set -u
BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
OUT=$BENCH/results/repeat
SUMMARY=$OUT/signatures.tsv
FRESH=$OUT/freshness.tsv
ALERT=$OUT/ALERTS.txt
YEAR=$(date +%Y)

[ -s "$FRESH" ] || printf "round\tarm\tarm_started\tfile_mtime\tverdict\tsignature\tnote\n" > "$FRESH"

seen=0
[ -s "$FRESH" ] && seen=$(($(wc -l < "$FRESH") - 1))

# arm_start_epoch <round> <arm> — the runner logs "[MM-DD HH:MM:SS] round R: ARM" when it
# launches an arm, and a second line with sig=/precision= when it finishes. Match only the
# launch line (nothing after the arm name) and take the last one, which is this round's.
arm_start_epoch() {
    local round=$1 arm=$2 stamp
    stamp=$(grep -hE "^\[[0-9-]+ [0-9:]+\] round $round: $arm\$" "$OUT"/repeat_*.log 2>/dev/null | tail -1 \
            | sed -E 's/^\[([0-9-]+ [0-9:]+)\].*/\1/')
    [ -n "$stamp" ] || { echo ""; return; }
    date -d "$YEAR-$stamp" +%s 2>/dev/null
}

while true; do
    if [ -s "$SUMMARY" ]; then
        rows=$(($(wc -l < "$SUMMARY") - 1))
        while [ "$rows" -gt "$seen" ]; do
            seen=$((seen + 1))
            line=$(tail -n +2 "$SUMMARY" | sed -n "${seen}p")
            [ -n "$line" ] || break
            round=$(echo "$line" | cut -f1); arm=$(echo "$line" | cut -f2)
            sig=$(echo "$line" | cut -f3)

            if [ "$sig" = "NO_OUTPUT" ]; then
                printf "%s\t%s\t-\t-\tNO_OUTPUT\t-\tarm produced nothing; runner recorded the failure itself\n" \
                    "$round" "$arm" >> "$FRESH"
                echo "[$(date +'%m-%d %H:%M:%S')] round $round $arm: NO_OUTPUT" >> "$ALERT"
                continue
            fi

            completed=$BENCH/predictions/${arm}_completed.gff3
            mtime=$(stat -c %Y "$completed" 2>/dev/null)
            started=$(arm_start_epoch "$round" "$arm")

            verdict=UNKNOWN; note="could not resolve start time or file mtime"
            if [ -n "$mtime" ] && [ -n "$started" ]; then
                if [ "$mtime" -ge "$started" ]; then
                    verdict=FRESH; note="written after this arm's run began"
                else
                    verdict=STALE
                    note="file predates this round's run by $((started - mtime))s — the signature was hashed from a leftover, discard this row"
                fi
            fi

            # Compare against the first FRESH signature recorded for this arm.
            first=$(awk -F'\t' -v a="$arm" '$2==a && $5=="FRESH" {print $6; exit}' "$FRESH")
            if [ -n "$first" ] && [ "$verdict" = FRESH ] && [ "$first" != "$sig" ]; then
                note="$note; DIVERGED from first round ($first)"
                echo "[$(date +'%m-%d %H:%M:%S')] round $round $arm: DIVERGED $first -> $sig" >> "$ALERT"
            fi
            [ "$verdict" = STALE ] && echo "[$(date +'%m-%d %H:%M:%S')] round $round $arm: STALE row — $note" >> "$ALERT"

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$round" "$arm" \
                "$([ -n "$started" ] && date -d @"$started" +'%m-%d %H:%M:%S' || echo -)" \
                "$([ -n "$mtime" ] && date -d @"$mtime" +'%m-%d %H:%M:%S' || echo -)" \
                "$verdict" "$sig" "$note" >> "$FRESH"
        done
    fi

    # The runner dying is itself worth recording — otherwise the tsv just stops growing
    # and silence reads the same as "still working".
    # Anchored: an unanchored -f pattern also matches pgrep's own command line, so the
    # test would always succeed and the runner's death would never be noticed.
    if ! pgrep -f "^bash $BENCH/run_repeat.sh" > /dev/null 2>&1; then
        echo "[$(date +'%m-%d %H:%M:%S')] run_repeat.sh is no longer running (last row: ${seen})" >> "$ALERT"
        exit 0
    fi
    sleep 120
done
