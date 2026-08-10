#!/bin/bash
# Re-run every arm from scratch, recording a structural signature each round.
# Round count: first argument, default 1 (2026-08-10 — the author's "10번" turned out to
# mean ten paper-critic passes, not ten benchmark rounds; one more full round is enough).
#
# The model is deterministic (do_sample=False, num_beams=2) and a forced ChrM regeneration
# reproduced its predecessor exactly on both the coordinate-only digest and the per-GM= CDS
# structure set — only the UUID transcript IDs differ, and nothing downstream reads those.
# So the expected result is identical rounds. The signature per round is what makes that
# a measurement instead of an assumption: if any round diverges, the arm and round are
# recorded and the run can be inspected rather than re-argued.
#
# Cost: ~13.4 h per round over nine arms (round 1 measured). Rounds are written as they
# finish, so stopping early leaves every completed round intact and comparable.

set -u
cd /data/gpfs/assoc/pgl/data/Transgenic/transgenic || exit 1

BENCH=/data/gpfs/assoc/pgl/data/Transgenic/polishing_benchmark
OUT=$BENCH/results/repeat
LOG=$OUT/repeat_$(date +%Y%m%d_%H%M%S).log
SUMMARY=$OUT/signatures.tsv
LOCK=$BENCH/.arms_running.lock
ARMS="helixer annevo tair10self tair10selfutr tair10helixerframe tair10helixerframeutr helixer_reframed helixertairutr annevotairutr"

mkdir -p "$OUT"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "another run holds the lock (pid $(cat "$LOCK"))" >> "$LOG"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> "$LOG"; }
[ -s "$SUMMARY" ] || printf "round\tarm\tstructure_signature\tloci\tstructures\tprecision\n" > "$SUMMARY"

signature() {  # signature <completed.gff3>
    python3 - "$1" <<'PY'
import hashlib, re, sys
from collections import defaultdict
tx, gm = defaultdict(list), {}
for line in open(sys.argv[1]):
    if line.startswith("#"): continue
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[2] != "CDS": continue
    p = re.search(r"Parent=([^;]+)", f[8]); g = re.search(r"GM=([^;\s]+)", f[8])
    if p and g:
        tx[p.group(1)].append((int(f[3]), int(f[4])))
        gm[p.group(1)] = g.group(1)
per = defaultdict(set)
for t, segs in tx.items():
    per[gm[t]].add(tuple(sorted(segs)))
flat = sorted((k, sorted(v)) for k, v in per.items())
print(f"{hashlib.md5(repr(flat).encode()).hexdigest()[:16]}\t{len(per)}\t{sum(len(v) for v in per.values())}")
PY
}

ROUNDS="${1:-1}"
for round in $(seq 1 "$ROUNDS"); do
    say "===== round $round ====="
    for arm in $ARMS; do
        say "round $round: $arm"
        python3 revision/scripts/33_run_polishing_benchmark.py --tool "$arm" --force \
            --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold >> "$LOG" 2>&1
        completed=$BENCH/predictions/${arm}_completed.gff3
        if [ ! -s "$completed" ]; then
            printf "%s\t%s\tNO_OUTPUT\t-\t-\t-\n" "$round" "$arm" >> "$SUMMARY"
            say "round $round: $arm produced no output"
            continue
        fi
        sig=$(signature "$completed")
        json=$OUT/run${round}_${arm}.json
        python3 revision/scripts/34_score_as_additions.py \
            --input $BENCH/inputs/${arm}_Athaliana.gff3 --output "$completed" \
            --tool "$arm" --json "$json" >> "$LOG" 2>&1
        prec=$(python3 -c "
import json;print(json.load(open('$json'))['precision_vs_TAIR10_alternatives_pct'])" 2>/dev/null)
        printf "%s\t%s\t%s\t%s\n" "$round" "$arm" "$sig" "$prec" >> "$SUMMARY"
        say "round $round: $arm sig=$(echo "$sig" | cut -f1) precision=$prec"
    done
    say "===== round $round complete ====="
done
say "=== all rounds finished ==="
