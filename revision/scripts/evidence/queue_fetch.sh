#!/usr/bin/env bash
# Run-specific column regexes; every child failure propagates to the caller.
set -Eeuo pipefail
cd "${LONGREAD_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}"
Q=${1:?queue tsv}; LOG=${2:-queue}; P=${P:-6}
pids=()
while IFS=$'\t' read -r g l a r; do
  [ -n "${r:-}" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  ./longread_fetch.sh "$g" "$l" "$a" "^${r}$" >> "$LOG.log" 2>&1 &
  pids+=("$!")
done < "$Q"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 )) || { echo "fetch child failed (exit $status); see fetch log" >&2; exit "$status"; }
echo "$(date -Is) QUEUE $Q FINISHED" >> longread.log
