#!/usr/bin/env bash
# Run-specific column regexes; every child failure propagates to the caller.
set -Eeuo pipefail
cd "${LONGREAD_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}"
P=${P:-8}
pids=()
while IFS=$'\t' read -r g l a r; do
  [ -n "$r" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  ./longread_fetch.sh "$g" "$l" "$a" "^${r}$" >> ont_parallel.log 2>&1 &
  pids+=("$!")
done < ont_parallel.tsv
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 )) || { echo "fetch child failed (exit $status); see fetch log" >&2; exit "$status"; }
echo "$(date -Is) ONT PARALLEL FINISHED" >> longread.log
