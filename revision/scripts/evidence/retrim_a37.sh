#!/usr/bin/env bash
# A37 (v1.27): produce both EST ingestion arms from the screening that already ran.
#   usage: bash retrim_a37.sh [species ...]     (default: all nine training species)
#
# The blastn pass is NOT repeated. UniVec screening does not depend on the length floor - it finds
# vector, and the floor is applied afterwards by the trimmer - so the per-species hits.tsv from
# 2026-09-03 is the same object for both arms. Only the trim step differs, which is why restoring the
# primary floor costs minutes rather than the hour the screening took.
#
#   est.univec.fa.gz          primary,     >= 100 nt (A8/A21 as frozen, restored by A37)
#   est.univec.min121.fa.gz   sensitivity, >= 121 nt (the A36 arm, now labelled)
set -uo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic/evidence
REPO=/data/gpfs/assoc/pgl/data/Transgenic/transgenic
PY=${PY:-/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python}
SPECIES=${@:-Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera}

for sp in $SPECIES; do
  d=$ROOT/est/$sp/univec
  [ -f "$d/hits.tsv" ] || { echo "$(date -Is) $sp: no hits.tsv - screening never ran here" >&2; exit 3; }
  in=$ROOT/est/$sp/est.fa.gz
  # The file currently called est.univec.fa.gz was produced at 121. Name it for what it is before
  # writing the primary, so the two arms are never confused by a stale filename.
  if [ -f "$d/est.univec.fa.gz" ] && [ ! -f "$d/est.univec.min121.fa.gz" ]; then
    mv "$d/est.univec.fa.gz" "$d/est.univec.min121.fa.gz"
    [ -f "$d/est.univec.fa.gz.md5" ] && mv "$d/est.univec.fa.gz.md5" "$d/est.univec.min121.fa.gz.md5"
    [ -f "$d/report.tsv" ] && mv "$d/report.tsv" "$d/report.min121.tsv"
    [ -f "$d/summary.json" ] && mv "$d/summary.json" "$d/summary.min121.json"
  fi
  for arm in 100 121; do
    case $arm in
      100) out=$d/est.univec.fa.gz;        rep=$d/report.tsv;        sum=$d/summary.json;;
      121) out=$d/est.univec.min121.fa.gz; rep=$d/report.min121.tsv; sum=$d/summary.min121.json;;
    esac
    # "exists" must mean "complete", not "a file is there". An interrupted trim leaves a truncated
    # gzip that -s accepts, and a truncated arm that looks finished is worse than a missing one.
    if [ -s "$out" ] && gzip -t "$out" 2>/dev/null; then
      echo "$(date -Is) $sp arm=$arm exists"; continue
    fi
    rm -f "$out" "$out.md5"
    s=$(date +%s)
    # Write to a temporary name and rename only after gzip -t passes, so the final name never exists
    # in a half-written state - the same discipline the fetcher had to learn on 2026-09-03.
    # The temporary name must still END IN .gz: 61_univec_trim.py picks gzip vs plain text from the
    # output suffix (open_any), so a name like "<...>.fa.gz.part" silently produces an uncompressed
    # file that gzip -t then rejects. Keep the suffix and put the marker at the front.
    tmp="$(dirname "$out")/.partial.$(basename "$out")"
    "$PY" "$REPO/revision/scripts/61_univec_trim.py" --fasta "$in" --hits "$d/hits.tsv" \
        --out "$tmp" --report "$rep" --summary "$sum" --min-len "$arm" > "$d/trim.$arm.out" 2>&1 \
      || { echo "$(date -Is) $sp arm=$arm FAILED" >&2; rm -f "$tmp"; exit 4; }
    gzip -t "$tmp" 2>/dev/null || { echo "$(date -Is) $sp arm=$arm CORRUPT output" >&2; rm -f "$tmp"; exit 5; }
    mv "$tmp" "$out"
    md5sum "$out" > "$out.md5"
    echo "$(date -Is) $sp arm=$arm $(( $(date +%s)-s ))s $(tr -d '\n ' < "$sum")"
  done
done
