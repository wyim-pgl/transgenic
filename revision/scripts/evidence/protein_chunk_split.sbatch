#!/usr/bin/env bash
#SBATCH --job-name=protein_chunk_split
#SBATCH --account=cpu-s2-pgl-0
#SBATCH --partition=cpu-s2-core-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein_alignments/logs/%x_%j.out
#
# Split the section 1 frozen protein resource into N chunks for chunked alignment (issue #45).
#
# WHY. miniprot memory grows with the query set and lh3/miniprot#68 has no fix, so a 12,115,085
# protein run cannot be given a bound that holds -- 6150892 tasks 1 and 3 died at 64 GB, and task 2
# was observed stepping 29.9 -> 55.2 GiB across 0.35 M queries. Chunking bounds the query set
# structurally instead of guessing a ceiling.
#
# SOUNDNESS. Chunking is only valid if per-query results do not depend on which other queries share
# the run. That was MEASURED before this was written (job 6152995, pilot/split_equiv/REPORT.txt):
# 1 x 100k, a re-run of 1 x 100k, and 2 x 50k concatenated are identical at the GFF level (same
# records, same order, same md5 after dropping run-local MP ids) and at the acceptance level
# (coverage, introns, ambiguous, alignments modulo alignment_id, and summary.json all identical).
# The one thing that does NOT survive concatenation is the MP###### numbering, which restarts per
# chunk -- 46,833 collisions in the pilot. protein_merge.sbatch renumbers; do not skip it.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
FAA=$ROOT/evidence/protein/odb12_Viridiplantae.filtered.fa.gz
FAA_FROZEN_MD5=453cb32b02e0799950d7d5f4de5f62ac
EXPECTED_SEQS=12115085
NCHUNK=6
OUT=$ROOT/evidence/protein/chunks_v1
MAN=$OUT/MANIFEST.tsv

# --- Gate: the input must be the section 1 frozen object. ---
[ -f "$FAA" ] || { echo "REFUSED: missing $FAA" >&2; exit 3; }
MD5=$(md5sum "$FAA" | cut -d' ' -f1)
[ "$MD5" = "$FAA_FROZEN_MD5" ] || { echo "REFUSED: protein resource md5 $MD5 != section 1 $FAA_FROZEN_MD5" >&2; exit 3; }

if [ -s "$MAN" ]; then echo "chunks already built; $MAN exists"; cat "$MAN"; exit 0; fi
mkdir -p "$OUT"
CS=$(( (EXPECTED_SEQS + NCHUNK - 1) / NCHUNK ))
echo "splitting $EXPECTED_SEQS sequences into $NCHUNK chunks of $CS"

# Records go to chunks in ORDER. Order is what the pilot measured, and it is what makes the
# concatenation check below a check on this exact split rather than on some canonicalised form.
gzip -cd "$FAA" | awk -v cs="$CS" -v n="$NCHUNK" -v out="$OUT" '
  /^>/ { c = int(rec / cs); if (c > n-1) c = n-1; rec++
         f = sprintf("%s/chunk%d.faa.gz", out, c)
         if (!(f in seen)) { seen[f] = sprintf("gzip -1 > %s.tmp", f) }
         cur = seen[f]; cnt[c]++ }
  { print | cur }
  END { for (f in seen) close(seen[f])
        for (i = 0; i < n; i++) printf "%d\t%d\n", i, cnt[i] > (out "/counts.tsv") }'
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: split stage $i exited ${st[$i]}" >&2; rm -f "$OUT"/chunk*.tmp; exit 4; }
done

for i in $(seq 0 $((NCHUNK-1))); do
  f=$OUT/chunk$i.faa.gz
  [ -s "$f.tmp" ] && gzip -t "$f.tmp" 2>/dev/null || { echo "REFUSED: chunk $i missing or corrupt" >&2; exit 4; }
  mv "$f.tmp" "$f"
done

# --- Verification 1: the record counts must add up to the frozen count. ---
TOT=$(awk -F'\t' '{s+=$2} END{print s}' "$OUT/counts.tsv")
[ "$TOT" -eq "$EXPECTED_SEQS" ] || { echo "REFUSED: chunk counts sum to $TOT, expected $EXPECTED_SEQS" >&2; exit 5; }

# --- Verification 2: chunks concatenated in order must reproduce the input byte for byte.
# Counting records only proves nothing was lost; this proves nothing was reordered or altered. ---
SRC_MD5=$(gzip -cd "$FAA" | md5sum | cut -d' ' -f1)
CAT_MD5=$(for i in $(seq 0 $((NCHUNK-1))); do gzip -cd "$OUT/chunk$i.faa.gz"; done | md5sum | cut -d' ' -f1)
[ "$SRC_MD5" = "$CAT_MD5" ] || { echo "REFUSED: concatenated chunks md5 $CAT_MD5 != source $SRC_MD5" >&2; exit 5; }

{
  printf "chunk\tseqs\tfile\tmd5\n"
  for i in $(seq 0 $((NCHUNK-1))); do
    printf "%d\t%s\t%s\t%s\n" "$i" \
      "$(awk -F'\t' -v c=$i '$1==c{print $2}' "$OUT/counts.tsv")" \
      "$OUT/chunk$i.faa.gz" "$(md5sum "$OUT/chunk$i.faa.gz" | cut -d' ' -f1)"
  done
} > "$MAN.tmp"
mv "$MAN.tmp" "$MAN"

{
  echo "source           $FAA"
  echo "source_md5       $MD5   (protocol section 1, A19.1 stage-2 filtered)"
  echo "sequences        $EXPECTED_SEQS"
  echo "chunks           $NCHUNK"
  echo "chunk_size       $CS"
  echo "uncompressed_md5 $SRC_MD5   (source == chunks concatenated in order)"
  echo "reason           miniprot memory is unbounded in the query set; lh3/miniprot#68 has no fix"
  echo "equivalence      measured, job 6152995, pilot/split_equiv/REPORT.txt"
  echo "built            $(date -Is)"
} > "$OUT/PROVENANCE.txt"

cat "$MAN"
echo "CHUNK_SPLIT_DONE"
