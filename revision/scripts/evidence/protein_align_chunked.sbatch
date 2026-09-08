#!/usr/bin/env bash
#SBATCH --job-name=protein_align_ch
#SBATCH --account=cpu-s2-pgl-0
#SBATCH --partition=cpu-s2-core-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein_alignments/logs/%x_%A_%a.out
#SBATCH --array=0-53%24
#
# OrthoDB protein -> genome spliced alignment, CHUNKED. Protocol A19.2, issue #45.
# Replaces the unchunked array 6150892, which lost tasks 1 and 3 to OUT_OF_MEMORY at 64 GB.
#
# WHY CHUNKED. miniprot memory grows with the query set and does not track genome size: the two
# species that died (275 MB, 382 MB) are SMALLER than one that was still running (1,024 MB), and
# growth is a staircase, not a ramp -- task 2 sat at 29.9 GiB for hours, then stepped to 55.2 GiB
# across 0.35 M queries. No memory ceiling derived from a plateau is trustworthy. lh3/miniprot#68
# is the same signature upstream, on a host with 925 GB available, and it is closed with no fix.
# Chunking bounds the query set instead of guessing a ceiling.
#
# WHY THIS IS SOUND. Measured, not assumed (job 6152995, pilot/split_equiv/REPORT.txt): 1 x 100k,
# a re-run of the same 100k, and 2 x 50k concatenated are identical -- same records, same order,
# same md5 once run-local MP ids are dropped -- and identical again through protein_accept.py
# (coverage intervals, introns, ambiguous list, alignments modulo alignment_id, summary.json).
# The previous driver ASSERTED this independence in a comment without measuring it; asserting an
# unmeasured property is what produced the OOM in the first place.
#
# SIZING. 6 chunks of 2,019,181. The worst peak observed on the unchunked run was 63.0 GiB at
# 4.0 M queries (Osativa), so a 2.0 M chunk sits well below it, and 120 GB leaves room for a
# staircase step. This is a bound taken from the axis that actually killed the job, which the
# original two-point pilot held fixed and therefore could not see.
#
# LAYOUT. Task id -> species = id / 6, chunk = id % 6. Each task writes ONE chunk. Merging,
# renumbering and the final DONE marker belong to protein_merge.sbatch; a chunk is not a result.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
CHUNKDIR=$ROOT/evidence/protein/chunks_v1
CHUNKMAN=$CHUNKDIR/MANIFEST.tsv
OUT=$ROOT/evidence/protein_alignments
NCHUNK=6
PARAMS="--gff --outs=0.97 -N 20"

SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
HELD_OUT=(Zmays Slycopersicum)

SP=${SPECIES[$(( SLURM_ARRAY_TASK_ID / NCHUNK ))]}
K=$(( SLURM_ARRAY_TASK_ID % NCHUNK ))

# --- Gate 1: A19.4 item 5 -- a held-out species must never reach this aligner. ---
for h in "${HELD_OUT[@]}"; do
  [ "$SP" = "$h" ] && { echo "REFUSED: $SP is a held-out test species (A19.4 item 5)" >&2; exit 2; }
done

CD=$OUT/$SP/chunks
mkdir -p "$CD" "$OUT/logs"
GFF=$CD/chunk$K.gff.gz
DONE=$CD/chunk$K.DONE
IDX=$OUT/$SP/$SP.mpi

# --- Gate 2: section 3.3 item 1 -- align to the reference the manifest declares, verified by md5. ---
REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
EXPECTED_REF_MD5=$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")
[ -f "$REF" ] || { echo "REFUSED: missing genome for $SP: ${REF:-<none>}" >&2; exit 3; }
ACTUAL_REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)
[ -n "$EXPECTED_REF_MD5" ] && [ "$ACTUAL_REF_MD5" = "$EXPECTED_REF_MD5" ] \
  || { echo "REFUSED: reference md5 $ACTUAL_REF_MD5 != manifest ${EXPECTED_REF_MD5:-<none>} for $SP" >&2; exit 3; }

# --- Gate 3: the chunk must be one the splitter built and recorded, verified by md5. ---
[ -s "$CHUNKMAN" ] || { echo "REFUSED: no chunk manifest $CHUNKMAN; run protein_chunk_split.sbatch" >&2; exit 3; }
FAA=$(awk -F'\t' -v c="$K" '$1==c{print $3}' "$CHUNKMAN")
CHUNK_MD5=$(awk -F'\t' -v c="$K" '$1==c{print $4}' "$CHUNKMAN")
CHUNK_SEQS=$(awk -F'\t' -v c="$K" '$1==c{print $2}' "$CHUNKMAN")
[ -s "$FAA" ] || { echo "REFUSED: missing chunk $K file ${FAA:-<none>}" >&2; exit 3; }
ACTUAL_CHUNK_MD5=$(md5sum "$FAA" | cut -d' ' -f1)
[ "$ACTUAL_CHUNK_MD5" = "$CHUNK_MD5" ] \
  || { echo "REFUSED: chunk $K md5 $ACTUAL_CHUNK_MD5 != manifest $CHUNK_MD5" >&2; exit 3; }

# --- Completion marker bound to both inputs and the parameters, as in the unchunked driver.
# A bare marker means "this ran once", which is how nine stale markers once made an array exit 0. ---
if [ -e "$DONE" ]; then
  D_REF=$(awk -F= '$1=="ref_md5"{print $2}'   "$DONE")
  D_CH=$(awk -F= '$1=="chunk_md5"{print $2}'  "$DONE")
  D_PAR=$(sed -n 's/^params=//p'              "$DONE")
  [ "$D_REF" = "$ACTUAL_REF_MD5" ] && [ "$D_CH" = "$ACTUAL_CHUNK_MD5" ] && [ "$D_PAR" = "$PARAMS" ] \
    || { echo "REFUSED: $SP chunk $K marker was built from different inputs or parameters" >&2; exit 6; }
  if [ -s "$GFF" ] && gzip -t "$GFF" 2>/dev/null; then
    echo "$SP chunk $K already complete"; exit 0
  fi
  echo "REFUSED: $SP chunk $K marker matches the inputs but the output is incomplete" >&2; exit 6
fi

START=$(date -Is); SECONDS=0

# --- Index, once per species. Six chunk tasks race for it, so it is built under a lock. ---
(
  flock 9
  if [ ! -s "$IDX" ]; then
    "$BIN"/miniprot -t 32 -d "$IDX.tmp" "$REF" 2> "$OUT/$SP/miniprot.index.err" \
      && mv "$IDX.tmp" "$IDX" || { rm -f "$IDX.tmp"; exit 4; }
  fi
) 9>"$OUT/$SP/.index.lock"
[ -s "$IDX" ] || { echo "REFUSED: no index for $SP (build failed; see $OUT/$SP/miniprot.index.err)" >&2; exit 4; }

# --- Alignment (A19.2 frozen parameters). ---
set -o pipefail
"$BIN"/miniprot -t 32 $PARAMS "$IDX" "$FAA" 2> "$CD/chunk$K.err" | gzip -1 > "$GFF.tmp"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i exited ${st[$i]} for $SP chunk $K" >&2; rm -f "$GFF.tmp"; exit 4; }
done
{ [ -s "$GFF.tmp" ] && gzip -t "$GFF.tmp" 2>/dev/null; } \
  || { echo "REFUSED: missing or corrupt GFF for $SP chunk $K" >&2; rm -f "$GFF.tmp"; exit 4; }
MRNA=$(gzip -cd "$GFF.tmp" | awk -F'\t' '$3=="mRNA"' | wc -l)
[ "$MRNA" -gt 0 ] || { echo "REFUSED: $SP chunk $K produced 0 mRNA records" >&2; rm -f "$GFF.tmp"; exit 4; }
mv "$GFF.tmp" "$GFF"

{
  printf 'species=%s\n'      "$SP"
  printf 'chunk=%s\n'        "$K"
  printf 'chunk_seqs=%s\n'   "$CHUNK_SEQS"
  printf 'ref_md5=%s\n'      "$ACTUAL_REF_MD5"
  printf 'chunk_md5=%s\n'    "$ACTUAL_CHUNK_MD5"
  printf 'params=%s\n'       "$PARAMS"
  printf 'gff_md5=%s\n'      "$(md5sum "$GFF" | cut -d' ' -f1)"
  printf 'mrna_records=%s\n' "$MRNA"
  printf 'miniprot=%s\n'     "$("$BIN"/miniprot --version)"
  printf 'started=%s\n'      "$START"
  printf 'completed=%s\n'    "$(date -Is)"
  printf 'wall_seconds=%s\n' "$SECONDS"
  printf 'slurm_job=%s\n'    "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
} > "$DONE.tmp"
mv "$DONE.tmp" "$DONE"
echo "$SP chunk $K finished in ${SECONDS}s, $MRNA mRNA records"
