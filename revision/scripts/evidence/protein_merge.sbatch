#!/usr/bin/env bash
#SBATCH --job-name=protein_merge
#SBATCH --account=cpu-s2-pgl-0
#SBATCH --partition=cpu-s2-core-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein_alignments/logs/%x_%A_%a.out
#SBATCH --array=0-8
#
# Merge the six per-chunk alignments into the one per-species GFF the rest of the pipeline expects.
#
# A chunk is not a result. The pilot (job 6152995) showed chunked and unchunked output identical in
# every field EXCEPT the MP###### alignment ids, which restart at 1 in every chunk -- 46,833
# collisions in a two-way split. Concatenating without renumbering produces a GFF whose ids are
# ambiguous and whose Parent= links are wrong across chunk boundaries. That is silent: the record
# content is right, so nothing fails until something joins on alignment_id.
#
# Renumbering touches ONLY the ID= and Parent= attributes in column 9. It must not be a global
# substitution on MP: 578 of the ##PAF lines in the pilot chunk carry MP inside a query NAME, and a
# blanket s/MP[0-9]+/../ would rewrite the data itself.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
CHUNKDIR=$ROOT/evidence/protein/chunks_v1
OUT=$ROOT/evidence/protein_alignments
FAA=$ROOT/evidence/protein/odb12_Viridiplantae.filtered.fa.gz
FAA_FROZEN_MD5=453cb32b02e0799950d7d5f4de5f62ac
NCHUNK=6
PARAMS="--gff --outs=0.97 -N 20"

SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
CD=$OUT/$SP/chunks
GFF=$OUT/$SP/$SP.protein.gff.gz
PROV=$OUT/$SP/PROVENANCE.txt
DONE=$OUT/$SP/DONE
TMP=$GFF.merge.tmp

REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)

if [ -e "$DONE" ] && [ -s "$GFF" ] && gzip -t "$GFF" 2>/dev/null; then
  echo "$SP already merged"; exit 0
fi

# --- Gate: every chunk must be complete, and built from the same reference and parameters. ---
EXPECT_MRNA=0
for k in $(seq 0 $((NCHUNK-1))); do
  d=$CD/chunk$k.DONE; g=$CD/chunk$k.gff.gz
  [ -s "$d" ] && [ -s "$g" ] || { echo "REFUSED: $SP chunk $k not complete" >&2; exit 3; }
  gzip -t "$g" 2>/dev/null || { echo "REFUSED: $SP chunk $k gff is corrupt" >&2; exit 3; }
  [ "$(awk -F= '$1=="ref_md5"{print $2}' "$d")" = "$REF_MD5" ] \
    || { echo "REFUSED: $SP chunk $k was built from a different reference" >&2; exit 3; }
  [ "$(sed -n 's/^params=//p' "$d")" = "$PARAMS" ] \
    || { echo "REFUSED: $SP chunk $k was built with different parameters" >&2; exit 3; }
  EXPECT_MRNA=$(( EXPECT_MRNA + $(awk -F= '$1=="mrna_records"{print $2}' "$d") ))
done

# --- Merge with a running id offset. Chunk k's ids are 1..max_k; adding the running total keeps
# them unique AND keeps them in the order a single unchunked run would have produced. ---
OFF=0
rm -f "$TMP"
{
  echo "##gff-version 3"
  for k in $(seq 0 $((NCHUNK-1))); do
    MAXK=$(gzip -cd "$CD/chunk$k.gff.gz" | awk -F'\t' '$3=="mRNA"{ if (match($9,/^ID=MP[0-9]+/)) { v=substr($9,6,RLENGTH-5)+0; if (v>m) m=v } } END{print m+0}')
    NK=$(gzip -cd "$CD/chunk$k.gff.gz" | awk -F'\t' '$3=="mRNA"' | wc -l)
    [ "$MAXK" -eq "$NK" ] || { echo "REFUSED: $SP chunk $k has max id $MAXK but $NK mRNA records" >&2; exit 4; }
    gzip -cd "$CD/chunk$k.gff.gz" | awk -F'\t' -v OFS='\t' -v off="$OFF" '
      /^##gff-version/ { next }
      /^#/             { print; next }
      {
        if (match($9, /^ID=MP[0-9]+/)) {
          v = substr($9, 6, RLENGTH-5) + 0
          $9 = sprintf("ID=MP%06d%s", v + off, substr($9, RLENGTH+1))
        } else if (match($9, /^Parent=MP[0-9]+/)) {
          v = substr($9, 10, RLENGTH-9) + 0
          $9 = sprintf("Parent=MP%06d%s", v + off, substr($9, RLENGTH+1))
        }
        print
      }'
    OFF=$(( OFF + MAXK ))
  done
} | gzip -1 > "$TMP"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: merge stage $i exited ${st[$i]} for $SP" >&2; rm -f "$TMP"; exit 4; }
done

# --- Verification: count, uniqueness and contiguity of the renumbered ids. ---
gzip -t "$TMP" 2>/dev/null || { echo "REFUSED: merged GFF corrupt for $SP" >&2; rm -f "$TMP"; exit 5; }
read -r MRNA UNIQ MAXID <<<"$(gzip -cd "$TMP" | awk -F'\t' '
  $3=="mRNA" && match($9,/^ID=MP[0-9]+/) { n++; id=substr($9,6,RLENGTH-5)+0; seen[id]=1; if(id>mx)mx=id }
  END { u=0; for (i in seen) u++; print n, u, mx+0 }')"
[ "$MRNA" -eq "$EXPECT_MRNA" ] || { echo "REFUSED: $SP merged $MRNA mRNA, chunks declared $EXPECT_MRNA" >&2; rm -f "$TMP"; exit 5; }
[ "$UNIQ" -eq "$MRNA" ]        || { echo "REFUSED: $SP has $UNIQ distinct ids for $MRNA records" >&2; rm -f "$TMP"; exit 5; }
[ "$MAXID" -eq "$MRNA" ]       || { echo "REFUSED: $SP max id $MAXID != $MRNA records (ids not contiguous)" >&2; rm -f "$TMP"; exit 5; }
mv "$TMP" "$GFF"

GFF_MD5=$(md5sum "$GFF" | cut -d' ' -f1)
{
  echo "species          $SP"
  echo "scope            protein_training_evidence (A19.2, issue #45)"
  echo "reference        $REF"
  echo "reference_md5    $REF_MD5"
  echo "protein_resource $FAA"
  echo "protein_md5      $FAA_FROZEN_MD5   (protocol section 1, A19.1 stage-2 filtered)"
  echo "protein_seqs     12115085   (filter_summary.json sequences_out, 408 taxa)"
  echo "miniprot         $("$BIN"/miniprot --version)"
  echo "params           $PARAMS   (A19.2, frozen)"
  echo "chunked          $NCHUNK chunks, $CHUNKDIR/MANIFEST.tsv"
  echo "chunk_reason     miniprot memory is unbounded in the query set; array 6150892 lost two"
  echo "                 species to OUT_OF_MEMORY at 64 GB. lh3/miniprot#68 is the same signature"
  echo "                 upstream and is closed with no fix."
  echo "chunk_equivalence measured before use, job 6152995: 1x100k, a re-run of 1x100k and 2x50k"
  echo "                 concatenated are identical at the GFF and acceptance levels once the"
  echo "                 run-local MP ids are dropped. See pilot/split_equiv/REPORT.txt."
  echo "id_renumbering   chunk ids restart at 1; merged ids are 1..$MRNA, contiguous and unique."
  echo "gff              $GFF"
  echo "gff_md5          $GFF_MD5"
  echo "mrna_records     $MRNA"
  echo "merged           $(date -Is)"
  echo "acceptance       NOT APPLIED HERE. A19.2 identity >= 30 %, coverage >= 70 %, no frameshift,"
  echo "                 canonical introns <= 200 kb and mapping_ambiguous flagging run as a separate"
  echo "                 pass over this GFF, so the judgement can be revised without realigning."
  echo "run_location     pronghorn. A19 and #45 say Delta; author decision 2026-09-07. Input is the"
  echo "                 section 1 frozen object by md5, so location cannot change the input."
  echo "                 Log in protocol section 12."
} > "$PROV"

PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  printf "%s\tprotein_training_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$FAA" "$FAA_FROZEN_MD5" "$REF" "$REF_MD5" \
    "$("$BIN"/miniprot --version)" "miniprot -t 32 $PARAMS (${NCHUNK} chunks, merged)" "$GFF" "$GFF_MD5" \
    "A19.2 alignment only. Chunked after two OOMs on array 6150892; split equivalence measured in job 6152995. Ran on pronghorn, not Delta (author decision 2026-09-07)" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock"

{
  printf 'ref_md5=%s\n'      "$REF_MD5"
  printf 'faa_md5=%s\n'      "$FAA_FROZEN_MD5"
  printf 'params=%s\n'       "$PARAMS"
  printf 'gff_md5=%s\n'      "$GFF_MD5"
  printf 'mrna_records=%s\n' "$MRNA"
  printf 'chunks=%s\n'       "$NCHUNK"
  printf 'completed=%s\n'    "$(date -Is)"
} > "$DONE.tmp"
mv "$DONE.tmp" "$DONE"
echo "$SP merged: $MRNA mRNA records, ids 1..$MAXID"
