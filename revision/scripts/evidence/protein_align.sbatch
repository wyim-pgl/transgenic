#!/usr/bin/env bash
#SBATCH --job-name=protein_align
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein_alignments/logs/%x_%A_%a.out
#SBATCH --array=0-8
#
# OrthoDB protein -> genome spliced alignment. Protocol A19.2, issue #45.
#
# This produces the ALIGNMENT only. The A19.2 acceptance filter (identity >= 30 %, coverage >= 70 %,
# no frameshift, canonical introns <= 200 kb) and the per-organism support counts are a separate,
# re-runnable pass over the retained GFF. That split is deliberate and has precedent: A39 rescored 27
# audits from retained SAMs without realigning, because the expensive artifact and the judgement made
# from it must be able to move independently.
#
# Frozen command (A19.2 — "--outs=0.97 -N 20 style parameters fixed at freeze"):
#     miniprot -t 32 --gff --outs=0.97 -N 20 <index> <proteins>
# --gff is required, not cosmetic: the acceptance filter needs per-alignment Identity, Frameshift and
# intron coordinates, and the PAF default does not carry the exon structure.
#
# WHERE THIS RUNS. A19 says the resource is "downloaded on the ACCESS Delta allocation, where
# alignment runs", and #45 says "miniprot per training genome on Delta". This runs on pronghorn.
# The author decided this on 2026-09-07 after the two-point pilot below. The reasons on record:
# the input is the section 1 frozen object identified by md5 (so the location cannot change the
# input), miniprot is CPU work that would otherwise spend GPU allocation, and A19.4 item 3 places
# this "parallel with B5 training" rather than on the same machine. This deviation is recorded here,
# in every PROVENANCE.txt this job writes, and must be logged in protocol section 12.
#
# Sizing comes from a measured two-point pilot, NOT from extrapolating one point. Extrapolating EST
# memory onto long reads is exactly how tasks 8/49/119 ended up at a 32 GB ceiling with two OOMs
# (quarantine.md section 2f). Pilot: jobs 6150890 (Athaliana) and 6150891 (Gmax), 100,000 proteins,
# 32 threads:
#
#   species     genome        index   align(100k)   peak RSS
#   Athaliana   119.7 Mb        2 s       597 s     11.9 GB
#   Gmax       1011.1 Mb       16 s       981 s     19.0 GB
#
# An 8.45x genome costs 1.64x time and 1.59x memory, so cost goes as genome^0.23 -- strongly
# sub-linear. The second point is what makes this a measurement: a linear reading of the first point
# alone predicts 170 h for Gmax against a measured 33 h, and would have sized both knobs from a
# number that is wrong by 5x. Full run = 121.15x the pilot: 20 h (Athaliana) to 33 h (Gmax),
# 243 node-hours over nine species.
#
# --time=48:00:00 is 45 % over the largest estimate. --mem=64G is 3.4x the measured maximum; the
# nodes hold 256 GB and 32 cpus-per-task already caps this at two tasks per node, so the headroom
# costs no concurrency. Memory is dominated by the index and the streamed batch, not by the query
# count, but the long-read OOM is what a thin margin here looks like in practice.
#
# Only the nine TRAINING species run here. A19.4 item 5: test-species genomes are never aligned
# against this resource for training purposes.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
FAA=$ROOT/evidence/protein/odb12_Viridiplantae.filtered.fa.gz
FAA_FROZEN_MD5=453cb32b02e0799950d7d5f4de5f62ac      # protocol section 1, A19.1 stage-2 filtered
OUT=$ROOT/evidence/protein_alignments

SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
HELD_OUT=(Zmays Slycopersicum)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$OUT/logs" "$OUT/$SP"

GFF=$OUT/$SP/$SP.protein.gff.gz
IDX=$OUT/$SP/$SP.mpi
PROV=$OUT/$SP/PROVENANCE.txt
DONE=$OUT/$SP/DONE

# --- Gate 1: A19.4 item 5 — a held-out species must never reach this aligner. ---
for h in "${HELD_OUT[@]}"; do
  [ "$SP" = "$h" ] && { echo "REFUSED: $SP is a held-out test species (A19.4 item 5)" >&2; exit 2; }
done

# --- Gate 2: section 3.3 item 1 — align to the reference the manifest declares, verified by md5. ---
REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
EXPECTED_REF_MD5=$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")
[ -f "$REF" ] || { echo "REFUSED: missing genome for $SP: ${REF:-<none>}" >&2; exit 3; }
ACTUAL_REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)
if [ -z "$EXPECTED_REF_MD5" ] || [ "$ACTUAL_REF_MD5" != "$EXPECTED_REF_MD5" ]; then
  echo "REFUSED: reference md5 $ACTUAL_REF_MD5 != $(basename "$MANIFEST") ${EXPECTED_REF_MD5:-<none>} for $SP" >&2
  exit 3
fi

# --- Gate 3: the protein resource must be the section 1 frozen object, not a copy that drifted. ---
[ -f "$FAA" ] || { echo "REFUSED: missing protein resource $FAA" >&2; exit 3; }
gzip -t "$FAA" 2>/dev/null || { echo "REFUSED: corrupt protein resource $FAA" >&2; exit 3; }
FAA_MD5=$(md5sum "$FAA" | cut -d' ' -f1)
if [ "$FAA_MD5" != "$FAA_FROZEN_MD5" ]; then
  echo "REFUSED: protein resource md5 $FAA_MD5 != section 1 $FAA_FROZEN_MD5" >&2
  exit 3
fi

# --- Completion marker, bound to both inputs and the parameters. ---
# A bare marker means "this directory was written once", which is how nine stale markers once made an
# entire array exit 0 while the old results stayed in place. This one names what produced it, so a
# changed reference, a changed resource or changed parameters refuse instead of skipping.
PARAMS="--gff --outs=0.97 -N 20"
if [ -e "$DONE" ]; then
  D_REF=$(awk -F= '$1=="ref_md5"{print $2}'    "$DONE")
  D_FAA=$(awk -F= '$1=="faa_md5"{print $2}'    "$DONE")
  D_PAR=$(awk -F= '$1=="params"{sub(/^params=/,"");print}' "$DONE")
  [ -n "$D_REF" ] && [ -n "$D_FAA" ] || { echo "REFUSED: $SP has a marker with no input identity; retire the old result first" >&2; exit 6; }
  [ "$D_REF" = "$ACTUAL_REF_MD5" ] || { echo "REFUSED: $SP marker built from ref_md5=$D_REF, current is $ACTUAL_REF_MD5" >&2; exit 6; }
  [ "$D_FAA" = "$FAA_MD5" ]        || { echo "REFUSED: $SP marker built from faa_md5=$D_FAA, current is $FAA_MD5" >&2; exit 6; }
  [ "$D_PAR" = "$PARAMS" ]         || { echo "REFUSED: $SP marker built with params '$D_PAR', current is '$PARAMS'" >&2; exit 6; }
  if [ -s "$GFF" ] && gzip -t "$GFF" 2>/dev/null && [ -s "$PROV" ]; then
    echo "$SP already complete for ref_md5=$ACTUAL_REF_MD5 faa_md5=$FAA_MD5"; exit 0
  fi
  echo "REFUSED: $SP marker matches the inputs but the outputs are incomplete" >&2
  exit 6
fi

START=$(date -Is)
SECONDS=0

# --- Index. Cheap next to the alignment (2 s for A. thaliana) and reused by any rerun. ---
if [ ! -s "$IDX" ]; then
  "$BIN"/miniprot -t 32 -d "$IDX.tmp" "$REF" 2> "$OUT/$SP/miniprot.index.err" \
    || { echo "REFUSED: miniprot index failed for $SP" >&2; rm -f "$IDX.tmp"; exit 4; }
  mv "$IDX.tmp" "$IDX"
fi
INDEX_SECONDS=$SECONDS

# --- Alignment (A19.2 frozen parameters). ---
set -o pipefail
"$BIN"/miniprot -t 32 $PARAMS "$IDX" "$FAA" 2> "$OUT/$SP/miniprot.err" | gzip -1 > "$GFF.tmp"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the alignment pipeline exited ${st[$i]} for $SP" >&2; rm -f "$GFF.tmp"; exit 4; }
done
# Exit status alone does not say the output arrived: gzip of an empty stream exits 0.
{ [ -s "$GFF.tmp" ] && gzip -t "$GFF.tmp" 2>/dev/null; } \
  || { echo "REFUSED: missing or corrupt GFF for $SP" >&2; rm -f "$GFF.tmp"; exit 4; }
MRNA=$(gzip -cd "$GFF.tmp" | awk '$3=="mRNA"' | wc -l)
[ "$MRNA" -gt 0 ] || { echo "REFUSED: $SP produced 0 mRNA records" >&2; rm -f "$GFF.tmp"; exit 4; }
mv "$GFF.tmp" "$GFF"

GFF_MD5=$(md5sum "$GFF" | cut -d' ' -f1)
{
  echo "species          $SP"
  echo "scope            protein_training_evidence (A19.2, issue #45)"
  echo "reference        $REF"
  echo "reference_md5    $ACTUAL_REF_MD5"
  echo "protein_resource $FAA"
  echo "protein_md5      $FAA_MD5   (protocol section 1, A19.1 stage-2 filtered)"
  echo "protein_seqs     12115085   (filter_summary.json sequences_out, 408 taxa)"
  echo "miniprot         $("$BIN"/miniprot --version)"
  echo "params           $PARAMS   (A19.2, frozen)"
  echo "command          miniprot -t 32 $PARAMS $IDX $FAA"
  echo "index            $IDX"
  echo "index_seconds    $INDEX_SECONDS"
  echo "gff              $GFF"
  echo "gff_md5          $GFF_MD5"
  echo "mrna_records     $MRNA"
  echo "started          $START"
  echo "finished         $(date -Is)"
  echo "wall_seconds     $SECONDS"
  echo "acceptance       NOT APPLIED HERE. A19.2 identity >= 30 %, coverage >= 70 %, no frameshift,"
  echo "                 canonical introns <= 200 kb and mapping_ambiguous flagging run as a separate"
  echo "                 pass over this GFF, so the judgement can be revised without realigning."
  echo "run_location     pronghorn. A19 and #45 say Delta; author decision 2026-09-07 after the"
  echo "                 two-point pilot (jobs 6150890 Athaliana, 6150891 Gmax). Input is the section 1"
  echo "                 frozen object by md5, so location cannot change the input. Log in section 12."
  echo "slurm_job        ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
} > "$PROV"

PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  printf "%s\tprotein_training_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$FAA" "$FAA_MD5" "$REF" "$ACTUAL_REF_MD5" \
    "$("$BIN"/miniprot --version)" "miniprot -t 32 $PARAMS" "$GFF" "$GFF_MD5" \
    "A19.2 alignment only; acceptance filter and per-organism counts are a separate pass. Ran on pronghorn, not Delta (author decision 2026-09-07); input identified by the section 1 md5" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock"

{
  printf 'ref_md5=%s\n' "$ACTUAL_REF_MD5"
  printf 'faa_md5=%s\n' "$FAA_MD5"
  printf 'params=%s\n'  "$PARAMS"
  printf 'gff_md5=%s\n' "$GFF_MD5"
  printf 'mrna_records=%s\n' "$MRNA"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp"
mv "$DONE.tmp" "$DONE"
echo "$SP finished in ${SECONDS}s, $MRNA mRNA records"
