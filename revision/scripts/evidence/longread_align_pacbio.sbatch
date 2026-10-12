#!/usr/bin/env bash
#SBATCH --job-name=lr_align_pb
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/longread_align_v1/logs/%x_%A_%a.out
#SBATCH --array=0-48%4
#
# PacBio Iso-Seq CCS/FLNC training reads -> genome spliced alignment. PROTOCOL_B1 §4 (frozen
# PacBio command) + A18.3 role gate + A47.3 (issue #60 reversed, 2026-09-22).
#
# Sibling of longread_align.sbatch (ONT). Everything is the same except two things, both set by
# the protocol rather than chosen here:
#   1. the command is §4's PacBio line, quoted, not varied:
#          minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
#   2. there is no §3.3 orientation audit and never -uf. A47.3 rule 3: the raw Iso-Seq CCS runs
#      keep primers and poly(A) in random orientation, so the strand of a spliced alignment comes
#      from the splice motif; the FLNC-grade runs are aligned the same way so one command covers
#      the whole set. -uf is not a parameter of this script at all.
# Reads are the submitted sequence after seqkit fq2fa only; nothing is trimmed or processed here
# (A47 rule 1: no lima/refine). Primers and poly(A) are soft-clipped by the aligner.
#
# Run list: evidence/longread_runs_pacbio_v1.tsv (species, dataset, run, reads path), one task per
# run (§3.3 item 4). Output layout, DONE key, PROVENANCE and the central PROVENANCE.tsv row mirror
# the ONT driver so consumers find PacBio runs where they find ONT runs; the command line in
# PROVENANCE is what distinguishes them.
#
# WHAT THIS DRIVER DOES NOT DO: A18.5 acceptance (primary, MAPQ >= 20, query fraction >= 0.80,
# identity >= 98 % for CCS/FLNC) is the consumer's job, exactly as in the ONT driver. The BAM and
# PAF retained here are the raw alignment.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
RUNS=$ROOT/evidence/longread_runs_pacbio_v1.tsv
OUTROOT=$ROOT/evidence/longread_align_v1
THREADS=${SLURM_CPUS_PER_TASK:-32}

read -r SP DS RUN FA < <(awk -F'\t' -v i="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==i' "$RUNS")
[ -n "${RUN:-}" ] || { echo "REFUSED: no run at index $SLURM_ARRAY_TASK_ID in $RUNS" >&2; exit 2; }
FA=$ROOT/$FA
DSBASE=$(basename "$DS")
OUT=$OUTROOT/$SP/$DSBASE/$RUN
mkdir -p "$OUT" "$OUTROOT/logs"
BAM=$OUT/$RUN.bam
PAF=$OUT/$RUN.paf.gz
PROV=$OUT/PROVENANCE.txt
DONE=$OUT/DONE

# --- Gate 1: A18.3 role, by RUN IDENTITY (three Cui 2020 runs exist under two paths; the copy
#     under training/ont/ is the `excluded` duplicate, so a path lookup drops training evidence) ---
n_eli=$(awk -F'\t' -v r="$RUN" '$2==r && $12=="c2_training_eligible"' "$ROLES" | wc -l)
[ "$n_eli" -eq 1 ] || { echo "REFUSED: $RUN has $n_eli c2_training_eligible rows, expected exactly 1" >&2; exit 2; }

REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
GFF=$(awk -F'\t' -v s="$SP" '$1==s{print $6}' "$MANIFEST")
[ -f "$REF" ] && [ -f "$GFF" ] || { echo "REFUSED: missing reference or GFF for $SP" >&2; exit 3; }

# --- Gate 2: §3.3 item 1 ---
REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)
GFF_MD5=$(md5sum "$GFF" | cut -d' ' -f1)
[ "$REF_MD5" = "$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")" ] \
  || { echo "REFUSED: $SP reference md5 $REF_MD5 does not match $(basename "$MANIFEST")" >&2; exit 3; }
# The GFF identity is enforced here too, not only inside the audit's key: this driver must be
# checkable on its own rather than by trusting a marker written by another script.
[ "$GFF_MD5" = "$(awk -F'\t' -v s="$SP" '$1==s{print $7}' "$MANIFEST")" ] \
  || { echo "REFUSED: $SP GFF md5 $GFF_MD5 does not match $(basename "$MANIFEST")" >&2; exit 3; }
[ -s "$FA" ] && gzip -t "$FA" 2>/dev/null || { echo "REFUSED: missing or corrupt reads: $FA" >&2; exit 3; }
FA_MD5=$(md5sum "$FA" | cut -d' ' -f1)

# --- No Gate 3: PacBio has no §3.3 orientation audit and never uses -uf (A47.3 rule 3). ---

# --- Completion marker bound to reads, reference and the fixed PacBio preset ---
KEY="$FA_MD5:$REF_MD5:splice_hq"
if [ -e "$DONE" ]; then
  # Structural presence is not identity: a BAM that was truncated and rebuilt still passes
  # quickcheck. Compare against the hashes the marker itself recorded.
  HAVE=$(awk -F= '$1=="key"{print $2}' "$DONE")
  ok=1
  [ "$HAVE" = "$KEY" ] || ok=0
  [ "$ok" -eq 1 ] && { [ -s "$BAM" ] && [ -s "$BAM.bai" ] && [ -s "$PAF" ] && [ -s "$PROV" ]; } || ok=0
  [ "$ok" -eq 1 ] && "$BIN"/samtools quickcheck -v "$BAM" || ok=0
  [ "$ok" -eq 1 ] && gzip -t "$PAF" 2>/dev/null || ok=0
  [ "$ok" -eq 1 ] && [ "$(md5sum "$BAM" | cut -d' ' -f1)" = "$(awk -F= '$1=="bam_md5"{print $2}' "$DONE")" ] || ok=0
  [ "$ok" -eq 1 ] && [ "$(md5sum "$PAF" | cut -d' ' -f1)" = "$(awk -F= '$1=="paf_md5"{print $2}' "$DONE")" ] || ok=0
  if [ "$ok" -eq 1 ]; then
    echo "$RUN already aligned for key=$KEY"; exit 0
  fi
  echo "REFUSED: $RUN has a DONE marker that does not match the current inputs or its own outputs" >&2
  exit 6
fi

# §3.3 item 1 is a PRE-alignment audit, so the contig alias table is written from the REFERENCE
# before the aligner runs, not read back out of the BAM header afterwards. Same mapping either way
# (the BAM header @SQ lines are the reference's sequences), but the ordering the protocol asks for.
grep '^>' "$REF" | sed 's/^>//; s/[[:space:]].*//' | awk -v sp="$SP" '{
    n=$0; a=n; sub("^[Cc]hr","",a);
    if(a=="C") a="Pt";
    if(a=="M") a="Mt";
    if(a=="Pt" || a=="Mt" || a ~ /^[0-9]+$/) print sp"\t"n"\t"a"\tfrozen-naming";
    else print sp"\t"n"\t"n"\tunchanged"
  }' > "$OUT/contig_alias.tsv"
[ -s "$OUT/contig_alias.tsv" ] || { echo "REFUSED: empty contig alias table for $SP" >&2; exit 3; }

CMD="minimap2 -t $THREADS -ax splice:hq -G 200000 --secondary=no --cs=long $REF $FA"
START=$(date -Is); SECONDS=0
set -o pipefail
"$BIN"/minimap2 -t "$THREADS" -ax splice:hq -G 200000 --secondary=no --cs=long "$REF" "$FA" \
  2> "$OUT/minimap2.err" \
  | "$BIN"/samtools sort -@ 4 -m 2G -o "$BAM" - 2> "$OUT/samtools.err"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the alignment pipeline exited ${st[$i]} for $RUN" >&2; rm -f "$BAM"; exit 4; }
done
"$BIN"/samtools quickcheck -v "$BAM" || { echo "REFUSED: $BAM fails quickcheck" >&2; rm -f "$BAM"; exit 4; }
"$BIN"/samtools index "$BAM" || { echo "REFUSED: samtools index failed for $RUN" >&2; exit 4; }

"$BIN"/minimap2 -t "$THREADS" -x splice:hq -G 200000 --secondary=no --cs=long "$REF" "$FA" \
  2> "$OUT/minimap2.paf.err" | gzip -1 > "$PAF"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the PAF pipeline exited ${st[$i]} for $RUN" >&2; rm -f "$PAF"; exit 4; }
done
{ [ -s "$PAF" ] && gzip -t "$PAF" 2>/dev/null; } || { echo "REFUSED: missing or corrupt PAF for $RUN" >&2; rm -f "$PAF"; exit 4; }

"$BIN"/samtools flagstat "$BAM" > "$OUT/flagstat.txt" \
  || { echo "REFUSED: samtools flagstat failed for $RUN" >&2; exit 4; }

{
  echo "run            $RUN"
  echo "species        $SP"
  echo "dataset        $DS"
  echo "role           c2_training_eligible   (by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
  echo "reads          $FA"
  echo "reads_md5      $FA_MD5"
  echo "reference      $REF"
  echo "reference_md5  $REF_MD5"
  echo "uf             none   (PacBio: no §3.3 audit, never -uf; A47.3 rule 3)"
  echo "product        $(awk -F'\t' -v r="$RUN" '$2==r{print $13}' "$ROLES" | cut -c1-160)"
  echo "minimap2       $("$BIN"/minimap2 --version)"
  echo "samtools       $("$BIN"/samtools --version | head -1)"
  echo "command        $CMD"
  echo "started        $START"
  echo "finished       $(date -Is)"
  echo "wall_seconds   $SECONDS"
  echo "bam            $BAM"
  echo "bam_md5        $(md5sum "$BAM" | cut -d' ' -f1)"
  echo "paf            $PAF"
  echo "paf_md5        $(md5sum "$PAF" | cut -d' ' -f1)"
  echo "contig_alias   $OUT/contig_alias.tsv"
  echo "slurm_job      ${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
} > "$PROV" || { echo "REFUSED: could not write $PROV for $RUN" >&2; exit 4; }
[ -s "$PROV" ] || { echo "REFUSED: $PROV is empty for $RUN" >&2; exit 4; }

PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  # A task that was killed after appending and before writing DONE will run again. Without this
  # the central table grows a second row for the same run and key, and nothing says which is live.
  if grep -qF "	longread_training_evidence	" "$PROVTSV" 2>/dev/null \
     && grep -qF "run $RUN; " "$PROVTSV" 2>/dev/null; then
    grep -vF "run $RUN; " "$PROVTSV" > "$PROVTSV.tmp" && mv "$PROVTSV.tmp" "$PROVTSV"
  fi
  printf "%s\tlongread_training_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$FA" "$FA_MD5" "$REF" "$REF_MD5" "$("$BIN"/minimap2 --version)" "$CMD" \
    "$BAM;$PAF" "$(md5sum "$BAM" | cut -d' ' -f1);$(md5sum "$PAF" | cut -d' ' -f1)" \
    "run $RUN; PacBio splice:hq, no -uf (A47.3); counts are reads" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock" || { echo "REFUSED: could not append central provenance for $RUN" >&2; exit 4; }
grep -qF "run $RUN; " "$PROVTSV" || { echo "REFUSED: central provenance row for $RUN is not present after the append" >&2; exit 4; }

{
  printf 'key=%s\n' "$KEY"
  printf 'run=%s\n' "$RUN"
  printf 'preset=splice:hq\n'
  printf 'bam_md5=%s\n' "$(md5sum "$BAM" | cut -d' ' -f1)"
  printf 'paf_md5=%s\n' "$(md5sum "$PAF" | cut -d' ' -f1)"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp" || { echo "REFUSED: could not write the completion marker for $RUN" >&2; exit 4; }
mv "$DONE.tmp" "$DONE" || { echo "REFUSED: could not install the completion marker for $RUN" >&2; exit 4; }
echo "$RUN $SP splice:hq finished in ${SECONDS}s"
