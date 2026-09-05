#!/usr/bin/env bash
#SBATCH --job-name=lr_align
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/longread_align_v1/logs/%x_%A_%a.out
#SBATCH --array=0-128%4
#
# ONT long-read -> genome spliced alignment. Protocol B1 §4 (frozen command) + §3.3 + A18.3.
#
# One task per run. §3.3 item 4: "every run is aligned and scored separately; pooled results are
# reported in addition, never instead" - so there is no per-dataset shortcut here even though runs
# of a dataset share a library prep.
#
# -uf IS NOT A SETTING OF THIS SCRIPT. §3.3 item 2 decides it per run from a measurement, and this
# driver only reads that verdict:
#     PASS       -> -uf         (orientation agreement >= 95%)
#     FAIL       -> no -uf
#     UNRESOLVED -> no -uf      (too few eligible alignments to apply the frozen rule)
#     no audit, or an audit built from different inputs -> REFUSE the run
# The audit's DONE marker is bound to reads+reference+GFF md5s, so an audit that predates a changed
# input cannot be mistaken for a current one.
#
# Frozen §4 commands, quoted, not varied:
#     minimap2 -t 32 -ax splice       -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
#     minimap2 -t 32 -ax splice -uf   -k14 -G 200000 --secondary=no --cs=long REF.fa reads.fq.gz
# -t 32 is kept because §4 freezes it literally. Sixteen threads would use the cluster better
# (measured parallel efficiency at 32 was 54%), but running four 32-thread tasks instead of six
# 16-thread ones gives the same aggregate throughput without varying a frozen command.
# §4 also requires BOTH a sorted BAM and a PAF carrying --cs=long to be retained. The PAF is a
# second pass with -x. It cannot be derived from the BAM: measured 2026-09-04, paftools sam2paf
# reproduces the record count exactly but drops every ts:A: tag (56,501 -> 0 on Ptrichocarpa EST),
# and §5 reads the strand of a junction from that tag.
#
# WHAT THIS DRIVER DOES NOT DO. §4's acceptance rules - primary only (0x900 unset), MAPQ >= 20,
# supplementary/chimeric discarded, reads with primary blocks on different chromosomes or strands
# discarded - are NOT applied here. The retained BAM and PAF are the RAW alignment, deliberately,
# because discarding at this stage would make the counts unrecoverable and §4 asks for the
# alignment to be retained. Every consumer (§5 junction calling, the evidence scorer, issue #24)
# must apply those filters itself; an unfiltered read of these files is a protocol violation on
# the consumer's side, not a licence granted by this file.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
RUNS=$ROOT/evidence/longread_runs_v1.tsv
AUDITROOT=$ROOT/evidence/orientation_audit_v1
OUTROOT=$ROOT/evidence/longread_align_v1
THREADS=${SLURM_CPUS_PER_TASK:-32}

read -r SP DS RUN FA < <(awk -F'\t' -v i="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==i' "$RUNS")
[ -n "${RUN:-}" ] || { echo "REFUSED: no run at index $SLURM_ARRAY_TASK_ID in $RUNS" >&2; exit 2; }
FA=$ROOT/$FA
DSBASE=$(basename "$DS")
OUT=$OUTROOT/$SP/$DSBASE/$RUN
AUDIT=$AUDITROOT/$SP/$DSBASE/$RUN
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

# --- Gate 3: §3.3 item 2 verdict. No audit, no alignment. ---
[ -s "$AUDIT/DONE" ] && [ -s "$AUDIT/audit.json" ] \
  || { echo "REFUSED: $RUN has no §3.3 orientation audit at $AUDIT" >&2; exit 5; }
# The audit's key names the data it measured AND the scorer that measured it (A39). This driver
# checks the data part - it must be the objects we are about to align - and records the scorer,
# but does not pin a scorer md5: that would couple the aligner to a version number instead of to a
# meaning. What it does require is the meaning: an audit written under the superseded reading,
# which gated on the splice motif rather than the read, must not be accepted.
WANT_KEY="$FA_MD5:$REF_MD5:$GFF_MD5"
HAVE_KEY=$(awk -F= '$1=="key"{print $2}' "$AUDIT/DONE")
AUDIT_SCORER=${HAVE_KEY##*:}
case "$HAVE_KEY" in
  "$WANT_KEY"|"$WANT_KEY":*) ;;
  *) echo "REFUSED: $RUN audit was built from different inputs (have=$HAVE_KEY want=$WANT_KEY)" >&2; exit 5;;
esac
[ "$AUDIT_SCORER" != "$HAVE_KEY" ] \
  || { echo "REFUSED: $RUN audit predates A39 (its key names no scorer); rescore it" >&2; exit 5; }
AUDIT_MD5_REC=$(awk -F= '$1=="audit_md5"{print $2}' "$AUDIT/DONE")
AUDIT_MD5=$(md5sum "$AUDIT/audit.json" | cut -d' ' -f1)
[ "$AUDIT_MD5_REC" = "$AUDIT_MD5" ] \
  || { echo "REFUSED: $RUN audit.json changed since its DONE was written" >&2; exit 5; }
# The marker beside a measurement is not the measurement. A DONE saying PASS next to an
# md5-valid audit.json saying FAIL would have enabled -uf; the verdict is taken FROM THE JSON and
# the marker only has to agree with it.
STATUS_MARK=$(awk -F= '$1=="status"{print $2}' "$AUDIT/DONE")
read -r STATUS J_USEUF J_READS J_REF J_GFF J_A39 < <("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["status"], d["use_uf"], d.get("reads_md5",""), d.get("reference_md5",""),
      d.get("gff_md5",""), "A39" in d.get("protocol",""))
' "$AUDIT/audit.json")
# A39 is what makes the verdict mean "the reads are in transcript orientation" rather than "the
# splice motifs agree with the annotation". Without it the number is not a -uf gate at all.
[ "$J_A39" = "True" ] \
  || { echo "REFUSED: $RUN audit.json was not scored under A39; its verdict is not a -uf gate" >&2; exit 5; }
[ "$STATUS" = "$STATUS_MARK" ] \
  || { echo "REFUSED: $RUN audit.json says $STATUS but its DONE says $STATUS_MARK" >&2; exit 5; }
# use_uf and status must not disagree with each other either.
case "$STATUS:$J_USEUF" in
  PASS:True|FAIL:False|UNRESOLVED:False) ;;
  *) echo "REFUSED: $RUN audit.json status=$STATUS is inconsistent with use_uf=$J_USEUF" >&2; exit 5;;
esac
# and the measurement must have been made against the objects we are about to align.
[ "$J_READS" = "$FA_MD5" ] && [ "$J_REF" = "$REF_MD5" ] && [ "$J_GFF" = "$GFF_MD5" ] \
  || { echo "REFUSED: $RUN audit.json was computed from different objects" >&2; exit 5; }
case "$STATUS" in
  PASS)             UF="-uf"; UFNOTE="§3.3 item 2 PASS";;
  FAIL|UNRESOLVED)  UF="";    UFNOTE="§3.3 item 2 $STATUS";;
  *) echo "REFUSED: $RUN audit status '${STATUS:-<none>}' is not PASS|FAIL|UNRESOLVED" >&2; exit 5;;
esac
FRAC=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['fraction'])" "$AUDIT/audit.json")

# --- Completion marker bound to reads, reference and the audit that chose -uf ---
KEY="$FA_MD5:$REF_MD5:$AUDIT_MD5"
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

CMD="minimap2 -t $THREADS -ax splice $UF -k14 -G 200000 --secondary=no --cs=long $REF $FA"
START=$(date -Is); SECONDS=0
set -o pipefail
# shellcheck disable=SC2086  # $UF is deliberately unquoted: it is either "-uf" or nothing.
"$BIN"/minimap2 -t "$THREADS" -ax splice $UF -k14 -G 200000 --secondary=no --cs=long "$REF" "$FA" \
  2> "$OUT/minimap2.err" \
  | "$BIN"/samtools sort -@ 4 -m 2G -o "$BAM" - 2> "$OUT/samtools.err"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the alignment pipeline exited ${st[$i]} for $RUN" >&2; rm -f "$BAM"; exit 4; }
done
"$BIN"/samtools quickcheck -v "$BAM" || { echo "REFUSED: $BAM fails quickcheck" >&2; rm -f "$BAM"; exit 4; }
"$BIN"/samtools index "$BAM" || { echo "REFUSED: samtools index failed for $RUN" >&2; exit 4; }

# shellcheck disable=SC2086
"$BIN"/minimap2 -t "$THREADS" -x splice $UF -k14 -G 200000 --secondary=no --cs=long "$REF" "$FA" \
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
  echo "orientation    $UFNOTE, agreement $FRAC -> ${UF:-no -uf}"
  echo "audit          $AUDIT/audit.json"
  echo "audit_md5      $AUDIT_MD5"
  echo "audit_scorer   $AUDIT_SCORER   (md5 of 67_orientation_audit.py that produced the verdict)"
  echo "umi_status     unknown (§3.3 item 3: ENA metadata publishes no kit or UMI field for these runs; counts are READS, never molecules, and no endpoint deduplication is performed)"
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
    "run $RUN; $UFNOTE agreement $FRAC -> ${UF:-no -uf}; audit md5 $AUDIT_MD5; UMI unknown, counts are reads" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock" || { echo "REFUSED: could not append central provenance for $RUN" >&2; exit 4; }
grep -qF "run $RUN; " "$PROVTSV" || { echo "REFUSED: central provenance row for $RUN is not present after the append" >&2; exit 4; }

{
  printf 'key=%s\n' "$KEY"
  printf 'run=%s\n' "$RUN"
  printf 'uf=%s\n' "${UF:-none}"
  printf 'audit_status=%s\n' "$STATUS"
  printf 'bam_md5=%s\n' "$(md5sum "$BAM" | cut -d' ' -f1)"
  printf 'paf_md5=%s\n' "$(md5sum "$PAF" | cut -d' ' -f1)"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp" || { echo "REFUSED: could not write the completion marker for $RUN" >&2; exit 4; }
mv "$DONE.tmp" "$DONE" || { echo "REFUSED: could not install the completion marker for $RUN" >&2; exit 4; }
echo "$RUN $SP ${UF:-no-uf} finished in ${SECONDS}s"
