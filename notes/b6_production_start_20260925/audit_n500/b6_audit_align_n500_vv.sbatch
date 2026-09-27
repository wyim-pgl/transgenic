#!/usr/bin/env bash
#SBATCH --job-name=b6_audit_n500_vv
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/audit_align_n500/logs/%x_%A_%a.out
#SBATCH --array=8,127-137,178-186%6
#
# N=500 VARIANT (author decision 17, 2026-09-27): identical to b6_audit_align.sbatch except -N 500 (was 50),
# OUTROOT audit_align_n500 and the task list (Vvinifera, all sources: runs.tsv indices 8,127-137,178-186). Consumed by b6_build_c0.py
# --audit-n 500 (check_audit_cl requires exactly -N <audit-n>).
#
# B6 audit alignment: PROTOCOL_B1 A46 rule 4(b), command spec v2 SPEC-1 step 1.
# Used ONLY for leakage exclusion, NEVER for labels. One task per primary alignment unit, from
# the run list written by b6_audit_runs.py (9 EST species, 129 ONT runs, 49 PacBio runs = 187).
#
# The command is each source's frozen primary command with ONLY `--secondary=no` replaced by
# `--secondary=yes -N 50 -p 0.5`:
#     EST     minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF est.fa   (§3)
#     ONT     minimap2 -t 32 -ax splice [-uf] -k14 -G 200000 --secondary=no --cs=long REF reads
#             (-uf exactly as the run's §3.3 item 2 verdict, as in longread_align.sbatch)
#     PacBio  minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF reads (no -uf)
# The primary string is rebuilt here exactly as the primary driver builds it, and must equal the
# `command` line that driver recorded in its PROVENANCE.txt; the audit string is derived from it
# by one textual substitution and executed as that string. No other flag can drift.
# Only the BAM is produced (placements, flags, MAPQ); no PAF, no labels.
#
# Gates (fail closed, same as the primary drivers): A18.3 role; §3.3 item 1 reference (and GFF
# for long reads) md5 against b5_species_v1.tsv; EST: A21-screened FASTA, sidecar md5, A37
# primary floor 100 nt; ONT: §3.3 item 2 audit bound to reads/ref/GFF, A39 scoring, verdict
# consistency; primary DONE bound to the same reads/reference. DONE key = reads:ref:command md5.
#
# Output layout (the contract the C0 builder b6_build_c0.py reads; b6_audit_runs.py --list prints
# all 187 paths without running anything):
#     evidence/audit_align_v1/<species>/<dataset>/<run>/<run>.audit.bam   (+ PROVENANCE.txt, DONE)
#     EST     <dataset> = est, <run> = est_primary  -> <sp>/est/est_primary/est_primary.audit.bam
#     ONT     <dataset> = basename of the runs-TSV dataset dir, <run> = SRR id
#     PacBio  same as ONT (longread_runs_pacbio_v1.tsv)
#     logs    evidence/audit_align_v1/logs/b6_audit_align_<jobid>_<task>.out
#
# B6_DRY_RUN=1 runs every gate and prints the expanded PRIMARY/AUDIT commands without aligning
# or writing anything. B6_ROOT/B6_BIN/B6_PY/B6_RUNLIST exist for fixtures and dry runs only.
set -uo pipefail

ROOT=${B6_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic}
BIN=${B6_BIN:-/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin}
PY=${B6_PY:-/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python}
RUNLIST=${B6_RUNLIST:-$ROOT/evidence/audit_align_v1/runs.tsv}
DRY_RUN=${B6_DRY_RUN:-0}
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
EST_ROLES=$ROOT/evidence/DATASET_ROLES.est_v1.tsv
LR_ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
AUDITROOT=$ROOT/evidence/orientation_audit_v1
OUTROOT=$ROOT/evidence/audit_align_n500
THREADS=${SLURM_CPUS_PER_TASK:-32}
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SECONDARY_FROZEN="--secondary=no"
SECONDARY_AUDIT="--secondary=yes -N 500 -p 0.5"

refuse() { echo "REFUSED: $*" >&2; exit "${CODE:-2}"; }
md5() { md5sum "$1" | cut -d' ' -f1; }

[ -s "$RUNLIST" ] || refuse "missing run list $RUNLIST (build it with b6_audit_runs.py)"
IFS=$'\t' read -r IDX SOURCE SP DSBASE RUN READS UF PRIMARY_DIR < <(awk -F'\t' -v i="$TASK" '$1==i' "$RUNLIST")
[ "${IDX:-}" = "$TASK" ] || refuse "no row for task $TASK in $RUNLIST"
READS=$ROOT/$READS
PRIMARY_DIR=$ROOT/$PRIMARY_DIR
OUT=$OUTROOT/$SP/$DSBASE/$RUN
BAM=$OUT/$RUN.audit.bam
PROV=$OUT/PROVENANCE.txt
DONE=$OUT/DONE

# --- Gate 1: A18.3 role ---
CODE=2
case "$SOURCE" in
  est)
    ROLE=$(awk -F'\t' -v d="est/$SP" '$1==d{print $12}' "$EST_ROLES")
    [ "$ROLE" = "c2_training_eligible" ] || refuse "est/$SP has role '${ROLE:-<none>}', not c2_training_eligible"
    ROLES=$EST_ROLES;;
  ont|pacbio)
    n_eli=$(awk -F'\t' -v r="$RUN" '$2==r && $12=="c2_training_eligible"' "$LR_ROLES" | wc -l)
    [ "$n_eli" -eq 1 ] || refuse "$RUN has $n_eli c2_training_eligible rows, expected exactly 1"
    ROLES=$LR_ROLES;;
  *) refuse "unknown source '$SOURCE'";;
esac
case "$SOURCE:$UF" in est:none|pacbio:none|ont:none|ont:-uf) ;; *) refuse "invalid uf '$UF' for $SOURCE";; esac

# --- Gate 2: §3.3 item 1 ---
CODE=3
REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
[ -f "$REF" ] || refuse "missing genome for $SP: ${REF:-<none>}"
REF_MD5=$(md5 "$REF")
[ "$REF_MD5" = "$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")" ] \
  || refuse "$SP reference md5 $REF_MD5 does not match $(basename "$MANIFEST")"
if [ "$SOURCE" != est ]; then
  GFF=$(awk -F'\t' -v s="$SP" '$1==s{print $6}' "$MANIFEST")
  [ -f "$GFF" ] || refuse "missing GFF for $SP"
  GFF_MD5=$(md5 "$GFF")
  [ "$GFF_MD5" = "$(awk -F'\t' -v s="$SP" '$1==s{print $7}' "$MANIFEST")" ] \
    || refuse "$SP GFF md5 $GFF_MD5 does not match $(basename "$MANIFEST")"
fi
{ [ -s "$READS" ] && gzip -t "$READS" 2>/dev/null; } || refuse "missing or corrupt reads: $READS"
READS_MD5=$(md5 "$READS")

# --- Gate 3: source-specific input gates ---
CODE=5
case "$SOURCE" in
  est)
    [ -f "$(dirname "$READS")/DONE" ] || refuse "$SP has no A21 screening DONE"
    SIDECAR=$(awk 'NR==1{print $1}' "$READS.md5" 2>/dev/null)
    [ -z "$SIDECAR" ] || [ "$SIDECAR" = "$READS_MD5" ] || refuse "$READS.md5 says $SIDECAR, file is $READS_MD5"
    OBS_MIN=$(gzip -cd "$READS" | awk '/^>/{if(seen && n<m) m=n; seen=1; n=0; next}
         {gsub(/[[:space:]]/,""); n+=length} END{if(seen){if(n<m) m=n; print m}}' m=2147483647)
    { [ -n "$OBS_MIN" ] && [ "$OBS_MIN" -ge 100 ]; } || refuse "$READS minimum record ${OBS_MIN:-none} < 100 (A37 primary arm)";;
  ont)
    AUDIT=$AUDITROOT/$SP/$DSBASE/$RUN
    { [ -s "$AUDIT/DONE" ] && [ -s "$AUDIT/audit.json" ]; } || refuse "$RUN has no §3.3 orientation audit"
    WANT_KEY="$READS_MD5:$REF_MD5:$GFF_MD5"
    HAVE_KEY=$(awk -F= '$1=="key"{print $2}' "$AUDIT/DONE")
    case "$HAVE_KEY" in "$WANT_KEY":?*) ;; *) refuse "$RUN audit key $HAVE_KEY is not $WANT_KEY:<A39 scorer>";; esac
    AUDIT_MD5=$(md5 "$AUDIT/audit.json")
    [ "$(awk -F= '$1=="audit_md5"{print $2}' "$AUDIT/DONE")" = "$AUDIT_MD5" ] || refuse "$RUN audit.json changed since its DONE"
    read -r STATUS J_USEUF J_READS J_REF J_GFF J_A39 < <("$PY" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["status"], d["use_uf"], d.get("reads_md5",""), d.get("reference_md5",""),
      d.get("gff_md5",""), "A39" in d.get("protocol",""))
' "$AUDIT/audit.json")
    [ "$J_A39" = True ] || refuse "$RUN audit.json was not scored under A39"
    [ "$STATUS" = "$(awk -F= '$1=="status"{print $2}' "$AUDIT/DONE")" ] || refuse "$RUN audit.json/DONE status disagree"
    { [ "$J_READS" = "$READS_MD5" ] && [ "$J_REF" = "$REF_MD5" ] && [ "$J_GFF" = "$GFF_MD5" ]; } \
      || refuse "$RUN audit.json was computed from different objects"
    case "$STATUS:$J_USEUF" in
      PASS:True) WANT_UF=-uf;; FAIL:False|UNRESOLVED:False) WANT_UF=none;;
      *) refuse "$RUN audit status=$STATUS use_uf=$J_USEUF is not a valid verdict";;
    esac
    [ "$UF" = "$WANT_UF" ] || refuse "$RUN run list says uf=$UF, the orientation verdict says $WANT_UF";;
esac

# --- Gate 4: the primary alignment this audit mirrors ---
CODE=6
case "$SOURCE" in
  est)
    PRIMARY="minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long $REF $READS"
    [ "$(awk -F= '$1=="est_md5"{print $2}' "$PRIMARY_DIR/DONE")" = "$READS_MD5" ] \
      || refuse "$SP primary est_align DONE was built from a different screened FASTA";;
  ont)
    UFARG=""; [ "$UF" = -uf ] && UFARG="-uf"
    PRIMARY="minimap2 -t $THREADS -ax splice $UFARG -k14 -G 200000 --secondary=no --cs=long $REF $READS"
    [ "$(awk -F= '$1=="key"{print $2}' "$PRIMARY_DIR/DONE")" = "$READS_MD5:$REF_MD5:$AUDIT_MD5" ] \
      || refuse "$RUN primary DONE is bound to different reads/reference/orientation audit"
    [ "$(awk -F= '$1=="uf"{print $2}' "$PRIMARY_DIR/DONE")" = "$UF" ] || refuse "$RUN primary used a different -uf";;
  pacbio)
    PRIMARY="minimap2 -t $THREADS -ax splice:hq -G 200000 --secondary=no --cs=long $REF $READS"
    [ "$(awk -F= '$1=="key"{print $2}' "$PRIMARY_DIR/DONE")" = "$READS_MD5:$REF_MD5:splice_hq" ] \
      || refuse "$RUN primary DONE is bound to different reads/reference";;
esac
RECORDED=$(sed -n 's/^command[[:space:]]\{1,\}//p' "$PRIMARY_DIR/PROVENANCE.txt")
[ "$RECORDED" = "$PRIMARY" ] || refuse "rebuilt primary command differs from the recorded one:
  rebuilt : $PRIMARY
  recorded: $RECORDED"
case "$PRIMARY" in *" $SECONDARY_FROZEN "*) ;; *) refuse "primary command lacks $SECONDARY_FROZEN";; esac
AUDIT_CMD=${PRIMARY/ $SECONDARY_FROZEN / $SECONDARY_AUDIT }
CMD_MD5=$(printf '%s' "$AUDIT_CMD" | md5sum | cut -d' ' -f1)
KEY="$READS_MD5:$REF_MD5:$CMD_MD5"

if [ "$DRY_RUN" = 1 ]; then
  printf 'TASK\t%s\t%s\t%s\t%s\t%s\n' "$TASK" "$SOURCE" "$SP" "$DSBASE" "$RUN"
  printf 'PRIMARY\t%s\n' "$PRIMARY"
  printf 'AUDIT\t%s\n' "$AUDIT_CMD"
  printf 'OUT\t%s\n' "$OUT"
  printf 'BAM\t%s\n' "$BAM"
  printf 'KEY\t%s\n' "$KEY"
  exit 0
fi

# --- Completion marker bound to reads, reference and the exact audit command ---
if [ -e "$DONE" ]; then
  ok=1
  [ "$(awk -F= '$1=="key"{print $2}' "$DONE")" = "$KEY" ] || ok=0
  [ "$ok" -eq 1 ] && { [ -s "$BAM" ] && [ -s "$BAM.bai" ] && [ -s "$PROV" ]; } || ok=0
  [ "$ok" -eq 1 ] && "$BIN"/samtools quickcheck -v "$BAM" || ok=0
  [ "$ok" -eq 1 ] && [ "$(md5 "$BAM")" = "$(awk -F= '$1=="bam_md5"{print $2}' "$DONE")" ] || ok=0
  [ "$ok" -eq 1 ] && { echo "$RUN audit already aligned for key=$KEY"; exit 0; }
  refuse "$RUN has an audit DONE that does not match the current inputs or its own outputs"
fi

CODE=4
mkdir -p "$OUT" "$OUTROOT/logs" || refuse "cannot create $OUT"
read -r -a ARGV <<< "$AUDIT_CMD"
ARGV[0]="$BIN/${ARGV[0]}"
START=$(date -Is); SECONDS=0
"${ARGV[@]}" 2> "$OUT/minimap2.err" | "$BIN"/samtools sort -@ 4 -m 2G -o "$BAM" - 2> "$OUT/samtools.err"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { rm -f "$BAM"; refuse "stage $i of the audit pipeline exited ${st[$i]} for $RUN"; }
done
"$BIN"/samtools quickcheck -v "$BAM" || { rm -f "$BAM"; refuse "$BAM fails quickcheck"; }
"$BIN"/samtools index "$BAM" || refuse "samtools index failed for $RUN"
"$BIN"/samtools flagstat "$BAM" > "$OUT/flagstat.txt" || refuse "samtools flagstat failed for $RUN"
BAM_MD5=$(md5 "$BAM")

{
  echo "purpose        A46 rule 4(b) audit alignment: leakage exclusion only, never labels"
  echo "source         $SOURCE"
  echo "run            $RUN"
  echo "species        $SP"
  echo "dataset        $DSBASE"
  echo "role           c2_training_eligible   ($(basename "$ROLES"), md5 $(md5 "$ROLES"))"
  echo "reads          $READS"
  echo "reads_md5      $READS_MD5"
  echo "reference      $REF"
  echo "reference_md5  $REF_MD5"
  echo "uf             $UF"
  echo "primary        $PRIMARY_DIR"
  echo "primary_cmd    $PRIMARY"
  echo "command        $AUDIT_CMD"
  echo "command_md5    $CMD_MD5"
  echo "minimap2       $("$BIN"/minimap2 --version)"
  echo "samtools       $("$BIN"/samtools --version | head -1)"
  echo "started        $START"
  echo "finished       $(date -Is)"
  echo "wall_seconds   $SECONDS"
  echo "bam            $BAM"
  echo "bam_md5        $BAM_MD5"
  echo "slurm_job      ${SLURM_ARRAY_JOB_ID:-NA}_${TASK} on $(hostname)"
} > "$PROV" || refuse "could not write $PROV"
[ -s "$PROV" ] || refuse "$PROV is empty"

PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  if grep -qF "b6 audit $SOURCE $RUN $SP; " "$PROVTSV" 2>/dev/null; then
    grep -vF "b6 audit $SOURCE $RUN $SP; " "$PROVTSV" > "$PROVTSV.tmp" && mv "$PROVTSV.tmp" "$PROVTSV"
  fi
  printf "%s\tb6_audit_alignment\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$READS" "$READS_MD5" "$REF" "$REF_MD5" "$("$BIN"/minimap2 --version)" \
    "$AUDIT_CMD" "$BAM" "$BAM_MD5" "b6 audit $SOURCE $RUN $SP; exclusion only (A46 rule 4(b)); uf=$UF" >> "$PROVTSV"
} 9>"$PROVTSV.lock" || refuse "could not append central provenance for $RUN"

{
  printf 'key=%s\n' "$KEY"
  printf 'source=%s\n' "$SOURCE"
  printf 'run=%s\n' "$RUN"
  printf 'uf=%s\n' "$UF"
  printf 'command_md5=%s\n' "$CMD_MD5"
  printf 'bam_md5=%s\n' "$BAM_MD5"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp" || refuse "could not write the completion marker for $RUN"
mv "$DONE.tmp" "$DONE" || refuse "could not install the completion marker for $RUN"
echo "$SOURCE $RUN $SP audit alignment finished in ${SECONDS}s"
