#!/usr/bin/env bash
#SBATCH --job-name=est_align
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_align/logs/%x_%A_%a.out
#SBATCH --array=0-17
#
# EST -> genome spliced alignment, protocol B1 §3 (frozen command) + A8/A11/A21/A37.
#
# A37 (v1.27) replaced A36. There are now TWO labelled ingestion arms and A37 line 758 requires
# BOTH to be aligned with the §4 command and kept side by side:
#     tasks 0-8   primary      >= 100 nt   -> evidence/est_align/
#     tasks 9-17  sensitivity  >= 121 nt   -> evidence/est_align_min121/
# A37 item 3 ("wherever an EST-derived quantity appears, both arms appear") is what makes the
# sensitivity arm part of this run rather than a later errand.
#
# Three gates run before the aligner, and all fail closed:
#   1. A18.3 dataset-role manifest: refuse to align evidence that is not declared training-eligible.
#   2. §3.3 item 1: the reference FASTA must match the md5 that b5_species_v1.tsv declares for it.
#      The provenance note used to *say* this was checked while the code only read the path.
#   3. A21 UniVec screening + the A37 arm floor: refuse to align anything but the SCREENED FASTA,
#      and refuse if its observed minimum record length is below the floor the arm claims.
#      The 2026-09-03 run read est.fa.gz directly, skipped the screening, and nothing stopped it —
#      that BAM set is retired in evidence/RETIRED_DO_NOT_USE/est_align_prescreen_20260903/.
#
# The command below is quoted from PROTOCOL_B1_frozen_v1.md §3 and must not be varied:
#     minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long REF.fa est.fa
# --cs=long is what the evidence scorer reads; --secondary=no is what makes "one accepted EST supports
# at most one locus" (A11) enforceable; -G 200000 is the intron cap of §3.
#
# Only the nine TRAINING species run here. Z. mays and S. lycopersicum EST are b1_validation_only in
# evidence/DATASET_ROLES.est_v1.tsv (md5 a6375713, frozen into protocol §1) — they are the held-out
# test species and must not be aligned as part of the training-evidence pass.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
ROLES=$ROOT/evidence/DATASET_ROLES.est_v1.tsv

SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
SP=${SPECIES[$((SLURM_ARRAY_TASK_ID % 9))]}
UVD=$ROOT/evidence/est/$SP/univec

# --- Arm selection (A37). The §3 command is identical in both arms; only the input differs. ---
if [ "$SLURM_ARRAY_TASK_ID" -lt 9 ]; then
  ARM=primary;     MIN_LEN=100; OUT=$ROOT/evidence/est_align
  EST=$UVD/est.univec.fa.gz;        SUMMARY=$UVD/summary.json
else
  ARM=sensitivity; MIN_LEN=121; OUT=$ROOT/evidence/est_align_min121
  EST=$UVD/est.univec.min121.fa.gz; SUMMARY=$UVD/summary.min121.json
fi
mkdir -p "$OUT/logs" "$OUT/$SP"

BAM=$OUT/$SP/$SP.est.bam
PAF=$OUT/$SP/$SP.est.paf.gz
PROV=$OUT/$SP/PROVENANCE.txt
DONE=$OUT/$SP/DONE

# --- Gate 1: role manifest (A18.3) ---
ROLE=$(awk -F'\t' -v d="est/$SP" '$1==d{print $12}' "$ROLES")
if [ "$ROLE" != "c2_training_eligible" ]; then
  echo "REFUSED: est/$SP has role '${ROLE:-<none>}' in $(basename "$ROLES"), not c2_training_eligible" >&2
  exit 2
fi

REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
RAW=$ROOT/evidence/est/$SP/est.fa.gz
[ -f "$REF" ] || { echo "missing genome: $REF" >&2; exit 3; }
[ -f "$RAW" ] || { echo "missing EST: $RAW" >&2; exit 3; }

# --- Gate 2: §3.3 item 1 — the reference we align to must be the reference the manifest declares. ---
EXPECTED_REF_MD5=$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")
ACTUAL_REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)
if [ -z "$EXPECTED_REF_MD5" ] || [ "$ACTUAL_REF_MD5" != "$EXPECTED_REF_MD5" ]; then
  echo "REFUSED: reference md5 $ACTUAL_REF_MD5 != $(basename "$MANIFEST") ${EXPECTED_REF_MD5:-<none>} for $SP" >&2
  exit 3
fi

# --- Gate 3: A21 screening + the A37 arm floor. ---
# The primary arm can be produced here if the screening never ran. The sensitivity arm cannot:
# it is written only by retrim_a37.sh with an explicit --min-len 121, and A37 requires that arm to
# always be an explicit act rather than something a driver conjures as a side effect.
if [ "$ARM" = primary ] && [ ! -f "$UVD/DONE" ]; then
  echo "$SP: A21 screening not present, running univec_screen.sh"
  PATH="$BIN:$PATH" PY=$(command -v python3) \
    bash "$ROOT/transgenic/revision/scripts/evidence/univec_screen.sh" "$SP" 32 \
    > "$OUT/$SP/univec_screen.log" 2>&1
fi
if [ ! -f "$UVD/DONE" ] || [ ! -f "$EST" ]; then
  echo "REFUSED: $SP has no A21-screened FASTA for the $ARM arm ($EST). A21: only the screened FASTA is aligned." >&2
  exit 5
fi
[ -s "$EST" ] || { echo "REFUSED: empty screened FASTA: $EST" >&2; exit 5; }
gzip -t "$EST" 2>/dev/null || { echo "REFUSED: corrupt screened FASTA: $EST" >&2; exit 5; }

EST_MD5=$(md5sum "$EST" | cut -d' ' -f1)
SIDECAR_MD5=$(awk 'NR==1{print $1}' "$EST.md5" 2>/dev/null)
if [ -n "$SIDECAR_MD5" ] && [ "$SIDECAR_MD5" != "$EST_MD5" ]; then
  echo "REFUSED: $EST.md5 says $SIDECAR_MD5 but the file hashes to $EST_MD5" >&2
  exit 5
fi

# The arm label is only worth as much as the data behind it. A file named for a 100-nt floor that
# holds an 80-nt record is the failure this checks for, and it is cheap next to the alignment.
OBS_MIN=$(gzip -cd "$EST" |
  awk '/^>/{if(seen && n<m) m=n; seen=1; n=0; next}
       {gsub(/[[:space:]]/,""); n+=length}
       END{if(seen){if(n<m) m=n; print m}}' m=2147483647)
if [ -z "$OBS_MIN" ] || [ "$OBS_MIN" -lt "$MIN_LEN" ]; then
  echo "REFUSED: $EST observed minimum record length ${OBS_MIN:-none}, the $ARM arm claims >= $MIN_LEN" >&2
  exit 5
fi

# --- Completion marker, bound to the input it was produced from. ---
# A bare `touch DONE` made the marker mean "this directory was written once", so nine markers left
# over from the A36 >=121 run would have made every task of this array exit 0 while SLURM reported
# nine COMPLETED tasks and the A36 BAMs stayed in place. The marker now names the exact input md5,
# and a marker that does not match, or that predates this scheme, refuses instead of skipping.
if [ -e "$DONE" ]; then
  DONE_MD5=$(awk -F= '$1=="est_md5"{print $2}' "$DONE")
  [ -n "$DONE_MD5" ] || { echo "REFUSED: $SP has a legacy DONE marker with no est_md5; retire the old result first" >&2; exit 6; }
  [ "$DONE_MD5" = "$EST_MD5" ] || { echo "REFUSED: $SP DONE marker was built from est_md5=$DONE_MD5, current input is $EST_MD5" >&2; exit 6; }
  if [ -s "$BAM" ] && "$BIN"/samtools quickcheck -v "$BAM" && [ -s "$BAM.bai" ] \
     && [ -s "$PAF" ] && gzip -t "$PAF" 2>/dev/null && [ -s "$PROV" ] \
     && [ -s "$OUT/$SP/flagstat.txt" ] && [ -s "$OUT/$SP/contig_alias.tsv" ]; then
    echo "$SP $ARM already complete for est_md5=$EST_MD5"; exit 0
  fi
  echo "REFUSED: $SP DONE matches the input but the required outputs are incomplete" >&2
  exit 6
fi

CMD="minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long $REF $EST"
START=$(date -Is)
SECONDS=0

# Section 4 requires BOTH a sorted BAM and a PAF carrying --cs=long to be retained. Only the BAM was
# produced before; the PAF is a second pass with -x instead of -ax, the rest of the frozen command
# unchanged.
set -o pipefail
"$BIN"/minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long "$REF" "$EST" 2> "$OUT/$SP/minimap2.err" \
  | "$BIN"/samtools sort -@ 4 -m 2G -o "$BAM" - 2> "$OUT/$SP/samtools.err"
# Every stage, not just the first. Checking PIPESTATUS[0] alone reads minimap2's status and ignores
# samtools sort, so a sort that ran out of memory or disk produced a truncated BAM that this script
# then signed off as complete.
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the alignment pipeline exited ${st[$i]} for $SP" >&2; rm -f "$BAM"; exit 4; }
done
"$BIN"/samtools quickcheck -v "$BAM" || { echo "REFUSED: $BAM fails samtools quickcheck" >&2; rm -f "$BAM"; exit 4; }
"$BIN"/samtools index "$BAM" || { echo "REFUSED: samtools index failed for $SP" >&2; exit 4; }

"$BIN"/minimap2 -t 32 -x splice:hq -G 200000 --secondary=no --cs=long "$REF" "$EST" 2> "$OUT/$SP/minimap2.paf.err" \
  | gzip -1 > "$PAF"
st=("${PIPESTATUS[@]}")
for i in "${!st[@]}"; do
  [ "${st[$i]}" -eq 0 ] || { echo "REFUSED: stage $i of the PAF pipeline exited ${st[$i]} for $SP" >&2; rm -f "$PAF"; exit 4; }
done
# Exit status alone does not say the PAF arrived: gzip of an empty stream exits 0.
{ [ -s "$PAF" ] && gzip -t "$PAF" 2>/dev/null; } || { echo "REFUSED: missing or corrupt PAF for $SP" >&2; rm -f "$PAF"; exit 4; }

# Section 3.3 item 1: the alias table that maps our reference names onto the frozen contig names is
# stored with the results. The training genomes carry Chr1..ChrN/ChrM/ChrC; the frozen naming is
# 1..N/Mt/Pt. Written, not applied - renaming a finished BAM is a separate, auditable step.
# The awk program below was previously wrapped in '"'"' escaping carried over from a nested quoting
# context. In this file that sequence closes the single quote and reopens a DOUBLE one, so the shell
# expanded $1/$2 itself and `set -u` aborted the job with "$1: unbound variable" — after the BAM and
# PAF had been built. It had never run: the block was added on 2026-09-04, after the 2026-09-03 run.
"$BIN"/samtools view -H "$BAM" | awk -v sp="$SP" '$1=="@SQ"{
    n=$2; sub("^SN:","",n); a=n;
    sub("^[Cc]hr","",a);
    if(a=="C") a="Pt";
    if(a=="M") a="Mt";
    if(a=="Pt" || a=="Mt" || a ~ /^[0-9]+$/) print sp"\t"n"\t"a"\tfrozen-naming";
    else print sp"\t"n"\t"n"\tunchanged"
  }' > "$OUT/$SP/contig_alias.tsv"
[ -s "$OUT/$SP/contig_alias.tsv" ] || { echo "REFUSED: contig alias table is empty for $SP" >&2; exit 4; }

"$BIN"/samtools flagstat "$BAM" > "$OUT/$SP/flagstat.txt" \
  || { echo "REFUSED: samtools flagstat failed for $SP" >&2; exit 4; }

{
  echo "species        $SP"
  echo "arm            $ARM   (A37, v1.27)"
  echo "role           $ROLE   (from $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
  echo "reference      $REF"
  echo "reference_md5  $ACTUAL_REF_MD5"
  echo "est_raw        $RAW"
  echo "est_raw_md5    $(md5sum "$RAW" | cut -d' ' -f1)"
  echo "est_raw_count  $(awk -F'\t' -v d="est/$SP" '$1==d{print $8}' "$ROLES")"
  echo "est            $EST   (A21-screened, A37 $ARM arm)"
  echo "est_md5        $EST_MD5"
  echo "univec_db      UniVec_Core md5 $(cut -d' ' -f1 "$ROOT/evidence/univec/UniVec_Core.md5" 2>/dev/null)"
  echo "univec_version $(cat "$ROOT/evidence/univec/UniVec_Core.version" 2>/dev/null)"
  echo "univec_summary $(tr -d '\n ' < "$SUMMARY" 2>/dev/null)"
  echo "min_len        $MIN_LEN   (protocol A37, v1.27)"
  echo "observed_min   $OBS_MIN"
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
  echo "contig_alias   $OUT/$SP/contig_alias.tsv"
  echo "ref_md5_source b5_species_v1.tsv (training-species authority), verified above. Differs from the section 1 TAIR10 md5; author states the difference is headers/formatting only (2026-09-04) — stated, not measured."
  echo "slurm_job      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
} > "$PROV"

# Section 3.3 item 5: tool versions, command lines, dates and file MD5s go to one central table.
# Eighteen array tasks append here at once, so take a lock: two unlocked >> of a multi-field printf
# can interleave and produce a row that belongs to no species.
PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  printf "%s\test_training_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$EST" "$EST_MD5" \
    "$REF" "$ACTUAL_REF_MD5" "$("$BIN"/minimap2 --version)" "$CMD" \
    "$BAM;$PAF" "$(md5sum "$BAM" | cut -d' ' -f1);$(md5sum "$PAF" | cut -d' ' -f1)" \
    "A37 $ARM arm (min_len $MIN_LEN, observed min $OBS_MIN); ref md5 verified against b5_species_v1.tsv. For A. thaliana it differs from the section 1 TAIR10 object (513e5ef3 vs 987ca803); author states 2026-09-04 the two differ in headers/formatting only, same sequence - stated, not measured here" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock"

{
  printf 'est_md5=%s\n' "$EST_MD5"
  printf 'arm=%s\n'     "$ARM"
  printf 'min_len=%s\n' "$MIN_LEN"
  printf 'bam_md5=%s\n' "$(md5sum "$BAM" | cut -d' ' -f1)"
  printf 'paf_md5=%s\n' "$(md5sum "$PAF" | cut -d' ' -f1)"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp"
mv "$DONE.tmp" "$DONE"
echo "$SP $ARM finished in ${SECONDS}s"
