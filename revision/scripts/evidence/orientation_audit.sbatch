#!/usr/bin/env bash
#SBATCH --job-name=orient_audit
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/orientation_audit_v1/logs/%x_%A_%a.out
#SBATCH --array=0-128%12
#
# §3.3 item 2 — library orientation audit, ONE TASK PER RUN. This decides `-uf` for that run and
# nothing else; §3.3 item 4 requires every run to be scored separately, so a dataset-level verdict
# would not satisfy the protocol even though runs of one dataset share a library prep.
#
# Per run:
#   1. resolve the A18.3 role BY RUN IDENTITY, not by tree path (see the gate below for why)
#   2. draw a deterministic uniform 10,000-record sample (68_sample_fasta.py)
#   3. align it with the §4 ONT command MINUS -uf, which is what item 2 asks for
#   4. score with 67_orientation_audit.py -> PASS | FAIL | UNRESOLVED
#
# The verdict is written to evidence/orientation_audit_v1/<species>/<dataset>/<run>/ and the §4
# long-read driver refuses to align a run that has no matching audit, so `-uf` can never be set
# by anything except this measurement.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
SCRIPTS=$ROOT/transgenic/revision/scripts
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
RUNS=$ROOT/evidence/longread_runs_v1.tsv
OUTROOT=$ROOT/evidence/orientation_audit_v1
THREADS=${SLURM_CPUS_PER_TASK:-8}

read -r SP DS RUN FA < <(awk -F'\t' -v i="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==i' "$RUNS")
[ -n "${RUN:-}" ] || { echo "REFUSED: no run at index $SLURM_ARRAY_TASK_ID in $RUNS" >&2; exit 2; }
FA=$ROOT/$FA
OUT=$OUTROOT/$SP/$(basename "$DS")/$RUN
mkdir -p "$OUT" "$OUTROOT/logs"
DONE=$OUT/DONE

# --- Gate 1: A18.3 role, resolved by RUN IDENTITY ---
# A path lookup is wrong here. Three Cui 2020 runs sit in the tree twice: the canonical copy is
# declared under ont/Athaliana_cui2020_PRJNA594286 as c2_training_eligible (A18.3 names it
# explicitly), while the copy under training/ont/... is the duplicate that demote_duplicates() put
# to `excluded`. Looking the role up by the path we are walking returns `excluded` and would drop
# exactly the runs the protocol calls training evidence. Identity is the key; the manifest
# guarantees one canonical eligible row per run.
n_eli=$(awk -F'\t' -v r="$RUN" '$2==r && $12=="c2_training_eligible"' "$ROLES" | wc -l)
if [ "$n_eli" -ne 1 ]; then
  echo "REFUSED: $RUN has $n_eli c2_training_eligible rows in $(basename "$ROLES"), expected exactly 1" >&2
  exit 2
fi

REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
GFF=$(awk -F'\t' -v s="$SP" '$1==s{print $6}' "$MANIFEST")
[ -f "$REF" ] && [ -f "$GFF" ] || { echo "REFUSED: missing reference or GFF for $SP" >&2; exit 3; }

# --- Gate 2: §3.3 item 1 — the reference and the annotation are the declared objects ---
EXP_REF=$(awk -F'\t' -v s="$SP" '$1==s{print $5}' "$MANIFEST")
EXP_GFF=$(awk -F'\t' -v s="$SP" '$1==s{print $7}' "$MANIFEST")
REF_MD5=$(md5sum "$REF" | cut -d' ' -f1)
GFF_MD5=$(md5sum "$GFF" | cut -d' ' -f1)
[ "$REF_MD5" = "$EXP_REF" ] || { echo "REFUSED: $SP reference md5 $REF_MD5 != manifest $EXP_REF" >&2; exit 3; }
[ "$GFF_MD5" = "$EXP_GFF" ] || { echo "REFUSED: $SP GFF md5 $GFF_MD5 != manifest $EXP_GFF" >&2; exit 3; }

[ -s "$FA" ] || { echo "REFUSED: missing or empty reads: $FA" >&2; exit 3; }
gzip -t "$FA" 2>/dev/null || { echo "REFUSED: corrupt reads: $FA" >&2; exit 3; }
FA_MD5=$(md5sum "$FA" | cut -d' ' -f1)

# --- Completion marker bound to every input that can change the verdict ---
KEY="$FA_MD5:$REF_MD5:$GFF_MD5"
if [ -e "$DONE" ]; then
  HAVE=$(awk -F= '$1=="key"{print $2}' "$DONE")
  if [ "$HAVE" = "$KEY" ] && [ -s "$OUT/audit.json" ]; then
    echo "$RUN already audited for key=$KEY"; exit 0
  fi
  echo "REFUSED: $RUN has a DONE marker for a different input (have=${HAVE:-<none>}, want=$KEY)" >&2
  exit 6
fi

START=$(date -Is); SECONDS=0
SAMPLE=$OUT/sample.fa
SAMPLE_JSON=$("$PY" "$SCRIPTS/68_sample_fasta.py" --fasta "$FA" --out "$SAMPLE" --n 10000 --md5 "$FA_MD5") \
  || { echo "REFUSED: sampling failed for $RUN" >&2; exit 4; }

# §4 ONT command, deliberately WITHOUT -uf: item 2 measures orientation, so it must not assume it.
CMD="minimap2 -t $THREADS -ax splice -k14 -G 200000 --secondary=no --cs=long $REF $SAMPLE"
"$BIN"/minimap2 -t "$THREADS" -ax splice -k14 -G 200000 --secondary=no --cs=long "$REF" "$SAMPLE" \
  > "$OUT/sample.sam" 2> "$OUT/minimap2.err"
rc=$?
[ "$rc" -eq 0 ] || { echo "REFUSED: minimap2 exited $rc for $RUN" >&2; rm -f "$OUT/sample.sam"; exit 4; }
[ -s "$OUT/sample.sam" ] || { echo "REFUSED: empty SAM for $RUN" >&2; exit 4; }

EXTRA=$(printf '{"reads_fa":"%s","reads_md5":"%s","reference":"%s","reference_md5":"%s","gff":"%s","gff_md5":"%s","sample":%s,"command":"%s","slurm_job":"%s_%s"}' \
  "$FA" "$FA_MD5" "$REF" "$REF_MD5" "$GFF" "$GFF_MD5" "$SAMPLE_JSON" "$CMD" \
  "${SLURM_ARRAY_JOB_ID:-NA}" "${SLURM_ARRAY_TASK_ID:-NA}")
"$PY" "$SCRIPTS/67_orientation_audit.py" --sam "$OUT/sample.sam" --gff "$GFF" \
  --out "$OUT/audit.json" --run "$RUN" --species "$SP" --dataset "$DS" --extra "$EXTRA" \
  > "$OUT/audit.stdout" 2> "$OUT/audit.err"
arc=$?
# 0 = PASS or FAIL (both are verdicts), 4 = UNRESOLVED (also a verdict, but one to look at).
[ "$arc" -eq 0 ] || [ "$arc" -eq 4 ] || { echo "REFUSED: scorer exited $arc for $RUN" >&2; exit 4; }
[ -s "$OUT/audit.json" ] || { echo "REFUSED: no verdict written for $RUN" >&2; exit 4; }

STATUS=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$OUT/audit.json")
USEUF=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['use_uf'])" "$OUT/audit.json")
FRAC=$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['fraction'])" "$OUT/audit.json")

# The 10,000-record sample and its SAM are intermediates; the verdict and its provenance are the
# artifact. Keeping 129 SAMs would cost far more than regenerating one from the recorded seed.
gzip -1 -f "$OUT/sample.sam" && rm -f "$SAMPLE"

{
  echo "run            $RUN"
  echo "species        $SP"
  echo "dataset        $DS"
  echo "role           c2_training_eligible   (resolved by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
  echo "reads          $FA"
  echo "reads_md5      $FA_MD5"
  echo "reference      $REF"
  echo "reference_md5  $REF_MD5"
  echo "gff            $GFF"
  echo "gff_md5        $GFF_MD5"
  echo "sample         $SAMPLE_JSON"
  echo "command        $CMD"
  echo "status         $STATUS"
  echo "use_uf         $USEUF"
  echo "fraction       $FRAC"
  echo "minimap2       $("$BIN"/minimap2 --version)"
  echo "started        $START"
  echo "finished       $(date -Is)"
  echo "wall_seconds   $SECONDS"
  echo "slurm_job      ${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
} > "$OUT/PROVENANCE.txt"

# §3.3 item 5: one central table. 12 tasks append at once, so take the lock.
PROVTSV=$ROOT/transgenic/revision/results/evidence/PROVENANCE.tsv
mkdir -p "$(dirname "$PROVTSV")"
{
  flock 9
  [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
  printf "%s\tlongread_orientation_audit\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$SP" "$FA" "$FA_MD5" "$REF" "$REF_MD5" "$("$BIN"/minimap2 --version)" "$CMD" \
    "$OUT/audit.json" "$(md5sum "$OUT/audit.json" | cut -d' ' -f1)" \
    "§3.3 item 2, run $RUN: $STATUS, agreement $FRAC, use_uf=$USEUF; GFF md5 $GFF_MD5 verified against b5_species_v1.tsv" \
    >> "$PROVTSV"
} 9>"$PROVTSV.lock"

{
  printf 'key=%s\n' "$KEY"
  printf 'run=%s\n' "$RUN"
  printf 'status=%s\n' "$STATUS"
  printf 'use_uf=%s\n' "$USEUF"
  printf 'audit_md5=%s\n' "$(md5sum "$OUT/audit.json" | cut -d' ' -f1)"
  printf 'completed=%s\n' "$(date -Is)"
} > "$DONE.tmp"
mv "$DONE.tmp" "$DONE"
echo "$RUN $SP $STATUS use_uf=$USEUF fraction=$FRAC in ${SECONDS}s"
