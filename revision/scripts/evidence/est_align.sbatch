#!/usr/bin/env bash
#SBATCH --job-name=est_align
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_align/logs/%x_%A_%a.out
#SBATCH --array=0-8
#
# EST -> genome spliced alignment, protocol B1 §3 (frozen command) + A8/A11/A21/A36.
#
# Two gates run before the aligner, and both fail closed:
#   1. A18.3 dataset-role manifest: refuse to align evidence that is not declared training-eligible.
#   2. A21 UniVec screening + A36 length floor: refuse to align anything but the SCREENED FASTA.
#      A21 says "only the screened FASTA is aligned"; A36 (v1.26, author decision 2026-09-03) drops
#      records <= 120 nt at that same ingestion step. The 2026-09-03 run read est.fa.gz directly,
#      skipped both, and nothing stopped it — that BAM set is retired in
#      evidence/RETIRED_DO_NOT_USE/est_align_prescreen_20260903/.
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
OUT=$ROOT/evidence/est_align
BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
ROLES=$ROOT/evidence/DATASET_ROLES.est_v1.tsv

SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$OUT/logs" "$OUT/$SP"

# --- Gate 1: role manifest (A18.3) ---
ROLE=$(awk -F'\t' -v d="est/$SP" '$1==d{print $12}' "$ROLES")
if [ "$ROLE" != "c2_training_eligible" ]; then
  echo "REFUSED: est/$SP has role '${ROLE:-<none>}' in $(basename "$ROLES"), not c2_training_eligible" >&2
  exit 2
fi

REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
RAW=$ROOT/evidence/est/$SP/est.fa.gz
UVD=$ROOT/evidence/est/$SP/univec
EST=$UVD/est.univec.fa.gz
[ -f "$REF" ] || { echo "missing genome: $REF" >&2; exit 3; }
[ -f "$RAW" ] || { echo "missing EST: $RAW" >&2; exit 3; }

BAM=$OUT/$SP/$SP.est.bam
PROV=$OUT/$SP/PROVENANCE.txt
[ -f "$OUT/$SP/DONE" ] && { echo "$SP already done"; exit 0; }

# --- Gate 2: A21 screening + A36 floor. Produce it if absent, then refuse to continue without it. ---
if [ ! -f "$UVD/DONE" ]; then
  echo "$SP: A21 screening not present, running univec_screen.sh"
  PATH="$BIN:$PATH" PY=$(command -v python3) \
    bash "$ROOT/transgenic/revision/scripts/evidence/univec_screen.sh" "$SP" 32 \
    > "$OUT/$SP/univec_screen.log" 2>&1
fi
if [ ! -f "$UVD/DONE" ] || [ ! -f "$EST" ]; then
  echo "REFUSED: $SP has no A21-screened FASTA ($EST). A21: only the screened FASTA is aligned." >&2
  exit 5
fi

CMD="minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long $REF $EST"
START=$(date -Is)
SECONDS=0

"$BIN"/minimap2 -t 32 -ax splice:hq -G 200000 --secondary=no --cs=long "$REF" "$EST" 2> "$OUT/$SP/minimap2.err" \
  | "$BIN"/samtools sort -@ 4 -m 2G -o "$BAM" - 2> "$OUT/$SP/samtools.err"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "minimap2 failed rc=$rc for $SP" >&2; exit 4; }
"$BIN"/samtools index "$BAM"

{
  echo "species        $SP"
  echo "role           $ROLE   (from $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
  echo "reference      $REF"
  echo "reference_md5  $(md5sum "$REF" | cut -d' ' -f1)"
  echo "est_raw        $RAW"
  echo "est_raw_md5    $(md5sum "$RAW" | cut -d' ' -f1)"
  echo "est_raw_count  $(awk -F'\t' -v d="est/$SP" '$1==d{print $8}' "$ROLES")"
  echo "est            $EST   (A21-screened, A36 floor)"
  echo "est_md5        $(cut -d' ' -f1 "$UVD/est.univec.fa.gz.md5" 2>/dev/null)"
  echo "univec_db      UniVec_Core md5 $(cut -d' ' -f1 "$ROOT/evidence/univec/UniVec_Core.md5" 2>/dev/null)"
  echo "univec_version $(cat "$ROOT/evidence/univec/UniVec_Core.version" 2>/dev/null)"
  echo "univec_summary $(tr -d '\n ' < "$UVD/summary.json" 2>/dev/null)"
  echo "min_len        121   (protocol A36, v1.26)"
  echo "minimap2       $("$BIN"/minimap2 --version)"
  echo "samtools       $("$BIN"/samtools --version | head -1)"
  echo "command        $CMD"
  echo "started        $START"
  echo "finished       $(date -Is)"
  echo "wall_seconds   $SECONDS"
  echo "bam            $BAM"
  echo "bam_md5        $(md5sum "$BAM" | cut -d' ' -f1)"
  echo "slurm_job      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
} > "$PROV"

"$BIN"/samtools flagstat "$BAM" > "$OUT/$SP/flagstat.txt"
touch "$OUT/$SP/DONE"
echo "$SP finished in ${SECONDS}s"
