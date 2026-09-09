#!/usr/bin/env bash
#SBATCH --job-name=protein_accept
#SBATCH --account=cpu-s2-pgl-0
#SBATCH --partition=cpu-s2-core-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein_alignments/logs/%x_%A_%a.out
#SBATCH --array=0-8
#
# A19.2 acceptance pass over the merged per-species GFF (issue #45). Identity >= 30 %, coverage >= 70 %,
# no frameshift, canonical introns <= 200 kb, mapping_ambiguous flagging, per-organism support tables.
# A separate, re-runnable pass by design: the judgement can be revised without realigning (A39 precedent).
# protein_accept.py is Python stdlib + coreutils sort; it needs no genome and no container.
set -uo pipefail

ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin/python3
ACC=$ROOT/transgenic/revision/scripts/evidence/protein_accept.py
OUT=$ROOT/evidence/protein_alignments
SPECIES=(Athaliana Bdistachyon Gmax Osativa Ppatens Ptrichocarpa Sbicolor Sitalica Vvinifera)
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
GFF=$OUT/$SP/$SP.protein.gff.gz
DONE=$OUT/$SP/DONE
AD=$OUT/$SP/accept
export TMPDIR=$OUT/$SP/.accept_tmp; mkdir -p "$TMPDIR" "$AD"

# --- Gate: the GFF must be the one the merge declared complete. ---
[ -s "$DONE" ] && [ -s "$GFF" ] || { echo "REFUSED: $SP has no merged GFF/DONE" >&2; exit 3; }
D_MD5=$(sed -n 's/^gff_md5=//p' "$DONE"); A_MD5=$(md5sum "$GFF" | cut -d' ' -f1)
[ "$D_MD5" = "$A_MD5" ] || { echo "REFUSED: $SP GFF md5 $A_MD5 != DONE $D_MD5" >&2; exit 3; }

# --- Skip only a complete, identity-matching result. ---
if [ -s "$AD/$SP.summary.json" ] && [ -s "$AD/ACCEPT_DONE" ] && [ "$(sed -n 's/^gff_md5=//p' "$AD/ACCEPT_DONE")" = "$A_MD5" ]; then
  echo "$SP acceptance already complete for gff_md5=$A_MD5"; exit 0
fi
rm -f "$AD/ACCEPT_DONE"

START=$(date -Is); SECONDS=0
"$PY" "$ACC" --gff "$GFF" --species "$SP" --out-dir "$AD" || { echo "REFUSED: protein_accept.py exited $? for $SP" >&2; exit 4; }
for f in "$AD/$SP.tsv" "$AD/$SP.alignments.tsv" "$AD/$SP.introns.tsv" "$AD/$SP.summary.json"; do
  [ -s "$f" ] || { echo "REFUSED: $SP acceptance produced no $(basename "$f")" >&2; exit 4; }
done
{
  printf 'species=%s\ngff_md5=%s\nsummary_md5=%s\nalignments=%s\naccepted=%s\nstarted=%s\ncompleted=%s\nwall_seconds=%s\nslurm_job=%s\n' \
    "$SP" "$A_MD5" "$(md5sum "$AD/$SP.summary.json" | cut -d' ' -f1)" \
    "$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('alignments'))" "$AD/$SP.summary.json")" \
    "$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('accepted'))" "$AD/$SP.summary.json")" \
    "$START" "$(date -Is)" "$SECONDS" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
} > "$AD/ACCEPT_DONE.tmp" && mv "$AD/ACCEPT_DONE.tmp" "$AD/ACCEPT_DONE"
rm -rf "$TMPDIR"
echo "$SP acceptance done in ${SECONDS}s: $(cat "$AD/$SP.summary.json" | tr -d '\n' | cut -c1-200)"
