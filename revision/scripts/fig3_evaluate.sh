#!/bin/bash
# fig3_evaluate.sh — per-species GFFCompare of the regenerated Figure 3 test-set
# predictions (de novo 400M) against the training reference annotations.
#
# Anchor validation: A. thaliana base F1 should be ~92.2%, Z. mays ~71.1%
# (manuscript Fig. 3A values from the original test evaluation). The split here is
# reconstructed from manual_seed(123) rather than restored, and generation uses
# do_sample=True, so exact agreement is not expected — but a large divergence means
# the abstract and the first Results paragraph need updating to the new values.
#
# Predictions are produced by fig3_infer.py on the GPU host. This script copies them
# over (stage 1) and scores them (stage 2).
#
# Stage 1 is skipped automatically when all ten prediction files are already present
# locally, so on a machine that has them you need nothing but gffcompare.
#
# Credentials are read from the environment; nothing is hardcoded, because this
# script ships in the public repository:
#
#   export PGLGPU_HOST=<host-or-ip>
#   export PGLGPU_USER=<user>
#   export PGLGPU_PASS=<password>        # omit to use an SSH key / agent instead
#   export SSH_PY=/path/to/python        # a python with paramiko installed
#
# Usage: bash fig3_evaluate.sh [--force-pull]
set -uo pipefail

GPFS=/data/gpfs/assoc/pgl/data/Transgenic
GENOMES=$GPFS/genomes
OUT=$GPFS/transgenic/revision/results/fig3_regen
GFFCOMPARE=${GFFCOMPARE:-/data/gpfs/assoc/pgl/bin/conda/conda_envs/AnnotationSplitter/bin/gffcompare}   # v0.12.10

PGLGPU_HOST=${PGLGPU_HOST:-}
PGLGPU_USER=${PGLGPU_USER:-}
PGLGPU_PASS=${PGLGPU_PASS:-}
PRED_REMOTE=${PRED_REMOTE:-/home/framazan/fig3_test_preds}
SSH_PY=${SSH_PY:-$GPFS/.venv-ssh/bin/python}

declare -A REFS=(
  [A_thaliana]=Athaliana_167_TAIR10.gene.clean.gff3
  [G_max]=Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3
  [P_patens]=Ppatens_318_v3.3.gene_exons.clean.gff3
  [P_trichocarpa]=Ptrichocarpa_533_v4.1.gene_exons.clean.gff3
  [S_bicolor]=Sbicolor_730_v5.1.gene_exons.clean.gff3
  [B_distachyon]=Bdistachyon_314_v3.1.gene_exons.clean.gff3
  [S_italica]=Sitalica_312_v2.2.gene_exons.clean.gff3
  [V_vinifera]=Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3
  [O_sativa]=Osativa_323_v7.0.gene_exons.exon.gff3
  [Z_mays]=Zmays_493_RefGen_V4.gene_exons.exon.gff3
)

mkdir -p "$OUT/preds"

# ---- stage 1: fetch predictions from the GPU host ---------------------------
have=0
for sp in "${!REFS[@]}"; do
  [ -s "$OUT/preds/${sp}_test400M.gff3" ] && have=$((have + 1))
done

if [ "${1:-}" != "--force-pull" ] && [ "$have" -eq "${#REFS[@]}" ]; then
  echo "all ${#REFS[@]} prediction files already present locally; skipping the copy"
elif [ -z "$PGLGPU_HOST" ] || [ -z "$PGLGPU_USER" ]; then
  echo "ERROR: only $have/${#REFS[@]} prediction files are present locally and" >&2
  echo "       PGLGPU_HOST / PGLGPU_USER are not set, so they cannot be fetched." >&2
  echo "       Set them (see the header of this script) or copy the files into" >&2
  echo "       $OUT/preds/ by hand." >&2
  exit 1
elif [ ! -x "$SSH_PY" ]; then
  echo "ERROR: no paramiko-capable python at $SSH_PY." >&2
  echo "       Create one with:  python3 -m venv $GPFS/.venv-ssh &&" >&2
  echo "                         $GPFS/.venv-ssh/bin/pip install paramiko" >&2
  echo "       or point SSH_PY at an interpreter that already has paramiko." >&2
  exit 1
else
  echo "fetching predictions from $PGLGPU_USER@$PGLGPU_HOST:$PRED_REMOTE"
  PRED_REMOTE="$PRED_REMOTE" OUT_PREDS="$OUT/preds" \
  PGLGPU_HOST="$PGLGPU_HOST" PGLGPU_USER="$PGLGPU_USER" PGLGPU_PASS="$PGLGPU_PASS" \
  "$SSH_PY" - <<'PYEOF'
import os, paramiko

host = os.environ["PGLGPU_HOST"]
user = os.environ["PGLGPU_USER"]
pw = os.environ.get("PGLGPU_PASS") or None
remote = os.environ["PRED_REMOTE"]
out = os.environ["OUT_PREDS"]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# With no password set, fall back to the agent and the usual key locations.
c.connect(host, username=user, password=pw, timeout=30)
sftp = c.open_sftp()
for attr in sftp.listdir_attr(remote):
    if not attr.filename.endswith("_test400M.gff3"):
        continue
    dst = os.path.join(out, attr.filename)
    # Re-pull whenever the sizes differ: a file left over from an interrupted run
    # is partial, and skipping it would silently score an incomplete prediction set.
    if os.path.exists(dst) and os.path.getsize(dst) == attr.st_size:
        continue
    sftp.get(os.path.join(remote, attr.filename), dst)
    print(f"pulled {attr.filename} ({attr.st_size:,} bytes)")
sftp.close()
c.close()
PYEOF
  [ $? -ne 0 ] && { echo "ERROR: fetch failed" >&2; exit 1; }
fi

# ---- stage 2: per-species gffcompare ----------------------------------------
echo "species,base_sn,base_pr,exon_sn,exon_pr,intron_sn,intron_pr,ichain_sn,ichain_pr,tx_sn,tx_pr,locus_sn,locus_pr" > "$OUT/fig3A_metrics.csv"
for sp in "${!REFS[@]}"; do
  pred="$OUT/preds/${sp}_test400M.gff3"
  [ -s "$pred" ] || { echo "SKIP $sp (no predictions)"; continue; }
  # Score against the evaluated genes only. The full genome annotation counts the
  # 85% of genes that were never in the test split as missed, which drives
  # sensitivity to nonsense (A. thaliana: 14.0% Sn against 92.5% Pr). Build the
  # restricted references with 24_restrict_reference_to_test_genes.py.
  ref="$OUT/refs/${sp}.testset.gff3"
  if [ ! -s "$ref" ]; then
    echo "WARNING: $sp has no restricted reference; falling back to the whole genome," >&2
    echo "         which makes sensitivity uninterpretable. Run" >&2
    echo "         24_restrict_reference_to_test_genes.py first." >&2
    ref="$GENOMES/${REFS[$sp]}"
  fi
  [ -s "$ref" ] || { echo "SKIP $sp (reference $ref missing)"; continue; }
  o="$OUT/gffcompare_${sp}"
  $GFFCOMPARE -r "$ref" -o "$o" "$pred" 2>/dev/null
  python3 - "$sp" "$o.stats" >> "$OUT/fig3A_metrics.csv" <<'PYEOF'
import sys, re
sp, path = sys.argv[1], sys.argv[2]
vals = []
txt = open(path).read()
for level in ["Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus"]:
    m = re.search(rf"{level} level:\s+([\d.]+)\s+\|\s+([\d.]+)", txt)
    vals += [m.group(1), m.group(2)] if m else ["NA", "NA"]
print(sp + "," + ",".join(vals))
PYEOF
done

echo "=== fig3A_metrics.csv ==="
cat "$OUT/fig3A_metrics.csv"

# ---- anchor check -----------------------------------------------------------
echo
echo "=== anchor check against the published Fig. 3A values ==="
python3 - "$OUT/fig3A_metrics.csv" <<'PYEOF'
import csv, sys

ANCHORS = {"A_thaliana": 92.2, "Z_mays": 71.1}
rows = {r["species"]: r for r in csv.DictReader(open(sys.argv[1]))}
for sp, published in ANCHORS.items():
    r = rows.get(sp)
    if not r or r["base_sn"] == "NA":
        print(f"  {sp:12s} MISSING — cannot check the anchor")
        continue
    sn, pr = float(r["base_sn"]), float(r["base_pr"])
    f1 = 0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr)
    delta = f1 - published
    verdict = "matches" if abs(delta) <= 1.0 else "DIVERGES — update the manuscript"
    print(f"  {sp:12s} regenerated base F1 {f1:5.1f}%  published {published:5.1f}%  "
          f"delta {delta:+5.1f}  {verdict}")
print()
print("  A divergence above ~1 point means the abstract and the first Results")
print("  paragraph must be restated with the regenerated values, and Methods must")
print("  declare the reconstructed split seed (manual_seed(123)).")
PYEOF
