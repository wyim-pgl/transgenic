#!/bin/bash
#SBATCH --job-name=b6_blastx_audit
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#
# Stage C1 blastx artifact audit -- TEMPLATE ONLY.  NOT submitted by the
# implementing agent; submission requires explicit author authorization
# (exits the read-only discovery constraint).  12 h wall per reviewer note
# (~150 GB of GPFS reads at 8 CPUs leaves slack for slow GPFS).
#
# Required environment (set by the submitting controller, all absolute):
#   AUDIT_SCRIPT     path to est_blastx_audit.py in the reviewed snapshot
#   AUDIT_SCRIPT_SHA reviewed sha256 of that script (self-verified below)
#   BLASTX_ROOT      e.g. $R/evidence/est_blastx
#   OUT_DIR          e.g. $R/evidence/est_phase_routeB_20260916/audit/c1_<date>
#   SACCT_FILE       sacct -P --format=JobID,State,ExitCode capture WITH header
#                    and allocation/.batch/.extern for 6163851 + 6164049_0-7
# Audit requires original absolute input/engine paths to remain resolvable.
# Source-tree tmp/preflight failures are inventoried, not automatically waived.
# TMPDIR must be outside BLASTX_ROOT (filter reconstruction uses local scratch).
set -euo pipefail
for v in AUDIT_SCRIPT AUDIT_SCRIPT_SHA BLASTX_ROOT OUT_DIR SACCT_FILE; do
  [[ -n "${!v:-}" ]] || { echo "REFUSED: $v is unset" >&2; exit 2; }
done
actual=$(sha256sum "$AUDIT_SCRIPT" | cut -d' ' -f1)
[[ "$actual" == "$AUDIT_SCRIPT_SHA" ]] || {
  echo "REFUSED: audit script sha256 $actual != reviewed $AUDIT_SCRIPT_SHA" >&2; exit 2; }
[[ -d "$BLASTX_ROOT" && -f "$SACCT_FILE" ]] || {
  echo "REFUSED: BLASTX_ROOT/SACCT_FILE missing" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "REFUSED: OUT_DIR exists (no overwrite)" >&2; exit 2; }
exec python3 "$AUDIT_SCRIPT" --blastx-root "$BLASTX_ROOT" \
  --out-dir "$OUT_DIR" --sacct-file "$SACCT_FILE"
