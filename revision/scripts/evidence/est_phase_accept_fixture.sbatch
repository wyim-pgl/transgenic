#!/bin/bash
#SBATCH --job-name=b6_phase_btop_pin
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#
# RICH17 real-DIAMOND BTOP-pinning fixture -- TEMPLATE ONLY, NOT submitted by
# the implementing agent.  Same partition/limits as est_blastx_fixture.sbatch.
# Required environment (absolute paths, set by the submitting controller):
#   FIXTURE_SCRIPT     est_phase_accept_fixture.py in the reviewed snapshot
#   FIXTURE_SCRIPT_SHA reviewed sha256 (self-verified below)
#   DIAMOND            real DIAMOND 2.1.24 executable
#   OUT_DIR            new dir under $R/evidence/est_phase_routeB_20260916/acceptance/
#   TMPDIR_ABS         existing absolute scratch dir
set -euo pipefail
for v in FIXTURE_SCRIPT FIXTURE_SCRIPT_SHA DIAMOND OUT_DIR TMPDIR_ABS; do
  [[ -n "${!v:-}" ]] || { echo "REFUSED: $v is unset" >&2; exit 2; }
done
actual=$(sha256sum "$FIXTURE_SCRIPT" | cut -d' ' -f1)
[[ "$actual" == "$FIXTURE_SCRIPT_SHA" ]] || {
  echo "REFUSED: fixture sha256 $actual != reviewed $FIXTURE_SCRIPT_SHA" >&2; exit 2; }
[[ -x "$DIAMOND" && -d "$TMPDIR_ABS" ]] || { echo "REFUSED: DIAMOND/TMPDIR_ABS invalid" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "REFUSED: OUT_DIR exists (no overwrite)" >&2; exit 2; }
exec python3 "$FIXTURE_SCRIPT" --diamond "$DIAMOND" --out-dir "$OUT_DIR" \
  --tmpdir "$TMPDIR_ABS" --threads "${SLURM_CPUS_PER_TASK:-4}"
