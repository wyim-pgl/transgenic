#!/bin/bash
# Full analysis pipeline: Compare TransGenic against TAIR10 and AtRTD3
# Usage: bash scripts/05_run_full_analysis.sh <transgenic_predictions.gtf>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <transgenic_predictions.gtf>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/arabidopsis_transgenic.gtf"
    exit 1
fi

PREDICTED_GTF="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${REVISION_DIR}/data"
RESULTS_DIR="${REVISION_DIR}/results"
FIGURES_DIR="${REVISION_DIR}/figures"

# Check if predicted GTF exists
if [[ ! -f "${PREDICTED_GTF}" ]]; then
    echo "ERROR: Predicted GTF not found: ${PREDICTED_GTF}"
    exit 1
fi

# Check reference files
TAIR10_GTF="${DATA_DIR}/TAIR10/TAIR10.gtf"
ATRTD3_GTF="${DATA_DIR}/AtRTD3/AtRTD3.gtf"

for ref in "${TAIR10_GTF}" "${ATRTD3_GTF}"; do
    if [[ ! -f "${ref}" ]]; then
        echo "ERROR: Reference GTF not found: ${ref}"
        echo "Run 'bash scripts/01_download_references.sh' first"
        exit 1
    fi
done

echo "=============================================="
echo "TransGenic AS Evaluation - Full Analysis"
echo "=============================================="
echo "Predicted: ${PREDICTED_GTF}"
echo "TAIR10:    ${TAIR10_GTF}"
echo "AtRTD3:    ${ATRTD3_GTF}"
echo ""

mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

#######################################
# 1. GFFCompare Analysis
#######################################
echo "=== [1/4] GFFCompare Analysis ==="

echo ""
echo "--- vs TAIR10 ---"
python "${SCRIPT_DIR}/02_gffcompare_analysis.py" \
    -r "${TAIR10_GTF}" \
    -p "${PREDICTED_GTF}" \
    -o "${RESULTS_DIR}/vs_TAIR10" \
    --prefix "transgenic_vs_tair10"

echo ""
echo "--- vs AtRTD3 ---"
python "${SCRIPT_DIR}/02_gffcompare_analysis.py" \
    -r "${ATRTD3_GTF}" \
    -p "${PREDICTED_GTF}" \
    -o "${RESULTS_DIR}/vs_AtRTD3" \
    --prefix "transgenic_vs_atrtd3"

#######################################
# 2. Splice Event Detection
#######################################
echo ""
echo "=== [2/4] Splice Event Detection ==="

echo ""
echo "--- vs TAIR10 ---"
python "${SCRIPT_DIR}/03_splice_event_detection.py" \
    -r "${TAIR10_GTF}" \
    -p "${PREDICTED_GTF}" \
    -o "${RESULTS_DIR}/vs_TAIR10" \
    --prefix "splice_events_tair10"

echo ""
echo "--- vs AtRTD3 ---"
python "${SCRIPT_DIR}/03_splice_event_detection.py" \
    -r "${ATRTD3_GTF}" \
    -p "${PREDICTED_GTF}" \
    -o "${RESULTS_DIR}/vs_AtRTD3" \
    --prefix "splice_events_atrtd3"

#######################################
# 3. Generate Figures
#######################################
echo ""
echo "=== [3/4] Generating Figures ==="

echo ""
echo "--- Figure 6A: vs TAIR10 ---"
python "${SCRIPT_DIR}/04_generate_figure6.py" \
    -g "${RESULTS_DIR}/vs_TAIR10" \
    -s "${RESULTS_DIR}/vs_TAIR10" \
    -o "${FIGURES_DIR}/Figure6_vs_TAIR10"

echo ""
echo "--- Figure 6B: vs AtRTD3 ---"
python "${SCRIPT_DIR}/04_generate_figure6.py" \
    -g "${RESULTS_DIR}/vs_AtRTD3" \
    -s "${RESULTS_DIR}/vs_AtRTD3" \
    -o "${FIGURES_DIR}/Figure6_vs_AtRTD3"

#######################################
# 4. Generate Combined Summary
#######################################
echo ""
echo "=== [4/4] Combined Summary ==="

python - << 'PYTHON_SCRIPT'
import json
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")

# Load results
tair10_gff = json.load(open(results_dir / "vs_TAIR10/summary_report.json"))
atrtd3_gff = json.load(open(results_dir / "vs_AtRTD3/summary_report.json"))
tair10_splice = json.load(open(results_dir / "vs_TAIR10/splice_events_tair10_report.json"))
atrtd3_splice = json.load(open(results_dir / "vs_AtRTD3/splice_events_atrtd3_report.json"))

print("\n" + "="*60)
print("COMBINED SUMMARY: TransGenic AS Evaluation")
print("="*60)

print("\n[Transcript-level Metrics]")
print(f"{'Metric':<25} {'vs TAIR10':>15} {'vs AtRTD3':>15}")
print("-"*55)

t_metrics = tair10_gff['transcript_level_metrics']
a_metrics = atrtd3_gff['transcript_level_metrics']

print(f"{'Isoform Recall':<25} {t_metrics['isoform_recall']:>14.1%} {a_metrics['isoform_recall']:>14.1%}")
print(f"{'Isoform Precision':<25} {t_metrics['isoform_precision']:>14.1%} {a_metrics['isoform_precision']:>14.1%}")
print(f"{'Isoform F1':<25} {t_metrics['isoform_f1']:>14.1%} {a_metrics['isoform_f1']:>14.1%}")
print(f"{'Exact Matches':<25} {t_metrics['exact_matches']:>15,} {a_metrics['exact_matches']:>15,}")

print("\n[Splice Event Recovery (Recall)]")
print(f"{'Event Type':<25} {'vs TAIR10':>15} {'vs AtRTD3':>15}")
print("-"*55)

for event_type in ['SE', 'A5SS', 'A3SS', 'IR']:
    t_recall = tair10_splice['per_event_type'][event_type]['recall']
    a_recall = atrtd3_splice['per_event_type'][event_type]['recall']
    print(f"{event_type:<25} {t_recall:>14.1%} {a_recall:>14.1%}")

print("\n" + "="*60)
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  ${RESULTS_DIR}/vs_TAIR10/"
echo "  ${RESULTS_DIR}/vs_AtRTD3/"
echo ""
echo "Figures saved to:"
echo "  ${FIGURES_DIR}/Figure6_vs_TAIR10.{png,pdf,svg}"
echo "  ${FIGURES_DIR}/Figure6_vs_AtRTD3.{png,pdf,svg}"
echo ""
