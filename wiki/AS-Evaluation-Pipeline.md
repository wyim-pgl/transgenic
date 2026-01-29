# AS Evaluation Pipeline

Step-by-step workflow for Alternative Splicing evaluation of TransGenic predictions.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT FILES                                   │
├─────────────────────────────────────────────────────────────────────┤
│  TransGenic Predictions (GTF/GFF3)                                   │
│  Reference Annotations (TAIR10, AtRTD3, Rice, Maize)                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: GFFCompare Analysis                                        │
│  Tool: gffcompare                                                   │
│  Output: Transcript-level metrics (recall/precision/F1)             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Splice Event Detection                                     │
│  Events: SE, A5SS, A3SS, IR                                         │
│  Output: Event-level recall/precision per type                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Figure Generation                                          │
│  Output: Figure 6 (multi-panel comparison)                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Activate environment
micromamba activate transgenic-revision

# Run full analysis against both TAIR10 and AtRTD3
bash revision/scripts/05_run_full_analysis.sh <transgenic_predictions.gtf>
```

## Step-by-Step Guide

### Step 1: GFFCompare Analysis

**Script:** `revision/scripts/02_gffcompare_analysis.py`

```bash
python revision/scripts/02_gffcompare_analysis.py \
    -r revision/data/AtRTD3/AtRTD3.gtf \
    -p <transgenic_predictions.gtf> \
    -o revision/results/vs_AtRTD3 \
    --prefix transgenic_vs_atrtd3
```

**Output Files:**
| File | Description |
|------|-------------|
| `*.tmap` | Per-transcript matching results |
| `summary_report.json` | Key metrics (recall, precision, F1) |
| `isoform_distribution.csv` | Isoforms per gene comparison |
| `parsed_tmap.csv` | Parsed matching data |

**Class Codes:**
| Code | Meaning |
|------|---------|
| `=` | Exact intron chain match |
| `c` | Contained in reference |
| `k` | Contains reference |
| `j` | Multi-exon, partial junction match |
| `e` | Single exon overlap |
| `o` | Generic overlap |
| `i` | Intronic |
| `u` | Intergenic/unknown |

### Step 2: Splice Event Detection

**Script:** `revision/scripts/03_splice_event_detection.py`

```bash
python revision/scripts/03_splice_event_detection.py \
    -r revision/data/AtRTD3/AtRTD3.gtf \
    -p <transgenic_predictions.gtf> \
    -o revision/results/vs_AtRTD3 \
    --prefix splice_events
```

**Event Types Detected:**

| Event | Description | Detection Method |
|-------|-------------|------------------|
| **SE** | Exon Skipping | Exon present in one isoform, skipped in another |
| **A5SS** | Alt 5' Splice Site | Same acceptor, different donors |
| **A3SS** | Alt 3' Splice Site | Same donor, different acceptors |
| **IR** | Intron Retention | Intron retained as exonic region |

**Output Files:**
| File | Description |
|------|-------------|
| `*_report.json` | Event counts and metrics |
| `*_reference_*.bed` | Reference events (BED format) |
| `*_predicted_*.bed` | Predicted events (BED format) |

### Step 3: Figure Generation

**Single Reference Figure:**
```bash
python revision/scripts/04_generate_figure6.py \
    -g revision/results/vs_AtRTD3 \
    -s revision/results/vs_AtRTD3 \
    -o revision/figures/Figure6_vs_AtRTD3
```

**Comparison Figure (TAIR10 vs AtRTD3):**
```bash
python revision/scripts/06_generate_comparison_figure.py \
    -t revision/results/vs_TAIR10 \
    -a revision/results/vs_AtRTD3 \
    -o revision/figures/Figure6_comparison
```

**Figure 6 Panels:**
| Panel | Content |
|-------|---------|
| A | Transcript-level precision/recall bar plot |
| B | Isoform count distribution (violin plot) |
| C | Splice event recovery by type |
| D | Class code distribution (pie chart) |

## Output Directory Structure

```
revision/results/
├── vs_TAIR10/
│   ├── transgenic_vs_tair10.tmap
│   ├── summary_report.json
│   ├── isoform_distribution.csv
│   ├── splice_events_tair10_report.json
│   └── splice_events_*.bed
│
└── vs_AtRTD3/
    ├── transgenic_vs_atrtd3.tmap
    ├── summary_report.json
    ├── isoform_distribution.csv
    ├── splice_events_atrtd3_report.json
    └── splice_events_*.bed

revision/figures/
├── Figure6_vs_TAIR10.{png,pdf,svg}
├── Figure6_vs_AtRTD3.{png,pdf,svg}
└── Figure6_comparison.{png,pdf,svg}
```

## Interpreting Results

### Expected Patterns

| Metric | vs TAIR10 | vs AtRTD3 |
|--------|-----------|-----------|
| Isoform Recall | Higher | Lower |
| Isoform Precision | Similar | Similar |
| SE Recovery | Higher | Lower |

### Key Message for Reviewers

> TransGenic achieves high recall against standard references (TAIR10). Lower recall against AtRTD3 reflects missing low-abundance isoforms that require deep sequencing to detect, not incorrect splice boundary predictions.
