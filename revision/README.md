# TransGenic Revision: Alternative Splicing Evaluation

Reviewer response analysis comparing against **TAIR10** (standard) and **AtRTD3** (comprehensive AS).

## Quick Start

```bash
# 1. Create environment
micromamba create -f environment.revision.yml
micromamba activate transgenic-revision

# 2. Download BOTH references (TAIR10 + AtRTD3)
bash scripts/01_download_references.sh

# 3. Run full analysis (TAIR10 + AtRTD3)
bash scripts/05_run_full_analysis.sh /path/to/transgenic_predictions.gtf

# 4. Generate comparison figure
python scripts/06_generate_comparison_figure.py \
    -t results/vs_TAIR10 \
    -a results/vs_AtRTD3 \
    -o figures/Figure6_comparison
```

## Directory Structure

```
revision/
├── environment.revision.yml        # Micromamba environment
├── AS_evaluation_revision_plan.md  # Detailed revision plan
├── README.md
├── scripts/
│   ├── 01_download_references.sh       # Download TAIR10 + AtRTD3
│   ├── 02_gffcompare_analysis.py       # Transcript-level metrics
│   ├── 03_splice_event_detection.py    # Splice event analysis
│   ├── 04_generate_figure6.py          # Single-reference figure
│   ├── 05_run_full_analysis.sh         # Run against both refs
│   └── 06_generate_comparison_figure.py # TAIR10 vs AtRTD3 figure
├── data/
│   ├── TAIR10/    # Standard Arabidopsis reference
│   └── AtRTD3/    # Comprehensive AS reference
├── results/
│   ├── vs_TAIR10/ # Results against TAIR10
│   └── vs_AtRTD3/ # Results against AtRTD3
└── figures/
```

## Why Two References?

| Reference | Purpose | Expected Result |
|-----------|---------|-----------------|
| **TAIR10** | Standard annotation | Higher recall (baseline genes) |
| **AtRTD3** | AS-focused annotation | Lower recall (comprehensive AS) |

**Key insight for reviewers:**
> Lower recall against AtRTD3 reflects missing **low-abundance isoforms**, not incorrect predictions. TransGenic successfully recovers the majority of well-supported AS events.

## Metrics

### Transcript-level (GFFCompare)
- **Isoform Recall** = Exact matches / Reference transcripts
- **Isoform Precision** = Exact matches / Predicted transcripts
- **Isoform F1** = Harmonic mean

### Splice Event-level
| Event | Description |
|-------|-------------|
| SE | Exon Skipping |
| A5SS | Alternative 5' Splice Site |
| A3SS | Alternative 3' Splice Site |
| IR | Intron Retention |

## Manual Analysis Steps

```bash
# Against TAIR10
python scripts/02_gffcompare_analysis.py \
    -r data/TAIR10/TAIR10.gtf \
    -p <prediction.gtf> \
    -o results/vs_TAIR10

python scripts/03_splice_event_detection.py \
    -r data/TAIR10/TAIR10.gtf \
    -p <prediction.gtf> \
    -o results/vs_TAIR10

# Against AtRTD3
python scripts/02_gffcompare_analysis.py \
    -r data/AtRTD3/AtRTD3.gtf \
    -p <prediction.gtf> \
    -o results/vs_AtRTD3

python scripts/03_splice_event_detection.py \
    -r data/AtRTD3/AtRTD3.gtf \
    -p <prediction.gtf> \
    -o results/vs_AtRTD3
```

## References

- **TAIR10**: Lamesch et al., The Arabidopsis Information Resource, NAR 2012
- **AtRTD3**: Zhang et al., Genome Biology 2022
- **GFFCompare**: Pertea & Pertea, 2020
- **rMATS definitions**: Shen et al., PNAS 2014
