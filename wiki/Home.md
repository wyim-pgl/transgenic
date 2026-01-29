# TransGenic Wiki

Welcome to the TransGenic wiki! This documentation covers the genome annotation tool and Alternative Splicing (AS) evaluation pipeline.

## Contents

### Usage
- [[Genome Annotation Script]] - Run inference on genomes (options, examples, systemd usage)

### AS Evaluation (Revision)
- [[Reference Datasets]] - Download and setup reference genomes/annotations
- [[AS Evaluation Pipeline]] - Step-by-step analysis workflow
- [[Metrics Definitions]] - Detailed explanation of evaluation metrics
- [[Reviewer Response]] - Addressing reviewer concerns

## Quick Links

| Resource | Description |
|----------|-------------|
| [Main Repository](https://github.com/wyim-pgl/transgenic) | TransGenic source code |
| `src/run_genome_annotation.py` | Genome annotation script |
| `revision/` | Revision analysis scripts and data |
| `examples/` | Example prediction files |

## Quick Start

### Genome Annotation

```bash
# Basic usage
python src/run_genome_annotation.py genome.fa genes.gff3 -o output.gff3 --device cuda

# Production (large genome with systemd)
systemd-run --user --scope -p MemorySwapMax=0 \
    python3 src/run_genome_annotation.py \
    genome.fa genes.gff3 \
    -o output.gff3 \
    --batch_size 1 --num_workers 1 --device cuda --compile
```

### AS Evaluation

```bash
# Setup environment
micromamba create -f revision/environment.revision.yml
micromamba activate transgenic-revision

# Download references
bash revision/scripts/01_download_references.sh

# Run analysis
bash revision/scripts/05_run_full_analysis.sh <prediction.gtf>
```

## Revision Overview

### Reviewer Core Criticism
> Base-level F1 is inappropriate for evaluating alternative splicing (AS)

### Our Response
We provide comprehensive transcript-level and splice event-level metrics:

1. **Transcript-level accuracy** (GFFCompare)
   - Isoform Recall / Precision / F1

2. **Splice event recovery** (rMATS-style)
   - Exon Skipping (SE)
   - Alternative 5'/3' Splice Sites (A5SS/A3SS)
   - Intron Retention (IR)

3. **Isoform count distribution**
   - Per-gene isoform count comparison

## Reference Data

| Species | Reference | Transcripts | Genes |
|---------|-----------|-------------|-------|
| Arabidopsis | TAIR10 | 54,013 | 32,833 |
| Arabidopsis | AtRTD3 | 169,499 | 40,929 |
| Rice | IRGSP-1.0 | 45,973 | 38,993 |
| Maize | B73 NAM-5.0 | 77,341 | 44,303 |
