# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TransGenic is a DNA-to-annotation machine translation model using a HyenaDNA encoder + Longformer decoder. It generates gene structure annotations (CDS, UTR, introns) from raw DNA sequences in Gene Sentence Format (GSF). The project supports training on multiple plant species and inference on new genomes.

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Create a DuckDB training database (multi-species, with augmentation)
python scripts/create_database.py \
    --fasta At.fa Os.fa Zm.fa \
    --gff   At.gff3 Os.gff3 Zm.gff3 \
    --output training.db --mode train \
    --add-extra 200 --add-rc-iso-only --clean

# Create an inference database
python scripts/create_database.py \
    --fasta genome.fa --gff genes.gff3 \
    --output predict.db --mode predict --no-prefix

# Full inference pipeline (sort GFF3 → build DB → inference → GFF3 output)
python src/run_genome_annotation.py genome.fa genes.gff3 -o output.gff3 --device cuda

# Training (platform-specific)
python train/train_HyenaTransgenic.py --db training.db
python train/train_HyenaTransgenic_RTX4090.py --db training.db
python train/train_HyenaTransgenic_GB10.py --db training.db

# Resume training from latest checkpoint
python train/train_HyenaTransgenic_RTX4090.py --db training.db --resume auto

# Convert GFF3 to GSF format
python scripts/gff2gsf.py annotation.gff3 -o output.gsf

# Check system for environment selection
./scripts/check_system.sh
```

## Architecture

### Core Components

- **Encoder**: HyenaDNA (`LongSafari/hyenadna-large-1m-seqlen-hf`) — processes DNA sequences up to 1M nucleotides using sub-quadratic convolution
- **Decoder**: Longformer-based autoregressive decoder — generates GSF text annotations with sliding window attention
- **Downsampling**: 2-stage Conv1d with relative positional bias (6x compression from encoder to decoder)
- **Model class**: `transgenicForConditionalGeneration` in `src/transgenic/model/modeling_HyenaTransgenic.py`

### Model Variants

| Variant | Params | d_model | Layers | Heads | Notes |
|---------|--------|---------|--------|-------|-------|
| Base (400M) | ~400M | 768 | 12 | 6 | Published checkpoint, 92% F1 |
| Base (160M) | ~160M | 512 | 9 | 4 | Published checkpoint |
| Wide (1.17B) | ~1.17B | 1152 | 16 | 8 | GB10/RTX 4090 training target |

### Data Pipeline

1. **Preprocessing** (`src/transgenic/datasets/preprocess.py`):
   - `genome2GSFDataset()` — creates DuckDB database from FASTA + GFF3/BED files
   - Stores sequences padded to multiples of 6144nt, with GSF labels for training
   - `speciesPrefix` parameter prefixes chromosome names for multi-species DBs (e.g., `Zm_Chr01`)
   - Supports training augmentation: random flanking buffer (`addExtra`), reverse complement (`addRC`/`addRCIsoOnly`), CDS validation (`clean`)

2. **CLI Database Builder** (`scripts/create_database.py`):
   - Wraps `genome2GSFDataset()` with argparse CLI
   - Supports multiple `--fasta`/`--gff` pairs in one command
   - Auto-derives species prefix from FASTA filenames; `--no-prefix` to disable

3. **Dataset** (`src/transgenic/datasets/datasets.py`):
   - `isoformDataHyena` — PyTorch Dataset for loading from DuckDB
   - `hyena_collate_fn` — collate function for DataLoader
   - `exclude_prefix` parameter to exclude species for cross-species evaluation

4. **GSF Utilities** (`src/transgenic/utils/gsf.py`):
   - `gffString2GFF3()` — converts model output (GSF) back to GFF3 format
   - `reverseComplement_gffString()` — handles reverse strand annotations

5. **Post-processing** (`src/transgenic/utils/postprocess.py`):
   - `PredictionProcessor` — refines GSF predictions using segmentation probabilities
   - Validates start/stop codons, splice junctions, reading frames

### Gene Sentence Format (GSF)

GSF is a compact annotation format: `<features>><transcripts>`

Example: `0|CDS1|150|+|A;200|CDS2|350|+|B>CDS1|CDS2`
- Features: `start|type|end|strand|phase` separated by `;`
- Transcripts: feature IDs separated by `|`, multiple isoforms separated by `;`
- Phase encoding: A=0, B=1, C=2, .=UTR
- Coordinates: 0-indexed, end-exclusive (Python-style)

### HuggingFace Models

- `jlomas/HyenaTransgenic-768L12A6-400M` (400M params, 92% F1)
- `jlomas/HyenaTransgenic-512L9A4-160M` (160M params)

## Directory Structure

```
src/transgenic/           # Main package
├── model/                # Model definitions and tokenizer
│   ├── modeling_HyenaTransgenic.py   # HyenaDNA encoder variant (primary)
│   ├── modeling_NTTransgenic.py      # Nucleotide Transformer encoder variant
│   ├── configuration_transgenic.py   # HyenaTransgenicConfig
│   ├── tokenization_transgenic.py    # GFFTokenizer for GSF output
│   └── huggingface_integration.py    # AutoModel/AutoTokenizer registration
├── datasets/
│   ├── preprocess.py     # genome2GSFDataset() — FASTA+GFF3 → DuckDB
│   └── datasets.py       # PyTorch Dataset/DataLoader classes
└── utils/
    ├── gsf.py            # GSF ↔ GFF3 conversion
    ├── postprocess.py    # Prediction refinement (splice junctions, codons)
    └── sequence.py       # loadGenome(), reverseComplement(), validateCDS()

train/                    # Training scripts (Accelerate + W&B)
├── train_HyenaTransgenic.py          # Generic training script
├── train_HyenaTransgenic_RTX4090.py  # RTX 4090 optimized (torch.compile, TF32)
├── train_HyenaTransgenic_GB10.py     # GB10 optimized (no compile, unified memory)
└── ...                               # NT, T5, Segment, MLM variants

test/                     # Evaluation and benchmark scripts
scripts/                  # CLI utilities
├── create_database.py    # DuckDB builder (multi-species, species prefix)
├── gff2gsf.py            # GFF3 → GSF converter
├── check_system.sh       # System/GPU detection
└── ...

examples/                 # Jupyter notebooks and example data
src/run_genome_annotation.py  # Full inference pipeline CLI
```

## Training

### Training Scripts

All three main training scripts share the same CLI interface:

```bash
python train/train_HyenaTransgenic_RTX4090.py \
    --db training.db \
    --batch-size 4 \
    --accumulation-steps 64 \
    --num-workers 6 \
    --save-every-n-steps 5000 \
    --checkpoint-path checkpoints/ \
    --resume auto \
    --no-wandb
```

### Resume/Checkpoint System

- Checkpoints saved by `accelerator.save_state()` in `accelerate_epoch{N}_step{G}/` directories
- `meta.json` stores epoch, step (micro-batch), global_step (optimizer), best_eval_score
- `--resume auto` finds latest checkpoint by highest `global_step`
- Step-level resume: skips already-processed micro-batches within an epoch
- KeyboardInterrupt (Ctrl+C) triggers graceful checkpoint save

### Platform-Specific Optimizations

**RTX 4090** (24 GB VRAM, SM 8.9):
- `torch.compile("reduce-overhead")` — full Triton kernel fusion
- TF32 matmul, `pin_memory=True`, `prefetch_factor=4`
- Attention window: 768 (reduced from 1024 to fit VRAM)

**GB10** (128 GB unified memory, SM 12.1):
- `torch.compile` **disabled** — Triton's layernorm backward needs 180 KB shared memory, GB10 SM has only 101 KB
- `cudaMallocAsync` **disabled** — over-allocates on unified memory
- `pin_memory=False` — unified memory is already CPU-GPU coherent
- `set_per_process_memory_fraction(0.78)` — prevents Linux OOM killer
- Source-built PyTorch (SM 12.0 target) is 1.59x faster than pip wheels
- AdamW 8-bit optimizer for ~75% memory savings

## Environment Setup

- **x86 GPU (CUDA 12.4)**: `conda env create -f environment.yml`
- **CPU only**: `conda env create -f environment.cpu.yml`
- **GB10 ARM (CUDA 13.0)**: `conda env create -f environment.gb10.cuda.yml`
- **Python**: 3.12 (required for f-string nesting in model code)

## GFF3 Requirements

GFF3 files must be sorted using AGAT before processing:
```bash
agat_convert_sp_gxf2gxf.pl -g file.gff3 -o file.sorted.gff3
```

## Code Style

- Model files (`modeling_*.py`) and training scripts use **tabs** for indentation
- Scripts (`scripts/`) and utility files use **4 spaces** for indentation
- All source files have English docstrings and inline comments
- Type hints used in function signatures where applicable
