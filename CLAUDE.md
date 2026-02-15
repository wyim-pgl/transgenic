# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TransGenic is a DNA-to-annotation machine translation model using HyenaDNA encoder + Longformer decoder. It generates gene structure annotations (CDS, UTR, introns) from raw DNA sequences in Gene Sentence Format (GSF).

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Run single sequence inference
python examples/prompt_mode.py --genome GENOME.fas --gff ANNOTATION.gff3 --output OUTPUT.gff

# Convert GFF3 to GSF format
python scripts/gff2gsf.py annotation.gff3 -o output.gsf

# Check system for environment selection
./scripts/check_system.sh

# Training (single GPU)
python train/train_HyenaTransgenic.py

# Training (multi-GPU)
accelerate launch train/train_HyenaTransgenic.py
```

## Architecture

### Core Components

- **Encoder**: HyenaDNA (`LongSafari/hyenadna-large-1m-seqlen-hf`) - processes DNA sequences up to 1M nucleotides
- **Decoder**: Longformer-based autoregressive decoder - generates GSF text annotations
- **Model class**: `transgenicForConditionalGeneration` in `src/transgenic/model/modeling_HyenaTransgenic.py`

### Data Pipeline

1. **Preprocessing** (`src/transgenic/datasets/preprocess.py`):
   - `genome2GSFDataset()` - creates DuckDB database from FASTA + GFF3/BED files
   - Stores sequences padded to multiples of 6144nt, with GSF labels for training

2. **Dataset** (`src/transgenic/datasets/datasets.py`):
   - `isoformDataHyena` - PyTorch Dataset for loading from DuckDB
   - `hyena_collate_fn` - collate function for DataLoader

3. **GSF Utilities** (`src/transgenic/utils/gsf.py`):
   - `gffString2GFF3()` - converts model output (GSF) back to GFF3 format
   - `reverseComplement_gffString()` - handles reverse strand annotations

### Gene Sentence Format (GSF)

GSF is a compact annotation format: `<features>><transcripts>`

Example: `0|CDS1|150|+|A;200|CDS2|350|+|B>CDS1|CDS2`
- Features: `start|type|end|strand|phase` separated by `;`
- Transcripts: feature IDs separated by `|`, multiple isoforms separated by `;`
- Phase encoding: A=0, B=1, C=2, .=UTR

### Key HuggingFace Models

- `jlomas/HyenaTransgenic-768L12A6-400M` (400M params, 92% F1)
- `jlomas/HyenaTransgenic-512L9A4-160M` (160M params)

## Directory Structure

- `src/transgenic/` - main package (model, datasets, utils)
- `train/` - training scripts using Accelerate + W&B
- `test/` - evaluation scripts for different model configs
- `examples/` - notebooks and example scripts
- `scripts/` - utility scripts (gff2gsf.py, check_system.sh)

## Environment Setup

- **GPU (CUDA 12.4)**: `conda env create -f environment.yml`
- **CPU only**: `conda env create -f environment.cpu.yml`
- **GB10 ARM**: `conda env create -f environment.gb10.yml`

## GFF3 Requirements

GFF3 files must be sorted using AGAT before processing:
```bash
agat_convert_sp_gxf2gxf.pl -g file.gff3 -o file.sorted.gff3
```
