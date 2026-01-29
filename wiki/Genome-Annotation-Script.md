# Genome Annotation Script

`run_genome_annotation.py` - Genome annotation inference using TransGenic model

## Overview

This script takes a genome FASTA and gene region GFF3 as input, and predicts transcript structures using the TransGenic model.

```
Input: Genome FASTA + Gene regions GFF3
  ↓
[AGAT Sort] → [DuckDB Dataset] → [Model Inference] → [GFF3 Output]
```

## Usage

```bash
python run_genome_annotation.py <fasta_file> <gff_file> [options]
```

### Basic Example

```bash
python run_genome_annotation.py \
    genome.fa \
    genes.gff3 \
    -o output.gff3
```

### Production Example (Maize)

```bash
systemd-run --user --scope --unit=zmays_annotation_01 -p MemorySwapMax=0 \
    python3 run_genome_annotation.py \
    /home/pgl/data/genomes/Zmays_493_APGv4.fa \
    /home/pgl/data/genomes/Zmays_493_RefGen_V4.gene_exons.exon.gff3 \
    -o /home/pgl/results/transgenic/zmays_transgenic_inference.gff3 \
    --batch_size 1 \
    --num_workers 1 \
    --prefetch_factor 1 \
    --device cuda \
    --reject_output /home/pgl/results/transgenic/zmays_annotation_rejects.txt \
    --compile
```

---

## Command Line Arguments

### Positional Arguments (Required)

| Argument | Description |
|----------|-------------|
| `fasta_file` | Path to genome FASTA file |
| `gff_file` | Path to input GFF3 file defining regions to annotate |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `transgenic_inference.gff3` | Output GFF3 file path |
| `--reject_output` | `<output>.rejects.txt` | Path to save parsing failures/hallucinations |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `jlomas/HyenaTransgenic-768L12A6-400M` | HuggingFace model name or local path |
| `--device` | `cuda` (if available) | Device to use (`cuda` / `cpu`) |
| `--precision` | `auto` | Computation precision (`auto` / `fp32` / `fp16` / `bf16`) |
| `--compile` | `False` | Use `torch.compile()` to optimize model |

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | `1` | Inference batch size |
| `--max_length` | `2048` | Maximum generated sequence length (shorter = faster) |

### DataLoader Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num_workers` | `2` | Number of DataLoader workers (0 = main process loading) |
| `--prefetch_factor` | `2` | Batches to prefetch per worker (only when num_workers > 0) |

### Checkpoint & Resume Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint_path` | `<output>.ckpt.json` | Checkpoint file path |
| `--resume` | `False` | Resume from existing checkpoint |

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no_sort` | `False` | Skip AGAT sorting (if GFF3 is already sorted) |
| `--keep_db` | `False` | Keep intermediate DuckDB file (for debugging) |

---

## Detailed Option Explanation

### `--batch_size 1`

**Why use 1?**
- Genomic sequences have highly variable lengths (hundreds of bp to tens of kb)
- Variable length makes batch padding inefficient
- GPU memory management becomes difficult
- **Batch size 1 is most stable and memory efficient**

### `--num_workers 1` & `--prefetch_factor 1`

**Why use low values?**
- Prevents DuckDB file handle sharing issues
- Avoids fork-related deadlocks (script uses `spawn` context)
- Multiple workers cause memory spikes on large genomes
- **Stability first**: 1 worker is sufficient since GPU is the bottleneck

```python
# Internal script logic
if args.num_workers > 0:
    dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")  # spawn instead of fork
```

### `--compile` (torch.compile Detailed Explanation)

`--compile` enables **PyTorch 2.0+ `torch.compile()`** functionality.

#### What is torch.compile?

```python
# Internal script code
if args.compile:
    model = torch.compile(model)
```

Converts the PyTorch model to optimized code through **JIT (Just-In-Time) compilation**.

#### How It Works

```
[Standard PyTorch]
Python → Eager Execution (line by line)

[torch.compile]
Python → Graph Capture → Optimization → Compiled Kernel
                              ↓
                    - Operator Fusion
                    - Memory Planning
                    - CUDA Graph
```

#### Performance Comparison

| Mode | First Batch | Subsequent Batches | Total Time (10K samples) |
|------|-------------|-------------------|--------------------------|
| **Without compile** | Fast | 1x | Baseline |
| **With compile** | Slow (compiling) | 1.2-2x faster | **20-40% reduction** |

#### Pros and Cons

**Pros:**
- **Inference speed improvement** (20-50%)
- GPU memory optimization
- Reduced kernel launches through operator fusion

**Cons:**
- **Compilation overhead on first batch** (tens of seconds)
- Possible recompilation with variable sequence lengths
- Compatibility issues with some models/operations

#### When to Use?

| Situation | Use --compile |
|-----------|---------------|
| Large number of samples (>1000) | ✅ Recommended |
| Small number of samples (<100) | ❌ Overhead outweighs benefit |
| Long expected runtime | ✅ Recommended |
| Debugging/testing | ❌ Not needed |

#### Real-World Performance Example

```
Maize genome (44K genes):

Without --compile:
  - Average 0.5 sec/sample
  - Total 6 hours

With --compile:
  - First batch: 30 sec (compilation)
  - Subsequently average 0.35 sec/sample
  - Total 4.3 hours (28% reduction)
```

**Note:** `reduce-overhead` mode causes recompilation on every variable-length input common in genomics, making it slower. The script uses **default mode** instead.

### `--device cuda`

- Explicitly specifies GPU usage
- Auto-detection is possible but explicit specification is safer
- 10-100x faster than CPU

### `--reject_output`

**Hallucination Handling:**
- Model may generate incorrectly formatted output
- Parsing failures are logged to reject file
- Main inference continues uninterrupted

```python
# Information saved on failure
meta = {
    "gm_id": gm_id,
    "chrom": chrom,
    "start": start_coord,
    "pred_raw": pred_raw,
    "pred_clean": pred_clean,
    "error": f"{type(err).__name__}: {err}"
}
```

---

## Why Use systemd-run

```bash
systemd-run --user --scope --unit=zmays_annotation_01 -p MemorySwapMax=0 python3 ...
```

### Option Explanation

| Option | Description |
|--------|-------------|
| `--user` | Run in user session (no root required) |
| `--scope` | Create transient scope unit (one-time execution, not a service) |
| `--unit=zmays_annotation_01` | Specify unit name (for monitoring/management) |
| `-p MemorySwapMax=0` | **Disable swap usage** |

### Why Use systemd-run?

1. **Disable Swap (`MemorySwapMax=0`)**
   - Swap during GPU inference causes severe performance degradation
   - OOM killer terminating the process is better than swapping
   - Prevents DuckDB + PyTorch memory management conflicts

2. **Resource Isolation**
   - Resource limiting/monitoring via cgroup
   - Minimizes impact on other processes

3. **Process Management**
   - Check status: `systemctl --user status zmays_annotation_01`
   - Stop: `systemctl --user stop zmays_annotation_01`
   - View logs: `journalctl --user -u zmays_annotation_01`

### Alternative: Direct Execution

Running directly without swap disabled is possible:
```bash
python3 run_genome_annotation.py ...
```

However, systemd-run is recommended for large genomes (Maize 2.1GB)

---

## Pipeline Flow

```
1. Input Validation
   └── Verify FASTA, GFF3 files exist

2. GFF3 Sorting (optional)
   └── Uses AGAT (agat_convert_sp_gxf2gxf.pl)
   └── Skip with --no_sort

3. Dataset Creation
   └── genome2GSFDataset() → Creates DuckDB
   └── Extracts sequences per gene region

4. DataLoader Initialization
   └── isoformDataHyena dataset
   └── Supports checkpoint resume

5. Model Loading
   └── HuggingFace AutoModel
   └── torch.compile (optional)

6. Inference Loop
   └── Batch inference
   └── GFF3 string → coordinate conversion
   └── Checkpoint save (per batch)
   └── Reject handling

7. Cleanup
   └── Delete temp files
   └── Delete DuckDB (unless --keep_db)
```

---

## Output Files

| File | Description |
|------|-------------|
| `<output>.gff3` | Predicted transcript annotations |
| `<output>.ckpt.json` | Checkpoint (processed sample count) |
| `<output>.rejects.txt` | Failed prediction parsing (JSON lines) |

### Output GFF3 Format

```
##gff-version 3
Chr1  transgenic  gene   1000  5000  .  +  .  ID=xxx;GM=original_gene_id
Chr1  transgenic  mRNA   1000  5000  .  +  .  ID=xxx.t1;Parent=xxx
Chr1  transgenic  exon   1000  1500  .  +  .  Parent=xxx.t1
Chr1  transgenic  CDS    1100  1500  .  +  0  Parent=xxx.t1
...
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch_size (if already 1, reduce max_length)
--batch_size 1 --max_length 1024
```

### DataLoader Deadlock

```bash
# Disable workers
--num_workers 0
```

### AGAT Not Found

```bash
# Install AGAT
conda install -c bioconda agat

# Or skip sorting (if GFF is already sorted)
--no_sort
```

### Resume from Crash

```bash
# Add --resume to the same command
python run_genome_annotation.py ... --resume
```

---

## Performance Tips

1. **GPU required** - CPU is very slow
2. **SSD recommended** - Heavy DuckDB I/O
3. **Use `--compile`** - Effective for long runs
4. **Use checkpoints** - Large genomes may crash
5. **Disable swap** - Use systemd-run or swapoff

---

## Example Commands

### Arabidopsis (Small genome)

```bash
python run_genome_annotation.py \
    TAIR10_genome.fa \
    TAIR10_genes.gff3 \
    -o arabidopsis_transgenic.gff3 \
    --device cuda \
    --compile
```

### Rice (Medium genome)

```bash
python run_genome_annotation.py \
    Rice_IRGSP_genome.fa \
    Rice_genes.gff3 \
    -o rice_transgenic.gff3 \
    --batch_size 1 \
    --num_workers 1 \
    --device cuda \
    --compile
```

### Maize (Large genome, with systemd)

```bash
systemd-run --user --scope --unit=maize_annotation -p MemorySwapMax=0 \
    python3 run_genome_annotation.py \
    Zmays_genome.fa \
    Zmays_genes.gff3 \
    -o maize_transgenic.gff3 \
    --batch_size 1 \
    --num_workers 1 \
    --prefetch_factor 1 \
    --device cuda \
    --reject_output maize_rejects.txt \
    --compile
```
