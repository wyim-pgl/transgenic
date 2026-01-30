# TransGenic
TransGenic is a transformer for DNA-to-annotation machine translation. Gene annotations specify the structure of a gene within a DNA sequence by providing the composition of each mRNA transcript based on the coordinate locations of sub-genic features, including coding sequences (CDS), introns, and untranslated regions (UTR). TransGenic uses a HyenaDNA encoder with the Longformer decoder to predict a text-based annotation format from raw DNA sequence.

![TransGenic Workflow](Figures/Gemini_Generated_Image_mb4afmb4afmb4afm.png)

## Architecture

TransGenic uses an encoder-decoder architecture:
- **Encoder**: [HyenaDNA](https://github.com/HazyResearch/hyena-dna), a long-range genomic foundation model capable of processing sequences up to 1 million nucleotides using sub-quadratic convolution operations instead of full attention
- **Decoder**: [Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)-based autoregressive decoder that generates structured text annotations

This design enables the model to capture long-range dependencies in DNA while producing human-readable outputs.

## Key Features

- **De novo annotation**: Generate complete gene structures from unannotated DNA sequences
- **Splice variant prediction**: Predict alternative isoforms via prompt completion given an existing transcript
- **Compact output format**: Gene Sentence Format (GSF) reduces annotation redundancy for efficient generation
- **Plant-focused**: Trained on 9 phylogenetically diverse plant species
- **High accuracy**: Achieves 92% base-level F1 score on *Arabidopsis thaliana* test data

# Gene sentence format (GSF)
TransGenic produces output in a format modified from the standard Gene Feature Format (GFF). Gene sentence format (GSF) contains identical information as GFF but reduces the redundancy and length of output annotations. This permits generative decoding within reasonable memory requirements for the decoder's attention mechanisms.

Gene sentence format specifies gene model outputs in two parts, a feature list and a transcript list. The feature list specifies the coordinate locations of sub-genic features (CDS, 5'-UTR, and 3'-UTR) and the transcript list specifies the composition of spliced mRNA transcripts based on the components in the feature list.

## GSF Format Structure

GSF consists of two parts separated by `>`:
```
<feature_list>><transcript_list>
```

### Feature List
Each feature follows the format: `start|type|end|strand|phase`
- **start**: 0-indexed start coordinate (relative to extracted sequence)
- **type**: Feature type with unique number (CDS1, CDS2, five_prime_UTR1, three_prime_UTR1, etc.)
- **end**: End coordinate (exclusive, like Python slicing)
- **strand**: `+` (forward) or `-` (reverse)
- **phase**: Reading frame for CDS features
  - `A` = phase 0 (codon starts at position 0)
  - `B` = phase 1 (codon starts at position 1)
  - `C` = phase 2 (codon starts at position 2)
  - `.` = not applicable (for UTRs)

Multiple features are separated by `;`

### Transcript List
After the `>` separator, transcripts list their component features:
- Features are separated by `|`
- Multiple transcripts (isoforms) are separated by `;`

## Examples

### Example 1: Simple single-transcript gene (3 CDS)
**GFF:**
```
Chr1  source  gene  100  400  .  +  .  ID=gene1
Chr1  source  mRNA  100  400  .  +  .  ID=mRNA1
Chr1  source  CDS   100  150  .  +  0  ID=cds1
Chr1  source  CDS   200  280  .  +  2  ID=cds2
Chr1  source  CDS   350  400  .  +  1  ID=cds3
```
**GSF:**
```
0|CDS1|50|+|A;100|CDS2|180|+|C;250|CDS3|300|+|B>CDS1|CDS2|CDS3
```
Note: Coordinates are relative to the extracted sequence (gene start = 0).

### Example 2: Gene with alternative splicing (2 transcripts)
**GFF:**
```
Chr1  source  gene  100  350  .  +  .  ID=gene1
Chr1  source  mRNA  100  350  .  +  .  ID=mRNA1
Chr1  source  CDS   100  130  .  +  0  ID=cds1
Chr1  source  CDS   180  220  .  +  1  ID=cds2
Chr1  source  CDS   280  350  .  +  0  ID=cds3
Chr1  source  mRNA  180  350  .  +  .  ID=mRNA2
Chr1  source  CDS   180  220  .  +  1  ID=cds2
Chr1  source  CDS   280  350  .  +  0  ID=cds3
```
**GSF:**
```
0|CDS1|30|+|A;80|CDS2|120|+|B;180|CDS3|250|+|A>CDS1|CDS2|CDS3;CDS2|CDS3
```
- First transcript uses all three CDS: `CDS1|CDS2|CDS3`
- Second transcript skips CDS1 (alternative start): `CDS2|CDS3`
- Coordinates are relative to gene start (100 → 0)

### Example 3: Gene with UTRs
**GFF:**
```
Chr1  source  gene            500  900  .  +  .  ID=gene1
Chr1  source  mRNA            500  900  .  +  .  ID=mRNA1
Chr1  source  five_prime_UTR  500  550  .  +  .  ID=utr5
Chr1  source  CDS             550  650  .  +  0  ID=cds1
Chr1  source  CDS             700  800  .  +  1  ID=cds2
Chr1  source  three_prime_UTR 800  900  .  +  .  ID=utr3
```
**GSF:**
```
0|five_prime_UTR1|50|+|.;50|CDS1|150|+|A;200|CDS2|300|+|B;300|three_prime_UTR1|400|+|.>five_prime_UTR1|CDS1|CDS2|three_prime_UTR1
```
- UTRs use `.` for phase since they are non-coding
- Transcript includes UTRs in the proper order

## Converting GFF3 to GSF

Use `scripts/gff2gsf.py` to convert existing GFF3 annotations to GSF format:

```bash
# Basic usage (output to stdout)
python scripts/gff2gsf.py annotation.gff3

# Save to file
python scripts/gff2gsf.py annotation.gff3 -o output.gsf

# Use absolute coordinates instead of relative
python scripts/gff2gsf.py annotation.gff3 --absolute
```

**Output format** (tab-separated):
```
gene_id    GSF_string
AT1G01010  0|CDS1|150|+|A;200|CDS2|350|+|B>CDS1|CDS2
AT1G01020  0|five_prime_UTR1|50|+|.;50|CDS1|200|+|A>five_prime_UTR1|CDS1
```

# Using TransGenic
## Quick start

Try TransGenic instantly on Google Colab (no installation required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnnyLomas/transgenic/blob/main/examples/Transgenic_SingleSequence_Colab.ipynb)

### Minimal Example

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizers from HuggingFace
model_name = "jlomas/HyenaTransgenic-768L12A6-400M"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
gsf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
dna_tokenizer = AutoTokenizer.from_pretrained(
    "LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True
)

# Tokenize DNA sequence
seq = "ATGCGT...your_sequence...TGATGA"
input_ids = dna_tokenizer.batch_encode_plus(
    [seq], return_tensors="pt"
)["input_ids"][:, :-1]

# Generate annotation
model.eval()
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")
    model.to("cuda")

outputs = model.generate(
    inputs=input_ids,
    max_length=2048,
    num_beams=2,
    do_sample=True
)

# Decode to GSF format
gsf_prediction = gsf_tokenizer.batch_decode(
    outputs.detach().cpu().numpy(),
    skip_special_tokens=True
)[0]
print(gsf_prediction)
# Output: 0|CDS1|150|+|A;200|CDS2|350|+|B>CDS1|CDS2
```

For local development, run notebook examples from the `examples/` folder after setting up an environment as described below.

## Installation

### Quick Install (pip)

If you already have PyTorch installed:

```bash
# Clone and install
git clone git@github.com:JohnnyLomas/transgenic.git
cd transgenic
pip install -e .
```

### Full Environment Setup (conda)

For a complete environment with all dependencies, first clone the repository:

```bash
git clone git@github.com:JohnnyLomas/transgenic.git
cd transgenic
```

Run `./scripts/check_system.sh` to determine which environment file to use, then follow the appropriate instructions below.

#### Environment Options

- **x86 with NVIDIA GPU** (`environment.yml`): For Linux/Windows with GTX, RTX, or Tesla GPUs. Includes CUDA 12.4.
  ```bash
  conda env create -f environment.yml
  conda activate transgenic
  pip install -e .
  ```

- **x86 CPU only** (`environment.cpu.yml`): For systems without GPU (macOS, VMs, CPU-only machines). Slower but fully functional.
  ```bash
  conda env create -f environment.cpu.yml
  conda activate transgenic
  pip install -e .
  ```

- **GB10 ARM** (`environment.gb10.base.yml`): For NVIDIA Grace Blackwell aarch64 systems. Uses a two-step install to avoid dependency conflicts.
  ```bash
  conda env create -f environment.gb10.base.yml -y
  conda activate transgenic
  ./scripts/install_ml_stack_gb10.sh
  ```

#### Verify CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Pretrained Checkpoints on HuggingFace

All checkpoints were trained on 9 plant genomes covering diverse phyla, including dicot, monocot, and moss species. The highest performance on test set evaluation (92% base-level F1 in *Arabidopsis*) was achieved using the 400M parameter model. Both checkpoints used sequences padded with neighboring genomic sequence to the next multiple of 6144 nucleotides.

### Training Data
Nine phylogenetically diverse plant species:
- *Arabidopsis thaliana*, *Glycine max* (Soybean), *Oryza sativa* (Rice)
- *Sorghum bicolor*, *Populus trichocarpa* (Poplar), *Brachypodium distachyon*
- *Vitis vinifera* (Grape), *Setaria italica* (Millet), *Physcomitrella patens* (Moss)

### Available Models

| Model | Parameters | Hidden Size | Layers | Attention Heads | F1 Score |
|-------|------------|-------------|--------|-----------------|----------|
| [HyenaTransgenic-768L12A6-400M](https://huggingface.co/jlomas/HyenaTransgenic-768L12A6-400M) | ~400M | 768 | 12 | 6 | 92% |
| [HyenaTransgenic-512L9A4-160M](https://huggingface.co/jlomas/HyenaTransgenic-512L9A4-160M) | ~160M | 512 | 9 | 4 | - |

### Training Configuration
- **Learning rate**: 5e-5
- **Batch size**: 96 (effective)
- **Loss**: Cross Entropy
- **Mixed precision**: BF16
- **Input length**: Multiples of 6,144nt (max 49,152nt)

### Intended Uses
1. Generate *de novo* annotations for plant DNA sequences containing genes
2. Add alternatively spliced isoforms to known primary mRNA transcripts via prompt completion

## Inference

The general outline of an inference workflow is:
1. Create a [DuckDB](https://duckdb.org/) database from a FASTA and a [GFF3|BED] file which describes the sequences to be used for prediction
2. Initialize a [PyTorch Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for the database
3. Generate annotations using `model.generate`
4. Convert GSF outputs to a GFF3 formatted output file

### Example Notebooks

**[Single Sequence Inference](https://github.com/JohnnyLomas/transgenic/blob/main/examples/Transgenic_SingleSequence.ipynb)**
- Annotate a single DNA sequence using a pretrained model
- Basic workflow: load model → encode sequence → generate GSF → convert to GFF3

**[Multi-Sequence Inference](https://github.com/JohnnyLomas/transgenic/blob/main/examples/Transgenic_MultiSequence.ipynb)**
- Batch annotation of multiple gene regions from a genome
- De novo prediction from BED file (gene coordinates only)
- Splice variant prediction from GFF3 file (prompt completion with existing transcript)

### Example Data Files

The `examples/` folder includes *Arabidopsis thaliana* chromosome 4 data files for testing:

| File | Description |
|------|-------------|
| `ATH_Chr4.fas` | FASTA sequence file for chromosome 4 |
| `ATH_Chr4_gene.bed` | BED file with gene coordinates |
| `ATH_Chr4.sorted.gff3` | Sorted GFF3 annotation file |

### GFF3 Sorting Requirement

When building databases from GFF3 files, TransGenic expects the GFF3 to be sorted using a sort order similar to the one used by [AGAT (Another GFF Analysis Toolkit)](https://github.com/NBISweden/AGAT). To sort using AGAT:
```bash
agat_convert_sp_gxf2gxf.pl -g [file.gff3] -o [file.sorted.gff3]
```

See [AGAT documentation](https://agat.readthedocs.io/) for installation and usage.

## Training

Training scripts are located in the [`train/`](https://github.com/JohnnyLomas/transgenic/tree/main/train) folder. These scripts use the [Accelerate](https://huggingface.co/docs/accelerate) library for distributed training and [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_HyenaTransgenic.py` | Main training script for HyenaDNA encoder with Longformer decoder |
| `train_NTTransgenic.py` | Training with Nucleotide Transformer encoder |
| `train_HyenaT5Transgenic.py` | T5 decoder with HyenaDNA encoder |
| `train_NTT5Transgenic.py` | T5 decoder with Nucleotide Transformer encoder |
| `train_HyenaSegment.py` | Segmentation model training with HyenaDNA |
| `train_NTSegment.py` | Segmentation model training with Nucleotide Transformer |
| `train_HyenaMLM.py` | Masked language model pretraining |

### Training Workflow

#### 1. Prepare Training Data

Create a DuckDB database from your genome FASTA and sorted GFF3 annotation files using the preprocessing utilities:

```python
from transgenic.datasets.preprocess import genome2GSFDataset

# For training data (includes GSF labels)
genome2GSFDataset(
    genome="genome.fasta",
    gff3="annotations.sorted.gff3",
    db="training_data.db",
    anoType="gff",
    mode="train",
    maxLen=49152,        # Max sequence length (49,152bp = 8,192 tokens)
    addExtra=200,        # Random buffer for UTR boundaries
    staticSize=6144,     # Sequences padded to multiples of this size
    addRC=True,          # Add reverse complement augmentation
    addRCIsoOnly=True,   # Only augment genes with alternative splicing
    clean=True           # Validate CDS start/stop codons
)

# Append additional genomes to the same database
genome2GSFDataset(
    genome="genome2.fasta",
    gff3="annotations2.sorted.gff3",
    db="training_data.db",  # Same database
    ...
)
```

#### 2. Configure and Run Training

Edit the training script to set your database path and hyperparameters:

```python
# In train/train_HyenaTransgenic.py
db = "training_data.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf")
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [train_size, eval_size, test_size])

trainTransgenicFCGAccelerate(
    train_data,
    eval_data,
    lr=5e-5,
    num_epochs=10,
    schedule_lr=True,
    eval=True,
    batch_size=1,
    accumulation_steps=128,  # Effective batch size = batch_size * accumulation_steps
    checkpoint_path="checkpoints/",
    max_grad_norm=1.0,
    log_wandb=True
)
```

#### 3. Launch Training

```bash
# Single GPU
python train/train_HyenaTransgenic.py

# Multi-GPU with Accelerate
accelerate launch train/train_HyenaTransgenic.py
```

#### 4. Monitor Training

Training metrics are logged to Weights & Biases:
- Loss and perplexity per step/epoch
- Gradient norms for each layer
- Learning rate schedule

### Key Hyperparameters

The pretrained models used:
- **Learning rate**: 5e-5
- **Effective batch size**: 96-128 (via gradient accumulation)
- **Mixed precision**: BF16
- **Optimizer**: AdamW with weight decay 0.02
- **Scheduler**: Linear warmup
- **Gradient clipping**: max norm 1.0
- **Input length**: Multiples of 6,144nt (max 49,152nt)

## Test Scripts

The `test/` folder contains evaluation and benchmark scripts for different model configurations:

| Script | Description |
|--------|-------------|
| `test_AgroSegmentNT.py` | Segmentation evaluation with AgroNT + Segment-NT encoder |
| `test_HyenaSegmentNT.py` | Segmentation evaluation with HyenaDNA encoder |
| `testingAdjustCoords.py` | Combined segmentation + generation with coordinate refinement |
| `testingHyena.py` | HyenaDNA generation model (without post-processing) |
| `testingHyenaCompletion.py` | Prompt completion for splice variant prediction |
| `testingHyenaDual.py` | Separate decoder and segmentation model pipeline |
| `testingHyenaPostnoPost.py` | Compare raw vs post-processed predictions |
| `testingNT.py` | Nucleotide Transformer based generation + segmentation |
| `testingT5Hyena.py` | T5 decoder with HyenaDNA encoder |
| `testingT5Transgenic.py` | T5 decoder with AgroNT encoder + segmentation |
| `testSingle_tomato.py` | Single sequence inference example (tomato gene) |

## Scripts

The `scripts/` folder contains utility scripts:

| Script | Description |
|--------|-------------|
| `check_system.sh` | Check system architecture and GPU for environment selection |
| `gff2gsf.py` | Convert GFF3 annotations to GSF format |
| `install_ml_stack_gb10.sh` | Install PyTorch + HuggingFace stack for GB10 ARM |
| `test_torch_cuda_gb10.py` | CUDA verification test for GB10 |

## License

This project is licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International License (CC-BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).

You are free to share and redistribute the material for any purpose, including commercially, as long as you give appropriate credit and do not distribute modified versions.
