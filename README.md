# TransGenic 
TransGenic is a transformer for DNA-to-annotation machine translation. Gene annotations specify the structure of a gene within a DNA sequence by providing the composition of each mRNA transcript based on the coordinate locations of sub-genic features, including coding sequences (CDS), introns, and unstranslated regions (UTR). TransGenic uses a HyenaDNA encoder with the Longformer decoder to predict a text-based annotation format from raw DNA sequence. 

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
gffTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
dnaTokenizer = AutoTokenizer.from_pretrained(
    "LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True
)

# Tokenize DNA sequence
seq = "ATGCGT...your_sequence...TGATGA"
input_ids = dnaTokenizer.batch_encode_plus(
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
gsf_prediction = gffTokenizer.batch_decode(
    outputs.detach().cpu().numpy(),
    skip_special_tokens=True
)[0]
print(gsf_prediction)
# Output: 0|CDS1|150|+|A;200|CDS2|350|+|B>CDS1|CDS2
```

For local development, run notebook examples from the `examples/` folder after setting up an environment as described below.

## Set-up

```bash
# Clone the repo
git clone git@github.com:JohnnyLomas/transgenic.git
cd transgenic
```

### x86 with CUDA GPU

Standard installation using conda/mamba with the full environment file (includes CUDA):

```bash
# Create environment with all dependencies
mamba create -y -n transgenic && mamba activate transgenic
mamba env update -f environment.yml

# Install the transgenic module
pip install -e .
```

### x86 CPU Only (No GPU)

For systems without NVIDIA GPU:

```bash
# Create CPU-only environment
mamba create -y -n transgenic && mamba activate transgenic
mamba env update -f environment.cpu.yml

# Install the transgenic module
pip install -e .
```

### GB10 ARM CPU (NVIDIA Grace Blackwell)
`environment.gb10.base.yml` and `scripts/install_ml_stack_gb10.sh` are installation files for NVIDIA GB10 ARM CPU environments. This two-step approach is recommended for aarch64 platforms:

1. **Base environment via conda-forge**: Only basic dependencies (numpy, pandas, etc.) are installed through conda-forge
2. **PyTorch via pip from PyTorch index**: torch, torchvision, and torchaudio are installed exclusively from the official PyTorch wheel index

This separation provides the most stable setup on aarch64 architectures, avoiding common dependency conflicts.

```bash
# Remove existing environment (if present)
micromamba env remove -n transgenic -y || true

# Create base environment
micromamba env create -f environment.gb10.base.yml -y
micromamba activate transgenic

# Install ML stack (PyTorch CUDA + HuggingFace + transgenic)
chmod +x scripts/install_ml_stack_gb10.sh
./scripts/install_ml_stack_gb10.sh

# Optional: For development with editable install, run additionally:
# pip install -e .
```

## Pretrained Checkpoints on HuggingFace

All checkpoints were trained on 9 plant genomes covering diverse phyla, including dicot, monocot, and moss species. The highest performance on test set evaluation (92% base-level F1 in *Arabidopsis*) was achieved using the 400M parameter model. Both checkpoints used sequences padded with neighboring genomic sequence to the next multiple of 6144 nucleotides.

[HyenaTransgenic-768L12A6-400M](https://huggingface.co/jlomas/HyenaTransgenic-768L12A6-400M)
- Hidden size = 768, 12 layers, 6 attention heads (~400M parameters)

[HyenaTransgenic-512L9A4-160M](https://huggingface.co/jlomas/HyenaTransgenic-512L9A4-160M)
- Hidden size = 512, 9 layers, 4 attention heads (~160M parameters)

## Inference

The general outline of an inference workflow is:
1. Create a DuckDB database from a FASTA and a [GFF3|BED] file which describes the sequences to be used for prediction
2. Initialize a PyTorch Dataset and DataLoader for the database
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

NOTE: When building databases from GFF3 files, TransGenic expects the GFF3 to be sorted using a sort order similar to the one used by AGAT. To sort using AGAT:
```
agat_convert_sp_gxf2gxf.pl -g [file.gff3] -o [file.sorted.gff3]
```

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
| `gff2gsf.py` | Convert GFF3 annotations to GSF format |
| `install_ml_stack_gb10.sh` | Install PyTorch + HuggingFace stack for GB10 ARM |
| `test_torch_cuda_gb10.py` | CUDA verification test for GB10 |
