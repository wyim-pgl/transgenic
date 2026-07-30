#!/usr/bin/env python3
"""
TransGenic Genome Annotation Script

Run TransGenic inference on genomic sequences defined by a GFF3 file.

Usage:
    python run_genome_annotation.py <fasta_file> <gff_file> [options]

Example:
    python run_genome_annotation.py genome.fa genes.gff3 -o output.gff3 --device cuda --compile
"""

# =============================================================================
# Standard library imports for filesystem operations, subprocess management,
# argument parsing, and serialization.
# =============================================================================
import argparse
import os
import sys
import subprocess
import shutil
import warnings
import math
import json

# =============================================================================
# PyTorch imports.
# torch.multiprocessing is used instead of the standard multiprocessing module
# so that we can request a 'spawn' start method, which is critical for avoiding
# fork-related deadlocks when DuckDB file handles are shared across processes.
# =============================================================================
import torch
import torch.multiprocessing as mp
import duckdb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Attempt to import the transgenic package.  This is the core TransGenic
# library that contains dataset preprocessing, custom dataset/collate
# implementations, and GSF (Gene Structure Format) utilities.  If the import
# fails, the script terminates immediately with an informative message because
# none of the downstream pipeline steps can succeed without it.
# =============================================================================
try:
    import transgenic
except ImportError:
    # If strictly needed, one could try to append local path, but relying on environment is better
    # assuming user is in the correct environment or has installed the package
    print("Error: 'transgenic' package not found. Please activate the correct environment or install the package.", file=sys.stderr)
    sys.exit(1)

# genome2GSFDataset:  converts a FASTA + GFF3 pair into a DuckDB-backed dataset
#                     suitable for TransGenic inference (mode="predict").
# isoformDataHyena:   PyTorch Dataset that reads the DuckDB and yields tokenized
#                     samples for the Hyena-based TransGenic model.
# hyena_collate_fn:   Custom collate function that pads variable-length genomic
#                     sequences and groups metadata (gene model ID, chromosome,
#                     start/end coords) into aligned batch tuples.
# gffString2GFF3:     Parses a GSF prediction string (model output) and converts
#                     it into standard GFF3 format lines with correct genomic
#                     coordinates, adding the chromosome and offset.
from transgenic.datasets.preprocess import genome2GSFDataset
from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
from transgenic.utils.gsf import gffString2GFF3, get_cds_fingerprint, get_utr_span, dedup_gff3_file


def ensure_parent_dir(path):
    """Create parent directory for a path if it does not exist."""
    # Guard against empty/None paths (e.g. when default checkpoint is unused).
    if not path:
        return
    dirpath = os.path.dirname(os.path.abspath(path))
    # exist_ok=True prevents race conditions when multiple processes or
    # successive calls try to create the same directory.
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def check_agat_installed():
    """Checks if AGAT tool is available."""
    # AGAT (Another Gff Analysis Toolkit) provides agat_convert_sp_gxf2gxf.pl,
    # which sorts GFF3 features hierarchically (gene → mRNA → exon/CDS) and
    # validates the file structure.  TransGenic requires sorted GFF3 input so
    # that gene models are correctly grouped when building the DuckDB dataset.
    # shutil.which searches PATH for the executable; returns None if absent.
    if shutil.which("agat_convert_sp_gxf2gxf.pl"):
        return "agat_convert_sp_gxf2gxf.pl"
    return None

def sort_gff(input_gff, output_gff, agat_cmd):
    """Sorts a GFF3 file using AGAT."""
    print(f"Sorting {input_gff} to {output_gff} using {agat_cmd}...")

    # Build the command list for subprocess.  Currently only the
    # agat_convert_sp_gxf2gxf.pl path is supported; the if-guard is here
    # for future extensibility if additional sorting backends are added.
    cmd = []
    if agat_cmd == "agat_convert_sp_gxf2gxf.pl":
        cmd = [agat_cmd, "-g", input_gff, "-o", output_gff]

    try:
        # stdout is silenced (DEVNULL) because AGAT is verbose; stderr is
        # captured so it can be printed on failure for debugging.
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error executing AGAT command: {e}")
        # Print stderr if available
        if e.stderr:
            print(e.stderr.decode())
        sys.exit(1)

def main():
    # =========================================================================
    # STEP 0: Parse command-line arguments.
    # The two positional arguments (fasta_file, gff_file) are mandatory; all
    # others have sensible defaults.  The device defaults to CUDA when a GPU is
    # detected, falling back to CPU otherwise.
    # =========================================================================
    parser = argparse.ArgumentParser(description="Run Transgenic inference on genomic sequences defined by a GFF3 file.")

    parser.add_argument("fasta_file", help="Path to the genome FASTA file.")
    parser.add_argument("gff_file", help="Path to the input GFF3 file defining regions to annotate.")
    parser.add_argument("-o", "--output", default="transgenic_inference.gff3", help="Path to the output GFF3 file (default: transgenic_inference.gff3).")
    parser.add_argument("-m", "--model", default="jlomas/HyenaTransgenic-768L12A6-400M", help="HuggingFace model name or path (default: jlomas/HyenaTransgenic-768L12A6-400M).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (default: 1).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu).")
    parser.add_argument("--checkpoint_path", default=None, help="Path to a checkpoint JSON file (default: <output>.ckpt.json).")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint.")
    parser.add_argument("--keep_db", action="store_true", help="Keep the intermediate DuckDB database file.")
    parser.add_argument("--no_sort", action="store_true", help="Skip AGAT sorting (use if input GFF3 is already sorted).")
    parser.add_argument("--max_length", type=int, default=2048, help="Max generated sequence length (default: 2048). Reduce for faster runs.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers (default: 2). Set 0 if workers cause issues.")
    parser.add_argument("--precision", choices=["auto","fp32","fp16","bf16"], default="auto", help="Precision for inference: auto/fp32/fp16/bf16 (default: auto).")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when num_workers>0 (default: 2).")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to optimize model inference.")
    parser.add_argument("--reject_output", default=None, help="Path to save rejected/hallucinated generations (default: <output>.rejects.txt)")
    parser.add_argument("--bed", action="store_true", help="Treat input annotation as BED format instead of GFF3.")

    args = parser.parse_args()

    # =========================================================================
    # Pre-flight: ensure that directories for all output files exist before any
    # expensive processing begins.  This avoids discovering a bad path after
    # minutes of model loading or dataset creation.
    # =========================================================================
    ensure_parent_dir(args.output)
    # The reject file defaults to the output path with a ".rejects.txt" suffix
    # so that it lives alongside the main GFF3 output for easy discovery.
    reject_path = args.reject_output or f"{args.output}.rejects.txt"
    ensure_parent_dir(reject_path)

    # Verify input files exist early so the user gets an immediate error
    # message rather than waiting for downstream failures.
    if not os.path.exists(args.fasta_file):
        print(f"Error: FASTA file '{args.fasta_file}' not found.")
        sys.exit(1)
    if not os.path.exists(args.gff_file):
        print(f"Error: GFF file '{args.gff_file}' not found.")
        sys.exit(1)

    # The DuckDB database file is placed in the current working directory and
    # named after the FASTA basename.  This convention avoids collisions when
    # annotating multiple genomes from the same directory.
    db_path = f"{os.path.basename(args.fasta_file)}.transgenic.db"
    # When resuming, the existing database is reused to avoid the expensive
    # FASTA+GFF3 → DuckDB conversion step.
    reuse_db = args.resume and os.path.exists(db_path)

    # =========================================================================
    # STEP 1: Sort GFF3 using AGAT.
    # GFF3 files must be hierarchically sorted (gene → mRNA → exon/CDS) for
    # genome2GSFDataset to correctly group features into gene models.
    # Sorting is skipped when: (a) the user passes --no_sort, indicating the
    # GFF3 is already properly sorted, or (b) we are resuming and can reuse
    # the existing DuckDB (the sorted GFF was already consumed on the first run).
    # =========================================================================
    gff_to_use = args.gff_file
    temp_sorted_gff = None  # Track the temp file so we can clean it up later.

    if not args.no_sort and not reuse_db:
        agat_tool = check_agat_installed()
        if not agat_tool:
            print("Error: AGAT tool 'agat_convert_sp_gxf2gxf.pl' not found in PATH.")
            print("Please install AGAT (bioconda package 'agat') or use --no_sort if your GFF is already sorted.")
            sys.exit(1)

        # The sorted GFF is written to a temporary file (not in-place) so that
        # the original input GFF is never modified.  This temp file is cleaned
        # up in the finally block at the end of main().
        temp_sorted_gff = args.gff_file + ".sorted.tmp.gff3"
        sort_gff(args.gff_file, temp_sorted_gff, agat_tool)
        gff_to_use = temp_sorted_gff

    # =========================================================================
    # STEP 2: Create the DuckDB-backed dataset.
    # genome2GSFDataset reads the FASTA and (sorted) GFF3, extracts genomic
    # regions around each gene model, tokenizes the DNA sequences into the GSF
    # (Gene Structure Format) representation, and stores everything in a DuckDB
    # database for efficient random access during inference.
    # mode="predict" tells the preprocessor to prepare input-only records
    # (no target labels), suitable for inference rather than training.
    # =========================================================================
    if reuse_db:
        print(f"Resuming with existing database: {db_path}")
    else:
        # If the DB exists, we might want to start fresh or append.
        # For a clean run script, starting fresh is safer unless specified otherwise.
        # genome2GSFDataset creates tables if not exists.
        # We will remove it if it exists to ensure we process the current inputs,
        # unless logic is complex. The notebook creates a db.
        if os.path.exists(db_path):
            print(f"Removing existing database {db_path}...")
            try:
                os.remove(db_path)
                # DuckDB may leave behind write-ahead log (.wal) or temporary
                # (.tmp) sidecar files; clean those up to avoid stale state.
                if os.path.exists(db_path + ".wal"): os.remove(db_path + ".wal")
                if os.path.exists(db_path + ".tmp"): os.remove(db_path + ".tmp")
            except Exception as e:
                print(f"Warning: Could not remove existing DB: {e}")

        print("Converting input data to Transgenic Dataset (DuckDB)...")
        try:
            genome2GSFDataset(
                genome=args.fasta_file,
                gff3=gff_to_use,
                db=db_path,
                anoType="bed" if args.bed else "gff",
                mode="predict"       # Prediction mode: input-only, no targets.
            )
        except Exception as e:
            # If dataset creation fails, clean up the temp sorted GFF before
            # exiting so we don't leave orphan files on disk.
            print(f"Error during dataset creation: {e}")
            if temp_sorted_gff and os.path.exists(temp_sorted_gff):
                os.remove(temp_sorted_gff)
            sys.exit(1)

    # =========================================================================
    # STEP 3: Initialize the DataLoader.
    # The DataLoader wraps an isoformDataHyena dataset (which reads from the
    # DuckDB) and yields batches of tokenized input sequences along with
    # metadata (gene model ID, chromosome, start/end coordinates).
    # A custom SkipSampler is used instead of the default sequential sampler so
    # that resumed runs can jump ahead to the first unprocessed sample without
    # re-iterating over already-completed items.
    # =========================================================================
    print("Initializing DataLoader...")
    try:
        # mode="inference" tells the dataset to return input tokens only
        # (no teacher-forced target sequences).
        ds = isoformDataHyena(
            db_path,
            mode="inference"
        )

        # -- Checkpoint / resume logic --
        # The checkpoint is a tiny JSON file that records how many samples have
        # been fully processed.  On resume, SkipSampler uses this count as its
        # starting index so that already-processed samples are skipped.
        start_index = 0
        checkpoint_path = args.checkpoint_path or f"{args.output}.ckpt.json"

        ensure_parent_dir(checkpoint_path)

        def load_checkpoint(path):
            """Read the checkpoint file and return the number of completed samples."""
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as cp:
                        data = json.load(cp)
                        return int(data.get("completed", 0))
                except Exception as e:
                    # A corrupt checkpoint is non-fatal; we simply restart
                    # from the beginning rather than crashing.
                    print(f"Warning: Could not read checkpoint {path}: {e}")
            return 0

        def save_checkpoint(path, completed):
            """Persist the current progress (number of completed samples) to disk."""
            try:
                with open(path, "w") as cp:
                    json.dump({"completed": int(completed)}, cp)
            except Exception as e:
                # Checkpoint write failures are non-fatal; the run continues
                # and the user simply loses the ability to resume from this point.
                print(f"Warning: Could not write checkpoint {path}: {e}")

        if args.resume:
            start_index = load_checkpoint(checkpoint_path)
            # Defensive: ensure start_index is never negative (could happen if
            # the checkpoint file was manually edited with an invalid value).
            if start_index < 0:
                start_index = 0
            # If the checkpoint says everything is done, exit immediately
            # rather than spinning up the model for nothing.
            if start_index >= len(ds):
                print("Checkpoint indicates all samples processed; exiting.")
                sys.exit(0)

        # -- SkipSampler --
        # A custom sequential sampler that begins iteration at `start_idx`
        # instead of 0.  This is the mechanism that makes resume work: the
        # DataLoader only yields samples from start_idx onwards, so the model
        # never re-processes samples that were already written to the output.
        class SkipSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, start_idx=0):
                self.data_source = data_source
                self.start_idx = start_idx

            def __iter__(self):
                return iter(range(self.start_idx, len(self.data_source)))

            def __len__(self):
                return len(self.data_source) - self.start_idx

        sampler = SkipSampler(ds, start_index)

        # Assemble DataLoader keyword arguments.  pin_memory=True enables
        # faster host-to-device transfers when using CUDA.
        dl_kwargs = {
            "dataset": ds,
            "batch_size": args.batch_size,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "collate_fn": hyena_collate_fn,
        }
        if args.num_workers and args.num_workers > 0:
            # prefetch_factor controls how many batches each worker pre-loads.
            # A minimum of 2 ensures the GPU is never starved while waiting for
            # the next batch to be prepared.
            dl_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
            # persistent_workers keeps worker processes alive between batches,
            # avoiding the overhead of repeated process creation (especially
            # important because each worker must re-open the DuckDB connection).
            dl_kwargs["persistent_workers"] = True
            # CRITICAL: Use the 'spawn' multiprocessing context instead of the
            # default 'fork'.  Forking a process that has open DuckDB file
            # handles can cause deadlocks or corruption because DuckDB uses
            # internal locks that are not fork-safe.  'spawn' starts fresh
            # child processes that re-initialize all resources cleanly.
            try:
                dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            except Exception:
                # Fall back silently if 'spawn' is not available on this
                # platform (unlikely on Linux, but defensive).
                pass
        dl = DataLoader(**dl_kwargs)
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        sys.exit(1)

    # Sanity check: an empty dataset usually means the GFF3 contained no 'gene'
    # features, or the chromosome names in the GFF3 don't match the FASTA
    # sequence headers (a common user error).
    if len(ds) == 0:
        print("Warning: Dataset is empty. Check if GFF3 contains 'gene' features matching the FASTA headers.")
        sys.exit(0)

    # =========================================================================
    # STEP 4: Load the HuggingFace model and tokenizer.
    # The model is a Hyena-based sequence-to-sequence architecture that takes
    # tokenized genomic input and generates GSF-encoded gene structure
    # predictions.  trust_remote_code=True is required because the Hyena
    # architecture is defined in the model repository, not in the core
    # transformers library.
    # =========================================================================
    device = torch.device(args.device)
    if torch.cuda.is_available():
        # Disable TF32 (TensorFloat-32) for matrix multiplications.  TF32
        # trades off precision for speed by rounding FP32 mantissas to 10 bits.
        # For inference on genomic sequences, we prefer full FP32 precision to
        # avoid subtle numerical drift that could affect beam search ranking.
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            # "highest" precision maps to pure FP32 with no reduced-precision
            # fast paths.  Some older PyTorch builds may not support "highest",
            # in which case we fall back to "high" (which still avoids TF32).
            torch.set_float32_matmul_precision("highest")
        except Exception:
            torch.set_float32_matmul_precision("high")
        # Enable cuDNN auto-tuner.  This benchmarks multiple convolution
        # algorithms on the first call and caches the fastest one.  Because
        # genomic input lengths vary, the benefit is modest, but it helps
        # when many samples share similar lengths.
        torch.backends.cudnn.benchmark = True
    print(f"Loading model '{args.model}' on {device}...")

    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        model.eval()   # Set model to evaluation mode (disables dropout, etc.).
        model.to(device)
        if args.compile:
            try:
                # torch.compile with the default mode ("inductor") traces the
                # model and generates optimized kernels.  The "reduce-overhead"
                # mode is intentionally NOT used here because it aggressively
                # caches CUDA graphs, which break when input tensor shapes
                # change -- a common scenario in genomics where gene models
                # have highly variable lengths, causing frequent recompilation
                # that actually slows things down.
                model = torch.compile(model)
                print("Model compiled with torch.compile (default mode).")
            except Exception as e:
                # torch.compile may fail on older GPUs, unsupported ops, or
                # when the model uses dynamic control flow.  Falling back to
                # eager mode is always safe.
                print(f"Warning: torch.compile failed; continuing without compile: {e}")

        # Load the GSF tokenizer from the same model repository.  This
        # tokenizer is specific to the Gene Structure Format vocabulary and is
        # used to decode the model's integer output tokens back into GSF text.
        gffTokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Fail fast on a tokenizer/checkpoint vocabulary mismatch: the
        # published checkpoints have vocab_size 272 (legacy tokenizer), while
        # the current repo's default GFF tokenizer builds 288 tokens. Encoding
        # with the larger vocabulary against the smaller embedding table
        # would produce out-of-range token ids.
        model_vocab_size = getattr(model.config, "vocab_size", None)
        if model_vocab_size is not None and len(gffTokenizer.get_vocab()) > model_vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary ({len(gffTokenizer.get_vocab())} tokens) exceeds "
                f"the checkpoint's vocab_size ({model_vocab_size}). Load the tokenizer "
                f"from the checkpoint repository (AutoTokenizer.from_pretrained("
                f"'{args.model}', trust_remote_code=True)) or instantiate GFFTokenizer "
                f"with vocab_version='v1'."
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 5: Inference loop.
    # For each batch, the model generates GSF-encoded predictions via beam
    # search with sampling.  Each prediction is decoded, cleaned of special
    # tokens, and converted to GFF3 format.  Valid predictions are appended to
    # the output file; unparsable ("hallucinated") predictions are logged to a
    # separate rejects file.  A checkpoint is saved after every batch so the
    # run can be resumed if interrupted.
    # =========================================================================
    print(f"Running inference on {len(ds)} samples...")

    # Initialize or append to the output GFF3 file.
    # On a fresh run, write the mandatory GFF3 version header.
    # On resume, the file already has the header and previously written records,
    # so we leave it as-is and will append new results.
    output_exists = os.path.exists(args.output)
    if args.resume and output_exists and os.path.getsize(args.output) > 0:
        # File already has content from a previous run; do not overwrite.
        pass
    else:
        # Start a fresh output file with the GFF3 version pragma.
        with open(args.output, "w") as f:
            f.write("##gff-version 3\n")

    processed = 0   # Total number of samples processed (including resumed).
    rejected = 0    # Count of predictions that failed GFF3 conversion.
    duplicates = 0  # Count of duplicate predictions (identical CDS coordinates).
    seen_fingerprints = {}  # {fingerprint: utr_span} for deduplication.

    # On resume, rebuild fingerprints from the existing output file so that
    # predictions already written are not duplicated in the appended output.
    if args.resume and output_exists and os.path.getsize(args.output) > 0:
        gene_lines = []
        with open(args.output) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) >= 9 and cols[2] == "gene":
                    # Flush the previous gene block
                    if gene_lines:
                        fp = get_cds_fingerprint(gene_lines)
                        if fp is not None:
                            utr = get_utr_span(gene_lines)
                            if fp not in seen_fingerprints or utr > seen_fingerprints[fp]:
                                seen_fingerprints[fp] = utr
                    gene_lines = [line]
                elif len(cols) >= 9:
                    gene_lines.append(line)
            # Flush the last gene block
            if gene_lines:
                fp = get_cds_fingerprint(gene_lines)
                if fp is not None:
                    utr = get_utr_span(gene_lines)
                    if fp not in seen_fingerprints or utr > seen_fingerprints[fp]:
                        seen_fingerprints[fp] = utr
        print(f"Resume: loaded {len(seen_fingerprints)} fingerprints from existing output")

    try:
        if args.resume:
            # When resuming, credit the already-completed samples to the
            # processed counter so that checkpoint values remain correct
            # (checkpoint stores cumulative processed count, not per-session).
            processed = start_index

        # The progress bar tracks only the remaining samples (total - already
        # processed) so the ETA is accurate for this session.  Setting
        # initial=0 ensures the bar starts from zero even though `processed`
        # may be non-zero.
        remaining = len(ds) - processed
        progress = tqdm(total=remaining, desc="Inference", initial=0)
        for batch in dl:
            # Move input tensors to the target device.  non_blocking=True
            # allows the transfer to overlap with other operations when using
            # pinned memory (pin_memory=True in the DataLoader).
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)

            # Metadata fields from the batch tuple (set by hyena_collate_fn):
            # batch[3]: geneModel IDs (tuple/list) -- unique identifier for each gene model
            # batch[4]: Chromosome names            -- e.g. "chr1", "scaffold_42"
            # batch[5]: Region Start coords         -- 0-based start of the genomic window
            # batch[6]: Region End coords            -- 0-based end of the genomic window

            # Run the model in inference mode.  torch.inference_mode() is
            # stricter than torch.no_grad(): it also disables version counting
            # on tensors and enables additional backend optimizations, yielding
            # a small speed improvement over no_grad alone.
            with torch.inference_mode():
                outputs = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    num_return_sequences=1,   # One prediction per input sample.
                    max_length=2048,          # Cap output length to prevent runaway generation.
                    num_beams=2,              # Beam search with 2 beams for better quality.
                    do_sample=True            # Enable sampling within beam search for diversity.
                )
            # Decode the integer token IDs back to GSF text strings.
            # .detach().cpu().numpy() moves the tensor off the GPU and converts
            # to a numpy array, which is what the HuggingFace tokenizer expects.
            decoded = gffTokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

            # Process each prediction in the batch individually, because each
            # has its own metadata (chromosome, coordinates, gene model ID).
            for i, pred_raw in enumerate(decoded):
                # Strip residual special tokens that batch_decode may leave
                # behind (e.g. sentence boundary markers).  The "|</s>" pattern
                # is specific to the GSF tokenizer's end-of-sequence format.
                pred_clean = pred_raw.replace("|</s>", "").replace("</s>", "").replace("<s>", "")

                # Extract per-sample metadata from the batch.
                # When batch_size > 1, each metadata field is a tuple/list and
                # we index by `i` to get the value for this specific sample.
                gm_id = batch[3][i]
                chrom = batch[4][i]
                start_coord = batch[5][i]

                # Convert the cleaned GSF prediction string into GFF3 lines.
                # gffString2GFF3 parses the relative feature coordinates from
                # the GSF encoding and translates them to absolute genomic
                # coordinates using `chrom` and `start_coord`.  Internally it
                # converts from 0-based GSF offsets to 1-based GFF3 positions.
                # The GM= attribute is added to the GFF3 attributes column to
                # trace each output feature back to its source gene model.
                try:
                    gff_lines = gffString2GFF3(pred_clean, chrom, start_coord, f"GM={gm_id}")
                except Exception as err:
                    # -- Handling rejected / hallucinated predictions --
                    # The model may occasionally produce malformed GSF strings
                    # that gffString2GFF3 cannot parse (e.g. missing fields,
                    # impossible coordinates, or completely nonsensical output).
                    # Rather than crashing the entire run, we log the failure
                    # to a dedicated rejects file and continue with the next
                    # sample.  This is essential for large-scale annotation
                    # where a handful of bad predictions should not halt the
                    # processing of thousands of valid ones.
                    rejected += 1
                    # Build a metadata dictionary capturing everything needed
                    # to debug the rejection later.  The .item() call handles
                    # the case where gm_id is a 0-d tensor rather than a plain
                    # Python scalar.
                    meta = {
                        "gm_id": gm_id if not hasattr(gm_id, "item") else gm_id.item(),
                        "chrom": chrom,
                        "start": int(start_coord) if hasattr(start_coord, "__int__") else start_coord,
                        "pred_raw": pred_raw,
                        "pred_clean": pred_clean,
                        "error": f"{type(err).__name__}: {err}"
                    }
                    try:
                        # Append one JSON line per rejected sample.  Using
                        # append mode ("a") ensures concurrent or resumed runs
                        # accumulate all rejects in one file.
                        with open(reject_path, "a") as rej:
                            rej.write(json.dumps(meta) + "\n")
                    except Exception as log_err:
                        # If even the reject log fails (e.g. disk full),
                        # print a warning but do not crash.
                        print(f"Warning: failed to log rejected sample: {log_err}")
                    continue  # Skip to the next sample in this batch.

                # gffString2GFF3 returns [""] for unparseable predictions that
                # don't raise an exception (e.g. missing CDS1, empty transcripts).
                if gff_lines == [""] or gff_lines == []:
                    rejected += 1
                    continue

                # Deduplicate predictions by CDS coordinate fingerprint.
                # Two predictions with identical (chrom, strand, CDS coords)
                # represent the same gene predicted independently.  When a
                # duplicate is found, the version with longer total UTR span
                # is preferred (it captures more regulatory context).
                fp = get_cds_fingerprint(gff_lines)
                if fp is not None:
                    utr = get_utr_span(gff_lines)
                    if fp in seen_fingerprints:
                        if utr <= seen_fingerprints[fp]:
                            # New prediction has equal or shorter UTR; skip it.
                            duplicates += 1
                            continue
                        # New prediction has longer UTR; write it now and the
                        # superseded entry will be removed by dedup_gff3_file
                        # at the end of the run.
                        duplicates += 1
                    seen_fingerprints[fp] = utr

                # Append the successfully converted GFF3 lines to the output.
                # Opening the file in append mode for each sample (rather than
                # keeping a persistent file handle) ensures that data is flushed
                # to disk promptly, which is important for resumability -- if
                # the process is killed, all lines written so far are safe.
                with open(args.output, "a") as f:
                    for line in gff_lines:
                        f.write(line + "\n")

            # Update the cumulative processed count by the actual number of
            # samples in this batch (which may be less than batch_size for the
            # final batch).
            processed += len(batch[0])
            progress.update(len(batch[0]))
            # Persist the checkpoint after every batch.  This is the key to
            # resumability: if the process is interrupted, the checkpoint
            # records exactly how many samples were completed, and SkipSampler
            # will skip them on the next run.
            save_checkpoint(checkpoint_path, processed)
            progress.set_postfix({"processed": processed})

    except Exception as e:
        # Catch-all for unexpected errors during the inference loop (e.g. OOM,
        # CUDA errors, corrupted data).  The traceback is printed for debugging,
        # and execution falls through to the finally block for cleanup.
        print(f"Error during inference loop: {e}")
        # traceback could be useful
        import traceback
        traceback.print_exc()

    finally:
        # =====================================================================
        # Cleanup: always runs, whether inference completed normally, was
        # interrupted by an exception, or exited early.
        # =====================================================================

        # Remove the temporary sorted GFF3 file created in Step 1.  The
        # original input GFF is never modified, so this is safe to delete.
        if temp_sorted_gff and os.path.exists(temp_sorted_gff):
            os.remove(temp_sorted_gff)

        # Unless --keep_db was specified, remove the intermediate DuckDB
        # database.  Users may want to keep it for debugging or to avoid
        # re-running the expensive preprocessing step on subsequent runs.
        if not args.keep_db:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    # Also remove the DuckDB write-ahead log if present.
                    if os.path.exists(db_path + ".wal"): os.remove(db_path + ".wal")
                except OSError as e:
                    print(f"Warning: Failed to cleanup database file: {e}")

    # Final dedup cleanup: remove superseded gene blocks from the output
    # file.  During streaming writes, a prediction with longer UTR may have
    # replaced an earlier one in seen_fingerprints, leaving both versions in
    # the file.  This pass keeps only the last (longest-UTR) version.
    if os.path.exists(args.output):
        superseded = dedup_gff3_file(args.output)
        if superseded:
            print(f"Cleanup: removed {superseded} superseded gene blocks from output")

    # Print a summary of rejected predictions (if any) so the user knows to
    # review the rejects file.
    if rejected:
        print(f"Rejected {rejected} predictions; saved to {reject_path}")
    if duplicates:
        print(f"Removed {duplicates} duplicate predictions (identical CDS coordinates)")
    print(f"Done. Results saved to {args.output}")

if __name__ == "__main__":
    main()
