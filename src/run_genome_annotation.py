#!/usr/bin/env python3
"""
TransGenic Genome Annotation Script

Run TransGenic inference on genomic sequences defined by a GFF3 file.

Usage:
    python run_genome_annotation.py <fasta_file> <gff_file> [options]

Example:
    python run_genome_annotation.py genome.fa genes.gff3 -o output.gff3 --device cuda --compile
"""

import argparse
import os
import sys
import subprocess
import shutil
import warnings
import math
import json
import torch
import torch.multiprocessing as mp
import duckdb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Attempt to import transgenic
try:
    import transgenic
except ImportError:
    # If strictly needed, one could try to append local path, but relying on environment is better
    # assuming user is in the correct environment or has installed the package
    print("Error: 'transgenic' package not found. Please activate the correct environment or install the package.", file=sys.stderr)
    sys.exit(1)

from transgenic.datasets.preprocess import genome2GSFDataset
from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
from transgenic.utils.gsf import gffString2GFF3


def ensure_parent_dir(path):
    """Create parent directory for a path if it does not exist."""
    if not path:
        return
    dirpath = os.path.dirname(os.path.abspath(path))
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def check_agat_installed():
    """Checks if AGAT tool is available."""
    # Use AGAT's agat_convert_sp_gxf2gxf.pl (sorts by default)
    if shutil.which("agat_convert_sp_gxf2gxf.pl"):
        return "agat_convert_sp_gxf2gxf.pl"
    return None

def sort_gff(input_gff, output_gff, agat_cmd):
    """Sorts a GFF3 file using AGAT."""
    print(f"Sorting {input_gff} to {output_gff} using {agat_cmd}...")

    cmd = []
    if agat_cmd == "agat_convert_sp_gxf2gxf.pl":
        cmd = [agat_cmd, "-g", input_gff, "-o", output_gff]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error executing AGAT command: {e}")
        # Print stderr if available
        if e.stderr:
            print(e.stderr.decode())
        sys.exit(1)

def main():
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

    args = parser.parse_args()

    # Ensure output directory exists before attempting to write
    ensure_parent_dir(args.output)
    reject_path = args.reject_output or f"{args.output}.rejects.txt"
    ensure_parent_dir(reject_path)

    # Verify input files
    if not os.path.exists(args.fasta_file):
        print(f"Error: FASTA file '{args.fasta_file}' not found.")
        sys.exit(1)
    if not os.path.exists(args.gff_file):
        print(f"Error: GFF file '{args.gff_file}' not found.")
        sys.exit(1)

    # Define db path early
    db_path = f"{os.path.basename(args.fasta_file)}.transgenic.db"
    reuse_db = args.resume and os.path.exists(db_path)

    # 1. Sort GFF3
    gff_to_use = args.gff_file
    temp_sorted_gff = None

    if not args.no_sort and not reuse_db:
        agat_tool = check_agat_installed()
        if not agat_tool:
            print("Error: AGAT tool 'agat_convert_sp_gxf2gxf.pl' not found in PATH.")
            print("Please install AGAT (bioconda package 'agat') or use --no_sort if your GFF is already sorted.")
            sys.exit(1)

        # Create a temporary filename for sorted GFF
        temp_sorted_gff = args.gff_file + ".sorted.tmp.gff3"
        sort_gff(args.gff_file, temp_sorted_gff, agat_tool)
        gff_to_use = temp_sorted_gff

    # 2. Create DuckDB Dataset
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
                # Cleanup wal or tmp files if any
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
                anoType="gff",
                mode="predict"
            )
        except Exception as e:
            print(f"Error during dataset creation: {e}")
            if temp_sorted_gff and os.path.exists(temp_sorted_gff):
                os.remove(temp_sorted_gff)
            sys.exit(1)

    # 3. Initialize DataLoader
    print("Initializing DataLoader...")
    try:
        ds = isoformDataHyena(
            db_path,
            mode="inference"
        )

        start_index = 0
        checkpoint_path = args.checkpoint_path or f"{args.output}.ckpt.json"

        ensure_parent_dir(checkpoint_path)

        def load_checkpoint(path):
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as cp:
                        data = json.load(cp)
                        return int(data.get("completed", 0))
                except Exception as e:
                    print(f"Warning: Could not read checkpoint {path}: {e}")
            return 0

        def save_checkpoint(path, completed):
            try:
                with open(path, "w") as cp:
                    json.dump({"completed": int(completed)}, cp)
            except Exception as e:
                print(f"Warning: Could not write checkpoint {path}: {e}")

        if args.resume:
            start_index = load_checkpoint(checkpoint_path)
            if start_index < 0:
                start_index = 0
            if start_index >= len(ds):
                print("Checkpoint indicates all samples processed; exiting.")
                sys.exit(0)

        class SkipSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, start_idx=0):
                self.data_source = data_source
                self.start_idx = start_idx

            def __iter__(self):
                return iter(range(self.start_idx, len(self.data_source)))

            def __len__(self):
                return len(self.data_source) - self.start_idx

        sampler = SkipSampler(ds, start_index)

        dl_kwargs = {
            "dataset": ds,
            "batch_size": args.batch_size,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "collate_fn": hyena_collate_fn,
        }
        if args.num_workers and args.num_workers > 0:
            dl_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
            dl_kwargs["persistent_workers"] = True
            # Use 'spawn' to avoid fork-related deadlocks with DuckDB/file handles
            try:
                dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            except Exception:
                pass
        dl = DataLoader(**dl_kwargs)
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        sys.exit(1)

    if len(ds) == 0:
        print("Warning: Dataset is empty. Check if GFF3 contains 'gene' features matching the FASTA headers.")
        sys.exit(0)

    # 4. Load Model
    device = torch.device(args.device)
    if torch.cuda.is_available():
        # Highest precision: disable TF32 for true FP32 math
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    print(f"Loading model '{args.model}' on {device}...")

    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        model.eval()
        model.to(device)
        if args.compile:
            try:
                # 'reduce-overhead' is problematic for variable sequence lengths (common in genomics)
                # causing frequent recompilation. Using default mode instead.
                model = torch.compile(model)
                print("Model compiled with torch.compile (default mode).")
            except Exception as e:
                print(f"Warning: torch.compile failed; continuing without compile: {e}")

        gffTokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 5. Inference Loop
    print(f"Running inference on {len(ds)} samples...")

    # Initialize output file
    output_exists = os.path.exists(args.output)
    if args.resume and output_exists and os.path.getsize(args.output) > 0:
        pass
    else:
        with open(args.output, "w") as f:
            f.write("##gff-version 3\n")

    processed = 0
    rejected = 0

    try:
        if args.resume:
            processed = start_index

        # Use sample-based progress accounting to avoid inflated ETA
        remaining = len(ds) - processed
        progress = tqdm(total=remaining, desc="Inference", initial=0)
        for batch in dl:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)

            # Metadata
            # batch[3]: geneModel IDs (tuple/list)
            # batch[4]: Chromosome names
            # batch[5]: Region Start coords
            # batch[6]: Region End coords

            with torch.inference_mode():
                outputs = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    num_return_sequences=1,
                    max_length=2048,
                    num_beams=2,
                    do_sample=True
                )
            # Decode outputs
            decoded = gffTokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

            for i, pred_raw in enumerate(decoded):
                # Clean special tokens
                pred_clean = pred_raw.replace("|</s>", "").replace("</s>", "").replace("<s>", "")

                # Get metadata for this sample
                # If batch_size > 1, batch[3] is a tuple of IDs
                gm_id = batch[3][i]
                chrom = batch[4][i]
                start_coord = batch[5][i]

                # Convert prediction string to GFF3 format lines
                # gffString2GFF3 expects start_coord to be strictly the start of the region (0-based in logic)
                # The function adds 1 internally for GFF3 (1-based) output
                try:
                    gff_lines = gffString2GFF3(pred_clean, chrom, start_coord, f"GM={gm_id}")
                except Exception as err:
                    # Capture hallucinated / unparsable generations without killing the run
                    rejected += 1
                    meta = {
                        "gm_id": gm_id if not hasattr(gm_id, "item") else gm_id.item(),
                        "chrom": chrom,
                        "start": int(start_coord) if hasattr(start_coord, "__int__") else start_coord,
                        "pred_raw": pred_raw,
                        "pred_clean": pred_clean,
                        "error": f"{type(err).__name__}: {err}"
                    }
                    try:
                        with open(reject_path, "a") as rej:
                            rej.write(json.dumps(meta) + "\n")
                    except Exception as log_err:
                        print(f"Warning: failed to log rejected sample: {log_err}")
                    continue

                # Append to output file
                with open(args.output, "a") as f:
                    for line in gff_lines:
                        f.write(line + "\n")

            processed += len(batch[0])
            progress.update(len(batch[0]))
            save_checkpoint(checkpoint_path, processed)
            progress.set_postfix({"processed": processed})

    except Exception as e:
        print(f"Error during inference loop: {e}")
        # traceback could be useful
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if temp_sorted_gff and os.path.exists(temp_sorted_gff):
            os.remove(temp_sorted_gff)

        if not args.keep_db:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    if os.path.exists(db_path + ".wal"): os.remove(db_path + ".wal")
                except OSError as e:
                    print(f"Warning: Failed to cleanup database file: {e}")

    if rejected:
        print(f"Rejected {rejected} predictions; saved to {reject_path}")
    print(f"Done. Results saved to {args.output}")

if __name__ == "__main__":
    main()
