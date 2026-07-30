#!/usr/bin/env python3
"""
Prompt Mode Prediction with TransGenic

This script uses TransGenic to add splice variants to existing annotations.
It constructs a dataset with GSF labels using a sorted reference GFF3 annotation,
then provides features of the first transcript as input to the decoder to complete
the annotation with additional isoforms.

Usage:
    python prompt_mode.py --genome GENOME.fas --gff ANNOTATION.gff3 --output OUTPUT.gff

Example:
    python prompt_mode.py --genome ATH_Chr4.fas --gff ATH_Chr4.sorted.gff3 --output ath_chr4_completion.gff
"""

import argparse
import os
import sys
import warnings
import threading

# Suppress all warnings and HuggingFace noise BEFORE imports
# warnings.filterwarnings("ignore")
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["SAFETENSORS_FAST_GPU"] = "0"
# os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached models only
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Prevent transformers from checking HF Hub

import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("safetensors").setLevel(logging.CRITICAL)

# Suppress thread exceptions (e.g., safetensors auto-conversion)
threading.excepthook = lambda args: None

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transgenic.datasets.preprocess import genome2GSFDataset
from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
from transgenic.utils.gsf import gffString2GFF3
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt mode prediction using TransGenic"
    )
    parser.add_argument(
        "--genome",
        type=str,
        required=True,
        help="Path to genome FASTA file"
    )
    parser.add_argument(
        "--gff",
        type=str,
        required=True,
        help="Path to sorted GFF3 annotation file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="completion.gff",
        help="Output GFF3 file path (default: completion.gff)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to database file (default: derived from output name)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jlomas/HyenaTransgenic-768L12A6-400M",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum output sequence length (default: 2048)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=2,
        help="Number of beams for beam search (default: 2)"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=None,
        help="Number of sequences to process (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip database creation if it already exists"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress parsing error messages"
    )
    return parser.parse_args()


def get_single_decoder_input_ids(labels_list: list) -> list:
    """
    Extract elements of the first transcript to use as decoder input IDs.
    Returns a list of integers (token IDs).
    """
    labs = ",".join([str(i) for i in labels_list])

    # Token 17 separates transcripts, token 21 separates features
    try:
        last_element = labs.split(",17,")[1].split(",21,")[0].split(",")[-1]
        last_element_index = [
            f",{last_element}," in i
            for i in labs.split(",17,")[0].split(",21,")
        ].index(True)
    except (ValueError, IndexError):
        last_element_index = len(labs.split(",17,")[0].split(",21,")) - 1

    # Reconstruct the first transcript's token sequence
    first_transcript = ",21,".join(
        labs.split(",17,")[0].split(",21,")[0:last_element_index + 1]
    )
    return list(map(int, first_transcript.split(",")))


def get_batch_decoder_input_ids(labels: torch.Tensor, device: torch.device, pad_token_id: int = 0, decoder_start_token_id: int = 2) -> torch.Tensor:
    """
    Extract decoder input IDs for a batch of labels.
    Pads to the max length in the batch.
    Prepends decoder_start_token_id so HuggingFace generate() doesn't
    auto-prepend it (which would add +1 token and overflow max_length).
    """
    batch_ids = []
    max_prompt_len = 2047  # Reserve 1 for decoder_start_token_id
    for i in range(labels.shape[0]):
        ids = get_single_decoder_input_ids(labels[i].tolist())
        # Trim to max_prompt_len
        ids = ids[:max_prompt_len]
        # Prepend decoder_start_token_id so HF doesn't auto-prepend it
        ids = [decoder_start_token_id] + ids
        batch_ids.append(ids)

    # Pad to max length
    max_len = max(len(ids) for ids in batch_ids)
    padded = []
    for ids in batch_ids:
        padded.append(ids + [pad_token_id] * (max_len - len(ids)))

    return torch.tensor(padded, device=device)


def main():
    args = parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set database path
    db_path = args.db or args.output.replace(".gff", ".db")

    # Create database if needed
    if not args.skip_preprocessing or not os.path.exists(db_path):
        print(f"Creating database from {args.gff}...")
        genome2GSFDataset(
            args.genome,
            args.gff,
            db_path,
            anoType="gff",
            mode="train"
        )
    else:
        print(f"Using existing database: {db_path}")

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = transgenicForConditionalGeneration.from_pretrained(args.model)
    gffTokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Fail fast on a tokenizer/checkpoint vocabulary mismatch: the published
    # checkpoints have vocab_size 272 (legacy tokenizer), while the current
    # repo's default GFF tokenizer builds 288 tokens. Encoding prompt GFFs
    # with the larger vocabulary would produce out-of-range token ids.
    model_vocab_size = getattr(model.config, "vocab_size", None)
    if model_vocab_size is not None and len(gffTokenizer.get_vocab()) > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(gffTokenizer.get_vocab())} tokens) exceeds "
            f"the checkpoint's vocab_size ({model_vocab_size}). Load the tokenizer "
            f"from the checkpoint repository (AutoTokenizer.from_pretrained("
            f"'{args.model}', trust_remote_code=True)) or instantiate GFFTokenizer "
            f"with vocab_version='v1'."
        )

    model.to(device)
    model.eval()

    # Enable optimizations
    torch.backends.cudnn.benchmark = True

    # Initialize dataset and dataloader
    print("Initializing dataset...")
    # The dataset's prompt tokenizer must match the checkpoint vocabulary:
    # published checkpoints (vocab_size 272) require the legacy "v1" scheme,
    # newly trained isoform-aware models use the default 288-token "v2".
    gff_vocab_version = "v1" if model_vocab_size is not None and model_vocab_size <= 272 else "v2"
    ds_comp = isoformDataHyena(db_path, mode="train", gff_vocab_version=gff_vocab_version)
    dl_comp = DataLoader(
        ds_comp,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=hyena_collate_fn,
        prefetch_factor=8
    )

    # Clear output file if it exists
    if os.path.exists(args.output):
        os.remove(args.output)

    # Prediction loop
    print(f"Generating predictions...")
    total = args.num_sequences or len(dl_comp)
    error_count = 0

    # Redirect stderr if quiet mode
    if args.quiet:
        stderr_backup = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    for step, batch in enumerate(tqdm(dl_comp, total=total, miniters=5, mininterval=10)):
        if args.num_sequences and step >= args.num_sequences:
            break

        # Unpack batch: input_ids, attention_mask, labels, gene_model, chrom, start
        ii = batch[0].to(device)
        am = batch[1].to(device)
        lab = batch[2]
        gene_models = batch[3]
        chroms = batch[4]
        starts = batch[5]
        batch_size = ii.shape[0]

        # Get decoder input IDs for the batch (prepends decoder_start_token_id)
        dii = get_batch_decoder_input_ids(lab, device, decoder_start_token_id=model.config.decoder_start_token_id)

        # Generate annotation with beam search
        # Use max_new_tokens instead of max_length to avoid overflow when
        # decoder_input_ids is already close to the model's max sequence length.
        max_new_tokens = max(1, args.max_length - dii.shape[1])
        with torch.no_grad():
            outputs = model.generate(
                inputs=ii,
                attention_mask=am,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                decoder_input_ids=dii,
                use_cache=True
            )

        # Decode the outputs to GSF
        preds = gffTokenizer.batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True
        )

        # Process each sample in the batch
        for i in range(batch_size):
            pred = preds[i].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
            gff = gffString2GFF3(pred, chroms[i], starts[i], f"GM={gene_models[i]}")

            # Track parsing errors
            if gff == [""] or gff == []:
                error_count += 1

            # Write the GFF3 output (skip empty results from parsing errors)
            with open(args.output, "a") as f:
                for line in gff:
                    if line:
                        f.write(line + "\n")

    # Restore stderr if quiet mode
    if args.quiet:
        sys.stderr.close()
        sys.stderr = stderr_backup

    print(f"Output written to: {args.output}")
    if error_count > 0:
        print(f"Parsing errors: {error_count}/{total} sequences skipped")


if __name__ == "__main__":
    main()
