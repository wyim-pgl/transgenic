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
        help="Batch size for inference (default: 1)"
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
    return parser.parse_args()


def get_decoder_input_ids(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract elements of the first transcript to use as decoder input IDs.

    This parses the label sequence to find the first transcript's features,
    which are then used to condition the model to generate additional isoforms.

    Args:
        labels: Tensor of label token IDs
        device: Device to place the output tensor on

    Returns:
        Tensor of decoder input IDs for the first transcript
    """
    labs = ",".join([str(i) for i in labels.tolist()[0]])

    # Token 17 separates transcripts, token 21 separates features
    last_element = labs.split(",17,")[1].split(",21,")[0].split(",")[-1]

    try:
        last_element_index = [
            f",{last_element}," in i
            for i in labs.split(",17,")[0].split(",21,")
        ].index(True)
    except ValueError:
        last_element_index = len(labs.split(",17,")[0].split(",21,")) - 1

    # Reconstruct the first transcript's token sequence
    first_transcript = ",21,".join(
        labs.split(",17,")[0].split(",21,")[0:last_element_index + 1]
    )
    decoder_ids = torch.tensor(
        list(map(int, first_transcript.split(",")))
    ).unsqueeze(0).to(device)

    return decoder_ids


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

    model.to(device)
    model.eval()

    # Initialize dataset and dataloader
    print("Initializing dataset...")
    ds_comp = isoformDataHyena(db_path, mode="train")
    dl_comp = DataLoader(
        ds_comp,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=hyena_collate_fn
    )

    # Clear output file if it exists
    if os.path.exists(args.output):
        os.remove(args.output)

    # Prediction loop
    print(f"Generating predictions...")
    total = args.num_sequences or len(dl_comp)

    for step, batch in enumerate(tqdm(dl_comp, total=total)):
        if args.num_sequences and step >= args.num_sequences:
            break

        # Unpack batch: input_ids, attention_mask, labels, gene_model, chrom, start
        ii = batch[0].to(device)
        am = batch[1].to(device)
        lab = batch[2].to(device)
        gene_model = batch[3][0]
        chrom = batch[4][0]
        start = batch[5][0]

        # Get decoder input IDs from first transcript
        dii = get_decoder_input_ids(lab, device)

        # Generate annotation with beam search
        with torch.no_grad():
            outputs = model.generate(
                inputs=ii,
                attention_mask=am,
                num_return_sequences=1,
                max_length=args.max_length,
                num_beams=args.num_beams,
                do_sample=True,
                decoder_input_ids=dii,
                use_cache=False
            )

        # Decode the output to GSF
        pred = gffTokenizer.batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True
        )[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")

        # Convert the GSF to GFF3
        gff = gffString2GFF3(pred, chrom, start, f"GM={gene_model}")

        # Write the GFF3 output
        with open(args.output, "a") as f:
            for line in gff:
                f.write(line + "\n")

    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
