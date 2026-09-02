#!/usr/bin/env python
"""
Create a DuckDB database for TransGenic training or inference.

Converts genome FASTA + GFF3 annotation pairs into a DuckDB database that
serves as the data backend for TransGenic's PyTorch DataLoader pipeline.

Supports building multi-species databases by specifying multiple --fasta/--gff
pairs. Each pair is appended to the same database sequentially.

Examples:
  # Single species (training)
  python create_database.py \
    --fasta Arabidopsis.fa --gff Arabidopsis.gff3 \
    --output train.db --mode train --add-extra 200 --add-rc-iso-only --clean

  # Single species (inference)
  python create_database.py \
    --fasta Zea_mays.fa --gff Zea_mays.gff3 \
    --output predict.db --mode predict

  # Multi-species (training, 9 plant genomes)
  python create_database.py \
    --fasta At.fa Os.fa Sl.fa Vv.fa Sb.fa Bd.fa Pp.fa Mp.fa Cr.fa \
    --gff   At.gff3 Os.gff3 Sl.gff3 Vv.gff3 Sb.gff3 Bd.gff3 Pp.gff3 Mp.gff3 Cr.gff3 \
    --output 9species_train.db --mode train \
    --add-extra 200 --add-rc-iso-only --clean

  # Append more species to an existing database
  python create_database.py \
    --fasta Zm.fa --gff Zm.gff3 \
    --output 9species_train.db --mode train --append
"""
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    from transgenic.datasets.preprocess import genome2GSFDataset
except ImportError:
    print(
        "Error: 'transgenic' package not found.\n"
        "Install it with: pip install -e /path/to/transgenic\n"
        "Or activate the correct conda/micromamba environment.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a DuckDB database for TransGenic training or inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single species, training mode
  python create_database.py --fasta At.fa --gff At.gff3 --output train.db --mode train

  # Multi-species, training mode with augmentation
  python create_database.py \\
    --fasta At.fa Os.fa Sl.fa --gff At.gff3 Os.gff3 Sl.gff3 \\
    --output multi.db --mode train --add-extra 200 --add-rc-iso-only --clean

  # Inference mode (no labels)
  python create_database.py --fasta Zm.fa --gff Zm.gff3 --output predict.db --mode predict
        """,
    )

    # Required arguments
    parser.add_argument(
        "--fasta", type=str, nargs="+", required=True,
        help="Path(s) to genome FASTA file(s). Must match --gff in count and order.",
    )
    parser.add_argument(
        "--gff", type=str, nargs="+", required=True,
        help="Path(s) to sorted GFF3 annotation file(s). Must match --fasta in count and order.",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output DuckDB database file path.",
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["train", "predict"], default="train",
        help="'train' includes GSF labels; 'predict' stores only gene regions (default: train).",
    )

    # Training augmentation options
    aug = parser.add_argument_group("Training augmentation options")
    aug.add_argument(
        "--add-extra", type=int, default=0,
        help="Max random buffer (bp) added to each side for training augmentation. "
             "Helps model learn UTR boundaries (default: 0, recommended: 200).",
    )
    aug.add_argument(
        "--rc", choices=["none", "all", "isoform-only"], default=None,
        help="Reverse-complement augmentation: none | all genes | only genes with >= 2 transcripts "
             "(replaces --add-rc/--add-rc-iso-only, which remain as deprecated aliases).",
    )
    aug.add_argument(
        "--add-rc", action="store_true",
        help="Add reverse complement copies of all genes for data augmentation.",
    )
    aug.add_argument(
        "--add-rc-iso-only", action="store_true",
        help="Add reverse complement copies only for genes with alternative splicing "
             "(multiple mRNA transcripts). Mutually exclusive with --add-rc.",
    )
    aug.add_argument(
        "--clean", action="store_true",
        help="Validate CDS integrity (start/stop codons, reading frame) and skip invalid genes.",
    )

    # Sequence parameters
    seq = parser.add_argument_group("Sequence parameters")
    seq.add_argument(
        "--static-size", type=int, default=6144,
        help="Pad sequences to multiples of this size. Must match the encoder chunk size (default: 6144).",
    )
    seq.add_argument(
        "--max-len", type=int, default=49152,
        help="Skip genes whose padded region exceeds this length (default: 49152 = 8 x 6144).",
    )
    seq.add_argument(
        "--ano-type", choices=["gff", "bed"], default="gff",
        help="Annotation file format (default: gff).",
    )

    # Species prefix
    pfx = parser.add_argument_group("Species prefix (chromosome naming)")
    pfx.add_argument(
        "--species-prefix", type=str, nargs="*", default=None,
        help="Prefix(es) to prepend to chromosome names (e.g., 'Zm' turns Chr01 into Zm_Chr01). "
             "By default, the prefix is auto-derived from each FASTA filename "
             "(e.g., Zm.fa → 'Zm'). Provide explicit values to override "
             "(must match --fasta in count and order). Use --no-prefix to disable.",
    )
    pfx.add_argument(
        "--no-prefix", action="store_true",
        help="Do not add species prefix to chromosome names.",
    )

    # Behavior
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing database instead of overwriting.",
    )

    return parser.parse_args()


def resolve_rc_mode(args) -> str:
    """Single source of truth for RC augmentation. --add-rc-iso-only alone used to be a silent no-op."""
    if args.rc is not None:
        if args.add_rc or args.add_rc_iso_only:
            print("Error: --rc cannot be combined with the deprecated --add-rc/--add-rc-iso-only flags.", file=sys.stderr)
            sys.exit(1)
        return args.rc
    if args.add_rc_iso_only:
        print("Warning: --add-rc-iso-only is deprecated; use --rc isoform-only.", file=sys.stderr)
        return "isoform-only"
    if args.add_rc:
        print("Warning: --add-rc is deprecated; use --rc all.", file=sys.stderr)
        return "all"
    return "none"


def main():
    args = parse_args()

    # Validate matching fasta/gff counts
    if len(args.fasta) != len(args.gff):
        print(
            f"Error: --fasta ({len(args.fasta)} files) and --gff ({len(args.gff)} files) "
            f"must have the same number of arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate input files exist
    for f in args.fasta + args.gff:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Validate mutually exclusive RC options
    rc_mode = resolve_rc_mode(args)
    args.add_rc = rc_mode != "none"
    args.add_rc_iso_only = rc_mode == "isoform-only"

    # Resolve species prefixes for chromosome naming
    if args.no_prefix:
        prefixes = [""] * len(args.fasta)
    elif args.species_prefix is not None:
        if len(args.species_prefix) != len(args.fasta):
            print(
                f"Error: --species-prefix ({len(args.species_prefix)} values) "
                f"must match --fasta ({len(args.fasta)} files) in count.",
                file=sys.stderr,
            )
            sys.exit(1)
        prefixes = args.species_prefix
    else:
        # Auto-derive from FASTA filenames: Zm.fa → "Zm", Arabidopsis_thaliana.fa → "Arabidopsis_thaliana"
        prefixes = [os.path.splitext(os.path.basename(f))[0] for f in args.fasta]

    # Handle existing database
    if os.path.exists(args.output) and not args.append:
        print(f"Removing existing database: {args.output}", file=sys.stderr)
        os.remove(args.output)
        # Also remove DuckDB sidecar files
        for ext in (".wal", ".tmp"):
            sidecar = args.output + ext
            if os.path.exists(sidecar):
                os.remove(sidecar)

    # Create output directory if needed
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Process each genome/annotation pair
    total_start = time.time()
    for i, (fasta, gff, prefix) in enumerate(zip(args.fasta, args.gff, prefixes)):
        species = os.path.splitext(os.path.basename(fasta))[0]
        print(
            f"\n[{i+1}/{len(args.fasta)}] Processing {species}...",
            file=sys.stderr,
        )
        print(f"  FASTA: {fasta}", file=sys.stderr)
        print(f"  GFF3:  {gff}", file=sys.stderr)
        if prefix:
            print(f"  Prefix: {prefix}_ (e.g., Chr01 → {prefix}_Chr01)", file=sys.stderr)

        t0 = time.time()
        genome2GSFDataset(
            genome=fasta,
            gff3=gff,
            db=args.output,
            anoType=args.ano_type,
            mode=args.mode,
            maxLen=args.max_len,
            addExtra=args.add_extra,
            staticSize=args.static_size,
            addRC=args.add_rc,
            addRCIsoOnly=args.add_rc_iso_only,
            clean=args.clean,
            speciesPrefix=prefix,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", file=sys.stderr)

    total_elapsed = time.time() - total_start

    # Print summary
    db_size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Database created: {args.output} ({db_size_mb:.1f} MB)", file=sys.stderr)
    print(f"Species: {len(args.fasta)}, Mode: {args.mode}", file=sys.stderr)
    print(f"Total time: {total_elapsed:.1f}s", file=sys.stderr)
    if args.mode == "train":
        opts = []
        if args.add_extra:
            opts.append(f"addExtra={args.add_extra}")
        opts.append(f"rc={rc_mode}")
        if args.clean:
            opts.append("clean")
        if opts:
            print(f"Augmentation: {', '.join(opts)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
