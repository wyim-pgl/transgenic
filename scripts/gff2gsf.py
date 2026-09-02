#!/usr/bin/env python3
"""Convert GFF3 to Gene Sentence Format (GSF) with the frozen serializer (docs/gsf_spec_v1.md).

Output (tab-separated): gene_id<TAB>GSF. Coordinates are relative to the gene start by default
(legacy behaviour), to the padded 6,144-nt window with --window, or absolute with --absolute.
Records over the caps are reported on stderr and skipped, never truncated.
"""
import argparse
import os
import sys
import types


def _load_gsf_contract():
    try:
        from transgenic.utils import gsf_contract as gc  # type: ignore
        return gc
    except Exception:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "transgenic", "utils", "gsf_contract.py")
        mod = types.ModuleType("gsf_contract")
        mod.__file__ = path
        sys.modules["gsf_contract"] = mod
        with open(path) as fh:
            exec(compile(fh.read(), path, "exec"), mod.__dict__)
        return mod


def gff2gsf(gff_file, output, coords="gene"):
    gc = _load_gsf_contract()
    n = skipped = 0
    for gene in gc.parse_gff3(gff_file):
        if not any(gene.transcripts.values()):
            continue
        if coords == "absolute":
            ws = 0
        elif coords == "window":
            ws, _ = gc.pad_window(gene.start0, gene.end0)
        else:
            ws = gene.start0
        gsf = gc.gene_to_gsf(gene, ws)
        try:
            gc.check_caps(gsf, window_len=(gc.pad_window(gene.start0, gene.end0)[1] - ws) if coords == "window" else None)
        except gc.CapError as e:
            print(f"skip {gene.gene_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        output.write(f"{gene.gene_id}\t{gsf}\n")
        n += 1
    return n, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input GFF3 file (use - for stdin)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--absolute", action="store_true", help="Absolute chromosome coordinates (0-based half-open)")
    parser.add_argument("--window", action="store_true", help="Coordinates relative to the padded 6,144-nt window (database convention)")
    args = parser.parse_args()
    if args.absolute and args.window:
        parser.error("--absolute and --window are mutually exclusive")
    coords = "absolute" if args.absolute else ("window" if args.window else "gene")
    infile = sys.stdin if args.input == "-" else open(args.input)
    outfile = open(args.output, "w") if args.output else sys.stdout
    try:
        n, skipped = gff2gsf(infile, outfile, coords)
        print(f"{n} genes written, {skipped} skipped", file=sys.stderr)
    finally:
        if args.input != "-":
            infile.close()
        if args.output:
            outfile.close()


if __name__ == "__main__":
    main()
