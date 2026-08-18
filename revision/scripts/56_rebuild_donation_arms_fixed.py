#!/usr/bin/env python3
"""Rebuild the two UTR-donation arms with the minus-strand fix, under new names.

WHAT WAS WRONG

`helixertairutr` and `annevotairutr` hand a tool's own CDS the reference's UTR, to decide
whether UTR correction alone would close the prompt-transfer gap. Both were built by
`35_build_frame_shift_prompts.py:write_helixer_with_tair_utr`, which took the 5'/3' lists
`utr_segments` labels BY STRAND and filtered them BY COORDINATE — keeping a 5' interval only
where it started below the CDS and a 3' interval only where it ended above it. On the minus
strand `utr_segments` puts the 5' UTR on the high-coordinate side, so both tests rejected
every interval and the locus was written with no UTR rows at all:

    helixertairutr   plus 10,001 / 11,972 loci got a UTR      minus     35 / 11,989
    annevotairutr    plus  8,807 / 10,431 loci got a UTR      minus      2 / 10,412

Roughly half of each file was therefore in the no-UTR condition rather than the
reference-UTR condition it was built to measure, and both arms understated that cell.
Rescoring the plus-strand half alone gives 2.2% (Helixer) and 3.5% (ANNEVO) against the
0.9% and 1.2% first reported, with the `tair10selfutr` control flat across the same split
(18.1% plus / 18.4% minus) — so the split itself is not what moves the number.

WHAT THIS SCRIPT DOES

Reproduces `35`'s arm G and arm H construction exactly — same pairing, same locus order,
same donation rule — with the corrected `place_donated_utr`, and writes them as
`helixertairutr_fixed` and `annevotairutr_fixed`. The buggy inputs are left untouched: the
predictions already made from them are still valid measurements of what they actually
contained, and their provenance pins their md5s.

The plus-strand rescore above predicts what these runs should return. It is a genuine
prediction and not a certainty — it assumes the minus strand behaves like the plus strand
once it receives a UTR — so the two numbers are worth comparing rather than assuming.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "revision" / "scripts"
BENCH = ROOT.parent / "polishing_benchmark"


def load_builder():
    path = SCRIPTS / "35_build_frame_shift_prompts.py"
    spec = importlib.util.spec_from_file_location("build_frame_shift_prompts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(tair10_gtf: Path, primary_ids_path: Path, helixer_gff: Path, annevo_gff: Path,
          out_dir: Path) -> dict:
    m = load_builder()
    primary_ids = m.load_primary_ids(primary_ids_path)
    tair = m.load_tair10(tair10_gtf, primary_ids)
    with_utr = m.load_tair10_with_utr(tair10_gtf, primary_ids)
    stats: dict = {}

    # --- arm G: Helixer's CDS, TAIR10's UTR -------------------------------------------
    rows, pairing = m.pair_and_filter(tair, m.load_helixer_genes(helixer_gff))
    donor_by_helixer = {}
    for row in rows:
        info = with_utr.get(row["gene"])
        if not info or not info["exons"]:
            continue
        five, three = m.utr_segments(info["exons"], info["cds"], row["strand"])
        donor_by_helixer[row["helixer_gene"]] = {"five": five, "three": three}
    g = out_dir / "helixertairutr_fixed_Athaliana.gff3"
    stats["arm_G_pairing"] = pairing
    stats["arm_G"] = m.write_helixer_with_tair_utr(
        m.load_helixer_first_cds(helixer_gff), donor_by_helixer, g)
    stats["arm_G_path"] = str(g)

    # --- arm H: ANNEVO's CDS, TAIR10's UTR --------------------------------------------
    annevo_rows, annevo_pairing = m.pair_and_filter(dict(with_utr),
                                                     m.load_helixer_genes(annevo_gff))
    donor_by_annevo = {}
    for row in annevo_rows:
        info = with_utr.get(row["gene"])
        if not info or not info["exons"]:
            continue
        five, three = m.utr_segments(info["exons"], info["cds"], row["strand"])
        donor_by_annevo[row["helixer_gene"]] = {"five": five, "three": three}
    h = out_dir / "annevotairutr_fixed_Athaliana.gff3"
    stats["arm_H_pairing"] = annevo_pairing
    stats["arm_H"] = m.write_helixer_with_tair_utr(
        m.load_helixer_first_cds(annevo_gff), donor_by_annevo, h)
    stats["arm_H_path"] = str(h)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tair10-gtf", type=Path,
                    default=ROOT / "revision" / "data" / "TAIR10" / "TAIR10.gtf")
    ap.add_argument("--primary-ids", type=Path,
                    default=ROOT / "revision" / "data" / "TAIR10" / "primary_transcript_ids.txt")
    ap.add_argument("--helixer-gff", type=Path, default=BENCH / "inputs" / "helixer_Athaliana.gff3")
    ap.add_argument("--annevo-gff", type=Path, default=BENCH / "inputs" / "annevo_Athaliana.gff3")
    ap.add_argument("--out-dir", type=Path, default=BENCH / "inputs")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    for name in ("helixertairutr_fixed", "annevotairutr_fixed"):
        path = args.out_dir / f"{name}_Athaliana.gff3"
        if path.exists():
            raise SystemExit(f"{path} already exists — refusing to overwrite a staged input")

    stats = build(args.tair10_gtf, args.primary_ids, args.helixer_gff, args.annevo_gff,
                  args.out_dir)

    # The fix is the whole point of this rebuild, so verify it rather than assume it: a
    # donation that still fails on one strand must not be quietly staged for a GPU run.
    for arm in ("arm_G", "arm_H"):
        by_strand = stats[arm]["loci_with_utr_by_strand"]
        total = stats[arm]["loci_by_strand"]
        for strand in ("+", "-"):
            got, n = by_strand.get(strand, 0), total.get(strand, 0)
            if n and got / n < 0.5:
                raise SystemExit(
                    f"{arm}: only {got}/{n} loci on strand {strand} received a UTR — the "
                    f"minus-strand fix has not taken effect; refusing to stage this input"
                )
    print(json.dumps(stats, indent=2))
    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
