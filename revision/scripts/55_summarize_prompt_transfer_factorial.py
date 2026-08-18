#!/usr/bin/env python3
"""Collect the prompt-transfer factorial into one table, with the strand correction applied.

Two things this reports that the per-arm JSONs do not:

1. The `helixertairutr` / `annevotairutr` arms donated their UTR to the plus strand only.
   `35_build_frame_shift_prompts.py:write_helixer_with_tair_utr` filters strand-labelled
   5'/3' lists by raw coordinate, and on the minus strand both tests reject every interval:
   35 of 11,989 minus-strand Helixer loci received a UTR, and 2 of 10,412 for ANNEVO. Their
   genome-wide numbers are therefore a ~50:50 mixture of the cell they were meant to measure
   and the no-UTR cell. The plus-strand-restricted rescore is the one to quote; the
   `tair10selfutr` split is carried alongside it as the control that shows the restriction
   is itself neutral.

2. The fourth cell — reference CDS carrying a predicted UTR (`tair10cdshelixerutr`, built by
   `53_build_tair10cds_helixerutr_arm.py`) — which had never been run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT.parent / "polishing_benchmark"
DEPOSIT = ROOT / "revision" / "results" / "prompt_transfer"

# (label, CDS source, UTR source, result file, note)
CELLS = [
    ("tair10selfutr", "TAIR10", "TAIR10", "tair10selfutr", "the 18% cell"),
    ("tair10self", "TAIR10", "none", "tair10self", "no UTR rows at all"),
    ("tair10cdshelixerutr", "TAIR10", "Helixer", "tair10cdshelixerutr", "the missing cell"),
    ("helixertairutr", "Helixer", "TAIR10", "helixertairutr",
     "genome-wide; UNDERSTATED, minus strand got no UTR"),
    ("annevotairutr", "ANNEVO", "TAIR10", "annevotairutr",
     "genome-wide; UNDERSTATED, minus strand got no UTR"),
    ("helixertairutr (plus only)", "Helixer", "TAIR10", "helixertairutr_plus",
     "plus-strand rescore of the buggy build"),
    ("annevotairutr (plus only)", "ANNEVO", "TAIR10", "annevotairutr_plus",
     "plus-strand rescore of the buggy build"),
    ("helixertairutr_fixed", "Helixer", "TAIR10", "helixertairutr_fixed",
     "strand-corrected rebuild + re-run: quote this one"),
    ("annevotairutr_fixed", "ANNEVO", "TAIR10", "annevotairutr_fixed",
     "strand-corrected rebuild + re-run: quote this one"),
    ("tair10selfutr (plus only)", "TAIR10", "TAIR10", "tair10selfutr_plus",
     "control: restriction is neutral"),
    ("helixer", "Helixer", "Helixer", "helixer", "the tool's own annotation"),
    ("annevo", "ANNEVO", "ANNEVO", "annevo", "the tool's own annotation"),
]

FIELDS = ("loci_scored", "added_structures", "precision_vs_TAIR10_alternatives_pct",
          "precision_vs_AtRTD3_pct", "recall_of_TAIR10_alternatives_pct")


def find(stub: str, extra: list[Path]) -> dict | None:
    for directory in [BENCH / "results", DEPOSIT, *extra]:
        path = directory / f"{stub}_as_additions.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extra-dir", type=Path, nargs="*", default=[],
                    help="additional directories to search for {stub}_as_additions.json")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    table = []
    print(f"{'arm':30s} {'CDS':8s} {'UTR':8s} {'loci':>7s} {'added':>7s} "
          f"{'alt%':>6s} {'RTD3%':>6s} {'rec%':>5s}  note")
    for label, cds, utr, stub, note in CELLS:
        data = find(stub, args.extra_dir)
        if data is None:
            print(f"{label:30s} {cds:8s} {utr:8s} {'-':>7s} {'-':>7s} "
                  f"{'-':>6s} {'-':>6s} {'-':>5s}  NOT RUN")
            continue
        row = {"arm": label, "cds_source": cds, "utr_source": utr, "note": note,
               **{k: data[k] for k in FIELDS}}
        table.append(row)
        print(f"{label:30s} {cds:8s} {utr:8s} {data['loci_scored']:7,} "
              f"{data['added_structures']:7,} "
              f"{data['precision_vs_TAIR10_alternatives_pct']:6} "
              f"{data['precision_vs_AtRTD3_pct']:6} "
              f"{data['recall_of_TAIR10_alternatives_pct']:5}  {note}")

    # The filtered (structurally valid) counterparts, where they exist.
    print()
    print("structurally filtered (36 -> 34):")
    filtered = []
    for label, cds, utr, stub, _ in CELLS:
        data = find(f"{stub}_filtered", args.extra_dir)
        if data is None:
            continue
        filtered.append({"arm": label, **{k: data[k] for k in FIELDS}})
        print(f"{label:30s} added={data['added_structures']:6,} "
              f"alt={data['precision_vs_TAIR10_alternatives_pct']}% "
              f"RTD3={data['precision_vs_AtRTD3_pct']}% "
              f"recall={data['recall_of_TAIR10_alternatives_pct']}%")

    if args.json:
        args.json.write_text(json.dumps(
            {"unfiltered": table, "filtered": filtered,
             "strand_bug": {
                 "function": "35_build_frame_shift_prompts.py:write_helixer_with_tair_utr",
                 "description": "strand-labelled 5'/3' UTR lists filtered by raw coordinate; "
                                "on the minus strand every interval is rejected",
                 "helixertairutr_minus_loci_with_utr": "35 / 11989",
                 "annevotairutr_minus_loci_with_utr": "2 / 10412",
             }}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
