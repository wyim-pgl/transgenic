#!/usr/bin/env python3
"""Normalize busco_summary_final.csv.

The summarize step split each BUSCO run directory name on the *last* underscore,
so tool names that themselves contain an underscore (`tiberius_softmasked`,
`transgenic{160,400}M_prompt_denovo`) leaked their prefix into the Species
column, e.g.

    A_thaliana_tiberius,softmasked,98.8,...
    A_thaliana_transgenic400M_prompt,denovo,74.6,...

This script rejoins the two columns and re-splits them against the known tool
vocabulary, writing a corrected CSV. Numbers are untouched.

Usage:
    python 18_normalize_busco_summary.py [--in CSV] [--out CSV]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Longest first so that `transgenic400M_prompt_denovo` wins over `transgenic400M`.
TOOLS = sorted(
    [
        "annevo",
        "helixer",
        "tiberius",
        "tiberius_softmasked",
        "transgenic160M",
        "transgenic160Mprompt",
        "transgenic160M_prompt_denovo",
        "transgenic400M",
        "transgenic400Mprompt",
        "transgenic400M_prompt_denovo",
    ],
    key=len,
    reverse=True,
)

SPECIES = [
    "A_thaliana",
    "B_distachyon",
    "B_rapa",
    "G_max",
    "L_sativa",
    "O_sativa",
    "P_patens",
    "P_trichocarpa",
    "S_bicolor",
    "S_italica",
    "S_lycopersicum",
    "V_vinifera",
    "Z_mays",
]


def split_key(key: str) -> tuple[str, str]:
    """Split `Species_tool` into its two parts using the known vocabularies."""
    for tool in TOOLS:
        suffix = "_" + tool
        if key.endswith(suffix):
            species = key[: -len(suffix)]
            if species in SPECIES:
                return species, tool
    raise ValueError(f"cannot split {key!r} into a known species and tool")


def main() -> int:
    here = Path(__file__).resolve()
    default_in = here.parents[3] / "transgenic_comparison" / "busco_summary_final.csv"
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=default_in)
    ap.add_argument("--out", dest="dst", type=Path, default=None)
    args = ap.parse_args()

    dst = args.dst or args.src.with_name("busco_summary_final.normalized.csv")

    with args.src.open(newline="") as fh:
        rows = list(csv.reader(fh))

    header, body = rows[0], rows[1:]
    fixed = 0
    out_rows = []
    for row in body:
        key = row[0] + "_" + row[1] if row[0] not in SPECIES else row[0]
        if row[0] in SPECIES:
            species, tool = row[0], row[1]
        else:
            species, tool = split_key(key)
            fixed += 1
        out_rows.append([species, tool, *row[2:]])

    out_rows.sort(key=lambda r: (SPECIES.index(r[0]), r[1]))

    with dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"read   {len(body)} rows from {args.src}")
    print(f"fixed  {fixed} rows with a leaked tool prefix in the Species column")
    print(f"wrote  {len(out_rows)} rows to {dst}")

    combos = {(r[0], r[1]) for r in out_rows}
    missing = [
        (s, t) for s in SPECIES for t in TOOLS if (s, t) not in combos
    ]
    if missing:
        print("\nmissing species x tool cells:")
        for s, t in sorted(missing):
            print(f"  {s:16s} {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
