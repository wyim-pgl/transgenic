#!/usr/bin/env python3
"""Count what untranslated-region evidence each prompt annotation actually carries.

The Discussion states that Helixer emits untranslated regions while ANNEVO emits none, its
exon and CDS coordinates being identical. Both halves of that claim were measured by hand
and never deposited, so this script is the artefact behind them. It reads only the prompt
annotations in `polishing_benchmark/inputs/` and consults no reference, so it says what a
tool supplies, not whether the supply is right — `37_analyse_prompt_requirements.py` answers
the second question.

Two counts are reported per annotation because the manuscript needs both and they differ:

    loci_emitting_utr            loci with at least one UTR row anywhere in the locus
    prompted_transcripts_with_utr   loci whose FIRST transcript carries one, the first
                                 transcript being what completion mode is actually prompted
                                 with (`load_prompt` in 37 and `supplied_structures` in 36
                                 both take it that way)

An annotation that stores no UTR rows may still imply untranslated regions through exons
that extend past the CDS, so exon-minus-CDS is measured too. When every exon row is
coordinate-identical to a CDS row of the same transcript, that route is closed as well and
the annotation carries no untranslated evidence in either form.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_PARENT = re.compile(r"Parent=([^;]+)")
_ID = re.compile(r"ID=([^;]+)")


def audit(gff: Path) -> dict:
    rows: dict = defaultdict(int)
    first_of_gene: dict = {}
    gene_of_tx: dict = {}
    cds: dict = defaultdict(list)
    exon: dict = defaultdict(list)
    utr: dict = defaultdict(int)

    with gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind = f[2]
            rows[kind] += 1
            if kind == "gene":
                first_of_gene.setdefault(f[8].split(";")[0].split("=")[1], None)
                continue
            if kind in ("mRNA", "transcript"):
                tid, parent = _ID.search(f[8]), _PARENT.search(f[8])
                if tid and parent:
                    gene_of_tx[tid.group(1)] = parent.group(1)
                    if first_of_gene.get(parent.group(1)) is None:
                        first_of_gene[parent.group(1)] = tid.group(1)
                continue
            parent = _PARENT.search(f[8])
            if not parent:
                continue
            tx, segment = parent.group(1), (f[0], int(f[3]), int(f[4]), f[6])
            if kind == "CDS":
                cds[tx].append(segment)
            elif kind == "exon":
                exon[tx].append(segment)
            elif kind.endswith("UTR"):
                utr[tx] += 1

    if not first_of_gene:
        raise ValueError(f"no gene rows in {gff} — refusing to report counts from a file "
                         "this parser did not understand")

    identical_tx = identical_rows = 0
    exon_rows_beyond_cds = 0
    for tx in set(cds) | set(exon):
        c, e = sorted(cds.get(tx, [])), sorted(exon.get(tx, []))
        if c and c == e:
            identical_tx += 1
            identical_rows += len(c)
        exon_rows_beyond_cds += len(set(e) - set(c))

    utr_rows = sum(v for k, v in rows.items() if k.endswith("UTR"))
    loci_with_utr = {gene_of_tx[tx] for tx in utr if tx in gene_of_tx}
    prompted_with_utr = sum(1 for tx in first_of_gene.values() if tx and utr.get(tx))

    return {
        "annotation": str(gff),
        "rows": dict(sorted(rows.items())),
        "loci": len(first_of_gene),
        "transcripts": len(gene_of_tx),
        "utr_rows": utr_rows,
        "loci_emitting_utr": len(loci_with_utr),
        "loci_emitting_utr_pct": round(100 * len(loci_with_utr) / len(first_of_gene), 1),
        "prompted_transcripts_with_utr": prompted_with_utr,
        "prompted_transcripts_with_utr_pct": round(
            100 * prompted_with_utr / len(first_of_gene), 1),
        "transcripts_with_exon_rows_identical_to_cds": identical_tx,
        "rows_where_exon_equals_cds": identical_rows,
        "exon_rows_extending_beyond_cds": exon_rows_beyond_cds,
        "carries_no_untranslated_evidence": utr_rows == 0 and exon_rows_beyond_cds == 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, nargs="+", required=True,
                    help="one or more prompt annotations")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = {"annotations": [audit(path) for path in args.input]}
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
