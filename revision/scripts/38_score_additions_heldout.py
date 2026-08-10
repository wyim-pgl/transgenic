#!/usr/bin/env python3
"""Additions-only precision on an arbitrary prompted prediction file.

Definitions are copied verbatim from revision/scripts/28_score_added_isoforms.py so that
the control run over the full 27,413-locus standardized prompt file reproduces its JSON.
The only change is that the prediction path is a command-line argument, and an optional
second definition of the supplied prompt (first transcript of a labels GFF3) can be used
as a sensitivity check.

Usage:
    python score_heldout_additions.py --pred <pred.gff3> [--labels <labels.gff3>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/gpfs/assoc/pgl/data/Transgenic")
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"


def cds_by_transcript(path: Path, gff3: bool) -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    lr = re.compile(r"GM=([^;\s]+)") if gff3 else re.compile(r'gene_id "([^"]+)"')
    tr = re.compile(r"Parent=([^;]+)") if gff3 else re.compile(r'transcript_id "([^"]+)"')
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = lr.search(f[8]), tr.search(f[8])
            if not (g and t):
                continue
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            d[locus.replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    return {g: {k: tuple(sorted(v)) for k, v in tx.items()} for g, tx in d.items()}


def chain(struct: tuple) -> tuple:
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def tx_index(name: str) -> int:
    m = re.search(r"\.t(\d+)$", name)
    return int(m.group(1)) if m else 0


def score(pred: dict, ref: dict, art: dict, primary: dict, supplied_from: dict | None,
          tag: str) -> dict:
    """supplied_from: locus -> structure to treat as the prompt. None = TAIR10 primary."""
    added = struct_hit = chain_hit = art_hit = 0
    alt_total = alt_recovered = 0
    loci_with_addition = 0
    loci_where_prompt_structure_present = 0
    loci_with_supplied_defined = 0
    for locus, txs in pred.items():
        if supplied_from is None:
            supplied = ref.get(locus, {}).get(primary.get(locus))
        else:
            supplied = supplied_from.get(locus)
        if supplied is not None:
            loci_with_supplied_defined += 1
            if supplied in set(txs.values()):
                loci_where_prompt_structure_present += 1
        additions = [s for s in txs.values() if s != supplied]
        additions = list({s for s in additions})
        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        art_here = set(art.get(locus, {}).values())
        alt_total += len(alt_ref)
        if additions:
            loci_with_addition += 1
        added += len(additions)
        struct_hit += sum(1 for s in additions if s in alt_ref)
        chain_hit += sum(1 for s in additions
                         if len(chain(s)) >= 1 and chain(s) in {chain(x) for x in alt_ref})
        art_hit += sum(1 for s in additions if s in art_here)
        alt_recovered += len(alt_ref & set(additions))

    return {
        "definition_of_supplied_prompt": tag,
        "loci_scored": len(pred),
        "loci_with_supplied_prompt_defined": loci_with_supplied_defined,
        "loci_where_supplied_structure_is_among_predictions": loci_where_prompt_structure_present,
        "loci_with_at_least_one_addition": loci_with_addition,
        "added_transcripts": added,
        "reference_alternative_transcripts": alt_total,
        "added_matching_TAIR10_alternative_exact_CDS": struct_hit,
        "added_matching_TAIR10_alternative_intron_chain": chain_hit,
        "added_matching_any_AtRTD3_transcript": art_hit,
        "precision_vs_TAIR10_alternatives_pct": round(100 * struct_hit / added, 1) if added else None,
        "precision_vs_AtRTD3_pct": round(100 * art_hit / added, 1) if added else None,
        "recall_of_TAIR10_alternatives_pct": round(100 * alt_recovered / alt_total, 1) if alt_total else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--labels", type=Path,
                    help="labels GFF3 for the same loci; its first transcript per locus is "
                         "used as an alternative definition of the supplied prompt")
    ap.add_argument("--name", default="")
    ap.add_argument("--restrict", type=Path,
                    help="file of locus IDs (AT1G01010 form); keep only these loci")
    ap.add_argument("--exclude", type=Path,
                    help="file of locus IDs; drop these loci")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    pred = cds_by_transcript(args.pred, True)
    if args.restrict:
        keep = {l.strip() for l in args.restrict.read_text().splitlines() if l.strip()}
        pred = {g: t for g, t in pred.items() if g in keep}
    if args.exclude:
        drop = {l.strip() for l in args.exclude.read_text().splitlines() if l.strip()}
        pred = {g: t for g, t in pred.items() if g not in drop}
    ref = cds_by_transcript(DATA / "TAIR10" / "TAIR10.gtf", False)
    art = cds_by_transcript(DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf", False)
    primary = {}
    for line in (DATA / "TAIR10" / "primary_transcript_ids.txt").read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    results = [score(pred, ref, art, primary, None, "TAIR10 primary_transcript_ids.txt (script 28)")]

    if args.labels:
        lab = cds_by_transcript(args.labels, True)
        first = {}
        for locus, txs in lab.items():
            order = sorted(txs, key=tx_index)
            if order:
                first[locus] = txs[order[0]]
        results.append(score(pred, ref, art, primary, first,
                             "first transcript of the labels GFF3 for the same locus"))
        # how often the two definitions agree
        agree = sum(1 for locus in pred
                    if locus in first and first[locus] == ref.get(locus, {}).get(primary.get(locus)))
        results[-1]["loci_where_labels_first_tx_equals_TAIR10_primary"] = agree

    out = {"name": args.name or args.pred.name, "prediction_file": str(args.pred),
           "labels_file": str(args.labels) if args.labels else None, "scorings": results}
    print(json.dumps(out, indent=1), file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
