#!/usr/bin/env python3
"""Resolve the AtRTD3 provenance of the additions that AtRTD3 supports.

Script 28 counts additions whose CDS structure equals some AtRTD3 transcript at the
same locus (204 of 1,103 = 18.5%) but does not record *which* AtRTD3 transcript
matched. AtRTD3 merges three sources -- AtIso Iso-Seq, AtRTD2 short-read assemblies
and Araport11 -- so "AtRTD3 supports it" is only long-read evidence when the matching
transcript came from AtIso. The origin is published per transcript in field 4 of
atRTD3_07082020.bed (gene;transcript;origin); the distributed GTF carries a uniform
"PBRI" source column and cannot answer this. The bed is retrieved from
https://ics.hutton.ac.uk/atRTD/RTD3/atRTD3_07082020.bed and its origin labels reproduce
the published composition exactly: 132,166 Isoseq, 24,831 atRTD2, 12,506 Araport11.

The matching logic below is copied from 28_score_added_isoforms.py so the counts it
reproduces are the manuscript's own.

Usage:
    python 46_atrtd3_provenance.py [--species A_thaliana] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"
BED = DATA / "AtRTD3" / "atiso" / "atRTD3_07082020.bed"


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


def load_origin(path: Path) -> dict:
    """transcript_id -> origin, from field 4 of the published AtRTD3 bed."""
    origin = {}
    with path.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            parts = f[3].split(";")
            if len(parts) < 3:
                continue
            origin[parts[1]] = parts[2]
    return origin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="A_thaliana")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--tsv", type=Path)
    args = ap.parse_args()

    pred = cds_by_transcript(
        CMP / "standardized_results" / f"{args.species}_transgenic400Mprompt_beam1.gff3", True)
    ref = cds_by_transcript(DATA / "TAIR10" / "TAIR10.gtf", False)
    art = cds_by_transcript(DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf", False)
    origin = load_origin(BED)

    primary = {}
    for line in (DATA / "TAIR10" / "primary_transcript_ids.txt").read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    added = art_hit = 0
    rows = []
    unmapped = []
    for locus, txs in pred.items():
        supplied = ref.get(locus, {}).get(primary.get(locus))
        additions = list({s for s in txs.values() if s != supplied})
        added += len(additions)

        # CDS structure -> AtRTD3 transcript ids sharing it at this locus
        struct2ids = defaultdict(list)
        for tid, struct in art.get(locus, {}).items():
            struct2ids[struct].append(tid)

        for s in additions:
            if s not in struct2ids:
                continue
            art_hit += 1
            ids = sorted(struct2ids[s])
            origins = [origin.get(t, "UNMAPPED") for t in ids]
            unmapped.extend(t for t, o in zip(ids, origins) if o == "UNMAPPED")
            rows.append({
                "locus": locus,
                "n_cds": len(s),
                "atrtd3_transcripts": ids,
                "origins": origins,
                "any_isoseq": any(o == "Isoseq" for o in origins),
                "all_isoseq": all(o == "Isoseq" for o in origins),
            })

    any_iso = sum(1 for r in rows if r["any_isoseq"])
    all_iso = sum(1 for r in rows if r["all_isoseq"])
    # Origin of the matched transcripts themselves (an addition may match several).
    tx_origins = Counter(o for r in rows for o in r["origins"])
    # Per-addition label when no matching transcript is Iso-Seq.
    non_iso_labels = Counter(
        ",".join(sorted(set(r["origins"]))) for r in rows if not r["any_isoseq"])

    out = {
        "species": args.species,
        "tool": "transgenic400Mprompt_beam1",
        "provenance_source": str(BED),
        "added_transcripts": added,
        "added_matching_any_AtRTD3_transcript": art_hit,
        "additions_with_an_IsoSeq_supporting_transcript": any_iso,
        "additions_where_all_supporting_transcripts_are_IsoSeq": all_iso,
        "pct_of_AtRTD3_supported_additions_with_IsoSeq": round(100 * any_iso / art_hit, 1) if art_hit else None,
        "pct_of_all_additions_with_IsoSeq_support": round(100 * any_iso / added, 1) if added else None,
        "matched_transcript_origin_counts": dict(tx_origins),
        "non_IsoSeq_addition_origin_labels": dict(non_iso_labels),
        "unmapped_transcript_ids": sorted(set(unmapped)),
    }
    for k, v in out.items():
        print(f"  {k:<52} {v}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    if args.tsv:
        with args.tsv.open("w") as fh:
            fh.write("locus\tn_cds_segments\tatrtd3_transcripts\torigins\tany_isoseq\tall_isoseq\n")
            for r in sorted(rows, key=lambda x: x["locus"]):
                fh.write("\t".join([
                    r["locus"], str(r["n_cds"]),
                    ",".join(r["atrtd3_transcripts"]), ",".join(r["origins"]),
                    str(r["any_isoseq"]), str(r["all_isoseq"])]) + "\n")
        print(f"written: {args.tsv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
