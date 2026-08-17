#!/usr/bin/env python3
"""Resolve the AtRTD3 provenance of the showcase loci in panelC_prompted.tsv.

25_scan_panelC_prompted.py reports, per locus, how many AtRTD3 transcripts share the
predicted CDS intron chain and how many carry the junction TAIR10 lacks, but not which
source those transcripts came from. AtRTD3 merges AtIso Iso-Seq, AtRTD2 short-read
assemblies and Araport11; only the first is long-read evidence. Origins come from
field 4 of atRTD3_07082020.bed (gene;transcript;origin), retrieved from
https://ics.hutton.ac.uk/atRTD/RTD3/atRTD3_07082020.bed.

The chain construction is copied from script 25 so the transcript sets recovered here
are the same ones its counts describe.

Usage:
    python 47_panelC_provenance.py [--out panelC_provenance.tsv]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRED = (ROOT / "transgenic_comparison" / "standardized_results"
        / "A_thaliana_transgenic400Mprompt_beam1.gff3")
TAIR = ROOT / "transgenic" / "revision" / "data" / "TAIR10" / "TAIR10.gtf"
ATRTD = (ROOT / "transgenic" / "revision" / "data" / "AtRTD3"
         / "atRTD3_TS_21Feb22_transfix.gtf")
BED = (ROOT / "transgenic" / "revision" / "data" / "AtRTD3" / "atiso"
       / "atRTD3_07082020.bed")
PANELC = (ROOT / "transgenic" / "revision" / "results" / "fig4_forensics"
          / "panelC_examples" / "prompted_full" / "panelC_prompted.tsv")
OUT = (ROOT / "transgenic" / "revision" / "results" / "fig4_forensics"
       / "panelC_examples" / "prompted_full" / "panelC_provenance.tsv")


def _cds_by_transcript(path: Path, gff3: bool) -> dict:
    out: dict = defaultdict(lambda: defaultdict(list))
    if gff3:
        locus_re, tx_re = re.compile(r"GM=([^;\s]+)"), re.compile(r"Parent=([^;]+)")
    else:
        locus_re = re.compile(r'gene_id "([^"]+)"')
        tx_re = re.compile(r'transcript_id "([^"]+)"')
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = locus_re.search(f[8]), tx_re.search(f[8])
            if not (g and t):
                continue
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            out[locus.replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    return out


def intron_chains(path: Path, gff3: bool) -> dict:
    chains: dict = {}
    for locus, txs in _cds_by_transcript(path, gff3).items():
        chains[locus] = {}
        for tx, segs in txs.items():
            s = sorted(segs)
            chains[locus][tx] = tuple((s[i][1], s[i + 1][0]) for i in range(len(s) - 1))
    return chains


def load_origin(path: Path) -> dict:
    origin = {}
    with path.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            parts = f[3].split(";")
            if len(parts) >= 3:
                origin[parts[1]] = parts[2]
    return origin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    pred = intron_chains(PRED, gff3=True)
    tair = intron_chains(TAIR, gff3=False)
    art = intron_chains(ATRTD, gff3=False)
    origin = load_origin(BED)

    rows = []
    with PANELC.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))

    out_rows = []
    unmapped = []
    for r in rows:
        locus, tx = r["locus"], r["predicted_transcript"]
        chain = pred[locus][tx]
        art_here = art.get(locus, {})
        tair_junctions = {j for c in tair.get(locus, {}).values() for j in c}
        novel = [j for j in chain if j not in tair_junctions]

        sharing = sorted(k for k, v in art_here.items() if v == chain)
        carrying = sorted(k for k, v in art_here.items()
                          if novel and all(j in v for j in novel))
        sh_org = [origin.get(t, "UNMAPPED") for t in sharing]
        ca_org = [origin.get(t, "UNMAPPED") for t in carrying]
        unmapped.extend(t for t, o in zip(sharing + carrying, sh_org + ca_org)
                        if o == "UNMAPPED")

        out_rows.append({
            "locus": locus,
            "category": r["category"],
            "predicted_transcript": tx,
            "n_sharing_chain": len(sharing),
            "n_sharing_isoseq": sum(1 for o in sh_org if o == "Isoseq"),
            "sharing_origins": ",".join(sorted(set(sh_org))) or "NA",
            "chain_has_isoseq": any(o == "Isoseq" for o in sh_org),
            "chain_all_isoseq": bool(sh_org) and all(o == "Isoseq" for o in sh_org),
            "n_carrying_novel_junction": len(carrying),
            "n_carrying_isoseq": sum(1 for o in ca_org if o == "Isoseq"),
            "carrying_origins": ",".join(sorted(set(ca_org))) or "NA",
            "sharing_transcripts": ",".join(sharing),
        })

    n = len(out_rows)
    any_iso = sum(1 for r in out_rows if r["chain_has_isoseq"])
    all_iso = sum(1 for r in out_rows if r["chain_all_isoseq"])
    junction_rows = [r for r in out_rows if r["category"] == "junction"]
    junction_iso = sum(1 for r in junction_rows if r["chain_has_isoseq"])
    tx_origins = Counter(o for r in out_rows for o in r["sharing_origins"].split(","))

    summary = {
        "loci": n,
        "loci_with_an_IsoSeq_transcript_sharing_the_chain": any_iso,
        "loci_where_all_chain_sharing_transcripts_are_IsoSeq": all_iso,
        "pct_loci_with_IsoSeq_chain_support": round(100 * any_iso / n, 1) if n else None,
        "junction_category_loci": len(junction_rows),
        "junction_category_loci_with_IsoSeq_chain_support": junction_iso,
        "sharing_origin_label_counts": dict(tx_origins),
        "unmapped_transcript_ids": sorted(set(unmapped)),
    }
    for k, v in summary.items():
        print(f"  {k:<54} {v}", file=sys.stderr)

    cols = ["locus", "category", "predicted_transcript", "n_sharing_chain",
            "n_sharing_isoseq", "sharing_origins", "chain_has_isoseq",
            "chain_all_isoseq", "n_carrying_novel_junction", "n_carrying_isoseq",
            "carrying_origins", "sharing_transcripts"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in out_rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"written: {args.out}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
