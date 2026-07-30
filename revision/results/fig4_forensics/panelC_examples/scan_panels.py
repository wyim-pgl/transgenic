#!/usr/bin/env python3
"""Classify every A. thaliana prompted locus into the three Figure-4 panel types.

Panels, as the manuscript legend defines them:

  A  reproduced the alternative transcripts shared by TAIR10 and AtRTD3
  B  recovered additional TAIR10 isoforms supported by AtRTD3
  C  predicted an alternatively spliced isoform supported by AtRTD3 but absent from TAIR10

A and B overlap in the legend's wording, so they are separated here by how much the model
recovered: A is the clean two-chain case (AT1G43770, the forensically anchored panel), B is
the richer case where three or more distinct TAIR10 chains come back, which is what
"additional isoforms" reads as next to A.

Everything is judged on the CDS intron chain - the ordered tuple of intron coordinates -
because that is what defines a splice isoform. UTR-level matching was what put AT1G43770 in
the wrong panel in the first forensic pass.

Usage:
  python3 scan_panels.py <labels.gff3> <prediction.gff3> <AtRTD3.gtf> [--out panels.tsv]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict


def cds_chain(exons: list) -> tuple:
    e = sorted(exons)
    return tuple((e[i][1] + 1, e[i + 1][0] - 1) for i in range(len(e) - 1))


def parse_gff3(path: str) -> dict:
    """{locus: {transcript: [(s,e)…]}} keyed on the GM= attribute."""
    tx = defaultdict(list)
    loc_of, span_of, strand_of, chr_of = {}, {}, {}, {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            p = re.search(r"Parent=([^;]+)", f[8])
            gm = re.search(r"GM=([^;]+)", f[8])
            if not p:
                continue
            key = p.group(1)
            tx[key].append((int(f[3]), int(f[4])))
            if gm:
                loc_of[key] = gm.group(1).replace(".TAIR10", "").replace("-rc", "")
            strand_of[key], chr_of[key] = f[6], f[0]
    by_locus = defaultdict(dict)
    for k, exons in tx.items():
        loc = loc_of.get(k, "?")
        if loc == "?":
            continue
        by_locus[loc][k] = exons
        xs = [x for seg in exons for x in seg]
        span_of[loc] = (min(xs), max(xs)) if loc not in span_of else (
            min(span_of[loc][0], min(xs)), max(span_of[loc][1], max(xs)))
    return by_locus, span_of, strand_of, chr_of


def parse_gtf(path: str) -> dict:
    tx = defaultdict(list)
    loc_of = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            t = re.search(r'transcript_id "([^"]+)"', f[8])
            g = re.search(r'gene_id "([^"]+)"', f[8])
            if not (t and g):
                continue
            tx[t.group(1)].append((int(f[3]), int(f[4])))
            loc_of[t.group(1)] = g.group(1)
    by_locus = defaultdict(dict)
    for k, exons in tx.items():
        by_locus[loc_of[k]][k] = exons
    return by_locus


def chains(d: dict) -> set:
    return {cds_chain(v) for v in d.values() if len(v) >= 2}


def main() -> int:
    labels, pred, atrtd = sys.argv[1], sys.argv[2], sys.argv[3]
    out = "panels.tsv"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    print("parsing…", file=sys.stderr)
    T, span, strand, chrom = parse_gff3(labels)
    P, _, _, _ = parse_gff3(pred)
    A = parse_gtf(atrtd)
    print(f"  TAIR10 {len(T)} loci | prediction {len(P)} loci | AtRTD3 {len(A)} loci",
          file=sys.stderr)

    rows = []
    for loc, ptx in P.items():
        tset, pset, aset = chains(T.get(loc, {})), chains(ptx), chains(A.get(loc, {}))
        if not tset or not pset:
            continue
        reproduced = pset & tset          # TAIR10 chains the model got back
        shared = reproduced & aset        # …that AtRTD3 also documents
        novel = (pset & aset) - tset      # AtRTD3-supported, absent from TAIR10
        unsupported = pset - tset - aset  # predicted with no support anywhere

        if novel:
            panel = "C"
        elif len(reproduced) >= 3 and len(shared) >= 2 and not unsupported:
            panel = "B"
        elif len(reproduced) == 2 and len(shared) >= 2 and not unsupported:
            panel = "A"
        else:
            continue

        s0, s1 = span.get(loc, (0, 0))
        rows.append(dict(locus=loc, panel=panel, chrom=chrom.get(
            next(iter(ptx)), "?"), strand=strand.get(next(iter(ptx)), "?"),
            kb=round((s1 - s0) / 1000, 1), n_tair=len(tset), n_pred=len(pset),
            n_art=len(aset), reproduced=len(reproduced), shared=len(shared),
            novel=len(novel), unsupported=len(unsupported),
            max_introns=max(len(c) for c in pset)))

    rows.sort(key=lambda r: (r["panel"], -r["shared"], -r["reproduced"], r["kb"]))
    cols = ["locus", "panel", "chrom", "strand", "kb", "n_tair", "n_pred", "n_art",
            "reproduced", "shared", "novel", "unsupported", "max_introns"]
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    from collections import Counter
    print("\npanel counts:", dict(Counter(r["panel"] for r in rows)))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
