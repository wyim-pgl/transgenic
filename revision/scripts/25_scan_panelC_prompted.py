#!/usr/bin/env python3
"""Find every locus where a prompted TransGenic isoform matches AtRTD3 but not TAIR10.

This scans the *same* prediction the manuscript's isoform numbers come from —
`transgenic_comparison/standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3`,
the top-beam prompted run over all 27,413 A. thaliana evaluation loci. Earlier versions
of this analysis used a per-locus extract whose provenance could not be established;
that file is not referenced here, so every locus reported below can be re-derived from
files the manuscript already cites.

A locus qualifies when a predicted transcript's CDS intron chain
  * is identical to the chain of an AtRTD3 transcript at that locus,
  * is identical to no TAIR10 transcript there, and
  * contains at least two introns.

Each hit is labelled with the weakest discriminating feature that applies:
  junction     - the chain uses at least one splice junction no TAIR10 isoform uses
  combination  - every junction already occurs in TAIR10; only the chain is new
`junction` loci are the ones where a reader can point at the novel feature in a figure.

Usage:
    python 25_scan_panelC_prompted.py [--out panelC_prompted.tsv]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRED = (ROOT / "transgenic_comparison" / "standardized_results"
        / "A_thaliana_transgenic400Mprompt_beam1.gff3")
TAIR = ROOT / "transgenic" / "revision" / "data" / "TAIR10" / "TAIR10.gtf"
ATRTD = (ROOT / "transgenic" / "revision" / "data" / "AtRTD3"
         / "atRTD3_TS_21Feb22_transfix.gtf")
OUT = (ROOT / "transgenic" / "revision" / "results" / "fig4_forensics"
       / "panelC_examples" / "prompted_full" / "panelC_prompted.tsv")

MIN_INTRONS = 2


def _cds_by_transcript(path: Path, gff3: bool) -> dict:
    """{locus: {transcript: [(start, end), …]}} over CDS rows only."""
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
            # Reverse-complement augmentation rows are a training artefact and are
            # emitted in window-local coordinates; they must never enter a comparison.
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            out[locus.replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    return out


def intron_chains(path: Path, gff3: bool) -> dict:
    """{locus: {transcript: chain}}; chain is the ordered tuple of intron intervals."""
    chains: dict = {}
    for locus, txs in _cds_by_transcript(path, gff3).items():
        chains[locus] = {}
        for tx, segs in txs.items():
            s = sorted(segs)
            chains[locus][tx] = tuple((s[i][1], s[i + 1][0]) for i in range(len(s) - 1))
    return chains


def scan() -> list[dict]:
    pred = intron_chains(PRED, gff3=True)
    tair = intron_chains(TAIR, gff3=False)
    art = intron_chains(ATRTD, gff3=False)
    print(f"  prediction {len(pred):,} loci | TAIR10 {len(tair):,} | AtRTD3 {len(art):,}",
          file=sys.stderr)

    hits = []
    for locus, ptx in pred.items():
        art_here = art.get(locus)
        if not art_here:
            continue
        tair_chains = set(tair.get(locus, {}).values())
        tair_junctions = {j for c in tair_chains for j in c}
        art_chains = set(art_here.values())
        for tx, chain in sorted(ptx.items()):
            if len(chain) < MIN_INTRONS or chain in tair_chains or chain not in art_chains:
                continue
            novel = [j for j in chain if j not in tair_junctions]
            matches = sorted(k for k, v in art_here.items() if v == chain)
            # Two different kinds of long-read support, kept apart because they answer
            # different questions: how many AtRTD3 transcripts reproduce the whole chain,
            # and how many merely use the junction that TAIR10 lacks. The second is the
            # larger number and the one a figure's amber band refers to.
            carrying = sum(1 for c in art_here.values()
                           if all(j in c for j in novel)) if novel else 0
            hits.append({
                "locus": locus,
                "category": "junction" if novel else "combination",
                "novel_junctions": len(novel),
                "introns": len(chain),
                "atrtd_match": matches[0],
                "atrtd_sharing_chain": len(matches),
                "atrtd_carrying_novel_junction": carrying,
                "atrtd_transcripts": len(art_here),
                "tair_chains": len(tair_chains),
                "predicted_transcript": tx,
            })
            break                      # one novel isoform per locus is enough
    # Directly visible features first, then by how much structure supports them.
    hits.sort(key=lambda h: (h["category"] != "junction",
                             -h["novel_junctions"], -h["introns"], h["locus"]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    hits = scan()
    n_j = sum(1 for h in hits if h["category"] == "junction")
    print(f"  {len(hits)} qualifying loci: {n_j} junction, {len(hits) - n_j} combination",
          file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["locus", "category", "novel_junctions", "introns", "atrtd_match",
            "atrtd_sharing_chain", "atrtd_carrying_novel_junction",
            "atrtd_transcripts", "tair_chains", "predicted_transcript"]
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for h in hits:
            fh.write("\t".join(str(h[c]) for c in cols) + "\n")
    print(f"written: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
