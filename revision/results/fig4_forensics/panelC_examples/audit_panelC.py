#!/usr/bin/env python3
"""Audit every Panel-C candidate locus under one common rule set.

For each locus reports, from the extracted per-locus GFF3/GTF here:
  - the TAIR10 CDS intron-chains,
  - which predicted transcript is novel (chain absent from TAIR10) and which
    AtRTD3 transcript it matches exactly,
  - the highlighted feature (novel exon if the novel isoform has an exon that
    overlaps no TAIR10 exon, otherwise the novel splice junction),
  - AtRTD3 support for that feature, split into the three categories that were
    previously conflated: identical feature / overlap-only / splice-site sharing.

Run: python3 audit_panelC.py [locus ...]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCI = ["AT4G22540", "AT1G78940", "AT2G37450", "AT3G29185", "AT3G13740", "AT3G56730"]

UTR_FEATS = {"five_prime_UTR", "three_prime_UTR", "five_prime_utr", "three_prime_utr"}


def parse(path: Path, gtf: bool) -> dict:
    tx: dict = defaultdict(lambda: {"cds": [], "utr": [], "exon": [], "strand": ".", "chr": ""})
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        # Only structural rows define a transcript. Skipping gene/mRNA rows matters:
        # in the GFF3 an mRNA row carries Parent=<gene>, so keying on Parent alone
        # invents a phantom empty transcript per gene (this used to add a blank track).
        if f[2] not in ({"CDS", "exon"} | UTR_FEATS):
            continue
        m = (re.search(r'transcript_id "([^"]+)"', f[8]) if gtf
             else re.search(r"Parent=([^;]+)", f[8]))
        if not m:
            continue
        rec = tx[m.group(1)]
        rec["strand"], rec["chr"] = f[6], f[0]
        seg = (int(f[3]), int(f[4]))
        if f[2] == "CDS":
            rec["cds"].append(seg)
        elif f[2] == "exon":
            rec["exon"].append(seg)
        else:
            rec["utr"].append(seg)
    return dict(tx)


def chain(segs: list) -> tuple:
    e = sorted(segs)
    return tuple((e[i][1] + 1, e[i + 1][0] - 1) for i in range(len(e) - 1))


def overlaps(a: tuple, b: tuple) -> bool:
    return not (a[1] < b[0] or a[0] > b[1])


def audit(locus: str) -> dict:
    pred = parse(HERE / f"{locus}_pred.gff3", gtf=False)
    tair = parse(HERE / f"{locus}_tair.gff3", gtf=False)
    art = parse(HERE / f"{locus}_atrtd.gtf", gtf=True)

    tair_chains = {t: chain(r["cds"]) for t, r in tair.items()}
    tair_junctions = {j for c in tair_chains.values() for j in c}
    tair_exons = [s for r in tair.values() for s in (r["cds"] + r["utr"])]
    # AtRTD3 GTF carries exon (+CDS); use exon for structure, fall back to CDS
    art_struct = {t: (r["exon"] or r["cds"]) for t, r in art.items()}
    art_cds_chain = {t: chain(r["cds"]) for t, r in art.items() if r["cds"]}

    novel_tx = novel_match = None
    novel_exons: list = []
    novel_junctions: list = []
    for pt, pr in sorted(pred.items()):
        pc = chain(pr["cds"])
        if len(pc) < 2 or pc in set(tair_chains.values()):
            continue
        hits = sorted(t for t, c in art_cds_chain.items() if c == pc)
        if not hits:
            continue
        novel_tx, novel_match = pt, hits[0]
        novel_junctions = [j for j in pc if j not in tair_junctions]
        novel_exons = [s for s in (pr["cds"] + pr["utr"])
                       if not any(overlaps(s, b) for b in tair_exons)]
        break

    mode = "exon" if novel_exons else "junction"
    if mode == "exon":
        feat = (min(s for s, _ in novel_exons), max(e for _, e in novel_exons))
    elif novel_junctions:
        feat = novel_junctions[0]
    else:
        feat = None

    # AtRTD3 support for the highlighted feature, three disjoint categories
    identical, site_shared, overlap_only, absent = [], [], [], []
    for t, segs in sorted(art_struct.items()):
        if feat is None:
            continue
        if mode == "exon":
            if feat in segs:
                identical.append(t)
            elif any(s == feat[0] or e == feat[1] for s, e in segs):
                site_shared.append(t)
            elif any(overlaps(feat, s) for s in segs):
                overlap_only.append(t)
            else:
                absent.append(t)
        else:
            if feat in chain(segs):
                identical.append(t)
            elif any(j[0] == feat[0] or j[1] == feat[1] for j in chain(segs)):
                site_shared.append(t)
            else:
                absent.append(t)

    tair_supports = False
    if feat is not None:
        tair_supports = (any(feat in (r["cds"] + r["utr"]) for r in tair.values()) if mode == "exon"
                         else feat in tair_junctions)

    return dict(locus=locus, pred=pred, tair=tair, art=art, novel_tx=novel_tx,
                novel_match=novel_match, mode=mode, feat=feat,
                n_tair_chains=len(set(tair_chains.values())), n_art=len(art_struct),
                identical=identical, site_shared=site_shared,
                overlap_only=overlap_only, absent=absent,
                tair_supports=tair_supports,
                chrom=next(iter(tair.values()))["chr"] if tair else "",
                strand=next(iter(tair.values()))["strand"] if tair else ".")


if __name__ == "__main__":
    for loc in (sys.argv[1:] or LOCI):
        a = audit(loc)
        if a["novel_tx"] is None:
            print(f"\n### {loc}: NO novel AtRTD3-matched isoform found — NOT a Panel-C locus")
            continue
        f = a["feat"]
        if f is None:
            print(f"\n### {loc}  {a['chrom']}({a['strand']})  "
                  f"TAIR10 {len(a['tair'])} tx / {a['n_tair_chains']} chains | "
                  f"pred {len(a['pred'])} tx | AtRTD3 {a['n_art']} tx")
            print(f"    novel predicted tx : {a['novel_tx'].split('.')[-1]}  "
                  f"== AtRTD3 {a['novel_match']} (exact CDS chain)")
            print("    highlighted        : NONE — every junction of the novel chain is "
                  "used by some TAIR10 isoform; the novel part is the COMBINATION only")
            continue
        print(f"\n### {loc}  {a['chrom']}({a['strand']})  "
              f"TAIR10 {len(a['tair'])} tx / {a['n_tair_chains']} chains | "
              f"pred {len(a['pred'])} tx | AtRTD3 {a['n_art']} tx")
        print(f"    novel predicted tx : {a['novel_tx'].split('.')[-1]}  "
              f"== AtRTD3 {a['novel_match']} (exact CDS chain)")
        print(f"    highlighted        : {a['mode']} {f[0]:,}-{f[1]:,} "
              f"({f[1]-f[0]+1} bp)   TAIR10 has it: {a['tair_supports']}")
        print(f"    AtRTD3 identical   : {len(a['identical']):2d}  {a['identical']}")
        print(f"    AtRTD3 site-shared : {len(a['site_shared']):2d}  {a['site_shared']}")
        if a["overlap_only"]:
            print(f"    AtRTD3 overlap-only: {len(a['overlap_only']):2d}  {a['overlap_only']}")
        print(f"    AtRTD3 absent      : {len(a['absent']):2d}")
