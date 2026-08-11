#!/usr/bin/env python3
"""Rebuild the alternative-transcript-only AtRTD3 reference using the rule stated
in Methods and already applied to TAIR10: remove the FIRST transcript of each gene
(file order).

The original builder (14_make_altonly_references.py) removed AtRTD3 transcripts whose
IDs matched the TAIR10 primary set instead, which leaves the primary transcript of
every AtRTD3 gene absent from TAIR10 in the "alternative-only" reference.

Writes to revision/results/altonly_fixed/ only; nothing existing is overwritten.
"""

import re
from pathlib import Path

REV = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision")
OUT = REV / "results/altonly_fixed"

GENE_RE = re.compile(r'gene_id "([^"]+)"')
TX_RE = re.compile(r'transcript_id "([^"]+)"')


def first_transcript_per_gene(gtf):
    """First transcript ID encountered per gene, in file order."""
    seen_gene = set()
    first = {}
    for line in open(gtf):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        g = GENE_RE.search(f[8])
        t = TX_RE.search(f[8])
        if not (g and t):
            continue
        g = g.group(1)
        if g not in seen_gene:
            seen_gene.add(g)
            first[g] = t.group(1)
    return first


def drop_transcripts(gtf, out_gtf, drop_ids):
    kept = dropped = 0
    with open(out_gtf, "w") as out:
        for line in open(gtf):
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.split("\t")
            if len(f) < 9:
                out.write(line)
                continue
            t = TX_RE.search(f[8])
            if t and t.group(1) in drop_ids:
                dropped += 1
                continue
            kept += 1
            out.write(line)
    return kept, dropped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    atrtd = REV / "data/AtRTD3/AtRTD3.gtf"
    first = first_transcript_per_gene(atrtd)
    print(f"AtRTD3 genes: {len(first)}")
    (OUT / "AtRTD3_primary_transcript_ids.txt").write_text(
        "\n".join(f"{g}\t{t}" for g, t in first.items()) + "\n")
    drop = set(first.values())
    k, d = drop_transcripts(atrtd, OUT / "AtRTD3.altonly_firsttx.gtf", drop)
    print(f"AtRTD3 altonly_firsttx: kept {k} feature lines, dropped {d}")


if __name__ == "__main__":
    main()
