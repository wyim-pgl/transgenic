#!/usr/bin/env python3
"""Filter a TransGenic GFF3 to the top-beam prediction per locus.

TransGenic exports both beam-search hypotheses per locus (identical GM
attribute appearing twice). This keeps the FIRST gene record per GM value
(file order = beam rank) together with all of its child features.

Usage: python 13_beam1_filter.py in.gff3 out.gff3
"""

import sys


def main(inp, outp):
    seen_gm = set()
    keep_gene_ids = set()
    keep_tx_ids = set()
    lines = open(inp).read().splitlines(True)

    def attr(line, key):
        for kv in line.rstrip("\n").split("\t")[8].split(";"):
            kv = kv.strip()
            if kv.startswith(key + "="):
                return kv.split("=", 1)[1]
        return None

    # pass 1: select top-beam gene IDs per GM
    for line in lines:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        gm = attr(line, "GM")
        gid = attr(line, "ID")
        key = gm if gm else gid
        if key in seen_gm:
            continue
        seen_gm.add(key)
        keep_gene_ids.add(gid)

    # pass 2: collect transcripts of kept genes
    for line in lines:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] not in ("mRNA", "transcript"):
            continue
        if attr(line, "Parent") in keep_gene_ids:
            keep_tx_ids.add(attr(line, "ID"))

    # pass 3: write
    n_gene_in = n_gene_out = 0
    with open(outp, "w") as out:
        for line in lines:
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.split("\t")
            if len(f) < 9:
                out.write(line)
                continue
            feat = f[2]
            if feat == "gene":
                n_gene_in += 1
                if attr(line, "ID") in keep_gene_ids:
                    n_gene_out += 1
                    out.write(line)
            elif feat in ("mRNA", "transcript"):
                if attr(line, "ID") in keep_tx_ids:
                    out.write(line)
            else:
                parent = attr(line, "Parent")
                if parent and any(p in keep_tx_ids for p in parent.split(",")):
                    out.write(line)
    print(f"genes: {n_gene_in} -> {n_gene_out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
