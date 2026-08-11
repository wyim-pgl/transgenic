#!/usr/bin/env python3
"""Audit the alt-only AtRTD3 reference: how many genes retained their primary transcript."""
import re, sys, collections

REV = "/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision"

def gene_tx(path):
    """gene -> ordered list of unique transcript ids (file order)."""
    d = collections.OrderedDict()
    seen = collections.defaultdict(set)
    for line in open(path):
        if line.startswith("#"): continue
        f = line.split("\t")
        if len(f) < 9: continue
        g = re.search(r'gene_id "([^"]+)"', f[8])
        t = re.search(r'transcript_id "([^"]+)"', f[8])
        if not (g and t): continue
        g, t = g.group(1), t.group(1)
        if t not in seen[g]:
            seen[g].add(t)
            d.setdefault(g, []).append(t)
    return d

full = gene_tx(f"{REV}/data/AtRTD3/AtRTD3.gtf")
alt  = gene_tx(f"{REV}/data/AtRTD3/AtRTD3.altonly.gtf")
primary = set(p.replace(".TAIR10","") for p in
              open(f"{REV}/data/TAIR10/primary_transcript_ids.txt").read().split())

ntx_full = sum(len(v) for v in full.values())
ntx_alt  = sum(len(v) for v in alt.values())
print(f"AtRTD3 full   : {len(full)} genes, {ntx_full} transcripts")
print(f"AtRTD3 altonly: {len(alt)} genes, {ntx_alt} transcripts")
print(f"removed       : {ntx_full-ntx_alt} transcripts, {len(full)-len(alt)} genes lost entirely")
print(f"TAIR10 primary id set: {len(primary)}")

# per-gene removal counts
rem = collections.Counter()
gene_first_kept = 0     # genes whose FILE-ORDER-FIRST transcript still present in altonly
genes_no_removal = []
for g, txs in full.items():
    kept = set(alt.get(g, []))
    n_removed = len(txs) - len(kept)
    rem[n_removed] += 1
    if txs[0] in kept:
        gene_first_kept += 1
print("\nper-gene #transcripts removed -> #genes:")
for k in sorted(rem): print(f"  {k:>3} removed : {rem[k]:>7} genes")
print(f"\ngenes whose file-order-FIRST transcript survives in altonly: {gene_first_kept} "
      f"({100*gene_first_kept/len(full):.1f}% of {len(full)})")

# how many AtRTD3 genes have any TAIR-primary-matching transcript
hit = sum(1 for g,txs in full.items() if any(t in primary for t in txs))
print(f"AtRTD3 genes containing >=1 TAIR10-primary id: {hit}")

# distribution of transcripts/gene in full
n = collections.Counter(len(v) for v in full.values())
single = n[1]
print(f"AtRTD3 single-transcript genes: {single}")
# single-transcript genes that survived intact (i.e. whole gene leaked as 'alt')
single_intact = sum(1 for g,txs in full.items() if len(txs)==1 and len(alt.get(g,[]))==1)
print(f"  ...of which fully retained in altonly (primary leaked): {single_intact}")
