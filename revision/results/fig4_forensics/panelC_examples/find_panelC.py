import sys, re
from collections import defaultdict
DEST, ATRTD = sys.argv[1], sys.argv[2]

def cds_introns(exons):
    e=sorted(exons)
    return tuple((e[i][1]+1, e[i+1][0]-1) for i in range(len(e)-1))

def parse(path, gtf=False):
    tx_cds=defaultdict(list); tx_loc={}; tx_strand={}; tx_chr={}
    for line in open(path):
        if line.startswith("#") or "\t" not in line: continue
        f=line.rstrip("\n").split("\t")
        if len(f)<9 or f[2]!="CDS": continue
        if gtf:
            t=re.search(r'transcript_id "([^"]+)"',f[8]); g=re.search(r'gene_id "([^"]+)"',f[8])
            if not (t and g): continue
            key=t.group(1); loc=g.group(1)
        else:
            p=re.search(r'Parent=([^;]+)',f[8]); gm=re.search(r'GM=([^;]+)',f[8])
            if not p: continue
            key=p.group(1); loc=gm.group(1).replace(".TAIR10","").replace("-rc","") if gm else "?"
        tx_cds[key].append((int(f[3]),int(f[4]))); tx_loc[key]=loc; tx_strand[key]=f[6]; tx_chr[key]=f[0]
    chains={k:cds_introns(v) for k,v in tx_cds.items()}
    return chains, tx_loc, tx_strand, tx_chr

pc,pl,ps,pchr = parse(f"{DEST}/raw_TAIR10_hyenaTest_prediction_noPost.gff3")
tc,tl,_,_     = parse(f"{DEST}/raw_TAIR10_hyenaTest_labels.gff3")
ac,al,_,_     = parse(ATRTD, gtf=True)

# index chains by locus
def by_loc(chains, locs):
    d=defaultdict(set)
    for k,c in chains.items():
        d[locs[k]].add(c)
    return d
Tloc=by_loc(tc,tl); Aloc=by_loc(ac,al)

# find predicted transcripts: chain in AtRTD3, NOT in TAIR10, multi-intron (>=2 introns for clarity)
cands=[]
seen=set()
for tx,chain in pc.items():
    loc=pl[tx]
    if loc in ("?",) or len(chain)<2: continue
    T=Tloc.get(loc,set()); A=Aloc.get(loc,set())
    if chain in A and chain not in T:
        # this predicted isoform is AtRTD3-supported and absent from TAIR10
        n_introns=len(chain)
        # prefer loci where the model ALSO reproduced a TAIR10 isoform (shows primary + novel)
        also_primary = len(pc_by_loc.get(loc,set()) & T) if (pc_by_loc:=by_loc(pc,pl)) else 0
        key=(loc,chain)
        if key in seen: continue
        seen.add(key)
        cands.append((loc, tx, n_introns, len(T), len(A), also_primary, ps[tx], pchr[tx]))

# rank: more introns (clearer splice), and model also got primary, and modest AtRTD3 count
cands.sort(key=lambda x:(-x[2], -x[5], x[4]))
print(f"Found {len(cands)} clean Panel-C predicted isoforms (>=2 introns, in AtRTD3, absent from TAIR10)\n")
print(f"{'locus':11s} {'nIntron':>7s} {'nTAIR':>5s} {'nAtRTD':>6s} {'primary?':>8s} {'strand':>6s}  predicted_tx")
for loc,tx,ni,nt,na,ap,strand,chrom in cands[:20]:
    print(f"{loc:11s} {ni:7d} {nt:5d} {na:6d} {ap:8d} {strand:>6s}  {tx[:40]}")
