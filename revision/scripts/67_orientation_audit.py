#!/usr/bin/env python3
"""§3.3 item 2 — library orientation audit. Decides `-uf` for ONE run.

Protocol §3.3 item 2 (frozen): "align 10,000 reads without -uf; compute the fraction of spliced
alignments whose inferred transcript strand agrees with the annotated strand at confidently
annotated multi-exon genes (single-strand loci only). -uf is enabled for a run only if agreement
>= 95%; otherwise the run is aligned without -uf throughout."

Reads SAM on stdin (or --sam) and a reference GFF3, writes a JSON verdict.

The three places this can quietly go wrong, and what is done about them:

1. `ts:A:` is relative to the READ, not the genome. minimap2 infers it from the splice signal
   (GT-AG vs CT-AC), so it exists only on spliced alignments where the strand could be called at
   all. Reading it as a genomic strand inverts every reverse-complemented alignment, which is
   roughly half of them, and drives any run toward 50%. It is converted with FLAG 0x10.

2. The denominator. §3.3 says "the fraction of spliced alignments ... at confidently annotated
   multi-exon genes (single-strand loci only)", so the denominator is not "spliced alignments"
   and not "aligned reads". Every exclusion below removes the alignment from BOTH numerator and
   denominator; an alignment that cannot be judged is not evidence of disagreement.

3. Power. A run whose sample yields a handful of eligible alignments produces a fraction that can
   sit either side of 95% by chance. Below --min-eligible the verdict is UNRESOLVED and `-uf` is
   never enabled; the frozen 95% rule itself is not touched.

Qualifying gene (operationalising "confidently annotated multi-exon ... single-strand loci"):
  - a `gene` feature with strand exactly + or -
  - carrying at least one transcript with >= 2 distinct, non-overlapping exons
  - not overlapped, by even one base, by any gene on the opposite strand
Confidence here is structural confidence in the frozen reference GFF. With only reference
annotation available it cannot honestly mean experimentally validated, and saying so is the point.

An alignment is scored only if its genomic span overlaps exactly ONE qualifying gene and at least
one of its introns lies inside that gene. Requiring the intron to MATCH an annotated junction would
restrict the audit to isoforms already in the annotation, which is circular.
"""
import argparse
import gzip
import json
import math
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
TS_RE = re.compile(r"\bts:A:([+-])")


def openf(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def parse_attrs(field):
    out = {}
    for part in field.rstrip(";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_qualifying_genes(gff_path):
    """Gene intervals that survive the §3.3 filter, indexed per contig."""
    genes = {}            # gene id -> [contig, start, end, strand]
    tx_gene = {}          # transcript id -> gene id
    tx_exons = defaultdict(list)
    with openf(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind, strand = f[2], f[6]
            if kind == "gene":
                if strand not in "+-":
                    continue
                a = parse_attrs(f[8])
                gid = a.get("ID")
                if gid:
                    genes[gid] = [f[0], int(f[3]), int(f[4]), strand]
            elif kind in ("mRNA", "transcript", "lncRNA"):
                a = parse_attrs(f[8])
                tid, par = a.get("ID"), a.get("Parent")
                if tid and par:
                    tx_gene[tid] = par
            elif kind == "exon":
                a = parse_attrs(f[8])
                par = a.get("Parent")
                if par:
                    tx_exons[par].append((int(f[3]), int(f[4])))

    # multi-exon: some transcript of the gene has >= 2 distinct, non-overlapping exons
    multi = set()
    for tid, exons in tx_exons.items():
        gid = tx_gene.get(tid)
        if gid is None or gid in multi:
            continue
        es = sorted(set(exons))
        if len(es) < 2:
            continue
        if any(es[i][1] < es[i + 1][0] for i in range(len(es) - 1)):
            multi.add(gid)

    by_contig = defaultdict(list)
    for gid, (contig, start, end, strand) in genes.items():
        by_contig[contig].append((start, end, strand, gid in multi))

    # single-strand locus: drop any gene overlapped by >=1 base by an opposite-strand gene
    qualifying = defaultdict(list)
    dropped_opposite = 0
    for contig, items in by_contig.items():
        items.sort()
        # for each gene, is there a gene of the OTHER strand overlapping it by >= 1 base?
        others = {"+": sorted([(s, e) for s, e, st, _ in items if st == "-"]),
                  "-": sorted([(s, e) for s, e, st, _ in items if st == "+"])}
        starts = {k: [s for s, _ in v] for k, v in others.items()}
        # prefix max of ends, so "is there an opposite gene starting <= end with its end >= start"
        pmax = {}
        for k, v in others.items():
            m, run = [], 0
            for _s, e in v:
                run = max(run, e)
                m.append(run)
            pmax[k] = m
        for start, end, strand, is_multi in items:
            if not is_multi:
                continue
            v, st, pm = others[strand], starts[strand], pmax[strand]
            i = bisect_right(st, end)
            if i > 0 and pm[i - 1] >= start:
                dropped_opposite += 1
                continue
            qualifying[contig].append((start, end, strand))
    # Per contig keep the sorted intervals plus the longest gene on it, so a lookup can start its
    # scan at (query_start - max_len) instead of at index 0. Without that bound the scan is
    # O(genes) per alignment and the audit spends minutes per run doing nothing but walking lists.
    index = {}
    for contig, v in qualifying.items():
        v.sort()
        index[contig] = (v, [x[0] for x in v], max((e - s2 + 1) for s2, e, _ in v))
    return index, dropped_opposite


def genes_overlapping(index, contig, start, end):
    """All qualifying genes whose interval overlaps [start, end], 1-based inclusive."""
    got = index.get(contig)
    if not got:
        return []
    items, starts, maxlen = got
    i = bisect_left(starts, start - maxlen)
    hits = []
    for j in range(i, len(items)):
        s, e, strand = items[j]
        if s > end:
            break
        if e >= start:
            hits.append((s, e, strand))
    return hits


def introns_from_cigar(cigar, pos):
    """1-based inclusive intron intervals implied by N operations."""
    ref = pos
    out = []
    for n, op in CIGAR_RE.findall(cigar):
        n = int(n)
        if op == "N":
            out.append((ref, ref + n - 1))
            ref += n
        elif op in "MDP=X":
            ref += n
    return out


def ref_end(cigar, pos):
    ref = pos
    for n, op in CIGAR_RE.findall(cigar):
        if op in ("M", "D", "N", "=", "X"):
            ref += int(n)
    return ref - 1


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def audit(sam_fh, qual, min_mapq):
    c = defaultdict(int)
    agree = 0
    eligible = 0
    for line in sam_fh:
        if line.startswith("@"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 11:
            continue
        c["records"] += 1
        flag = int(f[1])
        if flag & 0x4:
            c["unmapped"] += 1; continue
        if flag & 0x900:
            c["secondary_or_supplementary"] += 1; continue
        if int(f[4]) < min_mapq:
            c["low_mapq"] += 1; continue
        cigar = f[5]
        if "N" not in cigar:
            c["unspliced"] += 1; continue
        m = TS_RE.search(line)
        if not m:
            c["no_ts_tag"] += 1; continue
        # ts is relative to the read; FLAG 0x10 puts it on the genome
        ts = m.group(1)
        inferred = ts if not (flag & 0x10) else ("-" if ts == "+" else "+")
        contig, pos = f[2], int(f[3])
        end = ref_end(cigar, pos)
        hits = genes_overlapping(qual, contig, pos, end)
        if len(hits) == 0:
            c["no_qualifying_gene"] += 1; continue
        if len(hits) > 1:
            c["ambiguous_gene"] += 1; continue
        gs, ge, gstrand = hits[0]
        if not any(gs <= a and b <= ge for a, b in introns_from_cigar(cigar, pos)):
            c["no_intron_inside_gene"] += 1; continue
        eligible += 1
        if inferred == gstrand:
            agree += 1
    c["eligible"] = eligible
    c["agree"] = agree
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam", default="-", help="SAM from the no-uf alignment ('-' = stdin)")
    ap.add_argument("--gff", required=True)
    ap.add_argument("--out", required=True, help="JSON verdict")
    ap.add_argument("--run", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--min-mapq", type=int, default=20, help="§4: MAPQ >= 20")
    ap.add_argument("--min-eligible", type=int, default=1000,
                    help="below this the verdict is UNRESOLVED and -uf is never enabled")
    ap.add_argument("--threshold", type=float, default=0.95, help="§3.3 item 2, frozen")
    ap.add_argument("--extra", default="{}", help="JSON of provenance fields to embed")
    a = ap.parse_args()

    qual, dropped_opposite = load_qualifying_genes(a.gff)
    n_qual = sum(len(v[0]) for v in qual.values())
    fh = sys.stdin if a.sam == "-" else openf(a.sam)
    c = audit(fh, qual, a.min_mapq)

    eligible, agree = c["eligible"], c["agree"]
    fraction = agree / eligible if eligible else None
    if eligible < a.min_eligible:
        status, uf = "UNRESOLVED", False
        reason = (f"only {eligible} eligible alignments, below the floor of {a.min_eligible}; "
                  f"the frozen >= {a.threshold:.0%} rule is not applied to a sample this small")
    elif fraction >= a.threshold:
        status, uf = "PASS", True
        reason = f"agreement {fraction:.4f} >= {a.threshold}"
    else:
        status, uf = "FAIL", False
        reason = f"agreement {fraction:.4f} < {a.threshold}"

    lo, hi = wilson(agree, eligible)
    out = {
        "run": a.run, "species": a.species, "dataset": a.dataset,
        "status": status, "use_uf": uf, "reason": reason,
        "agree": agree, "eligible": eligible,
        "fraction": fraction,
        "wilson95": None if eligible == 0 else [round(lo, 6), round(hi, 6)],
        "threshold": a.threshold, "min_eligible": a.min_eligible, "min_mapq": a.min_mapq,
        "qualifying_genes": n_qual, "genes_dropped_opposite_strand": dropped_opposite,
        "counters": dict(c),
        "protocol": "B1 §3.3 item 2 (frozen); ts converted to genomic strand via FLAG 0x10",
    }
    out.update(json.loads(a.extra))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))
    # A run that cannot be judged must not read as a pass.
    sys.exit(0 if status == "PASS" else (4 if status == "UNRESOLVED" else 0))


if __name__ == "__main__":
    main()
