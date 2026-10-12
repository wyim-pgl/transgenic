#!/usr/bin/env python3
"""Read-only census of Iso-Seq primer and poly(A) residue at read ends.

Raw CCS from an Iso-Seq library keeps the cDNA primers (and the poly(A) tail)
at both ends in either orientation. lima removes primers and orients reads;
refine additionally trims poly(A) -> FLNC. Polished HQ isoforms carry neither.
So per file we report the fraction of the first N reads that still carry
a primer k-mer or a poly(A)/poly(T) run within the terminal window.
"""
import gzip
import re
import sys

N = int(sys.argv[1])
WIN = 100
comp = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s.translate(comp)[::-1]


# exact 14-16-mer cores of the common Iso-Seq cDNA primers
KMERS = {
    "SMART": ["AAGCAGTGGTATCAAC", "GGTATCAACGCAGAGT"],   # Clontech SMARTer 5'/3' shared core
    "ISOEXP": ["AAGTCGCAGGGTTG", "GCAATGAAGTCGCAGG"],    # Iso-Seq Express 5' primer
}
PATS = {k: re.compile("|".join(v + [rc(x) for x in v])) for k, v in KMERS.items()}
POLYA3 = re.compile("A{15,}")
POLYT5 = re.compile("T{15,}")


def reads(path):
    name, seq = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].strip(), []
            else:
                seq.append(line.strip().upper())
        if name is not None:
            yield name, "".join(seq)


print("file\tn\tmean_len\tprimer_any\tprimer_both\tsmart\tisoexp\tpolyA3_or_T5\tpolyA3\tpolyT5\tfirst_header")
for path in sys.argv[2:]:
    n = tot = pany = pboth = sm = ie = pa = a3 = t5 = 0
    first = ""
    for name, s in reads(path):
        if n == 0:
            first = name
        n += 1
        tot += len(s)
        h, t = s[:WIN], s[-WIN:]
        hits = {k: (bool(p.search(h)), bool(p.search(t))) for k, p in PATS.items()}
        head = any(v[0] for v in hits.values())
        tail = any(v[1] for v in hits.values())
        pany += head or tail
        pboth += head and tail
        sm += any(hits["SMART"])
        ie += any(hits["ISOEXP"])
        x3 = bool(POLYA3.search(s[-50:]))
        x5 = bool(POLYT5.search(s[:50]))
        a3 += x3
        t5 += x5
        pa += x3 or x5
        if n >= N:
            break
    f = lambda c: f"{c / n:.3f}" if n else "NA"
    print("\t".join([path, str(n), f"{tot / n:.0f}" if n else "NA", f(pany), f(pboth), f(sm), f(ie),
                     f(pa), f(a3), f(t5), first]), flush=True)
