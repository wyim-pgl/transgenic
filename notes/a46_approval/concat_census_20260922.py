#!/usr/bin/env python3
"""Read-only: internal Iso-Seq primer k-mer or internal poly(A)/poly(T) run (concatemer signal)."""
import gzip, re, sys
N = int(sys.argv[1]); EDGE = 150
comp = str.maketrans("ACGT", "TGCA"); rc = lambda s: s.translate(comp)[::-1]
K = ["AAGCAGTGGTATCAAC", "GGTATCAACGCAGAGT", "AAGTCGCAGGGTTG", "GCAATGAAGTCGCAGG"]
PR = re.compile("|".join(K + [rc(k) for k in K])); PA = re.compile("A{20,}|T{20,}")
print("file\tn\tinternal_primer\tinternal_polyA20\teither")
for path in sys.argv[2:]:
    n = ip = ia = ei = 0; name = None; seq = []
    def done(s):
        global n, ip, ia, ei
        mid = s[EDGE:-EDGE] if len(s) > 2 * EDGE else ""
        a, b = bool(PR.search(mid)), bool(PA.search(mid)); n += 1; ip += a; ia += b; ei += a or b
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if name is not None:
                    done("".join(seq).upper())
                    if n >= N: break
                name, seq = line, []
            else: seq.append(line.strip())
    print(f"{path}\t{n}\t{ip/n:.4f}\t{ia/n:.4f}\t{ei/n:.4f}", flush=True)
