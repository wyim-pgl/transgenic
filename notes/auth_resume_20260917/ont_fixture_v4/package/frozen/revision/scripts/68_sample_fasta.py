#!/usr/bin/env python3
"""Draw a deterministic uniform sample of records from a (gzipped) FASTA.

§3.3 item 2 says "align 10,000 reads". Which 10,000 matters. The first 10,000 records of an ONT
run are not a sample of it: these files are ordered by acquisition, by channel, or by whatever
order the parts were concatenated in, so the head of the file can be one flow-cell region or one
time window. Sampling until 10,000 spliced alignments are collected would be worse still -- that
conditions the sample on the outcome being measured.

So: uniform without replacement over the whole run, with the PRNG seeded from the input's md5, so
the same input always yields the same sample and the audit is reproducible without storing it.
Prints a JSON line with the seed, the record counts and an md5 over the selected read ids, which is
what the audit artifact records as evidence of which reads were used.
"""
import argparse
import gzip
import hashlib
import json
import random
import sys


def openf(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--md5", required=True, help="md5 of --fasta; seeds the PRNG")
    a = ap.parse_args()

    total = 0
    with openf(a.fasta) as fh:
        for line in fh:
            if line[0] == ">":
                total += 1
    if total == 0:
        print(f"REFUSED: {a.fasta} holds no FASTA records", file=sys.stderr)
        sys.exit(2)

    seed = int(a.md5[:16], 16)
    k = min(a.n, total)
    pick = set(random.Random(seed).sample(range(total), k))

    i = 0
    keep = False
    h = hashlib.md5()
    written = 0
    with openf(a.fasta) as fh, open(a.out, "w") as out:
        for line in fh:
            if line[0] == ">":
                keep = i in pick
                i += 1
                if keep:
                    h.update(line.split()[0].encode())
                    written += 1
            if keep:
                out.write(line)
    if written != k:
        print(f"REFUSED: wrote {written} records, expected {k}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"run_records": total, "sampled": k, "seed": seed,
                      "sample_ids_md5": h.hexdigest(),
                      "short_run": k < a.n}, sort_keys=True))


if __name__ == "__main__":
    main()
