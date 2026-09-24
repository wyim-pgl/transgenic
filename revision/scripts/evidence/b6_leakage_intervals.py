#!/usr/bin/env python3
"""Dump the A46 rule 4(a) excluded intervals of one species as TSV (the file b6_build_c0.py --overlap verifies).

    b6_leakage_intervals.py --corpus <corpus DuckDB> --species Athaliana --splits <split TSV> --flags <A22 flags>
                            [--build-flags <A22> <A30>] [--reference FASTA] [--gff GFF] --seed <corpus build seed> --out intervals.tsv

``--seed`` is required (the corpus build_manifest does not record it; a wrong seed fails closed).  Rows are written
sorted, so the TSV is byte-deterministic for one corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import leakage as LK  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--build-flags", nargs="+")
    ap.add_argument("--reference")
    ap.add_argument("--gff")
    ap.add_argument("--seed", type=int, required=True, help="corpus build seed (not in build_manifest)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        model = LK.build_leakage_model(corpus_db=a.corpus, species=a.species, splits_path=a.splits, flags_path=a.flags,
                                       build_flags=a.build_flags, gff_path=a.gff, fasta_path=a.reference, seed=a.seed)
        LK.write_intervals_tsv(model.intervals, a.out)
    except (LK.LeakageError, FileExistsError, OSError) as exc:
        print(f"b6_leakage_intervals failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(model.qc, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
