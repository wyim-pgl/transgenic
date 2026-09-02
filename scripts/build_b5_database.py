#!/usr/bin/env python3
"""Build the leakage-controlled B5 DuckDB from the frozen species manifest and split table (docs/gsf_spec_v1.md)."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from transgenic.datasets.build_b5 import build_b5_database  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--species-manifest", default="data/manifests/b5_species_v1.tsv")
    ap.add_argument("--split-table", default="data/splits/b5_orthogroup_split_v1.tsv")
    ap.add_argument("--rc", choices=["none", "all", "isoform-only"], default="isoform-only")
    ap.add_argument("--add-extra", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-len", type=int, default=49152)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--no-verify-md5", action="store_true")
    ap.add_argument("--only", nargs="*", help="species_id subset (smoke build)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--qc-flags", default=None, help="GeenuFF flags TSV from revision/scripts/62_geenuff_qc.py (protocol A22)")
    a = ap.parse_args()
    if os.path.exists(a.db):
        if not a.overwrite:
            sys.exit(f"{a.db} exists; the B5 database is immutable — use --overwrite only for smoke builds")
        os.remove(a.db)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    res = build_b5_database(a.db, a.species_manifest, a.split_table, rc=a.rc, add_extra=a.add_extra, seed=a.seed, max_len=a.max_len,
                            clean=a.clean, verify_md5=not a.no_verify_md5, only_species=set(a.only) if a.only else None, git_commit=commit,
                            qc_flags_path=a.qc_flags)
    for r in res:
        print(json.dumps({"species_id": r["species_id"], "rows": r["rows"], "rc_rows": r["rc_rows"], "rejected": len(r["rejected"])}))
    with open(a.db + ".rejected.json", "w") as fh:
        json.dump({r["species_id"]: r["rejected"] for r in res}, fh, indent=1)


if __name__ == "__main__":
    main()
