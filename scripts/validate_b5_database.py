#!/usr/bin/env python3
"""Audit a B5 database: split integrity (orthogroup, RC, strict held-out), excluded species, caps, NULL labels."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from transgenic.datasets.build_b5 import validate_b5_database  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--db", required=True)
ap.add_argument("--excluded", nargs="*", default=["Zmays"])
a = ap.parse_args()
report = validate_b5_database(a.db, excluded_species=a.excluded)
print(json.dumps(report, indent=1))
sys.exit(0 if report["ok"] else 1)
