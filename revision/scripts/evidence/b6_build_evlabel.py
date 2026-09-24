#!/usr/bin/env python3
"""WP2b ``evlabel`` builder (plan v4.1 §5; ``transgenic.evidence.b6_evlabel``).  A50.2: NOT a B6 input.

    b6_build_evlabel.py --merged evidence_c0.db --protein-placement protein_placement_b6_v1.<sp>.db [...]
                        --corpus b5_full_b6_v1.db --freeze b5_full_b6_v1.freeze.json
                        --roles data/manifests/DATASET_ROLES.b6_v1.tsv --out evlabel_b6_v1.db

Before any label is built, the plan §8 ``training`` binding is enforced: freeze keys, per (species, source)
inputs.corpus = freeze base_sha256, interval_manifest_sha256 = the freeze entry for that species, C0 schema, leakage
QC, A49.2 exhaustiveness and completeness policy.  Every species with accepted protein placements needs its
supplement (sidecar + PROVENANCE, built against this merged snapshot); absent protein evidence is recorded.  Output: ``evlabel_b6_v1.db`` + ``.sha256`` + ``.PROVENANCE.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_evlabel as EV  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--protein-placement", nargs="*", default=[])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        prov = EV.build(merged=a.merged, supplements=a.protein_placement, corpus=a.corpus, freeze=a.freeze,
                        roles=a.roles, out=a.out, argv=argv)
    except (EV.EvlabelError, ValueError, FileExistsError, OSError) as exc:
        print(f"b6_build_evlabel failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_sha256": prov["output_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
