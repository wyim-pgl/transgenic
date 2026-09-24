#!/usr/bin/env python3
"""WP2-prot protein-placement supplement (plan v4.1 §4; ``transgenic.evidence.b6_protein_placement``).

    # measure the stop convention on one miniprot GFF per species (freeze once if all agree, per species otherwise)
    b6_build_protein_placement.py --measure-stop-convention --gff <sp>.protein.gff.gz --reference <sp>.fa

    # build protein_placement_b6_v1.<species>.db (+ .sha256, + .PROVENANCE.json); no WAL, no c0-v1 change
    b6_build_protein_placement.py --species Athaliana --gff <the GFF the protein C0 recorded> --reference <fasta>
        --protein-c0 Athaliana.protein.c0.db --merged evidence_c0.db --stop-convention inside_last_block
        --out protein_placement_b6_v1.Athaliana.db

Every input is sha-verified: the GFF and the reference against the protein C0's recorded inputs, the protein C0
against its PROVENANCE output_sha256, and the merged snapshot against its sidecar and its merge_manifest row for
(species, protein).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_protein_placement as PP  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-stop-convention", action="store_true")
    ap.add_argument("--species")
    ap.add_argument("--gff", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--protein-c0")
    ap.add_argument("--merged")
    ap.add_argument("--stop-convention", choices=PP.STOP_CONVENTIONS)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.measure_stop_convention:
            from transgenic.evidence import leakage as LK

            genome = LK.load_build_b5().load_fasta(a.reference)
            print(json.dumps({"gff": a.gff, "gff_sha256": PP.sha256(a.gff),
                              "counts": PP.measure_stop_convention(a.gff, genome)}, sort_keys=True))
            return 0
        missing = [k for k in ("species", "protein_c0", "merged", "stop_convention", "out") if not getattr(a, k)]
        if missing:
            raise PP.PlacementError(f"build mode requires --{', --'.join(m.replace('_', '-') for m in missing)}")
        prov = PP.build(species=a.species, gff=a.gff, reference=a.reference, protein_c0=a.protein_c0, merged=a.merged,
                        out=a.out, stop_convention=a.stop_convention, argv=argv)
    except (PP.PlacementError, ValueError, FileExistsError, OSError) as exc:
        print(f"b6_build_protein_placement failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_sha256": prov["output_sha256"], "qc": prov["qc"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
