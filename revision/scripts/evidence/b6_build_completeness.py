#!/usr/bin/env python3
"""WP2c completeness pass (plan v4.1 §6; ``transgenic.evidence.b6_completeness``): A16 codes per population.

    b6_build_completeness.py --species Athaliana --merged evidence_c0.db --read-ends read_ends_b6_v1.Athaliana.db
        --roles data/manifests/DATASET_ROLES.b6_v1.tsv --reference <fasta>
        --cage-recipe data/manifests/CAGE_RECIPE.b6_v1.md --cage-recipe-sha256 <frozen sha>
        --cage-inputs data/manifests/CAGE_INPUTS.b6_v1.tsv --cage RUN=<run>.cage.bam [...]
        --admitted-chains train_track.tsv valid_track.tsv eval_input.tsv score.tsv
        --out completeness_b6_v1.Athaliana.db

Refuses without the CAGE recipe sha (D12).  Every CAGE_INPUTS run of the species must be supplied with ``--cage``.
Each population needs its own admitted-chain TSV: first line ``# population=<name>``, then the header population,
species, chrom, strand, introns, and a ``.sha256`` sidecar.  A header-only file is a valid, hash-bound empty set (for
example the score population of a species without scoring data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_completeness as CP  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--read-ends", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--cage-recipe", required=True)
    ap.add_argument("--cage-recipe-sha256", required=True)
    ap.add_argument("--cage-inputs", required=True)
    ap.add_argument("--cage", action="append", default=[], help="RUN=PATH of a mapped CAGE alignment (recipe §1)")
    ap.add_argument("--admitted-chains", nargs="+", required=True)
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        cage = {}
        for v in a.cage:
            k, sep, p = v.partition("=")
            if not sep:
                raise CP.CompletenessError(f"--cage expects RUN=PATH, got {v!r}")
            cage[k] = p
        prov = CP.build(merged=a.merged, read_ends=a.read_ends, roles=a.roles, admitted_chains=a.admitted_chains,
                        cage_recipe=a.cage_recipe, cage_recipe_sha256=a.cage_recipe_sha256, cage_inputs=a.cage_inputs,
                        cage_paths=cage, reference=a.reference, species=a.species, out=a.out, samtools=a.samtools,
                        argv=argv)
    except (CP.CompletenessError, ValueError, FileExistsError, OSError) as exc:
        print(f"b6_build_completeness failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_sha256": prov["output_sha256"], "cage_peaks": prov["cage_peaks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
