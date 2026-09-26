#!/usr/bin/env python3
"""WP2 rule 5 repair of the A40 tile corpus into the B6 corpus (plan v4.1 §2; ``transgenic.evidence.b6_corpus_repair``).

    b6_build_corpus.py --base b5_full_a40.db --splits <split TSV>
                       --flags Sp1=<Sp1>.geenuff_flags.tsv [--flags Sp2=... ...]
                       --build-flags Sp1=<Sp1>.geenuff_flags.tsv Sp1=<Sp1>.swissprot_flags.tsv [--build-flags Sp2=... ...]
                       --seed <corpus build seed> --interval-manifest <species>.b6-intervals-v2.jsonl [...]
                       (--measure-only [--report measure.json] | --apply --out b5_full_b6_v1.db --freeze freeze.json)

Flag files are per species, exactly as each species' C0 build takes them (--flags <Sp>.geenuff_flags.tsv
--build-flags <Sp>.geenuff_flags.tsv <Sp>.swissprot_flags.tsv): every build_manifest species needs its A22 file and
its build files, and a file naming another species is refused.  The bare form --flags <A22> --build-flags <A22> <A30>
is accepted only for a single-species corpus (one species' files would leave every other species' hard edge
crossers unmasked).  Per-species flag paths and sha256 go into the freeze and the report (``species_flags``).

``--measure-only`` counts every repair and the two row-set events (a train/valid tile over masked fraction 0.60; an
RC twin whose tile loses its last labelled gene) and writes nothing but the report.  ``--apply`` measures first and
refuses if either event is non-zero; otherwise it copies ``--base`` to ``--out`` (exclusive create, single-threaded
DuckDB, CHECKPOINT, no WAL left) and updates rows in place, then writes the freeze JSON (exclusive create).
``--interval-manifest`` takes one b6-intervals-v2 file per species (b6_leakage_intervals.py --manifest-out).
``--reference`` / ``--gff`` take SPECIES=PATH overrides for relocated build_manifest paths (sha256 still verified).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_corpus_repair as CR  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402


def _pairs(values, flag):
    out = {}
    for v in values or []:
        k, sep, p = v.partition("=")
        if not sep or not k or not p:
            raise CR.RepairError(f"{flag} expects SPECIES=PATH, got {v!r}")
        if k in out:
            raise CR.RepairError(f"{flag} names {k} twice")
        out[k] = p
    return out


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the frozen A40 tile corpus (closed DuckDB, no WAL)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--flags", action="append", required=True,
                    help="SPECIES=<Sp>.geenuff_flags.tsv (A22 rule 4(a) hard flags), repeated for every species")
    ap.add_argument("--build-flags", action="append", nargs="+", required=True,
                    help="SPECIES=PATH ... every flag file the A40 build used for that species (A22 + A30)")
    ap.add_argument("--seed", type=int, required=True, help="the A40 corpus build seed (verifies every repaired tile)")
    ap.add_argument("--interval-manifest", nargs="+", required=True, help="b6-intervals-v2 JSONL, one per species")
    ap.add_argument("--reference", action="append", help="SPECIES=FASTA override of the build_manifest path")
    ap.add_argument("--gff", action="append", help="SPECIES=GFF override of the build_manifest path")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--out", help="B6 corpus DB (--apply)")
    ap.add_argument("--freeze", help="freeze JSON (--apply)")
    ap.add_argument("--report", help="measurement JSON (--measure-only; exclusive create)")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    try:
        refs, gffs = _pairs(a.reference, "--reference"), _pairs(a.gff, "--gff")
        flags, build_flags = CR.parse_flag_args(a.flags, a.build_flags)
        common = dict(splits=a.splits, flags=flags, build_flags=build_flags, seed=a.seed,
                      manifests=a.interval_manifest, references=refs, gffs=gffs)
        if a.measure_only:
            if a.out or a.freeze:
                raise CR.RepairError("--out/--freeze are --apply options")
            _, out = CR.measure(a.base, **common)
            if a.report:
                with open(a.report, "x") as fh:
                    json.dump(out, fh, indent=1, sort_keys=True)
                    fh.write("\n")
            print(json.dumps({"events": out["events"], "counts": out["counts"]}, sort_keys=True))
            return 0
        if not (a.out and a.freeze):
            raise CR.RepairError("--apply requires --out and --freeze")
        record = CR.apply(a.base, a.out, a.freeze, argv=argv, **common)
    except (CR.RepairError, LK.LeakageError, FileExistsError, OSError) as exc:
        print(f"b6_build_corpus failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out_sha256": record["out_sha256"], "rows_repaired": record["rows_repaired"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
