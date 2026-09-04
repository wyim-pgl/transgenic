#!/usr/bin/env python3
"""A37 paired reporting for the EST alignment: both ingestion arms, side by side.

A37 item 3 (protocol v1.27): "Wherever an EST-derived quantity appears, both arms appear." A single
mapping rate cannot be read on its own here, because the two arms differ in which records they
admit and short records map far worse than long ones -- that asymmetry is the whole reason A36 was
withdrawn. Reporting one arm's rate is reporting a number that its own inclusion rule produced.

Reads what the alignment already wrote; recomputes nothing and needs no BAM:
  evidence/est_align/<sp>/{flagstat.txt,PROVENANCE.txt,DONE}          primary,     >= 100 nt
  evidence/est_align_min121/<sp>/{...}                                sensitivity, >= 121 nt

Mapping rate is primary-alignment based: supplementary records are removed from both numerator and
denominator, which is how the 2026-09-03 table was computed, so the columns are comparable.

  usage: 66_est_arm_report.py [--root <Transgenic>] [--format md|tsv]
Incomplete species are listed as pending rather than silently dropped: a report that quietly omits
what has not finished looks identical to a report of a finished set.
"""
import argparse
import os
import sys

SPECIES = ("Athaliana", "Bdistachyon", "Gmax", "Osativa", "Ppatens",
           "Ptrichocarpa", "Sbicolor", "Sitalica", "Vvinifera")
ARMS = (("primary", "est_align", 100), ("sensitivity", "est_align_min121", 121))
ROOT_DEFAULT = "/data/gpfs/assoc/pgl/data/Transgenic"


def read_flagstat(path):
    """Total, supplementary and mapped counts from samtools flagstat."""
    total = suppl = mapped = None
    with open(path) as fh:
        for line in fh:
            n = line.split(" ", 1)[0]
            if not n.isdigit():
                continue
            if "in total" in line:
                total = int(n)
            elif "supplementary" in line:
                suppl = int(n)
            elif "mapped (" in line and "primary" not in line:
                mapped = int(n)
    if None in (total, suppl, mapped):
        raise ValueError(f"{path}: could not read total/supplementary/mapped")
    return total, suppl, mapped


def read_provenance(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    return out


def arm_stats(root, sp, subdir):
    d = os.path.join(root, "evidence", subdir, sp)
    fs, pv, done = (os.path.join(d, f) for f in ("flagstat.txt", "PROVENANCE.txt", "DONE"))
    if not (os.path.exists(done) and os.path.exists(fs) and os.path.exists(pv)):
        return None
    total, suppl, mapped = read_flagstat(fs)
    prov = read_provenance(pv)
    records = total - suppl
    return {
        "records": records,
        "mapped": mapped - suppl,
        "rate": (mapped - suppl) * 100.0 / records if records else float("nan"),
        "min_len": prov.get("min_len", "?").split()[0],
        "observed_min": prov.get("observed_min", "?"),
        "raw_count": prov.get("est_raw_count", "?"),
        "seconds": prov.get("wall_seconds", "?"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--format", choices=["md", "tsv"], default="md")
    a = ap.parse_args()

    rows, pending = [], []
    for sp in SPECIES:
        got = {name: arm_stats(a.root, sp, sub) for name, sub, _ in ARMS}
        missing = [n for n, v in got.items() if v is None]
        if missing:
            pending.append(f"{sp}: {', '.join(missing)}")
        rows.append((sp, got))

    if a.format == "tsv":
        print("species\tarm\tmin_len\tobserved_min\trecords\tmapped\trate_pct\twall_s")
        for sp, got in rows:
            for name, _sub, floor in ARMS:
                v = got[name]
                if v is None:
                    print(f"{sp}\t{name}\t{floor}\t-\t-\t-\t-\t-")
                else:
                    print(f"{sp}\t{name}\t{v['min_len']}\t{v['observed_min']}\t{v['records']}\t"
                          f"{v['mapped']}\t{v['rate']:.2f}\t{v['seconds']}")
    else:
        print("| species | records >=100 | rate >=100 | records >=121 | rate >=121 | delta (pp) |")
        print("|---|---|---|---|---|---|")
        for sp, got in rows:
            p, s = got["primary"], got["sensitivity"]
            if p is None or s is None:
                have = ", ".join(n for n in ("primary", "sensitivity") if got[n]) or "neither arm"
                print(f"| {sp} | | *pending — have: {have}* | | | |")
                continue
            print(f"| {sp} | {p['records']:,} | {p['rate']:.2f} % | {s['records']:,} | "
                  f"{s['rate']:.2f} % | {p['rate'] - s['rate']:+.2f} |")
        done = [(sp, g) for sp, g in rows if g["primary"] and g["sensitivity"]]
        if done:
            tp = sum(g["primary"]["records"] for _, g in done)
            mp = sum(g["primary"]["mapped"] for _, g in done)
            ts = sum(g["sensitivity"]["records"] for _, g in done)
            ms = sum(g["sensitivity"]["mapped"] for _, g in done)
            print(f"| **{len(done)} species** | **{tp:,}** | **{mp*100.0/tp:.2f} %** | "
                  f"**{ts:,}** | **{ms*100.0/ts:.2f} %** | **{mp*100.0/tp - ms*100.0/ts:+.2f}** |")

    if pending:
        print(f"\npending ({len(pending)}):", file=sys.stderr)
        for x in pending:
            print("  " + x, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
