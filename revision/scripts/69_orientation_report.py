#!/usr/bin/env python3
"""Render the §3.3 item 2 orientation audit (A39) as a per-run table.

§3.3 item 4 wants every run recorded separately, and A39 made the gate a different statistic from
the one §3.3's words name, so both are shown: `sense` is what decides -uf, `motif` is the retained
diagnostic. A run missing its verdict is listed as pending, never dropped - the whole point of the
audit is that it accounts for every run.

  usage: 69_orientation_report.py [--root <Transgenic>] [--format md|tsv|summary]
"""
import argparse
import json
import os
import sys

ROOT_DEFAULT = "/data/gpfs/assoc/pgl/data/Transgenic"


def band(sense):
    """The three library classes the statistic separates. Labels, not thresholds: only the frozen
    95% decides -uf. These exist so a reader can see WHY a run failed."""
    if sense is None:
        return "-"
    if sense >= 0.95:
        return "stranded sense"
    # The antisense band is set from what the data actually shows: col0_DRP009401 sits at
    # 0.128-0.186, which is nowhere near 0.5 and is plainly a reverse-oriented library, not a
    # half-and-half one. A band boundary at 0.10 would have called it "mixed" and hidden that.
    # These are labels for a reader; only the frozen 95% decides -uf, so where they fall changes
    # no verdict.
    if sense <= 0.25:
        return "stranded antisense"
    if 0.35 <= sense <= 0.65:
        return "unstranded"
    return "mixed/unclear"


def load(root):
    runs_tsv = os.path.join(root, "evidence", "longread_runs_v1.tsv")
    audit_root = os.path.join(root, "evidence", "orientation_audit_v1")
    rows = []
    with open(runs_tsv) as fh:
        for line in fh:
            sp, ds, run, _fa = line.rstrip("\n").split("\t")
            j = os.path.join(audit_root, sp, os.path.basename(ds), run, "audit.json")
            rows.append((sp, os.path.basename(ds), run,
                         json.load(open(j)) if os.path.exists(j) else None))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--format", choices=["md", "tsv", "summary"], default="summary")
    a = ap.parse_args()
    rows = load(a.root)
    done = [r for r in rows if r[3]]
    pending = [r for r in rows if not r[3]]

    if a.format == "tsv":
        print("species\tdataset\trun\tstatus\tuse_uf\tsense_read_fraction\t"
              "motif_annotation_agreement\tvalid_for_orientation\tband")
        for sp, ds, run, d in rows:
            if d is None:
                print(f"{sp}\t{ds}\t{run}\tPENDING\t-\t-\t-\t-\t-")
            else:
                print(f"{sp}\t{ds}\t{run}\t{d['status']}\t{d['use_uf']}\t"
                      f"{d['sense_read_fraction']:.4f}\t{d['motif_annotation_agreement']:.4f}\t"
                      f"{d['valid_for_orientation']}\t{band(d['sense_read_fraction'])}")
    elif a.format == "md":
        print("| species | dataset | run | sense | motif | n | verdict | -uf |")
        print("|---|---|---|---|---|---|---|---|")
        for sp, ds, run, d in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            if d is None:
                print(f"| {sp} | {ds} | {run} | | | | *pending* | |")
            else:
                print(f"| {sp} | {ds} | {run} | {d['sense_read_fraction']:.4f} | "
                      f"{d['motif_annotation_agreement']:.4f} | {d['valid_for_orientation']} | "
                      f"{d['status']} | {'-uf' if d['use_uf'] else 'no -uf'} |")

    # A dataset-level view is reported IN ADDITION, never instead: §3.3 item 4 makes the run the
    # unit of decision. It is here because a dataset whose runs disagree is worth a second look.
    by_ds = {}
    for sp, ds, run, d in done:
        by_ds.setdefault((sp, ds), []).append(d["status"])
    split = {k: v for k, v in by_ds.items() if len(set(v)) > 1}

    print(f"\n{len(done)}/{len(rows)} audited", file=sys.stderr)
    for st in ("PASS", "FAIL", "UNRESOLVED"):
        n = sum(1 for _, _, _, d in done if d["status"] == st)
        print(f"  {st:<11} {n}", file=sys.stderr)
    if a.format == "summary":
        print("verdicts by species and band:")
        agg = {}
        for sp, _ds, _run, d in done:
            agg.setdefault(sp, {}).setdefault(band(d["sense_read_fraction"]), 0)
            agg[sp][band(d["sense_read_fraction"])] += 1
        for sp in sorted(agg):
            parts = ", ".join(f"{k} {v}" for k, v in sorted(agg[sp].items()))
            print(f"  {sp:<14} {parts}")
    if split:
        print("\ndatasets whose runs do not agree (worth a look, not an error):", file=sys.stderr)
        for (sp, ds), v in sorted(split.items()):
            print(f"  {sp}/{ds}: {sorted(set(v))}", file=sys.stderr)
    if pending:
        print(f"\npending: {len(pending)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
