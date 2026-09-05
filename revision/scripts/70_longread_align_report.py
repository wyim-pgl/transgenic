#!/usr/bin/env python3
"""Completeness and outcome table for the ONT long-read alignment (§3.3 item 4, §4).

Every run is aligned and scored separately, so the report is per run and a run that is missing is
named rather than omitted. "Complete" here means all of what §4 asks to be retained is present and
readable: sorted BAM, its index, the PAF carrying --cs=long, PROVENANCE.txt, flagstat and the
contig alias table. A directory holding four of those six is not a finished run.

Mapping rate is primary-alignment based (supplementary removed from numerator and denominator
alike), which is how the EST table was computed, so the two are comparable.

--mapq additionally counts alignments passing §4's acceptance rules (primary, MAPQ >= 20). That is
a full pass over every BAM and is off by default: the audit already showed two datasets losing
30-50 % of their reads there, so the number matters, but it costs a scan of ~500 GB.

  usage: 70_longread_align_report.py [--root R] [--format md|tsv|summary] [--mapq]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT_DEFAULT = "/data/gpfs/assoc/pgl/data/Transgenic"
BIN = "/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin"
REQUIRED = ("PROVENANCE.txt", "flagstat.txt", "contig_alias.tsv", "DONE")


def read_flagstat(path):
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
        raise ValueError(f"{path}: unreadable")
    return total, suppl, mapped


def kv(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split(None, 1)
            if len(p) == 2:
                out[p[0]] = p[1].strip()
    return out


def run_row(root, sp, ds, run, want_mapq):
    d = os.path.join(root, "evidence", "longread_align_v1", sp, ds, run)
    bam = os.path.join(d, f"{run}.bam")
    paf = os.path.join(d, f"{run}.paf.gz")
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(d, f))]
    for f, label in ((bam, "bam"), (bam + ".bai", "bai"), (paf, "paf")):
        if not (os.path.exists(f) and os.path.getsize(f) > 0):
            missing.append(label)
    if missing:
        return {"run": run, "species": sp, "dataset": ds, "complete": False, "missing": missing}

    total, suppl, mapped = read_flagstat(os.path.join(d, "flagstat.txt"))
    prov = kv(os.path.join(d, "PROVENANCE.txt"))
    done = dict(l.split("=", 1) for l in open(os.path.join(d, "DONE")).read().splitlines() if "=" in l)
    records = total - suppl
    row = {
        "run": run, "species": sp, "dataset": ds, "complete": True, "missing": [],
        "records": records, "mapped": mapped - suppl,
        "rate": (mapped - suppl) * 100.0 / records if records else float("nan"),
        "uf": done.get("uf", "?"), "audit_status": done.get("audit_status", "?"),
        "seconds": prov.get("wall_seconds", "?"),
        "bam_gb": os.path.getsize(bam) / 2**30, "paf_gb": os.path.getsize(paf) / 2**30,
    }
    if want_mapq:
        n = subprocess.run([f"{BIN}/samtools", "view", "-c", "-q", "20", "-F", "0x900", bam],
                           capture_output=True, text=True)
        if n.returncode == 0:
            row["accepted"] = int(n.stdout.strip())
            row["accepted_pct"] = row["accepted"] * 100.0 / records if records else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--format", choices=["md", "tsv", "summary"], default="summary")
    ap.add_argument("--mapq", action="store_true",
                    help="also count §4-acceptable alignments (primary, MAPQ>=20); scans every BAM")
    a = ap.parse_args()

    rows = []
    with open(os.path.join(a.root, "evidence", "longread_runs_v1.tsv")) as fh:
        for line in fh:
            sp, ds, run, _ = line.rstrip("\n").split("\t")
            rows.append(run_row(a.root, sp, os.path.basename(ds), run, a.mapq))

    done = [r for r in rows if r["complete"]]
    incomplete = [r for r in rows if not r["complete"]]

    if a.format == "tsv":
        hdr = "species\tdataset\trun\tuf\taudit\trecords\tmapped\trate_pct\tbam_gb\tpaf_gb\twall_s"
        print(hdr + ("\taccepted\taccepted_pct" if a.mapq else ""))
        for r in rows:
            if not r["complete"]:
                print(f"{r['species']}\t{r['dataset']}\t{r['run']}\tINCOMPLETE\t"
                      f"{','.join(r['missing'])}\t-\t-\t-\t-\t-\t-")
                continue
            line = (f"{r['species']}\t{r['dataset']}\t{r['run']}\t{r['uf']}\t{r['audit_status']}\t"
                    f"{r['records']}\t{r['mapped']}\t{r['rate']:.2f}\t{r['bam_gb']:.2f}\t"
                    f"{r['paf_gb']:.2f}\t{r['seconds']}")
            if a.mapq and "accepted" in r:
                line += f"\t{r['accepted']}\t{r['accepted_pct']:.2f}"
            print(line)
    elif a.format == "md":
        print("| species | run | -uf | records | mapped | rate | BAM GB |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(done, key=lambda x: (x["species"], x["run"])):
            print(f"| {r['species']} | {r['run']} | {r['uf']} | {r['records']:,} | "
                  f"{r['mapped']:,} | {r['rate']:.2f} % | {r['bam_gb']:.1f} |")

    print(f"\n{len(done)}/{len(rows)} runs complete", file=sys.stderr)
    if done:
        tr = sum(r["records"] for r in done)
        tm = sum(r["mapped"] for r in done)
        gb = sum(r["bam_gb"] + r["paf_gb"] for r in done)
        uf = sum(1 for r in done if r["uf"] == "-uf")
        print(f"  {tr:,} primary records, {tm:,} mapped ({tm*100.0/tr:.2f} %)", file=sys.stderr)
        print(f"  -uf on {uf} runs, off on {len(done)-uf}", file=sys.stderr)
        print(f"  {gb:.0f} GB of BAM+PAF retained", file=sys.stderr)
        if a.format == "summary":
            print("per species:")
            agg = {}
            for r in done:
                s = agg.setdefault(r["species"], [0, 0, 0])
                s[0] += 1; s[1] += r["records"]; s[2] += r["mapped"]
            for sp in sorted(agg):
                n, rec, mp = agg[sp]
                print(f"  {sp:<14} {n:>3} runs  {rec:>12,} records  {mp*100.0/rec:6.2f} % mapped")
    if incomplete:
        print(f"\nincomplete ({len(incomplete)}):", file=sys.stderr)
        for r in incomplete[:20]:
            print(f"  {r['species']}/{r['run']}: missing {', '.join(r['missing'])}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
