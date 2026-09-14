#!/usr/bin/env python3
"""Assign the B6 locus panels (A46; work plan B6 §2 item 10; MVP pilot §1.2) from the eligible pool.

Inputs
  --eligible   revision/protocols/b6_eligible_loci_v1.tsv from `71_b6_panels.py` (every
               eligible A. thaliana strict-held-out locus with its 1-based `shuffle_rank`,
               seed 20260914). If `<tsv>.sha256` exists beside it, the file is verified.
  --est-counts TSV with columns gene_id, est_molecules, produced on pronghorn from the
               screened A-EST BAM by `72a_est_molecules_per_locus.sh` (distinct primary
               placements — chromosome, FLAG-0x10 strand, POS, CIGAR — after samtools view
               -F 0x904, overlapping the locus; MVP §1.3 rule (a)).

Selection, walking the pool in `shuffle_rank` order:
  MVP   the first --mvp loci (default 50) that pass the MVP sub-filter of
        B6_MVP_PILOT_v1.md §1.2: length <= --max-len nt (default 20,000) AND
        est_molecules >= --min-est (default 3). The first --core of them (default 20)
        are `mvp_core`, the rest `mvp`. Loci walked but failing the sub-filter stay in
        the pool for the pilot/final split.
  pilot the first --pilot (default 300) non-MVP loci by shuffle_rank
  final every remaining locus
The four panel labels are disjoint and exhaust the pool. `est_molecules` is filled for
MVP rows (mvp_core and mvp) only and left empty elsewhere. A locus that must be
consulted but is missing from --est-counts is an error (loci longer than --max-len
never need a count). Deterministic: sorted inputs, no `hash()`, no randomness.

Coordinates are 1-based inclusive (`start`, `fin`, as in the input).

Usage:
  python revision/scripts/72_mvp_select_loci.py \\
      --eligible revision/protocols/b6_eligible_loci_v1.tsv \\
      --est-counts revision/protocols/b6_est_molecules_v1.tsv \\
      --out revision/protocols/b6_panels_v1.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

IN_COLUMNS = ("shuffle_rank", "species_id", "gene_id", "chromosome", "start", "fin", "strand", "length",
              "orthogroup_id", "n_transcripts", "exon_count", "intron_count")
OUT_COLUMNS = ("panel",) + IN_COLUMNS + ("est_molecules",)
PANELS = ("mvp_core", "mvp", "pilot", "final")


class SelectError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_eligible(path: Path) -> list:
    sha_file = Path(str(path) + ".sha256")
    if sha_file.exists():
        recorded = sha_file.read_text().split()[0]
        actual = digest(path)
        if recorded != actual:
            raise SelectError(f"{path} sha256 {actual} != recorded {recorded}")
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if tuple(reader.fieldnames or ()) != IN_COLUMNS:
            raise SelectError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    for r in rows:
        for k in ("shuffle_rank", "start", "fin", "length", "n_transcripts", "exon_count", "intron_count"):
            r[k] = int(r[k])
        if r["length"] != r["fin"] - r["start"] + 1:
            raise SelectError(f"{r['gene_id']}: length {r['length']} != fin - start + 1")
    rows.sort(key=lambda r: r["shuffle_rank"])
    if [r["shuffle_rank"] for r in rows] != list(range(1, len(rows) + 1)):
        raise SelectError("shuffle_rank is not a contiguous 1-based sequence")
    ids = [r["gene_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SelectError("duplicate gene_id in the eligible pool")
    return rows


def read_est_counts(path: Path) -> dict:
    counts: dict = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or reader.fieldnames[:2] != ["gene_id", "est_molecules"]:
            raise SelectError(f"--est-counts must have columns gene_id, est_molecules; got {reader.fieldnames}")
        for r in reader:
            if r["gene_id"] in counts:
                raise SelectError(f"duplicate gene_id in --est-counts: {r['gene_id']}")
            counts[r["gene_id"]] = int(r["est_molecules"])
    return counts


def select_panels(pool: list, counts: dict, mvp: int, core: int, pilot: int, max_len: int, min_est: int, report: list) -> list:
    """Label every pool row; `pool` must already be sorted by shuffle_rank."""
    if core > mvp:
        raise SelectError(f"core ({core}) > mvp ({mvp})")
    mvp_ids: list = []
    walked = n_too_long = n_low_est = 0
    for r in pool:
        if len(mvp_ids) >= mvp:
            break
        walked += 1
        if r["length"] > max_len:
            n_too_long += 1
            continue
        if r["gene_id"] not in counts:
            raise SelectError(f"--est-counts has no entry for {r['gene_id']} (shuffle_rank {r['shuffle_rank']})")
        if counts[r["gene_id"]] < min_est:
            n_low_est += 1
            continue
        mvp_ids.append(r["gene_id"])
    if len(mvp_ids) < mvp:
        raise SelectError(f"only {len(mvp_ids)} loci pass the MVP sub-filter (need {mvp}); pool exhausted")
    label = {gid: ("mvp_core" if i < core else "mvp") for i, gid in enumerate(mvp_ids)}
    report.append(f"[mvp] walked {walked} loci in shuffle_rank order (last walked shuffle_rank {pool[walked - 1]['shuffle_rank']}): "
                  f"{len(mvp_ids)} selected ({core} mvp_core), {n_too_long} dropped for length > {max_len}, "
                  f"{n_low_est} dropped for est_molecules < {min_est}")
    out, n_pilot = [], 0
    for r in pool:
        if r["gene_id"] in label:
            panel, est = label[r["gene_id"]], counts[r["gene_id"]]
        elif n_pilot < pilot:
            panel, est, n_pilot = "pilot", "", n_pilot + 1
        else:
            panel, est = "final", ""
        row = {k: r[k] for k in IN_COLUMNS}
        row.update({"panel": panel, "est_molecules": est})
        out.append(row)
    tally = {p: sum(1 for r in out if r["panel"] == p) for p in PANELS}
    tally["total"] = len(out)
    if tally["pilot"] < pilot:
        raise SelectError(f"pool too small for {pilot} pilot loci after MVP: {tally}")
    report.append("panel_counts=" + json.dumps(tally, sort_keys=True))
    return out


def write_tsv(rows, columns, path: Path) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{sha}  {path.name}\n")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eligible", default="revision/protocols/b6_eligible_loci_v1.tsv", help="pool from 71_b6_panels.py")
    ap.add_argument("--est-counts", required=True, type=Path, help="TSV gene_id, est_molecules from 72a_est_molecules_per_locus.sh")
    ap.add_argument("--out", default="revision/protocols/b6_panels_v1.tsv", help="panel TSV; <out>.sha256 is written beside it")
    ap.add_argument("--mvp", type=int, default=50, help="MVP panel size")
    ap.add_argument("--core", type=int, default=20, help="first N MVP loci labelled mvp_core")
    ap.add_argument("--pilot", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=20000, help="MVP sub-filter: gene length <= this (nt)")
    ap.add_argument("--min-est", type=int, default=3, help="MVP sub-filter: est_molecules >= this")
    args = ap.parse_args(argv)

    report: list = []
    try:
        eligible = Path(args.eligible)
        pool = read_eligible(eligible)
        report.append(f"eligible={eligible} sha256={digest(eligible)} rows={len(pool)}")
        counts = read_est_counts(args.est_counts)
        report.append(f"est_counts={args.est_counts} sha256={digest(args.est_counts)} entries={len(counts)}")
        report.append(f"mvp={args.mvp} core={args.core} pilot={args.pilot} max_len={args.max_len} min_est={args.min_est}")
        rows = select_panels(pool, counts, args.mvp, args.core, args.pilot, args.max_len, args.min_est, report)
        out = Path(args.out)
        sha = write_tsv(rows, OUT_COLUMNS, out)
        report.append(f"wrote {out} tsv_sha256={sha} -> {out}.sha256")
        report.append(f"script_sha256={digest(Path(__file__))} ({Path(__file__).name})")
    except SelectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
