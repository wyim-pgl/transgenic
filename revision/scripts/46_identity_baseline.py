#!/usr/bin/env python3
"""Score the identity (do-nothing) baseline with the authors' own transcript-level protocol.

Completion mode returns the supplied primary transcript almost without exception, so the
whole-annotation transcript-level figures quoted in the Results describe a file that is
95.4% prompt by construction. The control that number needs is the null prediction: hand
back the 27,413 supplied TAIR10 primary transcripts unchanged and score them the same way.

The prediction is built from the authors' own `primary_transcript_ids.txt` and `TAIR10.gtf`,
restricted to the 27,413 loci at which completion-mode predictions were generated (read from
`A_thaliana_transgenic400Mprompt_beam1.gff3`). Scoring reproduces the convention defined in
Methods and implemented in `02_gffcompare_analysis.py`:

    precision = query transcripts with GFFCompare class code '=' / all query transcripts
    recall    = distinct reference transcripts matched by '=' / released reference total
                (35,386 for TAIR10, 169,499 for AtRTD3; not GFFCompare's post-discard total)
    F1        = harmonic mean of the two

Usage:
    python 46_identity_baseline.py [--outdir revision/results/baselines]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"
GFFCOMPARE = Path("/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare")

PREDICTION = CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3"
TAIR10 = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3 = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

# Released reference totals, the denominators the Results re-base recall to.
REFERENCE_TOTALS = {"TAIR10": 35386, "AtRTD3": 169499}

# Completion mode, as published in Table S4b (same protocol, same binary).
COMPLETION = {
    "TAIR10": {"predicted": 29922, "matching_queries": 22260, "duplicates": 341,
               "distinct_ref_matched": 21919, "recall_pct": 61.9, "precision_pct": 74.4,
               "f1_pct": 67.6},
    "AtRTD3": {"predicted": 29922, "matching_queries": 21694, "duplicates": 442,
               "distinct_ref_matched": 21252, "recall_pct": 12.5, "precision_pct": 72.5,
               "f1_pct": 21.4},
}

GM = re.compile(r"GM=([^;\s]+)")
TX = re.compile(r'transcript_id "([^"]+)"')


def evaluation_loci(path: Path) -> set[str]:
    """Loci at which completion-mode predictions were generated, from the GM= tag."""
    loci: set[str] = set()
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = GM.search(f[8])
            if not m:
                continue
            gm = m.group(1)
            if gm.endswith("-rc"):
                continue
            loci.add(gm.replace(".TAIR10", ""))
    return loci


def primary_of_locus(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        tid = line.strip()
        if tid:
            out[tid.split(".")[0]] = tid
    return out


def write_echo(loci: set[str], primary: dict[str, str], dst: Path) -> tuple[int, int]:
    """Copy each locus's primary transcript out of TAIR10.gtf verbatim."""
    wanted = {primary[g] for g in loci if g in primary}
    written: set[str] = set()
    rows = 0
    with TAIR10.open() as src, dst.open("w") as out:
        for line in src:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9:
                continue
            m = TX.search(f[8])
            if not m or m.group(1) not in wanted:
                continue
            out.write(line)
            written.add(m.group(1))
            rows += 1
    missing = sorted(g for g in loci if g not in primary)
    if missing:
        print(f"  WARNING: {len(missing)} evaluation loci have no primary id "
              f"(e.g. {missing[:3]})", file=sys.stderr)
    return len(written), rows


def run_gffcompare(query: Path, ref: Path, prefix: Path) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(GFFCOMPARE), "-r", str(ref), "-o", str(prefix), str(query)],
                   check=True, capture_output=True)
    tmaps = list(prefix.parent.glob(f"{prefix.name}.*.tmap"))
    if not tmaps:
        raise SystemExit(f"no .tmap produced for {prefix}")
    return tmaps[0]


def score_tmap(tmap: Path, total_reference: int) -> dict:
    total_predicted = 0
    exact = 0
    matched_refs: set[str] = set()
    with tmap.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            total_predicted += 1
            if row["class_code"] == "=":
                exact += 1
                matched_refs.add(row["ref_id"])
    precision = exact / total_predicted if total_predicted else 0.0
    recall = len(matched_refs) / total_reference if total_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_reference_released": total_reference,
        "total_predicted": total_predicted,
        "matching_queries_exact": exact,
        "duplicate_exact_matches": exact - len(matched_refs),
        "distinct_ref_matched": len(matched_refs),
        "recall_pct": round(100 * recall, 1),
        "precision_pct": round(100 * precision, 1),
        "f1_pct": round(100 * f1, 1),
    }


def raw_stats(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    pat = re.compile(r"^\s*([A-Za-z ]+?) level:\s*([\d.-]+)\s*\|\s*([\d.-]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1).strip() + "_Sn"] = float(m.group(2))
            out[m.group(1).strip() + "_Pr"] = float(m.group(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "transgenic" / "revision" / "results" / "baselines")
    args = ap.parse_args()
    assert GFFCOMPARE.exists(), f"gffcompare v0.12.6 not found at {GFFCOMPARE}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    loci = evaluation_loci(PREDICTION)
    primary = primary_of_locus(PRIMARY_IDS)
    echo = args.outdir / "identity_baseline_primaries.gtf"
    n_tx, n_rows = write_echo(loci, primary, echo)
    print(f"  evaluation loci                {len(loci)}", file=sys.stderr)
    print(f"  echoed primary transcripts     {n_tx} ({n_rows} GTF rows)", file=sys.stderr)

    results = {}
    for name, ref in (("TAIR10", TAIR10), ("AtRTD3", ATRTD3)):
        prefix = args.outdir / f"identity_vs_{name}"
        print(f"  gffcompare vs {name} ...", file=sys.stderr)
        tmap = run_gffcompare(echo, ref, prefix)
        scored = score_tmap(tmap, REFERENCE_TOTALS[name])
        scored["gffcompare_raw"] = raw_stats(prefix.with_suffix(".stats"))
        results[name] = scored

    comparison = {}
    for name in ("TAIR10", "AtRTD3"):
        i, c = results[name], COMPLETION[name]
        comparison[name] = {
            "identity_recall_pct": i["recall_pct"],
            "completion_recall_pct": c["recall_pct"],
            "delta_recall_pp": round(c["recall_pct"] - i["recall_pct"], 1),
            "identity_precision_pct": i["precision_pct"],
            "completion_precision_pct": c["precision_pct"],
            "delta_precision_pp": round(c["precision_pct"] - i["precision_pct"], 1),
            "identity_f1_pct": i["f1_pct"],
            "completion_f1_pct": c["f1_pct"],
            "delta_f1_pp": round(c["f1_pct"] - i["f1_pct"], 1),
        }

    out = {
        "analysis": "identity (echo-the-prompt) transcript-level baseline",
        "date": date.today().isoformat(),
        "gffcompare": "v0.12.6",
        "gffcompare_binary": str(GFFCOMPARE),
        "prediction_is": "the TAIR10 primary transcript of each evaluation locus, verbatim",
        "evaluation_loci": len(loci),
        "echoed_transcripts": n_tx,
        "scoring_convention": (
            "precision = queries with class code '=' / all queries; "
            "recall = distinct reference transcripts matched / released reference total "
            "(35,386 TAIR10; 169,499 AtRTD3); F1 = harmonic mean. Identical to "
            "02_gffcompare_analysis.py and to the Methods definition."),
        "identity_baseline": results,
        "completion_mode_published": COMPLETION,
        "identity_vs_completion": comparison,
        "inputs": {
            "prediction_source_for_locus_set": str(PREDICTION),
            "primary_transcript_ids": str(PRIMARY_IDS),
            "TAIR10": str(TAIR10),
            "AtRTD3": str(ATRTD3),
        },
    }
    dst = args.outdir / "identity_baseline.json"
    dst.write_text(json.dumps(out, indent=1))

    csv_dst = args.outdir / "identity_baseline.csv"
    with csv_dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Reference", "Prediction set", "Reference transcripts (n)",
                    "Predicted transcripts (n)", "Exact matches (n)", "Duplicate matches (n)",
                    "Distinct reference transcripts matched (n)", "Isoform recall (%)",
                    "Isoform precision (%)", "Isoform F1 (%)"])
        for name in ("TAIR10", "AtRTD3"):
            i, c = results[name], COMPLETION[name]
            w.writerow([name, "Identity baseline (supplied primary returned unchanged)",
                        i["total_reference_released"], i["total_predicted"],
                        i["matching_queries_exact"], i["duplicate_exact_matches"],
                        i["distinct_ref_matched"], i["recall_pct"], i["precision_pct"],
                        i["f1_pct"]])
            w.writerow([name, "TransGenic 400M, reference-prompted (deduplicated)",
                        REFERENCE_TOTALS[name], c["predicted"], c["matching_queries"],
                        c["duplicates"], c["distinct_ref_matched"], c["recall_pct"],
                        c["precision_pct"], c["f1_pct"]])
    print(json.dumps(comparison, indent=1), file=sys.stderr)
    print(f"written: {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
