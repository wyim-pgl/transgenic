#!/usr/bin/env python3
"""Score the AUGUSTUS posterior-sampling baseline at a candidate budget matched to TransGenic.

`28_score_added_isoforms.py` scores every structure AUGUSTUS proposes — 43,433 of them
against TransGenic's 1,103 — and the manuscript notes four times that precision and
candidate count trade off. AUGUSTUS writes a posterior probability into column 6 of every
`mRNA` row of the scored file, so the two tools can be compared at equal budget without
re-running anything. Script 28 parses only `CDS` rows and never reads column 6.

This applies script 28's definitions unchanged — an addition is a distinct CDS structure
that differs from the locus's first prediction; the reference is the alternative transcripts
of TAIR10 at the same locus with the primary removed, plus all AtRTD3 transcripts there;
matching is on exact CDS coordinates — and adds one step: rank the additions by posterior
and keep the top N. A posterior threshold sweep is reported alongside the budget sweep,
because a top-N cut can land inside a group of tied posteriors whereas a threshold cannot.

Recall denominators are held at the full scored-locus reference set (5,554 distinct TAIR10
alternative CDS structures across the 25,597 loci AUGUSTUS predicted at), so recall stays
comparable across budgets.

Usage:
    python 47_augustus_budget_match.py [--budgets 1103 2000 5000 20000] [--outdir ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"

AUGUSTUS = CMP / "standardized_results" / "A_thaliana_augustusSampling.gff3"
TAIR10 = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3 = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

# TransGenic completion mode, additions only (Table S4d) — the row this is matched against.
TRANSGENIC = {
    "loci_scored": 27413, "added_structures": 1103,
    "matched_TAIR10_alt": 200, "precision_vs_TAIR10_alt_pct": 18.1,
    "matched_AtRTD3": 204, "precision_vs_AtRTD3_pct": 18.5,
    "TAIR10_alt_structures": 5580, "recall_of_TAIR10_alt_pct": 3.6,
}

TN = re.compile(r"\.t(\d+)$")


def cds_by_transcript(path: Path) -> dict:
    """TAIR10/AtRTD3 GTF -> {gene: {transcript: sorted CDS intervals}} (script 28's parser)."""
    d: dict = defaultdict(lambda: defaultdict(list))
    lr = re.compile(r'gene_id "([^"]+)"')
    tr = re.compile(r'transcript_id "([^"]+)"')
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = lr.search(f[8]), tr.search(f[8])
            if not (g and t):
                continue
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            d[locus.replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    return {g: {k: tuple(sorted(v)) for k, v in tx.items()} for g, tx in d.items()}


def read_augustus(path: Path) -> tuple[dict, dict]:
    """-> ({locus: {transcript: CDS structure}}, {transcript: posterior from GFF column 6})."""
    raw: dict = defaultdict(lambda: defaultdict(list))
    posterior: dict[str, float] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "mRNA":
                m = re.search(r"ID=([^;]+)", f[8])
                if m and f[5] not in (".", ""):
                    posterior[m.group(1)] = float(f[5])
                continue
            if f[2] != "CDS":
                continue
            m = re.search(r"Parent=([^;]+)", f[8])
            if not m:
                continue
            tx = m.group(1)
            raw[tx.split(".t")[0].replace("augSmp_", "")][tx].append((int(f[3]), int(f[4])))
    pred = {locus: {k: tuple(sorted(v)) for k, v in txs.items()} for locus, txs in raw.items()}
    return pred, posterior


def chain(struct: tuple) -> tuple:
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def build_candidates(pred: dict, posterior: dict, ref: dict, art: dict,
                     primary: dict) -> tuple[list[dict], int, int]:
    """One record per distinct added CDS structure, with its posterior and its match flags."""
    candidates: list[dict] = []
    alt_total = 0
    for locus, txs in pred.items():
        order = sorted(txs, key=lambda k: int(TN.search(k).group(1)) if TN.search(k) else 0)
        supplied = txs[order[0]] if order else None

        # Collapse identical emissions to one proposed structure, as script 28 does, and
        # score it at the best posterior AUGUSTUS gave any emission of it.
        best: dict[tuple, float] = {}
        for tx, struct in txs.items():
            if struct == supplied:
                continue
            p = posterior.get(tx, 0.0)
            if struct not in best or p > best[struct]:
                best[struct] = p

        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        alt_chains = {chain(x) for x in alt_ref}
        art_here = set(art.get(locus, {}).values())
        alt_total += len(alt_ref)

        for struct, p in best.items():
            candidates.append({
                "locus": locus,
                "posterior": p,
                "struct": struct,
                "hit_tair10_alt": struct in alt_ref,
                "hit_tair10_chain": len(chain(struct)) >= 1 and chain(struct) in alt_chains,
                "hit_atrtd3": struct in art_here,
            })
    return candidates, alt_total, len(pred)


def score(subset: list[dict], alt_total: int, loci_scored: int) -> dict:
    added = len(subset)
    struct_hit = sum(1 for c in subset if c["hit_tair10_alt"])
    chain_hit = sum(1 for c in subset if c["hit_tair10_chain"])
    art_hit = sum(1 for c in subset if c["hit_atrtd3"])
    recovered = {(c["locus"], c["struct"]) for c in subset if c["hit_tair10_alt"]}
    loci_with = len({c["locus"] for c in subset})
    return {
        "loci_scored": loci_scored,
        "loci_with_at_least_one_addition": loci_with,
        "added_structures": added,
        "reference_alternative_structures": alt_total,
        "matched_TAIR10_alt_exact_CDS": struct_hit,
        "matched_TAIR10_alt_intron_chain": chain_hit,
        "matched_AtRTD3": art_hit,
        "precision_vs_TAIR10_alt_pct": round(100 * struct_hit / added, 1) if added else None,
        "precision_vs_AtRTD3_pct": round(100 * art_hit / added, 1) if added else None,
        "recall_of_TAIR10_alt_pct": round(100 * len(recovered) / alt_total, 1) if alt_total else None,
    }


def tie_bounds(ordered: list[dict], budget: int, above: int, tied: int,
               flag: str) -> list[float]:
    """Worst- and best-case precision over every way of breaking the tie at the cut."""
    fixed = sum(1 for c in ordered[:above] if c[flag])
    slots = budget - above
    tie_hits = sum(1 for c in ordered[above:above + tied] if c[flag])
    lo = fixed + max(0, slots - (tied - tie_hits))
    hi = fixed + min(slots, tie_hits)
    return [round(100 * lo / budget, 1), round(100 * hi / budget, 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1103, 2000, 5000, 10000, 20000])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.4, 0.3, 0.2, 0.1])
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "transgenic" / "revision" / "results" / "baselines")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("  reading AUGUSTUS predictions ...", file=sys.stderr)
    pred, posterior = read_augustus(AUGUSTUS)
    print("  reading TAIR10 / AtRTD3 ...", file=sys.stderr)
    ref = cds_by_transcript(TAIR10)
    art = cds_by_transcript(ATRTD3)
    primary = {}
    for line in PRIMARY_IDS.read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    candidates, alt_total, loci_scored = build_candidates(pred, posterior, ref, art, primary)
    missing_posterior = sum(1 for c in candidates if c["posterior"] == 0.0
                            and c["locus"] not in ())
    print(f"  additions: {len(candidates)}  loci: {loci_scored}  "
          f"reference alternatives: {alt_total}", file=sys.stderr)

    # Deterministic order: posterior descending, then locus and coordinates, so that a
    # budget cut falling inside a group of tied posteriors is reproducible.
    ordered = sorted(candidates, key=lambda c: (-c["posterior"], c["locus"], c["struct"]))

    full = score(ordered, alt_total, loci_scored)
    budget_rows = []
    for n in sorted(set(args.budgets)):
        if n > len(ordered):
            continue
        row = score(ordered[:n], alt_total, loci_scored)
        cut_p = ordered[n - 1]["posterior"]
        tied = sum(1 for c in ordered if c["posterior"] == cut_p)
        above = sum(1 for c in ordered if c["posterior"] > cut_p)
        row.update({"budget": n, "posterior_at_cut": cut_p,
                    "candidates_tied_at_cut": tied, "candidates_strictly_above_cut": above})
        # The cut lands inside a group of tied posteriors, so the reported value depends on
        # an arbitrary choice. Bound what *any* tie-breaking rule could give: fill the
        # remaining slots with the worst, then the best, of the tied candidates.
        row["tie_sensitivity"] = {
            key: tie_bounds(ordered, n, above, tied, flag)
            for key, flag in (("precision_vs_TAIR10_alt_pct", "hit_tair10_alt"),
                              ("precision_vs_AtRTD3_pct", "hit_atrtd3"))
        }
        budget_rows.append(row)
    budget_rows.append({**full, "budget": len(ordered), "posterior_at_cut": ordered[-1]["posterior"],
                        "candidates_tied_at_cut": None, "candidates_strictly_above_cut": None,
                        "note": "as published (Table S4d)"})

    threshold_rows = []
    for t in sorted(set(args.thresholds), reverse=True):
        subset = [c for c in ordered if c["posterior"] >= t]
        if not subset:
            continue
        threshold_rows.append({**score(subset, alt_total, loci_scored), "posterior_threshold": t})

    published = {"matched_TAIR10_alt_exact_CDS": 574, "matched_TAIR10_alt_intron_chain": 1088,
                 "matched_AtRTD3": 4409, "added_structures": 43433, "loci_scored": 25597,
                 "reference_alternative_structures": 5554}
    agrees = all(full[k] == v for k, v in published.items())

    out = {
        "analysis": "AUGUSTUS posterior-ranked additions at a matched candidate budget",
        "date": date.today().isoformat(),
        "definitions": (
            "script 28 unchanged (addition = distinct CDS structure differing from the "
            "locus's first prediction; TAIR10 primary removed from the reference; exact CDS "
            "match), plus a ranking on the posterior in GFF column 6 of each mRNA row"),
        "structure_posterior_rule": "maximum posterior over the emissions sharing that CDS structure",
        "tie_handling": ("budget cuts break posterior ties by locus then coordinates; the "
                         "posterior threshold sweep is tie-free and is the check on this"),
        "recall_denominator": ("distinct TAIR10 alternative CDS structures across all scored "
                               "loci, held constant across budgets"),
        "additions_with_posterior": len(candidates) - missing_posterior,
        "budget_sweep": budget_rows,
        "posterior_threshold_sweep": threshold_rows,
        "transgenic_completion_additions_only": TRANSGENIC,
        "full_set_reproduces_TableS4d": agrees,
        "full_set_published_values": published,
        "inputs": {"augustus": str(AUGUSTUS), "TAIR10": str(TAIR10), "AtRTD3": str(ATRTD3),
                   "primary_transcript_ids": str(PRIMARY_IDS)},
    }
    dst = args.outdir / "augustus_budget_match.json"
    dst.write_text(json.dumps(out, indent=1))

    csv_dst = args.outdir / "augustus_budget_sweep.csv"
    with csv_dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Prediction set", "Candidate budget (n)", "Posterior at cut",
                    "Matched TAIR10 alt (n)", "Precision vs TAIR10 alt (%)",
                    "Matched AtRTD3 (n)", "Precision vs AtRTD3 (%)",
                    "Recall of TAIR10 alt (%)"])
        for r in budget_rows:
            w.writerow(["AUGUSTUS v3.5.0 posterior sampling, posterior-ranked",
                        r["budget"], r["posterior_at_cut"],
                        r["matched_TAIR10_alt_exact_CDS"], r["precision_vs_TAIR10_alt_pct"],
                        r["matched_AtRTD3"], r["precision_vs_AtRTD3_pct"],
                        r["recall_of_TAIR10_alt_pct"]])
        w.writerow(["TransGenic 400M, reference-prompted (additions only)",
                    TRANSGENIC["added_structures"], "n/a",
                    TRANSGENIC["matched_TAIR10_alt"], TRANSGENIC["precision_vs_TAIR10_alt_pct"],
                    TRANSGENIC["matched_AtRTD3"], TRANSGENIC["precision_vs_AtRTD3_pct"],
                    TRANSGENIC["recall_of_TAIR10_alt_pct"]])

    hdr = f"{'budget':>8} {'p@cut':>7} {'TAIR10-alt%':>12} {'AtRTD3%':>9} {'TAIR10-alt recall%':>19}"
    print(hdr, file=sys.stderr)
    for r in budget_rows:
        print(f"{r['budget']:>8} {str(r['posterior_at_cut']):>7} "
              f"{r['precision_vs_TAIR10_alt_pct']:>12} {r['precision_vs_AtRTD3_pct']:>9} "
              f"{r['recall_of_TAIR10_alt_pct']:>19}", file=sys.stderr)
    for r in threshold_rows:
        print(f"  p>={r['posterior_threshold']}: n={r['added_structures']} "
              f"TAIR10-alt {r['precision_vs_TAIR10_alt_pct']}% "
              f"AtRTD3 {r['precision_vs_AtRTD3_pct']}%", file=sys.stderr)
    print(f"  full set reproduces Table S4d: {agrees}", file=sys.stderr)
    print(f"written: {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
