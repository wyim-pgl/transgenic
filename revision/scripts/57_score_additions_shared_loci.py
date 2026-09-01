#!/usr/bin/env python3
"""Score TransGenic and AUGUSTUS additions on the loci at which BOTH tools predicted.

WHY THIS EXISTS

`28_score_added_isoforms.py` scores the two tools on different locus sets: TransGenic on
the 27,413 loci it predicted at (reference alternatives: 5,580) and AUGUSTUS on the 25,597
loci it returned a prediction for (5,554). The precision and recall figures in Table S4d
therefore have different denominators, which a reader can reasonably object to. This
script restricts both tools to the intersection of their predicted loci and scores every
number against ONE recall denominator: the distinct TAIR10 alternative CDS structures at
the shared loci.

DEFINITIONS (script 28, unchanged; the functions are imported from it, not copied)

    addition          a distinct predicted CDS structure at a locus that differs from the
                      supplied structure — TAIR10's curated primary for TransGenic, its own
                      first (.t1) prediction for AUGUSTUS; identical emissions collapse
    TAIR10 alternative any TAIR10 transcript at the locus other than the curated primary
    AtRTD3 match      equality with any AtRTD3 transcript at the locus
    matching          exact CDS coordinates; the CDS intron chain is reported alongside

The budget-matched AUGUSTUS row reuses `47_augustus_budget_match.py`'s ranking: each added
structure takes the best posterior AUGUSTUS gave any emission of it (GFF column 6 of the
mRNA row), additions are ordered by posterior descending with locus and coordinates as
deterministic tie-breakers, and the top N are kept where N is TransGenic's addition count
on the shared loci. Tie sensitivity bounds are reported as in script 47.

SANITY CHECKS (the script refuses to write results if either fails)

    1. on all TransGenic loci the re-implementation must return exactly the published
       1,103 / 200 / 204 / 5,580 (added / TAIR10-alt / AtRTD3 / reference alternatives)
    2. on all AUGUSTUS loci it must return exactly 43,433 / 574 / 4,409 / 5,554

Usage:
    python 57_score_additions_shared_loci.py [--outdir revision/results/baselines]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
s28 = importlib.import_module("28_score_added_isoforms")
s47 = importlib.import_module("47_augustus_budget_match")

ROOT = HERE.parents[2]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"

TRANSGENIC_PRED = CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3"
AUGUSTUS_PRED = CMP / "standardized_results" / "A_thaliana_augustusSampling.gff3"
TAIR10 = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3 = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

PUBLISHED = {
    "transgenic": {"loci_scored": 27413, "added_structures": 1103,
                   "matched_TAIR10_alt_exact_CDS": 200, "matched_TAIR10_alt_intron_chain": 343,
                   "matched_AtRTD3": 204, "reference_alternative_structures": 5580},
    "augustus": {"loci_scored": 25597, "added_structures": 43433,
                 "matched_TAIR10_alt_exact_CDS": 574, "matched_TAIR10_alt_intron_chain": 1088,
                 "matched_AtRTD3": 4409, "reference_alternative_structures": 5554},
}

TN = re.compile(r"\.t(\d+)$")


def load_primary(path: Path) -> dict[str, str]:
    primary: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()
    return primary


def first_prediction(txs: dict) -> tuple | None:
    """AUGUSTUS's own first (.t1) prediction stands in for the prompt (script 28)."""
    order = sorted(txs, key=lambda k: int(TN.search(k).group(1)) if TN.search(k) else 0)
    return txs[order[0]] if order else None


def locus_records(pred: dict, ref: dict, art: dict, primary: dict,
                  augustus: bool) -> dict:
    """locus -> (added structures, TAIR10 alternative structures, AtRTD3 structures).

    This is the body of script 28's per-locus loop, kept per locus so that any subset of
    loci can be summed without re-parsing.
    """
    records: dict = {}
    for locus, txs in pred.items():
        supplied = first_prediction(txs) if augustus else ref.get(locus, {}).get(primary.get(locus))
        additions = {s for s in txs.values() if s != supplied}
        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        art_here = set(art.get(locus, {}).values())
        records[locus] = (frozenset(additions), frozenset(alt_ref), frozenset(art_here))
    return records


def score(records: dict, loci, denominator: int | None = None) -> dict:
    """Sum script 28's counters over `loci`. `denominator` overrides the recall base."""
    loci = list(loci)
    added = struct_hit = chain_hit = art_hit = alt_total = recovered = loci_with = 0
    for locus in loci:
        additions, alt_ref, art_here = records[locus]
        alt_chains = {s28.chain(x) for x in alt_ref}
        alt_total += len(alt_ref)
        if additions:
            loci_with += 1
        added += len(additions)
        struct_hit += sum(1 for s in additions if s in alt_ref)
        chain_hit += sum(1 for s in additions
                         if len(s28.chain(s)) >= 1 and s28.chain(s) in alt_chains)
        art_hit += sum(1 for s in additions if s in art_here)
        recovered += len(alt_ref & additions)
    denom = alt_total if denominator is None else denominator
    return {
        "loci_scored": len(loci),
        "loci_with_at_least_one_addition": loci_with,
        "added_structures": added,
        "reference_alternative_structures": denom,
        "matched_TAIR10_alt_exact_CDS": struct_hit,
        "matched_TAIR10_alt_intron_chain": chain_hit,
        "matched_AtRTD3": art_hit,
        "precision_vs_TAIR10_alt_pct": round(100 * struct_hit / added, 1) if added else None,
        "precision_vs_AtRTD3_pct": round(100 * art_hit / added, 1) if added else None,
        "recall_of_TAIR10_alt_pct": round(100 * recovered / denom, 1) if denom else None,
        "TAIR10_alt_recovered": recovered,
    }


def check(name: str, got: dict, want: dict) -> bool:
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        print(f"  SANITY FAIL {name}: {bad}", file=sys.stderr)
        return False
    print(f"  sanity ok    {name}: {', '.join(f'{k}={v}' for k, v in want.items())}",
          file=sys.stderr)
    return True


def budget_matched_augustus(pred: dict, posterior: dict, ref: dict, art: dict,
                            primary: dict, loci: set, budget: int,
                            denominator: int) -> tuple[dict, dict]:
    """Script 47's ranking applied to the AUGUSTUS additions at `loci` only."""
    sub_pred = {locus: txs for locus, txs in pred.items() if locus in loci}
    candidates, alt_total, loci_scored = s47.build_candidates(sub_pred, posterior, ref, art, primary)
    if alt_total != denominator:
        raise RuntimeError(f"script 47 denominator {alt_total} != shared denominator {denominator}")
    ordered = sorted(candidates, key=lambda c: (-c["posterior"], c["locus"], c["struct"]))
    full = s47.score(ordered, alt_total, loci_scored)
    row = s47.score(ordered[:budget], alt_total, loci_scored)
    cut_p = ordered[budget - 1]["posterior"]
    tied = sum(1 for c in ordered if c["posterior"] == cut_p)
    above = sum(1 for c in ordered if c["posterior"] > cut_p)
    row.update({
        "budget": budget, "posterior_at_cut": cut_p,
        "candidates_tied_at_cut": tied, "candidates_strictly_above_cut": above,
        "tie_sensitivity": {
            key: s47.tie_bounds(ordered, budget, above, tied, flag)
            for key, flag in (("precision_vs_TAIR10_alt_pct", "hit_tair10_alt"),
                              ("precision_vs_AtRTD3_pct", "hit_atrtd3"))},
    })
    return row, full


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "transgenic" / "revision" / "results" / "baselines")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("  reading predictions and references ...", file=sys.stderr)
    tg_pred = s28.cds_by_transcript(TRANSGENIC_PRED, True)
    aug_pred, posterior = s47.read_augustus(AUGUSTUS_PRED)
    ref = s28.cds_by_transcript(TAIR10, False)
    art = s28.cds_by_transcript(ATRTD3, False)
    primary = load_primary(PRIMARY_IDS)

    tg_rec = locus_records(tg_pred, ref, art, primary, augustus=False)
    aug_rec = locus_records(aug_pred, ref, art, primary, augustus=True)

    # --- sanity 1 and 2: reproduce Table S4d on each tool's own locus set -------------
    tg_full = score(tg_rec, tg_rec)
    aug_full = score(aug_rec, aug_rec)
    ok = check("TransGenic, all its loci (script 28)", tg_full, PUBLISHED["transgenic"])
    ok &= check("AUGUSTUS, all its loci (script 28 --augustus)", aug_full, PUBLISHED["augustus"])
    if not ok:
        print("  refusing to write results: re-implementation does not reproduce script 28",
              file=sys.stderr)
        return 1

    # --- the locus sets ----------------------------------------------------------------
    tg_loci, aug_loci = set(tg_rec), set(aug_rec)
    shared = tg_loci & aug_loci
    only_tg, only_aug = tg_loci - aug_loci, aug_loci - tg_loci
    denominator = sum(len(tg_rec[l][1]) for l in shared)
    assert denominator == sum(len(aug_rec[l][1]) for l in shared)
    print(f"  loci: TransGenic {len(tg_loci)}  AUGUSTUS {len(aug_loci)}  shared {len(shared)}  "
          f"TransGenic-only {len(only_tg)}  AUGUSTUS-only {len(only_aug)}  "
          f"shared-locus TAIR10 alternatives {denominator}", file=sys.stderr)

    tg_shared = score(tg_rec, shared, denominator)
    aug_shared = score(aug_rec, shared, denominator)
    aug_budget, aug_full_47 = budget_matched_augustus(
        aug_pred, posterior, ref, art, primary, shared, tg_shared["added_structures"], denominator)
    # Script 47's candidate builder and this script's per-locus records must agree.
    for k in ("added_structures", "matched_TAIR10_alt_exact_CDS", "matched_AtRTD3",
              "matched_TAIR10_alt_intron_chain"):
        if aug_full_47[k] != aug_shared[k]:
            raise RuntimeError(f"script 47 candidates disagree on shared loci: {k} "
                               f"{aug_full_47[k]} != {aug_shared[k]}")
    print("  sanity ok    script 47 candidate builder agrees with per-locus records on "
          "shared loci", file=sys.stderr)

    # The dropped loci, so the reader can see what restricting to the intersection removes.
    tg_only_rows = score(tg_rec, only_tg)
    aug_only_rows = score(aug_rec, only_aug)

    out = {
        "analysis": "additions scored on the loci at which both TransGenic and AUGUSTUS predicted",
        "date": date.today().isoformat(),
        "definitions": ("script 28 unchanged: addition = distinct CDS structure differing from "
                        "the supplied structure (TAIR10 curated primary for TransGenic; own .t1 "
                        "prediction for AUGUSTUS); reference = TAIR10 alternative transcripts "
                        "at the locus with the primary removed, and separately all AtRTD3 "
                        "transcripts at the locus; exact CDS coordinate match, intron-chain "
                        "match reported alongside"),
        "locus_key": "TAIR10 gene id (GM= on TransGenic CDS rows; augSmp_<gene> transcript ids for AUGUSTUS)",
        "recall_denominator": ("distinct TAIR10 alternative CDS structures at the shared loci, "
                               "the same number for every row below"),
        "loci": {
            "transgenic_predicted": len(tg_loci),
            "augustus_predicted": len(aug_loci),
            "shared": len(shared),
            "transgenic_only": len(only_tg),
            "augustus_only": len(only_aug),
            "shared_TAIR10_alternative_structures": denominator,
            "note": ("The manuscript's '25,641 of 27,415' for AUGUSTUS is gffcompare's query-locus "
                     "count (transcript clusters; altonly/A_thaliana_augustusSampling_vs_TAIR10.stats), "
                     "whereas script 28 keys on gene ids, of which the standardized AUGUSTUS file "
                     "holds 25,597; 27,415 - 25,597 = 1,818 loci had no AUGUSTUS prediction. "
                     "TransGenic's file holds 27,413 gene ids."),
        },
        "shared_loci": {
            "transgenic400Mprompt_beam1": tg_shared,
            "augustusSampling_all_additions": aug_shared,
            "augustusSampling_budget_matched": aug_budget,
        },
        "dropped_loci": {
            "transgenic_only_loci": tg_only_rows,
            "augustus_only_loci": aug_only_rows,
        },
        "full_sets_as_published": {"transgenic400Mprompt_beam1": tg_full,
                                   "augustusSampling": aug_full},
        "sanity": {"transgenic_full_reproduces_TableS4d": True,
                   "augustus_full_reproduces_TableS4d": True,
                   "script47_ranking_agrees_on_shared_loci": True},
        "inputs": {"transgenic": str(TRANSGENIC_PRED), "augustus": str(AUGUSTUS_PRED),
                   "TAIR10": str(TAIR10), "AtRTD3": str(ATRTD3),
                   "primary_transcript_ids": str(PRIMARY_IDS)},
    }
    dst = args.outdir / "additions_shared_loci.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")

    csv_dst = args.outdir / "additions_shared_loci.csv"
    with csv_dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Prediction set", "Loci scored", "Added structures (n)",
                    "Matched TAIR10 alt (n)", "Precision vs TAIR10 alt (%)",
                    "Matched AtRTD3 (n)", "Precision vs AtRTD3 (%)",
                    "TAIR10 alt structures (denominator)", "Recall of TAIR10 alt (%)"])
        for name, r in (("TransGenic 400M, reference-prompted (additions only)", tg_shared),
                        ("AUGUSTUS v3.5.0 posterior sampling, all additions", aug_shared),
                        (f"AUGUSTUS v3.5.0 posterior sampling, posterior-ranked top "
                         f"{aug_budget['budget']}", aug_budget)):
            w.writerow([name, r["loci_scored"], r["added_structures"],
                        r["matched_TAIR10_alt_exact_CDS"], r["precision_vs_TAIR10_alt_pct"],
                        r["matched_AtRTD3"], r["precision_vs_AtRTD3_pct"],
                        r["reference_alternative_structures"], r["recall_of_TAIR10_alt_pct"]])

    hdr = f"{'set':<34} {'loci':>6} {'added':>6} {'T10alt':>6} {'%':>5} {'AtRTD3':>6} {'%':>5} {'recall%':>7}"
    print(hdr, file=sys.stderr)
    for name, r in (("TransGenic (shared)", tg_shared), ("AUGUSTUS all (shared)", aug_shared),
                    (f"AUGUSTUS top-{aug_budget['budget']} (shared)", aug_budget)):
        print(f"{name:<34} {r['loci_scored']:>6} {r['added_structures']:>6} "
              f"{r['matched_TAIR10_alt_exact_CDS']:>6} {r['precision_vs_TAIR10_alt_pct']:>5} "
              f"{r['matched_AtRTD3']:>6} {r['precision_vs_AtRTD3_pct']:>5} "
              f"{r['recall_of_TAIR10_alt_pct']:>7}", file=sys.stderr)
    print(f"written: {dst}\nwritten: {csv_dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
