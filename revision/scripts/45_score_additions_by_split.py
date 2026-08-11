#!/usr/bin/env python3
"""Additions-only precision of the headline prompted run, partitioned by the split
category each A. thaliana locus actually occupies in the training dataset.

The addition/hit definition is identical to revision/scripts/28_score_added_isoforms.py:
an addition is a predicted transcript whose sorted CDS coordinate tuple differs from
the TAIR10 primary supplied as the prompt; identical emissions collapse to one
structure; a hit is an exact CDS coordinate match to a TAIR10 alternative transcript
with the primary removed from the reference; loci whose GM ends in -rc are skipped.

Categories come from the seed-123 split reconstruction (validated to an exact set
match against revision/results/fig3_regen/fig3_test_genes.tsv):
  train       gene has at least one row (forward or -rc) in the 302,744-row train slice
  validation  no train row, at least one row in the 40,365-row validation slice
  test        rows only in the 60,550-row held-out test slice
  not_in_db   gene never entered the database
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path("/data/gpfs/assoc/pgl/data/Transgenic")
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"
SCRATCH = Path(__file__).resolve().parent
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRATCH


def cds_by_transcript(path: Path, gff3: bool) -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    lr = re.compile(r"GM=([^;\s]+)") if gff3 else re.compile(r'gene_id "([^"]+)"')
    tr = re.compile(r"Parent=([^;]+)") if gff3 else re.compile(r'transcript_id "([^"]+)"')
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


def chain(struct: tuple) -> tuple:
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def wilson(k: int, n: int, z: float = 1.959963985):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def load_categories():
    cat = {}
    with (SCRATCH / "at_gene_split.tsv").open() as fh:
        next(fh)
        for line in fh:
            g, c, _ = line.rstrip("\n").split("\t")
            cat[g.replace(".TAIR10", "")] = c
    with (SCRATCH / "at_genes_not_in_db.tsv").open() as fh:
        next(fh)
        for line in fh:
            g, reason = line.rstrip("\n").split("\t")
            cat[g.replace(".TAIR10", "")] = "not_in_db"
    return cat


def main():
    pred = cds_by_transcript(
        CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3", True)
    ref = cds_by_transcript(DATA / "TAIR10" / "TAIR10.gtf", False)
    art = cds_by_transcript(DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf", False)
    primary = {}
    for line in (DATA / "TAIR10" / "primary_transcript_ids.txt").read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    cat = load_categories()
    # the fig3_original locus list the manuscript currently calls "held-out"
    old_heldout = {l.strip() for l in (ROOT / "transgenic" / "revision" / "results" /
                                       "heldout_additions" / "heldout_loci_nonrc.txt"
                                       ).read_text().splitlines() if l.strip()}

    agg = defaultdict(lambda: {"loci": 0, "added": 0, "struct_hit": 0, "chain_hit": 0,
                               "art_hit": 0, "alt_total": 0, "alt_recovered": 0,
                               "loci_with_addition": 0})
    per_locus = {}
    for locus, txs in pred.items():
        supplied = ref.get(locus, {}).get(primary.get(locus))
        additions = list({s for s in txs.values() if s != supplied})
        ref_primary = supplied
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        art_here = set(art.get(locus, {}).values())
        c = cat.get(locus, "UNKNOWN_not_in_annotation")
        buckets = [c, "ALL"]
        if locus in old_heldout:
            buckets.append("fig3_original_heldout")
        else:
            buckets.append("fig3_original_remainder")
        if c in ("train", "validation"):
            buckets.append("train_plus_validation")
        for b in buckets:
            a = agg[b]
            a["loci"] += 1
            a["alt_total"] += len(alt_ref)
            if additions:
                a["loci_with_addition"] += 1
            a["added"] += len(additions)
            a["struct_hit"] += sum(1 for s in additions if s in alt_ref)
            a["chain_hit"] += sum(1 for s in additions
                                  if len(chain(s)) >= 1 and chain(s) in {chain(x) for x in alt_ref})
            a["art_hit"] += sum(1 for s in additions if s in art_here)
            a["alt_recovered"] += len(alt_ref & set(additions))
        per_locus[locus] = (c, len(additions),
                            sum(1 for s in additions if s in alt_ref))

    results = {}
    for name, a in sorted(agg.items()):
        results[name] = {
            "loci_scored": a["loci"],
            "loci_with_at_least_one_addition": a["loci_with_addition"],
            "added_transcripts": a["added"],
            "reference_alternative_transcripts": a["alt_total"],
            "added_matching_TAIR10_alternative_exact_CDS": a["struct_hit"],
            "added_matching_TAIR10_alternative_intron_chain": a["chain_hit"],
            "added_matching_any_AtRTD3_transcript": a["art_hit"],
            "precision_vs_TAIR10_alternatives_pct":
                round(100 * a["struct_hit"] / a["added"], 1) if a["added"] else None,
            "wilson95_ci_pct": wilson(a["struct_hit"], a["added"]),
            "precision_vs_AtRTD3_pct":
                round(100 * a["art_hit"] / a["added"], 1) if a["added"] else None,
            "recall_of_TAIR10_alternatives_pct":
                round(100 * a["alt_recovered"] / a["alt_total"], 1) if a["alt_total"] else None,
        }

    def fisher(a_name, b_name):
        a, b = agg[a_name], agg[b_name]
        table = [[a["struct_hit"], a["added"] - a["struct_hit"]],
                 [b["struct_hit"], b["added"] - b["struct_hit"]]]
        orr, p = fisher_exact(table)
        return {"groups": [a_name, b_name], "table_hit_miss": table,
                "odds_ratio": round(float(orr), 3), "p_value": round(float(p), 4)}

    tests = {
        "heldout_test_vs_train_plus_validation": fisher("test", "train_plus_validation"),
        "heldout_test_vs_train": fisher("test", "train"),
        "not_in_db_vs_train_plus_validation": fisher("not_in_db", "train_plus_validation"),
        "not_in_db_vs_train": fisher("not_in_db", "train"),
        "validation_vs_train": fisher("validation", "train"),
        "fig3_original_heldout_vs_remainder_reproduction":
            fisher("fig3_original_heldout", "fig3_original_remainder"),
    }

    out = {
        "generated": "2026-08-11",
        "definition": ("identical to revision/scripts/28_score_added_isoforms.py; loci "
                       "partitioned by the split category measured from the reconstructed "
                       "seed-123 random_split over the 403,659-row training dataset"),
        "prediction_file": str(CMP / "standardized_results" /
                               "A_thaliana_transgenic400Mprompt_beam1.gff3"),
        "categories": results,
        "fisher_tests": tests,
    }
    (OUTDIR / "additions_precision_by_split.json").write_text(json.dumps(out, indent=1))
    with (OUTDIR / "per_locus_category.tsv").open("w") as fh:
        fh.write("locus\tcategory\tadditions\tadditions_matching_TAIR10_alt\n")
        for l in sorted(per_locus):
            c, ad, hit = per_locus[l]
            fh.write(f"{l}\t{c}\t{ad}\t{hit}\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
