#!/usr/bin/env python3
"""Per-species / per-tier accounting of a merged B5 database (issue #50 reporting list).

Counts rows by species, tier and split, RC rows, rejections by class, and the A29/A32/A33 masking
counters that the tile builder writes into `geneList.qc_flags` as `name=<n>` tokens.
`tier_margin_unguaranteed` is required by PROTOCOL_B1_frozen_v1.md §A26 but is not written by the
builder, so it is reported as null rather than as zero.
"""
import argparse
import collections
import json
import re

import duckdb

QC_COUNTERS = ("leak_masked", "hard_masked", "decoy_masked", "component_masked", "dup_collapsed", "edge_partial")
# Written nowhere in the builder; see PROTOCOL_B1_frozen_v1.md:439. Reported as null, never as 0.
UNIMPLEMENTED_COUNTERS = ("tier_margin_unguaranteed",)
REJECT_CLASSES = (
    ("mask_fraction_a33", r"^masked fraction"),
    ("token_cap", r"tokens \d+ >"),
    ("transcript_cap", r"^transcripts \d+ >"),
    ("cds_cap", r"^CDS \d+ >"),
    ("no_cds", r"^no CDS"),
    ("canonical_order", r"canonical order"),
    ("chromosome_missing", r"missing from FASTA"),
)


def classify(reason):
    for name, pat in REJECT_CLASSES:
        if re.search(pat, reason):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)

    rows_by_species_tier = collections.defaultdict(dict)
    for sp, tier, n, rc in con.sql(
            "SELECT species_id, fin - start AS tier, count(*), sum(CASE WHEN is_rc THEN 1 ELSE 0 END) "
            "FROM geneList GROUP BY 1, 2 ORDER BY 1, 2").fetchall():
        rows_by_species_tier[sp][str(tier)] = {"rows": n, "rc_rows": int(rc)}
    by_species_split = collections.defaultdict(dict)
    for sp, split, n in con.sql("SELECT species_id, split, count(*) FROM geneList GROUP BY 1, 2").fetchall():
        by_species_split[sp][split or "NULL"] = n
    by_tier_split = collections.defaultdict(dict)
    for tier, split, n in con.sql("SELECT fin - start, split, count(*) FROM geneList GROUP BY 1, 2 ORDER BY 1").fetchall():
        by_tier_split[str(tier)][split or "NULL"] = n

    # qc_flags is a ';'-joined string of flag names and name=<n> counters.
    counters = {sp: {k: 0 for k in QC_COUNTERS} for sp in rows_by_species_tier}
    tiles_with = {sp: {k: 0 for k in QC_COUNTERS} for sp in rows_by_species_tier}
    for sp, qc in con.sql("SELECT species_id, qc_flags FROM geneList WHERE qc_flags IS NOT NULL").fetchall():
        for tok in qc.split(";"):
            if "=" in tok:
                k, _, v = tok.partition("=")
                if k in counters[sp]:
                    counters[sp][k] += int(v)
                    tiles_with[sp][k] += 1

    rejections = collections.defaultdict(lambda: collections.Counter())
    for sp, reason in con.sql("SELECT species_id, reason FROM rejected_records").fetchall():
        rejections[sp][classify(reason)] += 1

    totals = {
        "rows": con.sql("SELECT count(*) FROM geneList").fetchone()[0],
        "rc_rows": con.sql("SELECT count(*) FROM geneList WHERE is_rc").fetchone()[0],
        "distinct_rn": con.sql("SELECT count(DISTINCT rn) FROM geneList").fetchone()[0],
        "rows_by_split": dict(con.sql("SELECT split, count(*) FROM geneList GROUP BY 1").fetchall()),
        "rows_train_weight_zero": con.sql("SELECT count(*) FROM geneList WHERE train_weight = 0").fetchone()[0],
        "rows_null_label": con.sql("SELECT count(*) FROM geneList WHERE gff IS NULL").fetchone()[0],
        "gene_split_rows": con.sql("SELECT count(*) FROM gene_split").fetchone()[0],
        "build_manifest_rows": con.sql("SELECT count(*) FROM build_manifest").fetchone()[0],
        "rejected_rows": con.sql("SELECT count(*) FROM rejected_records").fetchone()[0],
    }
    report = {
        "db": a.db,
        "totals": totals,
        "rows_by_species_tier": {k: v for k, v in sorted(rows_by_species_tier.items())},
        "rows_by_species_split": {k: v for k, v in sorted(by_species_split.items())},
        "rows_by_tier_split": dict(by_tier_split),
        "mask_counters_by_species": {k: counters[k] for k in sorted(counters)},
        "tiles_carrying_counter_by_species": {k: tiles_with[k] for k in sorted(tiles_with)},
        "mask_counters_total": {k: sum(counters[s][k] for s in counters) for k in QC_COUNTERS},
        "unimplemented_counters": {k: None for k in UNIMPLEMENTED_COUNTERS},
        "rejections_by_species": {k: dict(v) for k, v in sorted(rejections.items())},
        "rejections_total": dict(sum((collections.Counter(v) for v in rejections.values()), collections.Counter())),
    }
    con.close()
    out = json.dumps(report, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
