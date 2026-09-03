#!/usr/bin/env python3
"""Per-species / per-tier accounting of a merged B5 database (issue #50 reporting list).

Counts rows by species, tier and split, RC rows, rejections by class, and the A29/A32/A33 masking
counters that the tile builder writes into `geneList.qc_flags` as `name=<n>` tokens.
`tier_margin_unguaranteed` is required by PROTOCOL_B1_frozen_v1.md:439 but the builder never writes it,
so it is recomputed here from gene_key_map coordinates and labelled as recomputed, not as recorded.
"""
import argparse
import collections
import json
import os
import re
import sys
import types

import duckdb

# loaded by path, the way tests/conftest.py does: importing the transgenic package pulls in torch.
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "transgenic", "utils", "gsf_contract.py")
gc = types.ModuleType("gsf_contract")
gc.__file__ = _PATH
sys.modules["gsf_contract"] = gc
with open(_PATH) as _fh:
    exec(compile(_fh.read(), _PATH, "exec"), gc.__dict__)

QC_COUNTERS = ("leak_masked", "hard_masked", "decoy_masked", "component_masked", "dup_collapsed", "edge_partial")
# tier_margin_unguaranteed is required by PROTOCOL_B1_frozen_v1.md:439 but the builder never writes it, so
# it cannot be read back out of the database. It is recomputed here from gene_key_map coordinates with the
# same predicate the protocol names (gsf_contract.covered_with_margin) and reported separately, clearly
# labelled as recomputed at report time rather than recorded at build time.
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


def tier_margin_unguaranteed(con):
    """A33.4, recomputed from gene_key_map because the builder records no such counter (PROTOCOL:439).

    Two different quantities, both reported:
    `exceeds_length_guarantee` — genes longer than 2*tier/3 - 2*EDGE_MARGIN, which is the bound the protocol
    measured (3,418 / 653 / 86 across the nine references); a gene over that length is not *guaranteed* to sit
    EDGE_MARGIN inside any tile of the tier.
    `not_covered_with_margin` — of those, the ones that also do not happen to land inside a tile with margin
    at any of the three offsets. This is what the protocol means by tier_margin_unguaranteed: "those 86 genes
    are recovered only when they happen to fall inside a tile with margin".

    contig_len is not stored in the database, so a tile edge coinciding with a contig edge cannot be credited:
    `not_covered_with_margin` is an upper bound.
    """
    rows = con.sql("SELECT species_id, start0, end0 FROM gene_key_map WHERE start0 IS NOT NULL AND end0 IS NOT NULL").fetchall()
    out = {"source": "recomputed from gene_key_map at report time; the builder records no such counter",
           "contig_edge_credit": False, "edge_margin": gc.EDGE_MARGIN, "genes_considered": len(rows),
           "exceeds_length_guarantee": {}, "not_covered_with_margin": {}, "not_covered_with_margin_by_species": {}}
    for tier in gc.WINDOW_TIERS:
        bound = 2 * tier // 3 - 2 * gc.EDGE_MARGIN
        per_species = collections.Counter()
        for sp, s0, e0 in rows:
            if not gc.covered_with_margin(s0, e0, tier):
                per_species[sp] += 1
        out["exceeds_length_guarantee"][str(tier)] = {"bound_nt": bound, "genes": sum(1 for _, s0, e0 in rows if e0 - s0 > bound)}
        out["not_covered_with_margin"][str(tier)] = sum(per_species.values())
        out["not_covered_with_margin_by_species"][str(tier)] = dict(sorted(per_species.items()))
    return out


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
    rn_ranges = {sp: (lo, hi, n) for sp, lo, hi, n in con.sql(
        "SELECT species_id, min(rn), max(rn), count(*) FROM geneList GROUP BY 1 ORDER BY 2").fetchall()}
    n = totals["rows"]
    checks = {
        "rn_dense_1_to_n": con.sql("SELECT min(rn), max(rn) FROM geneList").fetchone() == (1, n) and totals["distinct_rn"] == n,
        "species_blocks_contiguous": all(hi - lo + 1 == cnt for lo, hi, cnt in rn_ranges.values()),
        "species_blocks_disjoint_and_ordered": [r[0] for r in rn_ranges.values()] == sorted(r[0] for r in rn_ranges.values())
                                               and sum(r[2] for r in rn_ranges.values()) == n,
        "gene_split_copied_once": con.sql("SELECT count(*) FROM (SELECT species_id, gene_id FROM gene_split "
                                          "GROUP BY 1, 2 HAVING count(*) > 1)").fetchone()[0] == 0,
        "one_build_manifest_row_per_species": totals["build_manifest_rows"] == len(rn_ranges),
        "no_excluded_species": con.sql("SELECT count(*) FROM geneList WHERE species_id = 'Zmays'").fetchone()[0] == 0,
        "no_maize_gene_models": con.sql("SELECT count(*) FROM geneList WHERE geneModel LIKE 'Zm%' OR geneModel LIKE 'GRMZM%'").fetchone()[0] == 0,
        "no_null_split": con.sql("SELECT count(*) FROM geneList WHERE split IS NULL").fetchone()[0] == 0,
    }
    report = {
        "db": a.db,
        "checks": checks,
        "rn_ranges": {k: {"min": v[0], "max": v[1], "rows": v[2]} for k, v in rn_ranges.items()},
        "totals": totals,
        "rows_by_species_tier": {k: v for k, v in sorted(rows_by_species_tier.items())},
        "rows_by_species_split": {k: v for k, v in sorted(by_species_split.items())},
        "rows_by_tier_split": dict(by_tier_split),
        "mask_counters_by_species": {k: counters[k] for k in sorted(counters)},
        "tiles_carrying_counter_by_species": {k: tiles_with[k] for k in sorted(tiles_with)},
        "mask_counters_total": {k: sum(counters[s][k] for s in counters) for k in QC_COUNTERS},
        "tier_margin_unguaranteed": tier_margin_unguaranteed(con),
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
