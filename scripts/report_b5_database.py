#!/usr/bin/env python3
"""Per-species / per-tier accounting of a merged B5 database (issue #50 reporting list).

Counts rows by species, tier and split, RC rows, rejections by class, and the A29/A32/A33 masking
counters that the tile builder writes into `geneList.qc_flags` as `name=<n>` tokens.
`tier_margin_unguaranteed` is required by PROTOCOL_B1_frozen_v1.md:439 but older builders did not write it,
so it is recomputed here from gene_key_map coordinates and labelled as recomputed, not as recorded.
"""
import argparse
import collections
import csv
import json
import hashlib
import statistics
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
# tier_margin_unguaranteed is required by PROTOCOL_B1_frozen_v1.md:439 but older builders did not write it, so
# legacy databases need recomputation from gene_key_map coordinates with the
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
    """A33.4, recomputed from gene_key_map for compatibility with older builds (PROTOCOL:439).

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
    out = {"source": "recomputed from gene_key_map at report time; without contig lengths; compare separately with recorded builder counters",
           "contig_edge_credit": False, "edge_margin": gc.EDGE_MARGIN, "genes_considered": len(rows),
           "exceeds_length_guarantee": {}, "not_covered_with_margin": {}, "not_covered_with_margin_by_species": {}}
    for tier in gc.WINDOW_TIERS:
        bound = 2 * tier // 3 - 2 * gc.EDGE_MARGIN
        per_species = collections.Counter()
        for sp, s0, e0 in rows:
            if e0 - s0 > bound and not gc.covered_with_margin(s0, e0, tier):
                per_species[sp] += 1
        out["exceeds_length_guarantee"][str(tier)] = {"bound_nt": bound, "genes": sum(1 for _, s0, e0 in rows if e0 - s0 > bound)}
        out["not_covered_with_margin"][str(tier)] = sum(per_species.values())
        out["not_covered_with_margin_by_species"][str(tier)] = dict(sorted(per_species.items()))
    columns = {r[0] for r in con.sql("DESCRIBE build_manifest").fetchall()}
    out["recorded_by_species"] = {}
    if "tier_margin_unguaranteed" in columns:
        out["recorded_by_species"] = {
            sp: json.loads(value) if value is not None else None for sp, value in con.sql(
                "SELECT species_id, tier_margin_unguaranteed FROM build_manifest").fetchall()}
    return out


def canonical_order_audit(con):
    """Classify recorded ordering failures without replaying or changing the build.

    Equal coordinate keys prove the comparison failed on the third (block) key.
    Rejected labels are not stored, so this is not a recovered-tile count.
    """
    cases = []
    for sp, window, reason in con.sql(
            "SELECT species_id, gene_id, reason FROM rejected_records "
            "WHERE reason LIKE '%canonical order%' ORDER BY 1, 2").fetchall():
        match = re.search(r"\(\((\d+), (\d+)\) after \((\d+), (\d+)\)", reason)
        equal = bool(match and match.group(1, 2) == match.group(3, 4))
        decreasing = bool(match and tuple(map(int, match.group(1, 2))) < tuple(map(int, match.group(3, 4))))
        cases.append({"decreasing_span": decreasing, "species_id": sp, "window": window, "equal_span_tie": equal, "reason": reason})
    return {"source": "read-only rejected_records; comparison keys recorded by validator",
            "rejected_tiles": len(cases), "equal_span_ties": sum(c["equal_span_tie"] for c in cases),
            "decreasing_spans": sum(c["decreasing_span"] for c in cases),
            "unparsed_or_unexpected": sum(not c["equal_span_tie"] and not c["decreasing_span"] for c in cases),
            "by_species": dict(sorted(collections.Counter(c["species_id"] for c in cases).items())),
            "cases": cases,
            "limitation": "Does not reconstruct discarded labels or establish survival of later mask-fraction filters."}


def distribution(values):
    values = sorted(values)
    n = len(values)
    def quantile(p):
        x = (n - 1) * p
        lo = int(x)
        return values[lo] + (values[min(lo + 1, n - 1)] - values[lo]) * (x - lo)
    return {"n": n, "mean": statistics.mean(values) if n else None,
            "p25": quantile(.25) if n else None, "median": quantile(.5) if n else None,
            "p75": quantile(.75) if n else None, "max": max(values) if n else None}


def evaluation_population(con, reference_distributions=False):
    """Unique loci, deduplicated across tiers, offsets and RC; no metric redefinition."""
    con.sql("CREATE OR REPLACE TEMP VIEW _membership AS SELECT DISTINCT species_id, gene_id FROM window_genes")
    matrix = [dict(zip(("species_id", "gene_split", "tile_split", "unique_genes"), r)) for r in con.sql(
        "SELECT w.species_id, s.split, t.split, count(DISTINCT w.gene_id) "
        "FROM window_genes w JOIN gene_split s USING(species_id, gene_id) "
        "JOIN geneList t ON t.species_id=w.species_id AND t.gene_id=w.window_id AND t.is_rc=w.is_rc "
        "GROUP BY 1,2,3 ORDER BY 1,2,3").fetchall()]
    totals = {}
    for split, n, labelled, strict, strict_labelled in con.sql(
        "SELECT s.split, count(*), count(l.gene_id), count(*) FILTER (WHERE s.strict_holdout), "
        "count(l.gene_id) FILTER (WHERE s.strict_holdout) "
        "FROM gene_split s LEFT JOIN _membership l USING(species_id,gene_id) GROUP BY 1 ORDER BY 1").fetchall():
        totals[split] = {"assigned_genes": n, "ever_labelled_genes": labelled,
                         "pct": round(100 * labelled / n, 3), "strict_assigned": strict,
                         "strict_ever_labelled": strict_labelled}
    lengths = collections.defaultdict(list)
    for sp, labelled, length in con.sql(
        "SELECT s.species_id, l.gene_id IS NOT NULL, k.end0-k.start0 FROM gene_split s "
        "LEFT JOIN _membership l USING(species_id,gene_id) "
        "JOIN gene_key_map k USING(species_id,gene_id) WHERE s.split='test'").fetchall():
        if length is not None:
            lengths[(sp, labelled)].append(length)
    result = {"definition": "Ever-labelled unique (species_id, gene_id) in window_genes, not nominal split size or a universal metric denominator.",
              "split_totals": totals, "gene_split_by_tile_split": matrix,
              "test_gene_length_nt": {sp: {state: distribution(lengths[(sp, flag)])
                  for state, flag in (("labelled", True), ("unlabelled", False))}
                  for sp in sorted({k[0] for k in lengths})},
              "metric_scope": "Tile targets contain all eligible labelled genes, including train/valid genes in test tiles. Full-GFF evaluation and evidence callability have separate denominators. No scoring universe is changed.",
              "test_reference_transcript_count": None}
    if reference_distributions:
        labels = set(con.sql("SELECT species_id,gene_id FROM _membership").fetchall())
        test = set(con.sql("SELECT species_id,gene_id FROM gene_split WHERE split='test'").fetchall())
        counts = collections.defaultdict(list)
        seen = set()
        provenance = {}
        for sp, path, expected in con.sql("SELECT species_id,gff,gff_sha256 FROM build_manifest").fetchall():
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"{sp}: reference GFF SHA256 differs from build manifest")
            provenance[sp] = {"path": path, "sha256": expected}
            with open(path) as fh:
                for gene in gc.parse_gff3(fh, species_code=gc.species_code(sp)):
                    key = (sp, gene.gene_id)
                    if key in test:
                        seen.add(key)
                        counts[(sp, key in labels)].append(len(gene.transcripts))
        result["test_reference_transcript_count"] = {
            "definition": "Transcripts retained by the original GSF GFF parser before QC/caps/masking; not an isoform-scoring denominator.",
            "missing_test_gene_keys": len(test - seen), "references": provenance,
            "by_species": {sp: {state: distribution(counts[(sp, flag)])
                for state, flag in (("labelled", True), ("unlabelled", False))}
                for sp in sorted(provenance)}}
    return result


def labelled_gene_checks(con):
    # Gene-level assertions the validator structurally cannot make: geneList.orthogroup_id is NULL for every
    # tile row, so validate_split's group key degenerates to the window id and its cross-split check is inert.
    # window_genes holds only the genes a tile actually labelled (masked genes are dropped before the insert).
    con.sql("CREATE TEMP TABLE _lab AS SELECT DISTINCT w.species_id, w.gene_id, g.split AS tile_split "
            "FROM window_genes w JOIN geneList g ON g.species_id = w.species_id AND g.gene_id = w.window_id AND g.is_rc = w.is_rc")
    _rank = "CASE {} WHEN 'train' THEN 0 WHEN 'valid' THEN 1 WHEN 'test' THEN 2 END"
    return {
        "no_gene_labelled_below_its_split": con.sql(
            f"SELECT count(*) FROM _lab l JOIN gene_split s ON s.species_id = l.species_id AND s.gene_id = l.gene_id "
            f"WHERE {_rank.format('s.split')} > {_rank.format('l.tile_split')}").fetchone()[0] == 0,
        "no_strict_holdout_labelled_outside_test": con.sql(
            "SELECT count(*) FROM _lab l JOIN gene_split s ON s.species_id = l.species_id AND s.gene_id = l.gene_id "
            "WHERE s.strict_holdout AND l.tile_split <> 'test'").fetchone()[0] == 0,
        "no_orphan_window_genes": con.sql(
            "SELECT count(*) FROM window_genes w LEFT JOIN geneList g ON g.species_id = w.species_id "
            "AND g.gene_id = w.window_id AND g.is_rc = w.is_rc WHERE g.gene_id IS NULL").fetchone()[0] == 0,
    }

def export_population_tables(report, out_dir):
    """Render supplementary tables from the same report, avoiding hand-copied counts."""
    os.makedirs(out_dir, exist_ok=True)
    coverage = []
    for sp, splits in sorted(report["labelled_coverage_by_species"].items()):
        for split, v in sorted(splits.items()):
            coverage.append(dict(species_id=sp, gene_split=split, assigned_genes=v["genes"],
                                 ever_labelled_genes=v["labelled"], pct=v["pct"]))
    population = report["evaluation_population"]
    distributions = []
    for sp, states in population["test_gene_length_nt"].items():
        for state, values in states.items():
            row = dict(species_id=sp, test_gene_status=state)
            row.update({"length_nt_" + k: v for k, v in values.items()})
            tx = population["test_reference_transcript_count"]
            if tx is not None:
                row.update({"reference_transcripts_" + k: v for k, v in tx["by_species"][sp][state].items()})
            distributions.append(row)
    for name, rows in (("TableS_b5_labelled_coverage.csv", coverage),
                       ("TableS_b5_test_distributions.csv", distributions)):
        if rows:
            with open(os.path.join(out_dir, name), "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--tables-dir", help="export supplementary coverage/distribution CSVs")
    ap.add_argument("--reference-distributions", action="store_true", help="read and hash-check original manifest GFFs for transcript counts")
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
    gene_checks = labelled_gene_checks(con)
    # A gene is labelled only where its own split allows, so a test gene needs a test block: the labelled
    # fraction is far from 1 for test and valid and is not recoverable from the split table alone.
    labelled_coverage = {}
    for sp, split, lab, tot in con.sql(
            "WITH tot AS (SELECT species_id, split, count(*) n FROM gene_split GROUP BY 1, 2), "
            "     got AS (SELECT s.species_id, s.split, count(DISTINCT l.gene_id) n FROM _lab l "
            "             JOIN gene_split s ON s.species_id = l.species_id AND s.gene_id = l.gene_id GROUP BY 1, 2) "
            "SELECT t.species_id, t.split, coalesce(g.n, 0), t.n FROM tot t "
            "LEFT JOIN got g ON g.species_id = t.species_id AND g.split = t.split ORDER BY 1, 2").fetchall():
        labelled_coverage.setdefault(sp, {})[split] = {"labelled": lab, "genes": tot,
                                                       "pct": round(100.0 * lab / tot, 1) if tot else None}

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
    checks.update(gene_checks)
    report = {
        "db": a.db,
        "database_access": "read_only=True; temporary relations only; frozen file never updated",
        "canonical_order_audit": canonical_order_audit(con),
        "evaluation_population": evaluation_population(con, a.reference_distributions),
        "checks": checks,
        "labelled_coverage_by_species": labelled_coverage,
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
    if a.tables_dir:
        export_population_tables(report, a.tables_dir)
    out = json.dumps(report, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
