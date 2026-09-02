#!/usr/bin/env python3
"""Annotation-quality flags from GeenuFF for loss masking before B5 training (protocol A22).

Step 1 (run on the cluster; GeenuFF pinned in the container):
  import2geenuff.py --gff3 <species>.gff3 --fasta <species>.fa --species <species_id> --db-path qc/<species>.geenuff.sqlite3
Step 2 (this script): read the GeenuFF sqlite database, collect every error feature GeenuFF wrote
(feature types whose name contains 'error' or that GeenuFF marks as errors), attribute it to the
super-locus (gene) and transcript it overlaps, and write qc/<species>.geenuff_flags.tsv:
  species_id  gene_id  transcript_id  flag  start  end
plus a summary JSON. The exact flag names are those of the pinned GeenuFF version and are
recorded verbatim; the mapping to loss-mask actions lives in HARD_FLAG_PATTERNS below (A22).
The builder (build_b5.py --qc-flags) consumes the TSV.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

# A22: hard errors make a transcript ineligible as a training label; soft flags are recorded only.
# A30: swissprot_caution_* (63_swissprot_sensitivity.py) is hard, swissprot_note_* is soft.
# (kept identical to src/transgenic/datasets/qc_flags.py)
HARD_FLAG_PATTERNS = ("missing_start", "missing_stop", "wrong_starting_phase", "mismatched_ending_phase", "mismatched_phase",
                      "overlapping_exon", "too_short_intron", "empty_transcript", "empty_super_locus", "wrong_phase",
                      "swissprot_caution")
SOFT_FLAG_PATTERNS = ("missing_utr", "missing_utr_5p", "missing_utr_3p", "no_utr")


def is_hard(flag: str) -> bool:
    f = flag.lower()
    return any(p in f for p in HARD_FLAG_PATTERNS)


def read_flags_tsv(path: str) -> Dict[Tuple[str, str], Dict[str, Set[str]]]:
    """(species_id, gene_id) -> {transcript_id or '*': set(flags)}."""
    out: Dict[Tuple[str, str], Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            sp, g, t, flag = c[idx["species_id"]], c[idx["gene_id"]], c[idx.get("transcript_id", 2)] or "*", c[idx["flag"]]
            out[(sp, g)][t].add(flag)
    return out


def loss_mask_decision(gene_flags: Dict[str, Set[str]], transcript_ids: Iterable[str]) -> Tuple[float, List[str], List[str]]:
    """A22 policy. Returns (train_weight, transcripts_to_keep, hard_flags).
    - hard flag on the gene ('*') or on every transcript -> train_weight 0 (sample masked, kept in DB)
    - hard flag on some transcripts -> those transcripts are dropped from the label, weight 1
    - soft flags only -> weight 1, nothing dropped
    """
    tids = list(transcript_ids)
    hard_gene = {f for f in gene_flags.get("*", set()) if is_hard(f)}
    hard_by_tx = {t: {f for f in gene_flags.get(t, set()) if is_hard(f)} for t in tids}
    all_hard = sorted(hard_gene | {f for s in hard_by_tx.values() for f in s})
    if hard_gene:
        return 0.0, [], all_hard
    keep = [t for t in tids if not hard_by_tx[t]]
    if not keep:
        return 0.0, [], all_hard
    return 1.0, keep, all_hard


def extract_from_geenuff_db(db_path: str, species_id: str) -> List[Tuple[str, str, str, str, int, int]]:
    """Best-effort extraction from GeenuFF's sqlite schema: error features -> (species, gene, transcript, flag, start, end).
    GeenuFF stores features with a `type` column and links them to transcripts/super_loci through association tables; the
    exact table names differ between releases, so this walks the schema generically."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "feature" not in tables:
        raise ValueError(f"{db_path}: no 'feature' table; not a GeenuFF database?")
    cols = [r[1] for r in cur.execute("PRAGMA table_info(feature)")]
    type_col = "type" if "type" in cols else next((c for c in cols if "type" in c), None)
    if type_col is None:
        raise ValueError("feature table has no type column")
    start_col = "start" if "start" in cols else "start_position"
    end_col = "end" if "end" in cols else "end_position"
    rows = cur.execute(f"SELECT id, {type_col}, {start_col}, {end_col} FROM feature WHERE lower({type_col}) LIKE '%error%'").fetchall()
    # transcript / super-locus association (names vary: association_transcript_to_feature, transcript, super_locus)
    assoc = next((t for t in tables if "transcript" in t and "feature" in t and "association" in t), None)
    tx_table = "transcript" if "transcript" in tables else None
    sl_table = "super_locus" if "super_locus" in tables else None
    out = []
    for fid, ftype, s, e in rows:
        gene_id, tx_id = "", "*"
        if assoc and tx_table:
            r = cur.execute(f"SELECT t.given_name, t.super_locus_id FROM {assoc} a JOIN {tx_table} t ON a.transcript_id = t.id WHERE a.feature_id = ? LIMIT 1", (fid,)).fetchone()
            if r:
                tx_id = r[0] or "*"
                if sl_table and r[1] is not None:
                    g = cur.execute(f"SELECT given_name FROM {sl_table} WHERE id = ?", (r[1],)).fetchone()
                    gene_id = g[0] if g else ""
        out.append((species_id, gene_id, tx_id, str(ftype), int(s or 0), int(e or 0)))
    con.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geenuff-db", required=True)
    ap.add_argument("--species-id", required=True)
    ap.add_argument("--out", required=True, help="flags TSV")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args(argv)
    rows = extract_from_geenuff_db(a.geenuff_db, a.species_id)
    with open(a.out, "w") as fh:
        fh.write("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    counts = defaultdict(int)
    for r in rows:
        counts[r[3]] += 1
    summ = {"species_id": a.species_id, "n_flags": len(rows), "by_flag": dict(counts),
            "genes_with_hard_flags": len({r[1] for r in rows if is_hard(r[3])}), "hard_patterns": HARD_FLAG_PATTERNS}
    with open(a.summary, "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
