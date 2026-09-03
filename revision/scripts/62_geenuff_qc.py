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
                      "super_loci_overlap", "missmatching_strand", "truncated_intron",   # GeenuFF 702cbf3 names (inconsistent models)
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


# Error feature types of the pinned GeenuFF (weberlab-hhu/GeenuFF 702cbf3, 2024-03-23; geenuff.base.types.Errors), recorded
# for the provenance record. Extraction does not depend on this list: every feature whose type does not start with
# "geenuff_" (the biological feature types geenuff_transcript / geenuff_cds / geenuff_intron) is an error feature.
GEENUFF_ERROR_TYPES = ("missing_utr_5p", "missing_utr_3p", "empty_super_locus", "missing_start_codon", "missing_stop_codon",
                       "wrong_starting_phase", "mismatched_ending_phase", "overlapping_exons", "too_short_intron",
                       "super_loci_overlap_error", "missmatching_strands", "truncated_intron")
BIOLOGICAL_PREFIX = "geenuff_"


def extract_from_geenuff_db(db_path: str, species_id: str, seen: Optional[Dict] = None) -> List[Tuple[str, str, str, str, int, int]]:
    """GeenuFF sqlite (schema of 702cbf3) -> (species, gene, transcript, flag, start, end) per error feature.
    Features reach their transcript through transcript_piece: feature -> association_transcript_piece_to_feature ->
    transcript_piece -> transcript -> super_locus. gene = super_locus.given_name (the GFF gene ID; the builder resolves it
    through gene_id_original), transcript = transcript.given_name (the mRNA ID). An error feature without a transcript
    association is kept with gene "" and transcript "*" (gene-level unknown) so that it is counted, never silently lost."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("feature", "transcript", "transcript_piece", "super_locus", "association_transcript_piece_to_feature"):
        if t not in tables:
            raise ValueError(f"{db_path}: table {t!r} missing; expected the GeenuFF 702cbf3 schema (tables: {sorted(tables)})")
    types_seen = sorted(r[0] for r in cur.execute("SELECT DISTINCT type FROM feature"))
    if seen is not None:
        seen["feature_types_seen"] = types_seen
        seen["schema"] = {"association": "association_transcript_piece_to_feature", "path": "feature->transcript_piece->transcript->super_locus"}
    q = """SELECT f.id, f.type, f.start, f.end, t.given_name, sl.given_name
           FROM feature f
           LEFT JOIN association_transcript_piece_to_feature a ON a.feature_id = f.id
           LEFT JOIN transcript_piece tp ON tp.id = a.transcript_piece_id
           LEFT JOIN transcript t ON t.id = tp.transcript_id
           LEFT JOIN super_locus sl ON sl.id = t.super_locus_id
           WHERE f.type NOT LIKE ?
           ORDER BY f.id"""
    out = []
    done = set()
    for fid, ftype, s, e, tx_name, sl_name in cur.execute(q, (BIOLOGICAL_PREFIX + "%",)):
        key = (fid, tx_name)
        if key in done:                       # a feature attached to several pieces of the same transcript
            continue
        done.add(key)
        out.append((species_id, sl_name or "", tx_name or "*", str(ftype), int(s or 0), int(e or 0)))
    con.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geenuff-db", required=True)
    ap.add_argument("--species-id", required=True)
    ap.add_argument("--out", required=True, help="flags TSV")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args(argv)
    seen: Dict = {}
    rows = extract_from_geenuff_db(a.geenuff_db, a.species_id, seen)
    with open(a.out, "w") as fh:
        fh.write("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    counts = defaultdict(int)
    for r in rows:
        counts[r[3]] += 1
    summ = {"species_id": a.species_id, "n_flags": len(rows), "by_flag": dict(counts),
            "genes_with_hard_flags": len({r[1] for r in rows if is_hard(r[3]) and r[1]}),
            "flags_without_gene": sum(1 for r in rows if not r[1]),
            "hard_patterns": HARD_FLAG_PATTERNS, "geenuff_error_types_pinned": GEENUFF_ERROR_TYPES, **seen}
    with open(a.summary, "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
