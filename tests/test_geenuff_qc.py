"""revision/scripts/62_geenuff_qc.py against the real GeenuFF schema (weberlab-hhu/GeenuFF 702cbf3, 2024-03-23).

Measured on pgl-gpu 2026-09-02 with a TAIR10 slice: tables feature(id, given_name, type, start, start_is_biological_start,
end, end_is_biological_end, is_plus_strand, score, source, phase, coordinate_id), transcript(id, given_name, type, longest,
super_locus_id), transcript_piece(id, given_name, position, transcript_id), super_locus(id, given_name, aliases, type),
association_transcript_piece_to_feature(transcript_piece_id, feature_id). Biological features are typed `geenuff_cds`,
`geenuff_intron`, `geenuff_transcript`; error features carry the error name itself (`missing_utr_5p`, `too_short_intron`,
...) — nothing contains the word "error", and features reach their transcript only through transcript_piece.
"""
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "revision" / "scripts" / "62_geenuff_qc.py"


def _load(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def qc():
    return _load(SCRIPT, "geenuff_qc_62")


def _make_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE genome (id INTEGER PRIMARY KEY, species VARCHAR);
    CREATE TABLE coordinate (id INTEGER PRIMARY KEY, sequence VARCHAR, length INTEGER, seqid VARCHAR, sha1 VARCHAR, genome_id INTEGER);
    CREATE TABLE super_locus (id INTEGER PRIMARY KEY, given_name VARCHAR, aliases VARCHAR, type VARCHAR(58));
    CREATE TABLE transcript (id INTEGER PRIMARY KEY, given_name VARCHAR, type VARCHAR(43), longest BOOLEAN, super_locus_id INTEGER);
    CREATE TABLE transcript_piece (id INTEGER PRIMARY KEY, given_name VARCHAR, position INTEGER, transcript_id INTEGER);
    CREATE TABLE feature (id INTEGER PRIMARY KEY, given_name VARCHAR, type VARCHAR(24), start INTEGER, start_is_biological_start BOOLEAN,
                          end INTEGER, end_is_biological_end BOOLEAN, is_plus_strand BOOLEAN, score FLOAT, source VARCHAR, phase INTEGER, coordinate_id INTEGER);
    CREATE TABLE association_transcript_piece_to_feature (transcript_piece_id INTEGER, feature_id INTEGER);
    CREATE TABLE protein (id INTEGER PRIMARY KEY, given_name VARCHAR, super_locus_id INTEGER);
    CREATE TABLE association_protein_to_feature (protein_id INTEGER, feature_id INTEGER);
    """)
    con.execute("INSERT INTO coordinate VALUES (1, 'ACGT', 4, 'Chr1', 'x', 1)")
    con.executemany("INSERT INTO super_locus VALUES (?,?,?,?)", [(1, "AT1G01073.TAIR10", None, "gene"), (2, "AT1G01110.TAIR10", None, "gene"), (3, "AT1G09999.TAIR10", None, "gene")])
    con.executemany("INSERT INTO transcript VALUES (?,?,?,?,?)", [(15, "AT1G01073.1.TAIR10", "mRNA", 1, 1), (23, "AT1G01110.2.TAIR10", "mRNA", 0, 2),
                                                                 (24, "AT1G01110.1.TAIR10", "mRNA", 1, 2), (30, "AT1G09999.1.TAIR10", "mRNA", 1, 3)])
    con.executemany("INSERT INTO transcript_piece VALUES (?,?,?,?)", [(15, None, 0, 15), (23, None, 0, 23), (24, None, 0, 24), (30, None, 0, 30)])
    feats = [(3001, None, "geenuff_transcript", 32377, 1, 44676, 1, 1, None, None, None, 1),
             (3002, None, "geenuff_cds", 32500, 1, 44600, 1, 1, None, None, 0, 1),
             (3003, None, "geenuff_intron", 33000, 1, 33100, 1, 1, None, None, None, 1),
             (3079, None, "missing_utr_5p", 32377, 1, 44676, 1, 1, None, None, None, 1),        # soft, transcript 15 / gene 1
             (3081, None, "missing_utr_5p", 45647, 1, 52238, 1, 1, None, None, None, 1),        # soft, transcript 23 / gene 2
             (3090, None, "too_short_intron", 46000, 1, 46010, 1, 1, None, None, None, 1),      # hard, transcript 23 / gene 2
             (3091, None, "missing_start_codon", 60000, 0, 60003, 1, 1, None, None, None, 1),   # hard, transcript 30 / gene 3
             (3092, None, "empty_super_locus", 70000, 1, 70100, 1, 1, None, None, None, 1)]     # error without transcript association -> gene-level unknown
    con.executemany("INSERT INTO feature VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", feats)
    con.executemany("INSERT INTO association_transcript_piece_to_feature VALUES (?,?)",
                    [(15, 3001), (15, 3002), (15, 3003), (15, 3079), (23, 3081), (23, 3090), (30, 3091)])
    con.commit()
    con.close()


def test_extract_error_features_through_transcript_piece(qc, tmp_path):
    db = tmp_path / "g.sqlite3"
    _make_db(db)
    rows = qc.extract_from_geenuff_db(str(db), "Athaliana")
    by = {(r[1], r[2], r[3]) for r in rows}
    assert ("AT1G01073.TAIR10", "AT1G01073.1.TAIR10", "missing_utr_5p") in by
    assert ("AT1G01110.TAIR10", "AT1G01110.2.TAIR10", "missing_utr_5p") in by
    assert ("AT1G01110.TAIR10", "AT1G01110.2.TAIR10", "too_short_intron") in by
    assert ("AT1G09999.TAIR10", "AT1G09999.1.TAIR10", "missing_start_codon") in by
    assert not any(r[3].startswith("geenuff_") for r in rows)                       # biological features are not flags
    orphan = [r for r in rows if r[3] == "empty_super_locus"]
    assert orphan and orphan[0][1] == "" and orphan[0][2] == "*"                    # unassociated error kept, gene unknown
    assert {r[0] for r in rows} == {"Athaliana"}
    starts = {r[3]: (r[4], r[5]) for r in rows}
    assert starts["too_short_intron"] == (46000, 46010)


def test_cli_writes_flags_and_summary_with_hard_soft_counts(qc, tmp_path):
    db = tmp_path / "g.sqlite3"
    _make_db(db)
    out, summ = tmp_path / "flags.tsv", tmp_path / "summary.json"
    assert qc.main(["--geenuff-db", str(db), "--species-id", "Athaliana", "--out", str(out), "--summary", str(summ)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "species_id\tgene_id\ttranscript_id\tflag\tstart\tend"
    assert any(l.startswith("Athaliana\tAT1G01110.TAIR10\tAT1G01110.2.TAIR10\ttoo_short_intron\t") for l in lines)
    s = json.loads(summ.read_text())
    assert s["n_flags"] == 5 and s["by_flag"]["missing_utr_5p"] == 2
    assert s["genes_with_hard_flags"] == 2                                             # AT1G01110 (too_short_intron), AT1G09999 (missing_start_codon)
    assert s["feature_types_seen"] == ["empty_super_locus", "geenuff_cds", "geenuff_intron", "geenuff_transcript", "missing_start_codon", "missing_utr_5p", "too_short_intron"]
    assert "schema" in s and s["schema"]["association"] == "association_transcript_piece_to_feature"


def test_pinned_geenuff_error_names_map_to_hard_or_soft(qc):
    """Every error type of GeenuFF 702cbf3 has a decided A22 action: the UTR-only ones are soft, all others hard."""
    soft = {"missing_utr_5p", "missing_utr_3p"}
    for name in qc.GEENUFF_ERROR_TYPES:
        assert qc.is_hard(name) == (name not in soft), name
    assert qc.is_hard("super_loci_overlap_error") and qc.is_hard("missmatching_strands") and qc.is_hard("truncated_intron")
