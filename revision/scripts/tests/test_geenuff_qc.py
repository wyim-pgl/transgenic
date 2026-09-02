import sqlite3
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "62_geenuff_qc.py"
gq = types.ModuleType("geenuff_qc"); gq.__file__ = str(SCRIPT); sys.modules["geenuff_qc"] = gq
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), gq.__dict__)


def test_hard_vs_soft_flags():
    assert gq.is_hard("geenuff_error_missing_start_codon") and gq.is_hard("mismatched_ending_phase") and gq.is_hard("overlapping_exons")
    assert not gq.is_hard("missing_utr_5p") and not gq.is_hard("empty_intron_note")


def test_loss_mask_policy_a22():
    # gene-level hard error -> masked (weight 0)
    assert gq.loss_mask_decision({"*": {"empty_super_locus"}}, ["t1"]) == (0.0, [], ["empty_super_locus"])
    # one bad transcript of two -> keep the clean one, weight 1
    w, keep, hard = gq.loss_mask_decision({"t1": {"missing_stop_codon"}, "t2": {"missing_utr_3p"}}, ["t1", "t2"])
    assert w == 1.0 and keep == ["t2"] and hard == ["missing_stop_codon"]
    # every transcript bad -> masked
    assert gq.loss_mask_decision({"t1": {"wrong_starting_phase"}}, ["t1"])[0] == 0.0
    # soft only -> untouched
    assert gq.loss_mask_decision({"t1": {"missing_utr_5p"}}, ["t1"]) == (1.0, ["t1"], [])


def test_read_flags_tsv(tmp_path):
    p = tmp_path / "f.tsv"
    p.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\nAth\tg1\tt1\tmissing_stop_codon\t10\t20\nAth\tg1\t\tempty_super_locus\t0\t0\n")
    d = gq.read_flags_tsv(str(p))
    assert d[("Ath", "g1")]["t1"] == {"missing_stop_codon"} and d[("Ath", "g1")]["*"] == {"empty_super_locus"}


def test_extract_from_minimal_geenuff_like_db(tmp_path):
    db = tmp_path / "g.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE super_locus (id INTEGER PRIMARY KEY, given_name TEXT);
    CREATE TABLE transcript (id INTEGER PRIMARY KEY, given_name TEXT, super_locus_id INTEGER);
    CREATE TABLE feature (id INTEGER PRIMARY KEY, type TEXT, start INTEGER, end INTEGER);
    CREATE TABLE association_transcript_to_feature (transcript_id INTEGER, feature_id INTEGER);
    INSERT INTO super_locus VALUES (1, 'AT1G01010');
    INSERT INTO transcript VALUES (1, 'AT1G01010.1', 1);
    INSERT INTO feature VALUES (1, 'geenuff_cds', 100, 200), (2, 'geenuff_error_missing_stop_codon', 190, 200);
    INSERT INTO association_transcript_to_feature VALUES (1, 1), (1, 2);
    """)
    con.commit(); con.close()
    rows = gq.extract_from_geenuff_db(str(db), "Athaliana")
    assert rows == [("Athaliana", "AT1G01010", "AT1G01010.1", "geenuff_error_missing_stop_codon", 190, 200)]
