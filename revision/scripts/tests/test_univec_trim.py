import gzip
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "61_univec_trim.py"
uv = types.ModuleType("univec_trim"); uv.__file__ = str(SCRIPT); sys.modules["univec_trim"] = uv
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), uv.__dict__)


def test_classify_terminal_and_internal_thresholds():
    assert uv.classify(1, 30, 500, 24) == ("terminal", "strong")
    assert uv.classify(1, 30, 500, 19) == ("terminal", "moderate")
    assert uv.classify(1, 30, 500, 16) == ("terminal", "weak")
    assert uv.classify(480, 500, 500, 24) == ("terminal", "strong")   # within 25 nt of the 3' end
    assert uv.classify(200, 240, 500, 29) == ("internal", "moderate")
    assert uv.classify(200, 240, 500, 30) == ("internal", "strong")
    assert uv.classify(200, 240, 500, 22) == ("internal", "none")


def test_plan_terminal_trim_both_ends():
    action, pieces, tags = uv.plan_record(600, [(1, 40, 30, 600), (570, 600, 25, 600)], min_len=100)
    assert action == "trim" and pieces == [(41, 569)]


def test_plan_internal_strong_splits_and_keeps_long_pieces():
    action, pieces, _ = uv.plan_record(700, [(300, 340, 35, 700)], min_len=100)
    assert action == "split" and pieces == [(1, 299), (341, 700)]
    action, pieces, _ = uv.plan_record(400, [(300, 340, 35, 400)], min_len=100)   # right piece is 60 nt -> dropped
    assert action == "split" and pieces == [(1, 299)]


def test_plan_weak_and_internal_moderate_are_flag_only():
    action, pieces, tags = uv.plan_record(500, [(1, 20, 16, 500), (200, 240, 27, 500)], min_len=100)
    assert action == "keep" and pieces == [(1, 500)] and len(tags) == 2


def test_plan_drop_when_too_short_after_trim():
    action, pieces, _ = uv.plan_record(150, [(1, 40, 30, 150), (120, 150, 25, 150)], min_len=100)
    assert action == "drop" and pieces == []


def test_run_end_to_end(tmp_path):
    fa = tmp_path / "est.fa.gz"
    with gzip.open(fa, "wt") as fh:
        fh.write(">A1 desc one\n" + "A" * 600 + "\n>B1\n" + "C" * 700 + "\n>C1\n" + "G" * 500 + "\n>D1 short\n" + "T" * 80 + "\n")
    hits = tmp_path / "hits.tsv"
    hits.write_text("A1\tvec\t100\t40\t1\t40\t1\t40\t1e-5\t30\t600\n" "B1\tvec\t100\t41\t300\t340\t1\t41\t1e-5\t35\t700\n")
    out = tmp_path / "trim.fa.gz"; rep = tmp_path / "rep.tsv"; summ = tmp_path / "s.json"
    c = uv.run(str(fa), str(hits), str(out), str(rep), str(summ), min_len=100)
    recs = dict((l[1:].split()[0], None) for l in gzip.open(out, "rt") if l.startswith(">"))
    assert set(recs) == {"A1", "B1_part1", "B1_part2", "C1"}   # D1 dropped (80 nt < 100)
    assert c["trim"] == 1 and c["split"] == 1 and c["keep"] == 1 and c["drop"] == 1 and c["records_out"] == 4
    lines = rep.read_text().splitlines()
    assert lines[0].startswith("accession") and any(l.startswith("B1\t700\tsplit\t1-299;341-700") for l in lines)
    assert json.load(open(summ))["records_in"] == 4
