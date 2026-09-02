"""Grammar-constrained decoding (protocol A24)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
g = types.ModuleType("gsf_grammar"); g.__file__ = str(ROOT / "src/transgenic/utils/gsf_grammar.py"); sys.modules["gsf_grammar"] = g
exec(compile(Path(g.__file__).read_text(), g.__file__, "exec"), g.__dict__)

W = 6144


def toks(s):
    return ["<s>"] + s.split()


def test_start_digits_bounded_by_window_and_monotone():
    a = g.allowed_next(toks(""), W)
    assert "1" in a and "CDS1" not in a and "</s>" not in a
    a = g.allowed_next(toks("6 1"), W)           # 61.. can still become 6100 < 6143
    assert "0" in a and "CDS1" in a
    a = g.allowed_next(toks("6 1 4"), W)         # 614x must stay < 6143
    assert "3" not in a and "2" in a


def test_feature_names_follow_first_use_numbering_and_caps():
    a = g.allowed_next(toks("1 0"), W)
    assert {"CDS1", "five_prime_UTR1", "three_prime_UTR1"} <= a and "CDS2" not in a
    a = g.allowed_next(toks("0 CDS1 5 0 + A ; 1 0 0"), W)
    assert "CDS2" in a and "CDS1" not in a


def test_end_must_exceed_start_and_strand_is_consistent():
    a = g.allowed_next(toks("1 0 0 CDS1 1 0 0"), W)      # end == start -> must add another digit
    assert "+" not in a and "0" in a
    a = g.allowed_next(toks("1 0 0 CDS1 2 0 0"), W)
    assert {"+", "-"} <= a                                # strand may follow (more digits are still possible too)
    a = g.allowed_next(toks("1 0 0 CDS1 2 0 0 + A ; 3 0 0 CDS2 4 0 0"), W)
    assert "+" in a and "-" not in a                      # second feature cannot flip strand


def test_phase_letters_depend_on_feature_type():
    assert g.allowed_next(toks("1 0 0 CDS1 2 0 0 +"), W) == {"A", "B", "C"}
    assert g.allowed_next(toks("1 0 0 five_prime_UTR1 2 0 0 +"), W) == {"."}


def test_transcript_plan_and_membership():
    base = "1 0 0 CDS1 2 0 0 + A ; 3 0 0 CDS2 4 0 0 + B"
    a = g.allowed_next(toks(base), W)
    assert ";" in a and "<tx1>" in a and "<tx2>" in a and "<tx3>" not in a   # only two features exist
    assert g.allowed_next(toks(base + " <tx2>"), W) == {">"}
    a = g.allowed_next(toks(base + " <tx2> >"), W)
    assert a == {"CDS1", "CDS2"}
    a = g.allowed_next(toks(base + " <tx2> > CDS2"), W)
    assert "CDS1" not in a and "<iso>" in a and "</s>" not in a              # plan says two transcripts; CDS1 lies upstream of CDS2 on +
    a = g.allowed_next(toks(base + " <tx2> > CDS1 CDS2 <iso> CDS2"), W)
    assert a == {"</s>"} or a == {"</s>", "CDS1"} - {"CDS1"}


def test_utr_only_transcript_cannot_close():
    base = "1 0 0 five_prime_UTR1 2 0 0 + . ; 3 0 0 CDS1 4 0 0 + A <tx1> > five_prime_UTR1"
    a = g.allowed_next(toks(base), W)
    assert "</s>" not in a and "CDS1" in a


def test_validate_gsf_reports_each_violation_class():
    ok = "0|CDS1|300|+|A;400|CDS2|502|+|A>CDS1|CDS2"
    assert g.validate_gsf(ok, W) == []
    bad = "500|CDS1|300|-|A;100|CDS2|200|+|.;100|CDS2|200|+|A>CDS1|CDS2|CDS9;CDS2|CDS2"
    v = "\n".join(g.validate_gsf(bad, W))
    for key in ("reversed", "not coordinate-sorted", "phase", "mixed strands", "duplicate", "undefined", "repeated", "multiple of 3"):
        assert key in v, key


def test_v3_gene_separator_and_empty_label():
    assert "<empty>" in g.allowed_next(toks(""), W, v3=True) and "<empty>" not in g.allowed_next(toks(""), W)
    assert g.allowed_next(toks("<empty>"), W, v3=True) == {"</s>"}
    base = "1 0 0 CDS1 2 0 0 + A <tx1> > CDS1"
    a = g.allowed_next(toks(base), W, v3=True)
    assert {"</s>", "<gene>"} <= a
    a = g.allowed_next(toks(base + " <gene>"), W, v3=True)
    assert "<empty>" not in a and "0" not in a and "2" in a     # next gene starts at/after 200: a lone 0 is impossible, 1xxx is still possible
    a = g.allowed_next(toks(base + " <gene> 3 0 0"), W, v3=True)
    assert "CDS1" in a                                             # numbering restarts per gene
    a = g.allowed_next(toks(base + " <gene> 3 0 0 CDS1 4 0 0"), W, v3=True)
    assert {"+", "-"} <= a                                         # strand is free again for the new gene
