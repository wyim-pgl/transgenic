"""Tests for 37_analyse_prompt_requirements.py.

These measurements are the evidence for "the prompt's UTR decides whether completion mode
works", so the tests target the ways that claim could be manufactured by the code rather
than found in the data: a locus counted in the wrong bucket, an addition credited to the
wrong group, or a denominator that quietly excludes the cases that would weaken it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prompt_requirements", Path(__file__).resolve().parents[1] / "37_analyse_prompt_requirements.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["prompt_requirements"] = mod
spec.loader.exec_module(mod)


# --------------------------------------------------------------------------- fixtures

PRIMARY_CDS = [(200, 300), (400, 500)]
ALT_CDS = [(200, 300), (420, 500)]
# Exons extend past the CDS on both sides, so the primary carries a UTR.
PRIMARY_EXONS = [(100, 300), (400, 600)]
TRUE_UTR = [(100, 199), (501, 600)]


def _gtf(path, loci):
    lines = []
    for locus in loci:
        for tx, exons, cds in locus["transcripts"]:
            attr = f'transcript_id "{tx}"; gene_id "{locus["gene"]}";'
            for start, end in exons:
                lines.append(f"Chr1\tsrc\texon\t{start}\t{end}\t.\t+\t.\t{attr}")
            for start, end in cds:
                lines.append(f"Chr1\tsrc\tCDS\t{start}\t{end}\t.\t+\t0\t{attr}")
    path.write_text("\n".join(lines) + "\n")


def _gff3(path, loci):
    lines = ["##gff-version 3"]
    for locus in loci:
        gm = f";GM={locus['gm']}" if locus.get("gm") else ""
        gene = locus["gene"]
        rows = [s for _t, feats in locus["transcripts"] for _k, ss in feats for s in ss]
        lines.append(f"Chr1\tsrc\tgene\t{min(s for s, _ in rows)}\t{max(e for _, e in rows)}"
                     f"\t.\t+\t.\tID={gene}{gm}")
        for tx, feats in locus["transcripts"]:
            spans = [s for _k, ss in feats for s in ss]
            lines.append(f"Chr1\tsrc\tmRNA\t{min(s for s, _ in spans)}\t{max(e for _, e in spans)}"
                         f"\t.\t+\t.\tID={tx};Parent={gene}{gm}")
            for kind, segs in feats:
                for n, (start, end) in enumerate(segs, 1):
                    lines.append(f"Chr1\tsrc\t{kind}\t{start}\t{end}\t.\t+\t0\t"
                                 f"ID={tx}.{kind}{n};Parent={tx}{gm}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def reference(tmp_path):
    gtf, ids = tmp_path / "tair10.gtf", tmp_path / "primary.txt"
    _gtf(gtf, [{"gene": "AT1G01010",
                "transcripts": [("AT1G01010.1", PRIMARY_EXONS, PRIMARY_CDS),
                                ("AT1G01010.2", [(200, 300), (420, 500)], ALT_CDS)]}])
    ids.write_text("AT1G01010.1\n")
    return mod.load_reference(gtf, ids)


def _prompt(tmp_path, utr):
    path = tmp_path / "prompt.gff3"
    feats = [("CDS", PRIMARY_CDS)] + ([("five_prime_UTR", utr)] if utr else [])
    _gff3(path, [{"gene": "G1", "transcripts": [("G1.t1", feats)]}])
    return mod.load_prompt(path)


def _output(tmp_path, structures):
    path = tmp_path / "out.gff3"
    _gff3(path, [{"gene": "O1", "gm": "G1",
                  "transcripts": [(f"O1.t{i}", [("CDS", s)])
                                  for i, s in enumerate(structures, 1)]}])
    return mod.load_emitted(path)


# ------------------------------------------------------------------------ utr-accuracy

def test_correct_utr_and_wrong_utr_are_separated(tmp_path, reference):
    prompt = _prompt(tmp_path, TRUE_UTR)
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_accuracy(prompt, emitted, reference)
    assert result["utr_correct"]["loci"] == 1
    assert result["utr_correct"]["additions"] == 1
    assert result["utr_correct"]["correct"] == 1
    assert result["utr_wrong"]["loci"] == 0


def test_a_wrong_utr_lands_in_the_wrong_group(tmp_path, reference):
    prompt = _prompt(tmp_path, [(100, 150)])       # shorter than the reference's UTR
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_accuracy(prompt, emitted, reference)
    assert result["utr_wrong"]["loci"] == 1
    assert result["utr_correct"]["loci"] == 0


def test_a_missing_utr_counts_as_wrong_not_as_absent(tmp_path, reference):
    """No UTR is not a third category — the reference has one and the prompt does not."""
    prompt = _prompt(tmp_path, None)
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_accuracy(prompt, emitted, reference)
    assert result["utr_wrong"]["loci"] == 1


def test_re_emitting_the_prompt_is_not_an_addition(tmp_path, reference):
    prompt = _prompt(tmp_path, TRUE_UTR)
    emitted, _ = _output(tmp_path, [PRIMARY_CDS])
    result = mod.utr_accuracy(prompt, emitted, reference)
    assert result["utr_correct"]["additions"] == 0
    assert result["utr_correct"]["precision_pct"] is None   # N/A, never 0.0


def test_locus_whose_cds_is_not_a_reference_transcript_is_excluded(tmp_path, reference):
    """The whole point is holding CDS quality fixed; a wrong CDS must not enter either group."""
    path = tmp_path / "prompt.gff3"
    _gff3(path, [{"gene": "G1", "transcripts": [("G1.t1", [("CDS", [(200, 305)])])]}])
    prompt = mod.load_prompt(path)
    emitted, _ = _output(tmp_path, [[(200, 305)], ALT_CDS])
    result = mod.utr_accuracy(prompt, emitted, reference)
    assert result["utr_correct"]["loci"] == 0
    assert result["utr_wrong"]["loci"] == 0


# ----------------------------------------------------------------------- utr-closeness

def test_exact_utr_lands_in_the_exact_bucket(tmp_path, reference):
    prompt = _prompt(tmp_path, TRUE_UTR)
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_closeness(prompt, emitted, reference)
    assert result["exact"]["loci"] == 1
    assert result["exact"]["correct"] == 1


def test_a_five_base_difference_lands_in_the_1_10_bucket(tmp_path, reference):
    # Reference UTR starts at 100; starting at 105 differs by exactly 5 bases.
    prompt = _prompt(tmp_path, [(105, 199), (501, 600)])
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_closeness(prompt, emitted, reference)
    assert result["1-10"]["loci"] == 1
    assert result["exact"]["loci"] == 0


def test_a_large_difference_lands_in_a_far_bucket(tmp_path, reference):
    prompt = _prompt(tmp_path, [(500 + 1, 600)])   # the whole 5' side missing: 100 bases
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_closeness(prompt, emitted, reference)
    assert result["51-200"]["loci"] == 1


def test_every_locus_falls_in_exactly_one_bucket(tmp_path, reference):
    prompt = _prompt(tmp_path, [(105, 199), (501, 600)])
    emitted, _ = _output(tmp_path, [PRIMARY_CDS, ALT_CDS])
    result = mod.utr_closeness(prompt, emitted, reference)
    assert sum(bucket["loci"] for bucket in result.values()) == 1


# ----------------------------------------------------------------------- generated-utr

def test_generated_utr_matching_the_reference_is_counted_exact(tmp_path, reference):
    prompt = _prompt(tmp_path, None)               # prompted with CDS only
    path = tmp_path / "out.gff3"
    _gff3(path, [{"gene": "O1", "gm": "G1",
                  "transcripts": [("O1.t1", [("CDS", PRIMARY_CDS),
                                             ("five_prime_UTR", TRUE_UTR)])]}])
    emitted, utr_of = mod.load_emitted(path)
    result = mod.generated_utr(prompt, emitted, utr_of, reference)
    assert result["prompt_preserving_transcripts"] == 1
    assert result["carried_a_generated_utr"] == 1
    assert result["generated_utr_exactly_correct"] == 1


def test_generated_utr_that_is_close_but_wrong_is_not_counted_exact(tmp_path, reference):
    prompt = _prompt(tmp_path, None)
    path = tmp_path / "out.gff3"
    _gff3(path, [{"gene": "O1", "gm": "G1",
                  "transcripts": [("O1.t1", [("CDS", PRIMARY_CDS),
                                             ("five_prime_UTR", [(101, 199), (501, 600)])])]}])
    emitted, utr_of = mod.load_emitted(path)
    result = mod.generated_utr(prompt, emitted, utr_of, reference)
    assert result["carried_a_generated_utr"] == 1
    assert result["generated_utr_exactly_correct"] == 0


def test_transcript_without_a_generated_utr_is_counted_but_not_as_carrying(tmp_path, reference):
    prompt = _prompt(tmp_path, None)
    emitted, utr_of = _output(tmp_path, [PRIMARY_CDS])
    result = mod.generated_utr(prompt, emitted, utr_of, reference)
    assert result["prompt_preserving_transcripts"] == 1
    assert result["carried_a_generated_utr"] == 0
    assert result["exact_pct_of_those_carrying"] is None


def test_locus_whose_prompt_was_overwritten_is_not_counted(tmp_path, reference):
    """`prompt_preserving` must mean the prompt survived, or the denominator is inflated."""
    prompt = _prompt(tmp_path, None)
    emitted, utr_of = _output(tmp_path, [ALT_CDS])
    result = mod.generated_utr(prompt, emitted, utr_of, reference)
    assert result["prompt_preserving_transcripts"] == 0


# ---------------------------------------------------------------------------- guards

def test_utr_outside_returns_empty_when_exons_equal_cds():
    """ANNEVO's shape: exon coordinates identical to CDS, so there is no UTR to derive."""
    assert mod.utr_outside(PRIMARY_CDS, tuple(PRIMARY_CDS)) == ()


def test_agi_normalises_both_id_shapes():
    assert mod.agi("AT1G01010.TAIR10") == "AT1G01010"
    assert mod.agi("AT1G01010.1") == "AT1G01010"
