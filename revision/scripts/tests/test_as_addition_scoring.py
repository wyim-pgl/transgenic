"""Tests for 34_score_as_additions.py.

Every test here is written against a behaviour that could plausibly go wrong in the
direction that flatters the result, because that is how this benchmark has failed before:
a quantity that counted one side of a ledger and never the other. So the suite checks the
gain side as well as the loss side — additions that should NOT be counted, denominators
that should NOT silently shrink, and matches that should NOT be credited.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "as_additions", Path(__file__).resolve().parents[1] / "34_score_as_additions.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["as_additions"] = mod
spec.loader.exec_module(mod)


# --------------------------------------------------------------------------- fixtures

def _gff3(path: Path, loci: list[dict], extra_lines: list[str] | None = None) -> None:
    """Write a real 3-level gene -> mRNA -> CDS GFF3.

    Each locus dict takes:
        gene         gene id, written as ID= on the gene row
        seq          sequence name (default "Chr1")
        strand       (default "+")
        name         when given, written as Name= BEFORE ID= — GeMoMa's attribute order,
                     which is what makes the model's GM= value the Name and not the ID
        gm           when given, written as GM= on every row, as completion-mode output does
        transcripts  [(transcript_id, [(start, end), ...]), ...] in file order; the FIRST
                     one is what preprocess.py hands the decoder as the prompt
    """
    lines: list[str] = []
    for locus in loci:
        seq = locus.get("seq", "Chr1")
        strand = locus.get("strand", "+")
        segments = [s for _tx, segs in locus["transcripts"] for s in segs]
        gene_start = min(s for s, _e in segments)
        gene_end = max(e for _s, e in segments)
        attrs: list[str] = []
        if locus.get("name"):
            attrs.append(f"Name={locus['name']}")
        attrs.append(f"ID={locus['gene']}")
        gm = f";GM={locus['gm']}" if locus.get("gm") else ""
        lines.append(f"{seq}\tsrc\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
                     + ";".join(attrs) + gm)
        for tx, segs in locus["transcripts"]:
            lines.append(f"{seq}\tsrc\tmRNA\t{min(s for s, _e in segs)}\t"
                         f"{max(e for _s, e in segs)}\t.\t{strand}\t.\t"
                         f"ID={tx};Parent={locus['gene']}{gm}")
            for n, (s, e) in enumerate(segs, 1):
                lines.append(f"{seq}\tsrc\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                             f"ID={tx}.cds{n};Parent={tx}{gm}")
    lines.extend(extra_lines or [])
    path.write_text("\n".join(lines) + "\n")


def _gtf(path: Path, loci: list[dict]) -> None:
    """Write a TAIR10/AtRTD3-shaped GTF (transcript/exon/CDS rows, gene_id + transcript_id)."""
    lines: list[str] = []
    for locus in loci:
        seq = locus.get("seq", "Chr1")
        strand = locus.get("strand", "+")
        for tx, segs in locus["transcripts"]:
            attr = f'transcript_id "{tx}"; gene_id "{locus["gene"]}";'
            lines.append(f"{seq}\tsrc\ttranscript\t{min(s for s, _e in segs)}\t"
                         f"{max(e for _s, e in segs)}\t.\t{strand}\t.\t{attr}")
            for s, e in segs:
                lines.append(f"{seq}\tsrc\texon\t{s}\t{e}\t.\t{strand}\t.\t{attr}")
                lines.append(f"{seq}\tsrc\tCDS\t{s}\t{e}\t.\t{strand}\t0\t{attr}")
    path.write_text("\n".join(lines) + "\n")


PRIMARY = [(100, 200), (300, 400)]
ALT1 = [(100, 200), (320, 400)]
ALT2 = [(100, 200), (340, 400)]


@pytest.fixture
def refs(tmp_path):
    """A one-locus TAIR10 with one primary and two alternatives, plus an empty AtRTD3."""
    tair10 = tmp_path / "tair10.gtf"
    atrtd3 = tmp_path / "atrtd3.gtf"
    primary_ids = tmp_path / "primary.txt"
    _gtf(tair10, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY),
                                   ("AT1G01010.2", ALT1),
                                   ("AT1G01010.3", ALT2)]}])
    _gtf(atrtd3, [{"gene": "AT9G99999", "seq": "Chr9",
                   "transcripts": [("AT9G99999.1", [(10, 20)])]}])
    primary_ids.write_text("AT1G01010.1\n")
    return dict(tair10=tair10, atrtd3=atrtd3, primary_ids=primary_ids)


def _score(tmp_path, refs, input_loci, output_loci, extra_input_lines=None):
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff3(inp, input_loci, extra_input_lines)
    _gff3(out, output_loci)
    return mod.score(inp, out, refs["tair10"], refs["atrtd3"], refs["primary_ids"])


# ------------------------------------------------------------------ the headline counts

def test_addition_matching_a_tair10_alternative_is_counted(tmp_path, refs):
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1)]}])
    assert r["added_structures"] == 1
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1
    assert r["precision_vs_TAIR10_alternatives_pct"] == 100.0
    assert r["loci_with_at_least_one_addition"] == 1
    assert r["loci_scored"] == 1


def test_identical_emissions_collapse_to_one_addition(tmp_path, refs):
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1), ("O1.t3", ALT1)]}])
    assert r["added_structures"] == 1
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1


def test_structure_equal_to_the_prompt_is_not_an_addition(tmp_path, refs):
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1", "transcripts": [("O1.t1", PRIMARY)]}])
    assert r["added_structures"] == 0
    assert r["loci_with_at_least_one_addition"] == 0
    # N/A, never 0: 0/0 precision would read as "everything it added was wrong".
    assert r["precision_vs_TAIR10_alternatives_pct"] is None
    assert r["precision_vs_AtRTD3_pct"] is None


def test_the_prompt_is_the_first_transcript_in_file_order(tmp_path, refs):
    """A second input isoform is NOT the prompt, so re-emitting it IS an addition.

    Both definitions are reported: `added_structures` counts everything that differs from
    the prompt (what the model was actually given), the decomposition field counts only
    what differs from every transcript the input already held (32_score_polishing.py's
    definition). They coincide exactly when the input is single-transcript per locus.
    """
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY), ("G1.t2", ALT1)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1), ("O1.t3", ALT2)]}])
    assert r["added_structures"] == 2
    assert r["decomposition"]["added_structures_vs_all_input_transcripts"] == 1
    assert r["decomposition"]["input_loci_with_multiple_transcripts"] == 1
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 2


def test_recall_counts_reference_alternatives_recovered(tmp_path, refs):
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1)]}])
    assert r["reference_alternative_transcripts"] == 2
    assert r["recall_of_TAIR10_alternatives_pct"] == 50.0


def test_recall_is_na_when_the_reference_has_no_alternatives(tmp_path, refs):
    single = tmp_path / "single.gtf"
    _gtf(single, [{"gene": "AT1G01010", "transcripts": [("AT1G01010.1", PRIMARY)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1)]}])
    r = mod.score(inp, out, single, refs["atrtd3"], refs["primary_ids"])
    assert r["reference_alternative_transcripts"] == 0
    assert r["recall_of_TAIR10_alternatives_pct"] is None


# ------------------------------------------------------------------- the intron chain

def test_intron_chain_match_is_looser_than_exact_cds(tmp_path, refs):
    terminus_variant = [(90, 200), (320, 410)]      # ALT1's chain, different termini
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", terminus_variant)]}])
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 0
    assert r["added_matching_TAIR10_alternative_intron_chain"] == 1
    assert r["precision_vs_TAIR10_alternatives_intron_chain_pct"] == 100.0


def test_single_exon_addition_never_matches_on_intron_chain(tmp_path, refs):
    """A single-CDS structure has an empty chain, which would otherwise match every other
    single-exon reference transcript's empty chain — a free, meaningless hit."""
    single_exon_ref = tmp_path / "se.gtf"
    _gtf(single_exon_ref, [{"gene": "AT1G01010",
                            "transcripts": [("AT1G01010.1", PRIMARY),
                                            ("AT1G01010.2", [(100, 250)])]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", [(100, 200)])]}])
    r = mod.score(inp, out, single_exon_ref, refs["atrtd3"], refs["primary_ids"])
    assert r["added_structures"] == 1
    assert r["added_matching_TAIR10_alternative_intron_chain"] == 0


def test_chain_match_that_reuses_the_prompt_chain_is_reported_separately(tmp_path, refs):
    """The 48x swing between exact-CDS and intron-chain precision recorded in
    32_score_polishing.py's docstring was almost entirely terminus variants of the
    supplied transcript. That decomposition has to be visible, not inferable."""
    prompt_chain_variant = [(90, 200), (300, 410)]   # PRIMARY's chain, different termini
    ref = tmp_path / "ref.gtf"
    _gtf(ref, [{"gene": "AT1G01010",
                "transcripts": [("AT1G01010.1", PRIMARY),
                                ("AT1G01010.2", [(95, 200), (300, 405)])]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", prompt_chain_variant)]}])
    r = mod.score(inp, out, ref, refs["atrtd3"], refs["primary_ids"])
    d = r["decomposition"]
    assert r["added_matching_TAIR10_alternative_intron_chain"] == 1
    assert d["added_matching_TAIR10_alternative_intron_chain_reusing_prompt_chain"] == 1
    assert d["added_matching_TAIR10_alternative_intron_chain_distinct_from_prompt"] == 0


# ------------------------------------------------------------------------ AtRTD3 side

def test_atrtd3_match_is_counted_and_is_independent_of_tair10(tmp_path, refs):
    novel = [(100, 200), (360, 400)]
    atrtd3 = tmp_path / "art.gtf"
    _gtf(atrtd3, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.5", novel)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", novel)]}])
    r = mod.score(inp, out, refs["tair10"], atrtd3, refs["primary_ids"])
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 0
    assert r["added_matching_any_AtRTD3_transcript"] == 1
    assert r["precision_vs_AtRTD3_pct"] == 100.0


def test_tair10_gene_id_suffix_is_normalised_before_joining_to_atrtd3(tmp_path):
    """TAIR10.gtf writes gene_id "AT1G01010.TAIR10"; AtRTD3 writes "AT1G01010". Without
    normalisation every AtRTD3 lookup misses and the count is a silent, plausible 0."""
    tair10, atrtd3, primary_ids = (tmp_path / "t.gtf", tmp_path / "a.gtf",
                                   tmp_path / "p.txt")
    _gtf(tair10, [{"gene": "AT1G01010.TAIR10",
                   "transcripts": [("AT1G01010.1.TAIR10", PRIMARY),
                                   ("AT1G01010.2.TAIR10", ALT1)]}])
    _gtf(atrtd3, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.2", ALT1)]}])
    primary_ids.write_text("AT1G01010.1.TAIR10\n")
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("G1.t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("O1.t1", PRIMARY), ("O1.t2", ALT1)]}])
    r = mod.score(inp, out, tair10, atrtd3, primary_ids)
    assert r["added_matching_any_AtRTD3_transcript"] == 1
    assert r["diagnostics"]["reference_loci_without_AtRTD3_counterpart"] == 0


# ------------------------------------------------------------------------- the pairing

def test_gm_pairs_on_the_first_attribute_value_not_the_id(tmp_path, refs):
    """GeMoMa's gene rows lead with Name=, and preprocess.py derives GM= from whichever
    attribute comes first. Matching GM= against ID= silently sends every locus down the
    positional fallback — measured on the real GeMoMa run as gm_paired=0 of 29,561."""
    r = _score(tmp_path, refs,
               [{"gene": "gene_0", "name": "Ath_00001",
                 "transcripts": [("gene_0.t1", PRIMARY)]}],
               [{"gene": "A_thaliana_g000001", "gm": "Ath_00001",
                 "transcripts": [("A_thaliana_g000001.t1", PRIMARY),
                                 ("A_thaliana_g000001.t2", ALT1)]}])
    assert r["diagnostics"]["pairing"]["gm_paired"] == 1
    assert r["diagnostics"]["pairing"]["overlap_fallback"] == 0
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1


def test_gene_ids_are_never_compared_across_files(tmp_path, refs):
    """standardize_gff.py renumbers by sort position, so the output's ID=A_thaliana_g000001
    is a DIFFERENT locus from the input's ID=A_thaliana_g000001 whenever anything upstream
    was dropped. Pairing must follow GM=, which carries the real provenance."""
    far = [(1000, 1100), (1300, 1400)]
    far_alt = [(1000, 1100), (1320, 1400)]
    tair10 = tmp_path / "t.gtf"
    _gtf(tair10, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.2", ALT1)]},
                  {"gene": "AT1G01020",
                   "transcripts": [("AT1G01020.1", far), ("AT1G01020.2", far_alt)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "A_thaliana_g000001", "transcripts": [("t1", PRIMARY)]},
                {"gene": "A_thaliana_g000002", "transcripts": [("t2", far)]}])
    _gff3(out, [{"gene": "A_thaliana_g000001", "gm": "A_thaliana_g000002",
                 "transcripts": [("o1", far), ("o2", far_alt)]}])
    r = mod.score(inp, out, tair10, refs["atrtd3"], refs["primary_ids"])
    assert r["diagnostics"]["pairing"]["gm_paired"] == 1
    assert r["loci_scored"] == 1
    assert r["diagnostics"]["loci_without_output"] == 1
    assert r["added_structures"] == 1
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1


def test_ambiguous_first_attribute_values_are_reported_not_guessed(tmp_path, refs):
    """Two input genes sharing a Name= would make GM= ambiguous. Picking one at random
    would be a silent mis-attribution; those loci must fall to the positional fallback
    and the ambiguity must appear in the JSON."""
    far = [(1000, 1100), (1300, 1400)]
    r = _score(tmp_path, refs,
               [{"gene": "gene_0", "name": "dup", "transcripts": [("a", PRIMARY)]},
                {"gene": "gene_1", "name": "dup", "transcripts": [("b", far)]}],
               [{"gene": "O1", "gm": "dup",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}])
    assert r["diagnostics"]["ambiguous_input_gm_keys"] == 2
    assert r["diagnostics"]["pairing"]["gm_paired"] == 0
    assert r["diagnostics"]["pairing"]["overlap_fallback"] == 1


# --------------------------------------------------------- denominators and gain side

def test_locus_without_output_is_excluded_from_the_denominator_and_reported(tmp_path, refs):
    far = [(1000, 1100), (1300, 1400)]
    far_alt = [(1000, 1100), (1320, 1400)]
    tair10 = tmp_path / "t.gtf"
    _gtf(tair10, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.2", ALT1)]},
                  {"gene": "AT1G01020",
                   "transcripts": [("AT1G01020.1", far), ("AT1G01020.2", far_alt)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("t1", PRIMARY)]},
                {"gene": "G2", "transcripts": [("t2", far)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}])
    r = mod.score(inp, out, tair10, refs["atrtd3"], refs["primary_ids"])
    assert r["loci_scored"] == 1
    assert r["diagnostics"]["loci_without_output"] == 1
    # G2's alternative is not recoverable and must not dilute recall — but it must not
    # vanish either, or a reader cannot tell coverage from performance.
    assert r["reference_alternative_transcripts"] == 1
    assert r["diagnostics"]["reference_alternative_transcripts_at_loci_without_output"] == 1
    assert r["recall_of_TAIR10_alternatives_pct"] == 100.0


def test_output_locus_with_no_input_partner_is_counted(tmp_path, refs):
    """The excess/gain side. Every previous defect in this benchmark counted loss and
    never gain; an output locus that corresponds to no input locus is exactly that."""
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]},
                {"gene": "O2", "gm": "ghost", "seq": "Chr2",
                 "transcripts": [("o3", [(5000, 5100), (5300, 5400)])]}])
    assert r["diagnostics"]["output_loci"] == 2
    assert r["diagnostics"]["output_loci_without_input_partner"] == 1


def test_prompt_survival_splits_additions_from_replacements(tmp_path, refs):
    """An "addition" at a locus whose prompt was destroyed is a replacement, not an added
    isoform. 84-89% of `added_structures` sat there in the polishing run."""
    far = [(1000, 1100), (1300, 1400)]
    far_alt = [(1000, 1100), (1320, 1400)]
    tair10 = tmp_path / "t.gtf"
    _gtf(tair10, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.2", ALT1)]},
                  {"gene": "AT1G01020",
                   "transcripts": [("AT1G01020.1", far), ("AT1G01020.2", far_alt)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("t1", PRIMARY)]},
                {"gene": "G2", "transcripts": [("t2", far)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]},
                {"gene": "O2", "gm": "G2", "transcripts": [("o3", far_alt)]}])
    r = mod.score(inp, out, tair10, refs["atrtd3"], refs["primary_ids"])
    d = r["decomposition"]
    assert r["added_structures"] == 2
    assert d["prompt_survived_loci"] == 1
    assert d["prompt_destroyed_loci"] == 1
    assert d["added_at_loci_where_prompt_survived"] == 1
    assert d["added_at_loci_where_prompt_destroyed"] == 1
    assert d["added_matching_TAIR10_alternative_exact_CDS_at_prompt_survived_loci"] == 1


def test_additions_at_loci_with_no_reference_gene_are_disclosed(tmp_path, refs):
    """They cannot be right against TAIR10 and are kept in the headline denominator (the
    same convention 28_score_added_isoforms.py uses), so the restricted denominator has to
    be reported alongside or the precision floor reads as a measurement."""
    orphan = [(9000, 9100), (9300, 9400)]
    orphan_alt = [(9000, 9100), (9320, 9400)]
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("t1", PRIMARY)]},
                {"gene": "G2", "seq": "Chr8", "transcripts": [("t2", orphan)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]},
                {"gene": "O2", "gm": "G2", "seq": "Chr8",
                 "transcripts": [("o3", orphan), ("o4", orphan_alt)]}])
    d = r["decomposition"]
    assert r["added_structures"] == 2
    assert d["added_structures_at_reference_matched_loci"] == 1
    assert r["precision_vs_TAIR10_alternatives_pct"] == 50.0
    assert d["precision_vs_TAIR10_alternatives_at_reference_matched_loci_pct"] == 100.0
    assert r["diagnostics"]["input_loci_without_reference_match"] == 1


def test_the_field_that_reconciles_with_32_score_polishing_is_reported(tmp_path, refs):
    """`32_score_polishing.py.added_structures` differs from this module's headline on two
    axes at once — it counts only at reference-matched loci, and only structures absent
    from the whole input locus. Both restrictions applied together is the one field that
    has to reproduce it, so all four combinations are asserted here."""
    orphan = [(9000, 9100), (9300, 9400)]
    orphan_alt = [(9000, 9100), (9320, 9400)]
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("t1", PRIMARY), ("t2", ALT1)]},
                {"gene": "G2", "seq": "Chr8", "transcripts": [("t3", orphan)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1), ("o3", ALT2)]},
                {"gene": "O2", "gm": "G2", "seq": "Chr8",
                 "transcripts": [("o4", orphan), ("o5", orphan_alt)]}])
    d = r["decomposition"]
    assert r["added_structures"] == 3                                    # prompt, all loci
    assert d["added_structures_vs_all_input_transcripts"] == 2           # all-input, all loci
    assert d["added_structures_at_reference_matched_loci"] == 2          # prompt, matched
    assert d["added_structures_vs_all_input_transcripts_at_reference_matched_loci"] == 1


# ---------------------------------------------------------------- reference resolution

def test_reference_primary_comes_from_primary_ids_not_file_order(tmp_path, refs):
    """If file order picked the primary here, ALT1 would be treated as the primary and
    excluded from the alternatives, and a correct addition would score as wrong."""
    tair10 = tmp_path / "t.gtf"
    _gtf(tair10, [{"gene": "AT1G01010",
                   "transcripts": [("AT1G01010.2", ALT1),       # alternative, written FIRST
                                   ("AT1G01010.1", PRIMARY)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}])
    r = mod.score(inp, out, tair10, refs["atrtd3"], refs["primary_ids"])
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1
    assert r["diagnostics"]["reference_primary_fallback_to_file_order"] == 0


def test_exact_match_on_the_opposite_strand_is_flagged(tmp_path, refs):
    """CDS structures are compared as coordinate tuples, so an antisense overlapping gene
    can produce a coordinate-identical "match" that is biologically nothing."""
    tair10 = tmp_path / "t.gtf"
    _gtf(tair10, [{"gene": "AT1G01010", "strand": "-",
                   "transcripts": [("AT1G01010.1", PRIMARY), ("AT1G01010.2", ALT1)]}])
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "strand": "+", "transcripts": [("t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1", "strand": "+",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}])
    r = mod.score(inp, out, tair10, refs["atrtd3"], refs["primary_ids"])
    assert r["added_matching_TAIR10_alternative_exact_CDS"] == 1
    assert r["diagnostics"][
        "added_matching_TAIR10_alternative_exact_CDS_opposite_strand"] == 1


def test_noncoding_input_features_are_excluded_and_reported(tmp_path, refs):
    lnc = ["Chr1\tsrc\tlnc_RNA\t2000\t2100\t.\t+\t.\tID=nc1",
           "Chr1\tsrc\texon\t2000\t2100\t.\t+\t.\tID=nc1.e1;Parent=nc1"]
    r = _score(tmp_path, refs,
               [{"gene": "G1", "transcripts": [("t1", PRIMARY)]}],
               [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}],
               extra_input_lines=lnc)
    assert r["diagnostics"]["input_noncoding_features_excluded"] == {"lnc_RNA": 1}
    assert r["diagnostics"]["input_coding_loci"] == 1


# ------------------------------------------------------------------------------- CLI

def test_cli_records_every_argument_in_provenance(tmp_path, refs):
    inp, out, js = tmp_path / "in.gff3", tmp_path / "out.gff3", tmp_path / "r.json"
    _gff3(inp, [{"gene": "G1", "transcripts": [("t1", PRIMARY)]}])
    _gff3(out, [{"gene": "O1", "gm": "G1",
                 "transcripts": [("o1", PRIMARY), ("o2", ALT1)]}])
    rc = mod.main(["--input", str(inp), "--output", str(out), "--tool", "helixer",
                   "--tair10", str(refs["tair10"]), "--atrtd3", str(refs["atrtd3"]),
                   "--primary-ids", str(refs["primary_ids"]), "--json", str(js)])
    assert rc == 0
    p = json.loads(js.read_text())["provenance"]
    assert p["tool"] == "helixer"
    assert p["input"] == str(inp)
    assert p["output"] == str(out)
    assert p["reference_TAIR10"] == str(refs["tair10"])
    assert p["reference_AtRTD3"] == str(refs["atrtd3"])
    assert p["primary_ids"] == str(refs["primary_ids"])
    for key in ("input", "output", "tool", "tair10", "atrtd3", "primary_ids", "json"):
        assert key in p["cli_args"]
    assert p["argv"][0] == "--input"


def test_empty_side_raises_rather_than_reporting_zeros(tmp_path, refs):
    inp, out = tmp_path / "in.gff3", tmp_path / "out.gff3"
    _gff3(inp, [{"gene": "G1", "transcripts": [("t1", PRIMARY)]}])
    out.write_text("##gff-version 3\n")
    with pytest.raises(ValueError):
        mod.score(inp, out, refs["tair10"], refs["atrtd3"], refs["primary_ids"])
