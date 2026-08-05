import importlib.util, sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "score", Path(__file__).resolve().parents[1] / "32_score_polishing.py")
score_mod = importlib.util.module_from_spec(spec)
sys.modules["score"] = score_mod
spec.loader.exec_module(score_mod)


def _gff(path, rows):
    """rows: (seq, feat, start, end, gene, tx)"""
    with open(path, "w") as fh:
        for seq, feat, s, e, g, t in rows:
            attr = f"ID={t};Parent={g}" if feat == "CDS" else f"ID={g}"
            fh.write(f"{seq}\tx\t{feat}\t{s}\t{e}\t.\t+\t0\t{attr}\n")


def test_repair_and_damage_are_counted_separately(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    # locus A: input wrong, output right  -> repaired
    # locus B: input right, output wrong  -> damaged
    # locus C: input right, output right  -> preserved
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1"),
               ("Chr1", "CDS", 500, 600, "B", "B.1"),
               ("Chr1", "CDS", 900, 1000, "C", "C.1")])
    _gff(inp, [("Chr1", "CDS", 100, 190, "A", "A.i"),      # wrong end
               ("Chr1", "CDS", 500, 600, "B", "B.i"),      # correct
               ("Chr1", "CDS", 900, 1000, "C", "C.i")])    # correct
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o"),      # fixed
               ("Chr1", "CDS", 500, 690, "B", "B.o"),      # broken
               ("Chr1", "CDS", 900, 1000, "C", "C.o")])    # kept
    r = score_mod.score(inp, out, ref)
    assert r["repaired"] == 1
    assert r["damaged"] == 1
    assert r["preserved_correct"] == 1
    assert r["still_wrong"] == 0


def test_added_isoform_is_not_counted_as_damage(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1"),
               ("Chr1", "CDS", 300, 400, "A", "A.2")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o1"),
               ("Chr1", "CDS", 300, 400, "A", "A.o2")])
    r = score_mod.score(inp, out, ref)
    assert r["preserved_correct"] == 1
    assert r["damaged"] == 0
    assert r["added_structures"] == 1
    assert r["added_matching_reference"] == 1


# -- judgement calls beyond the brief: real 3-level gene -> mRNA -> CDS hierarchy ---
#
# The two tests above use the brief's flattened fixture (CDS parented straight to the
# "gene" column). GeMoMa, BRAKER3 and EGAPx are real 3-level annotations that already
# carry alternative isoforms of their own — the scenarios below exercise that shape
# directly with raw GFF3 lines, matching what 30_stage_external_annotations.py writes.


def _gff3_lines(rows: list[tuple]) -> str:
    """rows: (seq, feat, start, end, id_attr_or_None, parent_attr_or_None, extra_attrs)"""
    lines = []
    for seq, feat, s, e, fid, parent, extra in rows:
        attrs = []
        if fid:
            attrs.append(f"ID={fid}")
        if parent:
            attrs.append(f"Parent={parent}")
        if extra:
            attrs.append(extra)
        lines.append(f"{seq}\tx\t{feat}\t{s}\t{e}\t.\t+\t0\t{';'.join(attrs)}\n")
    return "".join(lines)


def test_representative_is_first_mrna_in_file_order(tmp_path):
    """Input already carries two isoforms of gene A; the first (file order) is what
    was actually supplied to the model, even though the second happens to already
    match the reference exactly. The transition table must classify on the first,
    and the output must not get credit for reproducing the input's own second
    isoform as if it were something newly added."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 190, "A.i1", "A", None),   # first in file order: wrong
        ("Chr1", "CDS", 100, 190, "A.i1.cds", "A.i1", None),
        ("Chr1", "mRNA", 100, 200, "A.i2", "A", None),   # second: already correct
        ("Chr1", "CDS", 100, 200, "A.i2.cds", "A.i2", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 190, "A.o1", "A", None),   # unchanged primary: still wrong
        ("Chr1", "CDS", 100, 190, "A.o1.cds", "A.o1", None),
        ("Chr1", "mRNA", 100, 200, "A.o2", "A", None),   # reproduces the input's own A.i2
        ("Chr1", "CDS", 100, 200, "A.o2.cds", "A.o2", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["still_wrong"] == 1          # classified on the first mRNA, not the second
    assert r["repaired"] == 0
    assert r["preserved_correct"] == 0
    assert r["added_structures"] == 0     # A.o2 duplicates the input's own A.i2 exactly
    assert r["added_matching_reference"] == 0


def test_genuinely_new_isoform_is_added_and_scored_against_reference(tmp_path):
    """A locus with only one input isoform; the output's second isoform is new and
    matches the reference — this one *should* be credited as an addition."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 400, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
        ("Chr1", "mRNA", 300, 400, "A.2", "A", None),
        ("Chr1", "CDS", 300, 400, "A.2.cds", "A.2", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i1.cds", "A.i1", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 400, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.o1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o1.cds", "A.o1", None),
        ("Chr1", "mRNA", 300, 400, "A.o2", "A", None),   # new, matches ref A.2
        ("Chr1", "CDS", 300, 400, "A.o2.cds", "A.o2", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["preserved_correct"] == 1
    assert r["added_structures"] == 1
    assert r["added_matching_reference"] == 1


# -- UTR-level (CDS+UTR) transition table: N/A unless the input shows real UTR signal --


def test_utr_level_is_na_when_input_has_no_exon_rows(tmp_path):
    """GeMoMa's shape: CDS only, no exon rows at all."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref)
    assert r["utr_level"]["status"] == "N/A"
    assert "exon" in r["utr_level"]["reason"]


def test_utr_level_is_na_when_input_exon_equals_cds(tmp_path):
    """BRAKER3's shape: exon rows present but identical to CDS everywhere (no UTR)."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "exon", 100, 200, "A.i.exon", "A.i", None),  # exon == CDS exactly
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.o", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o.cds", "A.o", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["utr_level"]["status"] == "N/A"
    assert "UTR" in r["utr_level"]["reason"]


def test_utr_level_is_computed_when_input_has_real_utr(tmp_path):
    """EGAPx's shape: exon rows extend beyond CDS (real UTR)."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    # reference: correct CDS (100-200) with UTR-extended exon (80-220)
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 80, 220, "A", None, None),
        ("Chr1", "mRNA", 80, 220, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
        ("Chr1", "exon", 80, 220, "A.1.exon", "A.1", None),
    ]))
    # input: right CDS, but wrong (unextended) UTR -> damaged at the UTR level only
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "exon", 90, 210, "A.i.exon", "A.i", None),
    ]))
    # output: same CDS, and now matches the reference's exon/UTR extent exactly
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 80, 220, "A", None, None),
        ("Chr1", "mRNA", 80, 220, "A.o", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o.cds", "A.o", None),
        ("Chr1", "exon", 80, 220, "A.o.exon", "A.o", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["preserved_correct"] == 1        # CDS-level: right in both input and output
    assert "status" not in r["utr_level"]     # a real transition table, not N/A
    assert r["utr_level"]["repaired"] == 1    # UTR-level: wrong extent in, right extent out


# -- non-coding features (lnc_RNA / pseudogene / bare "transcript") are excluded ----


def test_noncoding_features_excluded_from_loci_and_reported(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    # coding gene A plus a non-coding lncRNA locus (a CDS parented straight to it must
    # still be excluded, not just a locus with no CDS at all)
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "gene", 1000, 1100, "L", None, 'gene_biotype=lncRNA'),
        ("Chr1", "lnc_RNA", 1000, 1100, "L.rna", "L", None),
        ("Chr1", "CDS", 1000, 1050, "L.cds", "L.rna", None),  # must not create a locus
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["input_coding_loci"] == 1
    assert r["input_noncoding_features_excluded"] == {"lnc_RNA": 1}
    inp_structs = score_mod.cds_structures(inp)
    assert all(seq != "Chr1" or not (1000 <= start <= 1100)
               for seq, start, _end, _gene in inp_structs)


# -- fail loudly instead of silently scoring zero ----------------------------------


def test_two_genes_with_identical_span_are_not_collapsed(tmp_path):
    """Found against the real staged BRAKER3 file: 8 of its 26,635 genes have another
    gene with the exact same aggregate CDS span (adjacent/nested single-exon calls).
    A locus key of bare (seq, start, end) silently drops one of every such pair."""
    inp = tmp_path / "in.gff3"
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
        ("Chr1", "gene", 100, 200, "B", None, None),   # identical span, different gene
        ("Chr1", "mRNA", 100, 200, "B.1", "B", None),
        ("Chr1", "CDS", 100, 200, "B.1.cds", "B.1", None),
    ]))
    structs = score_mod.cds_structures(inp)
    assert len(structs) == 2


def test_exact_span_tie_is_broken_by_structure_not_by_which_came_first(tmp_path):
    """Found on the real TAIR10 self-test: 5 loci have a neighbour with the exact same
    aggregate exon span (coincidentally identical extents). Position alone ties; the
    candidate that actually shares a transcript structure with the predicted locus must
    win, or the second of the pair gets scored against the wrong reference model."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    # Q and R are two different genes with the exact same span on both sides.
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.1", "Q", None),
        ("Chr1", "CDS", 100, 200, "Q.1.cds", "Q.1", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.1", "R", None),
        ("Chr1", "CDS", 130, 200, "R.1.cds", "R.1", None),   # different structure than Q
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.i", "Q", None),
        ("Chr1", "CDS", 100, 200, "Q.i.cds", "Q.i", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.i", "R", None),
        ("Chr1", "CDS", 130, 200, "R.i.cds", "R.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.o", "Q", None),
        ("Chr1", "CDS", 100, 200, "Q.o.cds", "Q.o", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.o", "R", None),
        ("Chr1", "CDS", 130, 200, "R.o.cds", "R.o", None),
    ]))
    r = score_mod.score(inp, out, ref)
    # Both loci are, in truth, unchanged and correct — a position-only tie-break would
    # score one of them against the wrong reference model and report it as wrong.
    assert r["preserved_correct"] == 2
    assert r["repaired"] == 0
    assert r["damaged"] == 0
    assert r["still_wrong"] == 0


def test_score_raises_on_total_seqid_mismatch(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("ChrX", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("ChrX", "CDS", 100, 200, "A", "A.o")])
    with pytest.raises(RuntimeError, match="sequence-name-mismatch"):
        score_mod.score(inp, out, ref)
