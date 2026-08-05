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
    c = r["cds_level"]
    assert c["repaired"] == 1
    assert c["damaged"] == 1
    assert c["preserved_correct"] == 1
    assert c["still_wrong"] == 0


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
    c = r["cds_level"]
    assert c["preserved_correct"] == 1
    assert c["damaged"] == 0
    assert c["added_structures"] == 1
    assert c["added_matching_reference"] == 1


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
    c = r["cds_level"]
    assert c["still_wrong"] == 1          # classified on the first mRNA, not the second
    assert c["repaired"] == 0
    assert c["preserved_correct"] == 0
    assert c["added_structures"] == 0     # A.o2 duplicates the input's own A.i2 exactly
    assert c["added_matching_reference"] == 0


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
    c = r["cds_level"]
    assert c["preserved_correct"] == 1
    assert c["added_structures"] == 1
    assert c["added_matching_reference"] == 1


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


def test_utr_level_is_na_when_output_has_no_exon_rows(tmp_path):
    """C1: gating must look at the OUTPUT too — an input with real UTR paired with an
    output that carries no exon rows must not silently report a table of zeros."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 80, 220, "A", None, None),
        ("Chr1", "mRNA", 80, 220, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
        ("Chr1", "exon", 80, 220, "A.1.exon", "A.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 80, 220, "A", None, None),
        ("Chr1", "mRNA", 80, 220, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "exon", 80, 220, "A.i.exon", "A.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.o", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o.cds", "A.o", None),  # no exon rows at all
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["utr_level"]["status"] == "N/A"
    assert "output" in r["utr_level"]["reason"]


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
    assert r["cds_level"]["preserved_correct"] == 1   # CDS-level: right in both
    assert "status" not in r["utr_level"]              # a real transition table, not N/A
    assert r["utr_level"]["repaired"] == 1             # UTR-level: wrong extent in, right out


# -- non-coding features (lnc_RNA / pseudogene / bare "transcript") are excluded ----


def test_noncoding_features_excluded_from_loci_and_reported(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    # coding gene A plus a non-coding lncRNA locus — matching real EGAPx, which never
    # attaches a CDS row to an lncRNA transcript (verified against the staged file);
    # the lncRNA locus is excluded simply because no CDS row ever points to it. The
    # separate, adversarial case of a CDS actually parented to a declared non-coding
    # feature is covered by test_cds_row_parented_to_declared_noncoding_feature_raises.
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "gene", 1000, 1100, "L", None, 'gene_biotype=lncRNA'),
        ("Chr1", "lnc_RNA", 1000, 1100, "L.rna", "L", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["input_gene_rows"] == 2               # I6: A and L
    assert r["input_coding_loci"] == 1              # A only
    assert r["input_genes_without_cds"] == 1        # L
    assert r["input_noncoding_features_excluded"] == {"lnc_RNA": 1}
    inp_structs = score_mod.cds_structures(inp)
    assert all(seq != "Chr1" or not (1000 <= start <= 1100)
               for seq, start, _end, _gene in inp_structs)


def test_cds_row_parented_to_declared_noncoding_feature_raises(tmp_path):
    """M5: this must fail loudly, not silently drop the row — a CDS attached to a
    declared lncRNA/pseudogene is a real anomaly, not routine (unlike the same
    situation at the exon level, which is EGAPx's normal, expected shape)."""
    bad = tmp_path / "bad.gff3"
    bad.write_text(_gff3_lines([
        ("Chr1", "gene", 1000, 1100, "L", None, 'gene_biotype=lncRNA'),
        ("Chr1", "lnc_RNA", 1000, 1100, "L.rna", "L", None),
        ("Chr1", "CDS", 1000, 1050, "L.cds", "L.rna", None),
    ]))
    with pytest.raises(RuntimeError, match="non-coding feature"):
        score_mod.cds_structures(bad)


def test_coding_transcript_typed_as_bare_transcript_is_not_dropped(tmp_path):
    """M5: NONCODING_TYPES must not blanket-exclude the feature type string
    "transcript" — EGAPx's non-coding "transcript" rows are misc_RNA by gbkey, but a
    different GTF->GFF3 dialect can legitimately use "transcript" for a CODING
    transcript. That must still resolve to a normal locus."""
    p = tmp_path / "coding_transcript.gff3"
    p.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "transcript", 100, 200, "A.1", "A", "gbkey=mRNA"),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
    ]))
    structs = score_mod.cds_structures(p)
    assert len(structs) == 1
    (locus,) = structs.values()
    assert set(locus.values()) == {((100, 200),)}


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
    """Found on the real TAIR10 self-test: loci at the exon level (85 of them, in 40
    tied groups) share the exact same aggregate span as a neighbour — position alone
    ties. The candidate whose representative structure actually matches must win.

    The fixture gives Q and R the same outer CDS span (100-200) built from genuinely
    different internal structure (Q: two segments, R: one), and lists them in
    *opposite* order between the reference and the input/output, so a position-only
    tie-break (first-in-ref-file-order wins) would pick the wrong sibling for whichever
    locus is *not* first in the reference — the mistake this test would have missed if
    both files agreed on gene order, which is exactly how the previous version of this
    test failed to exercise its own tie-break (still passed with the bonus deleted).
    """
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    # Reference lists R before Q — opposite of input/output — so a stable sort on tied
    # starts would default to R unless the structural bonus corrects it.
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.1", "R", None),
        ("Chr1", "CDS", 100, 200, "R.1.cds", "R.1", None),          # one segment
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.1", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.1.cds1", "Q.1", None),         # two segments,
        ("Chr1", "CDS", 180, 200, "Q.1.cds2", "Q.1", None),         # same outer span
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.i", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.i.cds1", "Q.i", None),
        ("Chr1", "CDS", 180, 200, "Q.i.cds2", "Q.i", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.i", "R", None),
        ("Chr1", "CDS", 100, 200, "R.i.cds", "R.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.o", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.o.cds1", "Q.o", None),
        ("Chr1", "CDS", 180, 200, "Q.o.cds2", "Q.o", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.o", "R", None),
        ("Chr1", "CDS", 100, 200, "R.o.cds", "R.o", None),
    ]))
    r = score_mod.score(inp, out, ref)
    c = r["cds_level"]
    # Both loci are, in truth, unchanged and correct — a position-only tie-break would
    # score one of them against the wrong reference model and report it as wrong.
    assert c["preserved_correct"] == 2
    assert c["repaired"] == 0
    assert c["damaged"] == 0
    assert c["still_wrong"] == 0
    assert c["ties_broken_by_structure"] == 1  # I5: the tie is reported, not hidden


def test_score_raises_on_total_seqid_mismatch(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("ChrX", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("ChrX", "CDS", 100, 200, "A", "A.o")])
    with pytest.raises(RuntimeError, match="sequence-name-mismatch"):
        score_mod.score(inp, out, ref)


def test_score_raises_when_output_cannot_be_paired_to_any_input(tmp_path):
    """C1: this used to be silent — every locus booked as loci_without_output,
    loci_compared ended at 0, and repaired_pct/damaged_pct read 0.0 rather than
    signalling that nothing was actually compared."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    # structurally identical to a correct output, but on a different seqid and with no
    # GM= to recover it by identity
    _gff(out, [("1", "CDS", 100, 200, "A", "A.o")])
    with pytest.raises(RuntimeError):
        score_mod.score(inp, out, ref)


def test_loci_compared_zero_gives_none_percentages_not_zero(tmp_path):
    """M2/C1: `n = loci_compared or 1` silently turned "nothing was compared" into a
    reported 0.0% rather than None. Locus A has a reference match but no output
    partner at all (no GM, no overlap); locus B has no reference match but *does* pair
    to an unrelated output locus far away, which keeps `io_pairs` non-empty overall so
    the total-mismatch guard does not fire — isolating the loci_compared == 0 case."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i"),
               ("Chr1", "CDS", 5000, 5100, "B", "B.i")])
    _gff(out, [("Chr1", "CDS", 5000, 5100, "B", "B.o")])
    r = score_mod.score(inp, out, ref)
    c = r["cds_level"]
    assert c["loci_compared"] == 0
    assert c["repaired_pct"] is None
    assert c["damaged_pct"] is None
    assert c["added_precision_pct"] is None


# -- C2: "correct" against ANY reference transcript vs against the PRIMARY only -----


def test_cds_level_headline_matches_any_reference_transcript_not_just_the_first(tmp_path):
    """TAIR10 is coordinate-sorted, so the first-listed reference transcript is the
    leftmost isoform, not necessarily the primary. The input/output structure here
    matches the reference's SECOND-listed transcript exactly — cds_level (any match)
    must call this correct."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 150, "A.1", "A", None),   # listed first, shorter
        ("Chr1", "CDS", 100, 150, "A.1.cds", "A.1", None),
        ("Chr1", "mRNA", 100, 200, "A.2", "A", None),   # listed second, is the primary
        ("Chr1", "CDS", 100, 200, "A.2.cds", "A.2", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),   # matches A.2, not A.1
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.o", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o.cds", "A.o", None),
    ]))
    r = score_mod.score(inp, out, ref, primary_ids_path=None)
    assert r["cds_level"]["preserved_correct"] == 1
    assert r["cds_level"]["still_wrong"] == 0


def test_cds_level_primary_uses_curated_list_not_file_order(tmp_path):
    """Same fixture as above. Without a primary list, cds_level_primary falls back to
    file order (A.1, the leftmost — WRONG) and would misclassify this genuinely
    correct locus as still_wrong, exactly the ~8%-of-exact-hits bug the review
    measured. With primary_transcript_ids.txt correctly naming A.2, it agrees with
    cds_level."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    primary_ids = tmp_path / "primary_ids.txt"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 150, "A.1", "A", None),
        ("Chr1", "CDS", 100, 150, "A.1.cds", "A.1", None),
        ("Chr1", "mRNA", 100, 200, "A.2", "A", None),
        ("Chr1", "CDS", 100, 200, "A.2.cds", "A.2", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.o", "A", None),
        ("Chr1", "CDS", 100, 200, "A.o.cds", "A.o", None),
    ]))

    # without a primary list: falls back to file order (A.1) and gets it wrong
    r_no_primary = score_mod.score(inp, out, ref, primary_ids_path=None)
    assert r_no_primary["cds_level_primary"]["still_wrong"] == 1
    assert r_no_primary["cds_level_primary"]["preserved_correct"] == 0
    assert r_no_primary["cds_level_primary"]["primary_fallback_to_file_order"] == 1

    # with the curated list naming A.2: agrees with cds_level
    primary_ids.write_text("A.2\n")
    r = score_mod.score(inp, out, ref, primary_ids_path=primary_ids)
    assert r["cds_level_primary"]["preserved_correct"] == 1
    assert r["cds_level_primary"]["still_wrong"] == 0
    assert r["cds_level_primary"]["primary_fallback_to_file_order"] == 0
    assert r["cds_level"]["preserved_correct"] == 1  # headline agrees regardless


# -- I3: input<->output paired by GM=, input<->reference one-to-one ----------------


def test_output_gm_pairing_survives_positional_drift_toward_a_neighbor(tmp_path):
    """The output for gene A is damaged badly enough that it now sits positionally
    closer to unrelated neighbour gene B than to A. A position-only pairing could
    match it to B (or lose it as unmatched); GM=A recovers the correct pairing
    regardless of where the output ended up, and the damage is counted, not deleted."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.1", "A", None),
        ("Chr1", "CDS", 100, 200, "A.1.cds", "A.1", None),
        ("Chr1", "gene", 500, 600, "B", None, None),
        ("Chr1", "mRNA", 500, 600, "B.1", "B", None),
        ("Chr1", "CDS", 500, 600, "B.1.cds", "B.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
    ]))
    # damaged output for A, positioned overlapping B's territory, tagged GM=A
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 450, 600, "out1", None, "GM=A"),
        ("Chr1", "mRNA", 450, 600, "out1.t1", "out1", "GM=A"),
        ("Chr1", "CDS", 450, 600, "out1.t1.cds", "out1.t1", "GM=A"),
    ]))
    r = score_mod.score(inp, out, ref)
    c = r["cds_level"]
    assert r["pairing"]["gm_paired"] == 1
    assert c["loci_compared"] == 1        # A was found and scored, not dropped
    assert c["loci_without_output"] == 0
    assert c["damaged"] == 1              # structure no longer matches A


def test_split_predictions_are_counted_not_double_scored(tmp_path):
    """Two input loci both overlap the same single reference gene (GeMoMa splitting
    one TAIR10 gene into two of its own predictions). Only the better-overlapping one
    should be scored; the other is reported via split_predictions, not folded into
    loci_compared."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 300, "A", None, None),
        ("Chr1", "mRNA", 100, 300, "A.1", "A", None),
        ("Chr1", "CDS", 100, 300, "A.1.cds", "A.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 300, "A1", None, None),   # perfect match to ref A
        ("Chr1", "mRNA", 100, 300, "A1.i", "A1", None),
        ("Chr1", "CDS", 100, 300, "A1.i.cds", "A1.i", None),
        ("Chr1", "gene", 100, 250, "A2", None, None),   # partial overlap only
        ("Chr1", "mRNA", 100, 250, "A2.i", "A2", None),
        ("Chr1", "CDS", 100, 250, "A2.i.cds", "A2.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 300, "out1", None, "GM=A1"),
        ("Chr1", "mRNA", 100, 300, "out1.t1", "out1", "GM=A1"),
        ("Chr1", "CDS", 100, 300, "out1.t1.cds", "out1.t1", "GM=A1"),
        ("Chr1", "gene", 100, 250, "out2", None, "GM=A2"),
        ("Chr1", "mRNA", 100, 250, "out2.t1", "out2", "GM=A2"),
        ("Chr1", "CDS", 100, 250, "out2.t1.cds", "out2.t1", "GM=A2"),
    ]))
    r = score_mod.score(inp, out, ref)
    c = r["cds_level"]
    assert c["loci_compared"] == 1
    assert c["split_predictions"] == 1
    assert c["preserved_correct"] == 1   # the winning (perfect) match, A1


# -- I6: cross-tool denominators reconcile -----------------------------------------


def test_input_gene_rows_and_genes_without_cds_reconcile(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "A", None, None),
        ("Chr1", "mRNA", 100, 200, "A.i", "A", None),
        ("Chr1", "CDS", 100, 200, "A.i.cds", "A.i", None),
        ("Chr1", "gene", 1000, 1100, "L", None, "gene_biotype=lncRNA"),
        ("Chr1", "lnc_RNA", 1000, 1100, "L.rna", "L", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["input_gene_rows"] == 2
    assert r["input_coding_loci"] == 1
    assert r["input_genes_without_cds"] == 1


# -- M2: 1-based inclusive coordinate math ------------------------------------------


def test_one_base_overlap_is_not_scored_as_zero():
    a = ("Chr1", 1, 100, "a")
    b = ("Chr1", 100, 200, "b")   # shares exactly base 100
    assert score_mod._jaccard(a, b) > 0.0


# -- M4: attribute parsing does not match inside a longer key name -----------------


def test_attr_does_not_match_inside_a_longer_attribute_name():
    attrs = "GeneID=12345;ID=real1"
    assert score_mod._attr(attrs, "ID") == "real1"


# -- M7: provenance is recorded in the JSON -----------------------------------------


def test_provenance_is_recorded(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref, tool="gemoma")
    assert r["provenance"] == {
        "input": str(inp), "output": str(out), "reference": str(ref),
        "primary_ids": str(score_mod.PRIMARY_IDS), "tool": "gemoma",
    }


def test_provenance_records_the_primary_ids_path_actually_passed(tmp_path):
    """R4: with --primary-ids overridden, provenance and primary_source must both name
    the file actually used, not the module's default constant."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    custom_primary = tmp_path / "custom_primary.txt"
    custom_primary.write_text("A.1\n")
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref, primary_ids_path=custom_primary)
    assert r["provenance"]["primary_ids"] == str(custom_primary)
    assert r["cds_level_primary"]["primary_source"] == str(custom_primary)


# -- M6: the 2-level fallback is counted, not just silently used -------------------


def test_undeclared_parent_fallback_is_counted(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])   # flattened: triggers the fallback
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref)
    assert r["undeclared_parent_fallback"]["input"] == 1
    assert r["undeclared_parent_fallback"]["output"] == 1
    assert r["undeclared_parent_fallback"]["reference"] == 1


# ===================================================================================
# Fix round 2 (task-2-findings-round2.md)
# ===================================================================================


# -- R1: split_predictions/ties_broken_by_structure must reach utr_level too --------


def test_utr_level_split_predictions_is_reported(tmp_path):
    """split_predictions/ties_broken_by_structure were computed once per matched
    feature level but only ever attached to cds_level — even though on the real
    TAIR10 self-test they are 0 at CDS level and non-zero (40 splits, 5 ties) at
    exon level, exactly where they matter. Here A2's CDS does not overlap the
    reference gene's CDS at all (clean at cds_level), but its UTR-extended exon does
    overlap the same reference gene's exon (a split, only visible at utr_level)."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 400, 700, "R", None, None),
        ("Chr1", "mRNA", 400, 700, "R.1", "R", None),
        ("Chr1", "CDS", 500, 600, "R.1.cds", "R.1", None),
        ("Chr1", "exon", 400, 700, "R.1.exon", "R.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 400, 700, "A1", None, None),
        ("Chr1", "mRNA", 400, 700, "A1.i", "A1", None),
        ("Chr1", "CDS", 500, 600, "A1.i.cds", "A1.i", None),
        ("Chr1", "exon", 400, 700, "A1.i.exon", "A1.i", None),
        ("Chr1", "gene", 600, 750, "A2", None, None),
        ("Chr1", "mRNA", 600, 750, "A2.i", "A2", None),
        ("Chr1", "CDS", 650, 690, "A2.i.cds", "A2.i", None),      # outside R's CDS
        ("Chr1", "exon", 600, 750, "A2.i.exon", "A2.i", None),    # overlaps R's exon
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 400, 700, "outA1", None, "GM=A1"),
        ("Chr1", "mRNA", 400, 700, "outA1.t1", "outA1", "GM=A1"),
        ("Chr1", "CDS", 500, 600, "outA1.t1.cds", "outA1.t1", "GM=A1"),
        ("Chr1", "exon", 400, 700, "outA1.t1.exon", "outA1.t1", "GM=A1"),
        ("Chr1", "gene", 600, 750, "outA2", None, "GM=A2"),
        ("Chr1", "mRNA", 600, 750, "outA2.t1", "outA2", "GM=A2"),
        ("Chr1", "CDS", 650, 690, "outA2.t1.cds", "outA2.t1", "GM=A2"),
        ("Chr1", "exon", 600, 750, "outA2.t1.exon", "outA2.t1", "GM=A2"),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["cds_level"]["split_predictions"] == 0     # A2's CDS never overlaps R's
    assert r["utr_level"]["split_predictions"] == 1      # but its exon does
    assert r["utr_level_primary"]["split_predictions"] == 1


def test_utr_level_ties_broken_by_structure_is_reported(tmp_path):
    """Same wiring gap as above, for ties_broken_by_structure. Reuses the CDS-level
    tie-break fixture's reversed-gene-order trick, extended with exon rows that carry
    real UTR signal, so the identical-span tie exists at both levels simultaneously —
    letting this test also re-confirm cds_level's count as a side effect."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.1", "R", None),
        ("Chr1", "CDS", 100, 200, "R.1.cds", "R.1", None),
        ("Chr1", "exon", 90, 210, "R.1.exon", "R.1", None),
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.1", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.1.cds1", "Q.1", None),
        ("Chr1", "CDS", 180, 200, "Q.1.cds2", "Q.1", None),
        ("Chr1", "exon", 90, 150, "Q.1.exon1", "Q.1", None),
        ("Chr1", "exon", 180, 210, "Q.1.exon2", "Q.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.i", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.i.cds1", "Q.i", None),
        ("Chr1", "CDS", 180, 200, "Q.i.cds2", "Q.i", None),
        ("Chr1", "exon", 90, 150, "Q.i.exon1", "Q.i", None),
        ("Chr1", "exon", 180, 210, "Q.i.exon2", "Q.i", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.i", "R", None),
        ("Chr1", "CDS", 100, 200, "R.i.cds", "R.i", None),
        ("Chr1", "exon", 90, 210, "R.i.exon", "R.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "Q", None, None),
        ("Chr1", "mRNA", 100, 200, "Q.o", "Q", None),
        ("Chr1", "CDS", 100, 150, "Q.o.cds1", "Q.o", None),
        ("Chr1", "CDS", 180, 200, "Q.o.cds2", "Q.o", None),
        ("Chr1", "exon", 90, 150, "Q.o.exon1", "Q.o", None),
        ("Chr1", "exon", 180, 210, "Q.o.exon2", "Q.o", None),
        ("Chr1", "gene", 100, 200, "R", None, None),
        ("Chr1", "mRNA", 100, 200, "R.o", "R", None),
        ("Chr1", "CDS", 100, 200, "R.o.cds", "R.o", None),
        ("Chr1", "exon", 90, 210, "R.o.exon", "R.o", None),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["cds_level"]["ties_broken_by_structure"] == 1
    assert r["utr_level"]["ties_broken_by_structure"] == 1
    assert r["utr_level_primary"]["ties_broken_by_structure"] == 1


# -- R2: the overlap fallback must not steal an output already claimed via GM= ------


def test_overlap_fallback_excludes_outputs_already_claimed_via_gm(tmp_path):
    """Before the fix, the fallback searched the FULL output set, so a locus already
    cleanly claimed via GM= could also be claimed by a second, unrelated input locus
    falling back to position — silently double-scoring one output against two
    different inputs while output_merged read 0. Y has no GM= match; the only output
    locus that exists is tagged GM=X and physically sits where Y is, drifted there by
    "damage" — Y must not also claim it."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "RX", None, None),
        ("Chr1", "mRNA", 100, 200, "RX.1", "RX", None),
        ("Chr1", "CDS", 100, 200, "RX.1.cds", "RX.1", None),
        ("Chr1", "gene", 300, 400, "RY", None, None),
        ("Chr1", "mRNA", 300, 400, "RY.1", "RY", None),
        ("Chr1", "CDS", 300, 400, "RY.1.cds", "RY.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "X", None, None),
        ("Chr1", "mRNA", 100, 200, "X.i", "X", None),
        ("Chr1", "CDS", 100, 200, "X.i.cds", "X.i", None),
        ("Chr1", "gene", 300, 400, "Y", None, None),   # no GM= match for this one
        ("Chr1", "mRNA", 300, 400, "Y.i", "Y", None),
        ("Chr1", "CDS", 300, 400, "Y.i.cds", "Y.i", None),
    ]))
    out.write_text(_gff3_lines([
        ("Chr1", "gene", 300, 400, "O", None, "GM=X"),
        ("Chr1", "mRNA", 300, 400, "O.t1", "O", "GM=X"),
        ("Chr1", "CDS", 300, 400, "O.t1.cds", "O.t1", "GM=X"),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["pairing"]["gm_paired"] == 1
    assert r["pairing"]["overlap_fallback"] == 0      # Y must not also claim O
    assert r["cds_level"]["loci_compared"] == 1        # only X, not X and Y both against O


def test_overlap_fallback_collisions_are_folded_into_output_merged(tmp_path):
    """Two different input loci, neither with a usable GM= match, both fall back to
    the same unclaimed output locus by position — this must be reported as
    output_merged and excluded from scoring, not silently kept as two pairs to the
    same output. Z is cleanly GM-paired so the run does not trip the total-mismatch
    guard, isolating the X/Y collision."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "RX", None, None),
        ("Chr1", "mRNA", 100, 200, "RX.1", "RX", None),
        ("Chr1", "CDS", 100, 200, "RX.1.cds", "RX.1", None),
        ("Chr1", "gene", 210, 300, "RY", None, None),
        ("Chr1", "mRNA", 210, 300, "RY.1", "RY", None),
        ("Chr1", "CDS", 210, 300, "RY.1.cds", "RY.1", None),
        ("Chr1", "gene", 1000, 1100, "RZ", None, None),
        ("Chr1", "mRNA", 1000, 1100, "RZ.1", "RZ", None),
        ("Chr1", "CDS", 1000, 1100, "RZ.1.cds", "RZ.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 200, "X", None, None),
        ("Chr1", "mRNA", 100, 200, "X.i", "X", None),
        ("Chr1", "CDS", 100, 200, "X.i.cds", "X.i", None),
        ("Chr1", "gene", 210, 300, "Y", None, None),
        ("Chr1", "mRNA", 210, 300, "Y.i", "Y", None),
        ("Chr1", "CDS", 210, 300, "Y.i.cds", "Y.i", None),
        ("Chr1", "gene", 1000, 1100, "Z", None, None),
        ("Chr1", "mRNA", 1000, 1100, "Z.i", "Z", None),
        ("Chr1", "CDS", 1000, 1100, "Z.i.cds", "Z.i", None),
    ]))
    out.write_text(_gff3_lines([
        # X and Y both overlap this single, GM=-less output locus positionally
        ("Chr1", "gene", 100, 300, "O", None, None),
        ("Chr1", "mRNA", 100, 300, "O.t1", "O", None),
        ("Chr1", "CDS", 100, 300, "O.t1.cds", "O.t1", None),
        # Z is cleanly GM-paired, keeping the run from tripping the total-mismatch guard
        ("Chr1", "gene", 1000, 1100, "outZ", None, "GM=Z"),
        ("Chr1", "mRNA", 1000, 1100, "outZ.t1", "outZ", "GM=Z"),
        ("Chr1", "CDS", 1000, 1100, "outZ.t1.cds", "outZ.t1", "GM=Z"),
    ]))
    r = score_mod.score(inp, out, ref)
    assert r["pairing"]["gm_paired"] == 1        # Z only
    assert r["pairing"]["output_merged"] == 2    # X and Y both collided on O
    assert r["pairing"]["overlap_fallback"] == 0
    assert r["cds_level"]["loci_compared"] == 1  # Z only; X and Y excluded, not double-scored


# -- R4: primary_source/provenance report the path actually used -------------------


def test_provenance_records_the_primary_ids_path_actually_passed(tmp_path):
    """With --primary-ids overridden, provenance and primary_source must both name
    the file actually used, not the module's default constant."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    custom_primary = tmp_path / "custom_primary.txt"
    custom_primary.write_text("A.1\n")
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref, primary_ids_path=custom_primary)
    assert r["provenance"]["primary_ids"] == str(custom_primary)
    assert r["cds_level_primary"]["primary_source"] == str(custom_primary)


# -- R5: the tie-break override flag must reset when a later, outright winner appears --


def test_ties_broken_by_structure_resets_when_a_later_candidate_wins_outright():
    """A tie broken by the structural bonus must not keep counting once a later,
    strictly-better-scoring candidate wins the match outright — the final answer had
    nothing to do with the earlier tie. Before the fix, `overridden_by_bonus` was
    never reset, so the counter over-attributed C's outright win to B's earlier,
    irrelevant tie-break."""
    p_span = ("Chr1", 100, 300)
    pred = {p_span: {"p.i": ((999, 999),)}}
    a_span = ("Chr1", 100, 200)   # ties with b_span on Jaccard, wrong structure
    b_span = ("Chr1", 150, 250)   # ties with a_span on Jaccard, right structure -> wins the tie
    c_span = ("Chr1", 160, 300)   # strictly better Jaccard, wrong structure -> wins outright
    ref = {
        a_span: {"a.1": ((1, 1),)},
        b_span: {"b.1": ((999, 999),)},
        c_span: {"c.1": ((2, 2),)},
    }
    stats: dict = {}
    pairs = score_mod.match_by_overlap(pred, ref, stats)
    assert pairs[p_span] == c_span
    assert stats.get("ties_broken_by_structure", 0) == 0


# -- R6: a negative "genes without CDS" is reported as undefined, not negative ------


def test_genes_without_cds_is_none_not_negative_when_gene_rows_undercounts(tmp_path):
    """On this module's own flattened fixtures (no explicit "gene" rows at all),
    gene_rows - coding_loci goes negative; that is gene_rows undercounting, not a real
    "genes without CDS" figure, so it must report None rather than a negative number."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    out = tmp_path / "out.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i")])   # no "gene" row at all
    _gff(out, [("Chr1", "CDS", 100, 200, "A", "A.o")])
    r = score_mod.score(inp, out, ref)
    assert r["input_gene_rows"] == 0
    assert r["input_coding_loci"] == 1
    assert r["input_genes_without_cds"] is None


# -- R7: a row with no Parent at all gets its own message, not the noncoding one ----


def test_row_with_no_parent_at_all_raises_a_distinct_message(tmp_path):
    bad = tmp_path / "bad.gff3"
    bad.write_text("Chr1\tx\tCDS\t100\t200\t.\t+\t0\tID=orphan.cds\n")   # no Parent=
    with pytest.raises(RuntimeError, match="no Parent attribute"):
        score_mod.cds_structures(bad)


# -- R9: attribute parsing tolerates a space after the semicolon --------------------


def test_attr_tolerates_space_after_semicolon():
    assert score_mod._attr("a=1; ID=x", "ID") == "x"


# -- Task 3: baseline mode reports how far the input already is from the reference,
#    before any model touches it (no output file involved) --------------------------


def test_baseline_counts_correct_and_wrong_loci(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1"),
               ("Chr1", "CDS", 500, 600, "B", "B.1")])
    _gff(inp, [("Chr1", "CDS", 100, 200, "A", "A.i"),      # correct
               ("Chr1", "CDS", 500, 590, "B", "B.i")])     # wrong
    b = score_mod.baseline(inp, ref)
    assert b["loci_matched"] == 2
    assert b["loci_correct"] == 1
    assert b["loci_wrong"] == 1
    assert b["correct_pct"] == 50.0
    assert b["wrong_pct"] == 50.0


def test_baseline_resolves_split_predictions_like_score_does(tmp_path):
    """Same split scenario as test_split_predictions_are_counted_not_double_scored,
    but scored with no output at all. A locus one tool splits into two of its own
    predictions must still be counted once here, via the same _resolve_one_to_one
    machinery score() uses -- not once per splitting prediction."""
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    ref.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 300, "A", None, None),
        ("Chr1", "mRNA", 100, 300, "A.1", "A", None),
        ("Chr1", "CDS", 100, 300, "A.1.cds", "A.1", None),
    ]))
    inp.write_text(_gff3_lines([
        ("Chr1", "gene", 100, 300, "A1", None, None),   # perfect match to ref A
        ("Chr1", "mRNA", 100, 300, "A1.i", "A1", None),
        ("Chr1", "CDS", 100, 300, "A1.i.cds", "A1.i", None),
        ("Chr1", "gene", 100, 250, "A2", None, None),   # partial overlap only
        ("Chr1", "mRNA", 100, 250, "A2.i", "A2", None),
        ("Chr1", "CDS", 100, 250, "A2.i.cds", "A2.i", None),
    ]))
    b = score_mod.baseline(inp, ref)
    assert b["loci_matched"] == 1
    assert b["split_predictions"] == 1
    assert b["loci_correct"] == 1   # the winning (perfect) match, A1
    assert b["loci_wrong"] == 0


def test_baseline_raises_on_empty_input_instead_of_reporting_zeros(tmp_path):
    ref = tmp_path / "ref.gff3"
    empty = tmp_path / "empty.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    empty.write_text("")
    with pytest.raises(ValueError, match="no CDS-bearing transcripts"):
        score_mod.baseline(empty, ref)


def test_baseline_raises_on_total_seqid_mismatch(tmp_path):
    ref = tmp_path / "ref.gff3"
    inp = tmp_path / "in.gff3"
    _gff(ref, [("Chr1", "CDS", 100, 200, "A", "A.1")])
    _gff(inp, [("ChrX", "CDS", 100, 200, "A", "A.i")])
    with pytest.raises(RuntimeError, match="sequence-name-mismatch"):
        score_mod.baseline(inp, ref)
