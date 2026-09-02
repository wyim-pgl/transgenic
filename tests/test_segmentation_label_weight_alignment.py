"""§10 C2 label contract: 14 classes, weight channel, 0-based conversion, protocol A18.4 weights."""
import math
import pytest

CLASSES = ["protein_coding_gene", "lncRNA", "exon", "intron", "splice_donor", "splice_acceptor", "5UTR", "3UTR",
           "CTCF-bound", "polyA_signal", "enhancer_Tissue_specific", "enhancer_Tissue_invariant", "promoter_Tissue_specific", "promoter_Tissue_invariant"]


def test_class_list_matches_preprocess(gsf):
    assert list(gsf.SEG_CLASSES) == CLASSES


def test_labels_and_weights_have_same_shape_and_0based_positions(gsf):
    # GFF CDS 10..12 (1-based inclusive) inside a window starting at genomic 0-based 0 -> rows 9,10,11
    labels, weights = gsf.segmentation_labels([("CDS", 10, 12, "+")], L=20, window_start=0)
    assert len(labels) == 20 and len(weights) == 20 and all(len(r) == 14 for r in labels) and all(len(r) == 14 for r in weights)
    ex = CLASSES.index("exon")
    assert [labels[i][ex] for i in range(20)] == [1.0 if i in (9, 10, 11) else 0.0 for i in range(20)]
    assert all(labels[i][ex] == 0.0 for i in (8, 12))


def test_reference_labels_have_unit_weight_and_unlabelled_cells_zero(gsf):
    labels, weights = gsf.segmentation_labels([("CDS", 10, 12, "+")], L=20, window_start=0)
    ex = CLASSES.index("exon")
    assert weights[9][ex] == 1.0 and weights[0][ex] == 0.0


@pytest.mark.parametrize("source,n,geno,ri,expected", [
    ("protein", 1, "reference", False, 1 + 1.0 * math.log1p(1)),
    ("pacbio", 5, "reference", False, 1 + 1.0 * math.log1p(5)),
    ("ont", 5, "reference", False, 1 + 0.8 * math.log1p(5)),
    ("est", 5, "non_reference", False, 1 + 0.6 * 0.5 * math.log1p(5)),
    ("pacbio", 10000, "reference", False, 4.0),
    ("ont", 5, "reference", True, 0.25),
])
def test_evidence_weight_formula_a18_4(gsf, source, n, geno, ri, expected):
    assert gsf.evidence_weight(source, n, genotype=geno, retained_intron=ri) == pytest.approx(expected)


def test_evidence_adds_positives_only_and_never_removes_reference_labels(gsf):
    labels, weights = gsf.segmentation_labels([("CDS", 10, 12, "+")], L=20, window_start=0)
    intron = CLASSES.index("intron")
    ex = CLASSES.index("exon")
    gsf.add_evidence(labels, weights, cells=[(14, intron, 2.0), (9, ex, 0.5)])
    assert labels[14][intron] == 1.0 and weights[14][intron] == 2.0
    assert labels[9][ex] == 1.0 and weights[9][ex] >= 1.0  # reference cell keeps its label and never drops below 1


def test_junction_evidence_labels_only_intron_and_boundary_classes(gsf):
    labels, weights = gsf.segmentation_labels([], L=50, window_start=0)
    gsf.add_junction_evidence(labels, weights, donor0=10, acceptor0=30, weight=1.5)
    d, a, i = CLASSES.index("splice_donor"), CLASSES.index("splice_acceptor"), CLASSES.index("intron")
    assert labels[10][d] == 1.0 and labels[30][a] == 1.0 and all(labels[p][i] == 1.0 for p in range(10, 31))
    touched = {c for p in range(50) for c in range(14) if labels[p][c] == 1.0}
    assert touched <= {d, a, i}


@pytest.mark.parametrize("source,expected_sw", [("est", 1.0), ("pacbio", 0.9), ("ont", 0.7), ("protein", 0.5)])
def test_junction_family_weights_est_leads_a20(gsf, source, expected_sw):
    n = 5
    assert gsf.evidence_weight(source, n, family="junction") == pytest.approx(1 + expected_sw * math.log1p(n))
    assert gsf.junction_weight(source, n) == gsf.evidence_weight(source, n, family="junction")


def test_junction_family_est_outranks_every_other_source(gsf):
    n = 3
    est = gsf.junction_weight("est", n)
    assert est > gsf.junction_weight("pacbio", n) > gsf.junction_weight("ont", n) > gsf.junction_weight("protein", n)
    # exon family keeps the A18.4 order (protein/pacbio > ont > est)
    assert gsf.evidence_weight("est", n) < gsf.evidence_weight("ont", n) < gsf.evidence_weight("pacbio", n)
