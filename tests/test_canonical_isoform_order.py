"""§3 canonical ordering gsf-order-v1."""
import itertools
from conftest import gff


def test_transcript_permutation_invariance_and_idempotence(gsf):
    base = {"a": [("CDS", 100, 200, 0), ("CDS", 300, 400, 1)], "b": [("CDS", 100, 200, 0)], "c": [("CDS", 100, 200, 0), ("CDS", 300, 400, 1), ("CDS", 500, 600, 2)]}
    outs = set()
    for perm in itertools.permutations(base):
        text = gff("Chr1", "g", "+", {k: base[k] for k in perm})
        gene = next(gsf.parse_gff3(text.splitlines()))
        s = gsf.gene_to_gsf(gene, 0)
        outs.add(s)
        assert gsf.canonicalize(s) == s
    assert len(outs) == 1


def test_transcripts_sorted_by_intron_count_then_chain(gsf):
    text = gff("Chr1", "g", "+", {"x": [("CDS", 100, 200, 0), ("CDS", 300, 400, 1), ("CDS", 500, 600, 2)], "y": [("CDS", 100, 200, 0)], "z": [("CDS", 100, 200, 0), ("CDS", 300, 400, 1)]})
    s = gsf.gene_to_gsf(next(gsf.parse_gff3(text.splitlines())), 0)
    txs = s.split(">")[1].split(";")
    assert [t.count("|") for t in txs] == [0, 1, 2]


def test_identical_signatures_merge_and_ids_do_not_matter(gsf):
    t1 = gff("Chr1", "g", "+", {"AT1.1": [("CDS", 100, 200, 0)], "AT1.2": [("CDS", 100, 200, 0)]})
    t2 = gff("Chr1", "g", "+", {"zzz": [("CDS", 100, 200, 0)]})
    s1 = gsf.gene_to_gsf(next(gsf.parse_gff3(t1.splitlines())), 0)
    s2 = gsf.gene_to_gsf(next(gsf.parse_gff3(t2.splitlines())), 0)
    assert s1 == s2 and s1.count(";") == 0


def test_feature_numbering_follows_first_use_after_ordering(gsf):
    text = gff("Chr1", "g", "+", {"long": [("CDS", 100, 200, 0), ("CDS", 300, 400, 1)], "short": [("CDS", 300, 400, 0)]})
    s = gsf.gene_to_gsf(next(gsf.parse_gff3(text.splitlines())), 0)
    feats, txs = s.split(">")
    # mono-exonic transcript comes first and uses CDS1, which must therefore be the 300-400 feature
    # (the feature list itself stays coordinate-sorted, §3)
    assert txs.split(";")[0] == "CDS1"
    by_name = {f.split("|")[1]: f for f in feats.split(";")}
    assert by_name["CDS1"].split("|")[0] == "299" and feats.split(";")[0].split("|")[1] == "CDS2"


def test_rc_canonical_form_is_stable_under_double_rc(gsf):
    text = gff("Chr1", "g", "-", {"a": [("CDS", 100, 200, 1), ("CDS", 300, 400, 0)], "b": [("CDS", 300, 400, 0)]})
    gene = next(gsf.parse_gff3(text.splitlines()))
    ws, we = gsf.pad_window(gene.start0, gene.end0)
    s = gsf.gene_to_gsf(gene, ws)
    L = we - ws
    assert gsf.canonicalize(gsf.reverse_complement(gsf.reverse_complement(s, L), L)) == gsf.canonicalize(s)
