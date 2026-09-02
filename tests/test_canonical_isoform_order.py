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


def test_gene_key_rule(gsf):
    assert gsf.species_code("Athaliana") == "Ath" and gsf.species_code("Gmax") == "Gma"
    assert gsf.gene_key("Ath", "AT1G01010", "", 7) == "AT1G01010"            # ID first
    assert gsf.gene_key("Ath", "", "AT1G01010", 7) == "AT1G01010"            # Name when ID missing
    assert gsf.gene_key("Ath", "AT1G01010.TAIR10", "AT1G01010", 7) == "Ath000007"   # >10 chars -> generated (Name is not consulted)
    assert gsf.gene_key("Zma", "Zm00001d027230", "", 3) == "Zma000003"       # 14 chars
    assert gsf.gene_key("Osa", "a.b.c", "", 4) == "Osa000004"                 # more than one dot
    assert gsf.gene_key("Ppa", "", "", 12) == "Ppa000012"


def test_intron_count_ignores_utr_cds_boundaries(gsf):
    # 5'UTR|CDS contiguous, one real intron, CDS|3'UTR contiguous -> intron_count 1, not 3
    text = gff("Chr1", "g", "+", {"t1": [("five_prime_UTR", 100, 150, "."), ("CDS", 151, 250, 0), ("CDS", 300, 400, 2), ("three_prime_UTR", 401, 450, ".")],
                                  "t2": [("CDS", 151, 250, 0), ("CDS", 300, 400, 2), ("CDS", 500, 520, 0), ("CDS", 600, 620, 0)]})
    s = gsf.gene_to_gsf(next(gsf.parse_gff3(text.splitlines())), 0)
    txs = s.split(">")[1].split(";")
    assert txs[0].count("|") == 3 and txs[1].count("|") == 3  # t1 (1 intron, 4 features) sorts before t2 (3 introns)
    feats = {f.split("|")[1]: f for f in s.split(">")[0].split(";")}
    assert txs[0].split("|")[0].startswith("five_prime_UTR")
    assert gsf.exons_of([("five_prime_UTR", 99, 150, "."), ("CDS", 150, 250, "0"), ("CDS", 299, 400, "2"), ("three_prime_UTR", 400, 450, ".")]) == [(99, 250), (299, 450)]


def test_no_cds_records_are_rejected(gsf):
    import pytest
    with pytest.raises(gsf.CapError):
        gsf.check_caps("0|five_prime_UTR1|50|+|.>five_prime_UTR1")
    with pytest.raises(gsf.CapError):
        gsf.check_caps("0|five_prime_UTR1|50|+|.;60|CDS1|120|+|A>five_prime_UTR1|CDS1;five_prime_UTR1")
