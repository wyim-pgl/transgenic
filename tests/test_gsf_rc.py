"""§5 reverse complement as a pure transformation."""
from conftest import gff


def _gene_gsf(gsf, tx, strand="+"):
    text = gff("Chr1", "g", strand, tx)
    gene = next(gsf.parse_gff3(text.splitlines()))
    ws, we = gsf.pad_window(gene.start0, gene.end0)
    return gsf.gene_to_gsf(gene, ws), we - ws


def test_rc_is_an_involution(gsf):
    s, L = _gene_gsf(gsf, {"t1": [("five_prime_UTR", 100, 150, "."), ("CDS", 151, 250, 0), ("CDS", 300, 400, 2), ("three_prime_UTR", 401, 450, ".")],
                            "t2": [("CDS", 151, 250, 0), ("CDS", 300, 400, 2)]})
    assert gsf.reverse_complement(gsf.reverse_complement(s, L), L) == s


def test_rc_mirrors_coordinates_and_flips_strand(gsf):
    s, L = _gene_gsf(gsf, {"t1": [("CDS", 100, 200, 0)]})
    a, _, b, strand, _ = s.split(">")[0].split("|")
    r = gsf.reverse_complement(s, L)
    ra, _, rb, rstrand, _ = r.split(">")[0].split("|")
    assert (int(ra), int(rb)) == (L - int(b), L - int(a)) and rstrand == "-" and strand == "+"


def test_rc_recomputes_phases_five_to_three_prime(gsf):
    # CDS lengths 100 and 101: forward phases A (0) then C (2). RC flips the strand but not the
    # transcript: the 100-bp piece is still read first (A) and the 101-bp piece keeps phase
    # (3 - 100 mod 3) mod 3 = 2 (C). In the coordinate-sorted feature list the mirrored 101-bp piece
    # now comes first, so the list reads C then A; a wrong phase (e.g. 1 -> B) must not appear.
    s, L = _gene_gsf(gsf, {"t1": [("CDS", 100, 199, 0), ("CDS", 300, 400, 2)]})
    r = gsf.reverse_complement(s, L)
    phases = [f.split("|")[4] for f in r.split(">")[0].split(";")]
    assert phases == ["C", "A"]
    # and a deliberately wrong forward phase is corrected by the recomputation
    s2, L2 = _gene_gsf(gsf, {"t1": [("CDS", 100, 199, 0), ("CDS", 300, 400, 1)]})
    r2 = gsf.reverse_complement(s2, L2)
    assert [f.split("|")[4] for f in r2.split(">")[0].split(";")] == ["C", "A"]


def test_rc_keeps_utrs_in_transcription_order_and_is_canonical(gsf):
    s, L = _gene_gsf(gsf, {"t1": [("five_prime_UTR", 100, 150, "."), ("CDS", 151, 250, 0), ("three_prime_UTR", 251, 300, ".")]})
    r = gsf.reverse_complement(s, L)
    assert r.split(">")[1].split("|")[0].startswith("five_prime_UTR")
    assert gsf.canonicalize(r) == r


def test_rc_involution_for_five_prime_partial_model(gsf):
    # first CDS starts in phase 2 (5'-partial model): RC must keep that phase and stay an involution
    s, L = _gene_gsf(gsf, {"t1": [("CDS", 100, 200, 2), ("CDS", 300, 400, 0)]})   # 101 - 2 = 99 coding bases -> next phase 0
    assert s.split(">")[0].split(";")[0].split("|")[4] == "C"
    r = gsf.reverse_complement(s, L)
    assert gsf.reverse_complement(r, L) == s
    # after RC the (mirrored) first CDS still carries phase 2 and the second follows from 101 - 2 coding bases
    phases = {f.split("|")[1]: f.split("|")[4] for f in r.split(">")[0].split(";")}
    assert sorted(phases.values()) == ["A", "C"]
