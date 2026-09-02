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
    # CDS lengths 100 and 101: forward phases A (0) then C (2). After RC the transcript reads the
    # 101-bp piece first: phases A then (3 - 101 mod 3) mod 3 = 1 -> B.
    s, L = _gene_gsf(gsf, {"t1": [("CDS", 100, 199, 0), ("CDS", 300, 400, 2)]})
    r = gsf.reverse_complement(s, L)
    phases = [f.split("|")[4] for f in r.split(">")[0].split(";")]
    assert phases == ["A", "B"]


def test_rc_keeps_utrs_in_transcription_order_and_is_canonical(gsf):
    s, L = _gene_gsf(gsf, {"t1": [("five_prime_UTR", 100, 150, "."), ("CDS", 151, 250, 0), ("three_prime_UTR", 251, 300, ".")]})
    r = gsf.reverse_complement(s, L)
    assert r.split(">")[1].split("|")[0].startswith("five_prime_UTR")
    assert gsf.canonicalize(r) == r
