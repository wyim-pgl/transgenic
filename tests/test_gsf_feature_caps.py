"""§4 caps: reject, never truncate."""
import pytest
from conftest import gff


def _many(typ, n, start=100, phase=0):
    return [(typ, start + 10 * i, start + 10 * i + 5, phase) for i in range(n)]


@pytest.mark.parametrize("typ,n,ok", [("CDS", 150, True), ("CDS", 151, False), ("five_prime_UTR", 50, True), ("five_prime_UTR", 51, False), ("three_prime_UTR", 51, False)])
def test_feature_caps(gsf, typ, n, ok):
    feats = _many(typ, n, phase=0 if typ == "CDS" else ".")
    if typ != "CDS":
        feats = feats + [("CDS", 5000, 5100, 0)] if typ == "five_prime_UTR" else [("CDS", 10, 20, 0)] + feats
    text = gff("Chr1", "g", "+", {"t": feats})
    gene = next(gsf.parse_gff3(text.splitlines()))
    s = gsf.gene_to_gsf(gene, 0)
    if ok:
        gsf.check_caps(s)
    else:
        with pytest.raises(gsf.CapError) as e:
            gsf.check_caps(s)
        assert typ.lower().replace("_prime_", "") in str(e.value).lower().replace("_prime_", "")


def test_window_cap(gsf):
    text = gff("Chr1", "g", "+", {"t": [("CDS", 1, 49153, 0)]})
    gene = next(gsf.parse_gff3(text.splitlines()))
    with pytest.raises(gsf.CapError):
        gsf.check_caps(gsf.gene_to_gsf(gene, 0), window_len=gsf.pad_window(gene.start0, gene.end0)[1])


def test_rejected_records_are_reported_not_truncated(gsf):
    text = gff("Chr1", "g", "+", {"t": _many("CDS", 151)})
    gene = next(gsf.parse_gff3(text.splitlines()))
    rows, rejected = gsf.build_rows([gene], species_id="Athaliana", rc="none", split_lookup={"g": "train"}, return_rejected=True)
    assert rows == [] and rejected[0]["gene_id"] == "g" and "CDS" in rejected[0]["reason"]
