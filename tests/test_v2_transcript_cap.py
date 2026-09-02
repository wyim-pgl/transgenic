"""§4: more than 15 transcripts is rejected at build time (the tokenizer would drop <iso> separators)."""
import pytest
from conftest import gff


def _gene(gsf, n_tx):
    tx = {f"t{i}": [("CDS", 100, 200, 0)] + [("CDS", 300 + 10 * j, 305 + 10 * j, 0) for j in range(i)] for i in range(n_tx)}
    return next(gsf.parse_gff3(gff("Chr1", "g", "+", tx).splitlines()))


def test_fifteen_transcripts_accepted(gsf):
    s = gsf.gene_to_gsf(_gene(gsf, 15), 0)
    assert s.split(">")[1].count(";") == 14
    gsf.check_caps(s)


def test_sixteen_transcripts_rejected(gsf):
    s = gsf.gene_to_gsf(_gene(gsf, 16), 0)
    with pytest.raises(gsf.CapError) as e:
        gsf.check_caps(s)
    assert "transcript" in str(e.value).lower()
