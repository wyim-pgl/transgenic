"""§5/§7: RC rows inherit the forward split; --rc none|all|isoform-only."""
import pytest
from conftest import gff


def _genes(gsf):
    t = gff("Chr1", "single", "+", {"a": [("CDS", 100, 200, 0)]}) + gff("Chr1", "multi", "-", {"a": [("CDS", 1100, 1200, 0), ("CDS", 1300, 1400, 1)], "b": [("CDS", 1300, 1400, 0)]})
    return list(gsf.parse_gff3(t.splitlines()))


@pytest.mark.parametrize("mode,expected_rc", [("none", 0), ("all", 2), ("isoform-only", 1)])
def test_rc_modes(gsf, mode, expected_rc):
    rows = gsf.build_rows(_genes(gsf), species_id="Athaliana", rc=mode, split_lookup={"single": "train", "multi": "valid"})
    assert sum(r["is_rc"] for r in rows) == expected_rc


def test_rc_rows_inherit_split_and_gene_id(gsf):
    rows = gsf.build_rows(_genes(gsf), species_id="Athaliana", rc="all", split_lookup={"single": "train", "multi": "valid"})
    for r in rows:
        if r["is_rc"]:
            fwd = next(x for x in rows if x["gene_id"] == r["gene_id"] and not x["is_rc"])
            assert r["split"] == fwd["split"] and r["strand"] != fwd["strand"]
            assert r["gsf"] != fwd["gsf"]


def test_invalid_rc_mode_rejected(gsf):
    with pytest.raises(ValueError):
        gsf.build_rows(_genes(gsf), species_id="Athaliana", rc="iso", split_lookup={"single": "train", "multi": "valid"})
