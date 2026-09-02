"""GSF v3: every complete gene inside a window (protocol A26)."""
import random
import pytest
from conftest import gff


def _genes(gsf, text):
    return list(gsf.parse_gff3(text.splitlines(), species_code="Ath"))


def test_window_label_concatenates_genes_in_coordinate_order(gsf):
    text = (gff("Chr1", "gB", "-", {"t": [("CDS", 5001, 5300, 0)]}) + gff("Chr1", "gA", "+", {"t": [("CDS", 1001, 1300, 0), ("CDS", 1501, 1800, 0)]})
            + gff("Chr1", "gEdge", "+", {"t": [("CDS", 9901, 10300, 0)]}))
    genes = _genes(gsf, text)
    inside, partial = gsf.genes_in_window(genes, 0, 10000)
    assert [g.gene_id for g in inside] == ["gA", "gB"] and partial == 1
    label = gsf.window_to_gsf_v3(inside, 0)
    blocks = gsf.split_v3(label)
    assert len(blocks) == 2 and blocks[0].startswith("1000|CDS1|1300|+|A") and blocks[1].startswith("5000|CDS1|5300|-|A")
    assert gsf.canonicalize_v3(gsf.GENE_SEP.join(reversed(blocks))) == label
    gsf.check_caps_v3(label, window_len=10000)
    assert gsf.window_to_gsf_v3([], 0) == gsf.EMPTY_LABEL and gsf.count_tokens_v3(gsf.EMPTY_LABEL) == 3


def test_v3_token_count_and_caps(gsf):
    a = "0|CDS1|300|+|A>CDS1"
    b = "1000|CDS1|1300|-|A>CDS1"
    label = a + gsf.GENE_SEP + b
    assert gsf.count_tokens_v3(label) == gsf.count_tokens_v2(a) + gsf.count_tokens_v2(b) - 2 + 1
    with pytest.raises(gsf.CapError):
        gsf.check_caps_v3(b + gsf.GENE_SEP + a)            # overlapping/unsorted genes
    with pytest.raises(gsf.CapError):
        gsf.check_caps_v3(gsf.GENE_SEP.join([f"{i*10}|CDS1|{i*10+3}|+|A>CDS1" for i in range(97)]))   # > 96 genes


def test_v3_rc_is_an_involution_and_reorders_genes(gsf):
    label = "0|CDS1|300|+|A>CDS1" + gsf.GENE_SEP + "1000|CDS1|1300|-|A>CDS1"
    L = 6144
    r = gsf.reverse_complement_v3(label, L)
    assert gsf.split_v3(r)[0].startswith(f"{L-1300}|CDS1|{L-1000}|+|A") and gsf.reverse_complement_v3(r, L) == label
    assert gsf.reverse_complement_v3(gsf.EMPTY_LABEL, L) == gsf.EMPTY_LABEL


def test_tile_windows_cover_contig(gsf):
    tiles = gsf.tile_windows(100000, 30720, offset=5000)
    assert tiles[0] == (5000, 35720) and tiles[-1] == (100000 - 30720, 100000) and all(b - a == 30720 for a, b in tiles)
    assert gsf.tile_windows(20000, 30720) == [(0, 30720)]


def test_block_splits_and_leak_mask(gsf):
    import random
    rng = random.Random(3)
    blocks = gsf.block_splits(5 * gsf.BLOCK_LEN + 10, rng, forced_test=[(gsf.BLOCK_LEN + 5, gsf.BLOCK_LEN + 50)])
    assert len(blocks) == 6 and blocks[1][2] == "test" and all(sp in ("train", "valid", "test") for _, _, sp in blocks)
    assert gsf.tile_split(blocks, gsf.BLOCK_LEN - 10, gsf.BLOCK_LEN + 100) == "test"        # straddles the forced block
    g = gsf.Gene("x", "c", "+", 1000, 1500, {})
    seq = "A" * 3000
    masked = gsf.leak_mask(seq, 0, [g])
    assert masked[900:1600] == "N" * 700 and masked[:900] == "A" * 900 and masked[1600:] == "A" * 1400
