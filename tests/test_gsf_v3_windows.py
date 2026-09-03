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


# --- protocol A33: overlapping gene blocks, masking closure, stitching, edge invariant ---

def _g(gsf, gid, start1, end1, strand="+", cds=None):
    """Gene with one transcript of the given CDS segments (1-based inclusive, as in a GFF)."""
    segs = cds or [(start1, end1)]
    return gsf.Gene(gid, "Chr1", strand, start1 - 1, end1,
                    {f"{gid}.1": [gsf.Feature("CDS", s, e, "0") for s, e in segs]})


def test_a33_overlapping_blocks_accepted_and_ordered(gsf):
    """A33.1: blocks may overlap; the key (start, end, canonical block) must not decrease."""
    genes = [_g(gsf, "a", 1001, 3000), _g(gsf, "b", 2001, 2600, "-"), _g(gsf, "c", 2001, 5000)]
    label = gsf.window_to_gsf_v3(genes, 0)
    gsf.check_caps_v3(label, window_len=30720)                       # overlap no longer rejected
    blocks = gsf.split_v3(label)
    assert len(blocks) == 3
    keys = []
    for b in blocks:
        feats, _, _ = gsf._parse(b)
        keys.append((min(f[1] for f in feats.values()), max(f[2] for f in feats.values())))
    assert keys == sorted(keys) and keys[1][0] == keys[2][0] and keys[1][1] < keys[2][1]   # equal start -> shorter end first
    assert gsf.canonicalize_v3(label) == label                                            # already canonical
    # a label whose blocks are in the wrong order is rejected but canonicalises to the accepted one
    swapped = gsf.GENE_SEP.join([blocks[0], blocks[2], blocks[1]])
    with pytest.raises(gsf.CapError):
        gsf.check_caps_v3(swapped, window_len=30720)
    assert gsf.canonicalize_v3(swapped) == label


def test_a33_nested_and_antisense_round_trip_and_rc(gsf):
    nested = [_g(gsf, "outer", 1001, 9000, "+", [(1001, 1999), (8001, 9000)]), _g(gsf, "inner", 3001, 4000, "-")]  # 999 nt keeps phase 0
    label = gsf.window_to_gsf_v3(nested, 0)
    gsf.check_caps_v3(label, window_len=30720)
    assert label.count(gsf.GENE_SEP) == 1
    rc = gsf.reverse_complement_v3(label, 30720)
    gsf.check_caps_v3(rc, window_len=30720)
    assert gsf.reverse_complement_v3(rc, 30720) == label                                  # involution through canonical form
    assert gsf.canonicalize_v3(gsf.GENE_SEP.join(reversed(gsf.split_v3(label)))) == label  # permutation invariance


def test_a33_overlap_components_and_duplicate_collapse(gsf):
    a, b, c = _g(gsf, "a", 1001, 3000), _g(gsf, "b", 2500, 4000), _g(gsf, "c", 9001, 9500)
    comps = gsf.overlap_components([a, b, c])
    assert [sorted(g.gene_id for g in comp) for comp in comps] == [["a", "b"], ["c"]]
    dup1 = _g(gsf, "d1", 1001, 3000, "+", [(1200, 2800)])
    dup2 = _g(gsf, "d2", 900, 3100, "+", [(1200, 2800)])                                   # same CDS, longer span
    other = _g(gsf, "e", 1001, 3000, "-", [(1200, 2800)])                                  # same CDS, other strand -> kept
    kept, dropped = gsf.collapse_duplicate_genes([dup1, dup2, other])
    assert dropped == 1 and {g.gene_id for g in kept} == {"d2", "e"}


def test_a33_flank_jitter_and_masked_fraction(gsf):
    g = _g(gsf, "x", 1001, 1500)
    seq = "A" * 3000
    masked = gsf.leak_mask(seq, 0, [g], flanks=[50])
    assert masked.count("N") == 500 + 2 * 50
    assert gsf.leak_mask(seq, 0, [g], flanks=[150]).count("N") == 500 + 2 * 150
    assert gsf.FLANK_RANGE == (50, 150) and gsf.MASK_FRACTION_MAX == 0.60
    assert abs(gsf.masked_fraction("A" * 60 + "N" * 40) - 0.40) < 1e-9


def test_a33_same_locus_stitching_rule(gsf):
    multi_a = [(1000, 1200), (2000, 2300)]
    multi_b = [(1000, 1200), (2000, 2280)]                                                 # shares the intron 1200-2000
    assert gsf.same_locus(multi_a, "+", multi_b, "+") is True
    assert gsf.same_locus(multi_a, "+", multi_b, "-") is False                             # opposite strand: distinct
    assert gsf.same_locus(multi_a, "+", multi_b, "+", same_tile=True) is False             # co-emitted: distinct
    no_shared = [(1000, 1200), (2100, 2400)]                                               # different intron
    assert gsf.same_locus(multi_a, "+", no_shared, "+") is False
    small = [(1000, 1100), (2000, 2300)]                                                   # reciprocal overlap < 0.90
    assert gsf.same_locus(multi_a, "+", small, "+") is False
    mono_a, mono_b = [(1000, 3000)], [(1100, 3050)]
    assert gsf.same_locus(mono_a, "+", mono_b, "+") is True
    assert gsf.same_locus(mono_a, "+", [(4000, 6000)], "+") is False                       # no overlap
    assert gsf.STITCH_RECIPROCAL == 0.90 and gsf.STITCH_MONO_END == 1000


def test_a33_edge_margin_invariant_over_three_offsets(gsf):
    """A33.4: the three offsets guarantee every gene of length <= tier - 2*margin is >= 1,000 nt inside some tile."""
    for tier in gsf.WINDOW_TIERS:
        assert gsf.tier_offsets(tier) == [0, round(tier / 3), round(2 * tier / 3)]
        bound = 2 * tier // 3 - 2 * gsf.EDGE_MARGIN            # 18,480 / 38,960 / 84,016 nt
        step = max(1, tier // 211)
        for start in range(gsf.EDGE_MARGIN, 3 * tier, step):
            for length in (500, 5000, bound):
                assert gsf.covered_with_margin(start, start + length, tier), (tier, start, length)
        # well above the bound the guarantee is lost at most positions (tier - 2*margin admits a single tile start)
        assert not all(gsf.covered_with_margin(s, s + tier - 2 * gsf.EDGE_MARGIN, tier)
                       for s in range(gsf.EDGE_MARGIN, tier, step))
        # a gene at a contig edge has no sequence beyond it; the contig edge satisfies the margin on that side
        assert gsf.covered_with_margin(0, 5000, tier)
        assert gsf.covered_with_margin(3 * tier - 5000, 3 * tier, tier, contig_len=3 * tier)
