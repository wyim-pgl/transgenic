"""docs/gsf_spec_v1.md §1 (coordinates), §2 (grammar), DB contract §6 (EOF flush, chromosome ownership)."""
import pytest
from conftest import TAIR10_SINGLE_CDS_MINUS, TAIR10_LAST_GENE_MINUS, gff


def test_gff_1based_inclusive_converts_to_0based_half_open(gsf):
    # README example 1: CDS 100-150 with window start 100 -> [0, 51)
    text = gff("Chr1", "g1", "+", {"t1": [("CDS", 100, 150, 0), ("CDS", 200, 280, 2), ("CDS", 350, 400, 1)]})
    gene = next(gsf.parse_gff3(text.splitlines()))
    s = gsf.gene_to_gsf(gene, window_start=99)  # window starts at genomic 0-based 99 (= GFF 100)
    assert s == "0|CDS1|51|+|A;100|CDS2|181|+|C;250|CDS3|301|+|B>CDS1|CDS2|CDS3"


def test_readme_example_3_with_utrs(gsf):
    text = gff("Chr1", "g1", "+", {"t1": [("five_prime_UTR", 500, 550, "."), ("CDS", 550, 650, 0), ("CDS", 700, 800, 1), ("three_prime_UTR", 800, 900, ".")]})
    gene = next(gsf.parse_gff3(text.splitlines()))
    assert gsf.gene_to_gsf(gene, window_start=499) == (
        "0|five_prime_UTR1|51|+|.;50|CDS1|151|+|A;200|CDS2|301|+|B;300|three_prime_UTR1|401|+|.>five_prime_UTR1|CDS1|CDS2|three_prime_UTR1")


def test_one_bp_and_51bp_features(gsf):
    text = gff("Chr1", "g1", "+", {"t1": [("CDS", 10, 10, 0), ("CDS", 20, 70, 2)]})
    gene = next(gsf.parse_gff3(text.splitlines()))
    s = gsf.gene_to_gsf(gene, window_start=9)
    assert s.startswith("0|CDS1|1|+|A;10|CDS2|61|+|C>")  # 1-bp feature has length 1, 51-bp has length 51


def test_roundtrip_tair10_minus_strand_single_cds(gsf):
    gene = next(gsf.parse_gff3(TAIR10_SINGLE_CDS_MINUS.splitlines()))
    ws, we = gsf.pad_window(gene.start0, gene.end0)
    s = gsf.gene_to_gsf(gene, ws)
    back = gsf.gsf_to_gene(s, window_start=ws, chrom="ChrM", strand="-", gene_id="ATMG00010")
    assert gsf.gene_to_gsf(back, ws) == s
    assert back.chrom == "ChrM" and back.strand == "-"
    assert [(f.type, f.start1, f.end1) for f in back.transcripts["t1" if "t1" in back.transcripts else next(iter(back.transcripts))]] == [("CDS", 273, 734)]


def test_roundtrip_tair10_last_gene_utr_order_on_minus_strand(gsf):
    gene = next(gsf.parse_gff3(TAIR10_LAST_GENE_MINUS.splitlines()))
    ws, _ = gsf.pad_window(gene.start0, gene.end0)
    s = gsf.gene_to_gsf(gene, ws)
    feats, txs = s.split(">")
    # transcript list is transcription-oriented: 5'UTR first even though it has the highest coordinate
    assert txs.split("|")[0].startswith("five_prime_UTR")
    back = gsf.gsf_to_gene(s, window_start=ws, chrom="Chr5", strand="-", gene_id="AT5G67640")
    assert gsf.gene_to_gsf(back, ws) == s


def test_last_gene_of_file_is_emitted(gsf):
    text = gff("Chr1", "g1", "+", {"t1": [("CDS", 100, 200, 0)]}) + TAIR10_LAST_GENE_MINUS
    genes = list(gsf.parse_gff3(text.splitlines()))
    assert genes[-1].gene_id_original.startswith("AT5G67640") and genes[-1].gene_id == "AT5G67640.TAIR10"
    assert len(genes) == 2
    keyed = list(gsf.parse_gff3(text.splitlines(), species_code="Ath"))
    # "AT5G67640.TAIR10" is 16 characters -> generated key; "g1" is valid and kept
    assert [g.gene_id for g in keyed] == ["g1", "Ath000002"] and keyed[1].gene_id_original == "AT5G67640.TAIR10"


def test_chromosome_transition_keeps_each_genes_own_chrom_and_strand(gsf):
    text = gff("Chr1", "g1", "+", {"t1": [("CDS", 100, 200, 0)]}) + gff("Chr2", "g2", "-", {"t2": [("CDS", 300, 400, 0)]})
    genes = list(gsf.parse_gff3(text.splitlines()))
    assert [(g.gene_id, g.chrom, g.strand) for g in genes] == [("g1", "Chr1", "+"), ("g2", "Chr2", "-")]


@pytest.mark.parametrize("length,expected", [(6144, 6144), (6145, 12288), (100, 6144), (12288, 12288)])
def test_exact_multiple_padding_adds_no_extra_chunk(gsf, length, expected):
    ws, we = gsf.pad_window(1000, 1000 + length)
    assert we - ws == expected
    assert ws <= 1000 and we >= 1000 + length


def test_tiered_window_policy(gsf):
    import random
    ws, we, tier = gsf.pad_window_tiered(10000, 15000)          # 5 kb gene -> smallest tier
    assert tier == 30720 and we - ws == 30720 and ws <= 9000 and we >= 16000
    ws, we, tier = gsf.pad_window_tiered(10000, 45000)          # 35 kb + flanks -> 61,440
    assert tier == 61440
    ws, we, tier = gsf.pad_window_tiered(100000, 200000)        # 100 kb -> 129,024
    assert tier == 129024
    ws, we, tier = gsf.pad_window_tiered(100000, 228000)        # too long -> over the cap (rejected downstream)
    assert tier > gsf.MAX_WINDOW_V2
    rng = random.Random(1)
    tiers = {gsf.pad_window_tiered(10000, 15000, rng=rng, tier_up_prob=0.5)[2] for _ in range(50)}
    assert tiers == {30720, 61440}                                # augmentation picks the next tier sometimes, never two up
    for _ in range(20):
        ws, we, tier = gsf.pad_window_tiered(10000, 15000, rng=rng, tier_up_prob=0.0)
        assert ws + 1000 <= 10000 and 15000 + 1000 <= we and we - ws == 30720
