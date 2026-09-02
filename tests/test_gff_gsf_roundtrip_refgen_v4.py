"""RefGen_V4 style (numeric chromosome names, plus strand, UTR+CDS) and DB row contract §6 (SQL NULL)."""
from conftest import REFGEN_V4_PLUS, gff


def test_roundtrip_refgen_v4_numeric_chromosome(gsf):
    gene = next(gsf.parse_gff3(REFGEN_V4_PLUS.splitlines()))
    assert gene.chrom == "1" and gene.strand == "+"
    ws, we = gsf.pad_window(gene.start0, gene.end0)
    assert (we - ws) % 6144 == 0
    s = gsf.gene_to_gsf(gene, ws)
    back = gsf.gsf_to_gene(s, window_start=ws, chrom="1", strand="+", gene_id="Zm00001d027230")
    assert gsf.gene_to_gsf(back, ws) == s
    feats = s.split(">")[0].split(";")
    assert feats[0].split("|")[1] == "five_prime_UTR1" and feats[-1].split("|")[1] == "three_prime_UTR1"


def test_relative_coordinates_are_window_based_not_gene_based(gsf):
    gene = next(gsf.parse_gff3(REFGEN_V4_PLUS.splitlines()))
    ws, _ = gsf.pad_window(gene.start0, gene.end0)
    first = gsf.gene_to_gsf(gene, ws).split(";")[0].split("|")
    assert int(first[0]) == gene.start0 - ws  # 0-based gene start relative to the padded window


def test_build_rows_uses_sql_null_for_absent_labels(gsf):
    gene = next(gsf.parse_gff3(REFGEN_V4_PLUS.splitlines()))
    rows = gsf.build_rows([gene], species_id="Zmays", rc="none", split_lookup={gene.gene_id: "test"})  # verbatim GFF ID without a species code
    assert len(rows) == 1
    r = rows[0]
    assert r["predict"] is None and r["is_rc"] is False and r["split"] == "test"
    assert r["ordering_version"] == "gsf-order-v1" and r["window_policy"] == "sym6144-v1"
    assert r["gsf_token_count"] == gsf.count_tokens_v2(r["gsf"])


def test_build_rows_rejects_gene_without_split_entry(gsf):
    gene = next(gsf.parse_gff3(REFGEN_V4_PLUS.splitlines()))
    try:
        gsf.build_rows([gene], species_id="Zmays", rc="none", split_lookup={})
    except gsf.SplitError:
        return
    raise AssertionError("a gene absent from the split table must fail closed")
