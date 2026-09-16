# tests/test_gff_gsf_roundtrip_tair10.py

- test_gff_1based_inclusive_converts_to_0based_half_open · function · L6-L11 — def test_gff_1based_inclusive_converts_to_0based_half_open(gsf): # README example 1: CDS 100-150 with window start 100 -> [0, 51)
- test_readme_example_3_with_utrs · function · L14-L18 — def test_readme_example_3_with_utrs(gsf)
- test_one_bp_and_51bp_features · function · L21-L25 — def test_one_bp_and_51bp_features(gsf)
- test_roundtrip_tair10_minus_strand_single_cds · function · L28-L35 — def test_roundtrip_tair10_minus_strand_single_cds(gsf)
- test_roundtrip_tair10_last_gene_utr_order_on_minus_strand · function · L38-L46 — def test_roundtrip_tair10_last_gene_utr_order_on_minus_strand(gsf)
- test_last_gene_of_file_is_emitted · function · L49-L56 — def test_last_gene_of_file_is_emitted(gsf)
- test_chromosome_transition_keeps_each_genes_own_chrom_and_strand · function · L59-L62 — def test_chromosome_transition_keeps_each_genes_own_chrom_and_strand(gsf)
- test_exact_multiple_padding_adds_no_extra_chunk · function · L66-L69 — def test_exact_multiple_padding_adds_no_extra_chunk(gsf, length, expected)
- test_tiered_window_policy · function · L72-L87 — def test_tiered_window_policy(gsf)
