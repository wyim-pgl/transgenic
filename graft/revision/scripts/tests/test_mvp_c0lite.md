# revision/scripts/tests/test_mvp_c0lite.py

- load_builder · function · L16-L21 — def load_builder()
- record · function · L24-L25 — def record(name, start=500, cigar='20M', flag=0, ts='+', chrom='Chr1')
- inputs · function · L28-L62 — def inputs(tmp_path, records, panels=None, tiles=None, blocks=None)
- build · function · L65-L68 — def build(tmp_path, records, **kwargs)
- test_transitive_closure_late_supplementary_and_cause_priority · function · L72-L83 — def test_transitive_closure_late_supplementary_and_cause_priority(tmp_path, reverse)
- test_exact_key_cigar_flag_chrom_not_ts_and_clone_suffixes · function · L86-L99 — def test_exact_key_cigar_flag_chrom_not_ts_and_clone_suffixes(tmp_path)
- test_flag_ts_and_cigar_geometry · function · L104-L109 — def test_flag_ts_and_cigar_geometry(tmp_path, flag, ts, want)
- test_boundaries_intronic_leakage_and_no_primary_precedence · function · L112-L123 — def test_boundaries_intronic_leakage_and_no_primary_precedence(tmp_path)
- test_union_member_geometry_distinct_membership_and_no_intron_membership · function · L126-L137 — def test_union_member_geometry_distinct_membership_and_no_intron_membership(tmp_path)
- test_safe_output_cli_and_invalid_inputs · function · L140-L156 — def test_safe_output_cli_and_invalid_inputs(tmp_path)
- test_schema_and_checksum_fail_closed · function · L159-L171 — def test_schema_and_checksum_fail_closed(tmp_path)
- test_b5_short_contig_padded_tile_is_legal · function · L174-L179 — def test_b5_short_contig_padded_tile_is_legal(tmp_path): # gsf_contract.tile_windows(1000,30720) returns [(0,30720)].
- test_bounded_batches_preserve_edges_and_all_member_geometry · function · L182-L192 — def test_bounded_batches_preserve_edges_and_all_member_geometry(tmp_path)
- test_independent_qc_blocks_publication_of_corrupted_disposition · function · L195-L206 — def test_independent_qc_blocks_publication_of_corrupted_disposition(tmp_path)
- corrupt · function · L199-L201 — def corrupt(con,qc)
- test_empty_bam_is_supplied_empty_source_with_provenance · function · L209-L217 — def test_empty_bam_is_supplied_empty_source_with_provenance(tmp_path)
