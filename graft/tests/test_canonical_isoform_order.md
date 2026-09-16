# tests/test_canonical_isoform_order.py

- test_transcript_permutation_invariance_and_idempotence · function · L6-L15 — def test_transcript_permutation_invariance_and_idempotence(gsf)
- test_transcripts_sorted_by_intron_count_then_chain · function · L18-L22 — def test_transcripts_sorted_by_intron_count_then_chain(gsf)
- test_identical_signatures_merge_and_ids_do_not_matter · function · L25-L30 — def test_identical_signatures_merge_and_ids_do_not_matter(gsf)
- test_feature_numbering_follows_first_use_after_ordering · function · L33-L41 — def test_feature_numbering_follows_first_use_after_ordering(gsf)
- test_rc_canonical_form_is_stable_under_double_rc · function · L44-L50 — def test_rc_canonical_form_is_stable_under_double_rc(gsf)
- test_gene_key_rule · function · L53-L60 — def test_gene_key_rule(gsf)
- test_intron_count_ignores_utr_cds_boundaries · function · L63-L72 — def test_intron_count_ignores_utr_cds_boundaries(gsf): # 5'UTR|CDS contiguous, one real intron, CDS|3'UTR contiguous -> intron_count 1, not 3
- test_no_cds_records_are_rejected · function · L75-L80 — def test_no_cds_records_are_rejected(gsf)
