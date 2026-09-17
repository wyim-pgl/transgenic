# tests/test_geenuff_qc.py

- _load · function · L22-L27 — def _load(path, name)
- qc · function · L31-L32 — def qc()
- _make_db · function · L35-L66 — def _make_db(path)
- test_extract_error_features_through_transcript_piece · function · L69-L83 — def test_extract_error_features_through_transcript_piece(qc, tmp_path)
- test_cli_writes_flags_and_summary_with_hard_soft_counts · function · L86-L98 — def test_cli_writes_flags_and_summary_with_hard_soft_counts(qc, tmp_path)
- test_pinned_geenuff_error_names_map_to_hard_or_soft · function · L101-L106 — def test_pinned_geenuff_error_names_map_to_hard_or_soft(qc)
- test_loss_mask_policy_a22 · function · L120-L129 — def test_loss_mask_policy_a22(qc): # gene-level hard error -> masked (weight 0)
- test_read_flags_tsv · function · L132-L138 — def test_read_flags_tsv(qc, tmp_path)
