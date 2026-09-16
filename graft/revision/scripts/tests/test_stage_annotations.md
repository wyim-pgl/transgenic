# revision/scripts/tests/test_stage_annotations.py

- test_normalise_strips_species_prefix · function · L13-L14 — def test_normalise_strips_species_prefix()
- test_normalise_leaves_tair_names_alone · function · L17-L19 — def test_normalise_leaves_tair_names_alone()
- test_normalise_maps_bare_numbers · function · L22-L23 — def test_normalise_maps_bare_numbers()
- test_stage_drops_unplaced_scaffolds · function · L26-L41 — def test_stage_drops_unplaced_scaffolds(tmp_path)
- _gff_lines · function · L44-L52 — def _gff_lines(seqid_gene_counts: dict) -> str
- test_validate_passes_for_healthy_meta · function · L58-L67 — def test_validate_passes_for_healthy_meta()
- test_validate_rejects_unrecognised_seqids · function · L70-L80 — def test_validate_rejects_unrecognised_seqids(tmp_path)
- test_validate_rejects_missing_chromosome · function · L83-L91 — def test_validate_rejects_missing_chromosome(tmp_path)
- test_validate_rejects_too_few_genes · function · L94-L102 — def test_validate_rejects_too_few_genes(tmp_path)
- test_validate_rejects_nonempty_dropped_seqids · function · L105-L115 — def test_validate_rejects_nonempty_dropped_seqids()
- test_main_aborts_before_writing_when_a_source_is_missing · function · L121-L140 — def test_main_aborts_before_writing_when_a_source_is_missing(tmp_path, monkeypatch, capsys)
