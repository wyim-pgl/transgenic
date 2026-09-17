# tests/test_segmentation_label_weight_alignment.py

- test_class_list_matches_preprocess · function · L9-L10 — def test_class_list_matches_preprocess(gsf)
- test_labels_and_weights_have_same_shape_and_0based_positions · function · L13-L19 — def test_labels_and_weights_have_same_shape_and_0based_positions(gsf): # GFF CDS 10..12 (1-based inclusive) inside a window starting at genomic 0-based 0 -> rows 9,10,11
- test_reference_labels_have_unit_weight_and_unlabelled_cells_zero · function · L22-L25 — def test_reference_labels_have_unit_weight_and_unlabelled_cells_zero(gsf)
- test_evidence_weight_formula_a18_4 · function · L36-L37 — def test_evidence_weight_formula_a18_4(gsf, source, n, geno, ri, expected)
- test_evidence_adds_positives_only_and_never_removes_reference_labels · function · L40-L46 — def test_evidence_adds_positives_only_and_never_removes_reference_labels(gsf)
- test_junction_evidence_labels_only_intron_and_boundary_classes · function · L49-L55 — def test_junction_evidence_labels_only_intron_and_boundary_classes(gsf)
- test_junction_family_weights_est_leads_a20 · function · L59-L62 — def test_junction_family_weights_est_leads_a20(gsf, source, expected_sw)
- test_junction_family_est_outranks_every_other_source · function · L65-L70 — def test_junction_family_est_outranks_every_other_source(gsf)
