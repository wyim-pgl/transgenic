# revision/scripts/tests/test_make_training_input_manifest.py

- md5 · function · L26-L27 — def md5(path: Path) -> str
- build_bundle · function · L30-L41 — def build_bundle(root: Path, species=SPECIES) -> Path
- run · function · L44-L46 — def run(root: Path)
- test_valid_bundle_is_accepted_and_writes_a_manifest · function · L49-L58 — def test_valid_bundle_is_accepted_and_writes_a_manifest(tmp_path)
- test_test_species_content_under_a_training_filename_is_refused · function · L61-L69 — def test_test_species_content_under_a_training_filename_is_refused(tmp_path)
- test_absent_species_manifest_is_refused · function · L72-L79 — def test_absent_species_manifest_is_refused(tmp_path)
- test_a_missing_training_genome_is_refused · function · L82-L88 — def test_a_missing_training_genome_is_refused(tmp_path)
- test_an_extra_genome_is_refused · function · L91-L96 — def test_an_extra_genome_is_refused(tmp_path)
- test_a_manifest_naming_other_than_nine_species_is_refused · function · L99-L103 — def test_a_manifest_naming_other_than_nine_species_is_refused(tmp_path)
- test_awk_header_error_is_not_masked_by_the_row_count_check · function · L106-L115 — def test_awk_header_error_is_not_masked_by_the_row_count_check(tmp_path)
- test_awk_programs_avoid_interval_expressions · function · L118-L135 — def test_awk_programs_avoid_interval_expressions()
