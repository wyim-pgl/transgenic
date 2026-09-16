# revision/scripts/tests/test_est_blastx_fixture.py

- TracebackTests · class · L11-L81 — class TracebackTests(unittest.TestCase)
- setUpClass · method · L13-L17 — def setUpClass(cls)
- setUp · method · L19-L20 — def setUp(self)
- row · method · L22-L25 — def row(self, dna, btop, cigar, qg, sg, translated, reverse=False)
- test_replays_both_frameshift_directions_and_reverse_complement · method · L27-L36 — def test_replays_both_frameshift_directions_and_reverse_complement(self)
- test_replays_inframe_insertions_deletions · method · L38-L42 — def test_replays_inframe_insertions_deletions(self)
- test_rejects_corrupted_btop_translation_endpoint_and_cigar · method · L44-L51 — def test_rejects_corrupted_btop_translation_endpoint_and_cigar(self)
- test_getseq_argv_uses_supported_options · method · L53-L73 — def test_getseq_argv_uses_supported_options(self): # Contract-only interception: no DIAMOND output or engine pass is simulated.
- GetseqChecked · class · L55-L56 — class GetseqChecked(Exception)
- check_getseq · function · L63-L67 — def check_getseq(command, log)
- test_fixture_has_required_cases_and_species_sources · method · L75-L81 — def test_fixture_has_required_cases_and_species_sources(self)
