# tests/test_mvp_dataset.py

- Tokenizer · class · L17-L20 — class Tokenizer
- __call__ · method · L19-L20 — def __call__(self, sequence, **kwargs)
- spawned_sample · function · L23-L24 — def spawned_sample(ds, queue)
- DatasetTests · class · L27-L187 — class DatasetTests(unittest.TestCase)
- setUp · method · L28-L51 — def setUp(self)
- dataset · method · L53-L58 — def dataset(self, evidence=True, **kwargs)
- test_legacy_and_species_branches · method · L60-L74 — def test_legacy_and_species_branches(self)
- test_tracks_orientation_mask_and_metadata · method · L76-L87 — def test_tracks_orientation_mask_and_metadata(self)
- test_collate_dna_not_label_length_and_validation · method · L89-L102 — def test_collate_dna_not_label_length_and_validation(self)
- test_validation · method · L104-L108 — def test_validation(self)
- test_fail_closed_metadata · method · L110-L120 — def test_fail_closed_metadata(self)
- test_bad_corpus_and_token_lengths · method · L122-L134 — def test_bad_corpus_and_token_lengths(self)
- test_missing_or_nonboolean_orientation_and_all_n_metadata · method · L136-L150 — def test_missing_or_nonboolean_orientation_and_all_n_metadata(self)
- test_private_rng_dropout_eval_and_thinning · method · L152-L168 — def test_private_rng_dropout_eval_and_thinning(self)
- test_pickle_spawn_and_pid_worker_lifecycle · method · L170-L187 — def test_pickle_spawn_and_pid_worker_lifecycle(self)
