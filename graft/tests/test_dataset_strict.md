# tests/test_dataset_strict.py

- dataset_cls · function · L23-L25 — def dataset_cls()
- _b5_db · function · L28-L37 — def _b5_db(path, rows)
- _make · function · L40-L41 — def _make(dataset_cls, db, **kw)
- test_empty_sequence_raises_under_strict · function · L44-L52 — def test_empty_sequence_raises_under_strict(tmp_path, dataset_cls)
- test_empty_sequence_substitutes_without_strict · function · L55-L63 — def test_empty_sequence_substitutes_without_strict(tmp_path, dataset_cls, capsys)
- test_unreadable_row_raises_under_strict · function · L66-L79 — def test_unreadable_row_raises_under_strict(tmp_path, dataset_cls, monkeypatch)
- Broken · class · L71-L73 — class Broken
- sql · method · L72-L73 — def sql(self, *a, **k)
- test_strict_defaults_off_so_published_behaviour_is_unchanged · function · L82-L85 — def test_strict_defaults_off_so_published_behaviour_is_unchanged(tmp_path, dataset_cls)
- test_b5_trainer_constructs_the_dataset_strict · function · L88-L95 — def test_b5_trainer_constructs_the_dataset_strict()
- test_a40_empty_target_matches_contract_through_dataset · function · L98-L121 — def test_a40_empty_target_matches_contract_through_dataset(tmp_path, dataset_cls, monkeypatch, gsf)
- Encoder · class · L102-L105 — class Encoder
- __call__ · method · L104-L105 — def __call__(self, seq, **kwargs)
