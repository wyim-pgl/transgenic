# revision/scripts/tests/test_epoch_budget.py

- power_curve · function · L20-L21 — def power_curve(n, a=1.20, b=1.5, c=0.9)
- test_parse_trainer_log_line_with_tensors · function · L24-L35 — def test_parse_trainer_log_line_with_tensors(): # Arrange: the exact stderr format of train/train_HyenaTransgenic.py
- test_parse_csv_with_optional_train_column · function · L38-L44 — def test_parse_csv_with_optional_train_column(tmp_path)
- test_best_and_patience_stop_on_overfitting_curve · function · L47-L54 — def test_best_and_patience_stop_on_overfitting_curve(): # eval improves to epoch 6 then rises; patience 3 -> stop after epoch 9
- test_plateau_curve_without_overfitting_recommends_plateau_plus_margin · function · L57-L63 — def test_plateau_curve_without_overfitting_recommends_plateau_plus_margin()
- test_power_law_fit_recovers_parameters_and_extrapolates · function · L66-L73 — def test_power_law_fit_recovers_parameters_and_extrapolates()
- test_short_run_reports_insufficient_data · function · L76-L78 — def test_short_run_reports_insufficient_data()
- test_cli_end_to_end_json · function · L81-L88 — def test_cli_end_to_end_json(tmp_path)
