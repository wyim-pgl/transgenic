# tests/test_mvp_readiness_review.py

- test_sidecar_change_invalidates_check_and_preflight_bindings · function · L14-L35 — def test_sidecar_change_invalidates_check_and_preflight_bindings(tmp_path, monkeypatch, name, mutation)
- test_optional_sidecars_must_be_canonical_regular_files · function · L40-L54 — def test_optional_sidecars_must_be_canonical_regular_files(tmp_path, monkeypatch, name, kind)
- test_staging_preserves_complete_loader_inventory · function · L58-L74 — def test_staging_preserves_complete_loader_inventory(tmp_path, monkeypatch, present)
- test_staging_refuses_loader_inventory_drift_during_copy · function · L78-L95 — def test_staging_refuses_loader_inventory_drift_during_copy(tmp_path, monkeypatch, mutation)
- copy_then_mutate · function · L84-L91 — def copy_then_mutate(source, destination, expected, label)
- test_explicit_accelerate_config_survives_conflicting_default · function · L98-L137 — def test_explicit_accelerate_config_survives_conflicting_default(tmp_path, monkeypatch)
- execute · function · L108-L110 — def execute(command, log, timeout)
- test_accelerate_config_identity_checked · function · L141-L162 — def test_accelerate_config_identity_checked(tmp_path, monkeypatch, mutation)
- test_prepare_cannot_leave_accelerate_config_unbound_for_later_addition · function · L165-L171 — def test_prepare_cannot_leave_accelerate_config_unbound_for_later_addition(tmp_path, monkeypatch)
- test_accelerate_config_must_be_regular_and_nonsymlink · function · L175-L183 — def test_accelerate_config_must_be_regular_and_nonsymlink(tmp_path, monkeypatch, kind)
- test_changed_accelerate_config_stops_before_training_after_warmup · function · L186-L205 — def test_changed_accelerate_config_stops_before_training_after_warmup(tmp_path, monkeypatch)
- warmup · function · L196-L199 — def warmup(command, log, timeout)
