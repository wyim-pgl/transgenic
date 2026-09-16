# tests/test_train_ckpt_mode.py

- _load · function · L22-L27 — def _load(path, name)
- rt · function · L31-L32 — def rt()
- test_nonreentrant_is_the_default_on_one_gpu_and_under_ddp · function · L37-L40 — def test_nonreentrant_is_the_default_on_one_gpu_and_under_ddp(rt)
- test_retired_opt_in_is_a_no_op · function · L43-L46 — def test_retired_opt_in_is_a_no_op(rt): # the 2026-09-09 experiment switch must not break old launch lines, and must not change the outcome
- test_reentrant_is_allowed_on_one_gpu_only · function · L49-L52 — def test_reentrant_is_allowed_on_one_gpu_only(rt)
- test_no_grad_ckpt_experiment_switch_wins · function · L55-L57 — def test_no_grad_ckpt_experiment_switch_wins(rt)
- test_trainer_uses_the_resolver_and_has_no_bare_enable · function · L60-L64 — def test_trainer_uses_the_resolver_and_has_no_bare_enable()
- _func_source · function · L69-L74 — def _func_source(name: str) -> str
- test_directory_ops_are_main_process_only_and_save_state_is_collective · function · L78-L112 — def test_directory_ops_are_main_process_only_and_save_state_is_collective(fn)
- test_save_calls_are_not_gated_on_rank_at_the_call_site · function · L115-L122 — def test_save_calls_are_not_gated_on_rank_at_the_call_site()
- test_eval_loss_is_reduced_across_ranks_before_the_stopper · function · L127-L134 — def test_eval_loss_is_reduced_across_ranks_before_the_stopper()
