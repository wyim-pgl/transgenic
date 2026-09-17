# tests/test_mvp_runtime.py

- OptimizerToy · class · L21-L28 — class OptimizerToy(torch.nn.Module)
- __init__ · method · L22-L28 — def __init__(self)
- test_recipe_validation_and_schedule_points · function · L31-L44 — def test_recipe_validation_and_schedule_points()
- test_optimizer_groups_are_named_disjoint_exhaustive_and_uniform_decay · function · L47-L64 — def test_optimizer_groups_are_named_disjoint_exhaustive_and_uniform_decay()
- test_factory_rejects_conflicting_ddp_handler · function · L67-L73 — def test_factory_rejects_conflicting_ddp_handler()
- test_all_250_updates_match_post_step_lambdalr_parameters_and_state · function · L76-L110 — def test_all_250_updates_match_post_step_lambdalr_parameters_and_state()
- _probe · function · L113-L123 — def _probe(ranks, case, timeout=120)
- test_actual_accelerate_cpu_gloo_core · function · L127-L142 — def test_actual_accelerate_cpu_gloo_core(ranks)
- test_collective_optimizer_contract_rejects_before_writes · function · L145-L152 — def test_collective_optimizer_contract_rejects_before_writes()
- test_literal_four_rank_250_by_24_budget · function · L155-L168 — def test_literal_four_rank_250_by_24_budget()
- test_collective_failures_and_contract_rejections · function · L175-L189 — def test_collective_failures_and_contract_rejections(case)
