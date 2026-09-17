# src/transgenic/training/mvp_runtime.py

- MVPRecipe · class · L22-L54 — class MVPRecipe
- __post_init__ · method · L33-L54 — def __post_init__(self) -> None
- make_accelerator · function · L57-L79 — def make_accelerator(recipe: MVPRecipe, **kwargs)
- build_optimizer · function · L82-L104 — def build_optimizer(model: torch.nn.Module, recipe: MVPRecipe) -> torch.optim.AdamW
- lr_scale · function · L107-L118 — def lr_scale(completed_updates: int, recipe: MVPRecipe) -> float
- _collective_true · function · L121-L123 — def _collective_true(accelerator, condition: bool) -> bool
- _all_ints · function · L126-L128 — def _all_ints(accelerator, value: int) -> list[int]
- _set_lrs · function · L131-L134 — def _set_lrs(optimizer, recipe: MVPRecipe, completed_updates: int) -> None
- _step_was_skipped · function · L137-L139 — def _step_was_skipped(optimizer) -> bool
- _validate_accelerator · function · L142-L208 — def _validate_accelerator(accelerator, model, optimizer, train_loader, recipe: MVPRecipe) -> None
- _batch_parts · function · L211-L226 — def _batch_parts(batch: Any, recipe: MVPRecipe)
- run_fixed_updates · function · L229-L311 — def run_fixed_updates(accelerator, model, optimizer, train_loader, recipe: MVPRecipe) -> dict
