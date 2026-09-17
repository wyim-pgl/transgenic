# train/train_HyenaTransgenic.py

- _wandb · function · L26-L31 — def _wandb()
- _count_parameters · function · L59-L61 — def _count_parameters(model: torch.nn.Module) -> int
- _write_json · function · L64-L73 — def _write_json(path: str, obj: dict)
- _read_json · function · L76-L79 — def _read_json(path: str) -> dict
- _pending_grads_path · function · L82-L83 — def _pending_grads_path(ckpt_dir: str, rank: int) -> str
- _save_pending_grads · function · L86-L100 — def _save_pending_grads(model, ckpt_dir: str, rank: int) -> int
- _load_pending_grads · function · L103-L115 — def _load_pending_grads(model, ckpt_dir: str, rank: int) -> int
- _find_latest_checkpoint · function · L118-L143 — def _find_latest_checkpoint(checkpoint_path: str) -> str | None
- train · function · L150-L743 — def train( train_ds, # Training dataset (torch Dataset or Subset) eval_ds, # Evaluation dataset lr, # Peak learning rate (after warmup) num_epochs, # Total number of training epochs schedule_lr, # Whether to use LR scheduler (warmup + decay) do_eval, # Whether to run evaluation after each epoch batch_size, # Mini-batch size per forward pass max_grad_norm=1.0, # Max gradient norm for clipping checkpoint_path="checkpoints/", # Directory for saving checkpoints output_dir="saved_models/", # Directory for final model output accumulation_steps=16, # Gradient accumulation steps # Effective batch size = batch_size * accumulation_steps num_workers=8, # Number of DataLoader workers notes="", # Experiment notes logged to wandb encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", # Pretrained encoder to load resume_from_checkpoint: str | None = None, # Path to Accelerate checkpoint (or "auto") save_every_epoch: bool = True, # Save full resumable checkpoint every epoch save_every_n_steps: int = 5000, # Save full checkpoint every N optimizer steps unlink=False, # If True, don't share weights between enc/dec log_wandb=True, # Enable Weights & Biases logging b5_config: dict | None = None, # frozen recipe (configs/b5_400m_v1.json); None = legacy wide run seed: int = 234, # shuffling/init seed (B5: the run seed) patience: int = 3, # best-validation early stopping (B5) run_dir: str | None = None, # B5 checkpoint layout root (epoch_NN/, best, TRAINING_DONE) benchmark_steps: int = 0, # >0: time N optimizer steps, print JSON, exit )
- _save_state · function · L407-L422 — def _save_state(epoch: int, step: int, global_step: int, best_score)
- _save_best · function · L424-L430 — def _save_best(score_name, score_val)
- _save_epoch_b5 · function · L432-L453 — def _save_epoch_b5(epoch_1based: int, eval_loss, train_loss, is_best: bool, global_step: int)
- _save_latest_b5 · function · L455-L477 — def _save_latest_b5(epoch0: int, step: int, global_step: int)
- _on_signal · function · L481-L483 — def _on_signal(signum, frame)
