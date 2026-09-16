# train/train_HyenaTransgenic_GB10.py

- linear_decay · function · L64-L80 — def linear_decay(step, total_steps, start_value=0.5, end_value=0.0)
- get_attr · function · L83-L101 — def get_attr(obj, names)
- set_attr · function · L104-L120 — def set_attr(obj, names, val)
- _repeat_list · function · L123-L125 — def _repeat_list(val, n)
- _count_parameters · function · L128-L130 — def _count_parameters(model: torch.nn.Module) -> int
- _write_json · function · L133-L136 — def _write_json(path: str, obj: dict)
- _read_json · function · L139-L142 — def _read_json(path: str) -> dict
- _find_latest_checkpoint · function · L145-L170 — def _find_latest_checkpoint(checkpoint_path: str)
- trainTransgenicFCGAccelerate · function · L173-L524 — def trainTransgenicFCGAccelerate( train_ds: isoformData, # Training dataset (random_split subset) eval_ds: isoformData, # Evaluation dataset (random_split subset) lr, # Base learning rate (e.g., 5e-5) num_epochs, # Number of training epochs schedule_lr, # Whether to use linear warmup+decay LR schedule eval, # Whether to run evaluation after each epoch batch_size, # Micro-batch size per gradient accumulation step max_grad_norm=1.0, # Maximum gradient norm for clipping checkpoint_path="checkpoints_FCG/", # Directory for resumable Accelerate checkpoints output_dir="saved_models_FCG/", # Directory for final model outputs accumulation_steps=32, # Number of micro-batches per optimizer step notes="", # Free-text notes logged to W&B encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", # Pretrained encoder HF model ID resume_from_checkpoint: str | None = None, # Path to Accelerate checkpoint (or "auto") save_every_epoch: bool = True, # Save a resumable checkpoint at every epoch boundary save_every_n_steps: int = 5000, # Save full checkpoint every N optimizer steps num_workers: int = 4, # DataLoader worker count unlink=False, # Whether to untie decoder embedding and LM head weights log_wandb=True): # Whether to log metrics to Weights & Biases
- _save_training_state · function · L366-L383 — def _save_training_state(epoch: int, step: int, global_step: int, best_score)
