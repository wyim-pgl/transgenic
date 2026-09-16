# train/train_HyenaTransgenic_RTX4090.py

- TrainingProfile · class · L47-L75 — class TrainingProfile
- dataloader_kwargs · method · L61-L75 — def dataloader_kwargs(self, *, is_eval: bool = False) -> Dict
- _count_parameters · function · L93-L94 — def _count_parameters(model: torch.nn.Module) -> int
- _write_json · function · L97-L99 — def _write_json(path: str, obj: dict)
- _read_json · function · L102-L104 — def _read_json(path: str) -> dict
- _find_latest_checkpoint · function · L107-L132 — def _find_latest_checkpoint(checkpoint_path: str) -> Optional[str]
- train · function · L153-L727 — def train( train_ds, eval_ds, profile: TrainingProfile, *, encoder_d_model: int, decoder_d_model: int, encoder_layers: int, decoder_layers: int, encoder_heads: int, decoder_heads: int, lr: float, num_epochs: int, schedule_lr: bool, do_eval: bool, max_grad_norm: float, checkpoint_path: str, output_dir: str, encoder_model: str, notes: str, resume_from_checkpoint: Optional[str], save_every_epoch: bool, save_every_n_steps: int = 2000, wandb_log_interval: int = 10, unlink: bool, log_wandb: bool, train_sampler=None, train_dl_kwargs: Optional[Dict] = None, )
- _save_state · function · L362-L427 — def _save_state(epoch: int, step: int, global_step: int, best_score: Optional[float], running_loss_sum: float = 0.0, processed_batches: int = 0, skipped_batches: int = 0)
- _save_best · function · L429-L462 — def _save_best(tag: str, value: float)
