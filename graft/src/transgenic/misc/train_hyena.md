# src/transgenic/misc/train_hyena.py

- linear_decay · function · L15-L20 — def linear_decay(step, total_steps, start_value=0.5, end_value=0.0)
- trainTransgenicFCGAccelerate · function · L22-L234 — def trainTransgenicFCGAccelerate( train_ds:isoformData, eval_ds:isoformData, lr, num_epochs, schedule_lr, eval, batch_size, max_grad_norm=1.0, checkpoint_path="checkpoints_FCG/", safetensors_model=None, output_dir="saved_models_FCG/", accumulation_steps=32, notes="", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", unlink=False, log_wandb=True)
