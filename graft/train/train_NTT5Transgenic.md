# train/train_NTT5Transgenic.py

- trainT5Transgenic · function · L17-L180 — def trainT5Transgenic( train_ds:isoformData, eval_ds:isoformData, lr, num_epochs, schedule_lr, eval, batch_size, max_grad_norm=1.0, checkpoint_path="checkpoints_FCG/", safetensors_model=None, output_dir="saved_models_FCG/", accumulation_steps=32, notes="", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", unlink=False, log_wandb=True)
