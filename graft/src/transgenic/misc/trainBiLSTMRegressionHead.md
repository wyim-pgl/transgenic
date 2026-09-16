# src/transgenic/misc/trainBiLSTMRegressionHead.py

- trainTransgenicFCGAccelerate · function · L15-L430 — def trainTransgenicFCGAccelerate( train_ds:isoformData, eval_ds:isoformData, lr, num_epochs, schedule_lr, eval, batch_size, max_grad_norm=1.0, checkpoint_path="checkpoints_BiLSTM/", safetensors_model=None, output_dir="saved_models_BiLSTM/", accumulation_steps=32, notes="", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", unlink=False, log_wandb=True)
