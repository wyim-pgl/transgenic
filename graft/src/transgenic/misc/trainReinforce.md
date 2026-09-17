# src/transgenic/misc/trainReinforce.py

- trainReinforce · function · L8-L141 — def trainReinforce( train_ds, decoder_checkpoint, segment_checkpoint, tokenizer, batch_size = 1, checkpoint_path = "checkpoints_PPO", output_dir= "saved_models_PPO", log_wandb=True, wandb_config={} ): # Set device
