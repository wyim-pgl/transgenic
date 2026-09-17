# src/transgenic/misc/trainHyenaMLM.py

- trainTransgenicFCGAccelerate · function · L16-L205 — def trainTransgenicFCGAccelerate( train_ds:isoformData, eval_ds:isoformData, lr=1e-4, num_epochs=10, schedule_lr=True, eval=True, batch_size=1, max_grad_norm=1.0, checkpoint_path="checkpoints_FCG/", safetensors_model=None, output_dir="saved_models_FCG/", accumulation_steps=32, notes="", log_wandb=True)
