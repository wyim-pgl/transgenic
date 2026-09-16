# src/transgenic/misc/train_transgenic.py

- linear_decay · function · L19-L24 — def linear_decay(step, total_steps, start_value=0.5, end_value=0.0)
- split_tokens · function · L26-L42 — def split_tokens(predicted_tokens)
- listFlatten · function · L44-L51 — def listFlatten(l)
- trainTransgenicFCGAccelerate · function · L54-L359 — def trainTransgenicFCGAccelerate( train_ds:isoformData, eval_ds:isoformData, lr, num_epochs, schedule_lr, eval, batch_size, max_grad_norm=1.0, checkpoint_path="checkpoints_FCG/", safetensors_model=None, output_dir="saved_models_FCG/", accumulation_steps=32, notes="", encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", unlink=False, generation_mixin=False, log_wandb=True)
