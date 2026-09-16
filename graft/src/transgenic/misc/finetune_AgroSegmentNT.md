# src/transgenic/misc/finetune_AgroSegmentNT.py

- trainSegmentNTAccelerate · function · L18-L194 — def trainSegmentNTAccelerate( train_ds:isoformData, eval_ds:isoformData, lr, num_epochs, schedule_lr, eval, batch_size, max_grad_norm=1.0, checkpoint_path="checkpoints_FCG/", safetensors_model=None, output_dir="saved_models_FCG/", accumulation_steps=32, notes="", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", segmentation_model="InstaDeepAI/segment_nt_multi_species", unlink=False): # start a new wandb run to track this script
