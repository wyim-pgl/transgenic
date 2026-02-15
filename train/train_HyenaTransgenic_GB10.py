#!/usr/bin/env python
"""
TransGenic training script — NVIDIA GB10 (Blackwell) variant.

Trains the TransGenic seq2seq model (HyenaDNA encoder + Longformer decoder)
for gene structure prediction from raw DNA sequences. This version is
specifically tuned for the NVIDIA GB10 platform:

  - 128.5 GB unified memory, GPU capped at ~100 GB via set_per_process_memory_fraction
  - SM 12.1 architecture — requires source-built PyTorch with SM 12.0 target
  - torch.compile DISABLED: Triton's fused layernorm backward kernel needs
    180 KB SM shared memory, but GB10 only has 101 KB. Forward compiles OK
    but backward fails every batch, so the model never learns.
  - cudaMallocAsync DISABLED: over-allocates on unified memory systems
  - pin_memory=False: not beneficial on unified memory (already coherent)
  - num_workers=4: fewer workers needed since data loading is ~0% of time

Architecture (wide config, ~1.17B params):
  Encoder: HyenaDNA (pretrained), d_model=1152, 16 layers
  Decoder: Longformer, d_model=2304, 16 layers, 8 heads, window=1024
  Downsampling: 2-stage Conv1d with relative positional bias (6x compression)

Training recipe:
  - bf16 mixed precision via Accelerate (no loss scaling needed on Ampere+)
  - 8-bit AdamW optimizer (bitsandbytes) for ~75% optimizer memory savings
  - Gradient checkpointing (recompute activations to save memory)
  - Gradient accumulation: effective batch = batch_size(8) * accumulation(32) = 256
  - Linear warmup (5%) + linear decay LR schedule
  - Resumable via Accelerate checkpoints with epoch metadata

Data:
  DuckDB database containing genomic regions + GFF annotations.
  'Zm' (maize) genes excluded at dataset level for cross-species evaluation.

Usage:
  python train_HyenaTransgenic_GB10.py --db /path/to/data.db [--no-wandb]

See train_HyenaTransgenic.py for the generic (4090-optimized) version.
"""

import os
# HuggingFace model cache directory — must be set before importing transformers
os.environ['HF_HOME'] = './HFmodels'
# cudaMallocAsync disabled — over-allocates on GB10 unified memory with GPU cap.
# The default cudaMalloc allocator manages the memory pool more conservatively.
# os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:cudaMallocAsync')

import torch, wandb, gc, time, sys, math, json, argparse, glob
from tqdm import tqdm
import torch.optim as optim            # Standard PyTorch optimizers (unused, kept for reference)
import bitsandbytes as bnb             # 8-bit optimizer library for AdamW8bit
from torch.nn.utils import clip_grad_norm_  # Gradient clipping to prevent exploding gradients
from transformers import get_linear_schedule_with_warmup  # LR schedule: warmup then linear decay
from accelerate import Accelerator     # HuggingFace multi-GPU / mixed-precision wrapper
from safetensors.torch import save_model  # Fast, safe model serialization format

# Dataset and model imports
from transgenic.datasets.datasets import isoformData, isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.tokenization_transgenic import GFFTokenizer
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig


def linear_decay(step, total_steps, start_value=0.5, end_value=0.0):
	"""Linear decay function for auxiliary scheduling (e.g., label smoothing).

	Args:
		step: Current training step.
		total_steps: Total number of steps for full decay.
		start_value: Initial value at step 0.
		end_value: Final value at step >= total_steps.

	Returns:
		Linearly interpolated value between start_value and end_value.
	"""
	if step >= total_steps:
		return end_value  # Clamp at end value after schedule completes

	decay_rate = (start_value - end_value) / total_steps  # Compute per-step decrement
	return start_value - (decay_rate * step)


def get_attr(obj, names):
	"""Recursively get a nested attribute by dotted name list.

	Handles both regular attributes and ModuleList indexing.
	Example: get_attr(model, ["encoder", "layers", "0", "self_attn"])

	Args:
		obj: Root object to traverse.
		names: List of attribute name strings (numeric strings index ModuleLists).

	Returns:
		The nested attribute value.
	"""
	if len(names) == 1:
		return getattr(obj, names[0])  # Base case: single attribute
	elif type(obj) == torch.nn.modules.container.ModuleList:
		return get_attr(obj[int(names[0])], names[1:])  # Index into ModuleList
	else:
		return get_attr(getattr(obj, names[0]), names[1:])  # Traverse named attribute


def set_attr(obj, names, val):
	"""Recursively set a nested attribute by dotted name list.

	Mirror of get_attr for setting values. Used for LoRA injection
	or parameter freezing by name path.

	Args:
		obj: Root object to traverse.
		names: List of attribute name strings.
		val: Value to set at the terminal attribute.
	"""
	if len(names) == 1:
		setattr(obj, names[0], val)  # Base case: set the attribute
	elif type(obj) == torch.nn.modules.container.ModuleList:
		return set_attr(obj[int(names[0])], names[1:], val)  # Index into ModuleList
	else:
		set_attr(getattr(obj, names[0]), names[1:], val)  # Traverse named attribute


def _repeat_list(val, n):
	"""Create a list by repeating a value n times. Used for attention_window config."""
	return [val for _ in range(n)]


def _count_parameters(model: torch.nn.Module) -> int:
	"""Count total number of parameters in a model (trainable + frozen)."""
	return sum(p.numel() for p in model.parameters())


def _write_json(path: str, obj: dict):
	"""Write a dictionary as pretty-printed JSON to a file."""
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: str) -> dict:
	"""Read a JSON file and return as a dictionary."""
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _find_latest_checkpoint(checkpoint_path: str):
	"""Find the most recent Accelerate checkpoint by global_step in meta.json.

	Searches for directories matching ``accelerate_*`` inside *checkpoint_path*,
	reads each ``meta.json``, and returns the directory with the highest
	``global_step``.  Falls back to the most-recently-modified directory if
	no ``meta.json`` is found.
	"""
	dirs = [d for d in glob.glob(os.path.join(checkpoint_path, "accelerate_*")) if os.path.isdir(d)]
	if not dirs:
		return None
	best, best_step = None, -1
	for d in dirs:
		meta = os.path.join(d, "meta.json")
		if os.path.exists(meta):
			try:
				step = int(_read_json(meta).get("global_step", 0))
				if step > best_step:
					best_step = step
					best = d
			except Exception:
				pass
	if best is None:
		dirs.sort(key=os.path.getmtime)
		best = dirs[-1]
	return best


def trainTransgenicFCGAccelerate(
	train_ds: isoformData,              # Training dataset (random_split subset)
	eval_ds: isoformData,               # Evaluation dataset (random_split subset)
	lr,                                  # Base learning rate (e.g., 5e-5)
	num_epochs,                          # Number of training epochs
	schedule_lr,                         # Whether to use linear warmup+decay LR schedule
	eval,                                # Whether to run evaluation after each epoch
	batch_size,                          # Micro-batch size per gradient accumulation step
	max_grad_norm=1.0,                   # Maximum gradient norm for clipping
	checkpoint_path="checkpoints_FCG/",  # Directory for resumable Accelerate checkpoints
	output_dir="saved_models_FCG/",      # Directory for final model outputs
	accumulation_steps=32,               # Number of micro-batches per optimizer step
	notes="",                            # Free-text notes logged to W&B
	encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",  # Pretrained encoder HF model ID
	resume_from_checkpoint: str | None = None,  # Path to Accelerate checkpoint (or "auto")
	save_every_epoch: bool = True,       # Save a resumable checkpoint at every epoch boundary
	save_every_n_steps: int = 5000,      # Save full checkpoint every N optimizer steps
	num_workers: int = 4,                # DataLoader worker count
	unlink=False,                        # Whether to untie decoder embedding and LM head weights
	log_wandb=True):                     # Whether to log metrics to Weights & Biases
	"""
	Main training loop for TransGenic on NVIDIA GB10.

	Trains the seq2seq model (HyenaDNA encoder → Conv1d downsampling → Longformer decoder)
	using gradient accumulation, 8-bit AdamW, bf16 mixed precision, and gradient checkpointing.
	Supports resuming from Accelerate checkpoints and logging to Weights & Biases.

	The effective batch size = batch_size * accumulation_steps (e.g., 8 * 32 = 256).
	"""

	# Create output directories if they don't exist
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Initialize Weights & Biases run for experiment tracking
	if log_wandb:
		wandb.init(
			entity="transgenic-paper",       # W&B team/organization
			project="transgenic",             # W&B project name
			config={                          # Log all hyperparameters
			"learning_rate": lr,
			"schedule_lr": schedule_lr,
			"architecture": "Hyena",          # Encoder type identifier
			"dataset": "10G_static6144_addExtra200_addRCIsoOnly_clean",
			"epochs": num_epochs,
			"max_grad_norm": max_grad_norm,
			"accumulation_steps": accumulation_steps,
			"Optimizer": "AdamW",
			"Checkpoints": checkpoint_path,
			"Outputs": output_dir,
			"Notes": notes,
			}
		)

	print(f"Training transgenic with Hyena. {checkpoint_path=} {output_dir=}", file=sys.stderr)

	# Enable TF32 for matmul — uses Blackwell tensor cores for ~8x throughput over FP32
	torch.set_float32_matmul_precision('high')
	# Let cuDNN benchmark and cache the fastest convolution algorithm for each input shape
	torch.backends.cudnn.benchmark = True

	# Initialize Accelerator with bf16 mixed precision (better dynamic range than fp16)
	accelerator = Accelerator(mixed_precision="bf16")
	device = accelerator.device
	# Cap GPU memory at ~100 GB on GB10's 128.5 GB unified memory
	# Without this cap, PyTorch can allocate all 128 GB and trigger the Linux OOM killer
	torch.cuda.set_per_process_memory_fraction(0.78)  # 128.5 * 0.78 ≈ 100 GB
	print(f"Using: {device} (GPU mem capped at {torch.cuda.get_device_properties(0).total_memory * 0.78 / 1e9:.0f}GB)", file=sys.stderr)

	# Set random seeds for reproducibility across data shuffling and weight init
	torch.manual_seed(234)
	torch.cuda.manual_seed_all(234)

	# Create DataLoaders
	# pin_memory=False: not useful on unified memory (CPU ↔ GPU share the same physical memory)
	# num_workers=4: sufficient since data loading is <1% of training time (compute-bound)
	# persistent_workers=True: keep worker processes alive between epochs to avoid respawning cost
	train_dl = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=num_workers, collate_fn=hyena_collate_fn, persistent_workers=True)
	eval_dl = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=num_workers, collate_fn=hyena_collate_fn, persistent_workers=True)

	# ── Model Architecture ─────────────────────────────────────────────────
	# Wide config: ~3x parameters vs base model
	# (1152/768)^2 * (16/12) ≈ 3.0x parameter scaling
	base_d_model = 1152  # Hidden dimension (encoder output, decoder input after 2x expansion)
	layers = 16          # Number of transformer layers in both encoder and decoder
	attn_heads = 8       # Number of attention heads per layer

	attentionWindow = _repeat_list(1024, layers)  # Longformer local attention: 1024 tokens per side
	ffn_dim = base_d_model * 4                     # Feed-forward dimension: 4x hidden size = 4608

	config = HyenaTransgenicConfig(
		d_model=base_d_model,          # Base hidden dimension
		encoder_layers=layers,          # HyenaDNA encoder depth
		decoder_layers=layers,          # Longformer decoder depth
		encoder_n_layer=layers,         # HyenaDNA internal layer count (must match encoder_layers)
		encoder_ffn_dim=ffn_dim,        # Encoder feed-forward inner dimension
		decoder_ffn_dim=ffn_dim,        # Decoder feed-forward inner dimension
		attention_window=attentionWindow,  # Per-layer Longformer attention window sizes
		dropout=0.1,                    # Dropout probability (regularization)
		encoder_attention_heads=attn_heads,  # Encoder multi-head attention heads
		decoder_attention_heads=attn_heads,  # Decoder multi-head attention heads
		encoder_model=encoder_model,    # HuggingFace ID for pretrained HyenaDNA weights
		unlink=unlink,                  # Whether to untie embedding ↔ LM head
	)
	model = transgenicForConditionalGeneration(config)  # Instantiate the full seq2seq model

	print(f"Model params: {_count_parameters(model):,}", file=sys.stderr)

	# Enable gradient checkpointing: trades ~30% more compute for ~60% less activation memory
	# Critical for fitting the 1.17B-param model on GB10's 100 GB GPU budget
	model.gradient_checkpointing_enable()
	model.to(device)   # Move model to GPU
	model.train()      # Set training mode (enables dropout, disables eval shortcuts)

	# torch.compile disabled on GB10 — Triton fused layernorm backward kernel needs
	# 180 KB SM shared memory but GB10 (SM 12.1) only has 101 KB per SM.
	# Forward pass compiles fine but backward fails on EVERY batch, preventing any learning.
	# Set TRANSGENIC_COMPILE=1 to opt-in on GPUs with larger SM shared memory (4090, A100, H100).
	if os.environ.get("TRANSGENIC_COMPILE"):
		try:
			model = torch.compile(model)
			print("Model compiled with torch.compile (inductor).", file=sys.stderr)
		except Exception as e:
			print(f"Warning: torch.compile failed; continuing without compile: {e}", file=sys.stderr)

	# ── Commented-out encoder freezing ─────────────────────────────────────
	# These blocks show previous experiments with:
	# 1. Freezing all encoder parameters (pure decoder fine-tuning)
	# 2. Differential learning rates for pretrained vs new parameters
	# Both are left for reference but currently all parameters are trained.
	#for param in model.transgenic.encoder.parameters():
	#	param.requires_grad = False

	#pretrained_params = []
	#new_params = []
	#for name, param in model.named_parameters():
	#	if name in new_keys.missing_keys:
	#		new_params.append(param)
	#	else:
	#		pretrained_params.append(param)

	# ── Optimizer ──────────────────────────────────────────────────────────
	# 8-bit AdamW: quantizes optimizer states (momentum, variance) to int8
	# Saves ~75% optimizer memory vs FP32 AdamW (~3.4 GB vs ~13.6 GB for 1.17B params)
	optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.02)
	# Alternative: differential LR for pretrained vs new parameters (commented out)
	#optimizer = optim.AdamW([
	#	{'params': pretrained_params, 'lr': lr / 2},
	#	{'params': new_params, 'lr': lr}
	#], weight_decay=0.01)
	optimizer.zero_grad()  # Initialize gradient buffers to zero

	# ── Learning Rate Schedule ─────────────────────────────────────────────
	# Compute total optimizer steps: (batches_per_epoch / accumulation_steps) * num_epochs
	steps_per_epoch = max(1, math.ceil(len(train_dl) / accumulation_steps))
	t_total = steps_per_epoch * num_epochs  # Total optimizer steps across all epochs
	lr_scheduler = None
	if schedule_lr:
		warmup_steps = int(t_total * 0.05)  # 5% warmup: LR ramps from 0 → lr
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=warmup_steps,     # Steps with linearly increasing LR
		num_training_steps=t_total         # Steps with linearly decreasing LR
		)
		print(f"LR schedule: {warmup_steps} warmup steps, {t_total} total steps", file=sys.stderr)

	# ── Accelerator Setup ──────────────────────────────────────────────────
	# Wraps model/optimizer/dataloaders for automatic mixed precision and distributed training
	if lr_scheduler is None:
		model, optimizer, train_dl, eval_dl = accelerator.prepare(model, optimizer, train_dl, eval_dl)
	else:
		model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
			model, optimizer, train_dl, eval_dl, lr_scheduler
		)

	# ── Resume from Checkpoint ─────────────────────────────────────────────
	# Accelerate checkpoints save full training state: model weights, optimizer state,
	# LR scheduler, RNG state, and dataloader position for deterministic resume.
	start_epoch = 0
	resume_step = 0                              # Micro-batch step to resume from (within start_epoch)
	global_step = 0                              # Total optimizer steps completed so far
	if resume_from_checkpoint is not None:
		accelerator.load_state(resume_from_checkpoint)  # Restore all training state
		meta_path = os.path.join(resume_from_checkpoint, "meta.json")
		if os.path.exists(meta_path):
			meta = _read_json(meta_path)
			start_epoch = int(meta.get("epoch", meta.get("next_epoch", 0)))
			resume_step = int(meta.get("step", 0))
			global_step = int(meta.get("global_step", 0))
		print(f"Resumed from {resume_from_checkpoint}; epoch={start_epoch}, step={resume_step}, global_step={global_step}", file=sys.stderr)

	def _save_training_state(epoch: int, step: int, global_step: int, best_score):
		"""Save a full Accelerate checkpoint for resumable training.

		Creates a directory with model weights, optimizer state, LR scheduler,
		RNG state, and a meta.json with epoch, step, and global_step.
		"""
		ckpt_dir = os.path.join(checkpoint_path, f"accelerate_epoch{epoch}_step{global_step}")
		os.makedirs(ckpt_dir, exist_ok=True)
		accelerator.save_state(ckpt_dir)  # Save full training state
		_write_json(
			os.path.join(ckpt_dir, "meta.json"),
			{
				"epoch": epoch,
				"step": step,
				"global_step": global_step,
				"best_eval_score": None if best_score is None else float(best_score),
			},
		)

	# ── Training Loop ──────────────────────────────────────────────────────
	best_eval_score = None  # Track best model for saving
	try:
		for epoch in range(start_epoch, num_epochs):
			total_loss = 0  # Accumulate loss across all batches in the epoch
			for step, batch in enumerate(tqdm(train_dl, miniters=10, disable=False)):
				# Skip already-processed steps when resuming mid-epoch
				if epoch == start_epoch and step < resume_step:
					continue

				# 'Zm' (maize) genes are already filtered at the dataset level via exclude_prefix
				ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)

				# Skip oversized batches that would cause OOM
				# Threshold: batch_size * seq_length > 100,000 for the ~100 GB GPU cap
				if ii.shape[0] * ii.shape[1] > 100_000:
					continue

				dii = None  # decoder_input_ids: None triggers automatic shift_tokens_right from labels
				try:
					outputs = None
					# Forward pass: encoder(DNA) → downsample → decoder(GFF) → cross-entropy loss
					outputs = model(input_ids=ii, attention_mask=am, decoder_input_ids=dii, labels=lab, return_dict=True)
					total_loss += outputs.loss.detach().float()  # Accumulate for epoch-level reporting

					# Scale loss by accumulation steps for correct gradient magnitude
					outputs.loss = outputs.loss / accumulation_steps
					# Backward pass: compute gradients (uses gradient checkpointing internally)
					accelerator.backward(outputs.loss)

					# Optimizer step: only update weights every accumulation_steps micro-batches
					if (step+1) % accumulation_steps == 0:
						global_step += 1
						# Clip gradients to prevent exploding gradient problem
						clip_grad_norm_(model.parameters(), max_grad_norm)
						optimizer.step()             # Apply 8-bit AdamW update
						if lr_scheduler is not None:
							lr_scheduler.step()      # Advance learning rate schedule

						# Log training metrics to Weights & Biases
						if log_wandb:
							wandb_log = {
								"epoch": epoch,
								"step": step,
								"global_step": global_step,
								"loss": outputs.loss.detach().float() * accumulation_steps,  # Unscaled loss
								"mean_loss": (total_loss) / (step + 1),  # Running mean loss
							}
							if lr_scheduler is not None:
								wandb_log["lr"] = lr_scheduler.get_last_lr()[0]  # Current learning rate
							# Log per-parameter gradient norms every 10 optimizer steps
							# (expensive: iterates all params, so don't do it every step)
							if global_step % 10 == 0:
								for name, param in model.named_parameters():
									if (param.grad != None) & (param.requires_grad):
										grad_norm = param.grad.norm().detach().item()
										wandb_log[f"{name}_grad_norm"] = grad_norm
							wandb.log(wandb_log)
						optimizer.zero_grad()  # Reset gradients for next accumulation window

						# Save full resumable checkpoint every save_every_n_steps optimizer steps
						if save_every_n_steps and global_step % save_every_n_steps == 0:
							_save_training_state(epoch, step + 1, global_step, best_eval_score)
							print(f"Checkpoint saved at epoch {epoch}, step {step+1}, global_step {global_step}", file=sys.stderr)

					del outputs  # Free GPU memory from forward pass outputs

				except Exception as e:
					# Handle OOM or other errors gracefully: skip the batch and continue
					print(f"Error in batch: {batch[3]}, {e}")
					optimizer.zero_grad()       # Clear any partial gradients
					model.zero_grad()           # Clear model-level gradient buffers
					del outputs                 # Free any allocated forward tensors
					torch.cuda.empty_cache()    # Return GPU memory to the allocator pool
					gc.collect()                # Force Python garbage collection
					time.sleep(1)               # Brief pause to let GPU memory settle
					continue

			# ── End of Epoch ───────────────────────────────────────────────
			train_epoch_loss = total_loss / len(train_dl)    # Average loss per batch
			train_ppl = torch.exp(train_epoch_loss)          # Perplexity = exp(loss)

			# Run evaluation if enabled
			if eval:
				eval_loss = 0
				for batch in tqdm(eval_dl, miniters=10, disable=False):
					with torch.no_grad():  # Disable gradient computation for eval speed + memory
						outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[2].to(device), return_dict=True)
					eval_loss += outputs.loss.detach().float()

				eval_epoch_loss = eval_loss / len(eval_dl)   # Average eval loss per batch
				eval_ppl = torch.exp(eval_epoch_loss)        # Eval perplexity
				print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
				if log_wandb:
					wandb_log = {"epoch_train_ppl": train_ppl, "epoch_train_loss": train_epoch_loss, "epoch_eval_ppl": eval_ppl, "epoch_eval_loss": eval_epoch_loss}
					wandb.log(wandb_log)
			else:
				print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)

			# Save best model based on eval loss (or train loss if eval is disabled)
			if eval:
				if best_eval_score is None or eval_epoch_loss < best_eval_score:
					best_eval_score = eval_epoch_loss
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
					print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
			else:
				if best_eval_score is None or train_epoch_loss < best_eval_score:
					best_eval_score = train_epoch_loss
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
					print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)

			# Save a full resumable checkpoint at every epoch boundary
			if save_every_epoch:
				_save_training_state(epoch + 1, 0, global_step, best_eval_score)

			# Clean up GPU memory between epochs
			torch.cuda.empty_cache()
			gc.collect()
			total_loss = 0            # Reset epoch loss accumulator
			train_epoch_loss = 0
			train_ppl = 0

	except KeyboardInterrupt:
		# Graceful shutdown: save checkpoint with epoch+step so training can resume mid-epoch
		print("KeyboardInterrupt: saving resume checkpoint...", file=sys.stderr)
		_save_training_state(
			epoch=epoch if 'epoch' in locals() else 0,
			step=step + 1 if 'step' in locals() else 0,
			global_step=global_step,
			best_score=best_eval_score,
		)
		raise

	# Finish W&B logging
	if log_wandb:
		wandb.finish()


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train HyenaTransgenic model (GB10 variant)")
	parser.add_argument("--db", type=str,
		default="/home/framazan/data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db",
		help="Path to DuckDB training database")
	parser.add_argument("--resume", type=str, default=None,
		help="Resume from checkpoint. 'auto' = find latest in checkpoint dir, or specify path")
	parser.add_argument("--batch-size", type=int, default=8,
		help="Micro-batch size per forward pass (default: 8)")
	parser.add_argument("--num-workers", type=int, default=4,
		help="Number of DataLoader workers (default: 4)")
	parser.add_argument("--accumulation-steps", type=int, default=32,
		help="Gradient accumulation steps (default: 32, effective batch = batch_size * this)")
	parser.add_argument("--save-every-n-steps", type=int, default=5000,
		help="Save full checkpoint every N optimizer steps (default: 5000)")
	parser.add_argument("--checkpoint-path", type=str, default="checkpoints_HyenaWide/",
		help="Checkpoint directory (default: checkpoints_HyenaWide/)")
	parser.add_argument("--no-wandb", action="store_true",
		help="Disable Weights & Biases logging")
	args = parser.parse_args()

	# Resolve --resume auto to the latest checkpoint
	resume_ckpt = None
	if args.resume:
		if args.resume == "auto":
			resume_ckpt = _find_latest_checkpoint(args.checkpoint_path)
			if resume_ckpt:
				print(f"Auto-detected checkpoint: {resume_ckpt}", file=sys.stderr)
			else:
				print("No checkpoint found for auto-resume; starting fresh.", file=sys.stderr)
		else:
			resume_ckpt = args.resume

	# Global random seed for dataset splitting reproducibility
	torch.manual_seed(123)

	# Load the full dataset, excluding maize (Zm) for cross-species evaluation
	db = args.db
	ds = isoformDataHyena(db, mode="train", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False, exclude_prefix="Zm")

	# Split dataset: 75% train, 10% eval, 15% test
	total = len(ds)
	train_size = int(total * 0.75)
	eval_size = int(total * 0.10)
	test_size = total - train_size - eval_size  # Remainder goes to test
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [train_size, eval_size, test_size])

	# Launch training with GB10-optimized hyperparameters
	trainTransgenicFCGAccelerate(
		train_data,
		eval_data,
		lr=5e-5,                  # Base learning rate for 8-bit AdamW
		num_epochs=10,            # Total training epochs
		schedule_lr=True,         # Enable warmup + linear decay schedule
		eval=True,                # Run evaluation after each epoch
		batch_size=args.batch_size,  # Micro-batch size
		accumulation_steps=args.accumulation_steps,  # Effective batch = batch_size * this
		checkpoint_path=args.checkpoint_path,  # Checkpoint directory
		output_dir="saved_models_HyenaWide/",  # Final model directory
		max_grad_norm=1,          # Gradient norm clipping threshold
		notes="HyenaTransgenic ~3x params (d_model=1152, layers=16, heads=8), training from scratch",
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",  # Pretrained encoder
		resume_from_checkpoint=resume_ckpt,
		save_every_n_steps=args.save_every_n_steps,
		num_workers=args.num_workers,
		unlink=False,             # Tie decoder embedding ↔ LM head weights
		log_wandb=not args.no_wandb  # Disable W&B if --no-wandb flag is set
	)
