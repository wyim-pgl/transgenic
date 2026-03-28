#!/usr/bin/env python
"""Optimized HyenaTransgenic training entrypoint for RTX 4090 24GB GPUs.

Changes vs the baseline trainer:
- Uses a RTX 4090 specific profile (4 sample micro-batches, 64 accumulation steps)
  to maintain a large effective batch (256) without exceeding 24GB VRAM.
- Enables pinned host memory, higher worker count, and a larger prefetch factor
  to keep the GPU saturated even though micro-batches are small.
- Automatically skips oversized batches (>48k tokens) that would trigger OOMs.
- Enables torch.compile in "reduce-overhead" mode (works well on 4090) for extra throughput.

Override anything via CLI flags if you need a different trade-off.
"""
from __future__ import annotations

# import argparse (removed)
import dataclasses
import gc
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import bitsandbytes as bnb
import torch
import wandb
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from transgenic.datasets.datasets import isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.tokenization_transgenic import GFFTokenizer

# Chr4 evaluation module (500-sample gffcompare after each epoch)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from testing_ch4 import evaluate_chr4


@dataclass
class TrainingProfile:
	"""Hyper-parameters tuned for a specific hardware target."""

	name: str
	batch_size: int
	accumulation_steps: int
	eval_batch_size: Optional[int]
	num_workers: int
	prefetch_factor: int
	pin_memory: bool
	max_tokens_per_batch: int
	attention_window: int
	snapshot_interval: int

	def dataloader_kwargs(self, *, is_eval: bool = False) -> Dict:
		batch_size = self.eval_batch_size if is_eval and self.eval_batch_size else self.batch_size
		kwargs = dict(
			batch_size=batch_size,
			shuffle=not is_eval,
			# Pin host memory so GPU DMA can overlap host->device copies with compute.
			pin_memory=self.pin_memory,
			num_workers=self.num_workers,
			persistent_workers=self.num_workers > 0,
			collate_fn=hyena_collate_fn,
		)
		if self.prefetch_factor and self.num_workers > 0:
			# Prefetch multiple batches per worker to hide DuckDB latency.
			kwargs["prefetch_factor"] = self.prefetch_factor
		return kwargs

PROFILE_PRESETS: Dict[str, TrainingProfile] = {
	"rtx4090": TrainingProfile(
		name="rtx4090",
		batch_size=1,
		accumulation_steps=128,
		eval_batch_size=1,
		num_workers=2,
		prefetch_factor=2,
		pin_memory=True,
		max_tokens_per_batch=90000,
		attention_window=1024,
		snapshot_interval=2500,
	),
}


def _count_parameters(model: torch.nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def _write_json(path: str, obj: dict):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _find_latest_checkpoint(checkpoint_path: str) -> Optional[str]:
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





# def _compile_model(model: torch.nn.Module, mode: str):
#     if mode == "none":
#         return model
#     mode_arg = None if mode == "default" else mode
#     try:
#         # reduce-overhead/max-autotune lower launch overhead on Ada GPUs.
#         model.lm_head = torch.compile(model.lm_head, mode=mode_arg)
#         model.transgenic.decoder = torch.compile(model.transgenic.decoder.lm_head, mode=mode_arg)
#         print(f"torch.compile enabled (mode={mode}).", file=sys.stderr)
#         return model
#     except Exception as exc:
#         print(f"Warning: torch.compile failed (mode={mode}): {exc}", file=sys.stderr)
#         return model


def train(
	train_ds,
	eval_ds,
	profile: TrainingProfile,
	*,
	lr: float,
	num_epochs: int,
	schedule_lr: bool,
	do_eval: bool,
	max_grad_norm: float,
	checkpoint_path: str,
	output_dir: str,
	encoder_model: str,
	notes: str,
	resume_from_checkpoint: Optional[str],
	save_every_epoch: bool,
	save_every_n_steps: int = 2000,
	unlink: bool,
	log_wandb: bool,
	train_sampler=None,
	train_dl_kwargs: Optional[Dict] = None,
):
	if torch.cuda.is_available():
		device_name = torch.cuda.get_device_name(0)
		total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
	else:
		device_name = "cpu"
		total_mem_gb = 0

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(123)
	torch.manual_seed(123)
	# cuDNN autotune squeezes extra throughput out of the 4090.
	torch.backends.cudnn.benchmark = True

	# Accelerator handles gradient accumulation + bf16 casting on our behalf.
	accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=profile.accumulation_steps)
	device = accelerator.device

	os.makedirs(checkpoint_path, exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)
	accelerator.wait_for_everyone()

	if log_wandb:
		wandb.init(
			entity="transgenic-paper",
			project="transgenic",
			config={
				"profile": profile.name,
				"learning_rate": lr,
				"schedule_lr": schedule_lr,
				"architecture": "Hyena",
				"dataset": "10G_static6144_addExtra200_addRCIsoOnly_clean",
				"epochs": num_epochs,
				"max_grad_norm": max_grad_norm,
				"accumulation_steps": profile.accumulation_steps,
				"batch_size": profile.batch_size,
				"effective_batch_size": profile.batch_size * profile.accumulation_steps,
				"optimizer": "AdamW8bit",
				"checkpoints": checkpoint_path,
				"outputs": output_dir,
				"notes": notes,
				"device": device_name,
				"device_mem_gb": total_mem_gb,
			},
		)

	# Build dl_kwargs from caller or profile. Train DL is rebuilt per-epoch
	# with a deterministic generator seed, so only store the kwargs here.
	if train_dl_kwargs is not None:
		dl_kwargs = dict(train_dl_kwargs)
	else:
		dl_kwargs = profile.dataloader_kwargs(is_eval=False)
	eval_dl_kwargs = profile.dataloader_kwargs(is_eval=True)

	# Remove sampler/shuffle from dl_kwargs – they are set per-epoch below.
	dl_kwargs.pop("sampler", None)

	eval_dl = makeDataLoader(eval_ds, **eval_dl_kwargs)

	# Build a temporary train_dl just to compute steps_per_epoch.
	_tmp_dl = makeDataLoader(train_ds, **{**dl_kwargs, "shuffle": False})
	steps_per_epoch = max(1, math.ceil(len(_tmp_dl) / profile.accumulation_steps))
	total_opt_steps = steps_per_epoch * num_epochs
	del _tmp_dl

	base_d_model = 1152
	layers = 16
	heads = 8
	config = HyenaTransgenicConfig(
		d_model=base_d_model,
		encoder_layers=layers,
		decoder_layers=layers,
		encoder_n_layer=layers,
		encoder_ffn_dim=base_d_model * 4,
		decoder_ffn_dim=base_d_model * 4,
		attention_window=[profile.attention_window] * layers,
		dropout=0.2,
		encoder_attention_heads=heads,
		decoder_attention_heads=heads,
		encoder_model=encoder_model,
		unlink=unlink,
	)
	model = transgenicForConditionalGeneration(config)
	model.gradient_checkpointing_enable()
	model = model.to(device)

	optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.02)
	optimizer.zero_grad(set_to_none=True)

	lr_scheduler = None
	if schedule_lr:
		warmup_steps = int(total_opt_steps * 0.05)
		lr_scheduler = get_linear_schedule_with_warmup(
			optimizer=optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=total_opt_steps,
		)
		if accelerator.is_main_process:
			print(f"LR schedule: {warmup_steps} warmup steps, {total_opt_steps} total", file=sys.stderr)

	# Prepare model, optimizer, eval_dl (and optionally lr_scheduler).
	# Train DataLoader is prepared each epoch inside the loop.
	if lr_scheduler is None:
		model, optimizer, eval_dl = accelerator.prepare(model, optimizer, eval_dl)
	else:
		model, optimizer, eval_dl, lr_scheduler = accelerator.prepare(
			model, optimizer, eval_dl, lr_scheduler
		)

	start_epoch = 0
	resume_step = 0                             # Micro-batch step to resume from (within start_epoch)
	global_step = 0                             # Total optimizer steps completed so far
	best_eval_score: Optional[float] = None
	# Running-loss accumulators restored from checkpoint for correct epoch-level metrics.
	resume_loss_sum = 0.0
	resume_processed = 0
	resume_skipped = 0
	if resume_from_checkpoint is not None:
		resume_from_checkpoint = os.path.abspath(resume_from_checkpoint)
		is_best_model_dir = os.path.basename(os.path.normpath(resume_from_checkpoint)) == "best_model"

		if is_best_model_dir:
			model_path = os.path.join(resume_from_checkpoint, "model.safetensors")
			state_path = os.path.join(resume_from_checkpoint, "training_state.json")
			if not os.path.exists(model_path):
				raise FileNotFoundError(f"Best-model weights not found: {model_path}")

			unwrapped_model = accelerator.unwrap_model(model)
			model_state = load_file(model_path, device="cpu")
			load_result = unwrapped_model.load_state_dict(model_state, strict=False)

			if accelerator.is_main_process and (load_result.missing_keys or load_result.unexpected_keys):
				print(
					f"Best-model resume loaded with missing={len(load_result.missing_keys)} "
					f"unexpected={len(load_result.unexpected_keys)}",
					file=sys.stderr,
				)

			# Try to restore optimizer and scheduler states if available
			if os.path.exists(state_path):
				meta = _read_json(state_path)
				start_epoch = int(meta.get("epoch", 0))
				resume_step = int(meta.get("step", 0))
				global_step = int(meta.get("global_step", 0))
				best = meta.get("best_eval_score")
				best_eval_score = None if best is None else float(best)
				# Attempt to restore optimizer and scheduler state
				try:
					accelerator.load_state(resume_from_checkpoint)
				except Exception as e:
					print(f"Warning: Could not restore optimizer/scheduler state from best_model: {e}", file=sys.stderr)
			elif accelerator.is_main_process:
				print("Warning: training_state.json not found in best_model; resuming with global_step=0.", file=sys.stderr)
		else:
			accelerator.load_state(resume_from_checkpoint)
			meta_path = os.path.join(resume_from_checkpoint, "meta.json")
			if os.path.exists(meta_path):
				meta = _read_json(meta_path)
				start_epoch = int(meta.get("epoch", meta.get("next_epoch", 0)))
				resume_step = int(meta.get("step", 0))
				global_step = int(meta.get("global_step", 0))
				best = meta.get("best_eval_score")
				best_eval_score = None if best is None else float(best)
				# Restore running loss accumulators so epoch mean is correct.
				resume_loss_sum = float(meta.get("running_loss_sum", 0.0))
				resume_processed = int(meta.get("processed_batches", 0))
				resume_skipped = int(meta.get("skipped_batches", 0))

		# Ensure scheduler resumes at the saved optimization step (no warmup restart).
		if lr_scheduler is not None and global_step > 0:
			lr_scheduler.step(min(global_step, total_opt_steps))

		accelerator.wait_for_everyone()
		print(f"Resumed from {resume_from_checkpoint}; epoch={start_epoch}, step={resume_step}, global_step={global_step}", file=sys.stderr)

	def _save_state(epoch: int, step: int, global_step: int, best_score: Optional[float],
	                running_loss_sum: float = 0.0, processed_batches: int = 0,
	                skipped_batches: int = 0):
		ckpt_dir = os.path.join(checkpoint_path, f"accelerate_epoch{epoch}_step{global_step}")
		accelerator.save_state(ckpt_dir)
		# Save decoder tokenizer at checkpoint root so AutoTokenizer.from_pretrained(ckpt_dir)
		# loads the GFF tokenizer used to decode model outputs.
		try:
			gff_tokenizer = GFFTokenizer()
			gff_tokenizer.save_pretrained(ckpt_dir)
		except Exception as e:
			print(f"Warning: Could not save decoder tokenizer with checkpoint: {e}", file=sys.stderr)

		# Save encoder tokenizer in a dedicated subdirectory to avoid overriding decoder tokenizer files.
		try:
			tokenizer = AutoTokenizer.from_pretrained(encoder_model, trust_remote_code=True)
			encoder_tok_dir = os.path.join(ckpt_dir, "encoder_tokenizer")
			os.makedirs(encoder_tok_dir, exist_ok=True)
			tokenizer.save_pretrained(encoder_tok_dir)
			# Copy extra tokenizer files if present and non-empty
			import shutil
			tokenizer_dir = None
			if hasattr(tokenizer, 'vocab_files_names') and hasattr(tokenizer, 'pretrained_vocab_files_map'):
				# Try to get the original directory
				vocab_map = list(tokenizer.pretrained_vocab_files_map.values())
				if vocab_map and isinstance(vocab_map[0], dict):
					# Get the first file path
					first_path = list(vocab_map[0].values())[0]
					if first_path:
						tokenizer_dir = os.path.dirname(first_path)
			if not tokenizer_dir:
				# Fallback: try encoder_model path
				tokenizer_dir = encoder_model if os.path.isdir(encoder_model) else None
			extra_files = ["tokenizer.json", "added_tokens.json", "chat_template.jinja"]
			if tokenizer_dir:
				for fname in extra_files:
					src = os.path.join(tokenizer_dir, fname)
					dst = os.path.join(encoder_tok_dir, fname)
					if os.path.isfile(src):
						try:
							if os.path.getsize(src) > 0:
								shutil.copy2(src, dst)
						except Exception as e:
							print(f"Warning: Could not copy {fname} to checkpoint: {e}", file=sys.stderr)
		except Exception as e:
			print(f"Warning: Could not save encoder tokenizer with checkpoint: {e}", file=sys.stderr)
		# Save model config as config.json for HF compatibility
		try:
			if hasattr(model, 'config'):
				model.config.save_pretrained(ckpt_dir)
		except Exception as e:
			print(f"Warning: Could not save config.json with checkpoint: {e}", file=sys.stderr)
		_write_json(
			os.path.join(ckpt_dir, "meta.json"),
			{
				"epoch": epoch,
				"step": step,
				"global_step": global_step,
				"lr": float(optimizer.param_groups[0]["lr"]),
				"best_eval_score": None if best_score is None else float(best_score),
				"running_loss_sum": float(running_loss_sum),
				"processed_batches": int(processed_batches),
				"skipped_batches": int(skipped_batches),
				"epoch_seed": 123 + epoch,
			},
		)

	def _save_best(tag: str, value: float):
		accelerator.wait_for_everyone()
		if accelerator.is_main_process:
			best_dir = os.path.join(checkpoint_path, "best_model")
			os.makedirs(best_dir, exist_ok=True)

			unwrapped_model = accelerator.unwrap_model(model)
			# Export model + config in standard HF format for easy inference loading.
			unwrapped_model.save_pretrained(best_dir, safe_serialization=False)

			# Export decoder tokenizer at root and encoder tokenizer in a subdirectory.
			gff_tokenizer = GFFTokenizer()
			gff_tokenizer.save_pretrained(best_dir)
			tokenizer = AutoTokenizer.from_pretrained(encoder_model, trust_remote_code=True)
			encoder_tok_dir = os.path.join(best_dir, "encoder_tokenizer")
			os.makedirs(encoder_tok_dir, exist_ok=True)
			tokenizer.save_pretrained(encoder_tok_dir)

			# Save resume metadata so LR/global_step can be restored from best_model.
			_write_json(
				os.path.join(best_dir, "training_state.json"),
				{
					"epoch": int(epoch),
					"step": 0,
					"global_step": int(global_step),
					"lr": float(optimizer.param_groups[0]["lr"]),
					"best_eval_score": None if best_eval_score is None else float(best_eval_score),
					"tag": tag,
					"value": float(value),
				},
			)

		accelerator.wait_for_everyone()
		print(f"New best ({tag}={value:.4f}) saved to {checkpoint_path}/best_model.", file=sys.stderr)

	print(
		f"Using device={device} ({device_name}, {total_mem_gb:.1f}GB) | profile={profile.name} | "
		f"micro_batch={profile.batch_size} accumulation={profile.accumulation_steps} "
		f"(effective={profile.batch_size * profile.accumulation_steps})",
		file=sys.stderr,
	)
	print(f"Model params: {_count_parameters(model):,}", file=sys.stderr)

	try:
		for epoch in range(start_epoch, num_epochs):
			model.train()

			# ── Build a fresh DataLoader with a deterministic per-epoch seed ──
			# This guarantees the shuffle order for epoch E is ALWAYS the same,
			# regardless of whether we are starting fresh or resuming mid-epoch.
			g = torch.Generator()
			g.manual_seed(123 + epoch)

			if train_sampler is not None:
				# Recreate the WeightedRandomSampler with the epoch-specific generator
				# so the same sequence of draws is produced on resume.
				epoch_sampler = torch.utils.data.WeightedRandomSampler(
					weights=train_sampler.weights,
					num_samples=train_sampler.num_samples,
					replacement=train_sampler.replacement,
					generator=g,
				)
				epoch_dl_kwargs = {**dl_kwargs, "sampler": epoch_sampler, "shuffle": False}
			else:
				epoch_dl_kwargs = {**dl_kwargs, "generator": g}

			train_dl_raw = makeDataLoader(train_ds, **epoch_dl_kwargs)
			train_dl = accelerator.prepare(train_dl_raw)

			# ── Handle mid-epoch resume via accelerator.skip_first_batches ──
			# Because the generator seed is deterministic, the batch order is
			# identical to the original run.  skip_first_batches fast-forwards
			# through exactly the batches the model already trained on.
			start_step_offset = 0
			if epoch == start_epoch and resume_step > 0:
				print(f"Skipping first {resume_step} batches (deterministic replay)...", file=sys.stderr)
				active_dl = accelerator.skip_first_batches(train_dl, resume_step)
				start_step_offset = resume_step
				# Restore running-loss accumulators from checkpoint so epoch mean is correct.
				total_loss = resume_loss_sum
				processed_batches = resume_processed
				skipped_batches = resume_skipped
				print(f"Restored running loss: sum={total_loss:.4f}, processed={processed_batches}, skipped={skipped_batches}", file=sys.stderr)
			else:
				active_dl = train_dl
				total_loss = 0.0
				processed_batches = 0
				skipped_batches = 0

			progress = tqdm(active_dl, leave=False, miniters=10)
			for step, batch in enumerate(progress, start=start_step_offset):
				ii, am, lab, *meta = batch
				# Extract transcript counts for regression head if available
				# (last element of batch from hyena_collate_fn when in training mode)
				tx_counts = None
				if len(meta) >= 5 and meta[-1] is not None and isinstance(meta[-1], torch.Tensor):
					tx_counts = meta[-1]
				tokens = ii.shape[0] * ii.shape[1]
				if profile.max_tokens_per_batch and tokens > profile.max_tokens_per_batch:
					# Drop pathological long batches instead of crashing with CUDA OOM.
					skipped_batches += 1
					continue

				processed_batches += 1
				
				# # Mark the beginning of a step for CUDA Graphs to prevent buffer overwriting errors
				# torch.compiler.cudagraph_mark_step_begin()

				with accelerator.accumulate(model):
					outputs = model(input_ids=ii, attention_mask=am, labels=lab,
					                transcript_counts=tx_counts, return_dict=True)
					loss_value = float(outputs.loss.detach())
					total_loss += loss_value
					accelerator.backward(outputs.loss)

					if accelerator.sync_gradients:
						global_step += 1
						# Only clip/step when gradients from all micro-batches are in sync.
						accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
						optimizer.step()
						if lr_scheduler is not None:
							lr_scheduler.step()
						# Save full resumable checkpoint every save_every_n_steps optimizer steps
						if save_every_n_steps and global_step % save_every_n_steps == 0:
							_save_state(epoch, step + 1, global_step, best_eval_score,
							            running_loss_sum=total_loss,
							            processed_batches=processed_batches,
							            skipped_batches=skipped_batches)
							print(f"Checkpoint saved at epoch {epoch}, step {step+1}, global_step {global_step}", file=sys.stderr)

					if log_wandb:
						log_payload = {}
						if step % 70 == 0:
							log_payload = {
								"epoch": epoch,
								"step": step,
								"global_step": global_step,
								"loss": loss_value,
								"mean_loss": total_loss / processed_batches,
								"skipped_batches": skipped_batches,
							}
							if lr_scheduler is not None:
								log_payload["lr"] = lr_scheduler.get_last_lr()[0]
						
						# Only log grad norms if grads are available (i.e., after optimizer step)
						if accelerator.sync_gradients:
							for name, param in model.named_parameters():
								if param.grad is not None and param.requires_grad:
									log_payload[f"{name}_grad_norm"] = param.grad.norm().detach().item()
						
						if log_payload:
							wandb.log(log_payload)

					if accelerator.sync_gradients:
						optimizer.zero_grad(set_to_none=True)
						torch.cuda.empty_cache()
						gc.collect()
				del outputs, ii, am, lab, meta

			# Free the per-epoch DataLoader to release memory before eval.
			del train_dl, train_dl_raw, active_dl, progress
			if processed_batches == 0:
				raise RuntimeError("No batches processed – lower batch_size or max_tokens_per_batch.")

			train_epoch_loss = total_loss / processed_batches
			train_ppl = math.exp(train_epoch_loss)

			if do_eval:
				model.eval()
				eval_loss_sum = 0.0
				eval_batches = 0
				for batch in tqdm(eval_dl, leave=False):
					# Evaluation runs in no_grad to maximize throughput and cut VRAM in half.
					ii, am, lab, *_ = batch
					with torch.no_grad():
						outputs = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
					eval_loss_sum += float(outputs.loss.detach())
					eval_batches += 1
					del outputs
				eval_epoch_loss = eval_loss_sum / max(1, eval_batches)
				eval_ppl = math.exp(eval_epoch_loss)

				print(
					f"epoch={epoch}: train_loss={train_epoch_loss:.4f} train_ppl={train_ppl:.2f} "
					f"eval_loss={eval_epoch_loss:.4f} eval_ppl={eval_ppl:.2f} (skipped={skipped_batches})",
					file=sys.stderr,
				)
				if log_wandb:
					wandb.log(
						{
							"epoch": epoch,
							"epoch_train_loss": train_epoch_loss,
							"epoch_train_ppl": train_ppl,
							"epoch_eval_loss": eval_epoch_loss,
							"epoch_eval_ppl": eval_ppl,
						}
					)
				if best_eval_score is None or eval_epoch_loss < best_eval_score:
					best_eval_score = eval_epoch_loss
					_save_best("eval_loss", eval_epoch_loss)
			else:
				print(
					f"epoch={epoch}: train_loss={train_epoch_loss:.4f} train_ppl={train_ppl:.2f} (skipped={skipped_batches})",
					file=sys.stderr,
				)
				if log_wandb:
					wandb.log(
						{
							"epoch": epoch,
							"epoch_train_loss": train_epoch_loss,
							"epoch_train_ppl": train_ppl,
						}
					)
				if best_eval_score is None or train_epoch_loss < best_eval_score:
					best_eval_score = train_epoch_loss
					_save_best("train_loss", train_epoch_loss)

			if save_every_epoch:
				_save_state(epoch + 1, 0, global_step, best_eval_score,
				            running_loss_sum=0.0, processed_batches=0,
				            skipped_batches=0)

			# ── Chr4 gffcompare evaluation (500 random gene regions) ─────────
			try:
				unwrapped = accelerator.unwrap_model(model)
				chr4_metrics = evaluate_chr4(
					unwrapped, device,
					n_samples=500,
					batch_size=2,
					max_gen_len=2048,
					seed=16,
					verbose=True,
				)
				print(
					f"  Chr4 eval: tx_sens={chr4_metrics.get('transcript_sensitivity', 'N/A')}"
					f"  tx_prec={chr4_metrics.get('transcript_precision', 'N/A')}"
					f"  exon_sens={chr4_metrics.get('exon_sensitivity', 'N/A')}"
					f"  exon_prec={chr4_metrics.get('exon_precision', 'N/A')}"
					f"  isoforms={chr4_metrics.get('isoforms_predicted', 'N/A')}"
					f"  tx/locus={chr4_metrics.get('transcripts_per_locus', 'N/A')}",
					file=sys.stderr,
				)
				if log_wandb:
					wandb.log({f"chr4/{k}": v for k, v in chr4_metrics.items()})
			except Exception as e:
				print(f"Warning: Chr4 evaluation failed: {e}", file=sys.stderr)
			finally:
				model.train()  # Restore training mode

			torch.cuda.empty_cache()
			gc.collect()

	except KeyboardInterrupt:
		print("KeyboardInterrupt: saving for resume...", file=sys.stderr)
		# Ensure all variables are defined and valid
		try:
			safe_epoch = epoch if 'epoch' in locals() and isinstance(epoch, int) else 0
		except Exception:
			safe_epoch = 0
		try:
			safe_step = step + 1 if 'step' in locals() and isinstance(step, int) else 0
		except Exception:
			safe_step = 0
		try:
			safe_global_step = global_step if 'global_step' in locals() and isinstance(global_step, int) else 0
		except Exception:
			safe_global_step = 0
		try:
			safe_best_score = best_eval_score if 'best_eval_score' in locals() else None
		except Exception:
			safe_best_score = None
		try:
			safe_loss_sum = total_loss if 'total_loss' in locals() else 0.0
		except Exception:
			safe_loss_sum = 0.0
		try:
			safe_proc = processed_batches if 'processed_batches' in locals() else 0
		except Exception:
			safe_proc = 0
		try:
			safe_skip = skipped_batches if 'skipped_batches' in locals() else 0
		except Exception:
			safe_skip = 0
		try:
			_save_state(
				epoch=safe_epoch,
				step=safe_step,
				global_step=safe_global_step,
				best_score=safe_best_score,
				running_loss_sum=safe_loss_sum,
				processed_batches=safe_proc,
				skipped_batches=safe_skip,
			)
			print(f"Checkpoint saved at epoch {safe_epoch}, step {safe_step}, global_step {safe_global_step}", file=sys.stderr)
		except Exception as e:
			print(f"ERROR: Failed to save checkpoint on KeyboardInterrupt: {e}", file=sys.stderr)
		raise
	finally:
		if log_wandb:
			wandb.finish()


if __name__ == "__main__":
	# All configuration is now set via the variable list below.
	# Modify these values directly to change training behavior.
	config = {
		"db": "/home/framazan/data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db",
		"profile": "rtx4090",
		"batch_size": 1,
		"accumulation_steps": 128,
		"attention_window": 1024,
		"max_tokens_per_batch": 90000,
		"num_workers": 2,
		"prefetch_factor": 4,
		"checkpoint_path": "/home/framazan/checkpoints_new/",
		"output_dir": "/home/framazan/saved_models/",
		"epochs": 7,
		"lr": 2e-5,
		"schedule_lr": True,
		"do_eval": True,
		"max_grad_norm": 1.0,
		"notes": "Transgenic RTX 4090 run with isoform tokne",
		"encoder_model": "LongSafari/hyenadna-large-1m-seqlen-hf",
		"resume": "",  # or "auto" or checkpoint path
		"save_every_n_steps": 100,
		"save_every_epoch": True,
		"unlink": False,
		"log_wandb": True,

	}

	# Set up the profile using config values
	base_profile = PROFILE_PRESETS[config["profile"]]
	profile = dataclasses.replace(base_profile,
		batch_size=config["batch_size"],
		accumulation_steps=config["accumulation_steps"],
		attention_window=config["attention_window"],
		max_tokens_per_batch=config["max_tokens_per_batch"],
		num_workers=config["num_workers"],
		prefetch_factor=config["prefetch_factor"],
	)

	# Handle resume logic
	resume_ckpt = None
	if config["resume"]:
		if config["resume"] == "auto":
			resume_ckpt = _find_latest_checkpoint(config["checkpoint_path"])
			if resume_ckpt:
				print(f"Auto-detected checkpoint: {resume_ckpt}", file=sys.stderr)
			else:
				print("No checkpoint found for auto-resume; starting fresh.", file=sys.stderr)
		else:
			resume_ckpt = config["resume"]

	torch.manual_seed(123)

	dataset = isoformDataHyena(
		config["db"],
		mode="train",
		encoder_model=config["encoder_model"],
		global_attention=False,
		exclude_prefix="Zm",
	)

	total = len(dataset)
	train_size = int(total * 0.75)
	eval_size = int(total * 0.10)
	test_size = total - train_size - eval_size
	train_data, eval_data, _ = torch.utils.data.random_split(dataset, [train_size, eval_size, test_size])

	# Build isoform-rebalanced sampler for training if weights are available.
	# Multi-isoform genes get sqrt(isoform_count) weight boost to gently
	# upsample them during training, improving alternative transcript discovery.
	train_sampler = None
	sample_weights = dataset.get_sample_weights()
	if sample_weights is not None:
		# Extract weights for the training subset indices
		train_weights = [sample_weights[i] for i in train_data.indices]
		train_sampler = torch.utils.data.WeightedRandomSampler(
			weights=train_weights,
			num_samples=len(train_weights),
			replacement=True,
		)
		print(f"Isoform rebalancing enabled: {sum(1 for w in train_weights if w > 1.0)} multi-isoform samples upsampled", file=sys.stderr)

	train(
		train_ds=train_data,
		eval_ds=eval_data,
		profile=profile,
		lr=config["lr"],
		num_epochs=config["epochs"],
		schedule_lr=config["schedule_lr"],
		do_eval=config["do_eval"],
		max_grad_norm=config["max_grad_norm"],
		checkpoint_path=config["checkpoint_path"],
		output_dir=config["output_dir"],
		encoder_model=config["encoder_model"],
		notes=config["notes"],
		resume_from_checkpoint=resume_ckpt,
		save_every_epoch=config["save_every_epoch"],
		save_every_n_steps=config["save_every_n_steps"],
		unlink=config["unlink"],
		log_wandb=config["log_wandb"],
		train_sampler=train_sampler,
		train_dl_kwargs=profile.dataloader_kwargs(is_eval=False),
	)

