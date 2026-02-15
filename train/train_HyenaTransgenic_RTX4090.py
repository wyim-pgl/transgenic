#!/usr/bin/env python
"""Optimized HyenaTransgenic training entrypoint for RTX 4090 24GB GPUs.

Changes vs the baseline trainer:
- Uses a RTX 4090 specific profile (4 sample micro-batches, 64 accumulation steps)
  to maintain a large effective batch (256) without exceeding 24GB VRAM.
- Shrinks the Longformer attention window to 768 by default, reducing decoder
  memory/compute ~25% while keeping receptive field >= GSF seqlens.
- Enables pinned host memory, higher worker count, and a larger prefetch factor
  to keep the GPU saturated even though micro-batches are small.
- Automatically skips oversized batches (>48k tokens) that would trigger OOMs.
- Enables torch.compile in "reduce-overhead" mode (works well on 4090) and TF32
  math across matmul + cuDNN kernels for extra throughput.

Override anything via CLI flags if you need a different trade-off.
"""
from __future__ import annotations

import argparse
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
from safetensors.torch import save_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from transgenic.datasets.datasets import isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration


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
    pin_memory_device: str
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
            pin_memory_device=self.pin_memory_device if self.pin_memory else "cpu",
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
        batch_size=4,
        accumulation_steps=64,
        eval_batch_size=4,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
        pin_memory_device="cuda",
        max_tokens_per_batch=48_000,
        attention_window=768,
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


def _configure_tf32(enabled: bool):
    torch.set_float32_matmul_precision('high')
    # Allow TF32 kernels (tensor cores) for matmuls + convolutions when latency matters.
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def _compile_model(model: torch.nn.Module, mode: str):
    if mode == "none":
        return model
    mode_arg = None if mode == "default" else mode
    try:
        # reduce-overhead/max-autotune lower launch overhead on Ada GPUs.
        compiled = torch.compile(model, mode=mode_arg)
        print(f"torch.compile enabled (mode={mode}).", file=sys.stderr)
        return compiled
    except Exception as exc:
        print(f"Warning: torch.compile failed (mode={mode}): {exc}", file=sys.stderr)
        return model


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
    save_every_n_steps: int = 5000,
    unlink: bool,
    log_wandb: bool,
    compile_mode: str,
    tf32: bool,
):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        device_name = "cpu"
        total_mem_gb = 0

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(234)
    torch.manual_seed(234)
    # TF32 + cuDNN autotune squeeze extra throughput out of the 4090.
    _configure_tf32(tf32)
    torch.backends.cudnn.benchmark = True

    # Accelerator handles gradient accumulation + bf16 casting on our behalf.
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=profile.accumulation_steps)
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if log_wandb and accelerator.is_main_process:
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

    dl_kwargs = profile.dataloader_kwargs(is_eval=False)
    eval_dl_kwargs = profile.dataloader_kwargs(is_eval=True)

    train_dl = makeDataLoader(train_ds, **dl_kwargs)
    eval_dl = makeDataLoader(eval_ds, **eval_dl_kwargs)

    steps_per_epoch = max(1, math.ceil(len(train_dl) / profile.accumulation_steps))
    total_opt_steps = steps_per_epoch * num_epochs

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
        dropout=0.1,
        encoder_attention_heads=heads,
        decoder_attention_heads=heads,
        encoder_model=encoder_model,
        unlink=unlink,
    )
    model = transgenicForConditionalGeneration(config)
    model.gradient_checkpointing_enable()
    # torch.compile fuses Hyena/Longformer kernels where possible.
    model = _compile_model(model, compile_mode)

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

    if lr_scheduler is None:
        model, optimizer, train_dl, eval_dl = accelerator.prepare(model, optimizer, train_dl, eval_dl)
    else:
        model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, eval_dl, lr_scheduler
        )

    start_epoch = 0
    resume_step = 0                             # Micro-batch step to resume from (within start_epoch)
    global_step = 0                             # Total optimizer steps completed so far
    best_eval_score: Optional[float] = None
    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)
        meta_path = os.path.join(resume_from_checkpoint, "meta.json")
        if accelerator.is_main_process and os.path.exists(meta_path):
            meta = _read_json(meta_path)
            start_epoch = int(meta.get("epoch", meta.get("next_epoch", 0)))
            resume_step = int(meta.get("step", 0))
            global_step = int(meta.get("global_step", 0))
            best = meta.get("best_eval_score")
            best_eval_score = None if best is None else float(best)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Resumed from {resume_from_checkpoint}; epoch={start_epoch}, step={resume_step}, global_step={global_step}", file=sys.stderr)

    def _save_state(epoch: int, step: int, global_step: int, best_score: Optional[float]):
        ckpt_dir = os.path.join(checkpoint_path, f"accelerate_epoch{epoch}_step{global_step}")
        accelerator.save_state(ckpt_dir)
        if accelerator.is_main_process:
            _write_json(
                os.path.join(ckpt_dir, "meta.json"),
                {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "best_eval_score": None if best_score is None else float(best_score),
                },
            )

    def _save_best(tag: str, value: float):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
            print(f"New best ({tag}={value:.4f}) saved.", file=sys.stderr)

    if accelerator.is_main_process:
        print(
            f"Using device={device} ({device_name}, {total_mem_gb:.1f}GB) | profile={profile.name} | "
            f"micro_batch={profile.batch_size} accumulation={profile.accumulation_steps} "
            f"(effective={profile.batch_size * profile.accumulation_steps})",
            file=sys.stderr,
        )
        print(f"Model params: {_count_parameters(accelerator.unwrap_model(model)):,}", file=sys.stderr)

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0.0
            processed_batches = 0
            skipped_batches = 0

            progress = tqdm(train_dl, disable=not accelerator.is_main_process, leave=False)
            for step, batch in enumerate(progress):
                # Skip already-processed steps when resuming mid-epoch
                if epoch == start_epoch and step < resume_step:
                    continue

                ii, am, lab, *meta = batch
                tokens = ii.shape[0] * ii.shape[1]
                if profile.max_tokens_per_batch and tokens > profile.max_tokens_per_batch:
                    # Drop pathological long batches instead of crashing with CUDA OOM.
                    skipped_batches += 1
                    continue

                processed_batches += 1

                with accelerator.accumulate(model):
                    outputs = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
                    loss_value = float(outputs.loss.detach())
                    total_loss += loss_value
                    accelerator.backward(outputs.loss)

                    if accelerator.sync_gradients:
                        global_step += 1
                        # Only clip/step when gradients from all micro-batches are in sync.
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        if log_wandb and accelerator.is_main_process:
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
                            # Log per-parameter gradient norms every 10 optimizer steps
                            if global_step % 10 == 0:
                                for name, param in model.named_parameters():
                                    if param.grad is not None and param.requires_grad:
                                        log_payload[f"{name}_grad_norm"] = param.grad.norm().detach().item()
                            wandb.log(log_payload)

                        # Save full resumable checkpoint every save_every_n_steps optimizer steps
                        if save_every_n_steps and global_step % save_every_n_steps == 0:
                            _save_state(epoch, step + 1, global_step, best_eval_score)
                            if accelerator.is_main_process:
                                print(f"Checkpoint saved at epoch {epoch}, step {step+1}, global_step {global_step}", file=sys.stderr)

                del outputs

            if processed_batches == 0:
                raise RuntimeError("No batches processed – lower batch_size or max_tokens_per_batch.")

            train_epoch_loss = total_loss / processed_batches
            train_ppl = math.exp(train_epoch_loss)

            if do_eval:
                model.eval()
                eval_loss_sum = 0.0
                eval_batches = 0
                for batch in tqdm(eval_dl, disable=not accelerator.is_main_process, leave=False):
                    # Evaluation runs in no_grad to maximize throughput and cut VRAM in half.
                    ii, am, lab, *_ = batch
                    with torch.no_grad():
                        outputs = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
                    eval_loss_sum += float(outputs.loss.detach())
                    eval_batches += 1
                    del outputs
                eval_epoch_loss = eval_loss_sum / max(1, eval_batches)
                eval_ppl = math.exp(eval_epoch_loss)

                if accelerator.is_main_process:
                    print(
                        f"epoch={epoch}: train_loss={train_epoch_loss:.4f} train_ppl={train_ppl:.2f} "
                        f"eval_loss={eval_epoch_loss:.4f} eval_ppl={eval_ppl:.2f} (skipped={skipped_batches})",
                        file=sys.stderr,
                    )
                if log_wandb and accelerator.is_main_process:
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
                if accelerator.is_main_process:
                    print(
                        f"epoch={epoch}: train_loss={train_epoch_loss:.4f} train_ppl={train_ppl:.2f} (skipped={skipped_batches})",
                        file=sys.stderr,
                    )
                if log_wandb and accelerator.is_main_process:
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
                _save_state(epoch + 1, 0, global_step, best_eval_score)

            torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("KeyboardInterrupt: saving for resume...", file=sys.stderr)
        _save_state(
            epoch=epoch if 'epoch' in locals() else 0,
            step=step + 1 if 'step' in locals() else 0,
            global_step=global_step,
            best_score=best_eval_score,
        )
        raise
    finally:
        if log_wandb and accelerator.is_main_process:
            wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser(description="RTX 4090-optimized HyenaTransgenic training")
    parser.add_argument("--db", type=str, required=True, help="Path to DuckDB training database")
    parser.add_argument("--profile", choices=PROFILE_PRESETS.keys(), default="rtx4090")
    parser.add_argument("--batch-size", type=int, help="Override micro-batch size")
    parser.add_argument("--accumulation-steps", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--attention-window", type=int, help="Override Longformer sliding window size")
    parser.add_argument("--max-tokens-per-batch", type=int, help="Skip batches larger than this token budget")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader worker count")
    parser.add_argument("--prefetch-factor", type=int, help="Override DataLoader prefetch factor (workers>0)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints_rtx4090/")
    parser.add_argument("--output-dir", type=str, default="saved_models_rtx4090/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-schedule", action="store_true", help="Disable LR warmup+linear decay")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation to maximize throughput")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--notes", type=str, default="HyenaTransgenic RTX 4090 run")
    parser.add_argument("--encoder-model", type=str, default="LongSafari/hyenadna-large-1m-seqlen-hf")
    parser.add_argument("--resume", type=str, default=None,
        help="Resume from checkpoint. 'auto' = find latest in checkpoint dir, or specify path")
    parser.add_argument("--save-every-n-steps", type=int, default=5000,
        help="Save full checkpoint every N optimizer steps (default: 5000)")
    parser.add_argument("--no-save-every-epoch", action="store_true")
    parser.add_argument("--unlink", action="store_true", help="Do not share encoder/decoder embeddings")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cuDNN kernels")
    return parser.parse_args()


def override_profile(base: TrainingProfile, args) -> TrainingProfile:
    profile = dataclasses.replace(base)
    if args.batch_size:
        profile.batch_size = args.batch_size
    if args.accumulation_steps:
        profile.accumulation_steps = args.accumulation_steps
    if args.attention_window:
        profile.attention_window = args.attention_window
    if args.max_tokens_per_batch:
        profile.max_tokens_per_batch = args.max_tokens_per_batch
    if args.num_workers is not None:
        profile.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        profile.prefetch_factor = args.prefetch_factor
    return profile


if __name__ == "__main__":
    args = parse_args()
    base_profile = PROFILE_PRESETS[args.profile]
    profile = override_profile(base_profile, args)

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

    torch.manual_seed(123)

    dataset = isoformDataHyena(
        args.db,
        mode="train",
        encoder_model=args.encoder_model,
        global_attention=False,
        exclude_prefix="Zm",
    )

    total = len(dataset)
    train_size = int(total * 0.75)
    eval_size = int(total * 0.10)
    test_size = total - train_size - eval_size
    train_data, eval_data, _ = torch.utils.data.random_split(dataset, [train_size, eval_size, test_size])

    train(
        train_ds=train_data,
        eval_ds=eval_data,
        profile=profile,
        lr=args.lr,
        num_epochs=args.epochs,
        schedule_lr=not args.no_schedule,
        do_eval=not args.no_eval,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        encoder_model=args.encoder_model,
        notes=args.notes,
        resume_from_checkpoint=resume_ckpt,
        save_every_epoch=not args.no_save_every_epoch,
        save_every_n_steps=args.save_every_n_steps,
        unlink=args.unlink,
        log_wandb=not args.no_wandb,
        compile_mode=args.compile_mode,
        tf32=not args.no_tf32,
    )
