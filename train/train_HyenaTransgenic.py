#!/usr/bin/env python
"""
Train HyenaTransgenic: a seq2seq model with HyenaDNA encoder + Longformer decoder.

Input:  DNA sequences (tokenized by HyenaDNA's character-level tokenizer)
Output: Gene Structure Format (GSF) annotations (tokenized by GFFTokenizer)

The model learns to predict gene isoform annotations from raw genomic DNA.
Training data is stored in DuckDB and loaded via isoformDataHyena.
"""
import os

# --- Environment variables (must be set BEFORE importing torch) ---
os.environ['HF_HOME'] = './HFmodels'                       # HuggingFace model cache directory
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',           # Use CUDA async memory allocator
                       'backend:cudaMallocAsync')           # Reduces alloc/dealloc overhead vs native

import torch, wandb, gc, time, sys, math, json, argparse, glob
from transgenic.training.b5_runtime import (load_b5_config, model_kwargs, accumulation_steps as _acc_steps, EarlyStopper,
                                             CheckpointLayout, split_row_numbers, parse_args as _b5_parse_args, benchmark_summary)
from tqdm import tqdm
import bitsandbytes as bnb                                  # Provides 8-bit quantized optimizers
from torch.nn.utils import clip_grad_norm_                  # Prevents gradient explosion
from transformers import get_linear_schedule_with_warmup    # LR: warmup then linear decay to 0
from accelerate import Accelerator                          # Handles mixed precision + multi-GPU
from safetensors.torch import save_model                    # Fast, safe model serialization format

# --- Project imports ---
from transgenic.datasets.datasets import isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def _write_json(path: str, obj: dict):
    """Write a dict to a JSON file (used for checkpoint metadata)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: str) -> dict:
    """Read a JSON file and return as dict (used to restore checkpoint metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_checkpoint(checkpoint_path: str) -> str | None:
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_ds,                                   # Training dataset (torch Dataset or Subset)
    eval_ds,                                    # Evaluation dataset
    lr,                                         # Peak learning rate (after warmup)
    num_epochs,                                 # Total number of training epochs
    schedule_lr,                                # Whether to use LR scheduler (warmup + decay)
    do_eval,                                    # Whether to run evaluation after each epoch
    batch_size,                                 # Mini-batch size per forward pass
    max_grad_norm=1.0,                          # Max gradient norm for clipping
    checkpoint_path="checkpoints/",             # Directory for saving checkpoints
    output_dir="saved_models/",                 # Directory for final model output
    accumulation_steps=16,                      # Gradient accumulation steps
                                                # Effective batch size = batch_size * accumulation_steps
    num_workers=8,                              # Number of DataLoader workers
    notes="",                                   # Experiment notes logged to wandb
    encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",  # Pretrained encoder to load
    resume_from_checkpoint: str | None = None,  # Path to Accelerate checkpoint (or "auto")
    save_every_epoch: bool = True,              # Save full resumable checkpoint every epoch
    save_every_n_steps: int = 5000,             # Save full checkpoint every N optimizer steps
    unlink=False,                               # If True, don't share weights between enc/dec
    log_wandb=True,                             # Enable Weights & Biases logging
    b5_config: dict | None = None,              # frozen recipe (configs/b5_400m_v1.json); None = legacy wide run
    seed: int = 234,                            # shuffling/init seed (B5: the run seed)
    patience: int = 3,                          # best-validation early stopping (B5)
    run_dir: str | None = None,                 # B5 checkpoint layout root (epoch_NN/, best, TRAINING_DONE)
    benchmark_steps: int = 0,                   # >0: time N optimizer steps, print JSON, exit
):
    """
    Main training function.

    Uses Accelerate for mixed precision (bf16), gradient accumulation,
    and 8-bit AdamW optimizer to reduce memory footprint.
    Supports resuming from Accelerate checkpoints.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Initialize wandb for experiment tracking ----
    if log_wandb:
        wandb.init(
            entity="transgenic-paper",
            project="transgenic",
            config={
                "learning_rate": lr,
                "schedule_lr": schedule_lr,
                "architecture": "Hyena",
                "dataset": "10G_static6144_addExtra200_addRCIsoOnly_clean",
                "epochs": num_epochs,
                "max_grad_norm": max_grad_norm,
                "accumulation_steps": accumulation_steps,
                "batch_size": batch_size,
                "optimizer": "AdamW8bit",
                "checkpoints": checkpoint_path,
                "outputs": output_dir,
                "notes": notes,
            },
        )

    print(f"Training transgenic with Hyena. {checkpoint_path=} {output_dir=}", file=sys.stderr)

    # ---- Device and precision setup ----
    torch.set_float32_matmul_precision('high')  # Enable TF32 on tensor cores (3x faster matmul)
    torch.backends.cudnn.benchmark = True       # cuDNN auto-tuner: picks fastest conv algorithm

    accelerator = Accelerator(mixed_precision="bf16")  # bf16: better dynamic range than fp16,
    device = accelerator.device                        # no loss scaling needed, native on Ampere+
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"Using: {device} ({gpu_props.name}, {gpu_props.total_mem / 1e9:.0f}GB)", file=sys.stderr)

    # ---- DataLoaders ----
    torch.manual_seed(seed)                     # Seed for reproducible data shuffling / init
    torch.cuda.manual_seed_all(seed)            # Seed all GPUs for reproducibility
    layout = CheckpointLayout(run_dir) if run_dir else None
    stopper = EarlyStopper(patience=patience)
    dl_kwargs = dict(
        shuffle=True,                           # Shuffle training data each epoch
        batch_size=batch_size,
        pin_memory=True,                        # Pin CPU memory for faster DMA to GPU
        num_workers=num_workers,                # Parallel data loading workers
        collate_fn=hyena_collate_fn,            # Pads variable-length sequences in a batch
        persistent_workers=True,                # Keep workers alive between epochs
    )                                           # (avoids re-creating DuckDB connections)
    train_dl = makeDataLoader(train_ds, **dl_kwargs)
    eval_dl = makeDataLoader(eval_ds, **dl_kwargs)

    # ---- Model configuration (wide variant: ~1.17B params) ----
    # Roughly 3x params vs base model: (1152/768)^2 * (16/12) ~ 3.0
    d_model, layers, heads = 1152, 16, 8        # Hidden dim, num layers, attention heads (legacy wide run)
    if b5_config is not None:
        kw = model_kwargs(b5_config)
        kw.update(unlink=unlink)
        config = HyenaTransgenicConfig(**kw)
        print(f"B5 recipe: {b5_config.get('name')} encoder {kw['encoder_d_model']}x{kw['encoder_layers']} decoder {kw['decoder_d_model']}x{kw['decoder_layers']}", file=sys.stderr)
    else:
        print("WARNING: no --config given; building the legacy 1.17B wide configuration (not the B5 recipe)", file=sys.stderr)
    config = config if b5_config is not None else HyenaTransgenicConfig(
        d_model=d_model,                        # Model hidden dimension
        encoder_layers=layers,                  # Number of HyenaDNA encoder layers
        decoder_layers=layers,                  # Number of Longformer decoder layers
        encoder_n_layer=layers,                 # Hyena operator internal layers
        encoder_ffn_dim=d_model * 4,            # Feed-forward hidden dim (standard 4x expansion)
        decoder_ffn_dim=d_model * 4,
        attention_window=[1024] * layers,       # Longformer sliding window size (tokens per side)
        dropout=0.1,                            # Dropout rate for regularization
        encoder_attention_heads=heads,
        decoder_attention_heads=heads,
        encoder_model=encoder_model,            # Load pretrained HyenaDNA weights into encoder
        unlink=unlink,                          # Whether to unlink encoder/decoder embeddings
    )
    model = transgenicForConditionalGeneration(config)  # Instantiate the full seq2seq model
    print(f"Model params: {_count_parameters(model):,}", file=sys.stderr)

    model.gradient_checkpointing_enable()       # Trade compute for memory: recompute activations
    model.to(device)                            # during backward instead of storing them
    model.train()                               # Set model to training mode (enables dropout etc.)

    # ---- torch.compile (set TRANSGENIC_NO_COMPILE=1 to disable) ----
    # Uses Inductor backend for kernel fusion: fuses multiple ops into single GPU kernels,
    # reducing memory bandwidth overhead and kernel launch latency.
    # NOTE: Does not work on GB10 (SM shared memory too small: 101KB < 180KB needed by Triton)
    #       Works on 4090, A100, H100, and other GPUs with >= 164KB shared memory per SM.
    if not os.environ.get("TRANSGENIC_NO_COMPILE"):
        try:
            model = torch.compile(model)        # Lazy compilation: compiles on first forward pass
            print("torch.compile (inductor) enabled.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: torch.compile failed, continuing eager: {e}", file=sys.stderr)

    # ---- Optimizer ----
    # 8-bit AdamW: quantizes optimizer states (momentum, variance) from fp32 to int8,
    # reducing optimizer memory from ~8 bytes/param to ~2 bytes/param (~75% savings).
    if b5_config is not None and b5_config.get("optimizer", "AdamW") == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(b5_config.get("weight_decay", 0.02)))
    else:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,                                  # Peak learning rate
            weight_decay=0.02,                      # L2 regularization (decoupled from LR in AdamW)
        )
    optimizer.zero_grad(set_to_none=True)       # Release grad memory instead of zeroing (faster)

    # ---- Learning rate scheduler ----
    # Linear warmup for first 5% of steps (stabilizes early training),
    # then linear decay to 0 over remaining steps.
    steps_per_epoch = max(1, math.ceil(len(train_dl) / accumulation_steps))  # Optimizer steps/epoch
    t_total = steps_per_epoch * num_epochs      # Total optimizer steps across all epochs
    lr_scheduler = None
    if schedule_lr:
        warmup_steps = int(t_total * 0.05)      # 5% of total steps for warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,      # LR ramps from 0 to `lr` over these steps
            num_training_steps=t_total,         # LR decays from `lr` to 0 over remaining steps
        )
        print(f"LR schedule: {warmup_steps} warmup steps, {t_total} total steps", file=sys.stderr)

    # ---- Accelerate prepare ----
    # Wraps model, optimizer, dataloaders, and scheduler for:
    #   - Automatic mixed precision (bf16 forward, fp32 optimizer)
    #   - Distributed training (if multiple GPUs available)
    #   - Gradient accumulation handling
    if lr_scheduler is None:
        model, optimizer, train_dl, eval_dl = accelerator.prepare(
            model, optimizer, train_dl, eval_dl)
    else:
        model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, eval_dl, lr_scheduler)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    resume_step = 0                             # Micro-batch step to resume from (within start_epoch)
    global_step = 0                             # Total optimizer steps completed so far
    if resume_from_checkpoint is not None:
        state_dir = resume_from_checkpoint
        if os.path.isdir(os.path.join(resume_from_checkpoint, "accelerate_state")):   # B5 epoch directory
            state_dir = os.path.join(resume_from_checkpoint, "accelerate_state")
            if layout is not None and layout.read_state() and layout.read_state().get("stopper"):
                stopper.load_state(layout.read_state()["stopper"])
        accelerator.load_state(state_dir)       # Restores model, optimizer, scheduler, RNG
        meta_path = os.path.join(resume_from_checkpoint, "meta.json")
        if os.path.exists(meta_path):
            meta = _read_json(meta_path)
            start_epoch = int(meta.get("epoch", meta.get("next_epoch", 0)))
            resume_step = int(meta.get("step", 0))
            global_step = int(meta.get("global_step", 0))
        print(f"Resumed from {resume_from_checkpoint}; epoch={start_epoch}, step={resume_step}, global_step={global_step}", file=sys.stderr)

    def _save_state(epoch: int, step: int, global_step: int, best_score):
        """Save full Accelerate state (model + optimizer + scheduler + RNG) for resuming."""
        ckpt_dir = os.path.join(checkpoint_path, f"accelerate_epoch{epoch}_step{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        accelerator.save_state(ckpt_dir)        # Saves everything needed to resume training
        _write_json(os.path.join(ckpt_dir, "meta.json"), {
            "epoch": epoch,                     # Epoch to resume from
            "step": step,                       # Micro-batch step to resume from within epoch
            "global_step": global_step,         # Total optimizer steps completed so far
            "best_eval_score": None if best_score is None else float(best_score),
        })

    def _save_best(score_name, score_val):
        """Save only model weights (safetensors) when a new best score is achieved (legacy layout)."""
        save_model(accelerator.unwrap_model(model),  # unwrap removes Accelerate wrapper
                   f"{checkpoint_path}/model.safetensors")
        print(f"New best model saved with {score_name}={score_val}", file=sys.stderr)

    def _save_epoch_b5(epoch_1based: int, eval_loss, train_loss, is_best: bool, global_step: int):
        """B5 layout: epoch_NN.tmp -> epoch_NN with model.safetensors, accelerate_state/, eval.json; best symlink."""
        tmp = layout.begin_epoch(epoch_1based)
        save_model(accelerator.unwrap_model(model), os.path.join(tmp, "model.safetensors"))
        accelerator.save_state(os.path.join(tmp, "accelerate_state"))
        _write_json(os.path.join(tmp, "meta.json"), {"epoch": epoch_1based, "step": 0, "global_step": global_step,
                                                     "best_eval_score": stopper.best, "seed": seed})
        layout.finish_epoch(epoch_1based, eval_loss, train_loss, extra={"global_step": global_step, "is_best": is_best,
                            "stopper": stopper.state()}, is_best=is_best)
        layout.write_state({"epoch": epoch_1based, "global_step": global_step, "stopper": stopper.state(), "seed": seed})

    # ========================================================================
    # Training loop
    # ========================================================================
    best_eval_score = None
    try:
        _bench_sec, _bench_tok, _bench_tokens, _bench_t0 = [], [], 0, time.perf_counter()
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0                      # Accumulated loss for epoch-level metrics

            for step, batch in enumerate(tqdm(train_dl, miniters=10)):
                # Skip already-processed steps when resuming mid-epoch
                if epoch == start_epoch and step < resume_step:
                    continue

                # Transfer tensors to GPU asynchronously (overlaps with computation)
                # Requires pin_memory=True in DataLoader to be effective
                ii = batch[0].to(device, non_blocking=True)    # input_ids: DNA sequence tokens
                am = batch[1].to(device, non_blocking=True)    # attention_mask: 1=real, 0=padding
                lab = batch[2].to(device, non_blocking=True)   # labels: GSF annotation tokens

                # OOM guard: skip batches where batch_size * seq_len exceeds threshold.
                # Tune this value based on your GPU's VRAM (24GB for 4090, 80GB for A100).
                if ii.shape[0] * ii.shape[1] > 100_000:
                    if layout is not None:
                        raise RuntimeError(f"B5: oversized batch {ii.shape} — the frozen DB must not contain windows over 49,152 nt")
                    continue

                try:
                    # Forward pass: encoder processes DNA, decoder generates GSF tokens
                    outputs = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)

                    # Detach loss from computation graph for logging (saves memory)
                    total_loss += outputs.loss.detach().float()

                    # Gradient accumulation: divide loss by number of accumulation steps
                    # so that gradients are averaged across the effective batch.
                    # Effective batch size = batch_size (16) * accumulation_steps (16) = 256
                    accelerator.backward(outputs.loss / accumulation_steps)

                    # Only update weights every `accumulation_steps` mini-batches
                    if (step + 1) % accumulation_steps == 0:
                        global_step += 1
                        # Clip gradients to prevent explosion (common in transformer training)
                        clip_grad_norm_(model.parameters(), max_grad_norm)

                        optimizer.step()                        # Update model weights

                        if lr_scheduler is not None:
                            lr_scheduler.step()                 # Advance learning rate schedule

                        # --- Logging to wandb ---
                        if log_wandb:
                            wandb_log = {
                                "epoch": epoch,
                                "step": step,
                                "global_step": global_step,
                                "loss": outputs.loss.detach().float() * accumulation_steps,  # Undo the division
                                "mean_loss": total_loss / (step + 1),  # Running average loss
                            }
                            if lr_scheduler is not None:
                                wandb_log["lr"] = lr_scheduler.get_last_lr()[0]  # Current learning rate

                            # Log per-parameter gradient norms every 10 optimizer steps.
                            # This is expensive (iterates all params) so we don't do it every step.
                            if global_step % 10 == 0:
                                for name, param in model.named_parameters():
                                    if param.grad is not None and param.requires_grad:
                                        wandb_log[f"{name}_grad_norm"] = param.grad.norm().detach().item()
                            wandb.log(wandb_log)

                        # Release gradient memory (faster than filling with zeros)
                        optimizer.zero_grad(set_to_none=True)

                        # Save full resumable checkpoint every save_every_n_steps optimizer steps
                        if save_every_n_steps and global_step % save_every_n_steps == 0:
                            _save_state(epoch, step + 1, global_step, best_eval_score)
                            print(f"Checkpoint saved at epoch {epoch}, step {step+1}, global_step {global_step}", file=sys.stderr)

                    del outputs                                 # Free GPU memory from output tensors

                except Exception as e:
                    # Recovery from OOM or other errors: clear all state and skip to next batch.
                    # batch[3] contains gene names for debugging which samples caused the error.
                    print(f"Error in batch: {batch[3]}, {e}", file=sys.stderr)
                    optimizer.zero_grad(set_to_none=True)       # Clear optimizer gradient state
                    model.zero_grad(set_to_none=True)           # Clear model gradient buffers
                    torch.cuda.empty_cache()                    # Return cached GPU memory to allocator
                    gc.collect()                                # Force Python garbage collection
                    continue

            # ---- End of epoch: compute metrics ----
            train_epoch_loss = total_loss / len(train_dl)       # Average loss over all batches
            train_ppl = torch.exp(train_epoch_loss)             # Perplexity = exp(loss)

            if do_eval:
                # Evaluation loop (no gradient computation needed)
                eval_loss = 0
                for batch in tqdm(eval_dl, miniters=10):
                    with torch.no_grad():                       # Disable autograd (saves memory + speed)
                        outputs = model(
                            input_ids=batch[0].to(device, non_blocking=True),
                            attention_mask=batch[1].to(device, non_blocking=True),
                            labels=batch[2].to(device, non_blocking=True),
                            return_dict=True,
                        )
                    eval_loss += outputs.loss.detach().float()

                eval_epoch_loss = eval_loss / len(eval_dl)
                eval_ppl = torch.exp(eval_epoch_loss)
                print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)

                if log_wandb:
                    wandb.log({"epoch_train_ppl": train_ppl, "epoch_train_loss": train_epoch_loss,
                               "epoch_eval_ppl": eval_ppl, "epoch_eval_loss": eval_epoch_loss})

                # Save model if eval loss improved (best model selection)
                if layout is not None:
                    is_best, should_stop = stopper.update(epoch + 1, float(eval_epoch_loss))
                    best_eval_score = stopper.best
                    _save_epoch_b5(epoch + 1, float(eval_epoch_loss), float(train_epoch_loss), is_best, global_step)
                    if is_best:
                        print(f"New best validation loss {float(eval_epoch_loss):.6f} at epoch {epoch + 1}", file=sys.stderr)
                    if should_stop:
                        print(f"Early stopping: no improvement for {patience} epochs (best epoch {stopper.best_epoch})", file=sys.stderr)
                        layout.mark_done()
                        break
                elif best_eval_score is None or eval_epoch_loss < best_eval_score:
                    best_eval_score = eval_epoch_loss
                    _save_best("eval_epoch_loss", eval_epoch_loss)
            else:
                # No eval: track best based on training loss instead
                print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
                if best_eval_score is None or train_epoch_loss < best_eval_score:
                    best_eval_score = train_epoch_loss
                    _save_best("train_epoch_loss", train_epoch_loss)

            # Save full resumable checkpoint at every epoch boundary
            if save_every_epoch and layout is None:
                _save_state(epoch + 1, 0, global_step, best_eval_score)

            # Free GPU memory between epochs (helps with fragmentation)
            torch.cuda.empty_cache()
            gc.collect()
            total_loss = 0                      # Reset for next epoch
        if layout is not None:
            layout.mark_done()

    except KeyboardInterrupt:
        # Graceful shutdown: save checkpoint with epoch+step so training can resume mid-epoch
        print("KeyboardInterrupt: saving resume checkpoint...", file=sys.stderr)
        _save_state(
            epoch=epoch if 'epoch' in locals() else 0,
            step=step + 1 if 'step' in locals() else 0,
            global_step=global_step,
            best_score=best_eval_score,
        )
        raise                                   # Re-raise so the process exits with correct code

    if log_wandb:
        wandb.finish()                          # Finalize wandb run (upload remaining data)


# ---------------------------------------------------------------------------
# Main: dataset loading and training invocation
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = _b5_parse_args()
    if args.config:
        # ------------------------------------------------------------------ B5 path (issue #17)
        cfg = load_b5_config(args.config)
        acc = args.accumulation_steps or _acc_steps(cfg, args.batch_size)
        max_epochs = args.max_epochs or int(cfg["max_epochs"])
        patience = args.patience or int(cfg.get("patience", 3))
        # membership comes from the frozen split column; the query also refuses excluded species and NULL splits
        split_row_numbers(args.db, "train"); split_row_numbers(args.db, "valid")
        vv = cfg.get("gff_vocab_version", "v2")
        train_data = isoformDataHyena(args.db, mode="train", encoder_model=cfg["encoder_model"], global_attention=False, split="train", gff_vocab_version=vv)
        eval_data = isoformDataHyena(args.db, mode="train", encoder_model=cfg["encoder_model"], global_attention=False, split="valid", gff_vocab_version=vv)
        print(f"B5 split sizes: train={len(train_data)} valid={len(eval_data)} seed={args.seed} acc={acc} max_epochs={max_epochs} patience={patience}", file=sys.stderr)
        layout = CheckpointLayout(args.output_dir)
        resume_ckpt = layout.resume_dir(args.resume)
        if args.resume and resume_ckpt:
            print(f"Resuming from {resume_ckpt}", file=sys.stderr)
        _write_json(os.path.join(args.output_dir, "run_config.json"), {"config": args.config, "recipe": cfg, "db": os.path.abspath(args.db),
                    "seed": args.seed, "batch_size": args.batch_size, "accumulation_steps": acc, "max_epochs": max_epochs, "patience": patience})
        train(
            train_data, eval_data,
            lr=float(cfg["lr"]), num_epochs=max_epochs, schedule_lr=True, do_eval=True,
            batch_size=args.batch_size, accumulation_steps=acc, num_workers=args.num_workers,
            checkpoint_path=args.output_dir, output_dir=args.output_dir, max_grad_norm=1,
            notes=f"B5 {cfg.get('name')} seed {args.seed}", encoder_model=cfg["encoder_model"],
            resume_from_checkpoint=resume_ckpt, save_every_epoch=True, save_every_n_steps=args.save_every_n_steps or 10**9,
            unlink=False, log_wandb=not args.no_wandb,
            b5_config=cfg, seed=args.seed, patience=patience, run_dir=args.output_dir, benchmark_steps=args.benchmark_steps,
        )
        sys.exit(0)
    # ---------------------------------------------------------------------- legacy wide run (unchanged behaviour)
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
    print("WARNING: legacy row-level random_split after RC augmentation (twin leakage); B5 runs must pass --config", file=sys.stderr)
    torch.manual_seed(123)                      # Seed for reproducible train/eval/test split
    ds = isoformDataHyena(
        args.db,
        mode="train",                           # Training mode: returns (input_ids, attn_mask, labels)
        encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",  # Tokenizer for DNA sequences
        global_attention=False,                  # No global attention tokens (use sliding window only)
        exclude_prefix="Zm",                    # Exclude Zea mays (maize) genes for held-out testing
    )
    total = len(ds)
    train_size = int(total * 0.75)
    eval_size = int(total * 0.10)
    test_size = total - train_size - eval_size  # Remainder goes to test (avoids off-by-one)
    train_data, eval_data, _ = torch.utils.data.random_split(
        ds, [train_size, eval_size, test_size]  # _ = test set (not used during training)
    )
    train(
        train_data, eval_data,
        lr=5e-5, num_epochs=10, schedule_lr=True, do_eval=True,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps or 16, num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path, output_dir="saved_models_HyenaWide/", max_grad_norm=1,
        notes="HyenaTransgenic wide (d_model=1152, 16 layers, 8 heads)",
        encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
        resume_from_checkpoint=resume_ckpt, save_every_n_steps=args.save_every_n_steps or 5000,
        unlink=False, log_wandb=not args.no_wandb,
    )
