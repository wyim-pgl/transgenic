#!/usr/bin/env python
"""
Profile training bottlenecks for TransGenic on NVIDIA GB10.

Runs a small number of training batches (default 10) with explicit
torch.cuda.synchronize() barriers between each phase of the training
step. This forces the GPU pipeline to drain, giving accurate wall-clock
measurements for each phase:

  1. Data loading  — time to unpack batch tensors from DataLoader
  2. Host→Device   — CPU-to-GPU memory transfer (via PCIe / unified memory)
  3. Forward pass   — encoder + decoder inference
  4. Backward pass  — gradient computation (usually the bottleneck: ~70%)
  5. Optimizer step  — 8-bit AdamW parameter update

Results from source-built PyTorch 2.11 (SM 12.0 Blackwell kernels):
  Forward: 0.458s, Backward: 1.268s, Total: 1.932s per batch (batch_size=2)
  This is 1.59x faster than pip-installed PyTorch.

Usage:
  python profile_training.py 2>&1 | tee profile_output.txt
"""
import os
os.environ['HF_HOME'] = './HFmodels'  # Cache HuggingFace models locally

import torch, sys, time
import bitsandbytes as bnb                # 8-bit optimizer library
from accelerate import Accelerator        # HuggingFace distributed training wrapper
from transgenic.datasets.datasets import isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig

NUM_BATCHES = 10  # Number of batches to profile (excluding warmup)

# Enable TF32 for matrix multiply (Ampere+ tensor cores: 8x throughput vs FP32)
torch.set_float32_matmul_precision('high')
# Let cuDNN auto-tune convolution algorithms for consistent input sizes
torch.backends.cudnn.benchmark = True

# Initialize Accelerator with FP16 mixed precision for profiling
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
# Cap GPU memory at ~80GB on GB10 (128.5GB unified memory)
# Prevents the Linux OOM killer from terminating the process
torch.cuda.set_per_process_memory_fraction(0.62)

# ── Dataset Setup ──────────────────────────────────────────────────────────
# Use a tiny subset of the training data — just enough for profiling
db = '/home/wyim/data/transgenic_data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db'
ds = isoformDataHyena(db, mode='train', exclude_prefix='Zm')  # Exclude maize (Zm) genes
# Take only (NUM_BATCHES * 2 + 10) samples to ensure enough data for profiling
subset = torch.utils.data.Subset(ds, range(NUM_BATCHES * 2 + 10))
# Small batch_size=2 to fit in memory during profiling
dl = makeDataLoader(subset, shuffle=False, batch_size=2, num_workers=2, collate_fn=hyena_collate_fn)

# ── Model Setup ────────────────────────────────────────────────────────────
# Wide architecture: d_model=1152, 16 layers, 8 heads (~1.17B params)
config = HyenaTransgenicConfig(
    d_model=1152, encoder_layers=16, decoder_layers=16, encoder_n_layer=16,
    encoder_ffn_dim=4608, decoder_ffn_dim=4608,   # FFN = 4x d_model = 4608
    attention_window=[1024]*16,                     # 1024-token Longformer attention window per layer
    dropout=0.1, encoder_attention_heads=8, decoder_attention_heads=8)
model = transgenicForConditionalGeneration(config)
model.gradient_checkpointing_enable()  # Trade compute for memory (recompute activations in backward)
model.to(device)
model.train()  # Set to training mode (enables dropout, gradient tracking)

# 8-bit AdamW optimizer (saves ~75% optimizer state memory vs FP32 AdamW)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5, weight_decay=0.02)
# Wrap model, optimizer, and dataloader with Accelerator for mixed precision
model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", file=sys.stderr)
print(f"Profiling {NUM_BATCHES} batches...\n", file=sys.stderr)

# ── Warmup (not profiled) ─────────────────────────────────────────────────
# Run one full training step to warm up CUDA kernels, JIT compilation,
# and cuDNN autotuner. This avoids one-time overhead skewing results.
batch = next(iter(dl))
ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
out = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
accelerator.backward(out.loss)
optimizer.step(); optimizer.zero_grad()
del out; torch.cuda.empty_cache()  # Free warmup memory before profiling
print("Warmup done.\n", file=sys.stderr)

# ── Profile Loop ──────────────────────────────────────────────────────────
# Timing accumulators for each phase
t_data, t_h2d, t_fwd, t_bwd, t_opt = [], [], [], [], []

torch.cuda.synchronize()  # Ensure all prior GPU work is finished
for i, batch in enumerate(dl):
    if i >= NUM_BATCHES:
        break  # Stop after profiling NUM_BATCHES batches

    # ─── Phase 1: Data loading time ───
    # Measures tensor unpacking from DataLoader (CPU-side only)
    t0 = time.perf_counter()
    ii_cpu, am_cpu, lab_cpu = batch[0], batch[1], batch[2]
    t1 = time.perf_counter()
    t_data.append(t1 - t0)

    # ─── Phase 2: Host-to-device (CPU → GPU) transfer ───
    # Copies tensors from CPU RAM to GPU memory
    # On GB10 with unified memory, this is essentially a no-op pointer remap
    ii = ii_cpu.to(device)
    am = am_cpu.to(device)
    lab = lab_cpu.to(device)
    torch.cuda.synchronize()  # Wait for all transfers to complete
    t2 = time.perf_counter()
    t_h2d.append(t2 - t1)

    # ─── Phase 3: Forward pass ───
    # Runs encoder (HyenaDNA) + downsampling + decoder (Longformer) + loss
    out = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
    torch.cuda.synchronize()  # Wait for all GPU compute to finish
    t3 = time.perf_counter()
    t_fwd.append(t3 - t2)

    # ─── Phase 4: Backward pass ───
    # Computes gradients via backpropagation (with gradient checkpointing)
    # This is typically the bottleneck (~69% of total time)
    accelerator.backward(out.loss)
    torch.cuda.synchronize()  # Wait for all gradient computation to finish
    t4 = time.perf_counter()
    t_bwd.append(t4 - t3)

    # ─── Phase 5: Optimizer step ───
    # Updates model weights using 8-bit AdamW and resets gradients
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()  # Wait for parameter updates to complete
    t5 = time.perf_counter()
    t_opt.append(t5 - t4)

    # Print per-batch timing breakdown
    seq_len = ii.shape[1]  # DNA sequence length for this batch
    total = t_data[-1] + t_h2d[-1] + t_fwd[-1] + t_bwd[-1] + t_opt[-1]
    print(f"Batch {i:2d} | seq={seq_len:6d} | data={t_data[-1]:.3f}s  h2d={t_h2d[-1]:.3f}s  "
          f"fwd={t_fwd[-1]:.3f}s  bwd={t_bwd[-1]:.3f}s  opt={t_opt[-1]:.3f}s  "
          f"total={total:.3f}s  loss={out.loss.item():.4f}", file=sys.stderr)
    del out  # Free forward pass memory

# ── Summary Statistics ────────────────────────────────────────────────────
def avg(lst): return sum(lst) / len(lst) if lst else 0

print(f"\n{'='*70}", file=sys.stderr)
print(f"PROFILE SUMMARY ({NUM_BATCHES} batches, batch_size=2)", file=sys.stderr)
print(f"{'='*70}", file=sys.stderr)
# Average time and percentage for each phase
print(f"  Data loading:     {avg(t_data):.3f}s avg  ({avg(t_data)/avg([sum(x) for x in zip(t_data,t_h2d,t_fwd,t_bwd,t_opt)])*100:.1f}%)", file=sys.stderr)
print(f"  Host→Device:      {avg(t_h2d):.3f}s avg", file=sys.stderr)
print(f"  Forward:          {avg(t_fwd):.3f}s avg", file=sys.stderr)
print(f"  Backward:         {avg(t_bwd):.3f}s avg", file=sys.stderr)
print(f"  Optimizer step:   {avg(t_opt):.3f}s avg", file=sys.stderr)
total_avg = avg(t_data) + avg(t_h2d) + avg(t_fwd) + avg(t_bwd) + avg(t_opt)
print(f"  Total per batch:  {total_avg:.3f}s avg", file=sys.stderr)
# Breakdown into compute vs data vs optimizer fractions
print(f"\n  Compute (fwd+bwd): {(avg(t_fwd)+avg(t_bwd))/total_avg*100:.1f}%", file=sys.stderr)
print(f"  Data (load+h2d):   {(avg(t_data)+avg(t_h2d))/total_avg*100:.1f}%", file=sys.stderr)
print(f"  Optimizer:         {avg(t_opt)/total_avg*100:.1f}%", file=sys.stderr)
# Peak GPU memory usage
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"\n  Peak GPU memory:  {peak_mb:.0f} MB ({peak_mb/1e3:.1f} GB)", file=sys.stderr)
