#!/usr/bin/env python
"""Profile 10 training batches with CUDA events to identify bottlenecks."""
import os
os.environ['HF_HOME'] = './HFmodels'

import torch, sys, time
import bitsandbytes as bnb
from accelerate import Accelerator
from transgenic.datasets.datasets import isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig

NUM_BATCHES = 10

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
torch.cuda.set_per_process_memory_fraction(0.62)

# Dataset
db = '/home/wyim/data/transgenic_data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db'
ds = isoformDataHyena(db, mode='train', exclude_prefix='Zm')
subset = torch.utils.data.Subset(ds, range(NUM_BATCHES * 2 + 10))
dl = makeDataLoader(subset, shuffle=False, batch_size=2, num_workers=2, collate_fn=hyena_collate_fn)

# Model
config = HyenaTransgenicConfig(
    d_model=1152, encoder_layers=16, decoder_layers=16, encoder_n_layer=16,
    encoder_ffn_dim=4608, decoder_ffn_dim=4608, attention_window=[1024]*16,
    dropout=0.1, encoder_attention_heads=8, decoder_attention_heads=8)
model = transgenicForConditionalGeneration(config)
model.gradient_checkpointing_enable()
model.to(device)
model.train()

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5, weight_decay=0.02)
model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", file=sys.stderr)
print(f"Profiling {NUM_BATCHES} batches...\n", file=sys.stderr)

# Warmup 1 batch (not profiled)
batch = next(iter(dl))
ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
out = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
accelerator.backward(out.loss)
optimizer.step(); optimizer.zero_grad()
del out; torch.cuda.empty_cache()
print("Warmup done.\n", file=sys.stderr)

# Profile loop
t_data, t_h2d, t_fwd, t_bwd, t_opt = [], [], [], [], []

torch.cuda.synchronize()
for i, batch in enumerate(dl):
    if i >= NUM_BATCHES:
        break

    # --- Data loading time ---
    t0 = time.perf_counter()
    ii_cpu, am_cpu, lab_cpu = batch[0], batch[1], batch[2]
    t1 = time.perf_counter()
    t_data.append(t1 - t0)

    # --- Host-to-device transfer ---
    ii = ii_cpu.to(device)
    am = am_cpu.to(device)
    lab = lab_cpu.to(device)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    t_h2d.append(t2 - t1)

    # --- Forward pass ---
    out = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    t_fwd.append(t3 - t2)

    # --- Backward pass ---
    accelerator.backward(out.loss)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    t_bwd.append(t4 - t3)

    # --- Optimizer step ---
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    t_opt.append(t5 - t4)

    seq_len = ii.shape[1]
    total = t_data[-1] + t_h2d[-1] + t_fwd[-1] + t_bwd[-1] + t_opt[-1]
    print(f"Batch {i:2d} | seq={seq_len:6d} | data={t_data[-1]:.3f}s  h2d={t_h2d[-1]:.3f}s  "
          f"fwd={t_fwd[-1]:.3f}s  bwd={t_bwd[-1]:.3f}s  opt={t_opt[-1]:.3f}s  "
          f"total={total:.3f}s  loss={out.loss.item():.4f}", file=sys.stderr)
    del out

# Summary
def avg(lst): return sum(lst) / len(lst) if lst else 0
print(f"\n{'='*70}", file=sys.stderr)
print(f"PROFILE SUMMARY ({NUM_BATCHES} batches, batch_size=2)", file=sys.stderr)
print(f"{'='*70}", file=sys.stderr)
print(f"  Data loading:     {avg(t_data):.3f}s avg  ({avg(t_data)/avg([sum(x) for x in zip(t_data,t_h2d,t_fwd,t_bwd,t_opt)])*100:.1f}%)", file=sys.stderr)
print(f"  Host→Device:      {avg(t_h2d):.3f}s avg", file=sys.stderr)
print(f"  Forward:          {avg(t_fwd):.3f}s avg", file=sys.stderr)
print(f"  Backward:         {avg(t_bwd):.3f}s avg", file=sys.stderr)
print(f"  Optimizer step:   {avg(t_opt):.3f}s avg", file=sys.stderr)
total_avg = avg(t_data) + avg(t_h2d) + avg(t_fwd) + avg(t_bwd) + avg(t_opt)
print(f"  Total per batch:  {total_avg:.3f}s avg", file=sys.stderr)
print(f"\n  Compute (fwd+bwd): {(avg(t_fwd)+avg(t_bwd))/total_avg*100:.1f}%", file=sys.stderr)
print(f"  Data (load+h2d):   {(avg(t_data)+avg(t_h2d))/total_avg*100:.1f}%", file=sys.stderr)
print(f"  Optimizer:         {avg(t_opt)/total_avg*100:.1f}%", file=sys.stderr)
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"\n  Peak GPU memory:  {peak_mb:.0f} MB ({peak_mb/1e3:.1f} GB)", file=sys.stderr)
