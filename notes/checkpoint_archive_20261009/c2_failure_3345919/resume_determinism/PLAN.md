# Actual C2 resume determinism diagnostic

The recovery 3361386 was deliberately signalled to stop: all four ranks saved update256 (step6144, pending0). Trainer exited normally; Slurm1:0 is the deliberate failure of the script's minimum400 postcondition. Controller3276938 is absent; journal remains blocked_determinism/jobs[]. Original checkpoint200 and production path unchanged.

All original/recovery overlapping draw rows are byte-equal, including rank3's available continuation4801–6144. Other original ranks only flushed4800 rows. Recovery rank0 row5040 is rn184594; canonical full order reconstruction runs as a diagnostic gate.

One four-GPU2h allocation, immutable original200, unchanged production config/4099 target, bf16, deterministic kernels, accumulation24 per rank. Three separate Python/accelerate launches:

1. A:233574a from200 to210.
2. B:d7b1a37 from200 to220; snapshot210 after atomic checkpoint publication inside collective checkpoint_io.
3. C:d7b1a37 from B210 to220.

Every case starts with its own empty Inductor and Triton cache. No extra forward priming. Trace outer DDP inputs, shapes, strides, CPU/CUDA/Python/NumPy RNG before/after forward, losses, accumulation-boundary gradients and DDP telemetry. Compare exact model tensors, optimizer, scheduler and all rank RNG payloads, not ZIP/safetensor serialization ordering. Input and stochastic execution equality are reported separately. Passing A=B isolates the policy change; B=C tests the resume boundary. Neither alone proves historical original200→210 equality under an unrecorded compiler history.

Heavy input verification is performed once within this allocation; the canonical binding record must equal original200's identity. Each phase verifies its allocation-bound record, paths, binding SHA and input-file size/mtime; inputs are mounted read-only. The diagnostic wrapper reuses this record and skips B6Corpus's redundant corpus rehash. All training/checkpoint identity checks remain active. This exception is confined to probe.py, and must never enter production.

No production promotion, resubmission or source mutation is performed by these scripts. The stopped continuation controller remains disabled. Comparison failure is a nonzero exit with retained traces/checkpoints. Analysis and any fix must preserve the existing hard scientific gate.
