# RTX PRO 6000 Blackwell: verified training sources

Checked 2026-10-21. Published measurements exist, including full-parameter training. None establishes current B6 runtime on one RTX. No GPU jobs, rentals, or installations were performed.

## Strongest accessible full-training comparison

[Harvard Kempner benchmark](https://handbook.eng.kempnerinstitute.harvard.edu/technical_blog/choosing_gpu_partition.html), updated July 15, 2026, reports the same single-GPU script on three independent allocations per GPU, PyTorch 2.8.0/CUDA 12.8. It trains a synthetic GPT-style model (~0.94B parameters; 16 layers; width 2048; 16 heads; batch 12; sequence 1024). The included code performs BF16 autocast forward, cross-entropy, backward, and AdamW on all parameters; fused AdamW has a fallback. Eight warm-up and 30 timed steps exclude data loading and convergence measurement.

| GPU | Training tokens/s |
|---|---:|
| H100 80 GB | 73,899 |
| H200 141 GB | 77,163 |
| RTX PRO 6000 Blackwell 96 GB | 41,370 |

Derived RTX/H100 throughput is **0.5598**; equal-work runtime is **1.7863×**. This is a measured training proxy, not a B6 benchmark or four-GPU scaling test. The linked [cluster GPU inventory](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/gpu_types_and_use_cases.html) identifies `kempner_rtx` as **Server Edition**. Actual benchmark power cap is undisclosed; do not relabel it 600 W Workstation.

## Exxact: confirmed original LoRA measurement

[Exxact, June 4, 2026](https://www.exxactcorp.com/blog/benchmark/lora-fine-tuning-benchmark-on-nvidia-gpus): **RTX PRO 6000 Server Edition**, 96 GB; AMD EPYC 9124 host. Model `NousResearch/Meta-Llama-3.1-8B-Instruct`; LoRA rank 16 on `q_proj,v_proj`; frozen BF16 base, FP32 adapters; batch 2/GPU; sequence 512; Hugging Face Transformers/DDP. Package versions are not specified. Throughput averages steps 1–60 after warm-up.

| RTX PRO 6000 SE count | Aggregate tokens/s | Average GPU power | Scaling efficiency |
|---|---:|---:|---:|
| 1 | 4,962 | 404.6 W | — |
| 2 | 9,541 | 662.6 W total | 96.1% |

**No H100 measurement appears in this benchmark.** The author attributes strong PCIe scaling to the small adapter-gradient payload. It cannot substantiate equivalent scaling for full-model training. Average measured draw does not identify the configured power limit. This is neither a 600 W Workstation test nor a full-training result.

## Other original measurements and access limitations

- [The Kaitchup, Benjamin Marie, June 16, 2025](https://kaitchup.substack.com/p/faster-and-cheaper-than-the-h100): accessible introduction says the author ran A100/H100/RTX PRO 6000 head-to-head Qwen3 QLoRA, LoRA, and full fine-tuning. Exact tables, configurations, and runtime ratios are behind the subscription boundary. Notebook link also leads to subscriber material. The public cost claim does **not** verify a particular throughput ratio, model size, H100 variant, or RTX edition; exclude it from numerical extrapolation.
- [QRUN, runs September 2–3, 2026](https://qrun.cloud/measurements): same Qwen2.5-7B BF16 LoRA workload, first 30,000 `alpaca-cleaned` examples, one epoch, batch 4 × accumulation 4, maximum length 1024, no quantization/checkpointing, AdamW. Reports RTX PRO 6000 96 GB at **0.78× H100 SXM 80 GB speed**. Billed lifetimes are 0.70 versus 0.44 hours, including startup/teardown; their ratio is not the per-step ratio. Different providers/hosts; RTX edition, LoRA rank, and package versions unavailable. Linked raw CSV could not be retrieved (tool failure; direct HTTP 403), so the ratio remains an author-reported result.
- [Alio Leuchtmann, Hugging Face community article, June 5, 2026](https://huggingface.co/blog/AlioLeuchtmann/rtx-6000-pro-vs-gh200-sft-benchmark): claims a 600 W workstation RTX versus 96 GB GH200 full-SFT comparison, Qwen3.5 0.8B/2B/4B/9B, BF16, batch 1, checkpointing, 8-bit AdamW, 32 measured steps. Reports RTX advantage up to 1.8× at short lengths and GH200 advantage 1.7–1.8× at long lengths. However, retrieved content exposes chart captions without numerical curves; its claimed companion harness/CSV repository has no exposed link. Treat as a lower-confidence lead, not a reproducibly verified ratio or B6 ETA.

## Implication for B6

Task context supplied by the parent: B6 is a roughly 400M encoder–decoder model with a HyenaDNA encoder, autoregressive decoder, 31 evidence channels, and chain memory; full-model BF16 training reaches DNA length 129,024 and labels 8,192. These operations and shapes differ substantially from the published tests. Memory fit and kernel compatibility remain unmeasured on RTX.

The supplied 24–36-hour remainder on four GH200s is itself an estimate, not completed runtime. A **conditional illustration only** is `(24–36 h) × 4 × (73899/41370) = 7.15–10.72 days`, assuming H100/GH200 equivalence, transferable workload ratio, unchanged effective batch, and ideal four-GPU scaling. None is demonstrated here. The earlier 4–12-day claim must remain an assumption, not a published or local B6 benchmark.

## Local capability check

The active workspace's `/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` returned `Quadro K620, 2048 MiB, 537.58` on2026-10-21 (exit0). This is an inventory observation only. No RTX PRO6000 is exposed in this workspace, and no B6 runtime benchmark was executed on that GPU. Direct validation requires an accessible target RTX host; no GPU rental, job submission, or migration was performed for this research request.
