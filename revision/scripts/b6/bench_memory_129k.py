#!/usr/bin/env python3
"""B6 memory bench at the 129,024-nt tier (plan B6_WP2_WP4_PLAN_v4 section 10 CP2; A46 rule 15).

Run conditions (all fixed, recorded in the output JSON): the B6 400M model of configs/b6_400m_win_v3.json with
the pretrained Hyena encoder, 2 or 4 ranks under DDP (A51), bf16, torch.compile (Inductor, suppress_errors off, whole
model), non-reentrant activation checkpointing, one 129,024-nt tile per rank per micro-batch, dense tracks
(all 31 channels non-zero on every base), 511 real chain elements (column 0 is the dummy) and a worst-case
decoder target of max_label_tokens (8,192). Each step is a forward, backward and AdamW update.

Gate: peak torch.cuda.max_memory_reserved <= 100 GB on every rank (plan 10). Status starts as FAIL; PASS needs
steps > warmup >= 1, every rank completing all measured updates, verified compiled execution (Dynamo graphs and
Inductor kernels or cache loads), no post-warm-up recompile, and the frozen conditions of GATE_CONDITIONS. Any
other settings report NOT_A_GATE_RUN. Exit 0 = PASS, 2 = FAIL, 3 = NOT_A_GATE_RUN, 4 = refused arguments, anything else = error.

Synthetic batches only: memory depends on shapes and density, not on evidence content, and the bench must not
wait for WP3 caches. Launch through revision/scripts/b6/bench_memory_129k.sbatch (never on a login node).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "./HFmodels")         # same cache the B5 trainer uses (train_HyenaTransgenic.py:14)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

GIB = 10 ** 9
EXIT_REFUSED = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/b6_400m_win_v3.json")
    p.add_argument("--tier", type=int, default=129024)
    p.add_argument("--label-tokens", type=int, default=None, help="default: config max_label_tokens (8192)")
    p.add_argument("--chain-elements", type=int, default=511, help="real elements (slots 1..511)")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4, help="steps before the compile guard and timing start")
    p.add_argument("--gate-gb", type=float, default=100.0)
    p.add_argument("--out", required=True, help="JSON output path (rank 0 writes it)")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args(argv)
    if not (a.steps > a.warmup >= 1):
        print(f"REFUSED: need steps > warmup >= 1 (got steps={a.steps}, warmup={a.warmup}); "
              "a gate needs measured updates", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    return a


# The frozen CP2 gate (plan 10): only a run under exactly these conditions can certify PASS.
# ``architecture_sha256`` is the digest of the BUILT model's architecture (b6_runtime.model_architecture), so a
# config that shrinks the model (e.g. decoder_layers 12 -> 1) cannot certify the gate (review 2 H3).
def _frozen_architecture_sha256():
    from transgenic.training.b6_runtime import FROZEN_ARCHITECTURE_SHA256
    return FROZEN_ARCHITECTURE_SHA256


# A51 (author 2026-09-24): per-rank peak memory does not depend on the DDP world size (per-rank micro-batch fixed;
# gradient-bucket buffers are per rank), so the gate is certified on 2 or 4 ranks. 1 rank omits the DDP buffers
# and is a pre-check only (NOT_A_GATE_RUN). Production training stays on 4 GPUs.
GATE_WORLD_SIZES = (2, 4)
GATE_CONDITIONS = {"tier": 129024, "chain_elements": 511, "n_tok": 512, "gate_gb": 100.0,
                   "label_tokens": 8192, "architecture_sha256": _frozen_architecture_sha256()}


def gate_verdict(conditions, ranks, *, steps, warmup, gate_gb):
    """PASS / FAIL / NOT_A_GATE_RUN (torch-free, unit-tested).

    ``conditions``: the run's world_size, tier, chain_elements, n_tok, gate_gb, label_tokens. ``ranks``: one
    dict per rank with measured_updates, compile_verified, peak_reserved_bytes and error. A run whose settings
    differ from GATE_CONDITIONS never certifies anything; otherwise PASS needs every rank to have completed all
    steps - warmup measured updates with verified compiled execution, no error, and peak <= gate_gb."""
    off = {k: (conditions.get(k), v) for k, v in GATE_CONDITIONS.items() if conditions.get(k) != v}
    if off:
        return "NOT_A_GATE_RUN", f"settings differ from the frozen gate: {off}"
    if conditions.get("world_size") not in GATE_WORLD_SIZES:
        return "NOT_A_GATE_RUN", f"world_size {conditions.get('world_size')} is not a gate size {GATE_WORLD_SIZES} (A51)"
    if len(ranks) != conditions["world_size"]:
        return "NOT_A_GATE_RUN", f"{len(ranks)} rank reports for world_size {conditions['world_size']}"
    for r in ranks:
        if r.get("error"):
            return "FAIL", f"rank {r.get('rank')}: {r['error']}"
        if r.get("measured_updates") != steps - warmup:
            return "FAIL", f"rank {r.get('rank')}: {r.get('measured_updates')} measured updates, need {steps - warmup}"
        if r.get("compile_verified") is not True:
            return "FAIL", f"rank {r.get('rank')}: compiled execution not verified"
    worst = max(int(r["peak_reserved_bytes"]) for r in ranks)
    if worst > gate_gb * GIB:
        return "FAIL", f"peak reserved {worst} > {gate_gb} GB"
    return "PASS", f"peak reserved {worst} <= {gate_gb} GB on all {len(ranks)} ranks"


EXIT_CODES = {"PASS": 0, "FAIL": 2, "NOT_A_GATE_RUN": 3}


def synthetic_batch(torch, cfg, args, device, rank):
    """One dense, contract-shaped 129 kb micro-batch (plan 7.7), different per rank."""
    g = torch.Generator().manual_seed(args.seed * 100 + rank)
    L = args.tier
    if L % 6:
        raise SystemExit(f"tier {L} is not a multiple of 6")
    T = int(args.label_tokens or cfg.get("max_label_tokens", 8192))
    n = args.chain_elements
    ids = torch.randint(7, 11, (1, L), generator=g)                        # Hyena char ids A, C, G, T
    tracks = (torch.rand(1, L, 31, generator=g) * 4.17 + 0.01).to(torch.float16)   # dense, <= log1p(64)
    idx = torch.zeros(1, 512, 2, dtype=torch.int64)
    feat = torch.zeros(1, 512, 6, dtype=torch.int16)
    mask = torch.zeros(1, 512, dtype=torch.bool)
    idx[0, 1:n + 1] = torch.randint(0, L // 6, (n, 2), generator=g)
    feat[0, 1:n + 1] = torch.stack([torch.randint(0, 32, (n,), generator=g), torch.randint(0, 32, (n,), generator=g),
                                    torch.randint(1, 64, (n,), generator=g), torch.randint(0, 3, (n,), generator=g),
                                    torch.randint(0, 2, (n,), generator=g), torch.randint(1, 65, (n,), generator=g)],
                                   dim=1).to(torch.int16)
    mask[0, 1:n + 1] = True
    labels = torch.randint(4, int(cfg.get("vocab_size", 290)), (1, T), generator=g)
    labels[0, -1] = 2
    batch = dict(input_ids=ids, attention_mask=torch.ones(1, L, dtype=torch.bool), evidence_tracks=tracks,
                 labels=labels, chain_idx=idx, chain_feat=feat, chain_mask=mask)
    return {k: v.to(device) for k, v in batch.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, gather_object
    from transgenic.model.modeling_b6 import B6TransgenicForConditionalGeneration, b6_config
    from transgenic.training import b6_runtime as rt

    cfg = rt.load_b6_config(args.config)
    acc = Accelerator(mixed_precision="bf16",
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = B6TransgenicForConditionalGeneration(b6_config(**rt.b6_model_kwargs(cfg)))
    architecture = rt.model_architecture(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(acc.device).train()
    kernels_before = rt.inductor_kernel_count()
    model = rt.compile_b6(model, backend="inductor")
    optimizer = rt.build_b6_optimizer(model, float(cfg["lr"]), float(cfg["weight_decay"]),
                                      expected_alphas=int(cfg["decoder_layers"]))
    model, optimizer = acc.prepare(model, optimizer)
    batch = synthetic_batch(torch, cfg, args, acc.device, acc.process_index)
    guard = rt.CompileGuard(args.warmup)
    step_seconds, error, measured, compile_verified = [], None, 0, False
    graphs_before, kernels_before = rt.compiled_graph_count(), rt.inductor_activity()
    torch.cuda.reset_peak_memory_stats(acc.device)
    try:
        for step in range(1, args.steps + 1):
            torch._dynamo.maybe_mark_dynamic(batch["labels"], 1)
            t0 = time.perf_counter()
            out = model(**batch)
            acc.backward(out.loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(acc.device)
            if not torch.isfinite(out.loss.detach()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            guard.observe(step, tuple(batch["input_ids"].shape), rt.compiled_graph_count(),
                          bool(torch._dynamo.config.suppress_errors))
            if step == args.warmup:
                rt.check_compiled(graphs_before, rt.compiled_graph_count())
                rt.check_inductor_kernels("inductor", kernels_before, rt.inductor_activity())
                compile_verified = True
            if step > args.warmup:
                step_seconds.append(time.perf_counter() - t0)
                measured += 1
            reserved = torch.cuda.memory_reserved(acc.device)
            peaks = acc.gather(torch.tensor([reserved], device=acc.device)).tolist()
            if rt.memory_abort(peaks, None, write=False):
                error = f"MEMORY_ABORT threshold exceeded at step {step}: {peaks}"
                break
    except (RuntimeError, FloatingPointError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    rank = dict(rank=acc.process_index, peak_reserved_bytes=int(torch.cuda.max_memory_reserved(acc.device)),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(acc.device)),
                mean_step_seconds=(sum(step_seconds) / len(step_seconds)) if step_seconds else None,
                measured_updates=measured, compile_verified=compile_verified,
                device=torch.cuda.get_device_name(acc.device), error=error)
    ranks = gather_object([rank])
    label_tokens = int(args.label_tokens or cfg.get("max_label_tokens", 8192))
    conditions = dict(world_size=acc.num_processes, tier=args.tier, chain_elements=args.chain_elements,
                      n_tok=int(batch["chain_mask"].shape[1]), gate_gb=args.gate_gb, label_tokens=label_tokens,
                      architecture_sha256=rt.architecture_sha256(architecture))
    status, reason = gate_verdict(conditions, ranks, steps=args.steps, warmup=args.warmup, gate_gb=args.gate_gb)
    if acc.is_main_process:
        import transformers
        summary = dict(bench="b6_memory_129k", status=status, reason=reason, gate_conditions=GATE_CONDITIONS,
                       conditions=conditions, architecture=architecture, gate_bytes=int(args.gate_gb * GIB),
                       worst_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in ranks), ranks=ranks,
                       dense_channels=31, compile="inductor",
                       suppress_errors=bool(torch._dynamo.config.suppress_errors), checkpointing="non-reentrant",
                       mixed_precision="bf16", steps=args.steps, warmup=args.warmup,
                       config=os.path.abspath(args.config), config_sha256=rt.file_sha256(args.config),
                       torch=torch.__version__, transformers=transformers.__version__,
                       inductor_activity=rt.inductor_activity(), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(json.dumps(summary), flush=True)
    acc.wait_for_everyone()
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
