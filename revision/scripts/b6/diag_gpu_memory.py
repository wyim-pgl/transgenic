#!/usr/bin/env python3
"""B6 1-GPU memory diagnostic: per-micro-batch CUDA memory under the trainer's settings and a mixed shape sequence.

Why: the real-data 4-GPU smoke OOMed at ~step 165 in the compiled backward (bf16 (6, s38, 21504) decoder
cross-attention scores; 87.09 GiB allocated, 6.03 GiB reserved but unallocated), while the CP2 bench
(bench_memory_129k.py, one fixed 129,024 nt x 8,192 label shape, one update per batch) peaked at 58.64 GiB reserved.
This tool MEASURES the candidate differences instead of assuming one: deterministic algorithms, per-rank gradient
accumulation, varying DNA tiers and label lengths (recompiles), and the caching allocator configuration.

What it mirrors from train/train_HyenaTransgenic_B6.py (run_b6, 826e68e): Accelerator(bf16, DDP kwargs
find_unused_parameters=True); torch.set_float32_matmul_precision("high"); when --deterministic on, exactly the two
statements the trainer runs inline (torch.use_deterministic_algorithms(True); cudnn.deterministic=True,
cudnn.benchmark=False; there is no shared helper); the trainer's build_model seeding; non-reentrant checkpointing via
resolve_checkpointing; rt.compile_b6 (Inductor, suppress_errors off, optimize_ddp off); rt.build_b6_optimizer; the
linear warm-up schedule; maybe_mark_dynamic(labels, 1) before every forward; backward of loss / per_rank_acc; at
each accumulation boundary clip_grad_norm_(1.0), optimizer.step, scheduler.step, zero_grad(set_to_none=True); and the
CompileGuard observation (recorded here, not raised). Model and synthetic batch come from bench_memory_129k.py.

Known differences (recorded in the JSON): one process, so no DDP wrapper or gradient buckets; synthetic batches
(dense tracks, 511 chain elements by default) instead of the WP3 loader; no evaluation pass.

Never sets PYTORCH_CUDA_ALLOC_CONF; it records the value it inherited. Exit codes: 0 completed (or stopped by
--max-minutes), 5 CUDA OOM (recorded with shape and memory_summary), 4 refused arguments, 1 any other error.
`--summarize A.json B.json ...` prints the comparison table without torch or a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_memory_129k as bench  # noqa: E402  (also puts <repo>/src on sys.path and sets HF_HOME)

from transgenic.training import b6_runtime as rt  # noqa: E402

GIB = 2 ** 30
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_OOM = 0, 1, 4, 5
DEFAULT_CONFIG = "configs/b6_400m_win_v3_wp5_pilot.json"
DEFAULT_SHAPES = ("129024:4480,129024:8192,61440:3000,30720:1500,129024:6000,"
                  "129024:4480,61440:3000,129024:8192")
DEFAULT_WORLD_SIZE = 4                     # production (WP5 pilot block world_size); used to derive per-rank accum
INACTIVE_SPLIT_KEY = "inactive_split_bytes.all.current"
STAT_KEYS = (INACTIVE_SPLIT_KEY, "num_alloc_retries", "num_ooms", "reserved_bytes.all.current",
             "allocated_bytes.all.current", "active_bytes.all.peak")


class Refused(ValueError):
    pass


# ---- torch-free helpers (unit-tested on CPU) ---------------------------------------------------------------------

def parse_shapes(text: str) -> List[Tuple[int, int]]:
    """'tier:labels,tier:labels,...' -> [(tier, labels), ...]; tiers must be multiples of 6, labels >= 2."""
    shapes = []
    for item in (s.strip() for s in str(text).split(",")):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise Refused(f"shape {item!r} is not tier:labels")
        try:
            tier, labels = int(parts[0]), int(parts[1])
        except ValueError:
            raise Refused(f"shape {item!r}: tier and labels must be integers") from None
        if tier <= 0 or tier % 6:
            raise Refused(f"shape {item!r}: tier {tier} must be a positive multiple of 6")
        if labels < 2:
            raise Refused(f"shape {item!r}: labels {labels} must be >= 2")
        shapes.append((tier, labels))
    if not shapes:
        raise Refused("empty --shapes")
    return shapes


def check_shapes_against_config(shapes: Sequence[Tuple[int, int]], cfg: Dict) -> List[str]:
    """Refuse shapes the model cannot take; return notes for tiers outside window_tiers (allowed, flagged)."""
    max_seq = int(cfg.get("max_encoder_seqlen", 129024))
    max_lab = int(cfg.get("max_label_tokens", 8192))
    tiers = set(int(t) for t in cfg.get("window_tiers", ()))
    notes = []
    for tier, labels in shapes:
        if tier > max_seq:
            raise Refused(f"tier {tier} > max_encoder_seqlen {max_seq}")
        if labels > max_lab:
            raise Refused(f"labels {labels} > max_label_tokens {max_lab}")
        if tiers and tier not in tiers:
            notes.append(f"tier {tier} is not a config window tier {sorted(tiers)}")
    return sorted(set(notes))


def shape_sequence(shapes: Sequence[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """The first n micro-batch shapes, cycling through ``shapes``."""
    return [tuple(shapes[i % len(shapes)]) for i in range(int(n))]


def per_rank_accumulation(cfg: Dict, world_size: int, batch_size: int = 1) -> int:
    """The trainer's per-rank accumulation: rt.accumulation_steps(cfg, batch_size) // world (run_b6)."""
    total = rt.accumulation_steps(cfg, batch_size)
    if total % world_size:
        raise Refused(f"accumulation_steps {total} is not divisible by the world size {world_size}")
    return total // world_size


def _on_off(value: str) -> bool:
    v = str(value).lower()
    if v in ("on", "1", "true", "yes"):
        return True
    if v in ("off", "0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got {value!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma-separated tier:labels micro-batches, cycled")
    p.add_argument("--accum", type=int, default=None,
                   help="micro-batches per optimizer update (default: the trainer's per-rank value, "
                        "accumulation_steps(cfg, 1) // --world-size)")
    p.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE,
                   help="production world size used only to derive the default --accum")
    p.add_argument("--deterministic", type=_on_off, default=True, metavar="{on,off}",
                   help="apply the trainer's --deterministic settings (default on, as the WP5 pilot sbatch)")
    p.add_argument("--updates", type=int, default=4, help="optimizer updates to run")
    p.add_argument("--chain-elements", type=int, default=511)
    p.add_argument("--dense-tracks", dest="dense_tracks", action="store_true", default=True,
                   help="all 31 channels non-zero (default, as the bench)")
    p.add_argument("--no-dense-tracks", dest="dense_tracks", action="store_false", help="all-zero tracks")
    p.add_argument("--compile-warmup-steps", type=int, default=3, help="trainer default (CompileGuard)")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop cleanly at the first update boundary after this wall time (status time_budget)")
    p.add_argument("--seed", type=int, default=rt.SEED)
    p.add_argument("--out", default=None, help="JSON report (rewritten after every micro-batch)")
    p.add_argument("--summarize", nargs="+", default=None, metavar="JSON",
                   help="print the comparison table of existing reports and exit (no torch, no GPU)")
    a = p.parse_args(argv)
    if a.summarize:
        return a
    try:
        if not a.out:
            raise Refused("--out is required")
        a.shape_list = parse_shapes(a.shapes)
        if a.updates < 1:
            raise Refused("--updates must be >= 1")
        if a.accum is not None and a.accum < 1:
            raise Refused("--accum must be >= 1")
        if a.world_size < 1:
            raise Refused("--world-size must be >= 1")
        if not 0 <= a.chain_elements <= rt.CHAIN_ELEMENTS - 1:
            raise Refused(f"--chain-elements must be in [0, {rt.CHAIN_ELEMENTS - 1}] (slot 0 is the dummy)")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    return a


def is_oom(exc: BaseException, oom_types: Tuple[type, ...] = ()) -> bool:
    """torch.OutOfMemoryError, or a RuntimeError carrying the CUDA OOM text (compiled graphs can rewrap it)."""
    if oom_types and isinstance(exc, oom_types):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def shape_key(tier: int, labels: int) -> str:
    return f"{int(tier)}:{int(labels)}"


def summarize_report(report: Dict) -> Dict:
    """Per-shape maxima (GiB) and the run-level facts the sbatch table prints."""
    per_shape: Dict[str, Dict] = {}
    for mb in report.get("micro_batches", []):
        k = shape_key(mb["tier"], mb["labels"])
        s = per_shape.setdefault(k, dict(n=0, peak_allocated_gib=0.0, peak_reserved_gib=0.0,
                                         max_inactive_split_gib=0.0))
        s["n"] += 1
        s["peak_allocated_gib"] = max(s["peak_allocated_gib"], mb["max_allocated_bytes"] / GIB)
        s["peak_reserved_gib"] = max(s["peak_reserved_gib"], mb["max_reserved_bytes"] / GIB)
        s["max_inactive_split_gib"] = max(s["max_inactive_split_gib"], mb.get("inactive_split_bytes", 0) / GIB)
    mbs = report.get("micro_batches", [])
    ups = report.get("updates", [])
    oom = report.get("oom") or None
    return dict(
        label=report.get("label") or os.path.basename(str(report.get("out", ""))),
        status=report.get("status"),
        deterministic=report.get("settings", {}).get("deterministic"),
        alloc_conf=report.get("env", {}).get("PYTORCH_CUDA_ALLOC_CONF"),
        micro_batches=len(mbs), updates=len(ups),
        peak_allocated_gib=max((m["max_allocated_bytes"] for m in mbs), default=0) / GIB,
        peak_reserved_gib=max((m["max_reserved_bytes"] for m in mbs), default=0) / GIB,
        update_peak_reserved_gib=max((u.get("max_reserved_bytes", 0) for u in ups), default=0) / GIB,
        compiled_graphs=(mbs[-1]["compiled_graphs"] if mbs else None),
        alloc_retries=(mbs[-1].get("num_alloc_retries") if mbs else None),
        oom_shape=(shape_key(oom["tier"], oom["labels"]) if oom else None),
        oom_micro_batch=(oom.get("micro_batch") if oom else None),
        oom_phase=(oom.get("phase") if oom else None),
        per_shape=per_shape)


def format_table(reports: Iterable[Dict]) -> str:
    lines = []
    for r in reports:
        s = summarize_report(r)
        lines.append(f"== {s['label']}: status={s['status']} deterministic={s['deterministic']} "
                     f"PYTORCH_CUDA_ALLOC_CONF={s['alloc_conf'] or '(unset)'} micro_batches={s['micro_batches']} "
                     f"updates={s['updates']} compiled_graphs={s['compiled_graphs']} "
                     f"alloc_retries={s['alloc_retries']}")
        lines.append(f"   peak allocated {s['peak_allocated_gib']:.2f} GiB, reserved {s['peak_reserved_gib']:.2f} GiB "
                     f"(optimizer update reserved {s['update_peak_reserved_gib']:.2f} GiB); OOM: "
                     + (f"{s['oom_shape']} at micro-batch {s['oom_micro_batch']} ({s['oom_phase']})"
                        if s["oom_shape"] else "none"))
        lines.append(f"   {'shape':>14} {'n':>4} {'alloc_GiB':>10} {'resv_GiB':>10} {'inact_GiB':>10}")
        for k, v in sorted(s["per_shape"].items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))):
            lines.append(f"   {k:>14} {v['n']:>4} {v['peak_allocated_gib']:>10.2f} {v['peak_reserved_gib']:>10.2f} "
                         f"{v['max_inactive_split_gib']:>10.2f}")
    return "\n".join(lines)


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def summarize_main(paths: Sequence[str]) -> int:
    reports = []
    for p in paths:
        try:
            with open(p) as fh:
                r = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"== {p}: no readable report ({type(exc).__name__}: {exc})")
            continue
        r.setdefault("out", p)
        reports.append(r)
    if reports:
        print(format_table(reports))
    return EXIT_OK if len(reports) == len(paths) else EXIT_ERROR


# ---- GPU part ----------------------------------------------------------------------------------------------------

def make_batch(torch, cfg, args, tier: int, labels: int, index: int, device):
    """The bench's dense contract batch at (tier, labels), different per micro-batch index."""
    ns = SimpleNamespace(tier=tier, label_tokens=labels, chain_elements=args.chain_elements, seed=args.seed)
    batch = bench.synthetic_batch(torch, cfg, ns, torch.device("cpu"), index)
    if not args.dense_tracks:
        batch["evidence_tracks"] = torch.zeros_like(batch["evidence_tracks"])
    return {k: v.to(device, non_blocking=False) for k, v in batch.items()}


def _mem(torch, device) -> Dict:
    st = torch.cuda.memory_stats(device)
    out = dict(allocated_bytes=int(torch.cuda.memory_allocated(device)),
               reserved_bytes=int(torch.cuda.memory_reserved(device)),
               max_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
               max_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
               inactive_split_bytes=int(st.get(INACTIVE_SPLIT_KEY, 0)),
               num_alloc_retries=int(st.get("num_alloc_retries", 0)), num_ooms=int(st.get("num_ooms", 0)))
    return out


def _dynamo_counters() -> Dict:
    from torch._dynamo.utils import counters
    return {k: dict(counters[k]) for k in ("stats", "recompiles", "frames") if k in counters}


def run(args) -> int:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from torch.nn.utils import clip_grad_norm_
    from transformers import get_linear_schedule_with_warmup
    from transgenic.model.modeling_b6 import B6TransgenicForConditionalGeneration, b6_config

    cfg = rt.load_b6_config(args.config)
    try:
        notes = check_shapes_against_config(args.shape_list, cfg)
        accum = args.accum or per_rank_accumulation(cfg, args.world_size)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    acc = Accelerator(mixed_precision="bf16",
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if acc.num_processes != 1:
        print(f"REFUSED: run on one process (got {acc.num_processes})", file=sys.stderr)
        return EXIT_REFUSED
    device = acc.device
    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                                  getattr(torch.cuda, "OutOfMemoryError", None)) if t is not None)
    env = {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG",
                                          "TORCH_LOGS", "TORCHINDUCTOR_CACHE_DIR", "TRANSGENIC_NO_GRAD_CKPT",
                                          "TRANSGENIC_CKPT_REENTRANT", "SLURM_JOB_ID")}
    report = dict(tool="b6_diag_gpu_memory", status="running", label=os.environ.get("DIAG_LABEL"),
                  out=os.path.abspath(args.out), env=env,
                  settings=dict(config=os.path.abspath(args.config), config_sha256=rt.file_sha256(args.config),
                                shapes=[list(s) for s in args.shape_list], accum=accum,
                                accum_source=("--accum" if args.accum else
                                              f"accumulation_steps(cfg, 1)={rt.accumulation_steps(cfg, 1)} // "
                                              f"world {args.world_size}"),
                                deterministic=bool(args.deterministic), updates=args.updates,
                                chain_elements=args.chain_elements, dense_tracks=bool(args.dense_tracks),
                                compile_warmup_steps=args.compile_warmup_steps, max_minutes=args.max_minutes,
                                seed=args.seed, mixed_precision="bf16", compile="inductor (rt.compile_b6)",
                                checkpointing=None, shape_notes=notes),
                  differences_from_production=["1 process: no DDP wrapper / gradient buckets",
                                               "synthetic batches (bench_memory_129k.synthetic_batch)",
                                               "no evaluation pass"],
                  micro_batches=[], updates=[], oom=None, error=None)

    def save():
        write_json(args.out, report)

    torch.set_float32_matmul_precision("high")
    if args.deterministic:
        # Exactly train_HyenaTransgenic_B6.run_b6 (spec.deterministic); needs CUBLAS_WORKSPACE_CONFIG=:4096:8.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(rt.SEED)                      # train_HyenaTransgenic_B6.build_model
    torch.cuda.manual_seed_all(rt.SEED)
    model = B6TransgenicForConditionalGeneration(b6_config(**rt.b6_model_kwargs(cfg)))
    ckpt = rt.resolve_checkpointing(os.environ, acc.num_processes)
    report["settings"]["checkpointing"] = ckpt
    if ckpt != "nonreentrant":
        print(f"REFUSED: checkpointing {ckpt}; unset TRANSGENIC_NO_GRAD_CKPT / TRANSGENIC_CKPT_REENTRANT",
              file=sys.stderr)
        return EXIT_REFUSED
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()
    n_layers = len(model.transgenic.decoder.layers)
    kernels_before, graphs_before = rt.inductor_activity(), rt.compiled_graph_count()
    model = rt.compile_b6(model, backend="inductor")
    guard = rt.CompileGuard(args.compile_warmup_steps)
    optimizer = rt.build_b6_optimizer(model, float(cfg["lr"]), float(cfg["weight_decay"]), expected_alphas=n_layers)
    optimizer.zero_grad(set_to_none=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, args.updates // 20), max(args.updates, 2))
    model, optimizer, scheduler = acc.prepare(model, optimizer, scheduler)
    report.update(device=torch.cuda.get_device_name(device),
                  device_total_bytes=int(torch.cuda.get_device_properties(device).total_memory),
                  torch=torch.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  suppress_errors=bool(torch._dynamo.config.suppress_errors),
                  optimize_ddp=bool(torch._dynamo.config.optimize_ddp),
                  model_static_allocated_bytes=int(torch.cuda.memory_allocated(device)))
    save()

    seq = shape_sequence(args.shape_list, args.updates * accum)
    t_start = time.time()
    phase, tier, labels, i = "setup", None, None, 0
    exit_code = EXIT_OK
    try:
        for i, (tier, labels) in enumerate(seq, start=1):
            phase = "batch"
            torch.cuda.reset_peak_memory_stats(device)
            inputs = make_batch(torch, cfg, args, tier, labels, i, device)
            torch._dynamo.maybe_mark_dynamic(inputs["labels"], 1)
            t0 = time.perf_counter()
            phase = "forward"
            out = model(**inputs)
            phase = "backward"
            acc.backward(out.loss / accum)
            torch.cuda.synchronize(device)
            loss = float(out.loss.detach().float())
            mb = dict(micro_batch=i, tier=tier, labels=labels, seconds=round(time.perf_counter() - t0, 3),
                      loss=loss, **_mem(torch, device), compiled_graphs=rt.compiled_graph_count())
            mb["allocated_after_backward_bytes"] = mb.pop("allocated_bytes")
            report["micro_batches"].append(mb)
            print(f"mb {i:4d} {tier:>6}:{labels:<5} {mb['seconds']:7.2f}s peak_alloc {mb['max_allocated_bytes'] / GIB:6.2f} "
                  f"peak_resv {mb['max_reserved_bytes'] / GIB:6.2f} after_bwd {mb['allocated_after_backward_bytes'] / GIB:6.2f} "
                  f"inactive_split {mb['inactive_split_bytes'] / GIB:5.2f} GiB retries {mb['num_alloc_retries']} "
                  f"graphs {mb['compiled_graphs']} loss {loss:.4f}", flush=True)
            if i % accum == 0:
                phase = "update"
                u_idx = i // accum
                torch.cuda.reset_peak_memory_stats(device)
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                up = dict(update=u_idx, after_micro_batch=i, **_mem(torch, device),
                          compiled_graphs=rt.compiled_graph_count(), guard_error=None)
                up["allocated_after_step_zero_grad_bytes"] = up.pop("allocated_bytes")
                try:
                    guard.observe(u_idx, tuple(inputs["input_ids"].shape), rt.compiled_graph_count(),
                                  bool(torch._dynamo.config.suppress_errors))
                except RuntimeError as exc:          # the trainer raises here; the diagnostic records it
                    up["guard_error"] = str(exc)
                if u_idx == args.compile_warmup_steps:
                    try:
                        rt.check_compiled(graphs_before, rt.compiled_graph_count())
                        rt.check_inductor_kernels("inductor", kernels_before, rt.inductor_activity())
                        report["compile_verified"] = True
                    except RuntimeError as exc:
                        report["compile_verified"] = str(exc)
                report["updates"].append(up)
                print(f"update {u_idx:3d} after mb {i}: alloc_after_step {up['allocated_after_step_zero_grad_bytes'] / GIB:6.2f} "
                      f"peak_resv {up['max_reserved_bytes'] / GIB:6.2f} GiB graphs {up['compiled_graphs']}"
                      + (f" GUARD: {up['guard_error']}" if up["guard_error"] else ""), flush=True)
                if args.max_minutes is not None and (time.time() - t_start) / 60.0 >= args.max_minutes:
                    report["status"] = "time_budget"
                    break
            save()
        if report["status"] == "running":
            report["status"] = "completed"
    except Exception as exc:                         # noqa: BLE001 - record every failure in the JSON
        if is_oom(exc, oom_types):
            report["status"] = "oom"
            report["oom"] = dict(micro_batch=i, tier=tier, labels=labels, phase=phase,
                                 message=str(exc)[:4000], memory=_mem(torch, device),
                                 memory_stats={k: torch.cuda.memory_stats(device).get(k) for k in STAT_KEYS},
                                 memory_summary=torch.cuda.memory_summary(device, abbreviated=False))
            print(f"OOM at micro-batch {i} shape {tier}:{labels} phase {phase}", file=sys.stderr, flush=True)
            print(report["oom"]["memory_summary"], file=sys.stderr, flush=True)
            exit_code = EXIT_OOM
        else:
            report["status"] = "error"
            report["error"] = f"{type(exc).__name__} at micro-batch {i} ({phase}): {str(exc)[:4000]}"
            print(report["error"], file=sys.stderr, flush=True)
            exit_code = EXIT_ERROR
    report.update(elapsed_seconds=round(time.time() - t_start, 1), dynamo_counters=_dynamo_counters(),
                  final_compiled_graphs=rt.compiled_graph_count(), inductor_activity=rt.inductor_activity(),
                  finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save()
    print(format_table([report]), flush=True)
    return exit_code


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize:
        return summarize_main(args.summarize)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
