#!/usr/bin/env python3
"""B6 evidence-mode trainer (plan B6_WP2_WP4_PLAN_v4 section 9; protocol A46 rules 8, 9, 11, 15).

Gate: ``main`` refuses to start unless the config's gate_status says the CP2 memory bench passed (a GATED config
carries the bench record, b6_runtime.check_gate_evidence), or --allow-ungated is given for an integration test.
The production config is GATED since author decision 15 (2026-09-26; bench 3208285 on 2 ranks per A51, CP4
3210279/3210585). Production never uses --allow-ungated; the WP5 pilot config refuses it.

What it does, relative to the B5 trainer (train/train_HyenaTransgenic.py, which stays untouched):
  * model = B6TransgenicForConditionalGeneration (tracks K = 31 + chain memory + 12 gated chain blocks);
  * one arm per run (B6, C2, C3, C4, C6, C7; rule 11), all at seed 123 and aug_seed 20261037; arms differ only
    by the dataset arm (C3 donor evidence) and the tensor transform ``b6_runtime.apply_arm``, which uses no RNG;
  * loss = L_GSF only; stop rule on de novo validation L_GSF (patience 3, ceiling 22) for B6 and C2, evidence
    validation L_GSF logged; C3/C4/C6/C7 stop at exactly N_pilot = 4,142 optimizer updates;
  * every arm publishes ``step_04142/`` (model.safetensors, meta.json, PRUNE_EXEMPT) at update 4,142;
  * WP5 pilot (--stop-after-updates 4142 or configs/b6_400m_win_v3_wp5_pilot.json): B6/C2 stop at exactly 4,142
    optimizer updates on every rank, ``step_04142/`` is marked PILOT_FINAL (meta pilot_final, never best-validation),
    TRAINING_DONE is written; the stop target is in the run identity, so a resume with another target is refused.
    The LR schedule is the full-run schedule truncated at 4,142 (A46 rule 11 matched budget with C3/C4/C6/C7);
  * AdamW as B5 with every chain gate alpha outside weight decay; the B5 LR schedule, including its world-size
    accounting (steps_per_epoch from the unsharded loader length), so the schedule is B5's;
  * torch.compile with suppress_errors off; zero Inductor kernels after warm-up or a recompile of an already
    compiled shape after warm-up raises (rule 9);
  * MEMORY_ABORT above 110 GB reserved on any rank; runtime identity with a required key set, verified on
    every link; per-layer alpha recorded at every link start.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

os.environ.setdefault("HF_HOME", "./HFmodels")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))

from transgenic.training import b6_runtime as rt  # noqa: E402

GATE_STATUS = rt.GATE_STATUS


class MemoryAbort(SystemExit):
    pass


@dataclass
class B6TrainSpec:
    arm: rt.ArmPreset
    cfg: Dict
    run_dir: str
    bindings: Dict
    seed: int = rt.SEED
    batch_size: int = 1
    num_workers: int = 0
    accumulation_steps: Optional[int] = None      # recipe total; divided by the world size (B5 rule)
    max_epochs: int = rt.MAX_EPOCHS
    patience: int = rt.PATIENCE
    n_pilot: int = rt.N_PILOT
    save_every_n_steps: int = 200
    mixed_precision: str = "bf16"
    compile_backend: Optional[object] = "inductor"     # None = eager (debugging only; recorded in identity)
    compile_warmup_steps: int = 3
    max_grad_norm: float = 1.0
    resume: Optional[str] = None
    link: Dict = field(default_factory=dict)
    stop_at_step: Optional[int] = None                 # test hook: behave as if USR1 arrived at this update
    stop_after_updates: Optional[int] = None           # WP5 pilot: hard stop at exactly N_pilot optimizer updates
    required_world_size: Optional[int] = None          # WP5 pilot config: 4 ranks (refuse any other world size)
    memory_probe: Optional[Callable[[], int]] = None   # test hook: reserved bytes of this rank (default CUDA)
    deterministic: bool = False                        # torch.use_deterministic_algorithms (GPU byte stability)
    record_sha256_fn: Optional[Callable] = None        # WP3 b6_bindings.canonical_record_sha256 (main always sets it)
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr, flush=True)


def _versions() -> Dict[str, str]:
    import accelerate
    import torch
    import transformers
    try:
        import duckdb
        duck = duckdb.__version__
    except ImportError:
        duck = "absent"
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "accelerate": accelerate.__version__, "duckdb": duck}


def _unwrap(accelerator, model):
    """The bare B6 model under DDP and torch.compile wrappers, in either nesting order.

    Not ``accelerator.unwrap_model``: accelerate 1.10.1 ``extract_model_from_parallel`` raises
    KeyError('_orig_mod') for DDP(OptimizedModule), because ``has_compiled_regions`` is true for the DDP wrapper
    and it then reads ``model.__dict__['_orig_mod']`` from the wrapper itself (seen in the 2-rank smoke run).
    """
    import torch
    m = model
    while True:
        if isinstance(m, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            m = m.module
        elif isinstance(m, torch._dynamo.eval_frame.OptimizedModule):
            m = m._orig_mod
        else:
            return m


def run_b6(spec: B6TrainSpec, model, train_ds, eval_ds, *, collate_fn, train_sampler, eval_sampler=None,
           accelerator=None) -> Dict:
    """Train one arm. ``model`` is built by the caller right after torch.manual_seed(seed) (A34 init);
    ``train_sampler`` must expose set_epoch(epoch) and a deterministic per-epoch order (WP3 B6EpochSampler)."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    arm, cfg, log = spec.arm, spec.cfg, spec.log
    acc = accelerator or Accelerator(mixed_precision=spec.mixed_precision,
                                     kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    world, is_main, barrier, device = acc.num_processes, acc.is_main_process, acc.wait_for_everyone, acc.device
    total_acc = spec.accumulation_steps or rt.accumulation_steps(cfg, spec.batch_size)
    if total_acc % world:
        raise ValueError(f"accumulation_steps {total_acc} is not divisible by the world size {world}")
    per_rank_acc = total_acc // world
    budget = rt.budget_steps(arm, spec.n_pilot)
    pilot_target = rt.pilot_stop_target(spec.stop_after_updates, spec.n_pilot)
    if spec.required_world_size is not None and world != spec.required_world_size:
        raise ValueError(f"world size {world} != {spec.required_world_size} required by the WP5 pilot config")
    layout = rt.CheckpointLayout(spec.run_dir)
    stopper = rt.EarlyStopper(patience=spec.patience)
    torch.set_float32_matmul_precision("high")
    if spec.deterministic:
        # CUDA backward kernels with atomics (gather/scatter-add, embedding) make two identical GPU runs differ
        # bitwise; byte-stable step checkpoints across a resume on GPU need deterministic kernels. Requires
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment; an op without a deterministic kernel raises.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    # Each loader owns a generator. Creating a DataLoader iterator draws its worker base seed from the loader's
    # generator, or from the GLOBAL torch RNG when it has none; that global draw lands after the restored RNG
    # state on a resumed link but before it in an uninterrupted run, so dropout/LayerDrop masks diverge and the
    # step checkpoint is not byte-stable across a resume (seen in the resume-determinism test). WP3 draws use
    # blake2b uniforms, never torch RNG, so the worker seeds carry no augmentation state.
    train_dl = DataLoader(train_ds, batch_size=spec.batch_size, sampler=train_sampler, collate_fn=collate_fn,
                          num_workers=spec.num_workers, pin_memory=acc.device.type == "cuda",
                          generator=torch.Generator().manual_seed(spec.seed))
    eval_dl = DataLoader(eval_ds, batch_size=spec.batch_size, sampler=eval_sampler, collate_fn=collate_fn,
                         num_workers=spec.num_workers, pin_memory=acc.device.type == "cuda",
                         generator=torch.Generator().manual_seed(spec.seed + 1))

    if not hasattr(train_sampler, "epoch_rows"):
        raise TypeError("train_sampler must expose epoch_rows(epoch): the canonical per-epoch row order (rule 11)")
    init_weights_sha256 = rt.state_digest(model)            # the weights as built, before training or resume
    ckpt = rt.resolve_checkpointing(os.environ, world)
    if ckpt == "reentrant":
        raise RuntimeError("B6 requires non-reentrant checkpointing (plan 9); unset TRANSGENIC_CKPT_REENTRANT")
    if ckpt == "nonreentrant":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()
    n_layers = len(model.transgenic.decoder.layers)
    compiled = spec.compile_backend is not None
    kernels_before = rt.inductor_activity() if compiled else 0
    graphs_before = rt.compiled_graph_count() if compiled else 0
    if compiled:
        model = rt.compile_b6(model, backend=spec.compile_backend)
    guard = rt.CompileGuard(spec.compile_warmup_steps)
    optimizer = rt.build_b6_optimizer(model, float(cfg["lr"]), float(cfg["weight_decay"]), expected_alphas=n_layers)
    optimizer.zero_grad(set_to_none=True)
    # B5 parity (train_HyenaTransgenic.py:347-352): the unsharded loader length over the per-rank accumulation.
    steps_per_epoch = max(1, math.ceil(len(train_dl) / per_rank_acc))
    t_total = steps_per_epoch * int(cfg["max_epochs"])
    # The WP5 pilot keeps this full-run schedule and stops at N_pilot (truncation, not a schedule sized to N): rule 11
    # compares it with the C3/C4/C6/C7 step-N checkpoints, which train on this same schedule.
    warmup_steps = int(t_total * float(cfg.get("lr_warmup_fraction", 0.05)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)
    model, optimizer, train_dl, eval_dl, scheduler = acc.prepare(model, optimizer, train_dl, eval_dl, scheduler)

    # ---- identity (rule 9) and resume ----
    run = dict(recipe_sha256=cfg["_sha256"], arm=arm.name, seed=spec.seed, aug_seed=rt.AUG_SEED_BASE + spec.seed,
               init_policy=cfg.get("init_policy", "A34 partial-load audit; torch.manual_seed(seed) before build"),
               init_weights_sha256=init_weights_sha256,
               data_order_policy="B5 order; sha256 of the canonical first 1,000 rows per epoch (data_order.jsonl)",
               batch_size=spec.batch_size, accumulation_steps=total_acc, world_size=world,
               effective_batch=spec.batch_size * total_acc, max_epochs=spec.max_epochs, patience=spec.patience,
               budget=arm.budget, n_pilot=spec.n_pilot, stop_after_updates=pilot_target,
               channel_dropout_p=rt.CHANNEL_DROPOUT_P,
               thinning_set=list(rt.THINNING_SET), withhold_source_rate=rt.WITHHOLD_SOURCE_RATE,
               withhold_library_rate=rt.WITHHOLD_LIBRARY_RATE,
               compile_policy=(f"torch.compile backend={spec.compile_backend if isinstance(spec.compile_backend, str) else 'custom'};"
                               f" suppress_errors=False; optimize_ddp={torch._dynamo.config.optimize_ddp};"
                               f" fallback=failure" if compiled else "eager (not a recipe run)"),
               checkpointing=ckpt, mixed_precision=spec.mixed_precision, memory_abort_bytes=rt.MEMORY_ABORT_BYTES,
               deterministic=bool(spec.deterministic),
               loss="L_GSF", model_version=_unwrap(acc, model).config.b6_model_version, **_versions())
    identity = rt.build_run_identity(run, spec.bindings, spec.record_sha256_fn)

    start_epoch = resume_step = global_step = pending_micro = 0
    resume_dir = layout.resume_dir(spec.resume)
    if resume_dir is not None:
        cfg_path, _ = rt.identity_paths(spec.run_dir)
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"resume refused: {cfg_path} is missing, so the run's identity cannot be checked")
        with open(cfg_path) as fh:
            rt.check_run_config_b6(json.load(fh), identity)
        rt.check_checkpoint_identity(resume_dir, identity)      # before any state of that checkpoint is loaded
        state_dir = os.path.join(resume_dir, "accelerate_state")
        acc.load_state(state_dir if os.path.isdir(state_dir) else resume_dir)
        with open(os.path.join(resume_dir, "meta.json")) as fh:
            meta = json.load(fh)
        start_epoch, resume_step = int(meta["epoch"]), int(meta["step"])
        global_step, pending_micro = int(meta["global_step"]), int(meta.get("pending_micro", 0))
        if meta.get("stopper"):
            stopper.load_state(meta["stopper"])
        if pending_micro and rt.load_pending_grads(_unwrap(acc, model), resume_dir, acc.process_index) == 0:
            raise RuntimeError(f"{resume_dir} declares {pending_micro} pending micro-batches but no gradients")
        if pilot_target is not None and global_step >= pilot_target:
            raise RuntimeError(f"resume refused: {resume_dir} is at update {global_step}, not before the WP5 stop "
                               f"target {pilot_target}")
        log(f"resumed from {resume_dir}: epoch {start_epoch} step {resume_step} global_step {global_step}")
    with rt.checkpoint_io(acc):
        if is_main:
            rt.record_identity(spec.run_dir, identity, _unwrap(acc, model).chain_alphas(),
                               dict(spec.link, arm=arm.name, resumed_from=resume_dir, global_step=global_step,
                                    evidence_channel=arm.evidence_channel, gate_status=cfg.get("gate_status")))
    log(f"arm {arm.name}: evidence_channel={arm.evidence_channel} budget={budget or 'stop rule'} "
        f"stop_after_updates={pilot_target} schedule t_total={t_total} warmup={warmup_steps} "
        f"per-rank accumulation {per_rank_acc} world {world} {cfg.get('gate_status')}")

    # ---- checkpoint writers (rank 0 owns directories; every rank saves its RNG state) ----
    def save_latest(epoch0, step, gstep, micro):
        tmp = layout.latest_state_dir() + ".tmp"
        with rt.checkpoint_io(acc):
            draw_log.flush()
            if is_main:
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
        with rt.checkpoint_io(acc):
            acc.save_state(os.path.join(tmp, "accelerate_state"))
        with rt.checkpoint_io(acc):
            if is_main:
                rt.write_checkpoint_identity(tmp, identity)
                with open(os.path.join(tmp, "meta.json"), "w") as fh:
                    json.dump({"epoch": epoch0, "step": step, "global_step": gstep, "pending_micro": micro % per_rank_acc,
                               "stopper": stopper.state(), "arm": arm.name, "seed": spec.seed,
                               "stop_after_updates": pilot_target,
                               "identity_sha256": rt.canonical_sha256(identity)}, fh)
                final, prev = layout.latest_state_dir(), layout.latest_state_dir() + ".prev"
                shutil.rmtree(prev, ignore_errors=True)
                if os.path.isdir(final):
                    os.rename(final, prev)
                os.rename(tmp, final)
                shutil.rmtree(prev, ignore_errors=True)

    def publish_step(gstep, epoch0):
        final = rt.step_checkpoint_dir(spec.run_dir, gstep)
        with rt.checkpoint_io(acc):
            if is_main:
                if os.path.lexists(final):
                    raise FileExistsError(f"refusing to replace published step checkpoint: {final}")
                tmp = final + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                weights = os.path.join(tmp, "model.safetensors")
                aliases = rt.save_weights_deterministic(_unwrap(acc, model), weights)
                rt.write_checkpoint_identity(tmp, identity)
                pilot_final = pilot_target is not None and gstep == pilot_target
                with open(os.path.join(tmp, "meta.json"), "w") as fh:
                    json.dump({"global_step": gstep, "epoch": epoch0, "arm": arm.name, "seed": spec.seed,
                               "alphas": _unwrap(acc, model).chain_alphas(), "tied_aliases": aliases,
                               "checkpoint_kind": "fixed_step", "selection": rt.FIXED_STEP_SELECTION,
                               "pilot_final": pilot_final, "stop_after_updates": pilot_target,
                               "model_sha256": rt.file_sha256(weights), "lr": scheduler.get_last_lr()[0],
                               "schedule_total_steps": t_total, "schedule_warmup_steps": warmup_steps},
                              fh, sort_keys=True)
                open(os.path.join(tmp, rt.PRUNE_EXEMPT_MARKER), "w").write("A28/plan 9: step checkpoint, never pruned\n")
                if pilot_final:
                    open(os.path.join(tmp, rt.PILOT_FINAL_MARKER), "w").write(
                        f"WP5 pilot-final checkpoint at exactly {gstep} optimizer updates (A46 rule 11); "
                        "never a best-validation checkpoint\n")
                os.rename(tmp, final)
        log(f"published {final}")

    def save_epoch(epoch1, gstep, micro, losses, is_best):
        tmp = layout.epoch_dir(epoch1) + ".tmp"
        pending = micro % per_rank_acc
        with rt.checkpoint_io(acc):
            if is_main:
                layout.begin_epoch(epoch1)
                rt.save_weights_deterministic(_unwrap(acc, model), os.path.join(tmp, "model.safetensors"))
        with rt.checkpoint_io(acc):
            acc.save_state(os.path.join(tmp, "accelerate_state"))
        with rt.checkpoint_io(acc):
            if pending:
                rt.save_pending_grads(_unwrap(acc, model), tmp, acc.process_index)
        with rt.checkpoint_io(acc):
            if is_main:
                rt.write_checkpoint_identity(tmp, identity)
                with open(os.path.join(tmp, "meta.json"), "w") as fh:
                    json.dump({"epoch": epoch1, "step": 0, "global_step": gstep, "pending_micro": pending,
                               "stopper": stopper.state(), "arm": arm.name, "seed": spec.seed,
                               "identity_sha256": rt.canonical_sha256(identity)}, fh)
                layout.finish_epoch(epoch1, losses["de_novo"], losses["train"],
                                    extra={"global_step": gstep, "eval_evidence_loss": losses["evidence"],
                                           "is_best": is_best, "stopper": stopper.state()}, is_best=is_best)
                layout.write_state({"epoch": epoch1, "global_step": gstep, "stopper": stopper.state(), "arm": arm.name})
                shutil.rmtree(layout.latest_state_dir(), ignore_errors=True)

    def reduce_sum(value):
        return float(acc.reduce(torch.as_tensor(value, dtype=torch.float32, device=device), reduction="sum"))

    def evaluate():
        model.eval()
        sums = {"de_novo": torch.zeros((), device=device), "evidence": torch.zeros((), device=device)}
        n = 0
        with torch.no_grad():
            for batch in eval_dl:
                rt.validate_b6_batch(batch)
                views = {"de_novo": rt.de_novo_view(batch), "evidence": rt.apply_arm(batch, arm)}
                for name, view in views.items():
                    inputs = rt.model_inputs(view)
                    if compiled:
                        torch._dynamo.maybe_mark_dynamic(inputs["labels"], 1)
                    sums[name] += model(**inputs).loss.detach().float()
                    if compiled:
                        guard.observe(state["link_updates"], ("eval",) + tuple(inputs["input_ids"].shape),
                                      rt.compiled_graph_count(), bool(torch._dynamo.config.suppress_errors))
                n += 1
        model.train()
        total = reduce_sum(float(n))
        return {k: reduce_sum(v) / max(total, 1.0) for k, v in sums.items()}

    stop = {"flag": False}
    state = {"global_step": global_step, "link_updates": 0, "compile_verified": not compiled}

    def verify_compile():
        """Every link (fresh or resumed) proves it runs compiled code: at the end of its own warm-up, or at its
        last update if it ends sooner. Counts are per process, never the global step."""
        rt.check_compiled(graphs_before, rt.compiled_graph_count())
        rt.check_inductor_kernels(spec.compile_backend, kernels_before, rt.inductor_activity())
        state["compile_verified"] = True

    draw_dir = os.path.join(spec.run_dir, "draw_log")
    os.makedirs(draw_dir, exist_ok=True)
    draw_log = open(os.path.join(draw_dir, f"rank{acc.process_index:02d}.tsv"), "a")

    def log_draws(meta):
        """Rule 11: the deterministic draw hash of every trained (row, epoch), per rank; arms must agree."""
        for row, ep, u in zip(meta["row"], meta["epoch"], meta["uniforms_sha256"]):
            draw_log.write(f"{row}\t{int(ep)}\t{u}\n")

    def log_data_order(epoch):
        """Rule 11: sha256 of the first 1,000 rows of the canonical epoch order, from the sampler itself, so the
        value is independent of sharding and of where a link resumed."""
        rows = [str(r) for r in train_sampler.epoch_rows(epoch)]
        digest = rt.data_order_hash(rows)
        if is_main:
            with open(os.path.join(spec.run_dir, "data_order.jsonl"), "a") as fh:
                fh.write(json.dumps({"epoch": epoch + 1, "first1000_sha256": digest, "n_rows": len(rows),
                                     "arm": arm.name}, sort_keys=True) + "\n")
        return digest

    def on_signal(signum, frame):
        stop["flag"] = True
        log(f"signal {signum}: latest_state at the next optimizer update, then exit")

    def optimizer_update(inputs, out, epoch, step, micro_done):
        """Everything after an accumulation window closes. Returns a result dict when the link must end."""
        state["global_step"] += 1
        gstep = state["global_step"]
        clip_grad_norm_(model.parameters(), spec.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["link_updates"] += 1
        if compiled:
            guard.observe(state["link_updates"], tuple(inputs["input_ids"].shape), rt.compiled_graph_count(),
                          bool(torch._dynamo.config.suppress_errors))
            if state["link_updates"] == spec.compile_warmup_steps:
                verify_compile()
        if spec.memory_probe is not None:
            reserved = int(spec.memory_probe())
        else:
            reserved = int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
        peaks = acc.gather(torch.tensor([reserved], device=device)).tolist()
        if rt.memory_abort(peaks, spec.run_dir, step=gstep, write=is_main):
            log(f"{rt.MEMORY_ABORT_MARKER}: reserved bytes per rank {peaks} > {rt.MEMORY_ABORT_BYTES}")
            raise MemoryAbort(4)
        if is_main and gstep % 10 == 0:
            log(json.dumps({"global_step": gstep, "epoch": epoch, "loss": float(out.loss.detach()),
                            "lr": scheduler.get_last_lr()[0], "alphas": _unwrap(acc, model).chain_alphas()}))
        if gstep == spec.n_pilot:
            publish_step(gstep, epoch)
        if budget is not None and gstep >= budget:
            with rt.checkpoint_io(acc):
                if is_main:
                    layout.mark_done()
            return {"done": True, "global_step": gstep, "reason": "n_pilot budget"}
        if pilot_target is not None and gstep >= pilot_target:
            # gstep is identical on every rank (same micro-batch count), so all ranks stop at this update together.
            with rt.checkpoint_io(acc):
                if is_main:
                    layout.mark_done()
            log(f"WP5 pilot: stopped at exactly {gstep} optimizer updates")
            return {"done": True, "global_step": gstep, "reason": "stop_after_updates",
                    "pilot_final": rt.step_checkpoint_dir(spec.run_dir, gstep)}
        if spec.stop_at_step is not None and gstep == spec.stop_at_step:
            stop["flag"] = True
        stop_now = bool(stop["flag"])
        if world > 1:
            stop_now = bool(acc.reduce(torch.tensor(int(stop_now), device=device), reduction="max").item())
        if (spec.save_every_n_steps and gstep % spec.save_every_n_steps == 0) or stop_now:
            save_latest(epoch, step + 1, gstep, micro_done)
        if stop_now:
            log("exiting for the job chain (no TRAINING_DONE)")
            return {"preempted": True, "global_step": gstep}
        return None

    def epoch_iter(epoch, skip):
        """One epoch of the prepared loader from per-rank batch ``skip`` on. The skip happens in the batch
        sampler (accelerate skip_first_batches), so skipped samples are never rasterised; the batch indices and
        their rank assignment are those of the uninterrupted epoch."""
        train_dl.set_epoch(epoch)
        loader = train_dl
        if skip:
            loader = acc.skip_first_batches(train_dl, skip)
            # DataLoaderShard.__iter__ calls set_epoch(self.iteration); the new loader starts at iteration 0 and
            # would silently switch the sampler back to epoch 0's order (accelerate 1.10.1).
            loader.set_epoch(epoch)
        for index, batch in enumerate(loader, start=skip):
            yield index, batch

    def require_pilot_target():
        if pilot_target is not None and state["global_step"] < pilot_target:
            raise RuntimeError(f"training ended at update {state['global_step']}, before the stop target "
                               f"{pilot_target}; raise max_epochs (no TRAINING_DONE written)")

    def train_loop():
        history = []
        micro_done = pending_micro
        for epoch in range(start_epoch, spec.max_epochs):
            total_loss = torch.zeros((), dtype=torch.float32, device=device)
            epoch_micro = 0
            order = log_data_order(epoch)
            skip = resume_step if epoch == start_epoch else 0
            for step, batch in epoch_iter(epoch, skip):
                rt.validate_b6_batch(batch)
                log_draws(batch["meta"])
                inputs = rt.model_inputs(rt.apply_arm(batch, arm))
                if compiled:
                    torch._dynamo.maybe_mark_dynamic(inputs["labels"], 1)
                out = model(**inputs)
                total_loss += out.loss.detach().float()
                acc.backward(out.loss / per_rank_acc)
                micro_done += 1
                epoch_micro += 1
                if micro_done % per_rank_acc == 0:
                    ended = optimizer_update(inputs, out, epoch, step, micro_done)
                    if ended is not None:
                        return dict(ended, history=history)
            seen = reduce_sum(float(max(1, epoch_micro)))
            losses = dict(train=reduce_sum(total_loss) / seen, **evaluate())
            gstep = state["global_step"]
            history.append(dict(epoch=epoch + 1, global_step=gstep, data_order_sha256=order, **losses))
            if is_main:
                log(json.dumps({"epoch_end": epoch + 1, "arm": arm.name, "data_order_first1000_sha256": order,
                                "evidence_channel": arm.evidence_channel, **losses}))
            is_best, should_stop = (stopper.update(epoch + 1, losses["de_novo"]) if arm.budget == "full"
                                    else (False, False))
            save_epoch(epoch + 1, gstep, micro_done, losses, is_best)
            if should_stop:
                require_pilot_target()
                with rt.checkpoint_io(acc):
                    if is_main:
                        layout.mark_done()
                return {"done": True, "global_step": gstep, "history": history, "reason": "early stop"}
        require_pilot_target()
        with rt.checkpoint_io(acc):
            if is_main:
                layout.mark_done()
        return {"done": True, "global_step": state["global_step"], "history": history, "reason": "max epochs"}

    previous = {}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, on_signal)
        except ValueError:          # not the main thread (a test driving run_b6 from a worker thread)
            pass
    try:
        result = train_loop()
        if not state["compile_verified"] and state["link_updates"]:
            verify_compile()                    # a link shorter than the warm-up is still checked
        return result
    finally:
        draw_log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# ---- CLI (integration point with WP3) ---------------------------------------------------------------------------

class WP3EpochKeys:
    """Sampler over WP3 dataset keys: one epoch of ``B6EpochSampler`` at a time, in the B5 order (rule 11).

    ``B6EpochSampler(order, epochs, start_epoch)`` yields (rn, epoch) and leaves ``order(epoch)`` to the trainer;
    ``set_epoch`` selects the epoch the prepared loader iterates, as for the B5 seedable sampler."""

    def __init__(self, sampler_cls, train_rns, seed):
        self.sampler_cls, self.train_rns, self.seed, self.epoch = sampler_cls, list(train_rns), int(seed), 0

    def order(self, epoch):
        return rt.b5_data_order(self.train_rns, self.seed, epoch)

    def epoch_rows(self, epoch):
        return [str(rn) for rn in self.order(epoch)]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self.sampler_cls(self.order, epochs=self.epoch + 1, start_epoch=self.epoch))

    def __len__(self):
        return len(self.train_rns)


def corpus_rows(corpus_path):
    """B5 train/valid lists in dataset index order (isoformDataHyena), and the C3 donor-map rows.

    The corpus schema is ``build_b5.ensure_schema`` (WP2 keeps it). geneList has no ``wid`` column: the window
    identity shared by a forward tile and its RC twin is ``gene_id`` (WP3's B6Corpus.twin_rns), selected as wid.
    ``train_weight`` is in the pinned schema; a corpus without the column (WP3's minimal fixture corpus) is read
    with the B5 COALESCE default of 1.0.
    """
    import duckdb
    con = duckdb.connect(str(corpus_path), read_only=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info('geneList')").fetchall()}
        need = {"rn", "gene_id", "start", "fin", "split", "is_rc", "gff"}
        if need - cols:
            raise ValueError(f"{corpus_path}: geneList lacks {sorted(need - cols)}")
        weight = "COALESCE(train_weight, 1.0) > 0" if "train_weight" in cols else "TRUE"
        keep = f"gff IS NOT NULL AND {weight}"
        split = {s: [r[0] for r in con.execute(f"SELECT rn FROM geneList WHERE split = ? AND {keep} ORDER BY rn",
                                                [s]).fetchall()] for s in ("train", "valid")}
        c3 = con.execute("SELECT rn, gene_id AS wid, fin - start AS tier, split, is_rc FROM geneList "
                         "WHERE split IN ('train', 'valid') ORDER BY rn").fetchall()
    finally:
        con.close()
    return split["train"], split["valid"], c3


def bind_freeze_n_pilot(freeze_path, record, train_rows, config_n_pilot):
    """N_pilot from the corpus freeze the bindings verified (its bytes are provenance["freeze"]), checked against the
    loader's train rows and the config (``rt.bind_n_pilot``; decision 18)."""
    with open(freeze_path, "rb") as fh:
        raw = fh.read()
    if hashlib.sha256(raw).hexdigest() != record["provenance"]["freeze"]:
        raise ValueError(f"{freeze_path}: freeze changed after the bindings verified it")
    return rt.bind_n_pilot(json.loads(raw), train_rows, config_n_pilot)


def _wp3():
    """WP3 modules (Writer B) and WP2 roles (Writer A). Imported by full module path; absent before the
    b6-integration merge."""
    try:
        from transgenic.datasets import b6_dataset
        from transgenic.evidence import b6_aug, b6_bindings, b6_context, b6_obs_cache, b6_roles
    except ImportError as exc:
        raise RuntimeError("the WP2/WP3 modules are not in this checkout; the B6 trainer runs only after the "
                           "b6-integration merge (plan 10)") from exc
    return b6_dataset, b6_aug, b6_bindings, b6_context, b6_obs_cache, b6_roles


class _Ref:
    """A reference FASTA for ``verify_inputs(references=...)``: the object carries its sha256 (WP3's contract)."""

    def __init__(self, path):
        self.path, self.sha256 = path, rt.file_sha256(path)


def verify_training_inputs(paths, *, roles, cache, b6_bindings, aug_seed):
    """WP3 verification of every training input, as the trainer runs it on each link. The freeze is passed as its
    FILE PATH (WP3 records its bytes and refuses a dict); the record's own record_sha256 is checked with WP3's
    canonical_record_sha256 before it enters the run identity."""
    refs = {sp: _Ref(fa) for sp, fa in sorted((paths.get("references") or {}).items())} or None
    record = b6_bindings.verify_inputs(accept=("training",), evidence=paths["evidence"], roles=roles, cache=cache,
                                       freeze=paths["freeze"], corpus=paths["corpus"], references=refs,
                                       aug_seed=aug_seed, draws=None)
    rt.identity_from_bindings(record, b6_bindings.canonical_record_sha256)       # fail closed before any training
    return record


def build_model(cfg):
    import torch
    from transgenic.model.modeling_b6 import B6TransgenicForConditionalGeneration, b6_config
    torch.manual_seed(rt.SEED)
    torch.cuda.manual_seed_all(rt.SEED)
    return B6TransgenicForConditionalGeneration(b6_config(**rt.b6_model_kwargs(cfg)))


def main(argv=None) -> Dict:
    args = rt.parse_args_b6(argv)
    cfg = rt.load_b6_config(args.config)
    cfg["_sha256"] = rt.file_sha256(args.config)
    if cfg["gate_status"] != rt.GATED_STATUS and not args.allow_ungated:
        raise SystemExit(f"REFUSED: {cfg['gate_status']} (plan 10 CP2). Pass --allow-ungated only for the "
                         "rule 15 integration test.")
    pilot = rt.resolve_wp5_pilot(cfg, args.arm, seed=args.seed, stop_after_updates=args.stop_after_updates,
                                 allow_ungated=args.allow_ungated) or {}
    arm = rt.ARMS[args.arm]
    b6_dataset, b6_aug, b6_bindings, b6_context, b6_obs_cache, b6_roles = _wp3()
    # INTEGRATION POINT (WP2/WP3 of 2026-09-24; re-check at the b6-integration merge). --bindings is a JSON manifest
    # {"evidence", "roles", "cache", "freeze", "corpus"[, "references": {species: fasta}]} of paths;
    # verify_inputs rehashes every one of them on every link, so a resume re-verifies (plan 8).
    with open(args.bindings) as fh:
        paths = json.load(fh)
    roles = b6_roles.load_roles(paths["roles"])
    cache = b6_obs_cache.ObsCache(paths["cache"])
    record = verify_training_inputs(paths, roles=roles, cache=cache, b6_bindings=b6_bindings,
                                    aug_seed=rt.AUG_SEED_BASE + args.seed)
    corpus = b6_context.B6Corpus(paths["corpus"], cache, expected_sha256=record["corpus_sha256"])
    train_rns, valid_rns, c3_rows = corpus_rows(paths["corpus"])
    n_pilot = bind_freeze_n_pilot(paths["freeze"], record, len(train_rns), cfg["n_pilot"])
    c3 = b6_aug.c3_donor_map(c3_rows, rt.AUG_SEED_BASE + args.seed) if arm.name == "C3" else None
    common = dict(arm=arm.dataset_arm, aug_seed=rt.AUG_SEED_BASE + args.seed,
                  encode_dna=b6_dataset.hyena_char_encoder(cfg["encoder_model"]),
                  tokenize_gsf=b6_dataset.gff_label_tokenizer(cfg.get("gff_vocab_version", "v3")), c3_donors=c3)
    train_ds = b6_dataset.B6Dataset(corpus, cache, mode="train_track", **common)
    eval_ds = b6_dataset.B6Dataset(corpus, cache, mode="valid_track", **common)
    spec = B6TrainSpec(arm=arm, cfg=cfg, run_dir=args.output_dir, bindings=dict(record), seed=args.seed,
                       record_sha256_fn=b6_bindings.canonical_record_sha256,
                       batch_size=args.batch_size, num_workers=args.num_workers,
                       max_epochs=args.max_epochs or int(cfg["max_epochs"]), patience=int(cfg["patience"]),
                       save_every_n_steps=args.save_every_n_steps, mixed_precision=args.mixed_precision,
                       compile_backend=None if args.compile_backend == "none" else args.compile_backend,
                       compile_warmup_steps=args.compile_warmup_steps, resume=args.resume,
                       deterministic=args.deterministic,
                       n_pilot=n_pilot, stop_after_updates=pilot.get("stop_after_updates"),
                       required_world_size=pilot.get("world_size"),
                       link={"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "argv": list(argv or sys.argv[1:])})
    return run_b6(spec, build_model(cfg), train_ds, eval_ds, collate_fn=b6_dataset.b6_collate_fn_torch,
                  train_sampler=WP3EpochKeys(b6_dataset.B6EpochSampler, train_rns, args.seed),
                  eval_sampler=[(int(rn), 0) for rn in valid_rns])


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("done") or result.get("preempted") else 1)
