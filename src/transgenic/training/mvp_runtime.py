"""Minimal fixed-update MVP training runtime.

Ownership contract: the caller constructs an Accelerator with ``make_accelerator``
(or identical settings), prepares only the model and optimizer, and supplies an
unprepared DataLoader.  Under DDP that loader must use one DistributedSampler.
Only DNA, attention mask, labels, and evidence tracks are moved to the device.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.utils.data.distributed import DistributedSampler


_PROJECTION_NAME = "transgenic.encoder.evidence_proj.weight"


@dataclass(frozen=True)
class MVPRecipe:
    max_updates: int = 250
    warmup_updates: int = 12
    batch_size: int = 1
    accumulation_steps: int = 24
    required_world_size: int = 4
    backbone_lr: float = 2e-5
    projection_lr: float = 2e-4
    weight_decay: float = 0.02
    clip_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_updates", "batch_size", "accumulation_steps", "required_world_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.warmup_updates, bool) or not isinstance(self.warmup_updates, int):
            raise TypeError("warmup_updates must be an integer")
        if not 0 < self.warmup_updates < self.max_updates:
            raise ValueError("warmup_updates must be between zero and max_updates")
        for name in ("backbone_lr", "projection_lr", "clip_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.weight_decay, bool) or
                not isinstance(self.weight_decay, (int, float))):
            raise TypeError("weight_decay must be numeric")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def make_accelerator(recipe: MVPRecipe, **kwargs):
    """Construct an Accelerator with the fixed runtime's required DDP/accumulation semantics."""
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    handlers = list(kwargs.pop("kwargs_handlers", []) or [])
    ddp_handlers = [handler for handler in handlers
                    if isinstance(handler, DistributedDataParallelKwargs)]
    if any(handler.find_unused_parameters is not True for handler in ddp_handlers):
        raise ValueError("DDP handler must set find_unused_parameters=True")
    if not ddp_handlers:
        handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    plugin = GradientAccumulationPlugin(
        num_steps=recipe.accumulation_steps,
        sync_with_dataloader=False,
        adjust_scheduler=False,
    )
    return Accelerator(
        gradient_accumulation_plugin=plugin,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=handlers,
        **kwargs,
    )


def build_optimizer(model: torch.nn.Module, recipe: MVPRecipe) -> torch.optim.AdamW:
    """Build exhaustive, uniform-decay AdamW groups with one exact projection parameter."""
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    projection = [(name, parameter) for name, parameter in named if name == _PROJECTION_NAME]
    if len(projection) != 1:
        raise ValueError(f"expected exactly one trainable {_PROJECTION_NAME}, found {len(projection)}")
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("duplicate trainable parameters are not allowed")
    projection_parameter = projection[0][1]
    backbone = [parameter for name, parameter in named if name != _PROJECTION_NAME]
    if not backbone:
        raise ValueError("backbone optimizer group is empty")
    groups = [
        {"name": "backbone", "params": backbone, "lr": recipe.backbone_lr,
         "weight_decay": recipe.weight_decay},
        {"name": "projection", "params": [projection_parameter], "lr": recipe.projection_lr,
         "weight_decay": recipe.weight_decay},
    ]
    grouped = [parameter for group in groups for parameter in group["params"]]
    if {id(parameter) for parameter in grouped} != {id(parameter) for _, parameter in named}:
        raise ValueError("optimizer groups leave trainable parameters unassigned")
    return torch.optim.AdamW(groups)


def lr_scale(completed_updates: int, recipe: MVPRecipe) -> float:
    """LambdaLR scale for ``completed_updates`` successful updates (post-step convention)."""
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError("completed_updates must be an integer")
    if completed_updates < 0:
        raise ValueError("completed_updates must be nonnegative")
    if completed_updates < recipe.warmup_updates:
        return completed_updates / recipe.warmup_updates
    if completed_updates < recipe.max_updates:
        return ((recipe.max_updates - completed_updates) /
                (recipe.max_updates - recipe.warmup_updates))
    return 0.0


def _collective_true(accelerator, condition: bool) -> bool:
    value = torch.tensor(int(condition), device=accelerator.device)
    return int(accelerator.reduce(value, reduction="sum").item()) == accelerator.num_processes


def _all_ints(accelerator, value: int) -> list[int]:
    tensor = torch.tensor([value], dtype=torch.int64, device=accelerator.device)
    return [int(item) for item in accelerator.gather(tensor).cpu().tolist()]


def _set_lrs(optimizer, recipe: MVPRecipe, completed_updates: int) -> None:
    scale = lr_scale(completed_updates, recipe)
    for group, peak in zip(optimizer.param_groups, (recipe.backbone_lr, recipe.projection_lr)):
        group["lr"] = peak * scale


def _step_was_skipped(optimizer) -> bool:
    """Narrow seam for exercising the AcceleratedOptimizer overflow result."""
    return bool(getattr(optimizer, "step_was_skipped", False))


def _validate_accelerator(accelerator, model, optimizer, train_loader, recipe: MVPRecipe) -> None:
    settings = accelerator.gradient_state.plugin_kwargs
    expected = {"num_steps": recipe.accumulation_steps, "sync_with_dataloader": False,
                "adjust_scheduler": False}
    errors = []
    if any(settings.get(key) != value for key, value in expected.items()):
        errors.append(f"invalid gradient accumulation settings; expected {expected}")
    if accelerator.step_scheduler_with_optimizer is not False:
        errors.append("step_scheduler_with_optimizer must be False")
    if accelerator.num_processes != recipe.required_world_size:
        errors.append(f"world size {accelerator.num_processes} != required {recipe.required_world_size}")
    if getattr(accelerator, "_dataloaders", []):
        errors.append("caller must not prepare the dataloader")
    if optimizer not in getattr(accelerator, "_optimizers", []):
        errors.append("caller must prepare the optimizer")
    prepared_models = getattr(accelerator, "_models", [])
    unwrapped = accelerator.unwrap_model(model)
    if model not in prepared_models and unwrapped not in prepared_models:
        errors.append("caller must prepare the model")
    ddp_handler = getattr(accelerator, "ddp_handler", None)
    if ddp_handler is None or ddp_handler.find_unused_parameters is not True:
        errors.append("Accelerator DDP handler must set find_unused_parameters=True")
    if accelerator.num_processes > 1:
        from torch.nn.parallel import DistributedDataParallel

        if (not isinstance(model, DistributedDataParallel) or
                model.find_unused_parameters is not True):
            errors.append("prepared DDP model must set find_unused_parameters=True")
        sampler = getattr(train_loader, "sampler", None)
        if (not isinstance(sampler, DistributedSampler) or
                sampler.num_replicas != accelerator.num_processes or
                sampler.rank != accelerator.process_index):
            errors.append("DDP loader must have one rank-matched DistributedSampler")

    groups = optimizer.param_groups
    if len(groups) != 2 or [group.get("name") for group in groups] != ["backbone", "projection"]:
        errors.append("optimizer groups must be exactly ordered backbone, projection")
    else:
        named = [(name, parameter) for name, parameter in unwrapped.named_parameters()
                 if parameter.requires_grad]
        projections = [parameter for name, parameter in named if name == _PROJECTION_NAME]
        backbone_params = groups[0].get("params")
        projection_params = groups[1].get("params")
        if not isinstance(backbone_params, (list, tuple)):
            errors.append("backbone optimizer group has no parameter list")
            backbone_params = []
        if not isinstance(projection_params, (list, tuple)):
            errors.append("projection optimizer group has no parameter list")
            projection_params = []
        grouped = list(backbone_params) + list(projection_params)
        grouped_ids = [id(parameter) for parameter in grouped]
        trainable_ids = {id(parameter) for _, parameter in named}
        if (len(projections) != 1 or len(projection_params) != 1 or
                projection_params[0] is not projections[0]):
            errors.append(f"projection group must contain exactly {_PROJECTION_NAME}")
        if (len(grouped_ids) != len(set(grouped_ids)) or
                set(grouped_ids) != trainable_ids):
            errors.append("optimizer groups must be disjoint and exhaustive")
        expected_backbone = {id(parameter) for name, parameter in named
                             if name != _PROJECTION_NAME}
        if {id(parameter) for parameter in backbone_params} != expected_backbone:
            errors.append("backbone optimizer group contains unrelated or missing parameters")
        if any(group.get("weight_decay") != recipe.weight_decay for group in groups):
            errors.append("optimizer weight decay does not match recipe")
    if not _collective_true(accelerator, not errors):
        detail = "; ".join(errors) if errors else "invalid setting on another rank"
        raise ValueError(f"Accelerator ownership/settings invalid: {detail}")


def _batch_parts(batch: Any, recipe: MVPRecipe):
    if not isinstance(batch, (tuple, list)) or len(batch) != 10:
        raise ValueError("training requires a collated 10-tuple batch")
    input_ids, attention_mask, labels, tracks = batch[0], batch[1], batch[2], batch[8]
    for name, value in (("DNA", input_ids), ("attention mask", attention_mask),
                        ("labels", labels), ("evidence tracks", tracks)):
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError(f"{name} must be a batched tensor")
        if value.shape[0] != recipe.batch_size:
            raise ValueError(f"{name} batch size must equal {recipe.batch_size}")
    if (tracks.ndim != 3 or input_ids.ndim < 2 or tracks.shape != (*input_ids.shape, 8) or
            not tracks.is_floating_point()):
        raise ValueError("evidence tracks must be floating point (batch, DNA length, 8)")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention mask must match DNA shape")
    return input_ids, attention_mask, labels, tracks


def run_fixed_updates(accelerator, model, optimizer, train_loader, recipe: MVPRecipe) -> dict:
    """Run exactly ``max_updates`` successful synchronized optimizer updates."""
    _validate_accelerator(accelerator, model, optimizer, train_loader, recipe)
    try:
        local_loader_length = len(train_loader)
    except TypeError as error:
        raise ValueError("train loader must have a finite length") from error
    loader_lengths = _all_ints(accelerator, local_loader_length)
    if any(length <= 0 for length in loader_lengths) or len(set(loader_lengths)) != 1:
        raise ValueError(f"per-rank loader lengths must be equal and nonempty: {loader_lengths}")

    model.train()
    optimizer.zero_grad()
    _set_lrs(optimizer, recipe, 0)
    completed = microbatches = epoch = 0
    lr_trace = []
    if len(set(_all_ints(accelerator, completed))) != 1:
        raise RuntimeError("startup successful-update counters diverged")

    while completed < recipe.max_updates:
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in train_loader:
            with accelerator.accumulate(model):
                try:
                    parts = _batch_parts(batch, recipe)
                    batch_valid = True
                except (TypeError, ValueError):
                    parts, batch_valid = None, False
                if not _collective_true(accelerator, batch_valid):
                    raise ValueError("invalid batch contract on at least one rank")
                input_ids, attention_mask, labels, tracks = (
                    value.to(accelerator.device) for value in parts
                )
                output = model(input_ids, attention_mask=attention_mask, labels=labels,
                               evidence_tracks=tracks)
                loss = getattr(output, "loss", None)
                loss_finite = (isinstance(loss, torch.Tensor) and loss.numel() == 1 and
                               bool(torch.isfinite(loss.detach()).item()))
                if not _collective_true(accelerator, loss_finite):
                    raise FloatingPointError("nonfinite loss on at least one rank")
                accelerator.backward(loss)
                microbatches += 1

                if accelerator.sync_gradients:
                    gradients_finite = all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()
                    )
                    if not _collective_true(accelerator, gradients_finite):
                        optimizer.zero_grad()
                        raise FloatingPointError("nonfinite gradients on at least one rank")
                    accelerator.clip_grad_norm_(model.parameters(), recipe.clip_grad_norm)
                    used_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                    optimizer.step()
                    skipped = not _collective_true(accelerator, not _step_was_skipped(optimizer))
                    optimizer.zero_grad()
                    if skipped:
                        raise RuntimeError("optimizer step skipped on at least one rank")
                    completed += 1
                    lr_trace.append({"update": completed, "backbone": used_lrs[0],
                                     "projection": used_lrs[1]})
                    if len(set(_all_ints(accelerator, completed))) != 1:
                        raise RuntimeError("successful-update counters diverged")
                    _set_lrs(optimizer, recipe, completed)
                    if completed == recipe.max_updates:
                        break
        if completed < recipe.max_updates:
            epoch += 1

    microbatch_counts = _all_ints(accelerator, microbatches)
    if len(set(_all_ints(accelerator, completed))) != 1:
        raise RuntimeError("final successful-update counters diverged")
    return {
        "successful_updates": completed,
        "microbatches_per_rank": microbatch_counts,
        "epochs_crossed": epoch,
        "effective_batch_size": (recipe.batch_size * recipe.accumulation_steps *
                                 accelerator.num_processes),
        "lr_trace": lr_trace,
        "final_lrs": [float(group["lr"]) for group in optimizer.param_groups],
    }
