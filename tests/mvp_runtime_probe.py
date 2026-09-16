"""Subprocess-only actual Accelerate CPU/Gloo probe for test_mvp_runtime.py."""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import transgenic.training.mvp_runtime as runtime
from transgenic.training.mvp_runtime import MVPRecipe, build_optimizer, make_accelerator, run_fixed_updates


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = torch.tensor([float(index % 7 + 1) / 7])
        mask = torch.ones(1, dtype=torch.bool)
        label = torch.tensor([float((index * 3) % 5) / 5])
        tracks = torch.full((1, 8), float(index % 3 - 1))
        return (x, mask, label, index, "chr", 0, 1, torch.tensor(1.0), tracks,
                {"rn": index})


class RecordingSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        super().set_epoch(epoch)


class ToyModel(torch.nn.Module):
    def __init__(self, mode="normal", rank=0):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1, bias=False)
        self.transgenic = torch.nn.Module()
        self.transgenic.encoder = torch.nn.Module()
        self.transgenic.encoder.evidence_proj = torch.nn.Linear(8, 1, bias=False)
        self.mode = mode
        self.rank = rank
        self.forward_calls = 0
        with torch.no_grad():
            self.backbone.weight.fill_(0.3)
            self.transgenic.encoder.evidence_proj.weight.fill_(-0.025)

    def forward(self, input_ids, attention_mask=None, labels=None, evidence_tracks=None):
        assert attention_mask is not None and evidence_tracks is not None
        self.forward_calls += 1
        prediction = (self.backbone(input_ids.float().unsqueeze(-1)).squeeze(-1) +
                      self.transgenic.encoder.evidence_proj(evidence_tracks.float()).squeeze(-1))
        loss = (prediction - labels.float()).square().mean()
        if self.mode == "nan_loss" and self.rank == 1 and self.forward_calls == 2:
            loss = loss * torch.tensor(float("nan"), device=loss.device)
        if self.mode == "nan_gradient" and self.rank == 1 and self.forward_calls == 2:
            loss.register_hook(lambda gradient: gradient * float("nan"))
        return SimpleNamespace(loss=loss)


def loader_for(accelerator, per_rank_length, *, invalid=False):
    dataset_size = per_rank_length * accelerator.num_processes
    dataset = ToyDataset(dataset_size)
    sampler = RecordingSampler(dataset, num_replicas=accelerator.num_processes,
                               rank=accelerator.process_index, shuffle=True, seed=123)
    if invalid:
        class InvalidDataset(ToyDataset):
            def __getitem__(self, index):
                return super().__getitem__(index)[:-1]
        dataset = InvalidDataset(dataset_size)
        sampler = RecordingSampler(dataset, num_replicas=accelerator.num_processes,
                                   rank=accelerator.process_index, shuffle=True, seed=123)
    worker_rng = torch.Generator().manual_seed(456 + accelerator.process_index)
    return DataLoader(dataset, batch_size=1, sampler=sampler, generator=worker_rng), sampler


def parameter_vector(model):
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def reference_core(world_size, recipe):
    model = ToyModel()
    optimizer = build_optimizer(model, recipe)
    runtime._set_lrs(optimizer, recipe, 0)
    streams = []
    for rank in range(world_size):
        indices = []
        epoch = 0
        while len(indices) < recipe.max_updates * recipe.accumulation_steps:
            dataset = ToyDataset(5 * world_size)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=True, seed=123)
            sampler.set_epoch(epoch)
            indices.extend(list(sampler))
            epoch += 1
        streams.append(indices)
    for update in range(recipe.max_updates):
        optimizer.zero_grad()
        for micro in range(recipe.accumulation_steps):
            for rank in range(world_size):
                index = streams[rank][update * recipe.accumulation_steps + micro]
                item = ToyDataset(5 * world_size)[index]
                output = model(item[0].unsqueeze(0), attention_mask=item[1].unsqueeze(0),
                               labels=item[2].unsqueeze(0), evidence_tracks=item[8].unsqueeze(0))
                (output.loss / (recipe.accumulation_steps * world_size)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip_grad_norm)
        optimizer.step()
        runtime._set_lrs(optimizer, recipe, update + 1)
    return parameter_vector(model)


def synchronized(accelerator, vector):
    gathered = accelerator.gather(vector.to(accelerator.device)).reshape(accelerator.num_processes, -1)
    return bool(torch.equal(gathered, gathered[0].expand_as(gathered)))


def run_success(case):
    literal = case == "literal"
    recipe = (MVPRecipe() if literal else
              MVPRecipe(max_updates=3, warmup_updates=1, accumulation_steps=4,
                        required_world_size=int(os.environ.get("WORLD_SIZE", "1")),
                        backbone_lr=0.03, projection_lr=0.05, weight_decay=0.02,
                        clip_grad_norm=0.25))
    accelerator = make_accelerator(recipe, cpu=True)
    model = ToyModel(rank=accelerator.process_index)
    optimizer = build_optimizer(model, recipe)
    step_count = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    model, optimizer = accelerator.prepare(model, optimizer)
    loader, sampler = loader_for(accelerator, 5 if not literal else 7)
    clip_count = 0
    original_clip = accelerator.clip_grad_norm_

    def counted_clip(*args, **kwargs):
        nonlocal clip_count
        clip_count += 1
        return original_clip(*args, **kwargs)

    accelerator.clip_grad_norm_ = counted_clip
    summary = run_fixed_updates(accelerator, model, optimizer, loader, recipe)
    unwrapped = accelerator.unwrap_model(model)
    vector = parameter_vector(unwrapped)
    summary.update({
        "world_size": accelerator.num_processes,
        "distributed_type": accelerator.distributed_type.name,
        "sampler_epochs": sampler.epochs,
        "optimizer_steps": step_count,
        "clip_calls": clip_count,
        "forward_calls": unwrapped.forward_calls,
        "parameters_synchronized": synchronized(accelerator, vector),
        "analytic_match": (bool(torch.allclose(vector, reference_core(accelerator.num_processes, recipe),
                                                rtol=2e-6, atol=2e-7))
                           if not literal else None),
    })
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(json.dumps(summary, sort_keys=True))


def run_failure(case):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    recipe = MVPRecipe(max_updates=2, warmup_updates=1, accumulation_steps=2,
                       required_world_size=world)
    if case == "invalid_settings":
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True)
    else:
        accelerator = make_accelerator(recipe, cpu=True)
    mode = case if case in {"nan_loss", "nan_gradient"} else "normal"
    model = ToyModel(mode=mode, rank=accelerator.process_index)
    optimizer = build_optimizer(model, recipe)
    if case != "unprepared_components":
        model, optimizer = accelerator.prepare(model, optimizer)
    if case == "empty":
        loader, _ = loader_for(accelerator, 0)
    elif case == "unequal":
        local_size = 8 if accelerator.process_index == 0 else 10
        dataset = ToyDataset(local_size)
        sampler = RecordingSampler(dataset, num_replicas=world, rank=accelerator.process_index,
                                   shuffle=False)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    else:
        loader, _ = loader_for(accelerator, 3, invalid=case == "invalid_batch")
    if case == "sampler_mismatch":
        dataset = ToyDataset(6 * world)
        wrong_rank = 0 if accelerator.process_index == 1 else accelerator.process_index
        sampler = RecordingSampler(dataset, num_replicas=world, rank=wrong_rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    if case == "prepared_loader":
        loader = accelerator.prepare(loader)
    if case == "skipped":
        runtime._step_was_skipped = lambda _: True
    rejected = False
    try:
        run_fixed_updates(accelerator, model, optimizer, loader, recipe)
    except (ValueError, RuntimeError, FloatingPointError):
        rejected = True
    flags = accelerator.gather(torch.tensor([int(rejected)], device=accelerator.device))
    all_rejected = bool(flags.sum().item() == world)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(json.dumps({"case": case, "all_ranks_rejected": all_rejected}))
    if not all_rejected:
        raise SystemExit(2)


if __name__ == "__main__":
    selected = sys.argv[1]
    if selected in {"core", "literal"}:
        run_success(selected)
    else:
        run_failure(selected)
