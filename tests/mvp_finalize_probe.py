"""Subprocess-only Accelerate CPU/Gloo probe for MVP finalization."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transgenic.training.mvp_finalize import PublicationDurabilityError, evaluate_and_publish_final
from transgenic.training.mvp_runtime import MVPRecipe, make_accelerator
import transgenic.training.mvp_finalize as finalize

SHA = "1" * 64


class Config:
    def to_dict(self):
        return {"model_type": "paired-toy", "vocab_size": 273}


class PairedDataset(Dataset):
    def __init__(self, size, case):
        self.size, self.case = size, case

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rn = index
        if self.case == "extra" and index == self.size - 1:
            rn = 99
        if self.case == "inconsistent" and index == 0:
            # The sampler pads rn=0 across ranks; rank-dependent corruption happens in model-independent metadata later.
            rn = 0
        ids = torch.tensor([index % 3], dtype=torch.long)
        mask = torch.ones(1, dtype=torch.bool)
        labels = torch.tensor([2], dtype=torch.long)
        tracks = torch.full((1, 8), float(index + 1))
        return (ids, mask, labels, index, "chr", 0, 1, torch.tensor(1.0), tracks,
                {"rn": rn, "has_evidence": True})


class Model(torch.nn.Module):
    def __init__(self, rank=0, inconsistent=False):
        super().__init__()
        self.shared = torch.nn.Embedding(273, 2)
        self.alias = self.shared
        self.scale = torch.nn.Parameter(torch.tensor(0.05))
        self.config = Config()
        self.rank = rank
        self.inconsistent = inconsistent
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None, evidence_tracks=None):
        self.forward_calls += 1
        signal = evidence_tracks.float().mean(dim=(1, 2)) * self.scale
        if self.inconsistent and self.rank == 1:
            signal = signal + 0.25
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 273), device=input_ids.device)
        logits[..., 2] = signal[:, None]
        loss = -logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).mean()
        return type("Output", (), {"logits": logits, "loss": loss})()


def metadata():
    identity = {"sha256": SHA, "verification": "recorded"}
    return {
        "identity_scope": "test",
        "initialization_audit": {"report_sha256": SHA, "verification": "recorded"},
        "data_identities": {name: dict(identity) for name in ("corpus", "evidence", "panel")},
        "sif_code_identity": {"sif": dict(identity), "code": dict(identity)},
        "runtime_packages": {"torch": torch.__version__, "accelerate": "0.34.2"},
        "normalized_recipe": {"batch_size": 1},
        "seed_augmentation": {"seed": 7, "augmentation": "fixed"},
    }


def main(case):
    accelerator = make_accelerator(MVPRecipe(max_updates=2, warmup_updates=1,
        accumulation_steps=2, required_world_size=2), cpu=True)
    size = int(case.split("-")[1]) if case.startswith("success-") else 3
    dataset = PairedDataset(size, case)
    sampler_rank = 0 if case == "bad_sampler" and accelerator.process_index == 1 else accelerator.process_index
    sampler = DistributedSampler(dataset, num_replicas=2, rank=sampler_rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    model = Model(accelerator.process_index, inconsistent=case == "inconsistent")
    backend_calls = {"count": 0}
    def backend(graph, example_inputs):
        backend_calls["count"] += 1
        return graph.forward
    model = torch.compile(model, backend=backend)
    model = accelerator.prepare(model)
    if accelerator.process_index == 1:
        accelerator.unwrap_model = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-main rank must not unwrap/save"))
    expected = list(range(size))
    if case == "missing":
        expected.append(77)
    final = Path(os.environ["MVP_FINALIZE_TEST_DIR"]) / "final"
    if case == "durability" and accelerator.is_main_process:
        finalize._fsync_parent = lambda path: (_ for _ in ()).throw(OSError("injected fsync"))
    failed = committed = False
    result = None
    try:
        result = evaluate_and_publish_final(accelerator, model, loader, expected, final,
            metadata=metadata(), training_summary={"successful_updates": 2, "effective_batch_size": 2})
    except PublicationDurabilityError as error:
        failed, committed = True, error.committed
    except (ValueError, RuntimeError, FileExistsError):
        failed = True
    flags = accelerator.gather(torch.tensor([int(failed), int(committed)], device=accelerator.device))
    summaries = [None] * 2
    if case.startswith("success-"):
        torch.distributed.all_gather_object(summaries, result)
    if accelerator.is_main_process:
        if case.startswith("success-"):
            target = accelerator.unwrap_model(model, keep_fp32_wrapper=True)._orig_mod
            clean = all(not key.startswith(("module.", "_orig_mod."))
                        for key in safe_open(final / "model.safetensors", framework="pt").keys())
            restored = Model()
            missing, unexpected = load_model(restored, final / "model.safetensors")
            print(json.dumps({"world_size": 2, "rns": [row["rn"] for row in result["records"]],
                "pair_count": result["pair_count"], "same_summary": summaries[0] == summaries[1],
                "primary_delta": result["primary_delta"], "composite_delta": result["composite_delta"],
                "record_fields": sorted(result["records"][0]), "backend_calls": backend_calls["count"],
                "compiled_forwards": target.forward_calls, "saved_keys_clean": clean,
                "tied_reload": not missing and not unexpected and restored.shared is restored.alias,
                "marker": (final / "TRAINING_DONE").is_file()}, sort_keys=True))
        elif case == "durability":
            print(json.dumps({"case": case, "all_committed": bool(flags[1::2].sum().item() == 2),
                              "marker": (final / "TRAINING_DONE").is_file()}, sort_keys=True))
        else:
            print(json.dumps({"case": case, "all_failed": bool(flags[0::2].sum().item() == 2),
                              "final_exists": final.exists()}, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
