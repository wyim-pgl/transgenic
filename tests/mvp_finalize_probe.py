"""Subprocess-only Accelerate CPU/Gloo probe for MVP finalization."""
from __future__ import annotations

import hashlib
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
    source_fields = {name: {"value": None, "provenance": "config-default"}
                     for name in finalize._AUDIT_SEMANTIC_FIELDS}
    target_fields = {name: {"value": None, "provenance": "target-config"}
                     for name in finalize._AUDIT_SEMANTIC_FIELDS}
    source_fields["evidence_channels"]["value"] = 0
    target_fields["evidence_channels"]["value"] = 8
    digest = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = {
        "schema_version": "b6-weights-init-v1", "mode": "weights-only",
        "requested_path": "/tmp/source/epoch", "resolved_checkpoint_dir": "/tmp/source/epoch",
        "resolved_weights_path": "/tmp/source/epoch/model.safetensors",
        "files": {
            "model.safetensors": {"path": "/tmp/source/epoch/model.safetensors",
                "before": {"sha256": SHA, "size": 1, "device": 1, "inode": 1, "mtime_ns": 1},
                "after": {"sha256": SHA, "size": 1, "device": 1, "inode": 1, "mtime_ns": 1}},
            "run_config.json": {"path": "/tmp/source/run_config.json",
                "before": {"sha256": SHA, "size": 1, "device": 1, "inode": 1, "mtime_ns": 1},
                "after": {"sha256": SHA, "size": 1, "device": 1, "inode": 1, "mtime_ns": 1}},
            "runtime_identity.txt": None, "config.json": None, "meta.json": None,
            "runtime.json": None},
        "recipe": {"canonical_sha256": SHA},
        "source_semantics": {"fields": source_fields, "sha256": digest(source_fields)},
        "target_semantics": {"fields": target_fields, "sha256": digest(target_fields)},
        "implementation": {
            "b5_runtime.py": {"path": "/tmp/source/b5_runtime.py", "sha256": SHA},
            "configuration_transgenic.py": {"path": "/tmp/source/configuration_transgenic.py", "sha256": SHA}},
        "compatibility": {"transition": "evidence_channels:0->8",
            "exception": "exactly missing transgenic.encoder.evidence_proj.weight; initialized to verified zeros",
            "same8_supported": False},
        "load": {"missing_keys": ["transgenic.encoder.evidence_proj.weight"],
                 "unexpected_keys": [], "safetensors_metadata": None},
        "provenance_caveat": "recorded loader output; source bytes are not re-certified here"}
    recipe = {"max_updates": 2, "warmup_updates": 1, "batch_size": 1,
              "accumulation_steps": 1, "required_world_size": 2,
              "backbone_lr": 2e-5, "projection_lr": 2e-4,
              "weight_decay": 0.02, "clip_grad_norm": 1.0}
    return {
        "identity_scope": "test",
        "initialization_audit": {"audit": audit,
            "audit_sha256": hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
            "verification": "recorded"},
        "data_identities": {name: dict(identity) for name in ("corpus", "evidence", "panel")},
        "sif_code_identity": {"sif": dict(identity), "code": dict(identity)},
        "runtime_packages": {name: "test-version" for name in
                             ("python", "torch", "transformers", "accelerate", "safetensors")},
        "normalized_recipe": recipe,
        "seed_augmentation": {"seed": 1, "aug_seed": 20260915},
        "runtime_hardware": {
            "measurement": "runtime_observation", "test_hardware": True,
            "world_size": 2, "distributed_type": "MULTI_CPU", "device_type": "cpu",
            "distinct_device_uuids": False,
            "ranks": [{"rank": rank, "local_rank": rank, "world_size": 2,
                       "distributed_type": "MULTI_CPU", "device_type": "cpu",
                       "device_index": None, "current_cuda_device": None,
                       "cuda_available": False, "cuda_device_count": 0,
                       "device_name": None, "device_uuid": None, "capability": None,
                       "bf16_supported": False, "torch_version": str(torch.__version__),
                       "torch_cuda_version": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version()}
                      for rank in range(2)],
        },
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
    if accelerator.process_index == 1:
        if case == "inventory-noniterable":
            expected = object()
        elif case == "inventory-throwing":
            class ThrowingIterable:
                def __iter__(self):
                    raise AssertionError("inventory must not be consumed")
            expected = ThrowingIterable()
        elif case == "inventory-infinite":
            import itertools
            expected = (value for value in itertools.count())
        elif case == "inventory-subclass":
            class HostileList(list):
                def __iter__(self):
                    raise AssertionError("list subclass must not be consumed")
            expected = HostileList(range(size))
    final = Path(os.environ["MVP_FINALIZE_TEST_DIR"]) / "final"
    payload_calls = 0
    if case == "byte-cap":
        finalize.MVP_MAX_SERIALIZED_BYTES = 1_000_000
        original_metric = finalize.paired_metric_record
        def padded_metric(*args, **kwargs):
            record = original_metric(*args, **kwargs)
            record["padding"] = "x" * 400_000
            return record
        finalize.paired_metric_record = padded_metric
        def forbidden_payload(*args, **kwargs):
            nonlocal payload_calls
            payload_calls += 1
            raise AssertionError("payload collective must not run")
        finalize._gather_record_payload_bytes = forbidden_payload
    if accelerator.is_main_process:
        if case == "durability":
            finalize._fsync_parent = lambda path: (_ for _ in ()).throw(OSError("injected fsync"))
        elif case == "publish-write-fail":
            finalize._write_payload_files = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected write"))
        elif case == "publish-rename-fail":
            finalize._rename_noreplace = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected rename"))
    object_calls = 0
    real_all_gather_object = torch.distributed.all_gather_object
    real_broadcast_object_list = torch.distributed.broadcast_object_list
    if case == "huge-status":
        def forbidden_object_collective(*args, **kwargs):
            nonlocal object_calls
            object_calls += 1
            raise AssertionError("status must not use an object collective")
        torch.distributed.all_gather_object = forbidden_object_collective
        torch.distributed.broadcast_object_list = forbidden_object_collective
        if accelerator.process_index == 1:
            finalize._validate_evaluation_inputs = lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("x" * 100_000))
    failed = committed = False
    error_info = None
    result = None
    try:
        result = evaluate_and_publish_final(accelerator, model, loader, expected, final,
            metadata=metadata(), training_summary={"successful_updates": 2, "effective_batch_size": 2})
    except PublicationDurabilityError as error:
        failed, committed = True, error.committed
        error_info = (type(error).__name__, str(error), error.committed)
    except (ValueError, RuntimeError, OSError) as error:
        failed = True
        error_info = (type(error).__name__, str(error), False)
    finally:
        torch.distributed.all_gather_object = real_all_gather_object
        torch.distributed.broadcast_object_list = real_broadcast_object_list
    flags = accelerator.gather(torch.tensor([int(failed), int(committed)], device=accelerator.device))
    summaries = [None] * 2
    errors = [None] * 2
    error_cases = {"publish-write-fail", "publish-rename-fail", "inventory-noniterable",
                   "inventory-throwing", "inventory-infinite", "inventory-subclass", "huge-status"}
    if case in error_cases:
        torch.distributed.all_gather_object(errors, error_info)
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
        elif case == "byte-cap":
            print(json.dumps({"case": case, "all_failed": bool(flags[0::2].sum().item() == 2),
                              "payload_calls": payload_calls, "final_exists": final.exists()}, sort_keys=True))
        elif case in error_cases:
            output = {"case": case, "all_failed": bool(flags[0::2].sum().item() == 2),
                      "same_error": errors[0] == errors[1], "error": errors[0],
                      "final_exists": final.exists()}
            if case == "huge-status":
                output["object_calls"] = object_calls
            print(json.dumps(output, sort_keys=True))
        else:
            print(json.dumps({"case": case, "all_failed": bool(flags[0::2].sum().item() == 2),
                              "final_exists": final.exists()}, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
