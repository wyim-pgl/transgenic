"""Paired MVP evaluation and strict atomic final-artifact publication.

The distributed contract is replicated DDP only.  It coordinates ordinary Python
failures, but rank death, communicator failure, and asymmetric DDP-forward failure
remain launcher/process-group-timeout concerns.
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict
import errno
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from . import weights_init as _weights_init
from .mvp_metrics import aggregate_paired_records, paired_metric_record
from .mvp_runtime import MVPRecipe

MVP_MAX_RECORDS = 100_000
MVP_MAX_SERIALIZED_BYTES = 64 * 1024 * 1024
MVP_STATUS_MESSAGE_MAX_BYTES = 2048
MVP_STATUS_PATH_MAX_BYTES = 4096
_STATUS_TRUNCATION_MARKER = "...[diagnostic truncated]"
_SHA256_LENGTH = 64
_REQUIRED_METADATA = {
    "identity_scope", "initialization_audit", "data_identities", "sif_code_identity",
    "runtime_packages", "normalized_recipe", "seed_augmentation", "runtime_hardware",
}
_IDENTITY_VERIFICATIONS = {"recorded", "independently_verified"}
_RUNTIME_PACKAGES = {"python", "torch", "transformers", "accelerate", "safetensors"}
_PROJECTION_KEY = _weights_init._PROJECTION_KEY
_AUDIT_SEMANTIC_FIELDS = set(_weights_init._SEMANTIC_FIELDS) - {
    "layer_norm_eps", "normalize_before", "scale_embedding",
}
_AUDIT_SNAPSHOT_FIELDS = {"sha256", "size", "device", "inode", "mtime_ns"}


class PublicationDurabilityError(OSError):
    """The final name is committed, but parent-directory durability is uncertain."""

    def __init__(self, message: str, final_dir: Path | str):
        super().__init__(message)
        self.committed = True
        self.final_dir = Path(final_dir)


def _canonical_json(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("metadata and summaries must be finite JSON values") from error


def _identity(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} identity must be an object")
    digest = value.get("sha256")
    if (not isinstance(digest, str) or len(digest) != _SHA256_LENGTH or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{name} SHA256 must be 64 lowercase hexadecimal characters")
    if value.get("verification") not in _IDENTITY_VERIFICATIONS:
        raise ValueError(f"{name} verification must distinguish recorded from independently_verified")


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == _SHA256_LENGTH and
            all(character in "0123456789abcdef" for character in value))


def _validate_initialization_audit(initialization: Any) -> None:
    if not isinstance(initialization, dict) or initialization.get("verification") not in _IDENTITY_VERIFICATIONS:
        raise ValueError("initialization_audit requires recorded or independently_verified provenance")
    audit = initialization.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("initialization_audit must embed the load_weights_only audit")
    digest = initialization.get("audit_sha256")
    if not _is_sha256(digest) or digest != hashlib.sha256(_canonical_json(audit)).hexdigest():
        raise ValueError("initialization_audit audit_sha256 does not bind the embedded audit")
    required = {"schema_version", "mode", "requested_path", "resolved_checkpoint_dir",
                "resolved_weights_path", "files", "recipe", "source_semantics",
                "target_semantics", "implementation", "compatibility", "load",
                "provenance_caveat"}
    if set(audit) != required or audit.get("schema_version") != _weights_init._SCHEMA_VERSION or audit.get("mode") != "weights-only":
        raise ValueError("initialization_audit is not the complete load_weights_only audit schema")
    if not isinstance(audit.get("requested_path"), str) or not audit["requested_path"]:
        raise ValueError("initialization_audit requested_path must be recorded")
    if not isinstance(audit.get("provenance_caveat"), str) or not audit["provenance_caveat"]:
        raise ValueError("initialization_audit provenance_caveat must be recorded")
    checkpoint = audit.get("resolved_checkpoint_dir")
    weights = audit.get("resolved_weights_path")
    if (not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute() or
            not isinstance(weights, str) or not Path(weights).is_absolute() or
            Path(weights).parent != Path(checkpoint)):
        raise ValueError("initialization_audit requires concrete resolved checkpoint and weights paths")
    files = audit.get("files")
    file_names = {"model.safetensors", "run_config.json", "runtime_identity.txt",
                  "config.json", "meta.json", "runtime.json"}
    if not isinstance(files, dict) or set(files) != file_names:
        raise ValueError("initialization_audit files must match the load_weights_only source inventory")
    for name, record in files.items():
        if record is None:
            if name in {"model.safetensors", "run_config.json"}:
                raise ValueError(f"initialization_audit requires {name}")
            continue
        if (not isinstance(record, dict) or set(record) != {"path", "before", "after"} or
                not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()):
            raise ValueError(f"initialization_audit requires concrete {name} source path")
        before, after = record["before"], record["after"]
        if (not isinstance(before, dict) or set(before) != _AUDIT_SNAPSHOT_FIELDS or before != after or
                not _is_sha256(before.get("sha256")) or
                any(isinstance(before[field], bool) or not isinstance(before[field], int) or before[field] < 0
                    for field in _AUDIT_SNAPSHOT_FIELDS - {"sha256"})):
            raise ValueError(f"initialization_audit {name} snapshot is incomplete or unstable")
    if files["model.safetensors"]["path"] != weights:
        raise ValueError("initialization_audit weights path disagrees with files record")
    semantic_fields = {}
    for side, channels in (("source_semantics", 0), ("target_semantics", 8)):
        semantics = audit.get(side)
        fields = semantics.get("fields") if isinstance(semantics, dict) else None
        encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8") if isinstance(fields, dict) else b""
        expected_origin = {"target-config"} if side == "target_semantics" else {
            "recipe-explicit", "mapping-derived", "config-default"}
        if (not isinstance(semantics, dict) or set(semantics) != {"fields", "sha256"} or
                not isinstance(fields, dict) or set(fields) != _AUDIT_SEMANTIC_FIELDS or
                any(not isinstance(entry, dict) or set(entry) != {"value", "provenance"} or
                    entry["provenance"] not in expected_origin for entry in fields.values()) or
                fields["evidence_channels"]["value"] != channels or
                semantics.get("sha256") != hashlib.sha256(encoded).hexdigest()):
            raise ValueError(f"initialization_audit {side} semantic field set/origins are invalid")
        semantic_fields[side] = fields
    for name in _AUDIT_SEMANTIC_FIELDS - {"evidence_channels"}:
        if semantic_fields["source_semantics"][name]["value"] != semantic_fields["target_semantics"][name]["value"]:
            raise ValueError(f"initialization_audit source/target semantic drift for {name}")
    compatibility, load = audit.get("compatibility"), audit.get("load")
    if (not isinstance(compatibility, dict) or
            set(compatibility) != {"transition", "exception", "same8_supported"} or
            compatibility["transition"] != "evidence_channels:0->8" or
            not isinstance(compatibility["exception"], str) or
            compatibility["same8_supported"] is not False or
            _PROJECTION_KEY not in compatibility["exception"] or
            "zero" not in compatibility["exception"].lower()):
        raise ValueError("initialization_audit compatibility must exactly record the verified 0->8 zero projection action")
    metadata_value = load.get("safetensors_metadata") if isinstance(load, dict) else None
    if (not isinstance(load, dict) or
            set(load) != {"missing_keys", "unexpected_keys", "safetensors_metadata"} or
            load["missing_keys"] != [_PROJECTION_KEY] or load["unexpected_keys"] != [] or
            (metadata_value is not None and
             (not isinstance(metadata_value, dict) or
              any(not isinstance(key, str) or not isinstance(value, str)
                  for key, value in metadata_value.items())))):
        raise ValueError("initialization_audit load must exactly match missing/unexpected keys and string safetensors metadata")
    recipe = audit.get("recipe")
    if (not isinstance(recipe, dict) or set(recipe) != {"canonical_sha256"} or
            not _is_sha256(recipe["canonical_sha256"])):
        raise ValueError("initialization_audit recipe must contain only the source run_config canonical_sha256")
    implementation = audit.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {"b5_runtime.py", "configuration_transgenic.py"}:
        raise ValueError("initialization_audit requires the exact loader implementation identities")
    for name, record in implementation.items():
        if (not isinstance(record, dict) or set(record) != {"path", "sha256"} or
                not isinstance(record["path"], str) or not Path(record["path"]).is_absolute() or
                not _is_sha256(record["sha256"])):
            raise ValueError(f"initialization_audit implementation identity is invalid for {name}")


def _validate_runtime_hardware(value: Any, scope: str) -> None:
    expected = {"measurement", "test_hardware", "world_size", "distributed_type",
                "device_type", "distinct_device_uuids", "ranks"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("runtime_hardware must contain the exact measured topology schema")
    ranks = value["ranks"]
    world = value["world_size"]
    if (value["measurement"] != "runtime_observation" or type(value["test_hardware"]) is not bool or
            type(value["distinct_device_uuids"]) is not bool or
            isinstance(world, bool) or not isinstance(world, int) or world < 1 or
            not isinstance(ranks, list) or len(ranks) != world or
            any(not isinstance(fact, dict) for fact in ranks)):
        raise ValueError("runtime_hardware is not a valid runtime observation")
    if sorted(fact.get("rank") for fact in ranks) != list(range(world)):
        raise ValueError("runtime_hardware ranks must be exactly 0..world_size-1")
    distributed = {fact.get("distributed_type") for fact in ranks}
    devices = {fact.get("device_type") for fact in ranks}
    if (value["distributed_type"] != (next(iter(distributed)) if len(distributed) == 1 else "MIXED") or
            value["device_type"] != (next(iter(devices)) if len(devices) == 1 else "mixed") or
            any(fact.get("world_size") != world for fact in ranks)):
        raise ValueError("runtime_hardware summary disagrees with measured rank facts")
    if scope == "production":
        if value["test_hardware"] or not value["distinct_device_uuids"]:
            raise ValueError("production runtime_hardware cannot be marked as test hardware")
        from .mvp_cli_runtime import _validate_public_hardware
        _validate_public_hardware(ranks)
    elif (not value["test_hardware"] or value["distinct_device_uuids"]):
        raise ValueError("test runtime_hardware must be marked as test hardware")


def _validate_metadata(metadata: Any, training_summary: Any) -> None:
    _canonical_json(metadata)
    _canonical_json(training_summary)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    missing = _REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"metadata missing required sections: {sorted(missing)}")
    scope = metadata["identity_scope"]
    if scope not in {"production", "test"}:
        raise ValueError("identity_scope must be production or test")
    _validate_initialization_audit(metadata["initialization_audit"])
    _validate_runtime_hardware(metadata["runtime_hardware"], scope)
    data = metadata["data_identities"]
    if not isinstance(data, dict):
        raise ValueError("data_identities must be an object")
    for name in ("corpus", "evidence", "panel"):
        if name not in data:
            raise ValueError(f"data_identities missing {name}")
        _identity(data[name], name)
    sif_code = metadata["sif_code_identity"]
    if not isinstance(sif_code, dict):
        raise ValueError("sif_code_identity must be an object")
    for name in ("sif", "code"):
        if name not in sif_code:
            raise ValueError(f"sif_code_identity missing {name}")
        _identity(sif_code[name], name)
    packages = metadata["runtime_packages"]
    if (not isinstance(packages, dict) or set(packages) != _RUNTIME_PACKAGES or
            any(not isinstance(value, str) or not value.strip() for value in packages.values())):
        raise ValueError(f"runtime_packages requires nonempty versions for {sorted(_RUNTIME_PACKAGES)}")
    recipe_data = metadata["normalized_recipe"]
    if not isinstance(recipe_data, dict) or set(recipe_data) != set(MVPRecipe.__dataclass_fields__):
        raise ValueError("normalized_recipe requires the exact MVPRecipe runtime fields")
    try:
        recipe = MVPRecipe(**recipe_data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"normalized_recipe is invalid: {error}") from error
    if scope == "production" and recipe != MVPRecipe():
        raise ValueError("production normalized_recipe must equal the fixed MVPRecipe")
    seeds = metadata["seed_augmentation"]
    seed = seeds.get("seed") if isinstance(seeds, dict) else None
    if (set(seeds) != {"seed", "aug_seed"} if isinstance(seeds, dict) else True) or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed_augmentation requires integer seed and aug_seed")
    from .mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE, validate_seed_profile
    profile = metadata.get("profile", LEGACY_PROFILE)
    validate_seed_profile(profile, seed)
    if ((profile == DEVELOPMENT_PROFILE or "run_role" in metadata) and
            metadata.get("run_role") != "mvp_finetune"):
        raise ValueError("MVP metadata run_role must be mvp_finetune")
    if type(seeds["aug_seed"]) is not int or seeds["aug_seed"] != 20260914 + seed:
        raise ValueError("aug_seed must equal 20260914 + seed")
    if (not isinstance(training_summary, dict) or
            isinstance(training_summary.get("successful_updates"), bool) or
            not isinstance(training_summary.get("successful_updates"), Integral) or
            isinstance(training_summary.get("effective_batch_size"), bool) or
            not isinstance(training_summary.get("effective_batch_size"), Integral)):
        raise ValueError("training_summary requires integer successful_updates and effective_batch_size")
    effective = recipe.batch_size * recipe.accumulation_steps * recipe.required_world_size
    if (training_summary["successful_updates"] != recipe.max_updates or
            training_summary["effective_batch_size"] != effective):
        raise ValueError(f"training_summary must match normalized_recipe successful_updates {recipe.max_updates} and effective_batch_size {effective}")
    if scope == "production" and (training_summary["successful_updates"] != 250 or effective != 96):
        raise ValueError("production training_summary requires successful_updates 250 and effective_batch_size 96")


def _validate_bounds(records: list[dict]) -> bytes:
    if len(records) > MVP_MAX_RECORDS:
        raise ValueError(f"paired record count exceeds MVP_MAX_RECORDS={MVP_MAX_RECORDS}")
    payload = _canonical_json(records)
    if len(payload) > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError(f"paired record payload exceeds {MVP_MAX_SERIALIZED_BYTES} bytes")
    return payload


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value, False
    marker = _STATUS_TRUNCATION_MARKER.encode("utf-8")
    prefix = encoded[:limit - len(marker)].decode("utf-8", errors="ignore")
    return prefix + _STATUS_TRUNCATION_MARKER, True


def _status(error: BaseException | None = None) -> dict:
    if error is None:
        return {"ok": True, "kind": "", "message": "", "truncated": False,
                "committed": False, "final_dir": ""}
    classes = {"ValueError", "RuntimeError", "FileExistsError", "OSError",
               "PublicationDurabilityError"}
    original_kind = type(error).__name__
    kind = original_kind if original_kind in classes else "RuntimeError"
    text = str(error)
    if kind == "RuntimeError" and original_kind != "RuntimeError":
        text = f"{original_kind}: {text}"
    message, truncated = _bounded_text(text, MVP_STATUS_MESSAGE_MAX_BYTES)
    final_dir = str(getattr(error, "final_dir", ""))
    if len(final_dir.encode("utf-8", errors="replace")) > MVP_STATUS_PATH_MAX_BYTES:
        raise RuntimeError("committed publication path exceeds bounded status schema")
    return {"ok": False, "kind": kind, "message": message, "truncated": truncated,
            "committed": bool(getattr(error, "committed", False)), "final_dir": final_dir}


def _status_bytes(status: dict) -> bytes:
    expected = {"ok", "kind", "message", "truncated", "committed", "final_dir"}
    if (not isinstance(status, dict) or set(status) != expected or
            type(status["ok"]) is not bool or type(status["truncated"]) is not bool or
            type(status["committed"]) is not bool or
            any(not isinstance(status[name], str) for name in ("kind", "message", "final_dir")) or
            len(status["message"].encode("utf-8")) > MVP_STATUS_MESSAGE_MAX_BYTES or
            len(status["final_dir"].encode("utf-8")) > MVP_STATUS_PATH_MAX_BYTES):
        raise ValueError("invalid bounded status schema")
    return _canonical_json(status)


def _decode_status(payload: bytes) -> dict:
    status = json.loads(payload)
    _status_bytes(status)
    return status


def _raise_status(status: dict) -> None:
    if status["ok"]:
        return
    if status["committed"]:
        raise PublicationDurabilityError(status["message"], status["final_dir"])
    classes = {"ValueError": ValueError, "RuntimeError": RuntimeError,
               "FileExistsError": FileExistsError, "OSError": OSError}
    raise classes.get(status["kind"], RuntimeError)(status["message"])


def _validate_transport_lengths(lengths: list[int]) -> int:
    if not lengths or any(isinstance(length, bool) or not isinstance(length, int) or length < 0
                          for length in lengths):
        raise ValueError("record transport lengths must be nonnegative integers")
    if any(length > MVP_MAX_SERIALIZED_BYTES for length in lengths):
        raise ValueError("local paired record payload exceeds serialized byte cap")
    if sum(lengths) > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError("aggregate paired record payload exceeds serialized byte cap")
    maximum = max(lengths)
    if len(lengths) * maximum > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError("padded paired record transport exceeds serialized byte cap")
    return maximum


def _transport_lengths(accelerator, local_error: Exception | None, payload: bytes) -> list[int]:
    values = [int(local_error is not None), len(payload) if local_error is None else 0]
    if accelerator.num_processes == 1:
        rows = [values]
    else:
        control = torch.tensor(values, dtype=torch.int64, device=accelerator.device)
        rows = accelerator.gather(control).reshape(accelerator.num_processes, 2).cpu().tolist()
    if any(row[0] for row in rows):
        raise ValueError("paired record collection or local serialization failed on at least one rank")
    lengths = [int(row[1]) for row in rows]
    _validate_transport_lengths(lengths)
    return lengths


def _gather_payload_bytes(accelerator, payload: bytes, lengths: list[int]) -> list[bytes]:
    maximum = _validate_transport_lengths(lengths)
    if accelerator.num_processes == 1:
        return [payload]
    padded = torch.zeros(maximum, dtype=torch.uint8, device=accelerator.device)
    if payload:
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(accelerator.device)
        padded[:len(payload)] = source
    gathered = accelerator.gather(padded).reshape(accelerator.num_processes, maximum).cpu()
    return [gathered[index, :length].contiguous().numpy().tobytes()
            for index, length in enumerate(lengths)]


def _gather_record_payload_bytes(accelerator, payload: bytes, lengths: list[int]) -> list[bytes]:
    return _gather_payload_bytes(accelerator, payload, lengths)


def _broadcast_payload_bytes(accelerator, payload: bytes) -> bytes:
    if accelerator.num_processes == 1:
        return payload
    size = torch.tensor([len(payload)], dtype=torch.int64, device=accelerator.device)
    torch.distributed.broadcast(size, src=0)
    length = int(size.item())
    if length > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError("broadcast paired summary exceeds serialized byte cap")
    tensor = torch.zeros(length, dtype=torch.uint8, device=accelerator.device)
    if accelerator.is_main_process and payload:
        tensor[:] = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(accelerator.device)
    torch.distributed.broadcast(tensor, src=0)
    return tensor.cpu().contiguous().numpy().tobytes()


def _collect_statuses(accelerator, status: dict) -> list[dict]:
    payload = _status_bytes(status)
    lengths = _transport_lengths(accelerator, None, payload)
    return [_decode_status(item) for item in _gather_payload_bytes(accelerator, payload, lengths)]


def _broadcast_status(accelerator, status: dict | None) -> dict:
    payload = _status_bytes(status) if accelerator.is_main_process else b""
    return _decode_status(_broadcast_payload_bytes(accelerator, payload))


def _collective_preflight(accelerator, operation) -> None:
    try:
        operation()
        local = _status()
    except Exception as error:
        local = _status(error)
    for status in _collect_statuses(accelerator, local):
        if not status["ok"]:
            _raise_status(status)


def _all_ints(accelerator, value: int) -> list[int]:
    if accelerator.num_processes == 1:
        return [value]
    tensor = torch.tensor([value], dtype=torch.int64, device=accelerator.device)
    return [int(item) for item in accelerator.gather(tensor).cpu().tolist()]


def _all_fixed_payloads_equal(accelerator, payload: bytes) -> bool:
    gathered = _gather_payload_bytes(accelerator, payload,
                                     [len(payload)] * accelerator.num_processes)
    return len(set(gathered)) == 1


def _validated_inventory(expected_rns) -> list[int]:
    if type(expected_rns) not in (list, tuple, range):
        raise ValueError("expected_rns must be a concrete list, tuple, or range")
    length = len(expected_rns)
    if length == 0 or length > MVP_MAX_RECORDS:
        raise ValueError(f"expected_rns length must be within 1..MVP_MAX_RECORDS={MVP_MAX_RECORDS}")
    if (any(isinstance(rn, bool) or not isinstance(rn, Integral) for rn in expected_rns) or
            len(set(expected_rns)) != length):
        raise ValueError("expected_rns must contain unique integers")
    return list(expected_rns)


def _validate_evaluation_inputs(accelerator, model, eval_loader, expected,
                                metadata, training_summary) -> None:
    _validate_metadata(metadata, training_summary)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch module")
    if not isinstance(expected, list):
        raise ValueError("expected_rns must be validated before evaluation")
    try:
        length = len(eval_loader)
    except TypeError as error:
        raise ValueError("eval_loader must have a finite length") from error
    if length <= 0 or getattr(eval_loader, "batch_size", None) != 1:
        raise ValueError("eval_loader must be nonempty with batch size one")
    if eval_loader.__class__.__module__.startswith("accelerate.data_loader"):
        raise ValueError("eval_loader must be unprepared")
    distributed_name = getattr(accelerator.distributed_type, "name", str(accelerator.distributed_type))
    if distributed_name not in {"NO", "MULTI_CPU", "MULTI_GPU"}:
        raise ValueError("only single-process or replicated DDP evaluation is supported")
    if accelerator.num_processes > 1:
        if not isinstance(model, DistributedDataParallel):
            raise ValueError("multi-rank finalization requires a prepared replicated DDP model")
        sampler = getattr(eval_loader, "sampler", None)
        if (not isinstance(sampler, DistributedSampler) or sampler.num_replicas != accelerator.num_processes or
                sampler.rank != accelerator.process_index or sampler.shuffle or sampler.drop_last):
            raise ValueError("DDP eval_loader requires rank/world-matched DistributedSampler shuffle=False drop_last=False")


def _to_device(value: Any, device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _save_target(accelerator, model):
    target = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:  # pragma: no cover - unavailable only on old unsupported Torch
        OptimizedModule = ()
    if OptimizedModule and type(target) is OptimizedModule:
        target = target._orig_mod
    if isinstance(target, (DistributedDataParallel, torch.nn.DataParallel)) or hasattr(target, "module"):
        raise ValueError("unknown parallel/compiled wrapper layout cannot be serialized safely")
    if not isinstance(target, torch.nn.Module) or not hasattr(target, "config"):
        raise ValueError("serialization target must be an unwrapped module with config")
    keys = target.state_dict().keys()
    if any(key.startswith(("module.", "_orig_mod.")) for key in keys):
        raise ValueError("serialization keys retain wrapper prefixes")
    return target


def _write_sync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_payload_files(stage: Path, target, paired_summary: dict,
                         metadata: dict, training_summary: dict) -> list[str]:
    recorded_metadata = dict(metadata)
    recorded_metadata["training_summary"] = training_summary
    _write_sync(stage / "metadata.json", _canonical_json(recorded_metadata))
    records = sorted(paired_summary["records"], key=lambda row: row["rn"])
    lines = b"".join(_canonical_json(record) for record in records)
    if len(lines) > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError(f"paired record payload exceeds {MVP_MAX_SERIALIZED_BYTES} bytes")
    _write_sync(stage / "paired_records.jsonl", lines)
    summary = {key: value for key, value in paired_summary.items() if key != "records"}
    _write_sync(stage / "paired_summary.json", _canonical_json(summary))
    save_model(target, str(stage / "model.safetensors"), force_contiguous=True)
    with (stage / "model.safetensors").open("rb") as handle:
        os.fsync(handle.fileno())
    config = target.config.to_dict() if hasattr(target.config, "to_dict") else target.config
    if not isinstance(config, dict):
        raise ValueError("model config must provide a JSON object")
    _write_sync(stage / "config.json", _canonical_json(config))
    return ["metadata.json", "paired_records.jsonl", "paired_summary.json",
            "model.safetensors", "config.json"]


def _write_manifest(stage: Path, names: list[str]) -> None:
    files = {name: {"size": (stage / name).stat().st_size, "sha256": _sha256(stage / name)}
             for name in names}
    _write_sync(stage / "manifest.json", _canonical_json({"files": files}))
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = stage / name
        if path.stat().st_size != expected["size"] or _sha256(path) != expected["sha256"]:
            raise OSError(f"published payload validation failed for {name}")
        if name == "model.safetensors":
            with safe_open(path, framework="pt") as model_file:
                list(model_file.keys())
        elif name.endswith((".json", ".jsonl")):
            if name.endswith(".jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    json.loads(line)
            else:
                json.loads(path.read_text(encoding="utf-8"))


def _write_done_marker(stage: Path) -> None:
    _write_sync(stage / "TRAINING_DONE", b"complete\n")
    descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "renameat2 RENAME_NOREPLACE requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def _fsync_parent(final_dir: Path) -> None:
    descriptor = os.open(final_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_final(accelerator, model, paired_summary: dict, final_dir: Path | str, *,
                   metadata: dict, training_summary: dict) -> dict:
    final = Path(final_dir)
    if len(str(final).encode("utf-8", errors="replace")) > MVP_STATUS_PATH_MAX_BYTES:
        raise ValueError("final_dir exceeds bounded publication-status path limit")
    _validate_metadata(metadata, training_summary)
    records = paired_summary.get("records") if isinstance(paired_summary, dict) else None
    if not isinstance(records, list):
        raise ValueError("paired summary requires records")
    _validate_bounds(records)
    if not final.parent.is_dir():
        raise ValueError("final parent must be a preexisting directory")
    if os.path.lexists(final):
        raise FileExistsError(final)
    target = _save_target(accelerator, model)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    renamed = False
    try:
        names = _write_payload_files(stage, target, paired_summary, metadata, training_summary)
        _write_manifest(stage, names)
        _write_done_marker(stage)
        _rename_noreplace(stage, final)
        renamed = True
        try:
            _fsync_parent(final)
        except OSError as error:
            raise PublicationDurabilityError(str(error), final) from error
        return paired_summary
    finally:
        if not renamed:
            shutil.rmtree(stage, ignore_errors=True)


def evaluate_and_publish_final(accelerator, model, eval_loader, expected_rns, final_dir, *,
                               metadata, training_summary, prepublication_prepare=None,
                               prepublication_check=None):
    """Evaluate off/on pairs collectively, then publish once on rank zero."""
    expected = None
    def validate_preflight():
        nonlocal expected
        expected = _validated_inventory(expected_rns)
        _validate_evaluation_inputs(accelerator, model, eval_loader, expected,
                                    metadata, training_summary)
    _collective_preflight(accelerator, validate_preflight)
    lengths = _all_ints(accelerator, len(eval_loader))
    if len(set(lengths)) != 1:
        raise ValueError(f"per-rank eval_loader lengths must be equal: {lengths}")
    identity = hashlib.sha256(_canonical_json({"expected_rns": expected, "metadata": metadata,
                                               "training_summary": training_summary})).hexdigest()
    if not _all_fixed_payloads_equal(accelerator, identity.encode("ascii")):
        raise ValueError("expected inventory, metadata, and training summary differ across ranks")
    local_records = []
    local_error = None
    for batch in eval_loader:
        try:
            local_records.append(paired_metric_record(model, _to_device(batch, accelerator.device)))
        except Exception as error:
            local_error = local_error or error
    payload = b""
    try:
        payload = _validate_bounds(local_records)
    except Exception as error:
        local_error = local_error or error
    transport_lengths = _transport_lengths(accelerator, local_error, payload)
    gathered_payloads = _gather_record_payload_bytes(accelerator, payload, transport_lengths)
    aggregate_status = None
    summary_payload = b""
    if accelerator.is_main_process:
        try:
            records = []
            for encoded in gathered_payloads:
                decoded = json.loads(encoded)
                if not isinstance(decoded, list):
                    raise ValueError("gathered paired record payload must decode to a list")
                records.extend(decoded)
            _validate_bounds(records)
            summary_payload = _canonical_json(aggregate_paired_records(records, expected))
            if len(summary_payload) > MVP_MAX_SERIALIZED_BYTES:
                raise ValueError("paired summary exceeds serialized byte cap")
            aggregate_status = _status()
        except Exception as error:
            aggregate_status = _status(error)
    aggregate_status = _broadcast_status(accelerator, aggregate_status)
    _raise_status(aggregate_status)
    summary = json.loads(_broadcast_payload_bytes(accelerator, summary_payload))
    if prepublication_prepare is not None:
        _collective_preflight(accelerator, prepublication_prepare)
    if prepublication_check is not None:
        _collective_preflight(accelerator, prepublication_check)

    publication_status = None
    if accelerator.is_main_process:
        try:
            _publish_final(accelerator, model, summary, final_dir,
                           metadata=metadata, training_summary=training_summary)
            publication_status = _status()
        except Exception as error:
            publication_status = _status(error)
    publication_status = _broadcast_status(accelerator, publication_status)
    accelerator.wait_for_everyone()
    _raise_status(publication_status)
    return summary
