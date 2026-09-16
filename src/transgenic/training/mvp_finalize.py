"""Paired MVP evaluation and strict atomic final-artifact publication.

The distributed contract is replicated DDP only.  It coordinates ordinary Python
failures, but rank death, communicator failure, and asymmetric DDP-forward failure
remain launcher/process-group-timeout concerns.
"""
from __future__ import annotations

import ctypes
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
from accelerate.utils import broadcast_object_list
from safetensors import safe_open
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .mvp_metrics import aggregate_paired_records, paired_metric_record

MVP_MAX_RECORDS = 100_000
MVP_MAX_SERIALIZED_BYTES = 64 * 1024 * 1024
_SHA256_LENGTH = 64
_REQUIRED_METADATA = {
    "identity_scope", "initialization_audit", "data_identities", "sif_code_identity",
    "runtime_packages", "normalized_recipe", "seed_augmentation",
}
_IDENTITY_VERIFICATIONS = {"recorded", "independently_verified"}


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


def _validate_metadata(metadata: Any, training_summary: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    missing = _REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"metadata missing required sections: {sorted(missing)}")
    if metadata["identity_scope"] not in {"production", "test"}:
        raise ValueError("identity_scope must be production or test")
    initialization = metadata["initialization_audit"]
    if not isinstance(initialization, dict):
        raise ValueError("initialization_audit must be an object")
    _identity({"sha256": initialization.get("report_sha256"),
               "verification": initialization.get("verification")}, "initialization audit report")
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
    for name in ("runtime_packages", "normalized_recipe", "seed_augmentation"):
        if not isinstance(metadata[name], dict) or not metadata[name]:
            raise ValueError(f"{name} must be a nonempty object")
    if (not isinstance(training_summary, dict) or
            isinstance(training_summary.get("successful_updates"), bool) or
            not isinstance(training_summary.get("successful_updates"), Integral) or
            isinstance(training_summary.get("effective_batch_size"), bool) or
            not isinstance(training_summary.get("effective_batch_size"), Integral)):
        raise ValueError("training_summary requires integer successful_updates and effective_batch_size")
    if metadata["identity_scope"] == "production" and (
            training_summary["successful_updates"] != 250 or
            training_summary["effective_batch_size"] != 96):
        raise ValueError("production training_summary requires successful_updates 250 and effective_batch_size 96")
    if (training_summary["successful_updates"] <= 0 or
            training_summary["effective_batch_size"] <= 0):
        raise ValueError("test training budgets must be positive")
    _canonical_json(metadata)
    _canonical_json(training_summary)


def _validate_bounds(records: list[dict]) -> bytes:
    if len(records) > MVP_MAX_RECORDS:
        raise ValueError(f"paired record count exceeds MVP_MAX_RECORDS={MVP_MAX_RECORDS}")
    payload = _canonical_json(records)
    if len(payload) > MVP_MAX_SERIALIZED_BYTES:
        raise ValueError(f"paired record payload exceeds {MVP_MAX_SERIALIZED_BYTES} bytes")
    return payload


def _status(error: BaseException | None = None, **values: Any) -> dict:
    if error is None:
        return {"ok": True, **values}
    return {"ok": False, "kind": type(error).__name__, "message": str(error),
            "committed": bool(getattr(error, "committed", False)),
            "final_dir": str(getattr(error, "final_dir", ""))}


def _raise_status(status: dict) -> None:
    if status["ok"]:
        return
    if status.get("committed"):
        raise PublicationDurabilityError(status["message"], status["final_dir"])
    classes = {"ValueError": ValueError, "RuntimeError": RuntimeError,
               "FileExistsError": FileExistsError, "OSError": OSError}
    raise classes.get(status.get("kind"), RuntimeError)(status["message"])


def _all_gather_objects(accelerator, value: Any) -> list[Any]:
    if accelerator.num_processes == 1:
        return [value]
    gathered = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _broadcast_status(accelerator, status: dict | None) -> dict:
    values = [status]
    broadcast_object_list(values, from_process=0)
    return values[0]


def _collective_preflight(accelerator, operation) -> None:
    try:
        operation()
        local = None
    except Exception as error:
        local = _status(error)
    failures = _all_gather_objects(accelerator, local)
    for failure in failures:
        if failure is not None:
            _raise_status(failure)


def _validate_evaluation_inputs(accelerator, model, eval_loader, expected_rns,
                                metadata, training_summary) -> None:
    _validate_metadata(metadata, training_summary)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch module")
    expected = list(expected_rns)
    if (not expected or len(expected) > MVP_MAX_RECORDS or
            any(isinstance(rn, bool) or not isinstance(rn, Integral) for rn in expected) or
            len(expected) != len(set(expected))):
        raise ValueError("expected_rns must be a nonempty finite unique integer inventory within MVP_MAX_RECORDS")
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
                               metadata, training_summary):
    """Evaluate off/on pairs collectively, then publish once on rank zero."""
    expected = list(expected_rns)
    _collective_preflight(
        accelerator,
        lambda: _validate_evaluation_inputs(accelerator, model, eval_loader, expected,
                                             metadata, training_summary),
    )
    lengths = _all_gather_objects(accelerator, len(eval_loader))
    if len(set(lengths)) != 1:
        raise ValueError(f"per-rank eval_loader lengths must be equal: {lengths}")
    identity = hashlib.sha256(_canonical_json({"expected_rns": expected, "metadata": metadata,
                                               "training_summary": training_summary})).hexdigest()
    if len(set(_all_gather_objects(accelerator, identity))) != 1:
        raise ValueError("expected inventory, metadata, and training summary differ across ranks")
    local_records = []
    local_error = None
    for batch in eval_loader:
        try:
            local_records.append(paired_metric_record(model, _to_device(batch, accelerator.device)))
        except Exception as error:
            local_error = local_error or error
    try:
        _validate_bounds(local_records)
    except Exception as error:
        local_error = local_error or error
    gathered = _all_gather_objects(accelerator, {"records": local_records,
                                                  "error": _status(local_error) if local_error else None})
    aggregate_status = None
    if accelerator.is_main_process:
        failure = next((item["error"] for item in gathered if item["error"] is not None), None)
        if failure is not None:
            aggregate_status = failure
        else:
            try:
                records = [record for item in gathered for record in item["records"]]
                _validate_bounds(records)
                aggregate_status = _status(summary=aggregate_paired_records(records, expected))
            except Exception as error:
                aggregate_status = _status(error)
    aggregate_status = _broadcast_status(accelerator, aggregate_status)
    _raise_status(aggregate_status)
    summary = aggregate_status["summary"]

    publication_status = None
    if accelerator.is_main_process:
        try:
            _publish_final(accelerator, model, summary, final_dir,
                           metadata=metadata, training_summary=training_summary)
            publication_status = _status(summary=summary)
        except Exception as error:
            publication_status = _status(error)
    publication_status = _broadcast_status(accelerator, publication_status)
    accelerator.wait_for_everyone()
    _raise_status(publication_status)
    return publication_status["summary"]
