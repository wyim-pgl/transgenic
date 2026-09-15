"""Audited, weights-only B5 (0 evidence channels) to MVP (8 channels) initialization.

``load_weights_only`` mutates the caller's fresh, unwrapped model.  Every
failure marks that target ``_weights_init_poisoned = True`` because a failed
safetensors load may already have copied some parameters.  Callers must discard
such a target; this helper provides no rollback and is not resume support.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_model

from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model import configuration_transgenic
from transgenic.training import b5_runtime
from transgenic.training.b5_runtime import REQUIRED_CONFIG_KEYS, model_kwargs

_SCHEMA_VERSION = "b6-weights-init-v1"
_PROJECTION_KEY = "transgenic.encoder.evidence_proj.weight"
_REQUIRED_RECIPE_KEYS = set(REQUIRED_CONFIG_KEYS) | {"encoder_n_layer"}

# These fields determine serialized layout or behavior used by the current
# constructor/forward path (encoder, LED decoder, generation, tying and heads).
_SEMANTIC_FIELDS = (
    "vocab_size", "max_encoder_position_embeddings", "max_decoder_position_embeddings",
    "encoder_layers", "encoder_ffn_dim", "encoder_attention_heads",
    "decoder_layers", "decoder_ffn_dim", "decoder_attention_heads",
    "encoder_layerdrop", "decoder_layerdrop", "use_cache", "is_encoder_decoder",
    "activation_function", "d_model", "encoder_d_model", "decoder_d_model",
    "dropout", "attention_dropout", "activation_dropout", "classifier_dropout", "init_std",
    "decoder_start_token_id", "pad_token_id", "bos_token_id", "eos_token_id",
    "attention_window", "encoder_model", "max_encoder_seqlen", "encoder_n_layer",
    "unlink", "max_transcript_count", "evidence_channels", "tie_word_embeddings",
    "return_dict", "output_attentions", "output_hidden_states", "torchscript",
    "scale_embedding", "normalize_before", "layer_norm_eps",
)
_MAPPED_FROM = {
    "d_model": "d_model_encoder",
    "encoder_d_model": "d_model_encoder",
    "decoder_d_model": "d_model_decoder",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(path: Path) -> dict[str, Any]:
    info = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "sha256": digest.hexdigest(), "size": info.st_size,
        "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
    }


def _regular_file(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.parent != parent or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{label} must be a regular file at its canonical checkpoint root")
    return resolved


def _resolve(init_from: os.PathLike[str] | str) -> tuple[Path, Path]:
    requested = Path(init_from)
    resolved = requested.resolve(strict=True)
    if resolved.is_dir():
        checkpoint = resolved
        weights = _regular_file(checkpoint / "model.safetensors", checkpoint, "weights")
    else:
        if requested.name != "model.safetensors" or resolved.name != "model.safetensors":
            raise ValueError("the exact weights filename must be model.safetensors")
        checkpoint = requested.parent.resolve(strict=True)
        if resolved.parent != checkpoint:
            raise ValueError("weights must be a regular file at its canonical checkpoint root")
        weights = _regular_file(requested, checkpoint, "weights")
    return checkpoint, weights


def _json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_value(item) for item in value]
    if isinstance(value, list):
        return [_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"semantic value is not JSON-serializable: {value!r}")


def _manifest(config: Any, recipe: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    kwargs = model_kwargs(recipe) if recipe is not None else {}
    for name in _SEMANTIC_FIELDS:
        if not hasattr(config, name):
            continue
        if recipe is None:
            provenance = "target-config"
        elif name in recipe:
            provenance = "recipe-explicit"
        elif name in _MAPPED_FROM:
            provenance = "mapping-derived"
        elif name in kwargs:
            provenance = "mapping-derived"
        else:
            provenance = "config-default"
        fields[name] = {"value": _value(getattr(config, name)), "provenance": provenance}
    return fields


def _manifest_record(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"fields": fields, "sha256": _sha256_bytes(_canonical(fields))}


def _check_equal(source: dict[str, dict[str, Any]], target: dict[str, dict[str, Any]]) -> None:
    names = (set(source) | set(target)) - {"evidence_channels"}
    for name in sorted(names):
        source_value = source.get(name, {}).get("value", "<absent>")
        target_value = target.get(name, {}).get("value", "<absent>")
        if source_value != target_value:
            raise ValueError(f"source/target semantic mismatch for {name}: {source_value!r} != {target_value!r}")


def load_weights_only(model: torch.nn.Module, init_from: os.PathLike[str] | str) -> dict[str, Any]:
    """Load only a canonical B5 ``model.safetensors`` into an 8-channel model.

    The sole compatibility exception is evidence channels 0 -> 8; the missing
    projection is zeroed. Same-8 initialization is deliberately unsupported.
    No optimizer, scheduler, pending-gradient or RNG state is read. Any raised
    exception poisons ``model``; discard it even when validation failed early.
    """
    if getattr(model, "_weights_init_poisoned", False):
        raise ValueError("weights-only target was poisoned by an earlier failed load; discard it")
    model._weights_init_poisoned = True
    requested = os.fspath(init_from)
    checkpoint, weights = _resolve(init_from)

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("weights-only target must not have preexisting grads")
    target_config = getattr(model, "config", None)
    if target_config is None or getattr(target_config, "evidence_channels", None) != 8:
        raise ValueError("weights-only target evidence_channels must be 8")

    run_root = checkpoint.parent
    run_config_path = _regular_file(run_root / "run_config.json", run_root, "run_config.json")
    runtime_identity = run_root / "runtime_identity.txt"
    paths: dict[str, Path | None] = {
        "model.safetensors": weights,
        "run_config.json": run_config_path,
        "runtime_identity.txt": (
            _regular_file(runtime_identity, run_root, "runtime_identity.txt")
            if os.path.lexists(runtime_identity) else None
        ),
    }
    for name in ("config.json", "meta.json", "runtime.json"):
        candidate = checkpoint / name
        if os.path.lexists(candidate):
            paths[name] = _regular_file(candidate, checkpoint, name)
        else:
            paths[name] = None

    before = {name: _snapshot(path) for name, path in paths.items() if path is not None}
    run_config = _json_file(run_config_path)
    if not isinstance(run_config, dict) or not isinstance(run_config.get("recipe"), dict):
        raise ValueError("parent run_config.json must contain an embedded normalized recipe")
    recipe = run_config["recipe"]
    missing_recipe = sorted(_REQUIRED_RECIPE_KEYS - set(recipe))
    if missing_recipe:
        raise ValueError(f"embedded normalized recipe missing keys {missing_recipe}")
    source_channels = recipe.get("evidence_channels", 0)
    if source_channels != 0 or isinstance(source_channels, bool):
        raise ValueError("only the evidence_channels 0->8 transition is supported; same-8 is deferred")

    source_config = HyenaTransgenicConfig(**model_kwargs(recipe))
    source_fields = _manifest(source_config, recipe)
    target_fields = _manifest(target_config, None)

    # model_kwargs is intentionally the historical B5 mapping. If a recipe has
    # additional known semantic keys, they may attest defaults but not override it.
    for name in _SEMANTIC_FIELDS:
        if name in recipe and name != "evidence_channels" and hasattr(source_config, name):
            if _value(recipe[name]) != _value(getattr(source_config, name)):
                raise ValueError(f"recipe semantic {name} contradicts the reconstructed B5 default")
    _check_equal(source_fields, target_fields)

    config_path = paths["config.json"]
    if config_path is not None:
        recorded_config = _json_file(config_path)
        if not isinstance(recorded_config, dict):
            raise ValueError("config.json must contain an object")
        for name, entry in source_fields.items():
            if name in recorded_config and _value(recorded_config[name]) != entry["value"]:
                raise ValueError(f"config.json semantic mismatch for {name}")

    target_state = model.state_dict()
    tied_keys = (
        "lm_head.weight", "transgenic.decoder_embed_tokens.weight",
        "transgenic.decoder.embed_tokens.weight",
    )
    if not all(key in target_state for key in tied_keys):
        raise ValueError(f"target is missing required tied storage keys: {tied_keys}")
    storages = {(target_state[key].untyped_storage().data_ptr(), target_state[key].storage_offset()) for key in tied_keys}
    if len(storages) != 1:
        raise ValueError("target decoder embeddings and LM head must use actual tied storage")

    with safe_open(weights, framework="pt", device="cpu") as handle:
        file_keys = set(handle.keys())
        safetensors_metadata = handle.metadata()
        if _PROJECTION_KEY in file_keys:
            raise ValueError("source checkpoint must not contain an evidence projection")
        unexpected_preflight = sorted(file_keys - set(target_state))
        if unexpected_preflight:
            raise ValueError(f"unexpected safetensors keys: {unexpected_preflight}")
        for key in sorted(file_keys):
            shape = tuple(handle.get_slice(key).get_shape())
            if shape != tuple(target_state[key].shape):
                raise ValueError(f"shape mismatch for {key}: {shape} != {tuple(target_state[key].shape)}")

    missing, unexpected = load_model(model, weights, strict=False, device="cpu")
    missing, unexpected = sorted(missing), sorted(unexpected)
    if missing != [_PROJECTION_KEY] or unexpected:
        raise ValueError(f"weights load refused: missing={missing}, unexpected={unexpected}")
    projection = model.get_parameter(_PROJECTION_KEY)
    with torch.no_grad():
        projection.zero_()
    if torch.count_nonzero(projection).item() != 0:
        raise RuntimeError("evidence projection zero verification failed")

    after = {name: _snapshot(path) for name, path in paths.items() if path is not None}
    for name in before:
        if before[name] != after[name]:
            raise RuntimeError(f"tracked source file changed during weights initialization: {name}")

    files = {
        name: None if path is None else {"path": str(path), "before": before[name], "after": after[name]}
        for name, path in paths.items()
    }
    implementation = {}
    for name, module in (("b5_runtime.py", b5_runtime), ("configuration_transgenic.py", configuration_transgenic)):
        path = Path(module.__file__).resolve(strict=True)
        implementation[name] = {"path": str(path), "sha256": _snapshot(path)["sha256"]}

    audit = {
        "schema_version": _SCHEMA_VERSION,
        "mode": "weights-only",
        "requested_path": requested,
        "resolved_checkpoint_dir": str(checkpoint),
        "resolved_weights_path": str(weights),
        "files": files,
        "recipe": {"canonical_sha256": _sha256_bytes(_canonical(recipe))},
        "source_semantics": _manifest_record(source_fields),
        "target_semantics": _manifest_record(target_fields),
        "implementation": implementation,
        "compatibility": {
            "transition": "evidence_channels:0->8",
            "exception": f"exactly missing {_PROJECTION_KEY}; initialized to verified zeros",
            "same8_supported": False,
        },
        "load": {"missing_keys": missing, "unexpected_keys": unexpected,
                 "safetensors_metadata": safetensors_metadata},
        "provenance_caveat": (
            "config-default fields are reconstructed under the current loader and are not "
            "checkpoint-attested historical defaults"
        ),
    }
    model._weights_init_poisoned = False
    return audit
