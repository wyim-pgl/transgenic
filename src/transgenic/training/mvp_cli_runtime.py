"""Heavy preflight and execution for the exploratory MVP CLI."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import stat
from typing import Any

from .mvp_cli import CandidateRecords, OUTPUT_USE, _digest, _json_record, _regular_file
from .mvp_runtime import MVPRecipe

_REQUIRED_EVIDENCE = {
    "real_tokenizer", "real_hyena_step0_three_tiers", "generation_cache_parity",
    "bf16_compile", "four_rank_smoke", "resource_budget",
}


def _snapshot_hash(value: os.PathLike[str] | str, label: str) -> tuple[Path, str]:
    path = _regular_file(value, label)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"{label} changed while it was hashed")
    return path, digest.hexdigest()


def _reject_wal(path: Path, label: str) -> None:
    wal = Path(str(path) + ".wal")
    if os.path.lexists(wal):
        raise ValueError(f"{label} has a WAL beside it: {wal}")


def verify_candidate_artifacts(candidate: CandidateRecords, db_value, evidence_value, panel_value) -> dict:
    """Bind candidate bytes, then compare only read-only identity metadata."""
    db, db_sha = _snapshot_hash(db_value, "corpus DB")
    evidence, evidence_sha = _snapshot_hash(evidence_value, "evidence DB")
    panel, panel_sha = _snapshot_hash(panel_value, "panel")
    _reject_wal(db, "corpus DB")
    _reject_wal(evidence, "evidence DB")
    accepted = candidate.acceptance
    expected = {
        "corpus": accepted.get("inputs", {}).get("corpus", {}).get("sha256"),
        "evidence": accepted.get("artifact", {}).get("sha256"),
        "panel": accepted.get("inputs", {}).get("panel", {}).get("sha256"),
    }
    measured = {"corpus": db_sha, "evidence": evidence_sha, "panel": panel_sha}
    for name in measured:
        if measured[name] != _digest(expected[name], f"accepted {name} SHA256"):
            raise ValueError(f"{name} SHA256 does not match the accepted identity")

    import duckdb
    with duckdb.connect(str(db), config={"access_mode": "READ_ONLY", "threads": 1}) as con:
        corpus_rows = con.execute(
            "SELECT rn, chromosome, start, fin, split, is_rc FROM geneList "
            "WHERE species_id='Athaliana' AND split IN ('train','valid') ORDER BY rn"
        ).fetchall()
    with duckdb.connect(str(evidence), config={"access_mode": "READ_ONLY", "threads": 1}) as con:
        evidence_rows = con.execute(
            "SELECT rn, chrom, start0, end0, split, is_rc FROM corpus_tile "
            "WHERE split IN ('train','valid') ORDER BY rn"
        ).fetchall()
    frozen = [(row["rn"], row["chromosome"], row["start0"], row["end0"],
               row["split"], row["is_rc"]) for row in candidate.inventory["records"]]
    if corpus_rows != frozen or evidence_rows != frozen:
        raise ValueError("candidate inventory metadata does not exactly match corpus/evidence metadata")
    return {
        "paths": {"corpus": str(db), "evidence": str(evidence), "panel": str(panel)},
        "sha256": measured,
        "train_rns": [row[0] for row in frozen if row[4] == "train"],
        "valid_rns": [row[0] for row in frozen if row[4] == "valid"],
    }


def _manifest_files(path: Path) -> None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("consumed code manifest is not valid JSON") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("consumed code manifest requires a nonempty files object")
    for source, expected in files.items():
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise ValueError("consumed code manifest paths must be absolute")
        _, measured = _snapshot_hash(source, "consumed code file")
        if measured != _digest(expected, "consumed code SHA256"):
            raise ValueError(f"consumed code identity mismatch: {source}")


def validate_launch_readiness(readiness_value, readiness_sha256: str,
                              candidate: CandidateRecords, paths, *, seed: int,
                              recipe: MVPRecipe) -> dict:
    """Validate a trusted operator readiness record; never synthesize evidence."""
    _, readiness = _json_record(readiness_value, readiness_sha256, "launch readiness")
    if (readiness.get("schema_version") != 1 or
            readiness.get("authorized_scope") != "exploratory_mvp" or
            readiness.get("production_authorized") is not False or
            readiness.get("ready_for_finetuning") is not True):
        raise ValueError("launch readiness does not authorize exploratory MVP finetuning")
    bindings = readiness.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("launch readiness bindings are missing")
    for key, identity_key in (("technical_acceptance_sha256", "acceptance"),
                              ("inventory_sha256", "inventory"),
                              ("receipts_sha256", "receipts")):
        if bindings.get(key) != candidate.identities[identity_key]:
            raise ValueError(f"launch readiness {key} binding mismatch")
    db, evidence, panel, sif, config, weights, run_config = paths
    actual_paths = {
        "corpus_sha256": db, "evidence_sha256": evidence, "panel_sha256": panel,
        "sif_sha256": sif, "config_sha256": config,
    }
    for key, value in actual_paths.items():
        _, measured = _snapshot_hash(value, key.removesuffix("_sha256"))
        if measured != _digest(bindings.get(key), f"readiness {key}"):
            raise ValueError(f"launch readiness {key} mismatch")
    source = bindings.get("source_checkpoint")
    if not isinstance(source, dict):
        raise ValueError("launch readiness source checkpoint binding is missing")
    measured_source_paths = {}
    for prefix, value in (("weights", weights), ("run_config", run_config)):
        path, measured = _snapshot_hash(value, f"source {prefix}")
        measured_source_paths[prefix] = path
        if str(path) != source.get(f"{prefix}_path") or measured != source.get(f"{prefix}_sha256"):
            raise ValueError(f"launch readiness source {prefix} identity mismatch")
    try:
        source_config = json.loads(measured_source_paths["run_config"].read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source run config is not valid JSON") from error
    if (not isinstance(source_config, dict) or source_config.get("seed") != 123 or
            source.get("seed") != 123):
        raise ValueError("launch readiness source checkpoint seed must exactly bind seed 123")
    manifest = bindings.get("consumed_code_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("launch readiness consumed code manifest binding is missing")
    manifest_path, measured = _snapshot_hash(manifest.get("path", ""), "consumed code manifest")
    if measured != _digest(manifest.get("sha256"), "consumed code manifest SHA256"):
        raise ValueError("launch readiness consumed code manifest mismatch")
    _manifest_files(manifest_path)
    if bindings.get("recipe") != asdict(recipe) or bindings.get("seed") != seed:
        raise ValueError("launch readiness fixed recipe or seed binding mismatch")
    evidence_records = readiness.get("evidence")
    if not isinstance(evidence_records, dict) or set(evidence_records) != _REQUIRED_EVIDENCE:
        raise ValueError("launch readiness evidence set is incomplete")
    for name, record in evidence_records.items():
        if (not isinstance(record, dict) or record.get("pass") is not True or
                not isinstance(record.get("receipt_sha256"), str) or
                not record["receipt_sha256"] or
                _digest(record["receipt_sha256"], f"{name} receipt SHA256") != record["receipt_sha256"]):
            raise ValueError(f"launch readiness evidence failed: {name}")
    return readiness


def _source_paths(init_from) -> tuple[Path, Path]:
    source = Path(init_from).resolve(strict=True)
    checkpoint = source if source.is_dir() else source.parent
    weights = checkpoint / "model.safetensors" if source.is_dir() else source
    run_config = checkpoint.parent / "run_config.json"
    return _regular_file(weights, "source weights"), _regular_file(run_config, "source run config")


def _validate_output(path_value) -> Path:
    final = Path(path_value).absolute()
    if os.path.lexists(final):
        raise ValueError(f"output path must be fresh: {final}")
    parent = final.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"output parent is not usable: {parent}")
    return final


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        versions[name] = package_metadata.version(name)
    return versions


def require_matching_identity(accelerator, value: Any) -> None:
    """Reject a preflight identity disagreement collectively before model construction."""
    import torch
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    tensor = torch.tensor(list(hashlib.sha256(payload).digest()), dtype=torch.uint8,
                          device=accelerator.device)
    gathered = accelerator.gather(tensor).reshape(accelerator.num_processes, -1)
    if not bool(torch.equal(gathered, gathered[0].expand_as(gathered))):
        raise ValueError("preflight identities differ across ranks")


def compile_bound_forward(model, backend="inductor"):
    """Compile only the bound forward; failures are fatal and never fall back."""
    import torch
    torch._dynamo.config.suppress_errors = False
    model.forward = torch.compile(model.forward, backend=backend)
    return model


def execute(args, candidate: CandidateRecords, *, validate_only: bool) -> dict:
    """Run validation, or the fixed public recipe after readiness authorization."""
    recipe = MVPRecipe()
    final = _validate_output(args.output_dir)
    artifacts = verify_candidate_artifacts(candidate, args.db, args.evidence_db, args.panel)
    sif, _ = _snapshot_hash(args.sif, "SIF")
    config_path, _ = _snapshot_hash(args.config, "config")
    weights, run_config = _source_paths(args.init_from)
    if validate_only:
        return {
            "status": "candidate_valid", "output_use": OUTPUT_USE,
            "runtime_profile": "mvp_fixed_250x96", "authorization_scope": "exploratory_mvp",
            "production_authorized": False, "candidate_identities": candidate.identities,
            "counts": {"train": len(artifacts["train_rns"]), "valid": len(artifacts["valid_rns"])},
        }
    readiness = validate_launch_readiness(
        args.launch_readiness, args.launch_readiness_sha256, candidate,
        (args.db, args.evidence_db, args.panel, sif, config_path, weights, run_config),
        seed=args.seed, recipe=recipe,
    )
    return _train(args, candidate, artifacts, final, recipe, weights, run_config, readiness)


def _train(args, candidate, artifacts, final, recipe, weights, run_config, readiness) -> dict:
    """Construct and run the reviewed runtime only after every external gate."""
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    from transgenic.training.b5_runtime import load_b5_config, model_kwargs, resolve_checkpointing
    from transgenic.training.mvp_finalize import evaluate_and_publish_final
    from transgenic.training.mvp_runtime import build_optimizer, make_accelerator, run_fixed_updates
    from transgenic.training.weights_init import load_weights_only

    forbidden = ("TRANSGENIC_NO_GRAD_CKPT", "TRANSGENIC_CKPT_REENTRANT",
                 "TRANSGENIC_NO_COMPILE", "TRANSGENIC_COMPILE_SCOPE")
    present = [name for name in forbidden if os.environ.get(name)]
    if present:
        raise ValueError(f"MVP refuses checkpoint/compile override environment variables: {present}")
    if resolve_checkpointing(os.environ, recipe.required_world_size) != "nonreentrant":
        raise ValueError("MVP requires nonreentrant gradient checkpointing")
    accelerator = make_accelerator(recipe, mixed_precision="bf16")
    require_matching_identity(accelerator,
                              {"candidate": candidate.identities, "artifacts": artifacts["sha256"]})
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg_data = load_b5_config(str(args.config))
    config = HyenaTransgenicConfig(**model_kwargs(cfg_data), evidence_channels=8)
    model = transgenicForConditionalGeneration(config)
    initialization = load_weights_only(model, args.init_from)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = build_optimizer(model, recipe)
    compile_bound_forward(model)
    model, optimizer = accelerator.prepare(model, optimizer)

    aug_seed = 20260914 + args.seed
    train_ds = isoformDataHyena(args.db, mode="train", split="train", strict=True,
        species="Athaliana", evidence_db=args.evidence_db, evidence_k=8,
        aug_seed=aug_seed, return_tile_ids=True)
    valid_ds = isoformDataHyena(args.db, mode="train", split="valid", strict=True,
        species="Athaliana", evidence_db=args.evidence_db, evidence_k=8,
        aug_seed=aug_seed, eval_fraction=1, return_tile_ids=True)
    if train_ds._index_map != artifacts["train_rns"] or valid_ds._index_map != artifacts["valid_rns"]:
        raise ValueError("dataset index map does not match accepted inventory")
    rank = accelerator.process_index
    train_generator = torch.Generator().manual_seed(aug_seed + 10_000 * rank)
    valid_generator = torch.Generator().manual_seed(aug_seed + 1_000_000 + rank)
    if args.num_workers == 0:
        train_ds._aug_generator = torch.Generator().manual_seed(aug_seed + rank)
        train_ds._aug_key = (os.getpid(), None)
    train_sampler = DistributedSampler(train_ds, num_replicas=accelerator.num_processes,
        rank=rank, shuffle=True, seed=args.seed)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=accelerator.num_processes,
        rank=rank, shuffle=False, drop_last=False)
    loader_args = {"batch_size": 1, "num_workers": args.num_workers,
                   "collate_fn": hyena_collate_fn, "pin_memory": True}
    train_loader = DataLoader(train_ds, sampler=train_sampler, generator=train_generator, **loader_args)
    valid_loader = DataLoader(valid_ds, sampler=valid_sampler, generator=valid_generator, **loader_args)
    summary = run_fixed_updates(accelerator, model, optimizer, train_loader, recipe)
    metadata = {
        "identity_scope": "production",  # Finalizer identity shape only; not authorization.
        "runtime_profile": "mvp_fixed_250x96", "authorization_scope": "exploratory_mvp",
        "production_authorized": False, "output_use": OUTPUT_USE,
        "initialization_audit": {"audit": initialization,
            "audit_sha256": hashlib.sha256((json.dumps(initialization, sort_keys=True,
                separators=(",", ":")) + "\n").encode()).hexdigest(), "verification": "independently_verified"},
        "data_identities": {name: {"sha256": value, "verification": "independently_verified"}
                            for name, value in artifacts["sha256"].items()},
        "sif_code_identity": {"sif": {"sha256": _snapshot_hash(args.sif, "SIF")[1], "verification": "independently_verified"},
                              "code": {"sha256": readiness["bindings"]["consumed_code_manifest"]["sha256"],
                                       "verification": "independently_verified"}},
        "runtime_packages": _versions(), "normalized_recipe": asdict(recipe),
        "seed_augmentation": {"seed": args.seed, "aug_seed": aug_seed},
        "runtime_contract": {"compile": "bound_forward_inductor_no_fallback", "precision": "bf16",
            "checkpointing": "nonreentrant", "construction_order": "weights_optimizer_compile_prepare",
            "rank_augmentation": "aug_seed+rank only for private dropout/thinning stream when num_workers=0"},
    }
    # Safety contract: reread all mutable external identities immediately before publication.
    final_artifacts = verify_candidate_artifacts(candidate, args.db, args.evidence_db, args.panel)
    if final_artifacts["sha256"] != artifacts["sha256"]:
        raise ValueError("candidate artifact identity changed during training")
    validate_launch_readiness(
        args.launch_readiness, args.launch_readiness_sha256, candidate,
        (args.db, args.evidence_db, args.panel, args.sif, args.config, weights, run_config),
        seed=args.seed, recipe=recipe,
    )
    evaluate_and_publish_final(accelerator, model, valid_loader, artifacts["valid_rns"], final,
                               metadata=metadata, training_summary=summary)
    return {"status": "complete", "output_use": OUTPUT_USE, "final": str(final)}
