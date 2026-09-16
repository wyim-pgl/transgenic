"""Heavy preflight and execution for the exploratory MVP CLI."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class _VerifiedInputs:
    db: Path
    evidence: Path
    panel: Path
    sif: Path
    config: Path
    weights: Path
    run_config: Path
    readiness: Path
    final: Path
    snapshots: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass(frozen=True)
class _TestExecution:
    """Private programmatic injection for CPU integration tests; never exposed by the CLI."""
    recipe: MVPRecipe
    compile_backend: Any
    cpu: bool = True
    fail_phase: str | None = None
    fail_rank: int = 1
    mutate_before_final_recheck: Any = None
    observation: dict[str, Any] | None = None


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


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    item = path.stat()
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns


def _reject_wal(path: Path, label: str) -> None:
    wal = Path(str(path) + ".wal")
    if os.path.lexists(wal):
        raise ValueError(f"{label} has a WAL beside it: {wal}")


def _close_dataset(dataset) -> None:
    """Idempotently close this dataset's supported DuckDB handles."""
    if dataset is None:
        return
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    for name in ("_con", "_evidence_con"):
        connection = getattr(dataset, name, None)
        if connection is not None:
            connection.close()
            setattr(dataset, name, None)


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
        grouped_splits = con.execute(
            "SELECT split, count(*) FROM corpus_tile GROUP BY split"
        ).fetchall()
        test_rns = [row[0] for row in con.execute(
            "SELECT rn FROM corpus_tile WHERE split='test' ORDER BY rn"
        ).fetchall()]
    unexpected = [split for split, _ in grouped_splits if split not in {"train", "valid", "test"}]
    if unexpected:
        raise ValueError(f"unexpected corpus_tile split values: {unexpected}")
    all_splits = candidate.inventory["counts"]["evidence_corpus_tile_all_splits"]
    measured_splits = {name: 0 for name in ("train", "valid", "test")}
    measured_splits.update(grouped_splits)
    if measured_splits != all_splits:
        raise ValueError("candidate evidence split counts do not exactly match corpus_tile")
    if len(test_rns) != len(set(test_rns)) or set(test_rns) & {row[0] for row in evidence_rows}:
        raise ValueError("candidate test inventory identities are not unique and disjoint")
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


def _expected_code_files() -> tuple[Path, ...]:
    """Exact project-owned Python closure for this checkout; dependencies are version-bound separately."""
    root = Path(__file__).resolve().parents[3]
    package = Path(__file__).resolve().parents[1]
    return tuple(sorted((*package.rglob("*.py"), root / "train" / "train_HyenaTransgenic_MVP.py"),
                        key=lambda path: path.as_posix()))


def _manifest_files(path: Path) -> None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("consumed code manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {"files", "external_packages"}:
        raise ValueError("consumed code manifest requires exact files and external_packages objects")
    files = manifest["files"]
    expected_paths = {str(source) for source in _expected_code_files()}
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise ValueError("consumed code manifest does not match the exact executed source set")
    for source, expected in files.items():
        _, measured = _snapshot_hash(source, "consumed code file")
        if measured != _digest(expected, "consumed code SHA256"):
            raise ValueError(f"consumed code identity mismatch: {source}")
    expected_versions = {name: package_metadata.version(name) for name in
                         ("accelerate", "duckdb", "safetensors", "torch", "transformers")}
    if manifest["external_packages"] != expected_versions:
        raise ValueError("consumed code manifest external package versions do not match runtime")


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


def _identity_payload(value: Any) -> bytes:
    """Encode and validate one rank's identity without distributed operations."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def require_matching_identity(accelerator, payload: bytes) -> None:
    """Compare already-validated identity bytes in one collective."""
    import torch
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


def _runtime_contract(test_execution: bool) -> dict[str, Any]:
    return {
        "execution_mode": "private_test_injection" if test_execution else "public_cli",
        "compile_backend": "eager_counting_test" if test_execution else "inductor",
        "precision": "no" if test_execution else "bf16",
        "checkpointing": "nonreentrant",
        "construction_order": "weights_optimizer_compile_prepare",
        "rng_streams": {
            "canonical_aug_seed": "20260914+seed (molecule ranking)",
            "private_num_workers_0": "aug_seed+rank",
            "train_worker_generator": "aug_seed+1000000+rank",
            "valid_worker_generator": "aug_seed+2000000+rank",
            "distributed_sampler": "seed",
            "nonzero_worker_dataset_rng": "dataset implementation unchanged",
        },
    }


def _validate_initialization_readiness(audit, readiness, weights: Path, run_config: Path) -> None:
    source = readiness["bindings"]["source_checkpoint"]
    files = audit.get("files", {})
    expected = (("model.safetensors", weights, source["weights_sha256"]),
                ("run_config.json", run_config, source["run_config_sha256"]))
    if (audit.get("resolved_weights_path") != str(weights) or
            audit.get("resolved_checkpoint_dir") != str(weights.parent)):
        raise ValueError("load_weights_only resolved paths disagree with readiness")
    for name, path, digest in expected:
        record = files.get(name, {})
        if (record.get("path") != str(path) or
                record.get("before", {}).get("sha256") != digest or
                record.get("after", {}).get("sha256") != digest):
            raise ValueError(f"load_weights_only {name} audit disagrees with readiness")


def execute(args, candidate: CandidateRecords, *, validate_only: bool,
            _test: _TestExecution | None = None) -> dict:
    """Run validation, or the fixed public recipe after readiness authorization."""
    recipe = _test.recipe if _test is not None else MVPRecipe()
    final = _validate_output(args.output_dir)
    artifacts = verify_candidate_artifacts(candidate, args.db, args.evidence_db, args.panel)
    db, evidence, panel = (Path(artifacts["paths"][name]) for name in ("corpus", "evidence", "panel"))
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
    readiness_path, _ = _snapshot_hash(args.launch_readiness, "launch readiness")
    readiness = validate_launch_readiness(
        readiness_path, args.launch_readiness_sha256, candidate,
        (db, evidence, panel, sif, config_path, weights, run_config),
        seed=args.seed, recipe=recipe,
    )
    paths = {"db": db, "evidence": evidence, "panel": panel, "sif": sif,
             "config": config_path, "weights": weights, "run_config": run_config,
             "readiness": readiness_path}
    verified = _VerifiedInputs(
        db, evidence, panel, sif, config_path, weights, run_config, readiness_path, final,
        tuple((name, _stat_identity(path)) for name, path in paths.items()),
    )
    return _train(args.seed, args.num_workers, candidate, artifacts, verified, recipe,
                  readiness, args.launch_readiness_sha256, _test=_test)


def _train(seed, num_workers, candidate, artifacts, inputs: _VerifiedInputs, recipe,
           readiness, readiness_sha256, *, _test: _TestExecution | None = None) -> dict:
    """Construct and run the reviewed runtime only from canonical verified bindings."""
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    from transgenic.training.b5_runtime import load_b5_config, model_kwargs, resolve_checkpointing
    from transgenic.training.mvp_finalize import _collective_preflight, evaluate_and_publish_final
    from transgenic.training.mvp_runtime import build_optimizer, make_accelerator, run_fixed_updates
    from transgenic.training.weights_init import load_weights_only

    forbidden = ("TRANSGENIC_NO_GRAD_CKPT", "TRANSGENIC_CKPT_REENTRANT",
                 "TRANSGENIC_NO_COMPILE", "TRANSGENIC_COMPILE_SCOPE")
    present = [name for name in forbidden if os.environ.get(name)]
    if present:
        raise ValueError(f"MVP refuses checkpoint/compile override environment variables: {present}")
    if resolve_checkpointing(os.environ, recipe.required_world_size) != "nonreentrant":
        raise ValueError("MVP requires nonreentrant gradient checkpointing")

    snapshots = dict(inputs.snapshots)
    paths = {name: getattr(inputs, name) for name in snapshots}

    def unchanged(names):
        for name in names:
            if _stat_identity(paths[name]) != snapshots[name]:
                raise ValueError(f"verified {name} identity changed at an open boundary")
        _reject_wal(inputs.db, "corpus DB")
        _reject_wal(inputs.evidence, "evidence DB")

    def phase(accelerator, name, operation):
        def checked():
            if (_test is not None and _test.fail_phase == name and
                    accelerator.process_index == _test.fail_rank):
                raise RuntimeError(f"injected {name} setup failure")
            operation()
        try:
            _collective_preflight(accelerator, checked)
        except Exception as error:
            raise type(error)(f"{name} phase failed: {error}") from error

    accelerator = make_accelerator(
        recipe, mixed_precision="no" if _test is not None else "bf16",
        **({"cpu": True} if _test is not None and _test.cpu else {}),
    )
    train_ds = valid_ds = None
    body_failed = False
    try:
        identity = {}
        phase(accelerator, "identity", lambda: identity.setdefault("payload", _identity_payload(
            {"candidate": candidate.identities, "artifacts": artifacts["sha256"]})))
        require_matching_identity(accelerator, identity["payload"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = {}
        def configure():
            unchanged(("config",))
            cfg["data"] = load_b5_config(str(inputs.config))
            unchanged(("config",))
        phase(accelerator, "config", configure)

        state = {}
        def initialize():
            unchanged(("weights", "run_config"))
            config_kwargs = model_kwargs(cfg["data"])
            if _test is not None:
                config_kwargs["load_pretrained_encoder"] = False
            config = HyenaTransgenicConfig(**config_kwargs, evidence_channels=8)
            state["model"] = transgenicForConditionalGeneration(config)
            state["initialization"] = load_weights_only(state["model"], inputs.weights.parent)
            state["model"].gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            unchanged(("weights", "run_config"))
        phase(accelerator, "initialization", initialize)

        def final_readiness():
            unchanged(tuple(paths))
            if _snapshot_hash(inputs.config, "config")[1] != readiness["bindings"]["config_sha256"]:
                raise ValueError("loaded config bytes disagree with readiness")
            _validate_initialization_readiness(
                state["initialization"], readiness, inputs.weights, inputs.run_config)
        phase(accelerator, "final-readiness", final_readiness)

        def compile_setup():
            state["optimizer"] = build_optimizer(state["model"], recipe)
            compile_bound_forward(state["model"],
                                  backend=_test.compile_backend if _test is not None else "inductor")
        phase(accelerator, "compile", compile_setup)
        model, optimizer = accelerator.prepare(state["model"], state["optimizer"])

        aug_seed = 20260914 + seed
        loaders = {}
        def datasets():
            nonlocal train_ds, valid_ds
            unchanged(("db", "evidence"))
            train_ds = isoformDataHyena(str(inputs.db), mode="train", split="train", strict=True,
                species="Athaliana", evidence_db=str(inputs.evidence), evidence_k=8,
                aug_seed=aug_seed, return_tile_ids=True)
            valid_ds = isoformDataHyena(str(inputs.db), mode="train", split="valid", strict=True,
                species="Athaliana", evidence_db=str(inputs.evidence), evidence_k=8,
                aug_seed=aug_seed, eval_fraction=1, return_tile_ids=True)
            unchanged(("db", "evidence"))
            if train_ds._index_map != artifacts["train_rns"] or valid_ds._index_map != artifacts["valid_rns"]:
                raise ValueError("dataset index map does not match accepted inventory")
            rank = accelerator.process_index
            private_aug_seed = aug_seed + rank
            train_worker_seed = aug_seed + 1_000_000 + rank
            valid_worker_seed = aug_seed + 2_000_000 + rank
            private_generator = None
            if num_workers == 0:
                private_generator = torch.Generator().manual_seed(private_aug_seed)
                train_ds._aug_generator = private_generator
                train_ds._aug_key = (os.getpid(), None)
            train_generator = torch.Generator().manual_seed(train_worker_seed)
            valid_generator = torch.Generator().manual_seed(valid_worker_seed)
            if _test is not None and _test.observation is not None:
                _test.observation["augmentation_streams"] = {
                    "rank": rank, "canonical_aug_seed": aug_seed, "sampler": seed,
                    "private": private_aug_seed, "train_worker": train_worker_seed,
                    "valid_worker": valid_worker_seed,
                }
                _test.observation["generator_ids"] = {
                    "private": id(private_generator), "train_worker": id(train_generator),
                    "valid_worker": id(valid_generator),
                }
            loader_args = {"batch_size": 1, "num_workers": num_workers,
                           "collate_fn": hyena_collate_fn, "pin_memory": True}
            loaders["train"] = DataLoader(train_ds, sampler=DistributedSampler(
                train_ds, num_replicas=accelerator.num_processes, rank=rank,
                shuffle=True, seed=seed), generator=train_generator, **loader_args)
            loaders["valid"] = DataLoader(valid_ds, sampler=DistributedSampler(
                valid_ds, num_replicas=accelerator.num_processes, rank=rank,
                shuffle=False, drop_last=False), generator=valid_generator, **loader_args)
        phase(accelerator, "dataset", datasets)

        summary = run_fixed_updates(accelerator, model, optimizer, loaders["train"], recipe)
        if _test is not None and _test.observation is not None:
            _test.observation["training_summary"] = dict(summary)
        initialization = state["initialization"]
        metadata = {
            "identity_scope": "test" if _test is not None else "production",
            "runtime_profile": "mvp_fixed_250x96", "authorization_scope": "exploratory_mvp",
            "production_authorized": False, "output_use": OUTPUT_USE,
            "initialization_audit": {"audit": initialization,
                "audit_sha256": hashlib.sha256((json.dumps(initialization, sort_keys=True,
                    separators=(",", ":")) + "\n").encode()).hexdigest(),
                "verification": "independently_verified"},
            "data_identities": {name: {"sha256": value, "verification": "independently_verified"}
                                for name, value in artifacts["sha256"].items()},
            "sif_code_identity": {"sif": {"sha256": _snapshot_hash(inputs.sif, "SIF")[1],
                                            "verification": "independently_verified"},
                                  "code": {"sha256": readiness["bindings"]["consumed_code_manifest"]["sha256"],
                                           "verification": "independently_verified"}},
            "runtime_packages": _versions(), "normalized_recipe": asdict(recipe),
            "seed_augmentation": {"seed": seed, "aug_seed": aug_seed},
            "runtime_contract": _runtime_contract(_test is not None),
        }

        def prepublication_prepare():
            if (accelerator.is_main_process and _test is not None and
                    _test.mutate_before_final_recheck is not None):
                _test.mutate_before_final_recheck()

        def prepublication_check():
            # Evaluation is complete; release read handles before the final independent DB audit.
            _close_dataset(train_ds)
            _close_dataset(valid_ds)
            unchanged(tuple(paths))
            final_artifacts = verify_candidate_artifacts(candidate, inputs.db, inputs.evidence, inputs.panel)
            if final_artifacts["sha256"] != artifacts["sha256"]:
                raise ValueError("candidate artifact identity changed during evaluation")
            validate_launch_readiness(
                inputs.readiness, readiness_sha256, candidate,
                (inputs.db, inputs.evidence, inputs.panel, inputs.sif, inputs.config,
                 inputs.weights, inputs.run_config), seed=seed, recipe=recipe)

        evaluate_and_publish_final(
            accelerator, model, loaders["valid"], artifacts["valid_rns"], inputs.final,
            metadata=metadata, training_summary=summary,
            prepublication_prepare=(prepublication_prepare if _test is not None and
                                    _test.mutate_before_final_recheck is not None else None),
            prepublication_check=prepublication_check)
        return {"status": "complete", "output_use": OUTPUT_USE, "final": str(inputs.final)}
    except BaseException:
        body_failed = True
        raise
    finally:
        cleanup_errors = []
        for dataset in (train_ds, valid_ds):
            try:
                _close_dataset(dataset)
            except Exception as error:
                cleanup_errors.append(error)
        try:
            accelerator.end_training()
        except Exception as error:
            cleanup_errors.append(error)
        if cleanup_errors and not body_failed:
            raise cleanup_errors[0]
