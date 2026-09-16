#!/usr/bin/env python3
"""Generate or check an exploratory-MVP launch-readiness record.

Track A (B6 phase 2). The record schema is defined by the consumer:
``transgenic.training.mvp_cli_runtime.validate_launch_readiness`` (f847cb8,
mvp_cli_runtime.py:175-236). This tool never synthesizes evidence: every
``evidence`` row binds a measured preflight receipt file (JSON with
``"pass": true``) plus its sibling ``<name>.DONE`` marker, and generation
fails closed if any receipt is missing, unreadable, or not passing.

Subcommands:
  generate  Build the readiness JSON (+ consumed-code manifest) from measured
            inputs. Output paths must be fresh; writes are atomic
            (tmp + renameat2(RENAME_NOREPLACE)). Prints the record SHA256.
  check     Directly call validate_candidate_records() and
            validate_launch_readiness() against a generated record.
            This exists because ``--validate-only`` on the MVP CLI skips the
            readiness requirement entirely (synthesis must-fix 3): the only
            way to exercise the readiness path without launching training is
            this direct call.

Scope: EXPLORATORY_MVP_NOT_FOR_PRODUCTION. ``production_authorized`` is
always false; this tool cannot grant production authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# The six evidence names required by mvp_cli_runtime._REQUIRED_EVIDENCE and
# the preflight stage that produces each receipt file (<name>.json + <name>.DONE).
EVIDENCE_RECEIPTS = {
    "real_tokenizer": "p1_identity",
    "real_hyena_step0_three_tiers": "p2_step0_determinism",
    "generation_cache_parity": "p3_generation_parity",
    "bf16_compile": "p4_compile_topology",
    "four_rank_smoke": "p4_compile_topology",
    "resource_budget": "p4_compile_topology",
}
_EXTERNAL_PACKAGES = ("accelerate", "duckdb", "safetensors", "torch", "transformers")


def require_allocation() -> str:
    """Operational guard, not authentication; never hash large inputs on login nodes."""
    allocation = os.environ.get("SLURM_JOB_ID") or os.environ.get("MVP_DATA_MOVER_ALLOCATION_ID")
    if not allocation:
        raise ValueError("compute/data-mover allocation required before copying or hashing")
    return allocation


def _sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'file changed during hashing: {path}')
    return digest.hexdigest()


def json_snapshot(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'JSON input/marker digest mismatch: {path}')
    return json.loads(payload)


def tooling_identity():
    directory = Path(__file__).resolve().parent
    return {str(directory / name): _sha256(directory / name) for name in
            ('make_launch_readiness.py', 'stage_checkpoint.py', 'run_bounded.py', 'run_training.py')}


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(destination: Path, payload: bytes) -> None:
    """Write payload to a fresh path atomically; refuse to replace anything."""
    from transgenic.training.mvp_finalize import _rename_noreplace

    destination = destination.absolute()
    if os.path.lexists(destination):
        raise SystemExit(f"FAIL: output path must be fresh: {destination}")
    staging = destination.parent / (destination.name + ".tmp." + str(os.getpid()))
    if os.path.lexists(staging):
        raise SystemExit(f"FAIL: staging path already exists: {staging}")
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(staging, destination)
    except BaseException:
        if os.path.lexists(staging):
            os.unlink(staging)
        raise
    parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _regular(path_value: str, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise SystemExit(f"FAIL: {label} is not a readable regular file: {path}")
    return path.resolve()


def _load_receipt(preflight_dir: Path, name: str, bindings_sha256: str) -> dict:
    """Fail-closed load of one measured preflight receipt + DONE marker."""
    directory = preflight_dir / name if (preflight_dir / name).is_dir() else preflight_dir
    receipt = directory / f"{name}.json"
    marker = directory / f"{name}.DONE"
    if (directory / f"{name}.FAILED.json").exists():
        raise ValueError(f"failed preflight cannot authorize readiness: {name}")
    if not receipt.is_file():
        raise SystemExit(f"FAIL: missing measured receipt: {receipt}")
    if not marker.is_file():
        raise SystemExit(f"FAIL: receipt {receipt.name} has no DONE marker: {marker}")
    try:
        receipt_sha = marker.read_text().strip()
        value = json_snapshot(receipt, receipt_sha)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"FAIL: receipt {receipt} is not valid JSON: {error}")
    if not isinstance(value, dict) or value.get("pass") is not True:
        raise SystemExit(f"FAIL: receipt {receipt} does not record a measured pass")
    expected_stage = EVIDENCE_RECEIPTS[name]
    if value.get("preflight") != expected_stage:
        raise SystemExit(f"FAIL: receipt {receipt} preflight stage "
                         f"{value.get('preflight')!r} != expected {expected_stage!r}")
    if marker.read_text().strip() != _sha256(receipt):
        raise ValueError(f"receipt marker digest mismatch: {name}")
    if (value.get("bindings_sha256") != bindings_sha256 or
            value.get("exit_code") != 0 or value.get("timed_out") is not False or
            value.get("signal") is not None):
        raise ValueError(f"receipt binding/execution failure: {name}")
    from run_bounded import validate_measurement
    validate_measurement(name, value.get("measurement"))
    return {
        "pass": True,
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": receipt_sha,
        "receipt_bytes": receipt.stat().st_size,
    }


def _consumed_code_manifest() -> dict:
    from importlib.metadata import version
    from transgenic.training.mvp_cli_runtime import _expected_code_files

    return {
        "files": {str(path): _sha256(path) for path in _expected_code_files()},
        "external_packages": {name: version(name) for name in _EXTERNAL_PACKAGES},
    }


def _source_checkpoint_binding(init_from: str) -> dict:
    """Bind the staged checkpoint identity exactly as mvp_cli_runtime._source_paths resolves it."""
    from transgenic.training.mvp_cli_runtime import _source_paths

    requested = Path(init_from).absolute()
    if any(part in {"best", "latest", "latest_state"} for part in requested.parts) or requested.resolve() != requested:
        raise ValueError("source checkpoint must be concrete, not a symlink or mutable alias")
    checkpoint = requested if requested.is_dir() else requested.parent
    for source_path in (checkpoint / 'model.safetensors', checkpoint.parent / 'run_config.json',
                        checkpoint.parent / 'runtime_identity.txt'):
        if source_path.resolve() != source_path.absolute():
            raise ValueError('checkpoint files and sidecars must not be symlinked')
    weights, run_config = _source_paths(init_from)
    try:
        config_value = json.loads(run_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"FAIL: source run_config.json is not valid JSON: {error}")
    if not isinstance(config_value, dict) or config_value.get("seed") != 123:
        raise SystemExit("FAIL: source checkpoint run_config.json must bind seed 123 "
                         "(B5 seed123 identity); refusing to fabricate the binding")
    binding = {
        "weights_path": str(weights),
        "weights_sha256": _sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "run_config_path": str(run_config),
        "run_config_sha256": _sha256(run_config),
        "seed": 123,
    }
    runtime_identity = run_config.parent / "runtime_identity.txt"
    if runtime_identity.is_file():
        binding["runtime_identity_path"] = str(runtime_identity.resolve())
        binding["runtime_identity_sha256"] = _sha256(runtime_identity)
    else:
        raise SystemExit(f"FAIL: staged run root lacks runtime_identity.txt: {runtime_identity} "
                         "(stage_checkpoint.py copies it; refusing an unattributed checkpoint)")
    return binding


def generate(args: argparse.Namespace) -> int:
    from transgenic.training.mvp_runtime import MVPRecipe

    acceptance_sha = args.acceptance_sha256
    inventory = _regular(args.inventory_record, "inventory record")
    receipts = _regular(args.receipts_record, "receipts record")
    acceptance = _regular(args.acceptance_record, "acceptance record")
    measured_acceptance = _sha256(acceptance)
    if measured_acceptance != acceptance_sha:
        raise SystemExit(f"FAIL: acceptance record SHA256 mismatch: expected {acceptance_sha}, "
                         f"measured {measured_acceptance}")

    from transgenic.training.mvp_cli import validate_candidate_records
    candidate = validate_candidate_records(acceptance, acceptance_sha, inventory, receipts)
    manifest = _consumed_code_manifest()
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        if manifest_path.read_bytes() != _canonical_bytes(manifest):
            raise ValueError("consumed code manifest changed since prepare")
    else:
        _atomic_write(manifest_path, _canonical_bytes(manifest))
    manifest_path = manifest_path.resolve()

    record = {
        "schema_version": 1,
        "authorized_scope": "exploratory_mvp",
        "production_authorized": False,
        "ready_for_finetuning": True,
        "output_use": "EXPLORATORY_MVP_NOT_FOR_PRODUCTION",
        "bindings": {
            "technical_acceptance_sha256": acceptance_sha,
            "inventory_sha256": _sha256(inventory),
            "receipts_sha256": _sha256(receipts),
            "corpus_sha256": _sha256(_regular(args.db, "corpus db")),
            "evidence_sha256": _sha256(_regular(args.evidence_db, "evidence db")),
            "panel_sha256": _sha256(_regular(args.panel, "panel")),
            "sif_sha256": _sha256(_regular(args.sif, "SIF")),
            "config_sha256": _sha256(_regular(args.config, "config")),
            "source_checkpoint": _source_checkpoint_binding(args.init_from),
            "consumed_code_manifest": {"path": str(manifest_path),
                                       "sha256": _sha256(manifest_path)},
            "tooling": tooling_identity(),
            "recipe": asdict(MVPRecipe()),
            "seed": args.seed,
        },
    }
    bindings = record["bindings"]
    for key, expected in (("corpus_sha256", candidate.acceptance['inputs']['corpus']['sha256']),
                          ("panel_sha256", candidate.acceptance['inputs']['panel']['sha256']),
                          ("evidence_sha256", candidate.acceptance['artifact']['sha256'])):
        if bindings[key] != expected:
            raise ValueError(f"accepted candidate input mismatch: {key}")
    # Operator-selected pins are mandatory; never follow a mutable 'best'.
    for prefix in ("weights", "run_config", "runtime_identity"):
        if bindings["source_checkpoint"][prefix + "_sha256"] != getattr(args, "expected_" + prefix + "_sha256"):
            raise ValueError(f"source checkpoint {prefix} pin mismatch")
    bindings_sha = hashlib.sha256(_canonical_bytes(bindings)).hexdigest()
    if args.command == "prepare":
        _atomic_write(Path(args.output), _canonical_bytes({"bindings": bindings, "bindings_sha256": bindings_sha}))
        return 0
    preflight_dir = Path(args.preflight_dir)
    record["evidence"] = {name: _load_receipt(preflight_dir, name, bindings_sha)
                          for name in sorted(EVIDENCE_RECEIPTS)}
    payload = _canonical_bytes(record)
    # Direct consumer validation BEFORE publishing a ready record.
    import tempfile
    from transgenic.training.mvp_cli_runtime import _source_paths, validate_launch_readiness
    paths = tuple(_regular(getattr(args, name), name) for name in
                  ("db", "evidence_db", "panel", "sif", "config")) + _source_paths(args.init_from)
    with tempfile.TemporaryDirectory(dir=Path(args.output).parent) as temporary:
        trial = Path(temporary) / "readiness.json"
        trial.write_bytes(payload)
        validate_launch_readiness(trial, hashlib.sha256(payload).hexdigest(), candidate,
                                  paths, seed=args.seed, recipe=MVPRecipe())
    _atomic_write(Path(args.output), payload)
    print(json.dumps({
        "status": "generated",
        "output_use": "EXPLORATORY_MVP_NOT_FOR_PRODUCTION",
        "readiness_path": str(Path(args.output).resolve()),
        "readiness_sha256": hashlib.sha256(payload).hexdigest(),
        "consumed_code_manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


def check(args: argparse.Namespace) -> int:
    """Directly exercise the consumer's readiness validation path (must-fix 3)."""
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import _source_paths, validate_launch_readiness
    from transgenic.training.mvp_runtime import MVPRecipe

    candidate = validate_candidate_records(args.acceptance_record, args.acceptance_sha256,
                                           args.inventory_record, args.receipts_record)
    weights, run_config = _source_paths(args.init_from)
    paths = tuple(_regular(value, name) for name, value in (
        ("corpus db", args.db), ("evidence db", args.evidence_db), ("panel", args.panel),
        ("SIF", args.sif), ("config", args.config))) + (weights, run_config)
    record = validate_launch_readiness(args.readiness, args.readiness_sha256, candidate, paths,
                                       seed=args.seed, recipe=MVPRecipe())
    bindings = record['bindings']
    if bindings.get('tooling') != tooling_identity():
        raise ValueError('readiness tooling identity changed')
    if _source_checkpoint_binding(args.init_from) != bindings['source_checkpoint']:
        raise ValueError('checkpoint sidecar identity changed')
    binding_sha = hashlib.sha256(_canonical_bytes(bindings)).hexdigest()
    for name, row in record['evidence'].items():
        current = _load_receipt(Path(row['receipt_path']).parent, name, binding_sha)
        if current != row:
            raise ValueError(f'measured receipt changed: {name}')
    print(json.dumps({
        "status": "readiness_valid",
        "output_use": "EXPLORATORY_MVP_NOT_FOR_PRODUCTION",
        "note": ("validate_launch_readiness() accepted this record for the given inputs; "
                 "this is NOT production authorization and NOT GPU certification"),
    }, sort_keys=True))
    return 0


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("db", "evidence-db", "panel", "sif", "config", "init-from",
                 "acceptance-record", "acceptance-sha256", "inventory-record",
                 "receipts-record"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "generate"):
        generate_parser = subcommands.add_parser(command)
        _add_input_arguments(generate_parser)
        generate_parser.add_argument("--preflight-dir", required=command == "generate")
        generate_parser.add_argument("--output", required=True)
        generate_parser.add_argument("--manifest-output", required=True)
        for prefix in ("weights", "run-config", "runtime-identity"):
            generate_parser.add_argument("--expected-" + prefix + "-sha256", required=True)
    check_parser = subcommands.add_parser("check", help="validate a readiness record directly")
    _add_input_arguments(check_parser)
    check_parser.add_argument("--readiness", required=True)
    check_parser.add_argument("--readiness-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_allocation()
        if args.command in {"prepare", "generate"}:
            return generate(args)
        return check(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
