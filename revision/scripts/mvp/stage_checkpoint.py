#!/usr/bin/env python3
"""Stage a pinned B5 checkpoint into the immutable layout MVP init consumes.

Track A must-fix 2 (B6 phase 2). ``weights_init.load_weights_only`` and
``mvp_cli_runtime._source_paths`` resolve ``--init-from <dir>`` as:

  <dir>/model.safetensors           weights (regular file, exact name)
  <dir>/../run_config.json          recipe authority (embedded normalized recipe)
  <dir>/../runtime_identity.txt     optional but staged here as required provenance

This script therefore produces a self-contained staged run root:

  <dest>/
    run_config.json          copied from the source run root, re-hashed
    runtime_identity.txt     copied from the source run root, re-hashed
    checkpoint/
      model.safetensors      copied from the pinned epoch dir, re-hashed
      eval.json, meta.json   copied when present (provenance; meta.json is
                             snapshotted by the loader when it exists)
    staging_receipt.json     measured source/staged hashes (atomic)
    STAGING_DONE             marker, written last (renameat2 NOREPLACE)

Pass ``--init-from <dest>/checkpoint`` to the MVP CLI. Because the staged
copy is disjoint from ``runs/seed123/``, later B5 epochs or ``best`` symlink
flips cannot mutate the MVP initialization.

The 2.09 GB copy and hashing MUST run inside a compute or data-mover
allocation (must-fix 6), never on a login node. This script only reads the
source; it never writes into the source run root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _entry in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _sha256(path: Path) -> str:
    from make_launch_readiness import _sha256 as stable_hash
    return stable_hash(path)


def _copy_verified(source: Path, destination: Path, expected: str | None, label: str) -> dict:
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"FAIL: {label} source is not a regular file: {source}")
    source_sha = _sha256(source)
    if expected is not None and source_sha != expected:
        raise SystemExit(f"FAIL: {label} source SHA256 {source_sha} != expected {expected}")
    shutil.copyfile(source, destination)
    with destination.open('rb') as handle:
        os.fsync(handle.fileno())
    staged_sha = _sha256(destination)
    if staged_sha != source_sha:
        raise SystemExit(f"FAIL: {label} staged copy SHA256 {staged_sha} != source {source_sha}")
    return {
        "source_path": str(source.resolve()), "staged_path": str(destination.resolve()),
        "sha256": staged_sha, "bytes": destination.stat().st_size,
        "expected_sha256": expected,
    }


def _atomic_json(destination: Path, value: dict) -> None:
    from make_launch_readiness import _atomic_write, _canonical_bytes
    _atomic_write(destination, _canonical_bytes(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-epoch", required=True,
                        help="pinned concrete epoch dir, e.g. .../runs/seed123/epoch_05 "
                             "(never the mutable 'best' symlink)")
    parser.add_argument("--dest", required=True, help="fresh staged run root")
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-run-config-sha256", required=True)
    parser.add_argument("--expected-runtime-identity-sha256", required=True)
    parser.add_argument("--quota-record", required=True)
    parser.add_argument("--quota-sha256", required=True)
    parser.add_argument("--max-copy-bytes", type=int, required=True)
    parser.add_argument("--reserve-bytes", type=int, required=True)
    args = parser.parse_args(argv)

    from make_launch_readiness import require_allocation, _atomic_write, _canonical_bytes
    require_allocation()
    requested = Path(args.source_epoch).absolute()
    source_epoch = requested.resolve(strict=True)
    if requested != source_epoch or any(part in {'best', 'latest', 'latest_state'} for part in requested.parts):
        raise SystemExit("FAIL: refusing to stage via the mutable 'best' symlink; "
                         "pin a concrete epoch directory")
    source_root = source_epoch.parent
    destination = Path(args.dest).absolute()
    if os.path.lexists(destination):
        raise SystemExit(f"FAIL: staging destination must be fresh: {destination}")

    import time
    quota_path = Path(args.quota_record)
    from make_launch_readiness import json_snapshot
    quota = json_snapshot(quota_path, args.quota_sha256)
    size = sum(p.stat().st_size for p in [source_epoch / 'model.safetensors',
               source_root / 'run_config.json', source_root / 'runtime_identity.txt'] +
               [source_epoch / name for name in ('eval.json', 'meta.json') if (source_epoch / name).exists()])
    if (quota.get('path') != str(destination.parent.resolve()) or
            type(quota.get('available_bytes')) is not int or
            type(quota.get('measured_at_unix')) not in (int, float) or
            not 0 <= time.time() - quota['measured_at_unix'] <= 3600 or
            args.reserve_bytes < 0 or not 0 < size <= args.max_copy_bytes or
            min(quota['available_bytes'], shutil.disk_usage(destination.parent).free) < size + args.reserve_bytes):
        raise ValueError('quota/headroom/copy budget gate failed')
    # Fresh incomplete directory is never a consumable destination.
    final_destination = destination
    destination = destination.with_name(destination.name + '.partial.' + str(os.getpid()))
    destination.mkdir()
    checkpoint = destination / "checkpoint"
    checkpoint.mkdir()
    receipts = {
        "model.safetensors": _copy_verified(
            source_epoch / "model.safetensors", checkpoint / "model.safetensors",
            args.expected_weights_sha256, "weights"),
        "run_config.json": _copy_verified(
            source_root / "run_config.json", destination / "run_config.json",
            args.expected_run_config_sha256, "run_config.json"),
        "runtime_identity.txt": _copy_verified(
            source_root / "runtime_identity.txt", destination / "runtime_identity.txt",
            args.expected_runtime_identity_sha256, "runtime_identity.txt"),
    }
    for optional in ("eval.json", "meta.json"):
        candidate = source_epoch / optional
        if candidate.is_file():
            receipts[optional] = _copy_verified(candidate, checkpoint / optional, None, optional)

    run_config = json.loads((destination / "run_config.json").read_text(encoding="utf-8"))
    if not isinstance(run_config, dict) or run_config.get("seed") != 123:
        raise SystemExit("FAIL: staged run_config.json does not bind seed 123")

    receipt = {
        "schema_version": 1,
        "purpose": "mvp_init_checkpoint_staging",
        "output_use": "EXPLORATORY_MVP_NOT_FOR_PRODUCTION",
        "source_epoch": str(source_epoch),
        "staged_init_from": str(checkpoint),
        "files": receipts,
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt['staged_init_from'] = str(final_destination / 'checkpoint')
    for row in receipts.values():
        row['staged_path'] = str(final_destination / Path(row['staged_path']).relative_to(destination))
    receipt['quota_sha256'] = args.quota_sha256
    descriptor = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _atomic_json(destination / "staging_receipt.json", receipt)
    _atomic_write(destination / 'STAGING_DONE', (_sha256(destination / 'staging_receipt.json') + '\n').encode())
    from transgenic.training.mvp_finalize import _rename_noreplace
    _rename_noreplace(destination, final_destination)
    parent = os.open(final_destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    checkpoint = final_destination / 'checkpoint'
    print(json.dumps({"status": "staged", "init_from": str(checkpoint),
                      "weights_sha256": receipts["model.safetensors"]["sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
