"""Strict exploratory-MVP CLI gates and wiring.

This is intentionally not a general training framework.  Public invocations use
one fixed 250x96 recipe and remain ``EXPLORATORY_MVP_NOT_FOR_PRODUCTION``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

OUTPUT_USE = "EXPLORATORY_MVP_NOT_FOR_PRODUCTION"
_SCOPE = "exploratory_mvp_candidate"
_INVENTORY_SCOPE = "exploratory_mvp_candidate"
_RECEIPTS_SCOPE = "exploratory_mvp_candidate"
_SHA_LEN = 64
_ACCEPTANCE_CHECKS = {
    "artifact_identity", "bam_identity", "corpus_identity", "inventory_counts",
    "inventory_exact_match", "panel_generation_ancestry_measured", "panel_identity",
    "production_remains_false", "qc_gate", "reference_identity", "source_commit", "success_pass",
}
_RECEIPT_LOCATIONS = {"local_git_object", "local_parent_repo", "pronghorn"}


@dataclass(frozen=True)
class CandidateRecords:
    acceptance: dict[str, Any]
    inventory: dict[str, Any]
    receipts: dict[str, Any]
    identities: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != _SHA_LEN or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _regular_file(value: os.PathLike[str] | str, label: str) -> Path:
    path = Path(value)
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return path.resolve()


def _json_record(path_value: os.PathLike[str] | str, expected_sha: str, label: str) -> tuple[Path, dict]:
    path = _regular_file(path_value, label)
    expected_sha = _digest(expected_sha, f"{label} SHA256")
    actual = _sha256(path)
    if actual != expected_sha:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha}, measured {actual}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, value


def _validate_inventory(value: dict) -> None:
    if value.get("schema_version") != 1 or value.get("scope") != _INVENTORY_SCOPE:
        raise ValueError("inventory must be schema 1 exploratory_mvp_candidate")
    records = value.get("records")
    fields = {"rn", "chromosome", "start0", "end0", "split", "is_rc"}
    if not isinstance(records, list) or not records:
        raise ValueError("inventory records must be a nonempty list")
    seen: set[int] = set()
    split_counts = {"train": 0, "valid": 0}
    previous = -1
    for row in records:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("inventory records require exact identity fields")
        rn = row["rn"]
        if isinstance(rn, bool) or not isinstance(rn, int) or rn <= previous or rn in seen:
            raise ValueError("inventory rn values must be unique ascending positive integers")
        previous = rn
        seen.add(rn)
        if row["split"] not in split_counts:
            raise ValueError("inventory contains an unknown split")
        split_counts[row["split"]] += 1
        if (not isinstance(row["chromosome"], str) or not row["chromosome"] or
                isinstance(row["start0"], bool) or not isinstance(row["start0"], int) or
                isinstance(row["end0"], bool) or not isinstance(row["end0"], int) or
                not 0 <= row["start0"] < row["end0"] or type(row["is_rc"]) is not bool):
            raise ValueError(f"inventory has malformed identity for rn={rn}")
    counts = value.get("counts")
    all_splits = counts.get("evidence_corpus_tile_all_splits") if isinstance(counts, dict) else None
    if (not isinstance(counts, dict) or counts.get("total") != len(records) or
            not isinstance(all_splits, dict) or set(all_splits) != {"train", "valid", "test"} or
            any(type(count) is not int or count < 0 for count in all_splits.values()) or
            all_splits["train"] != split_counts["train"] or
            all_splits["valid"] != split_counts["valid"] or
            counts.get("train") != split_counts["train"] or
            counts.get("valid") != split_counts["valid"]):
        raise ValueError("inventory counts do not reconcile with records")
    comparison = value.get("comparison")
    if (not isinstance(comparison, dict) or comparison.get("exact_ordered_identity_match") is not True or
            comparison.get("expected_count") != len(records) or comparison.get("source_count") != len(records)):
        raise ValueError("inventory exact-match comparison is not accepted")


def validate_candidate_records(
    acceptance_record: os.PathLike[str] | str,
    acceptance_sha256: str,
    inventory_record: os.PathLike[str] | str,
    receipts_record: os.PathLike[str] | str,
) -> CandidateRecords:
    """Validate the digest-bound CPU candidate without granting production use."""
    _, acceptance = _json_record(acceptance_record, acceptance_sha256, "acceptance record")
    if acceptance.get("schema_version") != 1 or acceptance.get("scope") != _SCOPE:
        raise ValueError("acceptance record must be schema 1 exploratory_mvp_candidate")
    decision = acceptance.get("decision")
    false_flags = ("production_training_accepted", "production_authorized",
                   "author_protocol_freeze", "scientific_approval")
    if (not isinstance(decision, dict) or decision.get("technical_acceptance") is not True or
            any(decision.get(name) is not False for name in false_flags)):
        raise ValueError("technical acceptance is required and all production flags must remain false")
    checks = acceptance.get("checks")
    if (not isinstance(checks, dict) or set(checks) != _ACCEPTANCE_CHECKS or
            any(value is not True for value in checks.values())):
        raise ValueError("acceptance requires the exact technical acceptance checks, all true")
    qc = acceptance.get("qc")
    if not isinstance(qc, dict):
        raise ValueError("acceptance QC is missing")
    valid, evidence = qc.get("valid_tiles_total"), qc.get("valid_tiles_with_evidence")
    if (isinstance(valid, bool) or not isinstance(valid, int) or isinstance(evidence, bool) or
            not isinstance(evidence, int) or not 0 < evidence <= valid):
        raise ValueError("valid evidence tile count must be positive and no greater than valid total")
    if qc.get("retained_overlapping_panel") != 0 or qc.get("retained_overlapping_test") != 0:
        raise ValueError("acceptance overlap QC must remain zero")
    source = acceptance.get("source_counts")
    count_names = ("members", "collapsed_rule_a", "collapsed_rule_b", "molecules_after_dedup",
                   "retained", "excluded_heldout_locus", "excluded_test_block", "excluded_unmapped")
    if (not isinstance(source, dict) or
            any(isinstance(source.get(name), bool) or not isinstance(source.get(name), int) or
                source[name] < 0 for name in count_names) or
            source["members"] - source["collapsed_rule_a"] - source["collapsed_rule_b"] != source["molecules_after_dedup"] or
            sum(source[name] for name in ("retained", "excluded_heldout_locus", "excluded_test_block", "excluded_unmapped")) != source["molecules_after_dedup"]):
        raise ValueError("acceptance source counts do not reconcile")
    companions = acceptance.get("companions")
    if not isinstance(companions, dict):
        raise ValueError("acceptance companion bindings are missing")
    inventory_sha = _digest(companions.get("inventory", {}).get("sha256"), "inventory companion SHA256")
    receipts_sha = _digest(companions.get("receipts", {}).get("sha256"), "receipts companion SHA256")
    _, inventory = _json_record(inventory_record, inventory_sha, "inventory companion")
    _, receipts = _json_record(receipts_record, receipts_sha, "receipts companion")
    _validate_inventory(inventory)
    if receipts.get("schema_version") != 1 or receipts.get("scope") != _RECEIPTS_SCOPE:
        raise ValueError("receipts must be schema 1 exploratory_mvp_candidate")
    receipt_rows = receipts.get("receipts")
    if (not isinstance(receipt_rows, list) or receipts.get("receipt_count") != len(receipt_rows) or
            any(not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes", "location"} or
                not isinstance(row["path"], str) or type(row["bytes"]) is not int or row["bytes"] < 0 or
                row["location"] not in _RECEIPT_LOCATIONS or
                _digest(row["sha256"], "receipt SHA256") != row["sha256"] for row in receipt_rows)):
        raise ValueError("receipt inventory is malformed")
    summary = acceptance.get("inventory_summary")
    counts = inventory["counts"]
    if (not isinstance(summary, dict) or summary.get("companion_sha256") != inventory_sha or
            summary.get("exact_match_to_evidence_corpus_tile") is not True or
            any(summary.get(name) != counts.get(name) for name in ("total", "train", "valid"))):
        raise ValueError("acceptance and inventory counts do not reconcile")
    return CandidateRecords(
        acceptance, inventory, receipts,
        {"acceptance": acceptance_sha256, "inventory": inventory_sha, "receipts": receipts_sha},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Exploratory MVP fixed-recipe trainer [{OUTPUT_USE}]",
        epilog=f"Outputs are always labeled {OUTPUT_USE}.",
    )
    parser.add_argument("--db", required=False)
    parser.add_argument("--evidence-db", required=False)
    parser.add_argument("--panel", required=False)
    parser.add_argument("--acceptance-record", required=False)
    parser.add_argument("--acceptance-sha256", required=False)
    parser.add_argument("--inventory-record", required=False)
    parser.add_argument("--receipts-record", required=False)
    parser.add_argument("--launch-readiness")
    parser.add_argument("--launch-readiness-sha256")
    parser.add_argument("--config", required=False)
    parser.add_argument("--init-from", required=False)
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--sif", required=False)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true",
                        help=f"validate candidate inputs only; {OUTPUT_USE}")
    return parser


def _required(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = ["--" + name.replace("_", "-") for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = ("db", "evidence_db", "panel", "acceptance_record", "acceptance_sha256",
              "inventory_record", "receipts_record", "config", "init_from", "output_dir", "sif", "seed")
    try:
        _required(args, common)
        if args.num_workers < 0:
            raise ValueError("--num-workers must be nonnegative")
        if not args.validate_only:
            _required(args, ("launch_readiness", "launch_readiness_sha256"))
        candidate = validate_candidate_records(args.acceptance_record, args.acceptance_sha256,
                                               args.inventory_record, args.receipts_record)
        # Heavy imports and all model/runtime construction stay behind record gates.
        from .mvp_cli_runtime import execute
        diagnostics = execute(args, candidate, validate_only=args.validate_only)
        print(json.dumps(diagnostics, sort_keys=True))
        return 0
    except (ValueError, OSError) as error:
        print(f"{OUTPUT_USE}: {error}", file=sys.stderr)
        return 2
