from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from transgenic.training.mvp_cli import (
    CandidateRecords,
    OUTPUT_USE,
    build_parser,
    validate_candidate_records,
)
from transgenic.training.mvp_cli_runtime import (
    compile_bound_forward,
    verify_candidate_artifacts,
    validate_launch_readiness,
)
from transgenic.training.mvp_runtime import MVPRecipe

CONTEXT = Path("/tmp/b6-mvp-cli-context")
ACCEPTANCE = CONTEXT / "b6_mvp_technical_candidate_20260915.json"
INVENTORY = CONTEXT / "b6_mvp_candidate_inventories_20260915.json"
RECEIPTS = CONTEXT / "b6_mvp_candidate_receipts_20260915.json"
ACCEPTANCE_SHA = "faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e"


def canonical(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def test_help_is_lazy_and_keeps_experimental_label():
    result = subprocess.run(
        [sys.executable, "train/train_HyenaTransgenic_MVP.py", "--help"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0
    assert OUTPUT_USE in result.stdout
    lazy = subprocess.run(
        [sys.executable, "-c", "import sys; import transgenic.training.mvp_cli; print('torch' in sys.modules)"],
        text=True, capture_output=True, timeout=10,
    )
    assert lazy.stdout.strip() == "False"


@pytest.mark.parametrize("option", [
    "--resume", "--max-updates", "--world-size", "--lr", "--eager", "--production",
])
def test_public_parser_rejects_non_mvp_options(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args([option, "x"])


@pytest.mark.parametrize("seed", ["0", "4", "123"])
def test_public_parser_rejects_unapproved_seed(seed):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--seed", seed])


def test_actual_mode_without_readiness_stops_before_file_or_runtime_access():
    arguments = []
    for name in ("db", "evidence-db", "panel", "acceptance-record", "acceptance-sha256",
                 "inventory-record", "receipts-record", "config", "init-from", "output-dir", "sif"):
        arguments.extend((f"--{name}", "missing"))
    arguments.extend(("--seed", "1"))
    result = subprocess.run(
        [sys.executable, "train/train_HyenaTransgenic_MVP.py", *arguments],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 2
    assert "--launch-readiness" in result.stderr
    assert "Accelerator" not in result.stderr


def test_real_candidate_companions_validate_without_production_authorization():
    candidate = validate_candidate_records(
        ACCEPTANCE, ACCEPTANCE_SHA, INVENTORY, RECEIPTS
    )
    assert candidate.acceptance["decision"] == {
        "author_protocol_freeze": False,
        "blockers": [],
        "production_authorized": False,
        "production_training_accepted": False,
        "scientific_approval": False,
        "technical_acceptance": True,
        "technical_acceptance_scope": "digest-bound CPU evidence data candidate for subsequent CLI/runtime/GPU readiness gates",
    }
    assert len(candidate.inventory["records"]) == 9082


def test_candidate_rejects_production_scope_even_if_technically_accepted(tmp_path):
    acceptance = json.loads(ACCEPTANCE.read_text())
    acceptance["decision"]["production_authorized"] = True
    path = tmp_path / "acceptance.json"
    canonical(path, acceptance)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="production flags must remain false"):
        validate_candidate_records(path, digest, INVENTORY, RECEIPTS)


def test_candidate_rejects_companion_digest_mismatch(tmp_path):
    inventory = json.loads(INVENTORY.read_text())
    inventory["records"][0]["rn"] = 99
    path = tmp_path / "inventory.json"
    canonical(path, inventory)
    with pytest.raises(ValueError, match="inventory.*SHA256"):
        validate_candidate_records(ACCEPTANCE, ACCEPTANCE_SHA, path, RECEIPTS)


def test_candidate_rejects_invalid_qc_reconciliation(tmp_path):
    acceptance = json.loads(ACCEPTANCE.read_text())
    acceptance["qc"]["valid_tiles_with_evidence"] = acceptance["qc"]["valid_tiles_total"] + 1
    path = tmp_path / "acceptance.json"
    canonical(path, acceptance)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="valid evidence tile count"):
        validate_candidate_records(path, digest, INVENTORY, RECEIPTS)


def test_candidate_rejects_malformed_source_count_as_validation_error(tmp_path):
    acceptance = json.loads(ACCEPTANCE.read_text())
    acceptance["source_counts"]["members"] = "1046825"
    path = tmp_path / "acceptance.json"
    canonical(path, acceptance)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source counts"):
        validate_candidate_records(path, digest, INVENTORY, RECEIPTS)


def tiny_candidate(tmp_path):
    import duckdb

    db = tmp_path / "corpus.db"
    evidence = tmp_path / "evidence.db"
    panel = tmp_path / "panel.tsv"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE geneList(rn INTEGER, chromosome VARCHAR, start INTEGER, fin INTEGER, split VARCHAR, is_rc BOOLEAN, species_id VARCHAR)")
        con.execute("INSERT INTO geneList VALUES (1, 'Chr1', 10, 20, 'train', false, 'Athaliana'), (2, 'Chr2', 30, 40, 'valid', true, 'Athaliana')")
    with duckdb.connect(str(evidence)) as con:
        con.execute("CREATE TABLE corpus_tile(rn INTEGER, chrom VARCHAR, start0 INTEGER, end0 INTEGER, split VARCHAR, is_rc BOOLEAN)")
        con.execute("INSERT INTO corpus_tile VALUES (1, 'Chr1', 10, 20, 'train', false), (2, 'Chr2', 30, 40, 'valid', true)")
    panel.write_text("locus\tstratum\nAT1G01010\tmvp\n")
    ids = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
           (("corpus", db), ("evidence", evidence), ("panel", panel))}
    records = [
        {"rn": 1, "chromosome": "Chr1", "start0": 10, "end0": 20, "split": "train", "is_rc": False},
        {"rn": 2, "chromosome": "Chr2", "start0": 30, "end0": 40, "split": "valid", "is_rc": True},
    ]
    acceptance = {"artifact": {"sha256": ids["evidence"]},
                  "inputs": {"corpus": {"sha256": ids["corpus"]}, "panel": {"sha256": ids["panel"]}}}
    candidate = CandidateRecords(acceptance, {"records": records}, {},
                                 {"acceptance": "a" * 64, "inventory": "b" * 64, "receipts": "c" * 64})
    return candidate, db, evidence, panel


def test_candidate_artifacts_bind_hashes_and_exact_metadata(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    result = verify_candidate_artifacts(candidate, db, evidence, panel)
    assert result["train_rns"] == [1]
    assert result["valid_rns"] == [2]


def test_candidate_artifacts_reject_wal_and_metadata_mutation(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    Path(str(db) + ".wal").write_text("mutable")
    with pytest.raises(ValueError, match="WAL"):
        verify_candidate_artifacts(candidate, db, evidence, panel)
    Path(str(db) + ".wal").unlink()
    candidate.inventory["records"][0]["chromosome"] = "Chr9"
    with pytest.raises(ValueError, match="metadata"):
        verify_candidate_artifacts(candidate, db, evidence, panel)


def readiness_record(paths, candidate, seed=1):
    db, evidence, panel, sif, config, weights, run_config, manifest = paths
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    evidence_rows = {name: {"pass": True, "receipt_sha256": "d" * 64} for name in (
        "real_tokenizer", "real_hyena_step0_three_tiers", "generation_cache_parity",
        "bf16_compile", "four_rank_smoke", "resource_budget")}
    return {
        "schema_version": 1, "authorized_scope": "exploratory_mvp",
        "production_authorized": False, "ready_for_finetuning": True,
        "bindings": {
            "technical_acceptance_sha256": candidate.identities["acceptance"],
            "inventory_sha256": candidate.identities["inventory"],
            "receipts_sha256": candidate.identities["receipts"],
            "corpus_sha256": digest(db), "evidence_sha256": digest(evidence),
            "panel_sha256": digest(panel), "sif_sha256": digest(sif),
            "config_sha256": digest(config),
            "source_checkpoint": {"weights_path": str(Path(weights).resolve()),
                "weights_sha256": digest(weights), "run_config_path": str(Path(run_config).resolve()),
                "run_config_sha256": digest(run_config), "seed": 123},
            "consumed_code_manifest": {"path": str(Path(manifest).resolve()),
                "sha256": digest(manifest)},
            "recipe": MVPRecipe().__dict__, "seed": seed,
        },
        "evidence": evidence_rows,
    }


def test_bound_forward_compile_has_no_eager_fallback():
    import torch

    class Model(torch.nn.Module):
        def forward(self, value):
            return value + 1

    def raising_backend(graph, inputs):
        raise RuntimeError("compile exploded")

    model = compile_bound_forward(Model(), backend=raising_backend)
    with pytest.raises(RuntimeError, match="compile exploded"):
        model(torch.tensor(1))
    assert torch._dynamo.config.suppress_errors is False


def test_actual_two_rank_tiny_training_to_atomic_final(tmp_path):
    env = dict(os.environ)
    env["MVP_CLI_TEST_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = ".testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps:src"
    command = ["/home/wyim/miniconda3/envs/pdb/bin/torchrun", "--standalone", "--nproc-per-node=2",
               "tests/mvp_cli_probe.py"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    output = json.loads([line for line in result.stdout.splitlines() if line.startswith("{")][-1])
    assert output == {
        "all_rank_rng_distinct": True, "asymmetric_identity_all_failed": True,
        "audit_bound": True, "backend_called": True,
        "epochs_crossed": 3, "marker": True, "saved_keys_clean": True,
        "successful_updates": 2, "world_size": 2,
    }


def test_launch_readiness_binds_recipe_sources_and_evidence(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    sif, config = tmp_path / "image.sif", tmp_path / "config.json"
    init = tmp_path / "init"
    init.mkdir()
    weights, run_config = init / "model.safetensors", init / "run_config.json"
    code = tmp_path / "code.py"
    manifest = tmp_path / "code-manifest.json"
    for path, payload in ((sif, b"sif"), (config, b"{}\n"), (weights, b"weights"),
                          (run_config, b'{"seed":123}\n'), (code, b"pass\n")):
        path.write_bytes(payload)
    canonical(manifest, {"files": {str(code.resolve()): hashlib.sha256(code.read_bytes()).hexdigest()}})
    paths = (db, evidence, panel, sif, config, weights, run_config, manifest)
    readiness = tmp_path / "readiness.json"
    canonical(readiness, readiness_record(paths, candidate))
    validate_launch_readiness(readiness, hashlib.sha256(readiness.read_bytes()).hexdigest(),
                              candidate, paths[:-1], seed=1, recipe=MVPRecipe())
    wrong_seed = readiness_record(paths, candidate)
    wrong_seed["bindings"]["source_checkpoint"]["seed"] = 456
    canonical(readiness, wrong_seed)
    with pytest.raises(ValueError, match="source checkpoint seed"):
        validate_launch_readiness(readiness, hashlib.sha256(readiness.read_bytes()).hexdigest(),
                                  candidate, paths[:-1], seed=1, recipe=MVPRecipe())
    canonical(readiness, readiness_record(paths, candidate))
    code.write_text("changed\n")
    with pytest.raises(ValueError, match="consumed code"):
        validate_launch_readiness(readiness, hashlib.sha256(readiness.read_bytes()).hexdigest(),
                                  candidate, paths[:-1], seed=1, recipe=MVPRecipe())
