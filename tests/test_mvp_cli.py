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
    _validate_inventory,
)
from transgenic.training.mvp_cli_runtime import (
    _collect_runtime_hardware,
    _expected_code_files,
    _manifest_files,
    _runtime_contract,
    _validate_initialization_readiness,
    _validate_public_hardware,
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


def test_candidate_rejects_fabricated_check_even_when_true(tmp_path):
    acceptance = json.loads(ACCEPTANCE.read_text())
    acceptance["checks"]["fabricated_pass"] = True
    path = tmp_path / "acceptance.json"
    canonical(path, acceptance)
    with pytest.raises(ValueError, match="exact technical acceptance checks"):
        validate_candidate_records(path, hashlib.sha256(path.read_bytes()).hexdigest(), INVENTORY, RECEIPTS)


@pytest.mark.parametrize("all_splits", [
    {"train": 7635, "valid": 1447},
    {"train": 7635, "valid": 1447, "test": 1388, "other": 0},
    {"train": 7635, "valid": 1447, "test": True},
])
def test_candidate_inventory_requires_exact_integral_evidence_split_counts(all_splits):
    inventory = json.loads(INVENTORY.read_text())
    inventory["counts"]["evidence_corpus_tile_all_splits"] = all_splits
    with pytest.raises(ValueError, match="inventory counts"):
        _validate_inventory(inventory)


def test_candidate_rejects_boolean_receipt_location_with_recomputed_bindings(tmp_path):
    acceptance = json.loads(ACCEPTANCE.read_text())
    receipts = json.loads(RECEIPTS.read_text())
    receipts["receipts"][0]["location"] = False
    receipts_path = tmp_path / "receipts.json"
    canonical(receipts_path, receipts)
    acceptance["companions"]["receipts"]["sha256"] = hashlib.sha256(receipts_path.read_bytes()).hexdigest()
    acceptance_path = tmp_path / "acceptance.json"
    canonical(acceptance_path, acceptance)
    with pytest.raises(ValueError, match="receipt inventory"):
        validate_candidate_records(acceptance_path, hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
                                   INVENTORY, receipts_path)


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
    inventory = {"records": records, "counts": {"total": 2, "train": 1, "valid": 1,
        "evidence_corpus_tile_all_splits": {"train": 1, "valid": 1, "test": 0}}}
    candidate = CandidateRecords(acceptance, inventory, {},
                                 {"acceptance": "a" * 64, "inventory": "b" * 64, "receipts": "c" * 64})
    return candidate, db, evidence, panel


def test_candidate_artifacts_bind_hashes_and_exact_metadata(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    result = verify_candidate_artifacts(candidate, db, evidence, panel)
    assert result["train_rns"] == [1]
    assert result["valid_rns"] == [2]


def test_candidate_artifacts_reject_wal_and_metadata_mutation(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    alias = tmp_path / "corpus-alias.db"
    alias.symlink_to(db)
    with pytest.raises(ValueError, match="regular file"):
        verify_candidate_artifacts(candidate, alias, evidence, panel)
    Path(str(db) + ".wal").write_text("mutable")
    with pytest.raises(ValueError, match="WAL"):
        verify_candidate_artifacts(candidate, db, evidence, panel)
    Path(str(db) + ".wal").unlink()
    candidate.inventory["records"][0]["chromosome"] = "Chr9"
    with pytest.raises(ValueError, match="metadata"):
        verify_candidate_artifacts(candidate, db, evidence, panel)


def test_candidate_artifacts_reject_fabricated_test_count(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    candidate.inventory["counts"]["evidence_corpus_tile_all_splits"]["test"] = 999999
    with pytest.raises(ValueError, match="split counts"):
        verify_candidate_artifacts(candidate, db, evidence, panel)


@pytest.mark.parametrize("split", ["mystery", None])
def test_candidate_artifacts_reject_unknown_or_null_evidence_split_with_updated_digest(tmp_path, split):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    import duckdb
    with duckdb.connect(str(evidence)) as con:
        con.execute("INSERT INTO corpus_tile VALUES (3, 'Chr3', 50, 60, ?, false)", [split])
    candidate.acceptance["artifact"]["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpected corpus_tile split"):
        verify_candidate_artifacts(candidate, db, evidence, panel)


def test_candidate_artifacts_reject_extra_known_split_row_with_updated_digest(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    import duckdb
    with duckdb.connect(str(evidence)) as con:
        con.execute("INSERT INTO corpus_tile VALUES (3, 'Chr3', 50, 60, 'train', false)")
    candidate.acceptance["artifact"]["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="split counts"):
        verify_candidate_artifacts(candidate, db, evidence, panel)


def test_code_manifest_requires_exact_checkout_sources_and_external_versions(tmp_path):
    files = _expected_code_files()
    manifest = tmp_path / "manifest.json"
    canonical(manifest, {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "external_packages": {"accelerate": "0", "duckdb": "0", "safetensors": "0",
                              "torch": "0", "transformers": "0"},
    })
    with pytest.raises(ValueError, match="external package version"):
        _manifest_files(manifest)
    from importlib.metadata import version
    canonical(manifest, {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "external_packages": {name: version(name) for name in
                              ("accelerate", "duckdb", "safetensors", "torch", "transformers")},
    })
    _manifest_files(manifest)
    fake = tmp_path / "pass.py"
    fake.write_text("pass\n")
    value = json.loads(manifest.read_text())
    value["files"] = {str(fake.resolve()): hashlib.sha256(fake.read_bytes()).hexdigest()}
    canonical(manifest, value)
    with pytest.raises(ValueError, match="exact executed source set"):
        _manifest_files(manifest)


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


def test_load_audit_checkpoint_mismatch_is_rejected_before_optimizer(tmp_path):
    checkpoint = tmp_path / "epoch"
    checkpoint.mkdir()
    weights, run_config = checkpoint / "model.safetensors", tmp_path / "run_config.json"
    weights.write_bytes(b"weights")
    run_config.write_bytes(b"config")
    readiness = {"bindings": {"source_checkpoint": {
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "run_config_sha256": hashlib.sha256(run_config.read_bytes()).hexdigest(),
    }}}
    audit = {"resolved_weights_path": str(weights), "resolved_checkpoint_dir": str(checkpoint),
             "files": {"model.safetensors": {"path": str(weights), "after": {"sha256": "0" * 64}},
                       "run_config.json": {"path": str(run_config), "after": {
                           "sha256": readiness["bindings"]["source_checkpoint"]["run_config_sha256"]}}}}
    with pytest.raises(ValueError, match="model.safetensors audit disagrees"):
        _validate_initialization_readiness(audit, readiness, weights, run_config)


def _gpu_fact(rank, *, uuid=None, **changes):
    fact = {
        "rank": rank, "local_rank": rank, "world_size": 4,
        "distributed_type": "MULTI_GPU", "device_type": "cuda",
        "device_index": rank, "current_cuda_device": rank,
        "cuda_available": True, "cuda_device_count": 4,
        "device_name": f"GPU {rank}", "device_uuid": uuid or f"GPU-{rank}",
        "capability": [8, 0], "bf16_supported": True,
        "torch_version": "2.6.0+cu124", "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
    }
    fact.update(changes)
    return fact


def test_public_hardware_gate_accepts_only_unit_synthetic_four_gpu_topology():
    measured = _validate_public_hardware([_gpu_fact(rank) for rank in range(4)])
    assert measured[2]["device_uuid"] == "GPU-2"


@pytest.mark.parametrize("change,message", [
    ({"device_type": "cpu", "cuda_available": False}, "CUDA device"),
    ({"world_size": 3}, "world size 4"),
    ({"bf16_supported": False}, "native bf16"),
    ({"local_rank": 0}, "rank/local-rank/device mapping"),
])
def test_public_hardware_gate_rejects_incorrect_rank_facts(change, message):
    facts = [_gpu_fact(rank) for rank in range(4)]
    facts[1].update(change)
    with pytest.raises(RuntimeError, match=message):
        _validate_public_hardware(facts)


def test_public_hardware_gate_rejects_duplicate_or_missing_gpu_uuid():
    facts = [_gpu_fact(rank) for rank in range(4)]
    facts[3]["device_uuid"] = facts[0]["device_uuid"]
    with pytest.raises(RuntimeError, match="distinct physical GPU UUID"):
        _validate_public_hardware(facts)
    facts[3]["device_uuid"] = None
    with pytest.raises(RuntimeError, match="UUID unavailable"):
        _validate_public_hardware(facts)


def test_hardware_probe_error_is_collectively_reachable():
    import torch

    class LocalAccelerator:
        num_processes = 1
        device = torch.device("cpu")

        def gather(self, tensor):
            return tensor

    def broken(_accelerator):
        raise RuntimeError("UUID API unavailable")

    with pytest.raises(RuntimeError, match="hardware probe phase failed.*UUID API unavailable"):
        _collect_runtime_hardware(LocalAccelerator(), _collector=broken)


def test_runtime_contract_records_actual_public_and_private_test_modes():
    rng = {
        "canonical_aug_seed": "20260914+seed (molecule ranking)",
        "private_num_workers_0": "aug_seed+rank",
        "train_worker_generator": "aug_seed+1000000+rank",
        "valid_worker_generator": "aug_seed+2000000+rank",
        "distributed_sampler": "seed",
        "nonzero_worker_dataset_rng": "dataset implementation unchanged",
    }
    assert _runtime_contract(False) == {
        "execution_mode": "public_cli", "compile_backend": "inductor", "precision": "bf16",
        "checkpointing": "nonreentrant", "construction_order": "weights_optimizer_compile_prepare",
        "rng_streams": rng,
    }
    assert _runtime_contract(True) == {
        "execution_mode": "private_test_injection", "compile_backend": "eager_counting_test",
        "precision": "no", "checkpointing": "nonreentrant",
        "construction_order": "weights_optimizer_compile_prepare", "rng_streams": rng,
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


def run_two_rank_probe(tmp_path, case):
    env = dict(os.environ)
    env["MVP_CLI_TEST_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = ".testdeps:/tmp/b6-dataset-verify.D2Is9a/.testdeps:src:tests"
    command = ["/home/wyim/miniconda3/envs/pdb/bin/torchrun", "--standalone", "--nproc-per-node=2",
               "tests/mvp_cli_probe.py", case]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads([line for line in result.stdout.splitlines() if line.startswith("{")][-1])


def test_actual_two_rank_execute_train_to_atomic_test_scope_final(tmp_path):
    output = run_two_rank_probe(tmp_path, "success")
    assert output == {
        "all_succeeded": True,
        "augmentation_streams": [
            {"canonical_aug_seed": 20260915, "private": 20260915, "rank": 0,
             "sampler": 1, "train_worker": 21260915, "valid_worker": 22260915},
            {"canonical_aug_seed": 20260915, "private": 20260916, "rank": 1,
             "sampler": 1, "train_worker": 21260916, "valid_worker": 22260916},
        ],
        "backend_called": True, "canonical_audit": True, "cpu_rank_facts_recorded": True,
        "effective_batch_size": 4, "epochs_crossed": 3, "identity_scope": "test",
        "marker": True, "pairwise_distinct_generators": True,
        "pairwise_distinct_stream_seeds_per_rank": True,
        "runtime_hardware": {"measurement": "runtime_observation", "test_hardware": True,
            "world_size": 2, "distributed_type": "MULTI_CPU", "device_type": "cpu",
            "distinct_device_uuids": False},
        "runtime_contract": {"checkpointing": "nonreentrant",
            "compile_backend": "eager_counting_test", "construction_order": "weights_optimizer_compile_prepare",
            "execution_mode": "private_test_injection", "precision": "no",
            "rng_streams": {
                "canonical_aug_seed": "20260914+seed (molecule ranking)",
                "private_num_workers_0": "aug_seed+rank",
                "train_worker_generator": "aug_seed+1000000+rank",
                "valid_worker_generator": "aug_seed+2000000+rank",
                "distributed_sampler": "seed",
                "nonzero_worker_dataset_rng": "dataset implementation unchanged",
            }},
        "saved_keys_clean": True, "successful_updates": 2, "update_budget": 2, "world_size": 2,
    }


def test_actual_two_rank_single_rank_hardware_probe_failure_rejects_before_model(tmp_path):
    output = run_two_rank_probe(tmp_path, "hardware-failure")
    assert output["all_rejected"] and output["no_artifact"] and output["dataset_closed"]
    assert not output["backend_called"] and not output["model_construction_started"]
    assert all("hardware probe phase failed" in error and "UUID probe failure" in error
               for error in output["errors"])


def test_actual_two_rank_single_rank_setup_failure_rejects_all_and_cleans_up(tmp_path):
    output = run_two_rank_probe(tmp_path, "failure")
    assert output["all_rejected"] and output["no_artifact"] and output["dataset_closed"]
    assert output["world_size"] == 2
    assert all("dataset phase failed" in error for error in output["errors"])


def test_actual_two_rank_identity_local_failure_rejects_all_without_nested_collective(tmp_path):
    output = run_two_rank_probe(tmp_path, "identity-failure")
    assert output["all_rejected"] and output["no_artifact"] and output["dataset_closed"]
    assert all("identity phase failed" in error for error in output["errors"])


def test_actual_two_rank_evaluation_time_mutation_rejects_without_artifact(tmp_path):
    output = run_two_rank_probe(tmp_path, "mutation")
    assert output["all_rejected"] and output["no_artifact"] and output["dataset_closed"]
    assert output["world_size"] == 2
    assert all("identity changed" in error for error in output["errors"])


def test_actual_two_rank_mutation_hook_exception_rejects_all_without_artifact(tmp_path):
    output = run_two_rank_probe(tmp_path, "mutation-error")
    assert output["all_rejected"] and output["no_artifact"] and output["dataset_closed"]
    assert all("injected mutation hook failure" in error for error in output["errors"])


def test_launch_readiness_binds_recipe_sources_and_evidence(tmp_path):
    candidate, db, evidence, panel = tiny_candidate(tmp_path)
    sif, config = tmp_path / "image.sif", tmp_path / "config.json"
    init = tmp_path / "init"
    init.mkdir()
    weights, run_config = init / "model.safetensors", init / "run_config.json"
    manifest = tmp_path / "code-manifest.json"
    for path, payload in ((sif, b"sif"), (config, b"{}\n"), (weights, b"weights"),
                          (run_config, b'{"seed":123}\n')):
        path.write_bytes(payload)
    from importlib.metadata import version
    code_files = _expected_code_files()
    canonical(manifest, {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        "external_packages": {name: version(name) for name in
                              ("accelerate", "duckdb", "safetensors", "torch", "transformers")},
    })
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
    value = json.loads(manifest.read_text())
    value["files"].pop(next(iter(value["files"])))
    canonical(manifest, value)
    tampered = readiness_record(paths, candidate)
    canonical(readiness, tampered)
    with pytest.raises(ValueError, match="exact executed source set"):
        validate_launch_readiness(readiness, hashlib.sha256(readiness.read_bytes()).hexdigest(),
                                  candidate, paths[:-1], seed=1, recipe=MVPRecipe())
