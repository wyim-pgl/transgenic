"""Two-rank production execute/_train integration using only private CPU test injection."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import torch
from safetensors import safe_open
from safetensors.torch import save_model

from test_mvp_dataset import DatasetTests, Tokenizer
from test_mvp_model import TinyHyena
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model import modeling_HyenaTransgenic as modeling
from transgenic.training.mvp_cli import CandidateRecords
from transgenic.training import mvp_cli_runtime as cli_runtime
from transgenic.training.mvp_cli_runtime import (
    _TestExecution, _expected_code_files, execute,
)
from transgenic.training.mvp_runtime import MVPRecipe

BACKEND_CALLS = 0


def canonical(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counting_backend(graph, inputs):
    global BACKEND_CALLS
    BACKEND_CALLS += 1
    return graph.forward


def source_recipe():
    return {
        "name": "b5-400m-test", "d_model_encoder": 8, "d_model_decoder": 16,
        "encoder_layers": 1, "decoder_layers": 1, "encoder_n_layer": 1,
        "encoder_attention_heads": 2, "decoder_attention_heads": 2,
        "encoder_ffn_dim": 32, "decoder_ffn_dim": 32,
        "attention_window": [4], "dropout": 0.0, "lr": 1e-4,
        "weight_decay": 0.02, "effective_batch_size": 4, "max_epochs": 5,
        "seeds": [123], "encoder_model": "tiny/local", "max_encoder_seqlen": 24,
        "max_decoder_position_embeddings": 32, "vocab_size": 288, "patience": 3,
        "optimizer": "AdamW", "lr_warmup_fraction": 0.05,
        "window_policy": "sym6144-v1", "gff_vocab_version": "v2",
        "max_label_tokens": 2048,
    }


def config(channels):
    return HyenaTransgenicConfig(**{
        "vocab_size": 288, "d_model": 8, "encoder_d_model": 8, "decoder_d_model": 16,
        "encoder_layers": 1, "decoder_layers": 1, "encoder_n_layer": 1,
        "encoder_attention_heads": 2, "decoder_attention_heads": 2,
        "encoder_ffn_dim": 32, "decoder_ffn_dim": 32, "max_encoder_seqlen": 24,
        "max_decoder_position_embeddings": 32, "attention_window": [4], "dropout": 0.0,
        "encoder_model": "tiny/local", "load_pretrained_encoder": False,
        "evidence_channels": channels,
    })


def build_fixture(root):
    fixture = DatasetTests()
    fixture.setUp()
    db, evidence = root / "corpus.db", root / "evidence.db"
    shutil.copy2(fixture.db, db)
    shutil.copy2(fixture.ev, evidence)
    fixture.doCleanups()
    with duckdb.connect(str(db)) as con:
        con.execute("UPDATE geneList SET sequence='ACGTACGTACGTACGTACGTACGT', fin=124 WHERE species_id='Athaliana'")
    with duckdb.connect(str(evidence)) as con:
        con.execute("UPDATE corpus_tile SET end0=124")
        con.execute("INSERT INTO tile_est VALUES (13,'m')")
    panel, sif = root / "panel.tsv", root / "image.sif"
    panel.write_text("locus\tstratum\nAT1G01010\tmvp\n")
    sif.write_bytes(b"sif")
    recipe_data = source_recipe()
    config_path = root / "config.json"
    canonical(config_path, recipe_data)
    run, epoch = root / "source", root / "source" / "epoch_01"
    epoch.mkdir(parents=True)
    torch.manual_seed(7)
    save_model(modeling.transgenicForConditionalGeneration(config(0)), epoch / "model.safetensors")
    canonical(run / "run_config.json", {"seed": 123, "recipe": recipe_data})
    manifest = root / "code-manifest.json"
    canonical(manifest, {
        "files": {str(path): digest(path) for path in _expected_code_files()},
        "external_packages": {name: version(name) for name in
                              ("accelerate", "duckdb", "safetensors", "torch", "transformers")},
    })
    for alias, target in ((root / "db-alias", db), (root / "config-alias", config_path),
                          (root / "init-alias", epoch)):
        alias.symlink_to(target)
    records = [
        {"rn": 7, "chromosome": "Chr1", "start0": 100, "end0": 124, "split": "train", "is_rc": False},
        {"rn": 9, "chromosome": "Chr1", "start0": 100, "end0": 124, "split": "train", "is_rc": True},
        {"rn": 13, "chromosome": "Chr1", "start0": 100, "end0": 124, "split": "valid", "is_rc": False},
    ]
    identities = {name: digest(path) for name, path in
                  (("corpus", db), ("evidence", evidence), ("panel", panel))}
    candidate = {
        "acceptance": {"artifact": {"sha256": identities["evidence"]},
                       "inputs": {"corpus": {"sha256": identities["corpus"]},
                                  "panel": {"sha256": identities["panel"]}}},
        "inventory": {"records": records, "counts": {"total": 3, "train": 2, "valid": 1,
            "evidence_corpus_tile_all_splits": {"train": 2, "valid": 1, "test": 0}}},
        "identities": {"acceptance": "a" * 64, "inventory": "b" * 64, "receipts": "c" * 64},
    }
    canonical(root / "candidate.json", candidate)
    recipe = MVPRecipe(max_updates=2, warmup_updates=1, accumulation_steps=2,
                       required_world_size=2, backbone_lr=3e-4, projection_lr=5e-4)
    evidence_rows = {name: {"pass": True, "receipt_sha256": "d" * 64} for name in (
        "real_tokenizer", "real_hyena_step0_three_tiers", "generation_cache_parity",
        "bf16_compile", "four_rank_smoke", "resource_budget")}
    readiness = {
        "schema_version": 1, "authorized_scope": "exploratory_mvp",
        "production_authorized": False, "ready_for_finetuning": True,
        "bindings": {
            "technical_acceptance_sha256": "a" * 64, "inventory_sha256": "b" * 64,
            "receipts_sha256": "c" * 64, "corpus_sha256": identities["corpus"],
            "evidence_sha256": identities["evidence"], "panel_sha256": identities["panel"],
            "sif_sha256": digest(sif), "config_sha256": digest(config_path),
            "source_checkpoint": {"weights_path": str((epoch / "model.safetensors").resolve()),
                "weights_sha256": digest(epoch / "model.safetensors"),
                "run_config_path": str((run / "run_config.json").resolve()),
                "run_config_sha256": digest(run / "run_config.json"), "seed": 123},
            "consumed_code_manifest": {"path": str(manifest.resolve()), "sha256": digest(manifest)},
            "recipe": recipe.__dict__, "seed": 1,
        },
        "evidence": evidence_rows,
    }
    canonical(root / "readiness.json", readiness)
    (root / "READY").write_text("ready")


def inputs(root):
    candidate_data = json.loads((root / "candidate.json").read_text())
    candidate = CandidateRecords(candidate_data["acceptance"], candidate_data["inventory"], {},
                                 candidate_data["identities"])
    readiness = root / "readiness.json"
    args = SimpleNamespace(
        db=str(root / "corpus.db"), evidence_db=str(root / "evidence.db"), panel=str(root / "panel.tsv"),
        sif=str(root / "image.sif"), config=str(root / "config.json"), init_from=str(root / "init-alias"),
        output_dir=str(root / "final"), launch_readiness=str(readiness),
        launch_readiness_sha256=digest(readiness), seed=1, num_workers=0,
    )
    return args, candidate


def main():
    torch.set_num_threads(1)
    root = Path(os.environ["MVP_CLI_TEST_DIR"])
    case = sys.argv[1] if len(sys.argv) > 1 else "success"
    rank = int(os.environ["LOCAL_RANK"])
    factory_config = patch.object(modeling.AutoConfig, "from_pretrained",
                                  side_effect=lambda *a, **k: SimpleNamespace())
    factory_model = patch.object(modeling.AutoModel, "from_config",
                                 side_effect=lambda cfg, **k: TinyHyena(cfg))
    tokenizer = patch("transgenic.datasets.datasets.AutoTokenizer.from_pretrained",
                      return_value=Tokenizer())
    with factory_config, factory_model, tokenizer:
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            build_fixture(root)
        deadline = time.monotonic() + 20
        while not (root / "READY").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("fixture setup timeout")
            time.sleep(0.02)
        args, candidate = inputs(root)
        recipe = MVPRecipe(max_updates=2, warmup_updates=1, accumulation_steps=2,
                           required_world_size=2, backbone_lr=3e-4, projection_lr=5e-4)
        def mutation_hook():
            if case == "mutation-error":
                raise RuntimeError("injected mutation hook failure")
            (root / "panel.tsv").write_text("mutated\n")

        observation = {}
        fail_phase = "dataset" if case == "failure" else ("identity" if case == "identity-failure" else None)
        injection = _TestExecution(
            recipe, counting_backend, fail_phase=fail_phase,
            mutate_before_final_recheck=(mutation_hook if case in {"mutation", "mutation-error"} else None),
            observation=observation,
        )
        error = None
        original_collector = cli_runtime._collect_local_hardware
        def hardware_collector(accelerator):
            if case == "hardware-failure" and accelerator.process_index == 1:
                raise RuntimeError("injected UUID probe failure")
            return original_collector(accelerator)
        collector_patch = (patch.object(cli_runtime, "_collect_local_hardware", hardware_collector)
                           if case == "hardware-failure" else nullcontext())
        try:
            with collector_patch:
                execute(args, candidate, validate_only=False, _test=injection)
        except Exception as caught:
            error = str(caught)
        result_path = root / f"result-{rank}.json"
        result_tmp = root / f".result-{rank}.tmp"
        canonical(result_tmp, {"error": error, "backend_called": BACKEND_CALLS > 0,
                               "observation": observation})
        result_tmp.replace(result_path)
        if rank == 0:
            deadline = time.monotonic() + 20
            while not all((root / f"result-{item}.json").exists() for item in range(2)):
                if time.monotonic() > deadline:
                    raise RuntimeError("result coordination timeout")
                time.sleep(0.02)
            results = [json.loads((root / f"result-{item}.json").read_text()) for item in range(2)]
            errors = [item["error"] for item in results]
            writable = True
            try:
                with duckdb.connect(str(root / "corpus.db")):
                    pass
            except Exception:
                writable = False
            if case == "success":
                metadata = json.loads((root / "final" / "metadata.json").read_text())
                clean = all(not key.startswith(("module.", "_orig_mod."))
                            for key in safe_open(root / "final" / "model.safetensors", framework="pt").keys())
                training = metadata["training_summary"]
                streams = sorted(
                    (item["observation"]["augmentation_streams"] for item in results),
                    key=lambda item: item["rank"],
                )
                generator_ids = [item["observation"]["generator_ids"] for item in results]
                pairwise_seeds = all(
                    len({item["private"], item["train_worker"], item["valid_worker"]}) == 3
                    for item in streams
                )
                pairwise_generators = all(len(set(item.values())) == 3 for item in generator_ids)
                hardware = metadata["runtime_hardware"]
                payload = {"all_succeeded": errors == [None, None],
                           "augmentation_streams": streams,
                           "cpu_rank_facts_recorded": all(
                               fact["rank"] == fact["local_rank"] and
                               fact["distributed_type"] == "MULTI_CPU" and
                               fact["device_type"] == "cpu" and
                               fact["current_cuda_device"] is None and
                               fact["device_uuid"] is None
                               for fact in hardware["ranks"]),
                           "backend_called": all(item["backend_called"] for item in results),
                           "canonical_audit": metadata["initialization_audit"]["audit"]["resolved_weights_path"] ==
                                              str((root / "source/epoch_01/model.safetensors").resolve()),
                           "effective_batch_size": training["effective_batch_size"],
                           "epochs_crossed": training["epochs_crossed"],
                           "identity_scope": metadata["identity_scope"],
                           "marker": (root / "final/TRAINING_DONE").is_file(),
                           "pairwise_distinct_generators": pairwise_generators,
                           "pairwise_distinct_stream_seeds_per_rank": pairwise_seeds,
                           "runtime_hardware": {
                               "measurement": hardware["measurement"],
                               "test_hardware": hardware["test_hardware"],
                               "world_size": hardware["world_size"],
                               "distributed_type": hardware["distributed_type"],
                               "device_type": hardware["device_type"],
                               "distinct_device_uuids": hardware["distinct_device_uuids"],
                           },
                           "runtime_contract": metadata["runtime_contract"],
                           "saved_keys_clean": clean,
                           "successful_updates": training["successful_updates"],
                           "update_budget": metadata["normalized_recipe"]["max_updates"],
                           "world_size": 2}
            else:
                payload = {"all_rejected": all(errors), "errors": errors,
                           "backend_called": any(item["backend_called"] for item in results),
                           "model_construction_started": any(
                               item["observation"].get("model_construction_started", False)
                               for item in results),
                           "no_artifact": not (root / "final").exists(), "dataset_closed": writable,
                           "world_size": 2}
            print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
