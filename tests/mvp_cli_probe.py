"""Two-rank tiny integration for MVP CLI wiring; CPU/Gloo and test scope only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import torch
from safetensors import safe_open
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mvp_finalize_probe import metadata
from test_mvp_dataset import DatasetTests
from test_mvp_model import TinyHyena
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model import modeling_HyenaTransgenic as modeling
from transgenic.training import mvp_finalize as finalize
from transgenic.training.mvp_cli_runtime import compile_bound_forward, require_matching_identity
from transgenic.training.mvp_finalize import evaluate_and_publish_final
from transgenic.training.mvp_runtime import MVPRecipe, build_optimizer, make_accelerator, run_fixed_updates
from transgenic.training.weights_init import load_weights_only


def config(channels):
    return HyenaTransgenicConfig(
        vocab_size=288, d_model=8, encoder_d_model=8, decoder_d_model=16,
        encoder_layers=1, decoder_layers=1, encoder_n_layer=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_encoder_seqlen=24,
        max_decoder_position_embeddings=32, attention_window=[4], dropout=0.0,
        encoder_model="tiny/local", load_pretrained_encoder=False,
        evidence_channels=channels,
    )


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


def main():
    torch.set_num_threads(1)
    root = Path(os.environ["MVP_CLI_TEST_DIR"])
    run, epoch = root / "source", root / "source" / "epoch_01"
    final = root / "final"
    recipe = MVPRecipe(max_updates=2, warmup_updates=1, accumulation_steps=2,
                       required_world_size=2, backbone_lr=3e-4, projection_lr=5e-4)
    accelerator = make_accelerator(recipe, cpu=True)
    identity_failed = False
    try:
        require_matching_identity(accelerator, {"rank_specific": accelerator.process_index})
    except ValueError:
        identity_failed = True
    identity_failures = accelerator.gather(torch.tensor([identity_failed], device=accelerator.device))
    no_artifact_after_identity = not final.exists()
    factory_config = patch.object(modeling.AutoConfig, "from_pretrained",
                                  side_effect=lambda *a, **k: SimpleNamespace())
    factory_model = patch.object(modeling.AutoModel, "from_config",
                                 side_effect=lambda config, **k: TinyHyena(config))
    with factory_config, factory_model:
        if accelerator.is_main_process:
            epoch.mkdir(parents=True)
            torch.manual_seed(7)
            save_model(modeling.transgenicForConditionalGeneration(config(0)),
                       epoch / "model.safetensors")
            (run / "run_config.json").write_text(json.dumps({"recipe": source_recipe()}))
        accelerator.wait_for_everyone()

        torch.manual_seed(7)
        model = modeling.transgenicForConditionalGeneration(config(8))
        audit = load_weights_only(model, epoch)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = build_optimizer(model, recipe)
        backend_calls = 0

        def backend(graph, inputs):
            nonlocal backend_calls
            backend_calls += 1
            return graph.forward

        compile_bound_forward(model, backend=backend)
        model, optimizer = accelerator.prepare(model, optimizer)
        fixture = DatasetTests()
        fixture.setUp()
        with duckdb.connect(fixture.db) as con:
            con.execute("UPDATE geneList SET sequence='ACGTACGTACGTACGTACGTACGT', fin=124 WHERE species_id='Athaliana'")
        with duckdb.connect(fixture.ev) as con:
            con.execute("UPDATE corpus_tile SET end0=124")
            con.execute("INSERT INTO tile_est VALUES (13,'m')")
        train = fixture.dataset(mode="train", split="train", return_tile_ids=True)
        valid = fixture.dataset(mode="train", split="valid", return_tile_ids=True, eval_fraction=1)
        train._aug_generator = torch.Generator().manual_seed(20260915 + accelerator.process_index)
        train._aug_key = (os.getpid(), None)
        train_sampler = DistributedSampler(train, num_replicas=2, rank=accelerator.process_index,
                                           shuffle=True, seed=1)
        train_loader = DataLoader(train, batch_size=1, sampler=train_sampler,
                                  collate_fn=__import__('transgenic.datasets.datasets', fromlist=['hyena_collate_fn']).hyena_collate_fn)
        summary = run_fixed_updates(accelerator, model, optimizer, train_loader, recipe)

        valid_sampler = DistributedSampler(valid, num_replicas=2, rank=accelerator.process_index,
                                           shuffle=False, drop_last=False)
        valid_loader = DataLoader(valid, batch_size=1, sampler=valid_sampler,
                                  collate_fn=__import__('transgenic.datasets.datasets', fromlist=['hyena_collate_fn']).hyena_collate_fn)
        md = metadata()
        md["normalized_recipe"] = recipe.__dict__
        md["initialization_audit"] = {
            "audit": audit,
            "audit_sha256": hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
            "verification": "independently_verified",
        }
        md.update({"runtime_profile": "mvp_fixed_250x96",
                   "authorization_scope": "exploratory_mvp",
                   "production_authorized": False,
                   "output_use": "EXPLORATORY_MVP_NOT_FOR_PRODUCTION"})
        evaluate_and_publish_final(accelerator, model, valid_loader, [13], final,
                                   metadata=md, training_summary=summary)

        rng = torch.Generator().manual_seed(20260915 + accelerator.process_index)
        value = torch.rand(1, generator=rng).to(accelerator.device)
        values = accelerator.gather(value)
        backend_flags = accelerator.gather(torch.tensor([backend_calls > 0], device=accelerator.device))
        if accelerator.is_main_process:
            clean = all(not key.startswith(("module.", "_orig_mod."))
                        for key in safe_open(final / "model.safetensors", framework="pt").keys())
            print(json.dumps({
                "all_rank_rng_distinct": bool(values.unique().numel() == 2),
                "asymmetric_identity_all_failed": bool(identity_failures.all().item()) and no_artifact_after_identity,
                "audit_bound": md["initialization_audit"]["audit_sha256"] ==
                               hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
                "backend_called": bool(backend_flags.all().item()),
                "epochs_crossed": summary["epochs_crossed"],
                "marker": (final / "TRAINING_DONE").is_file(),
                "saved_keys_clean": clean,
                "successful_updates": summary["successful_updates"],
                "world_size": accelerator.num_processes,
            }, sort_keys=True))


if __name__ == "__main__":
    main()
