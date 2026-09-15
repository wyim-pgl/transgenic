"""CPU tests for audited B5 (0-channel) to MVP (8-channel) weights initialization."""
import hashlib
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, save_model
from torch import nn

from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model import modeling_HyenaTransgenic as modeling


class TinyHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(16, config.d_model)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids), hidden_states=None)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class WeightsInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.config_patch = patch.object(modeling.AutoConfig, "from_pretrained", side_effect=lambda *a, **k: SimpleNamespace())
        self.model_patch = patch.object(modeling.AutoModel, "from_config", side_effect=lambda config, **k: TinyHyena(config))
        self.config_patch.start()
        self.model_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.model_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "run"
        self.epoch = self.run / "epoch_04"
        self.epoch.mkdir(parents=True)
        self.recipe = {
            "name": "b5-400m-test", "d_model_encoder": 8, "d_model_decoder": 16,
            "encoder_layers": 1, "decoder_layers": 1, "encoder_n_layer": 1,
            "encoder_attention_heads": 2, "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32, "decoder_ffn_dim": 32,
            "attention_window": [4], "dropout": 0.0,
            "lr": 1e-4, "weight_decay": 0.02, "effective_batch_size": 4,
            "max_epochs": 5, "seeds": [123], "encoder_model": "tiny/local",
            "max_encoder_seqlen": 24, "max_decoder_position_embeddings": 32,
            "vocab_size": 288, "patience": 3, "optimizer": "AdamW",
            "lr_warmup_fraction": 0.05, "window_policy": "sym6144-v1",
            "gff_vocab_version": "v2", "max_label_tokens": 2048,
        }
        self.run_config = {"config": "/stale/must-not-be-opened.json", "recipe": self.recipe,
                           "db": "/irrelevant/test.duckdb", "seed": 123, "batch_size": 1,
                           "accumulation_steps": 4, "max_epochs": 5, "patience": 3}
        self._write_json(self.run / "run_config.json", self.run_config)
        self.source = self._model(0)
        save_model(self.source, self.epoch / "model.safetensors")

    def _config(self, channels):
        return HyenaTransgenicConfig(
            vocab_size=288, d_model=8, encoder_d_model=8, decoder_d_model=16,
            encoder_layers=1, decoder_layers=1, encoder_n_layer=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32,
            max_encoder_seqlen=24, max_decoder_position_embeddings=32,
            attention_window=[4], dropout=0.0, encoder_model="tiny/local",
            load_pretrained_encoder=False, evidence_channels=channels,
        )

    def _model(self, channels=8):
        return modeling.transgenicForConditionalGeneration(self._config(channels))

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _file_tensors(path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key) for key in handle.keys()}

    def _load(self, model=None, path=None):
        from transgenic.training.weights_init import load_weights_only
        return load_weights_only(model or self._model(), path or self.epoch)

    def test_success_from_best_is_alias_safe_zeroes_projection_and_audits_hashes(self):
        os.symlink("epoch_04", self.run / "best")
        target = self._model()
        target.train()
        optimizer = torch.optim.AdamW(target.parameters())
        before_torch = torch.get_rng_state().clone()
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        with patch.object(torch, "load", side_effect=AssertionError("torch.load must not run")):
            audit = self._load(target, self.run / "best")

        self.assertTrue(target.training)
        self.assertFalse(target._weights_init_poisoned)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        for key, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        self.assertEqual(torch.count_nonzero(target.transgenic.encoder.evidence_proj.weight).item(), 0)
        self.assertIs(target.lm_head.weight, target.transgenic.decoder_embed_tokens.weight)
        self.assertIs(target.transgenic.decoder_embed_tokens.weight, target.transgenic.decoder.embed_tokens.weight)
        self.assertEqual(audit["requested_path"], str(self.run / "best"))
        self.assertEqual(audit["resolved_checkpoint_dir"], str(self.epoch.resolve()))
        self.assertEqual(audit["load"]["missing_keys"], ["transgenic.encoder.evidence_proj.weight"])
        self.assertEqual(audit["load"]["unexpected_keys"], [])
        self.assertEqual(audit["compatibility"]["transition"], "evidence_channels:0->8")
        self.assertEqual(audit["recipe"]["canonical_sha256"], hashlib.sha256(canonical(self.recipe)).hexdigest())
        for name in ("model.safetensors", "run_config.json"):
            record = audit["files"][name]
            self.assertEqual(record["before"]["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
            self.assertEqual(record["before"], record["after"])
        for name in ("config.json", "meta.json", "runtime.json"):
            self.assertIsNone(audit["files"][name])
        for side in ("source_semantics", "target_semantics"):
            manifest = audit[side]["fields"]
            self.assertEqual(audit[side]["sha256"], hashlib.sha256(canonical(manifest)).hexdigest())
        self.assertEqual(audit["source_semantics"]["fields"]["decoder_layerdrop"]["provenance"], "config-default")
        self.assertIn("current loader", audit["provenance_caveat"])
        json.dumps(audit)

    def test_exact_file_and_matching_optional_metadata_succeed(self):
        source_config = self._config(0).to_dict()
        source_config.update({"_name_or_path": "informational", "transformers_version": "ignored"})
        self._write_json(self.epoch / "config.json", source_config)
        self._write_json(self.epoch / "meta.json", {"epoch": 4})
        self._write_json(self.epoch / "runtime.json", {"host": "test"})
        audit = self._load(path=self.epoch / "model.safetensors")
        for name in ("config.json", "meta.json", "runtime.json"):
            self.assertIsNotNone(audit["files"][name])

    def test_stale_config_field_and_state_subtrees_are_never_read(self):
        (self.epoch / "accelerate_state").mkdir()
        (self.epoch / "accelerate_state" / "optimizer.bin").write_bytes(b"not torch")
        (self.epoch / "pending_rank0.pt").write_bytes(b"not torch")
        with patch("builtins.open", wraps=open) as opened, patch.object(torch, "load", side_effect=AssertionError):
            self._load()
        opened_paths = {os.fspath(call.args[0]) for call in opened.call_args_list if call.args}
        self.assertNotIn(self.run_config["config"], opened_paths)
        self.assertFalse(any("accelerate_state" in path or "pending_rank" in path for path in opened_paths))

    def test_semantic_recipe_or_optional_config_drift_fails_before_mutation(self):
        for location, key, value in (("recipe", "activation_function", "relu"),
                                     ("config", "use_cache", False)):
            with self.subTest(location=location):
                target = self._model()
                original = {k: v.clone() for k, v in target.state_dict().items()}
                if location == "recipe":
                    run_config = dict(self.run_config)
                    run_config["recipe"] = dict(self.recipe, **{key: value})
                    self._write_json(self.run / "run_config.json", run_config)
                else:
                    self._write_json(self.epoch / "config.json", {key: value})
                with self.assertRaisesRegex(ValueError, key):
                    self._load(target)
                self.assertTrue(target._weights_init_poisoned)
                for name, tensor in original.items():
                    self.assertTrue(torch.equal(tensor, target.state_dict()[name]), name)
                self._write_json(self.run / "run_config.json", self.run_config)
                (self.epoch / "config.json").unlink(missing_ok=True)

    def test_malformed_embedded_recipe_and_same8_source_fail(self):
        cases = [(dict(self.recipe), "encoder_n_layer"), (dict(self.recipe, evidence_channels=8), "0->8")]
        del cases[0][0]["encoder_n_layer"]
        for recipe, message in cases:
            with self.subTest(message=message):
                self._write_json(self.run / "run_config.json", dict(self.run_config, recipe=recipe))
                with self.assertRaisesRegex(ValueError, message):
                    self._load()
        self._write_json(self.run / "run_config.json", {"config": "ignored"})
        with self.assertRaisesRegex(ValueError, "recipe"):
            self._load()

    def test_rejects_grads_wrong_target_channels_and_marks_poisoned(self):
        target = self._model()
        next(target.parameters()).grad = torch.ones_like(next(target.parameters()))
        with self.assertRaisesRegex(ValueError, "grad"):
            self._load(target)
        self.assertTrue(target._weights_init_poisoned)
        next(target.parameters()).grad = None
        with self.assertRaisesRegex(ValueError, "poisoned"):
            self._load(target)
        with self.assertRaisesRegex(ValueError, "target.*8"):
            self._load(self._model(0))

    def test_missing_ordinary_key_and_unexpected_key_fail_and_poison(self):
        original = self._file_tensors(self.epoch / "model.safetensors")
        ordinary = next(key for key in original if key != "transgenic.encoder.evidence_proj.weight")
        variants = ({k: v for k, v in original.items() if k != ordinary}, dict(original, unexpected=torch.zeros(1)))
        for tensors in variants:
            with self.subTest(keys=len(tensors)):
                save_file(tensors, self.epoch / "model.safetensors")
                target = self._model()
                with self.assertRaisesRegex(ValueError, "missing|unexpected"):
                    self._load(target)
                self.assertTrue(target._weights_init_poisoned)
        save_model(self.source, self.epoch / "model.safetensors")

    def test_target_must_have_actual_tied_storage(self):
        target = self._model()
        target.lm_head.weight = nn.Parameter(target.lm_head.weight.detach().clone())
        with self.assertRaisesRegex(ValueError, "tied storage"):
            self._load(target)
        self.assertTrue(target._weights_init_poisoned)

    def test_source_projection_and_wrong_shape_fail_preflight(self):
        tensors = self._file_tensors(self.epoch / "model.safetensors")
        variants = [dict(tensors, **{"transgenic.encoder.evidence_proj.weight": torch.zeros(8, 8)}), dict(tensors)]
        shape_key = next(iter(variants[1]))
        variants[1][shape_key] = variants[1][shape_key].reshape(-1)
        for tensors, message in zip(variants, ("evidence projection", "shape")):
            with self.subTest(message=message):
                save_file(tensors, self.epoch / "model.safetensors")
                with self.assertRaisesRegex(ValueError, message):
                    self._load()

    def test_corrupt_file_dangling_link_wrong_filename_and_escaping_symlink_fail(self):
        good = (self.epoch / "model.safetensors").read_bytes()
        (self.epoch / "model.safetensors").write_bytes(b"corrupt")
        with self.assertRaises(Exception):
            self._load()
        (self.epoch / "model.safetensors").write_bytes(good)
        os.symlink("missing", self.run / "best")
        with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
            self._load(path=self.run / "best")
        wrong = self.epoch / "weights.safetensors"
        wrong.write_bytes(good)
        with self.assertRaisesRegex(ValueError, "model.safetensors"):
            self._load(path=wrong)
        external = Path(self.tmp.name) / "outside" / "model.safetensors"
        external.parent.mkdir()
        external.write_bytes(good)
        (self.epoch / "model.safetensors").unlink()
        os.symlink(external, self.epoch / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "checkpoint root"):
            self._load()
        with self.assertRaisesRegex(ValueError, "checkpoint root"):
            self._load(path=self.epoch / "model.safetensors")

    def test_source_identity_change_is_detected_after_consumption(self):
        import transgenic.training.weights_init as weights_init
        real_snapshot = weights_init._snapshot
        calls = 0

        def changed(path):
            nonlocal calls
            result = real_snapshot(path)
            calls += 1
            if calls > 2 and path.name == "model.safetensors":
                result = dict(result, sha256="0" * 64)
            return result

        target = self._model()
        with patch.object(weights_init, "_snapshot", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                weights_init.load_weights_only(target, self.epoch)
        self.assertTrue(target._weights_init_poisoned)


if __name__ == "__main__":
    unittest.main()
