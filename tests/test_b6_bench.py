"""CP2 bench and CP4 driver: the parts that can run without a GPU (argument defaults, batch contract)."""
import importlib.util
import os
import sys
import unittest

import torch

from transgenic.training import b6_runtime as rt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("b6_bench_memory", "revision/scripts/b6/bench_memory_129k.py")


class BenchTests(unittest.TestCase):
    def test_defaults_are_the_gate_conditions(self):
        a = bench.parse_args(["--out", "x.json"])
        self.assertEqual((a.tier, a.chain_elements, a.gate_gb, a.config),
                         (129024, 511, 100.0, "configs/b6_400m_win_v3.json"))

    def test_synthetic_batch_is_dense_and_on_contract(self):
        cfg = rt.load_b6_config(os.path.join(ROOT, "configs", "b6_400m_win_v3.json"))
        a = bench.parse_args(["--out", "x.json", "--tier", "3072", "--label-tokens", "64"])
        batch = bench.synthetic_batch(torch, cfg, a, torch.device("cpu"), 0)
        batch["meta"] = dict(row=["1"], epoch=[0], arm="B6", uniforms_sha256=["0"], resolved=[{}],
                             context_kind=["corpus"])
        rt.validate_b6_batch(batch)
        self.assertEqual(int(batch["chain_mask"].sum()), 511)
        self.assertEqual(torch.count_nonzero(batch["evidence_tracks"]).item(), 3072 * 31)
        self.assertEqual(batch["labels"].shape, (1, 64))
        other = bench.synthetic_batch(torch, cfg, a, torch.device("cpu"), 1)
        self.assertFalse(torch.equal(batch["input_ids"], other["input_ids"]))


class GateVerdictTests(unittest.TestCase):
    """Codex review 1 H3: the bench cannot certify PASS without running under the frozen conditions."""

    def conditions(self, **kw):
        c = dict(bench.GATE_CONDITIONS, world_size=4)
        c.update(kw)
        return c

    def ranks(self, n=4, **kw):
        base = dict(measured_updates=8, compile_verified=True, peak_reserved_bytes=90 * 10 ** 9, error=None)
        base.update(kw)
        return [dict(base, rank=r) for r in range(n)]

    def verdict(self, conditions=None, ranks=None, steps=12, warmup=4, gate_gb=100.0):
        return bench.gate_verdict(conditions or self.conditions(), self.ranks() if ranks is None else ranks,
                                  steps=steps, warmup=warmup, gate_gb=gate_gb)[0]

    def test_pass_only_under_frozen_conditions_with_measured_compiled_updates(self):
        self.assertEqual(self.verdict(), "PASS")
        for change in (dict(tier=30720), dict(chain_elements=100), dict(world_size=1), dict(gate_gb=120.0),
                       dict(label_tokens=64), dict(n_tok=256)):
            with self.subTest(change=change):
                self.assertEqual(self.verdict(conditions=self.conditions(**change)), "NOT_A_GATE_RUN")
        self.assertEqual(self.verdict(ranks=self.ranks(n=3)), "NOT_A_GATE_RUN")
        # A51: the gate is certified on 2 or 4 ranks; 1 rank (no DDP buffers) and 3 ranks are not gate runs
        self.assertEqual(self.verdict(conditions=self.conditions(world_size=2), ranks=self.ranks(n=2)), "PASS")
        self.assertEqual(self.verdict(conditions=self.conditions(world_size=1), ranks=self.ranks(n=1)), "NOT_A_GATE_RUN")
        self.assertEqual(self.verdict(conditions=self.conditions(world_size=3), ranks=self.ranks(n=3)), "NOT_A_GATE_RUN")
        self.assertEqual(self.verdict(conditions=self.conditions(world_size=2), ranks=self.ranks(n=4)), "NOT_A_GATE_RUN")
        self.assertEqual(self.verdict(ranks=self.ranks(measured_updates=0)), "FAIL")
        self.assertEqual(self.verdict(ranks=self.ranks(compile_verified=False)), "FAIL")
        self.assertEqual(self.verdict(ranks=self.ranks(error="RuntimeError: recompile")), "FAIL")
        self.assertEqual(self.verdict(ranks=self.ranks(peak_reserved_bytes=101 * 10 ** 9)), "FAIL")
        self.assertEqual(bench.EXIT_CODES, {"PASS": 0, "FAIL": 2, "NOT_A_GATE_RUN": 3})

    def test_arguments_without_measured_updates_are_refused(self):
        for argv in (["--steps", "0"], ["--steps", "4", "--warmup", "4"], ["--steps", "3", "--warmup", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as ctx:
                bench.parse_args(["--out", "x.json"] + argv)
            self.assertEqual(ctx.exception.code, bench.EXIT_REFUSED)


class ArchitectureGateTests(unittest.TestCase):
    """Codex review 2 H3: only the frozen 400M architecture, as built, can certify the memory gate."""

    def _built(self, **override):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.test_b6_model import TinyHyena
        from transgenic.model import modeling_HyenaTransgenic as b5
        from transgenic.model.modeling_b6 import B6TransgenicForConditionalGeneration, b6_config
        cfg = json.load(open(os.path.join(ROOT, "configs", "b6_400m_win_v3.json")))
        cfg.update(override)
        kw = rt.b6_model_kwargs(cfg)
        kw["load_pretrained_encoder"] = False
        with patch.object(b5.AutoConfig, "from_pretrained", side_effect=lambda *a, **k: SimpleNamespace()), \
                patch.object(b5.AutoModel, "from_config", side_effect=lambda c, **k: TinyHyena(c)):
            return B6TransgenicForConditionalGeneration(b6_config(**kw))

    def test_config_loader_refuses_a_smaller_model(self):
        import json
        import tempfile
        cfg = json.load(open(os.path.join(ROOT, "configs", "b6_400m_win_v3.json")))
        for key, value in (("decoder_layers", 1), ("d_model_decoder", 768), ("decoder_attention_heads", 2)):
            with self.subTest(key=key), tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(dict(cfg, **{key: value}), fh)
            with self.assertRaisesRegex(ValueError, key):
                rt.load_b6_config(fh.name)
            os.remove(fh.name)

    def test_built_frozen_model_matches_the_digest_and_a_one_layer_decoder_is_not_a_gate_run(self):
        frozen = rt.model_architecture(self._built())
        self.assertEqual(frozen, rt.FROZEN_ARCHITECTURE)
        self.assertEqual(rt.architecture_sha256(frozen), bench.GATE_CONDITIONS["architecture_sha256"])
        small = rt.model_architecture(self._built(decoder_layers=1))
        self.assertEqual(small["built_decoder_layers"], 1)
        ranks = [dict(rank=r, measured_updates=8, compile_verified=True, peak_reserved_bytes=10 ** 9, error=None)
                 for r in range(4)]
        conditions = dict(bench.GATE_CONDITIONS, architecture_sha256=rt.architecture_sha256(small))
        status, reason = bench.gate_verdict(conditions, ranks, steps=12, warmup=4, gate_gb=100.0)
        self.assertEqual(status, "NOT_A_GATE_RUN")
        self.assertIn("architecture_sha256", reason)
        self.assertEqual(bench.gate_verdict(dict(bench.GATE_CONDITIONS, world_size=4), ranks, steps=12, warmup=4,
                                            gate_gb=100.0)[0], "PASS")


if __name__ == "__main__":
    unittest.main()
