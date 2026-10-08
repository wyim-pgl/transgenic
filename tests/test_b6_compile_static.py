"""B6 compile policy: static DNA-length graphs, dynamic labels, bounded recompiles.

A recompile after a length change used to be compiled with L symbolic (Dynamo automatic_dynamic_shapes), and the
129,024 graph compiled that way OOM-ed on a GH200. compile_b6 pins every unmarked dim static, so only the labels dim
the trainer marks with maybe_mark_dynamic stays dynamic. The torch-free tests run in the duckdb-only environment;
the CPU compile test needs torch.
"""
import importlib.util
import sys
import types
import unittest
from unittest import mock

from transgenic.training import b6_runtime as rt

HAS_TORCH = importlib.util.find_spec("torch") is not None


def fake_torch(limit_name):
    """A torch stand-in exposing only what compile_b6 touches; limit_name is this torch version's recompile limit."""
    config = types.SimpleNamespace(suppress_errors=True, optimize_ddp=True, automatic_dynamic_shapes=True)
    setattr(config, limit_name, 8)
    setattr(config, "fail_on_recompile_limit_hit" if limit_name == "recompile_limit" else "fail_on_cache_limit_hit",
            False)
    torch = types.ModuleType("torch")
    torch._dynamo = types.SimpleNamespace(config=config)
    torch.compile = lambda model, backend: ("compiled", model, backend)
    return torch, config


class CompileB6PolicyTests(unittest.TestCase):
    def check(self, limit_name, fail_name):
        torch, config = fake_torch(limit_name)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            out = rt.compile_b6("model", backend="inductor")
        self.assertEqual(out, ("compiled", "model", "inductor"))
        self.assertIs(config.automatic_dynamic_shapes, False)
        self.assertIs(config.suppress_errors, False)
        self.assertIs(config.optimize_ddp, False)
        self.assertEqual(getattr(config, limit_name), rt.B6_RECOMPILE_LIMIT)
        self.assertIs(getattr(config, fail_name), True)

    def test_compile_b6_disables_automatic_dynamic_shapes(self):
        self.check("recompile_limit", "fail_on_recompile_limit_hit")

    def test_compile_b6_sets_cache_size_limit_on_older_torch(self):
        self.check("cache_size_limit", "fail_on_cache_limit_hit")

    def test_recompile_limit_covers_tiers_label_ranges_and_grad_modes(self):
        self.assertGreaterEqual(rt.B6_RECOMPILE_LIMIT, 3 * rt.B6_GRAPHS_PER_SHAPE * 2)
        self.assertLessEqual(rt.B6_RECOMPILE_LIMIT, 16)
        self.assertEqual(rt.B6_GRAPHS_PER_SHAPE, 2)


class CompileGuardTests(unittest.TestCase):
    def test_second_graph_for_seen_length_after_warmup_allowed_once(self):
        guard = rt.CompileGuard(warmup_steps=2)
        guard.observe(1, (1, 48), 1, False)
        guard.observe(2, (1, 48), 1, False)
        guard.observe(165, (1, 48), 2, False)              # labels crossed the Inductor fusion threshold
        guard.observe(166, (1, 48), 2, False)
        with self.assertRaisesRegex(RuntimeError, "recompile after warm-up"):
            guard.observe(167, (1, 48), 3, False)

    def test_new_tier_after_warmup_allowed_with_its_own_label_split(self):
        guard = rt.CompileGuard(warmup_steps=1)
        guard.observe(1, (1, 48), 1, False)
        guard.observe(10, (1, 96), 2, False)               # new tier
        guard.observe(20, (1, 96), 3, False)               # its label split
        guard.observe(30, (1, 48), 4, False)               # the first tier's label split
        with self.assertRaisesRegex(RuntimeError, "recompile after warm-up"):
            guard.observe(40, (1, 96), 5, False)

    def test_total_graph_cap_raises_on_too_many_tiers(self):
        guard = rt.CompileGuard(warmup_steps=0, max_total_graphs=3)
        guard.observe(1, (1, 8), 1, False)
        guard.observe(2, (1, 16), 2, False)
        guard.observe(3, (1, 32), 3, False)
        with self.assertRaisesRegex(RuntimeError, "total"):
            guard.observe(4, (1, 64), 4, False)

    def test_default_total_cap_is_the_recompile_limit(self):
        self.assertEqual(rt.CompileGuard(3).max_total_graphs, rt.B6_RECOMPILE_LIMIT)

    def test_suppress_errors_raises(self):
        with self.assertRaisesRegex(RuntimeError, "suppress_errors"):
            rt.CompileGuard(1).observe(1, (1, 48), 1, True)


@unittest.skipUnless(HAS_TORCH, "needs torch (container suite)")
class StaticLengthCompileTests(unittest.TestCase):
    def test_lengths_compile_static_and_labels_stay_dynamic(self):
        import torch
        import torch._dynamo as dynamo

        dynamo.reset()
        self.addCleanup(dynamo.reset)
        shapes = []

        def recording_backend(gm, example_inputs):
            shapes.append({n.name: tuple(str(s) for s in n.meta["example_value"].shape)
                           for n in gm.graph.nodes
                           if n.op == "placeholder" and isinstance(n.meta.get("example_value"), torch.Tensor)})
            return gm.forward

        class Tiny(torch.nn.Module):
            def forward(self, input_ids, labels):
                return input_ids.float().sum(1) + labels.float().mean(1)

        with dynamo.config.patch(automatic_dynamic_shapes=True):
            compiled = rt.compile_b6(Tiny(), backend=recording_backend)
            for length, n_labels in [(8, 3), (16, 4), (32, 5), (8, 6), (16, 7), (32, 9)]:
                labels = torch.ones(1, n_labels)
                dynamo.maybe_mark_dynamic(labels, 1)
                compiled(torch.ones(1, length), labels)
        self.assertEqual(len(shapes), 3)                       # one graph per length, reused on return to 8
        ids = [s for g in shapes for name, s in g.items() if "input_ids" in name]
        labs = [s for g in shapes for name, s in g.items() if "labels" in name]
        self.assertEqual(ids, [("1", "8"), ("1", "16"), ("1", "32")])
        self.assertEqual(len(labs), 3)
        self.assertTrue(all(not s[1].isdigit() for s in labs), labs)


if __name__ == "__main__":
    unittest.main()
