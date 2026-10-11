"""NGC 26.08's three label intervals and the explicit guard-only checkpoint upgrade."""
import json
from pathlib import Path
import unittest

from tests.test_b6_trainer import TrainerBase, rt


POLICY_PREFIX = ("torch.compile backend=inductor; suppress_errors=False; optimize_ddp=False;"
                 " automatic_dynamic_shapes=False (labels dim 1 dynamic, DNA length static);")
OLD_POLICY = POLICY_PREFIX + " recompile_limit=16; graphs_per_tier=2; fallback=failure"
NEW_POLICY = POLICY_PREFIX + " recompile_limit=18; graphs_per_tier=3; fallback=failure"


class LabelSpecializationTests(unittest.TestCase):
    def test_reset_compiler_counters_are_rejected(self):
        guard = rt.CompileGuard(warmup_steps=3, graphs_before=5)
        with self.assertRaisesRegex(RuntimeError, "counters were reset"):
            guard.observe(100, (1, 129024), 4, False)

    def test_real_c2_compilation_trace_accepts_all_three_label_intervals(self):
        # Real GPU replay 3360978, original rank3 rows1,3,7,3915,6718.
        # The 129024 tier saw labels80,4501,3924; measured guards split at3414 and4096.
        trace = [(1, (1, 30720), 1), (1, (1, 129024), 2), (1, (1, 61440), 3),
                 (164, (1, 129024), 4), (280, (1, 129024), 5)]
        guard = rt.CompileGuard(warmup_steps=3)
        for step, shape, count in trace:
            guard.observe(step, shape, count, False)
        with self.assertRaisesRegex(RuntimeError, "recompile after warm-up"):
            guard.observe(281, (1, 129024), 6, False)

    def test_three_regions_for_each_training_and_eval_tier_fit_the_default_budget(self):
        guard = rt.CompileGuard(warmup_steps=3)
        count = 0
        for mode in ((), ("eval",)):
            for length in (30720, 61440, 129024):
                for region in range(3):
                    count += 1
                    guard.observe(100 + count, mode + (1, length), count, False)
        with self.assertRaisesRegex(RuntimeError, "compile events in total"):
            guard.observe(200, ("unexpected", 1, 129024), 19, False)


class CompilePolicyRecoveryTests(TrainerBase):
    def identities(self):
        spec = self.spec("C2", stop_at_step=1)
        self.train_arm(spec)
        stored = json.loads(Path(spec.run_dir, "run_config_b6.json").read_text())
        return dict(stored, compile_policy=OLD_POLICY), dict(stored, compile_policy=NEW_POLICY)

    def test_guard_only_upgrade_accepts_old_run_and_checkpoint_without_rewriting_identity(self):
        old, new = self.identities()
        checkpoint = Path(self.tmp.name, "old_checkpoint")
        checkpoint.mkdir()
        rt.write_checkpoint_identity(str(checkpoint), old)
        path = checkpoint / rt.CHECKPOINT_IDENTITY
        before = path.read_bytes()
        rt.check_run_config_b6(old, new)
        rt.check_checkpoint_identity(str(checkpoint), new)
        self.assertEqual(path.read_bytes(), before)
        upgraded = Path(self.tmp.name, "upgraded_checkpoint")
        upgraded.mkdir()
        rt.write_checkpoint_identity(str(upgraded), new)
        rt.check_run_config_b6(old, new)  # the immutable run root still carries the original identity
        rt.check_checkpoint_identity(str(upgraded), new)

    def test_upgrade_is_directional_and_does_not_accept_any_other_policy_change(self):
        old, new = self.identities()
        with self.assertRaisesRegex(RuntimeError, "compile_policy"):
            rt.check_run_config_b6(new, old)
        for before, after in (("backend=inductor", "backend=eager"),
                              ("optimize_ddp=False", "optimize_ddp=True"),
                              ("suppress_errors=False", "suppress_errors=True"),
                              ("automatic_dynamic_shapes=False", "automatic_dynamic_shapes=True"),
                              ("recompile_limit=16", "recompile_limit=17"),
                              ("graphs_per_tier=2", "graphs_per_tier=1")):
            changed = dict(old, compile_policy=OLD_POLICY.replace(before, after))
            with self.subTest(change=after), self.assertRaisesRegex(RuntimeError, "compile_policy"):
                rt.check_run_config_b6(changed, new)
        for key in rt.IDENTITY_REQUIRED_KEYS:
            if key == "compile_policy":
                continue
            changed = dict(new, **{key: "different"})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "resume refused"):
                rt.check_run_config_b6(old, changed)

    def test_accepted_upgrade_is_explicit_in_link_log_and_preserves_first_identity(self):
        old, new = self.identities()
        run = Path(self.tmp.name, "audit")
        run.mkdir()
        rt.record_identity(str(run), old, [0.0], {"link": 1})
        before = (run / "run_config_b6.json").read_bytes()
        rt.record_identity(str(run), new, [0.0], {"link": 2})
        self.assertEqual((run / "run_config_b6.json").read_bytes(), before)
        lines = (run / "runtime_identity.txt").read_text().splitlines()
        transitions = [json.loads(line.split("=", 1)[1]) for line in lines
                       if line.startswith("link.compile_policy_transition=")]
        self.assertEqual(transitions, [{"from": OLD_POLICY, "to": NEW_POLICY,
                                       "reason": "NGC label fusion intervals; guard limits only",
                                       "evidence": "notes/c2_compile_recovery_20261011.md"}])
