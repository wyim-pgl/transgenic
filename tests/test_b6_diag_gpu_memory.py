"""1-GPU memory diagnostic (revision/scripts/b6/diag_gpu_memory.py): the parts that run without a GPU.

Argument parsing, the tier:labels shape parser, the per-rank accumulation the trainer uses, OOM classification and
the JSON summary/table. Torch is optional here: only the synthetic-batch check needs it.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transgenic.training import b6_runtime as rt  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


diag = _load("b6_diag_gpu_memory", "revision/scripts/b6/diag_gpu_memory.py")
PILOT_CFG = os.path.join(ROOT, "configs", "b6_400m_win_v3_wp5_pilot.json")
GIB = 2 ** 30

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _mb(i, tier, labels, alloc_gib, resv_gib, graphs, inactive_gib=0.0):
    return dict(micro_batch=i, tier=tier, labels=labels, max_allocated_bytes=int(alloc_gib * GIB),
                max_reserved_bytes=int(resv_gib * GIB), inactive_split_bytes=int(inactive_gib * GIB),
                compiled_graphs=graphs, num_alloc_retries=0)


class ShapeParserTests(unittest.TestCase):
    def test_parses_tier_label_pairs_in_order(self):
        self.assertEqual(diag.parse_shapes("129024:4480, 61440:3000,30720:1500"),
                         [(129024, 4480), (61440, 3000), (30720, 1500)])

    def test_refuses_malformed_or_impossible_shapes(self):
        for bad in ("", "129024", "129024:x", "129025:100", "0:100", "30720:1", "1:2:3"):
            with self.subTest(bad=bad):
                with self.assertRaises(diag.Refused):
                    diag.parse_shapes(bad)

    def test_sequence_cycles_through_the_shapes(self):
        shapes = [(30720, 10), (61440, 20)]
        self.assertEqual(diag.shape_sequence(shapes, 5),
                         [(30720, 10), (61440, 20), (30720, 10), (61440, 20), (30720, 10)])

    def test_default_mix_contains_the_required_shapes_and_fits_the_config(self):
        shapes = diag.parse_shapes(diag.DEFAULT_SHAPES)
        for s in [(129024, 4480), (129024, 8192), (61440, 3000), (30720, 1500), (129024, 6000)]:
            self.assertIn(s, shapes)
        self.assertGreater(len(shapes), len(set(shapes)))          # includes repeats
        cfg = rt.load_b6_config(PILOT_CFG)
        self.assertEqual(diag.check_shapes_against_config(shapes, cfg), [])

    def test_config_check_refuses_oversize_and_flags_off_tier(self):
        cfg = rt.load_b6_config(PILOT_CFG)
        with self.assertRaises(diag.Refused):
            diag.check_shapes_against_config([(129030, 10)], cfg)
        with self.assertRaises(diag.Refused):
            diag.check_shapes_against_config([(30720, 8193)], cfg)
        self.assertEqual(len(diag.check_shapes_against_config([(3072, 64)], cfg)), 1)


class ArgumentTests(unittest.TestCase):
    def test_defaults_match_the_trainer_run(self):
        a = diag.parse_args(["--out", "x.json"])
        self.assertEqual((a.config, a.deterministic, a.ddp, a.arm, a.dense_tracks, a.chain_elements, a.accum,
                          a.world_size, a.compile_warmup_steps),
                         ("configs/b6_400m_win_v3_wp5_pilot.json", True, True, "B6", True, 511, None, 4, 3))
        self.assertEqual(a.shape_list, diag.parse_shapes(diag.DEFAULT_SHAPES))

    def test_on_off_switches(self):
        a = diag.parse_args(["--out", "x.json", "--deterministic", "off", "--no-dense-tracks", "--accum", "6",
                             "--ddp", "off", "--arm", "C2", "--measure-minutes", "8"])
        self.assertEqual((a.deterministic, a.dense_tracks, a.accum, a.ddp, a.arm, a.measure_minutes),
                         (False, False, 6, False, "C2", 8.0))

    def test_full_run_schedule_matches_the_trainer_formula(self):
        cfg = rt.load_b6_config(PILOT_CFG)
        s = diag.full_run_schedule(cfg, 24)
        # rows = 4,099 * 96 / 2 = 196,752; ceil(rows / 24) = 8,198 updates per epoch; x 22 epochs; 5% warm-up
        self.assertEqual((s["train_rows_approx"], s["steps_per_epoch"], s["t_total"], s["warmup_steps"]),
                         (196752, 8198, 180356, 9017))

    def test_rank_out_paths(self):
        self.assertEqual(diag.rank_out_path("/o/a_trainer.json", 0), "/o/a_trainer.json")
        self.assertEqual(diag.rank_out_path("/o/a_trainer.json", 1), "/o/a_trainer.rank1.json")

    def test_refused_arguments_exit_with_the_refused_code(self):
        for argv in (["--out", "x.json", "--shapes", "7:10"], ["--out", "x.json", "--updates", "0"],
                     ["--out", "x.json", "--accum", "0"], ["--out", "x.json", "--chain-elements", "512"], []):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as cm:
                    diag.parse_args(argv)
                self.assertEqual(cm.exception.code, diag.EXIT_REFUSED)

    def test_summarize_mode_needs_no_out(self):
        a = diag.parse_args(["--summarize", "a.json", "b.json"])
        self.assertEqual(a.summarize, ["a.json", "b.json"])

    def test_default_accumulation_is_the_trainer_per_rank_value(self):
        cfg = rt.load_b6_config(PILOT_CFG)
        # run_b6: total = accumulation_steps(cfg, batch_size=1) = 96 / world 4 -> 24 micro-batches per rank.
        self.assertEqual(diag.per_rank_accumulation(cfg, 4), rt.accumulation_steps(cfg, 1) // 4)
        self.assertEqual(diag.per_rank_accumulation(cfg, 4), 24)
        with self.assertRaises(diag.Refused):
            diag.per_rank_accumulation(cfg, 5)

    def test_the_tool_never_sets_the_allocator_config(self):
        with open(os.path.join(ROOT, "revision", "scripts", "b6", "diag_gpu_memory.py")) as fh:
            src = fh.read()
        self.assertNotIn('environ["PYTORCH_CUDA_ALLOC_CONF"]', src)
        self.assertNotIn("environ.setdefault(\"PYTORCH_CUDA_ALLOC_CONF\"", src)
        self.assertNotIn("_set_allocator_settings", src)


class OomAndSummaryTests(unittest.TestCase):
    def test_oom_classification(self):
        class FakeOOM(RuntimeError):
            pass
        self.assertTrue(diag.is_oom(FakeOOM("boom"), (FakeOOM,)))
        self.assertTrue(diag.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.08 GiB")))
        self.assertFalse(diag.is_oom(RuntimeError("shape mismatch")))
        self.assertFalse(diag.is_oom(ValueError("out of memory")))

    def report(self, **kw):
        r = dict(label="a_trainer", status="oom", settings=dict(deterministic=True),
                 env=dict(PYTORCH_CUDA_ALLOC_CONF=None),
                 micro_batches=[_mb(1, 129024, 4480, 60.0, 70.0, 3, 1.5), _mb(2, 30720, 1500, 20.0, 72.0, 5),
                                _mb(3, 129024, 4480, 62.5, 80.0, 5, 6.0)],
                 updates=[dict(update=1, after_micro_batch=2, max_reserved_bytes=int(81 * GIB))],
                 oom=dict(micro_batch=4, tier=129024, labels=8192, phase="backward"))
        r.update(kw)
        return r

    def test_per_shape_maxima_and_run_facts(self):
        s = diag.summarize_report(self.report())
        self.assertEqual(s["per_shape"]["129024:4480"]["n"], 2)
        self.assertAlmostEqual(s["per_shape"]["129024:4480"]["peak_allocated_gib"], 62.5, places=6)
        self.assertAlmostEqual(s["per_shape"]["129024:4480"]["peak_reserved_gib"], 80.0, places=6)
        self.assertAlmostEqual(s["per_shape"]["129024:4480"]["max_inactive_split_gib"], 6.0, places=6)
        self.assertAlmostEqual(s["peak_reserved_gib"], 81.0, places=6)          # optimizer-update peak counts
        self.assertAlmostEqual(s["update_peak_reserved_gib"], 81.0, places=6)
        self.assertEqual((s["compiled_graphs"], s["oom_shape"], s["oom_phase"], s["micro_batches"], s["updates"]),
                         (5, "129024:8192", "backward", 3, 1))

    def test_table_lists_every_configuration_and_the_oom(self):
        ok = self.report(label="b_expandable", status="completed", oom=None,
                         env=dict(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        text = diag.format_table([self.report(), ok])
        self.assertIn("a_trainer: status=oom", text)
        self.assertIn("129024:8192 at micro-batch 4 (backward)", text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", text)
        self.assertIn("OOM: none", text)
        self.assertLess(text.index(" 30720:1500"), text.index("129024:4480"))      # shapes in numeric order

    def test_summarize_main_reads_json_and_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a_trainer.json")
            diag.write_json(p, self.report())
            with open(p) as fh:
                self.assertEqual(json.load(fh)["status"], "oom")
            self.assertFalse(os.path.exists(p + ".tmp"))
            self.assertEqual(diag.summarize_main([p]), diag.EXIT_OK)
            self.assertEqual(diag.summarize_main([p, os.path.join(d, "missing.json")]), diag.EXIT_ERROR)

    def test_run_maxima_include_the_oom_point_and_optimizer_updates(self):
        r = dict(label="a", status="oom", settings=dict(accum=2), env={},
                 micro_batches=[_mb(1, 129024, 4480, 60.0, 70.0, 3)],
                 updates=[dict(update=1, after_micro_batch=2, max_allocated_bytes=int(61 * GIB),
                               max_reserved_bytes=int(71 * GIB))],
                 oom=dict(micro_batch=3, tier=129024, labels=4480, phase="backward",
                          memory=dict(max_allocated_bytes=int(87 * GIB), max_reserved_bytes=int(93 * GIB),
                                      inactive_split_bytes=int(6 * GIB))))
        s = diag.summarize_report(r)
        self.assertAlmostEqual(s["peak_allocated_gib"], 87.0, places=6)
        self.assertAlmostEqual(s["peak_reserved_gib"], 93.0, places=6)
        self.assertAlmostEqual(s["per_shape"]["129024:4480"]["peak_reserved_gib"], 93.0, places=6)
        self.assertEqual(s["per_shape"]["129024:4480"]["n"], 2)
        r2 = dict(r, oom=None, status="completed")
        self.assertAlmostEqual(diag.summarize_report(r2)["peak_reserved_gib"], 71.0, places=6)    # update peak

    def test_classification_comes_from_the_json_status(self):
        self.assertEqual([diag.classify(x) for x in (None, dict(status="completed"), dict(status="time_budget"),
                                                     dict(status="oom"), dict(status="error"), dict(status="running"))],
                         ["missing", "finished", "finished", "oom", "error", "incomplete"])
        with tempfile.TemporaryDirectory() as d:
            ok, oom, run = (os.path.join(d, n) for n in ("ok.json", "oom.json", "run.json"))
            diag.write_json(ok, dict(label="ok", status="completed", micro_batches=[], updates=[]))
            diag.write_json(oom, self.report())
            diag.write_json(run, dict(label="run", status="running", micro_batches=[], updates=[]))
            self.assertEqual(diag.summarize_main([ok, oom]), diag.EXIT_OK)
            self.assertEqual(diag.summarize_main([ok, run]), diag.EXIT_ERROR)

    def test_growth_slope_per_100_micro_batches_after_warm_up(self):
        accum = 4
        mbs = []
        for i in range(1, 41):                      # warm-up = 1 update = micro-batches 1..4 (excluded)
            leak = (i * 0.01 if i > 4 else 50.0)   # 0.01 GiB per micro-batch after warm-up; junk inside warm-up
            mb = _mb(i, 129024 if i % 2 else 30720, 8192 if i % 2 else 1500, 60.0, 70.0, 5)
            mb.update(allocated_after_backward_bytes=int((10 + leak) * GIB), reserved_bytes=int(70 * GIB))
            mbs.append(mb)
        ups = [dict(update=u, after_micro_batch=u * accum, allocated_after_step_zero_grad_bytes=int((5 + u * 0.04) * GIB),
                    reserved_bytes=int(70 * GIB), inactive_split_bytes=0) for u in range(1, 11)]
        g = diag.growth_slopes(dict(settings=dict(accum=accum, compile_warmup_steps=1), micro_batches=mbs, updates=ups))
        self.assertAlmostEqual(g["update_first_mb"]["alloc"], 1.0, places=2)     # 0.01 GiB/mb -> 1 GiB / 100 mb
        self.assertEqual(g["update_first_mb"]["n"], 9)                             # mbs 5, 9, ..., 37
        self.assertAlmostEqual(g["update_first_mb"]["resv"], 0.0, places=6)
        self.assertAlmostEqual(g["after_update"]["alloc"], 1.0, places=2)         # 0.04 GiB per 4 mb
        self.assertAlmostEqual(g["shape:129024:8192"]["alloc"], 1.0, places=2)
        self.assertIn("shape:30720:1500", g)
        self.assertIsNone(diag.slope_per_100([(1.0, 5.0)]))
        text = diag.format_table([dict(label="g", status="completed", settings=dict(accum=accum, compile_warmup_steps=1),
                                       env={}, micro_batches=mbs, updates=ups)])
        self.assertIn("growth after warm-up", text)
        self.assertIn("update_first_mb", text)

    def test_ranks_aggregate_and_any_rank_oom_makes_the_configuration_oom(self):
        with tempfile.TemporaryDirectory() as d:
            r0 = os.path.join(d, "a_trainer.json")
            diag.write_json(r0, dict(label="a_trainer", status="completed", settings=dict(accum=2), env={},
                                     micro_batches=[_mb(1, 129024, 4480, 60.0, 70.0, 3)], updates=[]))
            self.assertEqual(diag.summarize_main([r0], ranks=2), diag.EXIT_ERROR)      # rank 1 missing
            diag.write_json(diag.rank_out_path(r0, 1), self.report())                    # rank 1 OOM
            self.assertEqual(diag.summarize_main([r0], ranks=2), diag.EXIT_OK)
        agg = diag.aggregate_ranks([dict(status="completed", micro_batches=[], updates=[]), self.report()])
        self.assertEqual((agg["classification"], agg["rank"], agg["oom_shape"], agg["ranks"]),
                         ("oom", 1, "129024:8192", ["finished", "oom"]))
        self.assertEqual(diag.aggregate_ranks([dict(status="completed"), None])["classification"], "missing")
        self.assertEqual(diag.aggregate_ranks([dict(status="error"), dict(status="running")])["classification"], "error")
        self.assertEqual(diag.parse_args(["--summarize", "a.json", "--ranks", "2"]).ranks, 2)

    def test_scheduler_advances_as_production_world_four(self):
        self.assertEqual(diag.scheduler_ticks_per_update(4, 1), 4)
        self.assertEqual(diag.scheduler_ticks_per_update(4, 2), 2)
        self.assertEqual(diag.scheduler_ticks_per_update(4, 4), 1)

    def test_zero_warmup_opens_the_measurement_window_immediately(self):
        t0 = 1000.0
        self.assertEqual(diag.measurement_start(t0, 0), t0)
        self.assertIsNone(diag.measurement_start(t0, 3))
        self.assertTrue(diag.budget_reached(t0 + 61, t0, diag.measurement_start(t0, 0), None, 1.0))
        self.assertFalse(diag.budget_reached(t0 + 61, t0, diag.measurement_start(t0, 3), None, 1.0))
        self.assertTrue(diag.budget_reached(t0 + 120, t0, None, 2.0, None))
        self.assertEqual(diag.parse_args(["--out", "x.json", "--compile-warmup-steps", "0"]).compile_warmup_steps, 0)
        with self.assertRaises(SystemExit):
            diag.parse_args(["--out", "x.json", "--compile-warmup-steps", "-1"])

    def test_empty_running_report_summarizes(self):
        s = diag.summarize_report(dict(status="running", micro_batches=[], updates=[]))
        self.assertEqual((s["peak_reserved_gib"], s["compiled_graphs"], s["oom_shape"]), (0.0, None, None))


SHA_A, SHA_B = "a" * 64, "0123456789abcdef" * 4


class RealReplayParsingTests(unittest.TestCase):
    def test_draw_log_last_rows_in_draw_order(self):
        lines = [f"{100 + i}\t0\t{SHA_A}\n" for i in range(50)] + ["\n"]
        rows = diag.parse_draw_log(lines, 40)
        self.assertEqual(len(rows), 40)
        self.assertEqual((rows[0], rows[-1]), ((110, 0, SHA_A), (149, 0, SHA_A)))
        self.assertEqual(diag.parse_draw_log([f"7\t1\t{SHA_B}"], 40), [(7, 1, SHA_B)])

    def test_draw_log_refuses_malformed_rows(self):
        for bad in ([f"7\t0"], [f"x\t0\t{SHA_A}"], ["7\t0\tnot-a-sha"], [f"7\t0\t{SHA_A.upper()}"], []):
            with self.subTest(bad=bad):
                with self.assertRaises(diag.Refused):
                    diag.parse_draw_log(bad, 40)

    def test_index_list_and_file(self):
        self.assertEqual(diag.parse_index_list("5, 6:1\n7"), [(5, 0), (6, 1), (7, 0)])
        for bad in ("5:x", "a", "1:2:3"):
            with self.assertRaises(diag.Refused):
                diag.parse_index_list(bad)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "idx.txt")
            with open(f, "w") as fh:
                fh.write("11\n12:0\n")
            self.assertEqual(diag.read_index_arg(f), [(11, 0), (12, 0)])
        self.assertEqual(diag.read_index_arg("13,14"), [(13, 0), (14, 0)])

    def test_trainer_module_loads_with_its_dataset_functions(self):
        tr = diag._load_trainer()
        for name in ("_wp3", "verify_training_inputs", "corpus_rows", "build_model"):
            self.assertTrue(callable(getattr(tr, name)), name)

    def test_real_mode_arguments(self):
        a = diag.parse_args(["--out", "x.json", "--real-bindings", "b.json", "--real-from-draw-log", "r.tsv"])
        self.assertEqual((a.real_bindings, a.real_from_draw_log, a.last, a.real_indices, a.real_verify),
                         ("b.json", "r.tsv", 40, None, False))
        for argv in (["--out", "x.json", "--real-bindings", "b.json"],
                     ["--out", "x.json", "--real-indices", "5"],
                     ["--out", "x.json", "--real-bindings", "b.json", "--real-indices", "5", "--last", "0"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as cm:
                    diag.parse_args(argv)
                self.assertEqual(cm.exception.code, diag.EXIT_REFUSED)

    def test_signature_diff_ignores_L_and_T_only(self):
        synth = {"input_ids": dict(shape=[1, 3072], dtype="torch.int64"),
                 "labels": dict(shape=[1, 64], dtype="torch.int64"),
                 "chain_idx": dict(shape=[1, 512, 2], dtype="torch.int64"),
                 "evidence_tracks": dict(shape=[1, 3072, 31], dtype="torch.float16")}
        same = {"input_ids": dict(shape=[1, 129024], dtype="torch.int64"),
                "labels": dict(shape=[1, 8000], dtype="torch.int64"),
                "chain_idx": dict(shape=[1, 512, 2], dtype="torch.int64"),
                "evidence_tracks": dict(shape=[1, 129024, 31], dtype="torch.float16")}
        self.assertEqual(diag.signature_diff(same, synth), [])
        real = dict(same, chain_idx=dict(shape=[1, 1024, 2], dtype="torch.int32"),
                    extra=dict(shape=[1, 4], dtype="torch.bool"))
        real.pop("labels")
        diff = diag.signature_diff(real, synth)
        self.assertTrue(any(x.startswith("extra: only in the real batch") for x in diff))
        self.assertTrue(any(x.startswith("labels: only in the synthetic batch") for x in diff))
        self.assertTrue(any("chain_idx: dtype torch.int32" in x for x in diff))
        self.assertTrue(any("chain_idx: fixed dims [1]" in x for x in diff))

    def real_report(self):
        def smp(order, rn, alloc, status="ok", match=True):
            return dict(order=order, rn=rn, epoch=0, status=status, draw_sha_match=match,
                        describe=dict(L=129024, attended=129000, label_len=8192, label_nonpad=8192, chain_elements=300),
                        memory=dict(max_allocated_bytes=int(alloc * GIB), max_reserved_bytes=int((alloc + 5) * GIB)))
        return dict(label="real_a_trainer", mode="real", status="oom", settings=dict(accum=24), env={},
                    micro_batches=[_mb(1, 129024, 8192, 58.0, 63.0, 3), _mb(2, 129024, 8192, 61.0, 66.0, 3)],
                    updates=[], samples=[smp(1, 11, 58.0), smp(2, 22, 61.0, match=False),
                                         smp(3, 33, 87.0, status="oom", match=None)],
                    oom=dict(micro_batch=3, rn=33, tier=129024, labels=8192, phase="backward",
                             memory=dict(max_allocated_bytes=int(87 * GIB), max_reserved_bytes=int(92 * GIB))),
                    signature_diff_vs_synthetic=["chain_idx: dtype torch.int32 (synthetic torch.int64)"])

    def test_real_table_sorted_by_peak_allocated_with_the_diff(self):
        text = diag.format_real_table(self.real_report())
        body = text.splitlines()[1:4]
        self.assertEqual([int(l.split()[1]) for l in body], [33, 22, 11])
        self.assertIn("oom", body[0])
        self.assertIn("DIFF", body[1])
        self.assertIn("chain_idx: dtype torch.int32", text)
        s = diag.summarize_report(self.real_report())
        self.assertAlmostEqual(s["peak_allocated_gib"], 87.0, places=6)
        self.assertEqual(s["classification"], "oom")
        self.assertIn("real-data replay (3 samples", diag.format_table([self.real_report()]))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class BatchTests(unittest.TestCase):
    def test_batch_is_the_bench_contract_at_the_requested_shape(self):
        import torch
        cfg = rt.load_b6_config(PILOT_CFG)
        a = diag.parse_args(["--out", "x.json"])
        b = diag.make_batch(torch, cfg, a, 3072, 64, 1, torch.device("cpu"))
        self.assertEqual((tuple(b["input_ids"].shape), tuple(b["labels"].shape)), ((1, 3072), (1, 64)))
        self.assertEqual(int(b["chain_mask"].sum()), 511)
        self.assertEqual(torch.count_nonzero(b["evidence_tracks"]).item(), 3072 * 31)
        a0 = diag.parse_args(["--out", "x.json", "--no-dense-tracks"])
        self.assertEqual(torch.count_nonzero(diag.make_batch(torch, cfg, a0, 3072, 64, 1,
                                                             torch.device("cpu"))["evidence_tracks"]).item(), 0)
        rt.validate_b6_batch(b)                     # the trainer's check passes on the diagnostic's batch
        self.assertEqual(rt.model_inputs(rt.apply_arm(b, rt.ARMS["C2"]))["chain_mask"].sum().item(), 0)
        d = diag.describe_batch(torch, b)
        self.assertEqual((d["L"], d["attended"], d["label_len"], d["label_nonpad"], d["chain_elements"]),
                         (3072, 3072, 64, 64, 511))
        self.assertEqual(d["track_nonzero_fraction"]["all"], 1.0)
        other = diag.make_batch(torch, cfg, a, 3072, 64, 2, torch.device("cpu"))
        self.assertFalse(torch.equal(b["input_ids"], other["input_ids"]))


if __name__ == "__main__":
    unittest.main()
