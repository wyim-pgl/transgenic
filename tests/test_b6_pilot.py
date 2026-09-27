"""WP5 pilot stop (work plan v1.2 WP5; A46 rule 11): B6 and C2 train for exactly N_pilot optimizer updates, archive
the step-N checkpoint as the pilot-final one (never best-validation) and exit; resume lands on N; the stop target is
part of the run identity; the pilot config is gated and refuses --allow-ungated.

N_pilot is scaled down (6 instead of 4,099) as in tests/test_b6_trainer.py; the production path is unchanged.
"""
import hashlib
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_b6_trainer import ROOT, TrainerBase, rt, trainer

PILOT_CFG = os.path.join(ROOT, "configs", "b6_400m_win_v3_wp5_pilot.json")
BASE_CFG = os.path.join(ROOT, "configs", "b6_400m_win_v3.json")


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


class PilotStopTests(TrainerBase):
    def test_full_budget_arms_stop_at_exactly_n_updates_and_mark_the_pilot_final_checkpoint(self):
        for arm in ("B6", "C2"):
            with self.subTest(arm=arm):
                spec = self.spec(arm, run=f"pilot_{arm}", stop_after_updates=6)      # 2 epochs reach 8 updates
                result = self.train_arm(spec)
                self.assertEqual((result["done"], result["global_step"], result["reason"]),
                                 (True, 6, "stop_after_updates"))
                run = spec.run_dir
                self.assertTrue(os.path.exists(os.path.join(run, "TRAINING_DONE")))
                step = rt.step_checkpoint_dir(run, 6)
                for name in ("model.safetensors", "meta.json", rt.PRUNE_EXEMPT_MARKER, rt.PILOT_FINAL_MARKER):
                    self.assertTrue(os.path.exists(os.path.join(step, name)), name)
                meta = json.load(open(os.path.join(step, "meta.json")))
                self.assertIs(meta["pilot_final"], True)
                self.assertEqual((meta["global_step"], meta["stop_after_updates"]), (6, 6))
                self.assertEqual(meta["checkpoint_kind"], "fixed_step")
                self.assertEqual(meta["selection"], rt.FIXED_STEP_SELECTION)
                self.assertEqual(meta["model_sha256"], rt.file_sha256(os.path.join(step, "model.safetensors")))
                self.assertEqual(result["pilot_final"], step)
                # optimizer updates, not micro-batches: accumulation 2 -> 12 trained rows for 6 updates
                rows = open(os.path.join(run, "draw_log", "rank00.tsv")).read().splitlines()
                self.assertEqual(len(rows), 12)
                self.assertFalse(os.path.exists(os.path.join(run, "epoch_02")))      # stopped inside epoch 2
                self.assertEqual(sorted(n for n in os.listdir(run) if n.startswith("step_")), ["step_00006"])
                ident = json.load(open(os.path.join(run, "run_config_b6.json")))
                self.assertEqual(ident["stop_after_updates"], 6)

    def test_non_pilot_step_checkpoint_is_fixed_step_but_not_pilot_final(self):
        spec = self.spec("B6", run="full")
        self.train_arm(spec)
        step = rt.step_checkpoint_dir(spec.run_dir, 6)
        meta = json.load(open(os.path.join(step, "meta.json")))
        self.assertIs(meta["pilot_final"], False)
        self.assertEqual(meta["stop_after_updates"], None)
        self.assertFalse(os.path.exists(os.path.join(step, rt.PILOT_FINAL_MARKER)))
        self.assertEqual(json.load(open(os.path.join(spec.run_dir, "run_config_b6.json")))["stop_after_updates"], None)

    def test_pilot_uses_the_full_run_schedule_truncated_at_n(self):
        """A46 rule 11 matched budget: the pilot's step-N weights and LR equal those of the full-duration run at
        step N (and of the C3/C4/C6/C7 n_pilot arms, which share that schedule): the schedule is not resized."""
        full, pilot = self.spec("B6", run="sched_full"), self.spec("B6", run="sched_pilot", stop_after_updates=6)
        self.train_arm(full)
        self.train_arm(pilot)
        a, b = (rt.step_checkpoint_dir(s.run_dir, 6) for s in (full, pilot))
        self.assertEqual(_read(os.path.join(a, "model.safetensors")), _read(os.path.join(b, "model.safetensors")))
        ma, mb = (json.load(open(os.path.join(d, "meta.json"))) for d in (a, b))
        self.assertEqual((ma["lr"], ma["schedule_total_steps"], ma["schedule_warmup_steps"]),
                         (mb["lr"], mb["schedule_total_steps"], mb["schedule_warmup_steps"]))
        # 8 rows / per-rank accumulation 2 = 4 scheduled steps per epoch x max_epochs 3 (cfg) = 12 > N = 6
        self.assertEqual(mb["schedule_total_steps"], 12)

    def test_resume_across_a_restart_lands_on_n(self):
        first = self.train_arm(self.spec("B6", run="chain", stop_after_updates=6, stop_at_step=3))
        self.assertEqual((first["preempted"], first["global_step"]), (True, 3))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "chain", "TRAINING_DONE")))
        meta = json.load(open(os.path.join(self.tmp.name, "chain", "latest_state", "meta.json")))
        self.assertEqual(meta["stop_after_updates"], 6)
        resumed = self.train_arm(self.spec("B6", run="chain", stop_after_updates=6, resume="auto"))
        self.assertEqual((resumed["done"], resumed["global_step"], resumed["reason"]), (True, 6, "stop_after_updates"))
        step = rt.step_checkpoint_dir(os.path.join(self.tmp.name, "chain"), 6)
        self.assertTrue(os.path.exists(os.path.join(step, rt.PILOT_FINAL_MARKER)))
        text = open(os.path.join(self.tmp.name, "chain", "runtime_identity.txt")).read()
        self.assertEqual(text.count("# link"), 2)

    def test_resume_refused_with_a_different_or_missing_stop_target(self):
        self.train_arm(self.spec("B6", run="tgt", stop_after_updates=6, stop_at_step=2))
        with self.assertRaisesRegex(RuntimeError, "stop_after_updates differs"):
            self.train_arm(self.spec("B6", run="tgt", resume="auto"))                  # full-budget resume
        self.train_arm(self.spec("B6", run="plain", stop_at_step=2))
        with self.assertRaisesRegex(RuntimeError, "stop_after_updates differs"):
            self.train_arm(self.spec("B6", run="plain", stop_after_updates=6, resume="auto"))
        with self.assertRaisesRegex(ValueError, "must equal N_pilot"):                  # any other target
            self.train_arm(self.spec("B6", run="tgt", stop_after_updates=5, resume="auto"))

    def test_stop_target_must_equal_n_pilot(self):
        for bad in (0, 5, 7, 4142, 4099):
            with self.subTest(target=bad), self.assertRaisesRegex(ValueError, "must equal N_pilot"):
                rt.pilot_stop_target(bad, 6)
        self.assertIsNone(rt.pilot_stop_target(None, 6))
        self.assertEqual(rt.pilot_stop_target(4099, rt.N_PILOT), 4099)

    def test_unreachable_target_fails_without_training_done(self):
        spec = self.spec("B6", run="short", stop_after_updates=6, max_epochs=1)          # 1 epoch = 4 updates
        with self.assertRaisesRegex(RuntimeError, "before the stop target"):
            self.train_arm(spec)
        self.assertFalse(os.path.exists(os.path.join(spec.run_dir, "TRAINING_DONE")))

    def test_required_world_size_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "world size 1 != 4"):
            self.train_arm(self.spec("B6", run="ws", stop_after_updates=6, required_world_size=4))


class PilotConfigGateTests(TrainerBase):
    def _main(self, *extra, config=PILOT_CFG, arm="B6"):
        return trainer.main(["--config", config, "--arm", arm, "--bindings", "x", "--output-dir",
                             os.path.join(self.tmp.name, "o"), *extra])

    def _copy(self, name, edit):
        cfg = json.load(open(PILOT_CFG))
        edit(cfg)
        path = os.path.join(self.tmp.name, name)
        json.dump(cfg, open(path, "w"), indent=2)
        return path

    def test_pilot_config_is_gated_and_pins_the_pilot(self):
        cfg = rt.load_b6_config(PILOT_CFG)
        self.assertEqual(cfg["gate_status"], rt.GATED_STATUS)
        self.assertEqual(cfg["wp5_pilot"], {"stop_after_updates": 4099, "arms": ["B6", "C2"], "world_size": 4,
                                            "seed": 123, "aug_seed": 20261037})
        self.assertEqual(rt.resolve_wp5_pilot(cfg, "C2", seed=123, stop_after_updates=None, allow_ungated=False),
                         {"stop_after_updates": 4099, "world_size": 4})
        base = json.load(open(BASE_CFG))
        for key, value in base.items():                    # the pilot config is the production recipe plus the block
            if key not in ("name", "notes", "arms"):
                self.assertEqual(cfg[key], value, key)

    def test_pilot_config_refuses_allow_ungated(self):
        with self.assertRaisesRegex(ValueError, "REFUSED.*--allow-ungated"):
            self._main("--allow-ungated")

    def test_pilot_config_refuses_other_arms_targets_and_ungated_status(self):
        with self.assertRaisesRegex(ValueError, "REFUSED.*arm C4"):
            self._main(arm="C4")
        with self.assertRaisesRegex(ValueError, "REFUSED.*4000"):
            self._main("--stop-after-updates", "4000")
        ungated = self._copy("ungated.json", lambda c: c.update(gate_status=rt.GATE_STATUS))
        with self.assertRaisesRegex(ValueError, "wp5_pilot.*GATED"):
            rt.load_b6_config(ungated)
        for key, bad in (("stop_after_updates", 4000), ("world_size", 2), ("seed", 456), ("aug_seed", 1),
                         ("arms", ["B6", "C3"])):
            path = self._copy(f"bad_{key}.json", lambda c, k=key, b=bad: c["wp5_pilot"].update({k: b}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"wp5_pilot {key}"):
                rt.load_b6_config(path)

    def test_valid_pilot_invocation_passes_the_gate_and_reaches_the_inputs(self):
        with patch.object(trainer, "_wp3", side_effect=RuntimeError("reached WP3 inputs")):
            with self.assertRaisesRegex(RuntimeError, "reached WP3 inputs"):
                self._main()
            with self.assertRaisesRegex(RuntimeError, "reached WP3 inputs"):
                self._main("--stop-after-updates", "4099", arm="C2")

    def test_stop_flag_without_pilot_config_still_needs_the_gate(self):
        ungated = self._copy("plain_ungated.json", lambda c: (c.pop("wp5_pilot"), c.pop("gate_evidence"),
                                                              c.update(gate_status=rt.GATE_STATUS)))
        with self.assertRaisesRegex(SystemExit, "UNGATED"):
            self._main("--stop-after-updates", "4099", config=ungated)
        with self.assertRaisesRegex(ValueError, "must equal N_pilot"):
            self._main("--stop-after-updates", "100", config=BASE_CFG)


class PilotLauncherTests(TrainerBase):
    SBATCH = os.path.join(ROOT, "revision", "scripts", "b6", "wp5_pilot.sbatch")

    def test_launcher_runs_the_gated_pilot_on_four_gpus_without_the_ungated_switch(self):
        text = open(self.SBATCH).read()
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        self.assertFalse([line for line in code if "--allow-ungated" in line])
        for needle in ("#SBATCH --gpus-per-node=4", "configs/b6_400m_win_v3_wp5_pilot.json", "N_PILOT=4099",
                       "--stop-after-updates $N_PILOT", "--seed 123", "--resume auto", "--num_processes=4"):
            self.assertIn(needle, text)

    def test_launcher_refuses_the_real_data_placeholder_and_other_arms(self):
        import subprocess
        env = dict(os.environ, ARM="B6")
        env.pop("BINDINGS", None)
        out = subprocess.run(["bash", self.SBATCH], env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 3)
        self.assertIn("placeholder", out.stderr)
        out = subprocess.run(["bash", self.SBATCH], env=dict(env, ARM="C4"), capture_output=True, text=True)
        self.assertEqual(out.returncode, 3)
        self.assertIn("B6 and C2 only", out.stderr)


class ProductionGateTests(TrainerBase):
    """Author decision 15 (2026-09-26): the production config is GATED on the recorded CP2 memory bench (A51)."""

    def test_production_config_is_gated_with_recorded_bench_evidence(self):
        cfg = rt.load_b6_config(BASE_CFG)
        self.assertEqual(cfg["gate_status"], rt.GATED_STATUS)
        ev = cfg["gate_evidence"]
        self.assertEqual(ev["memory_bench_job"], "3208285")
        self.assertIn(ev["memory_bench_world_size"], (2, 4))
        self.assertLessEqual(ev["memory_bench_peak_reserved_gb"], 100)
        self.assertEqual(ev["architecture_sha256"], rt.FROZEN_ARCHITECTURE_SHA256)

    def test_gated_status_without_valid_evidence_is_refused(self):
        def edited(name, edit):
            cfg = json.load(open(BASE_CFG))
            edit(cfg)
            path = os.path.join(self.tmp.name, name)
            json.dump(cfg, open(path, "w"), indent=2)
            return path
        cases = {"missing": lambda c: c.pop("gate_evidence"),
                 "one_rank": lambda c: c["gate_evidence"].update(memory_bench_world_size=1),
                 "over": lambda c: c["gate_evidence"].update(memory_bench_peak_reserved_gb=100.5),
                 "arch": lambda c: c["gate_evidence"].update(architecture_sha256="0" * 64),
                 "report": lambda c: c["gate_evidence"].update(memory_bench_report_sha256="short"),
                 "fields": lambda c: c["gate_evidence"].update(certified_gated_fields_sha256="0" * 64)}
        for name, edit in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, "gate_evidence"):
                rt.load_b6_config(edited(f"{name}.json", edit))


    def test_n_pilot_amendment_keeps_the_gate_but_memory_fields_break_it(self):
        """Decision 18 moved n_pilot 4,142 -> 4,099 after the CP2 bench; n_pilot (like names, notes, arms and the
        pilot block) is outside the certified fields, every memory-relevant field is inside them."""
        for path in (BASE_CFG, PILOT_CFG):
            cfg = rt.load_b6_config(path)
            self.assertEqual(cfg["n_pilot"], 4099)
            for n in (4142, 4099):                                        # the certified value and the amendment
                rt.check_gate_evidence(dict(cfg, n_pilot=n), path)
            memory_edits = {"effective_batch_size": 192, "max_encoder_seqlen": 262144, "decoder_layers": 13,
                            "window_tiers": [30720, 61440, 262144], "genes_per_window_cap": 192,
                            "checkpointing": "none", "max_label_tokens": 16384}
            for key, value in memory_edits.items():
                with self.subTest(config=os.path.basename(path), key=key), \
                        self.assertRaisesRegex(ValueError, "certified_gated_fields_sha256"):
                    rt.check_gate_evidence(dict(cfg, **{key: value}), path)

    def test_certified_fields_digest_is_that_of_the_bench_config_bytes(self):
        import subprocess
        from transgenic.training import b5_runtime
        try:
            blob = subprocess.run(["git", "-C", ROOT, "show", "ce22b2f:configs/b6_400m_win_v3.json"],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git history with ce22b2f unavailable")
        ev = json.load(open(BASE_CFG))["gate_evidence"]
        self.assertEqual(hashlib.sha256(blob).hexdigest(), ev["certified_config_sha256"])
        path = os.path.join(self.tmp.name, "certified.json")
        open(path, "wb").write(blob)
        self.assertEqual(rt.gated_fields_sha256(b5_runtime.load_b5_config(path)), ev["certified_gated_fields_sha256"])


# ---- two CPU ranks over gloo (the CP4 harness of tests/test_b6_checkpoint_io.py) --------------------------------

def _rank_worker(rank, root, link):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from tests.test_b6_trainer import TrainerBase, syn, trainer
    dist.init_process_group("gloo", init_method=(Path(root) / f"rendezvous{link}").resolve().as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    acc = Accelerator(cpu=True, mixed_precision="no", kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=60))])
    fixture = TrainerBase()
    fixture.setUp()
    # 8 rows over 2 ranks = 4 per rank per epoch; accumulation 4 -> 2 per rank -> 2 updates per epoch, 3 epochs = 6.
    spec = fixture.spec("C2", run_dir=str(Path(root) / "run"), accumulation_steps=4, max_epochs=3, n_pilot=3,
                        stop_after_updates=3, required_world_size=2,
                        stop_at_step=1 if link == 1 else None, resume=None if link == 1 else "auto")
    ds = syn.SyntheticB6Dataset(8, seed=1)
    ev = syn.SyntheticB6Dataset(4, seed=2, first_rn=5000)
    record = {}
    try:
        result = trainer.run_b6(spec, fixture.build(), ds, ev, collate_fn=syn.synthetic_collate,
                                train_sampler=syn.sampler_for(ds), eval_sampler=syn.eval_keys(ev), accelerator=acc)
        record = {k: result.get(k) for k in ("done", "preempted", "global_step", "reason")}
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    total = torch.tensor(1)
    dist.all_reduce(total)
    record["collective_after"] = int(total) == 2
    Path(root, f"link{link}_rank{rank}.json").write_text(json.dumps(record))
    fixture.doCleanups()
    dist.destroy_process_group()


def test_two_rank_pilot_stops_on_every_rank_at_n_across_a_restart(tmp_path):
    for link in (1, 2):
        mp.start_processes(_rank_worker, args=(str(tmp_path), link), nprocs=2, join=True, start_method="spawn")
    records = {(l, r): json.loads((tmp_path / f"link{l}_rank{r}.json").read_text()) for l in (1, 2) for r in (0, 1)}
    for rank in (0, 1):
        assert records[(1, rank)] == {"done": None, "preempted": True, "global_step": 1, "reason": None,
                                      "collective_after": True}, records
        assert records[(2, rank)] == {"done": True, "preempted": None, "global_step": 3,
                                      "reason": "stop_after_updates", "collective_after": True}, records
    run = tmp_path / "run"
    assert (run / "TRAINING_DONE").exists()
    step = run / "step_00003"
    assert (step / rt.PILOT_FINAL_MARKER).exists()
    meta = json.loads((step / "meta.json").read_text())
    assert (meta["pilot_final"], meta["global_step"], meta["stop_after_updates"]) == (True, 3, 3)
    assert sorted(p.name for p in run.iterdir() if p.name.startswith("step_")) == ["step_00003"]
    ident = json.loads((run / "run_config_b6.json").read_text())
    assert (ident["world_size"], ident["stop_after_updates"]) == (2, 3)
    shutil.rmtree(run)
