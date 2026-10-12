"""Failures at the pilot's last update must remain failures and allow an exact retry."""
import json
import os
from pathlib import Path
import subprocess
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

from tests.test_b6_trainer import TrainerBase, rt


class PilotFinalizationTests(TrainerBase):
    def test_short_final_resume_verifies_compile_before_publishing_success(self):
        import torch._dynamo as dynamo
        dynamo.reset()
        self.addCleanup(dynamo.reset)
        first = self.spec("C2", n_pilot=4, stop_after_updates=4,
                          compile_backend="eager", stop_at_step=2)
        self.assertTrue(self.train_arm(first)["preempted"])
        dynamo.reset()  # A resumed job starts in a fresh process and replays its compiler history.
        resumed = self.spec("C2", n_pilot=4, stop_after_updates=4,
                            compile_backend="eager", resume="auto")
        with patch.object(rt, "check_compiled", side_effect=RuntimeError("injected compile verification")):
            with self.assertRaisesRegex(RuntimeError, "injected compile verification"):
                self.train_arm(resumed)
        self.assertFalse(Path(resumed.run_dir, "TRAINING_DONE").exists())
        self.assertFalse(Path(rt.step_checkpoint_dir(resumed.run_dir, 4), rt.PILOT_FINAL_MARKER).exists())

    def published_but_not_done(self, run="run"):
        spec = self.spec("C2", run=run, stop_after_updates=6, save_every_n_steps=1)
        with patch.object(rt.CheckpointLayout, "mark_done", side_effect=OSError("injected publication crash")):
            with self.assertRaisesRegex(RuntimeError, "injected publication crash"):
                self.train_arm(spec)
        step = Path(rt.step_checkpoint_dir(spec.run_dir, 6))
        self.assertTrue((step / rt.PILOT_FINAL_MARKER).exists())
        self.assertFalse(Path(spec.run_dir, "TRAINING_DONE").exists())
        return spec, step

    def test_retry_after_publication_preserves_the_exact_published_checkpoint(self):
        spec, step = self.published_but_not_done()
        before = {p.name: (p.read_bytes(), p.stat().st_ino) for p in step.iterdir()}
        result = self.train_arm(self.spec("C2", stop_after_updates=6, save_every_n_steps=1, resume="auto"))
        self.assertEqual((result["done"], result["global_step"]), (True, 6))
        self.assertTrue(Path(spec.run_dir, "TRAINING_DONE").exists())
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_ino) for p in step.iterdir()})

    def test_retry_refuses_corrupt_published_weights_without_replacing_them(self):
        spec, step = self.published_but_not_done()
        weights = step / "model.safetensors"
        with weights.open("ab") as stream:
            stream.write(b"corruption")
        damaged = weights.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "published step checkpoint"):
            self.train_arm(self.spec("C2", stop_after_updates=6, save_every_n_steps=1, resume="auto"))
        self.assertEqual(weights.read_bytes(), damaged)
        self.assertFalse(Path(spec.run_dir, "TRAINING_DONE").exists())

    def test_retry_refuses_a_published_checkpoint_with_a_different_target(self):
        spec, step = self.published_but_not_done()
        meta_path = step / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["stop_after_updates"] = 7
        meta_path.write_text(json.dumps(meta))
        before = meta_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "published step checkpoint"):
            self.train_arm(self.spec("C2", stop_after_updates=6, save_every_n_steps=1, resume="auto"))
        self.assertEqual(meta_path.read_bytes(), before)
        self.assertFalse(Path(spec.run_dir, "TRAINING_DONE").exists())


    def test_retry_refuses_foreign_identity_and_missing_or_invalid_markers(self):
        for defect in ("identity", "missing_marker", "invalid_marker"):
            with self.subTest(defect=defect):
                spec, step = self.published_but_not_done(run=defect)
                if defect == "identity":
                    path = step / rt.CHECKPOINT_IDENTITY
                    value = json.loads(path.read_text())
                    value["seed"] = 789
                    path.write_text(json.dumps(value))
                elif defect == "missing_marker":
                    (step / rt.PILOT_FINAL_MARKER).unlink()
                else:
                    (step / rt.PRUNE_EXEMPT_MARKER).write_text("unrelated checkpoint")
                before = {p.name: p.read_bytes() for p in step.iterdir()}
                with self.assertRaisesRegex(RuntimeError, "published step checkpoint"):
                    self.train_arm(self.spec("C2", run=defect, stop_after_updates=6,
                                             save_every_n_steps=1, resume="auto"))
                self.assertEqual(before, {p.name: p.read_bytes() for p in step.iterdir()})
                self.assertFalse(Path(spec.run_dir, "TRAINING_DONE").exists())


class PilotLauncherExitTests(TrainerBase):
    def test_failed_marker_takes_precedence_over_stale_done_at_entry(self):
        run = Path(self.tmp.name, "entry")
        run.mkdir()
        for name in ("TRAINING_DONE", "TRAINING_FAILED"):
            (run / name).touch()
        script = Path(__file__).resolve().parents[1] / "revision/scripts/b6/wp5_pilot.sbatch"
        entry = script.read_text().split('RUN=/work/runs/b6_wp5_pilot_$ARM\n', 1)[1].split('mkdir -p "$RUN_HOST"', 1)[0]
        result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + entry],
                                env=dict(os.environ, RUN_HOST=str(run), ARM="C2"), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def launcher_exit(self, rc, done=False, final=False):
        run = Path(self.tmp.name, "launcher")
        run.mkdir()
        if done:
            (run / "TRAINING_DONE").touch()
        if final:
            step = run / "step_00006"
            step.mkdir()
            (step / "PILOT_FINAL").touch()
            (step / "model.safetensors").write_bytes(b"fixture")
        script = Path(__file__).resolve().parents[1] / "revision/scripts/b6/wp5_pilot.sbatch"
        # Execute the real post-training branch, with the external trainer represented by its exit status.
        tail = script.read_text().split("\nSTEP=$RUN_HOST/", 1)[1]
        result = subprocess.run(["bash", "-c", "set -euo pipefail\nSTEP=$RUN_HOST/" + tail],
                                env=dict(os.environ, RUN_HOST=str(run), N_PILOT="6", ARM="C2", TRAIN_RC=str(rc)),
                                text=True, capture_output=True)
        return result, run

    def test_failed_trainer_is_rejected_even_when_both_success_markers_exist(self):
        result, run = self.launcher_exit(1, done=True, final=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((run / "TRAINING_FAILED").exists())

    def test_successful_completed_trainer_is_accepted(self):
        result, run = self.launcher_exit(0, done=True, final=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((run / "TRAINING_FAILED").exists())

    def test_successful_preemption_is_accepted_without_done(self):
        result, run = self.launcher_exit(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((run / "TRAINING_FAILED").exists())


def _failed_verification_rank(rank, root):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from tests.test_b6_trainer import syn, trainer

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(root) / "rendezvous").resolve().as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    fixture = TrainerBase()
    fixture.setUp()
    acc = Accelerator(cpu=True, mixed_precision="no", kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=60))])
    spec = fixture.spec("C2", run_dir=str(Path(root) / "run"), accumulation_steps=4, max_epochs=3,
                        n_pilot=1, stop_after_updates=1, required_world_size=2, compile_backend="eager")
    ds, ev = syn.SyntheticB6Dataset(8, seed=1), syn.SyntheticB6Dataset(4, seed=2, first_rn=5000)
    inject = (patch.object(rt, "check_compiled", side_effect=RuntimeError("rank1 verification failed"))
              if rank == 1 else nullcontext())
    record = {}
    try:
        with inject:
            trainer.run_b6(spec, fixture.build(), ds, ev, collate_fn=syn.synthetic_collate,
                           train_sampler=syn.sampler_for(ds), eval_sampler=syn.eval_keys(ev), accelerator=acc)
    except RuntimeError as exc:
        record["error"] = str(exc)
    total = torch.tensor(1)
    dist.all_reduce(total)
    record["collective_after"] = int(total) == 2
    Path(root, f"rank{rank}.json").write_text(json.dumps(record))
    fixture.doCleanups()
    dist.destroy_process_group()


class CollectiveFinalizationTests(TrainerBase):
    def test_non_main_compile_verification_failure_prevents_all_success_publication(self):
        import torch.multiprocessing as mp
        mp.start_processes(_failed_verification_rank, args=(self.tmp.name,), nprocs=2,
                           join=True, start_method="spawn")
        for rank in (0, 1):
            record = json.loads(Path(self.tmp.name, f"rank{rank}.json").read_text())
            self.assertIn("rank1 verification failed", record["error"])
            self.assertIs(record["collective_after"], True)
        run = Path(self.tmp.name, "run")
        self.assertFalse((run / "TRAINING_DONE").exists())
        self.assertFalse((run / "step_00001").exists())
