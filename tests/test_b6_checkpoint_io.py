"""Real two-rank fault injection: checkpoint I/O failures must not desynchronise collectives."""
from datetime import timedelta
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_b6_ddp import _free_port


def _worker(rank, port, root, failure):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from tests.test_b6_trainer import TrainerBase, trainer, syn, rt
    device = os.environ.get("B6_CHECKPOINT_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    if device == "cuda":
        assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    acc = Accelerator(cpu=device == "cpu", mixed_precision="no", kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(backend="gloo" if device == "cpu" else "nccl", timeout=timedelta(seconds=30))])
    assert acc.num_processes == dist.get_world_size() == 2
    assert acc.process_index == rank
    fixture = TrainerBase()
    fixture.setUp()
    run = Path(root) / "run"
    spec = fixture.spec(run_dir=str(run), accumulation_steps=4, max_epochs=1,
                        n_pilot=1 if failure in ("publish", "preserve_step") else 10,
                        stop_at_step=1 if failure == "latest_state" else None)
    if failure == "preserve_step" and rank == 0:
        final = run / "step_00001"
        final.mkdir(parents=True)
        (final / "sentinel").write_text("prior checkpoint")
    dist.barrier()
    ds = syn.SyntheticB6Dataset(6, seed=1)
    ev = syn.SyntheticB6Dataset(4, seed=2, first_rn=5000)
    if failure in ("weights", "publish", "preserve_step"):
        owner, name, failing_rank = rt, "save_weights_deterministic", 0
    elif failure in ("state", "latest_state"):
        owner, name, failing_rank = acc, "save_state", 1
    elif failure == "identity":
        owner, name, failing_rank = rt, "record_identity", 0
    else:
        owner, name, failing_rank = rt, "save_pending_grads", 1
    original = getattr(owner, name)

    def failing_save(*args, **kwargs):
        if rank == failing_rank and failure != "preserve_step":
            raise OSError(122, "injected EDQUOT " + failure)
        return original(*args, **kwargs)

    record = {"error": None, "collective_after_failure": False}
    try:
        with patch.object(owner, name, side_effect=failing_save):
            trainer.run_b6(spec, fixture.build(), ds, ev, collate_fn=syn.synthetic_collate,
                           train_sampler=syn.sampler_for(ds), eval_sampler=syn.eval_keys(ev), accelerator=acc)
    except Exception as exc:
        record["error"] = str(exc)
    try:
        total = torch.tensor(1, device=acc.device)
        dist.all_reduce(total)
        record["collective_after_failure"] = int(total) == 2
    except Exception as exc:
        record["collective_error"] = str(exc)
    Path(root, f"rank{rank}.json").write_text(json.dumps(record))
    fixture.doCleanups()
    dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["weights", "state", "grads", "latest_state", "publish", "identity", "preserve_step"])
def test_checkpoint_failure_reaches_all_ranks_without_publication(tmp_path, failure):
    mp.start_processes(_worker, args=(_free_port(), str(tmp_path), failure), nprocs=2,
                       join=True, start_method="spawn")
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    expected = "refusing to replace published step checkpoint" if failure == "preserve_step" else "injected EDQUOT " + failure
    for record in records:
        assert expected in (record["error"] or ""), records
        assert record["collective_after_failure"], records
    if failure == "preserve_step":
        assert (tmp_path / "run/step_00001/sentinel").read_text() == "prior checkpoint"
    elif failure != "identity":
        stem = {"latest_state": "latest_state", "publish": "step_00001"}.get(failure, "epoch_01")
        assert (tmp_path / "run" / (stem + ".tmp")).is_dir()
        assert not (tmp_path / "run" / stem).exists()
    assert not (tmp_path / "run/TRAINING_DONE").exists()
