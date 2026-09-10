"""#59: real Accelerate checkpoints, with restart in a fresh Python process.

Run on pgl-gpu with its transgenic Python; local torch-free suites skip these.
No model download or training database is needed.
"""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_trainer_enables_seedable_sampler_and_uses_epoch_batches():
    tree = ast.parse((ROOT / "train/train_HyenaTransgenic.py").read_text())
    configs = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == "DataLoaderConfiguration"]
    assert len(configs) == 1
    assert any(k.arg == "use_seedable_sampler" and isinstance(k.value, ast.Constant)
               and k.value.value is True for k in configs[0].keywords)
    accelerators = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "Accelerator"]
    assert any(k.arg == "dataloader_config" and k.value is configs[0]
               for n in accelerators for k in n.keywords)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "epoch_batches" for n in ast.walk(tree))


@pytest.mark.parametrize("epoch,batch_size,workers,seed,seedable", [
    (0, 1, 0, 123, True),
    (3, 1, 0, 456, True),
    (0, 4, 2, 456, True),
    (3, 4, 2, 123, True),
    (0, 1, 0, 123, False),  # negative control: reproduces the original reshuffle
])
def test_checkpoint_restart_preserves_samples(tmp_path, epoch, batch_size, workers, seed, seedable):
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    results = []
    for mode in ("original", "resume"):
        output = tmp_path / f"{mode}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path),
                        str(epoch), str(batch_size), str(workers), str(seed), str(int(seedable)),
                        str(output)], check=True, capture_output=True, text=True)
        results.append(json.loads(output.read_text()))
    original, resumed = results
    full = original["prefix"] + original["remaining"]
    assert sorted(full) == list(range(64))
    if not seedable:
        assert resumed["remaining"] != original["remaining"]
        assert sorted(original["prefix"] + resumed["remaining"]) != list(range(64))
        return
    assert resumed["remaining"] == original["remaining"]
    assert sorted(original["prefix"] + resumed["remaining"]) == list(range(64))
    assert resumed["steps"] == original["steps"]
    assert resumed["next_epoch"] == original["next_epoch"]
    assert resumed["next_epoch"] != full


def _worker():
    import importlib.util
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DataLoaderConfiguration
    from torch.utils.data import DataLoader

    spec = importlib.util.spec_from_file_location("b5_runtime", ROOT / "src/transgenic/training/b5_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    mode, directory, epoch, batch_size, workers, seed, seedable, output = sys.argv[1:]
    epoch, batch_size, workers, seed = map(int, (epoch, batch_size, workers, seed))
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(
        use_seedable_sampler=bool(int(seedable))))
    torch.manual_seed(seed)
    loader = accelerator.prepare(DataLoader(list(range(64)), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, persistent_workers=workers > 0))
    checkpoint = str(Path(directory) / "checkpoint")
    skip = 20 // batch_size
    result = {"prefix": [], "remaining": [], "steps": []}
    if mode == "resume":
        accelerator.load_state(checkpoint)
    for step, batch in runtime.epoch_batches(loader, epoch, skip if mode == "resume" else 0):
        values = batch.tolist()
        # Stand in for model/dropout RNG consumption between loader batches.
        torch.rand(37)
        if mode == "original" and step < skip:
            result["prefix"].extend(values)
            if step + 1 == skip:
                accelerator.save_state(checkpoint)
        else:
            result["remaining"].extend(values)
            result["steps"].append(step)
    result["next_epoch"] = [value for _, batch in runtime.epoch_batches(loader, epoch + 1)
                            for value in batch.tolist()]
    Path(output).write_text(json.dumps(result))


if __name__ == "__main__":
    _worker()


def test_epoch_checkpoint_carries_the_unfinished_accumulation_window():
    """2026-09-10: the epoch save records pending_micro and each rank's pending gradients; the resume path restores
    them and starts micro_done from pending_micro, so a resume from an epoch checkpoint updates at the same
    boundaries as uninterrupted training (198,828 rows -> 12 pending micro-batches on 1 GPU, 3 per rank on 4)."""
    src = (ROOT / "train/train_HyenaTransgenic.py").read_text()
    assert 'pending = micro_done % accumulation_steps' in src
    assert '"pending_micro": pending' in src
    assert '_save_pending_grads(accelerator.unwrap_model(model), tmp, accelerator.process_index)' in src
    assert 'pending_micro = int(meta.get("pending_micro", 0))' in src
    assert '_load_pending_grads(accelerator.unwrap_model(model), resume_from_checkpoint, accelerator.process_index)' in src
    assert 'micro_done = pending_micro' in src
    # the train-loss average divides by this epoch's micro-batches, not the cross-epoch accumulation counter
    assert 'seen = max(1, epoch_micro)' in src and 'seen = max(1, micro_done)' not in src


def test_pending_grads_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_mod", ROOT / "train/train_HyenaTransgenic.py")
    # importing the trainer needs transformers/accelerate; skip where they are absent
    pytest.importorskip("accelerate"); pytest.importorskip("transformers")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = torch.nn.Linear(3, 2)
    m.weight.grad = torch.ones_like(m.weight) * 0.5
    m.bias.grad = None
    assert mod._save_pending_grads(m, str(tmp_path), 0) == 1
    m2 = torch.nn.Linear(3, 2)
    assert mod._load_pending_grads(m2, str(tmp_path), 0) == 1
    assert torch.equal(m2.weight.grad, torch.ones_like(m2.weight) * 0.5) and m2.bias.grad is None
    assert mod._load_pending_grads(m2, str(tmp_path), 1) == 0
