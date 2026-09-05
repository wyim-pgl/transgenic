"""#59: real Accelerate checkpoints, with restart in a fresh Python process.

Run on pgl-gpu with its transgenic Python; local torch-free suites skip these.
No model download or training database is needed.
"""
import ast
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
