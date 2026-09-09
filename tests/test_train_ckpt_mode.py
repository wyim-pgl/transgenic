"""DDP correctness of the B5 trainer (Codex review 2026-09-09).

1. Gradient checkpointing defaults to the non-reentrant form; reentrant is refused under DDP
   (job 3117555: "Expected to mark a variable ready only once").
2. Checkpoint directories are owned by rank 0 while accelerator.save_state runs on every rank.
3. The validation loss is reduced across ranks before the early-stopping decision.

The trainer imports torch at module load, so 2 and 3 are checked on the source text; 1 is a pure function
in b5_runtime.py and is exercised directly.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_HyenaTransgenic.py"


def _load(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def rt():
    return _load(ROOT / "src" / "transgenic" / "training" / "b5_runtime.py", "b5_runtime_ckpt")


# ---- 1. checkpointing form -------------------------------------------------------------------

def test_nonreentrant_is_the_default_on_one_gpu_and_under_ddp(rt):
    assert rt.resolve_checkpointing({}, 1) == "nonreentrant"
    assert rt.resolve_checkpointing({}, 2) == "nonreentrant"
    assert rt.resolve_checkpointing({}, 4) == "nonreentrant"


def test_retired_opt_in_is_a_no_op(rt):
    # the 2026-09-09 experiment switch must not break old launch lines, and must not change the outcome
    assert rt.resolve_checkpointing({"TRANSGENIC_CKPT_NONREENTRANT": "1"}, 1) == "nonreentrant"
    assert rt.resolve_checkpointing({"TRANSGENIC_CKPT_NONREENTRANT": "1"}, 2) == "nonreentrant"


def test_reentrant_is_allowed_on_one_gpu_only(rt):
    assert rt.resolve_checkpointing({"TRANSGENIC_CKPT_REENTRANT": "1"}, 1) == "reentrant"
    with pytest.raises(RuntimeError, match="mark a variable ready only once"):
        rt.resolve_checkpointing({"TRANSGENIC_CKPT_REENTRANT": "1"}, 2)


def test_no_grad_ckpt_experiment_switch_wins(rt):
    assert rt.resolve_checkpointing({"TRANSGENIC_NO_GRAD_CKPT": "1"}, 1) == "off"
    assert rt.resolve_checkpointing({"TRANSGENIC_NO_GRAD_CKPT": "1", "TRANSGENIC_CKPT_REENTRANT": "1"}, 4) == "off"


def test_trainer_uses_the_resolver_and_has_no_bare_enable():
    src = TRAINER.read_text()
    assert "resolve_checkpointing(os.environ, world_size)" in src
    assert "model.gradient_checkpointing_enable()" not in src, "a bare enable() is the reentrant default"
    assert 'gradient_checkpointing_kwargs={"use_reentrant": False}' in src


# ---- 2. checkpoint directory ownership -------------------------------------------------------

def _func_source(name: str) -> str:
    tree = ast.parse(TRAINER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(TRAINER.read_text(), node)
    pytest.fail(f"{name} not found in the trainer")


@pytest.mark.parametrize("fn", ["_save_latest_b5", "_save_epoch_b5", "_save_state"])
def test_directory_ops_are_main_process_only_and_save_state_is_collective(fn):
    src = _func_source(fn)
    tree = ast.parse(src)
    guarded, collective = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "is_main":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    guarded.append(ast.unparse(inner.func) if hasattr(ast, "unparse") else ast.dump(inner.func))
    body_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for c in body_calls:
        f = c.func
        if isinstance(f, ast.Attribute) and f.attr == "save_state":
            collective.append(c)
    assert collective, f"{fn} must call accelerator.save_state on every rank"
    # save_state must not sit inside an `if is_main:` block
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "is_main":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "save_state":
                    pytest.fail(f"{fn}: accelerator.save_state is gated on is_main; Accelerate writes per-rank RNG state")
    # every filesystem mutation is inside an `if is_main:` block
    fs_ops = {"rmtree", "makedirs", "rename", "begin_epoch", "finish_epoch", "write_state", "_write_json", "save_model"}
    unguarded = []
    guarded_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "is_main":
            for inner in ast.walk(node):
                guarded_nodes.add(id(inner))
    for c in body_calls:
        name = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
        if name in fs_ops and id(c) not in guarded_nodes:
            unguarded.append(name)
    assert not unguarded, f"{fn}: filesystem ops outside `if is_main:` -> {unguarded}"
    assert src.count("barrier()") >= 2, f"{fn}: barriers around the collective save are missing"


def test_save_calls_are_not_gated_on_rank_at_the_call_site():
    """Every rank must reach the collective save; only the filesystem ops inside are gated."""
    src = TRAINER.read_text()
    for call in ("_save_latest_b5(epoch, step + 1, global_step)",
                 "_save_epoch_b5(epoch + 1, float(eval_epoch_loss), float(train_epoch_loss), is_best, global_step)"):
        assert call in src
        before = src.split(call)[0].rsplit("\n", 3)[-3:]
        assert not any("if is_main" in line for line in before), f"{call} is gated on is_main at the call site"


# ---- 3. validation loss reduction ------------------------------------------------------------

def test_eval_loss_is_reduced_across_ranks_before_the_stopper():
    src = TRAINER.read_text()
    i_reduce = src.index('accelerator.reduce(eval_loss, reduction="sum")')
    i_count = src.index("accelerator.reduce(torch.tensor(float(len(eval_dl))")
    i_stop = src.index("stopper.update(epoch + 1, float(eval_epoch_loss))")
    assert i_reduce < i_stop and i_count < i_stop
    assert "eval_epoch_loss = eval_loss_sum / eval_batches" in src
    assert "eval_epoch_loss = eval_loss / len(eval_dl)" not in src
