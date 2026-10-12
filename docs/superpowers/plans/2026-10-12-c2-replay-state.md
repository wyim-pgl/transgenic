# C2 replay state preservation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Restore the complete mutable training state after compile-history priming, including failure paths.

**Architecture:** A focused context manager takes CPU snapshots of the original model registries, values, modes, gradients, and RNG. Replay runs within it; `finally` restores the original objects and state without touching compiler caches or optimizer/scheduler state.

**Tech Stack:** Python, PyTorch, NumPy, unittest; local interpreter `/home/wyim/miniconda3/envs/pdb/bin/python`.

**Spec:** `docs/superpowers/specs/2026-10-12-c2-replay-state-design.md`

## Global Constraints

- Implementation worktree `/tmp/b6-c2-recompile`; baseline d7b1a37; do not merge main.
- Preserve all registered parameters and their object identities, pending gradients (including None), all modules' complete buffer registries (including None and nonpersistent entries), and per-module training flags.
- Restore Python, NumPy, CPU torch and every already-initialized CUDA RNG state.
- Snapshot to CPU; do not initialize CUDA on a CPU-only path.
- Never update optimizer, scheduler or step counters; never reset compiler state.
- This unit alone does not authorize a C2 production run.

### Task 1: Implement and test the state-preservation boundary

**Files:**
- Create: `src/transgenic/training/b6_replay_state.py`
- Create: `tests/test_b6_replay_state.py`
- Read reference: main workspace `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/first_forward.py:16-40` (diagnostic snapshot/restore).

**Interfaces:**
- Consumes: original `torch.nn.Module` and initialized RNG families.
- Produces: `preserve_training_state(model)` context manager, `capture_rng()`, `restore_rng(state)`.

- [ ] Write regression tests before production code. A real module with a tied parameter, mixed child modes, a None buffer, nonpersistent buffer and pending gradient enters the context. During the body mutate values, replace a parameter, delete/replace buffers, register a new buffer and consume all RNG families. After exit require original object identities, literal values and registration sets. Repeat restoration under a custom BaseException. Use this minimal failing API test as the start, and extend the same fixture for each contract:

```python
def test_parameter_mutation_is_rolled_back():
    import torch
    from transgenic.training import b6_replay_state
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3)
    original = model.weight
    with b6_replay_state.preserve_training_state(model):
        with torch.no_grad():
            model.weight.fill_(9)
    assert model.weight is original
    assert torch.equal(model.weight, torch.tensor([[3., 3.]]))
```

- [ ] RED: run `PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s tests -p 'test_b6_replay_state.py' -v`. Record the expected missing implementation failure in the report.
- [ ] Implement the context with entry snapshot and unconditional restoration:

```python
@contextmanager
def preserve_training_state(model):
    snapshot = capture_model(model)
    rng = capture_rng()
    try:
        yield
    finally:
        restore_model(snapshot)
        restore_rng(rng)
```

Implement private helpers `capture_model(model)` and `restore_model(snapshot)` in the module to fulfill the registry/value contract. Deduplicate
objects by identity and retain references to originals. Use copy-back where shape,
dtype and layout are unchanged; handle replaced registrations from the stored
registry. Fail before replay if a tensor layout cannot be restored faithfully.

- [ ] GREEN: run the focused regression tests. Then run `tests/test_b6_compile_recovery.py` and `tests/test_b6_trainer.py` once to establish the integration baseline. Record exact commands and pass/skip totals.
- [ ] Self-review and commit only the new module/tests and this plan/spec. Request independent review via the controller. GPU state verification and full-history trainer integration follow in the next execution unit.
