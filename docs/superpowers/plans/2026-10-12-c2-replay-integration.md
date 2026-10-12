# C2 compiler replay integration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make compiled checkpoints carry and replay synchronized history, with an explicit auditable bootstrap for original C2 checkpoint200.

**Architecture:** A helper coordinates data reconstruction, fingerprint validation and state-preserving replay around the prepared model. The trainer records every actual forward context and atomically binds the journal to all checkpoint writers. A strict legacy manifest allows only the known checkpoint200 candidate.

**Tech Stack:** Python, PyTorch/Inductor/DDP, Accelerate, unittest.

**Spec:** `docs/superpowers/specs/2026-10-12-c2-replay-integration-design.md`

## Global Constraints

- Worktree `/tmp/b6-c2-recompile`; no main merge, remote writes or GPU job submission.
- No rank-specific replay, invented companion inputs, optimizer/scheduler/step advancement or numerical-policy changes.
- Preserve parameter objects, complete buffer registry, pending grads, modes and all RNG; retain compiler history.
- All train/eval views and latest/epoch/fixed-step checkpoint histories are covered.
- Missing/tampered/foreign history or reconstructed inputs fail closed on every rank before unmatched collectives.
- Legacy is explicit reconstructed provenance, fixed original200/world4/C2/batch1 candidate slots1..15+3915.
- Existing source checkpoints and input evidence remain unchanged.
- CPU tests alone do not authorize production; exact real4GPU proof remains required.

### Task 1: Integrate recording, replay and legacy bootstrap

**Files:** create `src/transgenic/training/b6_compile_replay.py` and `tests/test_b6_replay_integration.py`; modify `train/train_HyenaTransgenic_B6.py:55-80,114-537,642-684` and `src/transgenic/training/b6_runtime.py:856-881`. Existing test updates only for changed intended compiled-checkpoint contract, with justification. Do not edit separately owned state/journal modules; request interface changes from controller.

**Consumes:** reviewed preserve_training_state/capture_rng/restore_rng helpers, CompileHistory constructor/observe/save/load/check_inputs and safe RNG codecs. **Produces:** trainer compiled-history behavior, explicit legacy CLI, helper API documented for controller's validation harness.

- [ ] Read current graph spans and both prerequisite module interfaces. Write focused tests that fail because fresh compiled checkpoints lack a history digest and a compiled legacy resume is currently accepted silently. Build on TrainerBase's actual synthetic model/dataset.
- [ ] RED: run `PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_replay_integration -v`; capture expected failure evidence.
- [ ] Implement helper and trainer lifecycle per spec. Checkpoint metadata call pattern is:

```python
position = {key: meta[key] for key in ('epoch', 'step', 'global_step', 'pending_micro')}
meta['compile_history_sha256'] = history.save(history_path, checkpoint=position)
```

Call this for main-rank local I/O inside the existing coordinated checkpoint
transaction, before its publication; no collectives inside checkpoint_io.
Replay reconstruction uses the prepared batch_sampler and collate_fn, preserving
the loader state it changes. Stage preflight validation before prepared-model
collectives; use the state context for rollback even on BaseException.

- [ ] Add real normal-resume/eval/pending-gradient/changed-input regressions and exact legacy-manifest rejections. Add a bounded two-rank replay integration/failure test. Implement only behavior required by these scientific contracts, recording chosen public helper interfaces in the report.
- [ ] GREEN: run focused integration tests and prerequisite modules, then existing tests.test_b6_trainer, tests.test_b6_compile_recovery, tests.test_b6_compile_static, tests.test_b6_pilot and tests.test_b6_finalization once. Inspect all failures/expected skips; report commands and output. Self-review and scoped commit. Controller runs independent review before staging NGC proof.
