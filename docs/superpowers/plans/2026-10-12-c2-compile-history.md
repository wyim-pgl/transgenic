# C2 synchronized compile history implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Persist a validated global compile-event history containing every rank's real companion input and bind its numerical checkpoint position.

**Architecture:** A focused module joins each forward's per-rank graph/backend activity deltas, captures actual input/RNG/context only on a global event, and serializes a strict checkpoint-bound JSON artifact. Safe codecs and validation separate local I/O from distributed synchronization.

**Tech Stack:** Python, PyTorch distributed Gloo/NCCL, Accelerate, unittest.

**Spec:** `docs/superpowers/specs/2026-10-12-c2-compile-history-design.md`

## Global Constraints

- Work only in `/tmp/b6-c2-recompile`; no main merge, source deployment or GPU job.
- Every rank contributes its actual companion descriptor at the same global slot.
- Rank-specific replay and fabricated padding inputs are forbidden.
- RNG serialization is lossless explicit JSON/base64, never pickle.
- Missing/tampered files, world, checkpoint, identity, fingerprint, rank, context or input mismatches fail closed.
- Local errors propagate to every rank before any subsequent model collective.
- Reconstructed legacy entries require explicit provenance and are not observed history.
- Trainer wiring and GPU proof are separate follow-on work; this unit alone cannot launch production.

### Task 1: CompileHistory journal and corruption/distributed regressions

**Files:** create `src/transgenic/training/b6_compile_history.py` and `tests/test_b6_compile_history.py`.

**Interfaces:** consume the RNG dict keys python/numpy/torch_cpu/torch_cuda, an Accelerator, verified identity and fingerprint; produce the exact interfaces and JSON schema in the spec. No edits to the separate state module or trainer.

- [x] Write real single-rank tests for input signature checks, lossless RNG codec and persisted history binding. Start with the missing-module failure; confirm corrupt bytes and recomputed-SHA malformed data both fail after implementation.

```python
def test_tensor_value_change_is_rejected(self):
    inputs = {'labels': torch.tensor([[1, 4, 2]])}
    expected = input_signature(inputs)
    inputs['labels'][0, 1] = 5
    self.assertNotEqual(expected, input_signature(inputs))
```

- [x] RED: run `PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history -v` and record the expected missing implementation failure.
- [x] Implement the specified codecs, validation and persistence. Use canonical `json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)` bytes and SHA256; validate decoded contents separately from the digest. Copy incoming dictionaries through validated serialization so later caller mutation cannot change recorded history.
- [x] Add actual two-process Gloo/file-rendezvous tests: rank1 backend-activity-only compile must record both companions; rank1-only compile must save rank0's different input too; rank1 hash failure must reach both ranks. Bound worker join time and terminate failed test workers only. Implement synchronized local preflight and gathering until these regressions pass.
- [x] GREEN: run focused tests once complete and report pass/skip totals, commands, RED/GREEN evidence. Self-review the distinction between reconstructed and observed origins, then commit only owned module/tests and plan/spec. Controller sends an independent reviewer the committed diff.
