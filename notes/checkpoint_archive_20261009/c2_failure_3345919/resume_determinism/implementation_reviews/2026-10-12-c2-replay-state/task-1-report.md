# Task 1 report: C2 replay state preservation

Status: DONE; implementation and focused/integration regressions verified.
Baseline: `d7b1a37`. Worktree: `/tmp/b6-c2-recompile`.

## Implementation

Created `src/transgenic/training/b6_replay_state.py` with
`preserve_training_state(model)`, `capture_rng()`, and `restore_rng(state)`.
The original, unwrapped model supplies the boundary. Private snapshot records
retain complete per-module parameter/buffer registries, including None entries,
nonpersistent registration sets, original object references and heterogeneous
module training modes. All tensor value copies are detached CPU copies;
parameter/buffer/gradient objects are deduplicated by identity. Original pending
gradient objects and values are retained, including restoring None.

Restoration copies back when original metadata matches. If a replay changed a
retained tensor's shape/dtype/device/stride, a correctly strided tensor is
allocated on its original device and attached to the original object's data.
Original parameter object identity, tied registrations, buffer aliases, and
requires_grad flags survive. The complete registries are restored, removing new
registrations and replacing deleted or substituted entries. Module modes are
assigned directly, avoiding recursive train() overwriting child mode differences.

The outer finally restores model state after normal and BaseException exits;
a nested finally restores RNG even if model restoration itself fails. The helper
never imports/calls optimizer, scheduler, counters or compiler reset APIs.
Topology and arbitrary Python attributes are outside the spec.

Unsupported layouts fail at entry before replay: non-strided, quantized, meta,
Tensor subclasses outside Tensor/Parameter, and layouts for which PyTorch reports
internal overlap or cannot certify absence of overlap. Ordinary dense contiguous
and transposed strided tensors are supported. This conservative policy needs to
be checked on the actual GPU model before a production proof.

## RNG interface for the history journal

`capture_rng()` returns a plain dict, without custom classes:

- `python`: `random.getstate()` tuple (basic numbers/None/tuples).
- `numpy`: `np.random.get_state()` tuple with a copied ndarray of state words.
- `torch_cpu`: detached CPU ByteTensor state clone.
- `torch_cuda`: list of CPU ByteTensor state clones for visible CUDA devices if
  CUDA is already initialized; otherwise None.

`restore_rng()` accepts that shape. Persistence, JSON/base64 encoding and schema
validation belong to the history journal. A CPU-only entry does not initialize
CUDA. State comparisons and expected draws in tests use independent library RNG
snapshots, rather than the helper under test.

## TDD evidence

Before production code existed, created the minimal parameter-mutation test.
Command:

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_replay_state -v
```

RED result:

```text
ImportError: cannot import name 'b6_replay_state' from 'transgenic.training'
Ran 1 test in 0.001s
FAILED (errors=1)
```

Expanded the same test file with the complete contract before creating the
production module, and reran the same command. RED result:

```text
Ran 10 tests in 0.010s
FAILED (errors=10)
```

All ten errors were the expected missing-module import. Implemented the helper
and ran the same command. GREEN result:

```text
Ran 10 tests in 3.634s
OK
```

There were zero skips and no warnings. The local interpreter reports CUDA
available, so the CUDA regression actually ran, alongside the CPU path that
asserts CUDA stays uninitialized. This local CUDA coverage does not replace the
separately required NGC/GPU proof for C2 replay integration.

## Regression coverage

1. Literal parameter values and original object restored after mutation.
2. Full parameter/buffer registration sets, replacements, deletions, new entries,
   None entries, nonpersistent flags, tied parameter/buffer references, pending
   gradient object/value/None and heterogeneous modes restored.
3. Same complete fixture restored on a custom BaseException; exact exception
   instance propagates and all independently saved CPU RNG states match.
4. Changed tensor shape/dtype metadata recovers original parameter, buffer and
   gradient objects and values.
5. Changed noncontiguous buffer metadata recovers original transposed stride.
6. Sparse parameter, buffer and gradient layouts rejected before body entry.
7. Overlapping broadcast buffer rejected before body entry.
8. Python/NumPy/CPU torch helpers restore independently saved expected draws.
9. CPU context does not initialize CUDA.
10. Every visible initialized CUDA RNG and GPU parameter values restored, with
    independently saved CUDA draws.

## Integration baseline

Command:

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_recovery tests.test_b6_trainer tests.test_b6_replay_state -v > .superpowers/sdd/2026-10-12-c2-replay-state/task-1-green.log 2>&1
```

Result (exit0):

```text
Ran 44 tests in 149.160s
OK (skipped=2)
```

Totals: compile recovery6 passed; trainer27 passed +1 skipped for absent pyarrow;
replay state9 passed +1 skipped because the preceding trainer tests initialized
CUDA. The standalone focused run already passed that CPU noninitialization test.
Combined total42 passed,2 skipped,0 failures/errors. Full output was read and is
captured in `task-1-green.log` beside this report. Existing trainer tests emit
shared-tensor saving warnings, unclosed-file ResourceWarnings, an ImportWarning
and an expected argparse error from its rejection test; no unrelated production
code or tests were changed.

## Self-review and scope

Reviewed both new-file diffs against every state-boundary requirement. No
unresolved correctness issue found; rejected layouts are explicit rather than
silently converted. `git diff --check` returned exit0. Tests contain no mocks and
assert public model/RNG behavior from literal fixtures and independently captured
states. The snapshot CPU representation prevents a duplicate model resident on
GPU; the original tensor objects remain retained until restoration.

Scoped commit includes only the new module/test and the already-created replay
state plan/spec. The separately owned compile-history plan/spec remain outside
this commit. Controller owns independent review, graph refresh, the journal and
integration, NGC verification, scientific equality gate and production jobs.
No production C2 run was started or authorized by this unit alone.

## Commit and review handoff

Commit: `3d59cf2` — `feat(training): preserve model and RNG state during compile replay`.
Changed four files (module, test, replay-state plan and spec),509 inserted lines.
Staged diff whitespace verification passed before commit. Independent review is
requested from the controller; this implementer spawned no agents or jobs.
Graph context consulted: graft map; graft ask for replay/RNG model state;
graft skeleton for compile recovery tests; identified original diagnostic
`first_forward.py:16-40`. Estimated graft savings across these calls:1,491,005 tokens.

## Fix round 1: independent review findings

Read the complete independent review and verified the two concrete accepted
mutation defects with new regressions before modifying production code.

1. Delete a registered parameter/buffer, then assign a plain Tensor under the
   same name: ordinary attributes in __dict__ shadow the restored registries.
   Restore now removes only attributes under original parameter/buffer names.
   The regression checks ordinary lookup returns original objects and values,
   while an unrelated newly added body attribute remains unchanged.
2. Replace a registered slice's data, with or without replacing its whole-storage
   buffer's data: the prior fallback lost shared storage and offset. Snapshots
   now retain each original tensor's detached data view, in addition to the
   existing CPU value copy. This view retains storage and independent original
   metadata without allocating a second tensor storage or GPU model. Restoration
   reattaches the original data view if shape/stride/dtype/device/offset/storage
   identity differs, then copies CPU values back. The original allocation,
   distinct-object shared-storage relationships and nonzero offsets survive.
   Unchanged tensors still use copy-back without reattaching data. The tests
   verify offset2 and observable subsequent slice writes updating whole[2].

The retained view supersedes the original implementation/report's
empty_strided-allocation fallback. Exact storage identity uses PyTorch storage
_cdata; this is local in-memory snapshot metadata, not part of RNG persistence.
No additional previously supported strided views were rejected. Entry rejection
for sparse/quantized/meta/subclass/overlapping-or-unknown layouts is unchanged.

TDD RED command (before either fix):

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_replay_state -v > .superpowers/sdd/2026-10-12-c2-replay-state/task-1-fix-1-red.log 2>&1
```

Result exit1, expected failures:

```text
AssertionError: tensor([-3.]) is not Parameter containing: ...
AssertionError: 0 != 2  # both whole-buffer mutation subcases
Ran 12 tests in 1.942s
FAILED (failures=3)
```

TDD GREEN command (after both fixes):

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_replay_state -v > .superpowers/sdd/2026-10-12-c2-replay-state/task-1-fix-1-green.log 2>&1
```

Result exit0,12 passed,0 skipped, no warnings:

```text
Ran 12 tests in 4.002s
OK
```

Read full RED/GREEN logs and final scoped diff. Self-review found no additional
issue. Only this module and its regression tests changed; RNG interfaces, model
modes, gradient handling, compiler/optimizer boundaries and integration code are
unchanged. Per controller instruction, did not repeat the whole trainer suite.
Controller owns scoped independent re-review. No agents or remote jobs spawned.

Fix round1 commit: `91c2737` — `fix(training): restore replay registry lookup and storage views`.
Scoped staged whitespace check passed; commit includes module/test only.
