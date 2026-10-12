# Task 1 implementation report

Status: DONE. All owned implementation and unittest work complete; controller owns independent review and trainer integration.

## Implemented

Created `src/transgenic/training/b6_compile_history.py` and `tests/test_b6_compile_history.py` only; updated this task's plan/spec with controller clarifications. No changes to `b6_replay_state.py`, trainer, remote artifacts or GPU execution.

- Dense tensor input signatures bind SHA256 of contiguous CPU bytes, original dtype/shape/stride/requires_grad and `_dynamo_dynamic_indices`. Optional None and finite JSON scalars bind explicit scalar type/value/digest. Unsupported layouts/values fail closed; bfloat16 hashing uses uint8 views.
- Lossless explicit JSON/base64 RNG codecs preserve Python tuple state and Gaussian cache, NumPy MT19937 array/cache, CPU ByteTensor, and optional per-device CUDA ByteTensors. Python/NumPy/CPU states are checked using isolated generators without global RNG mutation. CUDA current 16-byte seed/offset is checked on CPU, including the Philox offset divisibility requirement, without CUDA initialization. Source verified against installed PyTorch `2.6.0+cu124` and [PyTorch v2.6.0 CUDAGeneratorImpl](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp).
- Each observe first collectively preflights local context/counters/origin, exchanges graph/backend activity deltas and checks rank agreement, then on a global event collectively preflights hashing/metadata/RNG encoding before separate byte gathering. Reuses existing `_collective_preflight`, `_transport_lengths` and `_gather_payload_bytes`; no collectives inside checkpoint I/O contexts. Every rank's actual companion descriptor is rank ordered and appended to the same slot.
- Controller-requested backend `activity_before`/`activity_after` are optional but must be supplied together as nonnegative monotonic integers. Positive graph **or** activity delta triggers a shared observed slot; non-events are omitted. Counters are never modified.
- Strict position keys and contexts: train/arm/training=True/grad_enabled=True, eval/de_novo|evidence/training=False/grad_enabled=False, nonnegative epoch/batch_index/optimizer_step excluding bool.
- Canonical bytes use exactly JSON `sort_keys=True`, compact separators and `allow_nan=False`; save returns SHA256. `save(..., checkpoint=...)` binds copied exact epoch/step/global_step/pending_micro; `load(..., checkpoint=...)` requires matching checkpoint, identity, fingerprint and world size. Digest is checked before JSON decode; duplicate fields are rejected, and strict decoded schema is checked independently of file digest.
- Incoming identity/fingerprint/context/meta/RNG data and returned payloads are copied through canonical serialization. Explicit bootstrap assignment also copies provenance. Reconstructed slots require nonempty explicit provenance and retain `reconstructed_legacy` origin. Forced observed slots are forbidden. The module does not infer an observed ledger for original checkpoint200 or manufacture reconstruction proof.

## TDD evidence

Initial RED command (before creating production module):

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history -v
```

Relevant output:

```text
ModuleNotFoundError: No module named 'transgenic.training.b6_compile_history'
Ran 1 test in 0.000s
FAILED (errors=1)
```

This was the task's required missing-implementation failure; tests exercised the proposed public interface before implementation existed.

After initial implementation, two test setup errors were corrected: eval fixture still used train view after the controller clarified enums; recomputed-digest fixture attempted to mutate an unsaved checkpoint=None instead of assigning the saved checkpoint context. No behavior expectation was weakened.

Self-review RED regressions (before correcting production gaps):

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history.CompileHistoryTest.test_bootstrap_assignment_copies_caller_provenance tests.test_b6_compile_history.CompileHistoryTest.test_malformed_cuda_rng_bytes_are_rejected -v
```

```text
FAIL: test_bootstrap_assignment_copies_caller_provenance
AssertionError: {'source': ['explicit', 'changed']} != {'source': ['explicit']}
FAIL: test_malformed_cuda_rng_bytes_are_rejected
AssertionError: ValueError not raised
Ran 2 tests in 0.017s
FAILED (failures=2)
```

Added copying bootstrap property and CPU validation of CUDA seed/Philox state. Both passed in full GREEN.

Additional strict-JSON RED command:

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history.CompileHistoryTest.test_duplicate_json_fields_fail_closed_even_with_recomputed_digest -v
```

```text
FAIL: test_duplicate_json_fields_fail_closed_even_with_recomputed_digest
AssertionError: ValueError not raised
Ran 1 test in 0.015s
FAILED (failures=1)
```

Corrected load using an object-pairs hook that refuses duplicate keys before decoded schema validation.

Final GREEN command:

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history -v
```

```text
Ran 24 tests in 23.272s
OK
```

24 passed, 0 failures/errors, 0 skips, output pristine. Four tests spawned actual two-process CPU Gloo groups with file rendezvous, 20-second process-group timeout and 45-second worker join bound. Cleanup terminates only these tests' still-running workers.

## Covering tests

- Input values, dtype, shape, stride, requires_grad, dynamic dimension marking; metadata and rank changes; unsupported values/nonfinite scalars/sparse tensors; bfloat16 and optional scalars.
- Exact independently saved next Python/NumPy/torch CPU draws plus Gaussian cache preservation; CUDA byte roundtrip; malformed Python/NumPy/CPU/CUDA RNG states and explicit typed/base64 payloads; CPU global RNG unchanged on rejected states.
- Canonical persistence/digest; missing or damaged files; missing expected digest; strict identity/fingerprint/world/checkpoint; invalid checkpoint integer/schema fields.
- Recomputed outer SHA cannot hide version, checkpoint, identity, fingerprint, world, context, rank order/cardinality, ordinal order/duplication, input dtype/shape/digest, RNG or absent reconstruction provenance damage. Duplicate JSON fields also fail despite a recomputed digest.
- Caller mutation cannot alter recorded history, returned payloads or assigned bootstrap provenance; empty history roundtrip; non-events omitted; negative graph deltas and malformed backend activity counters refused; full eval/no_grad context recorded.
- Actual two-rank Gloo: rank1-only graph event saves both distinct inputs identically; rank1-only backend activity with zero graph deltas saves both companions; rank1 hashing failure reaches both workers without hanging; context disagreement reaches both workers.

## Self-review and other verification

Read the scoped production/test diff. Found and fixed the three RED regressions above. Verified forced slots cannot receive observed origin, reconstructed provenance is explicit, loaded payloads receive the same entry/schema validation, and every local hashing/encoding failure occurs in collective preflight before descriptor gather.

`git diff --check --` the four owned module/test/plan/spec paths passed with exit 0. `graft build` refreshed the ignored worktree graph successfully (422 files, 6017 nodes, 14872 edges). Graft calls saved approximately 1,674,548 tokens by the tool's reported estimates.

Remaining scientific/operational boundaries: CPU/Gloo verifies journal transport and corruption handling; it does not establish compiler event compression equivalence in real four-rank GPU training. CUDA state layout is explicitly the current supported PyTorch Philox layout, bound by supplied runtime fingerprint. Rank0 selection, atomic checkpoint transaction, local save/load error propagation and legacy provenance proof remain caller/integration responsibilities as specified. Empty history cannot itself prove that a trainer has never previously compiled; integration owns that lifecycle condition.

Commit: `80a17a8` — `feat(b6): persist synchronized checkpoint-bound compile history` (four owned files only).

## Follow-up fix: bind real weak dynamic annotations

Controller/integration identified that production uses `torch._dynamo.maybe_mark_dynamic`, while the initial signature captured only strong markers. Verified the actual installed PyTorch `2.6.0+cu124` API by applying `maybe_mark_dynamic` to a real labels tensor and inspecting its resulting state and implementation: it creates `_dynamo_weak_dynamic_indices={1}` and no strong marker.

Added a real-tensor regression before changing production code. It records a real weak annotation using `maybe_mark_dynamic(labels, 1)` and requires `check_inputs` to refuse an unmarked clone, weak annotation on another axis, and strong annotation on the same axis created with real `mark_dynamic`. No attribute mock is used for this regression.

RED command:

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history.CompileHistoryTest.test_real_weak_dynamic_markers_are_bound_distinctly_for_replay -v
```

```text
FAIL: test_real_weak_dynamic_markers_are_bound_distinctly_for_replay
AssertionError: ValueError not raised
Ran 1 test in 0.006s
FAILED (failures=1)
```

Expected failure: replay validation accepted a clone that had lost the original weak annotation.

Production change: tensor descriptors now require two separate arrays, `dynamic_dims` for `_dynamo_dynamic_indices` and `weak_dynamic_dims` for `_dynamo_weak_dynamic_indices`. Each is sorted, unique, in-range, nonnegative integers excluding bool; unmarked arrays are empty. Original value/dtype/shape/stride/requires_grad binding is preserved. Weak annotation is not collapsed into the strong set. The current unpublished version-1 schema refuses older incomplete descriptors instead of silently inventing weak markers. Public function signatures are unchanged.

Additional covering test rejects missing, out-of-range, duplicate, bool and wrong-type weak arrays in persisted histories even when the outer file digest is recomputed.

GREEN command (only the focused journal suite):

```sh
PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_compile_history -v
```

```text
Ran 26 tests in 22.490s
OK
```

26 passed, 0 skips, output pristine, including all four two-rank Gloo regressions. Scoped `git diff --check` passed. Self-reviewed the fix diff for distinct marker binding and strict schema validation. Refreshed the ignored worktree graph. Files changed: journal module, journal tests, own design spec. No unrelated suites, trainer or replay-state changes. Follow-up graft reported 79,056 tokens saved.

Follow-up commit: `789ecf4` — `fix(b6): bind weak dynamic markers in compile history` (three owned files only).
