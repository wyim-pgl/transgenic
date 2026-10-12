# C2 synchronized compile history

This unit records the missing compiler input history. It follows the exact-resume
requirements in the main worktree's `notes/RESTART_HANDOFF_20261012.md` and uses
`b6_replay_state.capture_rng()` for pre-forward state. Trainer wiring and legacy
bootstrap are the next units; this unit is independently testable on CPU/Gloo.

## Interface and schema

Create `src/transgenic/training/b6_compile_history.py` with:

- `input_signature(inputs) -> dict`: per-name value SHA256, dtype, shape, stride,
  requires_grad, and explicitly marked dynamic dimensions for tensor inputs.
  Bind strong `_dynamo_dynamic_indices` as `dynamic_dims` and weak
  `_dynamo_weak_dynamic_indices` as `weak_dynamic_dims` distinctly; both fields
  are required sorted unique in-range integer arrays (empty when unmarked).
  Support None and JSON scalars; refuse unsupported values. Hash dense tensor
  bytes in CPU contiguous order while separately binding original layout.
- `encode_rng(state) -> dict` / `decode_rng(payload) -> dict`: lossless safe JSON
  representation of Python/NumPy/CPU/CUDA RNG (explicit types/base64 bytes, never
  pickle or executable decoding). The state helper keys are python, numpy,
  torch_cpu, torch_cuda; torch_cuda is None if CUDA was uninitialized.
- `CompileHistory(identity, fingerprint, accelerator)`: store canonical copied
  identity and runtime/compiler fingerprint, world_size and ordered `entries`.
- `observe(position, inputs, meta, rng_before, graphs_before, graphs_after,
  *, origin='observed', force=False, activity_before=None, activity_after=None) -> bool`: all ranks call after the same
  forward/backward. Gather graph deltas and optional backend activity deltas with
  the accelerator. Activity counters must be supplied together as nonnegative
  monotonic integers; activity captures lazy backward compilation and backend
  cache hits independently of Dynamo graph counts. If any rank's graph or
  activity delta is positive (or force is true), gather every rank's real descriptor and append one shared
  slot. The local descriptor contains rank, input signature, copied batch meta,
  safely encoded pre-forward RNG, delta, and activity_delta (zero when omitted). Position is a JSON mapping with
  phase, epoch, batch_index, view, training, grad_enabled and optimizer_step.
  Position uses exactly these seven fields: phase is train/eval, view is arm
  for train or de_novo/evidence for eval. Both training and grad_enabled are
  True for train and False for eval. epoch/batch_index/optimizer_step are
  nonnegative integers excluding bool. Positions must agree on all ranks. Graph deltas must be nonnegative. force is
  for explicitly reconstructed bootstrap slots; it cannot label them observed.
- `to_payload() -> dict`: version=1, identity, fingerprint, world_size,
  entries, bootstrap (None or explicit provenance), checkpoint (None before
  saving or the exact checkpoint context). Set history.bootstrap to an explicit
  nonempty JSON provenance mapping before reconstructed entries; assignment
  copies it, and the journal does not manufacture or approve the evidence. Entry fields: ordinal
  (0-based), position, origin, ranks (rank-ordered descriptors). Origin values
  are observed and reconstructed_legacy. A copied/validated payload must not
  alias caller mutable objects.
- `save(path, *, checkpoint) -> str`: write canonical JSON and return its SHA256. Caller owns
  rank0 selection, checkpoint transaction and collective error handling. Required
  checkpoint is exactly epoch, step, global_step, pending_micro (nonnegative
  integers excluding bool). Save binds its copied numerical checkpoint context
  so a same-run ledger cannot be transplanted between checkpoints.
- `CompileHistory.load(path, expected_sha256, identity, fingerprint, accelerator, *, checkpoint)`:
  check SHA before decode, schema/version, exact expected identity/fingerprint/
  world and exact supplied checkpoint context, ordinal order, rank cardinality/order, positions, input descriptor and
  RNG validity. Reject missing expected digest, damaged/missing file, altered
  world/identity/context, malformed RNG, missing rank, or duplicate ordinal.
  Use the existing directional identity checker only if an actual old/new
  identity difference is supported by it; compiler-history fingerprint stays
  strict. Empty history is valid only before any compiled forward.
- `check_inputs(entry, inputs, meta, rank)`: require exact saved local signature
  and metadata; used before replay to reject changed reconstructed data.

The distributed observe must propagate local descriptor/hash errors to every
rank before any rank enters the next model collective. A single failing rank
must not strand its peer. `rt.checkpoint_io` can surround *local-only* work;
its documentation forbids collectives inside that context. Use a separate
gather operation after local preflight. Do not place tensors in the persistent
JSON, invent companion inputs, perform rank-specific replay, or mutate compiler
counters. Record full train/eval contexts even when no_grad is active.

## Scientific limits

Observed event compression needs full checkpoint equality in real four-rank
training. The original checkpoint200 has no observed compile ledger. Its
candidate bootstrap is original slots1..15 and3915, all actual companions,
with original checkpoint/draw-prefix hashes, reconstruction provenance and an
independent proof. This module permits reconstructed origins but does not
manufacture or approve that proof. Bootstrap provenance is a JSON mapping to
be populated by the explicit bootstrap integration; load rejects reconstructed
entries when provenance is missing. A compiler fingerprint is supplied by the
trainer integration, which must bind actual operating cache/config/mode settings.

## Tests

Use unittest. Test exact RNG round-trip by comparing independently saved next
draws; input values/shape/stride/dtype changes must be rejected. File mutations,
identity/world/context damage, absent provenance and rank/ordinal damage must
be refused even when the test recomputes the outer file SHA. Spawn two real
Gloo ranks with file rendezvous and a timeout: only rank1 increments its graph count or its backend activity count,
yet both histories contain both actual descriptors and identical bytes. Add an
injected descriptor failure on rank1 and require both workers to exit with a
propagated error, without hanging. No GPU jobs are part of this unit.

CUDA RNG is validated on CPU as the current PyTorch Philox state: a 16-byte
seed/offset pair with offset divisible by four; this never initializes CUDA.
Its current layout is documented in [PyTorch v2.6.0 CUDAGeneratorImpl](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp).
