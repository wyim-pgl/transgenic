# C2 compiler history integration and explicit legacy bootstrap

User authorization and scientific gates: main worktree `notes/RESTART_HANDOFF_20261012.md`
and `notes/C2_REPLAY_PROGRESS_20261012.md`. Never restart the old controller or
claim a first-forward result proves full resume. Original200 and diagnostic256
are protected. No main merge. This unit supplies implementation; remote scientific
proof and operational launch are separate controller-owned actions.

## Existing modules and responsibility

`b6_replay_state.preserve_training_state(original_model)` restores model registries,
values, pending grads, modes and Python/NumPy/CPU/CUDA RNG via CPU snapshots.
`b6_compile_history.CompileHistory` records synchronized event unions and actual
companions with encoded pre-forward RNG, graph/backend activity deltas and input
hash/layout. Read its spec and finished implementation before integration. Its
save/load require exact checkpoint position `{epoch,step,global_step,pending_micro}`.
History file SHA goes in checkpoint meta as `compile_history_sha256`.

Add `b6_compile_replay.py` for fingerprinting, actual batch reconstruction,
synchronized replay, and strict legacy manifest validation. Keep trainer edits
focused on lifecycle/forward/checkpoint calls. Do not merge the whole trainer into
the helper. Public helper functions/classes can be chosen coherently by the
implementer, with names recorded in its report for the subsequent proof harness.

## Normal compiled training and checkpoint resume

Construct history after Accelerator.prepare and verified run identity are known.
Fresh compiled training starts empty. Before each actual forward take RNG and
Dynamo graph/backend activity baselines. After forward AND training backward,
all ranks call observe with the same position and actual local inputs/meta.
Use `rt.inductor_activity()` for Inductor; graph count still applies to custom
backends used in tests. Guard observation must charge to that same input after
backward, never charge a lazy backward compile to the next input. Eval records
each batch's de_novo then evidence view separately under eval/no_grad. The
position's optimizer_step is completed updates before this forward; train view
is arm. No rank may skip global union synchronization because it had no event.

Every latest/epoch/fixed-step writer embeds `compile_history.json` in the existing
temporary checkpoint directory and its digest in meta before publication. Bind
the precise checkpoint position, even fixed-step provenance. Save latest pending
gradients if pending_micro is nonzero; current usual latest saves are at update
boundaries, but the writer must not falsely advertise pending state it omitted.
Eager debug training keeps existing behavior and does not require a compile ledger.

Compiled resume must validate ledger SHA/schema/identity/fingerprint/world and
checkpoint position before replay. Missing ledger/digest is an error unless the
explicit legacy bootstrap argument is present; a bad existing ledger is never
overridden by a bootstrap. Coordinate local validation errors using
`rt.checkpoint_io` (no collectives inside that context). Do not discard pending
gradients or optimizer/scheduler/RNG state restored by Accelerate.

Replay all stored entries in common ordinal order on every rank. Reconstruct
actual data from the entry's prepared loader batch_sampler at epoch/batch_index:
set its epoch, iterate sampler keys only up to that batch, load only those sample
keys, collate with the original collate_fn and send tensors to acc.device. Apply
the recorded view and mark labels dimension1 dynamic exactly as training does.
Validate metadata plus every input byte/layout signature collectively before
the forward. This avoids rasterizing every skipped row and handles sharding and
padding using the actual prepared batch sampler. Preserve loader/sampler epoch,
iteration and generator state that reconstruction touches, including failure.

Within preserve_training_state(original_model), restore that rank's recorded
pre-forward RNG before each entry; set recorded train/eval and grad context;
call the same prepared model and acc.backward(loss/per_rank_acc) for train.
No optimizer/scheduler/clipping/update/step/DONE/draw-log action is allowed.
Rebuild the CompileGuard in actual replay order and reject graph-count deltas
that differ from the saved per-rank observation. Backend activity is a trigger
for recording, not an exact replay count: warm cache hits vs generated kernels
have different counts. Keep Dynamo history after restoring protected state.
Check peak/reserved memory against the existing B6 cap during replay too.
Replay failure propagates and does not mark the run complete.

## Runtime/compiler fingerprint

Bind torch and git build version, CUDA runtime, Triton version when available,
GPU name/compute capability (not rank device index), backend name, world size,
mixed precision, checkpointing, relevant model config, deterministic and matmul/
cudnn precision settings, Dynamo suppress_errors/optimize_ddp/automatic_dynamic_shapes/
recompile limit, effective FX/AOT local/remote cache controls, and relevant cache
environment settings. Fingerprint must be identical across ranks or fail before
replay. Do not disable caches to make tests pass. Effective runtime options must
be read, not inferred solely from env variables. Paths that intentionally vary
per node require an explicit documented compatibility choice; default shared
Inductor cache behavior is retained. Do not silently weaken an existing identity
field or change training math/config to obtain equality.

## Explicit legacy checkpoint200 bootstrap

Add optional `B6TrainSpec.legacy_compile_bootstrap` and CLI
`--legacy-compile-bootstrap PATH`. Only applicable to compiled C2/world4/batch1,
resume meta epoch0,step4800,global_step200,pending_micro0; reject other uses,
fresh runs, eager runs and checkpoints that already carry a ledger.

The external JSON manifest schema is:

```json
{
  "version": 1,
  "kind": "reconstructed_legacy_compile_bootstrap",
  "checkpoint": {"epoch": 0, "step": 4800, "global_step": 200, "pending_micro": 0},
  "world_size": 4,
  "checkpoint_files": {"meta.json": "SHA256", "accelerate_state/model.safetensors": "SHA256"},
  "draw_prefix_lines": 4800,
  "draw_prefix_sha256": {"0": "SHA256", "1": "SHA256", "2": "SHA256", "3": "SHA256"},
  "slots": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,3915],
  "provenance": "metadata-derived candidate; not an observed compiler ledger"
}
```

The shown checkpoint_files are illustrative keys: require the complete regular
file set actually present in the committed checkpoint, including optimizer,
scheduler, all four random-state files and identity, with matching SHA256s.
Reject missing/extra file declarations, unsafe relative paths, symlinks, bad SHA,
changed checkpoint bytes, wrong prefix hashes/count or malformed slots. Compute
large checkpoint hashes once on rank0 under coordinated local I/O; the allocation
must own this heavy check. Never rehash the 786GB evidence on a login node.
Verify original draw prefixes from `spec.run_dir/draw_log/rankNN.tsv`, then each
reconstructed batch against its slot's row/epoch/uniform hash before executing.

Use actual slots1..15 then3915 together on all ranks. This candidate covers all
known pre200 label regimes according to independent review, but is explicitly
unproven until historical and continuous/resume checks pass. The missing legacy
event RNG is unknown: execute candidate primers starting from restored checkpoint
RNG, capture their actual pre-forward states, and label every retained entry
reconstructed_legacy. Force retention of ALL16 slots including non-events;
do not compress them further. Record full manifest plus its digest in history
bootstrap provenance, and record actual graph/backend deltas and tensor hashes.
Undo replay numerical state/modes/RNG/gradients afterwards; retained compiler
state is the only intended change. Future checkpoints carry this bootstrap and
observed later events; subsequent resumes replay the recorded candidate exactly.

Independent legacy review at main
`notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/LEGACY_BOOTSTRAP_REVIEW_20261012.md`
contains source SHA/prefix SHA/actual companions. Original update210 historical
loss is3.2567708492279053; matching a newly cold run is insufficient.

## Tests and limits

Unittest regression tests must exercise actual helper behavior and trainer flows,
not grep source. Prove fresh compiled save/resume records and replays before the
first new training input; latest and epoch/eval paths bind history; pending grads
and RNG survive; missing/tampered history and changed input are refused; malformed
legacy manifests cannot authorize replay; source checkpoint and draw logs are not
changed by reconstruction. Use tiny real model/custom compile backend/CPU as
appropriate. Include a two-rank Gloo replay ordering or failure test if unit
journal tests alone do not cover integration. Existing trainer/pilot/finalization/
compile tests must pass. Test failures cannot be resolved by disabling compilation,
ignoring history, loosening equality, or changing numerical settings.

GPU scientific acceptance still requires original200→210 historical agreement,
continuous/resumed full checkpoint bitwise proof in actual4GPU NGC with operational
cache settings, future compile-event coverage, validation400 and another resume.
No production submission in this implementation task.
