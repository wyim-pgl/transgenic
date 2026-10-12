# C2 compile replay: preserved training state

The user authorized repairing and proving C2 resume determinism, then starting the
4099-update pilot without another approval. The scientific gate remains exact
continuous/resumed checkpoint equality. Root cause evidence is experiment 3364480
and the restart handoff in the main worktree, `notes/RESTART_HANDOFF_20261012.md`.

This first implementation unit supplies the state boundary needed by synchronized
compile-history replay. It does not claim to restore compiler history by itself.
The next unit records global forward event slots with each rank's real inputs,
binds the history to each checkpoint, and replays the same train/eval views. The
legacy checkpoint200 has no observed ledger and requires a separately proven,
explicit bootstrap. No production run may proceed on this unit alone.

## State boundary

Create `src/transgenic/training/b6_replay_state.py`, with a context manager
`preserve_training_state(model)` and helpers `capture_rng()` / `restore_rng(state)`.
The caller passes the unwrapped original model. Snapshot on entry, restore in
`finally`, including a BaseException from the replay. CPU copies must avoid a
second model resident on the GPU. Preserve all registered parameters and their
object identities, pending gradients (including None), all modules' complete
buffer registries (including None and nonpersistent entries), and per-module
training flags. Undo new buffers and restore deleted or replaced registrations.
Restore Python, NumPy, CPU torch and every already-initialized CUDA RNG state.
Do not initialize CUDA merely to snapshot a CPU-only model.

The context must not call an optimizer or scheduler, modify training counters,
or reset Dynamo/compiler caches. The caller owns those objects and replay order.
No general-purpose module topology or arbitrary Python attribute rollback is
required: B6 replay does not add/remove submodules. Unsupported tensor layouts
must fail explicitly before replay rather than silently change layout or values.
CPU torch tests cover real state mutations and exception restoration. CUDA-only
verification is separately required in NGC before using this in a GPU proof.

RNG helper state may be an in-memory Python object, suitable for later encoding
in the checkpoint ledger. Persistence/validation of the ledger is a separate unit.

## Verification

Tests must catch a missing restore, omitted None/nonpersistent/new/deleted buffer,
parameter replacement, loss of tied references, pending gradient mutation,
heterogeneous child train/eval modes, or a missing RNG family. An exception test
must use a BaseException subclass and assert both restored state and propagation.
Compare draws against independently saved RNG states, not the helper under test.
Existing compile recovery and trainer tests must retain their behavior.
