### Spec Compliance

- ❌ Issues found: original registered parameter/buffer lookup can remain replaced after restoration (`src/transgenic/training/b6_replay_state.py:121–125`); accepted shared-storage views can lose their storage relationship/offset (`:84–87`, `:109–119`). Both are concrete state-boundary defects in currently accepted mutations/layouts.
- ✅ Main contracts implemented: CPU snapshots with retained object references and identity deduplication (`:74–104`), gradients including None (`:96–98`, `:129–130`), full registration/nonpersistent sets and per-module modes (`:120–128`), Python/NumPy/CPU/already-initialized CUDA RNG (`:43–67`), and nested-finally restoration on BaseException (`:133–143`). No optimizer/scheduler/step/compiler mutation APIs occur in the complete module diff.
- ⚠️ Cannot verify from this diff: NGC compatibility with the actual B6 tensor layouts, operational compiler-history replay, DDP behavior or continuous/resumed GPU checkpoint equality. These are expected follow-on scope limits, not missing Task1 implementation.

### Strengths

- ✅ `src/transgenic/training/b6_replay_state.py:139–143`: RNG cleanup still runs if model restoration fails; exception handling covers BaseException without suppressing the normal successful-cleanup exception.
- ✅ `tests/test_b6_replay_state.py:107–131`: the real fixture checks tied object references, original pending gradient identity/value, None registrations, new/deleted/replaced entries, heterogeneous modes and exact BaseException instance propagation.
- ✅ `tests/test_b6_replay_state.py:133–161`: shape/dtype and transposed-stride changes are explicitly exercised, rather than assuming that copying state_dict is sufficient.
- ✅ `tests/test_b6_replay_state.py:195–245`: expected RNG draws come from independent library snapshots; initialized CUDA values/draws and CPU noninitialization are separately tested.
- ✅ Scoped structure:143-line production module,249-line real behavioral tests, matching plan/spec; no unrelated trainer edits in the review diff.

### Issues

#### Critical (Must Fix)

- None found in this scoped diff.

#### Important (Should Fix)

1. **Restored registries can be shadowed by ordinary attributes.** `src/transgenic/training/b6_replay_state.py:121–125` clears/repopulates `_parameters`/`_buffers` but leaves an ordinary attribute created under an original registration name in `module.__dict__`. A valid body can `del model.weight; model.weight = torch.tensor([-3.])` and similarly replace a deleted buffer with a plain Tensor. After exit, `_parameters['weight']` and `_buffers['running']` contain the original objects, yet `model.weight` and `model.running` return the replacement values. Forward code therefore sees corrupted training state even though registry-only checks pass. Remove conflicting ordinary attributes for the original registration names before restoring them, and add parameter/buffer regressions for delete→plain-Tensor assignment. This is narrowly required registration-name restoration, not arbitrary Python attribute rollback.

2. **Accepted shared-storage views are not faithfully restored after metadata/data replacement.** `src/transgenic/training/b6_replay_state.py:84–87` stores shape/stride/dtype/device but not original storage relationship or offset; `:109–119` can copy into a replacement allocation or allocate independent storage. Concrete accepted fixture: register `whole = torch.arange(8.)` and `slice = whole[2:6]`, then replay `slice.data = torch.zeros(4)`. Entry accepts both nonoverlapping strided tensors. Exit restores their object identities and numeric values but changes slice offset2→0 and breaks shared storage; assigning `slice[0] = 99` afterward no longer updates `whole[2]`. Numerical state and tied same-object aliases are insufficient to preserve this registered tensor behavior. Preserve supported storage/view relationships and offsets, or explicitly reject unsupported shared-storage/offset cases before body entry; cover this with a regression. No assertion is made that the production B6 currently contains such distinct-object aliases—its layout preflight remains a follow-on check.

#### Minor (Nice to Have)

- `.superpowers/sdd/2026-10-12-c2-replay-state/task-1-green.log:17`, `:170`: existing trainer ResourceWarnings/ImportWarning and shared-tensor-saving messages mean the combined baseline output is noisy. The task report documents this and the new focused tests are reported warning-free; these unchanged baseline warnings are not a blocker for this state module and should be cleaned up separately.

### Checks Run

- Read the task brief/report and complete `task-1-review.diff` once; did not separately reread changed files or crawl callers/source outside the scoped package. Consulted graph first, as required.
- Inspected the recorded integration log at its specified path:44 tests,149.160s, `OK (skipped=2)` at lines203–205; skips identify absent pyarrow and already-initialized CUDA. Did not rerun any suite.
- Named concrete risk **storage/view alias preservation**: ran a one-shot CPU probe with the two registered buffers above under the new context. Output: identities restored=True/True; values correct; shared-storage/offset before `(True,2)`, after `(False,0)`; subsequent slice write leaves whole[2]=2.0. No CUDA allocation or module mutation persisted outside that process.
- Named concrete risk **registration attribute shadowing**: ran a one-shot CPU probe with original parameter `[3.]` and buffer `[6.]`, deleting/reassigning plain tensors inside the context. Output: parameter/buffer registry originals=True/True; parameter/buffer attribute originals=False/False; remaining accessible values `[-3.]`/`[-6.]`.
- Both targeted probes used `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python` in `/tmp/b6-c2-recompile`. No suite rerun, remote command/job, implementation edit or spawned agent.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The core snapshot/finally design is focused and covers the ordinary path well, but two accepted mutation patterns leave observable registered training state changed after the boundary. Fix or preflight-reject those cases before trusting the context in compile-history recovery; operational GPU proof remains separately required.
