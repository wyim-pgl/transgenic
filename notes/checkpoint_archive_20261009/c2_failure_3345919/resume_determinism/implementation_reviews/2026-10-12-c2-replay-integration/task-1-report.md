# Task 1 implementation report

Status: implementation and verification complete; scoped commit `180589b`; independent review remains controller-owned.

## Implementation and review surface

- Added `src/transgenic/training/b6_compile_replay.py`: effective runtime/compiler/cache fingerprint; exact rank agreement; sampler-only reconstruction through the actual prepared batch sampler and original collator; state-preserving synchronized replay; exact per-rank graph-delta checks; memory cap; strict original200 bootstrap manifest validation and all16-slot reconstruction.
- Updated trainer lifecycle: validate compiled ledger before checkpoint state load; coordinate local load errors; retain restored pending gradients; replay before new training; observe every training forward after backward and every de_novo/evidence eval forward; bind ledger SHA and exact position in latest/epoch/fixed-step transactions. Latest writer now persists declared pending gradients.
- Added explicit `B6TrainSpec.legacy_compile_bootstrap` and `--legacy-compile-bootstrap PATH`. Reject fresh/eager usage and overriding any existing ledger/digest. Heavy complete-checkpoint hashing happens once on main rank, under checkpoint_io, in CUDA/world4 bootstrap execution. Pure manifest validation is CPU-testable. Operational allocation authorization remains the controller's responsibility.
- State dependency reviewed at91c2737; journal dependency read at789ecf4, including distinct strong and weak dynamic annotations. No changes to either owned dependency module.
- Cache paths are bound literally. No rank/node path normalization, cache disabling, numerical-policy changes, or identity relaxation. Fingerprint reads effective FX/AOT cache helpers as well as runtime config and environment.

## Public helper/proof interfaces

`replay_history(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard, *, memory_probe=None)` validates agreement and replays an existing journal. `loaders` maps `train`/`eval` to Accelerator-prepared loaders.

`execute_replay(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard, *, memory_probe=None, legacy_rows=None)` is the single priming boundary used by BOTH ordinary resume and bootstrap. The proof harness can wrap it with a depth counter to exclude primer hooks from actual-training traces. `legacy_rows=None` replays stored entries; non-None maps string rank -> string 1-based slot -> `[row, epoch, uniforms_sha256]` and is passed only after strict bootstrap validation. All protected model state, wrapper modes, pending gradient objects/values, loader/sampler epochs/generators and RNG are restored; compiler/guard state is retained.

`bootstrap_legacy(history, manifest_path, checkpoint_dir, run_dir, checkpoint, model, original_model, loaders, arm, accelerator, per_rank_acc, guard, *, batch_size, memory_probe=None)` enforces CUDA execution, validates on rank0, shares provenance/rows, and invokes execute_replay.

`validate_legacy_manifest(path, checkpoint_dir, run_dir, checkpoint, *, arm, world_size, batch_size)` is local-only and returns `(provenance, rows)`; provenance contains full manifest and SHA256. `reconstruct_batch(loader, epoch, batch_index, device)` loads only the selected prepared-sampler keys.

## RED/GREEN evidence

Required command prefix: `PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest`.

- Initial `tests.test_b6_replay_integration -v`: RED2 in38.075s, missing compile_history_sha256 and silently accepted ledger-less compiled resume. Log `/tmp/b6-replay-red.log`.
- Additional wrapper rollback RED: actual wrapper remained training after primer (1 expected failure). Log `/tmp/b6-replay-wrapper-red.log`. Fixed by preserving prepared-wrapper modes in addition to original-model state.
- Four edge checks GREEN in55.942s: bounded real two-rank Gloo/DDP replay with prepared sharding/padding; rank1 changed-input collective refusal; actual tiny B6 epoch resume with a pending microbatch matches next-update weight digest; pending gradients/RNG/buffers/modes survive success and KeyboardInterrupt. Log `/tmp/b6-replay-edge2.log`.
- First integration+dependency run GREEN52 in178.866s (one CPU-noninitialization test skipped because earlier integration tests initialized CUDA). Log `/tmp/b6-replay-focused.log`. Isolated state module GREEN12 in1.956s, no skips, including CUDA and CPU-noninitialization checks; log `/tmp/b6-replay-state-isolated.log`.
- Effective-cache override and real all16-slot forced-retention tests GREEN2 in1.955s. Log `/tmp/b6-replay-extra.log`.
- First original regression suite:75 tests, two changed-contract failures, one pre-existing pyarrow skip. Log `/tmp/b6-replay-regression.log`. Necessary updates only:
  - `test_b6_trainer.ReviewRound1Tests.test_every_link_verifies_compiled_execution`: a frozen graph counter is now rejected EARLIER by exact replay graph-delta validation; assert that specific refusal.
  - `test_b6_finalization.PilotFinalizationTests.test_short_final_resume_verifies_compile_before_publishing_success`: reset Dynamo between in-process links to simulate a real cold resumed process; retains injected final verification and no-DONE/no-final-checkpoint assertions.
- Both amended tests GREEN2 in57.143s. Log `/tmp/b6-replay-contract.log`.
- Final combined command: `tests.test_b6_replay_integration tests.test_b6_replay_state tests.test_b6_compile_history tests.test_b6_trainer tests.test_b6_compile_recovery tests.test_b6_compile_static tests.test_b6_pilot tests.test_b6_finalization -v`; log `/tmp/b6-replay-final.log`. GREEN129 in280.504s, exit0, two explained skips: CPU-context no-CUDA-initialization (already initialized earlier in this combined process; isolated state module passes all12), and the pre-existing missing-pyarrow WP3 fixture test. No failures/errors. Expected Accelerate tied-weight warnings and pre-existing test ResourceWarnings appear in the log. All16 integration tests pass, including real2rank replay/failure tests.

## Self-review and scientific limits

Closed wrapper-mode rollback gap with a real failing test; made snapshot/payload preparation and state loading coordinated local-only operations before prepared forward collectives; checked record-after-backward and all three publication paths. Graft rebuilt and git diff --check passed.

The test fixture originally assumed every eval view must compile; corrected it to assert retained actual eval contexts, because an evidence view may reuse the de_novo graph. Prepared DataLoaderShard also prefetches one future batch; replay still reconstructs only saved sampler keys, and the ordering test accounts for normal-loader prefetch. Neither correction changes production equality/event compression.

No remote writes, GPU submissions, main merge, source checkpoint edits, production authorization or historical-loss claim. Exact real4GPU original200→210 historical agreement, continuous/resume full-state comparison, future-event coverage, validation400, and another resume remain controller-owned scientific gates.

## Commit

`180589b feat(b6): replay checkpoint compiler history before resuming training` contains only the six owned/justified integration files. Worktree has only the controller-owned integration plan/spec untracked after commit. No main merge.

## Whole-branch review fix wave: effective cache paths

Finding: whole-branch review at5f1c7c7 identified that unset TORCHINDUCTOR_CACHE_DIR left the actual tempfile/user-derived local cache directory unbound. Two allocations could therefore use different default caches with identical fingerprints.

Fix: `_cache_paths()` adds the resolved absolute Inductor local directory to the fingerprint, using PyTorch's side-effect-free `default_cache_dir()` when no override exists. It never calls the mutating `cache_dir()` or `triton_cache_dir()` helpers. For default Triton routing, bind the same Inductor root plus `triton` and the documented `decimal_device_index` suffix policy. This policy is shared across ranks; it does not initialize/query a CUDA device or accidentally require different ranks' local indices to match. It leaves all cache settings untouched. Native Triton HOME defaults are bypassed by this Inductor-owned default route, so unused HOME fallback paths are not bound. For explicit Triton overrides, read the applicable native pure setting: modern `knobs.cache.dir`, or legacy stripped override/native `default_cache_dir()` fallback. Relative paths bind their actual absolute location; empty modern settings remain empty, preserving the native manager's eventual failure behavior.

Compatibility evidence: local torch2.6 source inspected directly. Controller also inspected actual NGC torch2.14.0a0+4fdf77b940.nv26.08 and Triton sources with read-only module introspection. Both PyTorch versions derive default Inductor roots from the pure resolver and default Triton paths as `<inductor>/triton/<decimal-device>`. NGC Triton uses `knobs.cache.dir`; its empty setting is not replaced with a fallback. The operational proof's explicit fixed cache paths are unchanged.

TDD evidence:

- `/tmp/b6-cache-path-red.log`: two expected assertion failures in0.006s. Real `default_cache_dir()` differs under allocation-a/allocation-b while old fingerprints remain equal; old fingerprint lacks explicit effective-path fields.
- `/tmp/b6-triton-blank-red.log`: one expected assertion failure in0.003s. Local Triton3.2's explicit empty override resolves to its native HOME cache, while the preliminary route descriptor incorrectly remained empty. Corrected using the native version's pure setting.
- Focused verification command: `PYTHONPATH=/tmp/b6-spec-review-deps:src:. /home/wyim/miniconda3/envs/pdb/bin/python -m unittest tests.test_b6_replay_integration.ReplayHelperTests tests.test_b6_replay_integration.TrainerHistoryTests.test_fresh_compiled_checkpoint_binds_history tests.test_b6_replay_integration.ResumeHistoryTests.test_resume_replays_before_new_rows_and_preserves_training_result -v`; log `/tmp/b6-cache-path-final.log`.
- Tests check actual default-root divergence, deterministic shared Triton suffix policy, explicit path preservation, native blank semantics, unchanged environment, no mkdir calls, no created allocation/cache directories, and no CPU CUDA initialization; actual fresh save and resume tests cover the fingerprint consumer.

Only `b6_compile_replay.py` and its integration tests changed. No129-suite repeat, cache behavior changes, remote writes/jobs, unrelated source changes, or additional agents. Graft rebuilt. Final focused result and commit appended below.

Fix-wave GREEN: all7 focused tests passed in37.499s, no skips/errors; command exited0. `git diff --check` passed. Scoped commit `188bf3e fix(b6): bind effective default compiler cache paths`; worktree clean after commit. Controller-owned one scoped re-review is the remaining code-review step.
