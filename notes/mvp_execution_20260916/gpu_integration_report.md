# MVP GPU integration — resumed 2026-09-16

## Delivery and ancestry

Branch: `mvp-gpu-integration-resumed-20260916`.
Native isolated worktree: `/tmp/pi-agent-259f45ab-10bc-4d1-ce65a6c5`.
Resumed directly from durable partial `544bcb0f5f66a197a11a1edb204291c672956fea`, including its 147-line integration test; no restart/reimplementation of reviewed A.

Verified merge bases before delivery (this commit adds no merges):

- Reviewed A: `d99b1f8a2a42edfe4877788d0f7d1ad7f9d53e50`.
- Resource FIX1 cherry-pick: `6ed59e212074114a3c214dfe189bdfe71bbd8367` (preceded by resource `3e15b49a9c6ed56f0a762035750ae5ca57f53064`).
- Local `main`: `9a8c8e1ba68286e845fb99be3cb76112a8bcd186`.

The narrow review delta is `544bcb0..HEAD`; combined GPU-probe review is `3b62306..HEAD`. The delivery commit contains only resource source, integration tests and this report. The durable parent already includes its autosaved ignore-file changes; they are not new delivery changes. Obtain the exact delivery ID with `git rev-parse mvp-gpu-integration-resumed-20260916` (also supplied in the final handoff).

Read the resource FIX1 report and both neutral FIX1 reviews in `/tmp/mvp-resource-context-20260916/`. `.atl/resume.md` was absent in this historical worktree. Graft map/build/query/caller checks preceded source inspection, and the local graph was refreshed after edits. Advisor consultation failed with `claude_code_version_too_old`; no advisor assessment or independent review is claimed.

## Root causes and integration

- The saved tests initially produced **2 failed, 3 passed in 11.31s**: resource still called the now-four-argument shared dataset factory with only two arguments. The actual factory raised `TypeError`; no successful resource execution was inferred from the autosave.
- `resource_probe.run` now calls exactly `p.datasets(args, bindings, cfg, model.config)` before compilation/wrapping. It retains the unwrapped `model_config` and calls shared `validate_decoder_dataset` on resource train and valid. Existing additional v3/8192 resource checks remain. Train validation precedes wrapping; valid validation uses the original config even when the prepared wrapper has no `.config`.
- Resource preloading uses `p.collective_batches` on the selected real-row generator, preserving 24 distinct rank-local samples and sharing retrieval failures before update entry. It invokes unchanged coordinated `p.one_update` directly, and uses `p.collective_compiler_check` directly before validation. Tier-specific steady-recompile checks remain local preflight operations.
- `accelerator.prepare`, coordinated batch/compiler/update helpers and paired validation are not nested in local-only preflight. Existing `validate_paired_batch` remains inside retrieval preflight before paired forwards; metric semantics and finalizer are unchanged.
- Resource early dispatch and normal compiled main ownership were already correctly combined in the durable parent. They are retained without source churn: offline flags precede dispatch/imports, resource owns its Accelerator, and normal main passes its sole owned Accelerator to `compiled_update`. Dynamic CPU failure tests cover both compiled modes at setup and compiled dispatch, with exactly one creation/cleanup.
- An added profile-rejection regression exposed a cleanup leak: profile/scratch validation raised after Accelerator creation but before `try/finally` (**1 failed, 10 passed in 26.21s**). Path resolution/validation now runs through local collective preflight inside the owned cleanup scope. This coordinates ordinary path errors and reaches `end_training` on rejection. This does not claim universal recovery from CUDA/DDP failure or dataset-close exceptions.

The original partial tests were strengthened rather than weakened: explicit train validation before prepare, a wrapper with no config, unconditional checks of both closed DuckDB handles, coordinated-helper call boundaries, normal-owner failure cases, and resource profile rejection. Final integration suite has 11 tests. CPU boundary fakes stop before real training/publication; they are not successful GPU-runtime surrogates.

## Verification ledger

Interpreter: `/home/wyim/miniconda3/envs/pdb/bin/python`.
Existing read-only-use dependencies: local `.testdeps` symlink to `/tmp/b6-p2-readiness-1535112/.testdeps`; existing `/tmp/mvp-decoder-resumed-testdeps` supplies missing `pysam`. No installations, environment rebuilds or dependency-file changes.

1. Original saved suite: **2 failed, 3 passed in 11.31s** (stale factory caller).
2. Added shared-helper assertions, before source fix: **3 failed, 3 passed in 7.83s**.
3. Factory/shared-helper fix: **6 passed in 8.54s**.
4. Extended ownership/profile suite before profile fix: **1 failed, 10 passed in 26.21s**.
5. Final focused command:
   ```sh
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=.testdeps:src \
     /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_gpu_integration.py
   ```
   **11 passed in 22.13s**; external timeout 120 seconds, not reached.
6. First combined command used the same environment with `tests/test_mvp_*.py`:
   **350 passed, 1 failed in 695.13s**. Sole failure: missing `pysam` while importing the real BAM-builder fixture from `test_mvp_tracks.py`. No source/fixture/validator changes were made for this environment failure. External timeout 1200 seconds, not reached.
7. Final expanded combined command:
   ```sh
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=.testdeps:src:/tmp/mvp-decoder-resumed-testdeps \
     /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
     tests/test_mvp_*.py revision/scripts/tests/test_mvp_c0lite.py
   ```
   **370 passed in 941.96s**; external timeout 1200 seconds, not reached. Includes CLI/runtime/finalizer, decoder/model/dataset, real-probe/readiness, resource FIX1, progress collision/cap, two-process paired-input rejection, bounded rank-local loader/compiler failures, tracks and BAM-builder tests. No suite invocation hit its external timeout. Timeout-path tests are expected negative tests, not a timed-out suite.
8. `py_compile` passed for resource source, real probes and integration tests. `git diff --check 544bcb0` and `git diff --check 3b62306` passed after the final source/test changes. Source A's real probes, production runtime, metrics, parser/P1 CLI, config and dependency pins are unchanged by this delivery.

CPU test versions were Torch **2.6.0+cu124**, Transformers **4.46.3**, Accelerate **0.34.2**, pysam **0.24.1**. CUDA availability was explicitly **False** under the final/broad test environment. Earlier focused boundary tests used CPU fixtures/no-op CUDA boundaries but did not hide host GPU discovery; a version-only query reported host availability True. No actual CUDA training/profiling was executed. These results do not certify the required Torch **2.14**, Transformers **4.57.6**, Accelerate **1.15.0** runtime; its pins and offline cache/config policy remain unchanged.

## Preserved limits and controller handoff

Immutable `rankN.SEQUENCE.profiling.json` publication and integer/measurement validation are unchanged. The authorization rule still rejects prior smoke plus proposed probe cap above **18,000 seconds (5 node-hours)**, and progress/receipt checks reject exhaustion. Controller must additionally include startup, prior attempts, parent verification and teardown in an outer allocation deadline; a child timeout/projection alone is not allocation-wide accounting.

The resource payload remains explicitly disposable, with **9 optimizer steps / 216 microbatches per rank / 4 ranks**. No fabricated 250-update completion metadata, PASS/readiness receipt or training authorization was created. The 250-update number remains projection math only.

**Combined source awaits independent review; GPU execution/readiness/training remain HOLD.** Preserve the existing P1 CLI. Before controller-authorized P1/P2 diagnostics, regenerate consumed-code/tooling bindings, prepared records and receipts (including the already-changed metric module), stage exact immutable inputs/cache layouts, and apply remaining-cap outer wall limits. Do not reuse old v2 receipts. No SSH, remote jobs, subagents, main merge, push or edits to source A were performed. Dependency symlink and graft cache are local/uncommitted.
