# Independent deployment execution review — S0 + P1

## Disposition: APPROVE bounded diagnostic

Approve **only one controller-authorized, one-node/four-GPU, 30-minute S0 + real_tokenizer diagnostic** using the exact release below. No blocking deterministic path, CLI, import-symbol, staging-layout, or false-PASS defect found in the reviewed integration. This is not a runtime PASS, resource-mode approval, training authorization, or approval of newer combined code.

The controller must still perform the plan's fresh queue/exclusion/quota observations, externally verify the ledger hash, confirm the staging evidence exists, and record the sole job ID and 0.5 node-hour / 2 GPU-hour authorization. No remote operation was performed for this review.

## Exact reviewed snapshot

Source commit: `d99b1f8a2a42edfe4877788d0f7d1ad7f9d53e50`.

All paths below are relative to `notes/mvp_execution_20260916/p1/`, except the plan:

| Artifact | SHA256 |
|---|---|
| `notes/MVP_P1_DEPLOYMENT_PLAN_20260916.md` | `01b42923c94e8a95078166f71c099c23c1b51c94139b38378fe66b349ada99ae` |
| `deploy.py`, `release/deploy.py` | `1c9d2084b48c2c55b014c6fc9586b14561a66376a4ed903035bb39a6eb5b82cd` |
| `p1.sbatch`, `release/p1.sbatch` | `d539d032f39f3759a34d543f1c0c353e2183804878edbd96bb6093f5ee968c1b` |
| `build_release.py` | `e68a4910fd1c65748d2055e9853037cca46bd2d3a8539c0467d0a2d806dad21d` |
| `test_deploy.py` | `edecdf9f3ac6db2ee1a92b7d13fa470ed17b6a5e9987cc1d19260d9a8518bbbf` |
| `release/code.tar` | `2a105e9d2827098d1947c3b8e7e58b689cae0e92bb52e911f100e78b8a3c895c` |
| `release/release.json` | `0ec3776350d4f19226c8fbee30892f28996fa11a604ec0bfaa2e1dcb99f17ae5` |
| `release/release.sha256` | `34a094c321a59ae4e0613c96d6797a2fdca32284682a2ed07074ca5ee52d203b` |
| `release/notes/b6_mvp_technical_candidate_20260915.json` | `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e` |
| `release/notes/b6_mvp_candidate_inventories_20260915.json` | `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98` |
| `release/notes/b6_mvp_candidate_receipts_20260915.json` | `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac` |

The manifest's 85 source-file hashes are incorporated by its exact hash above. Independently compared all 85 archive regular files to immutable Git objects and manifest hashes: PASS. No duplicate members or links/special members. The three accepted JSON companions are separately pinned artifacts, not falsely attributed to d99b. The inspected Track A worktree remained at d99b with no diff in consumed source/archive paths.

## Execution findings

- **Bootstrap and host compatibility:** inspected full helper, builder, bootstrap and tests. `deploy.py` uses Python 3.6-compatible host APIs and grammar; newer annotated project code executes only inside the SIF. Python 3.6 grammar checks passed locally, not an actual Python 3.6 execution claim. Bootstrap requires one node/four GPUs, uses an exclusive persistent started marker, verifies release checksums, sanitizes inherited application/container/Python overrides and launches once (`p1.sbatch:16–40`). The ledger's trusted outer hash is deliberately a controller check, not self-authentication by the ledger.
- **Archive and output isolation:** `deploy.py:150–249` validates tar paths/types/inventory, hashes extracted files, exclusively creates a fresh per-job subtree, copies only listed release files, and freezes copies. No mutable working-tree bytes enter source. Canonical path checks fail closed on aliases. Existing namespace siblings are not deleted or reused.
- **Cache/source preservation:** `deploy.py:120–147` resolves snapshot links within the model root, enforces both observed inventories and fixed revision refs, copies independent bytes and checks before/after/destination hashes. It never chmods or writes the source cache. Runtime cache content is measured in assets rather than assumed certified solely by the revision label. Writable HF dynamic modules, compiler caches, home and temporary files are separate scratch. Real HF behavior against read-only cache remains an intended compute diagnostic.
- **Mounts/environment:** `deploy.py:229–249` passes explicit offline/cache/PYTHONPATH settings after cleanenv/containall. Source code, model input cache, corpus and SIF are read-only. B5 seed123 is read-only and exposed only to S0. The new input tree is writable for checkpoint creation, then frozen and mounted read-only for P1; probe/scratch stay writable. No inherited `TRANSGENIC_RUN_DIR` is intentionally forwarded. Actual Apptainer bind support/site defaults remain runtime checks, not locally proven isolation against hostile image/site configuration.
- **Checkpoint and preparation:** helper `container_stage` matches `stage_checkpoint.py:76–182`: `--dest /mvp/inputs/b5_seed123_epoch05` produces `checkpoint/` plus parent run_config/runtime_identity. The quota path correctly names its parent `/mvp/inputs`. Concrete epoch05 exact inventory, sidecar absence, selected meta/eval and three primary identities are checked. No optimizer/RNG resume or mutable best link is used. `make_launch_readiness.py:186–318,356–379` accepts the supplied CLI; `prepare` publishes bindings only, not six-gate readiness. Its actual `_expected_code_files` closure (`mvp_cli_runtime.py:146–151`) and installed-package manifest run inside the SIF. The archive contains that closure and tooling/config inputs.
- **Actual tile selection:** helper `select_tiles` agrees with `datasets.py:177–487` and `real_probes.py:249–289`: train/Athaliana/strict membership, full-tier geometry and forward/RC pairing, retained evidence at fraction 1, actual tokenizer, nonzero EST, RC channel permutation, and at least one selected N-containing pair. Selection uses accepted RNs, not adjacency guesses or invented fixtures. Both S0 and P1 run the actual measured-batches checks. Missing signal/tier/N or an identity/shape mismatch stops. Availability is not inferred from local geometry pair counts.
- **Quota:** two actual compute quota calls preserve raw output, timestamp and hash. Conservative SI/IEC bounds plus filesystem free must cover 40 GiB reserve + 5 GiB envelope. Stage rechecks timestamp age, available space and ≤2.2 GB copy budget (`stage_checkpoint.py:103–118`). This is a conservative operational check, not an atomic group-quota reservation or a bound on arbitrary future cache growth. Concurrent B5 growth can cause a bounded diagnostic failure; no existing B5 data is overwritten.
- **Timeouts and success semantics:** bootstrap's 1700-second outer deadline + 20-second kill grace fits Slurm's 1800-second allocation limit; S0 work consumes that same budget. P1 invokes the reviewed runner once with 900 seconds, no warmup/retry. `run_bounded.py:80–134` owns and terminates its separately sessioned probe process group, including descendants, on normal/error/timeout paths; Slurm is the final allocation-wide backstop. No intentional detached service, submission, requeue, or training invocation exists in this execution path. This is not a local proof of site signal/cgroup behavior. `run_bounded.py:137–211` publishes DONE only after successful bound measurement validation. Outer interruption can omit FAILED; absent valid DONE is never PASS.

## Evidence move and recovery

**Safe and explicitly documented for this single attempt.** After successful large-file pin checks, `deploy.py:219–222` renames the controller-owned transferred evidence into the fresh job's `inputs/evidence/fullc0lite.db`. It does not move B5 corpus, SIF or source model cache. Failure after the rename leaves the evidence in that per-job tree; the original staging pathname is absent. Freeze may already have made its directory/file read-only.

The plan correctly says to preserve partial output and obtain separately reviewed controller recovery, not clear `S0_P1_STARTED`, remove collision guards, or automatically resubmit. For any recovery, first establish that the old allocation and descendants have ended, locate and rehash the retained evidence, preserve the old job/receipt provenance, and explicitly authorize the new transfer/staging arrangement. This approval supplies **no permission to move it back, overwrite a tree, or reuse a partial S0/P1 result**. A second diagnostic requires separate review/authorization; the current release cannot silently retry.

## Verification limits

Independently reran only focused local verification: seven release-ledger hashes; exact 85-file Git/manifest comparison; helper/bootstrap copy identity; host-helper/builder Python 3.6 grammar; `bash -n`. All passed. Read the nine unit tests and controller's reported PASS; did not repeat the broad suite. No SSH, jobs, GPU/SIF execution, or source edits. Graph queries preceded source inspection; graft auto-refreshed its local metadata. Advisor consultation failed with `claude_code_version_too_old`; no advisor approval is claimed.

Runtime version mismatch, cache/mount/import failure, missing actual tile evidence, OOM or deadline expiry remains possible and acceptable as a **failed diagnostic**, never a substituted/fabricated PASS. This d99b receipt cannot certify future combined resource/decoder code; new consumed code/tooling requires new bindings and review. Full training still requires the final reviewed snapshot, all fresh required gates, and separate budget authorization.
