# B6 headroom archive review — 2026-10-20

Verdict: **APPROVE WITH FIXES — one P2 scope-binding issue** in the operation wrapper. The unchanged archive implementation at `3b4eb0dfe0e83463775ce8e1ab05fa2f62d1e20b` passes this scoped review; its three pinned helper hashes match the local clean worktree. This review does not claim that the not-yet-created HDD destination, immutable remote staging, or a submitted allocation has passed execution checks.

## Spec

**P2 — Restrict each destructive call to the pinned epoch, rather than relying on a one-time candidate check.** `archive_guard.py:61–68` checks that the initial candidate set is exactly epochs09–11 and that no cleanup is pending. `archive.sbatch:20–24` then invokes the general CLI three times. Each invocation independently replans; `checkpoint_archive_cli.py:432–475` also automatically recovers pending cleanup before processing new candidates. Its `max_epoch=11` bounds candidate planning, but does not bound that pending-cleanup loop. Another archive operation between the initial check and a later pass could introduce a different pending transaction; the post-check detects an unexpected result only after removal has happened. The requirement is to archive **only** epochs09–11, and no cross-host exclusion for the whole wrapper operation has been established.

Fix: call the unchanged reviewed `prepare`, `transfer`, and `cleanup` APIs for each explicitly pinned epoch09/10/11. Before transfer, compare the prepared manifest's source/epoch/name/bytes to the pinned item. Use the existing CLI `storage_options` for pool/project/quota checks, preserve the rate/reserve limits, and publish per-epoch manifest/verified-receipt/cleanup-result artifacts exclusively. Do not enter generic pending-cleanup recovery. A failure must stop for inspection with its exact transaction artifacts preserved. The parent has selected this fix; its implementation needs a scoped re-review.

Otherwise the requested scope is represented correctly: source B5 seed123 only; epochs12–14/latest/best14 protected; all C2 paths excluded; 42,370,570,227 bytes total; 16GiB/one-checkpoint envelope; 16MiB/s copying; 50GiB HDD reserve; 1CPU/2GiB/2h allocation without a GPU request. Historical archival authorization covers the eligible completed checkpoints and no training recipe is changed.

## Quality and safety

No additional blocking findings. The reviewed core rejects unsafe paths and changed sources, checks actual destination-file Lustre pool/project as well as quota, independently hashes the copied payload, fsyncs files/directories and receipts, verifies durable publication again before cleanup, and uses recoverable source tombstones. The wrapper checks helper/config pins and protected metadata before and after, refuses an existing preflight via exclusive creation, and uses `set -euo pipefail` so a CLI failure cannot produce a completion record. Interrupted or uncertain submission must be reconciled by job/receipt evidence, never blindly retried.

Deployment conditions remain: create and verify the new destination's `ddn_hdd` layout/project22169 inheritance before payload; freeze and verify the operation manifest including the actual submitted script; record exclusive submission intent and actual Slurm job identity; confirm job/batch exits and completion artifacts; refresh project quota afterward. These are execution checks, not requests for new user approval. Copying alone has a 42.1-minute lower bound at the selected rate; the two-hour allocation also includes multiple complete hash reads. Timeout leaves preserved archives/transactions and must not be accepted as complete.

## Verification and reviewed artifact pins

- `bash -n archive.sbatch`: PASS.
- Python AST parsing and both JSON documents: PASS.
- Config digest equals `pins.json`; expected byte sum is exactly42,370,570,227.
- All three archive helper hashes equal clean `3b4eb0d` worktree files.
- No remote mutation, submission, transfer, source deletion, or implementation edit performed by this reviewer.

| Artifact | SHA256 |
|---|---|
| archive.json | `2309d1f5213a6ecf6bee4133aaf43601bcc9130d21d71707c696129c2aa6e386` |
| pins.json | `913b827b0dc3b1eb40c88d05706d827aeb4fc5ce541fc5469d897657425fc511` |
| archive.sbatch | `c6af4e3b3611b8a920d9f7d17c64389a73824242a28070bd701c4a1ff2071cf6` |
| archive_guard.py | `218bfea2b558a901544152c3006011051d3f8b5f8a34de07a23e4e2a93dac688` |
| PREFLIGHT_20261020.txt | `304a6f321a642350e40d3ba2f2c39753ebda839d1614a43e6ae074454b9a353b` |

## Scoped fix review — approved

**Current verdict: APPROVE. The original P2 is fixed; no new blocking findings.** This supersedes the initial wrapper verdict above. Runtime destination/staging/submission checks remain required as already described; this approval is not a claim of completed archival.

The batch now invokes only literal epochs9,10,11. Each transfer checks the requested epoch against the pinned list, refuses unresolved cleanup, prepares exactly that epoch through the unchanged reviewed core, and compares its epoch/name/bytes to the pinned candidate before any transfer or removal. It then uses the existing CLI storage policy with `core.transfer`, publishes the verified receipt exclusively, and calls `core.cleanup` with that exact manifest/receipt. No generic CLI replanning or pending-transaction recovery remains. Per-epoch artifacts and final cross-checks bind the prepared manifest, published receipt, cleanup result, source disappearance and same-allocation preflight. The protected-epoch loop no longer shadows the requested epoch.

Independent reviewer verification: `python3 notes/b6_pilot_resume_20261020/archive/test_archive_guard.py -v` **PASS**, 1 test,62.514s, exit0. Real temporary files exercised the added epoch08 after preflight, refusal of epochs8/12, fixed09–11 copying/publication/cleanup, preservation of08/12–14/latest/best, changed latest-state refusal, malformed cleanup-receipt refusal before completion, and independent destination rehashes. The test uses the local storage backend; actual Lustre enforcement remains the unchanged pinned core plus prospective host/allocation checks. Shell syntax, AST parsing, config binding and all three unchanged helper hashes were rechecked and passed. No broad unchanged-helper suite was repeated.

Updated reviewed pins:

| Artifact | SHA256 |
|---|---|
| archive.sbatch | `ff74368b9eb1eda69c193f7bcd43affd9d424c7b1079775d9fe3f24851e4bbd7` |
| archive_guard.py | `e1d8aa0d24ba142403b6b8b1609d1ad8156fb3d6348fcbeed899a7dc817c1623` |
| test_archive_guard.py | `f016c4f18fed9396e92cb1ade8c4c7b924d3820364d1911d4bb52ba9276a13a8` |

Config and pins.json hashes remain unchanged from the first review. The required earlier `graft ask` completed during review and automatically refreshed one graph node; no implementation or remote files were edited by the reviewer. Only this review report was intentionally written in the repository.

## Resource-only review — v2 allocation approved

**Verdict: APPROVE.** The sole script change is `#SBATCH --gpus-per-node=1`; removing that line reproduces the previously reviewed script hash exactly. The job remains one node, one CPU,2GiB RAM, two hours, `ghx4-interactive`, no requeue, and stdlib archive file I/O only. The fixed-epoch wrapper, config, pins and regression hashes are unchanged, so the transaction test was not repeated.

`STAGING_v1_cpu_only_rejected.json` records an actual `sbatch --test-only` rejection: “Jobs requesting no GPU resources cannot be submitted on DeltaAI.” It records `submitted=false`; no scheduler job was created. The one-GPU request accommodates that measured site requirement while retaining the existing archive purpose and bounded duration. It reserves at most two GPU-hours of hardware time, with charged account hours potentially differing under interactive billing; the supplied current balance is1,641 account hours. This is within the already authorized archival scope, not a new training run or a reason to request the same authorization again.

The same staging receipt verifies the actual destination `/work/hdd/bilv/wcyim/transgenic/checkpoint_archive` as pool`ddn_hdd`, project22169/inheritance`P`, and records the host-Python3.6 tiny transaction test PASS (1 test,0.123s). These observations supersede the earlier “destination not yet created” statements in this review and verification chronology. The actual payload files will still undergo the unchanged core's per-file storage checks inside the allocation.

Resource-review validation: `bash -n archive.sbatch` PASS; byte comparison confirms the one-GPU directive is the only script delta; guard/config/pins/regression hashes match the scoped approval. Freeze this script in the new immutable `ops/b6_headroom_20261020_v2`, leave the rejected v1 unused, rerun scheduler preflight, and retain the submission-intent/uncertain-result reconciliation policy. No job is claimed submitted or complete by this review.

**Approved v2 `archive.sbatch` SHA256:** `be75679f38b0841165eb841771670218097767b6bc90fa1710131294f7606278`.
