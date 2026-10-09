# Checkpoint archive requirements — 2026-10-09

User authorized implementation and repeated small-file login-node tests, then explicitly added transfer to pronghorn. This supersedes the earlier advice-only scope. Preserve every checkpoint's data: remove a source only after a durable, independently hash-verified archive exists. Do not change training math, existing trainer source, queued Slurm jobs, or the two separately documented finalization bugs in this feature.

## Platforms and scope

- Python 3.6+ standard library on DeltaAI; no torch or package installation required for archive tools.
- Source runs: B5 seed123 and B6/C2 WP5 runs. Immutable completed `epoch_NN` directories only in v1. Never archive `latest_state*`, `.tmp`, `step_*`/PILOT_FINAL, latest two completed epochs, or the `best` target. These best/latest protections apply to training-run inputs. An explicitly configured archive input may forward an already-verified immutable HDD archive after independently verifying its prior receipt and preserving upstream provenance; never infer archive status solely from its directory name.
- Copy entire epoch (weights, optimizer, RNG, pending gradients, metadata). Reject symlinks/special files inside checkpoint trees, overlapping roots and path traversal. Treat malformed checkpoint metadata as refusal, not proof of eligibility.
- Destination backends: DeltaAI HDD (actual Lustre `ddn_hdd` pool and project quota verified) and pronghorn (pull from DeltaAI over existing `ssh dtai` connection; source is NVMe or an already-verified HDD archive). Source deletion remains opt-in at CLI level; ordinary invocations only plan/dry-run.
- Actual 99 GiB production cleanup is not a test. Deploy and run read-only production inventory plus small disposable actual transfers. Do not enable unbounded production deletion on a login node.

## Required behavior

1. Plan deterministically from complete epoch metadata, keep newest two and best, apply minimum age and optional maximum epoch/bytes bounds.
2. Exclusive OS locks scoped to destination root and source run prevent competing archive workers. Fail closed if locking is unsupported. Real fcntl.flock worked on both DeltaAI pools with a second process correctly blocked on 2026-10-09.
3. Copy new files into an HDD-native staging directory; rename across NVMe/HDD is forbidden as migration (same filesystem preserves pool). Stream I/O with bounded memory, optional rate limit, compute SHA256, fsync copied files/directories, independently re-read destination, verify source stability/hash, write versioned receipt, atomically publish within destination.
4. A partial transfer, missing file, mismatched hash, changed source, conflicting existing destination, wrong pool/project, low quota, or authentication failure must leave source intact. Retrying verified identical archive is idempotent; conflicting archive never overwritten.
5. Source cleanup only after verified durable publication and another protection check. Use a recoverable transaction/tombstone for interruption during removal. Never delete unrelated files. Receipt binds canonical source root, epoch, inode/identity as appropriate, and all relative file hashes/sizes.
6. Pronghorn pull must use argv/strict shell quoting, BatchMode with bounded connection time, rsync without --remove-source-files/--delete, and local checksum verification. Source helper validates destination acknowledgment and current source against its prepared inventory before opt-in cleanup. Connection loss keeps source. Existing remote data is never overwritten blindly. Use a persisted manifest/receipt protocol rather than trusting rsync exit0.
7. Watcher/config supports multiple runs and both destinations, `--once` or bounded interval loop, JSON status/logs, nonzero exit on transfer errors, explicit apply/remove switches, minimum free space/maximum bytes and one-worker limits. Large I/O must not be enabled accidentally by dry-run. No self-submitted GPU jobs.
8. Keep production activation separate from installation. A production config/example and exact launch instructions make the feature usable after review. If host service credentials expire, show a blocked/error result while keeping sources; never silently claim migration.

## Validation

Real temporary files, not mock-only tests: selection/protection, byte-for-byte copy and manifest, restart after copy/publish/remove failures, corruption, source mutation, quota/pool mismatch, concurrent workers, symlink and overlap refusal, CLI dry-run/once/watch, remote failure and path quoting. Repeat seeded order/fault tests. Test on actual DeltaAI Python3.6 and Lustre tiny directories; test actual pronghorn pull with a tiny DeltaAI fixture. Record evidence and live quota before any real bulk migration.
