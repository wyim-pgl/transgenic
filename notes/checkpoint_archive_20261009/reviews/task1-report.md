# Task1 checkpoint archive core

## API contract for Task2

Import `revision/scripts/b6/checkpoint_archive.py`; stdlib only, Python3.6+.

- `plan(run_dir, min_age_seconds=0, max_epoch=None, max_bytes=None, now=None)` returns JSON `{version, source_root, candidates:[{epoch,name,bytes}], protected:[epoch], refused:[{epoch,name,reason}], bytes}`. Metadata/stat-only: no weights read/hash. `max_bytes` bounds total selected bytes. Raw core default age0 is deliberate for fixtures; production driver must set age300 or reviewed policy. Latest two validated completed epochs and best target protected; invalid best link causes global refusal. Only exact `epoch_[0-9]+` names qualify.
- `prepare(run_dir, epoch)` validates current eligibility, independently hashes entire raw checkpoint, checks stability, and persists source-side manifest under `.checkpoint_archive/manifests/<manifest_id>.json`. Returns manifest `{version, source_root, run_identity:[dev,ino], name, epoch, source_identity:[dev,ino], files:[{path,size,signature:[dev,ino,size,mtime_ns,ctime_ns],sha256}], directories:[relative path], bytes, manifest_id}`. Digest covers canonical JSON excluding manifest_id.
- `prepare_archive(archive_path)` explicitly selects **verified archive storage**, rather than training-run input. It independently verifies the prior receipt and all files, embeds full `upstream_receipt`, `upstream_signature`, `upstream_lock_root` in the new manifest and excludes the old reserved receipt file from ordinary payload. Active-run best/latest2 rules do not apply to this explicit archived-storage mode. Unverified storage cannot use this API successfully. Task2 config must distinguish `source_kind=run` and `source_kind=archive` explicitly.
- `archive_path(manifest, destination_root)` computes `<destination_root>/run-<SHA256(canonical_source_root)[:16]>/<epoch_name>`. Unique canonical run roots cannot collide accidentally. No silent overwrite.
- `transfer(manifest, destination_root, storage_check=None, required_pool=None, required_project=None, min_free_bytes=0, rate_limit_bytes=None)` returns receipt `{version, manifest_id, manifest, archive_path, storage}`. Full bounded streaming copy into destination-native `.stage-<manifest_id>`; creates files instead of source cross-pool rename. Rechecks source and best/latest2, independently rehashes destination, fsyncs files+dirs+receipt, renames within destination and fsyncs publication parents. A retained identical published archive is rehashed and barriers reestablished on retries; conflicting content/receipt is refused. Partial stages resume by hashing completed files and replacing only expected partial regular files. Unexpected files/dirs fail verification. Does not remove source.
- `storage_check(actual_path)` is an optional caller-supplied authoritative host boundary returning `{pool, project_id, quota_free_bytes, free_bytes?}`. For restrictions, missing pool/project/quota evidence fails closed. Runs for parent, stage, every copied or reused destination file, so directory layout cannot alone establish actual HDD placement. Core always checks `statvfs` as well; headroom is min(filesystem, supplied quota/free) and required bytes+reserve. CLI must require `ddn_hdd` plus project ID and validated quota on DeltaAI HDD. Generic local filesystem mode without these restrictions is for local/pronghorn fixtures and authenticated independently probed destination setup.
- `verify_archive(manifest, archive_path)` independently reads every file and directory; verifies exact persisted receipt binding/path/version; fsyncs files, receipt, directories and publication parent before returning receipt. It excludes only the root reserved `.archive_receipt.json` from checkpoint payload. Never trusts a `verified` flag.
- `cleanup(manifest, receipt, verifier=None)` is explicit removal. Local default independently rehashes and fsyncs destination and acquires source/destination root locks. Remote `verifier(manifest, receipt)` is a **trusted authenticated transport boundary** that MUST independently verify remote destination and return the exact durable receipt; a receipt assertion alone is insufficient. Remote source cannot prove pronghorn bytes cryptographically through this dictionary: Task2 authenticated pull coordinator must provide the independent destination verification acknowledgment and source helper must revalidate manifest/current source. Stale/mismatched return fails closed.
- Cleanup persists `.checkpoint_archive/cleanup/<manifest_id>.json`, then renames source to `.archive-removing-<manifest_id>` within source run. It rechecks source identities/content and current protections before rename and each unlink. Every removal fsyncs its parent. Restart hashes and validates all remaining tombstone contents against the prepared manifest, revalidates destination acknowledgment, and completes only expected paths. A new/replaced source directory or unexpected tombstone file is refused. Full prior receipt for archive-storage source is removed last. Completed retry returns `{status:'removed', manifest_id}` and never removes a replacement directory.
- `locked(*roots)` obtains nonblocking `fcntl.flock` in canonical root order on persistent `.checkpoint_archive.lock` files. Unsupported/busy locks raise `ArchiveError`. Lockfiles never unlinked. Mutating public APIs acquire locks themselves; callers must not hold the same lock while invoking them because flock is intentionally non-reentrant. Archive-storage mode also locks upstream destination root. Standalone verification is read-only plus durability barriers and does not acquire root locks; published archives must be immutable, and callers must serialize destructive storage operations.

Minimum complete raw checkpoint: nonempty root and accelerate_state model.safetensors, optimizer.bin, scheduler.bin; meta.json with nonnegative integer epoch/global_step/seed/pending_micro; matching eval.json epoch and global_step if present; nonempty contiguous per-rank random_states_0..N.pkl; pending_micro>0 requires each rank's `pending_grads/rankN.safetensors` (B5) OR `pending_grads_rankN.pt` (B6). All additional files and empty directories preserved. Symlinks/special files, reserved receipts in raw sources, overlapping roots, traversal/duplicate manifest paths refused.

## TDD / verification evidence

1. Tests written before implementation. Focused RED: `python3 -m unittest discover -s tests -p test_checkpoint_archive.py -k plan_protects -v` failed with `archive implementation is missing` (1 test, assertion failure).
2. First GREEN: full initial suite 12 tests passed in21.293s.
3. Second-tier RED: focused `-k previously_verified` showed missing `prepare_archive` API; implementation then expanded suite23 passed in46.160s.
4. Durability RED: focused `-k durability_barriers` asserted disk barrier failure must stop successful verification and failed because no OSError was raised. Added independent destination fsync barriers, repairing process death immediately after publication rename.
5. Expanded full suite31 tests passed in61.191s. Real subprocess SIGKILL tested mid-copy, immediately after publication rename, and after first source tombstone unlink; each restarted safely. Boundary-injected OS failures test copy ENOSPC, publication failure, fsync failure and unsupported locking. Actual cross-process lock contender exits nonzero. Other tests cover byte preservation/pending gradients, source mutation before copy/cleanup, stale remote acknowledgment, current best/latest2 changes, wrong pool/project/quota, file pool mismatch, malformed/incomplete metadata/RNG, corruption/conflicting receipt, idempotency, symlinks/overlap/traversal, metadata-only plan, and second-tier provenance.
6. Final rerun after self-review changes passed31 tests in64.592s; Python3.6 syntax parsing also passed on local Python. Actual Python3.6 execution remains host verification. `python3 -m py_compile revision/scripts/b6/checkpoint_archive.py tests/test_checkpoint_archive.py` passed.

No remote operations, production checkpoint mutation, job submissions, trainer edits, watchers or SSH added by this task. Host Python3.6/Lustre pool/fsync verification and authenticated pronghorn protocol are root/Task2 responsibilities. All fixture files are disposable. No crash simulation switches exist in production API/CLI.

## Limitations / review focus

Storage correctness depends on an authoritative storage_check callback, and remote receipt acknowledgment is a trusted authenticated transport boundary, not cryptographic proof. Published archives must remain immutable until their source cleanup finishes. A interrupted cleanup may leave a receipt-bound tombstone; cleanup(manifest,receipt) resumes it. Protection change or corruption during an interrupted removal leaves the remaining tombstone for investigation; it never guesses missing/changed data safe. Same-user uncontrolled edits during file unlink cannot be fully serialized against a trainer that does not take the archive lock; v1 operates completed immutable epochs, with repeated identity/hash/protection checks.

Graft context calls saved approximately135509 tokens total (1113+6516+49468+53533+24879).

Commit: `9ecd8a2` (source/tests only), feat: add durable verified checkpoint archive transactions.

Concrete Task2 integration extension needed: `transfer` requires local source access. An authenticated pronghorn pull needs a reviewed `publish_received` helper (or extraction of existing publication logic) to independently validate native destination staging, establish durability and publish the same receipt without source-local access. The source cleanup helper separately checks prepared manifest against current source. This fits PLAN Task2 allowance for reviewed API extension; do not improvise trust in rsync exit0.


## Independent review fix round1

Review: `.superpowers/sdd/PLAN/task1-review.md`, both Important findings verified with failing real-file tests before fixes.

- Shared `_classified_epochs` validates completion and rejects duplicate numeric aliases; shared `_protected_epochs` retains distinct latest-two completed numbers plus best. `plan` and every `_eligible` check use that definition. Tombstone cleanup includes its receipt-bound prepared epoch when checking retention.
- `_recover_receipt_temp` accepts only the exact stage-owned `.archive_receipt.json.tmp`, regular and unchanged during reading, with a complete canonical receipt schema bound to the current version, manifest ID, full manifest and destination path. It removes that recognized file and fsyncs the stage before normal payload verification. Invalid JSON/unrelated contents/symlinks remain refused; published conflict behavior unchanged. Temporary receipt storage evidence is not trusted for current placement/quota; transfer's current probes still apply. Partially written or malformed temporary receipt data remains a fail-closed refusal requiring investigation, because its complete transaction binding cannot be established.
- Public function signatures and manifest/receipt schema unchanged.

RED commands (`TMPDIR=/dev/shm` for disposable real files):

1. `python3 -m unittest discover -s tests -p test_checkpoint_archive.py -k incomplete_newer -v`:2 failures, candidates were `[2,3]` instead of `[2]`, and tombstone cleanup wrongly proceeded.
2. `... -k new_incomplete -v`:1 failure, cleanup did not raise after completed04 removal and incomplete05 arrival.
3. `... -k duplicate_numeric -v`:1 failure, cleanup proceeded with duplicate epoch04 aliases.
4. `... -k interrupted_receipt_write -v`:1 error, archive content mismatch on retry after the real temporary receipt was fsynced but replacement raised OSError.

GREEN:

- `TMPDIR=/dev/shm python3 -m unittest discover -s tests -p test_checkpoint_archive.py -v`:37 tests passed in0.704s.
- Focused `-k incomplete`:4 passed in0.069s; `-k receipt_temp`:1 passed in0.029s; `-k interrupted_receipt_write`:1 passed in0.031s.
- `python3 -m py_compile revision/scripts/b6/checkpoint_archive.py tests/test_checkpoint_archive.py`:exit0.

No host/production operations performed. Added6 tests (retention3, numeric aliases1, interrupted receipt1, unrecognized/symlink receipt1). Original31 real-file tests and real SIGKILL tests remain passing. tmpfs runs verify transaction logic; host Lustre durability/pool testing remains root verification. Root reports original9ecd8a2 all31 also passed on actual DeltaAI Python3.6.15 in0.488s, before these fixes.

Fix round graft savings57450 (51433+6017); task total192959. Fix commit recorded below after source/tests-only commit.

Fix round1 commit: `b80c638` (source/tests only), fix: retain completed checkpoints and recover staged receipts.

## Independent review fix round2

Review round1 correctly found routine SIGKILL-before-write/partial-write receipt temps still blocked retry. This supersedes the earlier intentional truncated-temp refusal limitation.

`_recover_receipt_temp` now recognizes ownership through the exact private `.stage-<manifest_id>/.archive_receipt.json.tmp` control pathname, never through trusting incomplete JSON. A source manifest that names that pathname as checkpoint payload is refused. Only regular files qualify; symlinks/special files remain refused. The original bytes are independently hashed, preserved using exclusive hard-link creation to a fresh UUID name under the destination run's `.receipt-recovery-<manifest_id>/` control directory, rechecked for identity/content stability, and fsynced together with the control directory. Only then is the stage's temporary name unlinked and its directory fsynced. This preserves empty/truncated/unrecognized bytes outside checkpoint payload and never overwrites existing files. A kill between link and unlink preserves both names and permits another safe retry.

Normal transfer still independently checks every checkpoint payload file against the known manifest, rehashes the current source, rechecks protections, reconstructs a fresh receipt from verified state/current storage probes and publishes durably. Quarantined receipt assertions are never used. Unexpected ordinary payload remains refused; published archives are never repaired/overwritten. Public APIs and JSON schema unchanged. Preserved small receipt control files remain for inspection; no automatic quarantine deletion introduced.

RED:

- `TMPDIR=/dev/shm python3 -m unittest discover -s tests -p test_checkpoint_archive.py -k sigkill_before_receipt -v`:child performed real receipt-temp open then died bySIGKILL, leaving size0; normal retry errored with JSONDecodeError/ArchiveError.
- Same command with `-k sigkill_during_receipt`:child wrote+fsynced real partial JSON then died bySIGKILL; normal retry errored with invalid temporary receipt.
- Same command with `-k unrecognized_receipt`:recovery incorrectly refused ambiguous `{}` instead of preserving it separately and reconstructing receipt.

GREEN:

- `TMPDIR=/dev/shm python3 -m unittest discover -s tests -p test_checkpoint_archive.py -k receipt -v`:7 passed in0.276s, including full temp replacement restart, empty/truncated SIGKILL restart, unrecognized bytes preservation, symlink refusal, arbitrary extra payload refusal and conflicting published receipt preservation.
- `TMPDIR=/dev/shm python3 -m unittest discover -s tests -p test_checkpoint_archive.py -v`:41 tests passed in1.041s.
- `python3 -m py_compile revision/scripts/b6/checkpoint_archive.py tests/test_checkpoint_archive.py`:exit0.

Root host verification must exercise the new exclusive hard-link quarantine boundary on actual Lustre. No host or production operations performed by this worker. All local fixture files actual disposable files under/dev/shm. Graft round2 saved40640; cumulative Task1 estimate233599. Source/test-only fix commit recorded below.

Fix round2 commit: `8ff381b` (source/tests only), fix: preserve and recover interrupted receipt control files.
