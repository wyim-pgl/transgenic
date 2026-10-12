# Task2 — bounded CLI and authenticated pull

Implementation commits: `1203a2b` initial GREEN source/tests; `a6066c0` review fixes, deadline protocol, host selectors, docs/example; `b6131cb` final descendant timeout fix. Latest source snapshot for root review/host testing: **3b4eb0d** (actual-host gate rsync fix), following test-only a3220c8. Earlier823bda9 trailing-slash fix remains included. Only owned core/CLI/tests/docs/example files were committed. Root's modified environment/spec/brief notes were not committed. Graft rebuilt successfully (418 files; cache git-ignored).

## Delivered files and core extensions

- `revision/scripts/b6/checkpoint_archive_cli.py`: Python3.6+ stdlib driver and DeltaAI stdin helper.
- `tests/test_checkpoint_archive_cli.py`:26 actual-file cases, real CLI/source-helper subprocesses and narrow executable SSH/rsync/lfs boundaries; no network operations by this worker.
- `deploy/deltaai/checkpoint_archive.example.json`:6 named targets (HDD/pronghorn for B5 and future B6/C2), age300, max1GiB+1checkpoint per target/pass, HDD reserve50GiB, pronghorn reserve1TiB,16MiB/s rate. The1GiB default deliberately excludes production checkpoints until a larger cap/execution location is reviewed. No service installed/started.
- `docs/checkpoint_archive.md`: installation, commands, exact schema, host selection, recovery and transport limitations.
- Core extensions `publish_received` and `pending_cleanup`; original41 core tests retained passing. Publication extracted into private `_publish_stage`, shared with local transfer, preserving receipt-temp quarantine, exact byte/topology verification, fsync and destination-native atomic rename. An internal verified-byte flag avoids redundant prepublication full reads; publication is still verified again after rename.

Public extension signatures:

```python
publish_received(manifest, destination_root, receiver,
                 storage_check=None, required_pool=None,
                 required_project=None, min_free_bytes=0)
pending_cleanup(run_dir)
```

`publish_received` validates manifest, locks destination, computes the existing deterministic target/private stage, checks headroom/placement, refuses unsafe/unexpected previous stage entries, calls `receiver(stage)` under that lock, independently hashes the exact payload/topology, checks every received file's storage, fsyncs payload/receipt/directories and publishes using existing recovery semantics. Existing target is independently verified and never overwritten. The callback is trusted authenticated transport, cannot authorize source deletion and must not recursively acquire the destination core lock.

`pending_cleanup` is metadata-only and validates every persisted unfinished cleanup journal's filename/version/manifest digest/source root/status/receipt binding. Completed journals are omitted. The CLI further filters by explicitly configured source kind and, for `archive`, **exact epoch path**, before budget accounting or recovery; another archive epoch's journal is never selected. Recovery binds archive_path to the configured destination root and independently verifies destination bytes/storage before cleanup. Default dry-run only reports pending journals.

## CLI contract

```sh
python checkpoint_archive_cli.py --config FILE [--target NAME ...] \
  [--once | --watch --cycles N --interval SECONDS] [--apply] [--remove-source]
```

Default is metadata-only plan. `--remove-source` requires `--apply`; watch requires positive bounded cycles/interval. One destination worker lock spans each target/pass, separate from nonreentrant core locks. Payload and checkpoint caps are positive, include pending-removal recovery and apply per target/pass. No unbounded watcher or implicit GPU job/service. Missing source reports waiting without creating it. Errors persist in the final nonzero exit even after a later successful bounded retry.

Example DeltaAI read-only plan:

```sh
/usr/bin/python3 /absolute/tools/checkpoint_archive_cli.py \
  --config /absolute/archive.json --target hdd-b5 --target hdd-b6 --target hdd-c2 --once
```

Example pronghorn read-only plan:

```sh
/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python /absolute/tools/checkpoint_archive_cli.py \
  --config /absolute/archive.json --target pronghorn-b5 --target pronghorn-b6 --target pronghorn-c2 --once
```

Tiny explicit fixture copy/removal:

```sh
python checkpoint_archive_cli.py --config /absolute/fixture.json --target fixture \
  --apply --remove-source --watch --cycles 3 --interval 30
```

Root must replace `REVIEWED_REVISION` in example helper paths with the deployed version. Deploy the core, CLI and `storage_preflight.py` as siblings and compare SHA256; pronghorn pulls over its authenticated `ssh dtai`, using DeltaAI `/usr/bin/python3`.

Runtime schema (unknown fields refused):

```json
{
  "version": 1,
  "targets": [{
    "name": "fixture",
    "transport": "ssh-pull",
    "source_kind": "run",
    "source_root": "/absolute/source/run",
    "destination_root": "/absolute/local/archive",
    "storage": {"kind": "local"},
    "min_age_seconds": 300,
    "min_free_bytes": 1099511627776,
    "max_bytes": 1073741824,
    "max_checkpoints": 1,
    "rate_limit_bytes": 16777216,
    "remote": {
      "host": "dtai",
      "python": "/usr/bin/python3",
      "helper": "/absolute/versioned/checkpoint_archive_cli.py",
      "connect_timeout": 10,
      "timeout": 120
    },
    "rsync_timeout": 120
  }]
}
```

`max_epoch` optionally bounds epoch (nonnegative integer). `rate_limit_bytes` optionally caps stream/rsync rate; rsync requires>=1024 and rounds down to integer KiB/s. `rsync_timeout` defaults120 seconds (I/O inactivity); remote.timeout bounds the complete transport subprocess/ACK channel. Local mode omits remote/rsync fields. DeltaAI policy instead requires:

```json
{"kind":"lustre","required_pool":"ddn_hdd","required_project":22169,"mount":"/work/hdd/bilv"}
```

Lustre probes current parent/stage/every payload file with actual lfs project/getstripe/quota; require directory inheritance P but accept regular file '-' project flag. Space is min(statvfs, project soft/hard pool headroom). Pronghorn local storage policy enforces statvfs/reserve; does not claim per-user quota reservation.

For verified HDD forwarding, explicitly select `source_kind:"archive"`, source_root equal to a concrete published epoch path. Core independently verifies prior receipt/content and includes full upstream provenance; raw epoch names cannot establish archive trust.

JSONL statuses: planned (bounded candidates, protected, refused, metadata-only pending), waiting, verified (full durable receipt), removed (manifest ID), deferred (pending exceeds pass budget), error. No copied-only success is claimed before verification. Remote removal errors carry phase before_ack/after_ack and source_state retained/cleanup_outcome_unknown when the ACK interaction fails.

## Exact helper frames and trust boundary

SSH invocation is strict shell quoting of absolute python/helper paths plus `--helper`; simple host aliases/hostnames only; BatchMode and connect timeout. Requests carry JSON over stdin with64MiB frame bounds, no eval. Ordinary request_for also includes optional `protocol_timeout` from remote.timeout. Fields below are exact; plan max_epoch may be null:

```json
{"action":"plan","source_kind":"run","source_root":"/source/run","min_age_seconds":300,"max_epoch":null,"max_bytes":1073741824,"protocol_timeout":120}
{"action":"prepare","source_kind":"run","source_root":"/source/run","epoch":2,"protocol_timeout":120}
{"action":"validate","source_kind":"run","source_root":"/source/run","manifest_id":"<64hex>","protocol_timeout":120}
{"action":"cleanup","source_kind":"run","source_root":"/source/run","manifest_id":"<64hex>","receipt":{},"ack_timeout":120,"protocol_timeout":120}
```

Successful ordinary response is `{"status":"ok","result":...}`. Plan includes `{waiting,pending,plan}`. Prepare returns the persisted complete manifest. Validate rehashes current source and checks current retention under source/upstream locks. Cleanup loads the source's persisted prepared manifest and matches its full receipt. While coordinator holds destination core lock after independent hashing+fsync, helper sends:

```json
{"status":"ack_required","nonce":"<fresh32hex>","manifest_id":"<64hex>"}
```

Coordinator replies exact `{nonce,manifest_id,receipt}`; helper refuses stale nonce/mismatched receipt/EOF, then core rehashes current source and protections before recoverable journal/tombstone cleanup. Final result is `{"status":"ok","result":{"status":"removed","manifest_id":"<64hex>"}}`. Destination lock is retained through that complete result; tested by attempting destination flock during actual helper unlink. Core locks remain nonreentrant.

FrameChannel uses nonblocking POSIX os.read/os.write + select and one monotonic deadline for complete frames, retains prefetched bytes, bounds reads/writes and rejects oversized data. A partial first byte cannot cause an unbounded readline. A blocked large stdin write also times out. Transport subprocesses use new sessions; stop kills the whole process group even after its leader exits, and uses bounded reap/pipe closure so a descendant cannot keep ordinary transport pipes hanging indefinitely.

Rsync uses -r/--checksum/--protect-args/--no-links/--no-devices/--no-specials, bounded inactivity+absolute subprocess timeouts, optional bwlimit, strict SSH -e quoting and '--' before path operands. Never uses --remove-source-files or --delete. Old archive receipt is excluded; byte/topology verification rejects changed/missing/special payload. Helper validates current source after rsync; source cleanup checks again after live ACK. A transport exit0/serialized dictionary alone is never proof of data durability.

## RED/GREEN and verification

- Initial CLI default-plan RED: absent implementation assertion (2 inherited cases before fixture class separation). Received publication RED: missing publish_received. Pending discovery RED: missing pending_cleanup. Initial11 cases GREEN2.946s, original41 GREEN0.911s.
- Checkpoint cap RED: plan offered [1,2] rather than max_checkpoints1 => fix bounded plan;18 cases GREEN6.921s. Intermediate snapshot1203a2b.
- Post-ACK status RED: SIGKILL failure lacked phase; fixed after_ack/cleanup_outcome_unknown. Target-selector RED argparse unknown option => implemented --target. Lockheldthroughunlink and Lustre regularfileflag/wrongpool/project cases20 GREEN12.202s.
- Independent review archive-scope RED: configuring only epoch02 incorrectly resumed epoch01 journal. Exact source kind/path filter GREEN0.118s.
- Protocol partialframe RED: '{' stalled beyond0.1s deadline; backpressured write RED missing bounded channel. Both FrameChannel cases GREEN0.440s. Full65 GREEN9.358s. Snapshota6066c0.
- Descendant transport RED: leader exited but pipe-holding child left timeout cleanup blocked>1s. Unconditional group termination/bounded reap GREEN0.223s. Snapshotb6131cb.
- Final seeded full suite:66 cases (41 core +25 CLI/extensions) seed17 GREEN9.459s; seed91 GREEN9.310s. Both actual disposable tmpfs files, with varied execution order and genuine helper SIGKILL.
- `python3 -m py_compile revision/scripts/b6/checkpoint_archive.py revision/scripts/b6/checkpoint_archive_cli.py tests/test_checkpoint_archive_cli.py`:exit0, rerun after final fix.
- `ast.parse(...,feature_version=(3,6))`:all3 Python files passed Python3.6 syntax; actual host execution remains root evidence.
- `graft build`:successful418-file graph refresh; graph cache ignored.

Full deterministic command:

```sh
TMPDIR=/dev/shm python3 -m unittest discover -s tests -p 'test_checkpoint_archive*.py' -v
```

Seeded command flattens fresh unittest discovery for that pattern, shuffles with random.Random(seed), runs a fresh suite for seeds17 and91. No source/receipt tests assert implementation strings. Core source extensions tested through actual received files/publication/receipt/corruption and resumable journals. CLI coverage additionally includes exact bytes/emptydirs, best/latest/live state preservation, copy-only then removal/idempotent watch, positive limits, default no hash/no state mutation, missing sources, explicit archived-source provenance, auth/rsync interruption/corruption/source mutation, special path quoting/host option injection, transport timeout, source staleACK/EOF, real SIGKILL tombstone restart/deferred budget, destination corruption, worker exclusion, destination lock during unlink and file-level Lustre evidence.

## Limits and handoff

The source helper trusts the authenticated reviewed coordinator's independent destination verification under lock, not a cryptographic storage attestation. Before ACK, failures preserve source data or prior tombstone. After ACK may have been delivered, source cleanup can be partial/complete despite a lost response; durable destination+journal remain, status reports ambiguity and retry safely rediscovers pending work. Receipt paths and complete manifests must match the configured destination; archive mode cannot recover another epoch.

Published archives must remain immutable until cleanup finishes; arbitrary same-user edits outside this protocol cannot be serialized. Pass caps bound migrated/resumed payload, not repeated checksum I/O. Large approved archives require reviewed larger caps/timeouts/execution location; defaults do not activate production cleanup. No training code/job/remote credentials or production files modified by this worker. Local subprocess transport doubles do not replace root's actual two-host/Lustre/host-Python verification. Root reported fixed1203a2b59 cases passing on actual DeltaAI; finalb6131cb actual host evidence and independent review remain root tasks.

Graft savings from this worker's context calls:1,548,881 estimated tokens (map1,469,615 + skeleton6,572 + storageask40,047 + testskeleton4,420 + quotaask28,227). Report is operational handoff and intentionally uncommitted; source/docs/test/example commits listed above.


## Final scoped review trailing-slash fix

Reviewer reproduced an archive source ending in '/' incorrectly deriving the removed epoch itself as the parent run, causing pending discovery to return waiting. Regression RED: actual receipt-backed archive cleanup interrupted into tombstone, then source_root=published_path+'/' produced only waiting and no removed status. Lexically normalize absolute paths at config/helper/inventory boundaries with normpath (no realpath/symlink following); source/destination filesystem checks retain the core's symlink refusal. Focused GREEN1case0.107s. Final full command above GREEN **67 cases (41 core +26 CLI/extensions) in9.806s**; py_compile rerun exit0. Commit **823bda9**, source/tests only. Earlier seeded66-case evidence remains valid for its exact snapshot. Source frozen after this fix; root real two-host testing and final review continue.

## Actual-host gate fix wave (supersedes previous source freeze)

Root actualDeltaAI d5b3168 gate found the backpressure test's fixed1,000,000byte write did not force backpressure on that host; its pipe can accommodate that amount. Test-only **a3220c8** now fills the real stdout pipe nonblocking in64KiB chunks then single bytes until EAGAIN, restores original blocking flags, and tests a small FrameChannel write. It asserts the exact `protocol deadline exceeded` error rather than accepting any ArchiveError. Focused1case GREEN0.224s; py_compilepassed. Production unchanged for this commit.

Root actualPG end-to-end gate then found an interrupted real rsync transfer left a generated temporary payload filename and normal retry refused the unknown entry. Local actual `/usr/bin/rsync`3.2.7 reproduction RED: hard process timeout2s with32KiB/s rate and1.5MiB source left `.model.safetensors.9i8iXK` plus ordinary payload files in its private stage; retry reported exactly `unexpected received stage contents`. Regression uses a narrow argv adapter converting only host:path to localpath and execs the actual rsync engine, receiver and timeout behavior. Source helper remains the actual CLI via executable SSH boundary. Inactivity timeout10s ensures hard process interruption occurs first; local rsync's graceful1s timeout cleaned temporary files before the2s kill and did not reproduce the host failure.

Production fix **3b4eb0d** uses `--inplace` exclusively for the destination-locked **unpublished** private stage. Incomplete data remains at exact expected manifest paths, then ordinary independent hash/topology/storage/fsync validation controls publication. Existing published targets are never rsync destinations; no published archive is overwritten. Unexpected preexisting payload (including legacy rsync temporary debris) and symlinks remain refused; nothing automatically wipes unexplained stage files.

Before receiver invocation, existing staged payload must have `st_nlink == 1`; multiple hardlinks are refused so in-place transport cannot overwrite unrelated data outside the stage. Root requested this protection. Its actual-file regression RED allowed overwrite/no refusal, then GREEN raises ArchiveError, preserves external linked bytes and source. Receipt control quarantine outside ordinary payload retains its existing reviewed semantics. No receipt/publicAPI schema changes.

Focused real-rsync interruption/retry plus outside-hardlink cases GREEN2cases3.655s. Full deterministic suite **69 cases (41 core +28 CLI/extensions) GREEN16.015s**, with real-rsync case executed (not skipped). All3Python files py_compileexit0. Documentation updated for unpublished-stage in-place operation and hardlink/unknown-payload refusal. Latest **3b4eb0d** source frozen for independent scoped review and root actual host reruns. Only owned files committed; no remote operations/production changes by worker.

Python3.6-compatible focused host command, cwd deployed repository:

```sh
TMPDIR=/dev/shm PYTHONPATH=tests /usr/bin/python3 -m unittest \
  test_checkpoint_archive_cli.CLITests.test_backpressured_protocol_write_obeys_deadline \
  test_checkpoint_archive_cli.CLITests.test_real_rsync_interrupted_private_stage_retries_without_payload_debris \
  test_checkpoint_archive_cli.ReceivedTests.test_received_stage_hardlink_cannot_overwrite_unrelated_file -v
```

Graft context savings this fixwave137,319 estimated tokens; cumulative worker1,686,200.
