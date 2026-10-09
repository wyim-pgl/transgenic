# Checkpoint archive operation

The Python 3.6+ standard-library tools copy complete immutable `epoch_NN` directories and preserve all weights, optimizer/RNG state, pending gradients, metadata, extra files and empty directories. Run inputs protect the newest two **complete** epochs and the current `best` target. Incomplete epochs do not weaken retention. Symlinks, special files, conflicting existing archives and overlapping local roots are refused. Training state such as `latest_state`, temporary and step/final directories is outside the archive selection.

Installation does not start a service or authorize bulk production cleanup. The example uses a **1 GiB payload cap per target/pass**, one checkpoint/pass, age300 seconds, HDD reserve50 GiB and pronghorn reserve1 TiB. Production checkpoints exceeding that cap remain unselected; increase the cap only after reviewing their sizes, live quota and the execution location. Login nodes are for repeated small disposable tests. A pass budget limits payload archived or resumed for deletion, rather than total bytes read during repeated hash verification. These tools do not change or submit training jobs.

## Install and plan

Deploy these three sibling files together into a versioned operational directory on both hosts, preserving their reviewed contents:

- `revision/scripts/b6/checkpoint_archive.py`
- `revision/scripts/b6/checkpoint_archive_cli.py`
- `revision/scripts/b6/storage_preflight.py`

Compare `sha256sum` for all three against the reviewed checkout. Copy `deploy/deltaai/checkpoint_archive.example.json` to an operational config. Replace `REVIEWED_REVISION` in each SSH helper path with the installed revision. The remote helper runs on DeltaAI; the pull coordinator runs on pronghorn, where `ssh dtai` must already authenticate without prompts. No credentials are stored in the config or printed in transport errors.

On DeltaAI, plan only HDD targets:

```sh
/usr/bin/python3 /absolute/tools/checkpoint_archive_cli.py \
  --config /absolute/archive.json --target hdd-b5 --target hdd-b6 --target hdd-c2 --once
```

On pronghorn, plan only pull targets:

```sh
/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python /absolute/tools/checkpoint_archive_cli.py \
  --config /absolute/archive.json --target pronghorn-b5 --target pronghorn-b6 --target pronghorn-c2 --once
```

`--target` may be repeated. Omitting it selects all configured targets, so host-specific selection is recommended for the combined example. Missing future B6/C2 run directories report `waiting` and are not created. Default execution is a metadata-only plan: it does not hash checkpoint binaries, create locks/stages/manifests, copy files or resume cleanup. Existing pending cleanup journals appear in the plan without modification.

Copy with `--apply`. Separately add `--remove-source` to authorize cleanup after durable verification. Removal without `--apply` is an error. For a bounded fixture watcher:

```sh
/usr/bin/python3 /absolute/tools/checkpoint_archive_cli.py \
  --config /absolute/fixture.json --target fixture --apply --remove-source \
  --watch --cycles 3 --interval 30
```

There is no unbounded watch mode. `--watch` requires a positive cycle count and interval; `--once` is the default. Any failed target/pass makes the final exit code nonzero, even if a later bounded retry succeeds. Watch waits its configured interval after errors. Explicit production activation must choose reviewed byte/checkpoint caps, free reserve, timeout and I/O execution location; the fixture command does not activate the supplied production example.

## Config schema

The top-level JSON object is `{ "version": 1, "targets": [...] }`. Unknown fields fail closed. Each target requires:

| Field | Meaning |
| --- | --- |
| `name` | Unique target selector/log name. |
| `transport` | `local` or `ssh-pull`; destination is always local to the coordinator. |
| `source_kind` | Explicit `run` or `archive`. |
| `source_root` | Absolute run directory, or exact published epoch path for `archive`. |
| `destination_root` | Absolute local archive root. |
| `storage` | `{ "kind": "local" }` for pronghorn/local fixtures, or Lustre policy below. |
| `min_age_seconds` | Nonnegative checkpoint age; production example300. |
| `min_free_bytes` | Nonnegative reserve after payload transfer; always combined with `statvfs`. |
| `max_bytes` | Positive payload cap per target/pass, including pending removal recovery. |
| `max_checkpoints` | Positive checkpoint cap per target/pass, including recovery. |

Optional `max_epoch` is a nonnegative integer; `rate_limit_bytes` is positive. Pull requires at least1024 bytes/s when configured because rsync uses integer KiB/s; conversion rounds down. Local copies rate-limit streamed reads/writes. An SSH target additionally requires:

```json
{
  "remote": {
    "host": "dtai",
    "python": "/usr/bin/python3",
    "helper": "/absolute/versioned/checkpoint_archive_cli.py",
    "connect_timeout": 10,
    "timeout": 120
  },
  "rsync_timeout": 120
}
```

`rsync_timeout` defaults120 and controls rsync I/O inactivity; `remote.timeout` also bounds each full SSH or rsync subprocess and the acknowledgment exchange. Large approved checkpoints need appropriately longer explicit timeouts for full source/destination hashes. A timeout is a failure, never permission to skip verification. Hosts must be simple SSH aliases/hostnames optionally containing `user@`; leading options and shell characters are rejected. Python/helper paths are absolute and shell-quoted; requests and acknowledgments are JSON on stdin, never evaluated code.

The DeltaAI HDD policy is:

```json
{
  "kind": "lustre",
  "required_pool": "ddn_hdd",
  "required_project": 22169,
  "mount": "/work/hdd/bilv"
}
```

The CLI probes actual `lfs project`, `lfs getstripe` and project pool quota for destination parents/stages and each payload file. Directories must inherit their project (`P`); regular files require the correct numeric project ID without the directory flag. Soft/hard quota headroom and filesystem free bytes must cover the transfer plus reserve. Files are created under the HDD-native stage, then renamed within that destination. An NVMe→HDD rename would preserve SSD placement and is not used. For pronghorn, `statvfs` plus the configured conservative reserve is enforced. That check does not reserve space or claim a per-user quota; host/fileset quota evidence remains part of deployment verification.

For second-tier forwarding, set `source_kind` to `archive` and `source_root` to the exact receipt-backed HDD epoch path. `prepare_archive` independently verifies its prior receipt and all bytes, preserves that receipt as upstream provenance and excludes the reserved old receipt from the new payload. Naming a raw run epoch as `archive` is insufficient. Run retention applies to run inputs; receipt-backed archived storage can be forwarded after verification.

## Durability and cleanup protocol

The deterministic target is `<destination_root>/run-<SHA256(canonical_source_root)[:16]>/<epoch_name>`. Source preparation persists a manifest binding canonical root, identities, all relative file paths, sizes, signatures and SHA256 hashes. A private `.stage-<manifest_id>` is owned by that transaction. Local copy and received-stage publication share the same receipt recovery and publication logic. Every payload is independently re-read, fsynced, bound to a versioned receipt and atomically published within the destination. Existing identical publication is independently verified again; conflicting bytes or receipts are never overwritten. Interrupted receipt control files are safely preserved outside the payload for inspection.

Pull uses BatchMode SSH, bounded timeouts and rsync `--protect-args`, without source-removal or destination-deletion options. Rsync success alone cannot authorize cleanup. The coordinator verifies and fsyncs the durable archive, holds the destination root lock, then asks the authenticated source helper for a fresh nonce challenge. It replies with the exact nonce, manifest ID and independently verified receipt while retaining the lock through the removal result. The helper loads its persisted prepared manifest, rejects stale/mismatched acknowledgment and rehashes current source/protection under source locks before cleanup. JSON receipt/acknowledgment dictionaries are bindings across a trusted authenticated transport; they are **not cryptographic proof of remote storage**.

Before a live acknowledgment, transport/authentication failure preserves source data or an already-existing tombstone. After an acknowledgment may have been delivered, disconnect/timeout can leave a partly or fully removed source; the verified durable destination remains and the source journal makes retry recoverable. Those errors include `phase: "after_ack"` and `source_state: "cleanup_outcome_unknown"`; do not interpret them as source retention. A retry discovers unfinished journals even when the epoch has already been renamed. It binds the receipt to the configured destination root, rechecks storage and all destination bytes, then resumes only expected source paths within the pass budget. Protected or changed sources/tombstones are refused for investigation. Ordinary dry-run never resumes removal.

Destination/source locks are persistent, nonblocking `flock` files. The CLI also holds a separate destination worker lock for the entire target pass; another compliant worker fails closed. Core public mutating APIs acquire their own locks and must not be called while holding the same core lock. Published archives are immutable until cleanup completes; arbitrary same-user edits outside this locking protocol are outside that guarantee.

## Status and testing

Stdout is JSON Lines: `planned` includes bounded candidates/protected/refused and metadata-only `pending`; `waiting` denotes an absent source; `verified` contains the durable receipt; `removed` identifies completed cleanup; `deferred` denotes recovery beyond the pass budget; `error` explains refusal. Copy-only success is `verified`; no removal status is emitted until cleanup confirms it. Logs can be redirected to a file with ordinary shell redirection. Transport stderr is withheld to avoid leaking credentials or host setup details. No success is silently reported after authentication loss.

Local verification:

```sh
TMPDIR=/dev/shm python3 -m unittest discover -s tests -p 'test_checkpoint_archive*.py' -v
python3 -m py_compile revision/scripts/b6/checkpoint_archive.py \
  revision/scripts/b6/checkpoint_archive_cli.py tests/test_checkpoint_archive_cli.py
```

The suite uses actual temporary trees, real CLI subprocesses, executable SSH/rsync boundaries where no local SSH server exists, real SIGKILL interruption and independent byte/receipt checks. Host verification additionally uses actual Python3.6, Lustre pool/project/quota/fsync behavior and a real pronghorn pull. Tiny tests must use generated fixtures and their own destination/config; production99 GiB cleanup is not a fixture test.
