# Task 2 independent protocol/data-safety review

Decision: **Needs fix** — two Important issues confirmed against fixed commit `1203a2b` (base `08736de`; previously approved core `8ff381b`). No Critical finding. This is the early protocol/core gate; intentionally unfinished documentation/config examples are excluded.

## Important: archive restart discovery exceeds the configured source scope

Fixed-snapshot locations: `revision/scripts/b6/checkpoint_archive_cli.py:152-157` and `:355-368`.

For `source_kind=archive`, `run_root` selects the parent of the explicitly configured epoch, then `inventory` returns every unfinished cleanup journal in that parent. `pass_target` checks destination binding but does not check the pending manifest's exact source path/source kind before calling local `core.cleanup`.

Focused real-file reproduction loaded the exact commit's core/CLI into memory, made upstream archives for epochs 1 and 2, forwarded epoch 1 to the configured final destination, and interrupted its cleanup after journal creation but before tombstone rename. A subsequent local pass configured **only upstream epoch 2**, `apply=True`, `remove=True`, `max_checkpoints=1`, removed the still-present upstream epoch 1 instead. Observed output:

```text
pending epoch before [1]
configured epoch2 exists True
unconfigured epoch1 exists False
```

The remote helper's `stored_manifest` exact-path check rejects the same mismatched transaction, so remote operation fails closed but can be blocked by an unrelated sibling's recovery. Filter pending transactions to the explicitly configured archive path and source kind before reporting/scheduling them; retain strict destination binding. Add local and SSH-pull tests with two sibling archives and only one configured input, including an absent configured path. Recovery must leave the sibling's journal/source untouched and not consume its budget.

## Important: cleanup protocol timeout does not bound complete message I/O

Fixed-snapshot locations: `revision/scripts/b6/checkpoint_archive_cli.py:197-203`, `:245-252`, and `:322-330`.

`read_json` uses `select` once and then calls blocking `readline`. After any partial message arrives, a peer/transport that stalls before newline bypasses the deadline indefinitely. This affects both the coordinator waiting for a challenge under its destination lock and the source helper waiting for an ACK under source/upstream locks. The coordinator's direct blocking request/ACK writes also happen outside `communicate(timeout=...)` and can block under backpressure.

Focused pipe reproduction ran the exact snapshot's `read_json(sys.stdin.buffer, 0.1)`, wrote a single `{` byte, and kept the pipe open. It remained blocked beyond 0.7 seconds; the reviewer terminated the disposable child. A first-byte timeout test does not cover this case.

Use a monotonic deadline that covers the full bounded JSON line, and bound protocol writes as well. Avoid buffered-reader prefetch/select mismatches. Test partial challenge, partial ACK, and a peer that stops reading a request/ACK larger than pipe capacity. On timeout before ACK completion the source must stay intact; after authorization preserve the verified destination/journal and report the possible cleanup outcome honestly.

## Other reviewed boundaries

- Fresh nonce plus exact manifest/receipt equality rejects stale ACKs; this is correctly a trusted same-user SSH boundary, not cryptographic remote-storage proof.
- `publish_received` independently hashes exact payload/topology, checks file storage placement, fsyncs payload/receipt/directories and publishes under destination lock. Source validation follows rsync and source cleanup validates again.
- `remove_remote` independently verifies the published destination while holding its root lock through the live handshake and returned outcome, without recursively taking the same core lock. Ambiguous transport failure retains the archive and points to journal recovery.
- Pending recovery independently checks destination storage and verification before cleanup, and charges manifest bytes/checkpoint count against the pass budget. The source-scope issue above remains the exception.
- Dry-run routes through metadata inventory and does not prepare/hash/copy checkpoint binaries or create source/destination directories. SSH helper argv is shell-quoted; host option injection is rejected; rsync uses protected arguments and neither source-removal nor delete options.
- Lustre probes check directory inheritance, numeric file project IDs, actual file stripe pool, project pool quota, and statvfs reserve. No new obvious Python 3.6 API incompatibility found in this snapshot.

No full suite rerun was performed: root supplied 18 new tests passing in 6.921s and 41 existing core tests passing in 0.911s. Reviewer validation was limited to the two focused reproductions above; it used actual disposable files/pipes and did not modify implementation code, git state, production files or host services. Actual DeltaAI Python 3.6/Lustre behavior, real rsync quoting, quota parsing and pronghorn SSH/pull tests remain root's responsibility. Documentation/config completion and later changes need their final gate.

No token/dollar tally was reported by the runtime. Graft's own estimated context savings across this review's graph calls: 1,644,901 tokens (includes the whole-repo map baseline; not billed-token usage).

## Final Task 2 re-review — b6131cb

Decision: **Approved, with one Minor usability follow-up below.** The two original Important findings are resolved. This approves the implementation/docs gate for the canonical paths used by the supplied examples; actual-host verification and production activation remain separate root responsibilities.

Reviewed the fixed `1203a2b..b6131cb` package (`a6066c0` and `b6131cb`), including new documentation and combined host-selectable example config. Exact snapshot source locations:

- CLI `:168-178` now filters pending journals by source kind and exact archive input before reporting/scheduling them. It preserves sibling transactions and applies the existing destination binding/budget controls only to the configured input.
- CLI `:219-270` uses nonblocking descriptor I/O and a monotonic deadline for full bounded frames, including partial reads and backpressured writes. One channel preserves prefetched bytes across request/ACK messages.
- CLI `:387-423` keeps independent destination verification and the destination lock through the exchange, conservatively labels attempted ACK delivery as potentially authorized, and reports ambiguous cleanup rather than claiming retained source.
- CLI `:118-130` kills the transport process group even if its leader has exited, then bounds reaping/pipe closure. This closes the descendant-held-pipe timeout escape.
- The core publication optimization removes a duplicate prepublication destination hash only after the caller already verified bytes under the same destination lock; postpublication verification/durability remain intact. No new safety regression found.
- `docs/checkpoint_archive.md` documents the three-file installation, checksum comparison, actual host Python choices, target selectors, explicit copy/removal flags, bounded watcher, source kind distinction, quota/pool checks, conservative caps, and honest ambiguous outcomes. The example config has B5/B6/C2 targets for both hosts, with small caps and no autoactivation. Source/destination/helper paths and operational constraints match the brief; placeholders are explicitly explained.

Reviewer verification: first compared all three relevant working files byte-for-byte to `git show b6131cb:<path>`, then ran the four focused regression tests for sibling-journal isolation, partial-frame deadline, backpressured-write deadline, and descendant-held-pipe timeout. **4 tests passed in 0.813s.** No broad suite rerun. Root reported the full 66-case suite passing in two seeded orders (17:9.459s, 91:9.310s), Python 3.6 syntax validation, and initial-snapshot 59-test runs on DeltaAI/pronghorn; the final snapshot still needs root's actual two-host tests. The implementer's final report was not yet present at review time; requirements and fixed code/docs supplied sufficient scope.

### Minor: normalize or explicitly reject noncanonical archive input spelling

CLI `:164-176` computes `dirname(source_root)` and compares exact strings without normalizing an otherwise accepted absolute input. A trailing slash makes the supposed parent equal the epoch itself. After interrupted cleanup tombstones that epoch, inventory silently reports `waiting=True, pending=[]` instead of discovering its journal. Repeated separators can also miss exact matching. This does not remove incorrect data; it prevents automatic recovery for a common path spelling. Existing examples use canonical paths.

Focused actual-file reproduction at `b6131cb`: create and forward one upstream archive, interrupt cleanup after tombstone rename, then call inventory with the receipt path and with the same path plus `/`:

```text
trailing_slash=False waiting=False pending=1
trailing_slash=True waiting=True pending=0
```

Normalize accepted absolute source paths lexically before parent selection/matching (including helper requests), or reject such spellings explicitly at config/protocol validation. Add a focused interrupted-recovery regression for an archive input ending in `/`. This is a recommended small follow-up, not a blocker for the canonical deployed config.

Re-review graft estimate: 87,022 tokens saved; cumulative review estimate 1,731,923. No billed token/dollar accounting was reported.
