# Real C1 reconciliation manifest proposal — 2026-09-16

## DRAFT / BLOCKED — not approved, not staged

Approved code commit `f6bef565f0bb432256b7e596a81b896afd2266c5` was read from `/tmp/b6-p2-phase-1535433`; all four deployment source hashes match committed bytes (see `reviewed_code_pins.json`). No source changes, remote writes, job actions, baseline changes, merge, push or subagents. Advisor unavailable due to client-version mismatch; no advisor review claimed.

Bundle: [reconciliation_proposal/](est_execution_20260916_1530/reconciliation_proposal/). Proposed manifest SHA256: `90ce4404d3d0d0547c21a1ccf7db96344b97413e10d4bdd167e03976025a9ca9`.

First remote action was unfiltered live queue at 23:08:34 UTC. Remote static capture timestamp is 2026-09-16T23:09:58.613982+00:00. Exactly one queue capture, invoked after the static command returned, has local timestamp 2026-09-16T23:09:58.284713+00:00 and retains **6165461 RUNNING est_c1_20260916_1530**. Host clock offset was not measured: command sequencing, not synchronized cross-host timestamps, establishes this order. The reviewed queue validator correctly rejects this row. No polling or attempt to omit the baseline. This is a failed quiescence observation, not quiescence evidence establishing safety.

## Captured facts

All19 allowlisted paths have exact numeric UID/GID/kind/recursive children and bounded small-file SHA256 snapshots. No unresolved/mismatched historical path found in this capture. All owners in top-level snapshots: [(3092957, 3092957)]. Preflight traversal was restricted to its closed expected layout; other sentinels were empty or jobid-only. Unknown listing, link, special or >1MiB file would stop that path. No raw13, large FASTA, DB, source-tree recursion or lock probe.

**32 unique small receipts, 489,199 total bytes**; all bytes preserved losslessly in `static_facts.json`, full per-file byte counts/hashes in [RECEIPTS.md](est_execution_20260916_1530/reconciliation_proposal/RECEIPTS.md). This includes five preflight receipt files, lineage/provenance/DONE, preflight logs/submission receipt/exitcode/sentinels, all three producing wrappers, and pilot stdout/empty stderr. Existing pinned recovered pilot extract is copied unchanged. SHA256SUMS binds every bundle file except itself.

Accounting contains exactly30 rows: nine search jobs plus preflight6163397, allocation/batch COMPLETED0:0 and each extern OUT_OF_MEMORY0:125. This anomaly remains unresolved. Offline verification passed canonical schema, all19 fixed job/wrapper bindings, decoded bytes/hashes, lineage/provenance/preflight validation, sentinel contents and exact committed code hashes. It is **not** a real-tree auditor run or an operational manifest PASS.

## Planned paths, not staging

Manifest references planned external captures under `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled/inputs/accounting.psv` and `queue.json`. Proposed deployment manifest name: `inputs/manifest.proposed.json`; pilot receipt: `inputs/pilot_submission_recovered.json`. These remote paths were neither created nor existence-probed. Controller must check exclusive fresh availability; do not overwrite. Local proposal is intentionally in main notes, not a production external capture location.

Schema references: reviewed auditor `load_reconciliation:554–625`, `reconciliation_paths:483–496`, `historical_snapshot:462–480`, `preflight_evidence:527–551`; C worktree plan/report read in full. Main operations report and its capture bundle were retained. No approval boolean was generated.

## Controller independent-review checklist

- [ ] Review source commit/pins, command transports, all receipt bytes/hashes,19 exact paths and job/species bindings; verify bundle SHA256SUMS.
- [ ] Wait for baseline6165461 completion without cancellation/resubmission; preserve baseline outputs. Independently inspect its final accounting/results.
- [ ] Obtain a **new full unfiltered queue capture after completion**, no relevant active/held/pending EST jobs; <=1h old at auditor startup. Current capture is blocked and must not be staged as acceptable evidence.
- [ ] Revalidate static paths/owners/listings/hashes and accounting before deployment. A capture is not continuous locking or proof of live lock ownership.
- [ ] Rebind changed queue hash in a newly reviewed immutable manifest and record its separate digest; current proposed digest cannot approve a future edited manifest.
- [ ] Confirm new external input/run/snapshot paths absent, exclusively stage a **new immutable auditor** and capture copies, verify all code/input hashes, and independently review invocation. Never patch baseline.
- [ ] Keep extern anomaly, ancestry/merged-taxid limitation, C2/C3, phase acceptance and production authorization unresolved; reconciliation does not clear scientific gates.

Graft estimated1,044,588 tokens saved across orientation/schema calls (tool estimate). No deployment or scientific acceptance claim.
