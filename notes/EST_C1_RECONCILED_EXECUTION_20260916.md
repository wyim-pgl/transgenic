# C1 reconciled execution proposal — 2026-09-16 PDT

## LOCAL ONLY — independent review required before staging/submission

New bundle: [reconciliation_refreshed/](est_execution_20260916_1530/reconciliation_refreshed/). Old proposal, baseline outputs and source files were not changed. Remote operations were read-only SSH: separate baseline final sacct, historical sacct, bounded small-file capture, path/interpreter check, then full unfiltered squeue. No remote writes, jobs, subagents, merges or pushes.

| Exact artifact | SHA256 |
|---|---|
| `manifest.proposed.json` | `c8b03a35bcf527b70d69fa4c11eb0ab2d90da1da3dd0b487d4dd85c6bfe00a2e` |
| `run.sbatch` | `763d655c2043b20bbe9882411589f6542112c8a65018e7eda638042a04080933` |
| historical/primary `accounting.psv` | `61120e921d3c8897e46608e02a8c46d9233812f71791c024a2626150bfbbb237` |
| `queue.json` | `fc30d8cb7e0f85d96416ba7587c4981853a30a54e4ad2f83081b5797a3722a6e` |
| recovered pilot receipt | `f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23` |

## Refreshed evidence and exact delta

- All **19 path snapshots / 32 unique small-file receipts / 489,199 bytes** equal the independently reviewed old facts. Every recursive kind, numeric UID/GID (3092957/3092957), child listing, hash and decoded byte is equal, including three historical wrappers, pilot logs and preflight evidence. No new static exception, ownership change or content change. Recovered pilot receipt equals the reviewed pinned local original; it is a local recovered submission record, not a newly discovered remote receipt.
- Historical sacct has exactly **30 unique rows**, byte-identical to the old capture: nine producers plus preflight; allocation/batch COMPLETED 0:0, extern OUT_OF_MEMORY 0:125. This same unfiltered historical capture is passed as primary `--sacct-file`; the auditor's primary nine-job accounting checks pass locally. Baseline accounting is separate and is not mixed into manifest evidence.
- New baseline sacct independently confirms **6165461 FAILED 1:0, elapsed 03:00:24**, batch FAILED 1:0 / MaxRSS 1549756K; extern OUT_OF_MEMORY 0:125 / MaxRSS 1480K. Its start/end strings have no timezone in sacct output and are not treated as UTC.
- Locally rechecked fetched baseline: 18/18 unit manifests PASS, rollup FAIL with `failed_units=[]`; tree failures are pilot receipt/log-layout and historical dispositions only. This does not substitute for the next full audit.
- New full queue: **68 rows**, local capture **2026-09-17 01:57:58.018983 UTC** (September 16 PDT). No filtering; all stdout preserved. Neither baseline6165461 nor MVP6165521 appears. The exact queue-validation statement block from reviewed auditor code passes at local verification time. No name/namespace waiver was added. If a later MVP row actually triggers the validator, retain the failure.
- Manifest entries, root, schema and accounting binding are unchanged. Only queue SHA changes from the old reviewed manifest (`90ce4404d3d0d0547c21a1ccf7db96344b97413e10d4bdd167e03976025a9ca9`) to the digest above. Queue timestamps/content and transport observations are new. No local generation is an approval decision.
- Read-only path check found `.../est_followup_20260916_1530/c1_reconciled` and planned inputs/code/audit/scratch/logs/run.sbatch absent; existing ancestors were not symlinks. This is an observation, not a reservation or continuous exclusion guarantee.

### Clock domains

Transport start/end and queue captured_at use the **local controller UTC clock**. Static facts/pathcheck captured_at use the **remote login host UTC clock**. Sequential calls establish baseline accounting returned → historical accounting returned → static capture returned → pathcheck returned → full queue returned. They do not establish synchronized cross-host wall-time ordering. Clock offset was not measured; remote timestamps can exceed later local timestamps. Captures are non-atomic and do not prove unheld flock or absence of all external writers.

## Exact bounded wrapper and deployment map

`run.sbatch` is a digest-fixed, one-shot proposal: cpu-s1-pgl-0, one node/task, **8 CPU / 16G / 12h / no-requeue**. It verifies code/input/interpreter hashes, claims an exclusive persistent execution.guard, rejects existing output/scratch/exitcode, keeps scratch outside original source/input trees and records batch exit. It never reads previous PASS manifests to skip hashing. It invokes all 18 units and final small-tree/producer artifact rechecks; expect roughly three hours from baseline, not a runtime guarantee.

Reviewed code commit **f6bef565f0bb432256b7e596a81b896afd2266c5**; four files copied locally under `code/`, byte-checked against both worktree and committed bytes:

- auditor `b4392729aaff73feb61e4ffef8df7b6ddc66235ec6aa1fd988babb0da46bebfb`
- launcher `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291`
- helper `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302`
- original sbatch template `5d122878c84e75d4b536a6c98413f03bb8ea5ec9f5a7892274d7a626449e8ad4`

Interpreter: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11`, observed Python3.11.14, executable SHA256 `74d2337199fe0880e01a38cd136c03f28de4383b833393657ce8420ce3c57415`. Wrapper uses `-I -B`, clears PYTHONPATH/PYTHONHOME, disables user site/bytecode. Executable hashing does not pin all linked libraries or stdlib.

For a separately authorized controller, R is `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled`:

| Local bundle item | Future remote target |
|---|---|
| `code/` (only the four listed files) | `R/code/`, preserve relative layout |
| `manifest.proposed.json`, `accounting.psv`, `queue.json`, `pilot_submission_recovered.json` | `R/inputs/` |
| `run.sbatch` | `R/run.sbatch` |
| no preexisting output | wrapper creates `R/audit`, `R/scratch`, `R/execution.guard`, `R/batch.exitcode` |
| new empty log directory | controller creates `R/logs` before submission |

Controller must independently inspect delta, captured receipts and wrapper, verify SHA256SUMS, refresh live state/path availability, exclusively create R (no overwrite/reuse), stage immutable/read-only inputs and code, verify deployed bytes and exact wrapper digest, and only then submit the single reviewed wrapper under separate authorization. No submission command was executed here. Do not stage collector/verifier as runtime code. No modification of raw data, old baseline, source trees, historical sentinels or locks is proposed.

## One-hour queue limitation — deliberately unchanged

The reviewed validator requires the saved queue capture to be nonfuture and **≤3600 seconds old at auditor startup**, measured against the compute host clock. This capture's nominal expiration is **2026-09-17 02:57:58.018983 UTC**, subject to clock differences. A long pending wait can honestly fail reconciliation; the auditor preserves full unit auditing even if reconciliation validation fails, so this can still consume ~3h and produce overall FAIL. The wrapper does not refresh/replace inputs or relax the gate.

Capturing at allocation start could address aging, but is **not implemented or approved**: it would require a separately reviewed evidence/manifest binding design, and this wrapper's own `est_` job name would be present in an unfiltered live queue and rejected by the existing validator. Do not solve that by silently filtering self or inventing an exception. If the window is missed, obtain newly reviewed capture/manifest/wrapper digests in a new bundle; do not mutate this reviewed revision or claim it remains current.

## Verification and remaining conditions

`verify_local.py` passed: old-static exact equality; recursive receipt/UID/GID validation; 19 authoritative bindings; committed source and copied code pins; 30 historical accounting rows; separate final baseline accounting; fetched18 PASS manifests; exact source queue-only checks; wrapper hash bindings; `bash -n`. This utility writes only local status/pins/checksums; do not mistake it for a read-only verifier or an independent review. Bundle SHA256SUMS covers all bundled files except itself.

No real-tree `load_reconciliation` or full current auditor execution was performed. Independent controller review is next, not completed. Extern OOM, authoritative ancestry/merged-taxid limitations, C2/C3, genomic phase acceptance and production authorization remain unresolved. Advisor attempt failed due to client/model version incompatibility; no advisor review claimed. Graft reported automatic local graph refresh on orientation; estimated tokens saved: 894,165.
