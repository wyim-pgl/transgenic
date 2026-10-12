# Job 6165875 bounded result review

## Verdict

Accepted as a hash-bound, terminal bounded diagnostic traversal only. Not a clean scheduler run: the extern step is OUT_OF_MEMORY, exit 0:125, and remains unresolved. Failed individual replay/geometry outputs are not accepted as positive evidence. No biological, phase, raster, training, full-C0, route-A, molecule-count or A11 completion is established.

## Observation and accounting

Two status snapshots, 2026-09-17 14:46:54 and 14:49:14 PDT, both terminal. Monitoring stopped after terminal verification, within the 12-minute/eight-snapshot bounds. Only job 6165875 was queried. No submission, retry, cancellation, remote write, original-source scan/hash, or code change occurred.

- Allocation: COMPLETED, exit 0:0, cpu-50, 2 CPUs, requested 8Gn.
- Batch: COMPLETED, exit 0:0, MaxRSS 26424K.
- Extern: OUT_OF_MEMORY, exit 0:125, MaxRSS 1480K. This repeats the known accounting anomaly; neither its cause nor resolution is established. Low recorded MaxRSS does not negate OOM.
- Start 14:40:30, end 14:45:48 PDT, elapsed 00:05:18.
- stdout and stderr are both zero bytes, retrieved and preserved. Reader stderr is separately reported empty, EOF observed, not truncated, reader exit 0.
- First snapshot's optional inventory command failed because python3 was absent from the login PATH. Its squeue/sacct observations succeeded. Output inventory and retrieval subsequently used bounded shell reads; final inventory independently confirms regular files and stable sizes. No ambiguous FAILED/INCOMPLETE worker marker was present.

Raw commands, timestamps, output and errors: `snapshot01.txt`, `snapshot02.txt`. Retrieved bytes: `raw/`. Per-file size checks and SSH return codes: `retrieval.json`.

## Independent local binding checks

`local_verification.json` records 16 passing checks, exact local file sizes/hashes, pinned local release inventory, manifest consistency, runtime identity and diagnostic summary. `diagnostic_link_verification.json` records 847 passing linkage checks. The marker was not accepted alone.

- Controller result SHA256: `675edd5e368505a06c92df2e80e411642354a7132e66bbb2c2a2e71b338e9446`, matches verification receipt.
- Worker report SHA256: `2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8`, matches marker bytes, result binding and verification.
- Diagnostics: 12,557,738 bytes, below 256 MiB; SHA256 `8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d`, matches report, result and verification.
- Ledger `46563a19f5699c55df12df7241af7f807e6d98bed0d7c0666138ce78804c40d6` and manifest `7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c` match approved pins and local bytes.
- Controller status VERIFIED_DIAGNOSTIC_COMPLETE, worker exit 0, no errors/interruption/deadline expiry. Ledger/import/runtime/binding postchecks all PASS. Start receipt's INCOMPLETE is the expected initial state, not the final result.
- All 496 retained domain/preimage IDs, 1,667 embedded byte-hash occurrences and 356 sequence-hash occurrences verified locally. Search identity includes matching provenance and commands hashes; HSPs link to searches, members, raw rows and placements.

## Real sample dispositions

Sitalica primary sample: first 16 distinct exact QNAMEs from the capped OrthoDB prefix, 376 physical lines read for selection. This is not a random or population-wide sample. All 16 members and 16 primary geometry candidates were retained.

- 462 raw search rows and 462 HSP diagnostics: OrthoDB 400, SwissProt 62.
- HSP_RULE_ACCEPTED: 385.
- Source-bound replay failures: query codon truth 30; subject truth 18; reported length versus M+I+D excluding frameshifts 29. Total 77, retained as failures, not biological negatives or accepted evidence.
- Of 385 geometry attempts, 380 report success and 5 report failure. All five failures belong to JK608593.1; details are retained in `diagnostic_link_verification.json`. A geometry success is a diagnostic property, not phase/raster acceptance.
- 324 retained protein records, 339 SAM header records. Worker reports selected-search EOF and selected-BAM EOF traversal complete; reported whole-stream counts are OrthoDB 1,371,252 rows, SwissProt 290,009 rows, BAM 66,219 records. These counts are worker attestations, not independently rescanned source counts.
- Exact per-QNAME source counts and dispositions are in `sample_dispositions.tsv`.

## Source and runtime identity

All 15 reported sources match manifest paths and specified hashes/receipts. Compute reports fresh whole-file hashes and unchanged before/after metadata. Fourteen sources report expected-hash validation. UniVec report.tsv deliberately has no historical expected hash; it has a fresh compute hash `10091f5ad99fd95001743359206d92f18b57b1286416c7229de84344e4e6345c` and retained relationships cross-bound to pinned FASTAs. It must not be described as historically pinned. Original source bytes were not reopened by this observer; verification covers retrieved attestations and embedded diagnostic bytes, not a second source scan.

Loaded six-module evidence closure matches the release ledger, including binder `93ea442d0dc60c1d8df2ee0b479361d89a188f9b43bba4327740f896efec27b2` and contract `fcb8a4b2413f12dc5c5b0c4d55676cedeac21b3fcb26cf849c089657650a3954`.

Before/after runtime identities agree: Python 3.11.14, executable SHA256 `74d2337199fe0880e01a38cd136c03f28de4383b833393657ce8420ce3c57415`; samtools 1.21 with htslib 1.23, executable SHA256 `a579aed31a13841cc0950a487b59729355d0b9fbcc39293e0b90b960133217bc`. Historical producer samtools 1.3.1 is distinct from the current reader. Shared libraries and stdlib are not individually pinned; runtime is not hermetic.

Worker elapsed 315.263 seconds, controller elapsed 316.887 seconds. Charged reads 12,669,342,513 bytes, charged output 12,649,898 bytes, expanded bases 310,959, retained bytes 560,027. These are worker counters, not scheduler I/O or whole-process-memory proof.

## Closed scope and remaining blockers

Bounded observation, retrieval and local diagnostic integrity review are complete. Extern OOM remains an explicit unresolved blocker to any clean-accounting claim. The 77 replay failures and five geometry failures remain failures; no fix, rerun or scientific promotion was performed. All scientific gates remain false and A11 remains unassessable.

The source re-review and local run_compute report-binding contract were consulted only to interpret this result. Parent graph query timed out; worktree graph returned no relevant launcher node, so targeted unindexed source spans were used. Advisor consultation failed with a client/model-version incompatibility, and supplied no approval.
