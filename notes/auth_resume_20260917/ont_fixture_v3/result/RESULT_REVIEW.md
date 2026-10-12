# Job 6165962: failed before workload; clean fixture rejected

## Terminal observation and scope

One scheduler snapshot at 2026-09-17 16:59:00 PDT found the exact job terminal and no row in `squeue -h -j 6165962`. Observation stopped immediately; no further polling, schedule, submission, cancellation, remote write, installation or workload was performed. Submitted 16:57:33.985680 PDT, scheduler start 16:57:34, end 16:57:36, elapsed two seconds, node cpu-50. The allocation is no longer active according to Slurm. This is scheduler termination evidence, not an independent compute-node process/cgroup reaping inspection.

`snapshot_01.txt` retains full scontrol output and unfiltered allocation/step sacct rows:

| Row | State | ExitCode | DerivedExitCode |
| --- | --- | --- | --- |
| 6165962 | FAILED | 1:0 | 1:0 |
| 6165962.batch | FAILED | 1:0 | not reported |
| 6165962.extern | OUT_OF_MEMORY | 0:125 | not reported |
| 6165962.0 (timeout) | FAILED | 1:0 | not reported |

No F4 pass, real-tool workload pass, or CLEAN fixture acceptance is established. The unexplained extern OUT_OF_MEMORY/0:125 independently blocks CLEAN acceptance. Small sampled RSS does not disprove an OOM or explain this accounting result.

## Exact failed gate

Actual complete `raw/stderr` reports:

```
fixture.py line 290, in run
require(resource.getrlimit(resource.RLIMIT_FSIZE) == (4194304, 4194304), 'startup file soft/hard limit must be 4194304 bytes')
ValueError: startup file soft/hard limit must be 4194304 bytes
```

This is the startup inherited file-size soft/hard-limit equality gate, before launch-environment capture, compute closure collection, cgroup verification, synthetic input generation and any alignment. The actual unequal tuple was not logged, so neither its values nor the mechanism producing the mismatch is proven. The batch script requests `ulimit -f 4096` before srun; this alone does not prove the task inherited the requested limits. No corrected job or diagnostic workload was run.

The private out directory exists and is empty. No receipt, COMPLETE, FAILED, command ledger, alignment, oracle table or partial scientific artifact exists. The gate occurs before the later workload try/cleanup block. Absence of FAILED is therefore not success. Stdout contains 17 manifest OK lines and an earlier RUNNING scontrol record, not a final receipt. Current terminal accounting supersedes that earlier RUNNING line.

Consequently the final-stdout/receipt/COMPLETE identity check, fixed two-junction oracle (20500..20699 supported by three names, 30500..30699 unsupported with two), six-name selection with no outside read, and UNRESOLVED/no-uf audit are all NOT REACHED, not passed. Intended synthetic panel, reference/authority and junction-bootstrap seams remain declared test-only boundaries, not verified biological or production evidence. Full ONT, training and canonical-authority acceptance remain blocked.

## Runtime and identity verification

`collection.json` retains the complete namespace inventory, hashes, small evidence bytes and explicitly enumerated runtime metadata/hash results. All retrieved bytes were independently hashed locally against captured remote SHA256. `verification.json` records a successful offline verification using `python3 -B verify_captures.py`.

Verified exact deployed pins:

- Manifest: `2cd93e7b93182a51dc9198d4f697af7b9f36b826e0286d1538f26c96af09b904`.
- Approval: `84e596d5360b9049f13a2b8fcea34d77484d9494f551532e45b40d8f7cdec91f`.
- Expected closure: `17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce`.
- All 17 deployed manifest payloads match their pins and local v3 bytes.

Post-terminal collection at 17:00:33.960706 PDT rehashed only the 180 unique file paths explicitly recorded in deployed expected_closure.json, covering all 417 repeated file records. All SHA256, size, resolved-path, recorded symlink, mode, UID and GID comparisons match. All 43 explicitly recorded parent-directory owner/group/mode checks match. No runtime directory discovery, environment scan or original-data access was performed. Runtime bytes were streamed into hashes remotely; only small hashes/metadata were retrieved and compared locally to the deployed pinned closure. This is a post-run login-host view of those explicit paths, not a compute-host closure observation. The compute closure gate was never reached, and no before/after compute runtime proof is invented.

## Resources and missing actual containment evidence

Scontrol records UID/GID 3092957, four CPUs, cpu-50 CPU_IDs 10-13, 4096 MiB and six-minute limit. Sacct AllocCPUS is four for every retained row. MaxRSS/MaxVMSize in KiB: batch 1496/163500, extern 1288/163476, timeout step 1340/298568; allocation-level maxima are blank. These are sampled scheduler records, not proven peak aggregate memory.

No cgroup_observed.json or launch_environment.json was produced. Actual workload PID membership, cgroup-v1 hierarchy, hierarchical RAM/memsw limits, and task affinity were not captured because execution stopped first. Scheduler allocation CPU IDs do not substitute for actual process affinity. No current-job cgroup containment acceptance can be claimed from the reviewed policy alone.

## Complete output/log size review and retrieval bounds

All output entries and complete stdout/stderr were inventoried. There are no files under out.

| Artifact | Logical bytes | Allocated bytes |
| --- | ---: | ---: |
| stdout | 1941 | 512 |
| stderr | 746 | 512 |
| empty out directory | 4096 | 1024 |
| Total, including directory metadata | 6783 | 2048 |

Regular-file total is 2687 logical / 1024 allocated bytes. Both conservative totals are below the 268435456-byte review threshold. Allocation is reported from actual st_blocks times 512, not inferred from logical length. No claim of a hard aggregate quota or transient/deleted-byte accounting is made.

Total raw remote retrieval was 1437798 bytes, including scheduler capture, successful collection and failed collection startup diagnostics. Every retrieved source file is below 4 MiB, as is the largest raw response (1435393 bytes). The observer used one scheduler snapshot, stopped on terminal state, and completed remote capture within roughly two minutes, below eight snapshots/eight minutes. Local extracted copies duplicate captured bytes but require no additional remote retrieval.

## Preserved evidence and work record

- `raw/stdout`, `raw/stderr`: complete Slurm logs.
- `raw/controller_approval.json`, `raw/release/MANIFEST.sha256`, `raw/release/runtime_evidence/expected_closure.json`: exact deployed identity evidence.
- `snapshot_01.txt`, `collection.json`, `collection.stderr`, `collection_summary.json`, `verification.json`: scheduler, inventory, closure, retrieval and local verification evidence.
- `collect_readonly.py`: exact read-only retrieval commands; no polling or submission logic.
- `verify_captures.py`: runnable offline verification, writes nothing.
- `collection_attempt1.stdout` and `.stderr`: preserved initial retrieval failure because login PATH had no python3. The subsequent read-only collection used the already enumerated explicit Python path, without installation or environment edits. This was a metadata retrieval correction, not a job retry.

Graph orientation timed out before source inspection; scoped reads used the specified review documents and actual traceback locations. Advisor consultation was unavailable because of client/model incompatibility; no advisor endorsement is claimed.

Observer and evidence verification complete. Failed fixture remains unaccepted. No remediation or resubmission is authorized by this result. All newly written files are confined to this new v3/result directory; existing v3 and v2 artifacts were not modified.
