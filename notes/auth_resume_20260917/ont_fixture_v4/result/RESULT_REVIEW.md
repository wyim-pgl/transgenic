# Job 6165965: technical workload pass; CLEAN/F4 acceptance blocked

## First check: actual file-size limits

Actual `raw/stderr` records `file_limit_before: [4194304, -1]`. Actual `raw/out/file_limit_observed.json` records before `[4194304, -1]`, after `[4194304, 4194304]`, enforced inside the allocated Python task. Here -1 denotes unlimited. `launch_environment.json` independently retains the resulting exact soft/hard tuple. The allocated-task correction worked on this attempt without relaxing the 4 MiB gate. This does not establish the old job 6165962 tuple or its propagation mechanism. The prior failure and all its evidence remain unchanged.

## Terminal scheduler result and acceptance decision

One exact-job scheduler snapshot at 2026-09-17 17:16:53 PDT found job 6165965 terminal and no squeue row. Polling stopped immediately. Submission was 17:14:54.545002 PDT, scheduler start 17:14:55, end 17:14:59, elapsed four seconds, cpu-50. `snapshot_01.txt` preserves scontrol and all allocation/step sacct rows, including extern.

| Row | State | ExitCode | DerivedExitCode |
| --- | --- | --- | --- |
| 6165965 | COMPLETED | 0:0 | 0:0 |
| 6165965.batch | COMPLETED | 0:0 | not reported |
| 6165965.extern | OUT_OF_MEMORY | 0:125 | not reported |
| 6165965.0 (timeout) | COMPLETED | 0:0 | not reported |

**The unexplained extern OUT_OF_MEMORY/0:125 blocks CLEAN and F4 acceptance, even though the payload passed.** The allocation-level COMPLETED result does not override its extern row. No explanation, resource-policy waiver, retry or broader authorization is inferred. Scheduler terminal state establishes the allocation is no longer active, not an independent compute-node process/cgroup reaping inspection.

All rows have AllocCPUS=4. Scontrol records 4096 MiB, six-minute limit, and allocation CPU_IDs 10-13. Sampled MaxRSS/MaxVMSize in KiB: batch 1496/163500, extern 1480/163476, timeout 1340/298568; allocation maxima are blank. These samples are not peak aggregate memory and do not refute the extern OOM. This attempt used four elapsed allocated seconds, or 16 allocated CPU-seconds, separately from the preserved prior attempt's eight allocated CPU-seconds.

## Independently verified technical result

Offline `python3 -B verify_captures.py` exits zero. It verifies captured bytes against remote hashes, the fixed oracle, actual BAM records and their CIGAR-derived junctions, not merely the receipt's success label.

| Junction, zero-based inclusive | Actual distinct read names | n_ONT | n_HiFi | supported |
| --- | --- | ---: | ---: | --- |
| fixture_chr:20500..20699, + | positive_0, positive_1, positive_2 | 3 | 0 | true |
| fixture_chr:30500..30699, + | below_threshold_0, below_threshold_1 | 2 | 0 | false |

The actual BGZF/BAM decodes to six primary forward-strand records. Three positive and two below-threshold reads have `500M200N500M`, beginning at 20000 and 30000 respectively. `unspliced` has `1000M` at 40000. BAM magic and BAI magic are valid. The stdlib verifier parses BAM records without installing software or launching remote tools. The 20509..20708 decoy junction is absent. The exact six names agree across selection list, selected FASTA and BAM; the seventh input name `outside` is excluded.

Actual orientation audit is UNRESOLVED, `use_uf=false`: five eligible alignments are below the unchanged floor of 1000. None of the actual nested commands contains `-uf`. Nine nested execution records show exit codes `[0, 0, 4, 0, 0, 0, 0, 0, 0]`. Exit 4 belongs specifically to the expected unresolved orientation audit, not a silent success or unexpected crash. Driver exit is zero. Real minimap2 2.28-r1209, samtools 1.3.1 and seqkit 2.13.0 versions, pinned executable hashes, command outputs and real BAM support the technical workload finding. Python is 3.11.14. The command ledger records the junction bootstrap's actual argv separately.

Final stdout's last JSON object equals `receipt.json`. `COMPLETE` equals the actual receipt SHA256 `b4101cf9e0b3cf9a8361978d244aa40d2c14a8facc553246dd28e651e774f7c4`. Every declared output and diagnostic hash/size matches. No FAILED artifact exists in the complete namespace inventory. Receipt status remains `FIXTURE_WORKLOAD_PASS_PENDING_CONTROLLER`, `real_fixture_controller_accepted=false`, `training_eligible=false`, decision null. This review does not promote that status.

The four declared seams remain visible: synthetic enum-labelled role/input root rather than canonical authority; explicit fixture panel hash and 50 overlapping test loci rather than the canonical panel; separately recorded actual junction bootstrap argv; synthetic reference binding rather than Git92. The result is a finite test-only real-tool check, not biological validation, full ONT authorization, training eligibility or canonical structural acceptance.

## Actual compute containment and evidence limits

Actual `cgroup_observed.json` records cgroup v1 memory, cpuset and devices membership under `/slurm/uid_3092957/job_6165965/step_0`. Both step and job memory limits and combined memory+swap limits are exactly 4294967296 bytes, with `memory.use_hierarchy=1`. Recorded failcnt is zero and under_oom is zero at this pre-workload observation. These are real compute outputs, not a substituted policy statement. They do not explain or negate the later extern terminal accounting.

The saved step task list is 27202 and 27203. Pinned `runtime_closure.py:106-135` verifies the current PID is in that list before returning. The launcher's own PID is not separately persisted, so the review does not invent which listed PID was Python or confuse the receipt's child PID 27380 with the launcher.

Pinned `fixture.py:277-284` checks `len(os.sched_getaffinity(0)) == 4` before limit enforcement. The completed run provides evidence that this assertion succeeded. However, the actual affinity CPU list was not saved in any retrieved diagnostic. Scheduler CPU_IDs 10-13 are allocation observations only, not a measured process affinity list. Exact actual affinity identifiers and post-run cgroup/process reaping are therefore not independently verified. No new diagnostic job or remote process inspection was used to fill those gaps.

## Runtime and deployment preservation

Verified exact deployed pins:

- Manifest: `13da604041044f3ce0192b2049d644a603b9bbef8d14d38c70fedc07ae92d3b9`.
- Approval: `981023667e3dc0b18a44ae62f6b8b6694dc177d84f19e5989bf26a713d804da5`.
- Expected closure: `17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce`.

All 17 deployed payload hashes match the manifest and local v4 files. Compute `runtime_closure_observed.json` has exactly the expected closure hash. Post-terminal collection at 17:17:26.367899 PDT rehashed only the 180 unique runtime paths enumerated by that pinned closure, covering all 417 repeated file records. All SHA256, sizes, resolved paths, recorded symlink targets, modes, UIDs and GIDs match. The 43 explicitly enumerated parent ownership/group/mode checks also match. This post-run view is from the login host, distinct from the saved compute observation.

The collector verifies fresh namespace pins before following runtime records, refuses any runtime path count other than 180, and never discovers runtime/environment directory contents. Only the approved job namespace was inventoried. No original data were accessed. Runtime bytes were hashed remotely without retrieval; only explicit metadata and hashes were returned.

## Output sizes and retrieval bounds

Complete output and scheduler-log inventory is retained in `collection.json` and `verification.json`; every captured raw output is under `raw/`. Actual `st_blocks * 512` supplies allocated sizes.

| Scope | Logical bytes | Allocated bytes |
| --- | ---: | ---: |
| OUT and logs, regular files | 1187871 | 1782272 |
| OUT and logs, including directories | 1224735 | 1791488 |

Both conservative totals are below 268435456 bytes. This is a post-run soft review threshold, not an aggregate quota or a statement about deleted/transient bytes.

Total raw remote response retrieval, including scheduler stdout/stderr and collection stdout/stderr, is 3030485 bytes, below 32 MiB. Retrieved source-file bytes total 2189617; the largest source file is 997375 bytes, below 4 MiB. The collector's base64 response is 3028385 bytes including SSH stderr. Locally extracted duplicates do not add remote retrieval. There were no refusals or truncated retrieved files. One snapshot and one read-only collection finished within about one minute of the first observation, comfortably below eight snapshots/eight minutes. No further remote action followed terminal capture.

## Reproduction and work record

- `verify_captures.py`: offline stdlib verification, writes nothing. Run from any directory with its absolute path.
- `verification.json`: successful verification output including decoded BAM records, explicit limitations and full output/log inventory.
- `snapshot_01.txt`, `snapshot_01.stderr`: unfiltered exact-job terminal evidence.
- `collection.json`, `collection.stderr`: complete bounded retrieval and post-run enumerated runtime verification.
- `raw/`: actual scheduler logs, approval, manifest, expected closure and every output, including receipts and observations.
- `collect_readonly.py`: prior bounded collector reused with fresh namespace/pins and a reduced raw-file budget. `remote_collector.py` retains the exact expanded remote read-only script. Do not rerun collection merely to reproduce the offline review.

Scoped documents FILE_LIMIT_FIX.md, REVIEW.md, controller_approval.json, staging_and_submission.txt and prior v3 RESULT_REVIEW.md were read. Graft query timed out; targeted specified artifacts and exact source guard ranges supplied context. Advisor consultation failed due to client/model incompatibility, so no advisor endorsement is claimed.

Observation, capture and local technical verification are complete. CLEAN/F4 remains blocked by extern OOM; exact saved affinity IDs and independent reaping evidence are absent. No new job, retry, cancellation, remote write, installation, environment discovery or original-data access occurred. All new files are confined to this v4/result directory. Job 6165962 failure evidence was neither edited nor superseded.
