# ONTF4 fixture v2: reviewed narrow amendment candidate

## Status and scope

ONTF4 real-tool execution is NOT DONE. No authorization JSON is supplied. `runtime.json.runtime_closure_approved` remains false. Controller explicitly accepted only the replacement of the newly self-imposed aggregate hard quota for this trusted, seven-read, below-150,000-byte fixture. This is not a full ONT policy waiver. Historical externOOM remains unresolved.

Only this new v2 directory is edited. The original 15 runtime files were copied first, together with tests/stub/plan. Package gate, wrapper, source pins and all nine frozen files remain byte-identical. The new bounded runtime collector and its expected metadata add two runtime manifest entries, outside the nine-file closure. Tests and login collection driver are review-only, not deployed runtime code.

## Scientific contract remains frozen

Run frozen CLI main functions and reuse unchanged scientific postflight. Preserve original tool argv, except the previously declared junction bootstrap seam, whose logical and actual argv are separately recorded. Synthetic role/read enum and reference authority are test-only. The process-local explicit panel-hash seam preserves loader schema/counts and duplicate checks. Fifty overlapping test IDs are not biological loci. Raw fixture-bound `panel_sha256_is_frozen=true` is not canonical attestation.

The precommitted exact junction oracle is unchanged:

| Inclusive zero-based junction | Distinct ONT names | Supported |
| --- | ---: | --- |
| fixture_chr:20500..20699 + | 3 | true |
| fixture_chr:30500..30699 + | 2 | false |

No extra rows, including +9 decoy 20509..20708. Six selected names include the unspliced read and exclude the outside-contig read. Selection and absence of extra rows do not independently prove the unspliced read's acceptance by the BAM parser. Any stronger claim requires real BAM/QC evidence. No HiFi input is not biological HiFi zero.

Orientation remains UNRESOLVED, exit 4, no-uf, min_eligible=1000 and threshold=.95. Eight driver commands plus audit yield nine ledger entries. Nested real samtools view is not a separate ledger entry. The junction path exercises frozen sam_lines, parse_sam, alignment_rejection, transcript strand and ONT3/HiFi2 thresholds, not cs-anchor/correction, molecule assignment, complete-chain/callability, full scoring or B1 execution. No changes to missingness, pooling, training or decisions.

## Resource and filesystem policy

Conditional candidate allocation is one node/task, four CPUs, 4 GiB, 360 seconds, no GPU/array/retry/requeue. No extra user approval is needed to adopt this candidate budget, but controller review, runtime/cgroup/ownership/duplicate checks remain gates before deployment or submission.

Required approval fields, in addition to scope/output/manifest bindings:

```
output_policy: finite_fixture_soft_aggregate
aggregate_review_limit_bytes: 268435456
aggregate_hard_limit_bytes: null
per_file_hard_limit_bytes: 4194304
cpus: 4
memory_mib: 4096
wall_seconds: 360
scheduler_logs: [<absolute stdout path>, <absolute stderr path>]
runtime_closure_sha256: <expected_closure.json SHA256>
scheduler_cgroup_reviewed: <controller decision, never worker-authored true>
runtime_closure_approved: <controller decision, never worker-authored true>
```

`scope` must be TEST_ONLY_REAL_TOOL_FIXTURE. `output` is the exact fresh private absolute OUT, disjoint from the entire release. `manifest_sha256` binds MANIFEST.sha256. ONT_FIXTURE_APPROVAL and ONT_FIXTURE_APPROVAL_SHA256 point to the separate controller-authored approval and its independently retained hash. Scheduler log paths must be distinct and outside OUT/release. Controller must verify these are the actual scheduler destinations and check owner, node, allocation, canonical-tree separation and no concurrent/duplicate work.

The executable verifies the policy fields. After creating OUT, it records resolved scratch/log paths, filesystem mount/type/source, available bytes/inodes and owner/mode. Missing log files use the existing parent for filesystem observation. Observed free space is not reserved. OUT must actually be owned by the caller with mode 0700. TMPDIR and CWD of the driver group are beneath OUT.

Non-POSIX Bash `ulimit -f 4096` gives 4 MiB; Python verifies both actual RLIMIT_FSIZE values are 4194304 before metadata/workload. The workload retains that limit and per-process CPU limit 240 seconds. Neither protects Slurm's separately opened log files. The logical post-run OUT threshold is checked before publication and again after receipt/COMPLETE, so its own final files count. It does not count scheduler logs, allocated-block overhead or transient/deleted files and is not a hard cap. Controller final review includes complete outputs plus scheduler logs, logical and allocated sizes, and rejects excess. Trusted finite tools, not arbitrary malfunction/malicious tools, are the basis for this narrow soft aggregate policy.

## Startup, runtime and containment

Controller submission must already strip native loader variables before the first Bash process starts: an in-script unset cannot undo that process's ELF preload. Bash privileged startup ignores BASH_ENV/exported functions. Shell utility PATH is fixed to /usr/bin:/bin. Before child interpreters, unset loader variables and all PYTHON variables; explicitly clean the srun launch environment while preserving every existing SLURM_* variable. New step variables from srun are retained. No environment dump containing credentials is recorded. Only the explicit relevant variable allowlist is recorded in OUT. The remote pinned interpreter is Python 3.11.14; fixture minimum is honestly >=3.10.

Metadata was collected read-only on login under clean env with the exact pinned Python/tools. `runtime_evidence/` contains the expected essential executable/library/import-origin/hash inventory, .pth contents/identities, isolated and ordinary-helper startup sys.path, owner/mode information, current tool versions and scheduler/configuration observations. No broad Conda tree hash, alignment, input scan, database access, install, deployment or job was performed. No unused NumPy/BLAS closure is claimed.

Compute preflight runs the same collector under the actual clean workload environment and compares the complete expected structure before alignment. HOME/TMPDIR are OUT rather than login /tmp; ordinary `-c` sys.path's empty script slot is labeled `<script-directory>`, while actual frozen helper script directories remain source-pinned. This is a bounded essential closure, not every possible runtime code path. Ownership, writable groups and a no-mutation basis still require controller assessment. Library resolution/paths/hashes differing across login and compute fail closed, not silently accepted.

Operational scontrol/srun/sbatch/timeout paths and hashes are recorded; candidate uses resolved scontrol/srun paths. Metadata ldd alone receives /usr/bin:/bin PATH so its shell utilities work; the workload PATH remains exactly the RNASEQ tool directory, including nested samtools view.

Site read-only metadata indicates proctrack/cgroup, task/affinity plus task/cgroup, RAM/swap constraints, AllowedRamSpace=100, AllowedSwapSpace=0. It also reports JobAcctGatherType=jobacct_gather/linux and JobAcctGatherParams=NoOverMemoryKill. These are observations, not evidence of a particular allocation. On compute, the harness verifies current PID membership and current-job memory hierarchy: v1 requires job memory.use_hierarchy=1, 4-GiB memory and combined memsw limits; v2 requires an effective and job-level 4-GiB memory.max. Unknown layouts fail closed. CPU affinity must be exactly four. Controller verifies actual batch/extern containment and reaping; current-process inheritance alone does not prove all scheduler domains.

Outer timeout is best effort at 300 seconds plus 10-second kill grace. Driver wait is 240 seconds plus five-second group cleanup. A narrow TERM/INT-to-exception context around wrapper.child executes its unchanged finally cleanup. Handler restoration follows cleanup. SIGKILL, preflight and postflight interruptions can still leave incomplete evidence. Preflight failures before the workload try block do not promise FAILED receipts. Every exception caught after workload start withdraws only the harness's own COMPLETE, keeps driver outputs/diagnostics, and publishes INCOMPLETE/FAILED where possible. The scheduler's verified 360-second cgroup reaping remains the aggregate termination boundary.

## Acceptance and remaining controller gates

1. Independently review v2 code, manifest and retained runtime closure, including site hooks and group-writable installation ownership. Author/pin runtime and containment decisions only after evidence is sufficient.
2. Verify immutable staged release and exact manifest, private fresh scratch/log filesystem observations, actual output destinations, job owner/node/resources, duplicate/no-overlap checks and operational executable identities. No staging/submission is authorized or performed by this worker.
3. One real compute fixture only if the controller approves. Actual compute closure/cgroup preflight must pass. No login alignment and no automatic retry/budget increase.
4. Independently check output identities, fixed scientific oracle/seams, complete logical and allocated output/log sizes, receipt/COMPLETE hash, final stdout equal to receipt, no FAILED anywhere, and clean job/batch/extern accounting including State, ExitCode, DerivedExitCode, Elapsed, TotalCPU, AllocCPUS, ReqMem, MaxRSS and OOM diagnostics. Any timeout/signal/OOM/nonzero/unexplained extern or partial evidence rejects success. A marker alone never proves success.
5. Only the controller can close the seam-limited ONTF4 sub-scope. Canonical authority, full50 callability, PASS/-uf, HiFi/pooling, full-run memory/throughput, biological claims, production wrapper authorization and historical externOOM remain outside this fixture.
