# Independent final deployment review: release_v5

## Verdict

**APPROVE these exact bytes as an explicit fixed-failure-recovery candidate, subject to the existing controller and compute-runtime gates. This is not permission to execute.** No concrete additional deployment blocker was established in this scoped review. release_v4 is superseded and is not approved by this report.

The four startup corrections and both query-boundary corrections are integrated. Job3163986 remains a real FAILED1:0 allocation, 51seconds on one node/four GPUs. It must never be relabeled as zero-cost cancellation. No automatic retry, removal of the persistent claim, source mutation, training, followup, cancellation or submission is authorized.

Scope: complete selected admission.py, deploy.py, build_release.py and p1.sbatch; adapted and new tests; implementation/reconciliation records; the prior queue-gate and query-capture reviews; affected archived checkpoint staging and profile interfaces. This is not a broad model review or a new approval framework.

## Independently measured immutable identities

| Artifact | SHA256 |
| --- | --- |
| release_v5.tar | `b3dafe316605a8ccaae61e94854fe527bd8591cb4600ae152afae017d2807458` |
| release_v5/release.json | `e0d2262a7aa94b49cdf08de764d6b807c75779dc32a5a7e83ff12810084c5389` |
| release_v5/release.sha256 | `10e2ec018cf00bb9180f8884b44a00a1d376734e7c5abfeb84e97be706facaa9` |
| code.candidate.tar | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |

Transport has exactly14 regular members, no duplicates, links, special files or escaping paths. Every member equals its selected release file. The ledger has exactly the expected13 entries and passes sha256sum. The nested source archive has exactly103 regular source files with matching manifest hashes and safe membership. Top-level tested implementation files equal the selected release bytes.

Frozen deployment_v2/release_v3.tar was compared directly, without rebuilding it. Source archive, source-manifest.json and operator-pins.json are byte-identical. Builder changes only its default output namespace; shell changes only names/paths. Complete wrapper diffs are retained. Before/after hashes preserve156 files across selected release, deployment_v1, deployment_v2, candidate and original final_p1. No prior release was modified.

## Integration findings

### Checkpoint staging and profile binding

`release_v5/deploy.py:115-126,361-411` permits only the three required regular files and the two known real nonsymlink resume directories. Unknown entries, aliases and wrong types reject. Resume directories are excluded rather than loaded, copied or deleted. Source inventory and regular-file metadata are compared across staging; meta/eval are repinned; destination is checked again.

I executed the **actual archived stage_checkpoint.main()** on tiny synthetic inputs through the selected container_stage wrapper, rather than manufacturing its output inventory. It copied and hashed real fixture files and produced:

- checkpoint/: exactly model.safetensors, meta.json, eval.json.
- Parent: checkpoint, run_config.json, runtime_identity.txt, staging_receipt.json, STAGING_DONE.

Thus the new three-file destination guard is compatible with the actual producer: provenance and completion files live in the parent, not inside checkpoint/. Both source resume directories and source regular-file identities were unchanged. Injecting an unknown source entry, a changed source regular file or an extra destination entry after real staging rejected before PREPARE.

Local-environment limitations are explicit: DrvFS rejected renameat2(RENAME_NOREPLACE) with EINVAL, so this inventory fixture used a no-existing-path os.rename adapter for publication only. Local free space was below40GiB, so this tiny-fixture reserve was zero; unchanged production reserve tests passed separately. These adaptations do not establish target-filesystem atomicity or live quota. Heavy PREPARE, tile and GPU work used explicit fixture outputs and are not claimed as executed.

The actual archived PREPARE argument parser accepts the wrapper's development-single and integer123. A wrong-seed receipt rejects immediately after PREPARE. The separate container_p1 path rechecks the binding and rejects a wrong-seed receipt even when its binding digest and outer file pin are recomputed in the fixture. Runtime role remains mvp_finetune, not B5 baseline. See deploy.py:239-244,395-403,500-525 and archived make_launch_readiness.py:377-414.

### Exact failed history, resources and conservative costs

`admission.py:299-389` distinguishes requestedCPU64/CPUsPerTask64/NumTasks1 from allocatedCPU288. Allocation must match NumCPUs and either64 or the trusted current node's CPUTot. It is not an arbitrary CPU override. Capacity is queried twice and must remain stable.

Live accounting requires3163986 FAILED1:0, elapsed51, node1/GPU4, requestedCPU64/allocatedCPU288, gh077 and the recorded start/end. Changed cancellation/zero-time history rejects.3163545 separately remains zero-allocation CANCELLED0:0. Unknown diagnostics and conflicting/missing history fail closed; recognized step rows do not become extra allocations.

The51seconds contribute51/3600node-hours and204/3600GPU-hours. With this attempt's full0.5/2 reservation and the unsubmitted panel1/4 reservation, minimum committed totals are **1.5141666667node-hours and6.0566666667GPU-hours**, before other prior usage. Both5/20caps are enforced independently. Current elapsed is not added twice. All startup, failed work and teardown remain charged.

### Read-only named evidence recovery

`deploy.py:83-112,278-359` binds only the specifically reviewed failed-job evidence pathname and original identity. Admission precedes large hashing/copying. Canonical regular identity, UID97434, no write bits and absent WAL/SHM/journal are required. Source metadata/hash checks surround copying; destination is exclusive0600 inside a private directory and independently checked before0444. Existing destinations and failed partial copies are not reused. No fallback search, rename of old evidence, chmod of old source, database rebuild or old-tree deletion exists.

Evidence1068773376 + checkpoint ceiling2200000000 + HF ceiling2000000000 =5268773376bytes, below5GiB by99935744bytes. Cache copying now enforces its aggregate payload ceiling. The unchanged40GiB reserve, inode headroom and fresh quota checks remain mandatory. Synthetic tests exercise wrong owner/mode, links, WAL, bad pins, source replacement, destination corruption and collision. No real database was hashed or copied here.

### Durable authority, fresh observations and financial integration

`admission.py:69-121,406-487` keeps durable external authorization separate from fresh compute observations. A24hour-old authority passes only with fresh successful observations and unchanged external identity. The example remains unapproved. Freshness, actual account/cluster/allocation, no overlap, B5 reserve and controller serialization remain required.

Two scheduler observations compare stable identity/resource fields, not full output or elapsed RunTime. A complete local live-path fixture advanced RunTime and LastSchedEval plus node CPULoad/FreeMem and was admitted. Changing final CPUsPerTask or CPUTot rejected. Queue membership/state/placement races also reject without retry. Pending placement/start prediction changes may conservatively stop admission; the code does not pretend snapshots guarantee future activity.

Accepted finance-conversion grounding from admission_review is retained, not reinterpreted as a new financial cap. Integration rechecks the account-script metadata/hash, QOS factor/flags, billing-to-GPU-minute consistency, displayed balance consistency and per-allocation billing=1000perGPU. Reserve uses balance minus one hour conservatively, full maximum B5 wall times, the20GPU-hour diagnostic envelope and20GPU-hour operational margin. Neither a historical balance nor a caller-supplied observation substitutes for live commands.

### Query boundary and host compatibility

`admission.py:124-190` enforces16MiB aggregate stdout+stderr raw bytes during binary capture. Reads are bounded by remaining capacity plus one overflow-detection byte. Timeout, non-timeout read failure, overflow and invalid UTF-8 all reject; error paths attempt process-group kill and a bounded two-second reap, close pipes and never unboundedly drain. Failed prefixes are evidence only, never admitted as truncated success. Decoding/JSON can expand stored representation by a bounded factor; this is not a claim of16MiB total process RSS.

Real harmless local children passed stdout-only, stderr-only and combined overflow tests, injected OSError cleanup, timeout, invalid decoding and failed spawn. Supplemental timeout used a forked child retaining pipes. No cleanup errors were recorded in those cases.

Both host modules parse as Python3.6 and avoid text=True. Existing actual-host3.6.15 smoke evidence was read, not rerun remotely. Its final admission/deploy hashes independently match the selected release. It is evidence of host import/API compatibility, not SIF/GPU readiness.

## Fresh verification and required handoff

Independent bounded commands, each with a180second ceiling:

- Adapted suite:42tests passed,15.338s.
- Unchanged original final_p1 suite:11tests passed,21.610s.
- Selected shell syntax and exact release ledger:exit0.
- Additional staging/profile, changing-observation and capture integration checks:passed, with fixture limitations above.

Evidence is exclusively in `../review_final/`: verify.py, integration.py, test-results.json, identities.json, both test logs, release-ledger.txt, shell-syntax.txt, integration-results.json, integration-log.txt, preservation snapshots and v2 comparison diffs. Initial supplemental harness runs exposed local quota/DrvFS limitations and fixture setup mistakes; those are not attributed to the release. Final evidence records the adapted local boundary accurately.

**Execution remains blocked until the controller separately authorizes it.** Required: exact external transport/ledger/authorization pins; genuine terminal-recovery and staging provenance; complete campaign reconciliation; retained single-attempt reservation and serialized diagnostic admissions through terminal reconciliation; preserved B5/account capacity and panel reservation. Compute must then pass fresh trusted Slurm/account/no-overlap/quota checks, actual data/runtime identities and existing S0/P1 gates. One1800second four-GPU allocation only,0.5node-hour/2GPU-hours within5/20; training=false and followup=false.

No jobs, remote actions, actual-data hashes, deployment invocation or implementation edits occurred. Only review artifacts were written under deployment_v3. Advisor consultation was unavailable due to client/model incompatibility. Graft was consulted before source exploration; no graph rebuild was requested.
