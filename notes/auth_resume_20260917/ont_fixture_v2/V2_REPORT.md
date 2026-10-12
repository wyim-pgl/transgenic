# V2 amendment report

## Disposition

Implementation and local checks complete. Candidate remains conditional, NOT submission-ready or ONTF4-complete. No controller approval was authored. `runtime_closure_approved` remains false in runtime.json. No transfer/deployment, job, cancellation, installation, commit, canonical data scan or real alignment was performed. All remote commands were bounded read-only runtime/scheduler metadata operations.

The controller-approved soft aggregate policy applies only to the seven-read, below-150-kB trusted fixture. Four CPUs, 4 GiB, six minutes, no GPU is the conditional candidate budget, not full ONT authorization or an externOOM waiver.

## Preserved and amended

- Original ont_fixture: all 28 files match `original_identities.json`. Its manifest and source were not edited.
- Copied all original 15 runtime entries plus tests/stub; all 12 package entries, including the exact nine-file frozen closure, remain byte-identical. Original CLI argument tails, oracle, scientific thresholds and declared seams are unchanged.
- Manifest now has 17 runtime entries: original 15 plus `runtime_closure.py` and `runtime_evidence/expected_closure.json`. Tests, reports and login collection driver are excluded from deployment.
- Approval verifies `finite_fixture_soft_aggregate`, review limit 268435456, hard aggregate null, per-file limit 4194304 and exact 4/4096/360 budget. Start scratch/log filesystem records capture resolved paths, mount/type, available bytes/inodes and ownership/mode.
- Honest >=3.10 guard; actual remote interpreter remains pinned 3.11.14. Bash 4096-block limit is verified by Python as both soft/hard 4194304 bytes.
- Top-level launch cleans loader/Python/BASH_ENV startup influence and uses explicit clean env, retaining SLURM allocation variables. Relevant launch/child environment fields are recorded without a general environment dump.
- TERM/INT exception bridge surrounds unchanged wrapper.child cleanup. Caught late failures withdraw only the harness COMPLETE and retain driver evidence. Logical OUT threshold includes final receipt and COMPLETE, but does not claim scheduler logs, allocated blocks or transient/deleted bytes.
- Compute preflight compares the expected essential runtime closure before alignment and checks actual PID/current-job memory-cgroup membership and limits. Controller assessment remains required for installation immutability, job ownership and batch/extern containment/reaping.

## Verification evidence

`timeout 90 /usr/bin/python3 -B -m unittest -v test_amendments test_fixture`: **17 tests passed, 17.095 seconds**, retained in `green_tests.txt`. System Python is 3.10.12, not remote-runtime validation. `bash -n fixture.sbatch` passed. `sha256sum --strict -c MANIFEST.sha256`: all 17 passed, `manifest_check.txt`.

`red_tests.txt` retains six expected pre-implementation failures for policy, filesystem record, final-output threshold, Python guard, limit/sanitization and signal helper. `red_closure_tests.txt` retains the missing runtime collector failure. Later integration/containment tests extend coverage; not every branch has an independent red-first trace.

### Actual exercised paths versus helpers/simulation

| Check | Evidence and limitation |
| --- | --- |
| Fixed scientific contract | Original six tests include frozen CLI preflight/default panel rejection, deterministic generator, closure pins and complete stub-tool driver/postflight. Stub BAM/index are text, not real-tool evidence. |
| fixture.run | Integrated test invokes real run(), verifies bad policy rejection before synthetic generation, then executes unchanged wrapper.child, frozen CLI, stub pipeline and real postflight. Final stdout failure is injected after publication; own COMPLETE disappears, INCOMPLETE/FAILED and driver junction output remain. |
| External boundaries in run test | Allocation environment/affinity, controller decision, runtime metadata/cgroup result and rlimits are simulated. OUT mode is simulated because local drvfs cannot enforce 0700. This does not certify real Slurm/runtime/resource behavior. |
| Signal cleanup | Real SIGTERM reaches the exception bridge while real wrapper.child is waiting; process group is gone and diagnostics retained. TERM/INT handler raising/restoration is checked separately. No SIGKILL or escaped-session containment proof. |
| Startup cleanup | Actual privileged Bash runs the candidate sanitization/clean-env fragments with injected BASH_ENV and startup variables. Injection marker is absent, loader/Python/unrelated variables are absent, SLURM sentinel survives. Full sbatch/srun is not executed. |
| File limit | Real local Bash limit inherited by Python is exactly (4194304, 4194304). Cluster launch has not run. |
| Aggregate review | Helper counts receipt/COMPLETE and rejects excess. Integrated run invokes final checks. No Slurm-log or allocated-byte accounting is claimed. |
| Filesystem metadata | Local helper and simulated run exercise observations. Actual remote scratch/log paths and ownership remain pending. |
| Runtime inventory | Full collector ran read-only on login; local Python inventory smoke test runs. Compute comparison is executable but not yet exercised on a compute node. |
| Cgroups | Missing-review refusal and v1 hierarchy/memory/memsw rejection tested with synthetic proc/sysfs reads. v2 branch and actual allocation membership remain unverified. |

## Runtime evidence collected

`runtime_evidence/login_observation.json`: login-0, x86_64 Linux/glibc 2.17, timestamp 2026-09-17T23:36:19.432727+00:00. Expected closure is 997375 bytes: four exact pinned tool identities and versions, essential dependency linkage/hashes, four operational executable identities/linkage, and 118 loaded module entries under each isolated and ordinary-helper Python startup, including applicable source/cache bytes and directory ownership/modes.

Minimap2, samtools and Python are dynamically linked; seqkit is static. RNASEQ bare-name resolution is explicit, including samtools view. Both site .pth files are retained with identities/content. coloredlogs activation is environment-conditional and the clean launch excludes its enabling variable; distutils-precedence imports _distutils_hack, whose loaded origin is captured. No loaded sitecustomize/usercustomize was observed. This is an essential trusted-path inventory, not a complete Conda or possible-code-path proof.

Several Python/site files are group-writable (for example .pth mode 0664, uid/gid 3092957). Do not infer immutable runtime from hashes or ownership alone. Controller must assess authorized writers and a no-mutation basis before approval.

Scheduler paths resolve under `/cm/shared/apps/slurm/20.11.7/bin`; timeout is `/usr/bin/timeout`. Retained configuration reports proctrack/cgroup, task/affinity and task/cgroup, ConstrainRAMSpace/ConstrainSwapSpace, AllowedRamSpace=100, AllowedSwapSpace=0 and MemorySwappiness=0. JobAcctGatherParams=NoOverMemoryKill is explicitly retained. These are site observations, not actual-job evidence.

The first metadata collection failed because ldd's shell utilities inherited the workload's RNASEQ-only PATH. The initial failure exit is retained in `runtime_evidence/metadata_failure.txt`; the cause and correction are recorded here; metadata ldd now gets /usr/bin:/bin only, while actual workload PATH remains RNASEQ-only. Subsequent collection succeeded. No library install or loader-path workaround was introduced.

## Independently retain these hashes

```
MANIFEST.sha256
6646242db4f02ffaa0bbebe703c0a750ba1ce91be20f6c658e288ad3dcd785aa

runtime_evidence/expected_closure.json
17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce

runtime_evidence/login_observation.json
c6a5b819116fc6d38a6b81b79b909b0f260b5adebd26cdbd009241521e741263
```

Per-file runtime hashes are in MANIFEST.sha256. The approval must separately bind both manifest and expected closure. No true approval artifact is provided; synthetic temporary test decisions are not authorizations.

## Exact remaining gates and limitations

1. Controller independent v2 review and runtime closure decision, including .pth effects, writable ownership/no-mutation basis, loader/library identities and login-to-compute differences. Advisor planning was attempted but failed due installed client/model incompatibility; no advisor review is claimed.
2. Controller-pinned approval with the exact schema/bindings in PLAN.md, correct separate scheduler log paths and reviewed scheduler cgroup semantics. Before submission/first Bash startup, strip native loader variables externally: in-script cleanup protects later interpreters but cannot undo the first shell's ELF preload. Shell utility PATH is fixed. No invented hard quota attestation.
3. Immutable release staging by the authorized controller, fresh private OUT/log paths, actual filesystem/owner/node checks, duplicate/no-overlap review and the approved single 4-CPU/4-GiB/360-second allocation. This worker performs none of those remote write/submission actions.
4. Actual compute preflight: matching runtime closure, genuine four-CPU affinity and current-job cgroup hierarchy/memory evidence. Unknown layouts or differing files fail closed. Verify batch/extern groups, swap accounting, containment and reaping independently.
5. One real-tool fixture, only after those gates. Outer timeout is best effort. Preflight failures and uncaught/hard signals can leave partial evidence without FAILED; no COMPLETE-only acceptance. No retry or budget increase.
6. Controller final review of complete OUT and scheduler logs, logical and allocated bytes, exact receipt/COMPLETE hash and final stdout, absence of FAILED, scientific oracle/seams, and clean job/batch/extern accounting/OOM diagnostics. Any excess, timeout, signal, nonzero or unexplained extern event rejects success.
7. Controller-only ONTF4 sub-scope decision. Full ONT and historical externOOM remain blocked independently. No claims about canonical authority, full50 callability, PASS/-uf, biological HiFi zero, cs-anchor/correction, molecule assignment, complete chains, B1 execution, full-run throughput or production wrapper approval.

Work record: local amendments, bounded login metadata and local verification complete; deployment, compute evidence and controller acceptance intentionally pending. No independent in-scope implementation TODO is deferred. Graft context query reported approximately 27175 tokens saved; initial orientation timed out.
