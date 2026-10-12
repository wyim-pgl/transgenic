# Independent scoped v2 review

## Disposition

Accept this narrow candidate conditionally for controller authorization of one tiny real-tool fixture. No mandatory source amendment was found for the documented site-v1 execution path. This review is not submission authorization, actual ONTF4 acceptance, production certification or an externOOM waiver. No remote writes, jobs, installs or full runtime experiment were performed.

The explicit controller amendment is implemented: soft aggregate 268435456-byte review threshold, no aggregate hard quota, hard per-file 4194304 bytes, four CPUs, 4096 MiB and 360 seconds. Do not restore the invented hard aggregate quota or require malicious-same-UID-resistant installation immutability.

## Independent verification

- `timeout 90s /usr/bin/python3 -I -B review_v2/run_tests.py`: 17 tests passed in 16.501 seconds, exit 0. Python 3.10.12. Runner redirects temporary directories into review_v2 without changing assertions, fixture paths or source. See local_tests.txt.
- All 17 manifest entries verified, exit 0; `bash -n fixture.sbatch` exit 0.
- identities.json records all v2 files outside review_v2 unchanged before/after tests. All 28 original fixture files remain unchanged and match original_identities.json.
- All package files, including the nine-file frozen closure, are byte-identical to v1. AST comparison confirms generate, entry, panel_seam, bindings, postflight and all other original functions except run are unchanged. See structural_checks.json.
- Read the complete amended fixture, collector, sbatch, runtime, plan/report and both test files, previous independent review, retained login observation and controller ownership evidence. Examined only necessary keys/values of the 997375-byte expected closure, not a full dump.
- Graft orientation and targeted query timed out; direct reads were limited to the specified review scope. Advisor consultation failed on installed-client/model incompatibility. No advisor endorsement is claimed.

## Amendment and scientific boundary

check_policy requires exact field types as well as values. Shell uses non-POSIX Bash `ulimit -f 4096`; Python verifies both inherited limits equal 4194304. Final threshold includes receipt and COMPLETE but deliberately excludes scheduler logs, allocated blocks and transient/deleted bytes. Start filesystem records are observations, not reservations. Controller review must include those separately accounted outputs.

The fixed oracle remains exact: fixture_chr inclusive zero-based 20500..20699, plus strand, three distinct ONT names, supported true; 30500..30699, plus strand, two names, supported false. Exact table equality excludes every extra row, including the +9 decoy. Six selected names include unspliced and exclude outside. Orientation remains UNRESOLVED, exit 4, no-uf, min_eligible=1000 and threshold .95. Nine ledger entries mean eight driver commands plus audit, not a separate samtools-view ledger entry.

The unchanged real frozen CLI and gate are exercised locally using stub tools. Stub BAM/index are not real alignment evidence. Only synthetic authority/read-enum, explicit panel-hash and declared junction-bootstrap seams are accepted. Selection does not independently establish unspliced-read acceptance by the BAM parser. No canonical authority, biological HiFi zero, full50 callability, PASS/-uf, cs-anchor/correction, molecule assignment, complete-chain, B1, throughput or production-wrapper claim follows.

## Startup and runtime trust

The >=3.10 guard is accurate for the frozen annotations; pinned remote Python remains 3.11.14. Privileged Bash suppresses BASH_ENV/functions, strips loader and Python variables for subsequent launches, fixes utility PATH and uses env -i while retaining SLURM variables. Actual startup-fragment tests pass. Controller must strip native loader influence before the first Bash starts, since an in-script unset is too late for its own ELF loader. Use the candidate shebang rather than invoking it through an unreviewed shell startup path.

The closure compares executable/library identities, versions, Python origins/source/cache bytes, site hooks, ownership and modes. Both Python modes inventory 118 modules. The ordinary empty script slot is normalized, and HOME/TMPDIR, hostname, time, PID and platform observations are not incorporated as ephemeral equality fields. ldd address noise is discarded. No guaranteed login-versus-compute mismatch was found. Local filesystem/library differences can still legitimately fail equality; do not bypass that failure. Parent ownership/modes are stable metadata, not evidence that a path can never change.

The two .pth contents are retained. Clean launch does not enable coloredlogs; distutils-precedence activates _distutils_hack, which is inventoried. This is an essential trusted runtime closure, not every possible import/plugin path. No unused NumPy/BLAS inventory is needed for this scope.

controller_runtime_ownership.txt identifies UID 3092957 as wyim, primary p-wyim group 3092957 with listed membership wyim, and environment directories owned by wyim:p-wyim with mode 2775. This supports a proportional same-user/site-admin trust decision despite group-writable Python files. The controller's reported absence of package-manager processes on the queried login host is not a cluster-wide no-writer proof. Approval should explicitly prohibit installation, upgrades and environment edits by this user during the job and accept ordinary trusted site administration. No adversarial same-UID or global never-writable proof is required.

Important boundary: fixture.run rechecks the four primary executable hashes after workload, but does not recollect every imported file/shared library afterward. Before/after runtime-byte evidence for the full reviewed closure therefore remains a controller acceptance check. A bounded metadata-only rehash of the already enumerated files is sufficient; no full environment scan or new real workload is needed. Preserve this condition rather than describing the executable's own postcheck as full-library immutability verification.

## Cgroups and cleanup

For the observed site-v1 layout, the checker requires current PID in current tasks, a matching current-job ancestor, hierarchical accounting enabled, job memory.limit_in_bytes=4294967296 and combined memory.memsw.limit_in_bytes=4294967296. Missing hierarchy/memsw evidence fails before alignment. Descendants inherit the job cap. A tighter descendant/ancestor cap can cause early failure but does not allow this job to exceed its checked aggregate cap. Synthetic v1 rejection tests pass. Runtime metadata is collected before this check, but scientific generation/alignment is after it.

The v2 branch checks effective minimum memory.max and job-level memory.max at 4294967296 plus actual process membership. Unknown layouts or absent job memory evidence fail closed. However, it only records memory.swap.max: it does not reject missing or nonzero swap values. Thus it is not independently a fail-closed combined RAM/swap validator. This is not a blocker for a controller approval explicitly limited to the documented no-swap site-v1 path. If the actual node is v2, do not treat scheduler_cgroup_reviewed=true alone as swap evidence: independently establish effective no-swap containment before allowing alignment, or stop this attempt. A future autonomous v2 approval would need a narrow swap check and v2 tests, not a broader wrapper rewrite. Actual v2 behavior was not tested here.

Controller must verify the actual mount/layout, allocation owner/node/four-CPU affinity, job/batch/extern hierarchy and scheduler containment/reaping. Retained proctrack/cgroup, task/cgroup and RAM/swap settings are supportive site observations. NoOverMemoryKill is an accounting setting, not evidence that memory cgroups are absent, and not permission to ignore OOM diagnostics.

TERM/INT bridge reaches unchanged wrapper.child finally cleanup. A real local SIGTERM test verifies its process group is gone. The late-publication failure test preserves diagnostics and withdraws the harness COMPLETE. This does not prove SIGKILL or escaped-session cleanup. Outer timeout remains best effort. The 360-second allocation is a wall-time request, not an exact all-processes-gone timestamp: retained KillWait and UnkillableStepTimeout permit scheduler cleanup latency. Require observed termination and clean accounting, not an invented instantaneous-reaping guarantee.

## Conditions for the single attempt and final acceptance

1. Controller separately authors and pins the exact PLAN.md approval fields, manifest and expected closure. Bind trusted-user/site-admin runtime policy and documented site-v1 containment assumptions. runtime.json remains unapproved metadata; this review does not change it.
2. Verify immutable staged candidate identity, fresh private OUT, separate real scheduler stdout/stderr paths, filesystem/ownership observations, canonical-tree separation and no duplicate/concurrent work. Clean first-shell startup externally. Only one 4-CPU/4-GiB/six-minute attempt, no automatic retry or budget increase.
3. Actual compute closure equality, four-CPU affinity and current-job cgroup preflight must pass. Unexpected layout/runtime drift is a refusal, not grounds to silently weaken comparison. Confirm batch/extern and no-swap conditions described above.
4. After execution, verify full reviewed runtime file hashes remain unchanged under the no-install policy. Check complete output plus scheduler logs, logical and allocated sizes against the soft review threshold; receipt/COMPLETE hash agreement; final stdout equal to receipt; unchanged scientific inputs/release; exact fixed oracle and command/exit linkage; no FAILED or unexplained partial evidence.
5. Require clean job, batch and extern State, ExitCode, DerivedExitCode, resource/accounting and OOM diagnostics, plus confirmed termination. Any timeout, signal, nonzero, OOM, excess or unexplained extern event rejects success. No marker-only acceptance. Only the controller may subsequently accept the seam-limited F4 result. Full ONT and historical externOOM gates remain unchanged.

## Independently retained hashes

```
MANIFEST.sha256
6646242db4f02ffaa0bbebe703c0a750ba1ce91be20f6c658e288ad3dcd785aa
runtime_evidence/expected_closure.json
17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce
runtime_evidence/login_observation.json
c6a5b819116fc6d38a6b81b79b909b0f260b5adebd26cdbd009241521e741263
controller_runtime_ownership.txt
e684ac14b70d99c3165f05ef18eb24b2454ff4f7e949771cc85673d6d05f6c41
```

Work record: scoped independent review, compatible local tests, syntax/manifest and nonmutation checks complete. Controller authorization, staging, compute checks, real run and final acceptance remain intentionally pending. No candidate source edits or independent implementation TODOs remain.
