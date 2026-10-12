# Queued authorization correction: release_v3

## Status and immediate blocker

Implementation and local verification are complete. **Execution remains blocked. No authorization, submission, cancellation, transfer, or training approval was created. Independent review of these new bytes is still required.**

A read-only observation during implementation found that old3163986 had changed from PENDING to FAILED. The subsequent allocation-only accounting query reports:

- Job3163986, owner wcyim/97434, account bilv-dtai-gh, ghx4, FAILED.
- Start2026-09-17T18:47:46, end2026-09-17T18:48:37, local CDT.
- ElapsedRaw51, one allocated node, four GPUs, AllocCPUS288, ReqTRES cpu64, node gh077.

See `review/old3163986-history.json`. This is incompatible with the expressly required CANCELLED and zero-allocation history. The implementation rejects it. It must not be relabeled CANCELLED, excluded from accounting, or repaired with a fabricated authorization. Controller reconciliation and any changed authority are separate from this implementation. The historical allocation also shows that exclusive placement can allocate288 CPUs despite a64-CPU request. The requested exact64-allocated-CPU gate is retained; the controller must resolve any scheduler incompatibility before execution rather than treating the directive as proof.

## Scope

Only new files under `notes/auth_resume_20260917/gpu/deployment_v2/` were intentionally written for the implementation and evidence. Prior top scripts/tests were copied, then adapted only for the gate and new immutable namespace. `deployment_v1/release_v2`, the original final_p1 tree, and the candidate source archive were not edited. The preservation verifier checks all files of deployment_v1 and final_p1 before/after its tests, the existing original preservation inventory, and the independently published old transport digest.

The entire prior `deployment_v1/review/INDEPENDENT_REVIEW.md` was read. Existing source/probe, profile123, quota, cache, mounts, preparation guards,1800s Slurm bound,1700s outer timeout,900s inner probe bound, and persistent one-shot marker remain. No new deployment feature, lock service, retry controller, or background refresh mechanism was introduced.

Graft was consulted before source inspection; its tool automatically refreshed its context graph. No manual source-worktree or graph edit was made. An advisor consultation was attempted but failed due to client/model incompatibility; no advisor approval is claimed.

## Contract and ordering

`admission.py` is the executable authorization schema and live checker. `python admission.py` prints the deliberately unapproved example also included as `authorization.example.json`. It is a template, not a record of true approval, cancellation, or reservation. All approval/provenance fields must be supplied by the controller after reviewing the exact package. `approved:false` and `issued_at_unix:null` intentionally prevent use as authority.

The canonical owner-checked authorization lives at:

`/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_controller_authorization_v3.json`

Required external pins are `DEPLOY_CONTROLLER_GATE_SHA256`, `DEPLOY_RELEASE_TRANSPORT_SHA256`, and `DEPLOY_RELEASE_LEDGER_SHA256`. Canonicalization is sorted-key compact JSON, finite numbers only, trailing LF. The controller must hash and pin the actual canonical bytes before submission. Pretty-printed example bytes are not accepted as canonical authority.

The true `issued_at_unix` must not be future-dated but has no300second expiration. Neither the file nor its queued digest is refreshed. The authority binds release JSON, transport and ledger identities; development-single/123/mvp_finetune; exactly one exclusive one-node/four-GPU/64-CPU attempt;1800s and0.5node-hours/2GPU-hours; unchanged5node-hours/20GPU-hours; no training/followup/requeue/restart/resume/retry; exact new namespace; campaign start2026-09-16T00:00:00 America/Chicago; old IDs and cancellation-history reference; staging/review references; one controller reservation identity retained through terminal reconciliation. B5 history IDs are identities for controller confirmation, not hardcoded claims about their runtime state.

The panel candidate remains an unsubmitted durable1node-hour/4GPU-hour reservation. Other declared unsubmitted reservations add both dimensions; unknown or already-submitted additional diagnostic admissions are rejected. The single controller must serialize admissions and retain the current reservation while pending and through terminal reconciliation.

Host execution validates small pinned authority/ledger identities, creates only the fresh job control directory, stores the unchanged authorization, then performs actual queries inside the1700s outer timeout. Source archive verification, payload copying, large hashes, evidence rename, cache copy, checkpoint staging and S0/P1 happen only after admission. The shell's full ledger hashing moved behind admission with the host's existing per-file pin checks. The controller must have independently accepted and protected the immutable wrapper bytes before execution; no new remote filesystem immutability claim is made.

## Live observations

There is no incoming fresh-observation JSON interface. Query arguments are constructed in the host checker and use absolute trusted executable paths. Each query has one15s timeout and no retry. Process-group termination covers descendants of the accounts script. Two successful scheduler snapshots are compared to detect changed admissions/resources; the second snapshot is a race check, never a retry after failure. Any failure stops immediately and preserves raw evidence and the exclusive claim.

Stored query JSON includes exact argv, stdout/stderr, their SHA256 values, exit status, timeout/error state, wall-clock and monotonic start/end times. Final decision evidence records admission or failure and derived costs/reserves. All observations must be successful, nonempty, warning-free, nonfuture, clock-consistent and at most300s old by monotonic time at decision. Accounting timestamps are interpreted explicitly in America/Chicago with ambiguous/nonexistent local times rejected.

The live checker obtains:

- Actual local scheduler cluster identity, account association, and complete visible queue census.
- Per-relevant-job scontrol records, with actual ownership, identity, allocation TRES, requested TRES, limits, requeue/restart state, command, workdir and placement.
- Full allocation-only sacct rows since the explicit campaign start, including duplicate-history detection. Steps are not counted as allocations. Registered old history must be present. Unregistered diagnostic rows and active diagnostic jobs fail closed. Missing/lagging/conflicting rows fail closed.
- A second per-job stable-resource snapshot and queue census. Other active allocations may not occupy the current node; terminal queue remnants are not treated as active occupancy.
- Account script identity/metadata, usage factor/flags, underlying billing/GPU-minute counters, and the actual accounts balance report.

Current elapsed startup is not added again: the current allocation reserves its entire0.5node-hour/2GPU-hour maximum exactly once. Completed/failed registered allocations use full allocation elapsed time and node/GPU counts; steps are ignored and duplicate allocation identities fail closed. All other declared reservations are added. Both cumulative dimensions are independently enforced.

## Verified finance mechanism

Read-only access is through `pronghorn` then its configured `dtai` host. The local machine has no direct dtai alias. No remote writes, cancellations or submissions were executed.

The installed accounts script is8705bytes, owner67674, mode755, SHA256:

`44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5`

Its source obtains `GrpTRESMins` billing counters from scontrol and divides expanded billing minutes by1000 and60. The observed account usage factor is1.000000 with NoDecay,UsageFactorSafe. The observed account billing-minutes/GPU-minutes agree after the1000 expansion within integer-counter rounding. Existing B5 allocation rows have billing4000 and four GPUs. These observations ground the conversion; the live checker verifies the script identity, current factor/flags, current counter consistency, and billing1000 per requested/allocated GPU for relevant cost calculations. Unknown formats, units, missing balance, changed script or inconsistency fail closed.

The operational reserve is the **full maximum wall time times allocated GPUs for every outstanding running B5 job, or requested GPUs for every outstanding pending B5 job**, plus the entire20GPU-hour diagnostic envelope and20GPU-hour operational margin. It does not guess remaining B5 duration or subtract already elapsed B5 work. The displayed balance is conservatively reduced by one hour for integer formatting. This is an operational controller reserve, not a training cap or measured training cost. Historical1154hours is never used as a runtime constant.

`review/machine-probe.json`, `scoped-probe.json`, and `unit-factor-probe.json` retain read-only reconnaissance commands and raw results. One machine-format probe unintentionally requested global job/accounting metadata and returned approximately6.3MB, broader than the intended small probe. Its raw record is retained for audit and is not part of the runtime transport or a substitute for live admission. Subsequent probes were scoped. The initial JobIDRaw squeue format probe was unsupported; the final implementation uses the subsequently verified `%i|%j|%u|%a|%T|%N` format. No runtime pass is inferred from these probes.

Read-only observations are not atomic with future scheduler or account activity. Snapshot comparisons cannot guarantee future activity by other users/controllers. Safety still requires the durable serialized controller reservation plus scheduler exclusivity. No distributed lock service is claimed.

## Tests and immutable identity

The first five new regression tests were written and run before the implementation, producing the expected five missing-feature failures in `review/tdd-red.txt`. Further admission and integration cases were added during implementation. The Python3.8 timezone incompatibility found by the expanded suite was corrected using native timezone conversion. Synthetic fixtures are explicitly labeled and are not authorizations or real scheduler measurements.

Final verification, with bytecode disabled and temporary writes under `review/tmp/`:

| Command | Result |
| --- | --- |
| `python -m unittest test_deploy test_candidate test_queued_gate test_admission_cases` |34passed,12.219s suite time, exit0 |
| Original final_p1: `python -m unittest test_deploy` |11passed,20.078s suite time, exit0 |
| `/bin/bash -n p1.sbatch` |exit0 |
| In release_v3: `/usr/bin/sha256sum -c release.sha256` |exit0 |

The interpreter was `/home/wyim/miniconda3/bin/python`. Exact commands, working directories and wall durations are in `review/test-results.json`. Tests cover24hour queued authority with unchanged digest and actual-runner fixture observations;300s freshness boundary; stale/future/failed/incomplete observations; bad allocation, active/missing old history, missing account, B5 maximum reserve, other diagnostics, each cap independently, self once, step deduplication, no stage effects before admission, bounded process failure, persisted failure evidence, and an actual local shell one-shot failure/refusal to rerun. Original/adapted archive/cache/quota/profile/mount guards remain exercised.

`review/verify.py` built the transport exclusively and verified exact transport inventory, source closure, release ledger, top/release script identity, unchanged operator pins and preservation. It is local verification, not independent review.

| Item | SHA256 |
| --- | --- |
| release_v3.tar | `ce8998b5f598d7cedbf4000d6f4a7319e43283e4d5ff6efb917b5c4bd7d9c95a` |
| release.json | `3941d6a86d2c21a131e0e2cf3dcb4d06d1b7a687480fd79d7ba6549b4ba03bf2` |
| release.sha256 | `1dc82402d9d16c75c8214a22d07d83bcc4f3109734efccc8d4a3d6045f514ef0` |
| Same103-file source archive | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |
| Same89-file comparison | `4a68eb3d299cd7e7cf7af9ce4e774f2ff18359f094d425cc9cde7becf4d67e97` |
| Same14-file decoder closure | `a651b9fbec9e477198d15c401276d8b45240d5d16212357e1c6eef73e7a67800` |

`HASHES.sha256` and `review/HASHES.sha256` pin implementation and review evidence respectively. The release and transport must not be patched. On filesystems where chmod does not establish immutability, the external digests and exact membership remain the identity.

## Handoff

Completed: wrapper-only queued gate split, executable unapproved schema/example, focused tests, unchanged original tests, finance mechanism grounding, release_v3/transport and local identity/preservation verification.

Blocked: independent new-code review; controller reconciliation of old3163986 FAILED with51allocated seconds; controller confirmation of feasible actual CPU geometry; actual approved durable authorization; all execution actions.

The controller must review this implementation, the exact new transport, the old-history evidence, and an independent new-code review before any execution decision. No stale instructions to cancel a PENDING old3163986 may be replayed from the prior review.
