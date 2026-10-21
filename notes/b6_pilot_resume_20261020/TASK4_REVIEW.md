# Task 4 independent spec and quality review

Reviewed base `e2bee04265a5e91255607d65f0aa4c7d07d72369` through `b4ab09c` (implementation `8b13234`), against `TASK4_BRIEF.md` and `TASK4_REPORT.md`. The supplied diff adds seven Task4 files and does not modify C2 or source198ce14. Review was read-only except this report; no agents, submissions, cancellations, deployments, or graph writes were performed.

**Spec verdict: changes required — one P1 finding.** The design implements the requested independent B6 operation, shared Task1 receipt acceptance, in-place activation, sequential retirement journal, immutable request/predecessor chain, and final acceptance. Cancellation has a race that violates the explicit running/released-job refusal requirement.

**Quality verdict: changes required — the same P1 mutation-boundary defect.** No separate blocking maintainability finding. The explicitly authorized producer copy is narrow; its model/state/RNG/history/epoch/final checks and signal supervision remain substantially unchanged. This verdict does not add an approval gate or replace parent-owned native/GPU validation.

## Finding

### P1 — Constrain cancellation to PENDING at the Slurm mutation boundary

`notes/b6_pilot_resume_20261020/continue_b6_history.py:258` (execution at lines261–263) constructs the durable cancellation argv as `['scancel', pin['job_id']]`. The controller checks PENDING/JobHeldUser immediately before calling it, but that observation and cancellation are separate scheduler operations. If the held allocation is released and starts between them, this command cancels the now-RUNNING job. It may then be observed as CANCELLED and retirement can advance as if the required held-only cancellation occurred.

The brief requires: “Cancel only the pinned same owned PENDING/JobHeldUser allocation; running/released/requeued/foreign/mismatched jobs refuse.” The unchanged C2 retirement implementation already uses `scancel --state=PENDING --user=<owner> <exact-job-id>`; the new B6 controller drops those server-side filters. Restore at least those filters while retaining the complete identity/held checks, durable per-job intent, terminal-CANCELLED observation, and refusal to reissue a recovered intent. A regression should change the scheduler job to RUNNING after the last observation and verify the pending-filtered command does not cancel it or advance retirement. A released-but-still-pending reason change remains a scheduler observation race; document any stronger atomic guarantee the live scheduler supports rather than claiming a reason filter exists.

Focused verification used the actual `Controller.retire_prepare` with an in-memory scheduler boundary and in-memory journal saves (no file or scheduler mutations). The boundary changes the allocation state after the final observation. Observed output:

```text
argv=['scancel', '3345918']
allocation_state_at_cancel=RUNNING
server_side_pending_guard=False
would_cancel_running=True
durable_phase_sequence=['retire_intent', 'retire_waiting', 'retire_waiting']
```

This reproduces the unsafe command construction and timing window; it does not claim a real Slurm job was cancelled.

## Strengths checked

- Host acceptance calls the committed Task1 predicates and verifies all eight prepared-schema files, raw/canonical hashes, and each prepared trace binding; it checks current small checkpoint metadata, source/helper/config/input/image provenance, and owned parent/batch accounting.
- Retirement ordering is3345918 then3345917; durable recovered cancellation/submission intents are observational only. Zero/multiple discovered submissions never trigger another sbatch. The controller checks full normalized submission argv and actual accounting identity/time.
- Activation embeds historical410 bytes and proof receipts. Reconciliation can accept a progressed live candidate after an uncertain first submission, while a fresh first submission explicitly rechecks live410 before sbatch. Later requests bind accepted predecessor receipts and checkpoint state.
- Separate B6 claim/lock and controller, source/helper manifests, streamed source hashing, host JSON size cap, B6 cache namespace with FX/AOT enabled, and100GiB reserve checks are present. No C2 global monkey-patching, C2 journal mutation, source modification, checkpoint-tree relocation, or generic arm switch was introduced.
- Producer/supervisor differences against unchanged C2 are localized to B6/profile/provenance/mmap/image/receipt bindings. Fresh input verification remains on every rank. Full checkpoint canonicalization, four-device RNG validation, pending gradients, compiled-key/tied-alias compatibility, history prefix, draw-prefix, epoch evaluation, exact4099 markers/export/DONE, nonzero exit refusal, and final recheck are retained.
- Tests cover real Task1 fixture consumption, crash recovery, uncertain progressed submission, missing rank artifacts, second-link pending-gradient/epoch state, and final4099 with periodic latest4000. The supplied35/35 result is reported evidence; this review did not rerun the full suite.

## Parent-owned checks

After the focused cancellation correction and regression, retain the already planned native hostPython3.6 import/CLI and NGC26.08 allocation checks. Live deployment must supply the reviewed full Slurm identity pins and resource/config/manifests, verify actual project22169/ddn_ssd reserve, and establish the real four-rank GPU proof before continuation. Native input/mmap/cache behavior, throughput-derived walltime, and eventual epoch/final execution remain runtime checks; local review and CPU fixtures do not establish scientific GPU PASS.

Review context used graft before outside-diff source checks. Reported graft estimate summed across calls:2,824,833 tokens saved (tool estimate, not measured review consumption).

## Scoped rereview — `01c3bc0` / `e4bb9a2`

**Final scoped spec verdict: changes required. Quality verdict: changes required. The original P1 is partially addressed but remains open.** No separate new defect was identified in the narrow diff.

The implementation now durably records state/user filters and retains identity checks, retirement order and recovery without retry. I inspected the regression and both actual logs: the new test failed against the old argv (1 failure,13.967s), and the updated host suite passed20/20 (89.035s). Its assertions correctly cover a job starting before the simulated cancellation call, no activation, no second-job cancellation and no recovered retry. The unchanged producer/chain suites were not rerun.

### P1 remains — state filtering is not necessarily performed at the cancellation mutation boundary

`notes/b6_pilot_resume_20261020/continue_b6_history.py:259` still omits `--ctld`. My initial review recommendation to add only `--state=PENDING --user=...` was incomplete, and the earlier description of those options as server-side filters is corrected here.

SchedMD documents that ordinary scancel first retrieves jobs, filters them, then issues individual signal RPCs; with `--ctld` and without interactive/sibling/step options, one controller RPC applies the filters. Thus the current argv moves the race inside scancel: it can read PENDING, then cancel after the job starts. See [official scancel PERFORMANCE documentation](https://slurm.schedmd.com/scancel.html).

The [upstream scancel source](https://github.com/SchedMD/slurm/blob/master/src/scancel/scancel.c) confirms the distinction: `_proc_cluster` selects local load/filter/cancel unless `opt.ctld`; `_cancel_job_id` sends a job-ID signal without the state predicate; `_ctld_signal_jobs` instead includes jobs, state and user in its request. This establishes the standard code path, not the installed cluster version or build.

The new fake scheduler applies `--state` atomically immediately before mutation, so it cannot expose the remaining client-filter-to-RPC window. Add explicit `--ctld` where supported by the installed scheduler, verify that support read-only, and make the regression distinguish client-side selection from controller-side filtering by starting the job after the former. Preserve exact job ID, owner/state filters, all existing held/identity checks and no-retry behavior. If the installed version does not provide that controller-filtered operation, record the limitation instead of claiming that `--state` alone is atomic.

The separately documented `Reason=JobHeldUser` race remains honestly qualified: state filtering does not atomically filter the hold reason. Native hostPython3.6/NGC/GPU checks remain parent-owned as before. No source, C2, graph, git or remote state was modified during rereview; only this report was appended.

## Final scoped rereview — round2 `2a48265` / `50a7fac`

**Spec verdict: PASS for the reviewed implementation, with the documented scheduler limitation below. Quality verdict: PASS. Original P1/I1: CLOSED. No remaining actionable finding was identified in the scoped fix.** This supersedes the preceding changes-required verdict for the reproduced PENDING-to-RUNNING cancellation race.

Reviewed `/tmp/b6-task4-fix2-e4bb9a2-50a7fac.diff`, the updated report, parent-captured `SCANCEL_CAPABILITIES.json`, and the actual round2 RED/GREEN logs. The only executable change adds `--ctld` to the durable exact-job, pending-state, owner-constrained cancellation argv. There are no interactive, sibling or step-ID options that would select the documented alternative path. Captured live capability evidence reports Slurm25.11.8 and explicit `--ctld` support. The fix therefore uses the controller-filtered operation identified in the preceding review.

The revised scheduler regression distinguishes stale client selection from controller-side filtering: a job starts after the PENDING snapshot, the old command cancels it, and the fixed command leaves it RUNNING. The logged RED run fails on that cancellation (1 test,12.175s); the logged GREEN host suite passes20/20 (129.677s). The test also checks the persisted exact argv, recovery refusal without another cancellation, empty retirements, no activation/submission and untouched3345917. Existing full identity/held checks, retirement ordering, durable intent and observation-only recovery are unchanged. No additional suite rerun was necessary for this one-option fix; no new implementation or scheduler action was performed by this review.

The report now explicitly retracts the round1 atomic-filter claim. `Reason=JobHeldUser` still is not an atomic controller predicate: a release that remains PENDING after the last observed hold can race cancellation. This is a disclosed scheduler limitation, not a claim that every hold-reason transition is prevented. The reviewed PENDING-to-RUNNING race is closed. Parent-owned nativePython3.6, NGC/four-rank GPU, actual proof and eventual production/final checks remain as previously recorded; this local PASS is not scientific GPU acceptance and creates no new approval gate.
