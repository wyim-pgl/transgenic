# Task 1 handoff

## F1 narrow correction (latest)

Changed only the runner-cap predicate to `type(timeout_seconds) is int and 0 < timeout_seconds <= outer_seconds - 120`. The 120-second reserve, five-node-hour cap and native gates are unchanged. No execution argv or mounts were fabricated.

Added eight tests: reject zero, -1, True, False, 1.0 and reserve-overrun 481; accept 1 and maximum 480 with valid synthetic inputs and outer_seconds=600. Before the fix, the 26-test suite reproduced exactly two failures (zero/-1 accepted). After the fix, **26/26 pass (1.325s)**, including all original 18. `followup.py --help` and `bash -n followup.sbatch` both exit zero. `verification.txt` below remains the historical 18-test transcript; this section records the new run.

Current hashes:
- followup.py: `690b7f636b60625da39b13e9ec7a796154dcd9f9cf947f28e3776ff08c94996b`
- test_followup.py: `f354b57092e867fbdeb48ba79c442c28e80156b9248b9ee71e9855e9a2ee5609`
- independent_review.md (unchanged): `5f3647c850f7f0bb2f1accf6a35942c044c42e902f88f0716c69be372f1c230a`

**G1/G2 remain open:** real completed P1/S0, reviewed concrete mounts/release, live accounting and resource authorization are still required. This local F1 fix does not authorize execution. No SSH, jobs, subagents or main-code changes. Controller independent recheck/re-review remains required.

## Original handoff (historical hashes/results below)

Delivered locally; GPU execution remains gated, not completed.

## Files

- `PLAN.md`: exact P1→three-tier step0→generation→bf16→four-rank→resource sequence; seed/readiness/training limits, cache mount retention, clock/node-hour ledger, controller input contract.
- `followup.py`: stdlib one-mode command generator and compute-only native runner. No submission, retry, source edits, fabricated measurement or authorization.
- `followup.sbatch`: bounded one-node/four-GPU/30-minute template, 1700s process timeout +20s kill grace, exclusive STARTED marker; controller must supply actual container argv/mounts and hashes.
- `test_followup.py`: 18 CPU gate/schema tests, including full synthetic S0/P1 command-building fixture. These are NOT GPU evidence.
- `source/revision/scripts/mvp/`: Git archive b660bf1 API excerpt, unchanged and not an alternate production source tree.
- `source-files.json`: SHA256 map of canonical Git archive regular source files at `/work/repo/` paths.
- `verification.txt`, `SHA256SUMS`: exact verification output and deliverable hashes.

## Verification

Initial negative-TDD run failed because the not-yet-written `followup.py` was absent. After implementation, 18/18 tests pass. CLI --help, bash -n and Python AST syntax checks pass. Final commands/results are in verification.txt (AST was also checked directly). No SSH, remote writes, job submission, GPU import, large dataset hashing, main-code change, merge or push was performed. An attempted advisor consultation failed due to local client/model version incompatibility; no advisor review is claimed. Graft auto-refreshed its local ignored graph while indexing the archive; no reviewed source was changed.

## Explicit blockers/controller actions

1. Actual complete final P1 wrapper/DONE and S0 bundle are not yet available here. Latest supplied snapshot remains PENDING 3163986. Obtain actual completion and live accounting before using tools; do not guess a job directory.
2. Controller independently reviews tools, fixes actual mount/release argv, current exclusions and immutable release hash list. Stable P1 `/scratch` and all followup import paths must remain mounted. Generation cache must survive through decoder consumption.
3. Complete live diagnostic accounting and absolute clock authorization are required; reserve FULL outer allocation, including failed time and setup, under cumulative five node-hours. Queue expiry fails closed. Current template is short-preflight-only unless resource genuinely fits its reviewed cap; there is no automatic cap increase.
4. Resource training max wall/three-seed budget, margins and real controller reference remain unselected. Resource mode refuses absent authorization; no invented author/approval.
5. All six GPU measurements are binding/seed-specific. Tool anchors actual seed-1 P1 3163986; seeds 2/3 require separately prepared and independently measured six-receipt sets and reviewed extension, not copied PASS or automatic reauthorization. Five-hour feasibility remains unresolved, not silently widened.

Primary SHA256:

- followup.py: `6183bb04f0dfa7bc5c62957da75254c7f60b3d782f535f60ffb50bad5017ac35`
- followup.sbatch: `6029e9ab9babad78e3f49f7bb14967020825740077a1aef88ece03bf5b29cc1f`
- test_followup.py: `0f09329cd866ebfff2775d3c375fbb1e6bb4587401ddd536ecf2201e45ed3e28`
- PLAN.md: `b948fdfe106dc1e33377ec70e8bb5bb896182ffb2c301235be0adbf80416716f`

Graft reported approximately 898,807 tokens saved across this worker's graph calls.
