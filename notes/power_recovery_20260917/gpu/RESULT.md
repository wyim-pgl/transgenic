# Power recovery: GPU/B5 read-only observation

Observed 2026-09-17 07:34:54–07:35:17 UTC (00:34–00:35 PDT; DeltaAI 02:34–02:35 CDT). This is a single bounded status observation plus a distinct filesystem metadata check, not a polling loop.

## Actual result

**GPU3163986 remains PENDING / Priority; S0/P1 not executed or measured.** `squeue`, `sacct`, and `scontrol` agree: owner wcyim, elapsed 0, no allocated TRES/node, Restarts=0. Requested one ghx4 node/4 GPUs/64 CPUs/30min; exclusions gh062,gh093 unchanged. Scheduler's projected start 2026-09-18T03:01:35 (DeltaAI local time) is only an estimate, not a reservation or completion promise.

Filesystem observation corroborates absence of execution outputs: job subtree, S0 input pins, real_tokenizer DONE/FAILED and both scheduler logs are all ENOENT. There is no P1 failure evidence, no successful receipt, and no actual launch-readiness. Pending ExitCode=0:0 is not success. Staged evidence still exists at 1,068,773,376 bytes; metadata only was inspected, not a content/hash validation.

Remote root: `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545`.
Expected job: `ROOT/final-p1-job-3163986`.
Expected logs: `ROOT/staging/final-p1-3163986.out` and `.err`.
Evidence: `ROOT/staging/evidence/fullc0lite.db`.

## Protected B5 and historical attempt

- B5 3131538 RUNNING on gh024, 00:57:25 at observation, 4 GPUs. Batch/extern RUNNING. No changes made.
- B5 3149356–3149363 PENDING / Dependency, elapsed 0; retained unchanged.
- Prior B5 3131537 allocation/batch/extern COMPLETED 0:0, elapsed 10:48:08, gh086. This is accounting only, **not evidence of full training completion or checkpoint acceptance**.
- Obsolete P1 3163545 remains CANCELLED by 97434, elapsed 0, no allocated TRES. No cancellation was performed in this investigation.
- Two recorded P1 attempts have consumed zero allocated node/GPU time so far. This narrow check is not an exhaustive campaign budget certification; the cumulative cap remains 5 node-hours/20 GPU-hours.

## Blocker and next feasible action

The GPU critical path is blocked solely on scheduling at this snapshot. Leave 3163986 and all B5 jobs untouched. No duplicate submission, automatic retry, follow-up GPU job, or training is permissible from this observation. Controller can continue independent CPU/local TODO work while the existing P1 waits; this investigator has no further executable GPU task under the assigned read-only scope.

At a later explicit recovery/status checkpoint, query once again. If terminal, collect bounded small S0 prepared/assets/tiles/consumed-code/pins, wrapper measurement/receipt/DONE-or-FAILED, and stdout/stderr; verify byte binding and semantic REAL_TOKENIZER result rather than scheduler state. Only a genuinely verified P1 PASS permits consideration of Task1 step0 → generation → bf16 → 4rank → resource, and then only after mount/release independent review, current accounting/no-overlap and resource authorization. All six gates and seed-specific bindings remain required before training.

If a future attempt fails after moving evidence, preserve the failed subtree/start marker. Never move/copy evidence back or reuse partial inputs automatically; an independently reviewed recovery with fresh exclusive output and remaining-budget proof is required. No failure repair was implemented here.

## Raw observations and scope

- `live_status_20260917T073453Z.txt`: exact nested SSH command, timestamp, return code, squeue/sacct/scontrol stdout and stderr.
- `p1_path_observations_20260917T073516Z.txt`: exact nested SSH command, timestamp, return code, seven targeted lstat observations.

Both SSH hops used BatchMode=yes and ConnectTimeout=15; local total command timeouts were 75s and 60s. No remote writes/submits/cancels, source changes, heavy hashes, DB reads/scans, recursive output search, or artifact edits. All local writes were limited to this new gpu directory.

Context read: `.atl/resume.md`, current top `.atl/todo.md`, `.atl/quarantine.md` current guidance; `notes/MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md`, `notes/WAIT_WORK_PLAN_20260916.md` Task1, and final_p1 submission/execution-scope/pre-replacement receipts. Graft map/ask preceded exploration; relevant `container_p1` covers `notes/mvp_execution_20260916/final_p1/deploy.py:381–403` (same release span). No source implementation was changed.
