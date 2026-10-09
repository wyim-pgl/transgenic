# Task2 integration brief (read with SPEC, PLAN and Task1 API report)

Implement only after Task1 review gate closes. Use root's Task1 report .superpowers/sdd/PLAN/task1-report.md. The core public API is stable; destination received-stage publication is the one planned extension.

## Real deployment facts

DeltaAI host Python3.6.15; source user wcyim. Pronghorn Python3.11.14 at /data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python, user wyim. Pronghorn ssh alias dtai is already authenticated and resolves gh-login03. rsync3.1.2 at pronghorn,3.2.7 at DeltaAI. Use pull, not reverse SSH. See ENVIRONMENT.md.

## Public CLI

One driver checkpoint_archive_cli.py beside core, stdlib only. It supports JSON config, metadata-only default plan, explicit --apply for copies, separate --remove-source for source cleanup, --once or bounded --watch --interval/--cycles. Config explicitly distinguishes source_kind run/archive, local HDD vs SSH-pull source and destination roots, remote host/python/helper path, minimum age/free reserve, per-pass bytes/checkpoint cap, transport timeouts/rate limit. Default examples must not bulk-delete data when merely installed.

Read-only production plan: B5 source /work/nvme/bilv/wcyim/transgenic/runs/seed123; future WP5 sources b6_wp5_pilot_B6 and b6_wp5_pilot_C2 can be missing and should report waiting without creating them. Destination HDD /work/hdd/bilv/wcyim/transgenic/checkpoint_archive; pronghorn /data/gpfs/assoc/pgl/data/Transgenic/archives/checkpoint_archive. User explicitly requested both destinations and repeated small login-node tests.

## Host storage checks

Use actual lfs project/getstripe/quota to enforce ddn_hdd/project22169 and pool soft/hard headroom, including each created file. File project flags differ from directory inheritance P; validate project numeric ID without demanding a directory-inherit flag on regular files. Reuse sibling storage_preflight.quota_headroom where sensible. Pronghorn destination uses statvfs plus configured reserve and external verified fileset quota facts; no false claim of per-user space reservation.

## Pull and cleanup boundary

Invoke source helper using strictly quoted argv through ssh BatchMode=yes, bounded connection timeout. JSON requests over stdin avoid shell injection. No runtime secrets printed. rsync --protect-args and bounded timeout; never --remove-source-files or --delete. Stage paths must be controlled; fresh files preserve destination storage layout. Rehash received payload against the prepared immutable manifest; publish only after durable receipt creation/fsync. Add publish_received to core with red/green tests rather than forge source path or receipt fields.

Source cleanup must occur only after coordinator has independently verified and durably published destination. Keep appropriate destination lock during verify/remote acknowledgment/removal. Core locks are nonreentrant; avoid recursively reacquiring the same flock. The authenticated coordinator is a trusted transport boundary, not cryptographic proof: source helper checks exact manifest/receipt binding and rehashes unchanged source/protection before removal. Reject stale acknowledgment or changed source. Never treat an unverified remote 'success' string as adequate.

Automatic restart must discover source cleanup journals/tombstones, not only normal epoch candidates (a tombstoned epoch no longer appears in plan). Add a reviewed metadata-only pending-transactions API/helper action if needed; validate stored receipt against configured destination/root and reverify its bytes before resuming cleanup. Default dry-run reports pending transactions without changing them. Keep pending transaction bytes within the configured per-pass limits.

Auth loss/timeout/changed source/low quota leaves source. Replay and ambiguous acknowledgment handling must be explicit, idempotent and testable. JSON status should distinguish copied, verified, removed, protected, waiting and error; dry-run does no binary hashing or destination/source metadata mutation. Watch must not spin after error or hide failure.

## Tests and report

Real-file CLI subprocess tests with executable transport doubles where no real SSH is available locally; test actual protocol and user-visible bytes/status, not source strings. Include path spaces/metacharacters and invalid remote host option injection. Repeated ordering/fault cases, actual tiny two-host tests later by root. Remote helper should be usable with inherited stdin pipes during actual SSH tests.

Own files CLI, tests/test_checkpoint_archive_cli.py, deploy/deltaai/checkpoint_archive.example.json, docs/checkpoint_archive.md; may extend core/test_core for received-stage API with explicit report. Commit source/docs/tests, not root operational notes. No remote operations, production data deletion, jobs, extra agents. Report exact CLI examples, runtime schema, transport guarantees/limits, RED-GREEN test outputs and commit in .superpowers/sdd/PLAN/task2-report.md.
