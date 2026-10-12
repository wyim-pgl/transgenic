# Independent scoped EST deployment fix re-review

## Verdict: REQUEST CHANGES for one remaining D2 preparation guard

D1 passes the scoped lifecycle review and bounded synthetic probes. The independent hash bootstrap, exclusive reservation, explicit append-mode log argv and no-execution behavior address the main D2 design gap. However, the helper reports successful preparation without verifying that its requested private permissions actually took effect. This is reproducible in the permitted local evidence directory and fails one of the supplied 25 tests. Scoped preparation approval is withheld pending a fail-closed permission check or an equally explicit, independently verified access-control prerequisite.

This is a new independent re-review, not a resumed original reviewer session. No source edits, remote actions, submissions, commits, real-input execution or broad binder re-review occurred. All generated artifacts are under this recovery directory, principally `est_deploy_fix_review_evidence/`. Do not submit or reuse the retained synthetic release or reservation.

## Remaining blocker D2.1: requested private modes are not verified

File: `notes/seed123_execution_20260917/est_deployment/prepare_controller.py:60-85`, especially namespace creation at line 60, stream creation at line 63, receipt chmod at line 77 and unconditional successful return at line 85. CLI lines 95-96 then print the receipt and executable submission argv.

Reviewed helper SHA256: `130312188bd022aa534b4701bf9cbe379539a29075585516fc6cb9b48ce0bd3b`.

The helper requests 0700 for the namespace and 0600 for streams/receipt, but never checks the resulting access modes. On this `/mnt/f` review filesystem, those calls return successfully while stat reports 0777 for all four objects. The helper nevertheless exits 0, writes `PREPARED_NOT_SUBMITTED`, and prints the sbatch argv. Thus the asserted private-namespace contract is not fail-closed on a filesystem that does not implement the requested POSIX modes.

Evidence:

- `suite.stderr`: 25 supplied tests ran in 55.092 seconds, 24 passed; `test_controller_preparation_contract` failed at `test_deployment.py:255`, with `511 != 448` (0777 versus 0700).
- `helper-permission-result.json`: independent real helper CLI invocation with `-I -B` exited 0; recorded namespace, stdout, stderr and preparation receipt modes are all `0o777`. The repeated identical CLI invocation exited 1, so exclusive no-reuse protection does work.
- `permission-reservation/preparation.json`, `prepare.stdout`, `prepare.stderr`, `refuse-reuse.stderr`: retained exact synthetic preparation evidence. The reserved stdout/stderr remain empty and the worker output is absent. No printed argv was executed.
- `adversarial-results.json`: independent in-process preparation reproduced the same effective modes, also verified stream contents survive refused reuse.

This is evidence about the local filesystem and missing helper validation, not proof that GPFS ignores permissions, that Windows ACLs permit another user access, or that the future cluster is insecure. Those access-control mechanisms were not inspected. The helper currently has no guard that distinguishes a valid private POSIX reservation from this unsupported environment.

Minimal resolution: after creating each reserved object, verify owner/private access modes before returning or printing usable submission argv. Fail closed and retain the failed reservation if the filesystem cannot establish the contract. Add a bounded regression for successful filesystem calls whose resulting modes are too permissive. Alternatively, provide a concrete independently checked access-control precondition and reject preparations lacking that evidence. Merely rerunning tests on a different filesystem would not add the missing guard. No broader binder or biological changes are needed.

## D1 findings: no remaining blocker found within scope

Reviewed complete launcher, shell, helper and supplied tests, and exact diffs against `est_deploy_fix_backup/`. Builder diff is empty. Consulted graft before the accepted worker source, then read only signal installation, cleanup and receipt/CLI lifecycle spans (`est_real_sample.py:245-269,490-602`). No accepted-worker computation or large input was run.

- `run_compute.py:180-207`: launcher handlers are installed before recorded preflight; start receipt is durable on successful filesystem operations. First TERM interrupts preflight, repeated TERM cannot repeatedly throw through launcher finalization.
- Worker TERM/ALRM save/restore the launcher handlers; scoped normal worker handling remains unchanged. Launcher USR1 remains independently available while the worker owns TERM/ALRM.
- `run_compute.py:230-264`: postchecks share one timer and include report binding. Deadline expiration prevents subsequent checks and success publication. Existing diagnostics are retained.
- `run_compute.py:267-301`: candidate-success publication is reconciled after interruptions, including signals pending when the receipt mask is installed. Successful verification is moved to `verification.interrupted.json`, final result becomes INCOMPLETE, and return is nonzero. There is no successful verification retry or worker-artifact deletion.
- `sample.sbatch:33-61`: only the first shell interruption is forwarded, an independent watchdog supplies USR1, waiting survives repeated signals, and an interrupted shell cannot return success. The supplied shell probe passed with a scaled watchdog and synthetic child.

The supplied regressions passed for TERM in preflight/postflight, repeated TERM, first TERM during result/verification publication, deadline during cleanup analogue, shared postcheck deadline, package/runtime failures and diagnostic/marker corruption.

Additional independent probes passed for TERM during start publication and report binding; TERM and USR1 pending at the receipt-mask boundary; USR1 during result and verification publication; and expiration specifically during report binding. For every applicable interruption probe, final status was INCOMPLETE, return was nonzero, active verification was absent, and the existing tiny diagnostics were byte-identical. Interrupted verification candidates were retained rather than deleted.

These conclusions retain the documented boundary: require process/job completion and independent receipt acceptance, not observation of an in-flight candidate receipt. Signals after the defined logical commit boundary are outside the work lifecycle. SIGKILL, OOM, interpreter/native failure, failed filesystem operations and uninterruptible I/O can leave partial evidence; missing final receipts are non-success. The 35/40-second design bounds interruptible work, not kernel/fsync latency. Startup before handler installation remains covered only by the external reservation/log contract. No cluster timing guarantee was measured.

## D2 checks that passed

`prepare_controller.py:44-59` directly reads and hashes the ledger and entrypoint byte streams before namespace creation. Externally supplied ledger, launcher and batch pins must agree. It does not import or execute release code, invoke the launcher, spawn a subprocess or submit anything. Supplied wrong-pin/changed-byte tests passed without executing the inserted marker payloads.

Reservations refuse existing namespaces, symlink paths, hardlinked entrypoints, parent traversal, relative/percent-expanded stream paths, collisions, outside paths and reserved names. Extra independent probes covered hardlinked launcher bytes, a symlinked release path and parent traversal. Exact argv has distinct absolute output/error paths, `--open-mode=append`, `--export=NIL`, and `--no-requeue`. Refused reuse preserved existing stream content. The permission issue above is the sole remaining blocker found here.

Independent review/pinning of the helper itself, a never-modified released package, exactly one authorized submission, recording submission errors externally, shared-path visibility and never reusing reservations remain external controller responsibilities. Hashing mutable launcher code from inside itself is not being used as the trust bootstrap.

## Commands and evidence

All children were bounded and used local Python 3.8.20, not the remote pinned 3.11.14 interpreter. `TMPDIR` was deliberately inside the recovery evidence directory to comply with the write restriction. Bytecode writes were disabled.

```sh
E=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_recovery_20260917_1400/est_deploy_fix_review_evidence
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$E" timeout 145s python3 -B "$E/review_checks.py"
```

Initial harness exit 1, accurately reflecting the supplied permission-contract failure and two initial independent-probe assertions. `commands.json` records child argv/timeouts/results: supplied suite timeout 90 seconds, shell syntax timeout 5 seconds, independent probes timeout 45 seconds. Shell syntax exited 0.

The initial independent report-binding deadline probe used 0.1 seconds. On this slower filesystem, the shared deadline correctly expired before binding, producing SKIPPED_DEADLINE rather than the probe's expected FAIL. This was a probe timing assumption, not a launcher defect. The review-only probe was changed to a 2-second shared budget and a 5-second binding pause, with a 5-second total assertion. Its mode assertion was changed to record the already-established blocker while continuing the remaining retention checks. No production test or source was changed.

The second independent probe run (`adversarial-second-command.json`) exited 0: 8 tests in 17.394 seconds. This does not mean the supplied permission test passed. First-run logs/results remain retained separately. The standalone permission reproduction (`helper_permission_probe.py`, outer timeout 30 seconds, each child timeout 10 seconds) then confirmed the helper's success despite mismatched modes and refusal of reuse. Its exact CLI argv/results are in `helper-permission-result.json`.

Advisor consultation was attempted twice but unavailable due client/model-version incompatibility; no advisor endorsement is claimed. Graft reported approximately 58,977 tokens saved.

## Exact reviewed pins and preservation

Preparation directory: `notes/seed123_execution_20260917/est_deployment/`.

```text
a75c460c6e0e078565a9177d0ac712ae9f78315f012abfe4fd299cd9c0aa0cf5  build_release.py
0c4321cb199374907dc99680a9ad8fb2c01202e7e17a4c62f6573543dc0697ea  run_compute.py
7fe7c107e5796043393ee6eea52d8db96cf327877577cf1679c658425ef31f1e  sample.sbatch
ee138ed82e98e2511571d5d4e596d846b8e8ce0b3b673082e47355373ec9567c  test_deployment.py
130312188bd022aa534b4701bf9cbe379539a29075585516fc6cb9b48ce0bd3b  prepare_controller.py
```

`before.json` and `after.json` match. Exact diffs are retained as `*.diff`; no builder changes were found. Final preservation/forbidden-character check is recorded in `final-verification.json`.

Completed: independent D1/D2 diff/lifecycle review, bounded 25-test suite, additional signal/deadline/namespace probes, retained preparation-only CLI reproduction and source preservation checks. Blocked: scoped preparation approval on D2.1. Next: minimal private-permission guard and scoped regression recheck. Deployment/runtime validation and biological acceptance remain separately unauthorized.
