# Scoped EST deployment D1/D2 fixes

## Status and scope

Implemented locally; ready for independent re-review, not deployment approval. No remote actions, submission, commits, real-input execution, dependency installation, accepted binder changes, or biological logic changes. Only the preparation launcher, batch script and existing tests were modified, and one standalone controller helper was added. `build_release.py` is byte-identical to its backup.

Before editing, all four original preparation files were copied to the newly created `est_deploy_fix_backup/` beside this report. Their hashes match the four original hashes in `EST_DEPLOY_REVIEW.md`. Before/after hashes are retained in `est_deploy_fix_final_hashes.json`.

Read the complete review and tiny signal probe. Graft map/query attempts timed out at 20 and 25 seconds; exact review-named files and worker signal/cleanup spans were used instead. Advisor consultation was attempted but unavailable because of client/model-version incompatibility. No advisor or independent-review endorsement is claimed.

## D1: launcher lifecycle and grace handling

`est_deployment/run_compute.py` now installs launcher TERM/INT/ALRM/USR1 handlers before package/runtime preflight. The compute entrypoint's duplicate preflight was moved inside this recorded lifecycle, including the launcher self-hash check. A fsynced `control/start.json` records INCOMPLETE before preflight, and JSON writes fsync their containing directory. Existing output namespaces remain exclusive and are not reused.

The accepted worker remains unchanged. Its scoped TERM/ALRM handlers save and restore the launcher handlers instead of the default TERM action. Exceptions, interruptions and grace expiry cannot qualify for diagnostic success. Launcher postchecks, including report binding, share a 35-second wall-clock alarm. Expired checks are recorded as `SKIPPED_DEADLINE`, rather than starting another potentially slow check.

`sample.sbatch` forwards only the first TERM/INT to Python, so repeated scheduler signals do not repeatedly interrupt worker cleanup. On that first signal, a separate shell watchdog sends USR1 after 40 seconds. The launcher owns USR1 even while the worker owns TERM/ALRM. This bounds interruptible cleanup/postchecks relative to the 60-second Slurm warning, leaving approximately 20 seconds for receipt I/O. The supervising shell keeps waiting across repeated signals and never returns success for an interrupted allocation. The worker's existing 1-second wait and bounded stderr drain are unchanged.

During receipt publication the launcher handler records interruptions without throwing. If the first interruption arrives while a candidate successful result or verification is being written, the candidate verification is retained as `verification.interrupted.json`, and an INCOMPLETE result is atomically installed. Repeated signals are masked during this reconciliation; pending signals at the receipt commit boundary are accounted for. There is no verification retry and no artifact deletion. Signals after the logical receipt commit are after completion, not a new work phase. Consumers must still wait for the process/job exit and require successful independent acceptance; a candidate receipt observed while the process is running is not acceptance.

Limitations remain explicit: SIGKILL, OOM, interpreter/native failure, filesystem failure, and uninterruptible I/O can leave partial records. The 35/40-second bounds concern interruptible work, not a guarantee that fsync or kernel I/O finishes before Slurm kills the allocation. Output creation/interpreter startup before handler installation can also leave only the external reservation/log evidence. Missing or partial final receipts are non-success, not permission to retry or reuse outputs.

## D2: independent preparation-only controller contract

New `est_deployment/prepare_controller.py` is standalone stdlib code. It never imports release code, invokes the mutable launcher, calls subprocesses, or submits jobs. The controller must independently review/pin this helper and supply the external ledger SHA plus trusted launcher and batch SHA256 values. It hashes the ledger and both byte streams directly, checks their agreement, and refuses wrong pins or changed bytes before creating a namespace. This is independent bootstrap verification, not the launcher's self-check.

The helper exclusively creates one fresh 0700 namespace, exclusively reserves distinct 0600 stdout/stderr files, and writes a fsynced `preparation.json`. It records the external ledger pin, both entrypoint pins, absolute stream paths, fresh launcher output root (`worker`), and exact reviewed argv. The actual worker sample is nested under that root as `worker/sample`. It rejects relative paths, symlinks, parent traversal, Slurm percent expansion, stream collisions, paths outside the namespace, and reserved-name collisions. Existing namespaces are always refused.

Printed argv contains explicit `--output`, `--error`, `--open-mode=append`, `--export=NIL` and `--no-requeue`. Append mode preserves the already reserved empty logs when Slurm opens them before the script begins, including batch startup failures. Scheduler submission errors occur before those streams are opened and must be recorded by the authorized controller in its own submission ledger. No helper code cleans up failures or retries reservations.

Contract: never modify a released package, never reuse a namespace or printed argv, including after submission failure. Preparation consumes the namespace permanently. Authorization and exactly one actual submission remain external controller responsibilities; this preparation helper cannot prevent a human from manually submitting the printed command twice. Private permissions and independent pins are not a hostile same-user/admin security boundary. The release and reserved namespace must be on a shared filesystem visible to the job owner.

Preparation-only recipe (placeholders require independently trusted values):

```sh
python3 -I -B /absolute/reviewed/prepare_controller.py \
  --release /absolute/reviewed/release \
  --ledger-sha "$TRUSTED_LEDGER_SHA" \
  --launcher-sha "$TRUSTED_LAUNCHER_SHA" \
  --batch-sha "$TRUSTED_BATCH_SHA" \
  --namespace /absolute/private-parent/never-used-run-id \
  --stdout /absolute/private-parent/never-used-run-id/stdout \
  --stderr /absolute/private-parent/never-used-run-id/stderr
```

This prints a receipt and reviewed command only. Do not submit any synthetic evidence package or reuse any evidence namespace from this work.

## Red/green evidence and bounded verification

Abbreviations:

```sh
D=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/est_deployment
R=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_recovery_20260917_1400
cd "$D"
```

Tests were added to the existing unittest runner. Before production changes:

```sh
timeout 25s python3 -B test_deployment.py \
  Deployment.test_term_preflight_retains_receipt \
  Deployment.test_term_postflight_retains_receipt \
  Deployment.test_controller_preparation_contract -v > "$R/est_deploy_fix_red.log" 2>&1
```

Exit 1: three expected failures. Both TERM children died before reaching final receipt publication; the independent helper was absent. The child probes use tiny packages and replace runtime identity only, with 5-second readiness/exit bounds, never real runtime/data access.

A second red cycle caught first TERM during receipt publication:

```sh
timeout 20s python3 -B test_deployment.py \
  Deployment.test_first_term_during_result_publication \
  Deployment.test_first_term_during_verification_publication -v \
  > "$R/est_deploy_fix_publication_red.log" 2>&1
```

Exit 1: first case retained a stale successful status; second case returned zero. Publication reconciliation was then added. These tests also send repeated TERM while the receipt writer is deliberately paused.

Final green commands:

```sh
timeout 90s python3 -B test_deployment.py -v > "$R/est_deploy_fix_green_final.log" 2>&1
bash -n sample.sbatch
python3 -B -c 'import ast,pathlib; [ast.parse(pathlib.Path(n).read_text(),filename=n) for n in ("build_release.py","run_compute.py","test_deployment.py","prepare_controller.py")]'
```

All exit 0. Final runner: **25 tests passed in 12.342 seconds**, versus 16 existing tests. Added coverage includes preflight/postflight TERM, repeated TERM, first TERM during both publication writes, shared postcheck deadline, USR1 during a tiny analogue of worker cleanup, shell forwarding exactly once with its watchdog, wrong external/entrypoint pins, changed launcher/batch bytes without execution, existing namespaces, stream-path rejection and explicit append-mode argv. The shell test substitutes a tiny child and a 0.3-second watchdog; it does not run Slurm. The postcheck timeout test scales the 35-second constant to 0.1 seconds. This is bounded local lifecycle evidence, not measurement of real cluster timing or runtime identity.

A separate actual helper CLI invocation with `-I -B` prepared a retained synthetic namespace. Its exact argv is `est_deploy_fix_evidence/preparation-command.json`; stdout, stderr, receipt, empty reserved streams and refused-reuse outputs are retained there. First preparation exited 0; the identical preparation invocation exited 1 with FileExistsError, retaining the original reservation. Neither invocation submitted anything, and its fresh worker output remains absent. Both subprocesses had 10-second timeouts.

That retained synthetic snapshot was built before the final one-line explicit INCOMPLETE assignment in publication reconciliation. Its ledger SHA is `96760e4d2f9c1357b64ee9eb93b116ce879f9e20997d5eb724d148c3dc42aa83`, and its launcher SHA is `2d40d5076958bfcb048ef3c2d30c21ea8be63a452bb76245cbd71e53d458a161`. It was deliberately not modified or reused. Final source bytes, including the assignment, were rebuilt into fresh temporary packages by the final 25-test run. The evidence snapshot's `current-sha256.json` is historical, not the final pin inventory.

## Final SHA256 inventory

Preparation directory `$D`:

```text
a75c460c6e0e078565a9177d0ac712ae9f78315f012abfe4fd299cd9c0aa0cf5  build_release.py
0c4321cb199374907dc99680a9ad8fb2c01202e7e17a4c62f6573543dc0697ea  run_compute.py
7fe7c107e5796043393ee6eea52d8db96cf327877577cf1679c658425ef31f1e  sample.sbatch
ee138ed82e98e2511571d5d4e596d846b8e8ce0b3b673082e47355373ec9567c  test_deployment.py
130312188bd022aa534b4701bf9cbe379539a29075585516fc6cb9b48ce0bd3b  prepare_controller.py
```

Completed: backups, D1/D2 fixes, two red cycles, bounded synthetic green checks, standalone preparation/refused-reuse evidence and final hash inventory. Next: independent reviewer recheck. Actual deployment, runtime validation and biological acceptance remain separate authorization gates. No current-scope implementation TODO remains.
