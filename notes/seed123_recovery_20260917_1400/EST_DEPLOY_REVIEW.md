# Independent EST deployment review

## Verdict: REQUEST CHANGES before deployment

Reviewed existing preparation only. No production edits, remote access, submission, commits, dependency installation, real-input reads or full binder reinvestigation occurred. Generated files are confined to this report and `est_deploy_review_evidence/` beside it. The accepted binder findings remain closed. Two deployment gaps remain: TERM outside the worker loses the controller failure receipt, and exclusive stream retention is delegated to an absent controller submission contract.

Worktree: `/tmp/est-real-source-binding-20260917`, HEAD `fca911e58fde44124b9359f9adc03a9d20059599`. Existing dirty state was observed, not reset. Read all four preparation files, the complete accepted `EST_SAMPLE_FIX1_REVIEW.md`, and complete `tasknotes/real_sample/sitalica.manifest.json`. Consulted worktree graft map/query/skeleton/import graph before source spans. Source inspection was limited to deployment interfaces, limits, imports and signal lifecycle.

## Blocking findings and exact minimal fixes

### D1. TERM outside the worker bypasses controller finalization

Locations: `est_deployment/run_compute.py:168-219`, `sample.sbatch:9,33`; accepted worker signal lifecycle at `revision/scripts/evidence/est_real_sample.py:256-260,499-502`.

The shell deliberately forwards TERM, but only the worker installs a Python TERM handler. During runtime preflight and after the worker restores the default handler, TERM terminates Python immediately. `except BaseException` and `finally` do not catch the operating system's default TERM termination. Controller results are written only at the end. Consequently even the configured graceful Slurm warning can leave no `control/result.json`, no controller failure status, and no completed independent after-check record.

Independent tiny-fixture reproduction (`signal_probe.py`, `signal-probe.json`):

| TERM point | Child return code | Worker COMPLETE retained | control/result.json retained | Verification receipt |
|---|---:|---|---|---|
| runtime-before | -15 | no | no | absent |
| runtime-after | -15 | yes | no | absent |

The probes patch runtime identity to a bounded sleep solely to reach each phase deterministically. They do not access the actual runtime or real data. The post-worker case preserves the worker's existing files, so this is loss of controller failure/finalization evidence, not false success publication.

Minimal fix: make the launcher own interruption handling for its full lifecycle, without replacing the worker's scoped cleanup semantics. Install a launcher TERM handler before preflight that the worker can restore on return; record interruption as INCOMPLETE, prevent successful verification after interruption, and preserve a durable start record plus final result. Ensure repeated TERM during finalization cannot skip receipt publication, and bound cleanup/postchecks within the Slurm grace window. Add two tiny subprocess regressions covering TERM during preflight and postflight. SIGKILL, OOM and uninterruptible I/O must still be explicitly treated as potentially partial runs, not promised recoverable finalization.

### D2. Exclusive stdout/stderr retention and deployment namespace are only an external comment

Locations: `est_deployment/sample.sbatch:11-12,25-33`; `run_compute.py:183-184,219-221`; `build_release.py:34-43`.

The Python runner invokes the worker in-process and does not capture process stdout/stderr. The batch file supplies neither output/error paths nor an enforceable reservation for them. It says the controller will supply NEW exclusive paths, but these four files include no such controller procedure or concrete reserved namespace. Slurm defaults are not proof of separate exclusive retention and must not be relied upon for no-overwrite behavior. The shell's output-directory existence check does not protect Slurm log files, which Slurm opens before the script runs.

Independent noisy-failure fixture emitted `REVIEW_STDOUT` and `REVIEW_STDERR` and then raised. The controller result retained the exception and postchecks, but neither stream token occurred in the output tree (`stream_probe.json`: both false). Both tokens were retained by this review's external stdout/stderr redirections, demonstrating that the missing guarantee is the external log contract, not unavoidable stream loss.

Minimal fix: supply a reviewed controller-side preparation/submission recipe or helper that exclusively creates one fresh private namespace, reserves distinct absolute stdout/stderr paths within it, records those paths with the external ledger SHA and fresh worker-output path, and requires explicit Slurm `--output`/`--error` arguments. It must refuse any existing namespace and retain logs on submission/startup/worker failure. This can remain preparation only; no remote action is required to write or test it. Do not attempt to fix Slurm log collisions solely inside the batch script. Alternatively, add exclusive launcher-owned stream capture, while still specifying how batch startup errors are retained.

The same contract must state that a released namespace is never modified or reused, and validate launcher/batch hashes from a trusted controller before executing package code. The builder provides a fresh content-pinned snapshot but does not make its directory filesystem-immutable. A mutable launcher's own hash check is not an independent trust bootstrap. These are deployment prerequisites, not a demand to redesign the accepted binder or provide a hostile-administrator security boundary.

## What passed

- Five-module closure is exact: binder, HSP contract, HSP expanded, phase accept and phase project, plus an empty package initializer. The four tracked dependencies match bytes at the dirty worktree's base commit. Binder, binder-test and manifest hashes match the accepted source-fix review.
- Builder requires exactly five caller-supplied hashes, validates each byte stream and the manifest before creating output, copies the approved manifest unchanged, uses exclusive output creation, and records a controller-pinnable ledger covering the launcher and batch file as well as code. No original application package initializer is executed.
- An actual-source package was built locally with explicit hashes using `-I -B`. Its manifest was only copied and hashed, never executed. Package verification and exact six-entry package/module import inventory succeeded under isolated local Python, with no real worker invocation.
- Runtime invocation is the intended manifest/hash/output CLI, not a test hook. The batch script uses the fixed Python path with `-I -B`, clean `env -i`, `--export=NIL`, and a non-exec supervising shell. Python path/version/executable SHA and samtools path/SHA/version are checked before and after work. Shared libraries/stdlib are explicitly not hermetically pinned. Actual cluster runtime checks were not run here.
- File and directory inventories, empty initializer, symlink/hardlink refusal, loaded module names/origins/objects, before/after package hashes, report/marker/diagnostics binding and negative biological gates fail closed in the supplied suite. Existing release/output paths are rejected. Unexpected preloaded evidence modules and working-directory shadow packages are covered.
- Resource intent is bounded: Slurm 1 node, 1 task, 2 CPUs, 8 GiB, 30 minutes, no requeue, TERM 60 seconds before expiry. The unchanged manifest selects 16 names from a 10,000-line prefix. Worker defaults are 40 GiB charged reads, 256 MiB output, 1,000,000 expanded bases, 1 MiB line/sequence, 32 MiB retained state, 10,000 records and 1,740 seconds. Manifest overrides can only reduce those limits. These are the accepted worker's accounting limits, not a claim of total physical disk I/O accounting for samtools or all wrapper activity.
- Package verification/building only read small local artifacts. Large manifest sources are opened only by the worker after the compute entry gate. `--verify-only` does not import the worker or access runtime/input paths. An absent Slurm job ID rejects compute entry locally before output creation. Job-ID presence is an accidental-login guard, not independent scheduler authentication; the controller must actually use the reviewed Slurm path.
- Ordinary Python exceptions, worker nonzero exit, corrupted markers/diagnostics, changed package bytes and runtime-postcheck failure preserve worker artifacts and prevent controller success publication in the supplied tests. Diagnostics readback has a 256 MiB cap outside the worker budget; wrapper metadata and executable checks are additional overhead. There is no biological approval: COMPLETE remains diagnostic traversal only.

## Executed commands and results

All individual subprocess argv, cwd, timeout, exit codes and output-file names are recorded verbatim in `est_deploy_review_evidence/commands.json`. `review_checks.py` and `signal_probe.py` are retained runnable review-only harnesses. Paths below abbreviate:

```sh
E=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_recovery_20260917_1400/est_deploy_review_evidence
D=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/est_deployment
R=/tmp/est-real-source-binding-20260917
P=/home/wyim/miniconda3/bin/python3
```

The executed outer command was:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$E" timeout 90s python3 -B "$E/review_checks.py" > "$E/review_checks.stdout" 2> "$E/review_checks.stderr"
```

Exit 0. Local Python was 3.8.20, not a claim of testing the remote pinned 3.11.14 interpreter. The harness imposed 40-second subprocess timeouts and disabled bytecode writes. Its key commands/results:

```sh
"$P" -B "$D/test_deployment.py" -v
# exit 0, 16 tests passed in 32.205 seconds
/bin/bash -n "$D/sample.sbatch"
# exit 0
"$P" -I -B "$D/build_release.py" --root "$R" --out "$E/actual-source-release" --source-hashes "$E/source-pins.json" --manifest "$R/tasknotes/real_sample/sitalica.manifest.json" --manifest-sha 7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c
# exit 0; ledger SHA 7f8b0c3028027db07ad3c1f0bedbaf912c21cf9338a921c13d7f7b74abefd01c
"$P" -I -B "$E/actual-source-release/run_compute.py" --verify-only "$E/actual-source-release" 7f8b0c3028027db07ad3c1f0bedbaf912c21cf9338a921c13d7f7b74abefd01c
# exit 0; package inventory/SHA verified, no compute/biological acceptance
"$P" -I -B "$E/actual-source-release/run_compute.py" "$E/actual-source-release" 7f8b0c3028027db07ad3c1f0bedbaf912c21cf9338a921c13d7f7b74abefd01c "$E/must-not-create"
# exit 1 as expected: fresh output and Slurm allocation required; output absent
PYTHONDONTWRITEBYTECODE=1 timeout 35s python3 -B "$E/signal_probe.py"
# harness exit 0; both TERM children return -15 with no controller result
```

Actual import-only command and four `git show BASE:path` comparisons are retained verbatim in `commands.json`; all exit 0. `actual_imports.stdout` contains six exact origins within the review package. `before.json` and `after.json` have identical hashes for all eleven reviewed production/source files. No runtime executable or large input was accessed. The failure probe's own stream outputs are in `review_checks.stdout` and `review_checks.stderr`. The signal probes intentionally leave partial synthetic outputs as recovery evidence.

Advisor consultation was attempted but unavailable due client/model-version incompatibility; no advisor endorsement is claimed. Graft context calls reported approximately 860,650 tokens saved.

## Reviewed SHA256 inventory

Preparation prefix: `/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/est_deployment/`.

```text
a75c460c6e0e078565a9177d0ac712ae9f78315f012abfe4fd299cd9c0aa0cf5  build_release.py
c254230d103887c8317325408b1ac0a8b8c0572e389f33f4d5afb5d01d02b2a3  run_compute.py
a9ee5613952f45994298b6bfc434a6b818b437764bab5ff8ee730879d0456308  sample.sbatch
f13f4eff98a283fb991860ad7f92933f15422ddedc4e0dda7527b18deaf2ece4  test_deployment.py
```

Worktree prefix: `/tmp/est-real-source-binding-20260917/`.

```text
93ea442d0dc60c1d8df2ee0b479361d89a188f9b43bba4327740f896efec27b2  revision/scripts/evidence/est_real_sample.py
fcb8a4b2413f12dc5c5b0c4d55676cedeac21b3fcb26cf849c089657650a3954  revision/scripts/evidence/est_hsp_contract.py
aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4  revision/scripts/evidence/est_hsp_expanded.py
0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3  revision/scripts/evidence/est_phase_accept.py
6b977fabf3b3401e162e8d8b800d215d34dbe978f54b364429cb2056a60fec4b  revision/scripts/evidence/est_phase_project.py
7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c  tasknotes/real_sample/sitalica.manifest.json
5920616a012ae1cdb2dd941dbafc5752b9643eaef564230597ab0ad9488044dc  tests/test_est_real_sample.py
```

Accepted source review SHA256: `0f0b6e76f9b5ffc1bfd49087880212a452076d63bd3bd0252455b4a83068993c` for `notes/seed123_execution_20260917/EST_SAMPLE_FIX1_REVIEW.md`.

Final read-only verification rehashed all eleven production/source files with Python: unchanged. The report passed the forbidden-character check. `git status --short` still listed only the original `.gitignore`, `.ignore`, binder, tasknote directory and binder test changes; `git diff --check` exited 0.

## Recovery handoff

Completed: bounded local deployment review, accepted source-pin comparison, actual package construction/import-only check, sixteen existing synthetic tests, stream-retention and two TERM lifecycle probes. Blocked: deployment approval pending D1 and D2. Next independent work is minimal preparation fixes and bounded regression re-review, not real-source execution or rebuilding the accepted binder. Do not reuse this review's package/outputs as a submitted run. Actual deployment/runtime validation and independent result acceptance remain separate authorization gates.
