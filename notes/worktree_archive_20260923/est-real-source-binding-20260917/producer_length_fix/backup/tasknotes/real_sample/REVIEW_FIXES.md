# Narrow review fixes — 2026-09-17

Implemented I1/I2/I3 in the existing uncommitted binder. No commit, remote command, job, package framework, existing-library edit, or actual large-data traversal was performed. Controller verification/deployment remains pending.

## Changes

- **I1:** require explicit `c1_provenance` and `c1_commands` references to absolute-path, SHA256-pinned file specs. Both freshly verified digests are now retained in the search preimage; HSP IDs inherit the search binding. Changing either contract file alone changes the affected search/HSP IDs, leaving the other source unchanged. The real manifest already has these fields and is unchanged.
- **I2:** on abort, attempt process-group SIGKILL regardless of leader liveness, tolerate ProcessLookupError, and reap with a bounded wait. Cleanup errors are recorded without suppressing report/marker publication. Actual reader return code is recorded separately. The descendant regression uses a ten-second fallback lifetime, a one-second binder deadline, a bounded liveness check, and unconditional own-PID cleanup.
- **I3:** use stdlib selectors to multiplex stdout/stderr on the main thread (no reader-thread Budget races). Stderr stores at most the first 64 KiB as exact base64 bytes, with observed total bytes, SHA256, truncation, capture size/limit and EOF truth. All consumed stderr bytes charge read usage; 64 KiB retention and 90 KiB serialized-output allowance are conservatively charged before launch. The final-status reserve remains separate. No undrained PIPE, stderr tempfile or unbounded retained log. On abort, final nonblocking drain is bounded; it never claims EOF unless observed.

`sam_stderr.total_bytes` and SHA256 describe bytes actually consumed. They describe the entire stream only when `eof=true`; abort/cap/cleanup failure can leave partial evidence. A cap can prevent launching the reader because the capture reservation itself must fit. OS/filesystem failure or SIGKILL can still prevent publication; existing missing-marker rejection remains required. A denied kill or uninterruptible process may produce cleanup errors/null return code rather than a claim that cleanup succeeded.

Scientific gates, partial raw artifacts, selected EOF requirements, caps, reference/dictionary requirements and existing projection policy remain unchanged. COMPLETE is still diagnostic traversal, never A11, full-C0, route-A, training, or molecule approval.

## Verification

Interpreter: `/tmp/mvp-final-source-env/bin/python3.12`, `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`; no dependency installs.

- `revision_red.txt`: **8 failed / 31 deselected before implementation**, covering independently changed provenance/commands, four invalid explicit-reference/spec cases, exited-leader descendant survival, and missing precise stderr. The stderr assertion decodes actual captured bytes; the marker is split in the recorded command and cannot satisfy the assertion by command echo.
- `revision_binder_green.txt`: initial eight regressions plus original binder tests passed.
- `revision_focused.txt`: **321 passed**, the exact requested ten-file 309-test baseline plus 12 new cases. Four additional hardening cases cover cleanup-error publication, stderr read/retention caps, and large binary stderr followed by valid SAM.
- `revision_review_boundaries.txt`: **7 passed / 3 deselected**, executing the reviewer's unchanged non-defect boundary tests. The three intentional observed-bug tests were read, not modified or represented as repaired tests.
- `git diff --check` and explicit no-index whitespace checks of the untracked binder/test files passed.
- Advisor consultation was attempted but failed due client/model-version incompatibility; no independent advisor endorsement is claimed.

Exact focused files:

```
tests/test_est_hsp_contract.py
tests/test_est_phase_conflicts.py
tests/test_est_approved_contract_integration.py
revision/scripts/tests/test_est_hsp_expanded.py
revision/scripts/tests/test_est_phase_project.py
revision/scripts/tests/test_est_molecule_closure.py
revision/scripts/tests/test_est_blastx_fixture.py
revision/scripts/tests/test_evidence_support.py
revision/scripts/tests/test_univec_trim.py
tests/test_est_real_sample.py
```

## SHA256 handoff

```
93ea442d0dc60c1d8df2ee0b479361d89a188f9b43bba4327740f896efec27b2  revision/scripts/evidence/est_real_sample.py
5920616a012ae1cdb2dd941dbafc5752b9643eaef564230597ab0ad9488044dc  tests/test_est_real_sample.py
7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c  tasknotes/real_sample/sitalica.manifest.json
```

Manifest hash did not change. Controller still owns immutable full-import-closure packaging, isolated runtime/before-after verification, independent acceptance, and the single bounded actual Sitalica run. Actual-source biological acceptance has not been established by these synthetic local regressions.
