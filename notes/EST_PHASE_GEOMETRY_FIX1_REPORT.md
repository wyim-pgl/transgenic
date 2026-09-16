# EST phase geometry fix 1 — 2026-09-16

## Scope

Branch `est-geometry-fix1-20260916`, started at `66b9c01` in isolated worktree `/tmp/pi-agent-efc0bb46-78ba-492-29d9c72f`. Read all three supplied neutral-context documents in `/tmp/est-geometry-context-20260916/`. Queried graft before source inspection (original boundary: `revision/scripts/evidence/est_phase_project.py:188–196`).

The sole P2 finding was real: `None` served both as an observation value and as iterator exhaustion. The fix uses a unique local object sentinel for both reads. A first `None` explicitly raises `ValueError`; any real second element, including `None`, returns the existing multi-HSP unresolved diagnostic with no segments. At most two entries are read, with no stream scan. Empty and single-valid inputs retain their behavior. No scientific policy, geometry, production eligibility or other API was changed.

## Regression and verification

Added six cases covering `[None]`, `[None, valid]`, `[valid, None, valid]`, empty input and a single-valid generator, plus bounded generators whose second entry is valid or `None` and whose third read raises.

Exact focused command:

```sh
python -m pytest revision/scripts/tests/test_est_phase_project.py -k observations -q
```

Before source changes: **4 failed, 2 passed, 51 deselected** (the four failures reproduce the sentinel defect). An earlier test-development run also caught an incorrect expected-value shape in the single-generator assertion; that assertion was corrected before this red run. After source changes: **6 passed, 51 deselected**.

Exact combined geometry and existing C-suite command:

```sh
python -m pytest revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py -q
```

Result: **176 passed in 6.93s** (prior 170 plus six new cases). `git diff --check` passed.

Only the source, test file and this report are committed. Graft generated an untracked local graph, excluded from the commit. No SSH, jobs, subagents, merge or push; no work performed in main or C worktrees. Independent controller re-review remains pending.
