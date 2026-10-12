# Scoped R1/R2 re-review

**Code verdict: APPROVE scoped fixes. R1 and R2 closed; no new blocking defect found in this delta.**

**Execution verdict: NOT APPROVED / pending real manifest review.** This is not a real-tree C1 PASS or submission authorization. Baseline6165461 was untouched; its live state was not checked. Extern and scientific/production gates remain unresolved.

Reviewed HEAD `f6bef565f0bb432256b7e596a81b896afd2266c5` against `21456324923c2c84610bd63e8b6ae868eae2f535` in `/tmp/b6-p2-phase-1535433`. Read the complete main `c1_reconcile.fix1.diff` and worktree REPORT including the appended Round1 section. Graft queried first. Scope was the original two findings and breakage introduced by their fixes, not a fresh whole-auditor certification.

## Findings disposition

- **R1 — CLOSED.** `revision/scripts/evidence/est_blastx_audit.py:622–625` now catches the specific external-parser `csv.Error` and `RecursionError` alongside the existing evidence exceptions, records reconciliation FAIL, and returns no waivers. No blanket Exception catch was added. The added programmer-TypeError test checks that unrelated errors still propagate.
- **R2 — CLOSED.** `est_blastx_audit.py:664–667, 676–678, 765–777, 807–814` binds both exact `run.sbatch` pins and all18 required pilot/array logs before the first unit audit. Final checks cover all20 files, reject missing/nonregular paths and symlink ancestors, compare initial size before hashing, and compare SHA256 for same-size changes. Failed bindings/final checks feed rollup FAIL without dropping unit reports. No end-time queue-age gate or large-payload rehash was introduced.
- **New delta findings:** none. Existing non-atomic filesystem/operator-trust limitations are unchanged; these checks do not claim continuous writer exclusion.

## Independent focused verification

Re-executed the original three reproductions using existing fixture helpers, temporary local synthetic trees, and Python3.11 with bytecode disabled—not the191-test suite:

```text
csv_field FAIL as required; 18 healthy unit reports retained; specific check FAIL; FAILED only
deep_json FAIL as required; 18 healthy unit reports retained; specific check FAIL; FAILED only
late_wrapper FAIL as required; 18 healthy unit reports retained; specific check FAIL; FAILED only
```

Each asserted rollup FAIL, `units_audited=18`, no failed scientific units,18 emitted unit JSONs, the relevant reconciliation/final-wrapper check FAIL, FAILED.json present and DONE absent. The late-wrapper case previously returned PASS. `git diff --check` over the exact reviewed commit range exited0.

Read the new regressions covering both wrappers and pilot/array logs under content mutation/deletion/same-byte symlink replacement, plus the positive two-hour clock advance/no-final-payload-rehash guard. Controller's191tests PASS12.78s is supplied evidence, not independently rerun here.

No source edits, SSH, scheduler operations or subagents. Only this requested review report was written persistently by the reviewer; Graft reported its automatic index refresh. Existing `.gitignore`, `.ignore`, and `.testdeps/` worktree state was not changed by this review.

Real manifest/capture approval and a separately reviewed immutable invocation remain prerequisites. Neither code approval nor operator-trusted capture hashes resolve extern OUT_OF_MEMORY or authorize production.

Graft estimated109,143 tokens saved this turn.
