# Independent scoped review — resource probe FIX1

## Verdict and scope

**FIX1 APPROVED: RP1–RP4 addressed; no new Critical/Important finding identified in the fix delta. Integration/execution remains HOLD.**

Reviewed detached `f479172fd64856902f8999a1d74e975f50622a13` against `4bd97399` in isolated native worktree `/tmp/pi-agent-4885ca28-37e8-426-f23eaa0a`. Consulted graft first, read the neutral original independent review and FIX1 diff under `/tmp/mvp-resource-context-20260916/`, read the committed implementation report, and inspected the actual committed source/test delta and relevant helper/caller definitions. This approves the narrow fixes for controller integration, not the unintegrated runtime or a resource receipt.

No source edits, SSH, jobs, CUDA execution, subagents, merge or push. Only this report is committed. Advisor consultation failed with `claude_code_version_too_old`; no advisor assessment was received.

## Findings disposition

- **RP1 — addressed.** `revision/scripts/mvp/resource_probe.py:287-302` initializes a per-rank sequence and publishes distinct immutable `rankN.SEQUENCE.profiling.json` files. Increment occurs after successful collective publication; previously written bytes remain intact. The local publisher converts its writer's `SystemExit` to `RuntimeError` before `_collective_preflight` exchanges status. A rank-specific collision therefore takes the ordinary coordinated failure path; failed runs need not produce equal snapshot counts across ranks. The shared fresh-path writer at `revision/scripts/mvp/make_launch_readiness.py:113-138` is unchanged. No receipt/DONE overwrite permission was introduced. Indexed filename consumers are limited to publisher/tests; the report documents numeric sequence selection for external readers.
- **RP2 — addressed.** `resource_probe.py:191-211` performs retrieval, transfer and `validate_paired_batch` inside preflight before any rank enters the paired forwards. The forward-containing call remains outside the local-only guard. The second validation inside `paired_metric_record` sees the same batch, with no intervening mutation in this path. This closes the identified ordinary input-contract failure ordering gap; it does not promise recovery from arbitrary model/backend failures inside DDP.
- **RP3 — addressed.** Exact integer checks cover schema/envelope, rank, update counts and RN values, with additional length/kernel checks. Existing seed/capacity/valid-count/memory checks remain strict. The 19-field bool/float/string regression matrix passes. Projection arithmetic is unchanged; the stricter acceptance set is intentional.
- **RP4 — addressed.** The extra EOF blank is removed and the prior report's inaccurate historical whitespace claim is corrected. Both `git diff --check 4bd97399..f479172` and `git diff --check 3b62306..f479172` pass independently.

## Metric semantics and consumers

`src/transgenic/training/mvp_metrics.py:64-114` extracts exactly the previous local validation statements into `validate_paired_batch`, returning the same RN/evidence scalars. Independent AST comparison against `4bd97399` confirmed:

1. Extracted validation statements are identical, excluding the new docstring/return.
2. The existing body from `was_training` through forward calls, weighted CE computation, result construction and mode restoration is identical.
3. Every other metric function is identical, including aggregation and duplicate handling.

`paired_metric_record(model, batch)` retains its signature and result contract. Graft traced consumers to resource paired validation, `evaluate_and_publish_final` (`mvp_finalize.py:628-696`), and metric tests. The finalizer is unchanged: this refactor neither changes its metric semantics nor claims to repair its inherited orchestration. No new shared-consumer regression was identified. Double validation in the resource path is small intentional local work, not duplicated collective work.

The RP2 negative test uses actual two-process Gloo status transport and DDP, injects float RN metadata on rank1 only, and verifies identical `ValueError` outcomes with zero forward-hook calls on both ranks. The CPU model is not a successful Hyena-forward surrogate; the test appropriately proves rejection before entry, not successful paired CUDA execution. Progress regressions execute the actual AST-extracted closure, exercise two successful snapshots and collision/deadline failure through single-rank preflight. Distributed progress failure is supported by source-level status-path tracing, not a newly claimed distributed progress test.

## Independent verification

Focused rerun only (the controller's broad run was not repeated):

```text
PYTHONPATH=/tmp/b6-p2-readiness-1535112/.testdeps:/tmp/pi-agent-4885ca28-37e8-426-f23eaa0a/src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q \
  tests/test_mvp_metrics.py tests/test_mvp_resource_fix1.py
74 passed in 13.41s
```

This includes all 60 FIX1 cases and metric regressions. Dependencies were read from the existing shared directory; none were installed. AST equality checks and both committed-range whitespace checks passed. Controller-provided **186 passed in 83.68s** and implementation-report **186 passed in 48.25s** are separate attributed runs, not this review's measurements. No result is pinned-SIF/CUDA/NCCL certification, proof of performance, or a resource PASS.

## Controller integration conditions (known, not new findings)

The separately accepted owner fixes `981489d` / `d99b1f8` remain absent here. Before execution, reconcile `datasets(args, bindings, cfg, model_config)`, validate both resource train/valid decoder datasets, reconcile the main-owned Accelerator and resource early-dispatch/cleanup ownership, and preserve coordinated `one_update`/batch/compiler contracts without wrapping collective work in local-only preflight. Review the combined callers and regenerate consumed-code/tooling bindings, including the changed metric module. This scoped approval permits proceeding to that integration review; it does not clear execution HOLD.

The original outer allocation deadline/accounting qualification also remains: child timeout and projected training costs alone do not enforce the five-node-hour allocation cap. Controller authorization, accounting for prior attempts/startup/parent validation/teardown, and bounded real CUDA diagnostics remain prerequisites. No training authorization or six-gate readiness is granted.
