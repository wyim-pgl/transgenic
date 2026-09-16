# Track A recovery report — 2026-09-16 PDT

## Scope and commits

Worktree: `/tmp/b6-p2-readiness-1535112`
Branch: `b6-p2-readiness-trackA-20260916`
Base: `f847cb80c7750326f0781f81481891ab0c9ea309`
Implementation commit: **`4c2214becd0113a33c03f40332b3859ce0908afc`**.
This report is committed separately after that implementation commit.

Preserved and finished the existing untracked readiness/staging scripts rather
than discarding them. Added bounded preflight and training launch wrappers,
35 CPU tooling tests, and the staging/deployment runbook.

Files:
- `revision/scripts/mvp/make_launch_readiness.py`
- `revision/scripts/mvp/stage_checkpoint.py`
- `revision/scripts/mvp/run_bounded.py`
- `revision/scripts/mvp/run_training.py`
- `revision/scripts/mvp/README.readiness.md`
- `tests/test_mvp_readiness_tools.py`

No existing training source or semantics changed. No main edit, new worktree,
reset/clean, SSH, remote write, Slurm submission, subagent, merge or push.
No GPU receipts generated outside explicitly synthetic temporary CPU fixtures.
`.testdeps`, `.ignore`, graph cache and `.gitignore` changes are **not committed**.

## Implemented boundaries

- Exact existing schema-1 exploratory readiness consumer contract retained
  (`src/transgenic/training/mvp_cli_runtime.py:175-236`). `prepare` binds the full
  CLI input set, source weights/sidecars, fixed recipe/seed, exact consumed-code
  manifest and these tools. `generate` requires all six measured receipts,
  digest-bound DONE markers, identical input binding identity and successful
  measurements. It directly calls the existing validator before publication.
- `check` directly calls the consumer (not `--validate-only`) and additionally
  checks live receipt bytes, marker hashes, sidecars and tooling identity.
  Generated-record mutation tests call `validate_launch_readiness` directly.
- Explicit checkpoint pins; concrete nonsymlink paths; no current-best selection.
  Staging requires fresh quota evidence plus filesystem capacity, byte ceiling
  and reserve; hashes source and copied bytes, fsyncs files/directories, publishes
  a fresh run root atomically with loader-compatible parent sidecars.
- Compute/data-mover guard precedes CLI large hashes/copies. It is an operational
  guard, not an authentication/security boundary. Full staging inventory,
  transfer-route decisions, pinned canonical companion hashes and quota planning
  are documented in `README.readiness.md`.
- P2 fail-closed deterministic bit-identity measurement gate for all three tiers,
  explicit `:4096:8` workspace and deterministic/eval settings. No tolerance
  guessed or relaxed, no changes to actual trainer determinism settings.
- Preflight process timeouts, 1 MiB log caps, process-group termination, bounded
  optional warmup retries only on exit 75. No real probe retry. Failure cannot
  leave a usable DONE marker. Training wrapper directly checks readiness and
  author budget, warms CUDA (at most three startup attempts), launches exactly
  one fixed four-rank bf16 Accelerate attempt, rejects validate-only/resume.
  SIGTERM/timeout records failure; process exit zero is explicitly **not** an
  artifact completion certificate.

## Verification commands and observed results

All commands below ran from `/tmp/b6-p2-readiness-1535112` using the existing
isolated `.testdeps` and `/home/wyim/miniconda3/envs/pdb/bin/python`. No shared
Python environment was modified.

### Baseline

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_cli.py
```

**45 passed in 152.77 s**, before implementation changes.

### Test-first / focused iterations

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_readiness_tools.py
```

Initial run: **5 failed** exposing the absent receipt-binding API, optional
checkpoint pins, missing allocation guard and absent bounded runner/gate.
After the initial implementation: 5 passed; later fixture development briefly
failed because the synthetic inventory omitted required comparison metadata
(fixed the fixture, not the consumer). Subsequent focused runs: 16, 24 and
33 passed as coverage grew.

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_readiness_tools.py -k 'candidate_input or training_wrapper'
```

**2 failed** before adding accepted-candidate input matching and the training
wrapper. Those cases pass in the final regression.

### Scoped regression

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_cli.py tests/test_mvp_finalize.py tests/test_mvp_runtime.py tests/test_weights_init.py tests/test_mvp_metrics.py tests/test_mvp_model.py tests/test_mvp_dataset.py tests/test_mvp_readiness_tools.py
```

First broad run (24 new tooling tests at that point): **176 passed, 1 failed in
581.82 s**. The unchanged legacy two-rank `hardware-failure` probe exceeded its
120-second subprocess deadline. No trainer/test timeout was modified.

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q tests/test_mvp_cli.py::test_actual_two_rank_single_rank_hardware_probe_failure_rejects_before_model
```

**1 passed in 17.14 s**. Process inspection found one orphan from that timed-out
owned test (`python -u tests/mvp_cli_probe.py hardware-failure`, PID 10816); it
was terminated with `kill -TERM 10816`. Root cause of the intermittent deadline
was not established; do not call it a fixed runtime bug.

Final full rerun of the scoped regression command above, after all source/test
changes: **188 passed in 475.22 s** (153 existing + 35 new). This is not a
whole-repository green claim.

### Static / entrypoint checks

```
PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python -m py_compile revision/scripts/mvp/make_launch_readiness.py revision/scripts/mvp/stage_checkpoint.py revision/scripts/mvp/run_bounded.py revision/scripts/mvp/run_training.py tests/test_mvp_readiness_tools.py
for tool in make_launch_readiness stage_checkpoint run_bounded run_training; do
  PYTHONPATH=.testdeps:src /home/wyim/miniconda3/envs/pdb/bin/python revision/scripts/mvp/$tool.py --help >/dev/null || exit
done
git diff --check
git diff --cached --check
```

All exited 0. Graft was built/queried before source inspection and rebuilt at
the end (258 indexed Python files). Context included exact readiness validator,
source-path resolution, initialization sidecar expectations and existing tests.

## Remaining gates / limitations — not completion claims

1. **Actual real-Hyena/tokenizer/GPU P1–P4 probe implementations and executions
   remain required.** This change supplies bounded execution and strict measured
   result schemas, not the real Hyena probe programs. No actual GPU, bf16,
   Inductor, cache parity, production SIF or filesystem certification occurred.
2. Checkpoint selection and all three expected digests remain author inputs;
   no epoch was selected. Budget authorization, staging route, live quota,
   allocation and job submission authorization remain unresolved.
3. Probe argv SHA256 does not certify imported code, cache contents or the
   executing container. Independent review must pin those probe sources and
   offline assets, deploy the immutable snapshot in the intended SIF, and verify
   real identity before accepting receipts. JSON measurements are trusted
   operator evidence, not cryptographic proof that a GPU ran.
4. The unchanged base runtime readiness validator trusts receipt digests. The
   stronger tooling check is additional workflow protection, not a new mandatory
   schema or protection against a deliberately handcrafted trusted record.
5. SIGKILL/node loss may prevent FAILED publication; missing success is never
   completion. No graceful save/public resume exists. Partially staged copies
   remain for inspection; no automatic cleanup or retry is implemented.
6. Independent review is pending with the controller. No launch authorization
   follows from this CPU test report.
