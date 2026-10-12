# Track A independent read-only review

Reviewed final `709ff26ff7f69be42f028a45b6c28ccbb5d68b60` (code `4c2214becd0113a33c03f40332b3859ce0908afc`) against base `f847cb80c7750326f0781f81481891ab0c9ea309` in `/tmp/b6-p2-readiness-1535112`.

## Verdicts

- **Spec: substantially implemented preparatory infrastructure; incomplete operational Track A.** The eight accepted must-fixes are addressed by code or explicit deployment instructions, with the initialization-binding gap below. Real P1–P4 programs, staging execution, GPU evidence and launch authorization are explicitly absent, not silently implemented by the tests.
- **Correctness: REQUEST CHANGES before deployment/readiness use.** Two medium-severity gaps remain: loader-consumed optional sidecars are outside readiness identity, and the supposedly fixed launch still consumes ambient Accelerate configuration. Neither establishes that invalid scientific output can pass the unchanged trainer's final gates. Both can make an accepted readiness check differ from the attempted run.
- **GPU launch readiness: BLOCKED.** No CPU result in this review or the implementation report certifies GPU/SIF readiness.

## Findings

### A1 — Medium: readiness does not bind optional loader sidecars or their absence

Locations: `revision/scripts/mvp/make_launch_readiness.py:173-208,305-313`; `revision/scripts/mvp/stage_checkpoint.py:134-137`. Consumer detail: `src/transgenic/training/weights_init.py:174-181,210-218` and `mvp_cli_runtime.py:441-454`.

`_source_checkpoint_binding` covers weights, parent run_config and runtime_identity, but not checkpoint `config.json`, `meta.json`, or `runtime.json`. The loader opens/snapshots all three when present; `config.json` additionally participates in semantic validation. Staging copies `meta.json`, yet prepare/generate/check do not bind even that file. The staging receipt is not subsequently consumed by readiness.

Consequently ordinary sidecar addition/removal/editing after preflight leaves the binding hash and six receipt bindings valid. A config addition can make the eventual loader reject a checkpoint after the wrapper has accepted readiness and started its GPU attempt. A metadata change also changes initialization provenance without invalidating preflight identity. This is not an allegation of malicious receipt fabrication.

**Focused CPU reproduction, observed:** use `tests/test_mvp_readiness_tools.py:launch_fixture` in a temporary directory, then write `checkpoint/config.json` containing `not JSON`, and invoke `make_launch_readiness.main(['check', *args, '--readiness', str(ready), '--readiness-sha256', sha256(ready)])`. Output was `status=readiness_valid`, return **0**. The loader explicitly JSON-parses that file. All fixture evidence was synthetic and temporary; this was a validator counterexample, not GPU evidence.

**Minimal fix:** bind the loader's complete optional-sidecar inventory, including explicit absence, and compare it during check. Alternatively, require an exact staged inventory and reject unbound loader sidecars. Add add/remove/change tests. Preserve the existing schema-1 consumer contract; stronger tooling bindings can remain additional fields.

### A2 — Medium: the fixed training command leaves Accelerate defaults ambient

Location: `revision/scripts/mvp/run_training.py:89-92`.

The wrapper fixes process count, machine count and bf16, and preserves the entire parsed trainer argv. However, it supplies neither a reviewed `--config_file` nor equivalent isolation from the user's default Accelerate configuration. A normal cached configuration can still supply GPU selection, Dynamo configuration and rendezvous settings. Those bytes are not part of the prepared/readiness/tool identity. Running in a pinned SIF does not itself exclude a home-mounted default config.

The inspected CPU dependency implementation is `.testdeps/accelerate/commands/launch.py:985-986,1012-1016,1026-1052`: it reads default configuration and copies `gpu_ids` and nested Dynamo settings despite explicit `--multi_gpu --num_machines 1 --num_processes 4 --mixed_precision bf16`.

**Focused parser-only reproduction, observed:** construct a `ClusterConfig` with `gpu_ids='0,1'` and `dynamo_config={'dynamo_backend':'EAGER'}`; supply it to `_validate_launch_command` through a substituted default-config loader, with a temporary default-config path and bf16 availability stubbed true. Parsing the wrapper's exact launch prefix yielded:

```text
{'num_processes': 4, 'mixed_precision': 'bf16',
 'gpu_ids': '0,1', 'dynamo_backend': 'EAGER', 'max_restarts': 0}
```

No training subprocess or GPU call ran. This demonstrates unresolved ambient options, not an observed GPU failure. Initial direct-dataclass harness attempts hit unset nested dictionaries; the final harness initialized those dictionaries explicitly. The deployed SIF's actual pinned Accelerate must be verified separately.

**Minimal fix:** use and bind an explicit reviewed Accelerate config (or demonstrably isolate default-config lookup), pin the intended GPU/Dynamo/rendezvous settings, and test conflicting ambient defaults. Explicit `--max_restarts 0` is worthwhile policy documentation, but **automatic retry was not reproduced**: the inspected parser's current default is already zero. Do not report an established restart bug.

## Contract and safety checks

Read the complete supplied `A.review.diff`, `TRACK_A_RECOVERY_REPORT.md`, `README.readiness.md`, discovery synthesis Track A, and exact `mvp_cli_runtime.py:175-236` contract. Queried graft before additional repository source inspection; inspected loader resolution and initialization auditing.

| Accepted must-fix | Assessment |
|---|---|
| 1. Exact readiness schema | Implemented: schema 1, exploratory-only authority, fixed recipe/seed, candidate/inputs/source/code identity and exact six evidence names; direct consumer validation before publication. |
| 2. Loader-compatible checkpoint | Correct staged parent-sidecar layout and three mandatory author pins; optional-sidecar binding gap A1. No implicit epoch selection. |
| 3. No validate-only shortcut | Implemented: explicit consumer check; training wrapper rejects validate-only. |
| 4. Full input staging | Full CLI inventory, companion pins, routes and rehash duties documented. Actual transfer/staging is deliberately not performed. |
| 5. SIGTERM/no public resume | Bounded process runner and failure records implemented; no fake save/resume/completion. |
| 6. Compute/data-mover hashing | CLI guard precedes large hashes/copies. Correctly documented as operational, not authentication. |
| 7. Quota/headroom | Fresh digest-bound quota record, filesystem free-space check, copy ceiling and reserve implemented. Whole-inventory capacity remains an operator gate. |
| 8. Deterministic P2 policy | Explicit deterministic/eval, `:4096:8`, all three tiers, finite outputs and exact hash equality; no inferred tolerance. Actual measurement programs absent. |

Other observations:

- Six receipts require measured pass, expected stage, matching binding digest, zero exit, no timeout/signal, and a digest-bearing DONE marker. FAILED prevents acceptance. Check rereads receipts and markers. This is appropriate within the documented trusted-operator evidence model, not proof a CUDA computation occurred.
- Source and copied checkpoint bytes are hashed; publication is fresh-path/no-replace with file/directory fsync. Failures preserve partial directories. No existing-file overwrite or data-loss path was established under the disjoint-root operator model.
- Trainer arguments are parsed with the existing parser and then forwarded intact after the fixed script operand; no dropped arguments or shell interpolation was found. Unsupported resume arguments are rejected. Readiness and authorization are checked before warmup/training.
- There is one wrapper training `run_command` call, after at most three startup attempts; only startup exit 75 is retryable. Actual probe execution is not retried. CPU tests do not prove the real CUDA warmup works.
- `run_command` enforces positive finite deadlines, caps written logs at 1 MiB and terminates/kills the process group, including after leader exit. Signal/timeout outcomes are forced nonzero. The timeout-exits-zero concern is therefore **not** a finding.
- The read loop stops on leader exit rather than explicitly draining stdout to EOF. Focused local trials (100 short echo children and 30 children writing 500,003 bytes) showed no truncation. Treat tail-draining as a follow-up hardening opportunity, not a reproduced blocker or evidence of leaked descendants.
- `PROCESS_EXIT.json` is clearly not artifact completion. Readiness hashing and warmup time sit outside the training subprocess timeout; the runbook explicitly requires allocation allowance. SIGKILL/node-loss failure publication is honestly disclaimed.

## Verification boundaries and remaining gates

Accepted the supplied evidence of 188 scoped tests passing in 475.22 s and controller's 35 new tests passing in 25.06 s; did **not** repeat either suite. The earlier unchanged two-rank probe timeout remains unexplained, not fixed by this tooling. Focused reproductions above used only temporary CPU fixtures. Verified final HEAD and changed-file list: trainer source is unchanged. Existing `.gitignore`, `.ignore`, `.testdeps` worktree state was not altered intentionally.

Before any launch:

1. Resolve A1/A2 and independently review the focused regression tests.
2. Implement and review actual real-tokenizer/Hyena P1–P4 programs; pin their executable/import identities and offline assets. argv hashing alone is explicitly insufficient.
3. Select the concrete source checkpoint and all three author pins; approve transfer route, full staging inventory, live quota and reserve.
4. Deploy an immutable snapshot containing these tools plus the reviewed trainer, inside the intended independently pinned SIF and allocation.
5. Obtain six genuine, input-bound GPU measurements with deterministic three-tier equality, cache parity, bf16/Inductor, four-rank topology and measured resource budget. No synthetic fixture flags may become real receipts.
6. Obtain per-seed budget and submission authorization; allow preprocessing/warmup/finalization overhead and account for total loss on interruption. No automatic resume or resubmission.

No source changes, SSH, Slurm jobs, subagents, merge or push were performed. Only this requested review note was written in main. Graft reported approximately 1,039,203 tokens saved across the context queries (tool estimate, not a verification metric).
