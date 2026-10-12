# Independent combined GPU integration review — 2026-09-16

## Verdict

Reviewed detached **772c38c3a428acedebb0ba3b1bef37f2777a0f35** in isolated native worktree `/tmp/pi-agent-bdbf9b31-8796-478-ffd8c11b`.

**Combined caller integration: ACCEPT for controller staging of bounded diagnostics. No new blocking integration defect found.** This is conditional diagnostic clearance, not permission to launch a job, a resource PASS, six-gate readiness, or training authorization. One inherited nonblocking schema weakness is recorded below; an unqualified claim that every identity/bindings schema is strict is not supported.

Read the committed integration report, neutral FIX1 reviews and combined diff under `/tmp/mvp-gpu-integration-context-20260916/`; checked `544bcb0..772c38c` first, then `d99b1f8..772c38c`, actual source definitions and exhaustive indexed helper uses. Graft orientation/query/skeleton/caller checks preceded source inspection. This is an independent combined-callsite review, not an inference from cherry-pick ancestry.

## Combined paths checked

- **Six modes / owner:** `revision/scripts/mvp/real_probes.py:601-657` sets offline flags before early resource dispatch. Resource does not create the normal-main Accelerator. The two compiled modes create one main-owned Accelerator and pass it to `compiled_update`; tokenizer, step0 and generation remain their original paths. `compiled_update:555-598` does not create a second owner. Resource creates its own owner at `resource_probe.py:270`.
- **Dataset signature and v3:** both actual probe callers use four arguments (`real_probes.py:633`, `resource_probe.py:344`). The shared factory explicitly chooses bound decoder version and validates/cleans on failure (`real_probes.py:236-249`). Resource retains the original model config before wrapping, validates train before prepare (`resource_probe.py:343-353`), and separately constructs and validates valid using that same config (`362-370`). No reliance on a DDP wrapper's `.config`. The common validator checks actual tokenizer vocabulary, model/config vocabulary and decoder/label caps (`src/transgenic/training/mvp_cli_runtime.py:491-499`). Production train and valid still explicitly select and validate v3 (`606-620`).
- **Collectives:** resource's selected 24 distinct local rows feed the actual `collective_batches` helper (`resource_probe.py:403-410`; `real_probes.py:533-542`). `one_update` remains coordinated and is called directly, not through `local`; its batch-contract agreement precedes forward and its finite/skip/update-count checks remain (`481-530`). Shared compiler check is direct at resource line431, before validation, and compiled-mode line592, before gather. `accelerator.prepare` is outside local-only preflight. The validation stage explicitly uses `collective=True` (`resource_probe.py:302-307,435`). No whole collective-containing function was placed in a local-only exception wrapper.
- **Paired inputs:** retrieval/device transfer and `validate_paired_batch` are coordinated before either DDP forward (`resource_probe.py:191-206`). The forward-containing metric remains outside preflight. The extracted metric validator and unchanged off/on metric body are visible in the combined diff (`src/transgenic/training/mvp_metrics.py:64-114`); production finalizer orchestration is not changed or newly certified by this integration.
- **Cleanup:** profile/scratch resolution now occurs inside owned try/finally through local preflight (`resource_probe.py:328-337`), closing the newly discovered ordinary path-rejection leak. Main uses nested finally for dataset/Accelerator cleanup (`real_probes.py:689-696`). Resource closes retained train/valid handles then ends training (`resource_probe.py:495-499`). As disclosed by the implementer, a dataset-close exception can still prevent subsequent resource cleanup; this inherited limitation is not universal distributed recovery. Backend failure/process death still needs bounded launcher teardown.
- **Compiler/performance:** resource retains fail-closed compiler-limit configuration, steady-update recompile rejection, full-tier selection, and directly shared training primitives. Full inventory scans, import/cache hashing and synchronization are intentional measured costs, not new fatal hot-path defects. No performance guarantee or broad refactor recommendation follows from CPU/source review.

## Resource accounting, publication and identity

`resource_probe.py:26-49,52-142` retains exact integer checks for budget schema/envelope/seeds and resource rank/count/RN/length/kernel fields, finite positive measurements, four-rank disjoint samples, all three tiers, one cold plus two steady updates, complete valid identity, and recomputed projection/headroom checks. Prior smoke seconds plus proposed probe cap must be **<=18,000 seconds**, not an enlarged allowance. Completion and progress reject elapsed exhaustion. The child timeout is required positive and within the bound (`253-258`). This validates a prospective record's arithmetic and digest, not the authenticity of a controller or actual allocation-wide authorization.

Immutable `rankN.SEQUENCE.profiling.json` snapshots retain previous bytes (`287-300`). Partial snapshots are explicitly not readiness. Publication calls only low-level payload/manifest writers, uses a fresh `PROBE_ONLY_CHECKPOINT`, and records **9 optimizer steps / 216 microbatches per rank / 4 ranks** (`438-451`; low-level serializer at `src/transgenic/training/mvp_finalize.py:517-537`). There is no fabricated 250-update completion or training DONE. The 250 value is projection math only.

The combined resource module is included in tooling identity (`make_launch_readiness.py:102-106`). Setup checks bound tooling/config/input/manifest/source/cache identities, actual pinned versions and parser semantics (`real_probes.py:36-63,137-206,209-233`). Resource reaudits candidate/SIF/source/sidecars/budget and imported modules/cache before measurement publication (`resource_probe.py:317-326,456-492`). Parent validation routes through real `validate_resource` rather than synthetic CUDA flags (`run_bounded.py:45-70,184-188`). Measurement digest must match the runner's prepared binding digest.

### Nonblocking inherited finding I1 — identity schema is not strictly typed

**Location:** `revision/scripts/mvp/real_probes.py:84` (also reached by resource validation at `resource_probe.py:151`). `schema_version != 1` accepts JSON `true` and `1.0`. Independently reproduced against real local executable/config-tree hashes without mocks, CUDA, measurement files or receipts:

```python
import sys
from pathlib import Path
sys.path.insert(0, 'revision/scripts/mvp')
import real_probes as p
path = str(Path(p.__file__).resolve())
root = str(Path('configs').resolve())
base = dict(probe_path=path, imports={path: p._sha256(Path(path))},
            asset_roots={root: p.tree_identity(root)})
for version in (1, True, 1.0):
    p.validate_identity({'probe_identity': dict(base, schema_version=version)})
    print('accepted', repr(version), type(version).__name__)
```

Observed: all three accepted. This does not bypass byte/hash checks or cause the genuine producer (which writes integer1) to fail, so it is not a blocker for bounded diagnostic staging and is not a new cherry-pick regression. For strict identity-schema acceptance, require exact integer type and add bool/float negatives. Prepared binding checks likewise rely on digest identity and Python equality, not a comprehensive strict schema validator (`real_probes.py:150-163`, `run_bounded.py:145-151`); e.g. the explicit seed-membership predicate itself is not type-strict. No claim of an end-to-end malformed-seed acceptance or GPU receipt forgery is made. Resource FIX1's strict numeric-field tests do not establish strictness of these separate inherited boundaries.

## Evidence and limits

- Independently verified both narrow and combined committed-range `git diff --check` commands passed; worktree was clean before writing this report.
- Independently ran only the focused identity-schema reproduction above. No reason to repeat the full suite or the controller's already-run integration tests was found.
- **Attributed, not rerun:** implementation reports 370 CPU tests passed in941.96s; controller reports 11 integration tests passed in18.72s and clean diff/no environment-file changes. Inspected the integration tests: real dataset factories/validators and negative owner/cleanup boundaries, not successful CUDA simulations.
- Advisor planning consultation was attempted but failed with `claude_code_version_too_old`; no advisor assessment is claimed.
- No source changes, SSH, jobs, subagents, GPU execution, merge or push. This report is the only committed change.

## Controller conditions

Controller may stage bounded P1/P2 diagnostics after regenerating consumed-code/tooling manifests, prepared bindings and receipts for this exact combined source, including `mvp_metrics.py` and `resource_probe.py`. Use immutable accepted inputs/checkpoint/SIF, both pinned offline cache layouts, independent writable scratch and reviewed bound argv. Do not reuse old v2 receipts or infer readiness from partial profiling.

An **outer allocation deadline/accounting plan** must include all prior smoke/attempts, startup, parent verification and teardown within the remaining 18,000-second one-node cap; the child timeout/projection is not that plan. Resource execution additionally requires the real prospective controller budget record and bound digest. No untimed warmups or automatic extra attempts are cleared here.

Actual SIF Torch2.14/CUDA13.4, Transformers4.57.6 generation/cache, Accelerate1.15.0, real Hyena gradients, bf16, Inductor and NCCL remain untested by this review. Controller-reported already-queued P1 diagnostic job3163545 targets d99, not this changed source; its outcome cannot certify this integration. No status query or submission was performed. **Training and six-gate readiness remain HOLD.**
