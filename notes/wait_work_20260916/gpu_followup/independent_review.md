# Independent Task 1 review

## Verdict

**NO EXECUTION.** Latest controller status supplied for this review is final P1 **3163986 PENDING**. No SSH/status refresh or job action was performed. Missing actual completed P1/S0 artifacts are legitimate execution gates, not a demand to fabricate future evidence.

The local sequence, wrapper/measurement API, native prerequisite validation and namespace-retention design match canonical b660bf1. One small, reproduced local validation defect prevents unconditional code sign-off (F1). Separately, actual container mounts/argv/release remain operationally unreviewed and must be fixed before submission (G1). Neither CPU tests nor this review authorize GPU work.

Paths below are relative to `notes/wait_work_20260916/gpu_followup/`, except the explicitly named final deployer.

## Findings and execution gates

### F1 — Medium: nonpositive runner caps pass the local builder

- **Location:** `followup.py:90-91`; contract `PLAN.md:37`; canonical consumer `source/revision/scripts/mvp/run_bounded.py:73-76`.
- `0 < timeout_seconds + 120 <= outer_seconds` accepts zero and negative integers down to -119. The actual consumer explicitly rejects every timeout <= 0. Thus a locally accepted plan can consume allocation/native-validation time only to fail before starting its probe.
- **Focused reproduction:** using the existing synthetic build fixture, change only `timeout_seconds` to `-1`; `build(execute=False)` succeeds and writes a runner ending `--timeout-seconds -1`. No native probe or GPU import was run.
- **Minimal fix:** require `type(timeout_seconds) is int and 0 < timeout_seconds <= outer_seconds - 120`; add a focused rejection test for zero and negative caps. This is a builder/consumer mismatch, not an unbounded-execution or false-PASS path: canonical `run_command` fails closed.

### G1 — High operational gate, acknowledged rather than a hidden code defect: concrete namespace/release absent

- **Location:** `PLAN.md:45-49`, `followup.sbatch:16-33`, `followup.py:143-150,168-173`.
- The template intentionally does not implement the actual container mount argv. Review that argv against `notes/mvp_execution_20260916/final_p1/deploy.py:202-251,254-292,381-403`, using the real completed job directory, not a guessed or old cancelled-job source.
- **Minimal closure:** controller supplies immutable reviewed release/argv and live exclusions; pinned SIF, `/work/repo`, cwd `/work/repo`, both HF asset roots, corpus, `/mvp/inputs`, `/mvp/probe`, `/image.sif`, Python path/offline environment and explicit `SLURM_JOB_ID`, `SLURM_JOB_NUM_NODES=1`, `ALLOCATION_RESERVED_SECONDS=1800`; writable distinct persistent output mount. Do not substitute the API excerpt for deployed source.
- Keep original P1 `/scratch` accessible read-only and every earlier recorded import path unchanged. New mode modules/inductor/triton use `<output>/cache`, not remapped `/scratch`. Retaining files elsewhere is insufficient. For decoding, additionally set `HF_MODULES_CACHE` to the **generation probe's exact original module-cache path**, before HF imports; merely mounting it alongside a fresh active module cache fails the decoder scaffold/loaded-module checks (`mvp_decode_runtime.py:144-225`). Review inherited HOME/XDG paths too; the builder changes module/compiler/TMP paths, not HOME/XDG.

### G2 — High operational gate: completion/accounting and resource authorization remain unavailable

- **Location:** `followup.py:60-79,109-119,127-153`; `PLAN.md:53-61`.
- **Minimal closure:** actual final-P1 wrapper/DONE/S0 plus current complete terminal-job accounting, serialized reservation/no-overlap evidence, and clock authorization covering the allocation. Final 3163986 must be COMPLETED; old 3163545 is cancellation/accounting history only, never substitute GPU evidence or source.
- Resource needs a separately selected real controller training budget, margins and inputs. Do not derive training authorization from the five-node-hour diagnostic cap or manufacture a controller reference. Missing authorization is correctly blocking.

## API and safety checks that hold

- **P1 identity:** `followup.py:40-57,94-119` checks wrapper stage, DONE digest, FAILED absence, zero exit/no timeout/signal, CUDA identity, binding, final allocation, P1 argv digest, all five S0 pins and executable digest. Final deployer's `container_p1` produces exactly that runner/argv shape. The seven tooling blobs were independently compared with `git show b660bf1:<path>`: canonical bytes = archived bytes = `source-files.json` entries. The map is a canonical source lookup, not permission to deploy the excerpt. Native setup validates the real consumed manifest and input/source hashes.
- **Sequential measured prerequisites:** local `wrapper()` alone is intentionally weaker than GPU certification. Execution calls canonical `_load_receipt` for every previous mode (`followup.py:147-148`), which calls `validate_measurement` and rehashes actual imported modules/assets. Step0 requires deterministic three-tier bitwise digests; compile/smoke commands use four-rank bf16 Accelerate with zero restarts; canonical `compiled_update` checks hardware, finite update and generated kernels. Generation uses the actual 30720 batch and exercises cache/evidence/reorder behavior, not a three-tier generation claim. Resource validates real rank/tier timing, memory, label coverage, disjoint samples and projections (`resource_probe.py:26-157`). No boolean-only fixture certifies these execution paths.
- **Clock and node-hours:** `budget()` rejects future observation times, expired/insufficient deadlines, nonfinite charges, duplicate/missing P1 rows and active/pending jobs; it charges prior node-seconds plus the full future allocation against 18000. A second check follows slow native validation. Complete history and no-overlap remain external controller attestations, not authenticated scheduler queries or locks; PLAN explicitly requires serialization and reservation/reconciliation. Current reservation must not be counted twice.
- **Outer timeout:** 1700 seconds plus 20-second kill grace wraps container startup and native receipt/import/asset validation as well as runner execution. Host STARTED/hash checks precede that timer; they are bounded/accounted by the full 1800-second Slurm allocation, not by the 1700-second timer. There is no guaranteed full inner runtime: at the maximum positive inner cap 1680, more than 20 seconds of container/gating overhead can cause outer termination. PLAN already requires choosing a smaller reviewed cap or stopping; the code does not evade the allocation limit. Do not claim guaranteed completion/headroom from 1700+20 arithmetic alone.
- **Six receipts / seed binding:** readiness `generate/check` uses all six receipts at the same binding and directly exercises the consumer (`make_launch_readiness.py:235-353`). Seed participates in that binding. Seed-2/3 PASS cannot be copied from seed 1; this tool deliberately anchors final seed-1 P1 and requires a separately reviewed extension. Training is not implemented/authorized here; canonical `run_training.py:36-117` separately checks authorization and measured projected runtime.

## Verification scope

Accepted the controller-reported **18 tests / 0.31s, bash syntax and six deliverable hashes PASS** without rerunning that suite. Performed only the focused negative-timeout reproduction and seven small canonical-tooling comparisons above. Read the requested files and graph-directed canonical API/final-deployer spans. No GPU measurement, actual future receipt validation, SSH, jobs, source edits, subagents, merge or push. Only this review document was intentionally written; graft automatically refreshed its local graph metadata. Advisor consultation failed with a client/model-version error; no advisor review is claimed.

Graft reported approximately 1,174,853 tokens saved across this review's graph calls.
