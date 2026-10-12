# Final isolated MVP source integration — 2026-09-16

## Result and immutable source

Branch **`mvp-final-source-20260916`**, isolated worktree `/tmp/pi-agent-cf3f429c-1b79-48f-e598f927`.
Executable/source integration commit: **`362eb899d8d43164ec1e0fb0f3b174ead9b1ba40`**. The following report-only commit does not change executable source. Obtain that final branch HEAD with `git rev-parse mvp-final-source-20260916`; its exact hash is supplied in the task response.

**Source integration implemented; independent final review and genuine readiness remain HOLD.** No GPU, SSH, scheduler queries/submissions, subagents, main merge or push. User-reported P1 job **3163545**, queued against d99, was not touched; its result cannot certify this source. Recovery `.atl/resume.md` was absent in this isolated checkout. Graft was rebuilt/queried here, not in main. Advisor consultation failed because the configured client does not support its model; no advisor approval is claimed.

Pins computed after the executable changes:

- Operator `runtime_preflight.decoder_code_sha256`: **`b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb`**.
- Relative component-source SHA256: **`555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`** (86 tracked files). Definition: canonical JSON mapping `{relative_path: sha256(file_bytes)}` of `git ls-files src/transgenic train/train_HyenaTransgenic_MVP.py revision/scripts/mvp configs`, sorted keys, compact separators, plus one newline. This relocation-independent comparison pin is **not** the absolute-path consumed-code manifest, a prepared binding, or authorization.

## Exact component ancestry

Base: reviewed **`772c38c3a428acedebb0ba3b1bef37f2777a0f35`**. Its recent ancestry is `d99b1f8 -> 3e15b49 -> 6ed59e2 -> 544bcb0 -> 772c38c`; d99 includes v3/collective fix `981489d`. This preserves resource profiling, paired-input preflight, the shared `validate_paired_batch` extraction and final combined caller integration.

Ordered source cherry-picks (original -> integrated):

| Original | Integrated | Component |
|---|---|---|
| eefe5b5 | 47095c8 | scoring/bootstrap/windows/adapters |
| 2276fcb | 530bb2d | count invariants/provenance |
| 1a9253d | 1b42b99 | raw-GFF count producer |
| 64b6d5b | 62c130b | producer implementation/init/config checks |
| 0f9f777 | e22b660 | COMPLETE panel preparation |
| 668299a | d2fd5b0 | panel decoder/350 transport |
| e6950b7 | ec5445b | COMPLETE consumption/atomic publication/runtime gate |
| 6366d8a | 7d8ff09 | completed wrapper/project applicability/HF scaffolding |

Only conflict: `notes/TRACK_B_RECOVERY_REPORT.md` modify/delete in 2276fcb, because the report did not exist in A. Retained the task commit's report. **No code conflict or semantic conflict resolution.** No separate report-only cherry-picks, a4e8808, old f4f1690 duplicate, environment files, main dirty files or untracked documents were imported.

## Integration correction and trace

`revision/scripts/mvp/mvp_decode_runtime.py:19-41` now explicitly supports the reviewed772 handshake **only**, with these actual `git show 772c38c:<path>` SHA256 values:

| File | SHA256 |
|---|---|
| real_probes.py | a40b46b3e2df1f5a80a4357d06c0e77bf1cdedd66c0fd22582df7de0541bbc58 |
| run_bounded.py | 8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932 |
| make_launch_readiness.py | 73728aee67611575448722cb4180062917ee0c86330749ff1ca648346e75a6d5 |

No receipt-selected code authority or gate-disable option was introduced. All three current integrated files are byte-identical to772. Resource module hash is `4e7e598b2e0df21bef8d47636781b79ce9f8b8a846632039baaa316dd69fc4f6`; it remains in the real seven-file tooling manifest (`make_launch_readiness.py:102-106`). Generation need not claim that it imported resource code: resource is tooling-bound, and any actually declared import is also rehashed.

The deep dependency trace found one additional concrete blocker: `mvp_produce_counts.py:18` imports `mvp_bootstrap.validate_paired_record` before decoder runtime preflight. This newly added project file was neither imported by A's generation probe nor in the closed decoder audit. Added **only** `src/transgenic/evidence/mvp_bootstrap.py` to `DECODER_ONLY`; did not exempt a directory or exempt model implementation. A regression failed on that omission before the fix. This addition changes the operator pin above, not bootstrap math.

Actual producer/consumer interfaces traced:

- `real_probes.py:655-688` inventories executable/import bytes and immutable asset roots, validates the exact consumed-code manifest, emits nested runtime versions and mode/device/real-runtime/generation results, runner binding, initialization/tile/assets digests and wall time. `run_bounded` is imported **after** that inventory, so absence of this import is legitimate. Its reviewed sibling bytes remain mandatory in the decoder.
- `run_bounded.py:145-199` verifies actual prepared tooling, canonical binding and command digests, bounds a real subprocess, validates measured identity/checks, writes `generation_cache_parity.json`, then `.DONE` with receipt SHA256 plus newline. The decoder's `completed_measurement` checks the official wrapper, integer schema/exit status, p3 name, true pass, false timeout, null signal, matching nested binding, independent receipt digest and DONE; FAILED sibling/raw measurement are rejected. It does not pretend that unused diagnostic fields (e.g. elapsed time/allocation label) are execution attestation.
- `mvp_decode_runtime.py:70-89,112-225` retains every declared import rehash, exact relative-root applicability for project code, pre-load model/config/tokenizer/encoder-init imports, probe identity precedence over decoder audit, fixed SIF/versions/offline/cache pins, exact-path dynamic-cache closure and narrowly verified empty HF initializer scaffolding. Numerical project code cannot be admitted solely by the decoder pin. Same-path original imports and caches must remain readable. The inherited identity-schema Python equality weakness noted by the GPU reviewer was not changed; no new strict-schema claim is made.
- `mvp_decode_panel.py:88-185` still verifies runtime before construction, after each model load, and before atomic final publication. CUDA device/allocation is mandatory operationally. COMPLETE panel, serialized final-temp recheck and no-replacement semantics remain unchanged. Post-link cleanup failure can leave a complete final name; no filesystem crash-durability claim.

## One training/probe/evaluation source closure

`mvp_cli_runtime._expected_code_files` (`146-172`) recursively includes **all** `src/transgenic/*.py` plus the MVP training entrypoint. Consequently the integrated manifest adds `mvp_bootstrap.py`, `mvp_windows.py`, and `mvp_decode.py`, even where training does not import them. No static old-A manifest may be reused. `make_launch_readiness._consumed_code_manifest` (`186-193`) uses that exact helper. Probe setup/final recheck and production launch consume it; `mvp_produce_counts.verify_training_code` (`73-109`) independently enforces the identical set at the bound training root. Revision-script decoder orchestration is separately covered by the explicit operator pin and decoder code/audit records, not falsely described as part of the training Python-package set.

New CPU tests exercise the real fresh manifest validator and the real finalizer-to-scorer fixture against this integrated closure, reject omission of new files, and prove future **manifest staging mtimes** do not alter content identity. They do not fabricate future training runs or bypass publication metadata. Existing 350-output integration covers actual finalizer/loaders/rasterizer/panel/transport/scorer with disclosed synthetic remote model/reference/generation/GFFCompare substitutions.

Byte comparison of **every pre-existing772 tracked file** under `src/transgenic`, the MVP entrypoint, `configs`, and `revision/scripts/mvp` passed. In particular model/config/tokenizer, encoder-init, weights-only/B5 initialization, full training/finalizer/runtime, paired metric validator and metric arithmetic are unchanged. The finalizer/scorer's reviewed semantic reconstruction therefore remains on the same implementation bytes. No protocol change: v3, **8192 total tokens**, beam2, no sampling, 50 loci and **350 outputs**, three trained off/on pairs plus seed123 off baseline, approved bootstrap policy. No HiFi or HSP policy selection, structural decision, coverage/withholding decision, or scientific approval.

## Verification and honest failures

CPU Python3.12.2, Torch2.6.0, Transformers4.46.3, Accelerate0.34.2, DuckDB1.5.5; **not the SIF runtime**. External dependencies only. No inner test deadline was modified.

1. Red tests before correction: real772 wrapper fixture rejected old d99 readiness hash; imported bootstrap missing from decoder audit. Both observed failing.
2. Handshake + combined GPU integration: **43 passed, 1 skipped in32.77s**.
3. Focused decoder/runtime/panel/scorer/track/source integration: **191 passed, 1 skipped in420.43s**. Log `/tmp/mvp-final-source-focused.log`.
4. Attempted entire `tests revision/scripts/tests` with outer timeout3600s: **exit139 / segmentation fault**, after several failures, at approximately22% of collected tests; stack in `mvp_decode_panel.py:141` in the350-row CPU fixture. Log `/tmp/mvp-final-source-full-cpu.log`. No complete result count and no claim this full-repository suite passes. Native-crash cause remains undetermined; the same decoder tests passed in focused and scoped reruns.
5. Complete scoped MVP/runtime/weights/B5/GSF/panel run with timeout3600s initially finished **41 failed, 558 passed, 1 skipped, 64 subtests passed in1039.60s**. Log `/tmp/mvp-final-source-mvp-cpu.log`. Investigated child failures: existing tests hard-code the base Python/torchrun and overwrite PYTHONPATH to `.testdeps`, losing the external Accelerate install; psutil was also missing. A first external-venv-only attempt did not fix those hard-coded children (4 failed/1 passed). No source changes were made for these environment failures.
6. Created external `/tmp/mvp-final-source-env` with system packages plus matching Accelerate0.34.2/DuckDB1.5.5 and psutil7.2.2. An ignored local `.testdeps` directory links these external packages for the unchanged hard-coded child harnesses. A diagnostic five-test rerun passed in52.37s (`/tmp/mvp-final-env-check.log`).
7. Same complete scoped suite, unchanged source/deadlines, corrected dependency visibility: **599 passed, 1 skipped, 64 subtests passed in1105.48s**. Log `/tmp/mvp-final-source-mvp-cpu-retry.log`. Includes real CPU Gloo ranks1/2/4, literal four-rank250x24 budget, collective failures, paired finalizer, GPU caller integration and decoder350 transport.

Final scoped command (outer tool timeout3600s):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.testdeps:src:revision/scripts/mvp:/tmp/mvp-decoder-resumed-testdeps:/tmp/b6-trackb-testdeps \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q -rs --tb=short \
  tests/test_mvp*.py tests/test_weights_init.py tests/test_b5_runtime.py \
  tests/test_gsf*.py revision/scripts/tests/test_mvp*.py --durations=20
```

The skipped test is the real dynamic-loader contract under pinned Transformers4.57.6, unavailable locally. Installed4.46.3 dynamic import/scaffolding tests run successfully; they do not establish4.57.6 behavior. The real bounded-publication fixture extracts actual772 runner/probe/readiness/resource source and all seven tooling files from Git. It runs actual tooling validation, real child execution, measurement validation and publication/DONE. **Only allocation and measurement are synthetic CPU substitutions**, explicitly labeled; no fake operational receipt is committed. Negatives retain old d99 hashes for each handshake component, arbitrary mismatches, mutation of absent-from-inventory late runner, declared-but-unloaded modules, relocated changed model/config/tokenizer/init, unknown project files, operator pin and completion fields.

## Operator export and fresh-binding recipe (not executed)

After independent review, export **one** integrated source to one retained canonical checkout for training, probes and evaluation. Do not overlay it with old branch files or generate manifests before all source files are fixed. Retain Git objects needed by frozen-reference authority; a source archive alone is insufficient for those checks.

```sh
# In the reviewed isolated branch, for offline transfer only:
git archive --format=tar --output=/tmp/mvp-final-source-20260916.tar \
  362eb899d8d43164ec1e0fb0f3b174ead9b1ba40
# Recompute the operator pin from reviewed code (never from a supplied receipt):
PYTHONPATH=src:revision/scripts/mvp python -c 'import hashlib,json; import mvp_decode_runtime as r; m={p:hashlib.sha256((r.ROOT/p).read_bytes()).hexdigest() for p in r.DECODER_ONLY}; print(hashlib.sha256((json.dumps(m,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest())'
```

In the intended retained canonical checkout **inside the approved SIF**, regenerate the absolute consumed-code/package manifest through the existing prepare workflow (`make_launch_readiness.py`), seven-file tooling identity, immutable input/source/cache/SIF identities and seed-specific prepared bindings. Do not substitute the relative comparison pin above for these manifests. Existing prepare must still validate accepted candidate artifacts and exact approved recipe. Re-audit original checkpoint sidecars and both pinned offline cache layouts; preserve original declared module-cache/import paths through decode. Do not reuse old receipts, old v2/d99 manifests, or the d99 queued P1 outcome for this source.

Runtime request requires independently reviewed expected receipt digest, actual SIF path/digest, and the new decoder pin above. Receipt must be the **fresh completed** `generation_cache_parity.json` plus retained matching `.DONE`, not `measurement.json`; device must be CUDA. Generate request/export after review, not by copying identities from untrusted receipt content. Same source files must remain byte-stable through all probes, training and decode. New root paths, package versions or source changes require fresh bindings; code changes also require a new decoder pin and review.

Fresh P1 and **all six readiness gates** remain required before finetuning: real tokenizer, real Hyena step0 all tiers, generation-cache parity, native bf16/Inductor, four-rank smoke, and resource budget. Resource execution additionally requires prospective controller authorization and allocation-wide accounting including prior attempts/startup/verification/teardown inside the18000s cap. Partial profiling is not readiness. No new submission is authorized here.

## Remaining actual SIF facts

Deployment pins remain SIF SHA256 `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`, Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA13.4, Transformers4.57.6, Accelerate1.15.0, Hyena revision `0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446`. These are **required pins, not newly observed runtime facts**. Actual SIF CUDA/bf16/Inductor/NCCL, real Hyena gradients and pinned-loader module closure, constrained8192/cache/EST decode, complete real three-seed training, real350 outputs, GFFCompare0.12.6 with frozen reference/evidence and resource feasibility remain unverified by this task. A16-new-token generation probe does not certify8192 constrained generation. Final independent review, six-gate readiness and scientific decisions remain separate HOLD conditions.
