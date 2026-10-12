# Decoder runtime integration fix 1 — 2026-09-16

## Scope and status

Branch `mvp-decoder-runtime-fix1-20260916` starts at `e6950b7` in an isolated native worktree. Addresses integrated independent review I1–I3; the old integration/review reports remain unchanged historical records. Only runtime helper, its caller, tests and this report change. No model/trainer, D1 rasterization, D2 publication implementation, panel contract or 350-row transport changes. No SSH, jobs, subagents, main merge or push.

**CPU verification passes; independent review is next. A genuine GPU receipt and operational SIF/8192-token decode remain pending.** No historical CUDA measurement was invented or claimed. Advisor consultation failed with a client/model-version incompatibility; no advisor approval is claimed.

## I1: project applicability and reviewed handshake

Removed the blanket current-ROOT exemption. Current loaded project files map by exact relative path from this checkout to the historical checkout identified by the probe's canonical `revision/scripts/mvp/real_probes.py` location. Their digests must match A's import inventory. Original declared files are still individually rehashed, including unused imports; relocation does not waive their retention. Concrete project model/config/tokenizer/encoder-init modules are imported and checked before construction, and live imports are checked after loading and before final publication. Loaded project model/config objects must be the actual project classes; loaded remote classes retain the immutable source/config checks.

The numerical wrapper, encoder initialization, configuration and tokenizer are not decoder-audit exceptions. The exact `DECODER_ONLY` tuple is a separate, closed orchestration/grammar/transport boundary, not a directory exemption or caller-selectable allowlist. It includes the decoder/panel/producer helpers, evidence decoder/windows/tracks, constrained grammar and GSF utilities. Its canonical relative-path/digest mapping must match the independently operator-supplied `runtime_preflight.decoder_code_sha256`. If A imported a file in that boundary, A's identity takes precedence; the decoder pin cannot override it. Unlisted project files cannot pass merely because they are under ROOT.

This distinction does **not** claim A's 16-new-token numerical generation probe certified constrained grammar or total-length-8192 generation. Those remain separately reviewed decoder code and pending genuine execution gates.

The helper hard-pins actual `git show d99b1f8:revision/scripts/mvp/{real_probes,run_bounded,make_launch_readiness}.py` SHA256 values. It checks the real probe and preparation handshake import identities against those constants and rehashes all three sibling files. Important source detail: A snapshots its imports **before** the final `run_bounded` import. Its absence from the snapshot is therefore supported, not filled in with an invented historical import; the independently pinned sibling bytes are still mandatory. A supplied arbitrary probe executable is no longer accepted solely for appearing in its own import inventory.

## I2: actual completed bounded receipt

Authority is the actual `d99b1f8` Git source, not a reconstructed wrapper convention. Input must be the independently digest-pinned `generation_cache_parity.json`, with exact sibling `generation_cache_parity.DONE` bytes `sha256(receipt) + '\n'`. Any sibling `generation_cache_parity.FAILED.json`, missing/mismatched DONE, raw measurement alone, failed execution or binding mismatch is rejected.

Checked wrapper fields: integer `schema_version=1`, `preflight=p3_generation_parity`, integer `exit_code=0`, `timed_out=false`, present `signal=null`, `pass=true`, 64-lowercase-hex `bindings_sha256` equal to the nested measurement binding. Only then is the nested measurement passed through existing genuine-generation/version/CUDA/asset checks. Receipt expected SHA256 remains an independent request input, never copied from the receipt itself. DONE identity joins the repeated input audit.

The test extracts the real three authority files from Git, verifies their fixed digests, invokes the real bounded runner and real subprocess execution/publication, and supplies an explicitly labeled **synthetic CPU measurement**. Allocation/tooling are explicit fixture substitutions. Real runner measurement-identity validation and completion publication execute; this is a wrapper-format regression, not evidence of GPU execution. The completed wrapper now reaches the next actual runtime/version check.

## I3: HF scaffolding and cache policy

For both bare `transformers_modules` and nested modules, exact original probe import path+digest remains mandatory. Dynamic-cache relocation is explicitly unsupported: retain the original bound `HF_MODULES_CACHE` and every original declared import. Immutable model/tokenizer caches likewise retain their original absolute-path tree identities; no relocation rewrite or blanket remote-hash membership exception was introduced.

HF-generated scaffolding is allowed only when the file is empty, named exactly `__init__.py`, under the bound module cache, and its relative path matches the imported package name exactly. Actual implementation Python still must match the pinned model/tokenizer snapshot source-hash closure as well as the genuine probe import identity. No placeholder files are added to immutable snapshots.

A real local dynamic import using installed CPU Transformers **4.46.3** exercises generated empty package initializers, including the bare root. A separate pinned **4.57.6** loader-contract test runs when that version is installed; here it is explicitly skipped because it is unavailable. Neither the 4.46.3 test nor a skipped 4.57.6 test establishes actual SIF behavior.

## Operational request change

`runtime_preflight` now contains:

- `receipt: {path: '/retained/probe/generation_cache_parity.json', sha256: '<independently supplied expected digest>'}`;
- `sif: {path: '/actual/container.sif', sha256: '<existing fixed SIF digest>'}`;
- `decoder_code_sha256: '<independently reviewed operator code pin>'`.

The decoder pin hashes UTF-8 `json.dumps(mapping, sort_keys=True, separators=(',', ':')) + '\n'`, where `mapping` is exactly `{relative: sha256(ROOT / relative) for relative in DECODER_ONLY}`. Establish it from the reviewed checkout outside the executing request; computing a digest of unreviewed code is not review or approval. No public override accepts unpinned code.

The caller now defaults to `cuda`, refuses CPU/non-CUDA requests, checks the actual device index and exercises a CUDA allocation before model/tokenizer loading. Tests explicitly monkeypatch this gate to CPU; there is no public CPU bypass.

The existing SIF environment/path and operator-pinned receipt remain trusted operational evidence, not cryptographic execution attestation. Genuine pinned-loader, CUDA, tokenizer, grammar/cache/EST and 8192-total-token compatibility still require controller-authorized execution.

## D1/D2 and publication limits

D1 coverage/junction separation, actual rasterizer verification, COMPLETE-panel validation and all 350 transport rows are unchanged. D2 still provides closed-temp atomic no-replace visibility, not directory-fsync/power-loss durability. A pre-link serialization/flush/fsync/close/recheck failure cannot publish the final name. A **post-link cleanup failure can leave a complete final file**, potentially alongside a FAILED diagnostic; that is not a partial-file publication. The earlier broad report wording about all failures leaving no evaluation file should not be applied to post-link cleanup.

## Verification

Existing external environment only: Python 3.12.2 at `/home/wyim/miniconda3/envs/pdb/bin/python`, external dependency paths `/tmp/mvp-decoder-resumed-testdeps:/tmp/b6-trackb-testdeps`. No generated environment or dependency files committed.

Focused final tests before the optional pinned-loader skip: **28 passed in 16.03 s** (`/tmp/mvp-runtime-fix1-focused.log`, full output and durations retained). Original 107-test suite: **107 passed in 409.07 s** (`/tmp/mvp-runtime-fix1-existing107.log`, full output and durations retained). Final combined run below includes all final source edits and the explicit pinned-loader skip:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 PYTHONPATH=src:revision/scripts/mvp:/tmp/mvp-decoder-resumed-testdeps:/tmp/b6-trackb-testdeps \
 /home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q -rs \
 tests/test_mvp_decoder_runtime.py tests/test_mvp_decoder_integration.py tests/test_mvp_decode.py \
 revision/scripts/tests/test_mvp_panel_inputs.py tests/test_mvp_score_producer.py \
 tests/test_mvp_score_producer_review.py tests/test_mvp_score_producer_integration.py \
 tests/test_mvp_tracks.py --durations=15
```

Full final output (`/tmp/mvp-runtime-fix1-final.log`, exit 0):

```text
.........................s.............................................. [ 52%]
................................................................         [100%]
============================= slowest 15 durations =============================
142.09s call     tests/test_mvp_decode.py::test_runner_exact_350_no_overwrite_and_partial_failure
76.53s call     tests/test_mvp_decoder_integration.py::test_full_publication_panel_decoder_counts
13.87s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[internal_schema]
11.15s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[wrong_request]
10.25s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[symlink_escape]
9.85s call     tests/test_mvp_tracks.py::test_real_bam_to_tracks_and_129024_cpu_smoke
9.22s setup    tests/test_mvp_decoder_integration.py::test_full_publication_panel_decoder_counts
8.63s setup    tests/test_mvp_decoder_integration.py::test_real_panel_high_junction_and_biological_zero
8.09s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[wrong_authority]
7.86s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[changed_jsonl]
7.60s call     tests/test_mvp_decoder_integration.py::test_real_panel_high_junction_and_biological_zero
7.09s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[coverage]
6.18s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[missing_marker]
6.15s setup    tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[wrong_sequence]
3.20s setup    revision/scripts/tests/test_mvp_panel_inputs.py::test_no_retained_flank_fallback
=========================== short test summary info ============================
SKIPPED [1] tests/test_mvp_decoder_runtime.py:183: pinned 4.57.6 not installed; CPU 4.46.3 test is not SIF validation
135 passed, 1 skipped in 404.94s (0:06:44)
```

New negatives cover relocated changed model/config/tokenizer/encoder-init, attempted decoder-audit override, unknown local project code, operator pin mismatch, arbitrary probe/handshake identities, mutation of declared-but-unloaded imports, mutation of the late-imported runner, every completed-wrapper rejection above, changed/nonempty/unbound/misnamed HF scaffolding, unsupported cache relocation, unpinned remote implementation and CPU-device rejection. Existing tests retain actual candidate/finalizer/loaders/panel/rasterizer/count-transport work with explicitly synthetic reference, remote model/tokenizer, GPU generation and GFFCompare substitutions. These limitations remain unchanged.

Independent review, real pinned-SIF receipt/decode, real frozen inputs and three completed seeds, genuine GFFCompare, structural scoring, coverage/withholding policy and scientific approval are all still pending.
