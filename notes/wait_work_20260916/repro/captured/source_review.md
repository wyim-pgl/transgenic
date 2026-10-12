# Independent final MVP source integration review — 2026-09-16

## Verdict and scope

Reviewed detached **b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6**, executable integration **362eb899d8d43164ec1e0fb0f3b174ead9b1ba40**, in native isolated worktree `/tmp/pi-agent-6fb47be4-03e7-43b-9e027b8f`.

**ACCEPT this exact source for controller staging/export, fresh bindings and separately authorized bounded GPU diagnostics. No new Critical/Important cross-component source defect found. NOT training-ready, production-decode approval, six-gate readiness, or scientific approval.** Genuine runtime applicability is still a gate, not an inference from CPU tests. This review resolves the old decoder-versus772 source-interface blocker; it does not assert that a deployed SIF receipt already passes the entire live closure.

Used the neutral final delta/report and independent GPU772 and decoder FIX1 reviews under `/tmp/mvp-final-review-context-20260916/`. Reused their unchanged-component conclusions only after the Git-byte and boundary checks below. Graph-first orientation, queries and skeletons preceded source inspection; relevant truncated definitions were opened. Recovery `.atl/resume.md`, `.atl/todo.md` and `.atl/quarantine.md` are absent in this worktree. Advisor consultation failed with `claude_code_version_too_old`; no advisor approval is claimed.

## Independent ancestry and byte checks

These are locally recomputed checks, not copied report assertions:

- `981489d`, `d99b1f8`, `772c38c` and `362eb899` are ancestors of the reviewed HEAD. `362eb899..b660bf1` changes only `notes/MVP_FINAL_SOURCE_INTEGRATION_REPORT.md`.
- Each integrated cherry-pick has the preceding listed commit as its direct parent, starting at772. Original/integrated patches over `src train revision configs tests` are byte-equal: `eefe5b5→47095c8`, `2276fcb→530bb2d`, `1a9253d→1b42b99`, `64b6d5b→62c130b`, `0f9f777→e22b660`, `668299a→d2fd5b0`, `e6950b7→ec5445b`, `6366d8a→7d8ff09`. No executable conflict resolution is hidden by the report-only conflict.
- All **74 pre-existing772 files** selected by `git ls-tree -r --name-only 772c38c src/transgenic train/train_HyenaTransgenic_MVP.py revision/scripts/mvp configs` equal their actual `git show 772c38c:<path>` bytes in the final checkout. This includes GPU resource/probe/runner, model/config/tokenizer/encoder initialization, learned-projection implementation, training/finalizer/runtime/metrics and recipes.
- `a4e8808` is not an ancestor. No tracked `.venv`, `.testdeps` or `site-packages` path exists; intersection with a4's added `.venv/` paths is empty. Final added/changed paths were checked, not merely ancestry.
- `git diff --check 7d8ff09..HEAD` and `git diff --check 772c38c..b660bf1` pass. Worktree was clean before this report.

Independently recomputed SHA256 values:

| Identity | Result |
|---|---|
| Relative component source, **86 tracked files** | `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44` |
| Closed decoder audit, **13 files** | `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb` |
| Actual772 `real_probes.py` | `a40b46b3e2df1f5a80a4357d06c0e77bf1cdedd66c0fd22582df7de0541bbc58` |
| Actual772 `run_bounded.py` | `8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932` |
| Actual772 `make_launch_readiness.py` | `73728aee67611575448722cb4180062917ee0c86330749ff1ca648346e75a6d5` |

Source-pin reproduction: enumerate `git ls-files src/transgenic train/train_HyenaTransgenic_MVP.py revision/scripts/mvp configs`; hash each file's bytes; hash `json.dumps(mapping, sort_keys=True, separators=(',', ':')) + '\n'`. Decoder pin uses the same encoding over the literal `DECODER_ONLY` paths. I extracted that literal and `PROBE_CODE` with Python AST and independently compared the latter against Git objects. These relative pins are **not** the absolute consumed-code manifest or authorization.

## Cross-component wiring checked

### Reviewed772 completion, tooling and runtime boundary

`revision/scripts/mvp/mvp_decode_runtime.py:19-41,44-84` now selects exactly the three772 handshake hashes above. There is no receipt-selected hash allowlist or directory exemption. `make_launch_readiness.py:102-106` includes all seven tooling files, including resource. Generation need not import resource; any declared resource import is still rehashed. `real_probes.py:657-688` inventories imports before its late `run_bounded` import, so that one missing inventory entry is legitimate, while runtime verification still checks its sibling bytes independently (`mvp_decode_runtime.py:175-178`).

The actual runner (`run_bounded.py:145-199`) validates prepared tooling, canonical binding and bound command, runs a bounded subprocess, validates nested measurement and binding, publishes `generation_cache_parity.json`, then digest-plus-newline `.DONE`. The decoder requires that completed wrapper, typed successful status, matching nested binding, independent receipt digest and DONE; raw measurement, FAILED sibling and malformed completion fail (`mvp_decode_runtime.py:44-66`). This is operator-trusted evidence, not cryptographic execution attestation.

Independently ran `tests/test_mvp_decoder_runtime.py:26-102`: the fixture extracts actual772 probe/readiness/runner/resource plus the remaining tooling files from Git, executes actual tooling validation and real child/publication code. Allocation and measurement are explicitly synthetic CPU substitutions. Its positive wrapper reaches the runtime-version gate; it does **not** pass or bypass the genuine runtime gate. Old-d99 per-file negatives, late-runner mutation, declared-but-unloaded imports, unknown project files and operator-pin checks also ran.

### Bootstrap addition is a closed audit dependency, not a model override

`mvp_produce_counts.py:18` imports `mvp_bootstrap.validate_paired_record` before runtime verification. The new module contains count validation/bootstrap policy, not the numerical model implementation. Adding only `src/transgenic/evidence/mvp_bootstrap.py` to the fixed 13-file audit (`mvp_decode_runtime.py:34`) is justified; leaving it absent would reject a legitimate decoder import not consumed by generation. Its dependencies are standard-library/numpy imports, not a new model factory or vendored dependency.

`mvp_decode_runtime.py:111-141` still gives probe-consumed project bytes precedence over audit entries. Actual model/config/tokenizer/encoder-init are imported before construction (`:201-205`) and are not audit exceptions. Every declared original import is retained/rehashed, external modules require exact original path/hash, and dynamic HF modules require exact cache-path/import identity plus immutable asset membership or the narrowly checked empty initializer. The operator pin cannot override changed model bytes or authorize arbitrary newly loaded modules. Module-cache relocation remains unsupported. Actual4.57.6 import closure, including any late decoder dependencies, must still be measured; unknown imports fail closed rather than being waived by this review.

### Fresh training manifest and historical scorer closure

`src/transgenic/training/mvp_cli_runtime.py:146-172` and `make_launch_readiness.py:186-193` share the recursive all-package-Python plus MVP-entrypoint closure. It now includes bootstrap, windows and decode even if training does not import them. Reusing a static772 manifest is invalid. `mvp_produce_counts.py:73-109` derives the retained historical training root from the bound B5 implementation, enforces that root's exact source set/canonical paths/hashes and five-package schema, and separately checks supported local initialization-semantic implementation bytes. It does not execute arbitrary historical Python or substitute the scorer checkout's larger source set. DuckDB is mandatory in the code manifest but not misrepresented as part of the finalizer's four-package runtime metadata.

The real fresh-manifest and finalizer-to-scorer regression tests ran here: new files are included, omission is rejected, and future manifest staging mtime alone does not change content identity. This does not validate future training completion. Revision-script orchestration remains separately operator/audit-bound, not falsely included in the training package manifest.

### Metrics, generation and panel transport

The actual `3e15b49..772c38c` metrics diff only extracts the pre-forward validator; off/on forwards, CE numerator/denominator arithmetic, composite loss and mode restoration are unchanged (`mvp_metrics.py:64-114`). Final metrics bytes equal772. Bootstrap numeric-field compatibility is independently tested (`tests/test_mvp_scoring.py:137-139`, passed here). Thus source integration does not change the finalizer/scorer paired-record semantics.

Model/config/generation/projection implementation remains byte-identical to reviewed772; the added B decoder adapter matches the reviewed cherry-picks. `mvp_decode.py:128-201` uses v3 constrained beam2/cache generation, no sampling, **8192 total tokens**, strict learned final-state loading including projection, and the approved weights-only zero-projection baseline. These are distinct from the short numerical parity probe; neither source identity nor 16-new-token parity certifies real8192 generation.

Compared actual producer policy (`mvp_prepare_panel_inputs.py:151-170`) to consumer (`mvp_verify_panel_inputs.py:13-84`): COMPLETE digest, exact typed policy/shared request fields, code/input identity, package versions, ordered50 loci, windows/sequence, float16 `[30720,8]` tracks, N masking, independently rerasterized evidence/provenance and support summaries agree. Decoder rechecks runtime before construction, after each load and at publication (`mvp_decode_panel.py:88-185`). Seed loops produce exactly three off/on pairs plus123/off per locus. `mvp_produce_counts.py:349-367,370-471` accepts the same350 identities/window/checkpoint/file records and consumes the emitted evaluation fields.

Inspected `tests/test_mvp_decoder_integration.py:223-307`: its actual finalizer/panel/loaders/rasterizer/transport/scorer integration checks350 outputs/comparisons, raw grammar, coordinate offsets, identical on tracks across seeds and zero off/baseline. Remote models, runtime, generation and GFFCompare are disclosed synthetic substitutions. Reused existing scoped test evidence for that unchanged fixture rather than rerunning the350 path or claiming GPU/scientific results.

## Evidence, unresolved failure and nonblocking debt

Independent focused command, from this worktree (external dependencies only; no source/deadline changes):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=$PWD/src:$PWD/revision/scripts/mvp:/tmp/mvp-final-source-env/lib/python3.12/site-packages:/tmp/mvp-decoder-resumed-testdeps:/tmp/b6-trackb-testdeps \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest -q -rs --tb=short \
  tests/test_mvp_decoder_runtime.py tests/test_mvp_final_source_integration.py
```

**35 passed, 1 skipped in35.42s.** Separate exact schema compatibility test: **1 passed in1.79s**. Skip explicitly reports pinned Transformers4.57.6 absent; installed4.46.3 behavior is not SIF validation.

Controller's **46 passed/1 skipped in36.68s** is attributed evidence, not rerun here. Read the scoped retry log's actual ending: **599 passed, 1 skipped, 64 subtests passed in1105.48s**. This corroborates the implementer's scoped result, not whole-repository success.

**Unresolved native crash — nonblocking for this bounded-diagnostic-only clearance, not dismissed or fixed.** I read `/tmp/mvp-final-source-full-cpu.log`: failures precede `Fatal Python error: Segmentation fault` around22%; current frame is `mvp_decode_panel.py:141` (`tracks[locus.gene_id].to(device)`), invoked by `tests/test_mvp_decode.py:190` inside the synthetic350 orchestration fixture. There is no native backtrace/root-cause diagnosis or completed full-suite summary. The report records exit139. Reproduction context is the implementer's full `tests revision/scripts/tests` run; it was **not repeated** here. Do not attribute this crash to missing Accelerate/psutil merely because those explain separate Python failures. Passing focused/scoped runs do not erase possible order-dependent native state. Conversely, the supplied evidence does not establish a reproducible final-source GPU probe defect or a new integration regression, so withholding all bounded diagnostic staging on that speculation is unwarranted. Preserve logs/environment and isolate the failure before any whole-suite-green/reliability claim; recurrence in the intended bounded runtime requires stop/investigation, not automatic retry.

Other inherited nonblocking debt/limits:

- `real_probes.py:84` and `mvp_decode_runtime.py:170` accept identity schema `true`/`1.0` through Python equality with1. GPU component review reproduced this; exact bytes are unchanged. It does not waive byte hashes or break genuine integer1 output. Strict identity-schema claims remain unsupported; exact integer checks are future hardening, not part of this report-only task.
- `mvp_decode_panel.py:25-42,162-185`: final temp is closed/fsynced and rechecked before atomic no-replace link. Failure after successful link can leave a complete final file alongside failure diagnostics; no directory-fsync crash-durability claim. This does not authorize consuming partial output.
- Original import/cache paths, retained historical source and checkpoint sidecars remain operational dependencies. CPU package visibility fixes are external test-environment repairs, not deployment evidence.

## Controller release conditions

Stage **one retained exact source tree** for probes/training/evaluation; retain required frozen-reference Git authority, not only a source archive. Inside the approved SIF regenerate absolute consumed-code/package and seven-file tooling manifests, canonical input/source/cache/SIF identities and seed-specific prepared bindings. Independently supply the completed receipt digest and decoder pin above; do not derive authority from receipt-selected hashes or reuse d99/v2/old772 manifests. Changed source or paths require corresponding fresh review/bindings.

Fresh P1 and all six real readiness gates remain mandatory before finetuning. Real pinned SIF Torch/CUDA/Transformers4.57.6/Accelerate, loader/import closure, Hyena gradients, bf16/Inductor/NCCL, resource feasibility, constrained8192 decode, real three-seed training and350 GFFCompare outputs remain unverified. Resource diagnostics additionally require prospective controller authorization and allocation-wide accounting of prior attempts/startup/verification/teardown within18000 seconds; no automatic extra attempts or unaccounted warmups.

No SSH, status queries, jobs, subagents, source edits, merge or push occurred. P1 **3163545**, queued against d99 per the last controller record, was untouched; no live status is asserted and its outcome cannot certify b660. Scientific structural/coverage/withholding decisions remain separate HOLDs. This report is the only intended commit change.
