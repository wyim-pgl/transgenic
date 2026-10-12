# Independent reference-publication fix re-review — 2026-09-16

## Verdict and scope

**Approve the scoped publication-integrity correction: the original Important late-WAL finding is closed. No new Critical or Important regression found in the reviewed delta. This is not deployment or actual-data approval.**

Reviewed detached `1de033a7148d778e41923d89cadca9671ec8b746` against `66f130d7abee262b2c3c840fc129ad8fd03ec29a` in an isolated native worktree. Implementation is `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558`; the later commit adds its report. Read the neutral prior independent review and `reference_publication.fix1.diff`, and the committed implementation report. The previously reviewed original-ID mapping algorithm was not broadly re-reviewed.

Only producer, CPU template, new publication regression tests and implementation report differ. This review changes only this report. No SSH, jobs, production-data query/hash, source edits, subagents, merge or push.

## Publication boundary verified

- `revision/scripts/mvp/mvp_produce_counts.py:327–369`: `stage_json` exclusively creates the sibling `.pending`, serializes JSON, flushes, fsyncs and exits the context (closing the original handle). `publish_staged` rejects occupied final names, reads staged JSON, invokes the callback, then uses atomic no-replace `os.link`. Successful staging links are unlinked; failed staging and external destinations are retained.
- `mvp_produce_counts.py:562–576`: counts/statistics serialization and closed receipt staging precede the final callback. It rechecks recorded input and output identities, then mapping sources/WAL, before linking `receipt.json`. Counts/statistics alone can remain diagnostic artifacts; they are not a successfully published producer receipt.
- `mvp_produce_counts.py:327–342`: mapping WAL absence is checked both before and after the complete mapping-source hash pass, using `lexists` (also rejecting dangling WAL symlinks). Device, inode, size, nanosecond mtime and ctime are compared across that pass. Atime is correctly excluded. The regression cases include same-size corpus changes after its hash and during the final source hash, not just size-changing appends.
- `tasknotes/reference_mapping/run_panel_cpu.py:76–81,127–133,161–179`: standalone compatibility publication uses the same closed staging helper. Main defers compatibility publication until reference-only or full-completion marker serialization has finished. The callback then verifies source/authority and reference/mapping identities before either final name is linked. The completion marker is last. Injected late mutation leaves FAILED and pending diagnostics, with neither final compatibility nor completion name.
- `run_panel_cpu.py:103–125`: truthful strict-parser failure observations remain distinct from mutation failure. Joint failure retains explicit error fields and individual fallback diagnostics; it is not silently converted into a successful joint parse. The positive fallback test still publishes the diagnostic report when sources are unchanged. No GFF repair, deduplication or scientific-policy change was introduced.

**Source/publication caveat:** these checks define a prepublication boundary, not a lock against later external writers. Mapping hashes/stat checks cover their complete pass, not an indefinitely frozen filesystem. Hard links are individually atomic and no-replace, not a transaction across compatibility and completion files. An I/O failure or destination collision between links can leave a compatibility report without completion; do not treat that as completed observations. There is no destructive rollback of external data and no claim of crash-durable multi-file publication.

## Independent focused execution

Environment: existing `/tmp/mvp-final-source-env/bin/python`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src`.

```sh
python -m pytest -q -rs tests/test_mvp_reference_publication.py \
  tests/test_mvp_reference_mapping.py::test_cpu_wrapper_uses_shared_mapping_and_diagnostic_fallback
```

**32 passed in 408.02s**: all 30 new regressions plus two unchanged positive CPU joint/fallback cases. This covers producer serialization targets, standalone CPU compatibility, reference-only/full completion, hashing-time mutation, positive completion and no-replace preservation. CPU completion fixtures stub packaging/allocation/raster work; they do not represent a real job.

Also executed the **original independent reproduction separately**, not merely a similarly named test: the full350 integration function with an in-memory `json.dump` hook writing `corpus.json.wal` immediately after `counts.json`. It now raised `ValueError: corpus WAL appeared`; injection occurred exactly once; final `result/receipt.json` was absent; `result/receipt.json.pending` and the unchanged injected WAL remained. Thus the formerly successful consumable receipt is no longer published. This standalone adversarial execution is not counted as an additional pytest-suite pass.

The implementation agent's **128 passed / 1 skipped / 656.89s** remains attributed evidence; this review did not duplicate that broad suite. Its Transformers-version skip is not SIF validation. The implementation report's prior red-phase result was not independently rerun by reverting source.

Independently repeated `git diff --check 66f130d HEAD` and the invariant comparison below; both exited zero, with the invariant diff empty:

```sh
git diff --exit-code 66f130d HEAD -- src train configs revision/protocols \
  revision/scripts/mvp/real_probes.py revision/scripts/mvp/run_bounded.py \
  revision/scripts/mvp/make_launch_readiness.py revision/scripts/mvp/mvp_decode_runtime.py
```

Model/training/source/configuration, panel/core20/windows, runtime and probe bytes remain unchanged by this fix. The changed production source is confined to the evaluation producer, plus the separate CPU template. Prediction/scoring logic is unchanged in the inspected delta.

## Independently recomputed pins and remaining release gates

Recomputed from reviewed local bytes; compact sorted relative-path/SHA256 JSON plus newline for closures. Operator membership comes from unchanged `DECODER_ONLY`; component membership is the tracked closure documented in the implementation report.

| Identity | SHA256 |
|---|---|
| Producer | `6069fb5fe0771015d044d820a4cbbff313d9f3d2982991e1c58c17bc8aa726dc` |
| Decoder operator closure | `b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088` |
| Evaluation component closure, 86 files | `ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e` |
| CPU template only | `09281c0bd38304e2adc317f8f9216fb3686627c64b768985dd84586dd0c75a23` |
| Unchanged panel | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |

All agree with the implementation report. This recomputation supplies operator pins, not actual-data approval or a real runtime preflight.

**The CPU wrapper intentionally retains old b660 hardcoded source/inventory/component guards and is not deployable as-is.** A new release must repin the evaluation commit, full source inventory and final staged entrypoint; the template hash above is not a future repinned entrypoint hash. Use a fresh immutable panel namespace because the producer identity changed. Do not rewrite prior panel manifests/COMPLETE markers or waive the identity gate. Preserve the separately bound b660 training source and 772 probe artifacts.

Actual accepted-corpus 50-ID compatibility, strict-parser/geometry/duplicate observations, and actual decoder package/SIF compatibility remain separate unresolved gates. Real first-gene prefix evidence and synthetic 50-locus tests do not certify the accepted nuclear panel. Controller retains the release decision.

Graft was consulted before source inspection; its reported savings total approximately 1,141,438 tokens. Advisor consultation failed with client/model-version incompatibility and provided no advice.
