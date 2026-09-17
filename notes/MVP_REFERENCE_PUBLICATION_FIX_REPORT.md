# Reference publication integrity fix — 2026-09-16

## Source and deployment gate

Implementation: **`9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558`**.
Branch: `mvp-reference-map-publication-20260916`, created directly from
`66f130d7abee262b2c3c840fc129ad8fd03ec29a` in an isolated worktree.
This report is a subsequent documentation-only commit; pins below identify the implementation.

**Controller independent re-review is required before packaging a new real-50 CPU job.**
Actual accepted-corpus 50-locus compatibility and decoder SIF compatibility remain unverified.
The CPU wrapper's b660 hardcoded source/inventory/component guards intentionally remain
unchanged: this is a template, not a deployable release. Packaging must repin the
new evaluation commit, full source inventory and final staged entrypoint. Preserve
old panel namespaces, manifests, COMPLETE markers, review/task histories, training
source and probe artifacts. Do not substitute template hashes for final release hashes.

No GPU operations, SSH, job submission/inspection, subagents, main-worktree edits,
merge, push, or production-data query/hash were performed. No independent approval
is claimed. The advisor consultation failed due client/model-version incompatibility.

## Narrow changes

- `revision/scripts/mvp/mvp_produce_counts.py:327–369`: mapping recheck now brackets
  potentially long source hashing with WAL checks (including dangling WAL symlinks).
  Source stat identities (device/inode/size/mtime/ctime, excluding read-updated atime)
  are compared across the whole hash pass, detecting same-size changes even after
  the affected source's hash has already returned.
- The producer serializes its receipt to an exclusively created `.pending` file,
  flushes/fsyncs/closes it, checks JSON readability and rechecks inputs, output
  identities and mapping before atomic no-replace hard-link publication. Counts and
  statistics can remain as diagnostics; no final receipt is published for detected
  late source/WAL mutation. Failed staging is retained. No decoder imports or
  general publication framework were added.
- `tasknotes/reference_mapping/run_panel_cpu.py:76–183`: standalone reference audit
  uses the same staging/publication helpers. In `main`, compatibility publication
  is deferred through serialization of either REFERENCE_ONLY or
  OBSERVATIONS_COMPLETE. Source/authority and reference/mapping checks run after
  serialization and before either final name is linked. A late mutation leaves
  pending diagnostics and FAILED, not compatibility/completion success names.
- Strict-parser failures remain observations with explicit failure fields and
  per-locus fallback. They are not confused with source-mutation publication
  failures. Mapping, empty-reference guard, prediction logic and scientific counts
  are unchanged.

Publication uses individual same-filesystem atomic no-replace links, not a
multi-file transaction or a source lock. The final marker is published last.
As with the existing decoder pattern, checks enforce the prepublication boundary;
this is not a claim that external writers can never mutate sources afterward.
No external data is overwritten or deleted; only successfully published owned
staging names are unlinked.

## Verification

Before the fix, the focused reproduction produced **2 failed, 1 passed** in 31.09s:
counts-serialization WAL and WAL-during-hash tests failed because no exception was
raised. The initial append-based source mutation was already detected by size;
the final regression strengthens this to a same-size mutation after hashing.

Intermediate focused run: **48 passed, 364.96s**, covering new publication tests
and existing mapping tests. After extending hash timing coverage and excluding
atime from stat comparisons, the final combined run was:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
/tmp/mvp-final-source-env/bin/python -m pytest -q -rs \
 tests/test_mvp_reference_publication.py \
 tests/test_mvp_reference_mapping.py \
 tests/test_mvp_score_producer.py \
 tests/test_mvp_score_producer_integration.py \
 tests/test_mvp_decoder_integration.py \
 tests/test_mvp_decoder_runtime.py \
 tests/test_mvp_final_source_integration.py
```

**128 passed, 1 skipped, 656.89s** (1800-second outer budget; no inner deadline changes).
This includes the prior 98-test scope plus 30 publication regressions. Skip:
pinned Transformers 4.57.6 unavailable; installed 4.46.3 is not SIF validation.

New tests actually inject WAL/source changes after counts, statistics and staged
receipt serialization; after CPU compatibility serialization on joint and
per-locus fallback paths; and after reference-only marker, panel counts and final
completion serialization. Every injection asserts that it ran and that final
success names are absent. Hash tests inject WAL and same-size source mutations
both after the corpus hash and during the last source hash. Positive CPU main
paths use the genuine tiny mapping/parser fixture with explicitly stubbed
packaging/allocation/raster work, not any real job. Producer tests reuse the
full350 actual-finalizer positive integration fixture. The retained first-gene
fixture remains real prefix evidence, not real-50 evidence. A collision injected
inside the publication callback verifies no-replace behavior and preservation of
external destination bytes and failed staging.

`git diff --check` passed. This invariant command returned an empty diff:

```sh
git diff --exit-code 66f130d -- src train configs revision/protocols \
 revision/scripts/mvp/real_probes.py revision/scripts/mvp/run_bounded.py \
 revision/scripts/mvp/make_launch_readiness.py revision/scripts/mvp/mvp_decode_runtime.py
```

Only producer, CPU template, new regression tests and this new report changed.
Original implementation/review/task artifacts were preserved. Graft was queried
before source inspection and rebuilt locally afterward (ignored graph cache).

## Recomputed implementation pins

| Identity | SHA256 |
|---|---|
| Producer | `6069fb5fe0771015d044d820a4cbbff313d9f3d2982991e1c58c17bc8aa726dc` |
| Decoder operator closure | `b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088` |
| Evaluation component closure, 86 files | `ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e` |
| CPU template (not future repinned entrypoint) | `09281c0bd38304e2adc317f8f9216fb3686627c64b768985dd84586dd0c75a23` |
| Unchanged panel | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |

Operator closure uses unchanged `mvp_decode_runtime.DECODER_ONLY`; component
closure uses tracked `src/transgenic/`, `configs/`, `revision/scripts/mvp/`, and
`train/train_HyenaTransgenic_MVP.py`. Both hash sorted compact relative-path/SHA256
JSON plus newline. These pins were recomputed locally, not substituted into
runtime guards. Fresh panel/evaluation packaging and independent controller review
remain mandatory.
