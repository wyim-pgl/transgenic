# Task 1 independent review

Reviewed `19049ba..19d319d` using the supplied diff, Task 1 brief and implementation report. Read-only inspection covered the relevant unchanged C2 tracing, clone and canonicalization helpers and the actual `198ce14` trainer interfaces, using graft before source inspection. No source edits, git changes, remote actions or full-suite reruns were performed.

**Spec verdict: changes required.** The fresh-run flow, frozen schedule, observed-history policy, exact comparison, allocation provenance and resource gates substantially implement the brief, but the required complete checkpoint state validation has a demonstrated acceptance gap.

**Quality verdict: changes required before deployment.** One Important finding; no Critical or Minor findings identified.

## Important finding

**I1 — Equal omissions in checkpoint payloads can publish READY.** `notes/b6_pilot_resume_20261020/b6_history_proof.py:408–417` checks that the model mapping is nonempty and that the outer state mapping contains the expected filenames, then compares both final artifacts. The inventory check at lines 235–245 likewise checks files, not their contents. The reused `probe.checkpoint` (`notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/probe.py:242–251`) unconditionally constructs those outer filename keys and canonicalizes whatever each file contains; it does not validate optimizer/scheduler/RNG schemas. Thus the outer state-key check cannot establish complete state. The model check also cannot detect a key omitted from both final checkpoints, even when snapshot400 still contains it.

This contradicts the brief's “complete canonical model/optimizer/scheduler/all four rank RNG states” and “all expected rows and state keys … no model-key omission” requirements. The host receipt predicate cannot repair the gap because it consumes the same equality claims and digests.

**Focused reproduction:** imported the existing `B6ProofTests` fixture with `/home/wyim/miniconda3/envs/pdb/bin/python`, `PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps`, and `PYTHONDONTWRITEBYTECODE=1`; created one small fixture in a temporary directory; replaced all six non-model state files in both `continuous` and `resumed` with `torch.save({})`; rewrote both final model files to retain `weight` while omitting the fixture's `buffer`; left snapshot400 intact; called the existing fixture's `publish(root)`. Exit 0, output:

```text
Published: b6_history_validation_ready
Both final checkpoints have empty optimizer/scheduler/all four RNG dicts and omit snapshot model buffer
```

**Required correction:** validate the loaded checkpoint structure before canonical hashing/comparison: nonempty optimizer state and parameter groups with required AdamW moments, scheduler fields, complete Python/NumPy/CPU/CUDA RNG contents for every rank, and model key/shape/dtype completeness against an independently established model schema. Bind the schema across snapshot400 and both final states. Add focused rejection tests for equal omissions, including empty payload dictionaries and a model key removed from both final artifacts. The current positive fixture uses the same generic `{'state': tensor}` payload for optimizer, scheduler and RNG files; replace these with small structurally valid artifacts so the tests exercise the required contract.

## Strengths

- The adapter keeps source198ce14 untouched, preserves the full4099 target and source-loaded LR schedule, and bounds only diagnostic execution through `stop_at_step=410`.
- The fresh reservation, separate `accelerate launch` invocations, source checkpoint publication hook and unchanged clone semantics implement the intended fresh0→410/snapshot400/new-process400→410 sequence.
- Public `CompileHistory.load`, null bootstrap, observed-only entries, exact positions, matching identities/fingerprints and nonzero compiler events provide strong history checks without importing C2-only witnesses.
- Exact per-rank trace row sets, finite losses, priming separation, input/context/shape/RNG/gradient comparisons and actual evidence/chain observations are explicit. The reviewed inherited trace reader binds snapshot bytes to the recorded publication event.
- Receipt publication recomputes current comparison artifacts and binds raw file digests in addition to canonical values. The shared host contract is narrow, stdlib-only and exposes its trust responsibilities.
- The wrapper preserves FX/AOT use and B6 cache paths, separates the100GiB source preflight from copy-peak fitting, and propagates command failures through `set -euo pipefail`.

## Parent verification outside this diff

- After correcting I1, obtain independent review of the fix and run the new targeted rejection tests plus the appropriate existing fixture checks. Existing reported17+5+38 passes are not evidence for this omitted-state case.
- Confirm deployed source/helper manifests cover the actual immutable releases and unchanged C2 dependencies, and independently pin image/config/bindings/input identities. The helper records actual allocated provenance; remote staging and those expected pins are outside this review.
- Run the actual four-rank NGC26.08 proof and inspect fresh verification, source and harness tests, live quota/copy-peak records, compiler fingerprint, exact comparison and receipt. Local CPU fixtures cannot establish GPU equality, elapsed time, cache behavior or live storage capacity.
- Check the parent controller's independent-file construction of `validate_ready(..., expected)` and its acceptance of current snapshot/candidate metadata, history and full-state provenance. That controller was outside this diff and was not reviewed here.

No additional approval requirement is introduced by this review.

## I1 fix rereview — e2bee04

Reviewed the supplied `19d319d..e2bee04` diff and appended implementation report, scoped to I1 and regressions introduced by its correction. **I1 is closed. Updated spec verdict: pass for the reviewed implementation. Updated quality verdict: pass; no remaining Critical, Important or Minor findings identified in this scoped review.** These verdicts supersede the changes-required verdicts above; actual allocated execution and parent deployment/controller checks remain outstanding.

The correction establishes the missing independent baseline. `b6_history_proof.py:268` captures model keys/shapes/dtypes, serialized aliases, optimizer parameters/groups and scheduler keys from actual prepared objects. The integration at line 1011 uses the same `_models[0]` and `get_state_dict(..., unwrap=False)` path as checkpoint saving, before loading or training. Each of eight phase/rank schema files is bound to its prepared trace record, and all canonical schemas must agree (`prepared_schema`, line 320). `complete_checkpoint`, line 348, validates snapshot400 and both final states against this baseline, including real optimizer/scheduler/RNG payload structure. The comparison also requires every optimizer slot initialized at snapshot400 to persist into both final checkpoints.

The added reuse of unchanged `production_link.py` is confined to `load_weights`, `finite` and `compatibility_model`; inspection found no invocation of its C2-specific checkpoint policy. The new dependency is required by both manifest verification and the batch dependency check. Model canonicalization retains the previous exact-digest representation. Scheduler validation correctly avoids assuming that its counters equal the global optimizer step.

**Independent focused verification:** reran the original review mutation against the fixed helper using the real small fixture, without rerunning either suite. In the first case both final checkpoints had all six state files replaced with empty dictionaries; in the second those files were restored and both final model files omitted the persistent buffer. Exit 0 with the following expected rejections, and `validation_receipt.json` absent after each case:

```text
Equal empty six-state payloads REJECTED: empty checkpoint state: optimizer.bin
Equal model-buffer omission REJECTED: model keys/shape/dtype differ from independently prepared schema
```

The command used the same Python/dependency paths as the original focused reproduction, with `PYTHONDONTWRITEBYTECODE=1`. Author-reported broader coverage is now22 proof tests and6 contract tests; those suites were not rerun during this scoped review. Inspection of the local Accelerate API confirmed that schema capture reads the saved model state without unwrapping and that model registration precedes GPU DDP wrapping. The parent's separate NGC1.15 API and actual RNG-layout checks remain the evidence for the target runtime; local inspection is not claimed as an NGC execution.

**Shared host contract:** `b6_receipt_contract.py:93` and `:170` now require the prepared-schema structure, canonical schema digest, all eight raw schema-file digests, and complete contracts for all three checkpoints. They reject model/schema disagreement, missing optimizer moments or snapshot-initialized slots, missing scheduler fields, missing rank RNG keys and wrong CUDA device counts. The ready receipt and independently constructed `expected` envelope remain unchanged because the comparison digest binds these new report fields.

For the future controller, retain and independently hash all eight `prepared_schema_rankN.json` files; compare their raw digests with `prepared_schema_file_sha256`, their canonical contents with `prepared_schema`/`prepared_schema_sha256`, and the corresponding prepared trace bindings. The pure host predicate validates the report's consistency; it does not itself read or establish those files. Also stage the unchanged `c2/production_link.py` dependency in the new B6 helper manifest. These are integration requirements for the parent-owned controller/staging, not unresolved defects in this fix.
