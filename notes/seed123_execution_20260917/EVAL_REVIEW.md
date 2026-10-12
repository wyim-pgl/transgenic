# Independent evaluation review — 2026-09-17

**Verdict: Important — one reproducible identity-validation gap in the standalone scorer.**

Scope: readonly inspection of `/tmp/mvp-seed123-integration-20260917`, base `9603e20`, after all three writers finished. Reviewed evaluation-owned decoder, runtime, producer, scorer, bootstrap and relevant tests plus `tasknotes/seed123_eval/{REPORT,INTERFACE}.md`. Training and panel-helper internals were not independently re-audited; their cross-interface calls were inspected. No source/index/HEAD edits, subagents, remote actions or jobs. Only this report and its adjacent bounded proof script were created in main's notes.

## Important finding

`revision/scripts/mvp/mvp_score.py:86-94,149-150`: development scorer validates explicit profile, role and seed but ignores an explicitly supplied `run_id`, both at payload top level and in `metadata`. It then publishes `run_id='mvp_seed123'` unconditionally. A caller declaring `run_id='b5_seed123'` therefore gets output relabeled as fine-tuned MVP rather than rejection. This is the exact baseline-versus-fine-tuned identity distinction required even though both numeric seeds are 123.

Reproduction (no implementation monkeypatches):

```sh
cd /tmp/mvp-seed123-integration-20260917
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 timeout --kill-after=5s 90s /tmp/mvp-final-source-env/bin/python \
 /mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/eval_review_proof.py
```

Observed:

```text
ACCEPTED_CONFLICT top input=b5_seed123 output=mvp_seed123
ACCEPTED_CONFLICT metadata input=b5_seed123 output=mvp_seed123
```

The valid fixture retains `pairs={'mvp_seed123': ...}`; only conflicting explicit metadata is injected. Expected: reject contradictory declared identity, like existing role/profile/seed checks. Suggested minimal correction: if `run_id` is present in either validated record, require exactly `mvp_seed123`; add both negative tests. This does **not** bypass `verify_run_inventory` or establish verified production provenance: standalone results still say UNVERIFIED and decision remains null. The finding concerns silent contradictory identity normalization at a public scorer boundary, not a claim that the producer accepts a swapped baseline checkpoint.

## Independently completed validation

All commands used Python 3.12 `/tmp/mvp-final-source-env`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=$PWD/src`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, pytest `-p no:cacheprovider`, and GNU timeout with kill-after cleanup.

1. `timeout --kill-after=10s 360s ... -m pytest -q -p no:cacheprovider tests/test_mvp_seed123_eval.py tests/test_mvp_scoring.py tests/test_mvp_score_producer.py tests/test_mvp_decoder_runtime.py --tb=short`
   **120 passed, 1 skipped in 47.93s.**
2. `timeout --kill-after=10s 360s ... -m pytest -q -p no:cacheprovider tests/test_mvp_decoder_integration.py -k full_publication --tb=short`
   **2 passed, 22 deselected in 176.33s.** Both legacy350 and development150 fixtures completed. Rehashed readiness mutations of profile, bool seed, role, failed resource gate, SIF and consumed-code hash were rejected in the development fixture.
3. Adjacent `eval_review_proof.py`: completed successfully. Also rejects profile values `True`, `123`, `[]`, `{}`, `None`; proves equal-tile primary CE rather than token pooling (5.0) and separate equal-tile composite diagnostic (10.0) with unequal denominators.
4. `git diff --check`: passed.

The prior extended reference/producer invocation that timed out remains **not green**; this review does not upgrade it. No whole-suite/native-crash-prone invocation was attempted.

## Contract observations

- Decoder dispatch and producer inventory distinguish `mvp_seed123` / `b5_seed123`; development is explicit, exactly 150 predictions (100 paired + 50 baseline), while legacy350 remains unchanged. Seed/fraction booleans, swapped role/checkpoint, missing/extra predictions, aliases, hardlinks and repeated stems are covered by passing negatives.
- `mvp_produce_counts.py:218-290` binds actual finalizer metadata, distinct final/baseline hashes, independent baseline pin equal to initialization audit, and training readiness/fixed recipe/SIF/code/source identity. No identity_scope=test exemption was added. Decoder checks before model loading and re-verifies before transport publication (`mvp_decode_panel.py:97-104,178-198`). Producer retains checked inputs through publication (`mvp_produce_counts.py:600-723`).
- Panel compatibility receipt is optional and transported/pinned; decoder calls the actual live panel loader, not a receipt-only shortcut. Helper internals remain the other reviewer's scope. Runtime `DECODER_ONLY` addition does not override a probe-consumed hash (`mvp_decode_runtime.py:132-137`).
- Missing or partial structural counts remain unknown/unassessable and suppress P/R; no zero fabrication. Development output decision is null, descriptive, with no three-seed verification flags. Existing hardcoded legacy pending-policy text is not a new authorization requirement: development producer explicitly records approved source-OR/missing-not-negative policy.
- **Weighting clarification, not classified as a new regression:** structural F1 uses pooled count ratios across loci, not an arithmetic mean of per-locus F1 (`mvp_score.py:65-73,161-167`); locus bootstrap resamples loci then recomputes that pooled statistic. Independent unequal-size example returns 1%, whereas arithmetic mean of locus F1 would be 50%. This behavior is inherited. If controller's intended “per-locus mean” requirement means macro-F1, that requirement is not met and needs a separate explicit metric-contract correction; this review does not silently certify macro-F1.
- Integration fixtures substitute GPU/runtime generation and GFFCompare with CPU stand-ins, and substitute synthetic panel authority pins explicitly. Their passing status is not actual GPU, accepted-panel or scientific certification.

## Outstanding operational boundaries

Actual accepted ONT structural-source producer linkage remains missing and disclosed; no measured structural P/R or GO/STOP certification follows. Integrated new-source freeze and reviewed `PROBE_CODE` refresh plus genuine runtime receipts remain controller follow-up. Legacy hardcoded probe pins are not grounds for a bypass. Training approval and the other panel review are separate from this evaluation verdict.

Advisor planning consultation was attempted once but unavailable (client/model-version error); no advisor approval is claimed. Graft was queried first; no graph rebuild/index mutation was performed. Reported graft context savings: approximately 1,121,725 tokens.
