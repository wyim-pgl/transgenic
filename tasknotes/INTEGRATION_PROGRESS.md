# B6 integration: post-merge fixture work (WP2 M7, WP4 M8)

Worktree /tmp/b6-integration, branch b6-integration, base HEAD b16e5af. Nothing committed. No protected file edited.
`src/transgenic/evidence/b6_track_spec.py` is unchanged (sha256 516835ca…8e43).

## Integration review 1 MEDIUM: training caches now enforce their corpus and freeze bindings

Source: notes/a46_approval/B6_INTEGRATION_REVIEW1_20260924.md (APPROVE_WITH_FIXES; M7 and M8 FIXED).

`src/transgenic/evidence/b6_bindings.py` `verify_cache`
- For a training-family cache, the manifest must bind `corpus_sha256` and `freeze_sha256` as 64-hex shas.
  Otherwise the error is "training cache binds no ...".
- Each binding must equal the verified upstream: `corpus_sha256 == ident["corpus_sha256"]` and
  `freeze_sha256 == ident["provenance"]["freeze"]`, the sha of the freeze file bytes. Otherwise the error is
  "cache was built against a different B6 corpus" or "... different corpus freeze".
- Eval_only caches keep their existing required set, with no corpus or freeze.
- `b6_build_obs_cache.py` already recorded both bindings and re-verifies its published cache, so it is unchanged.

`tests/b6_fixtures.py` `make_world` now writes the freeze file before the cache and binds `freeze_sha256` next to
`corpus_sha256`, the same shape the CLI builder produces. freeze.json keeps the serialization of `freeze_file`, so
`verify(w)` copies of an unchanged freeze stay bound.

`tests/test_b6_bindings.py`
- `test_training_cache_binds_the_verified_corpus_and_freeze`: the world cache binds both, and a republished
  unchanged cache passes.
- `test_training_cache_with_a_missing_or_mismatched_corpus_or_freeze_binding_fails`, 5 cases: corpus missing,
  mismatched, or not a sha; freeze missing or mismatched.
- `test_stale_cache_refused_with_another_valid_freeze_or_corpus`: this is the reviewer's case. Upstream verifies
  with a re-serialized freeze, and with a different corpus plus its own valid freeze, but the stale cache is refused.
- `legacy_world` replaces the freeze, so it now republishes its cache bound to that freeze.

`tests/test_b6_builders.py`: the dose-response test passes `world.freeze_path` instead of a re-serialized copy,
because the cache is now correctly refused against a copy with different bytes.

Verified against the committed pre-fix `b6_bindings.py`, swapped in temporarily and then restored, byte-identical
per cmp: the 6 new negative cases fail and the positive case passes.

Results after this fix:

    torch-free explicit list (F=$(ls tests/test_b6_*.py | grep -v -E "test_b6_(model|integration_contract|trainer|bench|ddp)\.py") + MVP pin):
        387 passed in 1967.70s (0:32:47)                          # 0 skipped; +7 new bindings tests
    torch venv (5 WP4 files, test_mvp_final_source_integration.py, test_b6_evlabel.py, test_b6_bindings.py):
        142 passed, 91 subtests passed in 576.28s (0:09:36)       # 0 skipped; +7 new bindings tests
    CP4 --data wp3-fixtures, 2 CPU ranks, aot_eager, accumulation 2, n-pilot 3, stop-at 1:
        {"status": "PASS", "failures": []}  2 twin pairs, label RC asserted, 3 links / 1 record, resume byte-identical

The sections below are the earlier M7/M8 round.

## WP2 M7: the WP3 verifier case no longer skips

`tests/test_b6_evlabel.py`
- Removed the `_wp3_bindings` / `WP3_REASON` skip machinery. Nothing in this case can skip now.
- New module fixture `wp3_cache` publishes a real WP3 training-family observation cache for the evlabel world through
  WP3's own CLI, `b6_build_obs_cache.main`. That call runs verify_upstream(stage="build"), extract_rows, write_cache
  and verify_cache. Header-only A48 zero lists for all three populations have QC `upstream` bound to the merged
  EST and protein C0 shas from `merge_manifest`. The reference is the world FASTA, and the freeze is passed as a FILE path.
  Any import or build failure is an error, not a skip.
- `test_wp3_training_verifier_rejects_an_evlabel_db_after_a_valid_binding`: `b6_bindings.verify_inputs` first accepts
  the valid binding. The test checks family training, evlabel "absent", corpus sha, cache manifest sha, and that
  provenance["freeze"] equals the sha of the freeze file. It then rejects the evlabel DB as evidence with
  `c0_schema.SchemaError` "not a C0 database", and as corpus with `BindingError` "... != freeze out_sha256".
  The old test used bare `Exception`.

## WP4 M8: real fixture labels and label RC in `--data wp3-fixtures`

`tests/b6_fixtures.py`
- `FIXTURE_GENES` holds three complete one-transcript genes:
  - gE1 '+' on e1's exons in tile [0,600).
  - gO3 '+' on o3's exons in tile [1800,2400).
  - gC1 '-' in tile [3000,3600).
  CDS phases are consistent with CDS lengths.
- `gsf_contract()` returns `b6_corpus_repair._gc()`, the same gsf_contract the corpus builder uses, loaded without
  the torch-importing `transgenic.utils` package.
- `tile_labels()` gives each forward tile `window_to_gsf_v3` of the genes inside it, or EMPTY_LABEL. Each RC twin
  gets `reverse_complement_v3(forward, TILE)`, and every label passes `check_caps_v3`. `write_corpus` now writes these
  instead of `gsf<rn>`.
- Twin pairs rn 1/2 and 7/8 carry a '+' gene and a '-' gene. rn 5 carries gO3. rn 3, 4 and 6 are `<empty>`.

`tests/test_b6_dataset.py`: new `test_fixture_rc_twins_carry_reverse_complement_v3_labels`.
- The corpus labels equal `tile_labels()`.
- The RC label is reverse_complement_v3 of the forward label in both directions and differs from it.
- The labels the dataset serves for each RC twin equal the tokens of the RC label.
- Both strands are covered.

`revision/scripts/b6/integration_4rank.py` (`wp3_fixture_data`)
- For every twin pair, the driver asserts that both labels are valid v3 GSF, that the forward label is non-empty,
  and that the RC label is reverse_complement_v3 of the forward label in both directions. It also asserts that the
  RC labels the B6Dataset serves equal the tokenizer output of reverse_complement_v3(forward).
- The report carries label_genes, label_fw and label_rc per pair. `labels_note` now states that label RC is asserted.
- Fixture mode now tokenizes labels with WP3's production `b6_dataset.gff_label_tokenizer("v3")`, the same ids as
  synthetic mode's `wp4_synthetic.gsf_token_ids`. The fixture's one-token-per-character `gsf_tokens` would make
  labels of about 135 tokens. That overran the tiny model's 64 decoder positions and caused an IndexError in
  `embed_positions` on the first run. The real tokenizer gives at most 47 tokens with max id 289, under vocab 290.
  WP3's own tests still use `gsf_tokens`.
- Docstring: `--data` is required and there is no synthetic default. The label RC is now listed among fixture-mode checks.

## Results

Torch-free venv. The lead's exact command stops at collection with 5 errors on the 5 `--ignore`d files, because the
shell glob names them explicitly and pytest 8.4.2 still collects them. I ran the same file set with the glob pre-filtered:

    cd /tmp/b6-integration && FILES=$(ls tests/test_b6_*.py | grep -v -E "test_b6_(model|integration_contract|trainer|bench|ddp)\.py$")
    PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp /tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q -p no:cacheprovider -rs $FILES tests/test_mvp_final_source_integration.py::test_reviewed_a_implementation_bytes_unchanged
    380 passed in 2406.49s (0:40:06)          # 0 skipped; the M7 case passes

Torch venv (scratchpad/b6venv): the 5 WP4 files, test_mvp_final_source_integration.py, test_b6_evlabel.py and test_b6_bindings.py:

    135 passed, 91 subtests passed in 543.79s (0:09:03)   # 0 skipped (the 3 former WP3-dependent WP4 skips now run)

CP4 driver on 2 CPU ranks with `CUDA_VISIBLE_DEVICES=` set. Without it, a visible GPU makes accelerate pick NCCL and
both ranks land on one device ("Duplicate GPU detected"). Launched with
`python -m torch.distributed.run --nproc_per_node=2 revision/scripts/b6/integration_4rank.py --model tiny --backend aot_eager --mixed-precision no`:

    --data wp3-fixtures --accumulation 2 --n-pilot 3 --stop-at 1   {"status": "PASS", "failures": []}
        2 twin pairs (0-600 rn 1/2, 3000-3600 rn 7/8), label RC asserted, 3 links re-verified with 1 distinct
        record_sha256, resume byte-identical, generation parity equal on both ranks (beams 1 and 2)
    --data synthetic --tier 48 --stop-at 2                          {"status": "PASS", "failures": []}
    --data synthetic --tier 48 --stop-at 4                          {"status": "PASS", "failures": []}
        28 RC pairs checked, 12 reachable updates, resume byte-identical

Negative check: a 1-rank fixture run with rn 2's label replaced by the forward label fails before training with
`AssertionError: Athaliana:Chr1:0-600: RC twin label != reverse_complement_v3(forward label)`.

## Unresolved / notes

- Synthetic mode used `--tier 48`. Writer C's notes give the settings but not the literal command, and the default
  tier of 30720 is the GPU size.
- A lead torch-free pytest (pid 612420, scratchpad/integration_tests.log) started after my edits began. Its results
  may mix file states, so rerun it.
- Still open from review 3 M8, and outside this task: four-rank GPU acceptance through the sbatch has not been run.
