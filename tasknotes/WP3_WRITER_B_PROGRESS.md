# WP3 Writer B progress (branch b6-wp3-tracks, base daf8436)

## CP0 — DONE
- `src/transgenic/evidence/b6_track_spec.py` final; sha in `tasknotes/TRACK_SPEC_READY.txt`
  (6f5f1d6d1cfb983a2422070b184f4be10fa6f9a978edc3c9fd6c96c726b59712).
  TRACK_SPEC_SHA eed27580…, CHAIN_SPEC_SHA e7d35fdd…, AUG_SPEC_SHA 93aa3102… (AUG_SPEC id b6-aug-uniform-blake2b-v2).
- `tests/b6_fixtures.py`: `stub_batch()` (numpy) / `stub_batch_torch()` satisfying §7.7; validator `TS.validate_batch`.
- `tests/test_b6_track_spec.py`: 10 passed.

## Readings recorded (fail-closed choices where the plan is silent)
- R1 Chains occupy slots 1..63, so at most 63 chains per tile (rule 6a "N_chain = 64" counts the dummy slot 0).
- R2 `AUG_SPEC_SHA` is sha256 of the canonical AUG_SPEC JSON, whose `id` is `b6-aug-uniform-blake2b-v2` (plan §8.5 wording read as the spec id).
- R3 `attention_mask` dtype = bool (B5 collate: `input_ids != pad_token_id`).
- R4 Fill: after a chain that does not fit is dropped whole, filling continues with the next chain (greedy).
- R5 Donor/acceptor counts are not clipped (rule 6 clips coverage only); N bases are not zeroed (plan §7.1 mask is 1 at N).
- R6 torch is absent from the test venv: collate/dataset produce numpy; torch conversion is optional and lazy.

## Next
b6_aug (draws) → b6_context → b6_obs_cache + builder → b6_tracks → b6_chains → b6_bindings → dataset/collate → zero-list --population → eval C0 → dose-response builder → throughput bench.

## Components done (2026-09-24)
- `b6_aug.py` (DrawSpec/resolve/resolve_fixed/resolve_off, arm transforms, C3 donor map) — tests in test_b6_tracks.py.
- `b6_obs_cache.py` (extract_rows from merged C0, parquet shards, manifest+sidecar, ObsCache loader with interval index).
- `b6_context.py` (Context.admit, corpus_context, donor_context (C3), eval_context, Reference, Panel, WindowManifest).
- `b6_tracks.py` (rasterise, LeakageError), `b6_chains.py` (build_chains, split, fill, chain-rc-v1).
- `datasets/b6_dataset.py` (B6Dataset, B6EpochSampler, b6_collate_fn[_torch]); imported by full path (package __init__ imports torch).
- `b6_bindings.py` (verify_inputs typed families, interval-manifest hash, §8.3 revalidation steps a-c).
- `revision/scripts/evidence/b6_build_eval_c0.py` (b6_build_c0 ingestion + null leakage model, policy eval_only_no_leakage_model).
- Tests: test_b6_track_spec (10), test_b6_tracks (25), test_b6_dataset (5), test_b6_bindings (13), test_b6_eval_path (5) — all pass.

## More readings
- R7 4(c): a junction whose donor or acceptor coordinate is in excluded_junction_coordinate is dropped from BOTH donor/acceptor channels (coverage kept) and splits chains.
- R8 Eval admission, training family (A. thaliana): b6_input dataset/run AND molecule_state excluded_heldout AND an accepted primary placement overlapping the panel locus. Eval family (Z. mays): b6_input only (states carry no meaning without a leakage model). Protein is never b6_input, so channel 24 is 0 at evaluation (D5 "low eval coverage of 24-30").
- R9 Route-B phase rows of a train_track EST whose HSP is leakage_excluded (protein query excluded) are dropped at cache build.
- R10 Chain slot hash aug_seed = the draw's aug_seed (training aug_seed; 20260914+d in dose-response), epoch carried by the resolved draw.
- R11 gsf_contract imports torch (postprocess); `a27-tiles-v1` loads tile_windows/WINDOW_TIERS from the pinned source via ast when torch is absent (no copy).

## Final (2026-09-24)
- Also done: `--population` in b6_routea_agreement.py (+ QC `population`, `upstream` shas), b6_build_obs_cache.py, b6_build_tracks.py (dose-response archive), revision/scripts/b6/bench_throughput.py; tests test_b6_routea_population (1), test_b6_builders (4), float16 phase RC mirror (tracks, 26 total).
- Full run: `tests/test_b6_*.py tests/test_mvp_final_source_integration.py` -> 250 passed, 2 failed. The byte pin `test_reviewed_a_implementation_bytes_unchanged` PASSES. The 2 failures import torch (tests/test_mvp_finalize.py:15); they fail identically on a clean export of base daf8436 (venv has no torch).
- b6_track_spec.py unchanged since marker (sha256 -c OK).
- Perf (synthetic dense 129,024-nt tile, 5,000 molecules, 57.5k rows): rasterise 0.85 s, build_chains 0.21 s per tile, one core. Real gate needs the real cache and Writer C's step rate.

## Fix round 1 (Codex review B6_WP3_CODE_REVIEW1_20260924.md) — DONE
- H1: b6_bindings split into verify_upstream(stage=build|consume) + verify_cache; verify_inputs = both. b6_build_obs_cache.py publishes nothing before verify_upstream(stage="build") passes (typed family keys, leakage + A49.2 gates, per-species --reference SPECIES=FASTA == C0 inputs.reference, zero-list population/species/upstream) and re-verifies the published cache. b6_build_tracks.py runs verify_inputs (new --evidence/--freeze/--corpus) before archiving; identity goes in the archive header. Cache bindings reference_sha256 is now {species: sha}; eval_context checks the window species' entry.
- H2: freeze interval_manifest_sha256 must be {species: sha} (b6_corpus_repair record shape), every covered species present, each C0 provenance entry compared with its species' hash. Fixture freeze = writer_a_freeze (exact record keys, copied by shape, module not imported).
- M3: legacy C0 without interval_manifest_sha256 is revalidated INSIDE the verifier: manifest read from freeze interval_manifests[sp] (canonical bytes, sidecar, == freeze hash); build_leakage_model rerun from the C0's recorded inputs (each re-hashed) and seed; §8.3 (a)(b)(c). The free-standing `revalidated` dict is removed. legacy_intervals_sha256 tested equal to b6_build_c0.intervals_sha256.
- M4: coverage = c0_provenance keys == merge_manifest keys; every covered (species, source) needs a c0_qc record with leakage_nonzero = 0; A49.2 needs by_source == covered audited sources with integer counters.
- M5: protein rows carry query_id (cache column); thinning ranks the query_id, molecule_id only for exclusion/QC. AUG_SPEC text changed -> AUG_SPEC_SHA 42cc22a0…; TRACK/CHAIN shas unchanged; b6_track_spec.py sha now 516835ca… (marker updated, lead told). Regression: rank(20261037, Athaliana, 29760:4) = 0.2536060591.
- Full run: tests/test_b6_*.py + MVP pin -> 260 passed, 2 failed (same two torch-import tests as before; byte pin passes).

## Rule 9 provenance map (lead follow-up for WP4) — DONE
- verify_upstream / verify_cache / verify_inputs records carry `provenance` {name: sha256}, sorted, every file re-hashed at verification; a missing file fails; plus `record_sha256` = sha256 of the canonical record (sort_keys, (",", ":"), excluding record_sha256).
- Names: `evidence.sha256` and/or `evidence.PROVENANCE.txt`; `c0:<species>/<source>.PROVENANCE.txt` (output_sha256 must equal merge_manifest input sha); `freeze`; `interval_manifest:<sp>`, `interval_manifest:<sp>.sha256` (training); `zero_list:<pop>/<sp>`, `zero_qc:<pop>/<sp>`; `cache_manifest`, `cache_manifest.sha256`.
- `freeze` argument is now the freeze FILE path (its bytes are recorded); builders pass the path.
- No protein-placement supplement is consumed by WP3, so none is listed.
- Full run: 263 passed, 2 failed (the same two torch-import MVP tests; byte pin passes). Spec file unchanged (marker 516835ca…).

## Codex round 2 residual MEDIUM (zero-list upstream) — DONE
- zero_list_check (b6_bindings.py ZERO_UPSTREAM): QC `species` must equal the list's species; `upstream` keys must be EXACTLY {est_c0, protein_c0} for a species with both merged C0s ({est_c0} when the species has no protein C0, e.g. Z. mays eval; an EST C0 is always required); each value must equal the merge-manifest sha of its OWN (species, source). merge_inputs is now keyed (species, source).
- Regression: tests/test_b6_bindings.py::test_zero_upstream_needs_both_producer_keys_each_matching_its_own_source (Codex reproduction {"protein_c0": EST hash}, est-only, swapped hashes, extra key; plus the correct pair passes) — PASS.
- Full run: tests/test_b6_*.py + MVP pin -> 264 passed, 2 failed (same two torch-import tests; byte pin passes). Spec file unchanged (516835ca…).

## Tightening (lead): protein C0 mandatory where present / for training — DONE
- zero lists: protein_c0 required whenever the merge holds a protein C0 for the species (est_c0-only only when none).
- training family: every covered species must have a merged protein C0, else BindingError ("without a merged protein C0").
- Tests: test_b6_bindings.py::test_protein_c0_present_makes_protein_c0_upstream_mandatory, ::test_training_family_species_without_protein_c0_fails, test_b6_eval_path.py::test_zmays_eval_c0_to_batch_with_no_corpus (asserts est_c0-only upstream, passes) — all PASS.
- Full run: 266 passed, 2 failed (same two torch-import MVP tests; byte pin passes). Spec unchanged (516835ca…).
