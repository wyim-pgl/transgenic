**REJECT** — one new HIGH integration mismatch blocks approval.

A. Round‑1 findings:

- **H1 — PARTIALLY.** Both public builders now verify upstream inputs: [cache builder:96](/tmp/b6-wp3-tracks/revision/scripts/evidence/b6_build_obs_cache.py:96), [dose builder:104](/tmp/b6-wp3-tracks/revision/scripts/evidence/b6_build_tracks.py:104). Reference identity is checked against C0 provenance at [b6_bindings.py:339](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:339). **Residual MEDIUM:** [zero-list validation:291](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:291) accepts any nonempty upstream mapping containing hashes of that species’ merged inputs. An in-memory reproduction accepted `{"protein_c0": EST_HASH}` without `est_c0`; the actual protein C0 identity was never checked. Require both producer keys and validate each against its corresponding species/source.

- **H2 — FIXED.** [b6_bindings.py:268](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:268) requires a species-indexed mapping with complete coverage; [line 357](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:357) compares each provenance entry with its species’ hash. This matches [WP2’s actual freeze producer:540](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_corpus_repair.py:540).

- **M3 — FIXED.** [b6_bindings.py:225](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:225) rehashes recorded inputs and reruns the leakage model with the recorded seed. [Line 359](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:359) verifies canonical manifest bytes, sidecar, complete legacy hash and projection, then binds the result to the evidence SHA. The unbound dictionary acceptance is removed.

- **M4 — FIXED.** [b6_bindings.py:165](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:165) establishes coverage from provenance and merge records. Every covered pair requires leakage QC; exhaustiveness requires complete audited-source coverage and nonnegative integer counters. In-memory probes rejected missing records, nonzero leakage, empty source maps, and fractional/negative counters.

- **M5 — FIXED.** Extraction preserves `query_id` at [b6_obs_cache.py:161](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_obs_cache.py:161); thinning uses it at [b6_tracks.py:32](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_tracks.py:32). The literal query ranks `0.2536060591` and is retained at fraction 0.5. Molecule identity remains available for exclusion/QC.

B. **New HIGH — the corrected augmentation spec is incompatible with the current WP4 consumer.**

[WP3’s updated spec:335](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_track_spec.py:335) produces augmentation SHA `42cc22a0…`, while [WP4’s copy:335](/tmp/b6-wp4-trainer/src/transgenic/evidence/b6_track_spec.py:335) still produces `93aa3102…`. [WP4 runtime:272](/tmp/b6-wp4-trainer/src/transgenic/training/b6_runtime.py:272) rejects the mismatch.

A controlled reproduction passed an otherwise valid binding record with WP4’s spec, then failed after changing only `spec` to WP3’s current values. Thus the supplied consumer rejects identities generated under the corrected thinning contract. Synchronize WP4’s shared spec and verify cross-checkout acceptance while retaining the hash check.

No additional CRITICAL/HIGH findings emerged in the reviewed fixes. Three focused tests passed, alongside in-memory probes. No full suite ran; no files were modified.