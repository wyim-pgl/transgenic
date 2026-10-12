**REJECT** — two HIGH findings block approval.

1. **HIGH — Public builders bypass required input verification.**  
   [b6_build_obs_cache.py:68](/tmp/b6-wp3-tracks/revision/scripts/evidence/b6_build_obs_cache.py:68), [b6_build_tracks.py:90](/tmp/b6-wp3-tracks/revision/scripts/evidence/b6_build_tracks.py:90).  
   The cache builder checks snapshot bytes and family, then records any supplied reference FASTA’s hash without comparing it with C0 `inputs.reference`. The dose-response CLI never calls `verify_inputs`; evaluation checks that reference against the cache’s self-recorded hash. Consequently, a different assembly with compatible chromosome names and lengths can supply DNA while evidence retains the original assembly’s coordinates. This path also skips leakage/exhaustiveness and zero-QC upstream validation, contrary to plan §8.  
   **Minimal fix:** require the typed upstream bundle before cache publication and dose consumption; verify reference identity against C0 provenance and validate zero-list population/upstream bindings.

2. **HIGH — WP2’s actual freeze format is incompatible with WP3’s verifier.**  
   [b6_bindings.py:209](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:209).  
   WP3 compares each C0 `interval_manifest_sha256` string with the entire freeze value. However, [WP2’s producer:540](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_corpus_repair.py:540) writes `{species: sha256}`. WP4 passes that freeze unchanged to WP3. Thus correctly produced integrated inputs cannot start training. An in-memory reproduction rejected the WP2-shaped freeze with `interval_manifest_sha256 != freeze`.  
   **Minimal fix:** require the species-indexed mapping, check complete species coverage, and compare each provenance entry with its species’ hash. Exercise the real producer format in the integration test.

3. **MEDIUM — Legacy-C0 revalidation is an unbound assertion.**  
   [b6_bindings.py:220](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:220).  
   A missing C0 manifest hash is accepted when `revalidated[(species, source)]` equals the freeze hash. That dictionary binds neither the C0 snapshot nor its recorded leakage-interval hash. Reusing a result from another C0 therefore passes without demonstrating §8.3’s complete interval rerun and projection checks. The existing test explicitly accepts a manually constructed dictionary.  
   **Minimal fix:** perform revalidation inside the verifier, or require a verified result binding the exact C0, legacy interval hash, recorded inputs and interval manifest.

4. **MEDIUM — Required leakage and exhaustiveness records can be absent without failure.**  
   [b6_bindings.py:153](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:153), [b6_bindings.py:165](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:165).  
   `_leakage_gate` accepts an empty `c0_qc` table. `_exhaustiveness_gate` accepts `{"Athaliana":{"by_source":{}}}`, because both loops become vacuous. Both cases were reproduced in memory. An incomplete snapshot can therefore pass these checks without establishing zero leakage or the A49.2 denominators.  
   **Minimal fix:** derive expected species/source coverage from the verified provenance and merge manifest, require every corresponding QC/audit record, and validate integer counters before recomputing the gates.

5. **MEDIUM — Protein thinning uses different hash bytes from approved D7.**  
   [b6_tracks.py:31](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_tracks.py:31), [b6_obs_cache.py:157](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_obs_cache.py:157).  
   D7 specifies ranking over protein `query_id`. Extraction discards that ID after deriving the organism; thinning instead hashes C0’s SHA256-derived `molecule_id`. For `Athaliana`, seed `20261037`, query `29760:4`, the prescribed rank is approximately `0.253606`, while the implemented key gives `0.595591`. At fraction 0.5, the required draw retains the query and this implementation drops it.  
   **Minimal fix:** preserve `query_id` for protein ranking, retain molecule identity for exclusion/QC, and update the spec hash and literal-key regression.

Verified correct:

- All **92 protected files** remain byte-identical to the pinned commit.
- **15 focused tests passed** with capture/cache writes disabled; no full suite ran and no files were modified.
- Strand-aware donor/acceptor coordinates, RC permutations, chain RC, and `(padding + position) // 6` match the contract.
- Chain fill ordering, whole-chain rejection, 63 real slots, deterministic slot hashes, and C3 donor checks are consistent with the plan.
- Labels come from corpus GSF; no `evlabel` consumption was found. Scoring rows are filtered during extraction, with a runtime admission guard.
- WP4’s spec and shared-fixture copies are byte-identical. The writer’s dummy-slot, greedy-fill, mask-at-N and lazy torch-conversion readings are supported.