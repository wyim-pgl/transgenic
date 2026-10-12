**REJECT** — three HIGH findings.

Reviewed tracked changes against `daf8436`, untracked implementation files, the specified plan/protocol sections, and adjacent worktree interfaces. No files modified; no full suite run.

1. **HIGH — Valid-only protein phase leaks into train-tile evlabel.**  
   [b6_evlabel.py:432](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:432) collects protein phase without applying `train_tiles_excluded`; [line 707](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:707) projects it onto every train/valid tile. A protein with an unmasked train placement and another placement triggering validation exclusion therefore contributes forbidden train phase labels. An in-memory reproduction produced **empty train cells but nonempty train phase** for that molecule. This violates plan §5’s population restriction.  
   **Minimal fix:** collect and project phase per scope, applying the same `in_scope()` filter as CDS cells; independently validate phase eligibility.

2. **HIGH — Evlabel verification does not bind C0 exclusions to the repaired corpus.**  
   [b6_evlabel.py:138](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:138) checks individual file hashes and supplement→merged identity, but never compares merged provenance against freeze `base_sha256` or `interval_manifest_sha256`. It also omits §8’s required C0 schema, leakage-QC and exhaustiveness gates. A correctly hashed merged DB from another split/exclusion build can consequently supply labels using inappropriate molecule eligibility. The verifier accepted a synthetic freeze containing **only `out_sha256`**.  
   **Minimal fix:** enforce the typed training binding contract before building labels, including required keys, per-species exclusion identity, QC/exhaustiveness and snapshot checks.

3. **HIGH — WP2’s freeze cannot pass the current WP3/WP4 consumer.**  
   [b6_corpus_repair.py:540](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_corpus_repair.py:540) publishes `interval_manifest_sha256` as `{species: sha}`. [WP3 b6_bindings.py:218](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:218) compares each C0’s scalar SHA directly against that entire mapping. Matching producer/consumer inputs therefore raise `BindingError`, including for one species; WP4 calls this verifier before training.  
   **Minimal fix:** agree on the species→SHA contract and compare against the entry for each provenance species, including legacy revalidation. Add a test consuming an actual WP2-produced freeze.

4. **MEDIUM — Omitting protein supplements silently drops required label families.**  
   [b6_build_evlabel.py:30](/tmp/b6-wp2-corpus/revision/scripts/evidence/b6_build_evlabel.py:30) makes supplements optional; [b6_evlabel.py:669](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:669) substitutes an empty list for missing species. Even when merged C0 contains accepted proteins, publication succeeds without their CDS/phase labels or protein-supported A50/A50.1 UTR labels.  
   **Minimal fix:** require a bound supplement for every species with applicable accepted protein placements; represent genuinely absent protein evidence explicitly.

5. **MEDIUM — Split terminal codons are evaluated using intronic bases.**  
   [b6_protein_placement.py:159](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_protein_placement.py:159) reads three contiguous genomic bases for the start and inside-block stop checks. Terminal codons spanning CDS blocks are consequently misclassified. Verified example: blocks `[(0,1),(7,15)]` splice to `ATGAAATAA`, but the function returns `start_codon_ok=False`, incorrectly suppressing valid 5′UTR labels.  
   **Minimal fix:** inspect transcript-oriented spliced CDS for the start and inside-block stop; retain explicit handling of the exterior-stop convention.

6. **MEDIUM — Legitimately empty admitted-chain populations block completeness construction.**  
   [b6_completeness.py:256](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_completeness.py:256) infers population exclusively from data rows and rejects header-only files. [Line 436](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_completeness.py:436) requires all four population files. A training species without scoring datasets cannot supply its legitimate empty score-chain set normally.  
   **Minimal fix:** encode population independently of chain rows and permit a hash-bound empty set.

7. **MEDIUM — The skipped trainer test becomes a guaranteed integration failure.**  
   [test_b6_evlabel.py:85](/tmp/b6-wp2-corpus/tests/test_b6_evlabel.py:85) requires the literal text `refuse_if_evlabel` in the trainer. The supplied WP4 trainer lacks it and implements refusal through binding/runtime checks instead. The current skip conceals this mismatch; it does not demonstrate an A50.2 leak.  
   **Minimal fix:** test rejection behavior through the agreed trainer interface rather than require a function name.

The writer’s explicit boundary deferral and disabled unapproved adapters follow the plan. Content-hash rebuild determinism is a reasonable storage workaround, but it does **not** establish the expressly requested byte-identical rebuild acceptance. The real 198,828-row gate and nine-species stop-convention measurement remain unverified; fixture results cannot close those gates.

Verified correct:

- All **92 protected files** are byte-identical to the pinned `681c5a0` versions; the policy-protected original trainer is unchanged.
- The roles manifest verifies successfully: **28 rows**, with the specified scoring/input and scope checks.
- Repair uses explicit edge-crossing seeds, clipped component geometry, label closure and forward-derived RC twins; row-loss events trigger refusal.
- A30-only manifest rows do not become C0 exclusions; the leakage refactor preserves historical RNG seeds and draw order.
- Negative-strand donor/acceptor cells retain their identities and correct half-open coordinates.
- Non-phase evlabel cells apply train/valid eligibility; completeness excludes scoring molecules from training populations and A-CCS23 from TES support.