# WP2 Writer A progress (plan v4.1, notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md)

Worktree /tmp/b6-wp2-corpus, branch b6-wp2-corpus, base daf8436 (contains T = e35a5bd). No commits by Writer A.

Test command:
    cd /tmp/b6-wp2-corpus && PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp \
      /tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q -p no:cacheprovider <tests>

| component | status | tests | open issues |
|---|---|---|---|
| WP2-roles (b6_roles.py + manifest copy) | done | tests/test_b6_roles.py 12 passed | manifest copied from main repo (untracked there); not committed |
| b6-intervals-v2 manifest (leakage.py; b6_leakage_intervals --manifest-out; C0 provenance interval_manifest_sha256) | done | tests/test_b6_interval_manifest.py 8 passed; test_b6_c0_leakage 15 passed | gene prep extracted to leakage.prepare_genes (behaviour-identical); variant world tests/test_b6_repair_world.py |
| WP2 repair (src/transgenic/evidence/b6_corpus_repair.py, revision/scripts/evidence/b6_build_corpus.py, b6_validate_corpus.py) | done (fixtures) | tests/test_b6_corpus_repair.py 18 passed | DuckDB files are not byte-deterministic across rebuilds (verified: even a 3-row UPDATE differs; force_compression does not help) -> freeze records content_sha256 (canonical over all tables) and rebuild identity is tested on it; out_sha256 = file bytes stays the consumer binding. Leakage per-tile replay extracted to leakage.tile_masks/verify_tile. Real-data run pending (commands in final report). |
| WP2-prot (src/transgenic/evidence/b6_protein_placement.py, revision/scripts/evidence/b6_build_protein_placement.py) | done (fixtures) | tests/test_b6_protein_placement.py 11 passed (evidence world tests/test_b6_evidence_world.py 1 passed) | stop convention is an explicit input; measure mode provided; the real nine-species measurement is pending (command in report) |
| WP2b evlabel (src/transgenic/evidence/b6_evlabel.py, revision/scripts/evidence/b6_build_evlabel.py, b6_validate_evlabel.py) | done (fixtures) | tests/test_b6_evlabel.py 19 passed, 1 skipped (Writer C trainer absent: it must call b6_evlabel.refuse_if_evlabel) | genotype for mixed strata = 0.5 (fail-closed); retained intron = A18.4 operational rule per source; boundary cells not emitted (recorded, not final) |
| WP2c (b6_read_ends.py + b6_build_read_ends.py; CAGE_RECIPE.b6_v1.md + sha; b6_completeness.py + b6_build_completeness.py) | done (fixtures) | tests/test_b6_completeness.py 16 passed | adapter 5C disabled unless a sheet row has adapter_5c_approved=true (draft sheets have none); admitted-chain sets are an input file per population (WP3 b6_chains to produce); CAGE peak calling lives in b6_completeness.py |

Full-suite run 2026-09-24: tests/test_b6_*.py + test_reviewed_a_implementation_bytes_unchanged -> 272 passed, 1 skipped (47 min).

## Fix round 1 (Codex review notes/a46_approval/B6_WP2_CODE_REVIEW1_20260924.md)
- H1 done: protein phase collected per scope inside _protein_cds (same in_scope filter as CDS); validator checks phase eligibility independently; regression: unit case + 4577_0:000017 (tte) end to end + validator tamper test.
- H2 done: b6_evlabel.verify_training_binding (local, plan §8 key set) runs before labels.
- H3 done: freeze keeps {species: sha}; CR.verify_freeze / CR.freeze_manifest_sha256 per-species consumer; test feeds the real WP2 freeze + C0 provenance.
- M4 done: missing supplement for a species with accepted protein primaries refused; provenance inputs.protein_evidence bound/absent.
- M5 done: start and inside stop read from spliced tx-oriented CDS; exterior stop unchanged.
- M6 done: admitted-chain file declares '# population=<name>'; header-only files valid.
- M7 done: trainer-name test replaced by behaviour tests through the §8 verifier (local; WP3 b6_bindings too once merged).
- Full suite after fix round 1 (2026-09-24): tests/test_b6_*.py + byte pin -> 291 passed, 0 skipped (49 min).

## Fix round 2 (notes/a46_approval/B6_WP2_CODE_REVIEW2_20260924.md)
- H2 done: verify_training_binding coverage = merge_manifest (provenance keys must equal it); c0_qc record + leakage_nonzero for every covered pair, no extra pairs; exhaustiveness by_source must equal the audited covered sources; counters must be nonnegative JSON ints (no coercion). Regression: the reviewer's EST+ONT manifest / EST-only provenance+exhaustiveness / ONT leakage=1 case, plus missing qc, extra qc, missing source, float, string, negative counters.
- M7 done: local-verifier behaviour test (valid binding first, then evlabel as evidence and as corpus rejected); WP3 case builds a valid binding with the freeze path and a real cache first, skipped with an explicit reason in the WP2 tree.
- Full suite + byte pin: 298 passed, 1 skipped (the WP3 case) in 46 min.
