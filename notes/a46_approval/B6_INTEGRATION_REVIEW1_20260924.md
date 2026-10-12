**APPROVE_WITH_FIXES** — one new MEDIUM integration finding; no new CRITICAL/HIGH established. Reviewed `daf8436` through `b16e5af`, including all four uncommitted changes.

**Spec: MEDIUM — cache corpus/freeze bindings are recorded but not enforced.**  
The [cache builder](/tmp/b6-integration/revision/scripts/evidence/b6_build_obs_cache.py:108) publishes `corpus_sha256` and `freeze_sha256`, but [verify_cache](/tmp/b6-integration/src/transgenic/evidence/b6_bindings.py:464) never compares either with the verified upstream identity. This leaves plan §7.5’s corpus binding incomplete.

Starting from real WP2-produced artifacts and the real WP3 cache, an in-memory probe changed the upstream corpus digest, then separately the freeze digest; cache verification accepted both. A fresh run can therefore combine a stale cache with another independently valid corpus/freeze using the same C0. This demonstrates a provenance-contract gap, not leakage: cached observations derive from C0, and the current corpus supplies DNA masks and labels.

**Required fix:** For training caches, require both recorded hashes and compare them against `ident["corpus_sha256"]` and `ident["provenance"]["freeze"]`. Add missing/mismatched-binding regressions.

**Post-merge items:**

| Item | Status | Evidence |
|---|---|---|
| **WP2 M7** | **FIXED** | The [test fixture](/tmp/b6-integration/tests/test_b6_evlabel.py:420) builds a real cache through WP3’s CLI against WP2’s repaired corpus/freeze. No skip remains. Read-only probes accepted the valid bundle through WP4’s verification entry point, then rejected evlabel as evidence with `SchemaError` and as corpus with `BindingError`. |
| **WP4 M8** | **FIXED in code; GPU acceptance pending** | [Fixture labels](/tmp/b6-integration/tests/b6_fixtures.py:191) now contain meaningful genes on both strands. The [driver](/tmp/b6-integration/revision/scripts/b6/integration_4rank.py:195) checks valid, nonempty GSF, bidirectional label RC and served RC tokens using `gff_label_tokenizer("v3")`. The new label-RC test passed against existing artifacts. Every simulated link re-verifies inputs, and the launcher requires explicit data selection with reachable fixture checkpoint counts. |

**Other integration checks passed:**

- **Producer → verifier → trainer identity:** Actual WP2 freeze/manifests and WP3 cache passed WP4 verification. Edited coverage/provenance failed canonical-record validation; resealed changed provenance produced a different trainer identity.
- **Combined leakage flow:** Scoring observations are filtered before caching and rejected at runtime admission; train/valid eligibility remains distinct; targets come from corpus GSF. All 22 train rows in the real-producer fixture rasterised and built chains; all eight test rows were refused.
- **Standards and wiring:** No actionable import or conflicting-definition issue found. Fourteen CLI `--help` checks succeeded. The shared track specification is byte-identical to both WP3 and WP4 tips.
- **Protected files:** **92/92 match `681c5a0` byte-for-byte**, including the working-tree state. Policy-protected `train/train_HyenaTransgenic.py` also remains unchanged.

No files were modified. Verification used source inspection, targeted read-only checks and existing artifacts; no full suite, training or GPU acceptance run was performed. The reported two-CPU-rank PASS does not complete four-rank GPU acceptance.