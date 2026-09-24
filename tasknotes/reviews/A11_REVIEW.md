# Independent A11 review — 2026-09-17

## Decision: CHANGES REQUESTED — one Important finding

Scope: the new bounded normalized-table consumer, its 39 tests and `tasknotes/a11_molecule_closure_20260917.md` in `/tmp/est-approved-contract-20260917`, HEAD/base `c9a351431b7dd8fc8ad45be35a9df29ca8445f81`. This is **not** a production C0/import/schema-migration review or an A46-wide approval. No demand for completed nine-species inventories is added to this library's acceptance criteria.

Read the approved decision record `notes/APPROVED_HSP_PHASE_A11_B5_20260917.md`, the complete `A11_IMPLEMENTATION_BOUNDARY.md`, exact worktree A11/A21 clauses (`PROTOCOL_B1_frozen_v1.md:552–566,327–334`) and main's A46 clauses (`:951–981`). Graft map/query/skeleton preceded source inspection. No code, index, HEAD, commits, remote jobs or subagents were changed/launched by this review. Review artifacts are in `a11_review_evidence/`. Concurrent workers' unrelated untracked files were left untouched.

## Important: conflicting libraries silently split versions of one accession

**Location:** `revision/scripts/evidence/est_molecule_closure.py:236–256` (automatic version keys), `:278–281,305–310` (newest-version keys), and metadata validation before those keys. The guard at `:264` catches a conflict only when an explicit relation is separately supplied. The component library guard at `:286` cannot detect identities already split into different components.

**Reproduced, not inferred:** supply `X.1` in canonical `lib` and `X.2` in canonical `lib2`, same species/assembly/primary arm and same normalized clone. Both libraries are individually valid, audits/completeness declarations are present, and no explicit relation is supplied (the API already promises automatic version edges).

1. With clean placements: `diagnostic_ready=True`, no blockers, **two ready components**, each exposing its version in `training_support_members`. Thus the older version is not suppressed.
2. Add an old-version secondary placement at `chr9:[100,200)` and a held-out exclusion at `chr9:[150,160)`: only `X.1` becomes excluded; **`X.2` stays ready with training support**. Independent `recheck_retained` passes because it trusts the incomplete membership partition, though it correctly rechecks geometry within that partition.
3. Changing the second record's download dataset does not remove the defect.
4. Control: make `lib2` an alias of `lib`; one component forms, only `X.2` supports, and the old placement excludes the component. An explicit `version` relation with the contradictory libraries correctly raises `relation library contradiction`.

Evidence: executable `a11_review_evidence/adversarial.py`, full output `adversarial.json`, especially `contradictory_libraries_two_ready_versions` and `old_exclusion_fails_to_propagate_and_rejoin_passes`.

**Why this violates the narrow contract:** A11 says versions are one accession and keeps only the newest parsed version for support; A46 rule 3/4 retains all versions for molecule-wide leakage. The implementation boundary explicitly requires conflicting identities to fail closed. Library-scoped clone/signature keys correctly preserve genuinely independent libraries, but cannot turn contradictory annotations of the *same accession* into two independent valid identities. This is not a request to merge independent preparations or change A11 policy. Universal `training_eligible=False` limits downstream authorization, but does not make a falsely ready diagnostic conformant.

**Minimal fix:** before constructing library-scoped keys, validate that an accession stem in the supplied species/assembly/arm has at most one resolved canonical biological library, across datasets and versions. Reject contradictory resolved libraries (or block the whole diagnostic explicitly); do not arbitrarily select a library and do not merge genuine different-library records by clone text/signature. Preserve existing unknown-library whole-input blocking. Add regression tests for the two outcomes above, a cross-dataset conflict, alias acceptance, and distinct-accession/different-library separation. Retain existing same-library version/latest-support tests. No new framework or production importer is needed.

## Other requested checks

| Area | Review result within the declared normalized-input boundary |
|---|---|
| Clone/library aliases, versions, mates, split parts | Known clone normalization and canonical library scoping work; explicit mates require matching known normalized clone. Report-proven nonoverlapping pieces union, including one-survivor handling without suffix inference. All input records remain. **Version/library conflict above is the exception.** |
| Latest support vs leakage | Same-library latest-only support works; every older/rejected placement remains in leakage checks. No representative placement replaces full geometry. |
| Known clone vs fallback | Known clones do not get signature keys; unavailable clones require exactly one primary, known transcript strand and exact ordered blocks. Same-signature known/unavailable records stay separate unless an explicit identity edge supplies linkage. Different known clone identities in one component raise. |
| Explicit parent proof/status | Parent needs existing distinct endpoints, nonempty provenance, same canonical library; component known-clone consistency still applies. It alone may link arms. A known/unavailable pair can be linked by the caller's explicit parent assertion; this is acceptable at this trusted normalized-table boundary, not independent proof verification. Empty provenance and contradictory known clones reject; an uninspected parent member blocks the whole input. No new parent-status policy is required here. |
| Unknown/incomplete metadata | Unknown library, uninspected clone, unresolved fallback, missing/incomplete member audit and false completeness declarations prevent all ready views; unknown rows are not dropped to certify the rest. |
| Primary/min121 | Separate support views; only primary enters `training_support_members`. Min121-only components are `never`; explicit parent linkage does not pool sensitivity support into training. |
| Intervals and exclusions | Zero-based half-open broad placement spans, either strand, all placement kinds; intronic spans overlap, boundary touches do not. All nine typed exclusions are exercised; valid orthogroups/partial intervals are context-bound. Shared junctions are supplied normalized exclusions, not independently derived junction annotation. |
| Audit thresholds | Secondary count ≥50 excludes; total placement count is separately used for `<50` exhaustiveness. ≥0.999 is integer-compared; tests cover 1000 vs 999 molecule boundary. MAPQ 4 excludes outside intervals and MAPQ 5 passes. Missing receipts prevent whole-input readiness. Audit command execution/completeness is caller-attested, not independently established. |
| Determinism and rejoin | Sorted IDs/edges/components are reproducible; additional 72 member/placement/relation-order permutations agree exactly. Retained rejoin catches disposition tampering and context/manifest changes; it cannot discover missing identity edges (the demonstrated finding). |
| No chain fabrication/count-policy drift | Original per-placement blocks remain unchanged; no joined chain, gap adjacency or invented junction is output. This primitive exposes support membership, not per-feature counts. Existing scorer's cap ten and support acceptance remain outside the primitive; focused scorer tests pass. |

Completeness booleans, report identities, hash syntax, provenance strings and upstream acceptance are **caller declarations, not truth/source attestation**. This is explicitly disclosed in both code and tasknote and is not a new blocker for this bounded consumer. The report's broad assertion that contradictory identities raise must be narrowed or made true by fixing the demonstrated exception. Actual inventory recovery, verified bytes/commands, full C0 serialization, raster/phase/trainer integration and production A46 certification remain out of scope.

## Verification and evidence

Executed twice, without bytecode/cache-provider writes to the worktree:

```text
cd /tmp/est-approved-contract-20260917
PYTHONDONTWRITEBYTECODE=1 /tmp/mvp-final-source-env/bin/python -m pytest -p no:cacheprovider \
  revision/scripts/tests/test_est_molecule_closure.py \
  revision/scripts/tests/test_evidence_support.py \
  revision/scripts/tests/test_univec_trim.py -q
89 passed (39 new + 50 focused existing); archived run 0.72 s
```

`git diff --check` passed. The selected worktree does not contain `test_mvp_c0lite.py`; no MVP test execution is claimed. Additional adversarial script exits 0 while asserting the observed defect and controls; **its success demonstrates the defect, not correctness**. No fix was applied. The attempted advisor consultation failed with a client/model-version error; no advisor endorsement is claimed.

Reviewed SHA256:

- Consumer: `53d2f8fbf58f8851e669d25701f01d5ced02682351d7c4ea45cd6b6af144eb1f`
- Tests: `88ddb85f0e262e7a8da1ed750f454e35228647dce7adb794299596ec6722583a`
- Tasknote: `cf3bf112a1c3e1c576a18b92a85348acb8dc19d9fcca7f221a865bf5c6d7123e`

**Next:** implement the single identity-consistency guard plus regressions, rerun the 89-test focused suite with new regressions and independently recheck this reproducer before scoped approval. Graft reported approximately 836,257 tokens saved in this review.
