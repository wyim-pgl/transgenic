# Bounded approved EST contract increment — controller acceptance 2026-09-17

User-approved whole reported DIAMOND HSP length>=30, evidence-local phase conflict removal preserving independent support, and existing A11 identity/leakage rules. This is a bounded pure-library implementation, not full C0/production phase/readiness acceptance.

## Source and tests
- Base5c46c75 geometry + unchanged reviewed a98f5c0 rich17 diagnostic integrated as c9a3514. Existing est_phase_accept/project/expanded bytes unchanged. src/train/configs/protocols/deploy unchanged. Main, GPU b660, eval9603, all remote jobs/data untouched.
- New modules: est_hsp_contract.py (raw13 replay/rich17 length selection/codon events); est_phase_conflicts.py (complete-HSP direct conflict sets, pair-local A/B, molecule/base dedup); est_molecule_closure.py (explicit normalized-member relations/all-member leakage/latest-only support).
- Controller four-case synthetic connection test joins HSP→splice geometry→A11 ID→phase counts for qframe±1×FLAG0/16; independent expected positions include split codon/intron. It is NOT actual BAM/routeA/fullC0 integration and synthetic IDs are not production namespace encodings.
- Controller final Python3.12 focused suite: **278 passed1.23s**, log final_tests.txt. Includes three new suites, four connection cases, old geometry/HSP/enginefixture/evidence scorer/UniVec. Whole-repo suite deliberately not run (known unresolved native crash; blind rerun prohibited). Negative scorer error messages in successful log are expected rejection tests.
- Initial combined run with defaultPython3.8 failed collection in unchanged59_evidence_support.py's tuple|None annotation. Preserved py38_collection_failure.txt; selected existing Python3.12 environment without source workaround, then passed. No global package modifications.
- Controller actual old144 fixture replay: exactraw13→rich17 144/144,126HSP-rule accepted/18below30, M9294 I30 D30 F96. Input bytes unchanged. Old132ELICITED/12NOT_ELICITED diagnostic statuses retained, not retroactively certified.

## Independent reviews
Main repo notes/approved_execution_20260917 holds HSP_REVIEW.md (APPROVE), PHASE_REVIEW.md(APPROVE), A11_REVIEW.md(Important), A11_FIX1_REVIEW.md(APPROVE), INTEGRATION_REVIEW.md(APPROVE).
The A11 Important was genuine: contradictory resolved libraries could split versions of one accession and lose older-version leakage. Fixed by pre-union same-stem canonical-library consistency guard, with failing regression then green; unknown identities still block the full ready view. Controller examined actual code and confirmed final reviewed hashes.

## Exact remaining boundaries
- Libraries consume supplied sequences/normalized metadata; complete-HSP/library/audit/reference/source authority are not proven by caller assertions or digest syntax.
- -1 frameshift HSPs may pass length yet yield self_overlap_policy_unresolved in old geometry. Preserve this disposition; never call empty segments resolved negative evidence. Production binder and full-scope handling remain unimplemented.
- RouteA miniprot replay/adapter, complete nine-species A11 metadata/audit/exclusion manifests, stable full-HSP/component provenance namespace, streaming publication/raster and real-source integration still required. No all-species raw scan, rawblastx rerun, DB rebuild, or training certification.
- HSP30 rule is now approved; fullA46/mainmerge/push are not. Keep branch/worktree; no merge or push.
