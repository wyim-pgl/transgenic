# A11 normalized membership/leakage primitive — 2026-09-17

## Scope and files

Implemented only `revision/scripts/evidence/est_molecule_closure.py` and `revision/scripts/tests/test_est_molecule_closure.py`, plus this report, in `/tmp/est-approved-contract-20260917` at base `c9a351431b7dd8fc8ad45be35a9df29ca8445f81`. No commits, shared-file edits, MVP/schema/trainer/phase changes, source scans, remote access, jobs or downloads. Other workers' HSP/phase files and preexisting ignore changes were left untouched. Graph refresh deferred to controller because graph output is shared.

Read the main `notes/approved_execution_20260917/A11_IMPLEMENTATION_BOUNDARY.md` completely; read exact A11/A21 clauses on selected base and A46 rules from main (older worktree lacks A46). Read actual-input inventory lines 81–128 and C1 final review lines 48–77 including the min121 correction. Graft used first.

`git diff 9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558 -- revision/scripts/59_evidence_support.py` was empty. Scorer source SHA256: `06f1bdc9a3af511e6de7434430f191d6b272d244ce7a44303d1c87dd29be25f4`. The four-line clone normalizer is copied verbatim to avoid loading unrelated scorer machinery; AST equality regression verifies reuse. No other scorer behavior is repurposed as leakage logic.

## Implemented API and semantics

`close_molecules(...)` is a pure, bounded normalized-table consumer with explicit Member, Placement, Relation, Audit and Exclusion records, biological-library aliases, one species/assembly, train-tile context, scientific-policy identifier, and three completeness declarations. Required arguments have no scientific-policy defaults.

- Validates species/assembly, accession/original-accession agreement, original retained half-open intervals and lengths, hash syntax, report identity/nonoverlapping pieces, library alias consistency, explicit clone status, placement geometry, relations and audit receipt counts. Invalid/contradictory/orphan inputs raise ValueError.
- Canonical biological libraries are supplied, never inferred from run/plate. Clone identity uses existing normalization. Explicit `clone_unavailable` alone allows same-library/arm/contig/transcript-strand/exact ordered block fallback from a unique primary placement; unknown/multiple primary geometry blocks readiness rather than guessing. Distinct known clones do not collapse by signature.
- Deterministic sorted union-find membership retains every original record and placement. Version, clone, report-proven split, explicit mate/parent and block-signature edge reasons are preserved. Same-library/arm accessions and clones may cross dataset boundaries; cross-arm linkage requires explicit parent provenance. Component IDs are sorted member tuples, implementation details, not protocol-frozen hash formulas.
- Report ID denotes a caller-normalized authoritative UniVec row; a one-survivor split needs no invented suffix or sibling. Original accession remains explicit, not parsed from `_partN`. No adjacency, combined chain or imputed junction is constructed.
- Latest-version support selection is library/arm/accession scoped and separate from all-member leakage. Support acceptance is caller-supplied, not re-scored. Min121 support remains a separate view, never contributes to `training_support_members`; min121-only components have `never` disposition. All results/components always have `training_eligible=False`.
- Every placement kind, old version, mate and rejected-support record participates in broad-span half-open overlap, either strand, including intronic span. Every overlapping exclusion cause is retained. Typed global classes: heldout, test_block, test_orthogroup, hard_flag, overlap_component, decoy_masked, shared_junction. Valid-orthogroup and partial spans require an explicit matching train context; they cannot silently become global. Shared-junction inputs here are **already normalized excluded intervals**, not a native junction-coordinate/annotation parser.
- Missing member audit or incomplete audit excludes that component and blocks the entire ready diagnostic. Any unknown library/uninspected clone, unresolved fallback, incomplete inventory/identity/exclusion declaration blocks every component's ready view. No unknown rows are dropped to certify a remainder.
- Secondary count >=50 and MAPQ<5 outside exclusions exclude the molecule. Total-placement count is distinct from secondary count. Explicit exhaustion counts are supplied-table molecules with total placements <50 / all molecules; >=0.999 is checked with integer arithmetic. Counts still describe supplied records if a receipt is missing, but missing completeness independently prevents readiness.
- Independent retained-span rejoin ignores stored causes and rejects a retained-overlap tamper; changed context or exclusion manifest is rejected. This is not cryptographic protection against changing the entire result/input object.

Returned originals, placements, relations, audits, exclusions, library aliases, edges, component support members, dispositions, causes and counts make the bounded diagnostic auditable. Caller IDs must uniquely namespace records across dataset/arm/name. Provisional `retained` disposition does not override `ready=False` or universal `training_eligible=False`.

## Actual verification

TDD initial run: **24 failures**, each explicit assertion that the primitive did not exist, before implementation. Subsequent failing regression rounds covered min121 disposition, unsupported unmapped support, changed recheck context/manifest and missing-audit exhaustion denominator before fixes. One initial implementation import needed postponed annotations for default Python 3.8; the 1000-molecule fixture was corrected to avoid biologically intentional normalization of terminal `-3`/`-5` clone directions.

Final command:

```
/tmp/mvp-final-source-env/bin/python -m pytest \
 revision/scripts/tests/test_est_molecule_closure.py \
 revision/scripts/tests/test_evidence_support.py \
 revision/scripts/tests/test_univec_trim.py -q
```

**89 passed** (39 new closure tests + 50 existing scorer/UniVec tests), 0.99 s. `git diff --check` and target `py_compile` passed. Default system `python` could not collect existing scorer tests because its Python 3.8 cannot evaluate their `tuple | None` annotation; selected existing Python 3.12 environment passed unchanged tests. The controller's broader baseline 70 is not claimed rerun here.

Coverage includes clone/library aliases and separation, known-vs-unavailable clones, block-signature distinctions, versions, split one-survivor/multiple pieces, explicit cross-arm linkage, no suffix/hash/adjacency inference, unknown metadata/global fail-close, invalid intervals/accessions/hash, orphan/contradictory links, every placement kind, rejected support, old-version secondary leakage, half-open touching, intronic span, all nine typed exclusion classes/context, saturation 49/50, low MAPQ, missing audit, exhaustion >=0.999, permutation invariance, duplicate rejection, independent tamper checks and exact normalizer reuse.

Advisor planning was attempted but failed with client/model-version incompatibility. No independent advisor/code review is claimed; controller should review before integration. Graft reported ~893,299 tokens saved (782,276 + 34,150 + 76,873).

## Still blocked / deliberately not certified

Completeness flags, report IDs, provenance strings, per-record hash strings and upstream acceptance are **caller statements, NOT source attestation**. Hash syntax is validated but actual sequence/manifest bytes are not read or verified. There is no full database serializer/importer/CLI, source metadata parser, audit-command runner/verifier, raw UniVec parser, annotation/junction exclusion builder, scorer acceptance reimplementation, feature counting, raster, phase or trainer integration. Existing library cap ten remains the scorer's per-feature cap; no molecule/phase cap was introduced.

Actual nine-species production closure remains blocked on authoritative biological-library/clone/mate/version inventory, complete report-to-member retained-interval joins, all-member placement/audit coverage and all typed/contextual exclusion manifests. Existing MVP tables do not establish those identities; zero suffix unions is not zero biological clone duplication. Accepted primary C1 query bindings remain valid; stale nearby min121 provenance does not reopen C1 or trigger screening/alignment/download reruns. This implementation does not certify broader draft A46 production readiness.

## Independent-review correction — accession/library contradiction

Addressed the single Important finding in main `notes/approved_execution_20260917/A11_REVIEW.md`. Read its preserved `a11_review_evidence/adversarial.py` without modification. The original implementation's library-scoped version/latest keys incorrectly let one accession stem annotated with two canonical libraries become two ready components; an excluded old-version placement then failed to exclude the newer version. The initial 89 passing tests did not establish correctness for this contradiction.

Minimal consumer delta: seven added lines collect resolved canonical libraries per `(species, assembly, arm, accession stem)` across versions and datasets and reject more than one before constructing union keys. No library is selected arbitrarily and no genuinely independent libraries are merged. Unknown libraries still block the whole ready diagnostic through the existing path; aliases resolving to one canonical library are accepted. All prior production/completeness limitations remain unchanged.

Added eight regression cases: clean conflicting versions and old-version-secondary exclusion, each within/across download datasets (four rejecting cases); clean/excluded canonical-alias controls (two); distinct-accession/different-library independence; unknown-library version whole-view blocking.

Actual red → guard → green: before the guard, **4 failed, 43 passed**, all four failures `DID NOT RAISE ValueError` for the reproduced contradiction. After the guard, the requested combined command with `PYTHONDONTWRITEBYTECODE=1 /tmp/mvp-final-source-env/bin/python3.12 -m pytest -p no:cacheprovider` and these four paths passed:

- `revision/scripts/tests/test_est_molecule_closure.py`: 47 cases
- `revision/scripts/tests/test_evidence_support.py` and `test_univec_trim.py`: 50 existing cases
- `tests/test_est_approved_contract_integration.py`: 4 controller cases

**101 passed in 0.86 s**; `git diff --check` passed. No commits; intentional edits restricted to the A11 consumer, its tests and this note. Graft's required first query reported an automatic one-file graph refresh and ~29,734 tokens saved; no explicit graph build was run. Preserved review artifacts and other worker/controller code were not modified.
