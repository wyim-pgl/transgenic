**REJECT — H2 remains HIGH, and a new HIGH WP3/WP4 interface mismatch blocks training.** No CRITICAL finding.

Reviewed the uncommitted changes, fix notes, specified plan/protocol, and current WP3/WP4 consumers. No files modified; no full suite run.

| Round-1 item | Status | Evidence |
|---|---|---|
| **H1 — train-phase leakage** | **FIXED** | [b6_evlabel.py:625](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:625) applies scope eligibility before collecting phase; projection is scope-specific. [Validator:168](/tmp/b6-wp2-corpus/revision/scripts/evidence/b6_validate_evlabel.py:168) independently rejects ineligible phase rows. |
| **H2 — incomplete training binding** | **PARTIALLY** | Required hashes, schema, QC and exhaustiveness checks were added, but [b6_evlabel.py:183](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:183) checks only sources represented in provenance. Missing-source coverage still permits bypass; details below. |
| **H3 — species-map incompatibility** | **FIXED** | [WP3 b6_bindings.py:409](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:409) now compares against `man[sp]`; legacy revalidation also uses the species entry. The original scalar-versus-mapping defect is closed. |
| **M4 — silently missing protein supplements** | **FIXED** | [b6_evlabel.py:246](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:246) refuses missing supplements for species with accepted primary proteins and records bound/absent evidence explicitly. |
| **M5 — split terminal codons** | **FIXED** | [b6_protein_placement.py:163](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_protein_placement.py:163) reads start and inside-stop codons from transcript-oriented spliced CDS, retaining the exterior-stop convention. |
| **M6 — empty admitted-chain populations** | **FIXED** | [b6_completeness.py:253](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_completeness.py:253) reads population independently of data rows, permitting hash-bound empty sets. |
| **M7 — guaranteed integration-test failure** | **PARTIALLY** | The literal function-name assertion is removed, but [test_b6_evlabel.py:418](/tmp/b6-wp2-corpus/tests/test_b6_evlabel.py:418) supplies `cache=None` to WP3. Once imported, that branch raises `AttributeError: 'NoneType' object has no attribute 'manifest'`, which neither expected regex accepts. It also passes the incompatible freeze dictionary described below. |

**Remaining HIGH — H2 still fails open for an omitted source.**  
[b6_evlabel.py:202](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:202) checks leakage counters only for existing provenance groups, without comparing those groups against `merge_manifest`. An in-memory reproduction using the actual merged schema and verifier accepted:

- EST and ONT present in `merge_manifest`;
- EST-only provenance and exhaustiveness metadata;
- ONT `leakage_nonzero = 1`.

The probe mocked filesystem/hash/open boundaries only. Thus a source can escape both exclusion-identity and leakage checks. Additionally, [line 213](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_evlabel.py:213) does not require all audited sources and silently coerces counts with `int()`. Require exact source coverage and nonnegative integer counters, matching WP3’s existing gates.

**New HIGH — WP4 passes the wrong freeze argument type.**  
[Trainer:547](/tmp/b6-wp4-trainer/train/train_HyenaTransgenic_B6.py:547) parses the freeze JSON and passes its dictionary to `verify_inputs`. [WP3:341](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:341) now requires the freeze **path** so its bytes can be hashed. Direct invocation reproduced `BindingError: freeze must be the path…`; training and resume stop before verification completes. Pass `paths["freeze"]` and exercise the actual consumer with valid bindings before testing evlabel rejection.

No additional CRITICAL/HIGH regressions were found in the H1 or M4–M6 fix paths. Settled round-1 items remain closed.