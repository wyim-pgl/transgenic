# Track C independent review

Reviewed final `2c91ecb295d88a1bc27199b0f84fe8af667abb08` (code `91e10efa61bc6ddfd9cbb850ac52727dd33e7531`) against base `950cf4cd9ef55db276ffa1b713d8925ed6dd6dd2` in `/tmp/b6-p2-phase-1535433`. Read all 1,294 lines of `C.review.diff`, the recovery report, Track C discovery requirements, authoritative workplan §§2.2/2.6/4.6, and quarantine §1m FLAG16/ts row. Queried the worktree graph before extra source inspection. Also checked the main-tree historical pilot/remaining8 launch scripts and remaining8 input manifest.

## Verdicts

- **Spec compliance: REQUEST CHANGES for C1; PARTIAL, appropriately blocked for C2/C3.** C1 has reproducible false-PASS and incomplete-reconciliation paths. C2/C3 are not complete implementations of the original track, but their expressly limited recovery scope is honestly documented and production remains blocked.
- **Implementation quality: REQUEST CHANGES.** Reuse of the original filter/validator, exclusive output creation and unconditional semantic gates are good choices. The synthetic tests do not cover several important negative cases below. Do not promote this C1 audit to launch-ready or evidence acceptance yet.
- **Scientific safety:** No C2/C3 acceptance/A46 bypass found. No raw HSP count is emitted as accepted phase evidence. However, C1 can currently emit PASS and outer DONE for contradictory markers and incomplete evidence inventory. That PASS must not be treated as proof of a reconciled source tree.

All code paths below are relative to `revision/scripts/evidence/` in the reviewed worktree.

## Findings

### 1. HIGH — contradictory markers on known container roots bypass the rollup

**Locations:** `est_blastx_audit.py:321-344` (especially known-directory skip at 334-335), `:379-425` (known-root traversal); `:293-304` does not inspect species-root markers.

`scan_dispositions` checks children, not its own root, and skips known directories. Consequently, pilot-root, remaining8-root and remaining8 species-root DONE/FAILED markers are never reconciled. Unit markers are checked correctly, but this does not catch container-level failures. A stale `batch.exitcode=0` does not justify ignoring an explicit FAILED marker.

**Focused observed repro:** Constructed the synthetic 18-unit layout using only the existing `make_unit` fixture helper, with normal accounting, source hashes, commands and per-species exit codes. Added exact DONE text plus `FAILED.json={}` at `sitalica_pilot_20260914/`. Called `audit(tree, fresh_out, sacct)`.

```text
contradictory pilot root: {'disposition': 'CONTRADICTORY', 'ok': False}
rollup: PASS units: 18 outer DONE: True
```

**Required correction:** Inspect markers on every container root independently of its child-layout exemption. Normal containers need not acquire a new DONE requirement, but present FAILED/contradictory/invalid markers must fail. Cover pilot, batch and species containers.

### 2. MEDIUM — original executable/code provenance is not fully reconciled

**Locations:** `est_blastx_audit.py:232-241,268-275,426-435`; original producer `est_blastx.py:310-315`.

The producer records `diamond_version`, `launcher_sha256` and `taxonomy_helper_sha256`. The auditor ignores all three. It verifies the executable against the hash in that same unit, but does not bind that hash/version to the approved engine snapshot; the rollup's current launcher hash does not reconcile the historical producer record. Thus “frozen command”/provenance PASS is weaker than the requested original-provenance binding.

**Focused observed repro:** On an otherwise passing `make_unit` fixture, set `diamond_version='NOT-DIAMOND'`, `launcher_sha256='0'*64`, and `taxonomy_helper_sha256='0'*64` in provenance. `audit_unit(...)` still returns **PASS**. The fixture itself omits these original producer fields, which explains why baseline tests miss this.

**Required correction:** Require and reconcile these fields against the reviewed historical code/engine identities, without executing an untrusted binary as an audit shortcut. Distinguish explicitly any historical identity that cannot be verified. Also retain the current audit implementation identities.

### 3. MEDIUM — source-specific timing and batch-log absence can pass

**Locations:** `est_blastx_audit.py:293-304,347-355`.

Timing passes if any one nonempty `*.time.txt` exists, irrespective of source. Batch artifacts pass if any one script or log matches. Consequently one source can have no timing at all, and all Slurm logs can be absent while the audit still passes. Historical launchers produce **both** `swissprot.time.txt` and `orthodb.time.txt` (`notes/b6_eight_species_blastx_20260915.sbatch:48-53`; pilot `:41-46`), not the test's single arbitrary `batch.time.txt`.

**Focused observed repro:** `batch.exitcode=0` plus only `only-one.time.txt='x'` yields no species-check failures; only `run.sbatch='x'`, with no logs, yields no batch-artifact failures. The integrated marker repro above also returned PASS with only `orthodb.time.txt` per species and no Slurm logs.

**Required correction:** Define and require the actual per-source timing and layout-specific script/log inventory. Explicitly report missing artifacts rather than allowing one present artifact to stand in for all categories. This does not require interpreting timing as a benchmark.

### 4. MEDIUM — nested unit payloads can pass without any hash inventory

**Locations:** `est_blastx_audit.py:215-220,321-344`; recovery-report claim of “complete unit-file SHA256 inventory.”

Only immediate regular files are hashed. A nested directory with valid DONE text is allowed by disposition scanning, while its ordinary files are ignored. Thus a passing unit can contain unmanifested payloads; the report's complete-inventory claim is false.

**Focused observed repro:** Added `orthodb/unexpected/DONE` with exact expected text and `orthodb/unexpected/payload='untracked'` to a passing unit. `audit_unit(...)` returned **PASS**; `files` contained only the eight immediate unit artifacts, not either nested file.

**Required correction:** Either reject unexpected nested content as unresolved or recursively inventory regular files by relative path. Preserve the unconditional unresolved-tmp failure and symlink rejection. Do not silently grant a closed inventory solely because a nested DONE exists.

### 5. MEDIUM — malformed JSON container types abort the entire audit instead of preserving available-unit inventory

**Locations:** `est_blastx_audit.py:183-189,221-232,420-421`.

The wrapper catches several input errors but not `AttributeError`, and does not validate the provenance object's type before `.get`. Unit manifests are written only after every unit has been processed. Therefore one malformed unit can abort the whole run before the other available units and final rollup are recorded.

**Focused observed repro:** Replace an otherwise complete unit's `provenance.json` with valid JSON `[]`. `audit_unit(...)` raises uncaught `AttributeError: 'list' object has no attribute 'get'` rather than returning a failed unit record.

This is **not a false success or phase-gate bypass**: the outer context will fail and normally write FAILED.json. It is a defect in the claimed malformed-provenance disposition and full audit inventory behavior.

**Required correction:** Validate top-level/nested JSON types explicitly and convert malformed unit evidence into a failed unit record while continuing other units. Preserve already-collected checks/hashes where possible.

## Compatibility and safety checks that held

- The real producer's `inputs` keys match `production_inputs(**inputs)`, and its all-string serialization is compatible: species/source are strings and paths are normalized inside the reused functions. No suspected string/integer round-trip bug was found.
- Historical wrappers use absolute query/filtered/database/output/executable paths and 16-thread command arguments, compatible with the auditor's reconstructed argument checks. Both pilot `query.sha256` and remaining8 `inputs.json` query resolution paths are present. Real archived payload bytes were not available locally, so this is source-contract compatibility, not a completed real-data audit.
- The remaining8 wrapper additionally validates manifest summary/count/receipt fields; the new audit does not replay those wrapper-only bindings. Its direct launcher provenance does bind the screening summary and query bytes. This limitation should remain explicit if claiming replay of the whole historical wrapper rather than just launcher inputs.
- Real launchers intentionally leave species `tmp/`, and the pilot creates `execution.lock/`. Their resulting FAIL/NEITHER dispositions are expected conservative policy, **not a request to rerun blastx or automatically waive/delete them**. Preflight containers likewise need separate reconciliation.
- Unit DONE/FAILED contradiction and descendant symlinks are explicitly detected; known child symlinks are checked before the known-name skip. Existing output paths are exclusively refused. A focused check of an existing output symlink returned `FileExistsError`; no overwrite occurred. The input-tree output/scratch guards use resolved paths. I did not test concurrent path replacement races or claim race-hardening.
- Accounting checks require all 27 expected allocation/batch/extern rows and preserve the anomaly with `clean_accounting=false`. A payload PASS is expressly not clean-accounting approval.
- C2 `run()` and `accept_stream()` immediately raise before I/O. Candidate decomposition is labelled hypothetical; diagnostic reports remain UNRESOLVED/INCONCLUSIVE or CANDIDATE_REQUIRES_REVIEW with authorization false. Outer fixture DONE is documented as diagnostic completion only.
- C3's four-way FLAG16/read-relative-ts composition agrees with quarantine §1m. Phase disagreements return absent support for both routes, not phase-label zero. `project()` unconditionally refuses production.

## Deliberate partial-scope omissions, not implementation defects

- No real RICH17 run or reviewed operational ≥30-aa definition; no certified slash/backslash semantics; no real-engine reverse-shift/boundary/discriminating acceptance demonstration. The diagnostic extension and hand-built tests cannot substitute for those gates.
- No `est_protein_hsp` population, genomic projector, intron-mod3 multi-exon fixtures, inconsistent-HSP resolution, six-channel raster, BAM ingestion or molecule aggregation. These remain workplan §§2.2/2.6 deliverables. The current primitives must not be represented as completed C3.
- A46 production authorization is still required under §4.6 for C2 **and** C3. No real-data C1 acceptance or Slurm submission is established by this review.

## Verification scope

Did not rerun the controller's 65-test suite or bash syntax checks. Ran only the focused temporary-fixture reproductions described above, using Python 3.12 and existing `.testdeps`, with bytecode writes disabled. No production code edits, SSH, jobs, subagents, merge or push. One advisor consultation was a review-risk checklist, not an independent approval. Only this requested review report was written persistently.

Graph tool estimated cumulative savings: approximately 952,586 tokens (tool estimate, not measured cost).
