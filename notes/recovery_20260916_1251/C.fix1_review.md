# Track C fix1 independent scoped review

## Verdict: REQUEST CHANGES — one new Important regression

Reviewed worktree `/tmp/b6-p2-phase-1535433`, final `b7547e2edc175f7eac9713c8465b5b20e396a86f` (code `53c8f7d1a0f6f5d69a2b2ff47d82cec361d06afe`) against `2c91ecb295d88a1bc27199b0f84fe8af667abb08`. Read the original independent review, entire `C.fix1.diff`, and recovery-report round one. Queried worktree graft before extra source inspection. All five original concrete defects are addressed, but narrowing the exception boundary introduces another reproducible whole-inventory abort. No new Critical finding or C2/C3 gate bypass found.

All source references below are relative to this worktree.

## Original findings

1. **ADDRESSED — contradictory container-root markers.** `revision/scripts/evidence/est_blastx_audit.py:421-441,535-560`: each scanned root receives an independent marker check; pilot, batch, species, logs and overall tree scans occur separately from known-child exemptions. Absent container markers remain permitted; failed/contradictory/invalid markers fail. This is an actual failed tree check, not merely reliance on the unconditional pilot-log failure.
2. **ADDRESSED — unbound historical producer identity.** `est_blastx_audit.py:73-80,96-105,142-156,318,371-375,572-574`: mandatory engine/version/launcher/helper fields bind to production constants; executable bytes must also reproduce the recorded engine hash. Independently computed historical launcher/helper SHA256 from `git show 950cf4c:<path>` and both main-tree wrapper hashes: all match the public constants. Both historical wrappers pin the same engine SHA `0fb259d7…fed07d`. Launcher/helper source is unchanged from `950cf4c`; the version regex agrees with the historical release/build acceptance contract (`est_blastx.py:230-246`). Audit runtime identities are separate. No untrusted binary execution or public override was introduced.
3. **ADDRESSED — arbitrary timing/script substitution.** `est_blastx_audit.py:391-406,467-494`: both named source timing files must be regular/non-symlink and nonempty; approved wrapper bytes and all 16 receipt-backed remaining8 log paths are separate requirements. Empty stderr is allowed. Verified names against both launch scripts and `notes/RESTART_HANDOFF_20260915.md:35-39`. Pilot logs unconditionally produce explicit UNRESOLVED FAIL; arbitrary logs cannot satisfy it and no Slurm default is inferred.
4. **ADDRESSED — unmanifested nested unit payloads could pass.** `est_blastx_audit.py:297-313`: all nested directories fail the closed-flat-inventory check, including a nested valid DONE directory. Immediate regular files remain hashed; symlink/tmp rejection remains. The revised report no longer claims recursive hashing of rejected payloads.
5. **ADDRESSED — provenance list aborts all-unit inventory.** `est_blastx_audit.py:96-139,278-285,313-314`: explicit top-level and consumed nested schema checks produce EvidenceError, retaining accumulated hashes/checks. Focused public-code repro replacing pilot OrthoDB provenance with `[]` produced 18 unit manifests, `units_audited=18`, eight retained hashes in that failed unit, and `malformed_or_unreadable: FAIL / provenance must be a JSON object`. No synthetic trust-anchor monkeypatch was used in this repro; consequently other synthetic units also fail real identity pins, as expected. The separate regression below concerns malformed FASTA, not failure of this JSON-list correction.

## New finding

### Important — malformed filtered FASTA now escapes the unit boundary and loses all 18 manifests

**Locations:** `revision/scripts/evidence/est_blastx_audit.py:278-285` (changed catch), `:348` (unwrapped generator consumption), `:538-558,561-575` (all manifests/rollup deferred until after unit processing).

`launcher.records()` deliberately raises ordinary `ValueError` for empty/malformed FASTA (`est_blastx.py:88-98`). Its consumption while building `subjects` is not wrapped in an evidence-validation boundary. The fix removes ordinary ValueError from `audit_unit`'s catch. Thus an ordinary damaged evidence file now aborts the entire audit, even after its hash mismatch has already been recorded. Base `2c91ecb` correctly converted this same error into a failed unit and inventoried the remaining units.

**Observed focused repro, same temporary 18-unit tree for base and fixed code:**

```python
# PYTHONDONTWRITEBYTECODE=1; Python 3.12; PYTHONPATH=.testdeps:revision/scripts/tests
import test_track_c_recovery as t
# p is a fresh TemporaryDirectory Path
 tree, sacct = t.make_audit_tree(p)
 root = tree / t.audit.PILOT_DIR
 (root / 'orthodb' / 'orthodb.filtered.fa').write_text('not FASTA\n')
 # Call base.audit(tree, p/'base', sacct), then t.audit.audit(tree, p/'fixed', sacct).
 # Base module loaded from git show 2c91ecb:revision/scripts/evidence/est_blastx_audit.py,
 # with __file__ set to the current auditor path, using the unchanged launcher/helper.
```

(The indented lines above belong within the temporary-directory context.) No provenance edits or anchor monkeypatches were needed.

```text
base unit: FAIL
base: audit FAILED: units=['Sitalica/orthodb'] tree_checks=[] n_units=18
base unit manifests: 18; rollup: True; FAILED: True
fixed: ValueError: empty or malformed FASTA: .../orthodb/orthodb.filtered.fa
fixed unit manifests: 0; rollup: False; FAILED: True
```

**Required correction:** Convert the filtered-FASTA validation error at its consumption boundary into a failed unit while preserving previously gathered hashes/checks; cover the integrated all-18 continuation case. Wrapping generator creation alone will not catch an exception raised during iteration. Retain the deliberate propagation of genuine programmer errors. This is inventory loss, not false success: outer FAILED remains and no acceptance gate opens.

## Incomplete production prerequisites — not new bugs

- Missing pilot submission/log-layout receipt must remain explicit UNRESOLVED FAIL pending controller-reviewed historical evidence and a scoped follow-up. No inferred default, waiver, remote inference or resubmission recommendation.
- Real engine bytes and independent literal banner receipt remain unavailable locally. The implemented authority is the archived engine hash plus historical version-validation contract, not a claim of observed engine execution.
- Original absolute input paths, real payload audit, tmp/execution.lock/preflight reconciliation and unresolved `.extern` accounting remain outstanding. Wrapper-only A37 receipt/count replay remains outside the stated scope.
- C2 semantic acceptance/A46 and C3 production projection remain unconditionally blocked. Git diff confirms all three C2/C3 Python files unchanged from the fix base. This review does not convert their deliberately incomplete production deliverables into completed work.

## Verification limits

Did not repeat the controller's 116-test suite or bash syntax checks. Ran only two focused temporary-fixture checks (base-vs-fixed malformed FASTA; fixed provenance list), without monkeypatching production anchors. Independently checked historical constants, unchanged historical dependencies/C2/C3 modules, and diff whitespace. No source edits, SSH, jobs, subagents, merge or push. Only this requested report was written persistently. One advisor consultation supplied review-risk suggestions, not independent approval. Graft estimated savings: 148,566 tokens (tool estimate, not measured cost).
