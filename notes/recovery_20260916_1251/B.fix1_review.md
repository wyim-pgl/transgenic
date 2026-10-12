# Track B fix-round-1 independent re-review

## Verdict

**B1 ADDRESSED. B2 ADDRESSED within the expressly accepted unverified-identity boundary. B3 ADDRESSED within the expressly accepted external-reference-gate boundary.**

**No new Critical/Important issues found in this fix diff.** Approve this correction round as a **partial preparatory implementation**, not scientific/operational Track B acceptance.

Reviewed `200f1141708b4f913a7e05ae88661fce4abaf98e` → `2276fcb1efdeb124a97a11dced8d73e4d48560e0`; verified live HEAD. Read the complete supplied `B.fix1.diff`, appended implementation report, changed documentation/tests, and actual API/CLI publication paths. Graft was queried first and automatically refreshed its local graph. `git diff --check` passed. The controller's **95 passed / 21 subtests in 3.97 s** is supplied verification, not a suite rerun by this reviewer. No additional execution repro was necessary for these straightforward guards. No source edits, SSH, jobs, subagents, merge, or push.

## Finding dispositions

### B1 — ADDRESSED

**Code:** `revision/scripts/mvp/mvp_score.py:95-120`.

- `recall_tp > precision_tp` now raises. Combined with the existing `precision_tp <= predicted`, this also excludes recall TP exceeding predicted junctions.
- `frozen` is initialized once **per locus**, outside the seed loop. Every supplied count object checks `f1_reference` and, when present, `supported` against that dictionary. Thus all six seed/fraction objects share the same denominators, without incorrectly requiring different loci to have the same denominator.
- Existing nonnegative-integer/count-schema checks, numerator bounds, and partial-structural-input rejection remain intact.
- Zero support is not reclassified as missing: structural availability remains true and undefined recall becomes UNASSESSABLE.

**Tests inspected:** independent subset rejection; separate cross-seed mutations of `f1_reference` and `supported`; explicit zero-support/UNASSESSABLE case. These directly cover the original defects. The replacement checks also retain off/on denominator equality by construction.

### B2 — ADDRESSED, limited remedy rather than authenticated run identity

**Code:** `revision/scripts/mvp/mvp_score.py:74-85,121-128,144-164`.

- `resolve(strict=True)` catches repeated canonical paths/symlink aliases; pairwise `samefile()` additionally rejects hardlinks to the same file.
- It does not reject independent files merely because their numerical contents coincide, which is correct for this limited remedy.
- The single successful `score()` return path unconditionally contains `pair_run_provenance_status=UNVERIFIED`, `three_seed_identity_verified=false`, and `three_seed_policy_verified=false`.
- The CLI calls this function and serializes the same result after adding hashes/count-provenance text; it does not strip or override the new fields. Both direct API and CLI therefore expose the limitation, including structural-present, structural-absent, and UNASSESSABLE-readout cases.
- `decision` remains null. `--final-decision` still calls the unconditional failing coverage-policy gate before any output publication.

**Tests inspected:** repeated path, symlink, hardlink rejection; distinct files with identical numbers accepted and marked unverified; actual frozen-panel CLI fixture now creates distinct files and asserts the provenance fields; final-decision invocation must fail without creating output.

The low-level bootstrap/metric helpers still return mathematical values without run receipts. They are not the `score()` result publication API and were not promoted to provenance validators by this fix.

### B3 — ADDRESSED, truthful external-gate boundary rather than manifest verification

**Code:** `revision/scripts/mvp/mvp_align_panel_longreads.py:157-163,203-225`.

- Both successful `execute()` modes use the same `boundary` dictionary: `reference_identity_status=UNVERIFIED`, `scientific_input_validation=false`, `external_frozen_reference_gate_required=true`, and `training_eligible=false`.
- Non-execution returns `PREPARATORY_BYTE_CHECKS_ONLY`.
- Successful execution returns `PREPARATORY_COMMANDS_COMPLETED`; the exact same report is written to `alignment.json` and returned. CLI stdout serializes `execute()` directly, preserving these fields.
- Failure continues to raise and, inside the execution try block, writes FAILED.json rather than a successful report. Failure/intermediate artifacts do not acquire the new success statuses.
- Module help, checksum-option help, and SCORING.md explicitly identify caller MD5s as byte assertions and require an external frozen-reference identity gate/anchored receipts for promotion. There is no new approval switch.
- Searches found no remaining old `VALIDATED_ONLY` or `SCORING_ALIGNMENT_COMPLETE` status consumers in indexed code, so no in-repo status-dependent regression was identified.

**Tests inspected:** both return modes; publication equality between executed `alignment.json` and returned report; no subprocess/output directory in the non-executed case. The execution fixture uses explicit stand-ins and a small panel, not real aligners. It tests the common report path through HiFi; the ONT branch reaches that same report construction. The report correctly disclaims external-tool certification.

Orientation audit JSON/DONE, BAMs, junction TSV/QC, and command logs remain intermediate artifacts; the new reference status is on the driver result/alignment report, not injected into every legacy artifact. This matches the limited claim. Their existence alone still must not authorize scientific use.

## Remaining partial-scope limitations (not newly introduced defects)

- Distinct pair files do not prove distinct seeds/runs. Copies can still be supplied, but outputs now explicitly decline to verify the three-seed policy. Producer receipts binding seed, run, file hash, and complete inventory remain necessary.
- The external reference gate is **required but not implemented/enforced as a receipt check**. Actual frozen FASTA/GFF identity remains unverified; `--execute` is still preparatory execution, not operational approval.
- Raw-GFF/count authenticity, GPU decoding/300 predictions, baseline and complete result-table production, real-tool alignment certification, coverage/withholding evidence, and author policy remain outside this correction round.
- The unchanged pure decision helper is a mathematical rule, not an approval mechanism. The public score CLI's unconditional final-decision block remains essential.
- Aggregate `decision_status` still names the coverage-policy blocker; it is not an exhaustive blocker list. Consumers must also honor the explicit false/unverified identity fields and count-provenance limitation.

**Separate verdicts:** code-quality/correctness corrections approved for the scoped diff; spec compliance remains partial, with no authorization for scientific claims, GO/STOP publication, production promotion, or training.

Graft estimated approximately 92,234 tokens saved in this re-review.
