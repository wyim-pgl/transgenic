# Narrow final review: release_v6

## Verdict

**APPROVE the exact alias-corrected release_v6 candidate under all previously accepted release_v5 conditions.** No additional deployment blocker was established. This approves candidate bytes, not execution or training. No new approval requirements are introduced.

The prior decision and conditions remain those in `../deployment_v3/review/INDEPENDENT_FINAL_REVIEW.md`: explicit fixed-failure recovery, independently pinned controller authority, serialized single-attempt reservation through terminal reconciliation, fresh compute admission and runtime gates, preserved B5/account capacity and panel reservation, read-only known-input recovery, no automatic retry, and no training or followup. The attempt remains one 1800-second four-GPU allocation, at most 0.5 node-hours/2 GPU-hours within the existing 5/20 diagnostic caps. Prior failed work remains charged.

## Exact scope and findings

Compared all 14 release files against approved `deployment_v3/release_v5`. All six production/configuration files also match their tested deployment_v4 counterparts. The sole semantic implementation change is `release_v6/admission.py:247`: `fields()` maps exactly `CPUs/Task` to `CPUsPerTask` before duplicate checking. Both aliases together still reject, in either order, including equal values. No other guard changed. Remaining production differences are v5-to-v6 namespace, build-output, authorization and job labels. Release metadata differs only in refreshed file hashes. Source archive, source manifest, operator pins and accepted companions are byte-identical.

The retained raw-line SHA256 is `95d819c95d7665cb830f177c75f62db80585245d77dda26ac48159c80ad26679`. Independent reproduction confirms the old parser lacks the internal key and the new parser yields task CPU64. This line is historical PENDING evidence, not a current allocation or execution authorization.

## Verification

Executed `python -B narrow_evidence/verify.py` successfully. Its Python unittest runner independently ran all five `FieldAliasTests` against the packaged release_v6 admission module: 5 PASS. Supplemental synthetic RUNNING allocation assertions accepted CPU64 and trusted node-capacity CPU288 while rejecting ten wrong/missing task-CPU, request, allocation and capacity cases. Historical PENDING timestamps were not treated as current allocation evidence.

Transport has exactly 14 unique regular members, matching release bytes; ledger has exactly 13 unique expected entries with matching hashes. All 305 current files named in `preservation-before.json` match their recorded hashes. `preservation-after.json` is correctly treated as the controller summary, not a hash map. The retained controller suite documents 47 PASS in 11.273 seconds; it was not rerun.

| Artifact | SHA256 |
| --- | --- |
| release_v6.tar | `26c1dc0a233a95ab86977813e68310eccfce91b081de3a63586cfd7b9427628a` |
| release_v6/release.json | `64a547b8ba5f54e87c1985f278564a9ea9fc7c8f3af07f6b6afe3bdf17f4f7d4` |
| release_v6/release.sha256 | `74a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b` |
| release_v6/code.candidate.tar | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |

Evidence: `narrow_evidence/verify.py`, `production.diff`, `unittest.txt`, `results.json`. Initial review-harness failures were incorrect summary-schema and historical NumCPUs assumptions, corrected only in the review harness; they were not production failures. Final verification passed.

No implementation changes, remote actions, transfers or jobs occurred. No broader model, runtime or financial review was repeated. Review artifacts are under deployment_v4. During initial orientation the required graft command reported an automatic graph refresh; no explicit graph rebuild was requested. Advisor consultation was unavailable because of client/model incompatibility and provided no review result.
