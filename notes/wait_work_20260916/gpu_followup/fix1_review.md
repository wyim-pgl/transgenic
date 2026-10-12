# F1 focused re-review

**F1 ADDRESSED. No new fix-induced breakage found. No job approval.**

- `followup.py:90-91` now requires exact `int` and `0 < timeout_seconds <= outer_seconds - 120`, rejecting nonpositive caps and booleans while retaining the 120-second reserve.
- `test_followup.py:97-115` adds a helper and eight tests: reject 0, -1, True, False, 1.0 and 481; accept 1 and 480 for outer=600. Rejections require no output directory; accepted cases check the actual runner cap.
- Independently verified current `followup.py` SHA256: `690b7f636b60625da39b13e9ec7a796154dcd9f9cf947f28e3776ff08c94996b`. Reversing only the predicate in memory reproduces the original reviewed source hash; removing only the new helper/eight tests reproduces the original test hash. Thus the source/test delta is exactly scoped to F1.
- Read existing `SHA256SUMS`; it records the new source hash and test hash `f354b57092e867fbdeb48ba79c442c28e80156b9248b9ee71e9855e9a2ee5609`, plus the report/original-review entries. Accepted controller-reported 26 tests PASS in 1.119s, bash syntax and SHA-ledger PASS; did not rerun the suite or ledger.

**G1 and G2 remain blocking:** actual reviewed mount/container argv, completed P1 evidence, current accounting/no-overlap and real resource-budget authorization remain missing. This correction does not establish any GPU PASS or authorize submission/execution.

Only this separate report was written; original review and hash ledger were not changed. No source edits, SSH, jobs or subagents. Graft reported approximately 25,787 tokens saved this turn.
