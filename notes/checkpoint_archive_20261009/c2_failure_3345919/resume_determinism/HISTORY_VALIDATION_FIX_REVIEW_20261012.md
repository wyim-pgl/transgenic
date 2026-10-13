# Validation400 receipt fix review — 2026-10-12

**Spec compliance: Approved for the validation extension and receipt producer. Quality: Approved. Prior Important finding is ADDRESSED in `/tmp/c2-validation400-receipt-fix.diff`.**

- `history_proof.py:472`: `_pinned_manifest(root)` now reasserts the current validation bootstrap against the immutable original receipt and committed draw-prefix evidence before receipt publication. Rebinding manifest/journal/meta/report hashes cannot substitute a different original checkpoint.
- `history_proof.py:476`: current continuous/resumed history SHA equality is now required independently of report.history_equal or the report's updated declared hashes. Each current journal still undergoes strict checkpoint/schema/fingerprint/specialization validation, and declarations must still match those actual hashes.
- `test_history_proof.py:452–516`: the added history_redeclared case changes a journal, updates its metadata, changes the report's history hash and rewrites comparison.json. The manifest case changes original checkpoint provenance and rebinds all three journals/metadata plus both report declarations. Both must raise and leave no validation_receipt.json; fixtures are restored afterward. These regressions reproduce the reviewed false READY paths rather than failing only because old declarations were stale.
- The two production assertions are local, small checks and introduce no extra model/optimizer read, source/cache policy change, scientific tolerance or public interface change. Original historical210, event6718, whole-checkpoint equality, strict predecessor and100GiB gates remain active.

Evidence: reviewed the complete narrow fix diff once; parent reports separate RED reproductions (history8.459s and fully rebound manifest9.994s), then full18 focused tests passing56.437s and diff-check success. No tests were rerun by this reviewer. No new correctness finding in this scoped repair.

Approval concerns implementation/staging, not an actual scientific pass. Failed initial proof3369338 remains disqualified as a validation predecessor. A newly reviewed/source-bound initial allocation must satisfy all initial gates before validation400; four-rank execution, historical equality, committed400 resume, source/job/receipt binding and final4099 completion remain required operational gates.

Review remained read-only except this report: no source edit, commit, remote command/job or spawned agent.
