# Independent C1 reconciliation review

## Verdicts

- **Code correctness: CHANGES REQUESTED (two P2 findings).** The closed reconciliation design is substantially implemented, but malformed evidence can abort before the promised 18-unit inventory, and final revalidation omits pilot/array producing wrappers and log bindings.
- **Execution readiness: NOT READY.** No operational reconciliation manifest is approved. Gather the authorized SMALL source facts, fix/review the findings, and independently review the real manifest/captures and a new immutable invocation before execution. This review does not authorize submission or alteration of baseline 6165461; its current running/queued/completed state is unknown here.

Reviewed `/tmp/b6-p2-phase-1535433`, HEAD `21456324923c2c84610bd63e8b6ae868eae2f535`, implementation `06835580c18905fe28f38eb43f28596b9bde06b4`, base `5c7a0aa`. Read the complete supplied review diff (including its truncated continuation), PLAN, REPORT, main operations note, actual recovered pilot receipt, relevant historical stat/wrapper/log inventory and receipt/runtime capture. File:line references below are to this worktree's `revision/scripts/evidence/est_blastx_audit.py`.

No SSH, scheduler calls, source edits, subagents, merge, push or broad test rerun. Focused reproductions used existing fixtures and disposable local temporary trees, with bytecode disabled. The controller's 175-test result is supplied evidence, not independently repeated. Graft ran first; it reported an automatic graph refresh, with no tracked graph/source changes shown by git status. Advisor consultation failed due to client-version incompatibility; no advisor assessment is claimed.

## Findings

### R1 — P2: malformed reconciliation evidence bypasses the inventory-preserving error boundary

**Locations:** `est_blastx_audit.py:511–515, 559–560, 622–624, 745–748`.

`load_reconciliation()` catches `OSError`, `ValueError`, and `UnicodeError`, but external parsers also raise `csv.Error` and `RecursionError`. Those escape before any units are audited. This violates the explicit “malformed input preserves unit audit” contract; the output has FAILED.json but neither rollup nor 18 unit reports. It is a fail-stop/inventory-loss bug, not a false PASS.

**Focused reproductions, both confirmed:**

1. Start with `test_c1_reconciliation.manifest_fixture()` and the existing `synthetic_identity` fixture. Keep primary sacct valid. Point the manifest accounting capture to an external file containing `JobID|State|ExitCode\n` followed by a JobID field of `csv.field_size_limit()+1` characters and `|COMPLETED|0:0\n`. Recompute that capture's binding and the manifest digest. Call `audit()`.
2. With the same valid synthetic tree, replace the external manifest bytes with `'[' * 1100 + '0' + ']' * 1100`, compute its digest, and call `audit()`.

Observed output:

```text
csv_field Error field larger than field limit (131072) unit_reports 0 rollup False
deep_json RecursionError maximum recursion depth exceeded while decoding a JSON array from a unicode string unit_reports 0 rollup False
```

**Minimal fix:** translate these specific external-parser exceptions to an ordinary failed `historical_reconciliation` check and return `{}`. Keep programming errors distinguishable; do not broadly suppress every exception. Add integration regressions asserting FAIL, rollup present, and all18 unit JSONs for both malformed inputs.

### R2 — P2: a producing wrapper can change during unit hashing and still yield PASS

**Locations:** `est_blastx_audit.py:591–593, 664–707, 780–787`.

The final loop rechecks only the 19 eligible target snapshots. For pilot and remaining8 these cover sentinels/scratch, not their `run.sbatch` or required logs. Their wrapper/log checks occur before the expensive unit audit. Consequently the final PASS can retain a successful historical producer binding after the live producing wrapper has changed. Preflight does not have this gap for its wrapper/logs because its full root is recursively snapshotted.

**Focused reproduction, confirmed:** start with the existing valid `manifest_fixture()` and synthetic identity anchors. Wrap `audit.audit_unit` so that its first invocation writes `changed after binding\n` to the synthetic pilot `run.sbatch`, then calls the original unit function. Invoke normal `audit()` with the pinned recovered receipt and valid manifest digest. No manifest/anchor is altered after validation.

```text
late_wrapper status PASS units 18
```

The same structural omission applies to the pilot/array required log set; no separate log-mutation reproduction is claimed.

**Minimal fix:** retain and recheck the small pilot/array producing-wrapper pins and receipt-backed required log inventories/hashes before final rollup, including regular-path/no-symlink checks. On change, add a failing tree check while retaining all unit reports. Do not rerun large unit hashes, demand a continuous scheduler lock, or automatically reject a long audit merely because its original queue capture has aged. Add the late-wrapper regression alongside the existing late-sentinel regression.

## Checks that are sound within the documented trust boundary

- **Exact allowlist:** independently enumerated 19 paths from `reconciliation_paths()` (`483–496`): three pilot sentinels/scratch, remaining8 submission sentinel plus eight species tmp paths, and seven preflight root/container/sentinel paths. No wildcard/user extension. Array mapping is alphabetical, consistent with the report; the historical inventory's traversal order is not task order.
- **Binding:** root pin and canonical external capture paths; whole manifest digest and capture hashes; fixed wrapper identities and fixed job lists (`499–508`, `554–624`). Snapshot equality binds recursive names, file hashes, kinds, numeric UID/GID (`462–480`). This is ownership equality to reviewed evidence, not proof of live lock ownership.
- **Markers and narrow dispositions:** snapshots reject links/special files, FAILED-prefixed entries, malformed DONE and unknown DONE variants. Scratch/xdg-cache/pilot execution directory/preflight submission directory must be empty; submission jobid bytes and preflight empty regular execution.lock are checked. Only NEITHER is reconciled; historical DONE is not inferred or written (`603–618`, `644–654`).
- **Preflight group:** all seven entries required together. Exact root/log/receipt/source-unit listings, two lineage pins, source DONEs, provenance `lineage_sha256` agreement, jobid6163397, successful submission receipt and batch exit are required (`527–551`, `601–607`). Root snapshot recursively binds the remaining provenance/receipt bytes; the code does not independently validate every provenance field's meaning. Those bytes still require controller review. Known root names, five receipt filenames, lock kinds, wrapper pin and lineage pins match inspected historical captures.
- **Pilot anchor:** independently hashed both main companion receipt and test fixture: each is `f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23`. Actual records540/541 bind session source, call/result IDs, explicit historical D/chdir/output/error command and returned6163851. Implementation requires this anchor, historical root, exact submission jobid, producing run.sbatch pin and both named logs; empty stderr is valid (`684–705`). The original session itself was not reread; this is acceptance of the reviewed recovered extract, not scheduler authentication.
- **Queue/accounting:** timezone-aware queue capture age is checked at initial validation, at most1hour, with exact command/result structure. Known historical IDs and BLASTX/lineage/est_-named queue entries are rejected regardless of state. Accounting binds exact allocation/.batch COMPLETED0:0 and `.extern OUT_OF_MEMORY0:125` rows to each entry's fixed jobs; primary sacct is cross-checked for search jobs, separate capture supplies preflight. Active/unknown extra states and duplicate bound rows are rejected (`511–524`, `566–597`). Freshness is a start-time gate, not a final-time or continuous quiescence assertion.
- **Long-running audit:** disposition-use snapshot rechecks and final target snapshot rechecks exist and detect persistent target mutations. R2 identifies the omitted producer dependencies, not absence of all revalidation.
- **Semantic separation:** final rollup keeps `clean_accounting`, `accepted_phase_evidence`, and `production_authorized` false and preserves the unresolved extern note (`794–797`). Neither report nor code resolves the OOM root cause or ancestry authority.
- **No destructive reconciliation path found:** new reconciliation logic only reads historical inputs. Output remains exclusively created outside the audited tree, and temporary work stays outside it. No sentinel deletion, DONE fabrication, subprocess submission, or execution of recovered receipt commands was added. This is not a claim of atomic protection against an actively adversarial filesystem writer.

## Readiness requirements, separate from code fixes

1. Obtain numeric UID/GID and exact recursive small-file hashes/listings for the real19 entries, especially missing preflight receipt/root-file bindings. Do not substitute current synthetic snapshots or guessed metadata.
2. Independently review and pin the external real manifest, accounting (including6163397), and fresh operator queue capture. Preserve the operator-trust boundary: signed/cryptographic Slurm proof is not requested.
3. Review a new immutable deployment/invocation that actually supplies the new receipt/manifest flags; the unchanged baseline sbatch template does not pass them. Do not patch the deployed baseline snapshot or presume job6165461 has finished.
4. Retain extern and scientific gates as unresolved even if a later real C1 reconciliation passes.

Graft reported approximately944,296 tokens saved across this review's calls (tool estimate).
