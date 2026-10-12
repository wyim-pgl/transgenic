# Task 1 independent spec and quality review

Review scope: `4aac577..9ecd8a2`, local transaction core and its real-file tests only. Read the supplied fixed diff once; recovered its truncated transfer/cleanup portion using the graph's exact source spans. Reviewed SPEC.md, Task 1 of PLAN.md, and task1-report.md. No code, tests, git state, production files, or jobs changed. No subagents dispatched.

## Decision: Needs fixes

No Critical findings. Two Important findings prevent approval.

### Important 1 — Incomplete epochs displace the latest completed checkpoints from protection

**Locations:** `revision/scripts/b6/checkpoint_archive.py:210-215` (`plan`), `revision/scripts/b6/checkpoint_archive.py:239-245` (`_eligible`).

Both paths derive the latest two from all names matching `epoch_[0-9]+`. `plan` computes this set before `_complete` refuses incomplete candidates; `_eligible` never checks completeness. Therefore newer incomplete directories count toward retention and expose an older checkpoint that is still one of the two latest **completed** checkpoints. This contradicts SPEC's explicit latest-two-completed protection. The report's characterization of counting malformed names as conservative is reversed for retention safety.

**Focused real-file reproduction:** Start with the supplied fixture's complete epochs 01–04 and best→01, then create an empty epoch_05. `plan` returns candidates `[2, 3]`. `prepare(run, 3)`, `transfer(...)`, and `cleanup(...)` all succeed; cleanup reports `removed` and epoch_03 no longer exists. Since epoch05 is incomplete, epochs03 and04 are still the newest two completed epochs and03 must remain protected.

**Required correction:** Compute retention from validated completed epochs, retaining/refusing incomplete entries separately. Share the same protection definition across planning and every cleanup recheck, including tombstone recovery, so an in-progress or malformed newer epoch cannot weaken retention. Add focused coverage for both a preexisting incomplete epoch and one introduced between prepare and cleanup; preserve the latest two completed epochs in each case.

### Important 2 — Interruption while writing a staging receipt permanently blocks retry

**Locations:** `revision/scripts/b6/checkpoint_archive.py:68-76` (`_save`), `revision/scripts/b6/checkpoint_archive.py:481-486` (`transfer`).

`_save` durably writes `.archive_receipt.json.tmp` before replacing `.archive_receipt.json`. A process death or replace error in that interval leaves the temporary receipt inside the transaction stage. On retry, `transfer` verifies the stage at line481 before rewriting the receipt. `_verify_bytes` excludes only the final receipt, so the implementation's own temporary file triggers `archive content mismatch`. Every subsequent retry fails at the same point; there is no supported recovery action. Source preservation works, but the required recoverable interrupted transfer does not.

**Focused real-file reproduction:** Intercept only the `os.replace` whose destination ends in `/.archive_receipt.json`, raising `OSError` after `_save` has written and fsynced the real temporary receipt. Restore normal operations and call `transfer` again. Retry raises `ArchiveError: archive content mismatch`; `.stage-<manifest_id>/.archive_receipt.json.tmp` remains and the source is intact.

**Required correction:** Safely recognize and recover the exact transaction-owned receipt temporary file before stage content verification, without relaxing rejection of unexpected files, symlinks, or conflicting published archives. A focused interruption/restart test should cover this boundary.

## Spec compliance otherwise

The core provides the requested plan/prepare/transfer/verify/opt-in-cleanup APIs with JSON manifests; bounded file streaming; independent source/destination hashes; canonical nonoverlapping roots; source identity and file signature binding; persisted receipts; destination-native staging; persistent ordered flock locks; storage callback checks; fsync barriers; conflict refusal; and resumable source tombstones. Cleanup locally revalidates the durable destination and repeats current source/protection checks. The published-archive forwarding mode independently verifies prior receipts and preserves provenance, consistent with the clarified spec.

The callback verifier is treated as the expressly documented trusted authenticated transport boundary. The absence of SSH/CLI/watcher integration or a received-stage publication API is outside Task 1 and is not a finding. Host Python3.6 execution and Lustre behavior remain Task 3 verification; the inspected core uses Python3.6-compatible standard-library constructs.

The supplied 31-test suite and reported SIGKILL coverage exercise important boundaries but miss the two cases above. I did not duplicate the full suite. I executed only the two focused real-file probes described above; both completed successfully as reproductions. No additional Minor findings are necessary for this review.

## Context tooling

`graft map` reported 1,467,331 estimated tokens saved; `graft skeleton revision/scripts/b6/checkpoint_archive.py` reported 6,017. Total reported estimate: **1,473,348 tokens**.

## Scoped re-review round 1 — b80c638

Scope: only `9ecd8a2..b80c638`, the prior two findings, and breakage introduced by those fixes. Read task1-fix1.diff once and the implementation report's fix appendix. No broad-suite rerun, source/test changes, or additional graph/source exploration.

**Verdict: Needs fixes. Finding 1 resolved; Finding 2 remains Important.**

### Finding 1 resolved

The shared `_classified_epochs` / `_protected_epochs` definition now derives retention from validated completed epochs and rejects duplicate numeric names. Planning and current protection checks use the same rule. Recovery includes the receipt-bound tombstone epoch in the retained set. The three focused regressions for incomplete epochs at planning, cleanup, and tombstone restart all passed.

### Finding 2 partially fixed, still unresolved for interruption during receipt writing

**Locations at b80c638:** `revision/scripts/b6/checkpoint_archive.py:68-76` and `revision/scripts/b6/checkpoint_archive.py:398-409`; recovery is invoked at `revision/scripts/b6/checkpoint_archive.py:517`.

The fix successfully handles interruption after the full temporary receipt has been written, including the originally demonstrated failed replacement. That focused regression passed. However, `_save` first creates/truncates the temporary receipt and only then writes/flushes/fsyncs its JSON. A process death anywhere before the full JSON has reached the file can leave empty or partial JSON. `_recover_receipt_temp` rejects it permanently, so an ordinary write interruption still prevents all subsequent retries.

**Focused real subprocess evidence:** In a disposable real-file fixture (`TMPDIR=/dev/shm`), intercept `os.open` only for the exact staging `/.archive_receipt.json.tmp` path, perform the real open, then SIGKILL the process before returning the descriptor. The child exited `-9`; the temporary receipt existed with size0. Two normal unmodified `transfer` retries both raised `ArchiveError: invalid temporary archive receipt: Expecting value: line 1 column 1 (char 0)`. The original source bytes remained intact. This is a valid crash point in the production write sequence, with no manual file corruption involved.

The implementation report's deliberate refusal of truncated temporary receipts preserves data safety but does not meet the existing interruption/retry requirement. It therefore does not close the finding. The correction should make the transaction's own partially written receipt recoverable while retaining refusal of unrelated files and symlinks. This may require changing how an unpublished receipt is written or establishing ownership independently of the receipt's completed JSON; it does not require trusting malformed JSON or modifying published archives.

**Validation performed:** Four existing focused tests passed in0.081s (planning retention, cleanup retention, tombstone retention, and complete-temporary-receipt restart). One additional disposable SIGKILL probe reproduced the unresolved failure as above. No test files were added and no full suite was duplicated. No new unrelated findings. Additional graft calls:0; prior tally unchanged.

## Scoped re-review round 2 — 8ff381b

Scope: only `b80c638..8ff381b`, unresolved Finding2, and new breakage in that fix. Read task1-fix2.diff once and the report appendix. Finding1 remains resolved and was not reopened.

**Verdict: Approved for Task1 core. Both Important findings are resolved.**

The new `_recover_receipt_temp` treats the exact private manifest-named staging control pathname as interrupted transaction state, refuses nonregular files and collision with checkpoint payload, and preserves the original bytes under an exclusively created UUID hardlink before removing the staging name. It checks identity/content stability, fsyncs the preserved file and control directory, and then unlinks/fsyncs the stage. A crash between linking and unlinking retains at least the original staging path; a crash afterward retains the durable control copy. Another retry can proceed safely. Empty, truncated, and unrecognized receipt bytes are never used as verification evidence. Normal independent payload/source verification and fresh receipt construction still gate publication. Existing published archives and unexpected checkpoint files remain protected.

I independently ran the seven receipt-focused real-file tests using `TMPDIR=/dev/shm python3 -m unittest discover -s tests -p test_checkpoint_archive.py -k receipt -v`: **7 passed in0.333s**. This includes subprocess SIGKILL before receipt writing and after partial JSON writing, successful retries with preserved control bytes, symlink refusal, unexpected payload refusal, and conflicting published receipt preservation. No broad suite duplicated. No additional findings introduced by this fix.

This closes the local Task1 review gate. Actual Python3.6/Lustre hardlink, pool and durability validation remains the separately planned host-verification gate; this approval does not claim those operational checks or production activation. Only this review report was modified by the reviewer. Additional graft calls:0; prior tally unchanged.
