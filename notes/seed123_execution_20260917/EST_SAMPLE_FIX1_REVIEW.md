# Independent I1–I3 fix re-review — 2026-09-17

## Verdict: original source findings closed; no new blocking source issue found

Scoped to `/tmp/est-real-source-binding-20260917`, HEAD `fca911e58fde44124b9359f9adc03a9d20059599`. This is **not deployment approval**. Full package closure, isolated runtime identity, before/after wrapper verification and controller acceptance remain the separate preparer's work. No actual large-source compute job, remote command, source/index/HEAD modification, graph rebuild or subagent was run. All executed readers used tiny synthetic local files.

Reviewed the original `EST_SAMPLE_REVIEW.md`, original adversarial fixtures, current `tasknotes/real_sample/REVIEW_FIXES.md`, current binder implementation and new regression cases. Binder/test are untracked at this HEAD; git cannot supply their original-version diff. Changes were assessed against the original finding spans, recorded old hashes, current code and regression behavior, not misrepresented as a committed diff.

## Findings disposition

- **I1 closed.** `est_real_sample.py:278–295,339–345` requires explicit absolute-path SHA256-pinned provenance/commands specs, hashes their files and rejects mismatches, then uses the **verified** SHA256 values in retained search preimages. HSP preimages inherit the search ID (`:443–445`). Independent tests vary provenance and commands separately, retain raw/query/protein bytes, and confirm changed orthodb search **and HSP** IDs with unchanged SwissProt IDs. Fresh/expected-hash flags and retained digest preimages agree.
- **I2 closed.** `:507–521` attempts group SIGKILL on abort regardless of leader liveness, tolerates missing groups, catches cleanup exceptions and performs bounded reaping. Original live-descendant bad-outcome assertion now fails because the child no longer exists. Focused tests also establish cleanup-error receipt publication and preserve the real reader return code. Denied kill/uninterruptible processes are reported as errors rather than guaranteed dead; no universal OS-level cleanup guarantee is claimed.
- **I3 closed.** `:211–250,399–405,522–542` multiplexes both pipes with main-thread selectors; there is no stderr thread or shared Budget race. Retains at most 64 KiB exact base64 bytes, observed byte count/hash, truncation, observed EOF and reader exit status. Retention/output reservation precedes spawn; consumed stderr charges read usage. Abort drain is nonblocking/time-bounded; incomplete capture does not claim stream EOF. Large binary stderr does not deadlock valid SAM, and nonzero exit remains FAILED even after both pipes reach EOF. Empty stderr and unterminated final SAM line succeed. Read/retention/output cap controls remain fail-closed.

### Important original-test interpretation

The unchanged original ten-case adversarial suite gives **2 failed, 8 passed**, not three failures: identity collision and surviving descendant bad-outcome assertions are now rejected; all seven unchanged boundary controls pass. The third original test, `test_reproduce_stderr_loss`, still passes because it searches for a **literal raw byte string in serialized files**, whereas the fix deliberately stores exact bytes as **base64**. This assertion is encoding-blind and no longer measures loss. It must not be advertised as evidence that the defect persists or that all three unchanged tests failed.

The original test file was preserved. `est_sample_fix1_review_evidence/test_independent.py` reuses the exact split-string error-emitting helper and checks decoded bytes, full observed SHA256/count, EOF, no truncation and exit 7. The original missing-error **outcome** is rejected. No source adjustment is required for this test-harness limitation.

## Executed evidence

Local interpreter `/tmp/mvp-final-source-env/bin/python` (3.12 environment); `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`, no installs.

- `focused321.txt`: **321 passed in 30.39s**, under a 90-second timeout. Exact ten-file set from `REVIEW_FIXES.md`/original 309-case baseline, including all **43 current binder cases**.
- `original_adversarial.txt`: **2 failed, 8 passed in 6.10s**, expected bad-outcome rejection and encoding caveat above; seven boundary controls unchanged.
- `independent.txt`: **5 passed in 3.59s**: separate provenance/commands identity binding, original stderr payload decoded, successful stdout EOF without newline/empty stderr, and nonzero exit after pipes close.
- `before.json` and `after.json`: byte-identical source/test/tasknote hash inventories, HEAD, status and index hash. `cmp` exit 0; `git diff --check` exit 0. Existing dirty/untracked worktree state preserved.
- `loaded_origins.json`: local test import origins and file hashes plus bound callable module attributes. This supports which local source was exercised, **not** deployed loaded-code attestation or full dependency/runtime closure.

All proofs are under `notes/seed123_execution_20260917/est_sample_fix1_review_evidence/`. Existing success, source mutation, selected EOF, replay/geometry, identity, caps, marker hashes and scientific gates were exercised by the complete focused suite and unchanged boundary controls. COMPLETE remains diagnostic traversal only; A11 remains unassessable and training/full-C0/route-A remain false.

Verified current file SHA256s (also in inventories):

```
93ea442d0dc60c1d8df2ee0b479361d89a188f9b43bba4327740f896efec27b2  revision/scripts/evidence/est_real_sample.py
5920616a012ae1cdb2dd941dbafc5752b9643eaef564230597ab0ad9488044dc  tests/test_est_real_sample.py
7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c  tasknotes/real_sample/sitalica.manifest.json
```

Manifest unchanged. Implementer's historical eight-red-test claim was read, not independently replayed against an old source snapshot. Advisor consultation failed due client/model-version incompatibility; no advisor endorsement is claimed. Graft-first orientation/query/skeletons reported approximately 876,968 tokens saved.

**Next gate:** separate preparer/controller verifies the immutable package and actual runtime before/after execution; only then may the authorized bounded actual-source run and its independent result acceptance proceed. This report alone authorizes neither deployment nor biological acceptance.
