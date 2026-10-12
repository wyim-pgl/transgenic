# Independent refreshed-delta and bounded execution review

## Decision: APPROVE one bounded reconciled C1 execution, subject to the unchanged startup gate

Exact approved revision:

- Manifest SHA256: `c8b03a35bcf527b70d69fa4c11eb0ab2d90da1da3dd0b487d4dd85c6bfe00a2e`
- Wrapper SHA256: `763d655c2043b20bbe9882411589f6542112c8a65018e7eda638042a04080933`
- Code commit: `f6bef565f0bb432256b7e596a81b896afd2266c5`

No blocker found for this exact one-shot C1 audit proposal. This is independent review approval, not a claim of deployed bytes, live remote quiescence, completed C1, scientific acceptance, or authorization for C2/training. Staging/submission remains a separately authorized controller action.

**The fixed queue expires nominally at 2026-09-17 02:57:58.018983 UTC.** The compute-host startup check must still enforce nonfuture age ≤3600 seconds; clock differences can shorten the usable window. No extension, self-row filtering, refreshed-in-place manifest, or freshness waiver is approved. Missing the window requires a newly reviewed immutable revision, not reuse of this approval as current evidence.

## Independent checks performed

Read-only local assertions at **2026-09-17 02:05:10.890641 UTC** independently established:

- All **28 SHA256SUMS entries** match, with exact bundle-file coverage excluding SHA256SUMS itself. This independently repeats, rather than merely trusts, the reported controller checksum/syntax result.
- All **19 manifest snapshots**, all captured snapshot structures and all **32 receipts / 489,199 decoded bytes** equal the prior independently reviewed facts. Strict base64 decoding, individual receipt sizes and SHA256 digests match. Errors remain empty; parsed saved static stdout equals static_facts.json. The saved remote capture program is byte-identical to the prior reviewed program. There is no static exception/content/ownership/listing delta requiring another full 32-file semantic review.
- The new manifest becomes exactly equal to the old manifest by restoring only its old queue SHA256. Entries, schema/root, accounting and queue path are unchanged. Both new capture hashes resolve to the supplied bytes. Recovered pilot submission receipt is byte-identical to the reviewed prior copy; it remains a recovered local submission record, not new remote evidence.
- All four bundled runtime/template source files match the prior pins and `git show` bytes at the reviewed commit. No new semantic code change is approved; the previous scoped 191-test result was not rerun or represented as a new test execution.
- Historical accounting is byte-identical to the old capture and saved accounting stdout, with exactly **30 unique rows**. Separate baseline accounting records 6165461 allocation/batch FAILED 1:0 and extern OUT_OF_MEMORY 0:125. Fetched baseline has **18 PASS unit manifests**, rollup FAIL, no failed units, and 12 failed tree checks confined to pilot log binding and dispositions. This is baseline evidence, not a substitute for the new complete audit.
- Queue JSON preserves all **68 rows** of saved queue stdout. Every row satisfies the unchanged three-field validator and its historical-job-prefix / `blastx`, `lineage`, `est_` name predicates; neither 6165461 nor 6165521 appears. Local age was **432.872 seconds**, within the gate at review time only.
- Transport records have successful return codes and sequential local start/end ordering. The actual saved queue command is `squeue --noheader --format="%i|%T|%j"`, without user/job/state filters or a postprocessing pipeline. Collector code copies its entire stdout into queue.json: no hidden application-side row removal was found.
- Saved path-check stdout agrees with pathcheck.json: proposed destinations absent, observed ancestors not symlinks. Wrapper pins agree with actual local input/code bytes and the captured resolved interpreter identity. `bash -n` passes.

## New wrapper review

The exact wrapper requests one node/task, **8 CPUs, 16G, 12 hours, no-requeue**, rejects restart count, validates canonical R, verifies nine code/input/interpreter pins, and validates code/inputs/logs directory locations. It rejects preexisting guard/audit/scratch/exitcode and claims an atomic persistent `execution.guard` directory. Two cooperating launches cannot both acquire that guard. The guard is deliberately not removed; this is not restartable. After acquisition, noclobber and the EXIT trap record batch status; failures before guard/trap acquisition need not produce batch.exitcode and must be diagnosed from scheduler/log evidence.

Scratch/output are separate from historical/source/input trees. Isolated Python `-I -B`, cleared PYTHONPATH/PYTHONHOME and disabled user-site/bytecode reduce import/write contamination. Runtime code does not consume baseline PASS manifests to skip work. Reviewed auditor `code/revision/scripts/evidence/est_blastx_audit.py:738–832` performs the two pilot and sixteen remaining unit audits, requires 18 units, and repeats historical snapshot and producer-wrapper/log bindings at completion. These final checks do not pretend to rehash every large unit payload a second time. The unchanged queue check at `:554–625` occurs at audit startup, not after the multi-hour audit.

Deployment must still exclusively create the new R without overwrite/reuse, stage only the four code files and prescribed inputs plus wrapper/log directory, make code/inputs read-only, verify deployed hashes including the exact wrapper, and refresh live queue/path availability before submission. A live conflicting job blocks proceeding even though the saved capture passed. The wrapper's own subsequent `est_` queue row is not a reason to filter any new capture. The wrapper does not itself reserve the historical tree, prove absence of external writers, or hash its own Slurm spool copy; those are controller/provenance boundaries, not protections to infer from execution.guard.

## Trust boundaries and retained limitations

Unsigned SSH transport and timestamps rely on the operator's honest capture and normal remote command environment. A command without explicit filters plus intact stdout establishes no application-side filtering; it is not cryptographic attestation against shell wrappers, Slurm visibility restrictions or a fabricated transcript. Likewise interpreter executable hashing does not pin all libraries/stdlib. No new inconsistency suggesting such substitution was found.

Remote static/path timestamps and local transport/queue timestamps are different clock domains with unmeasured offset. Captures are sequential but non-atomic; absence observations are not reservations, and empty historical sentinels are not proof that a flock is unheld. Exclusive deployment and final live controller checks remain mandatory.

An expired/invalid reconciliation deliberately fails while full unit auditing can still consume hours. No fallback acceptance is authorized. Extern OOM/clean-accounting, authoritative ancestry/merged-taxid limitations, phase/scientific acceptance, C2/C3 and production authorization remain unresolved/false as specified; these are not blockers to running this narrowly bounded diagnostic C1 audit and must not be promoted by a C1 PASS.

No SSH, scheduler action, source edit, collector/verifier execution or subagent was used. Only this review file was intentionally written; graft reported automatic graph refresh during orientation. Graft estimated 841,852 tokens saved.
