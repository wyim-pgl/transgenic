# STOP — v7 submission blocked: the release payload is namespace-bound to v6

2026-09-21 09:00 PDT. Author: Won <wyim@unr.edu>.

**No job was submitted. No control namespace was created. No authorization
artifact was created. No attempt journal was created. Nothing was written to
remote storage during this gate pass.** The only remote operations were read-only
queries.

I stopped at gate 5 because gate 5 and gate 6 cannot be satisfied as specified.

---

## Gate results

| Gate | Result |
|---|---|
| 3. Live account balance | **PASS** — see below |
| 5. v7 control namespace + canonical v7 authorization artifact | **BLOCKED — not attempted** |
| 6. Host checks | **BLOCKED — cannot be meaningful against this payload** |
| 4. Pre-submission live query | not reached |
| 7. Preservation confirmation | v6 still untouched; nothing done that could change it |

## Gate 3 — PASS (measured, not extrapolated)

`accounts` on the actual host, 2026-09-21 08:58 PDT:

```
Account                        Balance(Hours)   Deposited(Hours)
bilv-dtai-gh                              1068              1499
```

Requirement recomputed in the draft: `264 + 20 + 20 = 304`.
**1068 ≥ 304 — headroom is adequate.**

Corroboration worth noting: the 2026-09-17 record cited 1154 hours. 1154 − 1068 =
**86 hours**, which independently matches the 86.3833 GPU-hours I computed from
the two COMPLETED B5 jobs' elapsed times. Two unrelated measurements agree, so the
consumption arithmetic in the draft is sound.

`sreport` for the account since 2026-09-01 reports 6875 CPU-hours used
(0.03% of cluster), consistent with B5 activity. Raw capture:
`v7_account_balance.txt`.

## The blocker

The v7 namespace was staged from the transport byte-for-byte, exactly as
authorized. That is precisely the problem: **the release payload hardcodes the v6
namespace**, and copying it to a new path does not rebind it. Four pinned
members carry v6 path literals:

| File | Line | Literal |
|---|---|---|
| `admission.py` | 22 | `NAMESPACE = '/work/nvme/.../staging/dev123_p1_release_v6_20260917'` |
| `p1.sbatch` | 13 | `#SBATCH --chdir=/work/nvme/.../staging/dev123_p1_release_v6_20260917` |
| `controller_execution_scope.json` | 56 | `"namespace": "staging/dev123_p1_release_v6_20260917"` |
| `authorization.example.json` | 44 | `"namespace": "/work/nvme/.../staging/dev123_p1_release_v6_20260917"` |

### What would happen if the staged v7 `p1.sbatch` were submitted

1. `--chdir` sends the job into **`dev123_p1_release_v6_20260917`** — the
   preserved incident-evidence tree — not into v7. The v7 tree would not be used
   at all.
2. `p1.sbatch:39` then executes
   `(set -o noclobber; printf '%s\n' "$SLURM_JOB_ID" > S0_P1_STARTED)`, which
   **writes a new file into the preserved evidence tree**. That directly violates
   the preservation directive that has governed this work since 2026-09-17, and
   it is not reversible in the sense that matters: the tree would no longer be the
   untouched artifact of the stall.
3. `deploy.py host` would then run out of v6, where `source-manifest.json` is
   absent and `release.sha256` is 0 bytes, so `verify_source_manifest()` →
   `pinned(release/'source-manifest.json', ...)` fails closed. The job fails —
   but only *after* step 2 has already written into the evidence.
4. Independently, `admission.py`'s `NAMESPACE` constant pins v6, so the admission
   gate would reject v7 as an execution namespace in any case.

So the submission cannot succeed, and the failure mode is one that damages
preserved evidence and burns a GPU allocation to do it.

### Why gate 6 is also blocked

Re-running the host checks against this payload would validate a package that is
bound to the wrong namespace. The interpreter check (3.6.15 / `Popen(text=True)`)
and the `CPUs/Task` parser check would pass or fail on their own merits, but a
green result would be misleading: it would certify a package that cannot legally
execute from v7. Running them now would produce exactly the kind of green-looking
partial pass the instructions forbid.

## Why "Option A" did not and could not fix this

My earlier recommendation of Option A was correct as far as it went — it produced
a complete, verified 14-file tree without touching the evidence, and that part
stands. But I scoped it as a *staging completeness* problem, and it is also a
*namespace binding* problem. Staging the identical bytes at a new path closes the
first and not the second. I should have caught the binding when I recommended it;
I did not read the payload's path literals until preparing this submission. That
is my error, and it is better surfaced now than after an sbatch.

## What is actually required

Rebinding those four files changes their content, therefore their sha256,
therefore the ledger `release.sha256` (`74a740ea…f70d9b`), therefore
`release.json` (`64a547b8…f4f7d4`), therefore the transport tar
(`26c1dc0a…27628a`), therefore **every identity pin in the authorization**. This
is not a staging fix. It is a new release build — a genuine v7 package — and it
requires:

1. A rebuild producing new `admission.py`, `p1.sbatch`,
   `controller_execution_scope.json`, `authorization.example.json` bound to
   `dev123_p1_release_v7_20260921` (or to whatever namespace is chosen), with the
   `--job-name`, `--output`, `--error` and `--exclude` fields updated at the same
   time — the exclusion list `gh062,gh093` in the current file is stale, and the
   job name and log paths still say `v6-0917`.
2. New `release.sha256`, `release.json` and transport tar, with new pins.
3. Re-running the controller test suite (47 tests) against the rebuilt package.
4. **Fresh independent review of the rebuilt package.** The review currently in
   flight is reviewing a package that cannot be submitted, so its verdict — clean
   or not — does not carry over.
5. A revised execution decision re-pinning the new identities. The account facts
   and caps in the current draft remain valid and can be carried forward; only the
   identity pins change.

The currently staged `dev123_p1_release_v7_20260921` tree should be left in place
for now and **not** submitted from. Whether it is superseded by a rebuilt package
at the same path or at a new one is an author decision; overwriting it would
itself be a write to a verified tree.

## Unchanged

`dev123_p1_release_v6_20260917` remains untouched preserved evidence.
`3163986` remains terminal and was never cancelled or replayed. B5
`3149358`–`3149363` remain pending and untouched. No `sbatch`, no `srun`, no job
of any kind was submitted, and no submission command was prepared for execution.
