# State of record: what is actually on disk, and one correction

2026-09-21 09:40 PDT. Author: Won <wyim@unr.edu>. Read-only verification only;
no remote write was made in producing this document.

**Headline correction: the 08:36-verified v7 staging tree is GONE.** It was
deleted and replaced at 09:12 PDT under the overwrite authorization that was live
at the time. Any report stating "both v6 and v7 are preserved" is wrong and needs
correcting with the author. My own `V8_REBUILD.md` and `v8_rebuild_evidence.txt`
said this too; both are now fixed and annotated.

---

## 1. Sequence, as it actually happened

| Time PDT | Event |
|---|---|
| 08:36 | v7 namespace created; **v6-bound** payload staged; verified 14 files |
| ~08:59 | Binding defect found; **STOP** reported; no gate 5 write made |
| — | Team lead relays author approval: *rebuild bound to v7, overwrite the v7 path* |
| 09:11–09:12 | v7 contents **deleted**, v7-bound rebuild extracted in their place |
| 09:15 | `V7_REBUILD.md` written |
| ~09:25 | `V7_REVIEW_RECONCILIATION.md` written after the reviewer's report appeared |
| — | **Four messages arrive together**, including the v8 decision and the withdrawal of the overwrite instruction |
| 09:23–09:25 | `ACCOUNTS_SHA` re-measured, I/O floor measured, v8 built, v8 namespace created and staged |

The v8 instruction arrived **after** the v7 overwrite had already been executed.
That is the whole explanation: the overwrite was authorized when I performed it,
and was withdrawn afterwards — correctly, once the `FileExistsError` mechanics
were understood, but the deletion had already happened.

## 2. Current on-disk state, verified by inode

| Path | Files | Bound to | Evidence |
|---|---|---|---|
| `dev123_p1_release_v6_20260917` | 13 | **v6** | dir inode unchanged; file inodes `…497495`–`…497508` unchanged; `release.sha256` still **0 bytes, mode 0660**; `NAMESPACE` and `--chdir` both still read v6 |
| `dev123_p1_release_v7_20260921` | 14 | **v7** | dir inode `…131208` is the original 08:36 directory, but every file inode is now `…131234`–`…131248` — **new inodes**, i.e. the originals were unlinked and replaced. Dir mtime 09:12:13 PDT |
| `dev123_p1_release_v8_20260921` | 14 | **v8** | created 09:25 PDT, inode `…131249`; `NAMESPACE` and `--chdir` read v8 |

Nothing else in `staging/` changed. `dev123_p1_v6_control_20260917`,
`dev123_p1_controller_authorization_v6.json`, `evidence/`, the `3163986` logs,
`final_p1_release/`, `p1_release/`, `receipts/` and `OWNER.txt` all retain their
2026-09-16/17 timestamps.

## 3. Is the binding-defect evidence lost?

**No — but it does not live at the v7 path.** What was destroyed was a *copy*.
The defect itself is preserved in three independent places:

1. **Remote `dev123_p1_release_v6_20260917`** — still the v6-bound payload,
   verified this minute: `admission.py` `NAMESPACE` and `p1.sbatch` `--chdir`
   both still name `dev123_p1_release_v6_20260917`, with the original file
   inodes. This is the defect, in its original bytes, untouched.
2. **Local `notes/auth_resume_20260917/gpu/deployment_v4/release_v6/`** — the
   immutable local package, never modified, and its transport
   `release_v6.tar` still hashes to `26c1dc0a…27628a`.
3. **`V7_SUBMISSION_BLOCKER.md`** — the defect documented with file:line
   literals and the failure mechanism.

So the v7 tree was never the primary evidence; it was a second copy of bytes that
still exist in two other places. The claim that needs retiring is "v7 is
preserved as binding-defect evidence", not the evidence itself.

## 4. What was deleted, precisely

At 09:12 PDT, inside `dev123_p1_release_v7_20260921` only:
the 13 files staged at 08:36 (the v6-bound payload: `admission.py`,
`authorization.example.json`, `build_release.py`, `code.candidate.tar`,
`controller_execution_scope.json`, `deploy.py`, `operator-pins.json`,
`p1.sbatch`, `release.json`, `release.sha256`, and the three `notes/` members)
plus the `notes/` directory, via
`find "$V7" -mindepth 1 ! -name '.transport.tar' -delete`.

Nothing outside that directory was deleted, then or since. No deletion has
occurred since 09:12.

## 5. Attribution of the rebind inventory — correcting my own credit

The team lead's note says I found "five more than either I or the reviewer had".
That is not accurate and I do not want it in the record.

My 13 sites exceed the lead's 8 by five. Of those five, **the independent
reviewer had four of them** (its N1, N2, N3 and the third
`controller_execution_scope.json` string inside its K5):

| Extra beyond the lead's list | Reviewer had it? |
|---|---|
| `deploy.py:264` authority filename | **yes** (N1) |
| `deploy.py:290` job-dir prefix | **yes** (N2) |
| `build_release.py:98` output dir | **yes** (N3) |
| `controller_execution_scope.json:34` authority filename | **yes** (K5, "three v6 strings, not one") |
| test suite: 6 sites / 3 modules | **no** — outside the transport, so out of its scope |

So exactly **one** category was unique to me, and it is the least interesting
one. The genuinely valuable result is the opposite of a scoop: two independent
sweeps converged on the same nine transport sites, which is far better evidence
that the surface is fully mapped than either list alone.

**On the reviewer's stated blind spot** (relative paths, env-var names, URLs,
absolute paths under unanticipated roots): every one of the four shared extras
falls in that class — none is an absolute path literal. They are all paths
*constructed by joining a fragment to a root*:

- `deploy.py:264` — `ROOT / 'staging/dev123_p1_controller_authorization_v6.json'`
- `deploy.py:290` — `ROOT / ('dev123-p1-v6-job-' + job_id)`, a concatenation
- `build_release.py:98` — `HERE / 'release_v6'`
- `controller_execution_scope.json:34` — the relative `staging/…` string

That is the transferable lesson: a search for the absolute namespace path finds
`admission.py:22` and `p1.sbatch`, and misses every constructed one. The sweep
has to be for *fragments joined to roots*, which is how I found them and,
independently, how the reviewer did.

## 6. Items the lead asked about that are already complete

- **`ACCOUNTS_SHA` re-measured** — twice, 09:18 and 09:23 PDT:
  `44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5`, equal to
  the `admission.py:23` pin. Gate holds.
- **I/O floor measured** — sha256 at **400.1 MB/s**, Lustre read 5.3 GB/s;
  ~51.53 GB of hashing inside the outer budget; 800 s available after the 900 s
  probe; **128.8 s needed, 6.2× margin**; closes even at a quarter of the
  measured rate. Full arithmetic and caveats in `V8_REBUILD.md` §8.
- **v8 is already built and staged** — 47/47 tests, all pins verified remotely.
  No further rebuild will be run, and no further remote write will be made,
  until the target namespace is confirmed.

## 7. The truncated tail (E8)

The sentence that was cut off read: *E8 resolves cleanly — the two COMPLETED B5
jobs are inside `b5_registered_ids`, so `admission.py:334-336` routes them
through the terminal branch* `need(ident in rows and (ident in known or ident in
a['b5_registered_ids']))` *and reconciles them as known terminal history rather
than as unreconciled scheduler allocations.* Live `sacct -u wcyim -S 2026-09-16`
returns exactly `3131537, 3131538, 3149356, 3149357, 3149358, 3163545, 3163986`
plus the pending remainder — every id already registered. Separately, the `spent`
loop skips `b5-seed` rows, so B5 never counts against the 5/20 diagnostic cap.
This must still be re-confirmed at submission time, because the list is fixed and
a newly submitted B5 job would be rejected.

## 8. Standing state

No job submitted. No remote write since the v8 staging at 09:25 PDT. Nothing
deleted since 09:12 PDT. v6 untouched and verified this minute. `3163986`
terminal. B5 `3149358`–`3149363` pending and untouched. Awaiting confirmation of
the target namespace before any further action.
