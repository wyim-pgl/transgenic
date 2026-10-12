# GPU release_v6 staging reconciliation, 2026-09-21 08:30 PDT

Explicit recorded reconciliation required by
`notes/auth_resume_20260917/gpu/STAGING_IO_BLOCKER.md` ("Next safety conditions").
Read-only throughout. **No Slurm job was submitted, cancelled, requeued or
relabelled. Nothing in the remote staging tree was written, copied, deleted or
re-created.** Remote clock is CDT; all times below are PDT (CDT − 2 h).

Raw evidence alongside this file:
`staging_precondition_check.txt`, `staging_file_hashes.txt`, `staging_slurm_state.txt`.

---

## 1. Safety preconditions

| Condition (from the blocker note) | Status | Evidence |
|---|---|---|
| Storage I/O must return | **MET** | `ls`, `stat`, `find` and `sha256sum` over the whole tree — including `build_release.py`, the exact file fd 4 was blocked reading — all completed within their timeouts on the first attempt. 2.8 MB hashed with no stall. |
| Owned PID termination must be established | **MET** | `ps -p 1944304` → `PID_1944304_ABSENT` (08:26:50 PDT). `ps -u wcyim | grep -iE 'python|stag'` returns only the grep itself (08:27:15 PDT). No stager, no successor reader. |

Termination is now *established*, not inferred from a tool timeout — which is
exactly the distinction the blocker note refused to concede on 2026-09-17.

## 2. Per-file verification

Expected hashes are taken from two independent sources that agree: the immutable
local transport archive `deployment_v4/release_v6.tar`
(sha256 `26c1dc0a233a95ab86977813e68310eccfce91b081de3a63586cfd7b9427628a`,
recomputed today, matches the pin) and the pinned ledger `release.sha256`
(sha256 `74a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b`,
recomputed today, matches the pin). I computed every staged hash myself.

| # | Staged path | Size | Mode | Staged sha256 | vs archive | vs ledger | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `admission.py` | 27611 | 0444 | `6f75c811…67fa4` | = | = | **VERIFIED** |
| 2 | `authorization.example.json` | 2608 | 0444 | `25e75757…4fec1f` | = | = | **VERIFIED** |
| 3 | `build_release.py` | 5688 | 0444 | `06d5cab8…154049` | = | = | **VERIFIED** |
| 4 | `code.candidate.tar` | 1873920 | 0444 | `da161088…5cbec7` | = | = | **VERIFIED** |
| 5 | `controller_execution_scope.json` | 3332 | 0444 | `55666770…9c68aa` | = | = | **VERIFIED** |
| 6 | `deploy.py` | 29149 | 0444 | `3eba7fa9…7e1543` | = | = | **VERIFIED** |
| 7 | `notes/b6_mvp_candidate_inventories_20260915.json` | 859299 | 0444 | `9f790196…880a98` | = | = | **VERIFIED** |
| 8 | `notes/b6_mvp_candidate_receipts_20260915.json` | 11786 | 0444 | `73204a4e…c807bac` | = | = | **VERIFIED** |
| 9 | `notes/b6_mvp_technical_candidate_20260915.json` | 9832 | 0444 | `faab8387…6a5b51e` | = | = | **VERIFIED** |
| 10 | `operator-pins.json` | 11828 | 0444 | `45bd5764…6d28c72` | = | = | **VERIFIED** |
| 11 | `p1.sbatch` | 2010 | 0444 | `f3b4dd15…88901a` | = | = | **VERIFIED** |
| 12 | `release.json` | 25219 | 0444 | `64a547b8…f4f7d4` | = | = | **VERIFIED** |
| 13 | `release.sha256` (the ledger itself) | **0** | **0660** | `e3b0c442…52b855` (empty-file hash) | ≠ expected 1181 B / `74a740ea…f70d9b` | n/a | **INCOMPLETE — content absent** |
| 14 | `source-manifest.json` | — | — | file does not exist | expected 12174 B / `4eb7476a…79eb0a` | ≠ | **INCOMPLETE — never created** |

Full untruncated hashes are in `staging_file_hashes.txt`.

**Result: 12 of 13 ledger entries verified byte-exact; 12 of 14 transport members
verified byte-exact.** No staged file has wrong content. There is no corruption,
no partial-content file, no symlink, and no extra or unexpected entry
(`find` over the namespace returns exactly the 13 regular files above).

Local file modes read 0555/0777 only because the checkout is on a WSL DrvFs
mount that does not carry POSIX modes; they are not evidence. The transport
archive's recorded member modes are 0444 and the 12 completed staged payloads
are 0444, which matches the authorization's "payloads readonly" directive.

Control namespace and authority, also verified by my own hashing:

| Path | sha256 | Verdict |
|---|---|---|
| `dev123_p1_v6_control_20260917/release_v6.tar` | `26c1dc0a…27628a` | **VERIFIED** — identical to the local immutable archive and to the pin |
| `dev123_p1_controller_authorization_v6.json` | `de4a4df5…9df3` | **VERIFIED** — identical to the canonical authorization pin |
| `dev123_p1_v6_control_20260917/reservation.json` | `995d8ff7…4e8dfd` | present, 544 B, `"submitted": false` |

## 3. Explicit reconciliation of the two incomplete files

### `release.sha256`
- **Should be** (local authoritative package): 1181 bytes, sha256
  `74a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b`, 13 ledger
  lines, mode 0444.
- **Is**: 0 bytes, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  (the sha256 of the empty byte string — i.e. the file was created and never
  written), mode **0660**, mtime 2026-09-17 17:46:20 PDT, inode …497508.
- **Reconciliation**: the destination inode was allocated and the file opened for
  writing; no payload bytes were ever transferred. The mode is still the umask
  default 0660, not the finalized 0444, which independently confirms the copy of
  this member never reached its chmod step. The file is a zero-length stub, not a
  truncated or corrupted ledger. It has been left exactly as found.

### `source-manifest.json`
- **Should be**: 12174 bytes, sha256
  `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a`, mode 0444.
- **Is**: absent. No inode, no stub, no partial file, no tempfile beside it.
- **Reconciliation**: the copy loop never reached this member. Nothing to preserve
  and nothing was created.

**Neither file was written, overwritten, re-copied or removed by this task.**

## 4. What the stager actually completed

The staged inodes run `…497495` through `…497508` with no gaps (`…497501` is the
`notes/` directory). In creation order that is:

`admission.py`(495) → `authorization.example.json`(496) → `build_release.py`(497)
→ `code.candidate.tar`(498) → `controller_execution_scope.json`(499) →
`deploy.py`(500) → `notes/`(501) → `notes/…inventories`(502) →
`notes/…receipts`(503) → `notes/…technical`(504) → `operator-pins.json`(505) →
`p1.sbatch`(506) → `release.json`(507) → `release.sha256`(508) → **stopped**.

That is strict lexicographic order over the 14 transport members. The next and
final member in that order is `source-manifest.json`, which has no inode. So the
completed operations are, precisely:

1. Control namespace created; `reservation.json` written 17:45:14 PDT; transport
   `release_v6.tar` landed 17:45:29 PDT with the correct hash.
2. Canonical authorization written to the staging root 17:45:30 PDT with the
   correct hash.
3. Release namespace created; members 1–12 copied in full and chmod'd 0444;
   `notes/` created — all stamped 17:46:20 PDT.
4. `release.sha256` created (inode + open) and then the write blocked — leaving
   0 bytes at 0660.
5. `source-manifest.json` never begun.
6. No job submitted; no submission-attempt journal exists in the control
   namespace; `reservation.json` records `"submitted": false`.

This is consistent with, and refines, the 2026-09-17 evidence: `/proc/1944304`
fd 4 pointed at `build_release.py` (a *re-read* during the second reconciliation
pass, member 3 in order — not the copy frontier), while fd 3 held the transport
archive. The copy frontier itself was `release.sha256`.

## 5. Slurm and account state (read-only)

- Old `3163986` is **untouched** and still terminal: FAILED 1:0, 51 s,
  `.extern` COMPLETED 0:0, cpu=288 allocated against a 64-CPU request. Identical
  to the record in `OLD_P1_RECONCILIATION.md`.
- The B5 chain has **materially changed** since the authorization was written.
  `3149356` and `3149357` ran to **COMPLETED** on 2026-09-18/19 (10:48:49 and
  10:46:56). Six jobs (`3149358`–`3149363`) remain PENDING.
  - Real consumption by those two: 21.5958 node-hours / 86.3833 GPU-hours.
  - Forward B5 reserve is now 6 × 11 h × 4 GPU = **264 GPU-hours**, not the 352
    pinned in `EXECUTION_DECISION_V6.md`.
- No MVP diagnostic job is queued or running.

## 6. Verdict on authorization

**The durable authorization is intact but not exercisable on this tree. A fresh
execution decision must be authored before any GPU submission.**

Intact:
- Every identity pin in `EXECUTION_DECISION_V6.md` still verifies bit-exact, on
  both sides: transport `26c1dc0a…`, release JSON `64a547…f7d4`, ledger
  `74a740…0d9b`, canonical authorization `de4a4df5…9df3`, source candidate
  archive `da161088…5cbec7`, prior release_v5 tar `b3dafe31…07458`.
- The authorization was **not consumed**: no submission occurred, no attempt
  journal exists, `"submitted": false`.
- No evidence of tampering, truncation or substitution anywhere in the tree.

Not exercisable, for three independent reasons:

1. **Its own pre-exec gate fails closed.** The record requires "independently
   verify remote exact 14 files / 13 ledger entries before exec". The remote has
   13 files and 12 verified ledger entries. That gate is designed to fail on
   exactly this state, and it does. Nothing here may be reinterpreted to pass it.
2. **Completing the release requires a write**, which is outside this task's
   authority and outside the blocker note's "no automatic staging retry". See §7.
3. **The queue/account facts the record pinned have materially changed.** The
   authorization itself demanded a fresh live Slurm/account query before any
   large hash or staging; that query has now been run and it contradicts the
   pinned arithmetic (86.4 GPU-hours of B5 consumed; reserve 264 not 352). Those
   numbers must be re-pinned in a new record, not silently carried forward.

The cost ledger a new record must carry forward unchanged: old `3163986` FAILED
charges 0.0141666667 node-hour / 0.0566666667 GPU-hour (never zero); prior
`3163545` cancelled 0 remains included; campaign cap 5 node-hours / 20 GPU-hours;
new attempt ceiling 0.5 node-hour / 2 GPU-hours.

## 7. The write that would be needed (recommendation only — not performed)

No write was made. For author decision, there are two ways to reach a complete,
verifiable release, and they are not equivalent:

**Option A — fresh namespace (recommended).** Stage a new
`dev123_p1_release_v7_<date>` from the already-verified transport
`26c1dc0a…27628a`, which is *already present remotely* at the correct hash in the
control namespace, so no re-transfer is needed. The `v6` tree is left untouched
as preserved evidence of the stall. This respects the authorization's own "all
namespaces must be absent before exclusive creation" rule, needs no write to any
existing artifact, and yields a clean 14-file / 13-ledger-entry verification.

**Option B — in-place completion of `v6`.** Write exactly two paths:
`release.sha256` ← 1181 B / `74a740ea…f70d9b`, and create `source-manifest.json`
← 12174 B / `4eb7476a…79eb0a`, then chmod both to 0444. Mechanically possible
(`release.sha256` is still 0660, owner-writable). **Not recommended**: it
overwrites a preserved partial artifact that is itself the evidence of the
incident, and it is not covered by the existing authorization record.

Either option is a new write to remote storage and therefore needs an explicit
author decision. Neither was performed.

## 8. Remaining conditions before any GPU submission could be authored

1. Author decision on Option A or Option B; then perform that staging under it.
2. Re-verify the completed tree: exact 14 files, 13 ledger entries, every hash
   matched against the immutable archive, payload modes 0444.
3. Author a fresh execution decision (v7) that re-pins the current live
   queue/account facts (2 B5 COMPLETED, 86.3833 GPU-hours consumed, 6 PENDING,
   264 GPU-hour forward reserve) and carries forward the unchanged caps and the
   non-zero `3163986` charge.
4. Fresh live Slurm/account/node/reserve/overlap query at submission time,
   bounded capture, no query retry.
5. The known-host compatibility findings must still hold at submission time:
   `/usr/bin/python3` is 3.6.15 so `Popen(text=True)` fails — the verified
   interpreter path must be used; and exclusive GH200 allocation reports 288 CPUs
   against a 64-CPU request, which the corrected `CPUs/Task` parser must accept.
6. Independent review of the completed release and of the new execution decision.
7. B5 `3149358`–`3149363` remain untouched throughout; `3163986` remains terminal
   and is never cancelled or replayed.

Until all seven hold, no submission is authorized.

---

## Addendum, 2026-09-21 08:40 PDT — Option A executed

Author approved Option A. A fresh namespace
`staging/dev123_p1_release_v7_20260921` was created and populated from the
already-verified remote transport; all 14 members and 13 ledger entries now
verify at mode 0444. The v6 tree was re-hashed afterwards and is byte-for-byte
and metadata-for-metadata unchanged, including its 0-byte 0660 `release.sha256`.
No job was submitted.

See `V7_STAGING_VERIFICATION.md` and the draft `EXECUTION_DECISION_V7_DRAFT.md`
in this folder; raw evidence in `v7_stage_create.txt`, `v7_verification.txt`,
`v6_integrity_recheck.txt`.
