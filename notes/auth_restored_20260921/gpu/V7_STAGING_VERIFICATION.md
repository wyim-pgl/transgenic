# v7 release staging and verification, 2026-09-21 08:40 PDT

Sibling to `STAGING_RECONCILIATION.md` (which is accepted as-is and not revised).
This records Option A, executed under explicit author approval: a fresh v7
namespace staged from the already-verified remote transport, with the v6 tree
left untouched as preserved incident evidence.

**No Slurm job was submitted and no submission command was prepared for
execution. GPU spend remains a separate author decision that has not been given.**

Author: Won <wyim@unr.edu>. Times PDT (remote clock is CDT, PDT = CDT − 2 h).

Raw evidence: `v7_stage_create.txt`, `v7_verification.txt`, `v6_integrity_recheck.txt`.

---

## 1. Namespace

```
/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_release_v7_20260921
```

Date form `_20260921` matches the existing `_20260917` convention.

**Absence verified before creation.** The creation script's first action was
`[ -e "$V7" ] && abort`; it returned
`ABSENT_OK .../dev123_p1_release_v7_20260921`. This satisfies the
authorization's own "all namespaces must be absent before exclusive creation"
rule. Created 08:36:47 PDT.

## 2. Source of the payload

`staging/dev123_p1_v6_control_20260917/release_v6.tar` — the copy **already
present remotely**. Its sha256 was recomputed immediately before use and the
script would have aborted on any mismatch:

```
26c1dc0a233a95ab86977813e68310eccfce91b081de3a63586cfd7b9427628a   TRANSPORT_PIN_OK
```

No re-transfer from local, no re-download, no rebuild. `tar tf` over that archive
lists exactly 14 members and `tar -xpf` extracted all 14.

## 3. Writes performed

Exactly four operations, all confined to the new namespace:

1. `mkdir -m 700 dev123_p1_release_v7_20260921` (under `umask 077`)
2. `tar -xpf <remote transport> -C <v7>`
3. `find <v7> -type d -exec chmod 700 {} +`
4. `find <v7> -type f -exec chmod 444 {} +`

Nothing else was written anywhere. No step required a write outside items 1–4 of
the authorized scope.

## 4. Per-file verification

Computed in a **separate SSH invocation** from the creation pass — every value
re-read from disk, nothing carried over. Expected hashes come from the same two
independent sources used in the original reconciliation, which agree: the local
immutable `deployment_v4/release_v6.tar` (`26c1dc0a…27628a`) and the pinned
ledger (`74a740ea…f70d9b`).

| # | Path | Size | Mode | Computed sha256 | vs archive | vs ledger | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `admission.py` | 27611 | 0444 | `6f75c8113ff37376b3c62d4ae6f9a52d58d9ebbbbef76151a49b92119f467fa4` | = | = | **VERIFIED** |
| 2 | `authorization.example.json` | 2608 | 0444 | `25e75757eca9baed93522d73cb22cae9e0acfa0338ac5ef561b760e4dc4fec1f` | = | = | **VERIFIED** |
| 3 | `build_release.py` | 5688 | 0444 | `06d5cab84d436de615ca681b69760c1b22eaab7f3ae9fd3339d1bc536f154049` | = | = | **VERIFIED** |
| 4 | `code.candidate.tar` | 1873920 | 0444 | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` | = | = | **VERIFIED** |
| 5 | `controller_execution_scope.json` | 3332 | 0444 | `5566677068d0b22a9f1db7d0683c650e5bcc26f13d84c421aac1b8b7eb9c68aa` | = | = | **VERIFIED** |
| 6 | `deploy.py` | 29149 | 0444 | `3eba7fa940c31f37f3bc7ea3d9229b8723fde62749d850759b1d062e587e1543` | = | = | **VERIFIED** |
| 7 | `notes/b6_mvp_candidate_inventories_20260915.json` | 859299 | 0444 | `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98` | = | = | **VERIFIED** |
| 8 | `notes/b6_mvp_candidate_receipts_20260915.json` | 11786 | 0444 | `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac` | = | = | **VERIFIED** |
| 9 | `notes/b6_mvp_technical_candidate_20260915.json` | 9832 | 0444 | `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e` | = | = | **VERIFIED** |
| 10 | `operator-pins.json` | 11828 | 0444 | `45bd57640d7c55ee7ce7b1f20ab48b58729f5e58b880c4608813b99356d28c72` | = | = | **VERIFIED** |
| 11 | `p1.sbatch` | 2010 | 0444 | `f3b4dd15f3c589b7d329c311e537451a7ba087486c3d89cb74ce10a82288901a` | = | = | **VERIFIED** |
| 12 | `release.json` | 25219 | 0444 | `64a547b8ba5f54e87c1985f278564a9ea9fc7c8f3af07f6b6afe3bdf17f4f7d4` | = | = | **VERIFIED** |
| 13 | `release.sha256` (the ledger) | **1181** | **0444** | `74a740ea154532aa3c0b807d052fb9c05bdf02fcfb6fd86845d348be72f70d9b` | = | n/a (is the ledger) | **VERIFIED — v6 gap closed** |
| 14 | `source-manifest.json` | **12174** | **0444** | `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a` | = | = | **VERIFIED — v6 gap closed** |

## 5. Gate results — computed, not asserted

| Gate | Required | Measured | Result |
|---|---|---|---|
| Transport member count | 14 | `tar tf | grep -cv '/$'` → **14** | PASS |
| Staged regular files | 14 | `find -type f | wc -l` → **14** | PASS |
| Ledger entries verified | 13 | `sha256sum -c release.sha256` → **13 OK, 0 FAILED** | PASS |
| Ledger line count | 13 | `wc -l` → **13** | PASS |
| Every payload 0444 | all | `find ! -perm 0444` → **empty** | PASS |
| No owner-writable file | 0 | `find -perm -u=w | wc -l` → **0** | PASS |
| No symlinks | 0 | `find -type l | wc -l` → **0** | PASS |
| No extra/unexpected entries | 0 | 14 files + 2 dirs, nothing else | PASS |

The "exact 14 files / 13 ledger entries" gate that failed closed on v6 now
**passes on its own terms**, verified by running the ledger against the tree
rather than by inspection.

Note on mtimes: all v7 payloads read epoch 0 (1969-12-31). That is a property of
the immutable transport archive — every member in `release_v6.tar` is recorded at
epoch 0 — faithfully restored by `tar -xp`. It is not a defect or a clock fault.

Directory modes are 0700 (namespace root and `notes/`). Directory modes are not
pinned by any record; 0700 is at least as restrictive as v6's 0700/0770 and
leaves no group or world access.

## 6. v6 was not touched

Full re-hash of the v6 tree **after** the v7 write, compared against the
pre-write capture in `staging_file_hashes.txt`: identical file set, identical
sha256 for all 13 files, identical sizes, identical modes, **identical inodes**
(so nothing was replaced), identical mtimes, and `source-manifest.json` still
absent. Critically, v6's `release.sha256` is still **0 bytes at mode 0660** with
its original mtime — the incident evidence is intact. Details in
`v6_integrity_recheck.txt`.

`3163986` and B5 `3149358`–`3149363` were not queried in any mutating way and
were not touched. Queue re-read at 08:37 PDT shows the same 6 pending B5 jobs and
no MVP diagnostic job.

## 7. Remaining conditions before a submission could be authored

1. **Author approval of the v7 execution decision draft**
   (`EXECUTION_DECISION_V7_DRAFT.md`) — it is a draft, not an authorization.
2. **Independent review** of that draft and of this v7 staging record.
3. **Live account-balance query**, which I did **not** run. The 2026-09-17 record
   cited a displayed 1154 hours / conservative 1153; 86.3833 GPU-hours have been
   consumed by B5 since. That balance must be re-measured, not extrapolated.
4. **Fresh live Slurm/account/node/reserve/overlap query at submission time**,
   bounded capture, no query retry.
5. Host compatibility conditions confirmed on the actual host at submission time:
   `/usr/bin/python3` is 3.6.15 and rejects `Popen(text=True)`; exclusive GH200
   allocation reports 288 CPUs against the 64-CPU request.
6. The canonical authorization artifact for v7 must be issued and staged; the
   remote authority currently on disk is
   `dev123_p1_controller_authorization_v6.json` (`de4a4df5…9df3`), which names the
   v6 namespace. A v7 control namespace and authority do not yet exist.
7. v6 remains preserved and untouched; `3163986` stays terminal; B5 untouched.

Until 1–6 hold, no submission is authorized. Staging complete is not
authorization to run.
