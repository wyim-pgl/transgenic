# Reconciliation: my rebind inventory vs the independent review

2026-09-21 09:25 PDT. Author: Won <wyim@unr.edu>.
Sources: `V7_INDEPENDENT_REVIEW.md` (APPROVE_WITH_FIXES) and `V7_REBUILD.md`.

The reviewer and I worked from different angles and arrived at the same hard
rebind surface. Every item below was checked by me against the **rebuilt**
package, not assumed.

---

## 1. Hard v6 bindings (reviewer §5a: 9 sites, 6 files)

| Reviewer | Site | In my inventory? | State in rebuilt package |
|---|---|---|---|
| K1 | `admission.py:22` NAMESPACE | yes (#1) | rebound to v7 |
| K2 | `p1.sbatch:13` `--chdir` | yes (#5) | rebound |
| K3 | `p1.sbatch:2` `--job-name` | yes (#4) | rebound |
| K4 | `p1.sbatch:14,15` `--output`/`--error` | yes (#6,#7) | rebound |
| K5 | `controller_execution_scope.json` ×3 | yes (#8,#9,#10) | all three rebound |
| K6 | `authorization.example.json` namespace | yes (#11) | regenerated — see F4 |
| **N1** | `deploy.py:264` authority filename | **yes (#2)** | rebound |
| **N2** | `deploy.py:290` job dir prefix | **yes (#3)** | rebound |
| **N3** | `build_release.py:98` output dir | **yes (#12)** | rebound |

**Full agreement.** The three the reviewer flagged as missing from the team
lead's list (N1–N3) are the same three I marked `[+]` in `V7_REBUILD.md` §1 —
found independently before the review arrived. The reviewer counts 9 sites; I
count 13 because I list `p1.sbatch:14` and `:15` separately and include the test
suite (6 sites, 3 modules), which is controller-side and outside the transport.

**Nothing in the reviewer's list was missing from mine, and nothing in mine was
absent from theirs.** Verified on the staged tree: residual v6 token count is
**0** across `admission.py`, `p1.sbatch`, `controller_execution_scope.json`,
`authorization.example.json` and `deploy.py`.

## 2. The six required fixes

**F1 (HIGH) — rebind all nine sites.** Done; §1 above.

**F2 (HIGH) — strike "the release payload is unchanged from v6".** Done. That
sentence is gone from the draft. Replaced with the accurate and narrower claim:
the *source* payload is unchanged — `code.candidate.tar`, `source-manifest.json`,
`operator-pins.json` and the three `notes/` members all retain their original
hashes — while the namespace-bound members changed. The reviewer is right that
the original phrasing is what concealed the binding.

**F3 (HIGH) — the authorization cannot simply be "created".** Correct, and a real
ordering constraint I had stated loosely. `admission.py:72-78` enforces the
authorization's `namespace` equal to the compiled-in constant, and
`deploy.py:264` hardcodes the authority filename. Draft §5 and §8.5 are rewritten
to the right order: rebuild against the new namespace first (now done), then
issue the authorization against the new constant and the new pins, under the
filename the rebuilt `deploy.py` reads.

**F4 (MEDIUM) — `authorization.example.json` must be regenerated, not edited.**
Correct in principle, and I had edited it by substitution. **Verified rather than
assumed:** I regenerated it from `admission.example()` against the rebuilt
`admission.py` and compared byte-for-byte with the staged file — **IDENTICAL**.
The substitution happened to be exactly equivalent because `example()` derives
`namespace` from the same `NAMESPACE` constant I rebound, so there is no
divergence to fail closed on. The staged file is the compiled template's
serialization. Fix satisfied, and the reviewer's rule is the right one to follow
in future.

**F5 (MEDIUM) — the exclusion is enforced in four places.** Correct, and I had
only named two. The enforcement points are `admission.py:66` (template),
`admission.py:318` (placement not in the excluded set), `admission.py:319`
(**exact literal** `ExcNodeList` of `'gh[062,093]'` or `'gh062,gh093'`) and
`p1.sbatch:12`, plus `controller_execution_scope.json:58` as the declared scope —
five sites in total. **No defect was introduced**, because I retained
`gh062,gh093` unchanged, so all five remain consistent; verified by grep on the
rebuilt sources. The warning binds any future change: the exclusion cannot be
edited in `p1.sbatch` alone.

**F6 (LOW) — no change needed.** Acknowledged.

## 3. Environment-bound literals (reviewer §5b) — re-measured live

The reviewer is right that these are invisible to a namespace search-and-replace.
I measured them rather than carrying them forward on the record.

| # | Item | Measured 2026-09-21 09:18 PDT | Result |
|---|---|---|---|
| **E1** | `ACCOUNTS_SHA` pin on the site-owned `/sw/user/scripts/accounts` | `44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5` — **equals the pin**; file 8705 B, mode 755, owner `svcdeltaswmgt`, mtime 2025-08-05 | **PASS** — the balance gate holds, and it is the gate that produced the 1068 figure |
| E2 | `ROOT` campaign root | present | PASS |
| E3 | `SIF` | present, **11120398336 B** = pinned size, mode 750 | PASS (content hash verified on compute) |
| E4 | `CORPUS`, `SOURCE` | corpus present at `WORK/db/b5_full_a40_v1.db`, **23534252032 B** = pinned size; `runs/seed123` present | PASS |
| E5 | `EVIDENCE_SOURCE` | present, **1068773376 B** = `EVIDENCE_BYTES`, mode 444, owner wcyim | PASS |
| E6 | quota tooling | `/sw/user/scripts/quota` present, 4883 B, mode 755 | PASS (format unverified) |
| E7 | `ACCOUNT`/`USER`/`UID`/`CLUSTER`/`START` | account and user confirmed live; `START=2026-09-16` retained deliberately — this attempt is the same campaign's `3163986` recovery, not a new window | PASS, retained by decision |
| **E8** | `b5_registered_ids` fixed list | live `sacct -u wcyim -S 2026-09-16` returns exactly `3131537, 3131538, 3149356, 3149357, 3149358, 3163545, 3163986` plus the pending remainder — **every id is already in the registered list or `registered_prior_ids`**; no B5 job was submitted after the package was built | **PASS** |
| E9 | exclusion enforcement ×4 | see F5 | PASS |
| E10 | `train_b5.slurm` path pin | present, 21518 B | PASS |
| E11 | `3163986` pinned start/end/placement/cpus | terminal and immutable; re-confirmed FAILED 1:0, 51 s, cpu=288 | PASS, intentional |

**On E8 specifically**, which interacts with my own finding that two B5 jobs
completed: the completions do **not** break admission. `admission.py:333-335`
routes terminal jobs through `need(ident in rows and (ident in known or ident in
a['b5_registered_ids']))`, and `3149356`/`3149357` are both inside the registered
range `3149356-3149363`. They reconcile as known terminal history rather than as
unreconciled allocations. This still has to be re-confirmed at submission time,
because the list is fixed and any *newly submitted* B5 job would be rejected.

I also re-verified the checkpoint guard's subject directly: `epoch_05` contains
exactly `model.safetensors`, `meta.json`, `eval.json` and the two known resume
directories `accelerate_state` and `pending_grads` — no unknown entries, no
symlinks — and `meta.json` / `eval.json` hash to `844ecbc4…6943d7` and
`0df4e4a7…54df4f`, matching their pins.

### One correction to my own work
My first probe reported the corpus missing. That was my error: I derived the path
from `ROOT` when `deploy.py:24` uses `WORK = ROOT.parent`. Re-queried at the
correct path, the corpus is present at exactly the pinned byte count. No blocker
existed; I checked before reporting one.

## 4. Reviewer §5c (checked and clean) — agreed

Container-side bind targets, `operator-pins.json`, `source-manifest.json`, the
three `notes/` members, `release.json`'s `source_origin`, and the prefix-class
environment scrub in `p1.sbatch:20-26`. My independent scan reached the same
conclusions, including keeping `source_origin` unchanged as provenance.

## 5. Net position

The reviewer's inventory **confirms** mine and adds no site I missed. Its value
was in the enforcement mechanics — F4's regeneration rule, F5's four-point
exclusion coupling, F3's ordering constraint and E1/E8's live-drift risks — all
of which I have now verified against the rebuilt package rather than assumed.

**The rebuilt package still requires a fresh independent review before any
submission.** The existing APPROVE_WITH_FIXES covers the decision's transferable
content and the v6-bound package; it does not transfer to the new pins
(`73123b3f…`, `d65684a6…`, `902c9bd6…`).

No job was submitted. v6 remains untouched. `3163986` remains terminal. B5
`3149358`–`3149363` remain pending and untouched.
