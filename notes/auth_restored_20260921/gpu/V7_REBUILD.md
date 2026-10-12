# v7 release rebuild — namespace rebinding, 2026-09-21 09:15 PDT

Author: Won <wyim@unr.edu>. Times PDT (remote clock CDT, PDT = CDT − 2 h).
Raw evidence: `v7_rebuild_evidence.txt`. Supersedes the staging described in
`V7_STAGING_VERIFICATION.md`, which staged the v6-bound payload.

**No job was submitted.** Gates 4, 5, 6 and 7 remain unmet, and the rebuilt
package needs a fresh independent review before any submission can be authored.

---

## 1. Complete literal inventory

I scanned every payload member — `admission.py`, `deploy.py`, `build_release.py`,
`p1.sbatch`, `controller_execution_scope.json`, `authorization.example.json`,
`operator-pins.json`, `release.json`, `source-manifest.json` and the three
`notes/` members — plus the whole controller test suite, for v6 tokens,
namespace strings and absolute paths.

**13 bindings found and rebound.** The team lead's list had 8 of them; I found 5
more, marked **[+]**.

| # | File:line | Was | Now |
|---|---|---|---|
| 1 | `admission.py:22` | `NAMESPACE = '…/dev123_p1_release_v6_20260917'` | `…_v7_20260921` |
| 2 | **[+]** `deploy.py:264` | `ROOT / 'staging/dev123_p1_controller_authorization_v6.json'` | `…_v7.json` |
| 3 | **[+]** `deploy.py:290` | `ROOT / ('dev123-p1-v6-job-' + job_id)` | `'dev123-p1-v7-job-'` |
| 4 | `p1.sbatch:2` | `--job-name=dev123-p1-v6-0917` | `dev123-p1-v7-0921` |
| 5 | `p1.sbatch:13` | `--chdir=…/dev123_p1_release_v6_20260917` | `…_v7_20260921` |
| 6 | `p1.sbatch:14` | `--output=…/dev123-p1-v6-%j.out` | `dev123-p1-v7-%j.out` |
| 7 | `p1.sbatch:15` | `--error=…/dev123-p1-v6-%j.err` | `dev123-p1-v7-%j.err` |
| 8 | **[+]** `controller_execution_scope.json:34` | `"authorization": "staging/dev123_p1_controller_authorization_v6.json"` | `…_v7.json` |
| 9 | `controller_execution_scope.json:56` | `"namespace": "staging/dev123_p1_release_v6_20260917"` | `…_v7_20260921` |
| 10 | `controller_execution_scope.json:57` | `"job_namespace": "dev123-p1-v6-job-ID"` | `dev123-p1-v7-job-ID` |
| 11 | `authorization.example.json:44` | `"namespace": "…/dev123_p1_release_v6_20260917"` | `…_v7_20260921` |
| 12 | **[+]** `build_release.py:98` | `build(…, HERE / 'release_v6')` | `'release_v7'` |
| 13 | **[+]** test suite (6 sites across 3 modules) | v6 namespace / job prefix / `release_v6` dir | v7 equivalents |

Items 2 and 3 are the substantive additions. `deploy.py:264` is what the running
job uses to locate the controller authority, and `deploy.py:290` names the
per-job subtree — both would have pointed at v6 artifacts from a v7 run.

### Deliberately NOT changed

| Literal | Why |
|---|---|
| `source-manifest.json:110` `"source_origin": "/tmp/mvp-seed123-integration-20260917"` | Historical provenance of the candidate snapshot. Changing it would falsify the record. |
| `admission.py:45` evidence path under `final-p1-job-3163986/` | The authorized read-only input recovery file. Intentionally points at the old failed job. |
| `deploy.py` container/bind paths (`/work/repo`, `/mvp/inputs`, `/scratch/*`, SIF path) | Runtime layout, not namespace-bound. |
| `admission.py` tool paths (`/usr/bin/sacct`, `/sw/user/scripts/accounts`, …) | Host tool locations. |
| `operator-pins.json`, `notes/*.json` | No path or namespace literals; derived from source payload only. |

## 2. Node exclusions — measured, and deliberately unchanged

Live query at build time found **no node in `ghx4` down, drained, failing, in
maintenance or unresponsive** — 39 allocated, 43 mixed, 66 mixed-, 2 reserved.
`gh062` and `gh093` are both healthy with reason `none`. No B5 job is currently
placed, so there is no own-placement collision to avoid. Other accounts'
reservations are not schedulable by us regardless.

So the live data gives **no basis to add any exclusion**, and I retained
`gh062,gh093` rather than dropping them. Two reasons: their original rationale is
recorded only as "Fixed reviewed exclusions", so removing them would discard a
reviewed decision I cannot reconstruct; and the guard test is named
`test_new_namespace_and_preserved_exclusions`, which asserts that exact pair —
preservation is the reviewed intent. Excluding 2 of 150 nodes has no practical
scheduling cost.

This is the honest outcome of the refresh: measured, no change warranted. If the
author wants the pair re-evaluated on its merits, that is a separate decision.

## 3. Build and tests

Built in a **new** directory `deployment_v5/`, leaving `deployment_v4/` — the v6
build sources and preserved originals — untouched. Python 3.8.20.

`python3 -m unittest discover` → **Ran 47 tests, OK** (24.780 s).

The first run reported 41. Two causes, both mine and both fixed rather than
worked around: `test_recovery.py` (6 tests) had not been copied, and
`test_field_alias.py` needed the `scontrol-job-3163986.txt` fixture. 41 + 6 = 47
reconciles exactly with the recorded v6 suite size.

I diffed every test module against its v4 original with the v6/v7 tokens
normalised on both sides: **no differences**. The only test changes are the
namespace rebindings. No assertion was relaxed, deleted or rewritten to reach a
pass.

## 4. Transport recipe — validated before use

Before building the v7 transport I ran the same recipe against the unchanged
`release_v6` tree and reproduced `release_v6.tar` **byte-exactly**
(`26c1dc0a…27628a`). That proves the builder matches the controller's original
method (PAX format, mtime 0, mode 0444, uid/gid 0, sorted members), so the v7
transport is directly comparable rather than merely plausible.

## 5. New pins (computed by me, confirmed remotely)

| Artifact | New sha256 |
|---|---|
| **Transport `release_v7.tar`** (2887680 B) | `73123b3fe0f27e15aa024805c271e54b6ef834d08802a43549342cfcf8da3d16` |
| **`release.json`** | `d65684a69ec06bcf7018fe3c0fbaa0526c7e350da4f84630fbb26199f76cfd83` |
| **Ledger `release.sha256`** | `902c9bd6f6299b215b32c4c44c421efc9d6788e8f5a62d914b614067f7648d9f` |
| `admission.py` | `49b289ba8034649898260235e25eeb0195ad3874edbb97ab10bef7950137f8b8` |
| `deploy.py` | `0fd10399abfac8e35ff14512bc228aad06e4021fda5be554360d1810a9f6964c` |
| `p1.sbatch` | `6ec497c19b57e5cb531c3d265aa256dbf80375e23bb05d0586a724cd8d057f3d` |
| `controller_execution_scope.json` | `165c32eccb74af0d2f407e3b9e3308dd1549b2199fcbec8afc9c94e7fb45fb22` |
| `authorization.example.json` | `a8a097acf904fe07178881d1fe12ce26f10bc23ab479c5683852e63590ce5bf5` |
| `build_release.py` | `aef33dfc01e349732d8076b06e98f8e901f058cf5eeadbf613d0a7333e8c7c85` |

Unchanged, as expected — these carry no namespace binding:
`code.candidate.tar` `da161088…5cbec7`, `source-manifest.json` `4eb7476a…79eb0a`,
`operator-pins.json` `45bd5764…6d28c72`, and the three `notes/` members.

The old v6 pins (`26c1dc0a…`, `64a547b8…`, `74a740ea…`) are now **superseded** and
must not be used to validate v7.

## 6. Remote replacement and verification

Transport streamed into the v7 namespace, hash re-verified **on the remote host**
before use, old contents removed, extracted, chmod'd, temp tar deleted. The
script asserted the target path suffix and the transport hash, aborting on
either. Every write was inside `dev123_p1_release_v7_20260921`.

Independent verification pass afterwards:

| Gate | Measured | Result |
|---|---|---|
| Regular files | 14 | PASS |
| Ledger entries | `sha256sum -c` → 13 OK, 0 failed | PASS |
| Ledger lines | 13 | PASS |
| All payloads 0444 | `find ! -perm 0444` → empty | PASS |
| Owner-writable | 0 | PASS |
| Symlinks / hidden / extra | 0 / 0 / 0 | PASS |
| Remote hashes = locally computed pins | all 14 | PASS |
| Residual v6 tokens in staged package | 0 across all five bound files | PASS |

## 7. v6 untouched

Full re-hash after the replacement is identical to `v6_integrity_recheck.txt` in
every field — same 13 files, same hashes, same sizes, same modes, **same
inodes**, with `release.sha256` still 0 bytes at 0660. `3163986` and B5
`3149358`–`3149363` were not touched; queue re-read at 09:12 PDT is unchanged.

## 8. Reviewer reconciliation — outstanding

The independent reviewer's literal inventory has not reached me. When it does I
must reconcile it against the 13 above. If it names a literal I missed, that is a
finding about this build and the package must be rebuilt again — not waved
through. I am not treating my inventory as complete until that reconciliation
happens.

## 9. Remaining conditions before a submission could be authored

1. **Fresh independent review of the rebuilt package.** The in-flight review was
   of the v6-bound package and explicitly does not carry over.
2. Reviewer literal-inventory reconciliation (§8).
3. **Gate 5**: create the v7 control namespace and issue the canonical v7
   authorization artifact `dev123_p1_controller_authorization_v7.json`, each
   verified absent beforehand, with the attempt journal created exactly once. The
   v6 authority (`de4a4df5…9df3`) names the v6 namespace and must not be reused;
   the new one must carry the new pins from §5.
4. **Gate 6**: host checks re-run on the actual host against *this* package —
   interpreter (3.6.15 / `Popen(text=True)`), `CPUs/Task` parser against the
   288-vs-64 reality, checkpoint guard for `accelerate_state` / `pending_grads`.
5. **Gate 4**: fresh live Slurm/account/node/reserve/overlap query immediately
   before submission, bounded capture, no query retry.
6. **Gate 7**: v6 preserved, `3163986` terminal, B5 untouched — reconfirmed at
   submission time.
7. Team lead's confirmation to submit.

Gate 3 (account balance) remains satisfied: measured 1068 hours against the
recomputed 304 requirement, corroborated by 1154 − 1068 = 86 matching the
86.3833 GPU-hour derivation.
