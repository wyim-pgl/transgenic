# DRAFT execution decision v7 — one seed123 S0/P1 recovery attempt

**STATUS: DRAFT. This is not an authorization and not a submission.** It awaits
author approval and independent review. No GPU job has been submitted, and no
submission command has been prepared for execution. Nothing in this file may be
treated as consent to spend GPU time.

Author: Won <wyim@unr.edu>. Drafted 2026-09-21 PDT. Times PDT (remote CDT − 2 h).

Supersedes, if and only if approved, `auth_resume_20260917/gpu/EXECUTION_DECISION_V6.md`
— which is **not** exercisable: its own "exact 14 files / 13 ledger entries before
exec" gate failed closed on the incomplete v6 staging tree, and the queue/account
observations it pinned have since been overtaken by real consumption.

---

## 1. Why a new record is needed

`EXECUTION_DECISION_V6.md` remains sound on identity but stale on facts. Its
identity pins all still verify bit-exact (§3). What changed is consumption: two
B5 jobs ran to completion after it was written, so every figure that record
derived from the 352 GPU-hour forward reserve is now wrong and must be
**recomputed, not copied**.

## 2. Live account facts, re-pinned (measured 2026-09-21 08:28 PDT)

Source: `sacct -X` and `squeue`, read-only. Raw capture: `staging_slurm_state.txt`.

| Job | State | Elapsed | Alloc |
|---|---|---|---|
| 3149356 | **COMPLETED** 2026-09-19 04:54 PDT | 10:48:49 | cpu=64, gres/gpu=4, node=1 |
| 3149357 | **COMPLETED** 2026-09-19 16:20 PDT | 10:46:56 | cpu=64, gres/gpu=4, node=1 |
| 3149358–3149363 (6 jobs) | PENDING | — | — |

Consumption arithmetic, shown so it can be checked:

```
10:48:49 = 10 + 48/60 + 49/3600 = 10.8136111 h
10:46:56 = 10 + 46/60 + 56/3600 = 10.7822222 h
                                 ----------------
B5 consumed                    =  21.5958333 node-hours
                        x 4 GPU =  86.3833333 GPU-hours
```

Forward B5 reserve, recomputed from 6 remaining jobs at 11 h × 4 GPU each:

```
6 x 11 h            =  66 node-hours
6 x 11 h x 4 GPU    = 264 GPU-hours          (was 352 under the v6 record)
```

Every v6 figure derived from 352 is therefore void. The v6 record's headroom
comparison `B5 + 20 diagnostic + 20 margin = 392` recomputes to
`264 + 20 + 20 = 304`.

**Not measured, and required before approval:** the live account balance. The v6
record cited a displayed 1154 hours / conservative 1153 as of 2026-09-17; 86.3833
GPU-hours have been consumed since. I did not query the balance and this draft
does not assert one. It must be re-measured live, not extrapolated — see §8.

## 3. Caps and charges carried forward unchanged

These do not depend on the 352 figure and are carried forward verbatim:

- Old `3163986` FAILED, 51 s, charged **0.0141666667 node-hour /
  0.0566666667 GPU-hour**. Never zero. Re-confirmed terminal on 2026-09-21:
  FAILED 1:0, `.extern` COMPLETED 0:0, cpu=288 against a 64-CPU request.
- Prior `3163545` cancelled, **0**, remains included in the campaign ledger.
- Campaign cap: **5 node-hours / 20 GPU-hours**, unchanged.
- This one attempt: maximum **0.5 node-hour / 2 GPU-hours**, unchanged.
- Unsubmitted panel candidate reservation: 1 node-hour / 4 GPU-hours, unchanged.
- Combined committed floor: **1.5141666667 node-hours / 6.0566666667 GPU-hours**,
  unchanged (0.0141666667 + 0.5 + 1 node-hours; 0.0566666667 + 2 + 4 GPU-hours).

All startup, checks, copies, failures and teardown count against the attempt.

## 4. Identity pins — REVISED 2026-09-21 after the v7 rebuild

The v6 package was namespace-bound to `dev123_p1_release_v6_20260917` in 13
places, so it could not execute from a v7 namespace (see
`V7_SUBMISSION_BLOCKER.md`). It was rebuilt with those bindings corrected; see
`V7_REBUILD.md` for the full inventory. All pins below were computed by me from
the build output and confirmed byte-identical on the staged remote tree.

| Artifact | sha256 |
|---|---|
| **Transport `release_v7.tar`** | `73123b3fe0f27e15aa024805c271e54b6ef834d08802a43549342cfcf8da3d16` |
| **`release.json`** | `d65684a69ec06bcf7018fe3c0fbaa0526c7e350da4f84630fbb26199f76cfd83` |
| **Ledger `release.sha256`** | `902c9bd6f6299b215b32c4c44c421efc9d6788e8f5a62d914b614067f7648d9f` |
| `admission.py` | `49b289ba8034649898260235e25eeb0195ad3874edbb97ab10bef7950137f8b8` |
| `deploy.py` | `0fd10399abfac8e35ff14512bc228aad06e4021fda5be554360d1810a9f6964c` |
| `p1.sbatch` | `6ec497c19b57e5cb531c3d265aa256dbf80375e23bb05d0586a724cd8d057f3d` |
| `controller_execution_scope.json` | `165c32eccb74af0d2f407e3b9e3308dd1549b2199fcbec8afc9c94e7fb45fb22` |
| `authorization.example.json` | `a8a097acf904fe07178881d1fe12ce26f10bc23ab479c5683852e63590ce5bf5` |
| `build_release.py` | `aef33dfc01e349732d8076b06e98f8e901f058cf5eeadbf613d0a7333e8c7c85` |
| `source-manifest.json` (unchanged) | `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a` |
| `code.candidate.tar` (unchanged) | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |
| `operator-pins.json` (unchanged) | `45bd57640d7c55ee7ce7b1f20ab48b58729f5e58b880c4608813b99356d28c72` |
| Prior `release_v5.tar` (frozen) | `b3dafe316605a8ccaae61e94854fe527bd8591cb4600ae152afae017d2807458` |

The **source payload is unchanged** — `code.candidate.tar`,
`source-manifest.json`, `operator-pins.json` and the three `notes/` members all
carry their original hashes, so the 103-file candidate and its provenance are
untouched. Only namespace bindings changed.

**Superseded and not to be used for v7 validation:** transport
`26c1dc0a…27628a`, release JSON `64a547b8…f4f7d4`, ledger `74a740ea…f70d9b`.

The canonical authorization for v7 (`dev123_p1_controller_authorization_v7.json`)
**does not yet exist** and must carry the pins above. The v6 authority
`de4a4df50f3c908a81d5413fae1efce7e6939f02643d14c7c7431dc74aef9df3` names the v6
namespace and must not be reused.

The controller test suite passes **47/47** against the rebuilt package, with no
assertion relaxed — only the namespace rebindings changed.

## 5. Namespaces

- Release (rebuilt, staged and fully verified, 2026-09-21 09:12 PDT):
  `staging/dev123_p1_release_v7_20260921` — 14 files, 13 ledger entries OK,
  all payloads 0444, no owner-writable file, zero residual v6 bindings.
  See `V7_REBUILD.md`.
- Control namespace for v7: **does not yet exist.** Must be created, verified
  absent beforehand, before any attempt.
- Canonical v7 authorization artifact: **does not yet exist**, and note the
  ordering constraint (independent review F3). It cannot simply be "created"
  against a chosen namespace: `admission.py` compiles the namespace into a
  constant, `validate_authorization` enforces the authorization's `namespace`
  field equal to that constant (`admission.py:72-78`), and `deploy.py:264`
  hardcodes the authority *filename*. The correct order is therefore: **rebuild
  the release against the new namespace first — done — then issue the
  authorization against the new compiled-in constant and the §4 pins**, under the
  filename `dev123_p1_controller_authorization_v7.json` that the rebuilt
  `deploy.py` now reads. The v6 authority on disk
  (`de4a4df5…9df3`) names the v6 namespace and cannot stand in.
- `staging/dev123_p1_release_v6_20260917` is **preserved incident evidence**.
  It is never written, chmod'd, moved, completed or deleted.

## 6. Host compatibility findings carried forward, and how v7 handles each

| Finding (still holds) | v7 handling |
|---|---|
| `/usr/bin/python3` on the login/compute host is **3.6.15**, so `Popen(text=True)` fails immediately — reproduced against `/bin/true`. | The release already uses the corrected call form; `admission.py` (`6f75c811…67fa4`) is the reviewed post-fix artifact. Before any attempt, re-run the syntax / pure-validator / subprocess checks **on the actual host** with the interpreter that will be used. No installation. The SIF's Python 3.12.3 is a separate runtime and does not satisfy this check. |
| Exclusive GH200 allocation reports **288 CPUs** against a **64-CPU** request (`3163986` AllocTRES cpu=288). | The v6 field fix is carried in this same payload: the parser maps only `CPUs/Task` → internal `CPUsPerTask`, both aliases present in one response still reject, and validation compares the requested 64 against actual exclusive node capacity rather than assuming 64 allocated. No CPU, request or node-capacity limit is changed. |
| `epoch05` contains B5 resume-state directories `accelerate_state` and `pending_grads` (pending_micro 15), which the B5 producer legitimately writes. | Weights-only initialization skips exactly those two known real directories, which remain untouched and are never loaded or deleted. Unknown files, directories or symlinks still reject. The three pinned regular files (`model.safetensors`, `meta.json`, `eval.json`) must still match `844ecbc4…` / `0df4e4a7…`; large weights must be re-verified to the original `6e5e…` bytes before and after staging on compute. |

## 7. Exact scope of the one attempt

One exclusive GH200 node, 4 GPUs, 64 requested task CPUs, 1800 seconds maximum,
profile development-single / seed 123 / mvp_finetune. Ceiling 0.5 node-hour /
2 GPU-hours.

Input recovery is limited to the single named read-only evidence file from the
failed `3163986`:
`/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/final-p1-job-3163986/inputs/evidence/fullc0lite.db`
(1068773376 B, mode 0444, expected content SHA
`ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d`). It is
re-hashed, privately copied to a fresh destination and verified before and after,
with the old source left unchanged. No staging fallback, no directory search, no
move-back, no DB rebuild, no failure-tree cleanup, no alternate-storage bypass.

**`run_role == 'mvp_finetune'` is not a training instruction.** It is a
prepared-receipt *binding* value. At `deploy.py:241-243` it appears only inside
`validate_prepared()`, as a `require(...)` guard that rejects old or cross-profile
receipts — it selects nothing and starts nothing. I verified the actual execution
path independently: `p1.sbatch` runs `deploy.py host`, which invokes
`real_probes.py --mode real_tokenizer` (`deploy.py:513`) under `run_bounded.py`
with a 900 s probe timeout, and `p1.sbatch:43` states in terms "No separate warmup
and no TRAINING_DONE". No training step exists in this path. The name is
historical; a reader should not take §7's profile line as authorizing training.

This authorizes **one attempt only**. It does not authorize training, follow-ups,
automatic retry, extra seeds, full ONT, main merge or push, or scientific
acceptance of any result. Queued means queued, not GPU measurement. An uncertain
submission is reconciled, never retried. Failure preserves all artifacts and the
claim, and requires a separately reviewed next decision. The actual GPU-6 gates
remain unmet.

## 8. Conditions that must be satisfied before this draft can become an authorization

1. Author approval of this draft. **DONE** 2026-09-21.
2. Independent review. **PARTIALLY DONE** — `V7_INDEPENDENT_REVIEW.md` returned
   APPROVE_WITH_FIXES on the decision's transferable content (arithmetic, scope,
   `3163986` handling, host handling: no error found in any). Its six required
   fixes are addressed in `V7_REBUILD.md` §8. **A fresh independent review of the
   rebuilt package is still outstanding** — the reviewed package was the v6-bound
   one and its approval does not transfer to the new pins.
3. **Live account-balance query.** **DONE** — measured `Balance 1068 /
   Deposited 1499` against the recomputed `264 + 20 + 20 = 304` requirement;
   `1154 − 1068 = 86` corroborates the 86.3833 GPU-hour derivation.
4. Fresh live Slurm / account / node / reserve / overlap query at attempt start,
   bounded capture, no query retry, before any large hash or staging.
5. v7 control namespace created (verified absent beforehand) and the canonical v7
   authorization **issued against the rebuilt package's compiled-in namespace and
   the §4 pins** — in that order, per F3 — with the attempt journal created
   exactly once.
6. Host checks re-run on the actual host per §6 (interpreter, parser, checkpoint
   guard) and passing.
7. v6 preserved untouched; `3163986` left terminal and never cancelled or
   replayed; B5 `3149358`–`3149363` untouched.

Until all seven hold, no submission is authorized. A complete, verified staging
tree is not permission to run.
