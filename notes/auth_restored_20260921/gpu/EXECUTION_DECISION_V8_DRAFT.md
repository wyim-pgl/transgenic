# DRAFT execution decision v8 — one seed123 S0/P1 recovery attempt

**STATUS: DRAFT. Not an authorization, not a submission.** No GPU job has been
submitted and no submission command has been prepared for execution. Awaits a
fresh independent review of the v8 package and the team lead's confirmation.

Author: Won <wyim@unr.edu>. Drafted 2026-09-21 PDT. Times PDT (remote CDT − 2 h).

Supersedes `EXECUTION_DECISION_V7_DRAFT.md`, which is kept unedited alongside
this file as the record of what was decided at that point. That draft's pins are
void: the payload it described could not execute from any namespace.

---

## 1. Why v8 exists

The v6 package was bound to `dev123_p1_release_v6_20260917` in nine places across
six files. Restaging identical bytes at a new path (v7) did not rebind them, so
v7 was unsubmittable for the same reason v6 was. v8 is a genuine rebuild against
`dev123_p1_release_v8_20260921`, with all six files that the build ships rebound
*before* `build()` ran. Details: `V8_REBUILD.md`.

Both prior trees are preserved and nothing was deleted:
`…_v6_20260917` is the I/O-stall incident evidence; `…_v7_20260921` is the
binding-defect evidence.

## 2. Live account facts (measured 2026-09-21, unchanged from the v7 draft)

| Job | State | Elapsed |
|---|---|---|
| 3149356 | COMPLETED 2026-09-19 04:54 PDT | 10:48:49 |
| 3149357 | COMPLETED 2026-09-19 16:20 PDT | 10:46:56 |
| 3149358–3149363 | PENDING | — |

```
10:48:49 = 10.8136111 h ; 10:46:56 = 10.7822222 h
B5 consumed = 21.5958333 node-hours = 86.3833333 GPU-hours
forward reserve = 6 x 11 h x 4 GPU = 264 GPU-hours   (was 352 — void)
headroom requirement = 264 + 20 + 20 = 304
```

**Balance, measured:** `accounts` reports 1068 hours (deposited 1499). The gate
itself uses `max(0, bal−1)` (`admission.py:461`), i.e. **1067 ≥ 304**, leaving
763 GPU-hours of margin. **Gate 3 is discharged.**

**Corrected framing of the corroboration.** The v7 draft said `1154 − 1068 = 86`
matches the 86.3833 derivation exactly. It does not — both endpoints are
`math.floor` outputs (`admission.py:459`), so the agreement is to **±1**. It is
fully consistent and rules out a gross error, but the decimal is established by
the `sacct` elapsed values, not by the balance difference.

Independent-review finding worth recording: §2 and §3 are not arithmetic that
merely happens to agree with the gate — **they are the formulas the code
enforces**. `admission.py:198-202` requires `balance >= sum(b5) + 20 + 20` with
`b5` built live from `TimeLimit × GPUs / 3600`, and `admission.py:190-195`
computes the committed floor against 5/20.

## 3. Caps and charges — carried forward unchanged

- `3163986` FAILED, 51 s: **0.0141666667 node-hour / 0.0566666667 GPU-hour**.
  Never zero. Re-confirmed terminal: FAILED 1:0, `.extern` COMPLETED 0:0, cpu=288
  against a 64-CPU request.
- `3163545` cancelled: **0**, still included.
- Campaign cap **5 node-hours / 20 GPU-hours**; this attempt **0.5 / 2**;
  unsubmitted panel reservation 1 / 4.
- Committed floor **1.5141666667 node-hours / 6.0566666667 GPU-hours**.
- B5 never counts against the diagnostic cap: the `spent` loop skips `b5-seed`
  rows, and `admission.py:334-336` accepts the two completions as known terminal
  history because they are inside `b5_registered_ids`.

All startup, checks, copies, failures and teardown count against the attempt.
**No surviving figure derives from the void 352.**

## 4. Identity pins — v8

| Artifact | sha256 |
|---|---|
| **Transport `release_v8.tar`** | `90d904824f02447de2a77e11d90919ef8d4dac646f6eb25ae9a90bdd7f020dfd` |
| **`release.json`** | `f716d55256269c58da4f38028ea2660dfe6dc96db656651ea1a37f2761a58055` |
| **Ledger `release.sha256`** | `f56f3a29df50568792031d76ed9919ac2a5daef2c5340659836fcc7e1afa88c7` |
| `admission.py` | `f31bf313692b59fcc8ece9363fad9123e48dba129ea51039a725ae95589907d0` |
| `deploy.py` | `8cc70ce9e1a20fda8dbc24ddd2f4005a171ef400879da90d8a22c5bf47403ab9` |
| `p1.sbatch` | `efe2cc04b74b0914eaa8699c5f80fbee3ace0a690a8de46330aa4c3692a291f2` |
| `controller_execution_scope.json` | `e8b05fa4f25a820ef0f568e4390d60001c720087352b68113d23bfd0da8adbf6` |
| `authorization.example.json` | `1241804620118affb65ee09d3eeff29973f5a04d67dae9842949f679efcabca2` |
| `build_release.py` | `948acf4ea35bb279c65bc7fa4b63dac6d068ecc329b1a00ec4d6a08b760893e7` |
| `code.candidate.tar` (unchanged) | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |
| `source-manifest.json` (unchanged) | `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a` |
| `operator-pins.json` (unchanged) | `45bd57640d7c55ee7ce7b1f20ab48b58729f5e58b880c4608813b99356d28c72` |

The **103-file candidate source is untouched** — `code.candidate.tar`,
`source-manifest.json` and `operator-pins.json` carry their original hashes
across v6, v7 and v8. Only namespace bindings changed. *(This replaces the v7
draft's "only the namespace changes", which was false: the namespace was exactly
what that payload could not change.)*

v6 and v7 pins are superseded and must not be used to validate v8.

## 5. Namespaces — rebuild first, then authorize

- Release: `staging/dev123_p1_release_v8_20260921` — staged and verified
  2026-09-21 09:25 PDT; 14 files, 13/13 ledger, all 0444, zero residual v6/v7
  tokens.
- v8 control namespace: **does not yet exist.**
- Canonical `dev123_p1_controller_authorization_v8.json`: **does not yet exist,
  and the order is forced.** `admission.py:54` places `namespace=NAMESPACE`
  outside the template's `variable` set and `admission.py:76-78` enforces
  set-equality plus field-by-field equality against the compiled template, while
  `deploy.py:264` hardcodes the authority *filename*. An authorization therefore
  cannot be written against a chosen namespace — the release must be rebuilt
  against that namespace first (**done**), and the authorization then issued
  against the compiled-in constant and the §4 pins.
- `…_v6_20260917` and `…_v7_20260921` are preserved evidence: never written,
  chmod'd, moved, completed or deleted.

## 6. Host compatibility — carried forward, all still code rather than assertion

| Finding | v8 handling |
|---|---|
| `/usr/bin/python3` is 3.6.15 and rejects `Popen(text=True)` | No `text=True` anywhere in the payload; `ast.parse` clean. Must still be re-run **on the actual execution host** — this cannot be self-certified by the release. |
| Exclusive GH200 reports 288 CPUs against a 64-CPU request | `CPUs/Task` → `CPUsPerTask` alias mapping at `admission.py:247,302-305,462-464`; both aliases in one response still reject. |
| `epoch05` holds B5 resume state `accelerate_state`, `pending_grads` | Exact-inventory plus `lstat` type checks; the two directories are re-verified as excluded after staging. Verified live: `epoch_05` contains exactly `model.safetensors`, `meta.json`, `eval.json` and those two directories, with `meta.json` = `844ecbc4…6943d7` and `eval.json` = `0df4e4a7…54df4f`. This is a real fix, not symptom exclusion. |
| `admission.py:23` `ACCOUNTS_SHA` pins the **site-owned** `/sw/user/scripts/accounts` | Re-measured twice on 2026-09-21: `44302f2e…be35f5`, matches. Outside author control, so re-check at submission. |
| `admission.py:370-376` pins `3163986` start `18:47:46`, end `18:48:37`, placement `gh077`, cpus `288` | **Intentional and stated as such.** Terminal and immutable; tying the package to this specific incident is correct for a recovery attempt. |

## 7. Scope of the one attempt

One exclusive GH200 node, 4 GPUs, 64 requested task CPUs, 1800 s maximum,
profile development-single / seed 123. Ceiling 0.5 node-hour / 2 GPU-hours.

Input recovery is limited to the single named read-only file from `3163986`
(`fullc0lite.db`, 1068773376 B, mode 0444, `ed2d0eef…fe3d0d` — re-confirmed
present and hashing to its pin on 2026-09-21), re-hashed, privately copied and
verified before and after, with the source left unchanged. No staging fallback,
no directory search, no move-back, no DB rebuild, no failure-tree cleanup, no
alternate-storage bypass.

**The scope fields are stamped by the build tool, but read carefully — only some
are enforced.** `build_release.py` writes them into `release.json` at build time:

```
scope = 'S0_P1_DIAGNOSTIC_ONLY'      node_hours_max = 0.5   gpu_hours_max = 2
measured = False                      runtime_ready = False
training_authorized = False
review_status = 'PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW'
```

**M1 — correction.** These fields are **declarative only**. Neither executable
reads `review_status`, `runtime_ready`, `measured` or `scope` back out of the
manifest, so none of them fails closed. An earlier statement of mine (and one
relayed to the author) described `review_status` as a machine-enforced
precondition; that was wrong.

`release.json` is hash-pinned everywhere, so this **cannot be corrected in place**
— the v8 package will permanently read
`PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW` even after the review is complete, and
that is a known, accepted, uncorrectable-in-place artifact. **A v9 rebuild is
explicitly not justified for it.**

The authoritative record of review is instead the authorization's
`review_reference` field, which `admission.py:73-75` places in the `variable`
set for exactly this purpose. At gate 5 it must be populated with
`V8_INDEPENDENT_REVIEW.md` and its sha256
`fd9462368c2965b83f714709b94e9963d3f11cf35dda289d432cc1e8c699e9e9`.

What *is* genuinely enforced is the caps arithmetic (`admission.py:190-202`), the
namespace equality (`:72-78`), the pins, and the alias/placement/exclusion
checks. The scope label is documentation; the ceilings are code.

**`run_role == 'mvp_finetune'` is a role label, not a training instruction**, and
three independent lines of evidence say so: it appears only inside
`validate_prepared()` as a `require(...)` guard on a prepared receipt
(`deploy.py:241-243`); the executed path is `real_probes.py --mode
real_tokenizer` (`deploy.py:513`) under `run_bounded.py` at 900 s, with
`p1.sbatch:43` stating "No separate warmup and no TRAINING_DONE"; and it sits in
the same dict as `scope: S0_P1_DIAGNOSTIC_ONLY` and `training_authorized: False`.

This authorizes **one attempt only**. Not training, follow-ups, automatic retry,
extra seeds, full ONT, main merge or push, or scientific acceptance. Queued means
queued, not GPU measurement. An uncertain submission is reconciled, never
retried. Failure preserves all artifacts and the claim and requires a separately
reviewed next decision. The actual GPU-6 gates remain unmet.

## 8. The 1700 s budget — margin analysis (M2, L2, L3 applied)

**L3 — what this is and is not.** `3163986`'s recorded root cause was the
exact-inventory check rejecting `epoch05` over `accelerate_state` /
`pending_grads` — **not** an I/O stall. The separate 2026-09-17 I/O stall was a
*staging* incident on the login node, not a job failure. So the work below is a
**margin analysis against a hypothesis** — that the outer 1700 s deadline could
fire before the probe completes — and it must not be described as closing the v6
failure or the `3163986` failure. It closes neither; it retires a different risk.

**M2 — completeness.** The table below is the *dominant* term, not the complete
enumeration. Roughly fourteen further `sha()`/`pinned()` sites exist
(`deploy.py:248, 254, 267, 273, 274, 298, 304-305, 320, 409, 437, 487, 489, 506,
514-522`). The independent reviewer measured them at **9.6 MB, 0.02 s**. They are
folded into the total below; the conclusion is unchanged, but the earlier wording
that presented the table as the enumeration was wrong.

| Source | Bytes |
|---|---|
| `pinned(SIF)` `deploy.py:326` (L2: cited as :324 previously) | 11,120,398,336 |
| `pinned(CORPUS)` `deploy.py:326` | 23,534,252,032 |
| `pinned(EVIDENCE)` `deploy.py:326` | 1,068,773,376 |
| `recover_evidence` `:97, :106, :107` — three further full passes | 3,206,320,128 |
| HF cache <= 2 GB, 3 passes (`:200, :202`) | ~6,000,000,000 |
| checkpoint <= 2.2 GB, ~3 passes | ~6,600,000,000 |
| ~14 small `sha()`/`pinned()` sites (reviewer-measured) | 9,600,000 |
| **Total** | **51,529,743,872 B** |

Measured read-only on the host: **sha256 at 400.1 MB/s**, Lustre read 5.3 GB/s.
The binding constraint is sha256 CPU, not the filesystem.

```
outer budget                                1700 s
probe (run_bounded --timeout-seconds 900)    900 s
available for staging + hashing              800 s

51,529,743,872 B / 400.1 MB/s          =  128.8 s
copies ~5.27 GB read+write             =   ~12 s
generous allowance incl. startup       <= 300 s

break-even 51,529,743,872 / 800 s = 64.4 MB/s ; measured 400.1 = 6.21x margin
at a quarter of the measured rate (100 MB/s): 515 s < 800 s — still closes
```

**No budget or scope change is required.**

**Caveat, corrected.** I previously listed `sha256sum` vs Python `hashlib` as a
source of uncertainty. The reviewer is right that this is weaker than I allowed:
`deploy.py`'s `sha()` feeds 8 MiB blocks into `hashlib`, the same OpenSSL
primitive coreutils uses, so the two rates are directly comparable and this is
not a meaningful uncertainty. The caveats that do stand: measured on login node
`gh-login03` rather than a ghx4 compute node (same Lustre); **write throughput
estimated, not measured**, since writing outside the authorized namespaces is not
permitted; and Lustre performance varies with cluster load.

**Recommendation:** the attempt should record its own staging elapsed time, so
this margin becomes measured on the first real run instead of inferred.

## 8b. The attempt is one-shot in a stronger sense than stated (M3)

`p1.sbatch:38` executes, under `noclobber`, before `deploy.py host` runs at all:

```
(set -o noclobber; printf '%s\n' "${SLURM_JOB_ID:?}" > S0_P1_STARTED)
```

and `p1.sbatch:37` asserts `[[ ! -e S0_P1_STARTED ]]` beforehand. The v8
namespace is `drwx------ wcyim` with payloads at 0444.

**Consequence, stated here so it is not discovered on the day: any failure after
that line permanently consumes the namespace.** That includes failures entirely
outside our control — a node fault, a Lustre hiccup, a preemption, an interpreter
mismatch. The claim file is deliberately never removed on failure, so the
namespace cannot be reused, and a retry requires a **full v9 rebuild**: new
bindings, new ledger, new `release.json`, new transport, new pins, new
authorization, new independent review.

This is by design — it is what makes "one attempt" mechanically true rather than
merely promised — but it means the real cost of a failed attempt is not the
~51 s of GPU time, it is a complete rebuild-and-review cycle.

**Recovery path if it fires:** preserve the consumed namespace and the job's
`dev123-p1-v8-job-<id>` subtree untouched as evidence; reconcile the terminal
record and charge the actual elapsed time against the campaign cap; author a
separately reviewed next decision; rebuild as v9 against a fresh namespace. Never
delete `S0_P1_STARTED`, never reuse the namespace, never resubmit into it.

## 9. Conditions before this draft can become an authorization

1. **Fresh independent review of the v8 package.** Neither prior review
   transfers — both examined superseded pins. The build itself stamps
   `review_status = PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW`.
2. Author approval of this draft.
3. **Gate 3 — DISCHARGED**: 1067 ≥ 304, 763 GPU-hours margin.
4. **Gate 4**: fresh bounded live Slurm / account / node / reserve / overlap
   query immediately before submission; no query retry.
5. **Gate 5**: v8 control namespace created (verified absent) and the canonical
   v8 authorization issued against the compiled-in namespace and §4 pins, in that
   order; attempt journal created exactly once.
6. **Gate 6 — RESTATED as an in-job check (see §10). Not dischargeable as a
   pre-flight gate on the true execution host without spending allocation, and
   that spend is not justified.**
7. **Gate 7**: v6, v7, `3163986` and B5 `3149358`–`3149363` reconfirmed untouched
   at submission time.
8. Team lead's confirmation to submit.

Until 1, 2 and 4–8 hold, no submission is authorized. A complete, verified,
correctly bound staging tree is still not permission to run.


---

## 10. §8.6 — RESTATED: verified in-job, fails closed and fails fast

Adopted from the independent review, whose reasoning is better than my own
earlier framing and is recorded here in preference to it.

**There is no second interpreter to check.** `p1.sbatch:45` runs
`/usr/bin/python3 deploy.py host`. The interpreter that executes `host()` *is*
the host interpreter. Any 3.7+ syntax fails at compile time, before line 1 of
`host()` executes, costing seconds. The package side is independently verified:
`ast.parse` clean, no `text=True`, no walrus, no positional-only parameters, no
f-string `=`.

**The `CPUs/Task` parser cannot be pre-flighted at all.** It parses a live
`scontrol` response for a *running* allocation, so it requires an allocation to
exist. It runs inside `admission.live` at **`deploy.py:295`** — after the
controller gate, and well before the SIF/corpus hashing at `deploy.py:326`. A
mismatch aborts in seconds. *(Line-cite correction: `admission.live` is at :295,
not :292; same class of drift as L2.)*

**The checkpoint guard** runs during staging — `checkpoint_inventory` at
`deploy.py:367`, re-verified at `:382` (`'epoch05 inventory changed during
staging'`) and `:388` (`'resume state must never be staged'`), with the pinned
`meta.json`/`eval.json` checks at `:372, :386, :391`. Later than the parser,
still in-job, still fail-closed, still before the 900 s probe.

A pre-flight would only re-measure what the job measures first and cheapest, and
would spend an allocation to do it worse — on a node that may not even be the one
the attempt lands on.

### Inseparable caveat (M3) — fail-fast is cheap in GPU-hours, not free

`p1.sbatch:38` writes `S0_P1_STARTED` under `noclobber` **before**
`deploy.py host` starts. So even an abort at `deploy.py:295`, seconds into the
job, permanently consumes the v8 namespace and forces a full v9 rebuild: new
bindings, ledger, `release.json`, transport, pins, authorization and independent
review. **The cost of an in-job §8.6 failure is a rebuild cycle, not
allocation.** This restatement must never be read as "§8.6 is now free."

### What does NOT discharge §8.6 — stated so it cannot be mistaken later

- **`test_recovery.RecoveryTests.test_host_python36_api` does not discharge it.**
  It is a source-level assertion about the package's own text, executed on the
  build machine. It proves the package avoids the 3.6 API traps; it proves
  nothing about any host.
- **The `gh-login03` 3.6.15 reading does not discharge it.** A login node is not
  the execution host. The reading (in-memory `compile()` → ALL_OK on the exact
  staged bytes; `text=True` rejected; `universal_newlines=True` accepted, rc=0)
  is corroboration that lowers the residual risk. It is not a pass.

### Residual risk, now evidence-backed (for the author's acceptance decision)

The residual risk is that a ghx4 compute node's `/usr/bin/python3` differs from
`gh-login03`'s 3.6.15. It remains **accepted rather than discharged** — accepting
it is the author's call, not the controller's — but it is no longer unquantified.

**Direct evidence from a real compute-node execution.** Job `3163986`'s own
stderr, preserved at `staging/final-p1-3163986.err`, contains:

```
  File "deploy.py", line 249, in host
  File "/usr/lib64/python3.6/subprocess.py", line 438, in run
```

That traceback was produced **on compute node `gh077`** (confirmed by the job's
stdout, which prints `gh077.hsn.cm.delta.internal.ncsa.edu`). It shows two
things: the compute node's system interpreter is **Python 3.6**, the same
major.minor as the login node's; and `deploy.py host` ran **to completion** under
it, reaching line 249 where it launched the container. The failure was the
`epoch05 exact inventory changed` guard firing *inside* the container — a
different phase, and the very defect this recovery attempt fixes.

Patch level does not matter here: the `text=` keyword was added in 3.7, so any
3.6.x rejects it identically, and the staged package does not use it.

**Likelihood: low.** Both node classes take `/usr/bin/python3` from the same OS
image, and we have an observed compute-node execution of this same host phase
under 3.6. The unquantified remainder is that the attempt may land on a node
other than `gh077`, and that the image could have changed since 2026-09-17.

**Cost if it happens:** the job aborts within seconds at `deploy.py:295`, the GPU
charge is negligible against the 0.5 node-hour ceiling — but `S0_P1_STARTED` has
already been written by `p1.sbatch:38`, so the v8 namespace is permanently
consumed. **The real cost is a v9 rebuild-and-review cycle, not allocation.**

## 11. Required run-time records (L3)

The attempt **must** record its own staging elapsed time, so the §8 margin
becomes measured rather than inferred. This needs no code change — the package
already emits what is required:

- `p1.sbatch:41` runs `/usr/bin/date -Is` at job start;
- `deploy.py:326-328` writes `control/large-identities.json` with a
  `measured_at_unix` per large identity (SIF, corpus, evidence);
- the probe's own control artifacts carry their write times.

Terminal reconciliation must therefore report: job start, the three
`measured_at_unix` values, probe start, and the derived staging elapsed time,
compared against the 128.8 s predicted and the 800 s available. If the measured
figure departs materially from the prediction, that is a finding about the
budget model and must be recorded as such, whatever the attempt's outcome.

## 12. Required environment exports (p1.sbatch:33-35)

`p1.sbatch` refuses to start unless all three are present and match
`^[0-9a-f]{64}$`. The reviewed v8 values are:

```
DEPLOY_CONTROLLER_GATE_SHA256=8ac56646293b82cb6918e750a463351293df12f81a6807195b3d12736e0cb20d
DEPLOY_RELEASE_TRANSPORT_SHA256=90d904824f02447de2a77e11d90919ef8d4dac646f6eb25ae9a90bdd7f020dfd
DEPLOY_RELEASE_LEDGER_SHA256=f56f3a29df50568792031d76ed9919ac2a5daef2c5340659836fcc7e1afa88c7
```

Each was verified against the artifact it pins, not copied from a note:

| Variable | Pins | Consumed at | Verified |
|---|---|---|---|
| `DEPLOY_CONTROLLER_GATE_SHA256` | `staging/dev123_p1_controller_authorization_v8.json` | `deploy.py:265-267` `pinned(path, digest)` | remote sha256 of the staged authority = `8ac56646…cb20d` |
| `DEPLOY_RELEASE_TRANSPORT_SHA256` | `release_v8.tar` | `validate_controller_gate` → `validate_authorization`, compared to the authorization's own `transport_sha256` | remote sha256 in the v8 control namespace = `90d90482…020dfd` |
| `DEPLOY_RELEASE_LEDGER_SHA256` | `release_v8/release.sha256` | `deploy.py:273` `pinned(release/'release.sha256', …)` | staged ledger sha256 = `f56f3a29…1afa88c7` |

The authorization was validated locally against these three exact values through
the real `admission.validate_authorization` before it was staged, so the
env-to-authorization-to-artifact chain is closed rather than assumed.

**They survive the environment scrub.** `p1.sbatch:20-26` unsets by prefix class
(`APPTAINER*`, `SINGULARITY*`, `TRANSGENIC*`, `HF_*`, `TRANSFORMERS*`, `PYTHON*`,
`TORCH*`, `TRITON*`, `MVP_*`, `CUDA_VISIBLE_DEVICES`) — checked, and `DEPLOY_*`
matches none of them. They must be exported into the `sbatch` environment at
submission time.
