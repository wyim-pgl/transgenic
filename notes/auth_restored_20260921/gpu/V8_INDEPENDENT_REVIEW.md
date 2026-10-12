# Independent review — v8 release package

2026-09-21 09:55 PDT. Author: Won <wyim@unr.edu>. Times PDT.

I did not build this package. I am the fresh review the rebuild requires; the
`V7_INDEPENDENT_REVIEW.md` verdict examined superseded pins and does not carry
over. I read that report for its E1–E11 table and its rebind surface, and
re-derived its load-bearing conclusions rather than inheriting them.

**No Slurm job was submitted, cancelled, requeued or modified. No remote write
was made. Nothing was deleted. No source, package or record was edited. No
operator pin was issued.** All remote access was a bounded two-hop read-only
`ssh`, two calls.

---

## Verdict

**APPROVE_WITH_FIXES**

Nothing CRITICAL or HIGH remains. The rebind is complete and structural, nothing
was relaxed in the rebuild, `authorization.example.json` is genuinely the
generated artifact, the budget closes with the margin claimed, and the scope is
unchanged. Three MEDIUM items and three LOW items are listed below; none of them
blocks, and none of them requires another rebuild.

---

## 1. Is the rebind complete and structural? — YES, verified in the code

### 1a. The single-constant claim holds

The builder claims `deploy.py:282`, `admission.py:54` and `admission.py:305` all
derive from the one `NAMESPACE` constant, so rebinding it fixes three. **I read
all four sites and the claim is exactly right:**

- `admission.py:22` — `NAMESPACE = '.../staging/dev123_p1_release_v8_20260921'`
- `deploy.py:282` — `release = Path(admission.NAMESPACE)`. No literal. This is
  the site that made `--chdir` a red herring on the v6 payload.
- `admission.py:54` — `namespace=NAMESPACE` inside `example()`, outside the
  `variable` set at `admission.py:73-75`, so `validate_authorization`
  (`admission.py:76-78`) enforces equality against the compiled value.
- `admission.py:305` — `j.get('Command') == NAMESPACE+'/p1.sbatch' and
  j.get('WorkDir') == NAMESPACE`. String concatenation on the constant.

None of the three is an independent literal. The package is not still broken in
the way the v6 payload was.

The genuinely independent literals — the ones a single-constant rebind would
*not* have reached — are `deploy.py:264`
(`ROOT / 'staging/dev123_p1_controller_authorization_v8.json'`), `deploy.py:290`
(`'dev123-p1-v8-job-' + job_id`), `p1.sbatch:2,13,14,15`,
`controller_execution_scope.json` (`.namespace`, `.job_namespace`,
`.execution_gate.authorization`) and `build_release.py:98`. **I checked each by
reading it: all nine read v8.**

### 1b. My own token sweep, covering the prior reviewer's stated blind spots

The prior review disclosed that its sweep would miss relative paths, environment
variable names, URLs and absolute paths under unanticipated roots. I covered all
four, over every member of the staged package **including the ones the builder's
sweep did not name** (`release.json`, `source-manifest.json`, `operator-pins.json`,
the three `notes/*.json`, and the 103 files inside `code.candidate.tar`):

| Sweep | Result |
|---|---|
| Any `v6`/`v7` token, all 11 text members | **0** |
| Every absolute path literal, all members, classified | no staging path other than v8 |
| `/work/**` literals | only `.../dev123_p1_release_v8_20260921` (×3) and `.../dev123-p1-v8-` (×2); everything else is runtime layout, the SIF, the corpus, and the intentional `final-p1-job-3163986` evidence source |
| Environment-variable names read by the code | `DEPLOY_CONTROLLER_GATE_SHA256`, `DEPLOY_RELEASE_LEDGER_SHA256`, `DEPLOY_RELEASE_TRANSPORT_SHA256`, `HOME`, `SLURM_JOB_ID`, `SLURM_JOB_NUM_NODES`, `TZ` — none namespace-bearing |
| URLs, any scheme | **0** |
| Relative-path literals (`../`, `./x`) in the four executables | **0** |
| `v6`/`v7` inside `code.candidate.tar` | 9 hits, **all** genome filenames (`Gmax_880_v6.0.fa`, `Osativa_323_v7.0.*`), correctly untouched |

Confirmed on the remote staged tree as well: `grep -rlE` for six namespace token
patterns across the whole directory returned **0 files**.

**Finding: none. The builder's "zero residual tokens across all six bound files"
is correct, and I extend it to all fourteen members plus the source archive.**

### 1c. Identity, computed by me

Every pin the builder claims, recomputed locally from the build directory
`notes/auth_resume_20260917/gpu/deployment_v6/release_v8/` and independently
re-measured on the remote staged tree:

| Artifact | My sha256 | Matches claim |
|---|---|---|
| Transport `release_v8.tar` (2 887 680 B) | `90d90482…f020dfd` | yes |
| `release.json` | `f716d552…61a58055` | yes |
| Ledger `release.sha256` (1181 B) | `f56f3a29…7c7e1afa88c7` | yes |
| `admission.py` | `f31bf313…589907d0` | yes |
| `deploy.py` | `8cc70ce9…f47403ab9` | yes |
| `p1.sbatch` | `efe2cc04…c3692a291f2` | yes |
| `controller_execution_scope.json` | `e8b05fa4…da8adbf6` | yes |
| `authorization.example.json` | `12418046…f679efcabca2` | yes |
| `build_release.py` | `948acf4e…d4a08b760893e7` | yes |
| `code.candidate.tar` (unchanged) | `da161088…ecc5bcec7` | yes |
| `source-manifest.json` (unchanged) | `4eb7476a…3f4c79eb0a` | yes |
| `operator-pins.json` (unchanged) | `45bd5764…56d28c72` | yes |
| three `notes/*.json` | all three | yes |

Remote staged tree, measured by me: 12 top-level entries, 14 regular files,
2 directories, **0 symlinks, 0 hidden, 0 files not `0444`**; ledger 1181 B mode
`0444`; `sha256sum -c release.sha256` → **13 OK**; directory
`drwx------ wcyim`, inode `…131249`. `NAMESPACE` at `admission.py:22` and
`--chdir` at `p1.sbatch:13` both read v8.

---

## 2. Was anything relaxed in the rebuild? — NO, and the result is stronger than claimed

I did not take the builder's "I diffed every test module and found no
differences" on the record. I ran the diff myself against the v4 originals
(`notes/auth_resume_20260917/gpu/deployment_v4/`), with v6/v8 tokens normalised
on both sides, and I extended it beyond the tests to the executables:

| File | Diff vs v4 original, namespace normalised |
|---|---|
| `admission.py` | **byte-identical** |
| `deploy.py` | **byte-identical** |
| `build_release.py` | **byte-identical** |
| `controller_execution_scope.json` | **byte-identical** |
| `authorization.example.json` | **byte-identical** |
| `p1.sbatch` | one line: `--job-name` date suffix `0917`→`0921` |
| `test_capture.py`, `test_field_alias.py`, `test_queued_gate.py`, `test_recovery.py`, `test_candidate.py` | **byte-identical** |
| `test_admission_cases.py` | one line: fixture `JobName` date suffix |
| `test_deploy.py` | two lines: fixture directory `release_v6` → `release_v8` |

Assertion and test-definition counts are **identical per module** (21/26/9/53/11/23/38
asserts; 9/8/2/11/5/6/6 test defs = 47). No assertion was weakened, deleted or
rewritten.

**I re-ran the suite myself**, Python 3.8.20, in the build directory:

```
Ran 47 tests in 14.913s
OK
```

47/47, matching the builder. The 41→47 reconciliation is corroborated
structurally rather than on the record: `test_recovery.py` contributes exactly 6
test definitions, and 41 + 6 = 47 = the sum of the per-module counts above.

**Finding: none. Between v4 and v8 there is no logic change at all — only
namespace rebinding.** That is a materially stronger statement than "no
assertion was relaxed", and it is the single most reassuring result in this
review.

---

## 3. `authorization.example.json` — genuinely the generated artifact

`admission.py:76-78` enforces `set(a) == set(template)` and, for every key
outside the `variable` set, `type(a[k]) is type(template[k]) and a[k] ==
template[k]`. A hand-edited example diverges silently and fails closed at run
time, so this needed checking rather than accepting.

**I regenerated it myself** by importing the staged `admission.py` and calling
`example()`, then searching the serialization space:

```
EXACT MATCH: json.dumps(admission.example(), sort_keys=True, indent=2) + '\n'
sha256 = 1241804620118affb65ee09d3eeff29973f5a04d67dae9842949f679efcabca2
namespace = .../staging/dev123_p1_release_v8_20260921
```

Byte-exact against the staged file. It is the compiled template's serialization,
not a substitution that happens to look right. F4 is satisfied.

One correction to the record, LOW severity — see L1: the *build* does not
generate this file.

---

## 4. Budget reconciliation — arithmetic verified exact, enumeration incomplete but immaterial

I recomputed every figure from the same code the builder enumerated from:

```
SIF 11,120,398,336 + CORPUS 23,534,252,032 + EVIDENCE 1,068,773,376
  + 3 x EVIDENCE (recover_evidence, deploy.py:97,106,107)
  + HF ~6,000,000,000 + checkpoint ~6,600,000,000
  = 51,529,743,872 B  (51.53 GB)          <- builder: "~51,529,000,000"

measured rate 1,068,773,376 / 2.671161313 =  400.1 MB/s
sha time      51.53 GB / 400.1 MB/s       =  128.8 s
available     1700 - 900                  =  800 s
break-even    51.53 GB / 800 s            =  64.4 MB/s
margin                                    =  6.21x
at 100 MB/s (quarter rate)                =  515 s  -> closes
```

Every figure reproduces. The large terms are right: `deploy.py:323-326` pins
exactly `SIF`, `CORPUS` and `EVIDENCE_SOURCE` against their declared sizes, and
`recover_evidence` (`deploy.py:97,106,107`) does make three further full passes
over the 1.069 GB evidence DB.

**One point in the builder's favour that the caveats undersell:** `sha()`
(`deploy.py:57-66`) reads in 8 MiB blocks and feeds `hashlib.sha256`, which is
the same OpenSSL primitive `sha256sum` uses. At that block size the Python
interpreter overhead is negligible, so the `sha256sum`-vs-`hashlib` caveat is
weaker than stated and the 400.1 MB/s figure transfers. The login-node caveat
stands (gh-login03 is not a ghx4 node), but the same Lustre and the same Grace
CPU family make it a reasonable proxy, and the 6.2x margin absorbs a large
error.

The estimated — not measured — write term is also comfortably absorbed: the
whole copy volume is 5.27 GB, so even at a pessimistic 50 MB/s write the staging
total is ~235 s against 800 s available.

**Enumeration completeness — see M2.** The §8 table is *not* the complete list of
hashing sites. I found further `sha()`/`pinned()` calls the table omits at
`deploy.py:248, 254, 267, 273, 274, 298, 304-305, 320, 409, 437, 487, 489, 506,
514-522`. I measured what they cost: **~9.6 MB, or 0.02 s** at the measured rate.
The conclusion is unaffected; only the claim of completeness is overstated.

**Finding: the budget closes. Recorded as MEDIUM only for the enumeration
claim, not for the arithmetic.**

---

## 5. Scope — unchanged, and structurally locked

Verified by reading the code, not the decision document:

- **Executed path.** `deploy.py:513-523` is the only thing run:
  `real_probes.py --mode real_tokenizer`, wrapped in
  `run_bounded.py --name real_tokenizer ... --timeout-seconds 900`. Confirmed.
- **No training.** `run_training.py` is in the source manifest but is **never
  invoked** by `deploy.py` — I grepped for it and for `TRAINING_DONE` and found
  no call site. `p1.sbatch:43` says so in a comment and the code agrees.
- **`p1.sbatch`** carries no `TRAINING_DONE` write. Confirmed.
- **`release.json`** carries `scope: S0_P1_DIAGNOSTIC_ONLY`, `node_hours_max: 0.5`,
  `gpu_hours_max: 2`, `training_authorized: false`, `runtime_ready: false`,
  `measured: false`, `seed: 123`, `profile: development-single`. Confirmed by
  parsing the staged file.
- **`run_role: mvp_finetune` is still a guard label, not an authorization.** Its
  only consumer is `deploy.py:239-243`, `validate_prepared`, which *rejects* a
  prepared receipt whose bindings do not carry it —
  `'old/cross-profile prepared receipt rejected'`. It grants nothing.

The scope is not merely declared, it is locked. `admission.py:52-55` fixes
`scope`, `training_authorized`, `followup_authorized`, `node_hours_max`,
`gpu_hours_max`, `one_attempt`, `seconds_max` **inside the template and outside
the `variable` set**, so `validate_authorization` (`admission.py:76-78`) rejects
any authorization that widens them. Widening the scope would require editing
`admission.py`, which changes its hash, which breaks the ledger, `release.json`
and the transport pin. I verified that chain by reading it.

`p1.sbatch:11` `--time=00:30:00` (1800 s) against `node_hours_max 0.5`, and
4 GPUs x 0.5 h = `gpu_hours_max 2`. Consistent. Outer `timeout 1700s` at
`p1.sbatch:44` sits inside the 1800 s Slurm wall.

**Finding: none.**

---

## 6. Gates

| Gate | State | Basis |
|---|---|---|
| §8.3 balance | **DISCHARGED** | 1067 ≥ 304. I re-measured `ACCOUNTS_SHA` myself on the host: `sha256sum /sw/user/scripts/accounts` = `44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5`, equal to the `admission.py:23` pin. The gate that produced 1068 is intact. |
| §8.4 live Slurm/account/node/reserve/overlap | **OPEN** | Must be a fresh bounded query immediately before submission. My `squeue` read is a state observation, not this gate. |
| §8.5 v8 control namespace + canonical authorization | **OPEN** | I verified by `ls` that **neither** `staging/dev123_p1_controller_authorization_v8.json` **nor** `staging/dev123_p1_v8_control_20260921` exists. Correct — they must be created after the rebuild, against the compiled v8 `NAMESPACE`, under the filename `deploy.py:264` reads. |
| §8.6 host checks on the execution host | **OPEN — and NOT discharged** | See below. |
| §8.7 preservation reconfirmed at submission | **OPEN** | v6, v7, `3163986` and B5 all intact right now (§7), but the gate is a submission-time reconfirmation. |

### §8.6 has not been discharged, and cannot be self-certified

Saying this plainly, as asked. The only interpreter reading available to this
review is from `gh-login03.delta.ncsa.illinois.edu` — `/usr/bin/python3` is
**Python 3.6.15** there. **That is a login node, not a ghx4 compute node, and a
login-node result does not discharge §8.6.** The interpreter check
(3.6.15 / no `Popen(text=True)`), the `CPUs/Task` parser against the 288-vs-64
reality, and the checkpoint guard for `accelerate_state`/`pending_grads` must be
re-run on the host that will actually execute the job.

`test_recovery.RecoveryTests.test_host_python36_api` passes in my run, but it
ran under Python 3.8.20 on my machine. It is a source-level assertion about the
package, not a measurement of the execution host. **Nobody should treat my green
suite, or the login-node 3.6.15 reading, as satisfying §8.6.** Neither the
builder nor the decision draft claims otherwise — §10.3 of `V8_REBUILD.md` states
the constraint correctly. I am recording that no part of the record has quietly
converted it into a pass.

**No gate is being treated as satisfied when it is not.**

---

## 7. The v7 namespace — settled

Asked to determine from the evidence how the v7 tree came to be replaced despite
`build_release.py` refusing an existing output directory, and whether any
preserved artifact was lost.

**What happened.** `build()` never wrote into the existing v7 directory. The
sequence in `V7_V8_STATE_OF_RECORD.md` §1 and §4 is: the v7-bound package was
built locally in a *new* directory (`deployment_v5/`), shipped as a transport
tar, and at 09:11–09:12 PDT the **remote** v7 tree's contents were deleted by
`find "$V7" -mindepth 1 ! -name '.transport.tar' -delete` and the new payload
extracted in their place. The `FileExistsError` guard at `build_release.py` and
its exclusive `out.mkdir()` govern the *local build output path* only; they never
saw the remote directory. There is no contradiction, and no guard was
circumvented. The deletion was performed under an overwrite authorization that
was live at the time and was withdrawn only afterwards, when the v8 instruction
arrived. The builder's initial "nothing deleted / v7 untouched" was wrong; the
builder found and corrected it themselves at 09:40 PDT, annotating both
`V8_REBUILD.md` and `v8_rebuild_evidence.txt`. I take that as the standard of
candour and hold to it here.

**Was anything lost? No — and I verified the survivors rather than accepting
them.** What was destroyed was a *second copy* of the v6-bound payload. The
binding-defect evidence survives in two independent places, both of which I
checked myself this pass:

1. **Remote `dev123_p1_release_v6_20260917`** — 13 files;
   `release.sha256` still **0 bytes, mode 0660, inode `522420016499497508`**;
   `admission.py` still contains the v6 namespace literal. Untouched, original
   inodes. This is the defect in its original bytes.
2. **Local `notes/auth_resume_20260917/gpu/deployment_v4/`** — the immutable
   build sources and `release_v6.tar`, which I read from throughout §2 of this
   review. Intact.

(`V7_SUBMISSION_BLOCKER.md` documents the mechanism with file:line literals, a
third record.)

The claim that needs retiring is "v7 is preserved as binding-defect evidence".
The evidence itself is not at risk. **Nothing irreplaceable was lost.**

Current staging contents, read by me: `dev123_p1_release_v6_20260917` (v6-bound,
original), `dev123_p1_release_v7_20260921` (v7-bound rebuild, not the 08:36
tree), `dev123_p1_release_v8_20260921` (candidate),
`dev123_p1_v6_control_20260917`, `dev123_p1_controller_authorization_v6.json`,
plus the untouched `evidence/`, `final-p1-3163986.{out,err}`,
`final_p1_release/`, `p1_release/`, `receipts/`, `OWNER.txt`.

`squeue -u wcyim`: **only** `3149358`–`3149363`, all PENDING, all `b5-seed`. **No
P1 job is queued or running.**

---

## Findings by severity

### CRITICAL — none

### HIGH — none

### MEDIUM

**M1 — `release.json:1` `review_status` is declarative, not enforced, and
becomes permanently stale.**
I grepped `deploy.py` and `admission.py` for `review_status`, `runtime_ready`,
`measured` and `scope` as *read* from the manifest: there is no consumer. Only
`manifest['release_files']`, `manifest['source_files']`,
`manifest['source_provenance']` and
`manifest['accepted_companions_not_in_source_commit']` are read
(`deploy.py:249-251, 297-320`).
*Concrete failure this causes:* two things, neither fatal. First,
`PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW` will not block a submission — it is a
process marker, not the machine-encoded precondition it reads as, and nobody
should rely on it failing closed. Second, `release.json` is hash-pinned in the
ledger, the transport and the controller authorization, so the label **cannot be
corrected without a full v9 rebuild and new pins**; the archived package will
read "never reviewed" forever.
*Fix (no rebuild):* record this review's identity in the authorization's
`review_reference` field, which `admission.py:73-75` places in the `variable`
set precisely so it can carry a per-issuance value. State in the execution
decision that `release.json`'s `review_status` is frozen by design and that
`review_reference` is the authoritative record.

**M2 — `V8_REBUILD.md` §8 overstates the completeness of the hashing
enumeration.**
The table presents eight sources as the hashing work "enumerated from the code
rather than estimated". It omits `deploy.py:248, 254, 267, 273, 274, 298,
304-305, 320, 409, 437, 487, 489, 506, 514-522`.
*Concrete failure this causes:* none to the budget — I measured the omitted
volume at ~9.6 MB, 0.02 s at 400.1 MB/s, against a 660 s headroom. The failure is
to the *claim*: a reader who audits the table against the code will find sites
missing and may reasonably doubt the whole analysis.
*Fix:* add one line to §8 stating that the table enumerates the large terms and
that all remaining `sha()`/`pinned()` sites together contribute under 10 MB.

**M3 — one namespace buys exactly one attempt, and this should be stated
up front.**
`p1.sbatch:36` refuses to start if `S0_P1_STARTED` exists, and `:38` writes it
under `set -o noclobber` **before** `deploy.py host` runs and therefore before
any admission gate can stop the job. The v8 directory is `drwx------ wcyim` (I
verified), so the write will succeed.
*Concrete failure this causes:* any failure after `p1.sbatch:38` — including a
node fault or a Lustre hiccup that is nobody's fault — permanently consumes the
v8 namespace. A retry then needs a complete v9 rebuild, new pins, a new
authorization and another independent review. This is deliberate and I am not
asking for it to change, but it converts "one attempt" from a policy into a
physical property of the package, and the execution decision should say so
explicitly rather than leaving it to be discovered on the day.

### LOW

**L1 — `V8_REBUILD.md` §3 says `authorization.example.json` is "generated by the
build from `admission.example()`".** `build_release.py:78` only *copies* it
alongside the other five bound files; the build generates nothing. The substance
is nonetheless correct — I reproduced the file byte-exactly from
`admission.example()` — so this is a wording fix, not a defect. The earlier
`V7_REVIEW_RECONCILIATION.md` phrasing ("I regenerated it ... and compared
byte-for-byte") was the accurate one.

**L2 — line-number drift.** `V8_REBUILD.md` §8 cites `deploy.py:324` for the
SIF/CORPUS/EVIDENCE pins; the loop header is at `:323-324` and the `pinned()`
call is at `:326`. Cosmetic, but these documents are audited by file:line.

**L3 — do not describe the §8 budget work as closing the v6 failure.** The
recorded root cause of `3163986` is an exact-inventory check rejecting
`accelerate_state` and `pending_grads` (`V7_INDEPENDENT_REVIEW.md` §3, which I
re-read against `deploy.py:115-126, 367, 382, 385`), not a throughput stall. The
§8 analysis is a prudent margin check against a *hypothesis*, and I did not
independently verify the "I/O-stall" characterisation attached to the v6 tree in
`v8_rebuild_evidence.txt`. Keeping the two separate matters, because if the
attempt does stall, an analysis that was never the diagnosis should not be read
as having ruled it out. The builder's own recommendation — have the attempt
record its staging elapsed time — is the right one and should be carried into
the execution decision as a requirement, not a suggestion.

---

## What I verified by my own computation vs. accepted on the record

**Computed myself, locally:**
- All 14 member sha256 values and the transport hash, from the build directory
- The complete token sweep (§1b), including inside `code.candidate.tar`
- That `deploy.py:282`, `admission.py:54` and `admission.py:305` derive from
  `admission.py:22`, by reading all four
- That all nine independent namespace literals read v8, by reading each
- The v4→v8 diff of every executable and every test module, namespace normalised
- Per-module assertion and test-definition counts (47 total)
- The full test suite: **Ran 47 tests, OK**, Python 3.8.20
- Byte-exact regeneration of `authorization.example.json` from
  `admission.example()`, and its exact serialization form
- Every figure in the §8 budget arithmetic, plus the omitted sha sites and their
  9.6 MB cost
- That `run_training.py` and `TRAINING_DONE` have no call site in `deploy.py`
- That `run_role` is consumed only by `validate_prepared` (`deploy.py:239-243`)
- That no code path reads `review_status`, `runtime_ready`, `measured` or `scope`
  from `release.json`
- That `admission.py:52-55` + `:73-78` structurally prevent an authorization from
  widening the scope
- `sha()`'s 8 MiB block size (`deploy.py:57-66`), which strengthens the
  `sha256sum`-vs-`hashlib` caveat

**Measured myself, remotely, read-only (two bounded calls):**
- v8 staged tree: 12 entries / 14 files / 2 dirs / 0 symlinks / 0 non-0444;
  ledger 1181 B mode 0444; `sha256sum -c` → 13 OK; six key hashes equal to my
  local values; `NAMESPACE` and `--chdir` read v8; 0 residual tokens; directory
  `drwx------ wcyim` inode `…131249`
- v6 preserved: 13 files, `release.sha256` 0 B / mode 660 / inode `…497508`,
  `admission.py` still v6-bound
- `ACCOUNTS_SHA`: `sha256sum /sw/user/scripts/accounts` = `44302f2e…be35f5`
- Gate 5 artifacts **absent**: no `dev123_p1_controller_authorization_v8.json`,
  no `dev123_p1_v8_control_20260921`
- `squeue -u wcyim`: only `3149358`–`3149363` PENDING; no P1 job
- `/usr/bin/python3 --version` on `gh-login03` = 3.6.15 (**login node — does not
  discharge §8.6**)

**Accepted on the record, not independently verified:**
- The 400.1 MB/s sha256 and 5.3 GB/s Lustre measurements (single bounded
  read-only run on gh-login03; I verified the arithmetic built on them, not the
  measurements themselves)
- The sizes of the SIF (11.12 GB), corpus (23.53 GB), HF cache and checkpoint —
  taken from the code's declared pins and the E3–E5 table
- The 09:11–09:12 PDT deletion sequence and the timing of the overwrite
  authorization's withdrawal (reconstructed from `V7_V8_STATE_OF_RECORD.md`;
  I verified the *outcome* by inode, not the sequence)
- The 1067 ≥ 304 balance derivation (verified in the prior review; I re-measured
  only the `ACCOUNTS_SHA` gate that produces it)
- The root cause of `3163986` (per `V7_INDEPENDENT_REVIEW.md` §3; I re-read the
  code it cites but did not examine the job's own logs)

---

## Exact remaining conditions before a submission could be authored

1. **§8.5** — create `staging/dev123_p1_v8_control_20260921` (verified absent
   beforehand; it is absent now) and issue the canonical
   `staging/dev123_p1_controller_authorization_v8.json` **after** this review,
   against the compiled-in v8 `NAMESPACE` and the §1c pins, as canonical JSON
   (`deploy.py:269-271`), owner-matched (`deploy.py:266`), with the attempt
   journal created exactly once. Carry this review in `review_reference` (M1).
   The v6 authority must not be reused.
2. **§8.6** — run the host checks on the **ghx4 compute node that will execute
   the job**: interpreter 3.6.15 / no `Popen(text=True)`, the `CPUs/Task` parser
   against 288-vs-64, and the `accelerate_state` / `pending_grads` checkpoint
   guard. A login-node result does not count.
3. **§8.4** — a fresh bounded live Slurm/account/node/reserve/overlap query
   immediately before submission, no query retry.
4. **§8.7** — reconfirm at submission time that v6, v7, `3163986` and B5
   `3149358`–`3149363` are untouched.
5. **M1** — record `review_reference`; state in the execution decision that
   `release.json`'s `review_status` is frozen and non-enforcing.
6. **M3** — state in the execution decision that a failure after `p1.sbatch:38`
   permanently consumes the v8 namespace and that a retry requires a v9 rebuild.
7. **L3** — require the attempt to record its own staging elapsed time, so the
   6.2x margin is measured on the first real run rather than inferred.
8. The environment variables `DEPLOY_CONTROLLER_GATE_SHA256`,
   `DEPLOY_RELEASE_TRANSPORT_SHA256` and `DEPLOY_RELEASE_LEDGER_SHA256` must be
   exported at submission with the §1c values; `p1.sbatch:33-35` refuses to start
   without all three.
9. Team lead's confirmation.

M2, L1 and L2 are record corrections and do not gate a submission.

**APPROVE_WITH_FIXES.**
