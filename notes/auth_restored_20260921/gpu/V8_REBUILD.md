# v8 release rebuild — full rebind, I/O budget reconciled, 2026-09-21 09:30 PDT

Author: Won <wyim@unr.edu>. Times PDT. Raw evidence: `v8_rebuild_evidence.txt`.

**No job was submitted. v6 is untouched.** NOTE, corrected 09:40 PDT: the
sentence "nothing was deleted / v7 untouched" that first appeared here was
WRONG. The v7 tree's original contents were deleted and replaced at 09:12 PDT
under the then-live overwrite authorization, before the v8 decision arrived.
See `V7_V8_STATE_OF_RECORD.md`.
The rebuilt v8 package requires a fresh independent review and the team lead's
confirmation before any submission.

Namespace: `/work/nvme/.../staging/dev123_p1_release_v8_20260921`
(verified absent before `mkdir`; created 09:25 PDT).

---

## 1. Rebind inventory — what I swept and how

Three independent passes have now run over this payload (mine, the reviewer's,
the team lead's read of `build_release.py`), and each found something the others
missed. I therefore treated the instruction "assume a third gap exists" literally
and swept by pattern class rather than by known token.

**Method.** Four sweeps over every transport member and every build source:
(A) regex for version-ish, dated, `dev123*` and `staging/*` tokens;
(B) the candidate payload members separately (`source-manifest.json`,
`operator-pins.json`, `release.json`, three `notes/` members);
(C) every use of the `NAMESPACE` symbol, to catch derived enforcement;
(D) every exclusion enforcement point.

### 1a. Rebound literals — 9 sites, 6 files (plus 6 test sites)

| # | Site | Now bound to |
|---|---|---|
| 1 | `admission.py:22` `NAMESPACE` | `…/dev123_p1_release_v8_20260921` |
| 2 | `deploy.py:264` authority **filename** | `…_controller_authorization_v8.json` |
| 3 | `deploy.py:290` job dir prefix | `dev123-p1-v8-job-` |
| 4 | `p1.sbatch:2` `--job-name` | `dev123-p1-v8-0921` |
| 5 | `p1.sbatch:13` `--chdir` | `…_v8_20260921` |
| 6 | `p1.sbatch:14,15` `--output`/`--error` | `dev123-p1-v8-%j.{out,err}` |
| 7 | `controller_execution_scope.json:34` authority | `…_v8.json` |
| 8 | `controller_execution_scope.json:56` namespace | `…_v8_20260921` |
| 9 | `controller_execution_scope.json:57` job namespace | `dev123-p1-v8-job-ID` |
| 10 | `authorization.example.json:44` namespace | regenerated, not edited (§3) |
| 11 | `build_release.py:98` output dir | `release_v8` |
| 12 | test suite, 6 sites / 3 modules | v8 equivalents |

Verified on the **staged** tree: residual v6/v7 token count is **0** in
`admission.py`, `deploy.py`, `p1.sbatch`, `controller_execution_scope.json`,
`authorization.example.json` and `build_release.py`.

### 1b. C1 and C3 — enforcement points, not separate literals

The review's C1/C3 sites are **derived from the single `NAMESPACE` constant**, so
rebinding `admission.py:22` fixes all of them. Confirmed by sweep C:

- `deploy.py:282` — `release = Path(admission.NAMESPACE)`. This is the one that
  made `--chdir` a red herring: the job reads whatever `NAMESPACE` says, wherever
  it was submitted from. Now reads v8.
- `admission.py:54` — `namespace=NAMESPACE` inside the template, outside the
  `variable` set, so `validate_authorization` enforces equality against it.
- `admission.py:305` — `Command == NAMESPACE+'/p1.sbatch'` and
  `WorkDir == NAMESPACE`.

The review's conclusion that "there is no compliant submission of this payload at
all" was correct **for the v6-bound payload**. It is resolved for v8 not by
patching three sites but because all three read one constant that now names v8.

### 1c. Deliberately unchanged, with reasons

| Literal | Why |
|---|---|
| `deploy.py:21` `mvp_est8_20260916_1545` campaign root | Same campaign; this is the `3163986` recovery, not a new window |
| `admission.py` `START='2026-09-16'` | Same, deliberately retained |
| `admission.py:45` / `deploy.py:29` evidence path under `final-p1-job-3163986` | The authorized read-only recovery input |
| `admission.py:370-376` `3163986` start `18:47:46`, end `18:48:37`, placement `gh077`, cpus `288` | **Intentional.** Terminal and immutable, and pinning the exact incident is correct for a recovery attempt — this package is tied to that incident by design, not by oversight |
| `source-manifest.json` `source_origin` | Build provenance; changing it would falsify the record |
| `gh062,gh093` | See §4 |
| Container bind targets (`/work/repo`, `/mvp`, `/scratch`, `/corpus.db`) | Created by `deploy.py:337-358`; internal to the container namespace |

### 1d. Swept and clean — what I checked, so the next pass need not redo it

- `operator-pins.json` — no namespace, host, user, account or path literal. It is
  derived wholly from the source payload; its hash is **unchanged** across v6, v7
  and v8 (`45bd5764…6d28c72`), which is itself the proof.
- `source-manifest.json` — one match, `3124054`, which is a **sha256 substring,
  not a job ID** (independently the reviewer's conclusion too). Hash unchanged.
- Three `notes/` members — the numeric matches are B5 job IDs inside historical
  inventory **data**, not bindings. Hashes unchanged.
- `release.json` — regenerated by the build; its only path literal is
  `source_origin`, never dereferenced at run time.
- `p1.sbatch:20-26` — scrubs the environment by prefix class, so it needs no
  per-variable maintenance.

**On `build_release.py` specifically** (the file previously swept by regex only):
I read it end to end. Two mechanics matter and both were handled *before* the
build ran, not after — the build copies `deploy.py`, `admission.py`,
`authorization.example.json`, `p1.sbatch`, `build_release.py` and
`controller_execution_scope.json` from `HERE`, its own directory, and it **ships
itself**. So I rebound all six in a fresh build directory `deployment_v6/` and
then built. Rebinding the staged output afterwards would have desynced the
release from its own ledger.

## 2. Output path — v8, nothing deleted

`build()` refuses to write into an existing directory
(`if out.exists() or out.is_symlink(): raise FileExistsError`, then an exclusive
`out.mkdir()`). A new v8 namespace satisfies that constraint with no special
handling and no deletion, and matches the authorization's "namespace absent
before creation" rule. Built in `deployment_v6/` with `deployment_v4/` and
`deployment_v5/` left intact.

## 3. F4 — `authorization.example.json` regenerated, verified

Generated by the build from `admission.example()` and then checked: regenerating
it independently and comparing byte-for-byte with the staged file gives
**IDENTICAL**, with `namespace` reading the v8 path. No hand edit anywhere in the
v8 path.

## 4. Node exclusions — unchanged, coordinated across all five sites

Live query found **no** node in `ghx4` down, drained, failing or unresponsive;
`gh062`/`gh093` are healthy; no B5 job is placed. There is no basis to add an
exclusion, so I left the literal alone — which is also the only safe option
given the coupling, since `admission.py:319` demands the exact string
`'gh[062,093]'` or `'gh062,gh093'`. All five sites remain consistent:
`admission.py:66`, `:318`, `:319`, `p1.sbatch:12`,
`controller_execution_scope.json:58`.

## 5. New v8 pins

| Artifact | sha256 |
|---|---|
| **Transport `release_v8.tar`** (2887680 B) | `90d904824f02447de2a77e11d90919ef8d4dac646f6eb25ae9a90bdd7f020dfd` |
| **`release.json`** | `f716d55256269c58da4f38028ea2660dfe6dc96db656651ea1a37f2761a58055` |
| **Ledger `release.sha256`** | `f56f3a29df50568792031d76ed9919ac2a5daef2c5340659836fcc7e1afa88c7` |
| `admission.py` | `f31bf313692b59fcc8ece9363fad9123e48dba129ea51039a725ae95589907d0` |
| `deploy.py` | `8cc70ce9e1a20fda8dbc24ddd2f4005a171ef400879da90d8a22c5bf47403ab9` |
| `p1.sbatch` | `efe2cc04b74b0914eaa8699c5f80fbee3ace0a690a8de46330aa4c3692a291f2` |
| `controller_execution_scope.json` | `e8b05fa4f25a820ef0f568e4390d60001c720087352b68113d23bfd0da8adbf6` |
| `authorization.example.json` | `1241804620118affb65ee09d3eeff29973f5a04d67dae9842949f679efcabca2` |
| `build_release.py` | `948acf4ea35bb279c65bc7fa4b63dac6d068ecc329b1a00ec4d6a08b760893e7` |
| unchanged: `code.candidate.tar` | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |
| unchanged: `source-manifest.json` | `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a` |
| unchanged: `operator-pins.json` | `45bd57640d7c55ee7ce7b1f20ab48b58729f5e58b880c4608813b99356d28c72` |

v6 pins (`26c1dc0a…`, `64a547b8…`, `74a740ea…`) and v7 pins (`73123b3f…`,
`d65684a6…`, `902c9bd6…`) are both **superseded**.

Transport built with the recipe already validated by reproducing
`release_v6.tar` byte-exactly.

## 6. Tests

**47/47 OK** (11.772 s). Only the namespace rebindings changed; no assertion was
relaxed, deleted or rewritten.

## 7. Staging verification

14 files, 0 symlinks, 0 hidden, 0 owner-writable, 0 files not 0444; ledger
`sha256sum -c` → **13 OK, 0 FAILED**; all 14 remote hashes equal the locally
computed build outputs; 0 residual v6/v7 tokens.

**Preservation after the write:** v6 still 13 files with `release.sha256` at
0 bytes / mode 0660 / inode `…497508` — fully intact, and it is where the
original v6 binding is preserved. The v7 path holds the *v7-bound rebuild*
(`release.json` `d65684a6…`), **not** the original 08:36 staging tree, which was
deleted at 09:12 PDT. `3163986` and B5 `3149358`–`3149363` untouched.

## 8. The 1700 s budget — reconciled, not deferred

Enumerated from the code rather than estimated. Inside the outer
`timeout 1700s`, the sha256 work is:

| Source | Bytes |
|---|---|
| `pinned(SIF)` `deploy.py:324` | 11,120,398,336 |
| `pinned(CORPUS)` `deploy.py:324` | 23,534,252,032 |
| `pinned(EVIDENCE)` `deploy.py:324` | 1,068,773,376 |
| `recover_evidence` `:97,:106,:107` — three further full passes | 3,206,320,128 |
| HF cache ≤ 2 GB, 3 passes (`:200,:202`) | ~6,000,000,000 |
| checkpoint ≤ 2.2 GB, ~3 passes | ~6,600,000,000 |
| **Total** | **~51.53 GB** |

Measured on the host, read-only: **sha256 at 400.1 MB/s** (1.07 GB in 2.671 s)
and raw Lustre read at **5.3 GB/s**. The binding constraint is sha256 CPU, not
the filesystem.

```
outer budget                                1700 s
probe (run_bounded --timeout-seconds 900)    900 s
available for staging + hashing              800 s

sha256   51.53 GB / 400.1 MB/s          =  128.8 s
copies   5.27 GB read @ 5.3 GB/s        =    1.0 s
         5.27 GB write @ 500 MB/s (est) =   10.5 s
measured/estimated staging I/O          ~  140 s
generous allowance for startup/quota/small files  <= 300 s

HEADROOM 660 s (500 s at the generous figure)
break-even: 51.53 GB / 800 s = 64.4 MB/s required; measured 400.1 = 6.2x margin
even at 100 MB/s (quarter rate): 515 s < 800 s — still closes
```

**Conclusion: the budget closes with 6.2× margin. No budget or scope change is
required.** The dominant term is SIF + corpus (34.65 GB, 87 s), which are
read-only identity checks on files that never move.

Caveats stated plainly: measured on login node `gh-login03`, not a ghx4 compute
node (same Lustre); `sha256sum` vs Python `hashlib` are comparable but not
identical; **write throughput was estimated, not measured**, because writing
outside the authorized namespaces is not permitted; Lustre varies with load and
the dd figure may include client cache. The conclusion survives all four — it
still closes at a quarter of the measured rate.

**Recommendation:** have the attempt record its own staging elapsed time, so the
margin is measured on the first real run instead of remaining an inference.

## 9. `ACCOUNTS_SHA` — re-measured twice, matches

`/sw/user/scripts/accounts` hashes to `44302f2e…be35f5`, equal to the
`admission.py:23` pin, at 09:18 and again at 09:23 PDT (8705 B, mode 755, owner
`svcdeltaswmgt`, mtime 2025-08-05). The site-owned script has not moved, so the
gate that produced the 1068 figure holds. This remains the item most exposed to
drift outside the author's control and should be re-checked at submission time.

## 10. Remaining conditions

1. **Fresh independent review of the v8 package** — neither prior review
   transfers; both examined superseded pins.
2. Gate 5: v8 control namespace created (absent-verified) and the canonical
   `dev123_p1_controller_authorization_v8.json` **issued after this rebuild**,
   against the compiled-in v8 `NAMESPACE` and the §5 pins, with the attempt
   journal created once.
3. Gate 6: host checks on the actual execution host. **This cannot be
   self-certified by the release** — the interpreter check must run where the job
   will run.
4. Gate 4: fresh bounded live Slurm/account/node/reserve/overlap query
   immediately before submission.
5. Gate 7: v6, v7, `3163986` and B5 reconfirmed untouched at submission time.
6. Team lead's confirmation.

Gate 3 is discharged: `max(0, 1068−1) = 1067 ≥ 304`.
