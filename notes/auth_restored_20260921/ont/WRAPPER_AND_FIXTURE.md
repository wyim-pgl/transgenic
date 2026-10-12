> 🧊 **이 트랙은 2026-09-21 저자 결정으로 종료됐다 — "ONT full 사용하지 않는다."**
> 여기 기록된 wrapper 보강·커버리지·테스트 결과는 사실로 유효하지만 **소비처가 사라졌다.**
> v5 빌드와 원격 fixture 실행은 하지 않는다. 보존 목적의 문서다.
> MVP 패널 정렬(`mvp_align_panel_longreads.py`)은 **별개이며 이 wrapper에 의존하지 않는다.**
> 정본: `notes/auth_restored_20260921/ONT_FULL_DROPPED_20260921.md`.

# ONT full wrapper and real-tool fixture — 2026-09-21

Preparation and verification only. No Slurm job was submitted, cancelled, requeued or
modified. No commit, merge or push. No ONT artifact, failure record or receipt was edited
or deleted. The `.extern OUT_OF_MEMORY 0:125` operational gate is **not** waived,
reinterpreted or narrowed anywhere below, and CLEAN / F4 / full ONT remain BLOCKED.

Working copy: `notes/auth_restored_20260921/ont/ont_full_v2/`.
Unmodified originals preserved at `.../ont_full_v2/baseline/` and at their original paths.

---

## 1. Correction to the record: what job 6165965 actually exercised

`notes/seed123_recovery_20260917_1400/ont_full/wrapper.py` and
`notes/auth_resume_20260917/ont_fixture_v4/package/wrapper.py` are **byte-identical**
(`314d6027d9d146d63c6f1b2e2b71783ac5e1f63e95e6e31252614ddcf20728dd`). The fixture job
therefore deployed the F1–F3-fixed full wrapper. That does not mean it ran it.

`ont_fixture_v4/fixture.py` imports the wrapper as a module and reimplements the
orchestration itself. Verified by AST over both files:

| Wrapper functions (11) | Called by `fixture.py` (3) | Never reached by the real run (8) |
| --- | --- | --- |
| `require`, `identity`, `pinned`, `check_source_closure`, `fresh`, `save`, `stop_group`, `child`, `check_boundary`, `slurm`, `execute` | `child`, `identity`, `save` | `require`, `pinned`, `check_source_closure`, `fresh`, `stop_group`, `check_boundary`, `slurm`, `execute` |

So the real-tool evidence from 6165965 covers child-process handling, file identity and
atomic receipt writes. It covers **none** of the production entry point: no `execute()`,
no F1 closed-tree preflight, no F2 interpreter floor, no scheduler binding, no wall-budget
alarm, no publication path. `slurm()` has never run in any real allocation, and the
existing synthetic suite patches it out entirely.

This is a coverage finding about the fixture's design. It is not a defect found in the
wrapper, and it does not reduce what the RESULT_REVIEW independently verified.

## 2. What is real and what is still synthetic in the fixture

**Real, from `raw/out/runtime_observed.json` of job 6165965** — the actual approved runtime
tooling, pinned by SHA256 at canonical site paths:

| Tool | Version | Path | SHA256 (prefix) |
| --- | --- | --- | --- |
| minimap2 | 2.28-r1209 | `/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/minimap2` | `8a561f2c…` |
| samtools | 1.3.1 | `…/RNASEQ_bch709/bin/samtools` | `c1d24b1c…` |
| seqkit | 2.13.0 | `…/conda_envs/…/bin/seqkit` | `fe5cbb44…` |
| python | 3.11.14 | `…/conda_envs/sylvan/bin/python3.11` | `eaf04015…` |

The nested ledger (`nested_execution.jsonl`, 9 records) shows real minimap2 invoked 3×,
samtools 2×, seqkit 1×, exits `[0,0,4,0,0,0,0,0,0]` with the 4 belonging to the expected
UNRESOLVED orientation audit. Real BGZF/BAM output and the two-junction oracle were
independently verified offline.

**Still synthetic**, unchanged from the four seams the receipt itself declares:

1. synthetic enum-labelled role/input root, not canonical authority;
2. explicit fixture panel hash over 50 overlapping test loci, not the canonical panel;
3. synthetic reference binding, not Git92;
4. junction-bootstrap argv recorded separately.

Plus, newly identified here: **the orchestration itself is synthetic** — `wrapper.execute()`
was bypassed (§1).

The input scale limitation (7 reads / 64 kb) is unchanged and cannot be resolved locally.
**No approved runtime tool exists on this host:** `minimap2` is absent; local `samtools` is
1.13 (approved 1.3.1); local `seqkit` is v2.0.0 (approved 2.13.0). Exercising the approved
runtime on real-scale input requires an authorized Slurm job, which is outside this scope
and blocked by the unresolved gate. Nothing below simulates that.

## 3. What the full wrapper now covers

Four gaps were closed, each chosen because a real allocation already proved it necessary
and none of them existed in the wrapper.

- **Exact CPU affinity list persisted.** `slurm()` previously asserted only
  `len(os.sched_getaffinity(0)) == budget['cpus']` and discarded the identifiers — which is
  precisely why 6165965 left "exact affinity list not independently observed" open. It now
  returns `affinity=sorted(os.sched_getaffinity(0))` into the receipt. This closes the
  capability for future runs; it does **not** retroactively supply 6165965's list, which
  remains permanently unobserved.
- **Output file-size ceiling.** `enforce_output_limit(mib)` records the inherited
  `RLIMIT_FSIZE` tuple to stderr, sets both soft and hard limits inside the allocated task,
  and verifies the result. This ports the v4 `FILE_LIMIT_FIX` (which existed only in
  `fixture.py`) into the wrapper, where a full run needs it and had nothing. The ceiling is
  request-declared with no default, so it is not a fixture-sized number applied to a full run.
- **cgroup observation before *and* after the workload.** `cgroup_observation(spec)` resolves
  this task's memory controller via `/proc/self/cgroup` and records
  `memory.limit_in_bytes`, `memory.memsw.limit_in_bytes`, `memory.use_hierarchy`,
  `memory.failcnt`, `memory.memsw.failcnt` and `memory.oom_control`. 6165965 saved a
  **pre-workload view only**, which is explicitly cited as an evidence limit. This
  **records, it does not adjudicate**: it establishes no threshold, asserts no `failcnt`
  or `under_oom` value, and has no bearing on whether the extern accounting is explained.
  It must not be read as progress toward clearing the gate.
- **Scratch budget.** `check_scratch(paths, mib)` measures what the task actually wrote
  against a declared budget. The independent review recorded "Scratch has no request field
  or enforcement"; it now has both.

Request-contract change (no production request exists yet, so nothing is invalidated):
`budget.output_mib`, `budget.scratch_mib` and `cgroup={'required': True, 'root': …}` are now
required, fail-closed, with no defaults. A missing key raises `ValueError` with a named
message, not `KeyError`.

## 4. Exact commands and real results

All run from `notes/auth_restored_20260921/ont/ont_full_v2/`. Logs under `evidence/`.

**RED** — the 7 new tests against the unmodified baseline wrapper:

```
cp baseline/wrapper.py test_wrapper.py .red/ && cd .red && \
  timeout 300 /usr/bin/python3 -I -B test_wrapper.py -v
```
→ exit 1, `Ran 21 tests`, `FAILED (failures=4, errors=21)`. All 7 new tests fail; 3
pre-existing tests also fail because the harness now supplies the new request contract the
baseline wrapper does not honor. Log `evidence/red_python310.txt`
(`32914b7d531549b04c9c0e1a4a80e1c9378bb9467a71e958218c13dd0932555e`).

**GREEN** — both independently supported interpreters:

```
timeout 300 /usr/bin/python3 -I -B test_wrapper.py -v                      # 3.10.12
timeout 300 /tmp/mvp-final-source-env/bin/python3.12 -I -B test_wrapper.py -v   # 3.12.2
```
→ `Ran 21 tests … OK`, exit 0 on both (27.5 s and 33.7 s). Logs
`evidence/green_python310.txt` (`92256351…`), `evidence/green_python312.txt` (`94485fff…`).

**F2 guard still holds** on the unsupported local 3.8.20:

```
timeout 60 python -I -B wrapper.py --request /nonexistent/request.json \
  --request-sha256 0 --wrapper-sha256 0
```
→ exit 1, `ValueError: Python >=3.9 required; use the independently pinned deployment
interpreter`, raised before any path access. Log `evidence/python38_rejection.txt`.

Two of the new tests exercise **real** `RLIMIT_FSIZE` in real child processes: one asserts
the applied `[4194304, 4194304]` tuple, one confirms an oversized write is actually stopped
at the ceiling. The remaining new tests use declared synthetic substitutions (a fake
`scontrol` emitting real scontrol field syntax, and a fake cgroup v1 memory controller,
since a test host has no Slurm job cgroup). **These are not compute containment and not
real-tool scientific evidence.**

**F1 closure re-checked against the actual deployed trees** (this had only ever been tested
against synthetic copies). Both PASS:
`notes/auth_resume_20260917/ont_fixture_v4/package/frozen` and
`notes/wait_work_20260916/longread/result_review/release/frozen`.
An initial REJECT was my own relative-path call error, not a wrapper defect.

**Test-harness change:** the historical release path was hardcoded as `HERE.parents[1]`,
which silently breaks if the suite moves. It now walks up to locate
`wait_work_20260916/longread`. The unmodified baseline suite was confirmed green (14 tests)
in the new location before any wrapper change.

Identities: `wrapper.py` `f53b5cd50f8266c3e572b172a66f032345e62296961d5564001495cda75f1acc`
(baseline `314d6027…`, +72 lines); `test_wrapper.py`
`557380e0cca22ab8e1e1f47b08079ab5b5c1fdc395cce4f9671f09b4b90a0b3c`
(baseline `4cb563cb…`, +140 lines). Both originals verified unchanged at their original paths.

## 5. Still-unresolved gate conditions

Nothing here changes any of these.

- **`.extern OUT_OF_MEMORY 0:125` — UNRESOLVED.** Re-confirmed read-only on pronghorn at
  2026-09-21: `6165965 COMPLETED 0:0`, `.batch COMPLETED 0:0`,
  `.extern OUT_OF_MEMORY 0:125`, `.0 COMPLETED 0:0`; 4 s, 4 CPUs, cpu-50, unchanged. One
  `sacct` query, no writes, no submission.
- **Administrative logs not obtained.** The kernel / slurmstepd / cgroup history for
  6165965 identifying any OOM victim PID, step, effective extern memory limit and cleanup
  action is admin-side and cannot be produced from here. The support draft at
  `notes/wait_work_20260916/repro/SLURM_EXTERN_SUPPORT_DRAFT.md` remains **unsent**.
- **Exact affinity list and reaping for 6165965 — permanently unobserved.** §3 fixes the
  wrapper for future runs only. No diagnostic job was run to fill the gap.
- **F4 real end-to-end integration — still open.** The approved runtime has never traversed
  `wrapper.execute()` (§1), and the input remains 7 reads / 64 kb.
- **Runtime trust limit unchanged.** Hash-before-read plus later identity checking is not an
  immutable snapshot; non-isolated nested helpers still load site runtime packages.
- CLEAN / F4 / full ONT: **BLOCKED**. No structural availability, pooling, HiFi policy,
  scores, decision or training eligibility is claimed or implied.

## 6. What needs author or admin action

1. **Admin (blocking the gate):** kernel/slurmstepd/cgroup records for job 6165965 on
   cpu-50, 17:14:55–17:14:59 PDT 2026-09-17 — victim PID, owning step, effective extern
   memory/swap limit, process membership, cleanup action. Sending the existing draft needs
   explicit authorization; it has not been sent.
2. **Controller / independent review:** whether the four additions in §3 are accepted in
   preparation scope, and whether the request-contract change (§3) is approved.
3. **Author / controller decision — the one real fork:** the fixture currently bypasses
   `wrapper.execute()`. Making a future fixture drive the production entry point is a
   design change to the fixture, and building a v5 package that cannot be run under the
   present gate would be speculative. I did not build one. That call is not mine.
4. **Gate scope (already flagged, unchanged):** per `EXTERN_OOM_GATE_RECHECK_20260917.md`
   the extern-clean condition is a conservative *operational* review gate, not a scientific
   policy. Re-scoping it is a controller/independent-reviewer decision. Already-approved
   policy is not re-requested here.

Separately owned, untouched by this track: PacBio CCS/FLNC product identity, and the A-HiFi
ERR13994458 location/output comparison. The PacBio 6-run download is complete and was not
re-downloaded.

Author for attribution: Won <wyim@unr.edu>.

---

# Addendum — `execute()` bypass fixed, 2026-09-21 (author-approved)

Author accepted the four §3 additions and the request-contract change, and directed
that the `execute()` bypass be fixed now, design only. **No v5 package was built.**
All prohibitions still honored: no Slurm submission, no new ONT run, no retry of
6165965 or 6165962, no existing job touched, no commit/merge/push, nothing about the
`.extern OUT_OF_MEMORY 0:125` gate weakened or reinterpreted, CLEAN/F4/full ONT still
BLOCKED. The four additions are intact and still covered.

## A1. Authoritative per-function list

The 3 vs 3–5 discrepancy was mine. My first AST pass matched attribute access on the
name `wrapper` only, and missed `w.check_source_closure(...)` at `fixture.py:77`, where
the wrapper module is bound to `w` inside `verify_release`. A scope-aware pass that
resolves every alias of the wrapper module (`w` in `verify_release`, `wrapper` in
`main`) gives the authoritative count. Note `panel_seam` also binds a local `w`, but to
`driver.w` — a different module — which is why the analysis must be scope-aware.

Directly called by `ont_fixture_v4/fixture.py`: **4 of 11** — `check_source_closure`,
`child`, `identity`, `save`.

Also executed transitively (intra-wrapper edges `child → save, identity, stop_group`;
`identity → require`; `check_source_closure → require`): `require`, `stop_group`.

| Wrapper function | Under the v4 fixture | Now, through `execute()` |
| --- | --- | --- |
| `check_source_closure` | called directly | REACHED |
| `child` | called directly | REACHED |
| `identity` | called directly | REACHED |
| `save` | called directly | REACHED |
| `require` | transitive only | REACHED |
| `stop_group` | transitive only | REACHED |
| `check_boundary` | **never executed** | REACHED |
| `execute` | **never executed** | REACHED |
| `fresh` | **never executed** | REACHED |
| `pinned` | **never executed** | REACHED |
| `slurm` | **never executed** | REACHED |

So: 4 directly, **6 of 11 executed at runtime**, **5 of 11 never executed at all**
(`check_boundary`, `execute`, `fresh`, `pinned`, `slurm`). This supersedes the "3 of 11"
in §1; the conclusion there is unchanged. My earlier list of unreached functions wrongly
included `stop_group` — it does run, from `child`'s `finally`.

## A2. What changed

`fixture_execute.py` (new) replaces the v4 driver's duplicated orchestration. It pins
the release identities, builds the request, and hands control to `wrapper.execute()`,
which now performs the closure preflight, scheduler binding, budget enforcement,
workload supervision and publication itself. The frozen-closure check is deliberately
*not* repeated in the driver: it belongs to `execute()`, which runs it before any gate
import and again after postflight.

`fixture_substitutions.py` (new) is the one pinned, hashed file holding every binding the
fixture cannot supply outside a real allocation (`load_driver`, `read_binding`,
`reference_binding`, `verify_alignment`). It substitutes bindings only: it relaxes no
wrapper guard and touches none of the scheduler, cgroup, file-limit, scratch or closure
checks.

Three wrapper additions make substitution auditable rather than self-reported:

- `request['seams']` is now a mandatory list, fail-closed, copied into `receipt['seams']`.
- `request['substitutions']` is pinned by SHA256, loaded, applied to the gate, and its
  identity recorded in `receipt['substitutions']`. A production request sets it to null,
  so any substitution at all is visible in the receipt with its hash.
- `receipt['bindings']` records, via `binding_identity()`, the file each substitutable
  callable actually resolves to. This is the part with teeth: a substituted binding is
  **detected**, not declared. A patched `slurm` has no source file and reports
  `matches_expected: false` regardless of what the request claims.

**`slurm()` is no longer patched out.** It now runs for real inside `execute()` against a
declared stand-in `scontrol` that emits real scontrol field syntax: the field parser, the
ownership/node/TRES/memory checks, the time-remaining arithmetic and the affinity
assertion all execute. What remains substituted, and is declared in the seam list, is that
this is not a real allocation.

Declared seams in the fixture request: stand-in alignment workload (no approved runtime
tooling on this host); stand-in `scontrol`; synthetic cgroup v1 controller; gate data
bindings from the pinned substitution module. These are declared substitutions, **not**
compute containment and **not** real-tool scientific evidence.

## A3. Exact commands and real results

From `notes/auth_restored_20260921/ont/ont_full_v2/`.

**RED** — the new tests against the pre-addendum wrapper (round-1 wrapper reconstructed by
reversing the four edits, into a scratch `.red2/`):

```
cd .red2 && timeout 400 /usr/bin/python3 -I -B test_wrapper.py -v
```
→ exit 1, `Ran 29 tests`, `FAILED (failures=2, errors=5)`. All **7** `ExecuteDrivenFixture`
tests fail. Log `evidence/red2_python310.txt` (`284e2029…`).

The 8th new test, `test_driver_rejects_a_release_pin_mismatch_before_import`, exercises
`fixture_execute.load_wrapper`, which is new in this round but present in both the red and
green trees — so it cannot go red by construction. Stating that rather than implying 8/8.

Two of the seven initially passed against the old wrapper for an unrelated reason: they
asserted only `ValueError`, which the round-1 wrapper also raises when the real
`read_binding` meets a stand-in file. Both were tightened to assert the specific message
(`'seam'`, and `'pin mismatch'` naming `fixture_substitutions.py`) before the red log above
was taken.

**GREEN** — both supported interpreters:

```
timeout 400 /usr/bin/python3 -I -B test_wrapper.py -v                          # 3.10.12
timeout 400 /tmp/mvp-final-source-env/bin/python3.12 -I -B test_wrapper.py -v  # 3.12.2
```
→ `Ran 29 tests … OK`, exit 0 on both (35.1 s and 29.0 s). Logs
`evidence/green_python310.txt` (`690b3945…`), `evidence/green_python312.txt` (`38b86da2…`).

**F2 guard** unchanged on local 3.8.20: exit 1, `ValueError: Python >=3.9 required…`.
Log `evidence/python38_rejection.txt` (`d5361a18…`).

**Coverage, measured not asserted.** `test_every_wrapper_function_is_exercised_through_execute`
installs a `sys.settrace` hook filtered to the wrapper's own file and requires the set of
top-level functions entered to equal the set defined. Result: **15/15** reached through
`wrapper.execute()` — the 11 original plus the four §3 additions (`binding_identity`,
`enforce_output_limit`, `cgroup_observation`, `check_scratch`). Artifact
`evidence/execute_coverage.txt` (`104ef674…`). If a future change makes any wrapper
function unreachable from `execute()`, this test fails and names it.

Identities after this round:

| File | SHA256 |
| --- | --- |
| `wrapper.py` | `6a6608ca7dcfe919bd7616e0ddd0f3b5f4f086e9b2f2dda442547180f5ec4d28` |
| `test_wrapper.py` | `9c9b7de907dc93a39d7e5d30b343a6693279f8b866a48e46debd8706a669e0d2` |
| `fixture_execute.py` | `9ce0363323d0052406f0bfef000c67a0710ad5c7fec997d1cead56ce757611ca` |
| `fixture_substitutions.py` | `925e3055d7c49bf46f7cacd13c701c9dc1b2b1cf7db51c18df61ba3fdc9f6381` |

All originals remain unmodified at their original paths, including
`ont_fixture_v4/fixture.py`, which was read but never edited.

## A4. What this does and does not establish

Establishes: the production entry point, the scheduler binding and the four new runtime
boundaries are now structurally exercised end to end and locked in by a coverage test, so
the first real use of `execute()` will not be the first execution of `execute()`.

Does not establish: any real-tool result. The alignment workload, `scontrol`, the cgroup
controller and the gate's data bindings are declared stand-ins. No approved runtime tooling
exists on this host (§2), so a real approved-runtime run still requires an authorized job.
`slurm()` has still never met a real Slurm allocation — only a stand-in that reproduces its
output syntax. No structural availability, pooling, HiFi policy, score, decision or training
eligibility is claimed.

Unchanged and still blocking: the `.extern OUT_OF_MEMORY 0:125` gate; the admin
kernel/slurmstepd/cgroup records for job 6165965 (support draft still unsent); 6165965's
exact affinity list and reaping, permanently unobserved. Package build waits for the gate,
per the author's instruction.

Author for attribution: Won <wyim@unr.edu>.
