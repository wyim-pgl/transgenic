# A11 audit reconciliation and junction-counts binding — 2026-09-21

Owner: A11 audit / junction-counts binding. All times PDT. Author for
attribution: Won <wyim@unr.edu>.

## Scope actually executed

Four local, non-GPU-blocked items from
[`notes/TODO_FULL_RECONCILIATION_20260917.md`](../../TODO_FULL_RECONCILIATION_20260917.md)
§1 and `.atl/todo.md:12`. No Slurm job was submitted, no remote host was
contacted, no commit/merge/push was made, no frozen release, original DB or
existing receipt was modified or renamed, and no prior accepted number was
changed. The BioSample lookup that timed out was **not** retried and is **not**
recorded as absent.

Work was done in the two existing isolated worktrees, uncommitted:

| Worktree | Branch / base | Items |
|---|---|---|
| `/tmp/est-approved-contract-20260917` | `est-approved-contract-20260917` @ `fca911e` | A11 reconciliation, route-A fixture |
| `/tmp/mvp-seed123-integration-20260917` | `mvp-seed123-integration-20260917` @ `9603e20` | junction-counts binding |

The MVP worktree is a **shared** tree: other concurrent owners have uncommitted
edits in it (`mvp_score.py`, `mvp_produce_counts.py`, `src/transgenic/...`, plus
several `tasknotes/` and `tests/` files). Test counts below were obtained
against that shared working state, which is what the tree actually contains; it
is not a clean-`9603e20` measurement.

---

## 1. A11 full audit / exclusion / alias reconciliation

### Reconciliation of the real evidence against the normalized A11 API

Source of truth used: `notes/auth_resume_20260917/est/INVENTORY.md`
(SHA256 `fa18658de9be8dd0101cae8a39a7807bb6a12cef089be6feb7479054ac27b18e`,
computed here), `notes/approved_execution_20260917/A11_IMPLEMENTATION_BOUNDARY.md`,
and the 16 verified GenBank records in
`notes/auth_resume_20260917/est/public_metadata/nuccore_16.genbank.raw`
(SHA256 `85af56da812d225e1c46e7c969b1de67ce76a01b1da51c5409f3d56a9b4249d0`,
computed here — matches the value recorded in `PROGRESS.md`).

| A11 table | Real populated source located? | Reconciled state |
|---|---|---|
| `Member` identity fields (uid, species, assembly, dataset, arm, accession, lengths, intervals, screened hash, provenance, report row) | Yes, from the retained member receipt and UniVec report | **Available** — verified field-to-source mapping already recorded in INVENTORY.md |
| `Member.library` / `library_aliases` | **No** | **Blocking.** `CFGZ`, `CFBX`, `PRASADSSH` are observed header clone-library *text*; their biological-preparation identity and aliases are not proven |
| `Member.bioproject` | **No** | **Blocking.** Absent from the DBLINK sections of all 16 records; the 3 extra BioSample lookups hit `curl 28` SSL timeout with no response. Unverified, **not absent** |
| `Relation` (mate / parent / split / version) | **No populated table** | **Blocking** for mate/parent; split/version remain derivable from the authoritative report row |
| `Audit` (all-member, incl. secondary) | **No** | **Blocking.** The only primary alignment receipt ran `--secondary=no`; it cannot satisfy A46's `--secondary=yes -N 50 -p 0.5` audit |
| `Exclusion` (global + train_context intervals) | **No complete manifest** | **Blocking.** Dataset roles and the MVP panel/test intervals do not constitute that table |
| `inventory_complete` / `identity_complete` / `exclusions_complete` | — | All three remain **false** |

Conclusion, verified by test rather than asserted: with the real evidence as it
stands, every member is `unresolved_identity`, `diagnostic_ready` is `False`,
and `training_support_members` is empty. A11-complete, full-phase, full-C0,
production-raster and scientific acceptance all remain **FALSE**.

### Two documented rules that the code did not yet enforce (fixed, TDD)

Both rules already existed in writing; neither is a new policy.

1. **A download dataset must not become a biological library.**
   `A11_IMPLEMENTATION_BOUNDARY.md` and INVENTORY.md both state that
   `library` must never be replaced with `est/Sitalica`. Before this change,
   `library == dataset` (directly, or through an alias whose canonical target
   was the dataset) was accepted and would have united an entire download into
   one molecule component — exactly the "promote the MVP approximate
   clone_key to full A11" failure the author prohibited. Now it raises
   `library must not be a download dataset`.
2. **An explicit `unknown` BioProject marker is not completeness.**
   The API only required a non-empty string, so the literal marker `unknown`
   passed silently and could leave `diagnostic_ready=True`. It now adds
   `unresolved_identity:<uid>` to the blockers. It is a *blocker*, not an
   error and not a claim of absence — which is the correct handling for the
   timed-out BioSample lookup. Unknown **genotype** remains permitted, per the
   boundary document.

Changed: `revision/scripts/evidence/est_molecule_closure.py`
(SHA256 `7ec83d04b8e1f77258442a3e8734491bfff199ee93aa5d479c2198989332309f`),
tests appended to `revision/scripts/tests/test_est_molecule_closure.py`
(SHA256 `4b5f4fe0ce06d24856f7e4ebec36e58a487e7071dc185612cd08103f2b71f7e1`).
Four tests added, three of which were RED before the change; the fourth
(`test_observed_genbank_clone_library_text_stays_unresolved_not_absent`) is a
characterization pin over the real CFGZ/CFBX texts and passed both before and
after.

RED (`tasknotes/a11_reconciliation/red_tests.txt`): **3 failed, 48 passed in 0.49s**.

---

## 2. Route-A real trace fixture

A **certified** route-A observation cannot be built from anything held
locally, and this is a property of the recorded evidence, not of effort spent.
`est_phase_conflicts.Observation(route='A')` requires a reference-sequence
SHA256, a *normalized protein-sequence* SHA256 and a coding-direction codon
ordinal `c0`. None of the three exists in any retained receipt:

* the reference receipt for `Sitalica_312_v2.fa` carries MD5
  `d8f77d532d8c87f314f6175146fa6728` only;
* the protein receipt hashes the whole filtered FASTA
  (`fa0795b6…c1e25f`), which is not this protein's normalized sequence hash;
* raw GFF column 8 is miniprot's CDS phase, not `c0`; `c0` exists only after
  replaying the alignment against the exact reference and protein sequences.

In addition, the acceptance verdict and mapping-ambiguity membership of this
alignment ID were never inspected, and an `accepted` verdict alone does not
demonstrate A43-retained support.

What was therefore delivered is a **retained-trace fixture that is explicitly
not certified and refuses every substitution**:

* `revision/scripts/evidence/est_routea_fixture.py`
  (SHA256 `c26d73e9a1cf1b209f1751676a9a247fffea9bd7b47d4cbe34dbb8a166b7db80`)
  carries the exact recorded Sitalica `MP000002` trace — minus `scaffold_7`,
  protein `1000413_0:000002`, PAF length 172 / q `[0,160)`, coding-order CDS
  35312723–35312847 phase 0 Target 1–41 and 35312121–35312451 phase 1
  Target 42–160, miniprot `0.18-r281`, job `6153195_42` — transcribed from
  `notes/power_recovery_20260917/phase_input_inventory.md`
  (SHA256 `1468e5ec4b7300b6d484922255a0623bcba54e7570ad3a415a7835cc2ac615bd`)
  and INVENTORY.md. No raw alignment, reference or protein byte was read, and
  no field was invented; unavailable fields are `None`, never filled.
* `BLOCKING_GAPS` names all five gaps; `is_certified()` is `False`;
  `to_observation()` refuses and names them, and separately refuses the
  reference MD5, the whole-FASTA protein hash and raw GFF phase as `c0`.
* `revision/scripts/tests/test_est_routea_fixture.py`
  (SHA256 `20af9b24061eac816dfe2b78850e752fcb97346f96617f6b73c715688c2b913c`)
  pins all of that, and shows that once the three inputs are supplied the
  record validates as a route-A crosscheck with `hsp_sha256`/`member_id`/
  `placement_id`/`molecule_id` all `None`.

RED (`tasknotes/routea_fixture/red_tests.txt`): **8 failed in 0.09s**.

Closing this needs recovered artifacts (a reference-sequence SHA256 receipt,
the per-protein normalized hash, the acceptance/ambiguity membership for this
ID and a replay), **not** a new miniprot run.

### Additional-producer reference SHA binding — still blocked

`est_real_sample.py:31` compiles the registry pin
`PRODUCER_PATH_PROOF_SHA256 = a2843aa36a006730c1fd3fe02a34b76afd772f1e01328162f7c0f9af88a2b6a9`
(full value read from source here; it hashes caller-supplied registry JSON
bytes, so it cannot be verified against any file in this repository). The
resolver hard-requires `len(registry['producers']) == 2`, and every
`search_preimage` must carry `reference_sha256`. The Ptrichocarpa receipt
supplies reference MD5 `20c6f4bb6f8b52887a2a08addedac6ea` and no SHA256.
Binding the two additional Ptrichocarpa producer receipts therefore needs an
existing reference-SHA256 receipt **plus** a new compiled, independently
reviewed registry pin. Substituting the MD5 or Sitalica's hash is forbidden.
Nothing here was changed.

---

## 3. Independent junction-counts binding — implemented

The gap was precise and real: `mvp_align_panel_longreads.py:198` produces a
per-run `junctions.tsv` via `mvp_junction_table.py`, and
`mvp_score.junction_counts(predicted, supported, locus)` consumes a `supported`
junction set, but **nothing joined them**. The structural counts
(`predicted`, `precision_tp`, `supported`, `recall_tp`) had no producer bound
to actual independent evidence.

New: `revision/scripts/mvp/mvp_junction_binding.py`
(SHA256 `c340291b326c258625fd4855fe42cb4cbc6bfcf1187bd815e30cf1e1ff613356`).

* `load_run(table, report)` accepts a junction table only when the run's own
  `alignment.json` claims it by name **and MD5**, the receipt's
  `panel_sha256` equals the frozen `mvp_windows.PANEL_SHA256`, the run ID is a
  known long-read run, and the receipt's `source` agrees with the run
  inventory. Forged `supported` columns are rejected by the frozen
  `mvp_junction_table.read_table`.
* `bind(runs)` refuses duplicate run identity ("file labels do not establish
  distinct runs") and refuses **any multi-run pooling** by calling
  `mvp_score.require_coverage_policy`, which raises. Support thresholds are
  read counts, so pooling is a coverage decision, not arithmetic, and no
  HiFi/ONT-only coverage policy is author-approved.
* `supported_set(rows)` re-derives support from read counts through the frozen
  ONT 3 / PacBio(HiFi) 2 thresholds, never from a file's own boolean.
* `locus_counts(...)` returns `None` per locus when the independent evidence is
  absent — missing evidence is suppressed, never filled with zero.
* `receipt(...)` records `counts_are = reads (UMI status unknown; never
  molecules)`, `coverage_policy = PENDING`, `pooled_across_runs`, and
  `run_provenance_status = UNVERIFIED`.

Tests: `tests/test_mvp_junction_binding.py`
(SHA256 `98a3caa5944bb5858cd90f7a9a29c5525bba0c29a3fb81f6cc36d81dd589ab0c`),
10 cases. RED (`tasknotes/junction_binding/red_tests.txt`): **10 failed in 0.18s**.

This is a preparatory single-run binding. It decodes nothing, runs no
GFFCompare, verifies no reference genome, and makes no evaluation, coverage or
training decision.

---

## 4. Spec contract table and native-crash evidence

### 4a. SPEC ↔ later author decision ↔ source (`9603e20`) difference table

SPEC = `revision/protocols/B6_MVP_PILOT_v1.md` (331 lines). Its draft preamble
and old command sketches are **not** re-execution instructions and do not
invalidate later explicit approvals.

| Contract point | SPEC clause | Later author decision | Source at `9603e20` |
|---|---|---|---|
| Init checkpoint | seed-123 `best` symlink → `epoch_04` (L6, L208–209, L322 `--init-from .../seed123/best`) | pin one **concrete** checkpoint; observed best is `epoch_05`, eval `0.6394507885` | `stage_checkpoint.py:94–96` **refuses** any path containing `best`/`latest`/`latest_state`. Source is stricter than SPEC; SPEC's literal command line is stale |
| Optimizer steps | `N = 250` (L222), "fix N if it/s differs from 1.4" (L294) | 250 retained | Not pinned in a source constant; supplied on the launch command line. **Difference: no source-side guard that N == 250** |
| Seeds | 3 seeds 1/2/3 (L6, L219) | superseded by seed-123-only (`SEED123_ONLY_DECISION_20260917.md`) | `mvp_profiles.py` `DEVELOPMENT_PROFILE='development-single'` seeds `(123,)`; `mvp_score.score` requires pair key `mvp_seed123`, run role `mvp_finetune`, loci `seeds['123']`. Source already carries the later decision |
| Primary metric | paired tile bootstrap on Δ loss, 2,000 replicates, seed 20260914 (L243, L256–257) | pure weighted-CE primary, composite reported alongside, equal-tile off−on mean, δ 0.005 nats | `mvp_bootstrap.build_tiles`/`tile_bootstrap` (2,000 replicates); `mvp_score.score` computes composite separately. Consistent |
| Structural P/R | junction P/R as the contradiction check (L246) | — | `mvp_score.junction_counts` existed with **no producer**; now bound (§3). Partially missing counts suppress pooled P/R rather than becoming zero |
| Coverage policy | assumes aligned A-ONT1 / A-HiFi (L258–260) | HiFi identity vs ONT-only **undecided** | `mvp_score.require_coverage_policy` raises unconditionally: "coverage policy pending author approval (HiFi versus ONT-only)". Source correctly fails closed |
| Decision order | STOP evaluated first; unassessable withholding or missing structural scoring downgrades a prospective STOP to INCONCLUSIVE, never GO (L257–267, §5.4 L285) | unchanged | `mvp_bootstrap.decide` implements it; covered by the `test_stop_first` parametrization (`structural_available=False` → INCONCLUSIVE; `withholding_state='unassessable'` → INCONCLUSIVE) |

No spec clause was edited. The only open contract difference needing an author
decision is the coverage policy; the missing N==250 source-side guard is an
engineering gap, not a policy question.

### 4b. Native SIGSEGV evidence list

Status stays **UNRESOLVED** and preserved. No blind whole-suite rerun was
performed, and no non-reproduction is reported as a fix.

Primary artifact:
`notes/wait_work_20260916/repro/captured/mvp-final-source-full-cpu.log`,
7,273 bytes, SHA256 `142745e3dc7234fca0821af9015209e4255e77ecea55b6ff9b5f70f95d5a119f`
— **recomputed here and equal to the value in `repro/ledger.json`**.

| Evidence item | State | Detail |
|---|---|---|
| Faulting frame | **Have** | `mvp_decode_panel.py:141` in `decode_panel`; at `b660` line 141 is `ids,mask = ids.to(device),mask.to(device)` and line 142 is the `tracks[...]` conversion. An earlier review paraphrased 142 as 141 |
| Fixture | **Have** | `tests/test_mvp_decode.py:190 test_runner_exact_350_no_overwrite_and_partial_failure` |
| Exit status | **Have** | 139 (SIGSEGV), `Fatal Python error: Segmentation fault` |
| Both thread stacks | **Have** | crashing thread plus a `tqdm._monitor` waiter thread |
| Loaded native extensions | **Have** | 144 listed, incl. `torch._C*`, `pysam.libc*`, `numpy`, `scipy`, `pandas` |
| Interpreter / native env | **Partly** | log paths show `~/miniconda3/envs/pdb/lib/python3.12`; ledger records Python 3.12.2, torch 2.6.0, transformers 4.46.3, accelerate 0.34.2, duckdb 1.5.5 |
| Test-dependency tree | **NEW — recoverable** | `/tmp/b6-trackb-testdeps` still exists (14 entries): pytest 9.1.1, pluggy 1.6.0, packaging 26.3, pygments 2.21.0, iniconfig 2.3.0. This was on the crashing process's import path and can be hashed without rerunning anything |
| `argv` | **Missing, unrecoverable** | ledger records `exact_original_full_command_not_preserved: true` |
| `cwd` | **Missing, unrecoverable** | the run tree `/tmp/pi-agent-cf3f429c-1b79-48f-e598f927` **no longer exists** (checked 2026-09-21) |
| `PYTHONPATH` / env | **Missing** | not captured; only inferable from the two `/tmp` path prefixes in the traceback |
| Ordered predecessor node IDs | **Missing, but position is derivable** | the run used `-q`, so only progress characters exist. Counted exactly: lines 1–3 carry 72 progress characters each and line 4 carries 53 (ending at 22 %), then 4 more precede the fault → **273 items completed, fault during collected-item ordinal 274** (1-based). Total collected ≈ 1,200–1,225, a derived estimate, not a receipt. Failure positions before it are readable from the `F` marks; node IDs are not |
| Core dump | **Not available** | `/proc/sys/kernel/core_pattern` is `\|/wsl-capture-crash %t %E %p %s` and the shell's `ulimit -c` is `0`. That is *this* shell's limit, not proof of the historical process's limit, but no core file for that process exists anywhere that was checked |

Contradicting the "just rerun it" reading: the **same** test completes in
narrower scopes — `mvp-final-source-focused.log` records a 145.20 s passing
call and `mvp-final-source-mvp-cpu-retry.log` a 133.26 s passing call of
`test_runner_exact_350_no_overwrite_and_partial_failure`. So the fault depends
on accumulated in-process state across the whole-repo session, and a passing
narrow rerun does **not** establish a fix.

**Bounded next step that is not a blind rerun:** run `pytest --collect-only -q`
on the same pinned tree to recover the ordered node IDs up to ordinal 274. That
imports the test modules but executes no test. It needs the `b660` tree
re-materialized from git (the original `/tmp` worktree is gone) and is not
performed here.

---

## Test commands and real results

All runs used `/tmp/mvp-final-source-env/bin/python` (Python 3.12.2) with
`PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`. Python 3.8 was not used;
the repo's 3.8 interpreter cannot even import the package.

**EST worktree** (`/tmp/est-approved-contract-20260917`), the recorded
controller 278-test set plus the new route-A suite:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/mvp-final-source-env/bin/python3.12 -m pytest -p no:cacheprovider -q \
  tests/test_est_hsp_contract.py tests/test_est_phase_conflicts.py \
  revision/scripts/tests/test_est_molecule_closure.py tests/test_est_approved_contract_integration.py \
  revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_est_hsp_expanded.py \
  revision/scripts/tests/test_est_blastx_fixture.py revision/scripts/tests/test_evidence_support.py \
  revision/scripts/tests/test_univec_trim.py revision/scripts/tests/test_est_routea_fixture.py
```

**290 passed in 1.52s** — the accepted 278 unchanged, plus 4 A11 and 8 route-A.
The same set without the route-A file gave **282 passed in 1.83s**.
`git diff --check` passed. Logs: `tasknotes/a11_reconciliation/{red,green}_tests.txt`,
`tasknotes/routea_fixture/{red,green}_tests.txt`.

**MVP worktree** (`/tmp/mvp-seed123-integration-20260917`):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /tmp/mvp-final-source-env/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_mvp_junction_binding.py tests/test_mvp_scoring.py tests/test_mvp_scoring_driver.py \
  tests/test_mvp_score_producer.py tests/test_mvp_score_producer_integration.py \
  tests/test_mvp_score_producer_review.py
```

**95 passed in 65.85s** (baseline `tests/test_mvp_scoring.py` alone was
29 passed before the change; the 10 new cases are included).
`git diff --check` passed. Logs: `tasknotes/junction_binding/{red,green}_tests.txt`.

Not run: the whole repository suite in either worktree. The whole-repo run is
the one that segfaults (§4b) and a blind rerun is prohibited.

## Preserved and unchanged

- 385 rule-accepted HSPs, 414 length-pass / 48 truth failures, geometry
  380 success / 34 conflicts: untouched, not recomputed, not reconciled away.
- The legacy unresolved `-1 self-overlap` issue: preserved, not silently fixed.
- extern OOM `0:125`, full-ONT gate, GPU/B5 jobs, `eval9603`, `b660`, the
  original DB and every frozen release and receipt: untouched. Zero remote
  actions this session.

## Open, and what needs an author decision

Engineering-blocked (no decision needed, evidence needed):
1. Authoritative library/preparation identity + alias table, and a populated
   mate/parent relation table.
2. All-member A46 audit (`--secondary=yes -N 50 -p 0.5`) and the complete
   global + train-context exclusion manifest.
3. A reference-sequence SHA256 receipt for `Ptrichocarpa_533_v4.0.fa`, plus a
   new compiled and independently reviewed producer-registry pin to admit a
   third producer record.
4. Route-A closure: reference SHA256, normalized per-protein SHA256,
   acceptance/ambiguity membership for `MP000002`, and a replay for `c0`.
5. Ordered predecessor node IDs for the SIGSEGV, via `--collect-only` on a
   re-materialized `b660` tree.
6. No source-side guard that the MVP optimizer-step count is 250.

Author decisions outstanding:
- **Coverage policy** (HiFi acquisition versus ONT-only). Until it exists,
  `mvp_score.require_coverage_policy` raises and multi-run junction pooling
  stays refused. Single-run binding does not pre-empt that decision.
- Whether the BioProject/BioSample linkage follow-up should be retried at all,
  given the SSL timeout. It was not retried here and remains **unverified, not
  absent**.

Nothing in this record is a scientific, A11-complete, full-phase, full-C0 or
production-raster acceptance. Those all remain FALSE.
