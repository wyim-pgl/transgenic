# Track C recovery report

## Round-one C1 review fixes (supersedes the earlier C1 status below)

- FIX_BASE: `2c91ecb295d88a1bc27199b0f84fe8af667abb08`; source/tests fix commit: **`53c8f7d1a0f6f5d69a2b2ff47d82cec361d06afe`**. This report update is a separate documentation commit.
- Read all five findings verbatim in main `notes/recovery_20260916_1251/C.independent_review.md`; inspected historical wrappers and the log-layout record in `notes/RESTART_HANDOFF_20260915.md:38`. Only C1 and its CPU tests changed. `git diff --exit-code 2c91ecb` on the three C2/C3 Python files confirmed no changes; production/semantic gates remain intact.
- **Container markers:** every scanned root checks its own markers independently of known-child exemptions (pilot, remaining8 batch, species, source unit, and audit tree). Absent container markers are allowed. Present FAILED, contradictory markers, malformed DONE, marker directories and symlinks fail. No new normal-container DONE requirement.
- **Historical identity:** provenance must include `diamond_version`, `diamond_sha256`, `launcher_sha256`, and `taxonomy_helper_sha256`. Launcher SHA `5fa32fe8…d4e291` and helper SHA `32ea1a2e…d90302` were independently reproduced using `git show 950cf4c:<path> | sha256sum` and match both historical launch scripts. Engine SHA `0fb259d7…fed07d` is the approved pin in those scripts, not an arbitrary executable's self-declared identity. The recorded release/help lines must satisfy the exact historical 950cf4c acceptance contract (2.1.24 release, optionally .178, and .178 build). The auditor never executes the binary. Current audit/runtime hashes are recorded under separate audit-prefixed keys, not mistaken for historical producer identity.
- **Timing/scripts/logs:** require nonempty regular `swissprot.time.txt` AND `orthodb.time.txt`, exactly as both wrappers produce. A `.time.txt` directory cannot abort inventory or stand in for evidence. Wrapper category is satisfied only by matching the archived wrapper bytes, irrespective of copied script filename: pilot SHA `8e8f501ebaa0cbd15ee8d61aef127f5cb7ee3fdc3a070d3c7d18923440de2f4a`; remaining8 SHA `4cdad18b443c13ad564be640859419e86def7fb255fe82cabe797888913fdd7c`. Require every remaining8 `logs/6164049_<0-7>.{out,err}`; empty stderr is valid inventory. An arbitrary script/log no longer substitutes for missing categories. Timing content remains an inventory, not a benchmark or format-specific timing interpretation.
- **Closed unit inventory:** chose the smaller contract: hash all immediate regular files, reject ALL nested unit directories as UNRESOLVED even if they contain DONE. No claim to have recursively hashed rejected nested payloads. Existing tmp and symlink failures remain.
- **Malformed evidence:** explicit top-level/nested provenance and consumed batch-input schema/type checks. Unit failures retain hashes and checks already gathered and continue inventory of the other 17 units. The unit boundary catches only declared evidence/JSON/encoding/I/O errors, not generic TypeError/KeyError/IndexError/AttributeError; a programmer-TypeError regression verifies propagation. Historical launcher ValueError validation is converted at its input-validation boundaries.

### Remaining receipt gaps — intentional FAIL, not launch readiness

The pilot wrapper contains no `--output`/`--error` directive, and no local submission receipt establishing the pilot log paths was found. Slurm defaults or `slurm-6163851.out` are **not inferred as authority**. Consequently `batch_artifacts:pilot:logs` is currently an explicit **UNRESOLVED FAIL** even when all 18 units pass. All 18 unit records and the rollup are still written before failure. A controller-reviewed historical submission receipt and a scoped follow-up change are required to resolve this gate; there is no CLI waiver. Neither real payload failure nor blastx resubmission follows from this missing receipt.

An independent original engine version/banner receipt and the real binary bytes were not available locally. The implemented binding uses the archived wrapper's approved engine SHA and the historical launcher's version-validation contract; it does not claim an independently observed literal copyright/banner string. Unknown/mismatched historical pins explicitly fail. Original absolute files must remain resolvable for the real audit. No remote audit was performed. Wrapper-only A37 receipt/count bindings beyond the direct launcher's screened-query provenance remain outside replay scope, as the review noted. Existing tmp/execution.lock/preflight reconciliation and `.extern` OOM remain unresolved.

### Round-one test evidence

Baseline before edits: **65 passed**. New independent repro suite first ran against unfixed C1: **29 failed, 9 passed**, exposing all five findings. An additional directory-timing repro failed with IsADirectoryError before its fix. Synthetic unit fixtures now carry all required producer fields and named per-source timings; test-only monkeypatches substitute synthetic engine/wrapper trust anchors, never a public production override. Separate tests restore the real engine pin and prove an arbitrary self-hashed fixture binary fails. Integrated cases cover nine container-marker combinations, malformed provenance continuation with exactly one failed unit, malformed remaining8 manifest continuation, missing producer identity, and complete 18-unit FAIL rollups with no outer DONE.

Final exact command from this worktree:

```sh
PYTHONPATH=.testdeps /home/wyim/micromamba/envs/kimicli/bin/python3.12 -m pytest -q \
  revision/scripts/tests/test_est_blastx.py \
  revision/scripts/tests/test_est_blastx_fixture.py \
  revision/scripts/tests/test_track_c_recovery.py \
  revision/scripts/tests/test_track_c_audit_review.py
bash -n revision/scripts/evidence/est_blastx_audit.sbatch revision/scripts/evidence/est_phase_accept_fixture.sbatch
/home/wyim/micromamba/envs/kimicli/bin/python3.12 -m py_compile revision/scripts/evidence/est_blastx_audit.py
```

**116 passed** (existing 65 + 51 round-one cases); bash syntax, py_compile and staged diff whitespace checks passed. This is CPU fixture verification, not actual historical payload acceptance. Scoped independent re-review remains the controller's responsibility. No SSH, jobs, main edits, subagents, merge or push; dependency/graph/ignore files excluded from commits.

---

## Scope and commits

- Worktree only: `/tmp/b6-p2-phase-1535433`, branch `b6-p2-phase-20260916a`.
- Base: `950cf4cd9ef55db276ffa1b713d8925ed6dd6dd2`.
- Source/tests commit: **`91e10efa61bc6ddfd9cbb850ac52727dd33e7531`**. This report is a separate documentation commit; obtain its identity with `git log -1 -- notes/TRACK_C_RECOVERY_REPORT.md`.
- Inspected and revised all five interrupted files rather than replacing them blindly. Added `est_phase_project.py` (established primitives and explicit blocker only) and `test_track_c_recovery.py`.
- No SSH, remote writes, jobs, subagents, main edits, merge, push, reset, clean, new worktree, or blastx resubmission. No dependency installation. Existing `.gitignore`, `.ignore`, `.testdeps/` and graft cache remain outside commits.
- Independent review remains the controller's responsibility. Advisor consultation was planning guidance, not independent review or approval.

## Authority

Read Track C in `notes/b6_three_track_discovery_synthesis_20260916.md`, recovery records, and main `.atl/quarantine.md` §1m (specifically the read-relative `ts:A`/FLAG16 composition row). The supplied workplan and main workplan are byte-identical: SHA256 `5ff2eb9d8775c1ecf2434f10786416d6a4a63741e0b862dfc36ca4143dfd3e0b`.

Built/queried graft in this worktree before source inspection, then refreshed it after changes. Relevant base spans: `est_blastx.py:117-143` (pre-search exclusion), `:168-196` (screened-query provenance), `:293-315` (recorded inputs); `est_blastx_fixture.py:51-107` (replay); `67_orientation_audit.py:216-268` (strand composition). Graft's cumulative estimated savings across calls: approximately 1,038,369 tokens (tool estimate, not a measured cost).

## C1 implementation

`revision/scripts/evidence/est_blastx_audit.py` now performs fail-closed local inventory/reconciliation:

- 18 expected species/source units, required unit artifacts, complete unit-file SHA256 inventory, raw13 schema/count/hash and DB/filtered-FASTA hash reconciliation.
- Revalidates original screened query/source/lineage/roles with the existing launcher; reconstructs the **pre-search** taxonomic filter in scratch and compares provenance. Raw subjects must occur in that filtered FASTA. No top-25 post-search exclusion replacement.
- Binds frozen command arguments to query/filtered FASTA/DB/raw output paths and the recorded executable hash.
- Pilot `query.sha256` and remaining8 batch `inputs.json` + `inputs.sha256` layouts; resolvable query/hash and raw query-ID membership required. Missing input or malformed provenance returns FAIL.
- Species exit code, nonempty timing-file inventory and finite nonnegative provenance elapsed time. Batch scripts/logs hashed. Timing is inventoried, not interpreted as a performance benchmark.
- Recursive preflight/unknown-directory and unit tmp inventory. DONE/FAILED/neither, contradictory markers and symlinks are explicit; **all unreconciled tmp directories fail, including empty tmp**. Nothing is automatically deleted or waived.
- Captured accounting must be pipe-delimited with `JobID|State|ExitCode` header, including allocation/batch/extern for `6163851` and `6164049_0-7`. Missing/duplicate/changed rows fail, but inventory continues for all available units. `.extern OUT_OF_MEMORY 0:125` stays explicitly unresolved; `clean_accounting=false` even when payload inventory passes.
- Output and reconstruction scratch must be outside the source tree. Exclusive output and failure markers prevent reuse/overwrite.

The `.sbatch` is an unsubmitted template. C1 has **not** scanned the real ~16.4-GB raw13 payload. Synthetic success is not nine-species audit acceptance. Original absolute input and executable paths must remain resolvable; no relocation-map guessing. Historical preflight/tmp failures can intentionally make the first real rollup FAIL and require separately reviewed reconciliation, not blastx reruns.

## C2 recovered safely, semantic gate unresolved

The interrupted code could emit `accepted_hsp_frame_spans.tsv` using an unverified col9 >=30 interpretation. Removed that writer. Both `run()` and `accept_stream()` now refuse **before any I/O/output**; there is no bypass switch.

Preserved explicitly named `candidate_decompose_hsp()` only as a fixture hypothesis. Its slash/backslash cursor updates are **not certified semantics**. The RICH17 extension cross-checks input codons, gapped sequence/CIGAR/translated fields and candidate decompositions; records candidate length readings and raw17/provenance/code hashes; ambiguous readings are INCONCLUSIVE, unique readings still REQUIRE REVIEW. Its outer DONE is written only after diagnostic reporting, never interpreted as acceptance/A46 approval.

Real RICH17 DIAMOND execution was not performed. Selecting the operational >=30-aa definition (alignment columns vs ungapped residues vs constant-frame sub-span), verifying both shift directions/strands and boundary/discriminating cases, then reviewing/freezing the semantic contract are still required. CPU hand-built fixtures test plumbing only. Production C2 remains blocked by **both semantic review and A46**.

## C3 deliberately limited

Implemented and CPU-tested only:

- FLAG16 × read-relative `ts:A` composition (four cases), directly grounded in quarantine §1m;
- same-base/strand/protein-pair comparison of already-labelled route-A/B phases, returning neutral support for both on disagreement. `None` means zero support, not phase label 0;
- unconditional production projection refusal.

No genomic projector, multi-exon phase fixture, six-channel raster, BAM ingestion or molecule aggregation is claimed complete. Workplan §2.6 requires frameshift-aware **intron-length mod3**, six phase{0,1,2} × strand{+,-} channels, inconsistent-HSP rejection and route-A disagreement→0. It does not fully specify phase-0 anchoring, conflict scope/overlap handling, or the unresolved BTOP/HSP interpretation. Inventing these would violate this task. These requirements are preserved in the module and remain blocking decisions before C3 implementation/production. No raw HSP count is labelled accepted phase evidence.

## Actual verification

Baseline targeted suites: **22 passed** before implementation changes, using Python 3.12 and existing worktree dependencies. Earlier attempts were environment failures, not test failures: default Python 3.8 cannot import the installed pytest/pluggy; system Python 3.10 lacks `exceptiongroup`. Neither environment was modified.

TDD regression runs observed failures for unguarded acceptance, contradictory markers, hidden nested tmp/symlinks, missing timing, source-tree output, ambiguous diagnostic verdicts, command/input/provenance tampering, raw subjects outside the filtered source, absent C3 primitives, incomplete accounting inventory and source-tree scratch. Each was followed by a passing run. Existing partial behavior also received characterization coverage.

Final command (from the worktree):

```sh
PYTHONPATH=.testdeps /home/wyim/micromamba/envs/kimicli/bin/python3.12 -m pytest -q \
  revision/scripts/tests/test_est_blastx.py \
  revision/scripts/tests/test_est_blastx_fixture.py \
  revision/scripts/tests/test_track_c_recovery.py
```

**65 passed** (22 baseline + 43 recovery cases). Includes a complete synthetic 18-unit audit, both provenance layouts, failed accounting/preflight rollups preserving inventory, both-strand hand-built RICH17 diagnostic cases, 29/30/31 raw-boundary nonacceptance, and production gates creating no output.

`bash -n` passed for both recovered sbatch templates; `py_compile` passed for all four Track C Python modules. Staged `git diff --check` initially caught one trailing whitespace line; corrected before the successful source commit. No full-repository, real-DIAMOND, Slurm or production-data test claim.

## Round two — scoped filtered-FASTA iteration regression

FIX_BASE: `b7547e2edc175f7eac9713c8465b5b20e396a86f`. Read `notes/recovery_20260916_1251/C.fix1_review.md` completely. Original five findings remain addressed. The sole production change wraps **consumption** of `launcher.records()` in the filtered-subject set comprehension, converting its expected validation `ValueError` into `EvidenceError`. The unit-wide exception boundary is unchanged; no blanket ValueError/TypeError catch, policy/provenance change or C2/C3 change was introduced.

Three integrated 18-unit regressions cover malformed FASTA, empty FASTA, and one valid record followed by an empty record. The last case explicitly proves the generator yields a valid prefix before raising during later iteration. Each verifies 18 manifests, 17 passing units under the existing synthetic trust anchors, the failed unit's gathered hashes and prior checks, final rollup and FAILED marker, and no outer DONE. All three tests failed against FIX_BASE with escaping FASTA ValueError before the source fix. The existing programmer-TypeError propagation regression remains passing.

Baseline: **116 passed**. Final exact command (same worktree):

```sh
PYTHONPATH=.testdeps /home/wyim/micromamba/envs/kimicli/bin/python3.12 -m pytest -q revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py
bash -n revision/scripts/evidence/est_blastx_audit.sbatch revision/scripts/evidence/est_phase_accept_fixture.sbatch
```

Result: **119 passed**; both shell syntax checks and `git diff --check` passed. Source/tests/report are committed together; exact commit is available with `git log -1 -- notes/TRACK_C_RECOVERY_REPORT.md`. Existing missing pilot log receipt, real-data/identity limitations, `.extern` anomaly and C2/C3 gates are unchanged. No subagents, remote actions, main edits, merge or push. Controller scoped re-review remains pending.
