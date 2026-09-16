# Track C recovery report

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
