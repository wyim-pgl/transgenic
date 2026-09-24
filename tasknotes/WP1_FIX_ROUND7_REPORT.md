# WP1 fix round 7 — Codex review 4 (1 HIGH, 1 MEDIUM, 1 LOW), applied by the team lead

The round-6 writer stopped at a session limit before the A49.1 library-key rule arrived; the lead applied that rule and
then Codex review 4's three findings. Worktree `/tmp/b6-wp1-20260922`, uncommitted.

| # | Finding | Fix | Test |
|---|---|---|---|
| A49.1 | long-read `--library` must be an ENA experiment accession or `<BioSample>:<library_name>` | `c0_schema.is_longread_library` (shared); producer `longread_library` refuses anything else | `test_b6_c0_metadata.py::test_longread_library_is_explicit_and_never_the_run` (free text, `SRX_LIB1` refused; `SAMN49012086:HOTXU` accepted) |
| 1 HIGH | builder ingestion accepted legacy run-as-library metadata (checksum-valid) → split molecule, p1 stays train_track | `b6_build_c0.process_read`: long-read library must pass `S.is_longread_library` before molecule assignment | `test_b6_c0_build.py::test_legacy_run_as_library_metadata_is_refused_at_ingestion` |
| 2 MEDIUM | stale HSP library / est_accession survived rebuild and merge | builder `verify_hsp_placements`: every HSP row needs an EST member of its molecule with the same accession AND library in `--est-c0`; merge `hsp_gate` re-checks (`stale_library`) | `test_b6_c0_build.py::test_hsp_with_a_stale_library_is_refused`, `test_b6_c0_merge.py::test_merge_refuses_an_hsp_whose_library_no_est_member_has` |
| 3 LOW | per-source abort lost exhaustiveness counters | `decide_all`: `qc.update(ex.report())` before `ex.check()` | `test_capped_molecule_below_the_per_source_floor_aborts_the_build` asserts `(17, 16, 50)` counters in the ABORT record |

Fixture library values changed to valid keys (`SRX1000101/102/103/201/202`, `SRX1000001`). The HSP library binding is
by (molecule_id, est_accession, library) — the same binding the merge gate already used — rather than by the sidecar
qname, because a molecule's sidecar row names one member while the HSP names the member it was projected through.

## Test results

See the team lead's run (`wp1_round7_tests.log` in the session scratchpad); filled in below when it finishes.

Round 7 first run: 180 passed / 5 failed. Causes fixed: `qc` is a `Counter`, so recording `report()` and then
`check()` added every exhaustiveness counter twice (now `check()` is called without adding its return); the scale
fixture used the run as library (now `SRX9000001`). Rerun: **185 passed in 1268.63s** (exit 0, incl. the MVP byte pin).

## Round 8 (Codex review 5 HIGH, applied by the team lead)

Rejected clone-less EST members had no A11 alignment signature, so a MAPQ-10 twin with a held-out audit secondary
did not exclude its accepted twin. `b6_build_c0.rejected_est_signature` derives the signature from the frozen primary
(no flags 4/0x900) with the same `59.transcript_strand` call as the accepted path; unmapped members get none (QC
`rejected_est_members_with_signature` / `_without_signature`). Test
`test_b6_c0_build.py::test_rejected_clone_less_est_twin_closes_its_signature_molecule`.
Full suite: **186 passed in 1294.09s** (exit 0, incl. the MVP byte pin).

## Codex review 6: **APPROVE** — no CRITICAL, HIGH, MEDIUM or LOW findings.
