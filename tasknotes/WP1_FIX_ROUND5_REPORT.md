# WP1 fix round 5 — Codex re-review (3 HIGH, 3 MEDIUM) verified against A46 / A48 / A16 and fixed

Worktree: `/tmp/b6-wp1-20260922` (branch `b6-wp1-20260922`, uncommitted; the B6 files are untracked, so line anchors
are absolute line numbers of the current files).  Spec quotes: `PROTOCOL_B1_frozen_v1.md` (protocol line numbers),
`notes/a46_approval/A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md`, `EVIDENCE_TRANSCRIPT_MODEL_DESIGN_v1.md` (design).

| # | Finding | Verdict | Fix | Test (fails before / passes after) |
|---|---------|---------|-----|------------------------------------|
| 1 | Unknown-strand UMI siblings get different molecule keys | **confirmed** | `rejected_member_unit` keys on `None` like the accepted path | `test_b6_c0_build.py::test_unknown_strand_umi_siblings_share_one_molecule_and_close_together` |
| 2 | Excluded protein hits delete eligible HSPs before exclusion is applied | **confirmed** | MULTI_HSP_RULE moved from `b6_hsp_project._emit` to `b6_build_c0.load_hsp` (after eligibility); step-7 + merge re-tests; agreement reads the resolved set from the protein C0 | `test_b6_c0_build.py::test_hsp_phase_conflicts_are_resolved_after_protein_query_exclusion`, `::test_projection_that_resolved_conflicts_itself_is_refused`, `test_b6_c0_merge.py::test_merge_rechecks_phase_consistency_among_retained_hsps`, `test_b6_hsp_project.py::test_multi_hsp_conflict_within_one_source_is_deferred_to_the_builder` |
| 3 | Protein-C0 gate in `b6_routea_agreement` does not bind to its inputs | **confirmed** | `bind_inputs` (sha256 of EST C0, route-B TSV, sidecar, miniprot GFF vs `c0_provenance.inputs`), `verify_accept_table` (content binding of accept table / ambiguous list); fixture builds one protein C0 per case | `test_b6_routea_agreement.py::test_inputs_are_bound_to_the_protein_c0` (+ every case now built from its own protein C0) |
| 4 | Build-flags identity comparison misses omitted keys | **confirmed** | `identity_diff` over the union of keys | `test_b6_c0_merge.py::test_build_identity_is_compared_over_the_union_of_keys` (both directions) |
| 5 | Exhaustiveness enforced per source instead of per species | **partially confirmed** (species-level is right; the threshold is 99.9 %, not 99.9999 %) | builder records numerator / denominator per source (`Exhaustiveness.report`); merge enforces `leakage.species_exhaustiveness` | `test_b6_c0_build.py::test_capped_molecule_is_excluded_and_exhaustiveness_is_recorded_per_source_not_enforced`, `test_b6_c0_merge.py::test_species_exhaustiveness_is_enforced_in_the_merge_over_all_sources`, `::test_exhaustiveness_counters_and_one_audit_n_are_required` |
| 6 | Accepted reads never receive A16 completeness codes | **confirmed** (C0 must carry them; no producer exists; inputs missing) | explicit NULL policy in `c0_provenance` + counters + step-7 gate + merge gate; **author decision** written below | `test_b6_c0_build.py::test_a16_completeness_codes_null_policy_is_recorded_and_gated`, `test_b6_c0_merge.py::test_completeness_code_policy_is_required_and_rechecked` |

"Fails before" is established from the pre-fix code path (no committed baseline of these files exists to re-execute):
before, the rejected UMI sibling hashed `'?'` and the accepted one `None`; `_emit` dropped both P and Q; the
agreement script only checked that every accepted GFF protein had a molecule; `provenance_gate` iterated one side's
keys; the EST build aborted at 16/17 (`ABORT_EXHAUSTIVENESS`); the three code columns were `NULL` with no record.

## 1. Unknown-strand UMI siblings (CONFIRMED)

Spec.  A18.2 line 252: ONT unit = "A16: PCR-equivalence unit ...; direct RNA = one read; UMI when present"; rule 3
(line 989): "long read = the A16 unit ... A molecule is one unit everywhere it appears"; rule 4(b) (line 990): dropped
"if any placement of any member ... overlaps an excluded interval".

Code.  `59_evidence_support.assign_molecules` (line 514) keys a UMI unit on `("umi", sample, umi, locus, o.strand)`
and hashes `repr(key)`; the accepted path passes `o.strand = None` when `transcript_strand` finds no evidence, the
rejected path (`rejected_member_unit`, old lines 815–821) substituted `'?'` → `repr` differs → two molecules.

Fix.  `b6_build_c0.py` 830–862: `strand = None`, the strand comes from the same `ES.transcript_strand` call the
accepted path makes, and an `EvidenceError` there now fails closed (before it was swallowed into `'?'`).  `'?'`
remains only the schema spelling written to `evidence_alignment.transcript_strand` (line 782).

Test.  `test_b6_c0_world.write_ont_umi` (new fixture: reads k1 accepted at a clean span, k2 MAPQ 10 on the held-out
gene, same library/sample/UMI/locus, no `ts` tag, `uf=none`); the test asserts both rows carry `transcript_strand '?'`,
`mol(k1) == mol(k2)`, both `excluded_heldout`, `rejected_members_in_a16_unit == 1`, leakage QC 0.

## 2. Conflict resolution before eligibility (CONFIRMED)

Spec.  Rule 6 (line 992): "HSPs whose frame is inconsistent within one EST after projection are dropped"; rule 3
(line 989): "the leakage-closure unit is the protein query sequence with all its alignments ... one unit everywhere it
appears"; rule 6: protein queries touching excluded intervals are removed molecule-wide.  Which HSPs exist for the
rule-6 test therefore depends on the protein query's rule 4(b) state, which only the protein C0 build decides
(`decide_all` runs before `load_hsp`).

Code.  `b6_hsp_project._emit` (old 616–627) ran `geometry.resolve_hsps` per (molecule, source) over every projected HSP
and dropped the conflicted ones before any eligibility was known; `b6_build_c0.load_hsp` then joined the protein state
onto the survivors only.  Codex's scenario (P excluded phase 0, Q retained phase 1, same span) lost Q.

Design chosen (single producer, fail-closed).  `b6_hsp_project.py` now emits **every** projected HSP (no drop;
`MULTI_HSP_APPLIED_BY = 'b6_build_c0.load_hsp'` in its provenance and QC; per-source
`phase_conflict_candidates_pre_eligibility` and `cross_source_conflict_hsps` are information only; docstring 32–45,
constants 106–108, `_emit` 628–640).  `b6_build_c0.load_hsp` (1229–1320) streams rows molecule by molecule and applies
`est_phase_project.resolve_hsps` (`GEOM`, line 104) per (molecule, protein_source) **among HSPs that are not
leakage_excluded**; conflicted HSPs are dropped and counted (`hsp_dropped_phase_conflict_within_molecule[_orthodb|
_swissprot]`, `hsp_rows_dropped_phase_conflict_within_molecule`); excluded rows are written flagged as before.
`verify_hsp_provenance` (1151) refuses a projection whose provenance does not say `multi_hsp_rule_applied_by =
b6_build_c0.load_hsp` (an older projection that resolved first).  Two independent re-tests of the written rows:
`HSP_CONFLICT_SQL` (1513; strand / frame equality derived from `est_phase_project.base_label`) in the step-7 gate
(`hsp_phase_conflict_among_retained_recheck`, required 0 at 1707) and in `b6_merge_evidence_db.hsp_gate` (120).
Provenance records `multi_hsp_rule.applied_by` (1754).  Contract change (loud): the route-B TSV is no longer
conflict-resolved; the protein C0's `est_protein_hsp` is the resolved set, and `b6_routea_agreement.py` reads route B
from it (finding 3).  Also fixed on the way: an HSP of a `never` (validation-only) EST molecule was written without any
flag; it is now `leakage_excluded` with cause `est_molecule_never` (1299–1304, QC `hsp_rows_excluded_never_est_molecule`)
and the merge gate checks it (101).

Tests.  Builder: P (`4577_0:000002`, excluded) phase 0 and Q (`4577_0:000001`) phase 1 over e2's [10000,10150) → Q
survives with phase 1, P's rows flagged `protein_query`, `hsp_dropped_phase_conflict_within_molecule == 0`; adding an
eligible R (`39947_0:000003`, phase 2) over Q's span drops Q and R (count 2) and keeps P flagged.  Merge: an inserted
retained row in another frame over h1's interval is refused; the same frame from 3 nt later merges.  Projector: the
three conflict tests now assert emission + candidate counts.

## 3. Agreement inputs unbound (CONFIRMED)

Spec.  Rule 9 (line 997): one evidence DB sha256 identifies the run; rule 4(d)/13: any non-zero leakage aborts.  The
protein C0 records `inputs.est_c0`, `inputs.hsp`, `inputs.hsp_molecule_state`, `inputs["alignment:protein"]`
(`build`, 1600–1620 / `verify_est_c0`).

Code.  Old `load_protein_state` (162–185) opened the protein C0 and only required that every accepted GFF protein have
a molecule; a stale protein C0 with a clean placement of P zeroed while the current GFF placed P on held-out bases.

Fix (`b6_routea_agreement.py`, rewritten; docstring 26–50).  `bind_inputs` (113–134): `--est-c0`, `--route-b`, its
`.molecule_state.tsv` sidecar and `--miniprot-gff` are required and each file's sha256 must equal the sha256 the
protein C0 recorded for it; `verify_accept_table` (231–270): every GFF mRNA (per protein in file order = the C0's
`record_index`) must have a C0 `evidence_alignment` row whose `mapping_ambiguous` equals membership in the ambiguous
list and whose `accepted` equals (verdict accepted and not ambiguous); the accept-table protein must equal the GFF
Target; the C0 may hold no alignment the GFF lacks.  Route B is read from the protein C0 (`load_route_b`, 137–171):
rows flagged `leakage_excluded` / `train_tiles_excluded` are dropped and counted by cause
(`rows_dropped_excluded_heldout`, `rows_dropped_never_training`, `rows_dropped_protein_query_excluded`,
`rows_dropped_train_tiles_excluded`, `rows_dropped_protein_query_train_tiles_excluded`, `rows_used`).  The QC JSON
records `inputs_bound_to_protein_c0` (paths + sha256) and `gff_alignments_bound`.  The pyarrow-based sidecar
cross-check is gone (the protein C0 already cross-checked the sidecar at build time; the sidecar is now sha-bound).

Fixture.  `test_b6_routea_agreement.py` was rewritten: every case (`build_case`, 110–121) writes its route-B rows on
real placements of the world (e2 over gA [10000,10800), h1 over gH, v1 over gV), its miniprot GFF in the builder's
layout (`_write_gff`, `##PAF` + phased CDS) and an accept table / ambiguous list as `protein_accept.py` derives them,
builds a published protein C0 and runs the agreement against it — the GFF the test compares IS the GFF the protein C0
was built from.  Dropped tests: `test_unknown_route_b_source_fails_closed` (the c0-v1 CHECK makes an unknown source
unbuildable), `test_held_out_hsp_cannot_remove_a_zero_interval` and the `tte` half of the excluded-molecule case as
written (no excluded EST molecule of the world has an accepted primary on clean bases — s1's primary is
`supplementary_or_chimeric` — so a held-out HSP cannot overlap a train interval; the filter is covered by
`test_excluded_est_molecule_hsps_take_part_in_no_zeroing_decision`, `test_train_tiles_excluded_est_molecule_hsps_are_dropped`
and the pure-function assertions in `test_excluded_protein_query_takes_part_in_no_zeroing_decision`).

Test.  `test_inputs_are_bound_to_the_protein_c0`: a GFF with one more protein, a TSV with one more byte, an edited or
missing sidecar, the protein C0 passed as `--est-c0`, a flipped verdict (MP2), a hidden ambiguity (Z), a truncated GFF
→ each `ValueError` names the input and nothing is written; `--protein-c0` missing → `SystemExit`; the EST C0 as
`--protein-c0` → `C0Error`.

## 4. Identity over the union of keys (CONFIRMED)

Code.  Old `provenance_gate` line 143: `diff = sorted(k for k in ident if ident[k] != first[1][k])` — extra keys on the
first input were never visited (silent pass), extra keys on a later input raised `KeyError` (not `MergeError`).

Fix.  `b6_merge_evidence_db.identity_diff` (151–153) over `set(a) | set(b)` with `.get`; used at 170.

Test.  An ONT copy without `build_flags_1` and one with an extra `build_flags_2`, each merged in both orders → four
`MergeError`s naming the key; `identity_diff` unit-checked both ways.

## 5. Exhaustiveness per species (PARTIALLY CONFIRMED)

Spec (exact text, rule 4(b), line 990): "Exhaustiveness = the fraction of molecules with fewer than N = 50 placements,
reported per species and required to be **≥ 99.9 %**; otherwise N is raised and the audit rerun before any track is
built."  Species-level, as Codex says; the number Codex quotes (99.9999 %) is not in the spec — the threshold is 99.9 %
(`leakage.MIN_EXHAUSTIVENESS = 0.999`, unchanged).  Rule 4(d) keeps the QC table "per species × source".

Code.  `decide_all` (old 1304) called `LK.Exhaustiveness.check()` per species × source build and aborted the build
(`ABORT_EXHAUSTIVENESS`) below 99.9 % — one capped PacBio molecule among 500 aborted although the species was fine.

Fix.  `leakage.Exhaustiveness.report()` (516–527) returns `exhaustiveness_molecules` (denominator),
`exhaustiveness_below_cap` (numerator), `audit_n` and the per-source fraction for reporting; the builder records them in
`c0_qc` (1394) and no longer aborts (the capped molecule is still excluded as unauditable; docstring 19–22;
`EXHAUSTIVENESS_RULE` 109–111 in provenance).  `leakage.species_exhaustiveness` (541–567) sums the parts of one species,
requires one N and ≥ 0.999.  `b6_merge_evidence_db.exhaustiveness_gate` (185–204) runs it over every audited source
(est / ont / pacbio) before publication, fails closed when a source lacks the counters or the sources disagree on N,
and writes the species result to `c0_meta` key `exhaustiveness` (285) and the merge result.  `check()` is kept for the
in-memory `decide_molecules` and its leakage tests.

Tests.  Builder: e1 with 50 audit secondaries → `PASS`, e1/e2 `excluded_heldout` with `b_secondary_cap`, QC 16/17,
`c0_qc` rows present.  Merge: the EST C0 alone → `MergeError` "species-level audit exhaustiveness 0.941176 < 0.999 …
raise N"; with an ONT C0 of 1203 molecules (`test_b6_c0_world.write_ont_many`, 1200 distinct mono-exonic units) the
species sum 1219/1220 = 99.92 % publishes and records `by_source`; a missing counter and an `audit_n` of 60 vs 50 are
refused; `species_exhaustiveness` unit-checked at the 0.999 boundary.

## 6. A16 completeness codes (CONFIRMED — author decision required)

Spec (A16, line 625): "Every accepted read alignment carries three codes stored in the C0 tables: `complete_5p_code`
(5C/5I), `complete_3p_code` (3C/3I), `read_internal_code` (IC/IP/IM/IX), with the thresholds of the design document
§2. FLNC/"full-length" library labels do not imply 5C; direct-RNA reads default to 5I unless cap/CAGE clustering
establishes 5C."  Design §2 (lines 12–14): 5C needs a cap-trapped/TeloPrime primer or strand-switch adapter **and** an
end within 50 nt of a cap/CAGE TSS, or a TSS cluster with ≥ 5 units from ≥ 2 libraries whose medoid is within ±50 nt
of a same-strand CAGE peak; 3C needs untemplated poly(A) ≥ 10 nt through the A3 internal-priming filter or a TES
cluster with ≥ 2 tail-positive units; IC/IP need the **admitted** chain set.

Facts.  So the codes are required in C0 per read.  No producer exists: `59_evidence_support.py` (line 951) treats
"S12d terminal completeness" as a separate C0 analysis; no script computes `complete_5p_code` (grep over `src` and
`revision/scripts`).  The inputs are not available to this per-run streaming build: cap/CAGE peaks are not a B6 input
at all; TSS/TES clusters and the admitted chain set are species-wide aggregates over every molecule and library
(i.e. over the merged DB).  Inventing defaults (e.g. `5I` everywhere) would silently satisfy a `NOT NULL` reading while
mislabelling every cap-trapped read.

Fix (no value invented).  `COMPLETENESS_CODES_POLICY` (`b6_build_c0.py` 112–133) — policy `null_not_computed_at_c0`,
the A16 clause, the missing inputs, status `AUTHOR_DECISION_PENDING_2026-09-23` — is recorded in `c0_provenance` and
PROVENANCE.txt (1753); the step-7 gate counts accepted rows (`completeness_codes_accepted_rows`, `_null_accepted_rows`,
`_coded_accepted_rows`, `COMPLETENESS_COUNTS_SQL` 1525) and `completeness_codes_policy_violation`
(`completeness_policy_violation`, 1532: under the NULL policy any set code is a violation) is required to be 0 (1710).
The merge (`completeness_gate`, 207–218; re-count at 248–253) refuses an input without a known policy or whose rows
contradict it.  The c0-v1 schema is unchanged (the columns were already nullable).

**Author decision needed.**  Either (a) a post-C0 pass over the merged evidence DB computes the three codes from the
design §2 thresholds once TSS/TES clusters and the admitted chain set exist (and the cap/CAGE input for 5C is named as a
B6 input or 5C is declared unreachable and every read is 5I), or (b) A16 is amended to state the C0-time defaults.  Until
then every accepted row is NULL under a declared, gated policy.

## Files changed (absolute line anchors in the current files)

- `revision/scripts/evidence/b6_build_c0.py`: 19–22 (docstring step 3), 104 (`GEOM`), 106–134 (`EXHAUSTIVENESS_RULE`,
  `COMPLETENESS_CODES_POLICY`, `MULTI_HSP_APPLIED_BY`), 830–862 (`rejected_member_unit`), 1151–1154
  (`verify_hsp_provenance`), 1229–1320 (`load_hsp`: `emit_molecule`, never flag, streaming per molecule), 1394
  (`ex.report()`), 1513–1536 (`HSP_CONFLICT_SQL`, `COMPLETENESS_COUNTS_SQL`, `completeness_policy_violation`),
  1578–1585 (gate counters), 1707–1713 (gate requires), 1752–1756 (provenance keys), `code` hash now includes
  `est_phase_project.py`; the `ExhaustivenessError` status path is gone.
- `src/transgenic/evidence/leakage.py`: 516–527 (`Exhaustiveness.report`), 541–567 (`species_exhaustiveness`).
- `revision/scripts/evidence/b6_merge_evidence_db.py`: docstring 6–24, imports 49–56, `hsp_gate` 96–122 (never flag,
  conflict re-test), `identity_diff` 151–153, `provenance_gate` 156–182, `exhaustiveness_gate` 185–204,
  `completeness_gate` 207–218, `merge` 244–260 / 285 / 301.
- `revision/scripts/evidence/b6_hsp_project.py`: docstring 32–45, 106–108, provenance 538, report 602–612, `_emit`
  628–640.
- `revision/scripts/evidence/b6_routea_agreement.py`: rewritten (docstring, `bind_inputs`, `load_route_b` from the C0,
  `gff_mrnas`, `verify_accept_table`, `run`, CLI help).
- Tests: `tests/test_b6_c0_world.py` (`_write_meta(extra_cols)`, `write_ont_umi`, `write_ont_many`),
  `tests/test_b6_c0_build.py` (provenance helper adds `multi_hsp_rule_applied_by`; exhaustiveness test rewritten; 4 new
  tests at the end), `tests/test_b6_c0_merge.py` (5 new tests + `capped_dbs` fixture), `tests/test_b6_hsp_project.py`
  (3 conflict tests updated), `tests/test_b6_routea_agreement.py` (rewritten: 13 tests, each case builds its protein C0).
- Not touched: `c0_schema.py` (schema unchanged), `b6_leakage_intervals.py`, `b6_audit_runs.py`, `b6_est_molecule_map.py`,
  `b6_c0_metadata.py`, `est_phase_project.py`, the manifests, any MVP file.

Conventions kept: every new reader has an `ORDER BY`; the new checks are set-based SQL or one-molecule-at-a-time
streams; new QC counters are added (the projector's `drop_phase_conflict_within_molecule` no longer exists because
nothing is dropped there — replaced, not repurposed, by `phase_conflict_candidates_pre_eligibility`); c0-v1 schema
unchanged.  Contract changes to say out loud: (1) the route-B TSV is unresolved — the protein C0 is the resolved set;
(2) `b6_routea_agreement.py` reads route B from `--protein-c0` and requires its other inputs to be the C0's recorded
files; (3) a per-source build no longer aborts on exhaustiveness — the merge does, per species; (4) every C0 must
declare `completeness_codes.policy`.

## Test results

Run by the team lead (2026-09-23 13:35-13:56 PDT; the writer's own run was not recorded), from `/tmp/b6-wp1-20260922`,
`PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp`,
`/tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q -p no:cacheprovider`:

```
tests/test_b6_*.py tests/test_mvp_final_source_integration.py::test_reviewed_a_implementation_bytes_unchanged
  -> 176 passed in 1254.92s (0:20:54)   (exit 0; 175 B6 tests + the MVP byte pin)
```

## Open decisions for the author

1. Finding 6: compute the A16 codes in a post-C0 pass over the merged DB (needs a cap/CAGE input for 5C, TSS/TES
   clusters, the admitted chain set) or amend A16 with C0-time defaults; until then the declared NULL policy stands.
2. Finding 5: the per-source exhaustiveness is now reported only; confirm that the species-level enforcement in the
   merge (before any track is built) is the intended reading, or ask for an additional per-source floor.
3. Finding 2: conflict resolution among eligible HSPs only — `train_tiles_excluded` (but not leakage-excluded) HSPs take
   part; confirm, or restrict the resolution to fully train-eligible HSPs.
4. Finding 2 side fix: HSPs of `never` (validation-only) EST molecules are now `leakage_excluded` (cause
   `est_molecule_never`); confirm.
5. Carried over from round 4: route-B `protein_id` inherits the miniprot query's rule 4(b) state; ONT non-UMI rejected
   reads keep a per-read unit; roles manifests are outside the merge identity; the genotype alias.
