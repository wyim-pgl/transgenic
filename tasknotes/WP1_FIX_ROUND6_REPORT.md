# WP1 fix round 6 — Codex review 3 (1 HIGH, 1 MEDIUM) + author decisions of 2026-09-23, verified and fixed

Worktree: `/tmp/b6-wp1-20260922` (branch `b6-wp1-20260922`, uncommitted; line anchors below are absolute line numbers
of the current files).  Spec quotes: `PROTOCOL_B1_frozen_v1.md` (protocol line numbers) and
`notes/a46_approval/A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md`.  Nothing was committed, stashed or checked out; no
Slurm job was submitted; no file outside the worktree was written; no MVP file was touched.

| # | Finding / decision | Verdict | Fix | Test (fails before / passes after) |
|---|--------------------|---------|-----|------------------------------------|
| 1 | Run accessions split one biological molecule (`library = args.run`) | **confirmed** | `b6_c0_metadata.py` new required `--library` (never the run); builder QC `libraries_spanning_runs` / `molecules_spanning_runs` | `test_b6_c0_build.py::test_two_runs_of_one_library_form_one_a16_molecule_and_close_together`, `::test_two_runs_of_different_libraries_are_the_a11_split`, `test_b6_c0_metadata.py::test_longread_library_is_explicit_and_never_the_run` |
| 2 | Held-out evaluation HSPs left conflict-unresolved | **confirmed** | `b6_build_c0.load_hsp`: MULTI_HSP_RULE per population (training / excluded), never across; `HSP_CONFLICT_EXCLUDED_SQL` gate in the builder and the merge | `test_b6_c0_build.py::test_hsp_phase_conflicts_are_resolved_within_the_excluded_population_too`, `test_b6_c0_merge.py::test_merge_rechecks_phase_consistency_among_excluded_hsps_and_never_across` |
| 3 | Author: per-source exhaustiveness floor in addition to the species gate | **applied** (number assumed = 0.999) | `leakage.MIN_SOURCE_EXHAUSTIVENESS` + `source_exhaustiveness`; build aborts via `Exhaustiveness.check`; merge re-checks from recorded counters | `test_b6_c0_build.py::test_capped_molecule_below_the_per_source_floor_aborts_the_build`, `::test_exhaustiveness_counters_and_both_rules_are_recorded`, `test_b6_c0_merge.py::test_per_source_floor_and_species_sum_are_rechecked_in_the_merge_from_recorded_counters` |
| — | A16 completeness codes: author chose a post-C0 pass | nothing to change | declared NULL policy kept as is (`COMPLETENESS_CODES_POLICY`, gate unchanged) | existing `test_a16_completeness_codes_null_policy_is_recorded_and_gated` |
| — | Round-5 rulings (train_tiles_excluded HSPs take part in the training-population resolution; `never` EST molecules' HSPs flagged `est_molecule_never`) | confirmed by the author | unchanged | existing round-5 tests |

"Fails before" is established from the pre-fix code path in each case (the worktree has no committed baseline of these
files): before, the producer wrote `library = run` and the fixture mirrored it, so the two-run PacBio scenario gave
two molecules (the `different_libraries` test is exactly that outcome); before, an excluded molecule's conflicting
HSPs were both written (the round-5 resolution ran only among non-excluded rows); before, a 16/17 EST build passed
(`test_capped_molecule_is_excluded_and_exhaustiveness_is_recorded_per_source_not_enforced` asserted exactly that).

## 1. Run accessions split one biological molecule (CONFIRMED)

Spec.  A11 (protocol line 558): "A biological library is one independently prepared cDNA library; runs, plates, read
directions, accession series and download batches do not create libraries."  A16 (line 625): "PacBio: one source
FLNC/ZMW molecule = one unit"; ONT non-UMI: "reads from the same library with the same strand, the same corrected
intron chain and both ends within 10 nt are one PCR-equivalence unit".  A46 rule 3 (line 989): "long read = the A16
unit ... A molecule is one unit everywhere it appears"; rule 4(b) (line 990): dropped "if any placement of any member
... overlaps an excluded interval".

Code.  `b6_c0_metadata.longread_rows` (old line 281) wrote `"library": args.run` into the `'*'` row and the docstring
(old lines 28–29, 43–44) declared "one sequencing run = one library, A16" and "the library is the run, so a movie
shared by several runs still gives distinct units".  Every long-read unit key in `59_evidence_support.assign_molecules`
(lines 501–518) is `(species, assembly, source, library, ...)` and the builder's PCR grouping key is `(library,
unit_key)` (`assign_read_units`, line 951), so with `library = run` a ZMW / UMI / PCR-equivalence unit sequenced in two
runs of one preparation hashed to two molecules and rule 4(b) closed only the one whose member touched the excluded
interval — Codex's scenario, reproduced by the new test with the fixture's old `library=run` default.

Fix.
- `revision/scripts/evidence/b6_c0_metadata.py`: usage lines 9–13; `library` field contract lines 24–40 (source of the
  value, see "Contract changes"); `source_molecule` note 54–56; `longread_library` (289–299: required, stripped,
  not `unknown`, **refused when equal to the run accession**); `longread_rows` 301–306 writes it into the `'*'` row;
  argparse `--library` 361–363; `main` 375–382 validates it before any file is written and records
  `prov["library"]` explicitly in `<out>.PROVENANCE.json` (the `run` stays its own key).  The "one sequencing run =
  one library" text is gone.
- `revision/scripts/evidence/b6_build_c0.py`: nothing downstream assumed `library == run` — the unit keys already
  came from the metadata `library` (59's formula, `rejected_member_unit` line 866, the PCR group key line 951, the
  direct-RNA key includes `o.run` by 59's own design), `member_id` and the alignment / audit / DONE layout are keyed on
  `(dataset, run)` and unchanged, `molecule_member.run` is kept.  Added: docstring step 3 (lines 19–26),
  `assign_read_units` docstring (912–917), `run_library_qc` (911–921; qc `libraries`, `libraries_spanning_runs`,
  `molecules_spanning_runs`, set-based SQL over `stg.member` / `stg.unit`), called at 1729 after the units exist.
- Tests: `tests/test_b6_c0_world.py` `sam(..., as_score=)` (117), `PACBIO_LIBRARY`, `pacbio_role_row`, `PACBIO_RUN2` /
  `PACBIO_READS2` / `PACBIO_AUDIT_EXTRA2`, `write_pacbio(tmp, run, *, library, reads, audit_extra)` (317–352);
  `tests/test_b6_c0_metadata.py` `_ont` / `_pacbio_fasta` pass `--library`, assertions updated (library is the
  experiment, never the run), new `test_longread_library_is_explicit_and_never_the_run` (179–212).

Test.  `test_two_runs_of_one_library_form_one_a16_molecule_and_close_together`: run SRR2 (p1 ZMW m64011/42 accepted
at a clean locus, p3 ZMW 43) and run SRR5 (p6 ZMW m64011/42, accepted at another clean locus, audit secondary MAPQ 0
on the held-out gene, AS below the primary so it is not equal-best), both `--library SRX_PB1` → one `library`, `run`
kept per member, `mol(p1) == mol(p6) != mol(p3)`, p1 and p6 `excluded_heldout` with one molecule-wide cause set, p3
`train_track`, `leakage_nonzero == 0`, `heldout_retained == 0`, `libraries == 1`, `libraries_spanning_runs == 1`,
`molecules_spanning_runs == 1`.  `test_two_runs_of_different_libraries_are_the_a11_split`: the same reads with
`SRX_PB1` / `SRX_PB2` → two molecules, p1 stays `train_track` (the documented A11 split; the per-run `--library` is
the author's statement of the preparation).  `test_longread_library_is_explicit_and_never_the_run`: missing, blank,
`unknown` and `== run` → exit 1 with no output files; a value → `'*'` row and PROVENANCE.json carry it, `run` stays
separate; ONT takes an author key verbatim.

## 2. Held-out evaluation HSPs left conflict-unresolved (CONFIRMED)

Spec.  A46 rule 6 (line 992): "HSPs whose frame is inconsistent within one EST after projection are dropped"; rule 3
(line 989): "the leakage-closure unit is the protein query sequence with all its alignments ... one unit everywhere it
appears"; rule 10 evaluation reads the held-out (excluded) molecules.  Nothing in rule 6 limits the resolution to the
training population.

Code.  Since round 5, `load_hsp.emit_molecule` (old 1237–1270) resolved conflicts only among HSPs that are not
`leakage_excluded`; `HSP_CONFLICT_SQL` (old 1518) and the merge `hsp_gate` tested only retained pairs.  An
`excluded_heldout` EST molecule with two overlapping Swiss-Prot HSPs in phase 0 / 1 therefore had both written,
flagged — a permitted evaluation population holding rule-6-inconsistent evidence (before round 5 the projector dropped
both).

Fix (`b6_build_c0.py`).  `resolve(population)` (1263–1275) applies `est_phase_project.resolve_hsps` (unchanged, per
`protein_source`) to one population's rows; `emit_molecule` (1277–1311) runs it twice per molecule — training
population = rows not `leakage_excluded` (round 5, unchanged: protein-query-excluded rows cannot delete an eligible
HSP; `train_tiles_excluded` rows take part, as the author confirmed), excluded population = the molecule's
`leakage_excluded` rows — and never across.  The C0 has no notion of "evaluation input" finer than `molecule_state`,
so ALL leakage-excluded rows of a molecule form the second population (for an `excluded_heldout` or `never` molecule
that is every row of the molecule = the rule-10 evidence; for a `train_track` molecule it is its protein-query-excluded
rows, unusable in either population, resolved among themselves and counted the same way).  New QC counters (added, not
repurposed): `hsp_dropped_phase_conflict_within_molecule_excluded`, `..._excluded_orthodb`, `..._excluded_swissprot`,
`hsp_rows_dropped_phase_conflict_within_molecule_excluded` (1365–1369).  `_HSP_CONFLICT_SQL` template (1557–1567)
yields `HSP_CONFLICT_SQL` (retained pairs, unchanged) and `HSP_CONFLICT_EXCLUDED_SQL` (both `leakage_excluded`);
step-7 gate counter `hsp_phase_conflict_among_excluded_recheck` (1628) required 0 (1759–1762); provenance
`multi_hsp_rule.scope` names both populations (1807–1810).  `b6_merge_evidence_db.hsp_gate` (126–129) re-tests the
excluded population too; docstring 23–29.  Rule-3 protein-query inheritance is unchanged (the `leak` flag still comes
from the EST molecule state, the interval index and the protein query's rule 4(b) state).

Tests.  Builder: h1 (held-out gene, `excluded_heldout`) with Swiss-Prot hA phase 0 / hB phase 1 overlapping, plus e2
(train_track) with hP (protein `4577_0:000002`, excluded, phase 0) and hQ (retained, phase 1) on one span → written
rows are exactly hP (flagged) and hQ; `hsp_dropped_phase_conflict_within_molecule_excluded == 2 ==
..._excluded_swissprot`, training counter 0, both recheck counters 0.  Merge: a copied excluded row of h2 in another
frame → `MergeError` "1 leakage-excluded est_protein_hsp pair(s)"; a retained h1 and an excluded row in another frame
on the same span merge (populations are never tested against each other).

## 3. Author decision: per-source exhaustiveness floor (APPLIED)

Spec.  A46 rule 4(b) (line 990): "Exhaustiveness = the fraction of molecules with fewer than N = 50 placements,
reported per species and required to be >= 99.9 %".  Author decision 2026-09-23: BOTH the species-level gate AND a per
(species × source) floor ("종 소스별 하한"); no number given.  **Assumption:** the floor is 0.999, the species value
(the spec text names no other number); recorded in code and provenance so it can be changed in one place.

Consequence to state plainly: with the floor equal to the species threshold, every source at or above its floor
implies the species sum is too, so the species gate can no longer fail on its own — it is kept (it still enforces one
N over the species' sources and the presence of the counters, and it stays the gate if the author later sets a lower
per-source number).  The round-5 merge test that merged a 16/17 EST source with a 1203/1203 ONT source (species sum
99.92 %) is therefore no longer a legal outcome and was replaced (see tests); the species check is still exercised
directly (`species_exhaustiveness`).

Fix.  `src/transgenic/evidence/leakage.py`: `MIN_SOURCE_EXHAUSTIVENESS = MIN_EXHAUSTIVENESS` (80–85),
`Exhaustiveness(..., label=)` (514) and `check()` (534–538) delegate to the new `source_exhaustiveness(label, below,
total, audit_n, min)` (541–557; `ExhaustivenessError` with the "raise N and rerun the audit" wording; an empty source
has no fraction); `species_exhaustiveness` unchanged.  `b6_build_c0.py`: docstring step 3 (19–24), `EXHAUSTIVENESS_RULE`
(111–121: `scope: per_source_floor_and_species`, `rule: "per-source floor 0.999 (author 2026-09-23) + species >= 0.999
(rule 4(b))"`, both thresholds, `enforced_by`), `decide_all` builds the counter with a `species/source` label (1379) and
calls `ex.check()` (1437; the abort is restored — `ExhaustivenessError` propagates out of `build`, status `ABORT`,
nothing published, the QC record keeps the counters).  `b6_merge_evidence_db.exhaustiveness_gate` (192–219) re-applies
`source_exhaustiveness` to every audited input's RECORDED counters (a DB built by an older builder cannot bypass the
floor), then the species sum; the result records `min_source_exhaustiveness`; docstring 23–27.

Tests.  `test_capped_molecule_below_the_per_source_floor_aborts_the_build` (replaces the round-5 "recorded, not
enforced" test): one capped molecule among 17 → exit 1, "per-source audit exhaustiveness 0.941176 < 0.999", "raise
N", `Athaliana/est` in the message, QC status `ABORT`, no DB.  `test_exhaustiveness_counters_and_both_rules_are_recorded`:
a passing build records 17/17 and the provenance rule.  Merge: `test_per_source_floor_and_species_sum_are_rechecked_
in_the_merge_from_recorded_counters` (replaces the round-5 species-sum test and its 1200-read fixture): counters
tampered to 16/17 (EST) + 1203/1203 (ONT) → `MergeError` "per-source exhaustiveness floor: Athaliana/est ... 0.941176 <
0.999" although the species sum would pass; 999/1000 + 1203/1203 merges and records both thresholds; the pure
`species_exhaustiveness` / `source_exhaustiveness` checks.  `test_exhaustiveness_counters_and_one_audit_n_are_required`
unchanged.

## Contract changes (say out loud)

1. **`b6_c0_metadata.py --source ont|pacbio` now REQUIRES `--library <id>`.**  Its value is the run's biological
   library (A11): the ENA/SRA **experiment** accession (SRX/ERX/DRX — in the SRA model one experiment = one library
   preparation, its runs are sequencing runs of it), or an author-supplied library key when the preparation is known
   outside SRA (several experiments that are one preparation → one key, recorded in the author's notes).  It is never
   defaulted; `unknown` and a value equal to the run accession are refused.  The `'*'` row `library` column and
   `<out>.PROVENANCE.json` `library` carry it; `run` stays separate everywhere.  DATASET_ROLES has no experiment column,
   so the author (or the ONT protocol sheet) must supply one per run before the metadata is produced.  Every
   long-read `c0_metadata.tsv` produced so far (if any) was written with `library = run` and must be regenerated.
2. Two runs sharing a `--library` merge into one A16 molecule wherever the unit key matches (ZMW, UMI, PCR-equivalence
   chain); different `--library` values keep them apart (the A11 split).  New informational QC: `libraries`,
   `libraries_spanning_runs`, `molecules_spanning_runs`.
3. Route-B `est_protein_hsp` is now conflict-resolved within BOTH populations (retained; leakage-excluded), never
   across; rows dropped from the excluded population are counted under the new `*_excluded` QC keys; the builder gate
   and the merge gate both require `hsp_phase_conflict_among_excluded_recheck == 0`.  Provenance `multi_hsp_rule.scope`
   changed (provenance hash changes; no C0 has been built).
4. A per-source build ABORTS again below the per-source exhaustiveness floor (0.999, assumed); the merge re-checks the
   floor from recorded counters and then the species sum.  `EXHAUSTIVENESS_RULE` provenance changed accordingly.
5. c0-v1 schema unchanged.  No MVP file touched.

## Files changed (absolute line anchors in the current files)

- `revision/scripts/evidence/b6_c0_metadata.py`: 9–13, 24–40, 54–56, 289–306, 361–363, 375–382.
- `revision/scripts/evidence/b6_build_c0.py`: 19–26, 111–121, 911–921, 912–917, 1263–1311, 1365–1369, 1379, 1437,
  1557–1567, 1625–1628, 1729, 1759–1762, 1807–1810.
- `src/transgenic/evidence/leakage.py`: 80–85, 514, 534–557.
- `revision/scripts/evidence/b6_merge_evidence_db.py`: 23–29, 126–129, 192–219.
- Tests: `tests/test_b6_c0_world.py` (117, 317–352), `tests/test_b6_c0_build.py` (imports; 234–265; 713–789),
  `tests/test_b6_c0_metadata.py` (import json; 93–95; 102–110; 116; 147–148; 154–176; 179–212),
  `tests/test_b6_c0_merge.py` (225–267; 321–346).
- Not touched: `c0_schema.py`, `est_phase_project.py`, `59_evidence_support.py`, `b6_hsp_project.py`,
  `b6_routea_agreement.py`, `b6_leakage_intervals.py`, `b6_audit_runs.py`, `b6_est_molecule_map.py`, the manifests,
  the protocol, any MVP file.

Conventions kept: every new reader has an `ORDER BY` or is a set-based aggregate; `load_hsp` still streams one molecule
at a time; new QC counters are added, none repurposed; c0-v1 schema unchanged.

## Test results

Command (from `/tmp/b6-wp1-20260922`, `PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp`,
interpreter `/tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q -p no:cacheprovider`):

```
not run by the writer (session limit); the team lead ran the suite after round 7: see WP1_FIX_ROUND7_REPORT.md
```

## Open author decisions

1. Item 3: the per-source floor number — 0.999 (= species value) is an assumption; confirm or give the intended number
   (`leakage.MIN_SOURCE_EXHAUSTIVENESS`, one constant).  With an equal number the species gate is implied; with a lower
   per-source number both gates are live.
2. Finding 1: the source of `--library` per run — the SRA/ENA experiment accession by default; for runs whose
   experiments are known to be one preparation, an author key.  DATASET_ROLES has no such column; decide whether to add
   an `experiment`/`library` column to the manifest (then the producer can cross-check `--library` against it) or to
   keep it in the ONT protocol sheet.  A `--library` equal to the run accession is refused; confirm that no run must
   legitimately use its own accession as its library key.
3. Finding 2: the excluded population is "all leakage-excluded rows of the molecule" (no finer notion of evaluation
   input exists at C0); confirm, or define the evaluation population (e.g. only `excluded_heldout` / `never` molecules)
   so that protein-query-excluded rows of `train_track` molecules are left unresolved.
4. Carried over: A16 completeness codes post-C0 pass (author chose; separate task); route-B `protein_id` inherits the
   miniprot query's rule 4(b) state; ONT non-UMI rejected reads keep a per-read unit; roles manifests outside the merge
   identity; the genotype alias.
