# WP1 fix round 4 — Codex review (5 HIGH) verified against A46 / A18 and fixed

Worktree: `/tmp/b6-wp1-20260922` (branch `b6-wp1-20260922`, uncommitted; the B6 files are untracked there, so
line anchors below are absolute line numbers of the current files, not diff hunks).  Spec quotes: `PROTOCOL_B1_frozen_v1.md`
(protocol line numbers) and `notes/a46_approval/A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md`.

| # | Finding | Verdict | Fix | Test (fails before / passes after) |
|---|---------|---------|-----|------------------------------------|
| 1 | Rejected long-read members escape molecule-wide closure | **confirmed** (PacBio, UMI); residual for ONT PCR-equivalence noted | `b6_build_c0.rejected_member_unit` | `test_b6_c0_build.py::test_rejected_pacbio_member_closes_its_a16_molecule` |
| 2 | Protein-query exclusion does not reach phase evidence / zero-lists | **confirmed** (two halves: builder HSP rows and route-A/B agreement script) | `load_hsp` protein-state join; `b6_routea_agreement --protein-c0`; merge gate | `test_b6_c0_build.py::test_hsp_of_an_excluded_protein_query_is_leakage_excluded`, `test_b6_routea_agreement.py::test_excluded_protein_query_takes_part_in_no_zeroing_decision`, `::test_accepted_alignment_of_a_protein_absent_from_the_protein_c0_fails_closed`, `test_b6_c0_merge.py::test_merge_gate_rechecks_protein_query_flags_and_accepted_placements` |
| 3 | Route B admits EST placements rejected by C0 | **confirmed** | molecule map `accepted` column; projector `ACCEPTED_RULE`; builder `verify_hsp_placements`; merge gate | `test_b6_hsp_project.py::test_rejected_alignment_qnames_are_dropped_and_counted`, `test_b6_c0_build.py::test_hsp_projected_through_a_rejected_est_placement_is_refused`, `test_b6_c0_molecule_map.py::test_map_exports_the_c0_units_with_states_and_a_sha256_sidecar` (extended), merge test above |
| 4 | Merge accepts incompatible provenance across inputs | **confirmed** | `b6_merge_evidence_db.provenance_gate` | `test_b6_c0_merge.py::test_merge_refuses_a_protein_c0_projected_against_another_est_c0`, `::test_merge_refuses_inputs_with_different_exclusion_identity` |
| 5 | Frozen genotype vocabulary blocks 74 eligible runs | **confirmed** | `b6_c0_metadata.canonical_genotype` (alias `non_reference` → `known_nonreference` at ingestion) | `test_b6_c0_metadata.py::test_manifest_non_reference_is_normalised_to_the_c0_known_nonreference_token` |

"Fails before" is established from the pre-fix code path in each case (the worktree has no committed baseline of these
files to re-execute): before, p1 got a per-read unit; no protein-state join existed; no placement check existed; the
merge had no cross-input provenance check; `role_row` raised `MetadataError` on `non_reference` (Codex reproduced that
on SRR17876321).

## 1. Rejected long-read members (CONFIRMED)

Spec. A46 rule 4(b), protocol line 990: "A molecule unit is dropped from all training-side tracks if any placement of
any member (primary, supplementary, secondary-equivalent, `mapping_ambiguous`) overlaps an excluded interval on either
strand."  Rule 3, line 989: "long read = the A16 unit ... A molecule is one unit everywhere it appears."  A16, line 625:
"PacBio: one source FLNC/ZMW molecule = one unit".  Nothing in A16/A18.5 makes unit membership conditional on
acceptance; A18.5 (line 270) governs which *observations* enter, rule 4(b) governs which *members* close a molecule.

Code. `b6_build_c0.py:process_read` set `unit_kind/unit_key` only when `obs is not None` (old lines 719–736), and
`assign_read_units` (old 876–878) gave every unit-less member `sid("molecule", species, source, dataset, run, read)`,
a per-read molecule.  The rejected read's primary/supplementary placements and audit secondaries were tested only
against that lone unit, exactly Codex's scenario.

Fix (`b6_build_c0.py`): new `rejected_member_unit` (lines 797–831), called from `process_read` (line 737–739) for
every non-EST member whose primary was not accepted.  For PacBio (`source_molecule`) and UMI ONT it builds the same
namespace the PCR path already builds and calls `59.assign_molecules` itself, so the key formula is 59's, not a copy
(metadata-only for PacBio; the UMI key includes the transcript strand, read from the primary record's orientation,
`'?'` when undeterminable).  The unit is `direct`, so `assign_read_units` gives it the shared molecule id and
`decide_all` sees every member's placements.  QC: `rejected_members_in_a16_unit`, `rejected_members_per_read_unit`.

Test: `test_rejected_pacbio_member_closes_its_a16_molecule` (new PacBio fixture `test_b6_c0_world.write_pacbio`):
p1 MAPQ 60 at a clean locus and p2 MAPQ 10 on the held-out gene share ZMW `m64011/42`; both are now
`excluded_heldout`, p3 stays `train_track`, `leakage_nonzero == 0`, `heldout_retained == 0`.

Residual (open decision for the author): an ONT cDNA read without UMI whose primary was rejected has no *corrected*
intron chain, and the A16 PCR-equivalence unit is defined on that chain (A18.2 line 252), so it stays a per-read unit
(counted in `rejected_members_per_read_unit`).  Its own placements still exclude *it*; what it cannot do is drag an
accepted PCR duplicate along.  Since a PCR duplicate by definition shares the accepted sibling's primary placement
(same chain, ends within 10 nt), the sibling is excluded by its own primary whenever the duplicate's primary touches an
excluded interval; the only remaining gap is a duplicate's *audit secondary* differing from the sibling's, which needs
a chain for rejected reads and is not specified by A16.  Direct RNA is one read = one unit already.

## 2. Protein-query exclusion (CONFIRMED, both halves)

Spec. Rule 3, line 989: "Protein: ... the leakage-closure unit is the protein query sequence with all its alignments,
A43 `mapping_ambiguous` included. A molecule is one unit everywhere it appears."  Rule 6, line 992: "protein queries
touching excluded intervals are removed molecule-wide (rule 3)."  So the miniprot query *is* a rule-4 molecule, and
"everywhere it appears" covers its appearance as the route-B `protein_id` (OrthoDB subjects are the same sequences
miniprot aligned; Swiss-Prot subjects were never miniprot-aligned and have no protein-C0 molecule, so they are
untouched).  The protein C0 already decides per query over all placements (`load_protein`, `pl` rows for every
placement incl. ambiguous, then `decide_all`); what was missing is that nothing downstream *read* that decision.

Code. `load_hsp` (old 1136–1158) inherited only the EST molecule's state.  `b6_routea_agreement.load_route_a` (old
175–213) took every `accepted` non-ambiguous alignment from `accept/<sp>.alignments.tsv` without any protein state, so
an excluded query's clean placement could still generate a `route_a_b_disagreement` zero interval, and its route-B
rows entered the comparison.

Fix.
- `b6_build_c0.load_hsp` (lines 1192–1233): LEFT JOIN of `h.protein_id` to this build's `stg.state` via
  `stg.member.read`; `leakage_excluded |= state == 'excluded_heldout'`, `train_tiles_excluded |= tte`, cause
  `protein_query`.  QC: `hsp_rows_excluded_by_protein_query_only`, `hsp_rows_train_tiles_excluded_by_protein_query_only`.
- `b6_routea_agreement.py`: new required `--protein-c0` (line 404); `load_protein_state` (162–184) reads the published
  protein C0 (`open_published_c0` gate) into {query: (state, tte)}; `load_route_a` (228–) drops `excluded_heldout` /
  `never` / train-tiles-excluded queries (QC `A_leakage_excluded_alignments_removed`, `A_never_training_alignments_removed`,
  `A_train_tiles_excluded_alignments_removed`) and **fails closed** on an accepted alignment whose protein has no
  protein-C0 molecule; `training_rows` (199–) also drops route-B rows of such queries
  (`rows_dropped_protein_query_excluded`).  Docstring updated.
- `b6_merge_evidence_db.hsp_gate` (62–100): re-derives both flags from the merged protein `molecule_member`.

Tests: builder test (excluded query `4577_0:000002` → its HSP on the retained EST e2 becomes `(True, True,
'protein_query')`; a Swiss-Prot row stays `(False, False)`); route-A tests (H excluded / V tile-excluded queries take
part in no zeroing, control query still zeroes; absent query fails closed; `--protein-c0` required); merge test.
The route-A test module now builds a protein C0 in its fixture and names its miniprot queries `4577_0:P` etc.

Author check requested: the reading that the route-B `protein_id` inherits the miniprot query's rule 4(b) state rests
on rule 3's "one unit everywhere it appears" + rule 6's "removed molecule-wide".  It is the fail-closed direction
(more rows flagged, none dropped from the DB), and it can be relaxed by deleting the LEFT JOIN in `load_hsp` and the
two `p.*` counts in `hsp_gate` if the author rules that rule 6's sentence is route-A only.

## 3. Route B through rejected EST placements (CONFIRMED)

Spec. Rule 6, line 992: "each HSP ≥ 30 aa gives the EST a frame over its aligned span; the frame is projected onto the
genome through the EST's spliced alignment"; tracks are "built per tile from retained observations"; A18.5 (line 270)
defines the retained observation: "primary alignment, MAPQ ≥ 20, aligned query fraction ≥ 0.80, identity ≥ 95 %".

Code. `b6_hsp_project.load_placements`/`read_sam_primary` (306–322, 229–252) take the BAM primary (FLAG & 0x904 == 0)
for every current qname, never the C0 verdict; the molecule map (`b6_est_molecule_map`, old COLUMNS at 48–49) carried no
acceptance field; `b6_build_c0.load_hsp` joined `estc0.molecule_member` only (old 1136–1142) and never
`evidence_alignment.accepted`.  The EST C0 keeps rejected members in `molecule_member` and a rejected-only molecule
whose placements touch nothing excluded is `train_track` (e.g. `u1`, unmapped, in the synthetic world), so the state
join cannot distinguish.

Fix.
- `b6_est_molecule_map.py`: new column `accepted` (COLUMNS/ID_FIELDS lines 54–56; export query 98–104:
  `bool_or(accepted AND placement_kind = 'primary')` per member), `rejected_rows` in the result and PROVENANCE.
  This changes the *molecule-map* contract (part B input), not the c0-v1 schema.
- `b6_hsp_project.py`: `MOLECULE_MAP_COLUMNS`/`MAP_ID_FIELDS` + `accepted` (86–88), `ACCEPTED_RULE` (95–97),
  `validate_molecule_map` parses it (212), `project_row` drops and counts `rejected_alignment_qname_hsps` (437–439),
  `current` excludes rejected qnames (510), provenance/report carry `accepted_rule` (535, 597).  A map without the
  column is refused ("molecule map needs columns").
- `b6_build_c0.verify_hsp_placements` (1148–1166), called right after `cross_check_hsp_state` (1190): every HSP
  interval must lie inside an ACCEPTED primary `evidence_alignment` of its sidecar qname (same chrom, contained span);
  else fail closed.  QC `hsp_rows_inside_accepted_est_placement`.
- `b6_merge_evidence_db.hsp_gate`: the same containment re-derived from the merged tables through
  (`molecule_id`, `est_accession`) → member → accepted primary.

Tests: projector drop/count + refusal of a map without the column; builder refusal for `u1` (unmapped, train_track)
and for an interval outside an accepted qname's placement; molecule-map export flags exactly `{u1, s1, m1}`;
merge refusal for a moved interval.  One existing test fixture was corrected: `test_hsp_inherits_the_est_molecule_train_tiles_flag`
placed its HSP at 170000–170100, outside v1's accepted primary (100000–100700); it now lies inside (100000–100100)
and asserts the same inherited flag.

## 4. Merge provenance across inputs (CONFIRMED)

Spec. Rule 9, line 997: `runtime_identity.txt` records ONE "evidence DB sha256"; rule 13 / rule 4(d): "Any non-zero
leakage QC count aborts the run"; the protein build itself binds to one EST C0 (`verify_est_c0`: same
`leakage_intervals_sha256`, `inputs.est_c0.sha256` recorded).  The merge (`b6_merge_evidence_db.merge`, old 87–108)
checked schema, species/source uniqueness, leakage gate and PROVENANCE sha per input, but never that the protein DB's
recorded `inputs.est_c0.sha256` is the EST DB being merged, nor that inputs of one species share the exclusion identity.

Fix (`b6_merge_evidence_db.py` 103–162, 180–190): `_provenance` decodes each input's `c0_provenance`;
`build_identity` = {`leakage_intervals_sha256`, `seed`, sha256 of `corpus`/`splits`/`flags`/`build_flags_*`/
`reference`/`gff` inputs}; `provenance_gate` requires equality within a species and, for a protein input, that the
species' EST input is present and its file sha256 equals the recorded `inputs.est_c0.sha256`.  The merge result
carries `identity` per species.  Docstring updated.  Roles files are deliberately NOT part of the identity (a manifest
may gain rows between builds without changing any exclusion input); say so if the author wants them included.

Tests: an EST DB with identical molecule ids/states but other bytes (edited `junction`) is refused
("rebuilt after the projection"); a differing `leakage_intervals_sha256` or `seed` is refused; the recorded EST DB
merges.  The existing orphan test's first expectation changed from "orphan" to the earlier, more specific message
("no Athaliana EST C0 is among the inputs").

## 5. Genotype vocabulary (CONFIRMED)

Facts. `DATASET_ROLES.longread_v1.tsv` `genotype_stratum` values: reference 90, `non_reference` 77 (74 of them
`c2_training_eligible`), unknown 35, hybrid_pooled 13; SRR17876321 (Gmax scn_roots_PRJNA803218) is `non_reference` /
`c2_training_eligible`.  `DATASET_ROLES.est_v1.tsv`: unknown 10, reference 1.  Code: `c0_schema.GENOTYPE_STRATA`
(line 81) and `b6_c0_metadata.GENOTYPES` (68) name `known_nonreference`; `role_row` (old 88) refused anything else.

Spec. A18.5, line 270: "Genotype stratum (reference / known non-reference / hybrid-pooled / unknown) is recorded per
library and applied as the A18.4 multiplier" — the protocol names the stratum in prose; A47.3 (line 1112) and the A46
command spec v2 (line 27) cite the manifest token `non_reference` as data.  Neither text freezes a token; the c0-v1
schema CHECK does, and the schema is frozen.  Decision: keep `known_nonreference` canonical and normalise the manifest
token at ingestion with one explicit, documented alias (`GENOTYPE_ALIASES`, `canonical_genotype`,
`b6_c0_metadata.py` 68–80, `role_row` 98–104).  The manifest was not edited.  The c0-v1 schema is unchanged.

Test: alias → canonical, canonical passes, `nonreference`/empty/`non-ref` refused with the vocabulary in the message;
end to end through `b6_c0_metadata.main` for SRR17876321 the sidecar carries `known_nonreference`, which is in
`c0_schema.GENOTYPE_STRATA`.

## Files changed (absolute line anchors in the current files)

- `revision/scripts/evidence/b6_build_c0.py`: 737–739 (call), 797–831 (`rejected_member_unit`), 1148–1166
  (`verify_hsp_placements`), 1190 (call), 1192–1233 (`load_hsp` protein-state join, causes, QC).
- `revision/scripts/evidence/b6_merge_evidence_db.py`: 6–17 (docstring), 62–100 (`hsp_gate`), 103–162
  (`IDENTITY_INPUTS`, `_provenance`, `build_identity`, `provenance_gate`), 180–190, 232.
- `revision/scripts/evidence/b6_est_molecule_map.py`: 16–36 (docstring), 54–56 (columns), 95–127 (export/report).
- `revision/scripts/evidence/b6_hsp_project.py`: 17–22 (docstring), 84–97 (map contract, `ACCEPTED_RULE`), 212,
  437–439, 510, 535, 597.
- `revision/scripts/evidence/b6_routea_agreement.py`: 27–56 (docstring), 162–197 (`load_protein_state`,
  `protein_drop_reason`), 199–226 (`training_rows`), 228–244 (`load_route_a`), 353–357 (`run`), 404–405 (CLI).
- `revision/scripts/evidence/b6_c0_metadata.py`: 68–80, 98–104.
- Tests: `tests/test_b6_c0_world.py` (`PACBIO_*`, `write_pacbio`), `tests/test_b6_c0_build.py` (3 new tests at the
  end; `test_hsp_inherits_...` interval), `tests/test_b6_c0_merge.py` (3 new tests; orphan message), `tests/test_b6_routea_agreement.py`
  (fixture builds a protein C0; taxid-prefixed query names; `--protein-c0`; 2 new tests), `tests/test_b6_hsp_project.py`
  (`accepted` in fixture/MAP_ROW/invalid rows; 1 new test), `tests/test_b6_c0_molecule_map.py` (`accepted` assertions),
  `tests/test_b6_c0_metadata.py` (1 new test).
- Not touched: `src/transgenic/evidence/c0_schema.py`, `leakage.py`, `est_phase_project.py`, `b6_leakage_intervals.py`,
  `b6_audit_runs.py`, the manifests, any MVP file.

Conventions kept: every new reader has an `ORDER BY`; all new checks are set-based SQL over the staged/attached
tables (no whole-table Python lists); new QC counters are added, none repurposed; c0-v1 schema unchanged (the
molecule-map TSV contract gained one column, which `b6_hsp_project` refuses to run without).

## Test results

Command (from `/tmp/b6-wp1-20260922`, `PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp`,
interpreter `/tmp/b6-mvp-verify.iT7cl7/.venv/bin/python -m pytest -q -p no:cacheprovider`):

```
tests/test_b6_*.py                                   -> 169 passed in 735.52s (0:12:15)   (exit 0)
tests/test_mvp_final_source_integration.py::test_reviewed_a_implementation_bytes_unchanged -> 1 passed in 49.78s (exit 0)
```

The 169 include the 12 new/extended tests listed above.  No Slurm job was submitted; nothing was committed, stashed or
checked out; no file outside the worktree was written.

## Open decisions for the author

1. Finding 2b: route-B `protein_id` inherits the miniprot query's rule 4(b) state (rule 3 "one unit everywhere it
   appears").  Implemented fail-closed; confirm or restrict to route A.
2. Finding 1 residual: ONT non-UMI cDNA reads whose primary was rejected keep a per-read unit (no corrected chain);
   confirm that this is acceptable or specify a chain rule for rejected reads (A16 amendment).
3. Finding 4: roles manifests are excluded from the merge identity (only exclusion/corpus inputs + seed); confirm.
4. Finding 5: the alias `non_reference` → `known_nonreference` lives in the metadata producer; if the author prefers
   the manifest to carry the canonical token instead, the alias can be removed after the manifest is regenerated.
