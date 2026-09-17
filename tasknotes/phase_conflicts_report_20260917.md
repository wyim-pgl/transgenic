# Bounded phase-conflict library — 2026-09-17

## Scope / files

Base verified: `c9a3514` in `/tmp/est-approved-contract-20260917`.
No commit made. This task created only:

- `revision/scripts/evidence/est_phase_conflicts.py`
- `tests/test_est_phase_conflicts.py`
- `tasknotes/phase_conflicts_report_20260917.md`

Existing projector/scalar reconciliation, acceptance gates, MVP eight channels,
shared TODO/resume and other agents' files were not edited. No jobs, remote
access, source integration, datasets, or genome raster materialization.

Authority: main `notes/APPROVED_HSP_PHASE_A11_B5_20260917.md` and
`notes/EST_PHASE_DESIGN_20260916_1530.md` §3. Graft queried before source
inspection; `reconcile_phases` indexed span is
`revision/scripts/evidence/est_phase_project.py:213–224`. Its scalar semantics
remain unchanged. No need to import it for multi-observation set operations.

## Implemented contract

Frozen `Observation` records; pure `resolve_conflicts` function. Explicit
`complete_hsps=True` and `primary_unambiguous=True` attestations are mandatory.
All grouping domains include species and reference SHA256, plus contig/base
where relevant. Protein joins use normalized sequence SHA256, not accessions.
B carries full-HSP SHA256, member, placement, molecule and source identity.
Full-HSP digest means upstream identity/provenance of the entire HSP, not a
segment/window digest. Hash preimages are not recomputed here.

1. Find all B label disagreements at shared base on the same member/placement,
   ignoring protein identity for this within-member consistency check. Collect
   the whole-HSP rejection set before removal. Reject every direct participant,
   including self-conflicts, without score winners or iterative survivor bias.
2. On remaining observations compare A/B at exact base/strand/protein domain.
   Simultaneously neutralize only observations participating in unequal-phase
   pairs; no final-cell veto. Another protein/molecule's support survives.
   Opposite strands are not A/B-comparable. A is only a crosscheck.
3. Among remaining B observations, incompatible labels across members of one
   molecule make only that molecule/base unresolved. Other molecules survive.
4. Count distinct retained molecule IDs at each base/strand/c0, deduplicating
   HSPs, proteins, members and repeated source observations. Counts are sparse
   Python integers; no log/float16 conversion and no coverage cap of 64.

Return maps contain per-observation reason sets and HSP/molecule-base reason
unions, including partial retention. Exact duplicates collapse. Real c0=0 is
support, not absence. `A_missing` means no comparable A in the supplied input,
not an externally certified absence. `A_agrees` is only phase agreement, not
same-codon/residue agreement. `training_eligible` is always false on returned
results.

Malformed types, bool phases/positions, malformed hashes/IDs and conflicting
identity reuse fail closed with ValueError. Multiple placement IDs for the
same member are explicitly `placement_unresolved` and do not count; no new
MAPQ threshold or supplementary-placement policy is selected.

## Completeness and unresolved boundaries

The caller must provide every projected base of every accepted HSP competing
within an included member/placement, and relevant members of included
molecules. Whole-HSP rejection must precede genomic slicing: a conflict outside
a requested tile can invalidate evidence inside it. Completeness flags are
caller assertions, NOT a proof that a truncated window is complete. Bounded
memory is proportional to the supplied scope, not genome size; callers must
partition by complete conflict domains rather than arbitrary tiles.

Validated geometry, canonical upstream identities, primary/unambiguous
placement selection, molecule closure and route-A completeness are caller
responsibilities. This library does not parse route A, validate subject-residue
correspondence, certify full A11, or approve global A46/training. Existing
geometric projection remains diagnostic/preparatory, not production ready.

## Verification (actual runs)

- RED before implementation: `python -m pytest -q tests/test_est_phase_conflicts.py`
  -> **37 failed in 0.46s**, all at the explicit missing-library assertion.
  Full tool log: `/tmp/pi-bash-93283a6654bb71d2.log` (ephemeral).
- GREEN after implementation: same command -> **37 passed in 0.09s**.
- Regression run:
  `python -m pytest -q tests/test_est_phase_conflicts.py revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py`
  -> **191 passed in 6.59s**.
- `git diff --check` -> exit 0. New files reviewed directly (untracked files
  are not covered by ordinary git diff).

Tests cover chain conflicts collected simultaneously, whole-HSP rejection,
unrelated HSP retention, self-conflicts, opposite strands, pair-local independent
support, absent A/real zero, multilabel A/B permutations, cross-member molecule
ambiguity, duplicate/permutation invariance, uncapped 70-molecule counts,
cross-species/reference/contig domains, invalid records and scope attestations.

Advisor consultation was attempted but unavailable: client/model version
mismatch (HTTP 400). It supplied no guidance or review approval. Independent
controller review remains to be done. No production-readiness claim is made.
