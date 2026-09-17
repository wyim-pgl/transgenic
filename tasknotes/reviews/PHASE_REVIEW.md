# Independent phase-conflict review — 2026-09-17

## Verdict: scoped APPROVE

No reproducible Important/Critical defect found in the bounded, prevalidated-observation conflict API. This approves neither exact-geometry/source adapters nor production execution, source completeness, full A11, A46, or training readiness.

Reviewed checkout `/tmp/est-approved-contract-20260917`, verified HEAD `c9a351431b7dd8fc8ad45be35a9df29ca8445f81`. New implementation is uncommitted relative to that base; approval binds these contents, not unrelated concurrent workers:

| File | SHA256 |
|---|---|
| `revision/scripts/evidence/est_phase_conflicts.py` | `2765f03d885df55031a142c8a7a045064889de80e8c485325a2c54c0a598b9c8` |
| `tests/test_est_phase_conflicts.py` | `735f42a9975d32e2ba296008ac49788d2b4f2427b164c697494b6d30fe7a4d73` |
| `tasknotes/phase_conflicts_report_20260917.md` | `10e1987e2c50562693c02a450ee05915ade70bce90d7c2dea67509995aa82035` |

Authority read: main `notes/APPROVED_HSP_PHASE_A11_B5_20260917.md` and `notes/EST_PHASE_DESIGN_20260916_1530.md` §3. Current approval takes precedence over historical open-choice language: conflicting evidence only is removed; independent agreement survives. Within-member inconsistency still rejects both entire directly participating HSPs.

## Actual verification

From the reviewed checkout, with `PYTHONDONTWRITEBYTECODE=1`:

- `python -m pytest -p no:cacheprovider -q tests/test_est_phase_conflicts.py revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py`: **191 passed in 6.07s** (earlier independent run also 191 passed). Focused conflict suite contains 37 cases.
- `PYTHONPATH=/tmp/est-approved-contract-20260917 python /mnt/f/scratch/develop/Transgenic/transgenic_local/notes/approved_execution_20260917/phase_review_evidence/adversarial.py`: **6 targeted scenarios plus 1,000 deterministic randomized oracle cases**, each repeated after shuffle and duplicate injection, passed.
- Oracle independently enumerates observation pairs rather than using implementation phase-set groups. Checks counts, observation reasons, whole-HSP summaries, molecule/base summaries, and false training eligibility.

Durable evidence: [adversarial script](phase_review_evidence/adversarial.py), [adversarial output](phase_review_evidence/adversarial.log), [pytest output](phase_review_evidence/pytest.log).

## Reviewed behavior and trace scope

| Area | Assessment |
|---|---|
| Identity/domain partition | Species/reference partition all identities; contig/base partition local comparisons; protein sequence digest partitions A/B only. Member-to-molecule and full-HSP member/molecule/protein inconsistencies raise. Multiple member placements are explicitly unresolved. |
| HSP rejection | `est_phase_conflicts.py:122–149`: collects all direct participants before any removal. Whole-HSP flanks, self-conflicts and chained conflicts are rejected simultaneously; nonoverlapping unrelated HSPs survive. Strand disagreement counts within a member/placement. |
| A/B masking | `:151–163`: same base/strand/normalized protein; all disagreeing pairs evaluated simultaneously. Mask is observation/base-local, not whole-HSP or final-cell veto. Opposite A/B strands do not compare. Independent different-protein support survives, including support on the same molecule after another observation is masked. |
| Molecules/counts | `:165–181`: only post-pair retained B observations determine molecule/base ambiguity. Incompatible surviving member labels suppress that molecule/base, not others. Compatible members/proteins/HSPs/source observations deduplicate by molecule; distinct molecules remain distinct and no 64 cap is applied. Opposite-strand B on distinct molecules can both count. |
| Absence and validation | `:66–93,116–121`: zero is a genuine ordinal. Missing observation is absence; None/bool/invalid phase, invalid coordinate/type/hash/ID and A records carrying B identity fail closed. Attestations require literal True. Empty scope gives empty counts, not synthetic phase zero. |
| Ordering | Semantic mappings and reason sets are invariant to input shuffle/duplicates in tests and oracle. Mapping iteration/serialization order is not promised canonical; consumers needing byte-identical artifacts must sort explicitly. No evidence suppression depends on iteration order. |
| Returned trace | `:49–63,175–188`: observation keys preserve base/domain/source/HSP/member/molecule identity and reasons. HSP and molecule maps intentionally UNION reasons: a partially retained HSP can contain both AB_conflict and retained. Such a summary is not an all-or-nothing eligibility flag; only retained B observations contribute counts. A_agrees is phase-only, not subject-residue correspondence. |

## Deliberate caller-attestation boundary, not API defects

The module explicitly consumes already accepted, geometry-validated per-base labels and canonical, namespaced IDs. It validates digest syntax and limited identity consistency, not digest preimages, primary-alignment truth, contig bounds, query/strand/intron/frameshift replay, biological molecule closure, or external route-A availability. Complete-HSP/all-relevant-member scope and complete A crosschecks remain caller responsibilities; boolean attestations are not independent completeness proofs. Whole-HSP conflict resolution must precede genomic slicing. These limits are clearly stated in both implementation and task report, so they are not misreported as newly discovered bugs.

Per-observation reasons provide bounded diagnostic trace, not a production conflict-edge/interval receipt or source-specific raster. Same-codon correspondence and source-separated downstream reporting are outside this library. Returned training_eligible remains false. The task report's preparatory scope is consistent with code; its historical RED log was not independently reconstructed during this readonly review.

## Operational note

No source edits, index/HEAD changes, commits, agents, remote access or jobs were performed. Only this report and its own evidence directory were explicitly written. Graft was queried before source inspection; `graft map` automatically reported refreshing its graph for two changed files (not a requested source edit). Advisor attempt failed with client/model-version HTTP 400; no advisor guidance or approval is claimed. Graft reported approximately 878,819 tokens saved across context calls.
