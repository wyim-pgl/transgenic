# Independent interface review — 2026-09-17

## Verdict: scoped APPROVE

No Important/Critical **cross-module** finding in the bounded increment reviewed at `/tmp/est-approved-contract-20260917`, HEAD/base `c9a351431b7dd8fc8ad45be35a9df29ca8445f81`. Approval binds the file hashes below. It is not full-pipeline, production A46, full C0, nine-species database, source-authority, biological or training certification.

Read the current approval record, HSP_REVIEW/PHASE_REVIEW/A11_REVIEW, all three new module interfaces and implementations, the complete controller integration test, and the relevant existing geometry interface (`est_phase_project.py:1–185`). This review did not repeat three whole-module audits. The earlier A11 review's Important finding concerns the pre-fix hash: the present guard at `est_molecule_closure.py:186–190` rejects conflicting resolved canonical libraries for an accession stem within species/assembly/arm, including across datasets. Reviewed its regression cases at `revision/scripts/tests/test_est_molecule_closure.py:90–143`; they pass in the combined run. Separate A11 re-review remains its own report, not implied by this interface verdict.

## Interface findings and limits

- **HSP → geometry:** `est_hsp_contract.py:31–73` emits original-screened-query coordinates, explicit direction and local codon ordinal zero, matching `project_codons`' labelled-run convention. D consumes subject without inventing query bases; I emits a real translated query codon with no subject index. Whole-HSP col9=M+I+D≥30 is not replaced by query residue, paired residue, projected segment or per-frame-run thresholds. A 30-column D case legitimately has 29 events/87 query bases; a 15+15 frameshift case still has one accepted HSP, not two rejected short HSPs.
- **Frameshift/indel scope:** direct forward/reverse API probes confirmed D30→87 mapped bases, I30→90, +1→90 with the skipped base absent, and −1→zero emitted segments plus `self_overlap_policy_unresolved`. Geometry always remains training-ineligible. A −1 accepted length result is therefore not silently promoted to resolved geometry. DIAMOND I is not the same as a BAM CIGAR I: only mapped CIGAR runs project; insertion/clipping/gap dispositions and unresolved diagnostics must remain visible to the next consumer. Neither event labels nor geometric projection certify subject-codon correspondence.
- **Geometry → conflict resolution:** `est_phase_conflicts.py:1–24,104–188` explicitly requires already validated, complete accepted HSP observations, primary/unambiguous placement and all relevant molecule members. Resolve complete scopes before genomic slicing. HSP rejection uses a full-HSP identity across flanks, not one ID per codon/frame segment. A/B removal is local to disagreeing evidence at matching base/strand/protein digest; independent support survives. Molecule/base dedup happens after conflict removal. Summary reason unions are not whole-HSP retention flags.
- **A11 → molecule identity:** closure is scoped to species/assembly/context, preserving all-member placements for leakage while exposing primary and min121 separately. Only ready primary support members may feed the corresponding primary diagnostic; sensitivity members must not be pooled as extra primary evidence. Component IDs are sorted member tuples, not frozen globally canonical hashes. The controller's `':'.join(component['id'])` is harmless for its one-member literal fixture but is **not a production encoding**: delimiters and scope collisions require unambiguous serialization and namespace binding. Assembly must be bound to the phase reference digest; member, placement, source and arm/context identity must be consistently mapped. These are explicit next-consumer obligations, not newly discovered implementation defects.

## Four-case controller test

`tests/test_est_approved_contract_integration.py:17–63` parameterizes qframe ±1 × FLAG 0/16. All four run and pass. The 120-nt, 40-codon HSP passes the length rule; `5M7N115M` yields genomic blocks `[100,105)` and `[112,227)`, including a codon split at the splice. Expected strand is query direction XOR reverse FLAG; expected phase is `u % 3` forward or `(119-u) % 3` reverse. The expected dictionary is independently constructed from aligned-base offset, not copied from projected segments, and tests all 120 bases with exactly one molecule each. `ts` is supplied in the existing projector's original-query convention. A11 accepts the 120-nt primary member without imposing the separate min121 sensitivity threshold. Final false training eligibility and A_missing are asserted.

This is a correct **synthetic happy-path API connection**, not a production controller. It has no A records, conflict pair, multiple-member molecule, alternate placement, actual BAM, frameshift or real source binding. Those behaviors have focused component coverage; this four-case test must not be described as their end-to-end certification. In particular its raw13-only digest and hash of the length-contract string are synthetic identifiers, not the full source/search/member/protein provenance identity required by the conflict API. The explicitly missing production binder is not a new bug in this bounded increment.

## Fresh verification

Executed from `/tmp/est-approved-contract-20260917`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/mvp-final-source-env/bin/python3.12 -m pytest -p no:cacheprovider -q \
  tests/test_est_hsp_contract.py tests/test_est_phase_conflicts.py \
  revision/scripts/tests/test_est_molecule_closure.py tests/test_est_approved_contract_integration.py \
  revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_est_hsp_expanded.py \
  revision/scripts/tests/test_est_blastx_fixture.py revision/scripts/tests/test_evidence_support.py \
  revision/scripts/tests/test_univec_trim.py
```

**278 passed in 1.15s.** Python 3.8 was not used. Additional in-memory `python3.12 -c` interface probes used `runpy.run_path('tests/test_est_hsp_contract.py')['fixture']` for D30, I30, 15+1+15 and 15−1+15 in both directions, then `classify_rich17` → `project_codons` with full-length `nM` geometry: all eight mapped-base/reason/training-ineligibility assertions passed (results above). These probes deliberately reuse fixture construction; no independent biological validation is claimed.

```sh
git diff --check
git diff c9a3514 --name-status
git diff c9a3514 --exit-code -- revision/scripts/evidence/est_phase_project.py \
  revision/scripts/evidence/est_hsp_expanded.py revision/scripts/evidence/est_phase_accept.py src train scripts
git diff --cached --exit-code
git rev-parse HEAD
sha256sum revision/scripts/evidence/est_{hsp_contract,phase_conflicts,molecule_closure,phase_project,hsp_expanded,phase_accept}.py \
  tests/test_est_{hsp_contract,phase_conflicts,approved_contract_integration}.py \
  revision/scripts/tests/test_est_molecule_closure.py
```

Tracked baseline diff contains only the pre-existing `.gitignore` addition for `/graft/`; cached diff is empty and HEAD remains the base. Existing tracked source/model/data and pre-existing integration dependencies are unchanged against that base. New source/test files are untracked, alongside tasknotes, `.ignore` and local graph housekeeping; they must not be overlooked by relying only on `git diff`. No source, git index or HEAD was changed by this review; no engine rerun, remote job or subagent was launched. Only this report was explicitly written.

The existing controller replay JSON reports **144 exact raw13→rich17 rows, 126 accepted /18 below threshold**, consistent with the independent HSP review. A fresh Python SHA256 comparison of all five paths in its `input_sha256` against `fixture_root` passed. Thus those current fixture bytes match the recorded replay; this review did not rerun the engine or claim independent biological certification. Historical 132 ELICITED/12 NOT_ELICITED case status is not reclassified by these row counts.

## Reviewed SHA256

| Path | SHA256 |
|---|---|
| `revision/scripts/evidence/est_hsp_contract.py` | `fcb8a4b2413f12dc5c5b0c4d55676cedeac21b3fcb26cf849c089657650a3954` |
| `revision/scripts/evidence/est_phase_conflicts.py` | `2765f03d885df55031a142c8a7a045064889de80e8c485325a2c54c0a598b9c8` |
| `revision/scripts/evidence/est_molecule_closure.py` | `fd05c3ca0ef674b10962efac88861131aa3f54fbb1fa5f7e196c335740dc67b5` |
| `tests/test_est_hsp_contract.py` | `8d7cc3f0901c16569319a85baac3708d965f3f0c650d422b4684e9208ce3b39d` |
| `tests/test_est_phase_conflicts.py` | `735f42a9975d32e2ba296008ac49788d2b4f2427b164c697494b6d30fe7a4d73` |
| `revision/scripts/tests/test_est_molecule_closure.py` | `cc750e8fc6b54714235492fce6d34f0adb845fdba5e141f7e67533897f78febd` |
| `tests/test_est_approved_contract_integration.py` | `3eb1f36ffaa3928ba4d30fd82553696e2b06356f9bae09d9f93cc9cdc8ef8c6f` |
| `revision/scripts/evidence/est_phase_project.py` | `6b977fabf3b3401e162e8d8b800d215d34dbe978f54b364429cb2056a60fec4b` |
| `revision/scripts/evidence/est_hsp_expanded.py` | `aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4` |
| `revision/scripts/evidence/est_phase_accept.py` | `0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3` |

## Next-consumer requirements, not expanded approval

Bind actual source bytes/FASTA IDs, screened intervals and normalized protein identities; retain raw13/rich17 identities distinctly and construct complete full-HSP provenance digests. Bind reference/assembly and primary placement geometry; handle unresolved/self-overlap and unmapped event bases explicitly. Obtain complete HSP/member/A inventories before slicing and preserve evidence-local rejection. Encode scoped A11 component identities without collisions, carry all-member leakage exclusions while keeping support arms separate, and do not interpret readiness booleans or `training_support_members` as training authorization. None of this authorizes production A46/full-nine-species execution or modifies model/training behavior.

Graft was queried before source inspection; reported savings total approximately 887,279 tokens. One incorrectly shaped `graft grep` invocation failed without a graph and was corrected; no source/index change resulted.
