# Independent geometry review

Reviewed detached commit `66b9c01` (implementation `46ed7150035d955ccea7269c0148e798770204aa`), full Git diff against `5c7a0aa`, including all 188 test lines and the complete committed `notes/EST_PHASE_GEOMETRY_REPORT.md`. Built/queried the graph in the isolated worktree before inspecting source.

## Verdicts

- **Specification: conditional; full external-spec verification incomplete.** The implementation matches the diagnostic-only coordinate contract described in the request and committed report, except the malformed-observation refusal defect below. No geometry off-by-one or reverse/complement error identified. The requested main-working-copy `geometry.review.diff`, `geometry_report.md`, and `EST_PHASE_DESIGN_20260916_1530.md` are absent from this isolated checkout. The higher-priority isolation instruction prohibits working in main, so these were not read and no claim of independent verification against their exact F1–F4/F7–F8 wording is made. The full exact Git diff and committed report were reviewed instead.
- **Code quality: changes requested, one medium-severity input-boundary defect.** Compact affine implementation is otherwise appropriately small and standard-library-only. Production remains blocked; the finding cannot make its manifest training-eligible.

## Finding

### P2 — `None` observations masquerade as iterator exhaustion and bypass refusal

**Location:** `revision/scripts/evidence/est_phase_project.py:191–196` (`project_observations`).

Both iterator reads use `None` as an exhaustion sentinel. An actual malformed `None` entry therefore gets silently accepted as absence rather than rejected. Most importantly, `[valid_observation, None, valid_observation]` emits the first HSP's segments with no unresolved reason and never inspects the later HSP. This violates the stated boundary that all multi-HSP input is refused rather than silently selecting a subset. `[None, valid_observation]` is incorrectly reported as `no_observations`.

Focused reproduction, executed at the reviewed commit (no full-suite rerun):

```python
import runpy
t = runpy.run_path('revision/scripts/tests/test_est_phase_project.py')
m = t['m']
p, h, e = t['inputs']()
o = (h, e, p)
for items in ([o, o], [o, None, o], [None, o]):
    r = m.project_observations(items)
    print(r['reasons'], len(r['segments']))
```

Observed:

```text
['multi_hsp_policy_unresolved'] 0
['motif_strand_available'] 2
['no_observations'] 0
```

Expected: malformed input raises or returns an explicit refusal with no segments; a real second element must not be treated as exhaustion. Use a unique sentinel or handle `StopIteration` explicitly, and regress both malformed positions. This does not require choosing a scientific conflict policy.

## Checks with no finding

- **Identity and CIGAR (`:26–96`):** full original screened query is required, uppercase sequence hash and source interval length are validated, HSP identity must match placement identity. Optional BAM-oriented SEQ is checked after reverse complementation and removal of terminal hard clips. H advances the full-query cursor but not stored SEQ; S/I consume query only, D/N reference only, P neither, M/=/X both. Terminal clipping order, positive operation lengths, mapped-base presence, full query length, POS and reference overrun are checked. Reference dictionary authenticity is explicitly a caller responsibility.
- **Projection (`:106–185`):** one-based inclusive HSP bounds convert correctly to zero-based query coordinates; reverse placement uses `[L-hi,L-lo)` and `q=L-1-u`. Ordinal at the left genomic segment endpoint is `(c+(q0-q)*direction)%3`, with genomic step equal to HSP direction times placement orientation. Half-open segment lengths and source offset separation are correct.
- **Splices and indels:** N/D do not advance query ordinals or paint reference gaps. Producer ordinal carry across N is preserved; genomic anchors change with the intron length modulo three. I advances query ordinal and remains unpainted. No double intron correction found. Arbitrary supplied discontinuities remain separate rather than being extrapolated from signed frame.
- **Scope and ambiguity:** manifest declares `c0_first_codon_base`, six channel indices and `training_eligible: false`; documentation identifies them as local channels. Producer SHA is a reference, not verified authenticity. Self-overlap, read-relative ts/HSP conflict and secondary/supplementary placement explicitly refuse segments; normal multi-observation input refuses all HSPs. Missing ts does not erase protein-derived direction. Frame-magnitude acceptance and authentic HSP semantics remain upstream rather than silently invented here.
- **Compactness:** mapping allocates per CIGAR operation; projection allocates per event/run intersection, with sorted events and binary search into mapped runs. No per-genomic-base Python raster allocation. Sequence normalization, hashing and optional reverse-complement validation remain linear in query sequence length, as documented.
- **Deferred scope respected:** no route-A adapter, molecule aggregation, scientific phase-conflict policy, raster, CLI or training export added. F8 tests exercise geometric reflection, not float16 byte replay, and the report says so explicitly.

## Verification limits

The controller's reported 170-test run was not duplicated. The only executed behavioral check was the focused reproduction above; it confirmed the finding. No code edits, SSH, jobs, subagents, merges or pushes. Advisor consultation failed because its provider rejected the client version; no advisor review is claimed. This report is written only in the isolated worktree for controller transfer to the requested main notes location.
