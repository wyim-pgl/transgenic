# EST phase geometry fix 1 — independent scoped re-review

Reviewed detached `5c46c7547e2ade0e36a8aab914718dd0a0e5768f` against `66b9c01`. No source edits.

## Verdict

- **P2 ADDRESSED.** No new critical or important finding in the scoped fix.
- **Specification covered within the requested pure-geometry scope.** The previous external-spec access condition is closed: I read the complete neutral `EST_PHASE_DESIGN_20260916_1530.md`, authoritative `B6_EVIDENCE_MODE_WORKPLAN_v1.md` (the actual supplied filename), prior independent review, fix diff, and both committed geometry reports. This is not approval of the full design's downstream production pipeline.

## Fix verification

`revision/scripts/evidence/est_phase_project.py:188–199` uses a fresh private object for exhaustion on both reads. Empty input still returns `no_observations`; a first real `None` raises `ValueError`; any real second entry, including `None`, refuses all segments with `multi_hsp_policy_unresolved`. No third entry is read. Single-valid input still calls the same projector. The source diff changes only this boundary, not coordinate arithmetic or policy.

The six regressions at `revision/scripts/tests/test_est_phase_project.py:130–161` cover both first-None cases, the original valid/None/valid bypass, empty/single-generator preservation, and a generator that raises on a third read for either kind of second entry. No scientific multi-HSP selection rule is introduced.

## External design comparison

Mapper and projection inspected directly: `revision/scripts/evidence/est_phase_project.py:26–103,106–185`. Test paths below are relative to `revision/scripts/tests/test_est_phase_project.py`.

| Design fixture | Exact comparison and evidence |
|---|---|
| F1/F2 | Tests `:31–51` use positive introns 3/4/5. Plus low/high labels are `0120` / `12012`; minus labels are `2102` / `10210`, equivalent to local channels `5435` / `43543`. Plus anchors are `1 → (1+n)%3`; minus ascending-coordinate anchors are `0,(n-3)%3`, hence coding-order change is `−n mod3`. Split tuples `(103,104+n,105+n)` and `(105+n,104+n,103)` retain `012`. Matches the design exactly; no intron increment is added to codon ordinal. |
| F3 | The same parameterization exercises all four exact FLAG/frame/endpoints from the design. Strand equals FLAG16 XOR HSP sign. The committed ambiguity test `:164–175` covers one conflicting/missing-ts case; this review additionally checked all four combinations for opposite-ts refusal and missing-ts preservation of all labels. |
| F4 | Tests `:54–72,178–197` match I/D/P expected positions: insertion q3 is unpainted, deletion g103 absent, padding advances neither cursor. M/=/X map identically. Full-query H cursor reproduces F1; reverse BAM SEQ validation uses RC before removing terminal H, without double complementation. Actual reflected I/D/P CIGARs preserve c and reverse strand. |
| F7 | Tests `:80–90` match tile intersection `{102:2,103:0,108:1,109:2}`, include g112/exclude g113, retain one-base c2 support, and explicitly diagnose wholly clipped events. Source interval `[11,22]` gives original zero-based12 while remaining genomic100. Full fixture equality excludes intron/boundary painting. Split-part deduplication is deliberately downstream, not asserted by these tests. |
| F8 | Tests `:93–104` match local RC positions `1..5,10..13`, with ordinals `21021,0210` on minus, i.e. channels `54354,3543`. They verify dictionary involution, an actual reverse-placement/reversed-CIGAR projection, and a one-base genome translation with unchanged c and shifted anchor. No c→2−c relabelling occurs. Tile origins cannot affect this API because it outputs genomic segments without a tile-origin input. |

The interval arithmetic agrees with design §2: reverse full-query interval `[lo,hi)` becomes `[L-hi,L-lo)`; at a genomic segment start, original q is `L-1-u` on FLAG16. Ordinal is `(event_c+(q-event_q)*direction)%3`, genomic step is direction times placement orientation, and anchor `(g-step*c)%3` gives exactly the specified plus/minus formulas. Source-accession offset is kept out of the genomic transform.

## Scope limits retained, not defects in this review

F8's float16 byte-identical raster replay is **not implemented or claimed**: tests establish pure ordinal/strand geometry, not a production raster. Likewise F7 molecule deduplication, F5 authentic traceback semantics, F6 conflict/count policies, route-A phase conversion, A/B multi-support masking, provenance authenticity, exclusion closure, and nine-species execution remain outside this task. The full design/workplan requires those later; its complete production requirements are not satisfied by this diagnostic library. The manifest remains `training_eligible: false`; six indices are local, not final B6 K-axis indices. Supplied producer labels, not frame magnitude or annotation, determine c. Unresolved overlap/strand/nonprimary/multi-HSP inputs do not silently choose a scientific policy.

## Executed checks

- `python -m pytest revision/scripts/tests/test_est_phase_project.py -q`: **57 passed in 0.18s**.
- Additional read-only F3 matrix: all four FLAG/HSP combinations preserved labels with missing ts and refused opposite ts; passed.
- `git diff 66b9c01..HEAD -- revision/scripts/evidence/est_phase_project.py`: verified only the sentinel/None boundary changed.
- `git diff --check`: passed before report creation; report also checked before commit.

The controller's 176-test/6.29s run was not repeated or independently claimed. Graft was queried before source reads; its generated untracked graph is excluded from the report commit. Advisor consultation was attempted but failed on provider/client version incompatibility; no advisor review is claimed. No SSH, jobs, subagents, merge, push, or main-worktree work. Only this report is committed for controller extraction.
