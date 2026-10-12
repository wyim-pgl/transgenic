# MVP CPU evidence implementation plan

> For agentic workers: use subagent-driven-development; implement tests before production code. This plan covers only the first independently testable data increment, not model/trainer changes.

**Goal:** Build the approved exploratory Athaliana C0-lite evidence database and rasterize its eight EST channels.
**Architecture:** A one-pass BAM parser stages member placements in DuckDB, resolves molecule connected components, excludes whole components, and writes train/valid memberships. A CPU-only NumPy rasterizer consumes that database. No new service or general-purpose pipeline framework.
**Tech stack:** Python, pysam, DuckDB, NumPy, pytest. No torch needed at this stage.
**Spec:** `revision/protocols/B6_MVP_PILOT_v1.md`, approval SHA256 734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74.

## Global constraints

- Work only in the isolated agent worktree. Main contains uncommitted approved spec/panels: copy explicitly from the original checkout; do not modify original files.
- No remote submissions, production data builds, model/dataset/trainer changes, merge or push in this increment.
- Panel locus coordinates are 1-based inclusive; geneList start/fin and tile_blocks are 0-based half-open; junction endpoints are 0-based inclusive.
- Exact primary-placement dedup key: chromosome, FLAG-0x10 alignment strand, POS, CIGAR. Add clone/accession key unions, transitively, before exclusions.
- Genomic transcript strand composes FLAG and ts per A39; missing/unknown ts is unknown. Rule-(a) identity still uses FLAG, not transcript strand.
- Exclude all members' primary/supplementary placement spans overlapping any supplied panel row plus 150 nt, or any Athaliana test tile block. Secondary placements are ignored explicitly (accepted MVP limitation).
- Retained components supply train and valid tracks only. Usable heldout-excluded components supply evaluation input only. No-primary components are never usable evidence.
- Keep K=8, float16 output, exact blake2b thinning encoding/seed and RC donor/acceptor identities from spec; all channels zero on N/n.

## Recorded implementation rulings

User approved MVP and, after the proposed criteria, said “별도 워크트리 사용 진행해”. Apply these conservative definitions, recording them in DB provenance:

- Ruling: exclusion uses all 2,079 supplied panel rows, not only MVP 50; evaluation readouts remain MVP 50. Protects later panels; cost if wrong is reduced MVP evidence and possible inconclusiveness.
- Ruling: dedup components retain all member-placement provenance; union primary blocks and distinct junctions so one molecule counts at most once per base/junction. Supplementaries participate only in exclusion. Conflicting known transcript strands resolve to unknown and are counted in QC. Cost if wrong is reduced strand-specific signal, not silent arbitrary strand selection.
- Ruling: use full reference placement span including introns for leakage exclusion; blocks for membership/coverage. Cost is conservative extra exclusion.
- Ruling: preserve both panel/test overlap flags; usable primary-bearing component hitting panel gets heldout_locus cause precedence. Disposition of no-primary groups is established first and cannot be overridden by overlap causes.

## Task 1: C0-lite builder and rasterizer vertical slice

Files:
- Create `revision/scripts/mvp/mvp_build_c0lite.py`.
- Create `src/transgenic/evidence/mvp_tracks.py` (minimal package init only if needed).
- Create `revision/scripts/tests/test_mvp_c0lite.py`, `tests/test_mvp_tracks.py`.

Interfaces:
```python
build_c0lite(*, bam, panels, corpus_db, out, flank=150) -> dict
main(argv=None) -> int
rasterise_est(con, *, sequence, chrom, offset, L, is_rc,
              keep_fraction, aug_seed, rn=None,
              population="retained", source_supplied=True) -> numpy.ndarray
```
`sequence` is already output-oriented, offset is genomic window start, length must equal L. Population accepts retained/excluded_heldout only. rn uses tile_est; no rn queries explicit window geometry. Source dropped or evaluation fraction zero gives all zeros; supplied empty source has availability on non-N bases.

- [ ] Copy approved untracked inputs and current B6 protocol references to worktree, log their hashes and base HEAD. Do not copy .git, large data, or unrelated edits. Separate documentation baseline from implementation commits if committing.
- [ ] Establish a test interpreter without modifying shared Python: Python3.8 has pytest/numpy/duckdb but no pysam, Python3.10 has numpy/duckdb/pysam but no pytest. Prefer isolated venv with system packages plus pytest. Record baseline panel tests before implementation.
- [ ] Add failing synthetic tests for connected-component dedup, late supplementary leakage, FLAG/ts, coordinate boundaries, member union and population separation. Demonstrate intended failure before implementation.
- [ ] Implement minimal builder: validate source schemas/panels/species; stream BAM until EOF including unmapped; stage in bounded batches; derive deterministic IDs from sorted member names; apply exclusion after full union closure; retain member geometry/provenance; write the five specified public tables plus minimal provenance/helper tables needed for multi-placement fidelity.
- [ ] Populate distinct tile_est for both train and valid Athaliana tiles, including RC rows; independent final QC must find zero retained protected-span overlaps. Refuse output/input collisions and overwrite; publish output only after successful QC, leaving no successful-looking partial DB.
- [ ] Add failing raster tests, then implement exact molecular counts, half-open coverage difference arrays including D, inclusive intron donor/acceptor endpoints, separate block/junction queries, deterministic nested thinning, RC permutation [1,0,2,5,6,3,4,7], final N-zero mask. Coverage clips log1p(64); junction channels have no extra clipping.
- [ ] Run tiny BAM+corpus integration through DB to tracks, a 129024-base CPU timing smoke, targeted new tests and existing panel/evidence regressions. Record actual output, skipped coverage and timing; do not claim production readiness based only on fixtures.
- [ ] Write concise implementation report with files/commits, red/green commands and outputs, exact schema/interfaces, unresolved issues. Controller reviews actual diff before any production use.

## Required executable test cases

1. A/B share placement; B/C share clone key; C late supplementary overlaps panel: one excluded component regardless of input order.
2. Same endpoints but different CIGAR do not collapse; FLAG/chrom differences do not collapse; ts-only difference does not alter exact-placement key.
3. Version/.5/.3/_F/_R clone examples merge; unrelated names do not.
4. Panel [101,200] becomes [100,200), flanked [0,350); half-open placement starting350 does not overlap; ending350 with start<350 does. Protected locus wholly in N span excludes component.
5. M/=/X/D consume covered reference, N splits coverage with endpoint p+n-1, I/S/H do not consume reference.
6. All FLAG/ts combinations, especially FLAG16 ts+ gives minus; unknown/conflict coverage has no directional junctions.
7. Train+valid+RC joins present, test/other species absent; duplicate blocks yield distinct membership; intron-only overlap does not give membership.
8. Minus intron[100,199] gives donor199 acceptor100; RC swaps strand only, donor at L-1-199; test nonzero genomic offset.
9. Overlapping blocks/repeated junctions from same molecule count once.
10. Hash matches exact spec, subsets nested and stable across order/tiles.
11. Supplied empty versus dropped availability; N/n and RC masking all8 channels; retained/excluded inputs never mix.

## Preflight consistency ledger

| Boundary | Check | Resolution |
|---|---|---|
| Builder → raster | molecule IDs, geometry, strand, population | Use public tables plus provenance/helper fields for multi-placement components; document schema before raster code |
| Builder tests → spec | coordinate formats, whole-component exclusion | Conversions explicit; all supplied panel rows protected |
| Raster tests → spec | K8, RC, mask, nested hash | Exact values fixed above, no phase channels |
| Data increment → model increment | no torch or trainer dependency | Return NumPy array only; later adapters own tensors/collation |
