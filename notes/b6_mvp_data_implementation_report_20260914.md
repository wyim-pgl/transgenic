# MVP CPU evidence implementation report — 2026-09-14

**Exploratory / MVP; synthetic CPU verification only. Controller review required before production use.**

## Isolation, inputs, commits

- Verified working directory before changes: `/tmp/pi-agent-611e8432-78e4-474-e54d14c0` (isolated git worktree; initially clean/detached).
- Branch: `mvp-cpu-data-20260914`.
- Base: `9a8c8e1ba68286e845fb99be3cb76112a8bcd186`.
- Approved-input baseline: `5390864bfaa0999810178a38d8705c9bbf1bb780`.
- Implementation: `6e26a705c84a3b914943e8856be8cc6b2ca538ef`.
- This report is a separate subsequent documentation commit. All branch-local commits use author/committer Won Yim `<wyim@unr.edu>`, without coauthor attribution. No merge or push.
- Read-only input source: `/home/wyim/scratch/develop/Transgenic/transgenic_local`. Copied only the brief, approved spec, current English B6 workplan/readme and B1/Protocol M references, three panel/count TSVs and sidecars, scripts 71/72/72a, and panel tests. No source-checkout writes, `.git` copies, unrelated edits, or large data copies.
- Exact copied-path SHA256 ledger: `notes/b6_mvp_input_hashes_20260914.json` (16 entries, independently rechecked). Protocol M was already identical to the base, so has no diff.
- Approved spec SHA256 verified: `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`.
- Supplied panel SHA256: `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`; 2,079 rows: core 20, MVP 30, pilot 300, final 1,729. All rows are the protection universe; evaluation readouts remain the MVP 50.

## Delivered files and interfaces

- `revision/scripts/mvp/mvp_build_c0lite.py`
  - `build_c0lite(*, bam, panels, corpus_db, out, flank=150) -> dict` of integer QC metrics.
  - `main(argv=None) -> int`; CLI: `--bam BAM --panels TSV --db CORPUS --out NEW_DB [--flank 150]`.
  - CLI prints JSON QC on success, error to stderr/nonzero exit on failure. Existing outputs and input/output collisions are refused. Corpus must be a closed/checkpointed snapshot (no WAL).
- `src/transgenic/evidence/mvp_tracks.py`
  - `rasterise_est(con, *, sequence, chrom, offset, L, is_rc, keep_fraction, aug_seed, rn=None, population="retained", source_supplied=True) -> numpy.ndarray`.
  - Output is contiguous float16 `[L,8]`. Sequence is already output-oriented; offset is genomic zero-based; sequence length must equal L.
  - `population`: only `retained` or `excluded_heldout`. `rn` adds `tile_est` membership restriction. Evaluation windows omit `rn`, since heldout-excluded molecules have no train/valid membership.
  - Dropped source or fraction zero returns all zeros including availability. Supplied empty source has availability 1 on non-N bases. N/n masking is applied after RC to all eight channels.
- Tests: `revision/scripts/tests/test_mvp_c0lite.py` and `tests/test_mvp_tracks.py`.
- Existing namespace package needed no added `__init__.py`. Normal `PYTHONPATH=src` import works and does not import torch.

## Persisted schema (reopened read-only and inspected)

Types below: `V=VARCHAR`, `B=BIGINT`, `I=INTEGER`, `Bool=BOOLEAN`. Spans/blocks are zero-based half-open; intron endpoints are zero-based inclusive.

| Public table | Columns in order |
|---|---|
| `est_molecule` | `molecule_id V PRIMARY KEY, members V[], strand V, mapq I, chrom V, start0 B, end0 B, excluded Bool, cause V, overlaps_panel Bool, overlaps_test Bool, primary_count B, strand_conflict Bool` |
| `est_block` | `molecule_id V, chrom V, start0 B, end0 B` |
| `est_junction` | `molecule_id V, chrom V, intron_start0 B, intron_end0 B, tx_strand V, donor0 B, acceptor0 B` |
| `tile_est` | `rn B, molecule_id V`, composite primary key |
| `mvp_qc` | `metric V PRIMARY KEY, value B` |

`cause` is NULL for retained; otherwise `heldout_locus`, `test_block`, `unmapped`, or `supplementary_only`. No-primary components have NULL representative primary span/mapq and no block/junction tracks. Secondary-only members are classified as `unmapped` (no usable primary), explicitly recorded in provenance.

Helper/provenance tables:

- `est_member(member V PRIMARY KEY, clone_key V, molecule_id V)`.
- `est_placement(placement_id B PRIMARY KEY, member V, flag I, kind V, chrom V, alignment_strand V, start0 B, end0 B, cigar V, ts V, tx_strand V, mapq I)`; one row per input alignment record. `kind` is primary/supplementary/secondary/unmapped. Secondary names/flags remain auditable, but their geometry is deliberately NULL/ignored.
- `member_block(placement_id B, chrom V, start0 B, end0 B)` and `member_junction(placement_id B, chrom V, intron_start0 B, intron_end0 B)` retain per-primary-record geometry.
- `protected_span(chrom V, start0 B, end0 B, kind V)` stores flanked panel spans and Athaliana test blocks (`kind=panel/test`).
- `corpus_tile(rn B PRIMARY KEY, chrom V, start0 B, end0 B, split V, is_rc Bool)` stores Athaliana tile metadata, not sequences/labels.
- `mvp_provenance(key V PRIMARY KEY, value V)` stores JSON-encoded values: schema/spec version, input paths/SHA256, flank, all implementation rulings, ID/clone/count policies, batch size and DuckDB/pysam versions.

Temporary connectivity/overlap SQL tables do not persist. `est_molecule.chrom/start0/end0` are **only a representative primary placement**, chosen lexicographically by chrom/start/end/FLAG-strand/CIGAR/member. They must not be used as the entire component geometry. Mapq is maximum primary mapq. Track consumers use the unioned `est_block` and distinct `est_junction`; audits use member placements.

QC metrics: reads_in, secondary_ignored, max_stage_batch_rows, members_in, collapsed_rule_a, collapsed_rule_b, collapsed_total, molecules_after_dedup, excluded_heldout_locus, excluded_test_block, excluded_unmapped, excluded_supplementary_only, retained, strand_conflicts, panel_rows_protected, valid_tiles_total, valid_tiles_with_evidence, retained_overlapping_panel, retained_overlapping_test.

## Policy and implementation checks

- Read actual `59_evidence_support.py::transcript_strand` and B5 schema/tile construction. Reimplemented only the A39 FLAG/ts composition, not the scorer's molecule, cs, mapping-quality or supplementary-exclusion policy.
- One BAM alignment-decoding pass, `fetch(until_eof=True)`, including unmapped records. Raw placement/block/junction buffers flush at 4,096 rows each. Union-find holds unique names, **not all member records/geometry**, in Python memory; SQL stages retain the raw provenance.
- Exact key is chromosome, FLAG alignment strand, start0, CIGAR (start0 equality is equivalent to POS equality). Exact-placement unions precede clone unions and whole-component exclusion. Clone keys strip terminal `.digits` and `_F`/`_R` runs. IDs are SHA256 of UTF-8 compact JSON arrays of sorted unique names; root/input order is irrelevant. Collapse counts are successful name-component union edges, rule (a) first then (b), not raw alignment-record subtraction.
- Known primary strand conflict resolves to `?` and increments QC. One known strand plus unknown tags resolves to that known strand; all-unknown resolves to `?`. Supplementaries affect exclusion only.
- Panel `[101,200]` converts to `[100,200)`, then `[0,350)` at flank 150. Exclusion uses complete primary/supplementary spans including introns. No-primary disposition comes first; panel cause then takes priority over test cause; both overlap flags remain stored.
- Blocks consume M/=/X/D; N creates inclusive intron endpoints and splits coverage; I/S/H/P do not advance reference. Interval union handles nested/repeated member blocks. Distinct junctions use resolved component strand; unknown strand has NULL donor/acceptor.
- Membership is distinct retained block overlap with both train and valid Athaliana tiles, including RC. Test/other-species/intron-only membership is absent.
- Independent final QC recomputes retained protected-span overlap from raw placements, member mapping and protected intervals—not cached overlap flags. Injected wrong disposition is detected and publication prevented. A closed/checkpointed sibling `.partial-*` DB is published with an atomic no-clobber hard link only after QC.
- Raster queries blocks and junction endpoints independently. Defensive block union and per-molecule boundary dedup prevent duplicate counts. Coverage uses difference arrays and log1p capped at log1p(64); junctions use `np.add.at` and log1p without extra clipping.
- Thinning follows the spec's literal blake2b UTF-8/tab/Athaliana/8-byte/big-endian floating rank and strict `<` comparison, rather than a differently rounded integer-threshold policy. It is stable across tiles/order and nested across fractions.
- RC permutation is exactly `[1,0,2,5,6,3,4,7]`: reverse positions, swap strands, preserve donor/acceptor identities.
- Additional B5 compatibility finding: `gsf_contract.tile_windows` may return a zero-start full-tier window beyond a short contig. The builder accepts that legitimate padded case (30,720/61,440/129,024), rather than rejecting all `fin > reference_length` tiles.

## Test environment and red/green evidence

All commands below ran from the isolated worktree. `P=.venv-mvp/bin/python` abbreviates the interpreter.

Shared Python 3.8 had pytest/NumPy/DuckDB but no pysam. System Python 3.10 had NumPy 1.26.4, DuckDB 1.5.5 and pysam 0.23.3 but no pytest. Created an ignored worktree-local `.venv-mvp` with system packages and installed pytest 8.4.2 **only into that environment**. No shared package installs/upgrades.

Initial `python3.10 -m venv --system-site-packages .venv-mvp` failed because Debian ensurepip was absent. Recovery (successful):

```sh
/usr/bin/python3.10 -m venv --without-pip --system-site-packages /tmp/pi-agent-611e8432-78e4-474-e54d14c0/.venv-mvp
python3 -m pip --python /tmp/pi-agent-611e8432-78e4-474-e54d14c0/.venv-mvp/bin/python install 'pytest<9'
```

| Phase | Exact test command after interpreter prefix | Actual result |
|---|---|---|
| Baseline, before implementation | `-m pytest revision/scripts/tests/test_b6_panels.py -q` | 8 passed, 0.56 s |
| Builder RED, before builder existed | `-m pytest revision/scripts/tests/test_mvp_c0lite.py -q` | 15 assertion failures: `approved C0-lite builder is not implemented`, 33.66 s |
| Builder GREEN | `-m pytest revision/scripts/tests/test_mvp_c0lite.py -q --tb=short --basetemp=.venv-mvp/pytest-builder` | 15 passed, 132.41 s |
| Raster RED, before rasterizer existed | `-m pytest tests/test_mvp_tracks.py -q --tb=line --basetemp=.venv-mvp/pytest-raster-red` | 16 assertion failures: `eight-channel MVP rasterizer is not implemented`, 11.67 s |
| Raster GREEN | `-m pytest tests/test_mvp_tracks.py -q -s --tb=short --basetemp=.venv-mvp/pytest-raster-green` | 16 passed, 11.77 s |
| Padded-tile regression RED + hardening | `-m pytest revision/scripts/tests/test_mvp_c0lite.py -q -k 'padded or bounded or independent or empty_bam' --tb=short --basetemp=.venv-mvp/pytest-hardening-red` | 1 failed (`invalid Athaliana tile coordinates/reference`), 3 passed, 15 deselected; 25.34 s |
| Combined GREEN after padded-tile fix | `-m pytest revision/scripts/tests/test_mvp_c0lite.py tests/test_mvp_tracks.py -q -s --tb=short --basetemp=.venv-mvp/pytest-combined-green` | 35 passed, 159.75 s |
| Existing panel/evidence regressions | `-m pytest revision/scripts/tests/test_b6_panels.py revision/scripts/tests/test_evidence_support.py -q --tb=short --basetemp=.venv-mvp/pytest-regressions` | 51 passed, 0.96 s; no skips |
| Final fresh verification | `-m pytest revision/scripts/tests/test_mvp_c0lite.py tests/test_mvp_tracks.py revision/scripts/tests/test_b6_panels.py revision/scripts/tests/test_evidence_support.py -q --tb=short --basetemp=.venv-mvp/pytest-final` | **86 passed**, 175.23 s; no skips |

Additional successful checks: `git diff --cached --check`; `P -m compileall -q revision/scripts/mvp/mvp_build_c0lite.py src/transgenic/evidence/mvp_tracks.py`; normal namespace import via `PYTHONPATH=src` with `torch imported: False`; all 16 copied input hashes reverified; persisted fixture DB reopened read-only and all 12 tables/QC/provenance inspected.

The integrated fixture uses FLAG0/ts− and FLAG16/ts+ clone members with overlapping blocks and a repeated junction. Two members become one retained molecule; both valid rows (including RC) receive membership; both retained-overlap QC counts are zero. CPU raster timing for 129,024 bases: **0.044237 s**, output **2,064,384 bytes**. Earlier run: 0.041176 s. This is a tiny, warm local fixture—not production throughput or memory evidence.

## Limitations / deferred work

- No production BAM/corpus was built or inspected; no remote job, GPU, model, dataset, trainer, decoding, or result analysis was run/edited. Full-repository/model/GPU tests were not run; only the requested CPU slice and existing panel/evidence regressions were exercised.
- The full 1.5M-read/5M-block ingestion, union-find RAM, SQL spill space, DB size, per-window query cost and worker concurrency remain unbenchmarked. Batched `executemany` is deliberately simple, not a claim of meeting the draft one-hour build estimate. Unique-name union-find RAM remains O(names).
- Input SHA256 calculation adds a raw-byte BAM read after the single alignment-decoding pass, and hashes the corpus. Callers must supply immutable input snapshots; continuous concurrent input mutation is not supported. Chromosome/schema/species checks are not a substitute for a future full reference-manifest identity attestation.
- Current raster accepts only this schema. Callers must use retained memberships for train/valid and explicit windows for heldout evaluation; no trainer adapter or worker connection lifecycle was added.
- Secondary placements remain the explicitly accepted leakage-audit gap. Clone/accession stripping is only the approved MVP stand-in; no full A11, library keys, UniVec, audited secondary placements, or full C0 machinery.
- The exact protocol floating-rank calculation is preserved, including ordinary floating-point rounding; no alternative thinning encoding was introduced.
- No remaining fixture failure is known. Production use still requires controller diff review and a separately authorized small real-data resource/identity check before a full build.
