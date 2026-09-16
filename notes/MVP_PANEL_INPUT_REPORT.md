# MVP panel input implementation report (2026-09-16)

Base: `8803afb`; branch: `mvp-panel-inputs-20260916`.
Interface and decoder obligations: `notes/MVP_PANEL_INPUT_CONTRACT.md`.

## Implementation

Added only the panel producer, its CPU tests, and these two notes. No changes to
`mvp_tracks.py`, training Dataset/filter/DB, or B-producer files. Reused graph-located
`mvp_tracks.py:14-108`, `mvp_windows.py:94-160`,
`mvp_produce_counts.py:54-70`, `mvp_cli.py:122-186`, and
`mvp_cli_runtime.py:89-143` for rasterization, frozen panel/window policy,
commit-anchored FASTA/GFF authority, and accepted database identity gates.

Producer emits one reusable on tensor, genomic-forward FASTA, and selected-molecule
JSONL per locus. It compares seed 1/2/3 full-retention tensors exactly; off inputs
are zeros_like(on) using the same sequence. It reports biological channels 0..6
separately from availability 7, including zero-support windows and endpoint-only
junction support. Provenance uses independent block/endpoint queries, exact members,
and the molecule's exclusion cause. It never uses retained flank molecules as a
fallback, reads training examples, or uses RN membership for rasterization.
Existing acceptance verification does read corpus/evidence identity metadata.

Publication is exclusive-directory/exclusive-file, with canonical manifest and
SHA256-bound COMPLETE marker last. Input/request/code identities are freshly
rehash-checked before and after manifest publication; output bytes are also checked.
No fallback builder was needed: reviewed builder retains heldout primary geometry.

## Verified CPU tests

Final commands, from the isolated worktree:

```sh
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/.testdeps:$PWD/src" \
 /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
 revision/scripts/tests/test_mvp_panel_inputs.py tests/test_mvp_tracks.py -q
# 27 passed (28.69 s)
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/.testdeps:$PWD/src" \
 /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
 tests/test_mvp_cli.py -k candidate -q
# 16 passed, 29 deselected (11.81 s)
```

New tests cover empty tile_est with nonzero heldout EST, unchanged retained
population and unchanged DB bytes, no retained-flank fallback, overlapping block
molecule union, shared donors, minus donor/acceptor and RC permutation, N masking,
intron-only endpoints, exact members/geometry, all 50 actual frozen panel coordinates
with synthetic sequences, bound output hashes/shapes, zero-support publication,
no overwrite, mutations before AND after manifest publication, seed-independence
failure, real frozen-reference rejection, acceptance/dataset digest rejection,
and WAL rejection. CLI help and `git diff --check` also passed.

The first TDD run failed because the new module did not exist. Default Python 3.8
lacked torch. Python 3.12 with existing torch plus worktree-local duckdb/pytest/pysam
was used for final verification. A broader exploratory run had 62 passes and nine
missing-dependency failures (pysam/accelerate); pysam was then installed locally.
A subsequent broader builder/CLI run timed out at 240 s. Those broader runs are
not claimed passing. Final scoped runs above completed successfully.

## Limitations / not asserted

- No remote DB/FASTA was read, no SSH, scheduler jobs, merge, push, or subagents.
  No production artifact or scientific support claim was generated.
- The 50-locus publication integration uses real frozen panel coordinates but
  synthetic sequences and test-only mocked preflight. Actual reference and
  acceptance failures are tested separately; a fully accepted real-data end-to-end
  execution remains for the controller in its authorized environment.
- Production preflight inherits a torch import from the existing identity verifier;
  no GPU/model is used. Full source files and databases are rehashed several times
  deliberately. FASTA memory is one contig; provenance memory is one window's
  matching geometry, with two metadata queries per selected molecule. Full-data
  runtime/memory have not been profiled.
- Failed directories remain without COMPLETE and are not reusable. Hash checks
  detect changes across this invocation, not malicious transient replacement or
  later mutation; consumers must verify COMPLETE, manifest, and all artifact hashes.
- The advisor attempt was unavailable due to the configured client/model version;
  implementation reused existing gates rather than introducing replacement policy.
