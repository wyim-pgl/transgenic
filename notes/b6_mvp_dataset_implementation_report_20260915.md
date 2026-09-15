# MVP dataset/collate implementation — 2026-09-15

## Identity and scope

- Base: `0514b63`, resolved branch `pi-agent-1765f0e2-cf6c-4ed`. The concatenated branch/hash spelling in the request did not resolve; the brief confirmed this branch and hash.
- Implementation commit: `ad07302546a594e2b96726f742fa378a9067677b`.
- Isolated worktree: `/tmp/pi-agent-a92f1af6-926c-40f-6beab9f3`.
- Changed implementation only in `src/transgenic/datasets/datasets.py`; added `tests/test_mvp_dataset.py` and copied the binding brief. This report is a subsequent documentation commit.
- No model, rasterizer, builder, trainer, recipe, shared environment, main-worktree changes, remote work, GPU work, jobs, nested agents, merges or pushes. Author: WonYim <wyim@unr.edu>.

## Interfaces and rulings

Appended keyword-only options exactly as the brief specifies. Species filtering is parameterized in all three SQL branches; split precedence, rn ordering and weights are retained. Evidence-disabled samples remain 8-tuples, with no evidence connection, metadata query or augmentation draws. The original legacy row-query/fallback path remains unchanged; evidence mode forces strict handling even when `strict=False` was supplied.

Evidence requires Athaliana, eight channels, an explicit integer seed and finite probabilities/fractions. Corpus `is_rc` must be an explicit Boolean; the already-oriented sequence is passed unchanged. Region, sequence and post-SEP token lengths must agree. Frozen evidence `corpus_tile` metadata must match all six fields before tracks are returned, including dropped samples. Missing/duplicate metadata fails closed. Only successful per-tile validation is cached, never tracks.

Corpus and evidence use separate lazy read-only handles and PID ownership. Serialization clears handles, PIDs, generator and validation cache in a copied state; the live parent is untouched. Destructor tolerates partial initialization. A private CPU generator uses `(aug_seed + worker.seed) mod 2**64`, or `aug_seed mod 2**64` in the main process, and resets when PID/worker identity changes. Persistent workers keep their streams. This deliberately replaces the parent spec's generic global `torch.rand` sketch. Fixed `eval_fraction` bypasses augmentation draws even in training mode and retains labels. Zero fraction yields all-zero tracks through the existing rasterizer.

Default evidence samples are 9-tuples. Optional tenth metadata is `{'rn': actual_rn, 'has_evidence': EXISTS tile_est}`; availability is not inferred from dropout, masking or channel values. Collate produces int64 rn and Boolean has_evidence vectors. Uniform 7/8 legacy and 9/10 evidence input arities are supported; mixed arities and invalid track shapes/dtypes are rejected. Tracks are float16, left-padded to `dna_max_len`, independently of right-padded labels.

## Actual verification

Python: `/home/wyim/miniconda3/envs/pdb/bin/python` (3.12), torch 2.6.0+cu124, transformers 4.46.3. Only pinned DuckDB 1.5.5 was installed, with `python -m pip install --target .testdeps duckdb==1.5.5`; `.testdeps/.gitignore` ignores its entire local contents. No shared packages were installed or upgraded.

Initial test-first run (before implementation):

```sh
PYTHONPATH=$PWD/.testdeps:$PWD/src OMP_NUM_THREADS=1 \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s tests -p test_mvp_dataset.py
```

Eight initial tests ran and failed with 24 errors/subtest errors from missing new constructor kwargs; partially initialized legacy destructor also exposed its unsafe cleanup. Intermediate green attempts exposed a shadowed local `math` import (fixed) and a test substitute lacking `pad_token_id` (corrected to substitute the tokenizer call). A ninth orientation/all-N metadata regression was added afterward.

Final exact command:

```sh
PYTHONPATH=$PWD/.testdeps:$PWD/src:$PWD/tests CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest test_mvp_dataset test_mvp_model
```

**20 tests passed in 50.637 s:** 9 dataset integration tests and all prior 11 model tests. Dataset tests use separate real tiny DuckDB files and the actual rasterizer; only tokenizer loading/calls and worker identity are substituted. They cover offset/RC/N masking, distinct connections, SQL branches, legacy padding and RNG, missing/corrupt rows, metadata field mismatch/duplicates, missing/nonboolean orientation, token-length mismatch, dropout extremes, nested fractions, private/global RNG state, evaluation no-draw behavior, 9/10-tuples, DNA-vs-label padding, actual process spawn/pickling, simulated stale PID connection reopening, worker stream lifecycle and partial initialization. `git diff --check` passed.

Additional existing rasterizer regression attempt (read-only reuse of already-installed base pytest, after native pdb packages):

```sh
PYTHONPATH=$PWD/.testdeps:$PWD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/wyim/miniconda3/envs/pdb/bin/python -c "import sys; sys.path.append('/home/wyim/miniconda3/lib/python3.8/site-packages'); import pytest; sys.exit(pytest.main(['-q','tests/test_mvp_tracks.py']))"
```

**15 passed, 1 failed** in 1.44 s. Failure is unavailable `pysam` when the real-BAM smoke test imports builder fixtures. No pysam installation was authorized or attempted. An earlier attempt putting base site-packages directly on PYTHONPATH failed collection because Python 3.8 NumPy took precedence over native Python 3.12 NumPy; appending instead resolved that. A broad unittest discovery attempt also reported the known missing native pytest import for the pytest-based rasterizer file; the final explicit unittest command above excludes that unrelated collection issue. Full builder/BAM regressions are not claimed.

## Limitations / review gate

This is CPU-only synthetic-database integration, not production corpus, real Hyena tokenizer, real Hyena encoder, DDP, compile or training validation. The existing model suite substitutes its external Hyena factories. Exact resume and cross-worker-count invariance are not promised. PID handling was simulated and spawn was exercised; an actual forked DuckDB worker was not tested. Metadata is assumed frozen after first per-tile validation. Token validation checks the required length contract; it does not identify a malicious tokenizer that exchanges bases for specials while preserving length. Decoder target semantics are unchanged. Independent controller review remains required before integration.
