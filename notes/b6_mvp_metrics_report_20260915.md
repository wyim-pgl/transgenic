# B6 MVP CPU paired metrics implementation — 2026-09-15

## Scope and identity

- Exact base: `c7fa7597e579e1abc0fdcbb0a017a6c517bd47ed`.
- Isolated branch: `task/b6-mvp-metrics`.
- Implementation commit: the commit containing this report; resolve with `git log -1 --format=%H -- notes/b6_mvp_metrics_report_20260915.md`.
- Python: `/home/wyim/miniconda3/envs/pdb/bin/python` 3.12.2; torch 2.6.0+cu124; Transformers 4.46.3.
- CPU only. No subagents, remote actions, shared dependency changes, GPU/Slurm work, merge, push, trainer, runtime, initialization, bootstrap, or GO/STOP implementation.

The exact metric brief was read before source work. The worktree graph was built with `graft build`, then queried before opening source. Source authority was checked at `src/transgenic/model/modeling_HyenaTransgenic.py:898-1045`, `src/transgenic/datasets/datasets.py:369-487` and `:814-866`, and `revision/protocols/B6_MVP_PILOT_v1.md:237-244`.

## Implemented behavior

`src/transgenic/training/mvp_metrics.py` provides three small callables:

- `weighted_ce`: float32, already-aligned per-tile weighted CE with ignore ID -100; target-weight numerator and denominator are returned for audit. It requires int64 labels, a nonempty batch, vocabulary support through ID 272, valid targets, finite logits/results, and at least one retained target per tile.
- `paired_metric_record`: batch-size-one evaluation of the collated 10-tuple contract. It forwards identical DNA, mask, and labels twice, using explicit zero tracks off and caller-supplied fraction-1 tracks on. It does not pass transcript counts. Both calls run under the same model in eval/no-grad mode, and the prior mode is restored through failures. The record uses the collated database `rn` and flag and stores pure CE losses/audit terms plus each actual `outputs.loss` as separately named composite loss.
- `aggregate_paired_records`: validates an expected RN inventory after distributed records have been gathered, deduplicates padding copies by RN, preserves one record per tile, and rejects missing/extra IDs, invalid records, flag/denominator disagreement, and an empty evidence-positive subset. Duplicate floating fields must agree at `rtol=1e-6`, `atol=1e-8`. Primary and composite deltas are equal-tile arithmetic means over evidence-positive tiles; no pooled-token estimand is emitted.

## Test-first evidence and exact commands

Baseline model regression before production edits:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wyim/miniconda3/envs/pdb/bin/python -m unittest test_mvp_model
```

Result: 11 tests passed in 0.663 s.

Dataset baseline attempt:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wyim/miniconda3/envs/pdb/bin/python -m unittest test_mvp_dataset -v
```

Result: import error before tests ran: `ModuleNotFoundError: No module named 'duckdb'`. No remote install was attempted. The local pip cache contains only a CPython-3.8 DuckDB wheel, incompatible with this Python 3.12 environment.

Initial RED, before the production module existed:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wyim/miniconda3/envs/pdb/bin/python -m unittest test_mvp_metrics -v
```

Result: expected import failure, `ModuleNotFoundError: No module named 'transgenic.training.mvp_metrics'`.

First GREEN: 13 metric tests passed in 0.016 s. Additional fail-closed tests were then added for required-key handling, metadata types, int32/empty labels, and evidence-negative filtering. The second RED produced three failures and three errors for those missing checks. After the minimal corrections, 14 metric tests passed in 0.012 s.

Final combined metric and actual model regression:

```sh
PYTHONPATH="$PWD/src:$PWD/tests" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wyim/miniconda3/envs/pdb/bin/python -m unittest test_mvp_metrics test_mvp_model -v
```

Result: **25 tests passed in 0.566 s**: 14 metric tests and the existing 11 actual TransGenic-wrapper CPU tests.

Final syntax and whitespace checks:

```sh
PYTHONPATH="$PWD/src" /home/wyim/miniconda3/envs/pdb/bin/python -m py_compile src/transgenic/training/mvp_metrics.py tests/test_mvp_metrics.py && git diff --check
```

Result: exit 0, no output.

The metric tests cover hand-calculated unequal target weights, all weight boundaries and ignores, no second shift, equivalence to the inherited weighted-CE component, all-ignored/nonfinite/range/shape/vocabulary failures, unequal tile denominators versus pooled aggregation, actual composite retention, track forwarding, no transcript-count forwarding, eval/no-grad, mode restoration on errors, RN duplicates/missing/extra/inconsistency, nonfinite and internally inconsistent records, empty evidence subsets, evidence-negative filtering, and the `off - on` delta sign.

## Limitations and review gate

The orchestration test uses a tiny fake model; the existing model regression substitutes only the external Hyena factories while exercising the actual TransGenic wrapper. This does not claim real Hyena400M, production checkpoint, corpus, GPU, bf16, compile, DDP, or numerical reproducibility across devices. The dataset regression could not run because DuckDB is absent and remote/shared installation was prohibited. No full-suite pass is claimed. Caller-side distributed gathering, bootstrap, threshold decisions, trainer/runtime integration, and strict checkpoint initialization remain later tasks. Independent controller review remains required.
