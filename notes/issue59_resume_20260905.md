# Issue #59 implementation and remaining-work order — 2026-09-05

Base: `aa598f5` (#24). Changes in this session are uncommitted.
Read issues #54–#59 and #61 using `gh issue view --json` on `gpu` (the local
host has no `gh`; the remote CLI's default view fails on deprecated project cards).

## Recommended order

1. **#59: training blocker.** A28 already requires restart-stable sample order;
   fix this before the first B5 seed without changing the database recipe.
2. **#58: training blocker, author decision pending.** Resolve the empty-label
   token contract before training on the 25,159 affected rows.
3. **#56: leakage blocker, author decision pending.** Resolve the no-CDS masking
   exception before treating the frozen corpus as compliant with A29.
4. **#57: validation blocker plus reporting defects.** Check actual gene splits
   for tile databases; coordinate builder counters with any approved amendment.
5. **#55: corpus correctness defect requiring disposition.** Coordinate the
   ordering fix with other builder changes to avoid repeated frozen-DB rebuilds.
6. **#61: evaluation-definition blocker.** Establish the effective denominator
   and assess labelled versus unlabelled genes before interpreting metrics.
7. **#54: reporting.** Carry out the existing decisions about rice mapping and
   Arabidopsis masking in Methods and supplementary tables.
8. **Long-read fetch v2: blocker for future collection.** The design in outer
   `todo.md` §2c is fixed; collection is complete and the current no-`evidence/`
   write restriction prevents the coordinated caller migration and swap.
9. **Orientation provenance: cosmetic cleanup.** Defer until the running driver
   finishes and the no-`*.sbatch` restriction is lifted.

## Implemented: sample order on resume

The generic B5 trainer now prepares loaders with
`DataLoaderConfiguration(use_seedable_sampler=True)`. Its existing
`torch.manual_seed(seed)` precedes `accelerator.prepare`, so the sampler captures
the run seed before checkpoint RNG restoration. No new Accelerate API beyond the
flag proposed in #59 is required.

`b5_runtime.epoch_batches` explicitly sets the zero-based epoch before iteration
and skips the consumed prefix while preserving absolute micro-batch indices.
Setting the epoch is essential: a fresh loader starts at zero, even when metadata
requests a later epoch. The existing skip behavior is retained; this change does
not introduce Accelerate's optional skip-loader optimization.

This implements A28's existing sample-order requirement. No protocol amendment,
database rebuild, or change to frozen §3–§9 text is involved. A28 prohibits code
changes inside an existing job chain: apply to a new run, not an old chain whose
original permutation used the nonseedable sampler.

## Validation

`tests/test_b5_resume.py` runs real Accelerate save/load in separate Python
processes against 64 sample IDs, checkpointing after 20 samples. It checks epochs
0 and 3, seeds 123 and 456, batches of 1 and 4, and 0 and 2 persistent workers.
The resumed suffix, absolute batch indices, complete epoch membership, and next
epoch order match uninterrupted execution. A nonseedable negative control
reproduces duplicated/missed samples. A wiring check covers the trainer's sampler
configuration and use of the epoch helper.

On `gpu`, torch 2.5.1 / Accelerate 1.14.0: **6 passed** in 8.88 s, in isolated
`/tmp/transgenic-issue59.FGbfYC`; no real database or production checkout was used.
These tests establish sample-order continuity, not bitwise model/loss equivalence
or preservation of partially accumulated gradients.

Full local suite with `/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
-m pytest -q`: **473 passed, 6 skipped** in 62.52 s. Five additional skips are the
torch-dependent restart scenarios, all exercised on `gpu`; the baseline had one
skip. `git diff --check` passed.

## Decisions deliberately left open

These alternatives are explicitly unresolved in the issues and outer `todo.md`
§3; authorization to choose the work does not choose a scientific reading:

- **#58:** two-token `<empty> </s>` by intent (amend contract/spec), or three-token
  `<s> <empty> </s>` (fix tokenizer). Neither requires a DB rebuild.
- **#56:** quantified no-CDS exception, retain those genes for split masking, or
  clean the annotation and rebuild; these change the treatment of 1,605 held-out
  noncoding loci and the decoy denominator.
- **#55/#56:** whether and when to amend/refreeze the builder recipe and rebuild;
  do not modify the frozen 272,224-row B5 database in place.
- **#61:** how the manuscript distinguishes the nominal orthogroup test set from
  the actually labelled subset; do not silently redefine the split or denominator.
- **#59 ancillary observations:** scheduler totals and partial accumulation across
  checkpoint/epoch boundaries are outside this sample-order fix. This test does
  not resolve those semantics or claim full trajectory equivalence.

No author decision was needed to complete #59. No commits, GitHub issue edits,
`evidence/` writes, or `*.sbatch` edits were made.
