# MVP metric increment — 2026-09-15

User approved preserving inherited composite training loss, pure weighted CE paired primary, and composite logging alongside it. Threshold remains 0.005 nats; delta=loss_off-loss_on. This approval supersedes the pending metric question in the restart handoff, not A46 freeze.

## Global constraints
Work from exact c7fa759 in an isolated worktree, no main source changes, merge/push or remote modifications. Existing seed123 and all Slurm jobs untouched. CPU metric increment only; trainer/runtime/init-from remain subsequent increments. No new dependencies or shared environment upgrades. Read AGENTS and build/query graft in worktree before source exploration. Existing B5 behavior unchanged.

## Task 1: CPU paired metrics
Create src/transgenic/training/mvp_metrics.py and tests/test_mvp_metrics.py (minimal package initializer if needed). Use test-first and baseline model/dataset tests. Source authority modeling_HyenaTransgenic.py:898-1045, datasets.py:369-487/814-866 and approved B6_MVP_PILOT_v1.md:237-244; read complete relevant definitions after graft queries.

Pure per-tile CE uses logits already aligned to labels (no extra shift), ignore=-100, weights id2=5,id17=4,id21/id272=3,id22..271=2, others=1. Numerator=sum weighted negative log probability; denominator=sum target weights, not number of tokens. Compute CE float32, retain numerator/denominator per tile for audit. Validate shape, target range, finite results and positive denominator. All ignored fails closed. Vocabulary must support all prescribed IDs; fail closed rather than silently changing weights.

Paired evaluator batch1 only, same current model eval/no_grad, fraction1 tracks from caller, explicit zeros/off versus real/on, same DNA/mask/labels. Restore previous model mode even on failure. Do not pass transcript_counts (inherited trainer doesn't). Actual outputs.loss is separately recorded as composite, not reconstructed. Accept collated 10-tuple evidence batches with rn/has_evidence at index9 and tracks index8. No DDP collective implementation here: expose record collector and aggregator callable after caller gathers records.

Record rn (actual database identity), has_evidence, loss_off/on, composite_loss_off/on, CE numerators and denominators. Aggregator takes expected rn inventory, deduplicates distributed-padding copies by rn, rejects missing/extra rn, inconsistent/nonfinite records, disagreement in flags/denominators, and empty evidence-positive subset. Allow only tightly toleranced floating numerical differences for duplicate records; document tolerance. Primary=arithmetic mean of per-tile (loss_off-loss_on) over has_evidence=true; composite delta separately labelled. Do not replace with pooled token-weight CE. Preserve per-tile output for later bootstrap, do not implement bootstrap or GO/STOP here.

Tests: hand calculated unequal weights; boundaries and ignores; no double shift; equivalence to inherited CE; all ignored/nonfinite fail; unequal tile denominators distinguish tile mean vs pooled; actual composite kept; off/on track forwarding; eval/no_grad and restore on errors; rn duplicates/missing/extra/inconsistent; evidence-empty rejection; delta sign. Tiny fake model enough for orchestration, actual model tests regression. No realHyena/GPU claim.

## Remaining plan (not this task)
2. Strict weights-only init and provenance using enclosing run_config recipe.
3. Fixed-step trainer, two LRs and actual Accelerate/DDP scheduler/accumulation tests.
4. Production CPU data QC, immutable checkpoint/SIF identity and real GPU gates, independently reviewed before any fine tuning.

## Preflight ruling
Task1 metric output -> Task3 evaluator: per-tile records and expected rn inventory. Ruling: use equal-tile primary specified by MVP, not advisor's suggested global numerator/denominator aggregation; pooled CE would change the estimand/threshold. Task1 internal tests and source agree on target-weight denominator. No model forward modifications required.
