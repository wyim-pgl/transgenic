# Post-run acceptance (v2 scripts remain immutable after submission)

Reviewer identified that comparison.json alone does not check restored RNG records. Independent acceptance must require exactly one restored event per rank in A/B/C with rng_matches_checkpoint true and actual/expected RNG maps equal. Preserve all traces regardless of comparison return code. Do not promote on comparison.passed alone.

Also require: canonical_order.json complete/matching; fresh allocation-bound verified record; original/source hashes unchanged; snapshot200 A/B source digests equal; B210→C initial checkpoint digests equal; expected stop metadata and all four RNG files; complete per-rank 240/480/240 forward traces and exact inputs/losses/gradients/model/optimizer/scheduler comparison; no nonfinite values or failure markers. Diagnostic is not full4099 or epoch-eval coverage.

Require exact trace multiplicities (not dict-collapsed sets): A/C each240 forwards+10boundaries perrank, B480+20; no duplicate micro IDs. Exactly one finished event perrank, preempted=True, steps210/220/220. Exactly one B rank0 snapshot_published atmicro5040. All3launches and comparison must return0, with COMPARISON_PASS. Recheck armC2/seed123/world4/target4099 and nine required checkpoint files. Verify final input stats and both source manifests, original200 payload hashes. Frozen identity's53fields differ only allowed compile_policy.

Critical interpretation: even A=B and B=C does NOT alone establish equality to original uninterrupted210. Compare original logged210 loss3.2567708492279053 as an additional diagnostic. If historical mismatch remains unexplained, retain scientific gate and investigate initial compiler history/environment; do not automatically promote.
