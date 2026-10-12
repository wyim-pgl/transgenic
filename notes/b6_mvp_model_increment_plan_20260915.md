# MVP model-only increment — 2026-09-15

Approved parent spec: revision/protocols/B6_MVP_PILOT_v1.md SHA734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74. Implement in isolated worktree from pi-agent-611e8432-78e4-474 abca40f, which already includes approved CPU data implementation (86tests and reviewerPASS). No main/remote checkout changes, no GPU jobs, no dataset/trainer edits in this increment.

## Files/interfaces

- src/transgenic/model/configuration_transgenic.py: add evidence_channels=0, validated nonnegative integer (reject bool), roundtrip config. Do not edit frozen recipe JSON.
- src/transgenic/model/modeling_HyenaTransgenic.py: HyenaEncoder creates bias-free evidence_proj(K, resolved HyenaConfig.d_model) only if enabled. Mark projection for _init_weights to zero instead of generic random Linear initialization. Constructor zeros must survive parent post_init. Disabled configuration must preserve state keys and RNG consumption.
- HyenaEncoder.forward(...,evidence_tracks=None): fuse projection into per-base h before positional injection/downsampling. Validate enabled channel count, B/L/K shape, floating dtype/device. Cast tracks to h dtype; do not add synchronizing per-step finite assertions or cache fallback. Supplied tracks on disabled model fail explicitly. No tracks on enabled model means zero contribution.
- Thread evidence_tracks through transgenicModel.forward and transgenicForConditionalGeneration.forward to encoder. Precomputed encoder outputs are already fused: never inject a second time. Keep prepare_inputs_for_generation/cache internals unchanged unless a failing actual generation test proves change required.
- tests/test_mvp_model.py: stdlib unittest, actual model wrappers/downsampling/LED decoder/GenerationMixin; substitute only AutoConfig/AutoModel Hyena factories with a deterministic tiny embedding backbone, no downloads.

## Scope ruling

Audited --init-from checkpoint loader/config compatibility belongs to the next trainer/checkpoint increment, not this residual/generation change. Here test ordinary safetensors serialization/loading with shared aliases and document projection missing-key expectations. Do not introduce permissive runtime strict=False helpers or claim real seed123 checkpoint initialization certified. Zero output equivalence is step0 after shared weights are loaded; enabled-vs-disabled same seed alone is not enough because extra parameter construction consumes RNG. After fine-tuning zero evidence equals current backbone, not original seed123.

## TDD sequence

1. Establish tiny CPU baseline using /home/wyim/miniconda3/envs/pdb/bin/python: torch2.6.0+cu124,transformers4.46.3,safetensors, no pytest/accelerate. Use unittest without package installs. Run baseline before production edits.
2. Failing tests config0/8 roundtrip/reject invalid and disabled key/RNG path. Implement validated configuration and projection initialization minimally.
3. Failing tests on actual wrappers: copy identical shared weights, eval/no-dropout consistent dtype; evidence-enabled zero-initialized projection yields bitwise equal logits for omitted/zero/finite nonzero tracks vs B5. Test encoder width != decoder width. Assert no parameter for disabled path.
4. Failing tests nonzero projection changes encoder states/logits and receives finite nonzero gradient; invalid shapes/dtypes/disabled input/device mismatch fail. Implement encoder fusion+forward threading.
5. Generation hook with kwargs capture sees tracks tensor exactly once in encoder; cached vs no-cache sequences and same-prefix teacher-forced next-token argmax agree with nonzero projection. Numeric logits comparison uses documented tolerance. Confirm precomputed states not double-fused. No handcrafted encoder-output injection workaround.
6. Ordinary checkpoint save/load regression for disabled/enabled learned projection and alias-preserving safetensors. Explicitly distinguish missing-projection future initializer from evidence model roundtrip.
7. Run targeted tests, existing feasible model/config regressions; report unavailable real-Hyena/SIF tests as pending, not skips counted as success. CPU factory substitution does not certify Hyena400M/productionTransformers4.57.6/bf16/compile/DDP.

Write notes/b6_mvp_model_implementation_report_20260915.md with actual red/green commands, base/commit, files, limitations, any needed plan rulings. Commit only in worktree authorWon Yim<wyim@unr.edu>; no merge/push/nested agents. Controller independent review required.
