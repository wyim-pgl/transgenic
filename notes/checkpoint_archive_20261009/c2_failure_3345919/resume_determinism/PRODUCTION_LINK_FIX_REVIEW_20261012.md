# Production allocation helper fix review — 2026-10-12

**Spec compliance: Approved for this correction scope. Quality: Approved. Both prior Important findings are ADDRESSED in committed7266e14.** Reviewed the complete `/tmp/c2-production-link-fix-review.diff`, updated producer report and actual producer/controller contract report. Checked current production_link.py SHA256:

`5da60c8f994a7814ba2d0ac337f7f8320e59b887612bc6ec0622059e427d88c9`.

## Findings rechecked

1. **Compiled namespace compatibility — ADDRESSED.** `production_link.py:157–170` creates a validation-only compatibility view for either wholly bare names or one uniform leading _orig_mod. wrapper. Mixed namespaces, collisions and nested/embedded wrappers fail explicitly. Pending gradient shape/dtype checks resolve against that bare view (`:238–241`). Final export must remain bare; compiled resume coverage is normalized and legitimate source-declared tied aliases resolve to direct retained bare targets (`:262–270`). Alias targets must exist, alias tensors must be absent, and retained names precede aliases lexically, matching the source's sorted-first export convention. Original raw keys remain in stored contracts, inventories and canonical probe trees; no digest protocol change occurred.

2. **Complete CUDA RNG state acceptance — ADDRESSED.** `production_link.py:202–213` maps actual Accelerate RNG keys into public source encode_rng validation. It requires a list of exactly four contiguous strided state tensors and delegates one-dimensional CPU uint8/16-byte Philox/offset-divisibility, Python/NumPy/CPU checks to the existing side-effect-free source validator. CUDA device counts are captured in the private structural contract and compared on subsequent state validation (`:220–223`). The previously accepted short/missing-device state cannot emit READY.

## Evidence and regression assessment

- New real tiny compiled fixtures use torch.compile(...,backend='eager'), Accelerator(cpu=True).prepare/save_state, real backward and source pending-gradient/fixed-export writers. The real tied model retains compiled z_weight at resume while fixed export retains sorted a_weight and declares z_weight→a_weight. Pending and final compatibility checks now exercise that actual representation boundary.
- Zero-pending and pending canonical tests explicitly retain raw _orig_mod. names and compare the result to reviewed probe value-tree hashing. Namespace collision/mixed/nested and invalid tied-target mutations reject.
- CUDA regressions cover short/long byte states, wrong rank/dtype, invalid offset, missing/extra device, tuple representation and noncontiguous layout. The complete preflight→postflight malformed-rank3 regression checks that no link_receipt.json is created. A separate regression compares Python/NumPy/CPU RNG and CUDA initialization before/after validation.
- Reported RED reproduced both prior failures, including false READY. Reported GREEN:33 producer tests passed12.613s; the same three actual producer/controller tests passed15.978s; a fixture-cleanup focused regression passed0.309s. Two expected Accelerate tied-tensor removal warnings arise from the genuine save fixture; they are documented source behavior, not failed assertions. No suite or focused test was rerun by this reviewer because the diff and recorded behavioral coverage resolve both concerns.
- No new scoped correctness finding. No shell, request/receipt schema, controller, immutable training source, recipe, cache-policy or scientific-gate change appears in this fix. The actual controller contract tests corroborate unchanged receipt interoperability.

## Remaining boundaries

This approval concerns the production helper corrections and current request/receipt interface, not host3.6/Apptainer/NCCL execution, actual GPU scientific proof or final4099 completion. Separate controller in-place activation work prompted by Lustre's unsupported physical promotion remains outside this diff and needs its own review. It must preserve original200/diagnostic256 and all existing allocation/science/receipt/terminal/storage gates.

Review remained read-only except this report: no source edit, commit, remote action/job or spawned agent.
