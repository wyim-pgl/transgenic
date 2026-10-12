# History proof harness fix review — 2026-10-12

**Spec compliance: Approved for harness scope. Quality: Approved. All three prior Important findings are ADDRESSED, including the two small corrections identified during this re-review.**

Reviewed `/tmp/c2-history-proof-fix1.diff`, the appended fix report, and the final scoped corrections in `/tmp/c2-history-proof-final-delta.diff`. Independently checked the current `history_proof.py` SHA256 against the final package header:

`f38887aeabfa0721242823da2d0cab7330a7e3e69bb89f5e0f6931d06d93e9d2`.

The current test file SHA256 is `d9f2b8c2c7bab1f78c5c2a15417764b16ae6acb1ad37467e0c755c154d79bbb7`.

## Findings rechecked

1. **Strict history/bootstrap acceptance — ADDRESSED.** `history_proof.py:286–316` calls actual local-only CompileHistory.load with expected world4, the saved checkpoint identity, digest and exact numerical checkpoint position. All three journals—continuous220, resumed220 and at210—are validated. Their bootstrap must equal the root manifest plus its actual file digest; the manifest is compared to the pinned original receipt and literal committed draw-prefix hashes. First16 origins must be reconstructed and any later origins observed. The final correction at`:312–315` compares every seven-field position against the actual slots1..15/3915, including original optimizer_step=(slot-1)//24. The new `original_checkpoint_receipt.json` was independently byte-compared with retained3361666/B/DIAGNOSTIC_SOURCE.json; they are identical. It must be included in the immutable helper staging manifest.

2. **Enabled production operating caches — ADDRESSED.** `history_proof.py:130–138` wraps the actual runtime fingerprint before trainer execution; `:276–285` refuses disabled local FX/AOT caches, force_disable_caches, wrong world/backend/bf16/nonreentrant/determinism and wrong shared Inductor/Triton namespace. Acceptance repeats this validation. Remote cache settings remain actual runtime defaults. This enforces the requested proof configuration rather than merely recording a disabled configuration.

3. **Rank/lifecycle/snapshot/context evidence — ADDRESSED.** `history_proof.py:230–271` requires correct rank ownership for every row, one prepared/finished event in order, expected start/stop positions, actual world4/rank-specific NCCL reducer metadata, and the single continuous rank0 snapshot210 record. The snapshot record binds numerical position, receipt digest and copied checkpoint bytes. `TrainingTrace.pre:67–70` records actual module.training and torch.is_grad_enabled plus train arm view; these must match the expected context and are compared across continuations. Boundary RNG is compared alongside gradient digests. Invalid/missing/duplicate lifecycle/snapshot/context/rank evidence is refused.

## Additional corrections verified

- During this re-review, the initial fix's exact two-key finished-result comparison would reject the actual trainer result containing `history: []`. Named contract check: current trainer train_loop returns dict(ended, history=history) at `train/train_HyenaTransgenic_B6.py:559`; run_b6 returns it unchanged at`:591`, and main returns run_b6 at`:741`. The final correction at `history_proof.py:242–245` now validates required preempted220 fields, empty epoch history and absence of done while accepting that actual return.
- The initial fix checked only bootstrap indices/phase/epoch. The final correction validates all position fields, including optimizer step, and the test adds a recomputed-digest wrong-update mutation with a specific bootstrap rejection assertion.
- A targeted temporary-directory probe against the final live code confirmed wrong bootstrap optimizer positions are rejected and the actual finished result containing history=[] is accepted. No full suite was run by this reviewer.

## Evidence and scope

- Read both scoped diffs and the updated report. Because the two corrective edits arrived while review was active, also generated an in-memory delta between the supplied fix1 package and current harness/test files to verify the exact additional hunks; no source files or temporary review packages were written. Final controller-supplied delta/header matched those production corrections and the checked current SHA.
- The success fixture now uses complete real-schema histories and actual trainer return data. New negative regressions cover malformed ranks, disabled caches, wrong bootstrap slot/update, substituted original checkpoint receipt, rank ownership, lifecycle/context and missing/duplicate snapshot evidence. Recorded RED results expose these failures; reported focused GREEN is7 passed (latest controller result18.746s, zero skips). The report also records bash syntax and py_compile success. No broader suite rerun or remote operation occurred.
- No remaining scoped correctness finding. This approves staging this concrete harness version after the separate production integration gate. It does not claim GPU checkpoint equality, historical210 reproduction, event-compression equivalence, future compile-event/400/second-resume coverage,4099 completion or epoch/final publication. Those scientific/operational gates remain mandatory.
