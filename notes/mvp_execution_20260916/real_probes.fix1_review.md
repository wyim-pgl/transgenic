# Independent scoped review — real probes fix round 1

## Scope and verdict

Reviewed A `/tmp/b6-p2-readiness-1535112` at **d99b1f8a2a42edfe4877788d0f7d1ad7f9d53e50**, implementation **981489daa2ddbb040ab0056299655899dd2b28b2**, against **3b62306acfa989e3bcf2a15b6506068980a9c3fe**. Read the original independent review, complete `real_probes.fix1.diff`, and report's superseding round-1 section; checked affected live source and shared runtime helpers after Graft orientation. The implementation-to-HEAD delta contains documentation only.

**Scoped correctness: ACCEPT. R1 and R2 are addressed. No new Critical/Important regression identified in this fix.** This is not an exhaustive review of unrelated code or proof of GPU readiness.

**Bounded P1/P2: conditional GO for diagnostic execution only**, after controller-approved staging and binding regeneration below. The original R2 source-level objection to subsequent four-rank diagnostics is resolved; actual distributed CUDA behavior remains unverified. **Six-gate readiness and training remain BLOCKED.** Resource commit `4bd97399` is separate, under review and unmerged; it is neither evaluated nor credited here. Resource mode remains absent from reviewed A.

## R1 — HIGH decoder mismatch: addressed

- Actual MVP train **and** valid constructors explicitly pass the loaded bound config's `gff_vocab_version` (`src/transgenic/training/mvp_cli_runtime.py:606-620`). This fixes the readiness target, not just its probe.
- Probe construction receives loaded config and actual model config and explicitly selects the same decoder version (`revision/scripts/mvp/real_probes.py:233-246,624-628`). Shared `validate_decoder_dataset` (`mvp_cli_runtime.py:491-499`) checks dataset version, real tokenizer length, config/model vocabulary size, label cap and decoder positions. The bound v3 contract is **290 / 8192 / 8192**. Dataset defaults remain unchanged, but these callers no longer rely on v2/2048 defaults.
- The existing dataset behavior chooses 8192 for v3 and preserves the end token when truncating (`src/transgenic/datasets/datasets.py:231-236,414-419`); no cap was relaxed.
- Regression code uses real GFF tokenizer/dataset labels for `<empty>` (`[0,289,2]`), two genes, 2701 tokens and the 8192 cap, plus disagreement rejection (`tests/test_mvp_review_fixes.py:31-71`). These are actual label paths, not finite-loss proxies.
- The existing two-rank `execute -> _train -> final` fixture now observes explicit v3 kwargs, real 290-token tokenizer and 8192 cap on both train and valid on each rank (`tests/mvp_cli_probe.py:171-190,247-248`; success is exercised by `tests/test_mvp_cli.py:433-462`). Tiny external encoder/DNA-tokenizer and compute dimensions remain CPU fixtures, not real-Hyena evidence.

## R2 — MEDIUM rank-local divergence: addressed within requested boundary

- Compiled-mode main creates a shared Accelerator before guarded local setup/model/dataset/tier work (`real_probes.py:598-635`). Offline variables precede Accelerator/HF imports. `_collective_preflight` catches local exceptions, exchanges status and raises the same failing status on each rank (`mvp_finalize.py:398-417`).
- Compiler/optimizer configuration and local DataLoader construction are guarded. Exactly 24 batches are retrieved inside `collective_batches`; a rank-local loader exception is shared before any update (`real_probes.py:530-539,552-587`).
- `_batch_parts` validation is reduced across ranks before device transfer/forward, and update-count agreement is collective (`real_probes.py:489-518`). Finite-loss/gradient/skip checks remain intact.
- The post-update kernel-counter/eager-fallback check is collective **before** measurement gather (`real_probes.py:542-549,588-590`). Thus the original rank-1 early-raise path no longer leaves peers entering that gather alone.
- `accelerator.prepare`, `one_update`, hardware collection and final gather are **not** wrapped wholesale as local operations. Hardware collection already has its own local probe guard followed by its transport (`mvp_cli_runtime.py:334-374`). The new boundaries do not introduce nested collective exception handling.
- Actual two-process CPU fault programs call the production helpers with rank-1-only loader/kernel-counter failures (`tests/mvp_real_probe_faults.py`). Their tests invoke the real bounded runner and require original diagnostics on both ranks, nonzero exit, FAILED, no timeout and no DONE (`tests/test_mvp_review_fixes.py:87-111`). They do not run the whole CUDA probe or certify NCCL failure handling.

CUDA allocation/transfer/forward/backward, process death and failed collectives still require launcher teardown, bounded child execution and an outer allocation wall limit. This is deliberately not a universal distributed exception-recovery claim.

## Environment, cleanup and regression assessment

Nonempty `TRANSGENIC_RUN_DIR` is rejected before either path constructs the encoder (`real_probes.py:134-145`; `mvp_cli_runtime.py:517-523`), closing the inherited encoder-report-write hazard. Both refusal paths have explicit tests.

The shared decoder validator is small and used by both targets. Probe validation failure closes its dataset; main retains the dataset handle when later tier checks fail. Main's nested `finally` attempts Accelerator cleanup even if dataset close fails (`real_probes.py:683-690`). Trainer's existing cleanup still closes both datasets and ends training. No new distributed framework or unnecessary whole-function wrapper was introduced. Cleanup exceptions may still obscure a preceding probe exception; they cannot turn failure into an accepted runner receipt and are not a new Important blocker here.

No changes to LR, 250/12 training budget, sampling policy, metrics or publication authorization were found in this fix. The inherited first update's zero LR and cold-update/non-resource interpretation remain as recorded in the original review.

## Verification and staging conditions

- Independently checked HEAD, implementation-to-HEAD documentation-only delta, and successful `git diff --check 3b62306..HEAD`. Worktree has pre-existing `.gitignore`, `.ignore`, `.testdeps` changes; it is **not** globally clean.
- Controller reports **92 focused tests passed in 130.55s**. Writer reports **245 scoped passes in 644.32s** before the last import-order adjustment, then **92 focused passes in 112.55s** afterward. These results are attributed, not independently rerun. No broad suite or additional reproduction was needed for this source-level disposition.
- No GPUs, SSH, jobs, subagents, source edits, merge or push used. Only this review was written. Advisor consultation failed with `claude_code_version_too_old`; no advisor assessment was received.

Before diagnostic execution, regenerate **all consumed-code manifests, tooling bindings, prepared records and receipts** for this code. Old v2-label receipts cannot establish v3 readiness. Stage exact immutable corpus/evidence/checkpoint/SIF/code and both pinned HF cache layouts; keep fresh writable module/compiler scratch outside B5 and immutable roots. Sanitize environment, use reviewed real tile selections and exact bound argv through `run_bounded`, with controller authorization and an outer wall limit. Stop on failures; no fallback, cropping, fabricated receipts or automatic retries.

Actual aarch64 SIF Torch 2.14/CUDA 13.4, Transformers 4.57.6 generation/cache, Accelerate 1.15, real 400M gradients, native bf16, Inductor and NCCL remain untested here. P1/P2 diagnostic success would not authorize training; all required measured gates, separately reviewed resource integration/budget authorization and final readiness checks remain necessary.
