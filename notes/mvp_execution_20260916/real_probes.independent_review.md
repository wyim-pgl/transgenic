# Independent review: actual CUDA probes

## Scope and verdict

Reviewed `/tmp/b6-p2-readiness-1535112` at **3b62306acfa989e3bcf2a15b6506068980a9c3fe**, base `ac8ac2d`, including the complete controller diff, complete `notes/MVP_REAL_PROBES_REPORT.md`, all 626 probe lines, bounded runner/receipt contracts, and the actual model, generation, initialization, dataset and DDP training paths. Graft was consulted first in that worktree. Resource-mode future work on another branch is outside this review.

**Spec/correctness: changes required.** There is a confirmed decoder-input mismatch: the window-v3 model is exercised with the default v2 label tokenizer and 2048-token cap. This is inherited from the MVP trainer and reproduced in the new probe; copying the trainer does not make these inputs correct.

**Bounded execution:**

- **Conditional GO for single-process P1/P2 as diagnostic experiments only**, through `run_bounded`, with controller authorization, immutable inputs, a sanitized environment, fresh independent scratch, and an allocation-level wall limit. I found no unconditional single-process CUDA execution blocker established by this CPU review. Version/import/cache/determinism/OOM failures must remain failures.
- **Prefer the small label fix below before spending P1/P2 GPU time.** Without it, a successful receipt must not be represented as window-v3 decoder/input readiness. The current P1 is not merely an import check: it constructs the full model, reads six tiles and performs a labeled forward.
- **HOLD four-rank readiness execution pending the label fix and collective error-boundary hardening below.** The latter is a diagnosability/bounded-failure issue, not an established false-PASS route under the prescribed launcher.
- **NO six-gate PASS, training launch or resource authorization.** Explicitly missing `resource_budget` is an acknowledged scope limit, not a newly discovered hidden defect.

No CUDA/SIF execution, SSH, jobs, source edits or subagents were used. Only this review file was written. The advisor request failed with `claude_code_version_too_old`; no advisor assessment was received.

## Findings

### R1 — HIGH: v3 model/config silently paired with v2 decoder labels

**Locations:** `revision/scripts/mvp/real_probes.py:233-239` (`datasets`), used by `:568-587`; inherited counterpart `src/transgenic/training/mvp_cli_runtime.py:598-603`. Defaults and truncation are at `src/transgenic/datasets/datasets.py:178,227-235,414-421`. Tokenizer behavior is at `src/transgenic/model/tokenization_transgenic.py:174-190,192-239`. Bound window-v3 config: `configs/b5_400m_win_v3.json:55-58`.

`load_model` consumes the bound config, but `datasets` never receives/passes its `gff_vocab_version`. Thus the real window-v3 model (290 vocabulary entries; 8192 label positions) gets `GFFTokenizer(vocab_version='v2')`, with `maxlength=2048`. Multi-gene labels are not simply shortened: the v2 parser splits at `>` and consumes only its first two pieces, mishandling `<gene>` and dropping subsequent gene structure. `<empty>` becomes unknown plus v2 scaffolding. Long labels can additionally be silently truncated to 2048.

Neither ACGTN roundtrip, DNA/track RC checks, finite CE, nor B5/MVP logit equality detects this. Both comparison models get the same wrong labels. The four-rank update could pass while exercising corrupted/shortened decoder work, invalidating any claim that it validated the bound window-v3 input workload. This is a source-input/readiness problem, not evidence of a CUDA failure.

**Focused CPU reproduction executed successfully** (no model/HF downloads, no fixture changes):

```sh
cd /tmp/b6-p2-readiness-1535112
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.testdeps:src \
 /home/wyim/miniconda3/envs/pdb/bin/python -c '
from transgenic.model.tokenization_transgenic import GFFTokenizer
from transgenic.training.b5_runtime import load_b5_config
import inspect
from transgenic.datasets.datasets import isoformDataHyena
cfg=load_b5_config("configs/b5_400m_win_v3.json")
print(cfg["gff_vocab_version"],cfg["max_label_tokens"],cfg["vocab_size"])
print(inspect.signature(isoformDataHyena).parameters["gff_vocab_version"].default)
s="1|CDS1|9|+|A>CDS1<gene>20|CDS1|29|+|A>CDS1"
for v in ("v2","v3"):
 t=GFFTokenizer(vocab_version=v)
 print(v,[t._convert_token_to_id(x) for x in t._tokenize("<empty>")],t._tokenize(s))
'
```

Observed config `v3 8192 290`, dataset default `v2`. Empty IDs: v2 `[0,3,273,17,2]`, v3 `[0,289,2]`. The v2 two-gene token stream ends `'<tx1>', '>', 'CDS1<gene', '</s>'`; v3 preserves both gene blocks and the `<gene>` token.

**Minimum fix:** pass the already-loaded config's `gff_vocab_version` into the probe dataset, and do the same for both actual trainer datasets. Assert tokenizer vocabulary size and label cap agree with the bound model/config. Add focused real-tokenizer CPU cases for multi-gene, empty and >2048-token labels; retain the existing approved cap, rather than relaxing it. Regenerate consumed-code/tooling bindings after edits. Fixing only the probe would leave the training readiness target wrong.

### R2 — MEDIUM: rank-local failures can strand peers in the next collective

**Locations:** `revision/scripts/mvp/real_probes.py:518-548`, especially the local compiler-observation raise at `:544-545` immediately before `gather_object` at `:546`; local batch loading at `:539-542`, input validation at `:476`, cleanup at `:556-557`.

The smoke loop borrows finite-loss/gradient reductions but omits the trainer's collective phase guards. For example, if one rank has no observed new generated kernels, it raises and enters `end_training` while the others enter `gather_object`. A rank-specific loader/contract failure similarly bypasses the common error-status phase. These are precisely the kinds of negative cases a preflight must diagnose, even when healthy execution succeeds.

**Source-level reproduction / regression to add:** inject `metrics.generated_kernel_count <= before` on rank 1 only after `one_update`; ranks 0/2/3 follow `gather_object`, rank 1 follows the raise/finally path. Separately inject a loader exception on rank 1 before the first update. These distributed fault injections were **not run here**; the divergent control flow is visible directly in the reviewed function. No claim is made that actual SIF cache hits necessarily produce a zero counter.

The pinned `accelerate.commands.launch` parent should report a failed worker and the bounded runner rejects nonzero exit/timeout (`run_bounded.py:188-201`). Consequently this is **not a demonstrated false DONE/PASS**, nor an unbounded escape from the process-group timeout. It is an avoidable collective mismatch that can waste the authorized probe interval and obscure the original error.

**Minimum fix before four ranks:** reuse `_collective_preflight` (`mvp_finalize.py:409-417`) around local setup/loading and the post-update compiler check before entering the next collective; restore the collective batch-contract check used by `mvp_runtime.py:255-264`. Do not wrap an entire function containing collectives in a local-exception wrapper and assume that cures mid-collective failures. Keep launcher/process-group teardown as the backstop for CUDA/NCCL failures. No trainer budget override or new distributed framework is needed.

## What the code does establish, and what it does not

- **Source initialization:** `load_weights_only` verifies the 0→8 transition, tied storage, exact missing projection, config semantics and before/after source snapshots; the probe checks weights/run_config audit pins and optional sidecars. Postflight also rehashes runtime identity. The independent channels=0 B5 load is strict. I found no additional concrete checkpoint-binding defect in this path.
- **Dtype:** float16 EST tracks are explicitly checked; `HyenaEncoder.forward` converts them to the encoder hidden dtype before projection (`modeling_HyenaTransgenic.py:531-556`). The compiled modes use the existing bf16 Accelerator. I did not find a source-proven float16/float32 projection mismatch. Native bf16 and actual compilation still need CUDA evidence.
- **Generation:** the nonzero projection is restored in `finally`; encoder forwarding, expansion and same-cache/beam-path raw logits are checked, and cached/full-prefix tokens are compared. The observer independently constructs self-cache index selection. Leaving cross-cache tensors unchanged is appropriate for beams expanded from the same single encoder example; I found no concrete gather error there. This does **not** test heterogeneous multi-example beam batches.
- **EOS/cache limits:** `min_new_tokens=max_new_tokens=16` intentionally suppresses early EOS. This is a fixed-prefix cache/beam probe, not a demonstration of natural EOS termination or long-generation behavior. The tuple/`LegacyCacheWrapper` integration must run under actual Transformers 4.57.6; local toy success cannot certify that API. Unsupported cache behavior must fail, not be waived.
- **Optimizer meaning:** the inherited LR schedule gives the first optimizer step LR zero (`mvp_runtime.py:107-118`; probe `:470`). It does execute backward, reductions and AdamW state initialization, but does not demonstrate nonzero parameter learning. This matches the reviewed one-update primitive and is not a new defect; do not describe it as measured training convergence or a resource benchmark.
- **Receipt boundary:** `run_bounded` requires successful process exit, matching binding digest, validation, and publishes DONE last. Rank 0 can write a measurement before another rank's postflight finishes, but the launcher must finish successfully before the runner accepts it. A leftover measurement alone is not PASS. Legacy boolean-only measurements remain accepted intentionally; the controller must insist on these reviewed real-probe commands/identities, not handcrafted legacy receipts.
- **Identity scope:** import and immutable asset hashes are rechecked; hashes identify bytes, not proof of execution or independent SIF authentication. Prepare, execution, receipt validation and later checks must use the same pinned SIF paths. Do not transplant CPU package versions or host-side identities.

## Safety prerequisites for the diagnostic GO

1. Use controller-approved immutable corpus/evidence/checkpoint/code/cache mounts and the exact bound argv. Preserve both HF cache layouts. Keep writable module/Inductor/Triton trees outside all input/B5 locations; preferably start with fresh module scratch as well as fresh compiler scratch.
2. Sanitize inherited environment. In particular, **unset `TRANSGENIC_RUN_DIR`**: the actual encoder constructor writes `encoder_init_report.json` there (`modeling_HyenaTransgenic.py:495-500`), and the probe's forbidden-environment list does not reject it. An inherited B5 run directory would undermine the read-only intent. This is a conditional operational hazard, not evidence that the proposed clean environment already has that variable set.
3. Use the real bounded runner, no direct/manual success JSON, no automatic retries and an outer allocation wall limit. Its child timeout is not a deadline around subsequent runner-side identity hashing, which re-reads assets after the child exits.
4. Stop on missing real forward/RC/N tier coverage, version mismatch, load failure, determinism failure, numerical disagreement, OOM or timeout. Do not invent tile IDs, crop tiers, loosen comparisons or replace Hyena with a toy model.
5. After the label fix, run the single-process modes first. Only then attempt hardened four-rank smoke. Even all five successful measurements leave resource authorization and six-gate readiness incomplete.

## Verification record

- Independently verified HEAD `3b62306`; `git diff --check ac8ac2d..3b62306` returned success.
- Independently ran only the focused tokenizer reproduction above; no duplicate test suites and no CUDA measurements.
- Controller-provided result: 81 focused CPU tests in 45.58s. Writer report: latest full scoped run **233 passed, one fixture timeout**, followed by **82 focused passes including that timeout test**. These are attributed reports, not independently rerun results. **Do not call the full suite green.**
- Actual aarch64 SIF Torch `2.14.0a0+4fdf77b940.nv26.08` / CUDA `13.4`, Transformers `4.57.6`, Accelerate `1.15.0` remains untested by this review.
