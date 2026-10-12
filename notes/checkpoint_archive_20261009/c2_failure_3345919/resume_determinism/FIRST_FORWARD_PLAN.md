# First-forward intervention after3361666

Observed failure: B210→220 and C210→220 have identical per-rank inputs/RNG but609model tensors differ. The first mismatch is rank2/3 micro5041, before a resumed backward. Existing source and fixed-input comparisons clear the guard-policy change. DDP did not rebuild buckets. Generated rank2L61440forward kernels differ with initial labels18vs312 (77vs75kernels); this is not yet causal proof.

Keep the real four-rank DDP trainer, model, config, input loader, checkpoint210, strict identity, deterministic/bf16/checkpointing policy. Three separate process launches on one allocation, independent empty caches:

1. cold: first real forward5041 only.
2. primed: inside first real prehook, run exact historical Bmicro4801–4805 perrank with forward/backward, no optimizer step; restore all parameters, complete buffer registry (None/new/nonpersistent included), all RNG; measure actual5041.
3. primed_reset: same as2 then reset Dynamo before target. This preserves priming's DDP state while discarding compile history.

All cases disable local/remote FX and AOT cache lookup so filesystem caches cannot conceal the reset; this is diagnostic only. Existing generated-kernel caches remain untouched. Baseline cold must reproduce C5041, primed must reproduce B5041, and reset should return to cold logits before attributing the original discrepancy to compiler history. If those conditions fail, retain the result as a falsified/incomplete hypothesis; never a production PASS.

Outer hooks explicitly guard recursive primers, apply CompileGuard to every primer/target, verify exact historical tensor hashes and RNG, enforce memory cap, then raise only a dedicated BaseException sentinel after writing one target report/rank and collective barrier. Sentinel is caught outside real main; no target backward/optimizer/checkpoint/DONE. AcceleratedOptimizer.step is forbidden. Parameter and buffer values restored in place; replaced parameters refused. Registry restoration has CPU regression tests.

Single4GPU90minute allocation. Once-per-allocation real input verification; all source, reference, evidence, cache, original checkpoints mounted read-only. Each case receives a verified copy of B210. Final output is observations, not a production gate clearance. Production controller remains stopped and old C2/B6 held; main code remainsd7b1a37.
