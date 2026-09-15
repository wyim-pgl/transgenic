# MVP dataset/collate increment — 2026-09-15

Base pi-agent-1765f0e2-cf6c-4ed (0514b63), reviewed model+CPU data. Approved parent spec revision/protocols/B6_MVP_PILOT_v1.md. Work only in isolated worktree. Files src/transgenic/datasets/datasets.py and tests/test_mvp_dataset.py plus this brief/report. No model/rasterizer/builder/trainer/recipes modifications, remote jobs, merge or push.

## Binding interface and decisions

Append keyword-only options to isoformDataHyena existing constructor: species=None,evidence_db=None,evidence_k=8,aug_seed=None,channel_dropout_p=0.5,thin_fractions=(0.01,0.1,0.5,1.0),eval_fraction=None,return_tile_ids=False. Parameterize species SQL in split/exclude_prefix/unfiltered branches while preserving prior precedence/order/weights. Evidence-disabled default B5 remains8tuple, no evidence DB/query/RNG usage; species-only filtering is supported.

Evidence-enabled MVP requires species='Athaliana', K=8, explicit integer aug_seed, valid finite p/fractions in[0,1], nonempty thinning fractions. Enforce strict row handling (no fallback to another rn) for evidence mode, without changing legacy strict=False behavior. Return_tile_ids needs evidence mode. Validate explicit non-null boolean is_rc from corpus, never infer from strand.

Use separate lazy read-only evidence connection and PID, like current _con but separate file. Close/reopen inherited stale handles on PID change; __getstate__ returns copied state with both handles/PIDs cleared and worker generator/tile validation cache cleared; do not mutate parent's state. __del__ safely handles partially initialized objects. No unbounded new cache beyond per-tile metadata validation.

On tile first use, verify evidence corpus_tile row matches rn, chrom, start0,end0, split,is_rc. Frozen metadata mismatches fail instead of yielding plausible zeros. Fetch corpus is_rc/split without changing default legacy row query path. Require region span len==sequence len==input token len after existing trailing-token removal; reject tokenizer truncation or additionalspecial tokens in evidence mode. Sequence already output-oriented; pass unchanged to rasterizer which RC-transforms tracks.

Call existing rasterise_est(con,sequence=region_seq,chrom=chr,offset=start,L=fin-start,is_rc=is_rc,keep_fraction=f,aug_seed=aug_seed,rn=rn,population='retained',source_supplied=not_dropped). Convert npfloat16contiguous to torch.from_numpy. Dropout gives all8zero. No new phase tracks.

Use a private CPU torch.Generator, lazily derived deterministically from aug_seed and get_worker_info().seed; mainprocess uses aug_seed. Do not consume global torch/Python/NumPy RNG in evidence augmentation. Fixedworker/order reproducibility only; exactresume/crossworker-count invariance is NOT claimed. When worker/PID changes recreate RNG, persistentworkers keepstream. Fixed eval_fraction overrides augmentation even mode='train' (labels retained), consumes no augmentation draws. Parent spec's generic torch.rand sketch is replaced by explicit private generator to avoid interference; document in report.

Default evidence sample is9tuple (legacy8 + tracks). Opt-in return_tile_ids=True produces10tuple with last {'rn':actualrn,'has_evidence':bool}. has_evidence=EXISTS tile_est for rn, not postdropoutnonzero or availchannel. Collate metadata to int64 rn and bool has_evidence vectors. This narrow extension avoids inferring IDs from sampler order and is required by future DDP pairdedup; default9tuple spec preserved.

hyena_collate_fn accepts consistent7/8 legacy and9/10 evidence, rejects mixedarity. Preserve exact legacy output. Store dna_max_len separately from label_max_len; pad[L,8]tracks onLEFT withzeros to DNAlength, stack[B,L,8]float16; validate track shape/dtype relative to each sample tokens. Tenth metadata optional. No tokenizer or decoder label contract changes.

## Tests before code

Use offline tokenizer substitution only (noHFdownloads). Real tinyDuckDB fixtures and real approved rasterizer required for integration claims; don't use a DBmock as evidence. Local pdbPython has torch/transformers/pandas but lacksduckdb; permitted to install pinned duckdb1.5.5 wheel to ignored worktree-local testdeps/venv ONLY, no shared changes. Use unittest, no pytest needed. Explicitly report if real integration unavailable.

- Baseline legacy8tuple/collate outputs and torchglobalRNG unchanged with feature off; speciesfilters threeSQLbranches correct/order/weights.
- Distinct corpus/evidenceDB and metadata mismatch/missingis_rc/corruptrows failclosed, nofallback.
- ActualRC/offset/Nmaskedtracks align with tokens; mismatched tokenizerlength rejected.
- DNA and labellength differ: tracks leftpad to DNA, notlabels, validate uniformarity/badshapes.
- p0/p1, nestedfractions, deterministicprivateRNG and globalstateunchanged; evalfraction fixed/noaugdraws; main/worker lifecycle tests.
- 9tupledefault/10tupleoptin metadata, rnreal, has_evidence preserved even dropoutp1 orNzero, metadata types/shapes.
- Pickle/spawn and PIDchange connections/generator reset; partialinitcleanup harmless.
- Run prior11model tests and feasible data regressions with exactenv commands. Worktree only, no production or realHyena claim.

Report notes/b6_mvp_dataset_implementation_report_20260915.md with redgreencommands/results, interfaces/rulings/limitations, basecommit and implementationcommit. Independent controllerreview before integration. Commit authorWonYim<wyim@unr.edu>, no nestedagents.
