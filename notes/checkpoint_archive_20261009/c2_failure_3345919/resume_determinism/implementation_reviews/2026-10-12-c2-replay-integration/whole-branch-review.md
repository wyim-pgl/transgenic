# Whole-branch production review

Reviewed baseline `d7b1a37` through `5f1c7c74db3aed3858e63c0c97bf9384dde6e620`, including integration `180589b`, state fixes `91c2737`, and weak-dynamic fix `789ecf4`. Scope is the production branch and its tests/specs, not the separately reviewed diagnostic proof harness. Review was read-only except this report; no jobs, remote writes, source/checkpoint changes, merges, or additional agents.

**Verdict: needs one fix before NGC/four-GPU proof.** No Critical issues; one Important issue. The required GPU scientific gates remain pending independently of this code review.

## Important: effective default cache directories are absent from the fingerprint

Location: `src/transgenic/training/b6_compile_replay.py:71-85`, particularly `cache_env` at lines82-85.

The module promises literal cache-directory binding, and the integration spec requires actual runtime settings and an explicit compatibility choice for paths that vary by node. However, the fingerprint only records explicit cache environment variables and `inductor.global_cache_dir`. When `TORCHINDUCTOR_CACHE_DIR` is unset, the effective local Inductor directory is derived from `tempfile.gettempdir()` and the username; neither the resolved directory nor all its inputs are bound. `global_cache_dir` is a separate option and does not cover this default local directory.

A targeted probe against the installed PyTorch implementation confirmed:

```text
effective_default_a: /tmp/allocation-a/torchinductor_wyim
effective_default_b: /tmp/allocation-b/torchinductor_wyim
fingerprints_equal: True
bound_cache_env: {}
```

The probe imported `runtime_fingerprint`, used the same `PretrainedConfig`, backend `inductor`, precision `bf16`, world4 descriptor, and checkpointing setting, removed `TORCHINDUCTOR_CACHE_DIR` within a process-local environment patch, and set `tempfile.tempdir` to each path within a process-local mock. It called the real side-effect-free `default_cache_dir()` and the real fingerprint. It created no cache directories or files and changed no persistent environment. Reading PyTorch's `cache_dir()` confirmed that it uses this default when the explicit environment variable is absent. Thus ranks or resumed allocations can agree on the fingerprint while using different effective caches, without the explicit compatibility policy required by the spec.

Fix: include the effective resolved local cache directory in the fingerprint, covering the unset/default case without creating a directory, mutating environment, disabling caches, or changing training math. Audit other default cache paths covered by the stated literal-path policy as part of the same narrow change. If varying paths are intentionally compatible, document and test that explicit policy instead of silently omitting them. Add a focused regression demonstrating that differing defaults cannot pass the current literal-path policy, alongside the existing effective-cache-override test. Re-review the scoped fix before proof.

## Reviewed areas without another actionable finding

- **Trainer lifecycle and math:** history is constructed after preparation and verified identity; legacy absence/override restrictions precede replay; journal integrity and position checks precede state loading. Actual train observations follow backward, so lazy backward compilation is charged to that batch. Replay uses the same prepared model and accumulation divisor. Optimizer, scheduler, clipping and counters remain outside replay. The diff does not alter training loss, optimizer arithmetic, recipe, deterministic settings or precision settings.
- **Train/eval coverage:** train uses the arm view; eval records de_novo and evidence separately under no_grad. Prepared loader reconstruction follows its actual batch sampler and original collator, retaining rank sharding/padding. Production `WP3EpochKeys` supplies deterministic epoch-keyed samples; the production evaluation keys explicitly retain epoch0. Replay checks full local input signatures and batch metadata before model entry.
- **Distributed ordering and failures:** event union includes all ranks, including non-event companions. Observation performs collective preflight before descriptor transport. Replay establishes whole-payload agreement, then coordinates local reconstruction, signature checking, RNG/mode preparation, and post-forward graph checks. Prepared forward/backward remain outside checkpoint_io. Existing Gloo tests cover actual prepared sharding/padding and a one-rank changed-input refusal. This does not guarantee recovery from arbitrary failures inside a model's already-running collective.
- **State rollback:** CPU snapshots retain original parameter/buffer/gradient objects and original storage views; complete registries, None/nonpersistent entries, pending gradients, heterogeneous original modes, prepared-wrapper modes and Python/NumPy/CPU/CUDA RNG are restored. Unsupported layouts fail before replay. Compiler and CompileGuard state deliberately survive. Sampler reconstruction restores relevant epochs, iteration and generator states, including ordinary reconstruction failure.
- **Checkpoint publication:** latest, epoch and fixed-step writers all bind history SHA and exact numerical position within their existing temporary publication transaction. Latest/epoch pending state includes gradients. Fixed-step artifacts remain weight publications with provenance, rather than acquiring Accelerate optimizer state. Existing published fixed-step artifacts are compared against the candidate, including journal bytes.
- **Legacy bootstrap:** restricted compiled C2/world4/batch1 checkpoint200 position; no existing ledger can be overridden. Rank0 validates the complete regular-file set and all declared hashes, identity/model/optimizer/scheduler/four RNG files, draw-prefix hashes/counts, safe paths and explicit provenance. All16 actual slots are retained and marked reconstructed; each reconstructed companion is compared with its original draw row before execution. Bootstrap executes under state rollback and does not write source checkpoints or draw logs.
- **Journal/schema:** canonical copied payloads, independent digest/schema validation, exact checkpoint/identity/fingerprint/world binding, distinct strong/weak dynamic annotations, safe RNG encoding and explicit reconstructed provenance are integrated consistently.

## Evidence and limits

Read the whole supplied production/test diff, the three design specs, prerequisite/integration implementation reports and progress records, and graph-selected surrounding trainer/runtime/sampler code. Inspected installed Accelerate's replicated load path and PyTorch's default cache resolver for the concrete risks above. `git diff --check d7b1a37..5f1c7c7` passed; HEAD matched the requested head and the source worktree was clean before this report.

Did not rerun broad tests. Independently read the recorded final log tail:129 tests,0 failures/errors,2 skips in280.504s. The report explains those skips as CUDA already initialized before the CPU-noninitialization check and the pre-existing missing-pyarrow fixture. The isolated state log confirms12 tests,0 skips in1.956s. Reviewed test bodies, including real two-rank integration, pending epoch-gradient next-update equality, all16 forced slots, corruption refusal and BaseException rollback. These are useful implementation evidence, not four-GPU scientific acceptance.

After the fingerprint fix, proceed to the separately authorized NGC/four-GPU proof. Required acceptance remains original200→210 historical agreement (original update210 loss3.2567708492279053), continuous/resumed full checkpoint bitwise equality with operational cache settings, future compile-event coverage, validation400 and another resume. No production-launch approval or GPU-equivalence claim is supplied by this review.
