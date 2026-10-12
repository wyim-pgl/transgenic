# Post-P1 execution package — controller review required

## Current gate, not a result

Controller snapshot `../initial_live_status.txt` at 2026-09-16 21:14 PDT: final P1 **3163986 PENDING/Priority**, elapsed zero; old 3163545 cancelled without allocation, zero. No complete P1 wrapper/DONE is available to this worker. No GPU follow-up was submitted or run. Do not interpret local tests or generated argv as measurement PASS. The sole remote submitter is the controller.

Canonical source is b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6, accepted source pin `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`, decoder pin `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb`. `source/` is an unmodified Git archive excerpt for API review, NOT a runnable substitute for the complete deployed source. `source-files.json` maps Git archive regular files to SHA256 under their original `/work/repo/` namespace. No reviewed source is edited.

## Exact sequence

Each row is a separate single-attempt allocation; stop after any failure. Refresh live Slurm and accounting before every submission; do not submit the next row until receipt + DONE + native validation pass. Use fresh output paths, never clean/reuse failed directories.

1. **P1 completion gate**: actual `real_tokenizer.json`, sibling `real_tokenizer.DONE` and absence of `.FAILED.json`; zero exit, no signal/timeout, correct `p1_identity`, binding, real CUDA identity, allocation 3163986, command digest. Cross-check actual S0 `s0-inputs.json` pins for prepared, consumed-code, assets, tiles, tile-availability and original `p1.argv.json`. Obtain these from the ACTUAL job directory recorded by final deployment, not a guessed job-number path. Accounting must show P1 completed; unallocated cancellation is not proof of any GPU check.
2. **P2 `real_hyena_step0_three_tiers`**: full 30720/61440/129024 inputs, deterministic eval, bitwise B5/off/zero-projection/repeat logits at every tier. No cropped stand-in. Direct Python process through `run_bounded.py`.
3. **P3 `generation_cache_parity`**: real encoder + tokenizer, greedy and beam/cache behavior, evidence routing, off parity, nonzero projection sensitivity and reorder path from the canonical implementation. This uses its actual measured 30720 batch; it is not a three-tier generation claim. Direct Python through runner. Preserve its imported modules at original absolute names through decoding.
4. **P4 `bf16_compile`**, then **`four_rank_smoke`**: BOTH use `python -m accelerate.commands.launch --multi_gpu --num_machines 1 --num_processes 4 --mixed_precision bf16 --config_file <bound accelerate_config.path> --max_restarts 0 <real_probes.py> ...`. These are not CPU boolean substitutions or one-GPU proxies. One finite optimizer update, actual public topology and real compiled kernel/no-fallback checks are required by source.
5. **`resource_budget`**, only after all short checks: same four-rank launcher. Actual inventory, all three tiers × one cold plus two steady updates × 24 microbatches per rank, disjoint rank samples, maximum label coverage, full validation, publication/revalidation/import provenance. The measured result, not an extrapolated CPU fixture, must pass canonical `validate_resource`.
6. Only then `make_launch_readiness.py generate` and `check` with exact unchanged S0 input/initialization pins, seed, consumed manifest and the six actual receipt directories. For its preflight root, expose mode-named child directories at stable paths (read-only bind mounts allowed); do not rewrite wrapper fields. Generate must pass the direct consumer before publishing readiness. `run_training.py` requires separately reviewed argv/authorization and timeout larger than measured projected runtime; **not implemented/authorized by this package**.

## Runnable tool and controller inputs

`followup.py --help` is CPU-safe. It neither submits jobs nor imports torch during planning. A plan consumes ACTUAL small files and refuses missing P1. Example invocation (paths are operator arguments, not claimed existing job locations):

```
python3 followup.py --inputs ACTUAL_CONTROLLER_INPUTS.json \
  --mode real_hyena_step0_three_tiers --output /ABSOLUTE/FRESH/PLAN
```

Inside the pinned SIF/allocation, use the same command with `--execute` and a DIFFERENT fresh output. The controller supplies an absolute output mount retained through readiness/decode. The tool saves `command.json`, `runner.json`, `inputs.snapshot.json`, `plan.json`; execution writes native `receipt/<mode>.json`, `<mode>.DONE`, measurement and log. No warmup/retry override is passed.

Controller input JSON fields (no sample fabricated paths/measurements supplied):

- `commit`: full canonical commit above.
- `s0_control`: actual small-file S0 control directory; contains `s0-inputs.json`, its five pinned files, `p1.argv.json`.
- `ledger`: actual current controller accounting JSON path (below).
- `outer_seconds`: total ONE-node Slurm allocation reservation; supplied template reserves **1800**, not just runner time.
- `timeout_seconds`: explicit reviewed mode runtime cap, positive integer, at least 120 seconds below outer cap. This is an operator cap, not a prediction that a tier/resource run will finish. Template's 1700s outer process timeout also covers native input validation; choose runner cap accordingly or STOP for another separately reviewed bounded allocation within remaining budget.
- `receipts`: mapping from `real_tokenizer` and every prior mode name to their actual complete wrapper paths. No synthetic measurements.
- `resource` for resource mode only: exact keys `budget-authorization`, `budget-sha256`, `acceptance-record`, `inventory-record`, `receipts-record`, `panel`, `sif`. Profile directory is automatically fresh `<output>/profile`.

The bound probe argv is taken from actual P1 command, keeping db/evidence/config/assets/tiles/bindings paths and file hashes, changing ONLY mode and scratch (plus explicit resource arguments). Exact executable and all seven tooling hashes are checked against Git archive; consumed manifest file hashes must match that source. Before expensive execution the native `_load_receipt` rechecks all prerequisites including imports/assets in their original namespace. Execution cannot bypass native CUDA validation.

## Slurm and namespace handoff

`followup.sbatch` is a bounded 30-minute **template**, NOT ready for submission without the controller's actual mounts/release. Controller supplies fresh `FOLLOWUP_RELEASE`, reviewed `container.argv.json`, `release.sha256`, explicit log paths and refreshed B5 node exclusions. Retain P1's account/partition/one-node/four-GPU/exclusive/no-requeue envelope. Do not invent exclusions from the old snapshot or overlap B5 nodes. Record allocation reservation before submission. Only controller runs sbatch after independent review.

`container.argv.json` must launch the pinned SIF with clean environment, original `/work/repo` source, `/work/repo/HFmodels`, hub cache, `/corpus.db`, `/mvp/inputs`, `/mvp/probe` and `/image.sif` mappings from ACTUAL S0/P1 deployment. Keep SIF/inputs/source read-only. Pass SLURM identity and `ALLOCATION_RESERVED_SECONDS=1800` explicitly through cleanenv. Command must invoke deployed `followup.py --execute` with its archived verification companions. No arbitrary shell interpolation.

**Critical cache retention:** P1 imported paths include `/scratch/modules/...`. Remount the ORIGINAL P1 cache at `/scratch` read-only for native receipt validation. Do NOT map each new scratch back onto `/scratch`: that loses the original import identity. Each followup gets a distinct persistent absolute `<output>/cache/{modules,inductor,triton,tmp,home}`. All three writable caches satisfy the canonical scratch-root containment check. Compiler caches start empty for EVERY mode; never share warm inductor/triton caches. Keep earlier mode cache/import paths accessible at exactly their recorded absolute names. Generation modules in particular remain available unchanged until panel decoder verification and decoding finish. In fact readiness revalidates all six receipts, so retain ALL previous imported files/caches, original SIF paths and source until the campaign finishes. A copy relocated elsewhere is not sufficient. No cleanup automatically occurs.

## Accounting and unresolved resource authorization

Ledger schema: `observed_at`, `expires_at` are actual epoch seconds; `controller_record_reference` names the real controller record (not an invented author); `complete_diagnostic_history:true`, `no_overlap_confirmed:true` are controller attestations backed by live queue/accounting; `jobs` contains unique `job_id`, terminal `state`, integer `nodes`, and finite nonnegative `charged_seconds`. Include every diagnostic attempt, failed allocation, actual P1, old unallocated 3163545, and any previously reserved allocation. Use actual allocated elapsed rounded conservatively or the whole reservation, never GPU-active time. No active/PENDING diagnostics are allowed when launching a followup; while P1 is pending its reservation remains committed and execution is blocked. At submission reserve full future 1800 seconds atomically in controller ledger; after finish reconcile to actual or retain conservative reservation before next job. For current job's input ledger do not count its reservation twice: all OTHER jobs are rows; current reservation is `outer_seconds`. External controller serialization is mandatory; this tool does not provide cross-host ledger locking/authentication.

Require `sum(nodes * charged_seconds) + outer_seconds <= 18000`. The deadline must cover the entire next allocation, not just script startup. Queued work whose authorization expires fails closed when allocated: refresh by controller only, never automatic extension. Template timeout/SIGTERM/Slurm backstop include setup, hashing, failed attempts and teardown. There is no retry or continuation after timeout. Tool reruns ledger checks after slow native receipt validation.

Resource training budget remains **BLOCKED/unselected**. Do not infer a training max wall or three-seed allowance from the five-hour diagnostic allowance. Real controller authorization must satisfy canonical `resource_probe.validate_budget`: schema_version=1, authorized_scope=exploratory_mvp, seeds=[1,2,3], binding digest, resource_envelope={nodes:1,ranks:4,microbatches_per_rank:24}, nonempty actual controller_record_reference/intent/margin_rationale, positive finite safety_margin>1, memory_headroom_fraction<1, max_wall_seconds, max_three_seed_node_hours, max_probe_wall_seconds, launcher_overhead_seconds, and nonnegative prior_smoke_node_hours. The latter must include previous actual/failed/reserved diagnostics; the generator refuses underaccounting. No authorization record is manufactured here.

## Seed-specific constraint

`bindings.seed` participates in the canonical SHA256. `make_launch_readiness.generate` requires ALL six receipts at that exact binding; `check` and training consumer recheck it. Thus for seeds 2 and 3, independently prepare bindings and repeat **real_tokenizer, three-tier step0, generation, bf16 compile, four-rank smoke, resource** with the seed's binding/actual sampled inputs and a separately bound controller resource authorization. Do not copy a seed-1 PASS, alter a wrapper, copy DONE or reauthorize automatically. This package intentionally anchors to actual final seed-1 P1 allocation 3163986 and does not invent successor seed P1 jobs; seed-2/3 execution requires a separately reviewed controller extension/actual new P1 inputs. If repeating all receipts cannot fit the remaining five-hour diagnostic cap, readiness for those seeds stays BLOCKED; do not widen budget or change source to evade the consumer.

## Acceptance/verification

Negative tests use actual wrapper key schema but expressly synthetic local fixtures: absent P1, FAILED, missing/bad DONE, binding/source mismatch, CPU proxy, nonzero exit, expired/insufficient/NaN accounting, pending P1. These tests certify gate code only. Native API source spans: `real_probes.py:125-206,327-478,555-696`; `run_bounded.py:130-205`; `make_launch_readiness.py:148-183,235-353`; `resource_probe.py:26-157,241-499`; `run_training.py:42-124`, all under the archive directory.
