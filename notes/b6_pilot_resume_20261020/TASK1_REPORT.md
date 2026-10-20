# Task 1 — fresh B6 compiler-history proof

Status: **DONE** for implementation and local verification. Actual four-GPU NGC26.08 proof, remote staging, submission and independent review remain parent-owned gates. No remote mutation or job submission was performed.

Implementation commit: `48d526bd3491549ccbe28a440aa375fc2dec6cb4` on `b6-pilot-proof-20261020`, based on `19049ba`, author/committer Won Yim <wyim@unr.edu>. This report is a subsequent scoped documentation commit; use the branch log for its exact commit ID.

## Files and behavior

- `b6_history_proof.py`: B6-only fresh launch reservation/ticket, fresh allocation verification, bounded real trainer adapter, complete comparison and ready receipt publication/rechecking.
- `b6_history_proof.sbatch`: regular `ghx4`, four GPUs, six hours; exclusive job-ID root; source storage and independent proof-copy gates; NGC source/contract/harness tests; fresh verification; continuous0→410 with snapshot400; new-process400→410; acceptance publication.
- `b6_receipt_contract.py`: Python3.6 stdlib-only strict report/ready predicates shared with the parent-owned host controller.
- `test_b6_history_proof.py`:17 tests using real small CPU safetensors/optimizer/scheduler/four-rank RNG artifacts, actual public history save/load, complete diagnostic trace fixtures, actual trace hooks and source config loader.
- `test_b6_receipt_contract.py`:5 pure host tests for report/receipt consistency, rehashed contradictory comparisons, candidate pins, exact types, Python3.6 grammar and imports without heavy packages.

Only new files under this task's notes directory were committed. `/tmp/b6-c2-recompile` remains clean at `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`; existing C2 helpers and deployed bundles were untouched.

The B6 adapter is necessary because reviewed C2 `history_proof.run` requires an existing checkpoint and hardcodes armC2. It reuses unchanged C2 `TrainingTrace`, `_trace`, `probe.clone`, canonical tensor/state hashing and full checkpoint loading. It imports dependencies through an explicit path, never CWD. No source trainer changes, reconstructed bootstrap, legacy slot requirements, historical C2 loss scalar, cache reset, cache disabling or source-policy change were introduced.

The real trainer retains `--stop-after-updates4099`, frozen config, armB6, seed123, aug_seed20261037, world4, bf16, deterministic and nonreentrant checkpointing. Only `TrainSpec.stop_at_step=410` and diagnostic checkpoint frequency10 bound execution. Source config loading injects `lr_warmup_fraction=0.05`; the verifier binds the source-loaded effective config plus its raw file SHA, preserving that full-run schedule.

Fresh continuous starts without `latest_state` or `--resume`. An exclusive prelaunch ticket reserves the empty continuous directory before the four ranks open their traces. Snapshot400 is cloned at the reviewed committed-state rename hook without interrupting the continuous process. Resumed starts only from snapshot epoch0/step9600/global400/pending0, using a separate `accelerate launch` process.

## Acceptance and receipt

All three checkpoints use public `CompileHistory.load`, matching checkpoint metadata, full identity, full runtime fingerprint and embedded digests. Bootstrap must be null, entries nonempty/all observed, and compiler events nonzero. Snapshot400 and both410 positions are exact. Meta identity digest and effective recipe are bound; all three identities/fingerprints must agree.

Every rank has exactly9840 continuous and240 resumed real forward rows, and410/10 accumulation boundary rows. Priming is excluded from counters and real trace records using the reviewed context manager. Inputs, shapes, training/grad/view context, finite exact losses, all four RNG digests, boundary gradient/RNG hashes, final canonical model (every key)/optimizer/scheduler/four-rank RNG state, and final history bytes must agree. Complete expected checkpoint file sets are required. Raw state-file hashes also bind receipt validation, so extra serialization bytes cannot hide behind unchanged canonical tensors.

Actual input hooks count nonzero evidence tracks, validate masked dummy/chain indices and count valid chain elements, record all DNA tiers and bounded witnesses with input/track/chain hashes. Summaries retain each phase/rank. The biological coverage gate is global across the proof; a naturally empty rank does not introduce an extra rejection rule. All ranks/rows remain mandatory.

Comparison JSON includes explicit B6 kind/profile/arm/source/job pins, detailed mismatch lists, state equality, file/canonical/history/trace hashes, exact trace row counts, finite-loss acceptance, metadata positions, verified record binding, effective runtime fingerprint and observed summaries. Before publication, current tensor/state/trace/history acceptance is recomputed and compared strictly with persisted comparison content. Ready publication is exclusive, original-allocation only, version1 `b6_history_validation_ready`; candidate pins include identity SHA and the required400/410 schedule positions. No C2 predecessor or original-checkpoint receipt is fabricated. Read-only rechecking can run in a later allocation; it cannot republish the original allocation's receipt.

The host contract exposes:

```python
validate_report(report, expected_source, expected_job)
validate_ready(receipt, report, expected)
```

`expected` is the independently constructed full ready envelope: `version`, `kind`, `source_commit`, `validation_job_id`, `comparison_sha256`, `verified_record_sha256`, `provenance`, `runtime_fingerprint_sha256`, `candidate`. Candidate has relative run/checkpoint, epoch/step/global_step/pending_micro and raw meta/identity/history plus canonical full checkpoint digests. Provenance has image/source-manifest/helper-manifest SHA256s. The function docstring defines trust responsibilities. These predicates reject contradictory false reports even if declared hashes are rewritten. Full tensor/current-file checks remain allocated and are additional requirements.

## Deployment contract

Stage an immutable helper release with these exact manifested members:

```text
b6_history_proof.py
b6_history_proof.sbatch
b6_receipt_contract.py
test_b6_history_proof.py
test_b6_receipt_contract.py
c2/history_proof.py
c2/probe.py
```

The last two files are byte-for-byte unchanged copies from `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/`. Set `B6_C2_HELPER_DIR=/probe/c2` inside NGC; the batch sets its Apptainer environment value. Local tests use the explicit repository-relative directory.

The source `validation_sources.sha256` must cover the frozen config, trainer and b6_runtime/history/replay/replay_state source files. The existing immutable source manifest need not list `SOURCE_COMMIT`; its exact198ce14 content is checked separately. Do not modify an accepted C2 release to satisfy staging; use unchanged manifested dependencies and a new B6 helper release.

Required batch environment: `REPO_DIR`, `OPS_DIR`, `MMAP_DIR` (existing container-visible frozen mmap directory), `REFERENCE_CHECKPOINT` (host-visible complete B6 four-rank checkpoint). The NGC image is the existing26.08 SIF. Image SHA, helper/source manifest SHA, raw config/bindings SHA, verified training input identity, actual mmap path/input metadata inventory and the actual allocation job ID are recorded freshly inside the allocation. Initial weight identity is captured by the unchanged trainer and bound across all checkpoints/comparison/receipt. Large input/model/image hashing was not run on login by this task.

Stable compiler caches are `/work/compiler_cache/b6_history_v1/{inductor,triton}`; FX/AOT stay enabled, the existing image TF32/CUTLASS/TRITON settings are preserved, and no Dynamo reset occurs.

The host first invokes the existing source storage preflight (100GiB, project22169/ddn_ssd). An independent peak check estimates seven complete reference checkpoint sizes plus2GiB for bounded logs/traces. It checks the reference includes model/optimizer/scheduler/all four RNG files and metadata, rejects symlinks, verifies project/pool and separately fits quota/filesystem free space. These checks use explicit exceptions, so `PYTHONOPTIMIZE` cannot disable them. Parent supplied metadata-only reference `/work/nvme/bilv/wcyim/transgenic/runs/b6_smoke_233574a_B6/latest_state` (9files,5,727,670,043bytes), yielding42,241,173,949 estimated bytes. The actual batch measures and records current values; it does not trust this historical size report.

A fresh job-ID root contains `continuous`, `at400`, `resumed`, fresh/verified/storage/peak records, phase/test/verify/comparison logs, comparison and validation receipt. Existing roots refuse reuse. `set -euo pipefail` propagates failures; the terminal proof marker follows successful publication and revalidation. This wrapper does not activate a run, replace any canonical run, or create pilot4099 completion markers.

## Verification evidence

TDD RED: initial8 tests failed explicitly for missing B6 implementation (`/tmp/b6-proof-red.log`). Additional regressions reproduced the optimizer serialization-byte gap, fresh-launch/manifest requirements, excessive per-rank coverage, missing recipe/RNG validation, later-allocation audit restriction, optimized-Python resource bypass and missing source effective-config binding before fixes. Pure host contract's5 tests failed for the absent contract (`/tmp/b6-contract-red.log`).

Final integrated command, from `/tmp/b6-pilot-proof-20261020`:

```bash
SLURM_JOB_ID=999999 \
PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps \
/home/wyim/miniconda3/envs/pdb/bin/python \
notes/b6_pilot_resume_20261020/test_b6_history_proof.py -v
```

Result: **17/17 PASS**,100.124s, exit0, `/tmp/b6-proof-with-contract.log`. The synthetic test allocation is isolated from inherited real Slurm IDs. Test fixture data are small CPU artifacts; no actual production inputs/model weights are hashed.

Pure contract command:

```bash
/home/wyim/miniconda3/envs/pdb/bin/python \
notes/b6_pilot_resume_20261020/test_b6_receipt_contract.py -v
```

Result: **5/5 PASS**,0.114s, exit0, `/tmp/b6-contract-verified.log`. Includes Python3.6 AST grammar and `python -S` subprocess import from `/tmp` without Torch/NumPy/Accelerate/safetensors. The host copy-peak test executes the actual batch's stdlib heredoc with real small files under `python -O`: wrong project and insufficient capacity refuse; a valid complete fixture records the independently expected peak.

Unchanged source regressions, from `/tmp/b6-c2-recompile`:

```bash
PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest \
tests.test_b6_compile_history tests.test_b6_replay_state -v
```

Result: **38/38 PASS**,42.224s, exit0, `/tmp/b6-proof-source-regression.log`.

Additional checks passed: Bash `-n`, Python compile of the four new Python files, CLI `--help` executed from `/tmp`, `git diff --check`, source worktree cleanliness/SHA, deterministic worktree `graft build` (832files; graph remains an ignored local cache). Source public API checks also confirmed source config normalization without training or model/input hashing.

## Self-review and remaining gates

Self-review covered fresh/resume ambiguity and rank races, source API/config normalization, exact full-state/trace file inventories, snapshot/counter boundaries, observed-only history, B6 actual evidence, strict receipt/report contradiction handling, provenance, CWD-independent imports, host/NGC boundary, source/helper immutability and failure propagation.

No unresolved implementation/API question remains. Local environment was PyTorch2.6.0+cu124 with CPU fixtures, not the actual NGC26.08 four-GPU runtime. Actual GPU equality, live storage reserve, image/cache behavior, Slurm wall time and independent acceptance must be established by the parent's new allocation. A failing or timed-out proof cannot authorize continuation. Independent code/spec review and remote manifested staging are intentionally not claimed complete here.
