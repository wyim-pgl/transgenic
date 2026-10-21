# Task 4 — B6 automatic continuation

Implementation status: complete locally; independent review, deployment, real Slurm identities, proof allocation and GPU acceptance remain parent-owned. No remote mutation, submission, cancellation, merge or push was performed.

Base: `e2bee04265a5e91255607d65f0aa4c7d07d72369`. Implementation commit: `8b13234f0246eafb09c52d915678191791307df1`. Author/committer: Won Yim <wyim@unr.edu>. This report is committed separately after implementation verification.

## Scope and architecture

Only the seven Task4-owned files under this directory are added. Source `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4` and every existing C2 tool remain unchanged. The host controller is an independent class, not a subclass of C2. It imports only the unchanged C2 standard-library Slurm, submission normalization, safe file/JSON/pin/manifest, locking, publication and terminal-accounting infrastructure through an explicit dependency directory. Source/helper manifest members stream through `source_member_sha`; control JSON retains the16MiB limit.

`continue_b6_history.py` provides `init`, read-only `status`, `tick`, `reconcile` and `watch --poll-seconds 45`. `init` exclusively claims the logical B6 operation and adopts a proof job; it never submits proof. All mutations run under `.b6_history.lock` with the exclusive `.b6_history.owner.json` claim in WORK. No force, reset, retry, deletion, C2 alias or Nice-restoration tail is provided.

The host accepts the actual Task1 `validate_report` and `validate_ready` contract. It verifies exact source/config/bindings/helper/image provenance, successful owned parent/batch accounting, the narrow extern-only exception, actual current checkpoint metadata/identity/history, every one of the eight prepared-schema files and their raw and canonical digests, and the first prepared trace record binding on every phase/rank. It does not import Torch, hash large tensors/input files/image, or treat a Boolean PASS as scientific acceptance.

The two obsolete jobs are pinned by full accounting identity and normalized full argv/WorkDir/script, retired in order3345918 then3345917, and observed CANCELLED individually. A durably recovered cancellation intent only observes; a lost response never permits another scancel. Running, released, foreign, wrong-script and reactivated accepted jobs refuse. Activation is in place and publishes only an exclusive fsynced small receipt containing the initial410 metadata/identity/history text and hashes, accepted report/READY, source/provenance, active_run and both retirement observations. No checkpoint tree is moved; RENAME_NOREPLACE is not used.

A production submission intent is persisted before sbatch. Only that invocation can submit it. Recovered intents reconcile exactly one owner/UID/token/normalized-argv/time match, without resubmitting on zero or multiple matches. The comparison uses the actual accounting WorkDir and an intent-time window. A lost first submission response can be reconciled after the live candidate advances: historical410 remains bound by the activation receipt; a fresh submission still explicitly checks live410 before sbatch. Requests and predecessor receipts are immutable, with link1 anchored to READY and later links anchored to the accepted production receipt.

`b6_production_link.py/.sbatch` are separately named copies of the reviewed producer/supervisor. The diff is limited to the strict B6/source/profile/cache namespace; B6 READY/activation/predecessor binding; existing-mmap binding; image verification inside the allocation; and raw preflight/four-rank artifact hashes in the production receipt. Existing full model/optimizer/scheduler/RNG/pending-gradient, draw-prefix, epoch evaluation, history-prefix, final model/alias/marker and launcher signal/exit validations are retained. No C2 module global or arm setting is monkey-patched. The freshly loaded ordinary source trainer is called through the existing scoped observation hooks, which restore all original functions, including a fresh original input verification on all four ranks in every allocation. No scheduler counter is assumed equal to global_step.

The fixed profile is B6/seed123/aug_seed20261037/world4/bf16/deterministic/nonreentrant/target4099, using the unchanged full-run LR schedule and stable `/work/compiler_cache/b6_history_v1/{inductor,triton}`. FX/AOT remain enabled; the NGC26.08 image defaults are preserved. Per-link rank observations live in the request's execution directory; proof prepared-schema/trace files are never overwritten.

The host verifies all emitted preflight/four-rank artifact bytes against their receipt hashes, current small checkpoint bindings, unchanged history prefix and full identity, sampler/epoch/pending-gradient arithmetic, completed epoch metadata/evaluation and observed losses/data order. It requires positive progress for preemption and exact4099 for final acceptance. `final_verified` rechecks before `complete`; periodic latest4000 remains valid alongside final4099. No continuation is accepted on nonzero launcher exit, missing rank, failed marker, no receipt or no progress.

## Configuration and deployment contract

The exact config schema is in `validate_config`; the real-file fixture in `test_continue_b6_history.py` constructs a complete example. `proof_root` excludes the numeric proof job directory. `active_run` is derived as `proof_root/proof_job_id/resumed`; the operator cannot select it. `proof_identity` and each ordered `obsolete_jobs` entry include job_id, owner, uid, created (Slurm timestamp), token, full argv, absolute accounting work_dir and full normalized command. Owner is wcyim/UID97434; the CLI checks its process UID. The parent supplies source snapshot Git SHA, source/helper manifests, raw config/bindings SHA, NGC image SHA, expected runtime fingerprint and explicit reviewed resources.

Resources require account `bilv-dtai-gh`, partition `ghx4`, four GPUs, one node, explicit CPUs/exclusion/output, no requeue and a walltime no greater than48h. Project22169/ddn_ssd must have both quota and filesystem headroom of at least100GiB at activation, each submission and allocated launch. The unchanged storage helper's insufficient-headroom diagnostic becomes a persisted retryable wait; malformed/wrong-pool observations remain blocks. The independent proof-copy peak is never added to this100GiB reserve.

The proof release must retain all eight Task1 manifested members (both proof Python/test pairs, batch, and unchanged `c2/history_proof.py`, `c2/probe.py`, `c2/production_link.py`). The production release must manifest:

```text
continue_b6_history.py
b6_production_link.py
b6_production_link.sbatch
b6_receipt_contract.py
c2/continue_c2_history.py
c2/probe.py
```

The `c2/` members are byte-for-byte copies from the reviewed existing C2 directory. Additional test files may be manifested. Source `validation_sources.sha256` must include the frozen config, trainer, b6_runtime, b6_compile_history, b6_compile_replay, b6_replay_state, and the unchanged `revision/scripts/b6/storage_preflight.py` executed by both host and batch. Source `SOURCE_COMMIT` is independently checked; do not alter198ce14 trainer code or a deployed C2 helper release. The producer's mmap_dir is derived from accepted proof verification, bound in each request and checked against the actual cache object before each rank calls the original verifier.

CLI examples:

```bash
python3 continue_b6_history.py init --config /absolute/reviewed-config.json --state-dir /absolute/WORK/new-b6-operation
python3 continue_b6_history.py status --state-dir /absolute/WORK/new-b6-operation
python3 continue_b6_history.py watch --state-dir /absolute/WORK/new-b6-operation --poll-seconds 45
```

There is no new user-approval gate for the already authorized unchanged pilot. Ambiguous scientific/ownership state exits watch; transient observations and insufficient reserve persist as waits. A recovered unexecuted submission/cancellation intent can remain unresolved indefinitely: absence of a job or cancellation is deliberately not permission to retry.

## Verification

All commands ran from `/tmp/b6-pilot-proof-20261020` with local Python3.12.2/PyTorch2.6 CPU fixtures and the existing source/dependency overlays:

```bash
SLURM_JOB_ID=999999 PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps /home/wyim/miniconda3/envs/pdb/bin/python notes/b6_pilot_resume_20261020/test_continue_b6_history.py -v
SLURM_JOB_ID=999999 PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps /home/wyim/miniconda3/envs/pdb/bin/python notes/b6_pilot_resume_20261020/test_b6_production_link.py -v
SLURM_JOB_ID=999999 PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps /home/wyim/miniconda3/envs/pdb/bin/python notes/b6_pilot_resume_20261020/test_b6_link_contract.py -v
```

Final results: **35/35 PASS**: host19/19 (82.361s, `/tmp/b6-task4-host-verified.log`), producer10/10 (88.678s, `/tmp/b6-task4-producer-final.log`), chain6/6 (70.515s, `/tmp/b6-task4-chain-verified.log`); every command exited0. Tests consume the actual committed Task1 small CPU checkpoint/trace fixture and publisher, with real Git trees, manifests, proof receipts, activation, requests, preflight, model/optimizer/scheduler/RNG state, pending gradients, epoch evaluation and final artifacts. Only Slurm/storage and unavailable GPU allocation execution are simulated. The compiled/tied export test uses actual torch.compile, Accelerator.save_state and source deterministic export/pending-gradient APIs. Accelerate emits its expected saved-alias diagnostic for `_orig_mod.a_weight`; this is the tested first-storage-alias behavior, not a production warning/fallback waiver.

The full chain exercises410→epoch1/global416/pending17→4099, nine completed epoch evaluations, a second immutable predecessor request, all rank receipts, latest4000 and final4099, unchanged proof siblings and embedded410. Crash tests cover lost cancellation response, recovered unexecuted cancellation intent, partial retirements, activation publication before journal advancement, concurrent locking, zero/multiple submission matches, exact-match adoption, lost first sbatch response after live checkpoint progress, and accepted-job reactivation.

TDD evidence: `/tmp/b6-task4-red.log` initially12 missing-controller failures; producer RED had5 missing-producer failures (plus incidental discovery of the host class, corrected before final runs); `/tmp/b6-task4-chain-red.log` caught host acceptance after deletion of a rank record; `/tmp/b6-task4-requeue-red.log` caught reactivated accepted-job resubmission; `/tmp/b6-task4-mmap-red.log` caught missing explicit mmap binding; `/tmp/b6-task4-progress-reconcile-red.log` reproduced rejection of legitimate progressed-checkpoint reconciliation. Each corresponding final regression passes.

Additional checks: Python3.6 AST grammar for both executable Python files; isolated `python -S` import/CLI from `/tmp` with no Torch/NumPy/Accelerate/safetensors imports; Bash syntax; Python compile; scoped diff whitespace; deterministic graft refresh; exact198ce14 source SHA and clean source worktree. No unchanged source-state regressions were rerun because source behavior is unchanged.

## Self-review and limitations

Self-review inspected the producer/supervisor diff against the unchanged C2 originals, receipt trust boundaries, full script normalization, all mutation crash boundaries, logical ownership, live-vs-historical candidate checks, small-artifact binding, environment/cache/image provenance, epoch/pending-gradient/final arithmetic and failure propagation. No unresolved implementation/API question remains.

Actual Python3.6 execution, NGC26.08 four-rank CUDA execution, real Lustre quota and accounting observations, GPU proof equality and production throughput remain parent-owned validation gates. Local tests do not establish a scientific GPU PASS or authorize a replacement proof after failure. The next task after actual `complete` remains the existing WP5 pilot-panel evaluation.

## Independent review P1 — first retirement correction (superseded by round2)

**Correction:** round1 incorrectly treated state/user flags alone as controller-side predicates. Its simulated scheduler made the same assumption. Round2 below supersedes that claim and model; round1 did not close the client-filter-to-kill-RPC race.

Review base: `b4ab09ca8435d8477818817fb0668935427d49a8`. Fix commit: `01c3bc0ff97bc849498e73f274bcf35348f5c065`. The review reproduced an observation/cancellation race: the original B6 argv contained only `scancel <job-id>`, so a held job starting after the last observation could be cancelled while RUNNING. This supersedes any earlier implication that client-side held checks alone protected the cancellation boundary.

The durable per-job intent now records exactly `['scancel', '--state=PENDING', '--user='+owner, job_id]`, restoring the unchanged C2 scheduler-side filters. Full pinned identity and PENDING/JobHeldUser checks before the call, ordered retirement, terminal-CANCELLED observation, durable intent and no-retry recovery remain unchanged. A no-op filtered cancellation is not retirement evidence and cannot advance activation.

The new regression uses the actual controller, real Task1 proof artifacts and the on-disk journal, and a scheduler boundary that changes3345918 to RUNNING immediately before applying the cancellation. It verifies the job remains active, the exact filtered argv is durably recorded, recovery refuses the RUNNING job without another cancellation, the retirement list remains empty, no activation/sbatch occurs and3345917 remains untouched. The existing initial-refusal test now explicitly includes a released-but-PENDING `Reason=Priority` case alongside RUNNING, foreign UID and mismatched full script.

RED command: the host test command below with `ControllerTests.test_job_starting_after_final_observation_is_not_cancelled_or_activated -v` selected. Result:1 test,1 failure,13.967s, exit1 (`/tmp/b6-task4-r1-race-red.log`): the simulated RUNNING allocation was cancelled under the prior argv. GREEN verification:

```bash
SLURM_JOB_ID=999999 PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps /home/wyim/miniconda3/envs/pdb/bin/python notes/b6_pilot_resume_20261020/test_continue_b6_history.py -v
```

**20/20 host tests PASS**,89.035s, exit0 (`/tmp/b6-task4-r1-host-green.log`). Python compile and scoped diff whitespace checks also pass. The producer, batch/signal supervisor and integration-chain implementation are unchanged, so their previously passing checks were not rerun. No source, C2, graph output or remote state was modified.

Scheduler limitation: these server-side filters constrain allocation state and username; they do not atomically filter `Reason=JobHeldUser`. A release that leaves the job PENDING between the last held observation and scancel remains a reason-observation race. No unsupported reason filter or stronger atomic live-scheduler guarantee is claimed. Initial or subsequently observed released/running/foreign jobs still refuse; the restored filters specifically prevent cancellation of a job that has become RUNNING at the mutation boundary.

## Independent review P1 — round2 controller-side filtering

Review base: `e4bb9a288fe28fe82d9a0ad646070ec6fa2f0ea3`. Fix commit: `2a48265712f960859a86edfa5e5c1f26eb8fdecf`. The scoped reviewer corrected the initial recommendation: ordinary scancel loads and filters a client-side snapshot before sending job-ID kill RPCs without the predicates. Thus `--state=PENDING --user=...` alone leaves a race inside the CLI. The supplied review links the [SchedMD performance documentation](https://slurm.schedmd.com/scancel.html) and upstream `_proc_cluster`, `_filter_job_records`, `_cancel_job_id` and `_ctld_signal_jobs` paths establishing the distinction. The prior report's server-side description and first test's atomic-filter model were incorrect.

The durable intent now records exactly `['scancel', '--ctld', '--state=PENDING', '--user='+owner, job_id]`. No interactive, sibling or step-ID options are supplied. This chooses the controller request carrying the state/user/job predicates. Full pinned submission identity, the final held check, sequential retirement, terminal-CANCELLED observation and no-retry recovery are unchanged. Historical recovered intents are still observation-only and are never reissued.

Parent-owned read-only capability evidence is `notes/b6_pilot_resume_20261020/SCANCEL_CAPABILITIES.json` in the main workspace, observed2026-10-21T00:42:28.973313Z: installed `scancel --version` reports **slurm25.11.8**, and `--help` reports `--ctld` (“send request directly to slurmctld”), `--state` and `--user`. This task did not perform another remote command or a live cancellation.

The scheduler test now distinguishes the two paths: it first selects a PENDING client snapshot, starts the job before the kill RPC, then uses stale selection without `--ctld` or re-evaluates the request predicates at the controller boundary with `--ctld`. Against round1 code, the actual controller cancels that RUNNING job and the regression fails. With the fix, the job stays RUNNING, the exact `--ctld` argv is persisted, recovery refuses without retry, the retirement list remains empty, the controller remains `retire_waiting`, no activation/submission occurs and3345917 is untouched. Initial released-pending, foreign, running and mismatched-script refusals remain covered.

RED command: the host command below with `ControllerTests.test_job_starting_after_client_filter_snapshot_is_not_cancelled_or_activated -v` selected. Result:1 test,1 failure,12.175s, exit1 (`/tmp/b6-task4-r2-ctld-red.log`). GREEN command:

```bash
SLURM_JOB_ID=999999 PYTHONPATH=/tmp/b6-c2-recompile/src:/tmp/b6-spec-review-deps /home/wyim/miniconda3/envs/pdb/bin/python notes/b6_pilot_resume_20261020/test_continue_b6_history.py -v
```

**20/20 host tests PASS**,129.677s, exit0 (`/tmp/b6-task4-r2-host-green.log`). Python compile and scoped diff whitespace checks also pass. Only controller/tests/report change in this round; unchanged producer, batch and integration-chain suites were not rerun. No source, C2, graph output or remote state was changed.

`Reason=JobHeldUser` still has no claimed atomic controller predicate. A release that leaves the allocation PENDING after the final held observation remains a reason-observation race; `--ctld` closes the specifically reproduced PENDING-to-RUNNING race without inventing a hold-reason filter. Scientific GPU acceptance and all other parent-owned runtime gates remain outstanding.
