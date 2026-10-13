# C2 version2 deployment draft review — 2026-10-12

**Disposition: conditional draft review; not deployable and not approved for initialization.** The supplied configuration is consistent with the supplied staging/evidence records except for one concrete controller compatibility blocker for retiring old job3345920. Its runtime fingerprint is now populated from actual snapshot210 but remains provisional until the complete proof passes. Matching loss210, snapshot210, or completion of the continuous branch is not proof acceptance.

Reviewed read-only:

- `/tmp/c2-controller-config-v2.draft.json`, initially SHA256 `c022d604bfb36fa6ae5a54c09a9e9f515fc436cf5236a807b98063b2ad742386` with a null fingerprint; the parent then populated actual provisional snapshot210 context, yielding SHA256 `d0892e16718d94c5475f99fc16e56fe129601035851e95fc13e4ab9de2ccb311`. Resetting only that fingerprint to null reproduces the initial hash, confirming all other reviewed fields remained identical.
- `/tmp/c2-build-config-draft.py`, SHA256 `babad296f8af5d18c786d5f0645efb50cbd896fdc083afae40ecb32bccad5ef1`.
- Supplied deployment evidence, remote receipt-pin inventory, runtime/validation staging receipts and staged local helper files; source staging/submission records and host prerequisite record referenced by the builder.

The builder was inspected, not executed. No operation was initialized, configuration rewritten, remote command run, job submitted/cancelled or source file edited. Only this review note was written.

## P2 deployment blocker: relative terminal command for3345920

The draft correctly pins the actual live command:

```text
/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a/revision/scripts/b6/wp5_pilot.sbatch
```

However, the recorded actual `SubmitLine` for3345920 ends in:

```text
revision/scripts/b6/wp5_pilot.sbatch
```

The staged controller `Slurm.observe` at `continue_c2_history.py:325` sets terminal `command` to the final SubmitLine argument. While a job is active, `:329` replaces that value with scontrol's absolute `Command`. Consequently, after cancellation the same legitimate job changes from an absolute to a relative command representation. The exact identity comparisons at `retire():822` and `retired_old():835` refuse it. This prevents activation after a successful conditional cancellation. Setting the draft pin to the relative path would instead fail the required live held-job check, so changing the pin is not a valid workaround.

A focused read-only probe used the actual staged `Slurm.observe` parser with the recorded3345920 SubmitLine and synthetic terminal accounting responses. It did not create a controller operation or invoke Slurm. Result:

```json
{
  "configured_command": "/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a/revision/scripts/b6/wp5_pilot.sbatch",
  "observed_terminal_command": "revision/scripts/b6/wp5_pilot.sbatch",
  "exact_match": false
}
```

Reproducer outline: import `/tmp/c2-runtime-stage-1dd19f0/helpers/continue_c2_history.py`; subclass `Slurm.read` to return empty squeue, `3345920|CANCELLED|0:0` for state accounting, and `3345920|wcyim|97434|<recorded SubmitTime>||<recorded SubmitLine>` for identity accounting; call `observe('3345920','wcyim')`. Its `command` is relative and differs from the draft's correct absolute pin.

Required correction is evidence-backed command normalization, preserving exact SubmitLine argv, owner/UID and all existing submission/retirement checks. The parent separately confirmed that actual sacct supplies an absolute WorkDir for these jobs and assigned the original author to resolve relative scripts against that recorded WorkDir, reject missing/ambiguous working-directory evidence, and check agreement with live scontrol Command. That successor must be tested, reviewed, newly staged and pinned before initialization. Runtime bundle1dd19f0 is historical staged evidence to supersede, not modify in place.

## Checks that passed

| Area | Reviewed result |
|---|---|
| Owner and paths | `wcyim`, UID97434 match recorded job/account evidence. WORK and SIF match the staged scripts. Original remains `runs/b6_wp5_pilot_C2`; diagnostic is `runs/b6_wp5_pilot_C2_recovery_20261011`. No canonical/backup/operator-selected active path is present. Intended operation `WORK/ops/c2_in_place_198ce14_20261012` is structurally disjoint from source/helpers and proof/validation runs; actual absence and symlink checks remain initialization prerequisites. |
| Source | Commit `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`, snapshot Git `4f12c97a3798e09200de1f012e963c193e68964f`, directory `repo_b6_198ce14`, manifest `29206d99c14ce4260d4d6505bf1321ff494315a5ba97fe3220a013f0fd5969dc` match the verified source receipt. |
| Input/config | Bindings `b6_prep_20260925/bindings/train_v1.deltaai.json` SHA `64e7fa926ecbcc74062f9cff9c3978f3bdd9de461029fd4b32dd78ee9e08eab3`; configuration `configs/b6_400m_win_v3_wp5_pilot.json` SHA `e4e7d123fbda29a36a39b3ad4ed7fe4c505767aaf7e1085fcf6c63f9a8800d41` agree with supplied remote evidence and launcher paths. |
| Runtime stage | `ops/c2_compile_history_20261012/runtime_v2_1dd19f0`, manifest `b63dc958b1f0d8288d58ddea7baf267f1a80b70f2b6fd5d5e26dede55f28f34f`. All17 staged local manifest members match their hashes and the remote receipt, including the exact configured production script and reviewed controller/pure validator. This stage needs the retirement fix successor described above. |
| Validation stage | `ops/c2_compile_history_20261012/validation_v2_61d79f0`, manifest `a751eb193b607574df6ac6e44c224c59e57b75b1fd069a23bc6f816ffc1bbb91`. All8 staged local manifest members match their hashes and remote receipt, including `history_validation.sbatch`. |
| Proof adoption | ID3369694, comment `c2-history-198ce14-5d10eaad8598`, exact absolute script and full SubmitLine match accounting/submission evidence. Blank direct sacct Comment is correctly represented through the unique SubmitLine token. Immutable helper `v1_188bf3e` and manifest `5d10eaad8598b12b964ed0f576fb345f0486f3bdd278aac63331688dc9e57012` agree with the replacement proof submission record. Source198ce14 with that unchanged helper is intentional. Failed3369338 is not the adopted job. |
| Science roots/logs | Proof root `runs/c2_compile_history_proof`; validation root `runs/c2_compile_history_validation`, matching the validation script's actual ROOT/PRIOR definitions. Validation output override `c2-history-validation-%j.out` exactly matches its configured marker-log template. Production output override likewise matches `c2-history-production-{job_id}.out`. Production phase.root is not used as the training destination: the actual active run is derived later from accepted validation evidence, and execution receipts reside under the operation's links. |
| Tests | Proof: NGC58 with1 reviewed skip, harness7 with0 skips. Validation: NGC58 with1 reviewed skip, harness18 with0 skips. Production `tests={}` is intentional because production acceptance uses its allocated receipt, not science-test logs. |
| Protected original/diagnostic | Original small meta/identity/failure-marker pins agree with provided evidence; original position is epoch0/step4800/global200/pending0. Diagnostic meta/identity/provenance pins agree, with epoch0/step6144/global256/pending0. The supplied retained JSON text was independently hashed and matched its declared pins. Original failure marker's empty-byte hash is preserved intentionally. |
| Original receipt/bootstrap | Runtime and validation original-checkpoint receipt bytes match `c7ba61600300c8467f453c5476f42af80c795a217a6b9dab803966f5f37d44da`. Their original meta/identity entries agree with the draft. Reconstructing the expected bootstrap JSON from the pinned receipt, required slots and four prefix hashes using the helper's exact no-trailing-newline serialization yields configured `894b62c2fa793b66ac23710ee32aecfcbab738cbf54b9460cbd0756ddf7c59a2`. Actual proof bootstrap still must match at acceptance. |
| Staging receipt pins | The draft's entire proof_receipts set matches the supplied remote pin inventory. Local copies of source, validation and runtime remote receipts additionally match the exact pinned raw-byte SHA values. The proof submission identity is corroborated by accounting and the separate submission-state record; that local state record is not substituted for the differently named remote proof-submission receipt. |
| Protected jobs | B5 IDs3309786–3309790 use the recorded absolute `repo_seeds/deploy/deltaai/train_b5.slurm` command and `held=false`. B6 IDs3345917/3345918 use the recorded old-source absolute launcher and `held=true`; supplied observations are owned PENDING/JobHeldUser. Old C2 ID3345920 uses its correct live command and remains a separate retirement target. All require fresh scheduler identity/state checks when acted upon. |

## Resources and pending acceptance

Validation explicitly submits account `bilv-dtai-gh`, partition `ghx4`, time04:00:00, one node, four GPUs,64 CPUs/task, exclusions `gh062,gh093` and no-requeue. The controller's explicit argv overrides the validation script's two-hour interactive header and explicit output path overrides `%x`. The parent authorized this recovery resource choice based on measured setup time plus both training/replay/save phases; no fixed user two-hour cap was identified. Production retains the reviewed24:00:00 ghx4/four-GPU/64-CPU defaults and its existing900-second graceful signal behavior. Four-rank bf16/batch1/seed123/deterministic profile and accumulation remain enforced by the reviewed helpers/controller.

`validation_clone_peak_bytes=22,910,679,660` is exactly `4 × 5,727,669,915` bytes, about21.337GiB, covering three clones plus a publication temporary. The controller independently requires at least100GiB available and that the clone peak fits; it does not sum the two requirements. Storage interpreter `/usr/bin/python3` and `repo_b6_198ce14/revision/scripts/b6/storage_preflight.py` match the supplied host/source context. Previously recorded free space is not current submission authorization.

The later provisional runtime context in `/tmp/c2-proof210-runtime-context.json` (file SHA256 `44e8745e348833907449607c84166746e6e4c2e4bf2b8086e5471e0f4dff25ec`) identifies actual proof3369694 `at210/latest_state/compile_history.json`, history SHA256 `49028ddf20a382e4778d533680540cf900760a1371654fc44260c2fcb9135b17`. The draft fingerprint equals the context fingerprint exactly; its bootstrap hash, recipe SHA and epoch0/step5040/global210/pending0 metadata agree. It records four GH200 ranks, bf16, deterministic operation, nonreentrant checkpointing and enabled local FX/AOT caches. Observed image/compiler values include `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, `TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0`, CUTLASS and Triton tool paths. Preserve these proven-context values exactly; this review does not request TF32 changes or cache disabling. The parent is separately checking the production container's effective environment, including image defaults retained through `--cleanenv`. This context remains explicitly provisional and does not establish continuous/resumed equality or terminal proof success.

Before initialization, the parent must replace the runtime helper/manifest/staging-receipt pins with the independently reviewed command-normalization successor, confirm the provisional fingerprint against fully accepted actual proof evidence, and verify the resulting final draft/pins. The inspected builder still writes that field as null; rerunning it would discard the populated context, so it must not be treated as a finalized deployment builder. Parent+batch terminal status, complete comparison/history/input/RNG/gradient/historical checks, all test logs and terminal markers remain required. Validation400 and in-place activation remain subsequent gated operations. No additional configuration-path, pin, test-count or resource mismatch was found in this draft.
