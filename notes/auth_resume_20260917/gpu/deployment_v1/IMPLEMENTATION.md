# Development seed123 P1 deployment review candidate

## Current deliverable and status

Review and, only after controller gates, consider **`release_v2/`**. It is an immutable local review candidate, not independent review acceptance, runtime readiness, a measured GPU result or training approval. `release/` is an earlier immutable local build retained as superseded evidence. Do not transfer it. It retained the historical script's three-node exclusion list before the supplied live per-job override was reconciled. No frozen build was edited to make that correction.

All implementation artifacts are under this new `deployment_v1/`. Source `/tmp/mvp-seed123-integration-20260917`, original final_p1, original inputs and `gpu/candidate/` were not edited. No SSH, remote writes, jobs, cancellation, commits, source-worktree edits or large-input local staging. The required Graft context commands reported automatic graph index refreshes in the local repository; those are tool-managed metadata, not source-worktree changes. Advisor consultation failed with client/model-version incompatibility; no advisor approval is claimed.

Read `../PLAN.md` completely and verified every `candidate.sha256` entry. The plan is planning evidence; the current controller instruction supplies conditional replacement scope. Controller's later preflight is an observation, not a new measured result by this worker.

## Immutable identifiers

| Artifact | SHA256 |
| --- | --- |
| release_v2.tar, read-only transport archive | 5232741b2e7a0b722d9d63d3a62d23ff1cee17fd42486979b081bd17cd731ab9 |
| release_v2/release.json | 03099aa9d6e40a33e1b66365cc10d5c8a4cb7a264a2015e467baec93ebb3e281 |
| release_v2/release.sha256 | 466d23365d45718a4d3b023123ce8098161d57f5a33954bda5b6575dac88917e |
| code.candidate.tar, unchanged 103 files | da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7 |
| source-manifest.json, unchanged | 4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a |
| operator-pins.json | 45bd57640d7c55ee7ce7b1f20ab48b58729f5e58b880c4608813b99356d28c72 |
| source comparison, 89 components | 4a68eb3d299cd7e7cf7af9ce4e774f2ff18359f094d425cc9cde7becf4d67e97 |
| decoder, 14 exact DECODER_ONLY entries | a651b9fbec9e477198d15c401276d8b45240d5d16212357e1c6eef73e7a67800 |

`HASHES.sha256` additionally pins the implementation, tests, logs, documentation, read-only transport archive and both immutable build inventories. Local `/mnt/f` is DrvFS: chmod removes payload-file write permission (observed0555), but directory modes remain0777. Therefore filesystem-enforced directory immutability is not claimed. The exact reviewed file membership is sealed in `release_v2.tar`; its digest and nested ledger, not the directory mode, are the transfer identity. The archive was created exclusively with sorted names, epoch0 timestamps and numeric uid/gid0; no large inputs were included. Controller must independently record its digest and the selected release ledger digest. A ledger supplied with its own contents is integrity evidence, not independent trust or execution proof.

Source provenance is base `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558` plus exact dirty snapshot bytes, never a fake source commit. Candidate companions retain their separate accepted provenance. The archive includes `mvp_profiles.py`, panel compatibility and the separately pinned `mvp_verify_pinned_panel.py` helper, plus both producer scripts. No source files were regenerated from Git.

Source comparison uses the existing final_p1 test/reference-map selection: all files under `src/transgenic/`, `configs/`, `revision/scripts/mvp/`, and `train/train_HyenaTransgenic_MVP.py`. Decoder selection comes from the candidate's literal `DECODER_ONLY`; canonical JSON uses sorted keys, compact separators and trailing LF, as `mvp_decode_runtime.project_closure` does. Neither hash is a runtime consumed-code manifest. Installed-package/import identities and actual binding digest remain SIF outputs, not invented local receipts.

## Minimal changes and retained safeguards

- Adapted the four old final_p1 scripts in a new directory. Builder verifies the candidate ledger, exact manifest/archive pins, 103 member/hash closure, safe members, companions and small-input pins; refuses every existing destination; requests payload files0444 and directories0555. The DrvFS permission limitation described above is explicit; every payload file and the transport archive were verified non-writable.
- Host verifies candidate provenance/archive, creates fresh `dev123-p1-job-ID` and consumes only `staging/dev123_p1_release_20260917`. New logs use `dev123-p1-%j`. The existing owner namespace is deliberately retained so the already transferred evidence can be mapped without searching/copying old jobs.
- Exact observed old-job exclusions are `gh062,gh093`, copied from controller's `ExcNodeList` and `SubmitLine`, not historical sbatch default `gh062,gh093,gh086`. No B5 exclusions were randomized. Controller still reviews fresh B5 placement before submission.
- Actual prepare argv explicitly supplies `--profile development-single --seed 123`. Prepared records are checked for canonical digest, exact integer123, development profile and `run_role=mvp_finetune` after prepare and before P1. Old seed1/historical/baseline receipts are rejected. Initialization remains B5seed123/epoch05, with original checkpoint, sidecar, receipt, panel, corpus, evidence, SIF and HF pins.
- Original quota parsing, 40GiB reserve plus 5GiB staging envelope, 10000-file headroom, 2200000000-byte checkpoint ceiling, native checkpoint quota-age check, compute-only large hashes, exact cache layout/inventory, source-before/source-after/destination checks, offline cleanenv, RO source/cache/input mounts, override clearing, Slurm resources, no-requeue, no-overwrite and single-claim marker are retained.
- S0 still uses the original paired full-tile selection constant20260915. Production development augmentation is source-enforced20261037. No selection behavior was silently changed.
- Combined outer guard1700s plus20s kill grace, Slurm1800s, inner P1 guard900s remain unchanged. No performance prediction, GPU pass or runtime readiness is claimed.
- No followup launcher, training execution or evaluation data-authority transport was added. P1 does not need the later evaluation Git object.

## Conditional execution scope and remaining controller gates

`controller_execution_scope.json` records one replacement, one node/four GPUs/1800s, max0.5node-hour/2GPU-hours **inside** unchanged cumulative5node-hours/20GPU-hours, including startup, transfers, hashing, failures and teardown. `training_authorized=false`, `runtime_ready=false`, `measured=false`; schema2 training cap remains absent. No guessed budget fills any null field.

Controller's preflight file reports 2026-09-17 18:10:48 CDT (16:10:48 PDT), old3163986 PENDING, ownerwcyim UID97434, elapsed0, AllocTRESnull; old3163545 CANCELLED elapsed0. Only B5 jobs show GPU allocation in that observed since-Sep16 window. Staged evidence exists1068773376B, epoch05 weights2091010540B, SIF11120398336B; quota363.3G/500Gsoft/550Ghard and4568files. Controller separately reported the old inputs path absent. This supports zero observed prior MVP allocation in that window, not exhaustive or submission-time accounting.

Before any remote use, controller must independently accept this exact package, obtain full current accounting and pending reservations, confirm no overlap and preserve B5 jobs/account reserve/margin. Recheck old3163986 owner/PENDING/elapsed0/no allocated TRES, cancel only under controller authority, and confirm terminal cancellation. Any running/racing/uncertain state blocks replacement. No cancellation was performed here.

The host additionally fails closed unless controller supplies canonical owned `staging/dev123_p1_controller_gate.json`, externally pinned through `DEPLOY_CONTROLLER_GATE_SHA256`. No gate file is supplied or fabricated here. `validate_controller_gate` is the exact interface: scope/profile/seed/training false; selected release.json SHA; actual measurement timestamp no more than300seconds old at host entry; literal true attestations for full accounting, no overlap, independent new-package review, confirmed old owned-Pending0 cancellation, staging ownership mapping and B5 exclusion review; actual accounting/review references; finite nonnegative prior diagnostic node-hours and other pending reserved node-hours. Their sum plus0.5 must not exceed5. This is a controller attestation gate, not a scheduler query or schema2 budget validator. The accounting reference must carry the exhaustive evidence, including GPU-hour reconciliation; booleans alone do not establish facts.

A queue wait may expire that300second gate. Controller must explicitly arrange fresh review before execution; expiration means STOP, not automatic refresh, retry or second submission. The reviewed scope itself stays immutable and false for measured/runtime readiness. It is not changed into a fabricated true receipt.

Recheck namespace/staging ownership, absent new release/job destinations, evidence still at staging, source/cache small identities and live quota. Large hashes/copy occur only on compute after gates. If evidence moved into any old job, STOP; no partial-tree recovery. The already staged evidence would be renamed inside the same owner namespace into the fresh job, not recopied locally or recovered. Controller transport must preserve payload bytes/read-only modes, while providing a fresh owner-writable remote release directory for the retained exclusive `S0_P1_STARTED` marker. Do not copy that marker between releases or allocate twice.

## Exact local validation performed

From `/mnt/f/scratch/develop/Transgenic/transgenic_local`:

```sh
(cd notes/auth_resume_20260917/gpu/candidate && sha256sum -c candidate.sha256)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s notes/mvp_execution_20260916/final_p1 -p test_deploy.py -v
cd notes/auth_resume_20260917/gpu/deployment_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_candidate test_deploy -v
bash -n p1.sbatch
bash -n release_v2/p1.sbatch
(cd release_v2 && sha256sum -c release.sha256)
```

Results: **19 adapted/candidate tests PASS**, **11 original unchanged deployment tests PASS**, both shell syntax checks and all candidate/selected-release hashes pass. Tests cover candidate closure, wrong archive/manifest pins, no overwrite, explicit prepare profile/seed, old receipt rejection, conditional gate rejection, preserved exclusions, quota, archive safety, HF cache behavior, mounts, input readonly transition and original preservation. Original historical git tests were run unchanged against the original deployment, not mislabeled as candidate tests. In the adapted suite the historical export/aggregate assertions are replaced by exact candidate/aggregate checks.

TDD evidence: `tdd-red.txt` shows seven missing-adaptation failures before implementation; `tdd-gate-red.txt` shows the new execution-gate failure before that implementation; `tdd-exclusion-red.txt` shows the live-exclusion regression before correction. One intermediate preservation test failed from an incorrect test-only relative path, corrected without touching originals. Final evidence is `verification-v2-tests.txt` and `verification-original-tests.txt`; earlier logs are retained honestly, not replaced by a false pass.

Completed local implementation and verification. Blocked pending independent external review, fresh controller execution gates and all real S0/P1 measurements. Followups, full six-gate readiness, numeric schema2 training authority and training remain deferred. Graft reported estimated context savings1255137 tokens across this worker's calls.
