# Final unified source: immutable S0/P1 release — 2026-09-16

**LOCAL PREPARATION ONLY; independent deployment review remains pending.** Final source b660 is independently **ACCEPTED for staging/export, fresh bindings and separately authorized bounded diagnostics**, per `notes/mvp_execution_20260916/final_source.independent_review.md` (controller-supplied report commit `0a19b3a`, review6fb47be4). This is not training, decode, six-gate or scientific approval. The report was read; source/decoder pins agree with this package. No SSH, remote staging, scheduler queries, submissions, cancellation, GPU execution, merges or source edits were performed.

## Exact review target

New folder: `notes/mvp_execution_20260916/final_p1/`. Source commit **b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6**, executable integration **362eb899d8d43164ec1e0fb0f3b174ead9b1ba40**. The old `p1/` scripts, release, logs and job3163545 are untouched. `old_p1_preservation.json` records every old local file hash; a unit check compares the entire inventory and bytes.

| Artifact in new `release/` | SHA256 |
|---|---|
| code.tar | 0fb803cf0612623fa0023bb1f0f28b4b2f6a227a930833f50f74cd31a64e9cd2 |
| release.json | 7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689 |
| release.sha256 | 5b36bdf09a8b8eb548147654e23d6ad59fe18542642a05eb720f22be102fdfb8 |

Archive: **100 regular files, 1,832,960 bytes**. Fixed inclusion is all `src/`, `train/`, `revision/scripts/mvp/`, `configs/`, accepted `revision/protocols/b6_panels_v1.tsv`, plus exact producer dependencies `revision/scripts/07_selfconsistency_stats.py` and `24_restrict_reference_to_test_genes.py`. Every payload is compared with `git show b660bf1:<path>`; no source from the working tree, a4 branch, environments, dependencies or graph is included. Three accepted candidate JSONs remain separately pinned companions, explicitly not attributed to the source commit. Config/panel and all previous data/SIF/checkpoint pins are unchanged.

Locally recomputed source comparison pin: **555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44** (86 component files). Decoder operator pin: **b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb**. These are not prepared bindings or gate receipts. Tests compare every pre-existing772 file in the component scope against Git772 bytes, and require resource, decoder, bootstrap, windows and legacy helpers in the archive.

## Minimal deployer changes and retained execution

The old full deployment plan `notes/MVP_P1_DEPLOYMENT_PLAN_20260916.md` describes the retained quota, mount, offline-cache, sidecar, tile-selection and bounded-run implementation. Its old source/hash/path/submission choices are NOT applicable to this release. Changes here are source commit/archive additions; distinct release/job/log names; and an early missing-evidence guard. No gate implementation changed.

- Existing namespace remains `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545`.
- New submission directory: `staging/final_p1_release`.
- Exclusive per-job subtree: `final-p1-job-<SLURM_JOB_ID>`; job name `mvp8-final-p1-0916`; logs `staging/final-p1-<ID>.out/.err`.
- One exclusive ghx4 node, four GPUs,64 CPUs, no requeue,30min; combined timeout1700s with20s termination grace; inner REAL_TOKENIZER timeout900s unchanged. Initial exclusions gh062,gh093,gh086 must be refreshed by controller.
- Existing staging evidence MUST exist at `ROOT/staging/evidence/fullc0lite.db`. Missing evidence fails before creating the new job subtree. Old3163545 may already have moved it. NEVER move/copy it back, search old partial jobs, reuse an old input subtree, or delete collision markers automatically. A missing file is a blocker requiring separate reviewed recovery, not permission for another transfer.
- Compute hashes existing SIF/corpus/evidence against the unchanged byte-size/SHA pins. Only the still-staged verified evidence is renamed into the new job. Shared B5 seed123/corpus/SIF/cache remain read-only; checkpoint/cache copies and expensive hashes occur on compute, not login.
- Exact seed123 epoch_05 weights/run_config/runtime identity and eval/meta remain pinned. Epoch inventory weights/eval/meta only; absent config/runtime/epoch runtime_identity/parent config remain enforced. No best/latest/optimizer/RNG fallback. Both pinned HF layouts and revision0a629ab remain enforced and copied with source-before/source-after/destination hash checks.
- Actual pinned SIF runs `stage_checkpoint.py`, then `make_launch_readiness.py prepare`. The consumed-code manifest is freshly computed over the **full unified** SRC plus MVP entrypoint; new bootstrap/windows/decode files participate. Seven-file tooling includes resource_probe.py. Installed package, source/input/SIF and absolute canonical path bindings are made there, never fabricated locally or reused from d99.
- S0 checks full30720/61440/129024 actual train twins, retained EST, RC/N alignment; freezes prepared/assets/tiles/code inputs. Same SIF then runs **only** bounded real_tokenizer. No generate/readiness publication, P2/resource/generation/four-rank/native-bf16 execution or training.
- Expected outputs under the new job: `probe/p1_control/` quota/identity/prepared/consumed-code/assets/tiles/S0 pins/argv; `inputs/b5_seed123_epoch05/` staging receipt; `probe/real_tokenizer/` measurement/log and completed receipt+DONE or FAILED. Missing DONE, outer termination, or any drift is failure. No TRAINING_DONE.

## Strict controller handoff: no overlap

This document does not authorize submission. Final code review is accepted as scoped above; **this exact deployment/ledger still requires independent approval**. After that approval, controller must make and retain a fresh timestamped scheduler/accounting observation of old **3163545**, all own jobs and B5 chain, with ownership and state proof. Controller reports old P1 waiting with Priority as of20:08PDT; this is attributed historical status, not a live observation or cancellation permission. Do not rely on this document or an old pending observation.

1. If3163545 is PENDING: only the controller may cancel its **own still-pending** job after the two reviews, recording owner/state before cancellation and confirmed terminal cancellation afterward. If state races to RUNNING, stop; do not cancel a running job under this permission. Cancellation is not this preparer's role.
2. If RUNNING: let it finish and review its outputs/accounting. **No automatic second job**, and no final P1 while it runs.
3. If finished/failed/cancelled: review outputs, actual charged allocation and evidence location. Never reuse its d99 receipts as final-source gates. Uncertain/missing scheduler state is STOP, not cancellation or submission permission.
4. Independently verify evidence still exists in staging, new release destination absent, quota headroom, live node exclusions, controller authorization and remaining cumulative budget. No source authority transport dependency is introduced for P1.
5. Only then transfer this small immutable release into the exclusive new destination, verify the external ledger hash above and all ledger entries, and authorize at most one submission. Do not overwrite/rebuild a transferred release. Record returned jobID before any other operation; do not prequeue follow-ups. Run host helper only inside the allocation.

The original **cumulative5 node-hour /20 GPU-hour** diagnostic cap is unchanged and includes prior attempts, startup, copying, hashing, verification and teardown. This allocation costs at most0.5 node-hour /2 GPU-hours; it is not an additional cap allowance. Inadequate remaining budget means STOP. No automatic retries, cleanup, deadline extension or resource/training authorization.

## Later Git authority remains a separate gate

Panel/evaluation `frozen_reference` requires actual Git object **92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv** at its operational root. This source archive is NOT that Git object transport or authority check and contains no fabricated Git approval. Object availability in the eventual remote evaluation root is unverified and must be separately provisioned/reviewed before those consumers. **P1 does not call frozen_reference**, so this future transport requirement must not block otherwise approved P1. Retain source/import/cache paths for later runtime closure; any source/path/runtime change requires fresh appropriate bindings. P1 cannot certify P2, resource feasibility, all six gates,8192-token generation, training or scientific decisions.

## Local verification and exact independent-review files

All **9 adapted old checks plus2 new closure/preservation checks pass (11 total)**. Bash syntax, Python3.6 grammar/in-memory compile, seven ledger entries and exact Git-byte checks pass. Python3.6 grammar checking is not execution on actual Python3.6/SIF. Mount tests are mocked host construction, not successful Apptainer/CUDA checks. Initial new closure test exposed cwd-relative git ls-tree selection; corrected test to query the repository root, without modifying source or release. Advisor unavailable (client/model-version incompatibility); no advisor approval claimed.

Review all of:
- `final_p1/deploy.py`, `build_release.py`, `p1.sbatch`, `test_deploy.py`;
- `final_p1/old_p1_preservation.json`;
- `final_p1/release/{code.tar,deploy.py,p1.sbatch,release.json,release.sha256}` and its three `notes/b6_mvp_*_20260915.json` companions;
- this plan, with `final_source_report.md` as source integration context.

Local reproduction (repo root; never rebuild over existing release):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s notes/mvp_execution_20260916/final_p1 -p test_deploy.py -v
bash -n notes/mvp_execution_20260916/final_p1/p1.sbatch
(cd notes/mvp_execution_20260916/final_p1/release && sha256sum -c release.sha256 && sha256sum release.sha256)
```

Builder refuses existing `release/`. All old logs remain immutable; only the new folder and this expressly requested new plan were written. No live remote facts were observed in this task.
