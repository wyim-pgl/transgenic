# Executable S0 + P1 handoff — 2026-09-16

**Delivered for independent review, NOT submitted.** Only local notes/scripts/bundle were written. No remote writes, large remote hashes, GPU imports on login, jobs, source patches, merges, pushes or subagents. This is a single diagnostic allocation, not training authorization.

## Controller decision now

Use `notes/mvp_execution_20260916/p1/release/` after independent review. It contains an actual generated archive, checksum ledger, deployment helper and sbatch, not placeholder commands. The helper runs S0 then the unchanged reviewed `run_bounded.py → real_probes.py --mode real_tokenizer` once.

This release is **strictly d99b1f8a2a42edfe4877788d0f7d1ad7f9d53e50**. Combined resource code was partial at 544bcb0 when recovery resumed; its separate agent is continuing. This is not a review approval of that combined snapshot. Do not wait for it merely to review these scripts. If the combined snapshot becomes reviewed before submission, controller should regenerate/re-review a different immutable release and its bindings. **A d99b P1 receipt is diagnostic-only and cannot be reused as the final combined snapshot's gate:** changed consumed code/tooling changes bindings. No compatibility/reuse claim is made. This bundle does not contain resource mode.

### Fixed bundle identities

- `code.tar`: `2a105e9d2827098d1947c3b8e7e58b689cae0e92bb52e911f100e78b8a3c895c` (1,638,400 bytes; 85 regular source files).
- `release.json`: `0ec3776350d4f19226c8fbee30892f28996fa11a604ec0bfaa2e1dcb99f17ae5`.
- `release.sha256`: `34a094c321a59ae4e0613c96d6797a2fdca32284682a2ed07074ca5ee52d203b`.
- `release.json` pins every archive regular file, helper/sbatch, and three accepted JSONs. Bootstrap checks the checksum ledger; controller must compare its hash to this document before execution.

Archive includes exact Git bytes for **all `src/`, all `train/`, `revision/scripts/mvp/`, `configs/`, and the accepted panel TSV**. The builder compares archive member bytes against `git show d99b:<path>` and rejects symlinks/special entries. No working-tree source edits or dirty files enter it.

**Provenance correction discovered during assembly:** the three accepted candidate JSONs are untracked main artifacts and do not exist in d99b. They are included separately, with the fixed canonical SHA pins from the deployment report, explicitly labelled `accepted_companions_not_in_source_commit`. Panel/config are from d99b. Do not falsely attribute the accepted JSONs to that commit. All five small inputs are verified by fixed hashes in `build_release.py`; source/runtime use their staged `/work/repo/...` paths.

## What executes within the 30-minute allocation

`p1.sbatch`: bilv-dtai-gh / ghx4, **one node, four GPUs, 64 CPUs, exclusive, no requeue, 00:30:00**. No memory override, dependency, cancellation, warmup or resubmission. Initial exclusions gh062,gh093,gh086; controller refreshes live occupied nodes before submission. Single controller invocation claims `S0_P1_STARTED` exclusively. Existing outputs cause failure, not resume.

Combined S0/P1 process-group timeout is **1700 seconds + at most 20 seconds termination grace**, inside Slurm's 1800-second limit. Inner actual probe timeout is **900 seconds**; outer timeout also covers copying, hashing, imports, selection and runner's receipt validation. If S0 consumes too much time, P1 can be killed earlier by the outer deadline. No extension or automatic second allocation.

1. Require real one-node Slurm allocation and controller-owned canonical existing root `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545`. Exclusively create `p1-job-${SLURM_JOB_ID}` within that root, then fresh `release`, `code`, `inputs`, `cache`, `probe` children there. Existing namespace siblings are untouched. Copy only the manifest-listed release artifacts plus manifest/ledger into the job's release, verify the copied payload hashes and freeze it. An existing job subtree is a collision/blocker; do not delete it to force a retry.
2. Execute actual `/sw/user/scripts/quota -s` **on compute**; retain command, time, stdout/stderr/exit and derived quota record. Parse only the observed `/work/nvme/bilv` allocation table. Unrecognized format, warnings, missing/duplicate rows, unknown units, exhausted file quota or insufficient space fail closed. No substitution of `df` for quota.
3. Conservative rounded quota bound: used is IEC plus one displayed quantum; soft/hard limits are SI minus one quantum, then take min with filesystem free. Require **40GiB reserve + 5GiB new staging envelope**, with >10,000 file quota headroom. Re-measure immediately before checkpoint copy; reviewed stage enforces age≤3600s, copy≤2,200,000,000B and reserve40GiB. The reserve is a planning margin, not a measured B5 growth guarantee or group reservation.
4. Verify/extract immutable source; fresh compute hash/size-check existing corpus23,534,252,032B (`6840c93f…`), SIF11,120,398,336B (`1a5dee27…`), transferred evidence1,068,773,376B (`ed2d0eef…`). Corpus/SIF stay in place, reused read-only. Only after hash PASS, rename controller's transferred evidence from namespace `staging/evidence` into new per-job `inputs/evidence`; never overwrite B5 data.
5. Independently materialize just HF commit `0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446`, both tokenizer and hub encoder layouts. Check exact observed 3/4 snapshot inventories and refs. Resolve links inside their source model roots, copy bytes, compare source-before/source-after/destination hashes; output contains no symlinks or shared B5 blobs. Source cache is never written. Freeze new cache files read-only.
6. Inside pinned SIF call actual reviewed `stage_checkpoint.py`, source `/source/seed123/epoch_05`, destination `/mvp/inputs/b5_seed123_epoch05/checkpoint`, preserving parent run_config/runtime_identity layout. Enforce weights `6e5e4086…`, run_config `c3ca0b4d…`, parent runtime `44b6f578…`; epoch meta `844ecbc4…` and eval `0df4e4a7…`. Exact epoch inventory is weights/eval/meta only; config.json/runtime.json/epoch-local runtime_identity and parent config.json must be absent. Reviewed optional-sidecar map must be config=None, runtime=None, meta=132B with selected hash. No `best`, latest, optimizer or RNG resume.
7. Inside **the actual SIF**, invoke `make_launch_readiness.py prepare` with full explicit CLI. It generates the real consumer's `_expected_code_files` closure and installed external-package manifest, tooling, source-checkpoint and input bindings. **Never invoke `generate`, fabricate six receipts, or publish training readiness.** Source/config/panel/candidate/package/tooling pins are executable checks.
8. Generate `assets.json` using actual `real_probes.tree_identity` at `/work/repo/HFmodels` and `/work/repo/HFmodels/hub`. Build tile candidates from accepted JSON TRAIN records, then verify actual Athaliana TRAIN dataset membership and DB geometry. One SQL scan identifies full length/N availability; select real same-interval false/true twins with nonzero retained EST signal. Run actual `check_tracks` and `measured_batches` (full30720/61440/129024; no crops; actual tokenizer alignment, RC channel permutation, ≥one N). Emit `tile-availability.json` with attempts/selected RNs or BLOCKED reason. Impossible selection stops, never invents IDs. Local accepted inventory contains 1346/975/564 complete geometry pairs; this is **not** a real EST/N availability result.
9. Pin/freeze prepared/assets/tiles/consumed-code records in `s0-inputs.json`. Freeze new inputs and relaunch the same SIF with `/mvp/inputs:ro`. Verify S0 pins and fresh empty compiler directories. Generate exact absolute-interpreter `p1.argv.json`, hash it, and invoke reviewed `run_bounded` once. P1 itself checks actual Torch2.14.0a0+4fdf77b940.nv26.08/CUDA13.4, Transformers4.57.6 and Accelerate1.15.0 parser; real model/tokenizer, six full tile alignments and a CUDA forward. Actual loaded Python/extension/HF module files are hashed by the unchanged probe, in addition to full source closure. No GPU warmup occurs outside bounded execution.

### Container mapping / environment

In the table, new paths are relative to `ROOT/p1-job-${SLURM_JOB_ID}`; corpus/SIF/B5 source paths remain the fixed absolute paths in the helper. Both invocations use `/usr/bin/apptainer exec --nv --writable-tmpfs --cleanenv --containall --no-mount hostfs --pwd /work/repo`.

| Host artifact | Container path | Access |
|---|---|---|
| new `code/source` | `/work/repo` | ro |
| new `inputs/hf` | `/work/repo/HFmodels` | ro |
| existing corpus | `/corpus.db` | ro |
| exact existing SIF | `/image.sif` | ro |
| new `inputs` | `/mvp/inputs` | rw for S0 checkpoint creation; ro for P1 |
| concrete B5 run root | `/source/seed123` | ro, **S0 only** |
| new `probe` | `/mvp/probe` | rw |
| new `cache/p1` | `/scratch` | rw |
| transferred release | `/deployment` | ro |

The SIF defaults are not overlaid by a writable B5 checkout/home. Environment explicitly sets offline HF and both cache layouts; writable modules/Inductor/Triton/tmp/home live under `/scratch`. `TRANSGENIC*`, container-injection variables and Python/cache overrides are removed in the sbatch, including inherited `TRANSGENIC_RUN_DIR`; LD_PRELOAD/LD_LIBRARY_PATH are removed. No runtime pip/network fallback. Python bytecode writes disabled. Host orchestration is stdlib compatible with observed Python3.6.15; SIF Python is selected inside the image and recorded as the absolute `sys.executable` in bound argv.

## Controller-only transfer and submission (NOT executed here)

Review `deploy.py`, `p1.sbatch`, release manifest and this plan first. Commands below create only a small handoff on pronghorn, then the new Delta release child. Do not copy corpus/SIF again. They assume these exclusive destinations do not already exist; collision means inspect, not overwrite.

```bash
# Local controller shell, repository root; STOP on any command failure:
set -euo pipefail
ssh -o BatchMode=yes pronghorn 'mkdir -m 700 /tmp/mvp_p1_d99b_20260916'
rsync -rt notes/mvp_execution_20260916/p1/release/ \
  pronghorn:/tmp/mvp_p1_d99b_20260916/
ssh -o BatchMode=yes pronghorn "ssh -o BatchMode=yes dtai 'mkdir -m 700 /work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/p1_release'"
ssh -o BatchMode=yes pronghorn \
  'rsync -rt -e "ssh -o BatchMode=yes" /tmp/mvp_p1_d99b_20260916/ dtai:/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/p1_release/'
```

Then use one controller shell on DeltaAI (through pronghorn → dtai). Do not run `deploy.py` on login. These are small ledger hashes and scheduler queries only:

```bash
set -euo pipefail
cd /work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/p1_release
printf '%s  %s\n' 34a094c321a59ae4e0613c96d6797a2fdca32284682a2ed07074ca5ee52d203b release.sha256 | sha256sum -c -
sha256sum -c release.sha256
date -Is
squeue -u "$USER" -o '%.18i %.32j %.9T %.10M %.6D %R'
quota -s
# Exclude all current running jobs of this user, plus persistent problematic nodes.
# Repeat the queue review immediately before this one submission; preserve B5 chain.
EXCLUDE=$(/usr/bin/python3 -c 'import subprocess; s=subprocess.check_output(["squeue","-h","-u","wcyim","-t","RUNNING","-o","%N"]).decode(); print(",".join(sorted(set(["gh062","gh093","gh086"]+[x.strip() for x in s.splitlines() if x.strip()]))))')
# ONLY after independent deployment review and controller authorization:
JOB_ID=$(sbatch --parsable --exclude="$EXCLUDE" p1.sbatch)
printf 'Submitted exactly once: %s; budget 0.5 node-hour / 2 GPU-hours\n' "$JOB_ID"
```

Controller records the returned exact job ID and allocation budget before doing anything else; no prequeued follow-up. The release's exclusive started marker blocks second execution. No worker touches/cancels/submits B5 jobs. A partial S0 timeout can leave a moved evidence file or new read-only inputs; controller must inspect and authorize a separately reviewed recovery, not rerun/remove collision guards.

**Budget:** this one S0+P1 allocation≤0.5 node-hour /2 GPU-hours. Prospective cumulative diagnostic cap≤5 node-hours /20 GPU-hours, not five hours of training authorization. No P4 or training is launched here. Hash passes, tile selection, full-model load and all imported-module hashes may not fit30min; there is no measured throughput guarantee. Any deadline/OOM/import/cache/identity failure is a diagnostic failure. Full training still needs the final reviewed combined snapshot, six fresh gates and separate measured three-seed budget authorization.

## Expected receipts / interpretation

Receipt paths below are relative to `ROOT/p1-job-<returned-jobid>`; Slurm logs remain under namespace `staging/`.

- `probe/p1_control/`: raw quota records, large identities, prepared bindings, actual SIF consumed-code manifest, asset identities, selected tiles/availability, S0 pins and exact bound P1 argv.
- `inputs/b5_seed123_epoch05/`: reviewed staging receipt and `STAGING_DONE`, with concrete weights/sidecars.
- `probe/real_tokenizer/`: runner log, measurement and either measured `real_tokenizer.json` + `.DONE`, or `.FAILED.json`. Outer allocation termination can prevent failure JSON; absent DONE is never success.
- Slurm logs: `staging/p1-<jobid>.out/.err`. No `TRAINING_DONE` or launch-ready six-gate record is produced.

## Remaining one-time operations inputs (not supplied by a decoder)

1. Independent review of this exact helper/sbatch/ledger and per-job-path change; controller records reviewer disposition and selects **this d99b diagnostic** or a separately rebuilt/re-reviewed final immutable combined snapshot. No later-code binding is certified by this bundle.
2. Controller identity and explicit one-job S0+P1 authorization/budget (30min, 0.5 node-hour, four GPUs); record the returned job ID. Existing broad authorization does not manufacture a successful review or measured gate.
3. Fresh read-only queue/node/exclusion and quota observation immediately before the sole submission. Preserve the B5 chain. Verify the exclusive transfer destinations are absent and the already transferred evidence exists at namespace `staging/evidence/fullc0lite.db`; no extra database transfer is requested. Its expensive byte/hash check and the authoritative fresh quota checks run on compute.
4. After the job, inspect Slurm exit/accounting, staging receipt, tile availability and bounded probe DONE/measurement bindings. Any absent tier, absent N, wrong RC, identity drift, cache/mount/import failure, quota failure, timeout/OOM, collision or missing DONE means STOP; retain partial output. No automatic retry, cleanup, fallback RN, substituted tokenizer/version or resource/training approval.

No decoder input is needed for S0/P1. Actual cache writability, image bind mount support, runtime versions, full tiers and GPU execution remain compute checks, not local-test claims.

## Exact local reproduction and changed-file ledger

Run from the repository root, without rebuilding over an existing release:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s notes/mvp_execution_20260916/p1 -p test_deploy.py -v
bash -n notes/mvp_execution_20260916/p1/p1.sbatch
PYTHONDONTWRITEBYTECODE=1 python3 -c "import ast,pathlib; p=pathlib.Path('notes/mvp_execution_20260916/p1'); [(ast.parse((p/n).read_text(),filename=n,feature_version=(3,6)),compile((p/n).read_text(),n,'exec'),print(n,'PASS')) for n in ('deploy.py','build_release.py','test_deploy.py')]"
(cd notes/mvp_execution_20260916/p1/release && sha256sum -c release.sha256 && sha256sum release.sha256)
```

The builder still refuses an existing destination. Recovery preserved the original release at `/tmp/p1-original-release-8knast5h/release`, then regenerated into an absent local `release/` using `PYTHONDONTWRITEBYTECODE=1 python3 notes/mvp_execution_20260916/p1/build_release.py`. The first attempted cross-filesystem rename failed without modifying the old release; `shutil.move` then preserved it successfully. The supplied `notes/usage_recovery_20260916/p1_partial.tgz` was not modified.

Changed files in this resumed task (paths relative to `notes/mvp_execution_20260916/p1/` unless stated):

| File | SHA256 |
|---|---|
| `build_release.py` (resolve repo locally, no temporary-worktree dependency) | `e68a4910fd1c65748d2055e9853037cca46bd2d3a8539c0467d0a2d806dad21d` |
| `deploy.py`, `release/deploy.py` (per-job release/output; existing namespace preserved) | `1c9d2084b48c2c55b014c6fc9586b14561a66376a4ed903035bb39a6eb5b82cd` |
| `test_deploy.py` (three additional checks) | `edecdf9f3ac6db2ee1a92b7d13fa470ed17b6a5e9987cc1d19260d9a8518bbbf` |
| `release/release.json` | `0ec3776350d4f19226c8fbee30892f28996fa11a604ec0bfaa2e1dcb99f17ae5` |
| `release/release.sha256` | `34a094c321a59ae4e0613c96d6797a2fdca32284682a2ed07074ca5ee52d203b` |
| `notes/MVP_P1_DEPLOYMENT_PLAN_20260916.md` | hash reported externally after final write; not self-referential |

`p1.sbatch` and `release/p1.sbatch` are unchanged (`d539d032f39f3759a34d543f1c0c353e2183804878edbd96bb6093f5ee968c1b`). Regenerated `code.tar` and all three accepted companion JSONs are byte-identical to the previous release; their complete hashes are in `release.sha256`. No generated dependency/environment trees are in the release. Main source and AGENTS were not edited.

## Read-only live observations and local checks

At **2026-09-16 19:04:51 CDT (17:04:51 PDT)** Delta queue no longer listed3131537;3131538 was PENDING Priority and3149356–3149363 PENDING Dependency. This is not an accounting/completion claim. Allocation quota350G/500Gsoft/550Ghard;4482files. Previous gh086 occupancy is stale: refresh before submission. Exact commands found: `/usr/bin/apptainer`, `/usr/bin/python3`, `/sw/user/scripts/quota`, `/usr/bin/timeout`, `/usr/bin/sbatch`, `/usr/bin/squeue`. Additional read-only query confirmed Python3.6.15 and canonical root/corpus/SIF paths. No remote file writes occurred.

Resumed local verification: **9 stdlib unit tests PASS**, `bash -n` PASS, Python3.6 grammar and in-memory compile checks PASS; all seven release-ledger entries PASS. New tests verify per-job namespace preservation/collision refusal, SIF/corpus/B5 read-only mounts, stage-only writable inputs, isolated scratch, no TRANSGENIC_RUN_DIR argument, fresh quota records/40GiB+5GiB threshold/warning rejection, exact release inventory and all 85 archive files byte-for-byte against immutable d99b Git objects. Host mount tests are mocked command construction, not successful real Apptainer mounts. Tests cover real observed quota syntax/rounding and rejection, non-adjacent RN pairing/duplicate geometry, archive path/link refusal, both cache layouts/materialized links/source preservation, collision and escaping-link refusal. The real inventory pair counts above were computed locally. These are not compute/SIF/GPU tests. Initial test exposed local Python3.8 `Path.is_relative_to` incompatibility; helper now uses stdlib commonpath. Actual Delta3.6 discovery additionally removed newer subprocess API usage. Initial archive assembly correctly failed because accepted JSONs are not Git objects; corrected explicit companion provenance is described above.

Reviewed APIs used: `make_launch_readiness.py:186–318,356–379`; `stage_checkpoint.py:76–182`; `real_probes.py:36–94,125–289,598–690`; `run_bounded.py:137–211`; `mvp_cli_runtime.py:146–151`; dataset selection/retained evidence `datasets.py:237–487`. Graft orientation preceded source reads. Advisor was unavailable both before and during recovery (`claude_code_version_too_old`); no advice/review approval obtained. No SSH was performed during this resumed verification; live observations above remain historical.
