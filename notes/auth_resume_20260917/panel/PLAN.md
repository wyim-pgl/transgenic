# Actual50 immutable panel compatibility: candidate preparation

Status: BLOCKED, no actual50 compatibility execution, accepted receipt, deployment or submission. This directory is the only intentional write target. No GPU-agent files were changed. Coordination and all future approvals belong to the controller.

## Completed observations

- Read the scoped panel review and pinned-caller controller review. Current helper and wrapper match their approved SHA256 values. `consumer_closure_candidate.json` freezes the current 13-file report closure, including `mvp_profiles.py`, decoder and the reviewed runtime handshake repin. The wrapper is pinned separately because `consumer_closure()` does not include it.
- Reused `notes/power_recovery_20260917/runtime_ledger.json`, its accepted panel metadata, historical CPU runtime, terminal accounting, and `notes/MVP_DEPLOY_OPS_20260916_1545.md:46-62` rather than rescanning original data.
- `remote_metadata_explicit_python.json` records a fresh bounded stat-only SSH observation: all 15 historical input paths still resolve to themselves, their recorded sizes match, and no adjacent WAL was observed. The original panel manifest and COMPLETE are present. This is availability evidence, not renewed hash or numerical acceptance. No arrays or DBs were read.
- First metadata command found no generic `python3` on login PATH. Failure retained in `remote_metadata.json`. The second used the existing explicit transgenic-revision Python 3.11 executable only for stdlib metadata, not as the intended runtime.
- Pronghorn login-0 is x86_64 with Singularity 3.6.1, no `apptainer` on PATH. The intended image is absent at both its recorded `/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif` path and the specifically checked `/data/gpfs/assoc/pgl/singularity/transgenic-ngc-26.08.sif` candidate. No exhaustive filesystem search was performed.
- Prior deployment records identify that exact intended image as aarch64. Therefore native execution on the observed x86_64 Pronghorn host is incompatible even if someone copied it. No emulation, alternative image, package install or CPU Torch substitution is approved. Checking a login node does not certify every compute node; no compatible allocated node has been established.
- Prior small image build record pins Torch `2.14.0a0+4fdf77b940.nv26.08`, DuckDB `1.5.5`, CUDA `13.4`, Transformers `4.57.6`, Accelerate `1.15.0`. It does not establish the exact intended Python and NumPy versions. Those remain explicit null blockers, not guessed pins. Through the controller, request bounded `platform.python_version()` and NumPy/DuckDB/Torch version metadata from the GPU preparation owner in the intended image, with no panel/helper invocation.
- Historical panel CPU environment was Python 3.10.20, NumPy 1.26.4, DuckDB 1.5.5, Torch 1.13.1+cpu. It cannot replace the intended consumer environment: the receipt compares `current_packages` by exact whole-report equality.

## Independently reviewable request candidate

`candidate_request.json` contains only the exact original eight data paths and candidate acceptance binding, original accepted panel root, `profile: development-single`, integer seed123 and aug_seed20261037. No run inventory, training state or models are needed. The original historical request is untouched.

Proposed canonical request identity, to be approved independently and carried as constants rather than computed at launch:

- Path: `/data/gpfs/assoc/pgl/data/Transgenic/evidence/seed123_panel_compat_auth_resume_20260917/request.json`
- SHA256: `c5cc29ecea746375ebb4d5388d38fa2bb8db0d5c42088ac67bc47b4caa7da9a9`
- Size: 1264 bytes

`request_pin.json` binds these exact bytes to the proposed deployed path. This is frozen candidate content, not a claim that the remote request exists or has independent approval. If the controller chooses another namespace, issue a new pin document and review it. Do not change this candidate silently.

Accepted old root is exactly `/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916/outputs/job-6165628/panel`; manifest SHA256 is `7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e`. `historical_inputs.json` retains all 15 original path/hash/size records and historical authority. No copies, path aliases, modified producer sources, panel rebuilds or DB rebuilds are a substitute.

## Requested CPU budget, not allocation authority

Candidate upper bound: one task, 4 CPU, 16 GiB RAM, 60 minutes, zero GPUs, no requeue, maximum 4 allocated CPU-hours. Request controller and site approval before any job. `candidate_config.json` explicitly marks approval false and has no runnable scheduler submission script.

Rationale: historical 6165628 ran 18m53s with 4 CPU and 16 GiB requested; batch MaxRSS was 787524 KiB. The compatibility helper rereads the approximately 23.53 GB corpus and 1.07 GB evidence repeatedly at integrity boundaries, performs real DB preflight/selected-member checks, validates all 150 artifacts and three historical rasters, and adds the fourth seed123 raster for 50 loci. A 60-minute request provides a bounded roughly threefold historical wall allowance, not a measured guarantee. Historical batch RSS does not bound this different runtime or GPFS overhead. Its extern OUT_OF_MEMORY remains unresolved; a successful batch exit alone is insufficient. Controller must resolve whether this budget/site combination is acceptable, not suppress the extern issue or retry automatically.

Pronghorn partition metadata observed `cpu-s1-pgl-0`, 64 CPU/256000 MB node description, max14 days and up. This is capacity metadata only, not permission, availability or an architecture-compatible allocation. Choosing another architecture/site requires separately reviewed access to every original canonical GPFS path. Moving data to DeltaAI or rewriting requests is not an authorized workaround.

## Packaging recipe, not executed

1. Controller resolves runtime/site/canonical-path blockers first and reconciles the final consumer root with GPU preparation. Proposed root is `<namespace>/source` in `candidate_config.json`; it is not yet accepted. The CPU receipt and subsequent consumer must resolve all 13 report-closure files to identical canonical paths and bytes. A bind mount may preserve visible paths only if explicitly reviewed; do not assume host/container paths agree.
2. Freeze a complete immutable reviewed source tree from `/tmp/mvp-seed123-integration-20260917`, including the final runtime repin, decoder, profiles, panel helpers and separately pinned wrapper. The 13-file report closure is not the full Python import inventory. Package/review package initializers, imports, dependencies and exact tree inventory; reject unreviewed importable additions and stale bytecode. Do not use an installed older transgenic tree. Disable user-site and bytecode writes and provide explicit reviewed source/script import paths. No source import was used to generate the local ledger.
3. Preserve the real frozen Git authority object used by `mvp_produce_counts.frozen_reference`: it executes `git -C ROOT show AUTHORITY_COMMIT:AUTHORITY_PATH`. A flat source tarball without that object will fail. Freeze the relevant commit/tree/blob closure in a self-contained read-only repository; do not rely on a local worktree `.git` pointer or substitute a working-copy TSV. Preserve current reviewed dirty source bytes separately from base Git content. Compare `historical_inputs.json` authority to the final package's git-show output.
4. Independently pin full deployed tree, launcher, request, runtime/image metadata and final canonical consumer paths. Existing input trees remain read-only at exact original paths. Fresh output namespace must be outside the old panel, with existing canonical output parent and no receipt or `.pending` collision. Reserve separate exclusive stdout/stderr and submission-attempt records. Reject namespace reuse and keep failures; no cleanup retries. This session does not create remote directories, transfer bytes or set permissions.
5. After explicit deployment/budget approval, launch only inside a genuine Slurm CPU allocation with real scheduler job metadata, empty CUDA_VISIBLE_DEVICES, offline flags and bounded CPU threads. Never export a fake SLURM_JOB_ID. Match exact intended Python/NumPy/DuckDB/Torch versions and independently verify expected source/request/runtime bindings before importing the helper. Use the wrapper argv in `candidate_config.json`; replace its generic `python` only with the reviewed image interpreter invocation. The JSON is descriptive and blocked, not a shell script.
6. Wrapper must receive the independent fixed request digest and size. Run once. It invokes `verify_and_publish` once, checks serialized `verification_request` equality, returned/serialized report equality and postchecks. No helper, raster or original DB hash is permitted on login nodes.
7. Controller acceptance checks terminal allocation, batch and extern accounting; exact report request identity above; exact old manifest and all historical identities; current closure and package versions; 50 ordered locus proofs and all four augmentation seeds; explicit seed123 equality and exact DB-selected provenance; unchanged input paths/WAL absence; and externally located no-clobber output. Check request and receipt bytes again, then independently pin the accepted receipt. Do not trust a digest merely because the process printed it. Retain all diagnostics on failure.
8. Later GPU consumption repeats live proof and whole-report equality; this is not a cheap receipt-only gate. It additionally needs its own genuine runtime/module/cache/model/tokenizer/readiness approval. A CPU receipt grants neither GPU readiness nor scientific/training approval.

## Open controller TODOs

- BLOCKED: intended aarch64 SIF unavailable at checked Pronghorn paths and incompatible with observed native x86_64 architecture.
- BLOCKED: intended Python and NumPy exact versions unobserved; current small build record supplies only DuckDB/Torch pins.
- BLOCKED: compatible compute placement with all original canonical GPFS paths has not been established. No relocation/rebuild fallback.
- BLOCKED: final shared consumer canonical root, complete deployment inventory/import closure and Git authority packaging await independent acceptance.
- REQUESTED, not approved: CPU4/16GiB/60min budget and extern-OOM handling.
- NOT RUN: actual50 verifier, job submission, remote deployment, receipt acceptance or GPU preparation changes.

Graph context consulted before source. Source spans: `mvp_verify_pinned_panel.py:15-35`, `mvp_panel_compatibility.py:22-216`, `mvp_prepare_panel_inputs.py:30-45`, `mvp_produce_counts.py:55-71`, `mvp_decode_runtime.py:1-44`. Advisor consultation failed due to client/model incompatibility; no advisor endorsement. Graft reported about 1,254,530 tokens saved across this task's four map/ask calls.
