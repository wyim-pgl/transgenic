# CLAUDE.md — transgenic_local

Read [AGENTS.md](AGENTS.md) too (graft usage, session recovery, continuous TODO execution). This file adds the
working rules that the B6 production run depends on.

## Start here

1. `.atl/resume.md` (📌 block) → the handoff it links (now `notes/RESTART_HANDOFF_20261006.md`).
2. `.atl/todo.md` top checklist = the current instruction; `.atl/quarantine.md` §1r/§1q/§1p before retrying anything.
3. `.atl/intent.md` = why and which readiness gates must hold. Detailed log: `notes/b6_production_start_20260925/PROGRESS.md`.
4. Verify live Slurm state (`sacct -j <ids> -X`) before resubmitting anything.

## Code and branches

- Implementation lives in `/tmp/b6-integration` (branch `b6-integration`), not in main. Never merge to main or push
  unless the user asks. Feature work goes in its own worktree, then fast-forward or cherry-pick onto b6-integration.
- git refuses these repos on ownership: run git with `GIT_CONFIG_GLOBAL=<scratchpad>/ro.gitconfig`
  (`[include] path=/home/wyim/.gitconfig`, `[safe] directory = *`).
- Commits: `git -c user.name="Won Yim" -c user.email="wyim@unr.edu" commit`; no co-author, generated-by, or session
  trailers. Verify with `git log --format='%h %an <%ae> / %cn <%ce>' -1`.
- Deploy a commit to pronghorn as a new read-only `$PREP/source_vN` with `git archive <sha> | ssh pronghorn tar -x`,
  write `SOURCE_COMMIT`, and compare sha256 of the changed files. Never modify an existing `source_v*`.

## pronghorn

- `$PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925`. Production source: see the handoff
  (currently `source_v21`, `826e68e`; the production obs cache was built with `source_v17`, `f06a786`, and its memory-mapped derived files with `source_v20`).
- DuckDB/pyarrow code runs only inside `/data/gpfs/home/wyim/scratch/singularity/micromamba.sif` with
  `PYTHONPATH=$PREP/duckdb_runtime:...`. Host python is the sylvan env; there is no `python3`.
- Slurm: default time limit is 13 days (`SBATCH_TIMELIMIT` in `~/.bashrc`). A node allocates at most 232 GB
  (MemSpecLimit 24 GB); check `scontrol show node` AllocMem, not `sinfo` FREE_MEM. Size memory from measured MaxRSS
  (figures in the handoff and quarantine §1q). Ignore `.extern OUT_OF_MEMORY 0:125` (author decision); judge by the
  batch step, exit code, QC and hashes.
- Partitions: only `cpu-s1-pgl-0` or `cpu-s2-core-0` (account `cpu-s2-pgl-0`). **Never `cpu-s3-sponsored-0`** (author
  decision 2026-10-02). The pgl nodes are often full of the user's Sylvan jobs: use `sbatch --test-only` on both
  partitions and pick the one that starts first.
- Every DuckDB job needs an explicit `--memory-limit` below the allocation. The obs cache builder also holds one
  chromosome's rows as Python objects outside that limit (~80 GB for Athaliana Chr1): limit ≈ allocation − 100 GB.
- sbatch scripts under `set -euo pipefail`: never end with `cmd | head` (SIGPIPE → job FAILED although the payload
  succeeded) and add `|| true` to every command in an EXIT trap. An `afternotok` retry must check the payload's own
  success marker (e.g. `manifest.json`), not the Slurm state.
- Never delete outputs: move failed ones to `out_failed_<job>` / `<dir>.oom_<job>` and start a fresh `out/`.

## Changing pipeline code

- A fix must not change results unless the user decided so. Prove it: write the test first, show it fails on the base
  commit, then compare the full suite base vs new by the FAILED/ERROR set diff (`--continue-on-collection-errors`;
  ~165 failures are pre-existing container gaps such as missing torch). Rerun a failure that looks unrelated.
- Get an independent Codex review of each fix commit (user instruction, 2026-09-30) and address its findings before
  production use.
- Design choices, scientific rules, dropping data, or anything that changes outputs are the user's decision: present
  options with a recommendation and wait.

## DeltaAI

- Reach it only through pronghorn: `ssh pronghorn 'ssh -o BatchMode=yes dtai "..."'` (ControlMaster; NCSA Duo).
- Container `/work` = `/work/nvme/bilv/wcyim/transgenic`. B6 inputs: nvme `b6_prep_20260925/`; merged evidence DB on HDD
  `/work/hdd/bilv/wcyim/transgenic/b6_prep_20260925/merged/` bound read-only at `/hdd_prep`. The freeze records the
  interval manifests by pronghorn absolute paths (hash-bound): bind the nvme copy at that path. Pass binds with an
  exported `APPTAINER_BIND` and `sbatch --export=ALL,...` (a comma inside an `--export` value breaks it).
- Code: `repo_b6_9f003e1` (archive snapshot + local commit; git bundles of this repo fail on missing history objects).
  Never touch `repo_dev`, `repo_seeds` or `runs/seed123` (B5 seed chain).
- Do not run long CPU work on login nodes (a 33-minute `sha256sum` was killed). Re-hash inside a job; `verify_inputs`
  does it. Short gate jobs go to `ghx4-interactive` (≤ 2 h, starts in minutes); ghx4 has thousands pending.
- Re-sync C0 sidecars after any C0 finishes late; the DeltaAI verify job fails closed on a missing PROVENANCE file.
- Never stage symlinks under DeltaAI `/work` (the B5 preflight fails on dangling links): rsync with `--no-links` or `-L`,
  then check `find $WORK -xtype l` is empty.
- Training reads the obs cache through memory-mapped derived files: run `b6_prebuild_obs_mmap.py <cache>` once, serially,
  before training (and re-run its check before each submission), and train with `B6_OBS_CACHE_REQUIRE_PREBUILT=1`.
  Before any long GPU link on new data, run a short GPU smoke on `ghx4-interactive` that measures host RSS and GPU peak.

## GPU

- No GPU training job is submitted without explicit user confirmation, with the GPU-hour budget stated. The WP5 pilot
  (B6 and C2, 4,099 updates, ≈98 GPU-h each) was approved 2026-10-03 (first links OOM-failed 10-05; resubmission after
  the ObsCache fix and a ≈8 GPU-h smoke, approved 10-05); B5 seed 123 restart chain approved 10-04; WP6 and the B5 seeds 456/789 still
  need their own confirmation.

## Records

- After significant work update `PROGRESS.md`, `.atl/todo.md` and `.atl/resume.md`; add discarded approaches to
  `.atl/quarantine.md`. Commit notes/.atl only when the user asks (record branch `b6-production-start-20260925`).
