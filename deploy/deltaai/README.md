# TransGenic on NCSA DeltaAI / Delta with NVIDIA NGC containers

Purpose: run the B5 retraining seeds (configs/b5_400m_v1.json), the #18 throughput benchmark and the
B7 inference on the ACCESS allocation without building a conda stack on the cluster. The GPU image is
the NGC PyTorch container (multi-arch: works on DeltaAI's GH200/aarch64 and on Delta's A100/x86_64);
the CPU tools (OrthoFinder, minimap2, miniprot) come from Biocontainers images. Apptainer is the
container runtime on both systems.

## 1. Which system
| | DeltaAI | Delta |
|---|---|---|
| GPU | NVIDIA GH200 (Hopper, 96 GB HBM3 + 480 GB LPDDR unified), 4 per node | A100 40/80 GB (4 per node), A40 |
| CPU arch | **aarch64** (Grace) | x86_64 |
| Container image | NGC `pytorch:<tag>-py3` multi-arch manifest pulls the arm64 build automatically | same tag, amd64 build |
| Fit for B5 | best: one seed per GPU, 96 GB lets micro-batch grow | good: A100 80 GB; avoid A40 |
Everything below works on both; only the wheel architecture differs, which the NGC image hides.

**Compute assignment (protocol A25, 2026-09-02):** all three B5 seeds (123 primary, 456/789 confirmatory) run on ACCESS with the variable-context recipe `configs/b5_400m_ctx_v2.json` (windows up to 129,024 nt); the lab RTX 4090 is for inference, B7 and development. `bench_b5.slurm` must be run once per window tier.
Check the allocation's resource name in the ACCESS portal and record it in issue #18.

## 2. Build the GPU image (once, on a login node or locally)
```bash
module load apptainer            # name may differ; `module avail apptainer`
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache   # never $HOME (quota)
cd deploy/deltaai
bash build.sh                    # -> $SCRATCH/containers/transgenic-ngc.sif
```
`transgenic.def` starts from `nvcr.io/nvidia/pytorch:25.06-py3` (pin the tag you actually pulled in
`build.sh`; the tag is recorded into the image at /opt/transgenic/IMAGE_INFO) and pip-installs
transformers, accelerate, duckdb, safetensors, huggingface-hub, pandas, tqdm, wandb, pytest and the
repository itself in editable mode from a bind-mounted checkout. bitsandbytes is **not** installed
(the B5 recipe uses plain AdamW; the 8-bit optimizer of the RTX 4090 script is not part of B5).

## 3. Data layout on the cluster
```
$SCRATCH/transgenic/                 bind-mounted as /work inside the container
  repo/                              git clone of wyim-pgl/transgenic (main)
  db/b5_v1.duckdb                    immutable B5 database (built with scripts/build_b5_database.py)
  db/b5_v1.duckdb.sha256
  hf/                                HF cache (LongSafari/hyenadna-large-1m-seqlen-hf); set HF_HOME=/work/hf
  runs/seed123 | seed456 | seed789   checkpoints + logs (train.err holds the epoch=N eval lines)
```
DuckDB reads from scratch (parallel file system); copy the DB once, verify its sha256, never edit it.

## 4. Jobs
- `bench_b5.slurm` — #18: 300 optimizer steps on 1 GPU and on 4 GPUs (accelerate DDP), reports
  tokens/s, seconds/step, peak memory, and extrapolated hours/epoch for the B5 DB row count.
- `train_b5.slurm` — one seed per job (`--export=SEED=456`), 1 node × 1 GPU by default; set
  `GPUS=4` for DDP. Resumable: re-submit with the same run dir. Epoch decisions: run
  `revision/scripts/60_epoch_budget.py --log runs/seedNNN/train.err` after each epoch.
- `cpu_tools.slurm` — OrthoFinder / minimap2 / miniprot via Biocontainers images on CPU nodes.
All jobs use `apptainer exec --nv --bind $SCRATCH/transgenic:/work`. Set `NCCL_SOCKET_IFNAME`
and `OMP_NUM_THREADS` as in the templates; on DeltaAI add `--gpus-per-node=4 --gpu-bind=closest`.

## 4a. Per-epoch off-site copy (rsync)
`sync_watch.sh` copies every completed `epoch_NN/` directory once (marker in `.synced/`), re-copies
`train.err`/`*.json`/`best` every 5 min, and can prune old local epochs after a verified copy
(`KEEP_LOCAL=3`). `train_b5.slurm` starts it in the background when `SYNC_DEST` is set and stops it
after training. Because pronghorn logins use MFA, either (a) create a dedicated ssh key on the cluster
and authorise it on the destination (`RSYNC_SSH="ssh -i ~/.ssh/id_ed25519_sync"`), or (b) run the
watcher in **pull mode** from the destination host instead:
```bash
SRC=wyim@dt-login01.delta.ncsa.illinois.edu:/scratch/<proj>/transgenic/runs/seed456 \
  deploy/deltaai/sync_watch.sh pull /data/gpfs/assoc/pgl/data/Transgenic/runs/seed456
```
Optimizer states are skipped by default (`SYNC_OPTIMIZER=0`; ~3× the weight size) — resume happens
on the cluster from its own scratch copy. Checkpoint layout (`epoch_NN.tmp` → rename to `epoch_NN`,
`best` symlink, `TRAINING_DONE` marker) is part of the #17 trainer change.

## 5. Order
1. Account active → `build.sh` → `apptainer exec --nv transgenic-ngc.sif python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name())"`.
2. `pytest -q tests` inside the container (70 tests) → record versions in #18.
3. Copy the smoke DB (#15), run `bench_b5.slurm` → fill the #18 table → freeze seed placement.
4. Copy the full B5 DB (#16), submit `train_b5.slurm` for seeds 456 and 789.
5. B7 inference jobs reuse the same image.
Charging: DeltaAI/Delta bill GPU-hours per GPU; a 4-GPU DDP job costs 4× per hour but finishes ~3–3.5× faster.
