#!/bin/bash
#SBATCH --job-name=b5-bench
#SBATCH --account=bilv-dtai-gh            # ACCESS project on DeltaAI/Delta (e.g. bbxx-dtai-gh or bbxx-delta-gpu)
#SBATCH --partition=ghx4          # DeltaAI: ghx4 ; Delta: gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
# See train_b5.slurm: a shell variable here is rejected at submit. For the 4-GPU arm run
#   sbatch --gpus-per-node=4 deploy/deltaai/bench_b5.slurm
#SBATCH --cpus-per-task=16
# No --mem: see train_b5.slurm. 120g billed 1.08 GPUs under MAX_TRES.
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Issue #18: measure throughput of the frozen 400M recipe on this hardware.
set -euo pipefail
: "${SCRATCH:=/work/nvme/bilv/$USER}"   # DeltaAI has no /scratch and no $SCRATCH. Docs: /work/hdd for large job I/O, /work/nvme for many small I/Os; both Lustre, 1 TB default
command -v apptainer >/dev/null 2>&1 || module load apptainer
WORK=$SCRATCH/transgenic; SIF=$SCRATCH/containers/transgenic-ngc.sif
# SLURM_GPUS_ON_NODE is always a bare integer and is set for every request form.
# SLURM_GPUS_PER_NODE echoes the submitter's string (can be nvidia_gh200_120gb:4) and is
# unset for --gres=gpu:N or --gpus=N, which would silently run 1 process on 4 GPUs.
GPUS=${SLURM_GPUS_ON_NODE:-${GPUS:-1}}; STEPS=${STEPS:-300}
# The bundle carries only the A40 object (preflight: "db dir carries only the A40 object"),
# so the old b5_smoke.duckdb default could not have opened.
DB=${DB:-/work/db/b5_full_a40_v1.db}
# The benchmark must measure the recipe that will be TRAINED. train_b5.slurm runs
# b5_400m_win_v3.json (A26/A27; preflight calls it the canonical recipe), and win_v3 sets
# max_decoder_position_embeddings / max_label_tokens 8192, vocab 290 and a 96-gene cap that
# ctx_v2 leaves at the code defaults. Decoder length dominates step time and peak memory, so
# benchmarking ctx_v2 would report a CHEAPER configuration than the one being sized for.
# It is also the WRONG GRAMMAR: ctx_v2 is window_policy tier6144-v2 (single target gene) while
# the A40 database rows are tile6144-v3 whole-window labels. The switch is mandatory, not tuning.
CONFIG=${CONFIG:-configs/b5_400m_win_v3.json}
# One tier per run: window_tiers are 30720 / 61440 / 129024 and tier_up_prob 0.3 means an
# unpinned run samples a mixture, which is not a per-tier number. --benchmark-tier pins it.
TIER=${TIER:-129024}
export NCCL_DEBUG=WARN OMP_NUM_THREADS=8
apptainer exec --nv --writable-tmpfs --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config $CONFIG --db $DB --seed 123 --benchmark-steps $STEPS --benchmark-tier $TIER --no-wandb \
    --output-dir /work/runs/bench-g$GPUS-t$TIER
"
# Expected output: tokens/s, s/step, peak GB, hours/epoch (rows in DB / rows per step / 3600); paste into #18.
# Run the full grid before sizing a seed:
#   for t in 30720 61440 129024; do sbatch --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; done
#   for t in 30720 61440 129024; do sbatch --gpus-per-node=4 --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; done
