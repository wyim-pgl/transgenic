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
WORK=$SCRATCH/transgenic
# SIF is an env override like STEPS/TIER (sbatch --export=ALL,SIF=...), so another NGC tag can be tried
# without editing this file. The job prints which image it ran in; #18 must cite that line.
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc.sif}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
echo "image: $SIF"; apptainer exec "$SIF" cat /opt/transgenic/IMAGE_INFO 2>/dev/null || echo "image: (no IMAGE_INFO)"
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
# Triton must be told where apptainer --nv put the host driver. Inside the container the image's ldconfig
# points at /usr/local/cuda/compat/lib/libcuda.so.1, which does not exist, while the real library is
# bound at /.singularity.d/libs/libcuda.so.1; without this, torch.compile dies at the first forward with
# "libcuda.so cannot found!" (measured on the GB10 SIF, 2026-09-08; triton 3.3.0 honours the variable).
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
# Encoder weights come from the pre-populated cache, never from the network. train_HyenaTransgenic.py:14 sets
# HF_HOME='./HFmodels' itself (so the image's HF_HOME=/work/hf is never used); the model lives under
# HFmodels/hub/models--... and the tokenizer under HFmodels/models--... (cache_dir=./HFmodels). Both were
# shipped from the GB10 test bed and load with HF_HUB_OFFLINE=1 on DeltaAI (6.6 M params, 18 s). Offline
# mode makes a missing file a fast failure instead of six jobs downloading into one shared directory.
export HF_HOME=/work/repo/HFmodels HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN OMP_NUM_THREADS=8
# No runtime `pip install -e`. Measured on GB10 in the same SIF recipe: setuptools egg_info wants to write
# src/transgenic.egg-info INTO the bind-mounted repo ("Cannot update time stamp of directory"), so
# concurrent jobs race on one checkout and the trainer died at its first import. The package is a
# src-layout namespace package; PYTHONPATH=/work/repo/src imports it with zero writes.
# REPO_DIR: the checkout bound at /work/repo. Default is the bundle's repo/; experiments must point this at a
# separate clone, because /work/repo is re-read by every link of a running A28 seed chain and must not move.
REPO_DIR=${REPO_DIR:-$WORK/repo}
[ -d "$REPO_DIR/.git" ] || { echo "REFUSED: REPO_DIR is not a git checkout: $REPO_DIR" >&2; exit 3; }
echo "repo: $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-60)"
# The trainer loads the encoder from $REPO_DIR/HFmodels offline; a checkout without the cache dies at the first
# import after the GPU is allocated (round-1 bisection, 2026-09-09). Refuse before spending the slot.
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels (see preflight_b5.sh)" >&2; exit 3; }
# Driver-level memory ground truth (2026-09-09): torch.cuda.max_memory_allocated reported 101 GB on NGC 25.06 and
# ~11 GB on 26.08 for the same eager recipe, so the number PyTorch keeps is not comparable across images. nvidia-smi
# on the host samples memory.used every 2 s for the whole run; the per-GPU maximum is printed at the end.
MEMLOG=$WORK/runs/mem-g$GPUS-t$TIER${OUTTAG:+-$OUTTAG}-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > "$MEMLOG" 2>/dev/null &
SMI=$!
RC=0
apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && export PYTHONPATH=/work/repo/src &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config $CONFIG --db $DB --seed 123 --benchmark-steps $STEPS --benchmark-tier $TIER --no-wandb \
    --output-dir /work/runs/bench-g$GPUS-t$TIER${OUTTAG:+-$OUTTAG}
" || RC=$?
kill $SMI 2>/dev/null || true
echo "driver_peak_mem_mib_per_gpu: $(awk -F', ' '{if($2+0>m[$1])m[$1]=$2+0} END{for(i in m) printf "gpu%s=%d ", i, m[i]}' "$MEMLOG") samples=$(wc -l < "$MEMLOG")"
exit $RC
# Expected output: tokens/s, s/step, peak GB, hours/epoch (rows in DB / rows per step / 3600); paste into #18.
# Run the full grid before sizing a seed:
#   for t in 30720 61440 129024; do sbatch --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; done
#   for t in 30720 61440 129024; do sbatch --gpus-per-node=4 --export=ALL,TIER=$t deploy/deltaai/bench_b5.slurm; done
