#!/bin/bash
#SBATCH --job-name=mvp8-final-p1-0916
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --exclude=gh062,gh093,gh086
#SBATCH --chdir=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/final_p1_release
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/final-p1-%j.out
#SBATCH --error=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/final-p1-%j.err
set -euo pipefail
umask 077
# Controller refreshes B5 node exclusion immediately before submission; no worker submission.
# Clear all inherited container injection and application overrides, not just one known key.
for name in $(compgen -e); do
  case "$name" in
    APPTAINER*|SINGULARITY*|TRANSGENIC*|HF_*|TRANSFORMERS*|PYTHON*|TORCH*|TRITON*|MVP_*|CUDA_VISIBLE_DEVICES)
      unset "$name" ;;
  esac
done
unset LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV
export PATH=/usr/bin:/bin:/sw/user/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
[[ ${SLURM_JOB_NUM_NODES:?} == 1 ]]
[[ ${SLURM_GPUS_ON_NODE:?} == 4 ]]
[[ ! -e S0_P1_STARTED ]]
# Exclusive claim survives failures; never rerun this release automatically.
(set -o noclobber; printf '%s\n' "${SLURM_JOB_ID:?}" > S0_P1_STARTED)
/usr/bin/sha256sum -c release.sha256
/usr/bin/date -Is
/usr/bin/hostname
# 28m20s combined S0+P1 ceiling, 20s teardown, Slurm30m hard backstop.
# No separate warmup and no TRAINING_DONE; any failure exits nonzero.
/usr/bin/timeout --signal=TERM --kill-after=20s 1700s /usr/bin/python3 deploy.py host
