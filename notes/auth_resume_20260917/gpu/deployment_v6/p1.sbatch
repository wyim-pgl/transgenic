#!/bin/bash
#SBATCH --job-name=dev123-p1-v8-0921
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --exclude=gh062,gh093
#SBATCH --chdir=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_release_v8_20260921
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123-p1-v8-%j.out
#SBATCH --error=/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123-p1-v8-%j.err
set -euo pipefail
umask 077
# Fixed reviewed exclusions. Live B5 placement is checked before staging; no worker submission.
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
# Independent controller receipt is mandatory; this package does not create one.
[[ ${DEPLOY_CONTROLLER_GATE_SHA256:-} =~ ^[0-9a-f]{64}$ ]]
[[ ${DEPLOY_RELEASE_TRANSPORT_SHA256:-} =~ ^[0-9a-f]{64}$ ]]
[[ ${DEPLOY_RELEASE_LEDGER_SHA256:-} =~ ^[0-9a-f]{64}$ ]]
[[ ! -e S0_P1_STARTED ]]
# Exclusive claim survives failures; never rerun this release automatically.
(set -o noclobber; printf '%s\n' "${SLURM_JOB_ID:?}" > S0_P1_STARTED)
# Full payload hashing runs inside host only after fresh admission.
/usr/bin/date -Is
/usr/bin/hostname
# 28m20s combined S0+P1 ceiling, 20s teardown, Slurm30m hard backstop.
# No separate warmup and no TRAINING_DONE; any failure exits nonzero.
/usr/bin/timeout --signal=TERM --kill-after=20s 1700s /usr/bin/python3 deploy.py host
