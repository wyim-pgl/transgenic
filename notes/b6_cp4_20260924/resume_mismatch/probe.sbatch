#!/bin/bash
#SBATCH --job-name=b6-resume-diag
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=%x-%j.out
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
REPO=$WORK/repo_b6_devicefix_20260924
DIAG=$WORK/diagnostics/resume_mismatch_20260924
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
[ "${SLURM_GPUS_ON_NODE:-}" = 2 ] || exit 3
command -v apptainer >/dev/null 2>&1 || module load apptainer
cd "$REPO"
sha256sum -c source-a48579d.sha256 > "$DIAG/source-verification-$SLURM_JOB_ID.txt"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
BASE=$WORK/runs/b6_resume_diagnostic/job_$SLURM_JOB_ID
test ! -e "$BASE"
mkdir -p "$BASE"
cp -a "$REPO/HFmodels" "$BASE/hf_cache"
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu apptainer exec --nv "$SIF" python -c 'import torch; torch.ones(1).cuda(); torch.cuda.synchronize()'
done
for mode in eager unsplit; do
    set +e
    apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO":/work/repo:ro "$SIF" bash -lc "
        cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/runs/b6_resume_diagnostic/job_$SLURM_JOB_ID/hf_cache &&
        accelerate launch --num_processes=2 --mixed_precision=bf16 /work/diagnostics/resume_mismatch_20260924/run_probe.py \
        --case $mode --out /work/runs/b6_resume_diagnostic/job_$SLURM_JOB_ID/$mode
    " > "$BASE/$mode.log" 2>&1
    rc=$?
    set -e
    printf '%s %s\n' "$mode" "$rc" | tee -a "$BASE/case_exit_codes.txt"
    sha256sum -c source-a48579d.sha256 > "$BASE/source-after-$mode.txt"
done
printf 'DIAGNOSTIC_FINISHED: inspect both reports, never infer CP4 PASS from this job exit\n'
