#!/bin/bash
# Submitted with the same 2-GPU/1h30m CP4 allocation. Fail fast if NCCL fault tests fail.
set -euo pipefail
: "${REPO_DIR:?}"
: "${B6_RESULTS_DIR:?}"
: "${DATA:?}"
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
command -v apptainer >/dev/null 2>&1 || module load apptainer
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$B6_RESULTS_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
apptainer exec --nv "$SIF" python -c 'import torch; assert torch.cuda.device_count() == 2; [(torch.ones(1, device="cuda:"+str(i)), torch.cuda.synchronize(i)) for i in range(2)]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 apptainer exec --nv --bind "$REPO_DIR":/work/repo "$SIF" bash -lc '
    cd /work/repo && PYTHONPATH=/work/repo/src:/work/repo B6_CHECKPOINT_TEST_DEVICE=cuda \
    python -m pytest -q tests/test_b6_checkpoint_io.py
'
exec bash "$REPO_DIR/revision/scripts/b6/integration_4rank.sbatch"
