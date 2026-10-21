#!/bin/bash
#SBATCH --job-name=b6-headroom-archive
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${OPS_DIR:?immutable archive operation path required}"
cd "$OPS_DIR"
sha256sum --check --strict helpers.sha256
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/usr/bin/python3 archive_guard.py pre
for epoch in 9 10 11; do
    /usr/bin/python3 archive_guard.py transfer "$epoch"
done
/usr/bin/python3 archive_guard.py post
