#!/bin/bash
#SBATCH --job-name=panel-transport-review-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# PREPARATION ONLY. DO NOT SUBMIT: even this guard would consume an allocation.
# Account, partition, GPU resource shape, CPU/memory, exclusions, log paths and
# chdir deliberately absent. Controller must reconcile pending reservations and
# the shared <=5 node-hour campaign cap, then independently review final binding.
# <=1 node-hour panel ceiling is conditional, not permission or zero GPU billing.
# Do not add --gpus=0: no dedicated CPU placement was established on DeltaAI.
# No execution body is enabled before actual reviewed P1 stage receipt binding.
# Future reviewed body must retain no-clobber stdout/stderr/attempt records,
# verify real Slurm metadata, clear inherited injection, hide CUDA, use no --nv,
# run exact pinned proof once with timeout below scheduler ceiling, and retain
# failure evidence. No automatic resubmission, training or cleanup.
set -euo pipefail
printf '%s\n' 'BLOCKED: preparation-only candidate; actual reviewed P1 receipt, transport amendment, mount/resource bindings and execution adapter are absent.' >&2
exit 78
