#!/usr/bin/env bash
# submit_chain.sh -- pre-queued A28 job chain for one B5 seed (protocol A45, 2026-09-11).
#
#   bash deploy/deltaai/submit_chain.sh SEED [LINKS]
#
# Submits LINKS (default 3) copies of train_b5.slurm for SEED up front: link 1 without a dependency, link k+1 with
# --dependency=afterany:<link k>. Every link carries --comment=seed$SEED (how a running link finds its queued
# successors) and CHAIN_PRESUBMITTED=1 (the script then never resubmits anything: the queue already holds the
# chain, a finished or failed run cancels the remaining links, a node launch failure is pushed onto them as
# ExcNodeList). afterany, not afterok: a link that ends by forced preemption (watchdog, rc 137) or by a launch
# failure (exit 1) is continued by its successor from the last published checkpoint; a genuine trainer failure
# writes TRAINING_FAILED, which every later link refuses at its first line.
#
# Three links (3 x 48 h = 144 h) cover the 22-epoch ceiling at the measured 6.15 h/epoch on 4 GPUs (135 h).
# The scheduler does not accrue age priority for dependent jobs (no ACCRUE_ALWAYS), so the gain is that a
# successor is eligible the second its predecessor ends and can backfill onto the freed node; nothing more is claimed.
#
# Environment (exported to every link through --export=ALL): SIF, REPO_DIR, GPUS_PER_NODE (default 4), CPUS (64),
# EXCLUDE (gh062,gh093), plus anything else train_b5.slurm reads (SAVE_EVERY, TRANSGENIC_* switches).
set -euo pipefail
SEED=${1:?usage: submit_chain.sh SEED [LINKS]}
LINKS=${2:-3}
: "${USER:=$(id -un)}"
: "${SCRATCH:=/work/nvme/bilv/$USER}"
WORK=$SCRATCH/transgenic
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc-26.08.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_seeds}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS=${CPUS:-64}
EXCLUDE=${EXCLUDE:-gh062,gh093}
SCRIPT=$REPO_DIR/deploy/deltaai/train_b5.slurm
RUN_HOST=$WORK/runs/seed$SEED
IDFILE=$WORK/runs/seed${SEED}_chain_jobids.txt

[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
[ -d "$REPO_DIR/.git" ] && [ -f "$SCRIPT" ] || { echo "REFUSED: $REPO_DIR is not a checkout with $SCRIPT" >&2; exit 3; }
grep -q 'CHAIN_PRESUBMITTED' "$SCRIPT" || { echo "REFUSED: $SCRIPT predates the pre-queued chain (no CHAIN_PRESUBMITTED)" >&2; exit 3; }
case "$SEED" in ''|*[!0-9]*) echo "REFUSED: SEED must be an integer (it names the run directory and travels in --export), got '$SEED'" >&2; exit 3;; esac
case "$LINKS" in ''|*[!0-9]*|0) echo "REFUSED: LINKS must be a positive integer, got '$LINKS'" >&2; exit 3;; esac
# never two chains for one seed: an existing queued/running job with this seed's comment, or an existing run
# directory (unless RESUME_EXISTING=1 says the chain is meant to continue an earlier run), stops the submission.
EXISTING=$(squeue -u "$USER" -h -o '%i %k' | awk -v c="seed$SEED" '$2 == c { print $1 }' | tr '\n' ' ')
[ -z "$EXISTING" ] || { echo "REFUSED: seed $SEED already has queued/running links: $EXISTING" >&2; exit 4; }
if [ -e "$RUN_HOST" ] && [ "${RESUME_EXISTING:-0}" != 1 ]; then
  echo "REFUSED: $RUN_HOST exists; move it aside, or RESUME_EXISTING=1 to continue that run" >&2; exit 4
fi
[ -f "$RUN_HOST/TRAINING_DONE" ] && { echo "REFUSED: $RUN_HOST/TRAINING_DONE exists" >&2; exit 4; }
[ -f "$RUN_HOST/TRAINING_FAILED" ] && { echo "REFUSED: $RUN_HOST/TRAINING_FAILED exists; remove it deliberately first" >&2; exit 4; }

echo "seed $SEED: $LINKS pre-queued links, $GPUS_PER_NODE GPU x $CPUS CPU, exclude=$EXCLUDE"
echo "  repo:  $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-60)"
echo "  image: $SIF"
export SIF REPO_DIR
PREV=""; IDS=""
for k in $(seq 1 "$LINKS"); do
  DEP=${PREV:+--dependency=afterany:$PREV}
  # --export=ALL carries SIF/REPO_DIR and the caller's environment; the comma-free list below is what each link
  # needs to know about its place in the chain. GPUS is repeated on the command line (the #SBATCH directive says 1).
  OUT=$(sbatch --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS" --exclude="$EXCLUDE" --comment="seed$SEED" $DEP \
        --export=ALL,SEED=$SEED,GPUS=$GPUS_PER_NODE,CHAIN_N=$k,CHAIN_MAX=$LINKS,CHAIN_PRESUBMITTED=1,LAUNCH_RETRY=0 "$SCRIPT" 2>&1) \
    || { echo "sbatch failed for link $k: $OUT" >&2; [ -z "$IDS" ] || { echo "cancelling the links already queued: $IDS" >&2; scancel $IDS || true; }; exit 5; }
  JID=$(echo "$OUT" | grep -oE '[0-9]+$')
  [ -n "$JID" ] || { echo "no job id in: $OUT" >&2; [ -z "$IDS" ] || scancel $IDS || true; exit 5; }
  echo "  link $k: job $JID${PREV:+ (afterany:$PREV)}"
  IDS="${IDS:+$IDS }$JID"; PREV=$JID
done
mkdir -p "$WORK/runs"
printf '%s seed=%s links=%s jobs=%s repo=%s sif=%s\n' "$(date -Is)" "$SEED" "$LINKS" "$IDS" \
  "$(git -C "$REPO_DIR" rev-parse --short HEAD)" "$(basename "$SIF")" >> "$IDFILE"
echo "recorded in $IDFILE"
