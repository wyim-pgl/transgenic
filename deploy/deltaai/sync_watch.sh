#!/usr/bin/env bash
# Per-epoch rsync of a training run directory (checkpoints + logs) to another host.
#
#   sync_watch.sh <run_dir> <dest> [interval_s]
#     run_dir   e.g. $SCRATCH/transgenic/runs/seed456
#     dest      rsync destination, local path or user@host:/path (e.g. wyim@pronghorn:/data/gpfs/assoc/pgl/data/Transgenic/runs)
#     interval  poll interval in seconds (default 300)
#   Pull mode from the receiving side: SRC=user@delta:/scratch/.../runs/seed456 sync_watch.sh pull <local_dest>
#
# Checkpoint convention (implemented by the trainer in #17): each epoch is written to
# <run_dir>/epoch_NN.tmp/ and renamed to <run_dir>/epoch_NN/ when complete, containing
# model.safetensors, optimizer state (optional), trainer_state.json and eval.json. The watcher
# copies an epoch directory once, marks it with <run_dir>/.synced/epoch_NN, and re-copies
# train.err/train.out/*.json every interval. `best` is a symlink maintained by the trainer.
#
# Env: RSYNC_BWLIMIT (KB/s, default unlimited), KEEP_LOCAL (epochs to keep on the source after a
# verified copy; default: keep all), SYNC_OPTIMIZER=0 to skip optimizer.* files (default 1),
# RSYNC_SSH (ssh command, e.g. "ssh -i ~/.ssh/id_ed25519_sync"), ONESHOT=1 to sync once and exit.
set -uo pipefail
MODE=push
if [ "${1:-}" = "pull" ]; then MODE=pull; shift; RUN=${SRC:?SRC=user@host:/path/runs/seedNNN}; DEST=${1:?local dest}; else RUN=${1:?run_dir}; DEST=${2:?dest}; fi
INTERVAL=${3:-${INTERVAL:-300}}
RS=(rsync -a --partial --append-verify --human-readable --timeout=600)
[ -n "${RSYNC_BWLIMIT:-}" ] && RS+=(--bwlimit="$RSYNC_BWLIMIT")
[ -n "${RSYNC_SSH:-}" ] && RS+=(-e "$RSYNC_SSH")
[ "${SYNC_OPTIMIZER:-1}" = "0" ] && RS+=(--exclude 'optimizer*')
log() { echo "$(date -Is) sync: $*"; }

list_epochs() {  # complete epoch dirs on the source, oldest first
  if [ "$MODE" = push ]; then ls -d "$RUN"/epoch_[0-9]* 2>/dev/null | grep -v '\.tmp$' | xargs -rn1 basename | sort
  else local h=${RUN%%:*} p=${RUN#*:}; ${RSYNC_SSH:-ssh} "$h" "ls -d $p/epoch_[0-9]* 2>/dev/null | grep -v '\.tmp$' | xargs -rn1 basename | sort"; fi
}
mark_dir() { if [ "$MODE" = push ]; then echo "$RUN/.synced"; else echo "$DEST/.synced"; fi; }
mkdir -p "$(mark_dir)" 2>/dev/null || true

sync_once() {
  local n=0
  for ep in $(list_epochs); do
    [ -e "$(mark_dir)/$ep" ] && continue
    if "${RS[@]}" "$RUN/$ep/" "$DEST/$ep/"; then
      touch "$(mark_dir)/$ep"; log "copied $ep"; n=$((n+1))
      if [ "$MODE" = push ] && [ -n "${KEEP_LOCAL:-}" ]; then
        ls -d "$RUN"/epoch_[0-9]* | grep -v '\.tmp$' | sort | head -n -"$KEEP_LOCAL" | while read -r old; do
          [ -e "$(mark_dir)/$(basename "$old")" ] && rm -rf "$old" && log "pruned local $(basename "$old")"
        done
      fi
    else log "FAILED $ep (will retry)"; fi
  done
  # logs and small state files every pass; 'best' symlink copied as link
  "${RS[@]}" --include='*.err' --include='*.out' --include='*.json' --include='best' --exclude='*' "$RUN/" "$DEST/" >/dev/null 2>&1 || log "log sync failed"
  return $n
}

log "mode=$MODE run=$RUN dest=$DEST interval=${INTERVAL}s"
while true; do
  sync_once; [ "${ONESHOT:-0}" = "1" ] && exit 0
  if [ "$MODE" = push ] && [ -f "$RUN/TRAINING_DONE" ]; then sync_once; log "training finished; final sync done"; exit 0; fi
  sleep "$INTERVAL"
done
