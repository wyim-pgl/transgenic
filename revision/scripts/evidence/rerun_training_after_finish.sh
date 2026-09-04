#!/usr/bin/env bash
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
until grep -q "TRAINING LONGREAD ALL FINISHED" longread.log 2>/dev/null; do sleep 600; done
echo "$(date -Is) rerunning training driver for runs skipped by the tab-collapse bug" >> longread.log
./training_run_all.sh >> training_nohup.out 2>&1
