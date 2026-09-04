#!/usr/bin/env bash
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
until grep -q "ALL LANES FINISHED" est/lanes.log 2>/dev/null; do sleep 600; done
echo "$(date -Is) EST finished; starting long-read FASTA downloads" >> longread.log
./longread_run_all.sh >> longread.log 2>&1
