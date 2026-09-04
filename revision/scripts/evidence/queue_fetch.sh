#!/usr/bin/env bash
# 큐 파일(그룹\t라벨\t접근번호\t런) 하나를 런 단위로 병렬 수집한다.
#   usage: P=6 ./queue_fetch.sh <queue.tsv> <로그이름>
# longread_fetch.sh의 4번째 인자는 filereport '행 전체'에 걸리는 정규식이므로 접근번호 뒤에
# [[:space:]]를 붙여 고정한다(^…$로 앵커를 걸면 한 줄도 안 맞는다 — 2026-09-03에 겪음).
set -uo pipefail
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
Q=${1:?queue tsv}; LOG=${2:-queue}; P=${P:-6}
while IFS=$'\t' read -r g l a r; do
  [ -n "${r:-}" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  ./longread_fetch.sh "$g" "$l" "$a" "${r}[[:space:]]" >> "$LOG.log" 2>&1 &
done < "$Q"
wait
echo "$(date -Is) QUEUE $Q FINISHED" >> longread.log
