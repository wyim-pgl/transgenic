#!/usr/bin/env bash
# 실패하면 30분 기다렸다 다시 시도한다 (저자 지시 2026-09-03 17:2x).
#   usage: ROUNDS=8 WAIT=1800 P=2 ./ath_retry_loop.sh <queue.tsv> <로그이름>
#
# 왜 P를 낮추나: 이 데이터셋의 런은 개당 5.63 GB로, 오늘 성공한 것들보다 한 자릿수 크다.
# 7스트림 병렬에서는 스트림당 속도가 무너져 .part가 정지했고, 단일 스트림은 2 MB/s가 나왔다.
# longread_fetch.sh는 실패 시 .part를 지우므로(:56) 재개가 없다 — 한 번의 시도가 끝까지
# 가야 하고, 그래서 "빠른 스트림 적게"가 "느린 스트림 많이"보다 성공 확률이 높다.
set -uo pipefail
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
Q=${1:?queue tsv}; LOG=${2:-ath_retry_loop}; ROUNDS=${ROUNDS:-8}; WAIT=${WAIT:-1800}; P=${P:-2}

missing() {
  local n=0
  while IFS=$'\t' read -r g l a r; do
    [ -n "${r:-}" ] || continue
    [ -f "$g/$l/$r.DONE" ] || n=$((n+1))
  done < "$Q"
  echo "$n"
}

for round in $(seq 1 "$ROUNDS"); do
  m=$(missing)
  if [ "$m" -eq 0 ]; then
    echo "$(date -Is) ALL DONE (round $round 시작 전)" >> "$LOG.log"; exit 0
  fi
  echo "$(date -Is) round $round/$ROUNDS 시작 — 미수집 $m런, P=$P" >> "$LOG.log"
  P="$P" ./queue_fetch.sh "$Q" "${LOG}_r${round}" >> "$LOG.log" 2>&1
  m=$(missing)
  echo "$(date -Is) round $round 종료 — 미수집 $m런" >> "$LOG.log"
  if [ "$m" -eq 0 ]; then echo "$(date -Is) ALL DONE" >> "$LOG.log"; exit 0; fi
  [ "$round" -lt "$ROUNDS" ] && { echo "$(date -Is) ${WAIT}초 대기 후 재시도" >> "$LOG.log"; sleep "$WAIT"; }
done
echo "$(date -Is) GAVE UP after $ROUNDS rounds — 미수집 $(missing)런" >> "$LOG.log"
exit 1
