#!/usr/bin/env bash
# 런 단위 병렬 수집. longread_fetch.sh의 4번째 인자(접근번호 정규식)로 한 런씩 맡긴다.
# .DONE 표식이 중복을 막고, 런마다 .part 파일이 달라 서로 간섭하지 않는다.
# 단일 스트림이 약 1 MB/s에 묶여 있어(2026-09-03 실측) 병렬도가 곧 처리량이다.
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
P=${P:-8}
while IFS=$'\t' read -r g l a r; do
  [ -n "$r" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  ./longread_fetch.sh "$g" "$l" "$a" "${r}[[:space:]]" >> ont_parallel.log 2>&1 &
done < ont_parallel.tsv
wait
echo "$(date -Is) ONT PARALLEL FINISHED" >> longread.log
