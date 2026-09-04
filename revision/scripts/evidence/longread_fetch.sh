#!/usr/bin/env bash
# Stream ENA fastq.gz -> gzipped FASTA (no FASTQ stored). Resumable per run; records provenance.
# usage: longread_fetch.sh <group> <label> <ENA run/project accession> [run accession filter regex]
#   group = ont | pacbio ; output evidence/<group>/<label>/<run>.fa.gz + <run>.md5 + <run>.DONE
set -uo pipefail
GROUP=$1; LABEL=$2; ACC=$3; FILT=${4:-.}
ROOT=/data/gpfs/assoc/pgl/data/Transgenic/evidence
OUT=$ROOT/$GROUP/$LABEL; mkdir -p "$OUT"; cd "$OUT"
API="https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${ACC}&result=read_run&fields=run_accession,instrument_platform,library_strategy,fastq_ftp,submitted_ftp,submitted_format,fastq_md5,read_count,instrument_model&limit=0"
curl -sg -m 120 "$API" > filereport.tsv || { echo "ENA API failed for $ACC"; exit 2; }
SEQKIT=$(ls /data/gpfs/assoc/pgl/bin/conda/conda_envs/*/bin/seqkit 2>/dev/null | head -1)
[ -n "$SEQKIT" ] || { echo "seqkit not found"; exit 3; }
# NOTE (2026-09-02): tabs are IFS whitespace, so `read` collapses consecutive tabs and empty ENA fields
# (e.g. submitted_ftp) shift every later column; translate tabs to \001 first so empty fields survive.
tail -n +2 filereport.tsv | grep -E "$FILT" | tr '\t' '\001' | while IFS=$'\001' read -r run plat strat fq sub fmt md5 rc model; do
  [ -f "$run.DONE" ] && { echo "$run done"; continue; }
  if [ -n "${MAX_READS:-}" ] && [ "${rc:-0}" -gt "$MAX_READS" ]; then echo "$(date -Is) $run skipped: read_count=$rc > MAX_READS=$MAX_READS (subreads?)" >> log; continue; fi
  # An empty instrument_model is not "wrong instrument", it is "we cannot tell" -- the protocol's
  # instrument restrictions (A12 v1.3 Sequel II and later, A14 v1.5) cannot be evaluated at all.
  # Logging that as a skip made the run indistinguishable from a legitimate rejection and it left
  # the evidence set silently (issue #65). Mark it UNRESOLVED so it stays visible, and never DONE.
  if [ -n "${MODEL_RE:-}" ]; then
    if [ -z "$model" ]; then
      echo "$(date -Is) $run UNRESOLVED: instrument_model is empty, MODEL_RE=$MODEL_RE cannot be evaluated" >> log
      touch "$run.UNRESOLVED"; continue
    fi
    if ! echo "$model" | grep -Eq "$MODEL_RE"; then echo "$(date -Is) $run skipped: model=$model" >> log; continue; fi
  fi
  # A mixed BioProject offers whatever it holds: the accession filter matches the sequencing PLATFORM
  # (OXFORD_NANOPORE / PACBIO), never the library. PRJNA953663 carries ChIP-Seq, ONT WGS and ONT RNA-Seq
  # together, and the WGS run was selected as transcript evidence (issue #63). Allow only libraries that
  # are transcripts. OTHER has to be allowed because ENA labels direct-RNA runs that way (all 13 dRNA runs
  # here are OTHER), so it stays a loophole -- the accepted strategy is written to the log for auditing.
  STRAT_ALLOW=${STRAT_ALLOW:-'^(RNA-Seq|FL-cDNA|ncRNA-Seq|OTHER)$'}
  if ! echo "$strat" | grep -Eq "$STRAT_ALLOW"; then
    echo "$(date -Is) $run skipped: library_strategy=$strat is not transcript evidence" >> log; continue
  fi
  # Protocol v1.3/v1.5 admits PacBio only at CCS/FLNC level. MODEL_RE checks the instrument, which
  # passes for a Sequel II subreads run, and MAX_READS never fires because real subread sets here hold
  # 0.25-1.6 M reads. Reject the data product by name instead (author decision 2026-09-03, issue #60).
  if echo "$fq $sub" | grep -q '_subreads'; then
    echo "$(date -Is) $run skipped: subreads file (protocol v1.3/v1.5 admits CCS/FLNC only)" >> log; continue
  fi
  # prefer submitted HQ/FLNC fastq when it is a FASTQ; else ENA fastq (first file)
  url=""; case "$fmt" in *FASTQ*) url=$(echo "$sub" | tr ';' '\n' | grep -E 'hq|flnc|fastq' | head -1);; esac
  [ -n "$url" ] || url=$(echo "$fq" | tr ';' '\n' | head -1)
  [ -n "$url" ] || { echo "$run: no fastq URL" | tee -a log; continue; }
  echo "$(date -Is) $run $plat $model $strat reads=$rc url=$url" >> log
  # Download the raw fastq.gz to disk FIRST, then convert. The old form streamed
  # curl | zcat | fq2fa | gzip straight into the output and deleted the partial on failure,
  # so a transfer that died at minute 46 of 47 threw away everything: the partial was
  # *converted* output, which no byte-range resume can continue. PRJDB38182's runs are
  # 5.63 GB each and failed that way twice (2026-09-03). Now `curl -C -` resumes a raw
  # partial across attempts, and a stalled stream is cut early instead of hanging to -m:
  # --speed-limit/--speed-time abort below 10 KB/s for 120 s, which is cheap because the
  # bytes already on disk survive. The raw file is removed only after a successful convert.
  raw="$run.raw.gz"
  src_md5=$(echo "$md5" | tr ';' '\n' | head -1)
  ok=0
  for try in 1 2 3 4; do
    # --http1.1: over HTTP/2 this endpoint killed ranged requests within a few hundred KB with
    # `OpenSSL SSL_read: unexpected eof` (curl 56). Measured 2026-09-03: -C - over h2 died at
    # 71 KB while the same request with --http1.1 pulled 53.9 MB in 40 s. Keep it — but do NOT
    # read it as the cause of the day's failures. Half an hour later every protocol collapsed
    # together (FTP 6 KB/s, HTTPS 11 KB/s, both erroring) on a host that had served 150 GB
    # earlier the same day, which looks like ENA shedding our connections after we hammered it
    # with 7 parallel streams and repeated ranged retries. The real defence is below: resume,
    # low concurrency, and waiting between rounds.
    curl -sL -C - --http1.1 -m 36000 --retry 10 --retry-all-errors --retry-delay 5 \
         --speed-limit 10240 --speed-time 120 \
         "https://${url#ftp://}" -o "$raw" 2>> "$run.curl.err"
    crc=$?
    if [ ! -s "$raw" ]; then
      echo "$(date -Is) $run try=$try no bytes (curl rc=$crc)" >> log; sleep $((try*30)); continue
    fi
    if ! gzip -t "$raw" 2>/dev/null; then
      echo "$(date -Is) $run try=$try incomplete: $(stat -c%s "$raw") B so far (curl rc=$crc), resuming" >> log
      sleep $((try*30)); continue
    fi
    # ENA publishes fastq_md5; compare it now that we hold the actual file (issue #6).
    if [ -n "$src_md5" ]; then
      got=$(md5sum "$raw" | cut -d' ' -f1)
      if [ "$got" != "$src_md5" ]; then
        echo "$(date -Is) $run try=$try ENA md5 mismatch (published $src_md5, got $got) — refetching" >> log
        rm -f "$raw"; sleep $((try*30)); continue
      fi
      echo "$(date -Is) $run ENA fastq_md5 verified $src_md5" >> log
    fi
    if zcat "$raw" | "$SEQKIT" fq2fa 2>/dev/null | gzip -1 > "$run.fa.gz.part" && [ -s "$run.fa.gz.part" ]; then
      n=$(zcat "$run.fa.gz.part" | grep -c '^>'); if [ "$n" -gt 0 ]; then ok=1; break; fi
    fi
    echo "$(date -Is) $run try=$try convert produced no records" >> log
    rm -f "$run.fa.gz.part"; sleep $((try*30))
  done
  # The raw partial is deliberately KEPT on failure so the next attempt resumes it.
  # A FAILED marker, not just a log line. The while-loop runs in a subshell (it is fed by a pipe),
  # so a counter variable cannot survive it -- which is exactly why the completion line below could
  # not see failures and signed off "0 runs DONE, 0 skipped" on a dataset where every run had failed
  # (2026-09-03, PRJDB38182). A file survives, is greppable, and says which run.
  [ $ok -eq 1 ] || { echo "$(date -Is) $run FAILED (raw kept for resume: $(stat -c%s "$raw" 2>/dev/null || echo 0) B)" >> log; touch "$run.FAILED"; rm -f "$run.fa.gz.part"; continue; }
  mv "$run.fa.gz.part" "$run.fa.gz"; md5sum "$run.fa.gz" > "$run.md5"
  [ -n "$src_md5" ] && echo "$src_md5  $(basename "$url")" > "$run.source.md5"
  rm -f "$raw" "$run.curl.err"
  rm -f "$run.FAILED"
  echo "$(date -Is) $run reads=$n source_url=$url" >> log; touch "$run.DONE"
done
unresolved=$(ls *.UNRESOLVED 2>/dev/null | wc -l)
# Count only failures that are still failures: a run that failed earlier and later succeeded has
# had its marker removed, and a stale marker for a run now DONE must not inflate the count.
failed=$(ls *.FAILED 2>/dev/null | sed 's/\.FAILED$//' | while read -r rr; do [ -f "$rr.DONE" ] || echo x; done | wc -l)
echo "$(date -Is) $LABEL finished: $(ls *.DONE 2>/dev/null | wc -l) runs DONE, $(grep -c skipped log) skipped, $failed FAILED, $unresolved UNRESOLVED" >> log
# Neither an unresolved run nor a failed one leaves a finished dataset. Exit non-zero so a driver
# that chains datasets cannot read "no crash" as "everything accounted for".
#   4 = a run whose instrument_model ENA never published, so the protocol rule cannot be evaluated (#65)
#   5 = a run that was selected, attempted and did not arrive
# Before this, a dataset whose every run failed signed off "0 runs DONE, 0 skipped" and exited 0,
# which is indistinguishable from a dataset that had nothing to fetch. That is how six Col-0 runs
# of PRJDB38182 sat missing for four hours on 2026-09-03 while the tree looked complete.
if [ "$unresolved" -gt 0 ]; then
  echo "$(date -Is) $LABEL: $unresolved run(s) have no instrument_model and were not fetched; see *.UNRESOLVED" >> log
  exit 4
fi
if [ "$failed" -gt 0 ]; then
  echo "$(date -Is) $LABEL: $failed run(s) were attempted and did not complete; see *.FAILED (raw partials kept for resume)" >> log
  exit 5
fi
