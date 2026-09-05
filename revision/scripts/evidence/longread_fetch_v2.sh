#!/usr/bin/env bash
# ENA FASTQ -> one validated FASTA per run. See README.md for the v2 interface.
set -Eeuo pipefail
fail() { echo "longread_fetch: $*" >&2; exit 6; }
[[ $# -ge 3 && $# -le 4 ]] || fail 'usage: <group> <label> <accession> [RUN_RE]'
SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
GROUP=$1 LABEL=$2 ACC=$3
if [[ $# == 4 ]]; then
  case "$4" in *'[[:space:]]'|OXFORD_NANOPORE|PACBIO|PACBIO_SMRT) fail 'legacy whole-row fourth argument; use RUN_RE, PLATFORM_RE or SUBMITTED_RE';; esac
  [[ ! ${RUN_RE+x} ]] || fail 'RUN_RE specified twice'
  RUN_RE=$4
fi
[[ ! ${FILT+x} ]] || fail 'FILT is obsolete; use column filters'
# Unset optional filters impose no constraint. Explicitly empty regexes still require metadata.
STRAT_ALLOW=${STRAT_ALLOW-'^(RNA-Seq|FL-cDNA|ncRNA-Seq|OTHER)$'}
for key in RUN_RE PLATFORM_RE SUBMITTED_RE MODEL_RE STRAT_ALLOW; do
  if [[ ${!key+x} ]]; then
    status=0; grep -Eq -- "${!key}" <<< '' || status=$?
    [[ $status != 2 ]] || fail "invalid $key regex"
  fi
done
[[ ! ${MAX_READS+x} || $MAX_READS =~ ^[0-9]+$ ]] || fail 'invalid MAX_READS'
ROOT=${LONGREAD_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}
OUT=$ROOT/$GROUP/$LABEL
mkdir -p "$OUT"; cd "$OUT"
log() { echo "$(date -Is) $*" | tee -a log >&2; }
report=$(mktemp "$OUT/.filereport.XXXXXX")
trap 'rm -f "$report"' EXIT
fields=run_accession,instrument_platform,library_strategy,fastq_ftp,submitted_ftp,submitted_format,fastq_md5,read_count,instrument_model,submitted_md5
if [[ ${LONGREAD_FILEREPORT+x} ]]; then
  cp "$LONGREAD_FILEREPORT" "$report"
else
  curl --fail -sg --retry 3 -m 120 "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${ACC}&result=read_run&fields=$fields&limit=0" > "$report" || { log "ENA API failed for $ACC"; exit 5; }
fi
# Never parse TSV with whitespace IFS: translate tabs to \001 to preserve empty fields (#65).
IFS=$'\001' read -r -a header < <(head -1 "$report" | tr '\t' '\001')
declare -A col=()
for i in "${!header[@]}"; do col[${header[$i]}]=$i; done
for key in run_accession instrument_platform library_strategy fastq_ftp submitted_ftp submitted_format fastq_md5 read_count instrument_model; do
  [[ ${col[$key]+x} ]] || fail "filereport missing column $key"
done
report_hash=$(sha256sum "$report" | cut -d' ' -f1)
script_hash=$(sha256sum "$SCRIPT" | cut -d' ' -f1)
# Concurrent run callers each parse their private snapshot, never the shared report being replaced.
cp "$report" "$report.saved"; mv "$report.saved" filereport.tsv
SEQKIT=${SEQKIT:-}
if [[ -z $SEQKIT ]]; then
  for tool in /data/gpfs/assoc/pgl/bin/conda/conda_envs/*/bin/seqkit; do
    [[ ! -x $tool ]] || { SEQKIT=$tool; break; }
  done
fi
selected=0 unresolved=0 failed=0 done_count=0 skipped=0 row_number=0
mark() {
  log "$run $1: $2"
  rm -f "$run.FAILED" "$run.UNRESOLVED"
  printf '%s\n' "$2" > "$run.$1"
}
get() { value=${row[${col[$1]}]-}; }
while IFS=$'\001' read -r -a row; do
  [[ ${#row[@]} -gt 0 ]] || continue
  row_number=$((row_number+1))
  get run_accession; run=$value
  if [[ -z $run ]]; then
    run="row_$row_number"; selected=$((selected+1)); unresolved=$((unresolved+1))
    mark UNRESOLVED "run_accession is empty; cannot identify run"; continue
  fi
  [[ $run =~ ^[SED]RR[0-9]+$ ]] || fail "invalid or missing run_accession: $run"
  get instrument_platform; plat=$value
  get library_strategy; strat=$value
  get submitted_ftp; sub=$value
  get instrument_model; model=$value
  missing='' reject=0
  for pair in RUN_RE:run PLATFORM_RE:plat SUBMITTED_RE:sub MODEL_RE:model STRAT_ALLOW:strat; do
    key=${pair%:*}; field=${pair#*:}
    if [[ ${!key+x} ]]; then
      if [[ -z ${!field} ]]; then missing+="$key target is empty; "
      elif ! grep -Eq -- "${!key}" <<< "${!field}"; then reject=1
      fi
    fi
  done
  if (( reject )); then skipped=$((skipped+1)); log "$run skipped: column filter"; continue; fi
  selected=$((selected+1))
  # Evaluate missing filtered metadata even when a historical DONE exists; don't mutate that output.
  if [[ -n $missing ]]; then
    unresolved=$((unresolved+1)); mark UNRESOLVED "$missing"; continue
  fi
  [[ ! -f $run.DONE ]] || { done_count=$((done_count+1)); log "$run done (existing; separate re-audit required)"; continue; }
  get read_count; rc=$value
  if [[ ! $rc =~ ^[0-9]+$ || $rc == 0 ]]; then
    unresolved=$((unresolved+1)); mark UNRESOLVED 'missing/invalid ENA read_count'; continue
  fi
  if [[ ${MAX_READS+x} ]] && (( rc > MAX_READS )); then skipped=$((skipped+1)); log "$run skipped: read_count=$rc > MAX_READS=$MAX_READS"; continue; fi
  get fastq_ftp; fq=$value
  if [[ "$fq $sub" == *'_subreads'* ]]; then skipped=$((skipped+1)); log "$run skipped: subreads (CCS/FLNC only)"; continue; fi
  get fastq_md5; checksums=$value; urls=$fq; source=fastq
  get submitted_format; fmt=$value
  # Preserve submitted FASTQ preference, but never compare its bytes to fastq_md5.
  if [[ -n $sub && $fmt == *FASTQ* && $sub =~ hq|flnc|fastq ]]; then
    urls=$sub; source=submitted; checksums=''
    [[ ! ${col[submitted_md5]+x} ]] || { get submitted_md5; checksums=$value; }
    if [[ $fmt != FASTQ ]]; then
      # Check each position rather than accidentally ingest a mixed BAM/FASTQ submission.
      IFS=';' read -r -a formats <<< "$fmt"
      mixed=0; for f in "${formats[@]}"; do [[ $f == FASTQ ]] || mixed=1; done
      if (( mixed )); then unresolved=$((unresolved+1)); mark UNRESOLVED 'mixed submitted formats'; continue; fi
    fi
  fi
  IFS=';' read -r -a url_list <<< "$urls"
  IFS=';' read -r -a md5_list <<< "$checksums"
  bad=0
  [[ -n $urls && ${#url_list[@]} == ${#md5_list[@]} && $urls != *';' && $checksums != *';' ]] || bad=1
  for digest in "${md5_list[@]}"; do [[ $digest =~ ^[[:xdigit:]]{32}$ ]] || bad=1; done
  for url in "${url_list[@]}"; do [[ -n $url ]] || bad=1; done
  if (( bad )); then unresolved=$((unresolved+1)); mark UNRESOLVED "missing or unpaired ${source}_ftp/${source}_md5"; continue; fi
  if [[ ! -x $SEQKIT ]]; then failed=$((failed+1)); mark FAILED 'seqkit not found'; continue; fi
  log "$run $plat $model $strat reads=$rc files=${#url_list[@]} source=$source"
  raws=(); ok=1
  : > "$run.source.md5.part"
  for i in "${!url_list[@]}"; do
    url=${url_list[$i]}; digest=${md5_list[$i],,}
    # URL+digest identity prevents resuming bytes belonging to a different metadata revision.
    identity=$(printf '%s\n%s' "$url" "$digest" | sha256sum | cut -d' ' -f1)
    raw="$run.$source.$i.$identity.raw.gz"; raws+=("$raw")
    case "$url" in https://*|http://*) ;; ftp://*) url="https://${url#ftp://}";; *) url="https://$url";; esac
    file_ok=0
    for try in 1 2 3 4; do
      # Complete raws need no network access on restart. Incomplete raws resume with HTTP/1.1.
      if [[ ! -s $raw ]] || ! gzip -t "$raw" 2>/dev/null; then
        curl --fail -sL -C - --http1.1 -m 36000 --retry 10 --retry-all-errors --retry-delay 5 \
          --speed-limit 10240 --speed-time 120 "$url" -o "$raw" 2>> "$run.curl.err" || true
      fi
      if gzip -t "$raw" 2>/dev/null; then
        got=$(md5sum "$raw" | cut -d' ' -f1)
        if [[ $got == "$digest" ]]; then file_ok=1; break; fi
        log "$run file=$i ENA md5 mismatch: expected=$digest got=$got"
        # Retain corrupt sources for diagnosis, but permit a clean retry at the original path.
        mv "$raw" "$raw.bad.$(date +%s%N)"
      fi
      [[ $try == 4 ]] || sleep "${LONGREAD_RETRY_DELAY:-$((try*30))}"
    done
    if (( ! file_ok )); then ok=0; break; fi
    log "$run file=$i ENA ${source}_md5 verified $digest"
    printf '%s  %s\n' "$digest" "${url_list[$i]}" >> "$run.source.md5.part"
  done
  if (( ok )); then
    if ! { gzip -cd "${raws[@]}" | "$SEQKIT" fq2fa | gzip -1 > "$run.fa.gz.part"; }; then ok=0; fi
  fi
  if (( ok )); then
    n=$(gzip -cd "$run.fa.gz.part" | grep -c '^>') || n=0
    if [[ $n != "$rc" ]]; then log "$run read_count mismatch: converted=$n ENA=$rc"; ok=0; fi
  fi
  if (( ! ok )); then
    failed=$((failed+1)); mark FAILED 'download/conversion/count validation failed; raw sources kept for resume'
    rm -f "$run.fa.gz.part" "$run.source.md5.part"; continue
  fi
  mv "$run.fa.gz.part" "$run.fa.gz"
  md5sum "$run.fa.gz" > "$run.md5.part"; mv "$run.md5.part" "$run.md5"
  mv "$run.source.md5.part" "$run.source.md5"
  printf 'version=2\nscript_sha256=%s\nfilereport_sha256=%s\nreads=%s\nfiles=%s\n' "$script_hash" "$report_hash" "$n" "${#raws[@]}" > "$run.DONE.part"
  mv "$run.DONE.part" "$run.DONE"
  rm -f "$run.FAILED" "$run.UNRESOLVED" "${raws[@]}" "$run.curl.err"
  done_count=$((done_count+1)); log "$run DONE reads=$n"
done < <(tail -n +2 "$report" | tr '\t' '\001')
log "$LABEL finished: $selected selected, $done_count runs DONE, $skipped skipped, $failed FAILED, $unresolved UNRESOLVED"
(( selected > 0 )) || fail 'empty selection; check caller migration and column filters'
(( failed == 0 )) || exit 5
(( unresolved == 0 )) || exit 4
