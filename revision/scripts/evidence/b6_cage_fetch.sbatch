#!/usr/bin/env bash
#SBATCH --job-name=b6_cage_fetch
#SBATCH --account=cpu-s2-pgl-0
#SBATCH --partition=cpu-s2-core-0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/cage_v1/logs/%x_%A_%a.out
# Fetch one ENA FASTQ of the A49.1 cap/CAGE inputs (data/manifests/CAGE_INPUTS.b6_v1.tsv) per array task.
#
#   sbatch --array=0-<N-1>%4 b6_cage_fetch.sbatch   (%4: more concurrent transfers draw ENA 403s)
#
# Input  $OUTROOT/fetch.tsv  (header: species bioproject run experiment sample layout url md5 bytes), one row per
#        FASTQ, resolved from the ENA filereport of each designated BioProject (explicit run lists only).
# Output $OUTROOT/<species>/<bioproject>/<run>/<file>.fastq.gz  + <file>.DONE (md5=, bytes=, url=, fetched=)
# The file is written to a temporary name, its md5 must equal ENA's fastq_md5 and its size ENA's fastq_bytes, and only
# then is it renamed into place (fail-closed; a finished file with a matching DONE is skipped).
set -Eeuo pipefail
ROOT=${B6_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic}
OUTROOT=$ROOT/evidence/cage_v1
LIST=${B6_CAGE_LIST:-$OUTROOT/fetch.tsv}
fail() { echo "b6_cage_fetch: $*" >&2; exit 3; }
[[ -f $LIST ]] || fail "missing $LIST"
IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset}
LINE=$(awk -v n=$((IDX + 2)) 'NR == n' "$LIST")
[[ -n $LINE ]] || fail "no row $IDX in $LIST"
IFS=$'\t' read -r SP BP RUN EXP SAMPLE LAYOUT URL MD5 BYTES <<< "$LINE"
[[ $URL == https://ftp.sra.ebi.ac.uk/* && $MD5 =~ ^[0-9a-f]{32}$ && $BYTES =~ ^[0-9]+$ ]] || fail "bad row $IDX: $LINE"
OUT=$OUTROOT/$SP/$BP/$RUN
mkdir -p "$OUT"
F=$OUT/$(basename "$URL")
if [[ -f $F && -f $F.DONE ]] && grep -qx "md5=$MD5" "$F.DONE"; then
  echo "skip $F (DONE)"; exit 0
fi
curl --fail -sSI -m 60 https://ftp.sra.ebi.ac.uk/vol1/fastq/ > /dev/null || fail "no route to ENA from $(hostname)"
# ENA drops long transfers (curl 18) and throttles bursts with 403 (observed 2026-09-23, first run 6171128): retry up
# to ATTEMPTS times with growing back-off, RESUMING the same partial file (-C -), alternating https and ftp.  The
# partial is kept across attempts and across array jobs (fixed name); only the final size + md5 check decides.
TMP=$F.part
ATTEMPTS=${B6_CAGE_ATTEMPTS:-12}
FTP_URL=ftp://${URL#https://}
ok=0
for a in $(seq 1 "$ATTEMPTS"); do
  u=$URL; (( a % 2 == 0 )) && u=$FTP_URL
  if curl --fail -sS -L -C - -o "$TMP" "$u"; then ok=1; break; fi
  rc=$?
  echo "attempt $a/$ATTEMPTS failed (curl $rc) on $u" >&2
  # a 416 (range not satisfiable) after a complete transfer is fine: fall through to the checks
  [[ -f $TMP && $(stat -c %s "$TMP") == "$BYTES" ]] && { ok=1; break; }
  sleep $(( 30 * a ))
done
[[ $ok == 1 ]] || fail "download failed after $ATTEMPTS attempts: $URL (partial kept at $TMP)"
got_bytes=$(stat -c %s "$TMP")
[[ $got_bytes == "$BYTES" ]] || { rm -f "$TMP"; fail "$F: $got_bytes bytes != ENA $BYTES (partial removed)"; }
got_md5=$(md5sum "$TMP" | cut -d' ' -f1)
[[ $got_md5 == "$MD5" ]] || { rm -f "$TMP"; fail "$F: md5 $got_md5 != ENA $MD5 (partial removed)"; }
mv "$TMP" "$F"
printf 'md5=%s\nbytes=%s\nurl=%s\nrun=%s\nexperiment=%s\nsample=%s\nlayout=%s\nfetched=%s\nhost=%s\n' \
  "$MD5" "$BYTES" "$URL" "$RUN" "$EXP" "$SAMPLE" "$LAYOUT" "$(date -Is)" "$(hostname)" > "$F.DONE"
echo "ok $F"
