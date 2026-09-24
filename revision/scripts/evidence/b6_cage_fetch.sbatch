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
#   sbatch --array=0-<N-1>%8 b6_cage_fetch.sbatch
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
TMP=$F.part.$SLURM_JOB_ID
trap 'rm -f "$TMP"' EXIT
curl --fail -sS -L --retry 5 --retry-delay 30 -C - -o "$TMP" "$URL" || fail "download failed: $URL"
got_bytes=$(stat -c %s "$TMP")
[[ $got_bytes == "$BYTES" ]] || fail "$F: $got_bytes bytes != ENA $BYTES"
got_md5=$(md5sum "$TMP" | cut -d' ' -f1)
[[ $got_md5 == "$MD5" ]] || fail "$F: md5 $got_md5 != ENA $MD5"
mv "$TMP" "$F"
trap - EXIT
printf 'md5=%s\nbytes=%s\nurl=%s\nrun=%s\nexperiment=%s\nsample=%s\nlayout=%s\nfetched=%s\nhost=%s\n' \
  "$MD5" "$BYTES" "$URL" "$RUN" "$EXP" "$SAMPLE" "$LAYOUT" "$(date -Is)" "$(hostname)" > "$F.DONE"
echo "ok $F"
