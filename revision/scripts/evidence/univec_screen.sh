#!/usr/bin/env bash
# UniVec_Core vector screening of one EST set (protocol A21). Runs anywhere with BLAST+ 2.x and seqkit
# (Delta: apptainer exec $C/blast.sif ...; pronghorn: conda env). Resumable per chunk.
#   univec_screen.sh <species_label> [threads]
# Reads  evidence/est/<species>/est.fa.gz
# Writes evidence/est/<species>/univec/{hits.tsv,est.univec.fa.gz,report.tsv,summary.json,PROVENANCE.txt,DONE}
set -euo pipefail
SP=${1:?species label}; T=${2:-16}
ROOT=${EVIDENCE_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}
REPO=${TRANSGENIC_REPO:-$ROOT/../transgenic}
DBDIR=${UNIVEC_DIR:-$ROOT/univec}
BLASTN=${BLASTN:-blastn}; MAKEBLASTDB=${MAKEBLASTDB:-makeblastdb}; SEQKIT=${SEQKIT:-seqkit}; PY=${PY:-python3}
mkdir -p "$DBDIR"
if [ ! -f "$DBDIR/UniVec_Core.nsq" ]; then
  curl -fsSL --retry 3 -o "$DBDIR/UniVec_Core" https://ftp.ncbi.nlm.nih.gov/pub/UniVec/UniVec_Core
  curl -fsSL --retry 3 -o "$DBDIR/README.uv" https://ftp.ncbi.nlm.nih.gov/pub/UniVec/README.uv || true
  md5sum "$DBDIR/UniVec_Core" > "$DBDIR/UniVec_Core.md5"
  grep -m1 -i "build\|version" "$DBDIR/README.uv" > "$DBDIR/UniVec_Core.version" 2>/dev/null || date -Is > "$DBDIR/UniVec_Core.version"
  $MAKEBLASTDB -in "$DBDIR/UniVec_Core" -dbtype nucl -out "$DBDIR/UniVec_Core" > "$DBDIR/makeblastdb.log"
fi
IN="$ROOT/est/$SP/est.fa.gz"; OUT="$ROOT/est/$SP/univec"; mkdir -p "$OUT/chunks"; cd "$OUT"
[ -f DONE ] && { echo "$SP univec done"; exit 0; }
# chunked BLAST with VecScreen parameters; one hit table per chunk, resumable
[ -f chunks/SPLIT_DONE ] || { $SEQKIT split2 -s 200000 -O chunks -j "$T" "$IN" > /dev/null && touch chunks/SPLIT_DONE; }
for f in chunks/*.fa.gz chunks/*.fasta.gz chunks/*.fa chunks/*.fasta; do
  [ -e "$f" ] || continue; b=$(basename "$f"); [ -f "chunks/$b.hits" ] && continue
  ( [[ "$f" == *.gz ]] && zcat "$f" || cat "$f" ) | $BLASTN -task blastn -db "$DBDIR/UniVec_Core" -reward 1 -penalty -5 -gapopen 3 -gapextend 3 \
      -dust yes -soft_masking true -evalue 700 -searchsp 1750000000000 -num_threads "$T" \
      -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue score qlen" > "chunks/$b.hits.tmp" && mv "chunks/$b.hits.tmp" "chunks/$b.hits"
done
cat chunks/*.hits > hits.tsv
$PY "$REPO/revision/scripts/61_univec_trim.py" --fasta "$IN" --hits hits.tsv --out est.univec.fa.gz --report report.tsv --summary summary.json --min-len 100
md5sum est.univec.fa.gz > est.univec.fa.gz.md5
{ echo "date=$(date -Is)"; echo "input=$IN"; echo "input_md5=$(cut -d' ' -f1 "$ROOT/est/$SP/est.fa.gz.md5" 2>/dev/null)";
  echo "univec=UniVec_Core $(cat "$DBDIR/UniVec_Core.version") md5=$(cut -d' ' -f1 "$DBDIR/UniVec_Core.md5")";
  echo "blastn=$($BLASTN -version | head -1)"; echo "params=-task blastn -reward 1 -penalty -5 -gapopen 3 -gapextend 3 -dust yes -soft_masking true -evalue 700 -searchsp 1750000000000";
  echo "trimmer=revision/scripts/61_univec_trim.py min_len=100 terminal_nt=25"; echo "summary=$(cat summary.json | tr -d '\n ')"; } > PROVENANCE.txt
touch DONE; echo "$SP univec finished: $(cat summary.json | tr -d '\n ')"
