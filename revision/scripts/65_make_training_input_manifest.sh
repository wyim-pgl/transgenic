#!/usr/bin/env bash
# training_input/ 에 무엇이 들어갔는지 표로 만든다. 언제든 다시 돌려도 된다.
#   usage: bash training_input/make_manifest.sh
# 심볼릭 링크는 따라가서 실제 파일의 크기·md5를 적는다(링크 자체가 아니라 내용이 알고 싶은 것이므로).
set -uo pipefail
T=$(cd "$(dirname "$0")" && pwd)
OUT=$T/MANIFEST.tsv
purpose() { case "$1" in
  db)        echo "B5 frozen training database (272,224 rows; the training input)";;
  splits)    echo "orthogroup split table - replaces random_split (#14)";;
  configs)   echo "frozen training recipe (A25/A26; win_v3 is canonical)";;
  manifests) echo "species manifest + dataset-role manifests frozen into protocol section 1";;
  qc)        echo "GeenuFF and Swiss-Prot flags for A22 loss masking";;
  protein)   echo "OrthoDB v12 Viridiplantae, leakage-filtered - C2 label resource (A19)";;
  freeze)    echo "freeze records of the B5 database (content hashes, provenance)";;
  genomes)   echo "reference FASTA (symlink to Transgenic/genomes/)";;
  *)         echo "-";; esac; }
# Fail closed on a genome that is not one of the nine training species. Zmays and Slycopersicum are
# the held-out test species; a test genome sitting in a folder called training_input is the kind of
# thing nobody notices until it has been shipped. Four of them were here on the first pass.
MANI=$T/manifests/b5_species_v1.tsv
# A basename gate is not a content gate: renaming Zmays (a held-out TEST species) to an allowed
# training-species filename passed the old check. Compare the authoritative fasta_md5 of
# b5_species_v1.tsv column 5 instead, require all nine training species to be PRESENT (not merely
# that nothing forbidden is), and refuse outright when the manifest itself is missing -- the old
# `if [ -f "$MANI" ]` skipped the entire gate in exactly the case where it mattered most.
[ -f "$MANI" ] || { echo "REFUSED: required species manifest is absent: $MANI" >&2; exit 2; }

EXPECTED=$(mktemp "${TMPDIR:-/tmp}/ti_expected.XXXXXX") || exit 2
trap 'rm -f "$EXPECTED"' EXIT HUP INT TERM

# basename<TAB>authoritative fasta_md5, one line per species; exactly nine are required.
# NOTE: awk's `exit N` still runs END, so an END that calls exit would overwrite the code set
# here. The error is carried in `err` and END re-raises it instead of testing the row count.
awk -F '\t' '
  NR == 1 {
    if ($1 != "species_id" || $4 != "fasta" || $5 != "fasta_md5") { err = 10; exit err }
    next
  }
  NF {
    n++
    f = $4; sub(/^.*\//, "", f)
    if (seen[f]++)                       { err = 11; exit err }
    if ($5 !~ /^[[:xdigit:]]{32}$/)      { err = 12; exit err }
    print f "\t" tolower($5)
  }
  END {
    if (err) exit err
    if (n != 9) exit 13
  }
' "$MANI" > "$EXPECTED"
manifest_rc=$?
case "$manifest_rc" in
  0)  ;;
  10) echo "REFUSED: unexpected header in $MANI" >&2; exit 2 ;;
  11) echo "REFUSED: duplicate FASTA basename in $MANI" >&2; exit 2 ;;
  12) echo "REFUSED: invalid fasta_md5 in $MANI" >&2; exit 2 ;;
  13) echo "REFUSED: $MANI does not name exactly nine training species" >&2; exit 2 ;;
  *)  echo "REFUSED: could not parse $MANI (awk exit $manifest_rc)" >&2; exit 2 ;;
esac
[ "$(wc -l < "$EXPECTED")" -eq 9 ] || { echo "REFUSED: expected nine species rows, got $(wc -l < "$EXPECTED")" >&2; exit 2; }

bad=0
# Every declared training genome must be present AND have the declared content.
while IFS=$'\t' read -r name expected_md5; do
  g=$T/genomes/$name
  if [ ! -e "$g" ]; then
    echo "REFUSED: missing training genome $name" >&2; bad=1; continue
  fi
  got_md5=$(md5sum "$(readlink -f "$g")" 2>/dev/null | cut -d' ' -f1)
  if [ "$got_md5" != "$expected_md5" ]; then
    echo "REFUSED: $name md5 mismatch: expected $expected_md5, got ${got_md5:-unreadable}" >&2; bad=1
  fi
done < "$EXPECTED"

# And nothing else may sit in genomes/.
for g in "$T"/genomes/*; do
  [ -e "$g" ] || continue
  name=$(basename "$g")
  awk -F '\t' -v n="$name" '$1 == n { found = 1 } END { exit !found }' "$EXPECTED" \
    || { echo "REFUSED: $name is not a training genome in $(basename "$MANI")" >&2; bad=1; }
done

[ "$bad" -eq 0 ] || { echo "genomes/ failed the nine-species content gate; refusing to write the manifest" >&2; exit 2; }

# The manifest is written to a temporary file and renamed only once every input has been
# resolved, statted and hashed. Writing straight to MANIFEST.tsv meant a single unreadable file
# left a truncated manifest sitting under the name the bundle is audited by, and `2>/dev/null ||
# echo 0` turned an unhashable input into a plausible-looking row with an empty md5.
TMPOUT=$(mktemp "$T/.MANIFEST.tsv.XXXXXX") || exit 2
trap 'rm -f "$EXPECTED" "$TMPOUT"' EXIT HUP INT TERM

fail=0
{
  printf "category\tfile\tbytes\tmd5\tsource_path\tpurpose\n"
  for d in db splits configs manifests qc protein freeze genomes; do
    [ -d "$T/$d" ] || continue
    for f in "$T/$d"/*; do
      [ -e "$f" ] || continue
      case "$(basename "$f")" in rsync*.log|*.part) continue;; esac
      real=$(readlink -f "$f")
      if [ -z "$real" ] || [ ! -e "$real" ]; then
        echo "REFUSED: $d/$(basename "$f") does not resolve to an existing file" >&2; fail=1; continue
      fi
      bytes=$(stat -Lc%s "$f") || { echo "REFUSED: cannot stat $d/$(basename "$f")" >&2; fail=1; continue; }
      sum=$(md5sum "$real" | cut -d' ' -f1) || { echo "REFUSED: cannot hash $real" >&2; fail=1; continue; }
      [ -n "$sum" ] || { echo "REFUSED: empty md5 for $real" >&2; fail=1; continue; }
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$d" "$(basename "$f")" "$bytes" "$sum" "$real" "$(purpose "$d")"
    done
  done
} > "$TMPOUT"
[ "$fail" -eq 0 ] || { echo "one or more staged inputs could not be resolved, statted or hashed; refusing to write the manifest" >&2; exit 2; }

n=$(($(wc -l < "$TMPOUT")-1)); b=$(awk -F'\t' 'NR>1{s+=$3}END{printf "%.1f", s/1073741824}' "$TMPOUT")
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "TOTAL" "$n files" "$b GB" "-" "-" "generated $(date -Is)" >> "$TMPOUT"
chmod 0644 "$TMPOUT"
mv "$TMPOUT" "$OUT"
echo "wrote $OUT  ($n files, $b GB)"
