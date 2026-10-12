#!/usr/bin/env bash
# 72a_est_molecules_per_locus.sh — A-EST molecule counts per eligible locus (B6 MVP sub-filter,
# B6_MVP_PILOT_v1.md §1.2; consumed by 72_mvp_select_loci.py --est-counts).
#
# For every locus of the eligible pool (71_b6_panels.py output, 1-based inclusive start/fin)
# the count is the number of distinct A-EST molecules with a PRIMARY alignment overlapping the
# locus interval, after collapsing identical primary placements (rule (a) of
# B6_MVP_PILOT_v1.md §1.3: same chromosome, same FLAG-0x10 alignment strand, same 1-based POS,
# same CIGAR = one molecule; the
# clone/accession-prefix rule (b) is applied only in the C0-lite build). Exact command per locus:
#
#   samtools view -F 0x904 "$BAM" "${chrom}:${start}-${fin}" \
#     | awk -F'\t' '{ print (int($2/16)%2 ? "-" : "+") "\t" $4 "\t" $6 }' | sort -u | wc -l
#
#   -F 0x904 excludes unmapped (0x4), secondary (0x100) and supplementary (0x800) records, so
#   every read name contributes at most one placement; (strand, POS, CIGAR) identifies a
#   placement on the queried chromosome. Region queries are inclusive on both ends, matching
#   the TSV coordinates.
#
# Usage:
#   revision/scripts/72a_est_molecules_per_locus.sh ELIGIBLE_TSV [BAM] [OUT_TSV] [MAX_RANK]
#     ELIGIBLE_TSV  revision/protocols/b6_eligible_loci_v1.tsv
#     BAM           default evidence/est_align/Athaliana/Athaliana.est.bam (coordinate-sorted, indexed)
#     OUT_TSV       default revision/protocols/b6_est_molecules_v1.tsv (columns gene_id, est_molecules)
#     MAX_RANK      only loci with shuffle_rank <= MAX_RANK (default: all)
# Requires samtools on PATH. Deterministic; output order = shuffle_rank order of the input.
set -euo pipefail

ELIGIBLE=${1:?eligible TSV required}
BAM=${2:-evidence/est_align/Athaliana/Athaliana.est.bam}
OUT=${3:-revision/protocols/b6_est_molecules_v1.tsv}
MAX_RANK=${4:-0}

command -v samtools >/dev/null || { echo "samtools not on PATH" >&2; exit 1; }
[[ -s "$BAM" ]] || { echo "BAM missing: $BAM" >&2; exit 1; }
[[ -s "$BAM.bai" || -s "${BAM%.bam}.bai" || -s "$BAM.csi" ]] || { echo "BAM index missing for $BAM" >&2; exit 1; }
header=$(head -1 "$ELIGIBLE")
[[ "$header" == shuffle_rank$'\t'species_id$'\t'gene_id$'\t'chromosome$'\t'start$'\t'fin$'\t'* ]] || { echo "unexpected header: $header" >&2; exit 1; }

count_locus() {  # chrom start fin -> distinct primary placements (strand, POS, CIGAR); see header comment
  samtools view -F 0x904 "$BAM" "$1:$2-$3" | awk -F'\t' '{ print (int($2/16)%2 ? "-" : "+") "\t" $4 "\t" $6 }' | sort -u | wc -l
}

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
printf 'gene_id\test_molecules\n' > "$tmp"
tail -n +2 "$ELIGIBLE" | sort -t$'\t' -k1,1n | while IFS=$'\t' read -r rank _species gene chrom start fin _rest; do
  if [[ "$MAX_RANK" -gt 0 && "$rank" -gt "$MAX_RANK" ]]; then break; fi
  printf '%s\t%s\n' "$gene" "$(count_locus "$chrom" "$start" "$fin")" >> "$tmp"
done
mv "$tmp" "$OUT"
sha=$(sha256sum "$OUT" | cut -d' ' -f1)
printf '%s  %s\n' "$sha" "$(basename "$OUT")" > "$OUT.sha256"
echo "wrote $OUT ($(($(wc -l < "$OUT") - 1)) loci) sha256=$sha bam=$BAM"
