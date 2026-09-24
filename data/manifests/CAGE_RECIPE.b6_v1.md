# CAGE_RECIPE.b6_v1 -- cap/CAGE TSS peak recipe for the WP2c completeness pass (plan v4.1 D12)

Status: frozen by Writer A on 2026-09-24 as the plan's D12 default. A49.1 froze no processing, so this file fixes
it before WP2c runs. `b6_build_completeness.py` refuses to run without this file's sha256, which is recorded in the
completeness provenance. The JSON block below is the machine-readable part. The builder reads its parameters from
that block, so the text and the code cannot drift apart. Changing any value needs a new file version (`b6_v2`).

Inputs: the runs of `data/manifests/CAGE_INPUTS.b6_v1.tsv` (sha-bound), each fetched run's sha recorded. A species
with no CAGE row is all `5I` on the CAGE-dependent routes.

## 1. Mapping (every assay)

- Tool: minimap2 **2.28-r1209**, the version already frozen in `PROVENANCE.cpu_tools_v1.json`. No new aligner is
  introduced. samtools **1.21** is used for filtering and sorting.
- Reference: the species' B5 reference FASTA, whose sha must equal the corpus `build_manifest.fasta_sha256`.
- Command, read 1 only, single-end:
  `minimap2 -ax sr --secondary=no -t 8 <ref.fa> <read1.fastq.gz> | samtools view -b -F 0x904 -q 20 - | samtools sort -o <run>.cage.bam -`
- Uniqueness filter: primary alignments only (`-F 0x904` drops unmapped, secondary and supplementary records) and
  MAPQ >= 20. The builder re-applies both, fail-closed.

## 2. 5' tag extraction (per assay)

- The tag is the read-1 5' end: the first aligned reference base in read orientation. That is POS-1 for a
  forward record, and the last aligned base for a reverse record. Soft-clipped bases never move the tag.
- **Untemplated-G rule** (CAGE, nAnT-iCAGE, STRIPE-seq, RAMPAGE). Cap-dependent reverse transcription adds up to
  three non-templated G. If the first aligned read base(s), up to 3, are G and mismatch the reference, the tag
  moves inward past them. TSS-seq (oligo-capping) adds no G, so the rule is off for it.
- **STRIPE-seq**: read 1 begins with an 8-nt UMI and `TATAGGG` from the template-switching oligo. The first 15 nt
  are trimmed before mapping with `seqkit subseq -r 16:-1`, and the UMI is appended to the read name as `_<UMI>`.
  Tags are deduplicated on (chrom, strand, tag position, UMI).
- **RAMPAGE**: the 15-nt RT-primer barcode (read 2, positions 1-15) is appended to the read-1 name as `_<BC>` before
  mapping. Tags are deduplicated on (chrom, strand, tag position, barcode).
- **CAGE, nAnT-iCAGE, TSS-seq**: no trimming beyond the aligner's soft clipping and no deduplication. nAnT-iCAGE
  is PCR-free; CAGE and TSS-seq duplicates are kept as tags, following the published pipelines.

## 3. Peak calling (per species, all runs pooled)

- Same-strand tag positions sorted; a cluster is a maximal run in which consecutive tag positions differ by
  <= 20 nt, and a cluster needs >= 5 tags.
- The peak is the modal tag base, with ties broken by the lowest coordinate. Peak rows record tag and library
  counts.

## 4. Use in the codes (A16, design §2)

A read's 5' end within 50 nt of a same-strand peak makes it cap/CAGE-associated. Route (a) additionally needs an
author-approved adapter protocol row. Route (b) needs a TSS cluster whose medoid is within +-50 nt of a same-strand
peak.

```json
{
  "recipe_version": "CAGE_RECIPE.b6_v1",
  "mapper": {"tool": "minimap2", "version": "2.28-r1209",
             "command": "minimap2 -ax sr --secondary=no -t 8 {ref} {read1} | samtools view -b -F 0x904 -q 20 - | samtools sort -o {out} -"},
  "filter": {"mapq_min": 20, "exclude_flags": 2308},
  "assays": {
    "CAGE":        {"trim5": 0,  "g_rule": true,  "dedup": "none"},
    "nAnT-iCAGE":  {"trim5": 0,  "g_rule": true,  "dedup": "none"},
    "STRIPE-seq":  {"trim5": 15, "g_rule": true,  "dedup": "read_name_suffix_umi", "umi_len": 8},
    "RAMPAGE":     {"trim5": 0,  "g_rule": true,  "dedup": "read_name_suffix_umi", "umi_len": 15},
    "TSS-seq":     {"trim5": 0,  "g_rule": false, "dedup": "none"}
  },
  "g_rule_max": 3,
  "peak": {"max_gap": 20, "min_tags": 5, "summit": "modal_base_lowest_coordinate_tie"},
  "association_window": 50
}
```
