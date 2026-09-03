# Frozen B5 database (#16, #50)

`b5_full_v1.db` — the single leakage-controlled tile database the B5 seeds train on. Built as nine
per-species databases (one DuckDB file cannot take parallel writers) and merged by
`scripts/merge_b5_databases.py`.

Artifact lives on pgl-gpu at `/home/pgl/scratch1/wyim/transgenic_data/b5/merged/b5_full_v1.db`
(23,491,260,416 bytes, md5 `ebf4fb511073c546d05e3fa0fc0e4258`). The nine sources are kept at
`.../b5/full/<species>.db` with their own md5s in the freeze manifest.

| file | what it is |
|---|---|
| `b5_full_v1.freeze.json` | the freeze record: file md5, per-table content sha256, source md5s, split-table sha256, QC flag-file md5s, per-species rn ranges, and the git commit each species was built at |
| `b5_full_v1.validate.json` | `scripts/validate_b5_database.py` output — 0 violations |
| `b5_full_v1.report.json` | `scripts/report_b5_database.py` output — the #50 accounting |

**The file md5 is an artifact fingerprint, not a reproducibility claim.** A DuckDB file's bytes depend
on the write sequence and the free-space layout, so rebuilding from the same nine sources will not
reproduce it. `geneList_content_sha256` and `table_content_sha256` will: each row is reduced to an md5
inside DuckDB and the sorted digests folded into one sha256, so the hash is defined over the *set* of
rows and is independent of the file layout and of `rn`.

The nine builds were not all made from the same commit — six at `0aa5fab`, three at `fea0d1d`. The
builder is byte-identical between the two (`git diff 0aa5fab fea0d1d -- src/transgenic/datasets/build_b5.py
src/transgenic/utils/gsf_contract.py` is empty); `fea0d1d` only touched the trainer, `datasets.py` and
`b5_runtime.py`. Both commits are recorded in the freeze manifest rather than collapsed to one.

## Amendment, 2026-09-03

The merge was run with `--qc-flags` covering only the nine GeenuFF files and without `--species-manifest`,
so the manifest as first written recorded half the masking inputs and nothing about the build invocation.
`b5_full_v1.freeze.json` now also carries the nine `*.swissprot_flags.tsv` md5s (A30), the build driver
(`run_full_build.sh`, md5 `cc615a00…`), the species manifest the driver named
(`genomes_src/manifests/b5_species_v1.gpu.tsv` — gpu-local paths, not the repo copy), and each source's
`.db.rejected.json` md5, which is the only place the per-tile rejection reasons survive with their
mask-fraction values since that file is not merged in. The `amendments` key in the JSON records what was
added, when, and what was left untouched; `file_md5`, the content hashes, the row counts, the rn ranges and
the git commits are unchanged and were re-asserted against the artifact before writing.

Recording the Swiss-Prot files is a claim that the builds consumed them, so it was checked rather than
assumed. `run_full_build.sh:6` passes both flag families, and the effect is visible in the artifact: of the
535 genes that are `swissprot_caution`-hard but not GeenuFF-hard, **none** is labelled in any tile, while
918 of the 8,737 GeenuFF-only hard genes are — the A22 case where a hard flag hits some transcripts and not
the gene, which keeps the gene labelled at `train_weight` 1. Ids were resolved through `gene_key_map` on
`gene_id` / `gene_id_original` / `name_original`, the way `flags_for_gene` does; comparing the flag files'
GFF ids against the builder's generated keys directly matches nothing and looks like total failure.
