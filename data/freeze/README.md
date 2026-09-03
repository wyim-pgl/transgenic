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
