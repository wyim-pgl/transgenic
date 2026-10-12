# A46 PROVENANCE files

A46 **rule 9** records tool versions here rather than in the protocol text:

> "Tool versions are recorded in the PROVENANCE files whose sha256 values are listed in the run log;
> no version is restated here."

Approval (**rule 15** / **rule 12 Freeze clause**) requires *tool versions by PROVENANCE sha256*.
This directory is that record.

| file | sha256 | covers | status |
|---|---|---|---|
| `PROVENANCE.cpu_tools_v1.json` | `37caa5d9c64d43e3266f344fa77e487e354dae96ddda97854e1d8fcbeaee4fe7` | pronghorn CPU toolchain: minimap2 ×2, samtools ×2, seqkit, blastn, gffcompare, diamond, miniprot, python3 ×2 | **measured 2026-09-21** |
| `PROVENANCE.gpu_host_tools_v1.json` | `92b8babea3ab5c61a3dfb805a6e156b8f7ef370f3747f0270337b1c9d5fb559c` | pgl-gpu: python3 + **DuckDB 1.5.5**; frozen B5 corpus opened read-only and **md5 re-verified** | **measured 2026-09-21** |
| `PROVENANCE.univec_screen_v1.json` | `4fd697fee9de951b7384be8ebb8e826ba258d4b424cbc95c6ca28f9841b0da3f` | which blastn screened the 11 EST sets, UniVec_Core identity, A21 params | **measured 2026-09-21** |
| `PROVENANCE.gpu_runtime_v1.json` | `c084c76b827454e5a973fdeea995896ca94cd886e7e4beafc21455f5366710ed` | DeltaAI SIF: **sha256 verified** + torch/transformers/accelerate/CUDA + compile policy | **measured 2026-09-21** |
| `PROVENANCE.protein_db_v1.json` | `b356d9444cd642a9b561d111f50d59d812a8d16f584ad9fa334ef001a723ecf1` | OrthoDB v12 filtered stage2 + Swiss-Prot reviewed Viridiplantae: **sha256 verified against the build's own sidecar**, plus the diamond makedb commands and A19.1 filter counts | **measured 2026-09-21** |

Rule: **declare a tool by its sha256, never by its version string.** This directory contains a
worked example of why — see finding 3.

## Findings from the 2026-09-21 measurement

1. **`gffcompare` v0.12.6 exists and matches A46 rule 13 exactly** — `transgenic-revision` env,
   sha256 `a440e153…`. No substitution needed.
2. **`diamond` 2.1.24 confirmed independently.** The binary at
   `evidencemodeler/bin/diamond` has sha256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`,
   which **equals** the `diamond_sha256` recorded in
   `notes/est_execution_20260916_1530/fixture_result/engine_fixture/provenance.json` by a different
   pass on a different day. Two independent records agree — this is the canonical route-B binary.
3. **Two `minimap2` builds report the same version and are different binaries.**
   `RNASEQ_bch709` → `8a561f2c…`, `transgenic-revision` → `731e16c5…`, both print `2.28-r1209`.
   Recording "minimap2 2.28-r1209" in a run log would therefore **not** identify what ran.
4. **`samtools` differs by env: 1.3.1 (RNASEQ_bch709) vs 1.21 (transgenic-revision).**
   A46 rule 12 pins **1.3.1** for the EST-molecule count behind the MVP sub-filter. The project env's
   1.21 is **not** a drop-in substitute for that step.
5. **`python3` differs: 3.11.0 (RNASEQ_bch709) vs 3.11.14 (transgenic-revision).** The 3.11.14
   recorded for job 6165965's approved runtime is the `transgenic-revision` one.
6. **`blastn` is 2.5.0+**, and the UniVec screen's own receipts record exactly `blastn: 2.5.0+`
   for all 11 EST sets — **resolved**, see `PROVENANCE.univec_screen_v1.json`. The receipts give a
   version string, not a sha256, so the identification is weaker than findings 2–4.

## DuckDB — the pronghorn failure is a *location* problem, not a broken dependency

**On pronghorn it does not import, in any env tried:**

```
ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found
  (required by /data/gpfs/home/wyim/.local/lib/python3.11/site-packages/_duckdb.cpython-311-x86_64-linux-gnu.so)
```

`transgenic-revision`, `RNASEQ_bch709` and `clean_env` all fail identically because all three pick up
the same stale **user-site** wheel in `~/.local/lib/python3.11/`.

**On pgl-gpu it works.** `/home/wyim/micromamba/envs/transgenic/bin/python3` (3.12.13, sha256
`13a56117…`) carries **DuckDB 1.5.5**, and that host runs GLIBC **2.35**.

This is the sensible arrangement anyway: **the frozen B5 corpus lives on pgl-gpu**, so the DuckDB
work of rule 5 belongs there, not on pronghorn.

### Verified while checking it

Opening `b5_full_a40_v1.db` read-only with DuckDB 1.5.5 confirms two numbers A46 records:

| | measured | A46 |
|---|---:|---:|
| file size | 23,534,252,032 B | 23,534,252,032 B |
| `geneList` rows | 273,148 | 273,148 |

Other tables: `gene_key_map` 334,642 · `gene_split` 334,642 · `window_genes` 978,864 ·
`tile_blocks` 4,454 · `rejected_records` 8,130 · `build_manifest` 9. Rule 12 reads `geneList`,
`gene_key_map`, `gene_split` and `build_manifest` — all four present.

**Corpus md5 re-verified 2026-09-21.** `md5sum` on pgl-gpu returned
`dda7e78c880c993435654b57dcd1c498` — **equal to the value A46 records** (47.8 s wall for 23.5 GB).
Together with the size and `geneList` row count, the frozen corpus is now verified on all three
recorded quantities, which satisfies rule 15's hash check for this artefact.

### Action taken — the stale wheel is gone (2026-09-21, author instruction)

Removed from `~/.local/lib/python3.11/site-packages/` on pronghorn: `duckdb/` (1,133,222 B),
`duckdb-1.5.5.dist-info/` (21,305 B), `_duckdb-stubs/` (78,737 B) and
`_duckdb.cpython-311-x86_64-linux-gnu.so` (60,276,408 B, sha256
`3bafbe5412c0918004792d0c4bd58a667d48b8b6a840425a9e0e708ff016303e`, recorded before deletion).

`adbc_driver_duckdb/` was **left in place** — it is a separate package, not part of this wheel.

Verified after: all three envs (`transgenic-revision`, `RNASEQ_bch709`, `clean_env`) now raise a
plain `ModuleNotFoundError: No module named 'duckdb'` instead of the `GLIBC_2.23` `ImportError`.
The directory went from 71 to 67 entries; `graphify`, `lxml`, `networkx`, `PIL` and `statsmodels`
were spot-checked intact.

Reinstallable at any time with `pip install --user duckdb`, but do not — pronghorn's glibc is why
it failed. Use pgl-gpu.

## Status: this axis of rule 15 is now covered

All four PROVENANCE files are written and hashed. Every tool A46 rule 9 names has a measured record,
and three recorded artefact hashes were re-verified against their originals
(B5 corpus md5, SIF sha256, diamond sha256).

## Remaining gaps

- ~~`PROVENANCE.gpu_runtime_v1.json`~~ **done.** SIF sha256 re-measured on DeltaAI and it
  **matches A44 exactly** (`1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`,
  11,120,398,336 B, 28.4 s). Inside: python 3.12.3, torch 2.14.0a0+4fdf77b940.nv26.08,
  transformers 4.57.6, accelerate 1.15.0, numpy 2.1.0, CUDA 13.4, cuDNN 92500, duckdb 1.5.5.
  `torch._dynamo.config.suppress_errors` is **false**. Login-node read only; no GPU time.
- `transgenic-ngc.sif` (NGC 25.06, seed 123's image) was **not** re-hashed.
- ~~DuckDB version~~ **resolved: 1.5.5 on pgl-gpu.**
- ~~B5 corpus md5 re-verification~~ **done: matches.**
- ~~blastn identity for the UniVec screen~~ **done: the screening's own receipts record `blastn: 2.5.0+` for all 11 sets** (`PROVENANCE.univec_screen_v1.json`). Identified by version string, not sha256.

Do not fill any of these with a guess. A46 rule 15 permits deferring only the hashes of artefacts
that do not yet exist; a tool that exists but was not measured is a gap, not a deferral.
