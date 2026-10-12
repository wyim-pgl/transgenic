# MVP operational CPU smoke — 2026-09-15

**Restart update:** final retry **6164093 PASS**, allocation/batch0:0, 4m38s. Builder195.87s, peak149104KiB, DB16527360B, retained protected overlaps0. Train RC/N/off raster passed; sample valid evidence0, so valid nonempty-evidence path was not certified. Output `evidence/mvp_smoke_20260915_retry02/SUCCESS.json`. NOT_FOR_TRAINING remains binding. `.extern OOM0:125` separately unresolved. Full build6164118 is separate and running. Below FAILED sections preserve the earlier attempts, not the final smoke outcome.

**FAILED before input verification or builder execution. SAMPLE / NOT_FOR_TRAINING.**

## Scope and staging

One authorized CPU smoke submitted: **6164069**. No retry, requeue, full build, GPU, model/trainer changes, or changes to blastx array 6164049. Main code was not edited.

Read approved implementation report/brief using `git show`. Archived only approved builder, rasterizer, full panel/sidecar, approved spec and implementation report/brief from commit `abca40fb4941207fa9ad5bbeba211d11f3942748`.

Remote root `R=/data/gpfs/assoc/pgl/data/Transgenic`:
- Snapshot: `R/worktrees/b6-mvp-cpu-abca40f`; commit files in `approved/`, diagnostic/wrapper in `ops/`.
- Archive SHA256: `c91908566cdfa40d3ea9dba335af88b9f53097a7c7e41ba49f4753ac3530487c`.
- `manifest.json` records source commit, archive hash and matching local/remote per-file SHA256 values. Approved files set 0444 and directories 0555 (read-only permissions, not a filesystem immutability guarantee).
- Builder SHA256: `4c66a890a1e0cd304d3c6ce095bd4c8f9cac8d0d0fa800d8ecb5fbfaaf233738`.
- Rasterizer SHA256: `80a27c1909f41041c50af8d719856bbea914b80b804c6b6a4e07b715718922c6`.
- Spec SHA256: `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`.
- Full 2,079-row panel SHA256: `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`; archive bytes match the main uncommitted panel.
- Diagnostic SHA256: `e7e04af4e5bc2d491ef39199391303b7e360a08eae794a4be098814f2b88829c`.
- Output: `R/evidence/mvp_smoke_20260915`; persistent `SUBMISSION_LOCK/` and `jobid`. Preserve all files. Operational script hashes are in `operational_scripts.sha256`.

A login-node staging command initially could not find `python3`; manifest verification used the supplied container interpreter instead. This was not a job submission or production-data scan.

## Actual job status and failure

Requested account/partition `cpu-s1-pgl-0`, 4 CPUs, 16 GB, 01:00:00, `--no-requeue`.

| Slurm component | State | ExitCode | Elapsed | MaxRSS |
|---|---|---|---|---|
| Allocation 6164069 | FAILED | 1:0 | 00:00:02 | not reported |
| 6164069.batch | FAILED | 1:0 | 00:00:02 | 1496K |
| 6164069.extern | OUT_OF_MEMORY | 0:125 | 00:00:02 | 1480K |

Application failure is the import error below, **not an inferred OOM**. Extern OOM 0:125 is the known separate accounting condition and does not establish application OOM.

`slurm-6164069.err`:
```text
File ".../ops/smoke.py", line 4, in <module>
    import duckdb, numpy as np, pysam
ModuleNotFoundError: No module named 'duckdb'
```

Root cause: the operational wrapper added `PYTHONNOUSERSITE=1`, unintentionally disabling the user-site installation that supplied validated DuckDB 1.5.5. Read-only directory inspection confirms `duckdb`, `_duckdb.cpython-311-x86_64-linux-gnu.so`, and `duckdb-1.5.5.dist-info` under `/data/gpfs/home/wyim/.local/lib/python3.11/site-packages`. Explicit PYTHONPATH included only approved `src` and `R/worktrees/mvp-cpu-deps-20260914/site` (pysam), not this DuckDB location. This is a diagnostic environment regression, not an observed builder failure. No fix applied and no replacement job submitted.

`payload.exitcode`: 1. `/usr/bin/time -v` for the container/diagnostic payload: wall **1.46 s**, user 0.12 s, system 0.11 s, maximum RSS **32,912 KB**. These are startup-failure measurements, not builder performance.

## What did not run

The diagnostic had been staged to verify corpus/BAM/reference hashes on compute, create the first 10,000-record BAM retaining the complete original header, run the unchanged approved CLI with `/usr/bin/time -v`, then query retained train/valid and RC memberships and rasterize original read-only corpus sequences with provenance, shape, N-mask, all-N and fraction-zero checks. None reached execution past imports.

Therefore this job produced **no sampled BAM, C0-lite DB, builder timing, builder QC, raster results, tile provenance, or nonzero-count measurements**. Input identity values supplied from prior jobs were not revalidated by this job:
- Corpus: SHA256 `6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222`, 23,534,252,032 bytes, no WAL (prior verification).
- BAM: SHA256 `f4598638a377d2ed6a89a9a01242f9872207ab6e55c98da5fe3e104764f0760e`.
- Reference: SHA256 `63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973`.

Even a successful first-10,000-record smoke would remain **SAMPLE / NOT_FOR_TRAINING**: sampling can omit other placements of molecule members and cannot certify full-data leakage exclusion. A corrected environment and another job require separate authorization; the one-job allowance is consumed.

## Controller-authorized replacement (retry01)

Controller subsequently authorized exactly one replacement job. Before staging, reviewed the actual preserved `ops/smoke.py` and `ops/smoke.sbatch`: the wrapper alone sets `PYTHONNOUSERSITE=1`; the diagnostic imports DuckDB directly and retains all source/input hashes, panel size, sample/header, QC and raster assertions. Correction: explicitly `unset PYTHONNOUSERSITE`, retain `PYTHONDONTWRITEBYTECODE=1`, and change only operational/output paths to `ops_retry01` and `evidence/mvp_smoke_20260915_retry01`. This restores access to the previously validated user-site DuckDB without installing or changing packages. Prior job artifacts and approved builder/rasterizer stay untouched. No further retry is authorized if this replacement fails.

Replacement staged at `R/worktrees/b6-mvp-cpu-abca40f/ops_retry01`; old ops/output preserved. Python `compile()` and `bash -n` passed. Reversal assertions prove the script diff contains only the approved environment/output/script-path changes; approved-file hashes were rechecked unchanged. An initial staging command had a quoting SyntaxError before any write/submission; corrected staging completed without modifying the old files.

Replacement script SHA256:
- `smoke.py`: `9cd51eb1e50012468145bdd4438789c33de4627f90b0de513bb32284bf40af50`.
- `smoke.sbatch`: `a6145ace6e79a57df0779a2169f040391b2ec6dd54f83afd7f8c5f65e39300e5`.

Submitted exactly one replacement: **6164081**, same account/partition, 4 CPUs, 16 GB, one hour, no-requeue; new persistent lock/jobid in `R/evidence/mvp_smoke_20260915_retry01`.

Observed at elapsed **00:00:12**: allocation, batch and extern all **RUNNING** on `cpu-50` (displayed exit 0:0 is not a final result). Startup now successfully imports **DuckDB 1.5.5, NumPy 2.3.1, pysam 0.23.3**; rasterizer resolves to the approved snapshot. Stderr empty. Input hashing is in progress; no sample/build/QC/raster completion claimed. All SAMPLE / NOT_FOR_TRAINING and incomplete-placement leakage limitations remain.

## Controller-authorized replacement (retry02)

Retry01 **6164081 FAILED 1:0 after 00:01:26** (batch MaxRSS 110004K). Its preserved traceback confirms `FileNotFoundError: /usr/bin/time` in the diagnostic's first subprocess: host time exists, but container time does not. Input hashes and the 10,000-record sample completed according to the controller's handoff; builder did not execute. Extern OUT_OF_MEMORY 0:125 remains separate accounting, not evidence of application OOM.

Controller authorized exactly one further CPU replacement. Read the actual remote retry01 scripts and manifest before changing anything; reviewed root cause and command with advisor. New read-only `R/worktrees/b6-mvp-cpu-abca40f/ops_retry02` preserves every input/QC/raster assertion. Only retry paths and inner timing changed: builder argv starts directly with `sys.executable`; stdlib monotonic wall and before/after `resource.getrusage(RUSAGE_CHILDREN)` record CPU deltas and Linux peak RSS in KiB (not a subtracted peak). Scope explicitly identifies first subprocess builder and reaped children. Host `/usr/bin/time` and user-site DuckDB access remain unchanged.

Python compile, bash syntax, AST assertion preservation, exact reverse-diff checks, both approved manifest hash maps, unchanged retry01 hashes, and transferred retry02 hashes passed. No approved/core files changed. New operational SHA256:
- `smoke.py`: `9a783c83f9578be5943f670e102ccd9d780dfc18d00d1c9f976829b04c931ef0`.
- `smoke.sbatch`: `acd74595678cb5e977236c7153873b8303d39312327c0cacd3ee365d7abfa759`.

Submitted once under persistent exclusive `SUBMISSION_LOCK`, raw submission record and `jobid` at `R/evidence/mvp_smoke_20260915_retry02`: **6164093**. Same cpu-s1-pgl-0 account/partition, 4 CPUs, 16 GB, one hour, no-requeue. Observed **RUNNING at 00:00:13 on cpu-50**; imports DuckDB 1.5.5, NumPy 2.3.1 and pysam 0.23.3 succeeded; approved rasterizer resolved; stderr empty. No retry02 input-hash/build/QC/raster completion yet claimed. Stop and diagnose any subsequent failure; no additional retry authorized. **SAMPLE / NOT_FOR_TRAINING**: first 10,000 records may omit member placements and cannot certify full-data leakage exclusion.

