# Athaliana full-source MVP C0-lite CPU build — 2026-09-15

**Restart update, 15:13 PDT:** 6164118 remains RUNNING on cpu-51 (47m). Inputs passed hash checks; builder running, no final QC/SUCCESS yet. Current evidence: [restart snapshot](restart_status_20260915.txt). Below submission observations are historical; do not treat their 14-second elapsed value as current.

## One authorized submission

Submitted **6164118** once at cluster time `2026-09-15T14:25:31`. Bounded check: **RUNNING**, elapsed **00:00:14**, node **cpu-51**; allocation, batch, and extern all RUNNING. Displayed 0:0 is not a completed exit. Startup confirms DuckDB 1.5.5, NumPy 2.3.1, pysam 0.23.3 and approved rasterizer path; stderr empty. Input hashing is underway; no full build/QC completion claimed. **full_build_success: not established**.

- Account/partition: `cpu-s1-pgl-0`; 2 CPUs, 64 GiB, 12 hours; Requeue=0, Restarts=0. Hard allocation cap applies; the 10,000-record smoke does not establish full-size memory/runtime feasibility.
- `R=/data/gpfs/assoc/pgl/data/Transgenic`.
- New output/diagnostics: `R/evidence/mvp_full_20260915`; intended DB `fullc0lite.db`.
- Persistent exclusive `SUBMISSION_LOCK/`, `submission.raw`, and `jobid` protect this one submission. No automatic retry. Stop and report on failure; do not resubmit.
- Operational scripts are preserved in `ops/` with read-only permissions, `operational_scripts.sha256`, and exact submission command `bash ops/submit_once.sh` from the output directory. The script executes `sbatch --parsable ops/full_build.sbatch` once.
- Old sample artifacts preserved; no sample DB was copied or relabeled as full.

## Approved source and actual command

Read `notes/b6_mvp_operational_smoke_20260915.md` and report/plan through `git show abca40f:notes/b6_mvp_data_implementation_report_20260914.md` and corresponding plan. Source snapshot stays `R/worktrees/b6-mvp-cpu-abca40f/approved` at commit `abca40fb4941207fa9ad5bbeba211d11f3942748`. No source changes.

Builder and rasterizer SHA256 were rechecked before submission:
- Builder: `4c66a890a1e0cd304d3c6ce095bd4c8f9cac8d0d0fa800d8ecb5fbfaaf233738`.
- Rasterizer: `80a27c1909f41041c50af8d719856bbea914b80b804c6b6a4e07b715718922c6`.

The diagnostic invokes the unchanged actual CLI using container `sys.executable`:

```text
/data/gpfs/home/wyim/micromamba/bin/python R/worktrees/b6-mvp-cpu-abca40f/approved/revision/scripts/mvp/mvp_build_c0lite.py --bam R/evidence/est_align/Athaliana/Athaliana.est.bam --panels R/worktrees/b6-mvp-cpu-abca40f/approved/revision/protocols/b6_panels_v1.tsv --db R/training_input/db/b5_full_a40_v1.db --out R/evidence/mvp_full_20260915/fullc0lite.db
```

`R` above abbreviates the full root, not a literal argv value. Runtime `builder_command.json` records expanded argv. Builder internally uses threads=1; allocation remains 2 CPUs (corpus loading can use other defaults). Host `/usr/bin/time -v` wraps `/apps/singularity/3.6.1/bin/singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python .../ops/full_build.py`. No inner `/usr/bin/time` dependency. `PYTHONNOUSERSITE` explicitly unset; `PYTHONPATH` contains approved/src and `R/worktrees/mvp-cpu-deps-20260914/site`. No package changes.

## Staged gates and diagnostics (not yet completed)

Preflight verifies both matching manifest hash maps, all archived approved files, explicit builder/raster/panel hashes, all 2,079 panel rows, exact dependency versions, corpus size 23,534,252,032 bytes and no WAL. Full input SHA256 and corpus reference manifest checks run on compute:
- Full BAM: `f4598638a377d2ed6a89a9a01242f9872207ab6e55c98da5fe3e104764f0760e` (corrected documentation typo; diagnostic already used these correct bytes).
- Corpus: `6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222`.
- Reference: `63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973`.
- Panel: `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`.

Correct full BAM SHA256, as implemented in the diagnostic: **`f4598638a377d2ed6a89a9a01242f9872207ab6e55c98da5fe3e104764f0760e`**.

Post-builder checks preserve source hashes even on a nonzero builder exit if the diagnostic survives. SIGKILL/resource termination can prevent postchecks; their absence must not be interpreted as passing. Full success requires QC with both retained protected-overlap counts zero, `valid_tiles_with_evidence > 0`, 2,079 protected panels, more than 10,000 full-source records, and matching DB input provenance. Four evidence-bearing train/valid × forward/RC tiles are selected; strict raster checks cover shape/dtype/finiteness, N/all-N, fraction-zero, dropped-source, and exact RC permutation `[1,0,2,5,6,3,4,7]`. Missing evidence-bearing samples fail rather than weaken checks. Output/corpus no-WAL and source hashes are rechecked before exclusive `SUCCESS.json`, which alone sets `full_build_success: true`. Output SHA256, DB bytes, before/after filesystem disk availability, Slurm allocation/environment, builder child CPU/peak RSS, host payload time, and payload exit code are captured. Builder wall timing is explicitly labeled as including the small post-source-hash checks.

Python compile and both shell syntax checks passed before upload. Local/remote operational hashes matched:
- `full_build.py`: `dae774ccbf3fc6ecbdf58c8e16b88d69333433044913880f6273ee568ff205c1`.
- `full_build.sbatch`: `38bd6895b0853e84240ca391f30d270dbede86ca48a6717564ae14f0ffa1de25`.
- `submit_once.sh`: `1cbd9638f86d70a0dacf97f7c563cf094cce4f5f11394f3d73f30084c96191ae`.

## Scope and follow-up

This is **full-source candidate MVP**, not accepted production training data even if diagnostic success occurs. Prior 10,000-record smoke 6164093 succeeded, but its zero valid evidence was a geographic sampling limitation, not a complete-data validation result. No GPU/model/trainer/DS connection or blastx-derived phase channels were added. The full-build data project and model/DS-connection projects are separate. Blastx array **6164049** was untouched.

Later inspect allocation, batch, extern, `payload.exitcode`, builder result, `qc.json`, raster results and `SUCCESS.json` separately. In particular, any extern `OUT_OF_MEMORY 0:125` is a separate accounting condition and does not by itself prove application/resource OOM. No 12-hour wait was performed.
