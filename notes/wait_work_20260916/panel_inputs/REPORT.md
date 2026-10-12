# Task 2 — reviewable CPU panel-input execution package

**Prepared, not remotely executed.** Controller review and submission remain required. No GPU/checkpoint/P1 dependency. No SSH writes, submissions, subagents, main source changes, merge or push. Advisor was unavailable (client/model version incompatibility), not an approval.

## Read-only observations

`input_inventory.json` records pronghorn login-0 at 2026-09-17T04:19:59Z. Paths are taken from immutable `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv` and unchanged accepted candidate JSON, not caller-selected identities:

| Actual path under `/data/gpfs/assoc/pgl/data/Transgenic/` | Stat bytes |
|---|---:|
| `genomes/Athaliana_167_TAIR10.fa` |121183059|
| `genomes/Athaliana_167_TAIR10.gene.clean.gff3` |60503119|
| `training_input/db/b5_full_a40_v1.db` |23534252032|
| `evidence/mvp_full_20260915/fullc0lite.db` |1068773376|

Only stat, bounded directory discovery, small existing receipts and distribution METADATA were read. No login-node reference/DB hashing, DB queries or Torch imports. Existing SUCCESS/input-identity/raster receipts are historical observations, not fresh identities or actual 50-panel support. The earlier shell probe found no `python3` on PATH (exit127); explicit existing interpreters were subsequently used.

`runtime_signalp_cpu.txt`: existing micromamba SIF + `/data/gpfs/home/wyim/micromamba/envs/signalp6/bin/python` executes **metadata-only** Python3.10.20 and reports **torch1.13.1+cpu / numpy1.26.4**. No Torch/GPU import was performed remotely. Native transgenic-revision Python3.11 lacks Torch; miniforge3 SIF Python3.13 reports none of the required distributions. Existing DuckDB1.5.5 is CPython3.11 and must NOT be prepended to Python3.10.

Minimal missing piece: downloaded the **DuckDB1.5.5 CPython3.10 wheel only**, extracted locally, not installed into any existing environment. `cpu_dependency.json` binds every extracted file and confirms the wheel SHA256 against PyPI release JSON: `e238060db5ca59879882a6e9b015e2c65d5c64ddf281ba1d7a9a2033764152cf`. This is not an environment copy or a runtime certification. Real imports occur first on compute; failure stops before expensive input work. CPU Torch1.13 compatibility with this narrow verifier import path has not yet been executed. All numerical rasterization remains the canonical NumPy producer.

`cpu_partitions.txt` observes CPU partitions including default `cpu-s3-sponsored-0`, max8h. Controller must choose its approved account/CPU partition and recheck live queue before submission; no account entitlement is inferred here.

## Authoritative export and source

Use **`final_release/`**, not the earlier local `release/` or `reviewed_release/` drafts. These generated directories are deliberately git-ignored and retained locally. Local inspection source + Graft is `source_b660/`; it is not the transport.

- `final_release/source.tar`: exact100 reviewed source files from b660 Git objects, plus unchanged accepted3 JSON, fixed release inventory and sparse reference authority. No environment or all-branches/history copy.
- `final_release/authority.py`, `run_panel_cpu.py`, `panel_cpu.sbatch`, `cpu_dependency.json`.
- `final_release/cpu_site.tar`: only the downloaded ABI-matched DuckDB distribution; extract alongside `source/`.
- `final_release/transport.json`: hashes of all payloads. Review and independently hash this locally before transferring it; a self-consistent remote copy alone is not authorization.

Every exported source byte is compared directly with `git show b660bf1:<path>` and fixed reviewed release inventory `7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689`. Component pin remains `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`; decoder pin remains the reviewed `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb` (no decoder edit). Accepted companions remain `faab8387…`, `9f790196…`, `73204a4e…`.

`authority.py` transfers **five original objects**, calling Git's object writer on the original `cat-file` bytes and requiring identical OIDs: commit92d8020, root treeffff3bcb, data tree4b71774b, manifests tree25dd0a45, blobd41b313a. It neither constructs replacement trees/commits nor accepts a caller MD5. Object type/body hashes are rechecked, and the exact command the producer uses succeeds:

```
git -C ROOT show 92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv
```

Manifest SHA256 is `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. The object database is deliberately incomplete: unrelated trees, blobs and parent history are absent, no refs or alternates. General `git fsck`/history traversal is NOT supported; this is only authority for the pinned path. P1 does not need it; later panel/reference evaluation does.

## Shared reference interface — communicate to Task 3

1. **`final_release/reference_authority.tar`** contains `.git/`, `reference_authority.json`, `b5_species_v1.authority.tsv`; no source or environment. SHA256 `7002445d7007c6495d5fc00406faa2c7b373ef10b7c5ed93674bc9b04aae6ec3`. Extract ONLY into a new exact-code root without `.git`, never over another checkout. Run `authority.py --root ROOT` there. Preserve original OIDs; do not substitute the TSV working file for Git authority.
2. After actual CPU execution: **`outputs/job-ID/git_authority.json`**, **`reference_identity.json`**, **`reference_compatibility.json`**, **`reference.panel.gff3`** and **`request.json`**. `reference_identity.json` is the unchanged `mvp_produce_counts.frozen_reference` result from actual FASTA/GFF bytes. Task3 should invoke that same gate at its root and compare its fresh identities, not trust this JSON alone.
3. Panel windows are the canonical `mvp_windows.load_panel/evaluation_window`, 50 loci, 30720nt genomic-forward. Restricted GFF is an audit derivative, not a new frozen reference or a deduplication policy.

No Task3-folder edits were made.

## What the bounded job does

`panel_cpu.sbatch`: **1 task,4 CPUs,16GB,1h**, no GPUs/requeue/install/retry; outer timeout3500s +20s kill grace. Read-only `/data/gpfs` bind, with only the dedicated `PANEL_RELEASE/outputs` directory remounted writable. `CUDA_VISIBLE_DEVICES` empty; no `--nv`, model, checkpoint or GPU operations. Output is exclusive `outputs/job-$SLURM_JOB_ID`.

1. Verify fixed source inventory/component pin, no extra project Python files, accepted3 companions and genuine Git authority.
2. Verify the isolated CPU extension files, Python/package versions, then perform real CPU-only imports. Record Python/module/SIF hashes on compute. This does not certify the entire environment closure or decoder runtime.
3. Derive reference paths from the original Git object and DB paths from unchanged acceptance; persist `request.json`.
4. Rehash actual FASTA/GFF via canonical frozen-reference gate; derive contig lengths without an index. Use the **same accepted restrictor and strict `read_gff`/`select_transcripts`** as the count producer on the actual frozen-GFF panel subset. Report missing/duplicate IDs, parser/strand/geometry/window failures for all50; report canonical duplicate-structure groups without deduplication, changed multiplicity or F1 computation. If joint subset parsing fails, individual restrictions retain per-locus diagnostics. Scope is the actual panel subset, not certification of every non-panel gene in the genome.
5. Independently of reference parser compatibility observations, call unchanged **`mvp_prepare_panel_inputs.prepare`**. Its actual reference identity still must pass. It performs full candidate hash/identity gates and read-only DB access, rasterizes exactly50 `excluded_heldout` loci with `rn=None`, compares seeds1/2/3, retains selected provenance, and publishes canonical manifest/COMPLETE last.
6. Write **`panel_counts.json`**: selected-molecule counts, biological channels0–6 separately from channel7 availability, zero-support flags, source/request/reference identities. Zero support is observed, not fabricated or rejected. No retained/training fallback. `OBSERVATIONS_COMPLETE.json` is raw CPU observation completion, NOT a prediction/scientific verdict. Duplicate ambiguities remain reported even when observation collection finishes. On exceptions, preserve FAILED/partial directories; no in-place retry. Timeouts/native crashes may prevent FAILED publication; absence of observation completion is not success.

Candidate corpus hashing happens multiple times in the unchanged producer. A full attempt can use the whole hour or time out; no completion-time guarantee or resource increase is embedded.

## Controller execution sequence (not executed)

- Independently inspect scripts/objects/transport hashes and rerun the small tests. Review CPU1.13 import compatibility risk and dependency receipt.
- Choose a **new canonical retained release directory** on pronghorn; transfer only final payload files after hash review. Extract `source.tar` and `cpu_site.tar`; create a dedicated empty `outputs/`. Never overlay an existing release or modify original databases/reference/environment.
- Verify payload hashes, source/authority checks and no existing same-task job. Export absolute `PANEL_RELEASE` and `PANEL_CPU_SITE=$PANEL_RELEASE/cpu_site`; supply approved CPU account/partition and absolute Slurm stdout/stderr paths. Submit `panel_cpu.sbatch` only after review; no P1/checkpoint prerequisite.
- Inspect actual allocation/batch/extern accounting and all outputs. A metadata-only runtime probe or these local tests is not the actual50 validation. Missing CPU runtime support means stop and preserve diagnostics, not bypass canonical imports/gates.

**Future decoder caveat:** canonical `mvp_verify_panel_inputs` requires exact producer NumPy/DuckDB versions as well as paths and bytes. These CPU observations use NumPy1.26.4/DuckDB1.5.5. If eventual decoder SIF versions differ, regenerate with its required versions or obtain separately reviewed changes; do NOT weaken package checks. The staged candidate does not assert future decoder consumability or GPU readiness.

## Local verification

`tests.final_release.txt`: **9 tests PASS,24.306s**. Includes original object identity and byte equality, authority-only/full-source tar roundtrips, missing object, corrupt compressed object, valid wrong-content object under the expected OID, parent-Git fallback rejection, tampered authority/source/inventory rejection, login invocation refusal before outputs/data, package-script transport hashes, and duplicate reporting without modifying transcript inventories. No synthetic GFF is represented as actual data. Shell syntax and Python compile passed. Initial red missing-module test and one test-fixture permission failure are retained; mutation test corrected permissions before the successful rerun.

Graft-located canonical spans inspected: `mvp_prepare_panel_inputs.py:30–170`, `mvp_produce_counts.py:54–70,207–346`, `mvp_cli_runtime.py:89–143`, `mvp_windows.py:139–160`, `mvp_verify_panel_inputs.py:13–84`, all in the exact b660 export. Main's older source was not used as the implementation authority. Graft-reported savings across this task: approximately1,077,379 tokens (estimates only).
