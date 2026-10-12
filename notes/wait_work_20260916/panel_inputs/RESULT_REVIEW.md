# Task 2 actual job 6165559 — terminal result review

## Verdict

**CPU payload/50-locus input publication completed; reference compatibility FAILED (0/50). Not decoder-ready or scientific approval.** Allocation and batch success must remain separate from the **extern OUT_OF_MEMORY** accounting anomaly.

First live read at **2026-09-16 22:15:15 PDT** already found terminal state. No wait loop was needed. Exact job started **22:01:57**, ended **22:14:43**, elapsed **12m46s** on cpu-50, 4 CPUs/16GB/1h. Allocation and `.batch`: **COMPLETED, 0:0**; `.extern`: **OUT_OF_MEMORY, 0:125**. Requeue=0, Restarts=0. Persisted terminal sacct/scontrol evidence: `result_watch/terminal_status.txt`. Job command/workdir/logs match exactly `/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_input_audit_20260916_bindfix1`.

Both exact Slurm stdout and stderr are empty (0 bytes). No FAILED marker appears in the bounded job/panel inventory. `OBSERVATIONS_COMPLETE.json` identifies job6165559 and 763.288s payload runtime. These facts support payload completion, **not an explanation or resolution of extern OOM**. No failed-payload stderr root cause exists to report.

## Actual findings and next-controller blocker

- Exactly **50 distinct locus IDs**, matching ordered counts/manifest/reference-report rows; all 50 manifest track/sequence/provenance entries have matching regular-file sizes in the remote directory inventory.
- Selected molecule counts sum to **3443 over windows** (not a globally deduplicated molecule total), range **3–398 per locus**.
- Biological nonzero counts summed over loci, channels **0–6**: **34789, 52639, 71107, 138, 140, 193, 204**. **0/50 zero-biological-support loci**.
- Channel **7 availability** is separately **30720 per locus**, total **1536000**. Availability is not biological support.
- Publication metadata reports seeds **1/2/3**, augmentation seeds **20260915/20260916/20260917**, `seed_tracks_identical=true`, keep_fraction=1, EST/excluded_heldout, training_eligible=false, rn=null. This records the completed canonical producer's seed-equality gate; this metadata review did not independently rerasterize arrays.
- **Reference compatibility: 0/50.** Joint strict parser reports `ok=true` but **0 genes / 0 transcripts**. The fetched 16-byte restricted reference is exactly `##gff-version 3\n`. All 50 per-locus selection/geometry checks report **`ValueError: reference missing panel gene <ID>`**, e.g. Ath014705.
- Duplicate-ambiguous count **0** is **not evidence that the real reference lacks duplicates**: no panel transcripts reached duplicate-structure analysis. Parser acceptance of an empty subset is not reference compatibility.

The concrete blocker is an **empty restricted reference followed by missing-panel-gene failures**, despite successful frozen-reference byte identity and raster production. This does **not** establish that raw GFF lacks these biological loci, nor distinguish an identifier mismatch from restriction logic without a separately reviewed investigation. Preserve this job/namespace; controller may initiate a targeted identifier/restrictor investigation and review. **No fix, changed reference, deduplication, retry or new job was performed here.** Canonical wrapper reference audit stage: `final_release_bindfix1/run_panel_cpu.py:76–118`; observation completion deliberately permits compatibility findings at `:120–160`.

## Identity and publication validation

`result_watch/validate.py` executed successfully; detailed aggregates in `result_watch/validation.json`.

- Fetched transport bytes equal the independently reviewed local bindfix1 transport, SHA256 **1883df3c32c5b560030dbcb89543b3d2730d7c08518d47e4f932b62f88b18c0a**. Original submission receipt records all seven payloads OK. No remote tar hashing repeated on login.
- Actual source identity is **b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6**, component **555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44**. All **100 reported source-file hashes plus 3 companions** independently match bytes inside the reviewed local source tar; this is comparison to compute-produced identity, not another remote source read.
- Actual Git authority JSON exactly equals retained `authority.final_release.json`, including all **five original object IDs/types/sizes**. Commit **92d8020ee79eeedd7f1d50167b1c8dc94700fb16**, authoritative manifest SHA256 **3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783**.
- Reference identity agrees exactly across identity/report/panel authority; request paths match. Actual compute FASTA SHA256 **63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973** (121183059B); GFF **529510b60756c9a1ae5a590ec4097fd4b9773649c29e6a5543553118bb042c38** (60503119B). Neither raw file was fetched or hashed on login.
- Accepted panel/evidence IDs, rasterizer identity, request identity, COMPLETE→manifest, completion→counts, counts→manifest/reference report, and restricted-GFF hash all cross-check. Counts equal manifest summaries for all 50 loci; channel lengths/bounds/zero flags and 30720nt windows agree.

## Runtime boundary

Actual CPU record: Python **3.10.20**, NumPy **1.26.4**, DuckDB **1.5.5**, Torch **1.13.1+cpu**, empty CUDA_VISIBLE_DEVICES. Manifest NumPy/DuckDB versions exactly agree with that record. CPU container `/data/gpfs/assoc/pgl/singularity/micromamba.sif` compute SHA256 **9e2be34cec8cdfe642c58f4108833fd3e97635ce29becd5de8e48b3ae28332e1**; interpreter/module hashes retained in `cpu_runtime.json`.

**Eventual decoder SIF's actual NumPy/DuckDB record is not available in this review. Decoder compatibility remains UNKNOWN**, not PASS. Canonical verifier requires exact package equality (`mvp_verify_panel_inputs.py:13–84` in b660 export); CPU import/raster success does not certify that SIF, GPU readiness, training, predictions or scoring.

## Retained metadata and limits

All retrieved output bytes live under `result_watch/metadata/`; **14 files / 95867 bytes** total, far below the 10MB bound. `result_watch/collection.json` records the initial 13-file transfer and regular-file size inventory; the separately fetched 16B restricted GFF is included in the final exhaustive **`result_watch/output_metadata_hashes.json`**. Hashes are computed **locally over fetched bytes**. It includes empty Slurm logs, job observation, runtime, Git/source/reference identity, request, compatibility, counts, manifest, COMPLETE and transport.

Key local metadata SHA256:

| Artifact | SHA256 |
|---|---|
| panel/manifest.json | `3c31b90522dc8a1393480218f878b6c4e44e4339e211e69173f25005286c63ca` |
| panel_counts.json | `a6a8ce7089e24ac2e9ffbd213545870c406be3283dc311f768182844796f84ba` |
| reference_compatibility.json | `ebd6a22a7eb9b7d6c33a29ace3bfe05963635c96e8531b02a075de8a4c1f03e2` |
| reference.panel.gff3 | `897007ceaf55680e4fa9d1d9e6ebac641ede633c0a9f2426db36ecbc66177df9` |

No DBs, raw references, arrays, FASTAs, large provenance outputs or environments copied; payload file digests are producer-recorded, not independently rehashed in this review. No remote writes, source edits, submissions, cancellation/requeue/extensions, subagents, merge or push. Local collection/validation scripts are one-shot evidence utilities, not a recurring service. Advisor consultation failed due client/model version incompatibility and supplied no review approval. Graft-reported context savings: approximately **1025580 tokens** (estimates).
