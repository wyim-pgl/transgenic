# Panel transport v1: preparation-only review candidate

Status: metadata and local tests prepared. BLOCKED for transport, deployment, submission, actual50 verification and receipt publication. All new files are confined to this directory. No parent helper, prior request, archive, historical source or GPU package was changed. No remote call, payload transfer, DB/SIF read, installation, job or training was performed.

## Explicit amendment to prior no-copy condition

The parent PLAN and earlier helper approvals required physical originals at their original paths. They are not silently waived. The proposed new review explicitly permits only the named byte-identical replicas in `transport_manifest.json`, exposed at the original canonical paths by validated read-only mounts. Replicas are not originals. A canonical pathname match alone does not establish physical-source continuity. Independent review must approve this amendment, replica provenance, transport inventory and mounts before any transport.

Canonical G remains `/data/gpfs/assoc/pgl/data/Transgenic`. Proposed host namespace is `/work/nvme/bilv/wcyim/transgenic/seed123_panel_transport_20260917`. These are a proposal, not a created namespace. Fresh output/logs must use a separate writable backing parent and exclusive run child, never the historical panel, request, source or DB tree. Existing namespaces and receipt or pending collisions fail, including dangling symlinks. Failures and claims survive; no cleanup/retry fallback.

## Fixed inventory and evidence

`transport.py build` reads local pinned metadata and the small current source archive. It writes only a new metadata file, exclusively, and performs no network operation or historical payload read. `validate` regenerates from independent fixed metadata pins and requires whole-document equality, not self-declared hash agreement.

- Exactly 15 historical inputs plus manifest, COMPLETE and 150 numbered artifacts: 167 historical files. Manifest is accepted SHA256 `7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e`, 38854 bytes. COMPLETE is 65 bytes with that digest and newline. All 50 loci retain exact `00` through `49` sequence/track/provenance filenames, sizes and hashes from the accepted local manifest.
- Two DB entries are existing-byte reuse only. The 165 nonDB entries total 212152146 bytes before deduplication. This is an inventory ceiling, not a completed copy. No copying DB, SIF, Hugging Face data, weights or checkpoints is permitted.
- `historical_inputs.json` and locally recorded accepted manifest agree exactly on all 15 inputs and historical authority. No recomputation, DB rebuild, array generation, historical source rewrite or receipt relabeling is proposed.
- Current consumer is the separate `/work/repo` snapshot: 103 files, archive 1873920 bytes, SHA256 `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7`. The builder reads and verifies existing `gpu/candidate/code.candidate.tar` in place, does not extract or rewrite it, and checks every member against the separately pinned source manifest. It preserves the 13-file report closure and separately pinned proof wrapper. Historical producer source bytes remain under original G paths, even when their basenames match current source files.
- The exact existing five-object Git authority receipt and pinned `notes/wait_work_20260916/panel_inputs/authority.py` are recorded. They are not inside the 103-file archive. A future reviewed deployment must package the original object bytes separately, inventory the complete resulting `.git` tree and authority receipt/TSV, invoke that existing verifier, and compare its result to the fixed five objects and authority manifest digest. This revision does not claim fresh deployed Git verification.
- Request remains 1264 bytes, SHA256 `c5cc29ecea746375ebb4d5388d38fa2bb8db0d5c42088ac67bc47b4caa7da9a9`, mounted read-only at `/data/gpfs/assoc/pgl/data/Transgenic/evidence/seed123_panel_compat_auth_resume_20260917/request.json`. The recipe passes independent expected digest and size to the original proof wrapper. Request bytes are not rewritten.
- Intended runtime: aarch64, Python 3.12.3, NumPy 2.1.0, DuckDB 1.5.5, Torch 2.14.0a0+4fdf77b940.nv26.08, Git 2.43.0. SIF pin remains SHA256 `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`, 11120398336 bytes. These pins are independent constants, not discovered acceptance values. No fresh SIF hash was performed here.

## Candidate package and validation boundary

- `transport.py`: deterministic manifest generator/validator, lexical path/collision checks, symlink rejection, exact small-tree inventory validator, top-level RO/RO/RW mount-plan validator and no-reuse check. Local small-tree validation deliberately refuses DB/SIF reads and files above 2 MB. It is not a deployed 212 MB input verifier, a concurrency lock or proof of actual mounts.
- `transport_manifest.json`: fully enumerated historical and current source metadata, fixed pins, explicit null final backing/resource/approval fields. Any alteration, including filling those nulls, fails validation in this revision. A reviewed resolved binding must be a separate future artifact, not an edited self-approval.
- `candidate_launcher.py --describe` and `launch_recipe.json`: fixed wrapper argv, runtime/environment pins, required mounts and preimport/postproof checks. `container_argv` and interpreter remain null until real backing/resource/runtime bindings are reviewed. `--execute` fails closed. It never imports a project module, starts a subprocess, mounts data or calls the verifier.
- `candidate.sbatch`: syntactically valid but hard-stopped review skeleton with one-node/60-minute/no-requeue ceilings only. Missing final account/partition/GRES/CPU/memory/exclusions/log/cwd bindings are deliberately not guessed. Do not submit even this blocked file: a guard-only job still consumes allocation.
- `test_transport.py`: local stdlib tests for exact inventory/pins; missing, duplicate and tampered records; request/runtime/current-source changes; traversal, normalization, WAL and colliding paths; file tamper, unlisted imports/bytecode and symlinks; RO/RW and source/output mount collisions; existing/dangling namespace reuse; missing P1 receipt; launcher and scheduler fail-closed behavior. Temporary fixtures stay in this directory.

The mount-plan validator tests three top-level roots only. Existing DBs need explicit nested read-only overlays into G, with a reviewed complete allowed-overlay list and runtime mountinfo/stat verification. No such binding is invented now. A source Git authority overlay also requires exact separate inventory. The recipe records these remaining checks instead of pretending an unobserved mount tree has been validated.

## P1 lifecycle is a hard gate

P1 renames staged evidence. The final evidence backing pathname must be read from the actual independently reviewed terminal P1 stage receipt, pinned by the controller and checked against the stable deployment lifecycle. This revision has no such receipt. It performs no broad find, partial-run search, auto-selection, stale-path fallback or concurrent staging action. Unknown backing paths stay null. Merely editing a path or setting an approval boolean cannot unlock execution.

After that actual receipt exists, a narrowly scoped adapter must be implemented against its actual schema and independently reviewed. It must bind exact existing corpus/evidence bytes, source/input device-inode-stat provenance and allowed mount roots, reject symlinks and adjacent WAL, ensure no rename/mutation race, and check actual in-container mountinfo and hashes before and after the proof. DB/SIF hashing and DB validation belong only in the approved compute allocation, never on login.

## Resources and acceptance gates

Conditional ceiling is at most one node-hour of CPU workload in an aarch64 GPU-backed diagnostic allocation, within the existing total five-node-hour diagnostic cap. Hide CUDA and do not use `--nv`. This is neither dedicated zero-GRES CPU placement nor a zero GPU billing claim. No training is authorized. The historical extern OOM remains an acceptance concern; batch exit alone is insufficient.

Remaining gates, in order:

1. Independent amendment review explicitly approves named replicas rather than physical originals, the exact inventory/pins, unchanged helper proof and source separation. Controller pins this candidate package externally; its self-generated checksum file is not independent approval.
2. Actual reviewed terminal P1 stage receipt establishes final stable backing paths and ownership/lifecycle lock. No panel dependency is imposed on P1 itself.
3. Independent resolved transport/mount/source provenance review: exact transfer recipe for 165 nonDB entries, request deployment, reuse of existing DB/SIF only, full 103-file source plus five-object Git packaging and complete import inventory. Reject all unlisted importable files and stale bytecode. Final immutable container interpreter and allowed package import locations must be fixed.
4. Controller reconciles account, campaign consumed hours, active jobs and pending reservations, resolves prior extern OOM handling, and approves actual account/partition/resource shape/exclusions and <=1 node-hour panel slice inside <=5 node-hours total. Unknown fields block, not default.
5. Implement and review the small execution adapter against those real receipts, including hardened environment, genuine scheduler/allocation checks, physical provenance and mountinfo checks, exact package/SIF/source/request/import-closure pins, no-clobber logs/output, runtime timeout and failure retention. Re-pin launcher/sbatch/binding. Current candidate remains non-executable by design until then.
6. Explicit transport and submission authorization after those reviews. One proof only, no retries. Controller checks terminal allocation/batch/extern accounting, all 50 ordered loci and four augmentation seeds, seed123 equality and DB-selected provenance, unchanged historical identities, current exact closure/package equality, and postchecks. Independently pin accepted receipt after rereading its bytes. No accepted receipt or numerical compatibility is claimed here.

## Local verification and work record

Completed: local pinned-metadata inventory; immutable current archive inspection; blocked launch/sbatch recipe; local tamper/path/inventory/mount-plan tests. `tdd-red.txt` and `tdd-launcher-red.txt` retain expected preimplementation failures. `test-results.txt` records eight passing tests. No scientific helper was invoked.

Commands from this directory:

```sh
python3 -B transport.py validate
python3 -B candidate_launcher.py --describe
python3 -B -m unittest -v test_transport
bash -n candidate.sbatch
```

`python3 -B transport.py build` is exclusive creation only and intentionally refuses to overwrite an existing manifest. `python3 -B candidate_launcher.py --execute` exits 2. Direct local `bash candidate.sbatch` exits 78 without scheduler interaction. Do not run `sbatch`.

Blocked: all six execution/review gates above. No independent current-scope preparation TODO remains. Graft context was attempted first (`graft map`, then scoped ask) but both timed out; exact user-specified local metadata/helper paths were read as fallback. Advisor planning was attempted but unavailable due client/model version incompatibility, so no advisor endorsement is claimed.
