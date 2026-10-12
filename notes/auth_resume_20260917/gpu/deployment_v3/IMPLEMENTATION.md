# Explicit failed-attempt recovery candidate

## Status

Local candidate complete. **Final review target: release_v5.tar, not release_v4. Execution remains blocked.** No approved authorization, cancellation, submission, remote copy, installation, or commit was performed. Independent review of these final bytes is required next. This is an explicit recovery proposal, never an automatic retry.

Only deployment_v3 implementation and evidence were intentionally written. Preservation verification matched all121 files under deployment_v1, deployment_v2 and candidate before/after tests. The unchanged original final_p111-test suite also passed. Source candidate archive, initializer policy, scientific/runtime gates, panel identity and all original operator pins remain unchanged.

release_v4 was frozen before the controller supplied two additional query-boundary findings from `../admission_review/ADMISSION_REVIEW.md`. It remains unmodified and superseded. Its own ledger still verifies. The fixes are in a fresh release_v5 with its own namespace, authorization pathname and transport. Never patch either frozen release.

## Four startup corrections

### Weights-only epoch05 staging

`checkpoint_inventory()` requires exactly model.safetensors, meta.json and eval.json as real regular files. Optional accelerate_state and pending_grads must be real nonsymlink directories. Unknown entries, wrong types, links and aliased epoch paths reject.

The producer at `train/train_HyenaTransgenic.py:432-449` writes accelerator state and pending gradients as legitimate resumable state. These directories are named/excluded only: no contents are read, copied or deleted. Existing candidate stage_checkpoint.py copies the selected weights, run config, runtime identity and allowed sidecars, retaining its before/after hashes. The wrapper checks source file metadata and inventories before/after staging, repins meta/eval, requires a three-file staged checkpoint, and records the excluded directory names in checkpoint-weights-only.json. Initialization remains weights-only with new optimizer state.

### Exact terminal history and read-only evidence recovery

The authorization schema now requires a terminal_recovery_reference, fixed terminal_history and fixed evidence_recovery binding. A cancellation-history reference is no longer accepted. Template approved:false, issuance:null and absent provenance block execution.

Live accounting must show3163986 as FAILED1:0, elapsed51seconds, one node/four GPUs, allocatedCPU288/requestedCPU64, gh077 and the exact recorded start/end times. Job3163545 must remain CANCELLED0:0 with zero elapsed and no allocation. Other unknown history still rejects. This exception does not permit recovery from arbitrary failed-job artifacts.

The full51seconds contribute51/3600node-hours and204/3600GPU-hours. Including this new maximum0.5node-hour/2GPU-hour attempt and the retained unsubmitted panel1node-hour/4GPU-hour reservation gives1.5141666667node-hours/6.0566666667GPU-hours before other historical usage. Both unchanged5/20caps are enforced separately. Training and followup remain unauthorized; no restart, requeue, resume or retry is introduced.

The only recovery source is:

`/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/final-p1-job-3163986/inputs/evidence/fullc0lite.db`

Expected1068773376bytes, owner97434 and SHA256:

`ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d`

No old staging path, fallback, directory search, rename or database rebuild exists. After live admission and fresh quota checks, compute must check canonical regular identity, owner, no write bits and absence of adjacent WAL/SHM/journal. It hashes the source, creates the destination exclusively with0600 inside a0700evidence directory, copies, checks destination hash and size, rehashes source and compares inode/device/owner/mode/size/mtime/ctime before/after. Successful destination becomes0444. The source is never chmodded, moved or deleted. A collision or partial failure is retained and never overwritten. The recovery receipt records both identities. No evidence copy was performed during implementation.

The existing5GiB allowance remains unchanged:1068773376evidence +2200000000checkpoint ceiling +2000000000HF-cache ceiling =5268773376bytes, below5368709120bytes by99935744bytes. HF copying now enforces that aggregate cache ceiling. The40GiB reserve and fresh quota-before/quota-checkpoint checks remain. Actual checkpoint bytes are approximately2.091GB; the calculation uses the larger reviewed ceiling.

### Requested versus exclusive allocated CPUs

Current job ReqTRES must requestCPU64/node1/GPU4 and billing4000; CPUsPerTask64 and NumTasks1 are mandatory. AllocatedCPU must equal NumCPUs and be64, or equal the current node's CPUTot from a fresh bounded trusted `/usr/bin/scontrol -o show node <validated-node>` query. ArbitraryCPU overrides are not accepted and288is never declared user-requested. Capacity is rechecked before admission finalization. Requested/task resources join the existing stable-resource comparison.

One exclusive ghx4node, four GPUs,1800seconds maximum, no restart/requeue, account/cluster identity, no overlap, excluded nodes, full B5 maximum reserve, finance-unit checks and300second observation freshness remain unchanged.

### Actual Python3.6host compatibility

The initial text=True incompatibility was removed. Final query capture is binary selectors/os.read, avoiding text=True entirely; quota's existing universal_newlines remains supported. Both full modules compile and execute their import-only bodies under actual dtai `/usr/bin/python3`3.6.15 through pronghorn. Pure synthetic authority/capacity validators, harmless /bin/true and real query capture with an in-memory receipt sink passed. No remote file, model, database, job, SIF or installation was used.

`review/host-compat-streaming-result.json` binds the final admission/deploy source hashes and remote STDIN hash. SSH emits environment-module banners on stderr, preserved verbatim; this is a compatibility smoke check, not live admission or runtime approval. SIFPython3.12 is not claimed as the host interpreter.

## Additional accepted review fixes

The controller accepted both P2 findings in `../admission_review/ADMISSION_REVIEW.md` for this candidate. The query runner now uses selectors with binary os.read and a16MiB aggregate stdout+stderr raw-byte ceiling during capture. Reads are at most64KiB and at most remaining room plus one byte. Timeout is monotonic, per command, without retry.

Any capture exception, including non-timeout OSError, overflow or decoding failure, retains the original error, kills the detached process group and attempts a bounded two-second reap. Cleanup errors are recorded separately. Pipes close without a subsequent unbounded communicate/drain. Only bounded prefixes are persisted on failure; an error-bearing or truncated result can never be admitted. JSON escaping can enlarge persisted representation by a bounded factor; the capture buffer itself remains at the aggregate raw-byte ceiling.

No account/queue coverage or runtime gate was weakened. The prior independent review established the finance conversion mechanism, not final-package approval.

## Verification

Tests were written first. `review/tdd-red.txt` records missing recovery/inventory/allocation features and the reproduced unsupported subprocess API; `review/capture-red.txt` records all four expected capture assertions failing before correction. The added tests cover known directories versus unknown/symlink entries, evidence no-overwrite/owner/read-only/WAL/hash/source-replacement/destination-tamper guards, exact failed costs, requested64/exclusive288geometry, fixed recovery bindings, quota arithmetic and Python3.6syntax/API. Capture tests use real harmless children for stdout-only, stderr-only, combined-stream overflow and non-timeout read-error cleanup.

Final commands and raw results are in `review/test-results.json`:

- All34previous adapted guards plus8new test methods:42passed,9.911seconds.
- Unchanged original final_p1suite:11passed,17.431seconds.
- bash syntax check:exit0.
- release_v5ledger:all entries OK,exit0.
- Exact transport membership and every transported file hash verified.
- All121prior deployment/source-candidate files preserved.
- Actual host Python3.6.15compile/API/pure-validator smoke:exit0.

Advisor consultation failed due to tool/client model incompatibility; no advisor review is claimed. Graft was consulted before source exploration; its command reported a background graph rebuild already in flight. No source-worktree edit was needed.

## Final immutable identities

| Item | SHA256 |
| --- | --- |
| release_v5.tar | b3dafe316605a8ccaae61e94854fe527bd8591cb4600ae152afae017d2807458 |
| release_v5/release.json | e0d2262a7aa94b49cdf08de764d6b807c75779dc32a5a7e83ff12810084c5389 |
| release_v5/release.sha256 | 10e2ec018cf00bb9180f8884b44a00a1d376734e7c5abfeb84e97be706facaa9 |
| Unchanged candidate source archive | da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7 |

The final canonical authorization path is `staging/dev123_p1_controller_authorization_v5.json`; execution namespace is `staging/dev123_p1_release_v5_20260917`. External controller-gate, transport and ledger SHA256pins remain mandatory. Durable authorization can wait24hours or more, but live observations must still be fresh and complete. Top authorization.example.json matches the executable unapproved schema.

Completed: local narrowed recovery implementation, both query-boundary fixes, regression verification, actual host compatibility checks, immutable final transport and preservation evidence.

Blocked next: independent review of these exact final bytes, then any separately approved controller authorization and execution decision. No cancellation, submission, transfer or automatic retry is authorized by this document.
