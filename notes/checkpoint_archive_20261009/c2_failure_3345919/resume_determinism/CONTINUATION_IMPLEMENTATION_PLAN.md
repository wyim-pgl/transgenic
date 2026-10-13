# C2 automatic continuation implementation plan

> Use subagent-driven-development, test first, independent review. Existing user authorization covers validation and production4099 after scientific gates; no approval question or remote action belongs in implementation tasks.

**Goal:** A new controller observes the submitted proof, advances through reviewed validation400, promotes the verified resumed410 clone without another copy, and drives ordinary source198ce14 production links to verified4099.

**Spec:** CONTINUATION_DESIGN_20261012.md, with the concrete contracts/rulings below taking precedence. Source code is immutable198ce14 in `/tmp/b6-c2-recompile`. Initial failed job3369338's deployed v1 helpers never change; reviewed successor source198ce14 uses them in replacement proof3369694. Main holds diagnostic/operational helpers and records; no implementation branch merge.

## Adopted amendment: activation in place

`ACTIVATION_DESIGN_20261012.md` supersedes all physical promotion/backup/relocation requirements below. Actual shared Lustre rejects `RENAME_NOREPLACE` for absent targets. Preserve original200 at its existing path and activate the accepted validation resumed410 directory in place; no directory copy/move/delete/symlink. Use a new controller config/journal version with `original_run` and receipt-derived `active_run`, stable original-C2 ownership lock, immutable operation-only activation receipt, unchanged production request/receipt v1, and all existing scientific/storage/owner/submission/final gates. Client-local directory device/inode observations are audit-only, not persistent cross-host gates. Initial410 metadata is embedded in activation provenance; retained continuous410/at400 preserve baseline evidence. Once production starts, the active state evolves only under the accepted request/receipt chain. Source/helper deployment remains separately pinned and reviewed. No additional user approval is required by this implementation choice.

## Global constraints and decisions

- Implement only new helpers/controller under this notes directory. Never import/execute the old continue_c2.py or use its state directory/PID.
- Host controller is Python3.6 stdlib only. It can import the new history_proof.validate_prior_proof pure validator (now stdlib-only at import); it must never call tensor/history loading functions on login.
- Source/helper/config/bindings pins, actual effective runtime fingerprint, owner and scheduler states, exact strict proof/validation receipts, duplicate submission protection, protected paths and original checkpoint identity remain enforced.
- All large hashes, model/optimizer/RNG/gradient validation and ordinary fresh input verification run inside allocations.
- Existing storage_preflight100GiB is a hard PRE-SUBMISSION/pre-promotion gate. Also ensure planned validation clone peak fits available bytes. Do not invent an additional requirement to reserve100GiB plus the entire remaining multi-epoch training footprint before the first production link. Recheck before each link; low space blocks and is recorded, never bypassed. Cost of this ruling: later links may wait for separately verified archival if intervening epoch/cache/other-run growth reduces free space.
- Use the proof's effective fingerprint and fixed SIF path/version context. Do not demand an unavailable historical SIF file hash as a new prerequisite; a new allocation may record current image details, and no heavy image hash belongs on login.
- Promote resumed410 by same-filesystem rename; preserve original200 in a unique backup, diagnostic256 unchanged, continuous410 and at400 in validation root. No deletion/archival implementation in these tasks.
- Ordinary clean preemption requires a READY checkpoint receipt and zero trainer/parent/batch exit. Hard timeout/OOM/nonzero/no progress/missing receipt blocks; no automatic fallback or failure-marker removal.
- Final acceptance requires exact4099, fixed-step model/identity/history/markers/DONE, finite actual completed-epoch evaluations and normal exit. Do not add an unrequested evaluation of the final partial epoch. B5/heldB6 Nice0 restoration only after final verification; never release old held B6 scripts.
- No remote mutation, job submission, promotion or daemon launch by implementers. Controller owns staging and live deployment after independent review.

## Shared link request contract

All paths in requests are relative to the approved host WORK root, mapped inside the container under `/work`. Reject absolute/traversal/symlinked control paths. Source config is relative to source directory. Request SHA is passed separately to the allocation helper.

Required request fields (version1):

```text
version, operation_id, token, link_index,
source_commit, source_snapshot_git, source_dir, source_manifest_sha256,
helper_dir, helper_manifest_sha256,
run_dir, bindings, bindings_sha256, config_relative, config_sha256,
receipt_dir,
profile = {arm:C2,target:4099,seed:123,world_size:4,batch_size:1,mixed_precision:bf16,deterministic:true},
start_checkpoint = {relative_dir,epoch,step,global_step,pending_micro,
                    meta_sha256,identity_sha256,history_sha256,canonical_checkpoint_sha256},
predecessor = {relative_path,sha256},
expected_runtime_fingerprint,
completed_epoch_events
```

start_checkpoint.relative_dir is `latest_state` or a committed epoch_NN, never a fixed-step weights-only publication. Initial canonical_checkpoint_sha256 is validation's already verified resumed410 `probe.digest(probe.checkpoint(...))`. Later receipts use the same model/state value-tree canonicalization, plus a pending_grads map only if pending_micro>0; document and test that extension. No Python pickle is used for control files; only reviewed Torch loads of owned checkpoints inside the allocation.

receipt_dir is the new operation's `links/NNNN`; allocation output is `receipt_dir/execution_JOBID`, created once by the batch helper. Four rank result files and a final link_receipt.json live there. A receipt never overwrites previous evidence.

## Shared production receipt contract

```text
version=1, kind=c2_production_link_ready,
operation_id,token,link_index,job_id,request_sha256,source_commit,predecessor_sha256,
start_global_step,result,profile,runtime_fingerprint_sha256,
input_verification (fresh allocation ID, all four ranks and verified binding identity),
checkpoint (position,relative_dir,meta_sha256,identity_sha256,history_sha256,
            canonical_checkpoint_sha256,files:{relative_name:{size,sha256}}),
epoch_events,
final (null for preemption, otherwise validated4099 publication/marker/model hashes,
       expected completed epoch evaluations and their bindings)
```

All booleans and integers are strict; all hashes have explicit expected lengths. Use existing source constants for checkpoint/marker filenames and selection policy. A final receipt may represent final fixed-step publication rather than a resumable checkpoint; controller must never submit another link when result.done/final are valid. Header schema/paths can be refined coherently by the two implementers through the controller, but neither may silently diverge from the other.

### Task 1: Reviewed production allocation helper

Own `production_link.py`, `production_link.sbatch`, `test_production_link.py`, `PRODUCTION_LINK_IMPLEMENTATION.md`. No controller or science-harness edits.

- Read current source launcher and exact trainer/runtime graph spans from the design. Preserve proven USR1→SIGTERM-to-ranks/wait/exit behavior in the new batch helper; actual worker script is production_link.py run, so identify its rank descendants rather than the old trainer filename. Never signal unrelated jobs or kill the launcher as the clean-stop mechanism.
- Implement CLI preflight/run/postflight around the shared request. Validate request/source/helper/config/bindings/fingerprint and predecessor/start checkpoint. Bind WORK read-write, source/helper/evidence read-only, operational shared caches, exact4GPU/bf16/deterministic settings. Match original launcher partition/time/resource defaults; prevent inherited SBATCH/APPTAINERENV compile overrides from changing the reviewed request.
- `run` calls ordinary trainer.main with --resume auto and4099. Observers may record original verify_training_inputs/record_identity/run_b6 results, but must call originals and must not bypass verification, update optimizer/scheduler, or modify the training recipe. Record each rank's successful fresh verification and true result; refuse missing verification. Capture actual canonical train row/batch/accumulation counts needed for epoch acceptance.
- Postflight runs once in the allocation after all ranks exit0. Validate complete selected resume state and pending grads when needed, identity directionality (old16/2→new18/3 only), actual history schema/fingerprint/position, finite state tensors, progress and absence of failure flags. Hash bytes and canonical state here, not in the controller. For final4099 verify weights hash, exact final metadata/markers/nonemptyDONE and all prior completed epoch evaluations. Use true returned history/observations and sampler counts; do not infer a new final partial-epoch eval. Emit only a fully validated receipt.
- TDD cases: forged/stale request/receipt/source/history; missing fresh verifier on any rank; partial optimizer/RNG/gradient; no-progress; normal preempt/epoch pending; final4098/4100/NaN/staleDONE/missing epoch/PILOT_FINAL; source/helper change; real subprocess signal routing and nonzero-after-DONE. Use tiny owned tensor fixtures and real local process behavior; avoid source-text assertions. Keep tests deterministic and bounded.
- Run focused unittest, bash-n/py_compile and self-review. Commit only owned files/report; record exact RED/GREEN and request/receipt implementation details for Task2/reviewer. No remote action or spawned agents.

### Task 2: New host controller and durable promotion/submission

Own `continue_c2_history.py`, `test_continue_c2_history.py`, `CONTINUATION_IMPLEMENTATION.md`. Do not edit Task1 or science helpers.

- Implement init/status/tick/watch/reconcile with a new immutable config and durable journal, operation and canonical-C2 locks. Config supplies explicit source/snapshot/helpers/manifests, WORK, owner/UID, original/candidate/backup/diagnostic paths, existing proof ID/receipt pins (currently3369694), validation script, production script, protected job IDs and resource choices. No defaults to the old controller/state/source.
- One tick makes at most one durable mutation transition; waiting does no mutation. Adopt configured proof3369694, never resubmit it; preserve failed predecessor3369338. Parse strict pure initial-proof summary, predecessor identity, actual parent+batch states (allowed externOOM0:125 only), terminal markers and manifests. Validation job submitted only after all proof gates. Require validate400 summary/ready receipt and canonical digest/metadata bindings before any old-job retirement or promotion.
- Persist/fsync exact argv/token intent before external writes. sbatch runs once; timeout/unparsed output/accounting uncertainty never triggers automatic retry. Reconcile token+owner+time+exact submission identity; only one proved job may be adopted. No job found is not proof of non-submission. Block ambiguous observations while preserving the intent.
- Verify old3345920 ownership/PENDING/JobHeldUser before retiring it; already terminal cancellation can be reconciled. Running/released/foreign jobs block. Preserve protected B5/B6 scheduling until final verification.
- Promotion uses three known layouts: original200+candidate410+no backup; backup200+candidate410+no canonical; backup200+canonical410+no candidate. Verify small source/candidate artifacts, receipts and absence of writers, save intent, rename/fsync parents, emit relocation provenance, advance state. Everything else blocks; never replace even an empty destination, delete protected data, copy across filesystems or silently roll back.
- Build exact shared link requests and submit one ordinary production link at a time. Validate terminal success plus allocation receipt, small checkpoint/history/marker hashes and raw inventory presence/stat sizes; do not rehash tensors on login. Preserve epoch events across links. A clean positive-progress checkpoint loops; final verified4099 restores Nice0 idempotently for remaining owned B5/B6 IDs, keeping B6 held, and records completed C2 plus separate B6 follow-up block.
- Tests must exercise real temporary journal/file renames and concurrent processes, with a focused fake scheduler adapter for external Slurm only. Cover the12 fault groups in design at this task's boundary: duplicate tick/init, uncertain sbatch, forged science/receipt, metadata/source drift, held-owner/state, every rename crash window, low storage, no progress/failed/timedout/requeued links, final markers/epochs and idempotent nice restoration. No generic framework or shell-eval hooks.
- Run on local Python with Python3.6 grammar/stdlib import checks; actual host3.6 smoke and shared-filesystem lock checks are controller deployment tasks. Self-review/scoped commit/report; no remote actions or spawned agents.

The parent reviews both tasks and their interface together before any launch. Existing source/helper models remain pinned; validation and production receipt tools do not replace the real4GPU scientific gates. Archive operations or abnormal-preemption audits, if needed, are separate reviewed follow-ups rather than hidden recovery shortcuts.
