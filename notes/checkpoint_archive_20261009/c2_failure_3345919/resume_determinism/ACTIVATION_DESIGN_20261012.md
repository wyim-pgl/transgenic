# C2 in-place activation recommendation — 2026-10-12

Status: parent adopted this design for implementation after reviewing the actual filesystem evidence. Deployment still requires implementation, focused verification and independent review. No scientific acceptance or activation has occurred. This design supersedes directory promotion in CONTINUATION_IMPLEMENTATION_PLAN.md. Existing user authorization covers immediate production4099 after the scientific gates; it does not require a particular output directory.

## Recommendation and evidence

Activate the accepted validation allocation's `resumed` directory **in place**. Keep `runs/b6_wp5_pilot_C2` as the protected failed original200, with its failure marker. Keep diagnostic256, validation `continuous`410 and `at400`, all proof directories, and all original comparison/READY receipts unchanged. Record the relationship in one immutable activation receipt under the new operation. Production requests point to the activated directory's actual WORK-relative path.

This requires a controller change, not a training-source change. `production_link.py` already uses `request.run_dir` for routing, checkpoint selection, trainer output, postflight, and final publication. Its version1 request/receipt schemas can remain unchanged. The initial predecessor remains the original validation READY receipt; its `relative_run=resumed` is still true, so no relocation resolver is needed.

The parent-provided `/tmp/c2-host-filesystem-capabilities.json` was read during this review. It records host `gh-login03.delta.ncsa.illinois.edu`, kernel `6.4.0-150600.23.53-64kb`, direct native no-replace renames of an absent destination for both a file and directory returning `EINVAL(22)`, preserved sources, absent targets, and a successful exclusive hard link with matching filesystem identity. This is evidence about that host/mount/operation. It does not prove that every rename, every Lustre mount, every host, or every kernel lacks support. Existing-target `EEXIST` alone does not prove that a successful absent-target promotion is supported.

The strict `renameat2(RENAME_NOREPLACE)` implementation must not fall back to `exists()` plus `os.rename()`, ordinary replacement, directory copying, or a symlink. In-place activation removes the need for that operation. Ordinary reviewed trainer checkpoint writes retain their existing behavior; this proposal concerns ownership of the production run directory.

## Path meanings and ownership

Use an explicitly new controller configuration/journal version and a new operation directory. Prefer a single in-place workflow over adding a runtime switch that maintains both promotion strategies. Reject legacy promotion states/configurations rather than interpreting them under new semantics.

- Rename the controller's configuration concept `canonical` to `original_run`; remove the unused `backup` requirement. `original_run` remains `runs/b6_wp5_pilot_C2`, permanently protected at200. Do not invent a backup path that does not exist.
- Keep the logical C2 ownership lock and claim at the **same physical names** currently derived from that original path: the sibling `.b6_wp5_pilot_C2.history.lock` and `.b6_wp5_pilot_C2.history.owner.json`. Changing the active path must not create another independent C2 lock namespace. Keep the operation lock, owner UID, persistent claim, and claim/config identity checks.
- Derive the activation target solely from accepted validation evidence: `receipts.validation.root + '/resumed'`. It is not an operator-selected arbitrary output directory and is not mutable configuration.
- The activation receipt pins `active_run`; journal state holds the receipt path/hash. A private `active_run()` method validates and returns it. Every production path read goes through that method or through a request whose `run_dir` has been checked against it.
- Enforce safe relative paths and no symlink ancestors. Require the target to be the accepted validation `resumed` directory and disjoint from original200, diagnostic256, immutable proof data, source/helper directories, and operation control files. The intentional validation-parent/resumed-child relationship is permitted; `continuous` and `at400` are separate protected siblings.
- Record directory `st_dev`/`st_ino` only as audit observations, with the observing host. Do not make client-local device/inode numbers a persistent cross-host acceptance gate: earlier recovery work established that device identity can vary by client. Exact derived active path, no symlink ancestors, immutable activation/source/receipt bindings and allocated full checkpoint verification remain mandatory. Tests must verify activation itself leaves the directory inode unchanged.
- Extend competing-writer checks to the original path, activated path, validation target, and known production script/token identity. Keep the existing broad C2/owner/job checks and exact owner/command/state validation of obsolete3345920. Any ambiguous observation still blocks. The preserved original failure marker is not a failure marker for the different active run.

No directory is moved, copied, deleted, or symlinked by activation. No file is added inside original200, diagnostic256, or the validation evidence directories by activation. Only small control/evidence files are written under the new operation. The active `resumed` checkpoint and draw logs subsequently evolve through ordinary production training.

## Activation receipt and publication

Publish `STATE/activation_receipt.json` with an explicit `kind=c2_history_activation`, `mode=in_place`, and strict versioned fields. It is the immutable pointer, not a mutable `current` symlink. Include:

1. Operation ID, immutable configuration SHA, owner/UID, original run and diagnostic paths, active WORK-relative run, and audit-only target directory device/inode observations with host.
2. Source commit/snapshot, source/helper/config/bindings pins, actual runtime fingerprint hash, bootstrap hash, and the original checkpoint receipt pin.
3. Accepted proof and validation job IDs, comparison pins, validation READY pin, and obsolete-job retirement identity/evidence already held in the journal.
4. Initial start checkpoint at epoch0/step9840/global410/pending0, its original meta/identity/history byte hashes, and the verified canonical checkpoint digest.
5. Exact initial small meta/identity/history bytes embedded as UTF-8 strings with hashes, obtained through the existing bounded reads. This preserves initial candidate metadata even when `latest_state` advances, without copying checkpoint payloads. Retain validation `continuous`410 and `at400` as the actual unchanged state evidence.
6. Explicit semantics: original run preserved in place; candidate activated without relocation; validation comparison describes the checkpoint **at activation**, not an assertion that the active directory will remain at410 forever.

Construct the deterministic receipt payload in the durable activation intent. Do not regenerate timestamps or identity values differently during recovery. The publication primitive is small-file exclusive publication: create a unique temporary file in the same operation directory, write/fsync it, `os.link(temp, final)` exclusively, then fsync the parent. The parent smoke supports this primitive on the tested filesystem, and `production_link.exclusive_json` already uses this pattern (production_link.py:80–95). Use a collision-safe temporary name and preserve or clean only controller-owned temporary control files. Existing final bytes must match the exact intended payload; any different file, symlink, empty directory, or malformed content blocks without replacement. Existing `atomic_json` remains for intentionally mutable journal writes; do not use its `os.replace` behavior to publish the immutable activation receipt.

The stable ownership claim already points to the operation; that operation's receipt identifies the active run. No second global mutable pointer and no run-local receipt are necessary. Completion/status output should expose the receipt pin and actual active path.

## State transitions and crash handling

Retain proof adoption, proof acceptance, validation submission/acceptance, and exact held-job retirement unchanged. Only after terminal validation success and the retired old job have been rechecked:

`old_job_retired -> activation_intent -> candidate_activated -> production_submit_intent -> waiting_production`

Before writing intent, revalidate source/helper/script coverage, proof/READY/comparison pins, original200/diagnostic pins, all initial validation artifacts, exact410 candidate state and absence of failure/final markers, owner/claim, absence of writers, retired old-job state, and the existing100GiB quota/filesystem reserve. Recheck the same conditions immediately before receipt publication. This is the pre-activation replacement for the pre-promotion gate, not its removal.

| Crash or conflicting state | Required outcome |
|---|---|
| Before intent | All runs untouched; next tick can repeat read-only gates. |
| Intent durable, receipt absent | Revalidate the same candidate and gates; publish only the exact intent payload. No submission. |
| Temporary control file only | It never authorizes production. Revalidate; recover using another unique temp or audit the owned temp. Never infer success from its presence. |
| Receipt published, journal still `activation_intent` | Verify exact receipt bytes/hash and derived target path, revalidate410 while no submission intent exists, then roll forward to `candidate_activated`. Never republish different evidence. |
| Receipt differs, derived target path changes, symlink appears, foreign owner or writer | Block, preserve every existing file, record the reason. |
| `candidate_activated`, no submission intent | Revalidate receipt, preserved evidence, and current exact410 before constructing link1. |
| Submission intent exists, including uncertain sbatch result | Existing token/owner/time/exact-argv reconciliation only. Never retry or re-activate. No job found is still uncertain. |
| Production has started or an accepted production link exists | Do not require active `latest_state` to equal the historical410 files. Validate the current result through the exact request/receipt chain and allocation preflight/postflight. |

Keep one durable action transition per tick, normal fsync discipline, persistent ownership after process exit, and concurrent-process locking. No state reset, claim replacement, marker removal, or automatic migration from a partially completed legacy promotion is part of this change.

## Receipt revalidation over time

Before the first submission, all accepted proof and validation artifacts, including live resumed410, must still match. The first production request continues to bind the validation READY SHA and exact410 canonical value digest; allocation preflight rehashes/deserializes the real active checkpoint and rejects a substituted or changed state before training.

After submission, immutable evidence and evolving run state have different checks:

- Continue pinning the original proof trees, original200 receipt and small files, diagnostic256, proof/validation comparisons, validation READY, validation `continuous`410/`at400`, activation receipt, and the embedded initial candidate bytes.
- Do not rewrite validation READY/comparison/artifact records and do not redirect their old paths to current training state. In-place activation has no relocation.
- For the validation artifacts specifically under `resumed/latest_state`, the activation receipt preserves their historical bytes/hashes; it is the explicit transition from validated evidence to the owned training run. Check those historical records against the activation receipt after activation, and against the live files only before first submission. This is not a general “skip changed evidence” exception.
- Validate current active checkpoint/markers/epochs against the last request and allocation receipt. Preserve completed epoch events and draw-prefix continuity. No historical canonical digest is asserted to describe later weights.
- Verify the activation receipt hash and exact derived active path before building every request, accepting every link, and performing final priority restoration. Verify immutable evidence in controller transitions without trying to hash growing tensor payloads on login.

The production helper need not consume a new receipt schema. `request_sha256` binds the actual run path and exact initial state; the host binds that request to the activation receipt. Keep the first predecessor as validation READY and later predecessors as accepted production receipts. Do not silently substitute the activation receipt as a different predecessor kind.

## Exact implementation map

The following source ranges were inspected using the graph and then exact reads. Ranges refer to the controller with the original author's current P2 corrections; re-query the graph after implementation.

| File / covers span | Required change or retained behavior |
|---|---|
| continue_c2_history.py:236–239,256–279 | New explicit config version/`original_run` semantics; preserve the existing logical lock/claim filenames. Remove unused backup configuration; validate protected/active path relationships. |
| continue_c2_history.py:364–408 | Versioned init/status and claim validation; new operation only. Bind claim/config identity, never replace a foreign operation's claim. |
| continue_c2_history.py:416–444 | Keep all manifest/script/config/input pins. Query storage for the candidate/active run when known, so the measured project22169/ddn_ssd quota and filesystem correspond to the actual writer. Preserve `max(100GiB, clone_peak)` before validation and100GiB before activation/every production submission. |
| continue_c2_history.py:446–462 | Extend `no_writers` to actual active path and original namespace; do not lose existing C2-token/script detection, protected-job checks, or retired-job reactivation refusal. |
| continue_c2_history.py:544–552 | `original()` always verifies original200 at `original_run`; no backup remapping parameter. Keep original small-file pins and position checks. |
| continue_c2_history.py:609–661 | Scientific acceptance stays strict and unchanged in substance. Still validate proof220/210 and validation410/400, exact histories/slot6718, original receipt binding, READY canonical digest, historical210 match, marker/test-count/accounting checks. |
| continue_c2_history.py:663–685 | Replace relocation mapping with explicit pre-activation live410 versus post-activation historical evidence handling. Derive candidate only from the accepted validation root; add the validated active-run accessor. |
| continue_c2_history.py:695–722 | Retain exact owner/PENDING/JobHeldUser checks and cancellation reconciliation. Revalidate terminal old-job state before activation. |
| continue_c2_history.py:724–775 | Replace promotion preparation/rename states with activation intent/exclusive receipt publication/recovery. No call to `rename_new`, no `original_archived` or `candidate_promoted` state, no `RECOVERY_PROMOTION.json`. |
| continue_c2_history.py:777–810 | `run_dir=active_run()`; original check uses unchanged original path. Link1 checks exact410 plus activation; subsequent links retain positive-progress receipt chain. Keep request schema/version1 and initial READY predecessor. |
| continue_c2_history.py:827–856 | Epoch evidence is read below `active_run`, with unchanged consecutive epoch, finite evaluation, history and prior-event checks. |
| continue_c2_history.py:858–928 | Require request.run_dir equals activated run; resolve checkpoint/markers below it. Exact final `pilot_final` must be `/work/<active_run>/step_04099`, not the protected original path. All4099/position/inventory/input/epoch/model/DONE checks stay. |
| continue_c2_history.py:930–968 | Accept links against original200 in its original place and the pinned active run. Final revalidation/priority restoration expose actual output path/activation pin and retain separate held-B6 follow-up block. |
| production_link.py:286–315,335–368,467–496,499–564,582–664 | Existing request.run_dir plumbing suffices: schema, host paths, selected checkpoint, `--output-dir`, allocation checks and postflight all use it. No scientific/runtime-source refactor is required. |
| production_link.sbatch:56–73,89–99 | Existing host RUN_HOST is routed from the request; quota, trainer failure marker and receipts follow it. Do not accidentally bind the whole validation root read-only over its writable resumed child. |

Keep initial proof/helper/source manifests immutable. A controller successor requires a new deployment snapshot/manifest and explicit updated controller/config pins; do not amend already submitted proof or validation helper evidence. This task does not change the original author's uncommitted P2 corrections.

## Downstream path semantics and B6 follow-up

The actual final output is `<WORK>/<validation_root>/<validation_job_id>/resumed/step_04099`, bound by activation and final production receipt pins. `runs/b6_wp5_pilot_C2` still means failed original200. Status, completion evidence, archival inventory, export commands, later analyses, and any B6 comparison must state both roles honestly; do not claim the canonical old directory reached4099 and do not create an alias that hides the distinction.

The inspected selected-source launcher `/tmp/b6-c2-recompile/revision/scripts/b6/wp5_pilot.sbatch:53–56,84–89` hardcodes `runs/b6_wp5_pilot_$ARM`. With `ARM=C2` it selects the preserved failed original and refuses its failure marker. Do not use that launcher to continue activated C2 or remove that marker. The newly reviewed production helper supplies the explicit active path instead. The selected trainer `train/train_HyenaTransgenic_B6.py:313–372` publishes final checkpoints relative to `spec.run_dir`, so no source edit is needed to obtain the actual activated final path.

Graph searches over indexed main/source Python and literal searches over unindexed shell/config files found these additional canonical-path consumers: historical `continue_c2.py`, `prepare_resume_clone.py`, `storage_repro.py`, old diagnostic/proof/validation batch scripts, and historical archive inventories. Preserve these historical artifacts. The proof/validation launchers should continue reading original200 at the old path. Future archival must inventory the real active root separately, keep activation/final receipt provenance, and not treat historical inventory entries as production completion.

No read-only local search establishes the contents of every remotely staged launcher or external analyst script. The held-B6 follow-up already requires source/launcher/input provenance review and replacement planning; resolve C2 through the final operation evidence in that review. Do not release old held B6 scripts. B5/B6 Nice0 restoration remains gated on fully verified C2 completion, with B6 held. This path choice introduces no new scientific wait or user-approval flow.

## Alternatives considered

| Approach | Assessment |
|---|---|
| Activate existing resumed directory in place | Recommended. No checkpoint-tree transfer; actual source/helper interfaces already support an explicit run path; only ownership/provenance and controller path handling change. |
| Exclusive hard-link transfer of a checkpoint tree | The successful single-file hard-link smoke is useful for publishing a small receipt, but does not validate a tree migration. A tree requires exclusive destination creation, per-entry inventory/fsync/recovery, type/symlink checks, inode-alias handling, and reconciliation of partial trees. Growing logs and any in-place writes can alias retained evidence. Retiring the old original path still needs a separately safe operation. This is larger and riskier than the authorized requirement needs. |
| Execute atomic no-replace rename from another host | Not ruled out globally. It is viable only after an actual absent-target smoke on that host and the same mounted backing filesystem proves the operation, while preserving owner/lock/fsync/identity guarantees. A different host may use different client/kernel behavior; inference from a different filesystem is insufficient. Remote execution adds another crash/ownership surface and offers no necessary scientific benefit when the output path is unconstrained. Do not delay validated production merely to preserve the old pathname. |

## Acceptance tests and deployment evidence

Add focused behavioral regressions before implementation and retain the current43 controller tests' non-promotion safety coverage. Update promotion-specific cases to the new semantics; do not reduce their crash/concurrency/integrity assertions.

1. Real temporary runs: original200 and diagnostic256 stay at exactly the same paths/inodes/bytes; validation continuous410, at400, comparisons/READY and proof files stay unchanged. Active `resumed` is the identical directory inode before and after activation. No checkpoint-tree rename/copy/delete/symlink occurs; make `rename_new` raise if called.
2. Activation succeeds when native no-replace rename is deliberately unavailable, using only independently tested exclusive receipt publication. Pre-existing conflicting file/empty directory/symlink destinations block and remain intact; identical receipt recovery is idempotent.
3. Inject crashes before intent, after intent, during temp publication, after final link, after fsync, and before journal advance. Recovery creates one identical receipt and no early/duplicate production job. Run concurrent controllers and competing operations against the same stable logical C2 lock/claim.
4. Drift or forgery in proof/READY, original/diagnostic pins, initial candidate meta/identity/history/digest, source/helper/script/config/input pins, activation fields/hash/derived target path, owner/claim, or protected path relationships blocks. Both originally fixed P2 regressions remain covered.
5. Initial request uses the accepted resumed path and READY predecessor. Run the existing controller-to-production-helper integration fixture through allocation preflight, clean positive-progress preemption, a second link, and final4099 at a noncanonical nested path.
6. After real active latest_state changes, immutable evidence checks still pass only for the recorded initial snapshot/retained siblings, while current-state checks require the new production receipt. Changing continuous410/at400 or proof/READY still blocks. A forged old canonical-path `pilot_final` fails; the actual active-path final passes only with all existing exact4099 markers/epochs/normal exits.
7. Preserve strict low-storage blocking at validation, activation, and every production submission; no extra whole-run reserve. Keep exact held-old owner/state checks, no writer/duplicate submission behavior, unknown-submission reconciliation, timeout/OOM/nonzero/no-progress refusal, and four-rank fresh verification.
8. Host Python3.6 grammar/stdlib import and shared-filesystem flock/exclusive-link/fsync smoke; independent review of the controller successor and unchanged allocation interface; regression suite plus existing allocation tests, including current production P2 fixes. Pin the tested deployment files and verify immutable proof/validation manifests remain the expected versions.

Actual four-GPU proof3369694, terminal accounting/markers, validation400 with full continuous/resumed checkpoint equality, and all ordinary allocated preflight/postflight checks remain required. Local tests and an activation design do not satisfy those scientific gates. Once those gates and operational review pass, submit full4099 under the existing authorization without asking again.
