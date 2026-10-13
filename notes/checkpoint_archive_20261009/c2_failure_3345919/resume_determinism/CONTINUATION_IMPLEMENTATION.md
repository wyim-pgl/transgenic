# C2 in-place continuation controller — version 2

`continue_c2_history.py` activates the accepted validation allocation's `resumed` directory in place, then drives receipt-bound production links to4099. The protected failed original200 remains at `original_run`, including its failure marker. Diagnostic256, proof evidence, validation continuous410/at400, comparison and READY files remain unchanged. The controller does not move, copy, delete or symlink checkpoint directories; it writes activation provenance only inside the new operation.

This implements the adopted `ACTIVATION_DESIGN_20261012.md` amendment. Config and journal are **version2**; production request and receipt schemas remain **version1**. Legacy config fields and physical-promotion journal states are rejected without migration. The old controller is never imported, executed, or resumed. All model/optimizer/RNG/gradient validation and fresh input verification remain allocation work. Host Python3.6 uses only the standard library and the reviewed pure `history_proof.validate_prior_proof` function.

## Configuration

Use a new operation directory under WORK, disjoint from original/diagnostic/proof/validation runs, source, and helper directories. The complete executable configuration fixture is `test_continue_c2_history.ControllerTests.config`; it is not a deployment configuration. All data/control paths below are WORK-relative unless marked absolute; `config_relative` is relative to source.

Routing excerpt (the remaining required pins are listed below):

```json
{
  "version": 2,
  "operation_id": "c2-history-in-place-20261012",
  "owner": "wcyim",
  "uid": 97434,
  "work": "/work/nvme/bilv/wcyim/transgenic",
  "sif": "/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif",
  "original_run": "runs/b6_wp5_pilot_C2",
  "diagnostic": "REVIEWED_DIAGNOSTIC256_RELATIVE_PATH",
  "proof_job_id": "3369694"
}
```

Do not supply `canonical`, `backup`, `active_run`, or an unused scratch setting. The active path is derived exclusively from accepted validation evidence, never selected by the operator. A missing pin blocks the corresponding gate.

| Required fields | Meaning |
|---|---|
| `version=2`, `operation_id`, `owner`, `uid` | Unique operation, actual account and UID; UID must equal the running process. |
| `work`, `sif` | Absolute host paths. No unavailable historical SIF hash is introduced. |
| `original_run`, `diagnostic` | Protected failed original200 and diagnostic256. |
| `source_commit`, `source_snapshot_git`, `source_dir`, `source_manifest_sha256` | Reviewed implementation, actual snapshot Git HEAD, source directory and its `validation_sources.sha256` digest. |
| `bindings`, `bindings_sha256`, `config_relative`, `config_sha256` | Exact bindings/configuration paths and byte digests. |
| `expected_runtime_fingerprint` | Complete actual effective fingerprint from accepted scientific proof. |
| `original_pins`, `diagnostic_pins`, `proof_receipts` | Lists of `{relative_path,sha256}`. Original pins include original200 meta, identity and its failure marker. Proof receipts bind reviewed staging/submission evidence. |
| `original_checkpoint_receipt`, `bootstrap_sha256` | Original allocation-verified receipt pin and exact bootstrap bytes digest. |
| `proof_job_id`, `proof_identity` | Adopted ID and `{comment,command,submit_line}`; `submit_line` is exact accounting evidence. There is no initial-proof submission operation. |
| `phases` | Exactly `proof`, `validation`, `production`; see below. |
| `obsolete_job` | `{job_id,command}` for old C2 held3345920; command is its exact normalized absolute script path. |
| `protected_jobs` | Ordered `[{job_id,command,held}]` for B5/B6. Held B6 must stay PENDING/JobHeldUser even during Nice0 restoration. |
| `resources` | Explicit `{account,partition,time,cpus_per_task,exclude,output}`. Output is a WORK-relative Slurm pattern; use a phase marker-log pattern matching the explicit output/job-name choice. |
| `validation_clone_peak_bytes` | Available capacity must independently meet100GiB and fit the planned clone peak: `max(100GiB, peak)`, not their sum. No whole-production-footprint reserve. |
| `storage_python`, `storage_script` | Absolute host interpreter and WORK-relative reviewed preflight script. Require project22169/ddn_ssd and both quota/filesystem headroom. |

Each phase contains `helper_dir`, `helper_manifest_sha256`, `script`, `root`, `marker_log`, and `tests`. The exact script must be inside its helper directory and a member of its pinned `manifest.sha256`. Optional phase `resources` overrides the complete global resource object with an explicit reviewed choice. Science roots append the actual job ID. `marker_log` uses `{job_id}`; `tests` maps `ngc_tests.log` and `harness_tests.log` to exact reviewed `{ran,skipped}` counts. Production acceptance uses its allocation receipt rather than those test logs. Independently review complete deployed manifests and the explicit controller launch path.

Proof/validation launchers retain their reviewed fixed original200/SIF/bindings paths; config must match them. Source/helper manifest members are streamed because the source snapshot includes18–43MiB example assets. JSON/control files keep bounded reads; the host never substitutes payload hashing for allocation verification.

## Activation and immutable evidence

The logical C2 lock and ownership claim retain the old physical sibling names derived from `original_run`, for example `.b6_wp5_pilot_C2.history.lock` and `.b6_wp5_pilot_C2.history.owner.json`. An active-path change cannot create a second C2 ownership namespace. The operation lock, owner UID, persistent claim and config hash remain enforced. Existing claims and legacy operations are never silently replaced.

After strict proof and validation acceptance, retire old3345920 only with exact owner/UID/script and PENDING/JobHeldUser checks. Recheck terminal CANCELLED identity, terminal validation parent+batch success, all pins, original/diagnostic evidence, current exact410, absence of writers and100GiB available before activation intent and again immediately before publication.

The transitions are:

```text
old_job_retired -> activation_intent -> candidate_activated
 -> production_submit_intent -> waiting_production
 -> link_accepted -> next production link OR final_verified
```

`activation_prepare()` creates a deterministic payload in the durable intent. `activate()` writes/fsyncs a unique operation-local temporary file, publishes `activation_receipt.json` with exclusive `os.link`, and fsyncs the parent before journal advancement. An existing receipt must have identical intended bytes. A conflicting file, empty directory or symlink blocks without replacement. Only the controller's own temporary control file may be cleaned up. A leftover temporary file never authorizes production.

The receipt has version2, kind `c2_history_activation`, mode `in_place`, operation/config/owner bindings, original and diagnostic paths, and exactly `validation.root + '/resumed'` as `active_run`. It includes source/helper/config/bindings/fingerprint/bootstrap/original-receipt pins, accepted proof and validation comparison/READY pins, retirement evidence, exact initial410 checkpoint digests, and the initial meta/identity/history bytes as UTF-8 strings with their hashes. Its semantics explicitly state that comparison describes the checkpoint at activation while ordinary production may advance the active run.

Directory device/inode values are audit observations with the observing host. They are **not persistent acceptance gates across clients**. Exact derived safe path, no symlink ancestors, receipt/source bindings, disjoint protected paths, and allocation full-state verification remain mandatory. Tests verify activation itself preserves directory inodes and every protected byte.

Crashes after intent, temporary publication, exclusive final publication, directory fsync, or before journal advancement recover from the original intent payload. Recovery never regenerates timestamps or audit observations. A published-but-unjournaled identical receipt rolls forward; a conflict preserves all evidence and blocks. No checkpoint-tree rename primitive is present in this version.

`active_run()` validates the immutable receipt pin, strict payload, embedded initial bytes and exact derived target. Before link1, live resumed410 must still match every accepted artifact. Once a production submission intent or link exists, the pending action is reconciliation-only: historical resumed410 artifacts are checked against the embedded snapshot, while proof, READY/comparison, continuous410/at400 and original/diagnostic evidence stay pinned. Current state is checked through the production request/receipt chain. This exception is limited to the three recorded historical `resumed/latest_state` small files; it does not waive other immutable evidence checks.

## Submission, production and completion

Each tick performs one external action or activation transition, under both persistent locks. Submission stores/fsyncs exact argv, clean allowlisted environment, unique token, pins and timestamp before one `sbatch` call. Lost responses, malformed output or restart after intent never authorize automatic retry. Reconcile exact token/owner/UID/time/argv to one job; zero matches remain uncertain and multiple/conflicting matches block. A blank accounting Comment is resolved only from the unique comment token in exact SubmitLine; a nonempty disagreement or duplicate comment options is rejected.

Job observation also requests accounting `WorkDir`. For historical submissions whose script argument is relative, the command identity is normalized lexically against that recorded absolute directory. No current-working-directory or filesystem resolution is used. Missing/relative/ambiguous WorkDir blocks when it is required; an absolute script does not depend on it. A live absolute `scontrol Command` must match the normalized accounting path exactly, so held and subsequently CANCELLED observations retain the same identity. Full SubmitLine/argv, token, UID and submission time remain unchanged; basename or suffix equivalence is never accepted.

Requests remain `STATE/links/NNNN/request.json`; execution receipts remain `receipt_dir/execution_JOBID/link_receipt.json`. `run_dir` is checked against activation on every link. Initial predecessor is the unchanged validation READY receipt and its canonical410 digest. Later predecessors are accepted production receipts; the activation receipt never replaces this version1 predecessor contract. Initial completed epoch events are empty; subsequent events are carried forward exactly.

Ordinary source training keeps fresh allocation input verification, four rank records, exact source/fingerprint/profile, positive progress and complete selected state. Epoch step0 with pending gradients is valid only with all four gradient files and allocation-verified canonical pending-gradient digest. Host checks raw inventory presence/sizes and small hashes; it does not load or rehash tensors. Counts satisfy `rank_batches=ceil(train_rows/4)`, accumulation24, and `epoch*rank_batches+step=global_step*24+pending_micro`. FAILED/TIMEOUT/OOM/requeue/nonzero/missing receipt/no progress cannot create another link.

Exact4099 acceptance uses the actual active run, including `/work/<active_run>/step_04099`, never the protected original path. It requires normal parent+batch exit, final inventory/model bindings, fixed-step metadata, exact PILOT_FINAL/PRUNE_EXEMPT, nonempty hash-bound DONE, and all actual completed epochs' finite committed evaluations and epoch_end records. Expected final completed epochs are `(4099*24-1)//rank_batches`; no unfinished final-epoch evaluation is invented. Status/completion expose `active_run` and the activation receipt pin. Old `original_run` still means failed original200.

Only final verification permits idempotent Nice0 restoration for matching owned remaining B5/B6 jobs. B6 remains held and a separate provenance/replacement-plan follow-up block is recorded. No historical launcher is released or redirected, and no failure marker is removed.

```sh
python3 continue_c2_history.py init --config CONFIG_V2.json --state-dir /absolute/WORK/ops/NEW_OPERATION
python3 continue_c2_history.py status --state-dir /absolute/WORK/ops/NEW_OPERATION
python3 continue_c2_history.py tick --state-dir /absolute/WORK/ops/NEW_OPERATION
python3 continue_c2_history.py reconcile --state-dir /absolute/WORK/ops/NEW_OPERATION
python3 continue_c2_history.py watch --state-dir /absolute/WORK/ops/NEW_OPERATION --poll-seconds 60
```

`status` reads; `watch` exits on an explicit block for inspection. Restarting preserves pending intent. There is no force/reset/delete/claim-replacement/automatic-migration command.

## Verification and limits

Version1 had43 passing tests and two independently reviewed P2 fixes. Its directory-promotion route was unusable on the tested shared Lustre: parent-native absent-target RENAME_NOREPLACE returned EINVAL with sources intact. Version2 removes that route rather than weakening it. Submitted-script manifest coverage remains enforced; the no-replacement integrity regression now exercises exclusive immutable receipt publication, including a destination created at the publication boundary.

Version2 test-first evidence: exclusive-receipt tests first failed because the publisher was absent; legacy-version rejection failed because version1 was accepted; activation preservation failed because the workflow was absent; and protected-run operation placement failed because controls could be created there. These regressions pass after their implementations. Existing non-promotion gates were retained; obsolete rename-layout tests were replaced by immutable-publication crash, preservation, conflict and concurrency tests. One test-fixture restoration issue was corrected to restore exact receipt bytes rather than reserialize JSON with different spacing.

The full owned suite passed **52 tests in126.309s, zero skips**. It covers activation without any native rename, preserved original/evidence paths/bytes/inodes, exclusive publication conflicts, recovery after final publication and before journal commit, stray temporary files, audit-only cross-client device observations, embedded-byte/path/owner forgery, pending-intent reconciliation after the active state changes, and first→second→final production at a nested active path. Existing science, storage, metadata/source/helper/script drift, held-owner/state, concurrency, abnormal-stop, pending-gradient, final-marker/epoch and Nice0 gates remain covered.

Python3.6 grammar/compile and stdlib-only import checks pass. The producer author separately reports **3 real cross-helper tests passing**: actual tiny allocated checkpoint verification/postflight for410→420, controller acceptance and request2, and actual producer final4099 receipt accepted at the nested active path. Forged original-path final and post-emission metadata drift are rejected. No request/receipt schema or producer/training-source change was required by activation.

Local fixtures and cross-helper tests do not satisfy the actual four-GPU scientific proof or validation400. Independent review, immutable successor staging, actual host3.6/shared-filesystem flock/exclusive-link/fsync checks, and all live scientific/accounting/storage gates remain deployment prerequisites. No remote action, job submission, activation, cleanup or daemon launch was performed by this implementation task.

Deployment accounting correction: parent evidence showed that old3345920's active Command was absolute while SubmitLine ended in `revision/scripts/b6/wp5_pilot.sbatch`; accounting WorkDir was `/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a`. Five actual-shaped parser regressions first failed (eight failure reports including WorkDir subcases), then passed after the narrow normalization fix. They cover held→CANCELLED identity through the existing retirement gates, missing/relative/ambiguous WorkDir, live-path disagreement, and absolute/relative lexical normalization without changing argv. The focused new-parser/retirement/accounting set passed **10 tests in7.943s**, zero skips. Python3.6 grammar/compile, stdlib-only import and whitespace checks also passed. This isolated parser correction did not trigger a redundant full-suite rerun.
