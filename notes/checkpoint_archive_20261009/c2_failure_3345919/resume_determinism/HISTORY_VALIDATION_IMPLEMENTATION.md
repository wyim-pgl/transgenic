# Bounded checkpoint400 validation implementation

Production source remains **188bf3e9f6afcc6389b54c123d5bf533a90384cf** in `/tmp/b6-c2-recompile`. This change is confined to diagnostic helpers/tests, the new batch artifact, and this report on main. Initial proof3369338 and its immutable v1 helper deployment are untouched. No remote writes, submissions, source-branch changes, checkpoint edits, or production promotion were performed.

## Acceptance and API

`compare_pair(root, *, profile='resume210')` retains the initial200→220 /210→220 behavior. `validate400` uses200→410 /400→410, snapshot`at400`, exact trace ranges4801..9840 and9601..9840, and accumulation boundaries through9840. Both final histories must retain the observed rank3 graph event at epoch0/batch_index6717, optimizer_step279, input_ids[1,129024], labels[1,3924]; rank3 must have a positive graph delta. The original rank0 micro5040 scalar3.2567708492279053 remains mandatory. Existing bootstrap, lifecycle, rank, context, shared-cache, history-schema, model/state, input, RNG, gradient and snapshot receipt checks remain active.

`run --snapshot-step` accepts210(default) or400 and requires that snapshot precede stop. Programmatic legacy args lacking this field retain210.

`validate_prior_proof(report)` requires exact boolean successes, an empty model mismatch list, exactly six true optimizer/scheduler/RNG states, exactly four rank maps containing eight empty mismatch lists, and the exact historical210 expected/actual/equal fields. Immutable v1 reports without a profile are accepted; missing/string/false/truncated evidence and a validate400 predecessor are rejected.

`check-prior --root PATH --job-id DIGITS` additionally verifies the allocation record, final220 and snapshot210 metadata/history bindings, equal final history hashes, and the pinned original checkpoint/bootstrap receipt. It does not load model or optimizer tensors. A CLI regression succeeds even with all fixture model files removed.

Reports add `checkpoint_digests` for continuous/resumed canonical model/state value trees, calculated from `probe.checkpoint` results already in memory, and `checkpoint_history_sha256` from the history hashes already checked. No second model/state read is introduced. Successful validate400 reports add the required validation400 summary.

## Controller-approved receipt extension

The controller explicitly approved this extension after the original plan: `compare --profile validate400 --source-commit SHA --predecessor-root PATH --predecessor-job-id DIGITS`. The validation CLI requires these identity arguments and emits `validation_receipt.json` only after complete acceptance. The supplied full source commit must match `/work/repo/SOURCE_COMMIT`; the current SLURM job must match this allocation's verified_record, and the predecessor must pass the strict initial-proof checker.

Receipt version1/kind`c2_history_validation_ready` binds source_commit, validation/predecessor job IDs, predecessor comparison SHA, current comparison SHA and pinned original receipt SHA. Its candidate selects relative_run`resumed`/relative_checkpoint`latest_state`, epoch0/step9840/global_step410/pending_micro0, metadata SHA, history SHA and canonical checkpoint SHA. Receipt creation also rejects history substitution after comparison. Continuous410 and at400 are retained. The receipt is created exclusively, without overwriting prior evidence.

No new original checkpoint inode/device stat capture was added: the controller excluded it because cross-node device identity is unstable and the allocation already verifies original payload hashes under a read-only bind.

## Batch behavior

`history_validation.sbatch` requires REPO_DIR, OPS_DIR and PROOF_JOB_ID, preserves four GPUs/64 CPUs/ghx4-interactive/two hours, immutable source/helper checksums, read-only original/input/predecessor binds, and the operational shared cache namespace. It requires exact `sacct -X` COMPLETED0:0 and the strict predecessor check before cloning/training. The existing storage_preflight call and its100GiB minimum are retained without bypass.

It uses `/work/runs/c2_compile_history_validation/$SLURM_JOB_ID`, performs fresh allocation input verification once, clones original200, builds its explicit manifest, trains200→410 while cloning400, then launches a new process from the400 clone to410 without bootstrap. It compares validate400, publishes the receipt, rehashes source/helpers, and emits HISTORY_VALIDATION_PASS only after successful exit. It performs no production or dependency-submission action.

## Verification

TDD RED:14 tests ran in22.997s, with the seven original regressions passing and seven new missing-profile/prior-validator/snapshot/check-prior acceptance tests failing as expected. First GREEN:14 tests passed in27.168s.

Additional canonical-state digest test: RED1 failure in1.883s; GREEN1 pass in2.338s. Receipt identity/proof test: RED1 failure; GREEN1 pass in5.383s. Self-review history-substitution mutation: RED1 test with two failures in6.390s; GREEN1 pass in6.210s. A controlled local shell regression confirms a PENDING predecessor is rejected before storage or training execution.

Final command:

```bash
PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism -p test_history_proof.py -v
```

Final suite result: **17 tests passed in42.542s (exit0)**, including all seven initial regressions. `bash -n history_validation.sbatch`, Python py_compile of helper/tests, and git diff --check passed. The controller owns independent review, graph refresh, immutable v2 deployment, allocated NGC/four-GPU verification, submission journal and push. Scientific equality, runtime feasibility within two hours, and actual recorded slot6718 evidence remain allocated-job gates; local fixtures do not claim those gates passed.

Helper bytes at97c0408 (historical snapshot; superseded by compatibility follow-up below):

- `history_proof.py` SHA256`2829645e6bd9d1dd62d9112b89fd6ee3423dd6533e11090345820536613a2e35`
- `test_history_proof.py` SHA256`7e29f37001032eb15bb3b72a090ac7ef12097fcaf1aa1d765d34ff6e39793bc2`
- `history_validation.sbatch` SHA256`dce7af8a1da97ee9dc9989dbdebb68765b072e53c22dfefaf09c0928b45e359b`

## Controller-approved host Python3.6 interoperability follow-up

The controller additionally authorized a narrow change to probe.py: dataclasses is imported only inside its training run function, and history_proof.py likewise imports it only inside run. The allocated training closures retain the same dataclasses.replace behavior. Graft's exhaustive indexed search found no probe.dataclasses caller; its broader dataclasses search confirmed the two helper uses occur only in run.bounded.

A fresh `python -S` subprocess parses the actual history_proof.py and probe.py using `ast.parse(feature_version=(3,6))`, forbids imports of torch/numpy/accelerate/dataclasses, imports the real helper, accepts a complete prior report and rejects missing/string success evidence. Its final sys.modules check confirms none of the forbidden modules was loaded. This verifies Python3.6 grammar and the stdlib-only validator import path; it does not claim execution under a local Python3.6 interpreter or extend host support to check_prior/training paths.

RED: the new compatibility test failed in0.137s with blocked module import:dataclasses. GREEN: the new test and all three relevant prior-proof regressions passed (4 tests in11.044s, exit0):

```bash
PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism -p test_history_proof.py -k prior -v
```

Fresh py_compile of helper/probe/tests and git diff --check passed. The full17-test suite was not repeated for this isolated import relocation, per controller instruction; the new compatibility test brings the suite to18 tests. No immutable remote v1 bytes were changed.

Updated immutable v2 helper bytes:

- `history_proof.py` SHA256`4ab97872c850d22fc8398de99bac4acf9eda5a898979150ba0f8452f95718ec4`
- `probe.py` SHA256`9edf1eb412275f998cd0614d5b0ea12203433c52b0192da557aa860ab9ab65dc`
- `test_history_proof.py` SHA256`2409784293d524efa0421487618e9266aefdd9e230ffb4a37dc2b2c2394138f7`
- `history_validation.sbatch` SHA256`dce7af8a1da97ee9dc9989dbdebb68765b072e53c22dfefaf09c0928b45e359b`

## Independent-review receipt repair

Review reproduced two false READY cases after declaration hashes were recomputed: unequal current continuous/resumed history, and a substituted original manifest consistently rebound across all three journals. The publisher now reasserts actual current history equality and calls `_pinned_manifest(root)` on the current validation root before publishing. No extra tensor reads were added. Regressions update artifact, metadata, report and comparison declarations together.

RED for changed resumed history: receipt was incorrectly created (8.459s). After equality repair, fully rebound manifest RED likewise incorrectly created a receipt (9.994s). After both assertions, complete18-test suite passed56.437s. `git diff --check` passed. Independent scoped re-review approved both fixes with no remaining findings; immutable v1 helpers remain unchanged.
