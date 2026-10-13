# C2 activation and producer/controller contract verification — 2026-10-12

Current scope: controller configuration/journal version 2 and in-place activation from `ACTIVATION_DESIGN_20261012.md`. This replaces the earlier canonical/backup/candidate_promoted fixture. Only `test_c2_link_contract.py` and this report change; producer, controller, training source, deployment files, remote state and jobs are unchanged by this task.

## Actual mechanisms exercised

The contract fixture reuses the controller's scientific report/scheduler fixtures but fills the proof and validation checkpoint trees with real tiny Torch/safetensors model, AdamW, scheduler and four rank RNG states. Every rank has four valid 16-byte Philox device states. Compiler histories pass the real source schema validator and contain complete descriptors for the required 16 legacy positions plus validation's observed slot 6718. Real canonical checkpoint digests and sidecar hashes are computed before scientific fixture acceptance. This is a synthetic science-input boundary, not a claim that the real four-GPU proof passed.

The tests invoke actual `accept_science`, obsolete-job terminal-state fixture setup, `activation_prepare`, and `activate`. There is no synthetic candidate_promoted state, backup path, moved checkpoint tree, or manufactured activation receipt. The actual activation mechanism derives `evidence/validation/9001/resumed`, publishes the immutable receipt and reaches `candidate_activated`.

Production requests come from actual `Controller.build_request`. Each allocation fixture runs actual `production_link.preflight` and `production_link.postflight`, checks emitted `link_receipt.json` against the producer's return, and feeds that receipt to actual `Controller.validate_link_receipt` and `accept_link`. No production receipt or acceptance result is hand-built; no validator is replaced. Rank observation records simulate the training boundary using owned tensor fixtures. Training itself is not executed. Only external Slurm/quota and scientific observations use fixtures.

## Coverage

| Case | Required behavior exercised |
| --- | --- |
| Activation and two production links | Activation preserves the active directory's complete existing bytes and inodes. Request 1 binds the original validation READY and real 410 checkpoint; producer emits a positive-progress 420 preemption receipt; controller persists link_accepted. Request 2 binds that receipt and current 420 state, then producer emits exact 4099 final evidence at the same nested active path. Controller persists final_verified and a reopened instance resolves the same active path. |
| Forged old original-path final | A genuine final receipt is emitted first, then only result.pilot_final is corrupted to `/work/runs/c2/step_04099`. Both real controller validation and acceptance reject it; journal bytes and waiting_production phase remain unchanged. |
| Checkpoint corruption after emission | A genuine preemption receipt is emitted first, then current checkpoint metadata is changed. Both real controller validation and acceptance reject; journal bytes remain unchanged. |

Every case snapshots the preserved original200/failure marker, diagnostic256, proof tree, validation continuous410 and at400 trees, comparison and READY bytes/inodes before activation. These snapshots remain identical after activation and after each successful or refused production boundary. The activation receipt pin and bytes remain unchanged as active latest_state evolves. No final checkpoint appears under the protected original path. This verifies the narrow transition from historical resumed410 evidence to mutable active training state without pretending that later latest_state still describes410.

The final fixture intentionally has zero completed epochs for its sampler size, so no final partial-epoch evaluation is required. Existing finite completed-epoch behavior remains covered in the producer/controller suites; this integration task tests path ownership and receipt continuity.

## Validation

Command:

```bash
PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover \
-s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism \
-p test_c2_link_contract.py -v
```

The first v2 run passed the full positive two-link workflow. Both negative cases also raised the expected controller refusal, but the test harness had separately imported the controller twice, giving its exception classes different identities. The fixture now consistently uses the same real controller module as the controller fixture; no implementation or validator was changed.

Latest result: **3 tests PASS in 42.091 seconds**. `py_compile test_c2_link_contract.py` and scoped `git diff --check` pass. No controller/producer interface mismatch was observed. File pins recorded after the passing run at main `7e40971` (controller v2 was still in its author-owned working tree):

- `production_link.py`: `5da60c8f994a7814ba2d0ac337f7f8320e59b887612bc6ec0622059e427d88c9`
- `continue_c2_history.py`: `25046d6dcafe6789041532591f91ca31cd818ec6c0c7cf9152a43ab39af7b74f`

Final deployment pins and any changes from the controller's own remaining review must be assessed against the committed reviewed version; this task does not stage or alter that author's files.

Earlier historical checks: initial single-link producer/controller interface passed 3 tests in 13.538 seconds; after the approved compiled-state/CUDA RNG corrections at producer7266e14, those tests passed again in 15.978 seconds. The current v2 workflow supersedes their synthetic promotion setup.

These checks establish local activation/path/receipt interoperability and durable acceptance, not NGC runtime equivalence, real Slurm outcomes, storage availability, numerical resume equality or the pending scientific gates. Protected original200 remains at its original path; eventual production output must be reported from the activation-bound validation/resumed path.
