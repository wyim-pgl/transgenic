# C2 producer/controller contract verification — 2026-10-12

The assembled local interface passes three tests in `test_c2_link_contract.py`. No producer, controller, training source, deployment files, remote state, or jobs were modified by this task.

Each test builds a real temporary workspace with tiny Torch/safetensors model, optimizer, scheduler and four rank RNG files. The existing producer fixture supplies valid identity and compiler descriptors; the contract fixture retains the 16 required reconstructed legacy positions, valid rank descriptors and a pinned bootstrap manifest. Source/helper manifests and an actual temporary Git snapshot are checked by both programs. The test starts at the already-promoted candidate boundary; it does not claim to reproduce the scientific proof or promotion protocol.

The tests call the real `production_link.preflight` and `production_link.postflight`. The returned receipt is checked against the actual emitted `link_receipt.json`, then passed unchanged to real `Controller.validate_link_receipt` and `Controller.accept_link`. Neither acceptance validator is replaced. Rank observation records represent the training boundary using owned fixtures; model training itself is not executed. Only external Slurm and quota behavior use the existing fake scheduler adapter.

| Case | Producer evidence | Actual controller outcome |
| --- | --- | --- |
| Clean preemption, 410→420 | Fresh four-rank records, retained compiler history and draw prefix, complete finite checkpoint; real postflight emits LINK_CHECKPOINT_READY | Validator returns false for done; accept_link saves `link_accepted`; a new Controller instance reads that phase and the exact emitted receipt SHA |
| Exact final 4099 | Real finite final safetensors, metadata, identity/history, exact PILOT_FINAL/PRUNE_EXEMPT, nonempty DONE, all rank results; real postflight emits C2_PRODUCTION_FINAL_READY | Validator returns true; accept_link saves `final_verified`; reopened state agrees. Fixture sampler count intentionally leaves the final epoch incomplete, so zero completed-epoch evaluations is correct |
| Corruption after emission | A valid real preemption receipt is emitted first; checkpoint metadata is then changed | Both the actual validator and accept_link reject. Journal bytes remain identical and phase remains `waiting_production` |

Command:

```bash
PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile \
/home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover \
-s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism \
-p test_c2_link_contract.py -v
```

Observed result: **3 tests PASS in 13.538 seconds**. `py_compile test_c2_link_contract.py` also passed. This is new integration coverage against existing implementations, so no production implementation was changed to make it pass. The deliberate corrupt-artifact branch exercises rejection after a successful producer publication. No producer/controller schema mismatch was observed.

Tested file SHA256 values, recorded after the passing run at main `d332dcf`:

- `production_link.py`: `7ad77707ccc3e2df015d035c74c152a770dc7f350179728dbc32774975225e9c`
- `continue_c2_history.py`: `e56e00e06df23e58616131652a52c2a25ecd28b58e9ce914f0e3082bdc85ab82`

These tests verify receipt interoperability, filesystem validation and durable acceptance. They do not certify NGC runtime equivalence, actual Slurm behavior, storage availability, model numerical continuity, or the pending four-GPU scientific gates. The source bootstrap transport fix and proof job outcomes remain separate evidence. Broader producer/controller unit suites and independent review are recorded in their own implementation reports.
