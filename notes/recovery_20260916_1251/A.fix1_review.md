# Track A fix-round-1 independent scoped review

Reviewed worktree `/tmp/b6-p2-readiness-1535112`, verified HEAD `ac8ac2d34c3ae403017a4a0d72e7531528926626` (implementation `f6b171ffc3f2216504fb385031ad588cb1f797dd`), against `709ff26ff7f69be42f028a45b6c28ccbb5d68b60`.

Read the complete original `A.independent_review.md`, supplied `A.fix1.diff`, and worktree `notes/TRACK_A_RECOVERY_REPORT.md`, including round 1. Queried graft before additional repository source inspection. This is a fix-only review, not a fresh whole-Track-A or deployment certification.

## A1 disposition — resolved

`revision/scripts/mvp/make_launch_readiness.py:61-72,196-232,235-318,321-353` now records the exact loader inventory `config.json`, `meta.json`, `runtime.json`: digest/size when present, explicit null when absent. This matches the actual inventory read/snapshotted by `src/transgenic/training/weights_init.py:143-286`, including semantic parsing of `config.json`. Checkpoint-directory and canonical weights-file inputs resolve to the same adjacent inventory. Symlinks, dangling links and nonregular entries fail closed.

Prepare includes this inventory in the bindings digest; generate recomputes it before accepting the six bound receipts; check compares the complete live source binding. Consequently addition (including the original invalid-JSON config counterexample), removal and byte changes cannot reuse old preflight identity. Missing legacy inventory is not silently interpreted as absence.

`revision/scripts/mvp/stage_checkpoint.py:104-166` captures the chosen inventory, includes present sidecars in its capacity calculation, copies each against its digest, and compares both source and destination inventories before publication. Fresh destination/partial-directory creation prevents silently inheriting stale files. Presence/absence is recorded in the staging receipt; readiness independently remeasures the staged checkpoint rather than trusting that receipt. The added regression tests cover all three files and mutation kinds, filesystem types, staging presence/absence and copying-time drift.

The stronger checks remain tooling-level protection under the documented trusted-operator/immutable-deployment model. The unchanged trainer consumer does not independently enforce these additive fields; no such stronger claim is made here.

## A2 disposition — resolved

`revision/scripts/mvp/make_launch_readiness.py:54-58,102-106,269-282,335-342` makes the adjacent explicit config mandatory, binds its canonical path and bytes, includes it in tooling identity, and rejects mismatches. `revision/scripts/mvp/run_training.py:89-100` rechecks after warmup and supplies that exact path through `--config_file`, with `--max_restarts 0`, before the script operand. The original parsed trainer argv remains intact after that operand.

Reviewed the installed Accelerate **0.34.2** implementation, not merely the fixture assertions: `.testdeps/accelerate/commands/config/config_args.py:43-72`, `commands/launch.py:972-1124`, and `utils/launch.py:184-272`. Explicit config loading does not merge the ambient default config. The single-node environment preparation sets the requested CUDA visibility, bf16 and Dynamo NO.

Ran one focused CPU parser/environment reproduction, not a test-suite rerun: loaded the shipped config using the exact launcher prefix with a conflicting real temporary default config (GPU 2,3; two machines; other address/port/backend; Dynamo EAGER). Only bf16 capability and port-availability checks were substituted. No launcher subprocess or GPU computation ran. Observed:

```text
num_processes=4 num_machines=1 machine_rank=0
GPU IDs=0,1,2,3 mixed_precision=bf16 dynamo_backend=NO
main_process_ip=127.0.0.1 main_process_port=29500 rdzv_backend=static
max_restarts=0 nproc_per_node=4 master_port=29500
CUDA_VISIBLE_DEVICES=0,1,2,3
ACCELERATE_MIXED_PRECISION=bf16 ACCELERATE_DYNAMO_BACKEND=NO
training_script_args=['--seed', '1'] (preserved)
```

Parser nuance, not a blocking regression: effective `same_network` is **false**, despite `true` in the JSON, because this installed parser initializes the boolean rather than replacing it from config. This does not affect the fixed one-machine branch. That branch also uses torchrun's default master address, independently verified here as `127.0.0.1`, rather than forwarding `main_process_ip`. The required effective static single-node loopback rendezvous still holds. Do not infer that every JSON key overrides an argparse default or that this validates another Accelerate/SIF version.

## New Critical / Important findings

**None identified in the fix-only scope.** No new finding requiring severity/file-line/reproduction remediation was established. The parser nuance above is informational, not an operational gate failure for the fixed one-node recipe.

## Contract checks and verification boundaries

- Verified `git diff 709ff26ff7f69be42f028a45b6c28ccbb5d68b60 HEAD -- src/transgenic train` is empty. Exact schema-1 consumer (`mvp_cli_runtime.py:175-236`), trainer bound-forward compilation/no-eager-fallback and hardware gates remain unchanged.
- Reviewed all supplied new tests, including full trainer-argv parser comparison and post-warmup config mutation rejection. Accepted the reported 219 scoped tests in 480.62 s and controller's independent 66 tooling/review tests in 39.92 s plus clean diff check as supplied evidence; did not repeat those suites.
- Existing readiness/preflight artifacts require regeneration because tooling and additive binding identity changed.
- Real Hyena/tokenizer/GPU probe programs and genuine receipts remain absent; actual pinned SIF compatibility is unverified. These are incomplete deployment/readiness gates, not newly introduced defects in these fixes. Checkpoint selection/pins, transfer/quota, immutable deployment, real measurements and author budgets/submission approval still precede any launch.

## Partial-scope verdict

**APPROVE A1/A2 fixes within this CPU/static, fix-only scope. Operational Track A remains incomplete and GPU launch readiness remains BLOCKED.** This note grants no real-GPU claim or launch authorization.

No source edits, SSH, jobs, subagents, merge or push. Only this requested review note was written in main; the focused reproduction used temporary CPU-only data. Graft reported approximately 975,104 tokens saved across context queries (estimate, not a correctness metric).
