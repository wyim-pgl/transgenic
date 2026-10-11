# C2 compile guard recovery — 2026-10-11

Candidate status: CPU regressions pass; the full GPU specialization matrix and four-rank checkpoint recovery
still require validation before production deployment.

## Measured cause

Production233574a C2 job3345919 failed on rank3 at attempted update280, input shape(1,129024), compilation
event3 versus limit2. Its update200 checkpoint remains intact. This was a guard exception, not an OOM.

Unchanged production compiler/model code in NGC26.08 was replayed on GPU in job3360978. The190 inputs were
rank3 draws1–72,3913–3936,6625–6718, preserving accumulation-window positions. All augmentation hashes
matched and all190 forward/backward calls completed; peak allocated memory38.341572GiB.

| Original draw | Corpus row | DNA length | Label length | Cumulative graphs |
|---:|---:|---:|---:|---:|
| 1 | 91492 | 30720 | 33 | 1 |
| 3 | 146698 | 129024 | 80 | 2 |
| 7 | 117991 | 61440 | 1011 | 3 |
| 3915 | 10431 | 129024 | 4501 | 4 |
| 6718 | 6195 | 129024 | 3924 | 5 |

Replaying these measured graph observations through the original CompileGuard reproduces the exact production
exception at update280. Inductor's MixOrderReduction.can_fuse guards (installed scheduler.py:395,415) are
`1536*labels.size()[1] < 5242880` and `4096 <= labels.size()[1]`: there are three observed label intervals,
up to3413,3414–4095,and4096upwards. The previous two-event assumption was false.

Raw evidence resides in the primary checkout at
`notes/checkpoint_archive_20261009/c2_failure_3345919/3360978/` and
`real_guard_repro_red.log`. Original remote report:
`/work/nvme/bilv/wcyim/transgenic/runs/c2_recompile_diag/3360978/replay.json`.

## Bounded policy and compatibility

The candidate allows three events per shape and18 total/backend entries, the capacity of three DNA tiers,
three label intervals, and training/evaluation. This capacity remains a hypothesis until the complete GPU
matrix passes. Fourth per-shape events, event19 globally, increasing graph deltas, counter resets and enabled
error suppression still fail. Static DNA, dynamic labels, nonpartitioned DDP, fatal fallback, and memory limits
are unchanged. Model, inputs, losses, optimizer, schedule, initialization and RNG operations are unchanged.

Identity compatibility permits only the exact directional inductor policy16/2→18/3. The other52 required
fields remain strict. Reverse upgrades, other backends, automatic dynamic DNA, partitioned DDP, and other limits
are refused. Original run/checkpoint identities are preserved; every resumed link records the transition and
this evidence reference. Subsequent checkpoints carry the new identity.

## Validation ledger

- Real GPU baseline:190 samples complete, augmentation hashes match, noOOM; exact old guard failure reproduced.
- Candidate focused CPU tests:34 passed before the added counter-reset/subsequent-checkpoint cases.
- Finalization fix3f81fb6: NGC CPU57 run,56passed,1skipped; collective rank1 failure, immutable publication retry,
  corrupted publications and launcher failures covered. Independent review found no important defects.
- Pending: one live guard/cache across real190 plus all DNA/label intervals and train→eval(both views)→train;
  reversed interval order; exact real replay loss comparison; actual four-rank update200 checkpoint resume.

PyTorch's [dynamic-shape documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)
describes guarded specialization. The diagnosis above uses the installed version and measured inputs.
