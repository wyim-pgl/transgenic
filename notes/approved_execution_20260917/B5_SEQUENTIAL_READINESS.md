# B5 sequential readiness — 2026-09-17

## Controller correction — priority versus added prerequisites

User approved preserving seed123 and **MVP diagnostics first, then456→789 sequentially**. They did NOT require seed123 to finish before456 may start. The report's additional seed123-DONE/no-pending requirement below is a conservative scheduling proposal, **not an approved mandatory gate**. Actual release depends on protecting existing123, meeting MVP priority and verified resources/no harmful overlap. Do not ask the user to reapprove their seed decision. The single11h direct-sbatch/environment/retry changes below are also an unreviewed future proposal, not ready-to-run or permission to bypass submit_chain guards. P1 still pending is the current sufficient reason not to submit; controller will assign a bounded numeric tranche only after live accounting/reservations permit it. This does not mean the whole1154GPUh balance has been authorized for new runs.

## Decision / authority

**Preparation complete; NOT released for submission.** Latest user instruction approves preserving seed123, prioritizing urgent MVP diagnostics, then B5 **456 → 789 sequentially**. This supersedes the blanket `deferred / awaiting author choice` text in `.atl/resume.md` and `.atl/todo.md`; it does not approve unlimited GPU spending, immediate submission, changing seed123, or bypassing operational gates. This task performed bounded SSH reads only and writes only this report. No submit/cancel, remote writes, source edits, containers, GPU operations, large hashes, or checkout changes.

Authority read: current resume/todo and quarantine §1n; `revision/protocols/PROTOCOL_B1_frozen_v1.md:935–950` (A45); `.atl/archive/todo_before_20260916.md:197–207` (A45 and author-approved 11-hour transition); `notes/MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md`; latest runtime identity below. The old A45 three × 48h wording is historical, **not current launch argv**. Advisor consultation failed due client/model-version incompatibility; no advisor approval exists.

## Fresh bounded observations

Read through `ssh -o BatchMode=yes pronghorn` → `ssh -o BatchMode=yes dtai`, at **2026-09-17 12:33:51–12:35 CDT (10:33:51–10:35 PDT)**. Remote user `wcyim`, Slurm UID97434, account/QOS `bilv-dtai-gh`, partition `ghx4`.

| Item | Actual observation |
|---|---|
| P1 **3163986** | PENDING/Priority; elapsed0; 30min; 1 node/4GPU/64CPU; Requeue0; no allocated TRES. Source command is `mvp_est8_20260916_1545/staging/final_p1_release/p1.sbatch`. No P1 pass inferred. |
| B5 **3131538** | Now **COMPLETED0:0**, allocation/batch/extern all clean, elapsed10:46:52, limit11h, billing4000/4GPU/64CPU/454240M. Thus the earlier 10:06 PDT “RUNNING” snapshot is stale. |
| B5 next **3149356** | PENDING/Priority, eligible12:24:23 CDT; link4, 11h, no active allocation. |
| B5 **3149357–3149363** | Seven PENDING/Dependency links, each11h. Preserve all eight remaining links. No seed123 cancellation, edits or new submission. |
| Own456/789 queue | Full own queue contains only P1 and the eight seed123 links; no456/789 jobs. |
| Historical456/789 jobs | 3131539–3131544 all CANCELLED by97434, elapsed0, old48h limit, no allocation TRES. These are not successful runs or live work. |
| Run directories | `/work/nvme/bilv/wcyim/transgenic/runs/seed456` and `seed789` both absent (`lexists=false`); corresponding chain ID files absent. Fresh starts are the currently applicable branch. |
| seed123 markers | Neither TRAINING_DONE nor TRAINING_FAILED; no LINK_ACTIVE entry at read. Not scientifically complete merely because3131538 exited0. |
| seed123 checkpoint | best resolves to concrete epoch_06, best eval0.6322135329246521; latest_state metadata epoch6/global_step14049/step38934, seed123, bad_epochs0. Metadata read only; checkpoint tensor completeness not certified here. |

The scheduler displayed future StartTime estimates for both P1 and3149356; estimates are neither reservations nor proof they cannot overlap. **Preserve the existing seed123 chain even while it waits**; do not interpret the boundary as permission to replace/restart it. New456/789 work must not add overlap with seed123 or urgent MVP work.

## Verified source and initialization identity

`repo_seeds` remains **8f049eb3ba72ba7541076fe9fc8f6019b90c0f7f**. Small-file SHA256 comparisons confirmed remote source against inspected local/Git bytes:

| File | SHA256 |
|---|---|
| deploy/deltaai/train_b5.slurm | d901f3d5494b6cdcb2ffe86abf0d76aae140d0b28712ef32370164c614264bd0 |
| deploy/deltaai/submit_chain.sh | c420338e48c8c72773ad97204c481e9d92d318b9c5a8e6e26f4c470389734163 |
| train/train_HyenaTransgenic.py | 68cbefe65228d71e70f047f3cd4277f27a265236929b42af96cc5c45cf86be88 |
| src/transgenic/training/b5_runtime.py | b5a9d7675fec14762cbe1107fc956690e3ae9fb5c9b185b34b4351922d4bc2b9 |
| src/transgenic/model/modeling_HyenaTransgenic.py | 0daa2d9c9ffb4a5531b6c1b5914c48fa1791239af90f34e5000ff6d510082366 |
| configs/b5_400m_win_v3.json | 9b8bef9b642526835a0817e9b13ae50b7c7f274fe882256426cb78f90c080d79 |

Targeted remote `git diff --name-only HEAD` for scripts/trainer/config returned empty. This is not a whole-tree pristine attestation.

Fresh seed123 `runtime_identity.txt` (SHA256 `44b6f578ec50d373d01e764d391b2a559ce4d3012c52388fc160f94fef6f887e`) records NGC26.08, SIF SHA256 `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`, Torch `2.14.0a0+4fdf77b940.nv26.08`, repo8f049eb,4GPU/64CPU, nonreentrant checkpointing, compile on. This is a small existing receipt, **not a new 11GB SIF hash/import**.

B5 uses config `configs/b5_400m_win_v3.json`, original A40 corpus `/work/db/b5_full_a40_v1.db` inside the container, effective batch96 (24 accumulation per rank), patience3, ceiling22epochs, checkpoint every200 optimizer steps. Trainer source `:240–279,373–394,750–786` constructs the original recipe and resolves per-seed resume. `CheckpointLayout.resume_dir` at `b5_runtime.py:187–215` selects the largest global_step among the seed's completed epoch with `accelerate_state` and `latest_state` (or `.prev` if current publication is absent).

- **Fresh456/789 (observed now):** absent run directory means `--resume auto` finds no state. Original frozen B5 model initialization is used: cached pretrained Hyena encoder plus seed-specific newly initialized model components. Same original recipe/runtime, differing seed. No learned seed123 weights are supplied.
- **If the state changes before launch:** an existing directory is not blanket permission to restart. Verify same seed, recipe/db/run_config/runtime identity and a complete own Accelerate state (model/optimizer/scheduler/RNG, metadata and pending gradients when needed). Only then allow same-seed resume. Do not silently accept `resume auto` returning None in an existing partial run; it can otherwise become an unintended fresh start.
- TRAINING_DONE: verify actual stop/best checkpoint/accounting and skip completed seed. TRAINING_FAILED, ambiguous state, foreign seed, unexpected symlink, incomplete state, identity drift or unresolved active marker: STOP and preserve evidence; never remove a failure marker automatically or set RUNTIME_OVERRIDE.
- **Never initialize B5 seed456/789 from seed123 epoch05, epoch06, best or latest_state.** MVP seeds1/2/3 intentionally use pinned seed123 epoch05; that is a different experiment. MVP source b660bf1/executable362eb899 and eval-only9603e20 are not B5 replacement checkouts. New bestepoch06 does not repin MVP.

## Live budget, not historical estimates

Documented command: `.atl/archive/resume_before_20260916.md:367` gives `ssh dtai 'accounts; squeue -u $USER'`. Executed via the required pronghorn hop. **`accounts` reported balance1154 / deposited1499 hours for bilv-dtai-gh** at12:33:51CDT. Existing project documentation identifies these as GPU-hours; observed B5 billing4000 for4GPUs agrees with the established scale. The printed balance is rounded and may lag accounting; retain a margin and requery before spending.

Conservative reservations at this snapshot:

- Eight remaining seed123 links ×11h ×4GPU = **352 GPU-h**. Reserve the whole queued ceiling until links terminate/cancel naturally; do not count pending jobs as already charged or assume early stopping.
- Urgent MVP diagnostic cap remains **5 node-h /20 GPU-h cumulatively**, including prior attempts, startup/staging/hash/teardown. P1's0.5node-h/2GPU-h is within, not additional to, that20. Reserve20 conservatively here pending full diagnostic accounting.
- One proposed new B5 link = **11node-h /44GPU-h maximum**, with default memory (no `--mem` uplift), no scheduler requeue or self-submission.
- Arithmetic headroom before additional MVP training reservations/margin:1154−352−20 = **782GPU-h**. One first link per new seed would consume at most88, leaving694; **that is not a two-seed completion budget**. Seed789's tranche is not queued/reserved as an active job now.

Historical197–246GPU-h/seed used a6.15h/epoch benchmark; later measured first epoch was10h07. Do not claim either predicts full completion within today's balance, especially at22epochs. Full MVP training reservations, other account users' commitments, current charges and a controller-approved **total B5 continuation envelope** are not resolved by `accounts`. Before release record numeric `B5_total_cap`, charged/reserved B5, seed123 reserve, MVP diagnostics/training reserve and margin, with the inequality `new44 <= min(B5_cap_remaining, live_balance − other_remaining_reservations − margin)`. Each later44GPU-h link needs the same check. No implicit top-up or unlimited chain.

## Minimal sequential execution proposal — controller only, not executed

1. One named controller is the sole scheduler/submission/ledger writer. No background auto-submit watcher and no independent agent submissions. Keep current seed123 checkout/chain untouched.
2. **MVP-priority release condition:** actual P1 outputs/accounting are reviewed, urgent next diagnostics have either completed and been reviewed or received an explicit controller disposition that leaves no immediately runnable urgent GPU diagnostic pending; their bounded budget and any near-term MVP training reserve are recorded. Mere P1 pending, queue delay, P1 job exit0 alone or missing receipts does NOT release B5. Full six-gate MVP training readiness is not a new scientific prerequisite for B5; a diagnosed terminal MVP blocker may release B5 only with explicit controller priority/budget disposition, not by silent fallback. This condition is currently false (P1 pending).
3. Also require seed123 genuinely finished at its protocol stop, no owned active/pending seed123 links remaining, and no conflicting urgent MVP allocation/queue reservation. No cross-seed `afterany` shortcut: one seed's last allocated link can finish without completing its training.
4. Refresh own full queue, ownership/accounting/markers, selected-seed run identity, source hashes, quota/headroom and `accounts` immediately before one action. Serialize the observation→submission→job-ID recording transaction. Uncertain query = STOP. Source script's per-seed LINK_ACTIVE guard remains a backup; it does not provide cross-seed serialization or atomic protection from multiple submitters.
5. Release exactly **one11h seed456 link** under the bounded envelope. Reconcile accounting and checkpoint/stop state at termination. If continuation is legitimate, controller may submit one further11h link after fresh gates, never automatically. A trainer failure stops this path. Continue456 to verified TRAINING_DONE, not just one successful link.
6. Only after456's verified completion and no456 jobs remain, refresh MVP priority and budget gates and perform the same process for789. Do not prequeue789 while456 is unfinished.

### Source-verified future command shape (NOT launch-ready)

Run from `/work/nvme/bilv/wcyim/transgenic/repo_dev`, using the frozen script path below. The existing submit_chain helper has **no time override parameter** and defaults to48h through the script; do not run its historical `SEED 3` invocation. Smallest no-source-mutation option is a controller-submitted single link retaining A45 presubmitted mode (`CHAIN_MAX=1`), with all external preflight guards above. This deliberately avoids prequeueing unbounded successors under the new priority/budget instruction.

Explicit environment: `SCRATCH=/work/nvme/bilv/wcyim`, `SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif`, `REPO_DIR=/work/nvme/bilv/wcyim/transgenic/repo_seeds`, `SAVE_EVERY=200`, `WANDB_MODE=offline`. Use a controller-reviewed clean submission environment: no inherited `TRANSGENIC_*` debug overrides, RUNTIME_OVERRIDE, SYNC_DEST, stale EXCLUDE_NODES, benchmark flags or unintended SBATCH_* overrides. `CHAIN_PRESUBMITTED=1` blocks self-sbatch; set `LAUNCH_TRIES_MAX=1` to disable the script's default in-allocation second launch (LAUNCH_RETRY=0 alone does not disable it).

Exact prospective seed456 scheduler argv, only after every release gate and environment review:

```text
sbatch --parsable --no-requeue --account=bilv-dtai-gh --partition=ghx4
  --nodes=1 --gpus-per-node=4 --cpus-per-task=64 --time=11:00:00
  --exclude=gh062,gh093 --comment=seed456
  --export=ALL,SEED=456,GPUS=4,CHAIN_N=1,CHAIN_MAX=1,CHAIN_PRESUBMITTED=1,LAUNCH_RETRY=0,LAUNCH_TRIES_MAX=1
  /work/nvme/bilv/wcyim/transgenic/repo_seeds/deploy/deltaai/train_b5.slurm
```

No dependency is needed because the controller waits for verified release and quiescence before submission. No `--mem` override. Keep inherited script signal `B:USR1@900`. Fresh node incidents may require a reviewed extension of exclusions. For789, change only seed/comment after456 completion and all fresh gates. For same-seed continuation retain the same run path/config and `--resume auto`; record the controller's continuation ordinal separately (single-link CHAIN_N/MAX remain1). Record returned job ID immediately; no blind resubmit on a lost/ambiguous receipt. Guards in submit_chain are **not executed by this direct argv** and must be fulfilled by the sole controller before submission. Script may cancel same-seed successors on done/failure, but none are permitted in this one-link plan.

## Blockers and next safe action

1. P1 remains pending; MVP-priority release is absent.
2. seed123 is incomplete and still has eight preserved queued links; lack of a currently running allocation is not quiescence for a new seed.
3. Total B5 continuation cap and MVP training/other-account reservations are not yet bound;1154 balance is capacity evidence, not spending authorization.
4. Fresh launch-time source/run/queue/marker/quota checks and controller environment review remain necessary. Current456/789 absence supports fresh initialization, not future assumptions.

Safe immediate commands remain read-only: the same nested SSH `accounts`, full-own `squeue`, targeted `sacct` and `scontrol show job`; bounded small metadata/source reads. No submission is warranted by this report. No need to ask again whether456/789 are desired: that choice is approved; resolve these concrete execution gates instead.

Graft consulted before source exploration; reported token savings across calls:1,333,400 (tool estimates, not execution evidence).
