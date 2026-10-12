# Seed123 GPU source-freeze and deployment review plan

## Status and scope

Local candidate preparation complete. NOT a deployable release, controller authorization, runtime binding, six-gate pass or measured budget. Only this new `notes/auth_resume_20260917/gpu/` directory was intentionally written. No SSH, submit, cancel, remote write, source edit, reset, commit or GPU operation. This is authenticated continuation, not crash recovery. Advisor consultation failed with client/model-version incompatibility; no advisor approval.

Authoritative source remains `/tmp/mvp-seed123-integration-20260917`, base `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558` plus preserved dirty changes. Current resume/todo, FINAL_CHECKPOINT, TRAIN_REVIEW, runtime repin and both previous P1 plans were read. Current instruction supersedes old three-seed and456/789 instructions. Preserve all B5 seed123 jobs. Baseline `b5_seed123`/`b5_baseline` is distinct from `mvp_seed123`/`mvp_finetune`; same seed does not mean same trained weights.

Controller-supplied observation, not this worker's live query: 2026-09-17 18:01 CDT, P1 **3163986 PENDING/Priority, elapsed0**, B5 **3149356 Priority**, remaining B5 links Dependency. No fresh staging/quota/accounting fact is inferred from restored authentication.

## Prepared local artifacts and exact closure

`package_recipe.py` made `candidate/` using selected **working-tree bytes**, not `git archive HEAD`. No dirty change is attributed to base9603e20. Re-running refuses the existing candidate directory. Do not remove it to regenerate; a later revision needs a new reviewed directory.

- `code.candidate.tar`: **103 regular source/config/panel files**, SHA256 `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7`.
- `source-manifest.json`: exact exhaustive relative path/hash inventory and provenance, SHA256 `4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a`.
- `candidate.sha256`: full artifact ledger, SHA256 `d1ae4155fdd31fcb30e8d96f046bf40734b4bad3d65535df2766ba55fc9ae48a`.
- `expected-project-files.json`: **65** expected `/work/repo/` project files for comparison, NOT a runtime consumed-code manifest. No external installed-package measurement was invented.
- `model-config.candidate.json`: byte-identical `configs/b5_400m_win_v3.json`, SHA256 `9b8bef9b642526835a0817e9b13ae50b7c7f274fe882256426cb78f90c080d79`. Do not rewrite its historical B5 seed metadata to enforce MVP profile. The MVP profile is explicit CLI/prepared binding; fixed MVP250 updates/12 warmup/four ranks/effective96 remain source-enforced.
- `binding.candidate.json`: declarative seed/profile/role/input recipe, explicitly not prepared bindings.
- `budget.candidate.json`: incomplete schema2 REQUEST with null unapproved/unmeasured fields. Intentionally cannot pass `validate_budget`; never feed it to a probe or replace nulls with guessed values.
- Three original accepted candidate companions copied unchanged under `candidate/notes/`; `existing-script-hashes.json` records exact old deployer/followup/script/authority hashes. `source-status.txt` records the dirty source status.

Complete packaging inclusion rule: all tracked and nonignored untracked regular files under `src/`, `train/`, `revision/scripts/mvp/`, `configs/`, plus `revision/protocols/b6_panels_v1.tsv`, `revision/scripts/07_selfconsistency_stats.py`, `revision/scripts/24_restrict_reference_to_test_genes.py`. No tests, tasknotes, .git, graph, environments, pycache, datasets or model tensors. The two producer helpers must not be lost merely because P1 does not call them. Candidate is an archive review snapshot; it does not bundle a runnable new deployment wrapper.

New untracked dependencies explicitly included:

1. `src/transgenic/training/mvp_profiles.py`: integer123/development-single contract, imported by readiness, CLI/runtime, probes, resource authorization, training and evaluation.
2. `revision/scripts/mvp/mvp_panel_compatibility.py`: development panel baseline/fine-tune compatibility.
3. `revision/scripts/mvp/mvp_verify_pinned_panel.py`: independent request/helper pin caller. Pin/package it separately even though not part of numerical training consumed manifest or DECODER_ONLY.

The training consumer's closure is every `src/transgenic/**/*.py` plus `train/train_HyenaTransgenic_MVP.py` (`mvp_cli_runtime.py:146-151`). Full release closure is wider. Seven tooling bindings remain readiness, stage_checkpoint, run_bounded, run_training, accelerate_mvp.json, real_probes and resource_probe. Decoder-only closure additionally includes panel preparation/verification/compatibility, decoder runtime/orchestration/producer, bootstrap/windows/tracks, constrained decoding and GSF helpers (`mvp_decode_runtime.py:27-43`). Do not substitute a seven-file/tool-only manifest for the full source closure.

## Reviewed runtime handshake

TRAIN_REVIEW grants bounded software approval only. Current reviewed pins:

| File under revision/scripts/mvp/ | SHA256 |
| --- | --- |
| real_probes.py | 7f9507f4fced687e1886c659224d4b778b3486a2b390dac1e8f3c6954e46c7ae |
| make_launch_readiness.py | a5b2c895bd76699718655993d67430b9459c5db07d5f47636feb6acc77ce6de2 |
| run_bounded.py | 8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932 |
| mvp_decode_runtime.py | b0ffac46cff61f23262f5f8455a2217f823278b7b0225f4b9a445f8957db7d03 |

Recovery repin changed two PROBE_CODE pins/comments, not runtime exemptions/checks. Historical35PASS/1skip and TRAIN_REVIEW277PASS/22deselected plus one isolated CPU/Gloo success are software evidence, not this task's rerun or GPU evidence. Freeze the entire reviewed closure and calculate new source-comparison/decoder operator pins with the existing consumer algorithm when adapting deployment. Old b660 aggregate pins do not describe this candidate. No further automatic pin rewrite is appropriate.

## Existing immutable release, input receipts and reuse limits

Retain `notes/mvp_execution_20260916/final_p1/release/` unchanged:

- Old source b660bf1, code.tar `0fb803cf0612623fa0023bb1f0f28b4b2f6a227a930833f50f74cd31a64e9cd2`.
- Old release.json `7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689`; ledger `5b36bdf09a8b8eb548147654e23d6ad59fe18542642a05eb720f22be102fdfb8`.
- Old deploy.py `49922d81bc11eb17d651f0ea4132f0d1a8e6bcc3299dcc026d8bf6182c8b8cc6`; p1.sbatch `950e2a54e40e8d84471b8f0907b1fb168b967cb283d1a1107f040d7d0b4f035f`.
- Old P1 deployment binds **MVPseed1**, although initialization is B5seed123. Neither changing a JSON label nor replacing its source under the old namespace yields seed123 receipts.

Unchanged input identities, recorded in `binding.candidate.json`:

| Input | SHA256 |
| --- | --- |
| SIF,11120398336B | 1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6 |
| corpus,23534252032B | 6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222 |
| evidence,1068773376B | ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d |
| epoch05 weights | 6e5e408681dd5356548a8f4b93d28e561c22d885668a4e0015d3e314e28f8d22 |
| parent run_config | c3ca0b4dfc1c9fbedb0cfba4b42887b22d4a62bc5b0c84be6b711ec630a1ef18 |
| parent runtime_identity | 44b6f578ec50d373d01e764d391b2a559ce4d3012c52388fc160f94fef6f887e |
| epoch05 meta | 844ecbc41f19a9287eb0101f9047b3b0100f88d2bfed1a5057261296d86943d7 |
| epoch05 eval | 0df4e4a7350f3e1cec0884f835aa94f148cbf5042998c25d296e424c5554df4f |

Acceptance `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e`, inventory `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98`, receipts `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac`, panel `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`. Companion provenance remains separate from source Git commit.

Existing evidence transfer receipt records the file at namespace `staging/evidence/fullc0lite.db` on09-16, not current existence. Source root `/work/nvme/bilv/wcyim/transgenic/runs/seed123`; concrete `epoch_05` never best/latest/epoch06. Exact epoch inventory is weights/eval/meta; epoch config/runtime/runtime_identity and parent config absent. Staged optional sidecars config=None/runtime=None/meta132B. No optimizer/RNG resume.

Pinned HF commit `0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446` in BOTH tokenizer and hub encoder layouts under `repo_seeds/HFmodels`. Existing reviewed `copy_cache` verifies exact3/4 snapshot inventories and source-before/source-after/destination hashes, materializes private nonsymlink bytes and freezes them. This is a staging contract, not a newly verified cache receipt. Use actual pinned SIF versions Torch2.14.0a0+4fdf77b940.nv26.08/CUDA13.4, Transformers4.57.6, Accelerate1.15.0. No pip/network fallback or local-env substitution.

Runtime staging validity is **unverified now**: controller must check ownership, evidence still at staging, absent new destination, source/cache small identities and live quota. Large input/SIF/checkpoint hashing and staging remain on compute. If old P1 moved evidence, STOP; never recover by searching/copying/moving old partial input trees. Separate reviewed recovery is required.

## Minimal adaptation, not reimplementation

Reuse these existing scripts and their safeguards:

- `notes/mvp_execution_20260916/final_p1/{build_release.py,deploy.py,p1.sbatch,test_deploy.py}`.
- `notes/wait_work_20260916/gpu_followup/{followup.py,followup.sbatch}`.
- Frozen source `revision/scripts/mvp/{stage_checkpoint.py,make_launch_readiness.py,real_probes.py,run_bounded.py,resource_probe.py,run_training.py,accelerate_mvp.json}`.

They are NOT drop-in seed123 launchers. In a NEW owned deployment directory, controller authorizes a minimal adaptation and independent review:

1. Replace old git-only export with the reviewed dirty snapshot inventory; bind base+snapshot SHA truthfully, never claim dirty bytes came from a commit. Use new exclusive release/job/log names, proposed `staging/dev123_p1_release_20260917` and `dev123-p1-job-ID`. Leave old release untouched.
2. Retain deployer's quota parser,40GiB reserve+5GiB staging envelope,10000file headroom,2200000000B checkpoint-copy ceiling,≤3600s quota age; compute-only hashing; exact mounts and cleanenv/offline/cache/override guards. Retain source/cache/corpus/SIF read-only access. S0 inputs writable only for initial staging, then read-only. Keep 1700s outer+20s grace and900s P1 inner if controller approves the same1800s allocation; these are caps, not runtime predictions.
3. In `container_stage` readiness prepare argv, replace `--seed 1` with **`--profile development-single --seed 123`**. Keep all original explicit DB/evidence/panel/config/SIF/companion/source pins. No new model config is required. S0 paired fraction1 tile-selection helper's old augmentation constant is not the production augmentation identity; production uses20260914+123=20261037. Do not silently change selection behavior while repinning the launcher.
4. Followup adaptation must replace b660 source/tooling manifests, seed-in(1,2,3), hardcoded allocation3163986 and required completed3163986. Require the actual NEW P1 allocation and development-single/123/mvp_finetune; keep old3163986 and3163545 in complete accounting without requiring old3163986 to have succeeded. Call native `_load_receipt(..., bindings=bindings)` and `validate_budget(..., bindings=bindings)` for explicit profile checks. Never weaken source/binding/DONE/no-overlap/budget/timeout safeguards to make old receipts pass.
5. Preserve original P1 `/scratch/modules` imports at exact paths read-only in followups. Give each mode fresh distinct persistent module/Inductor/Triton/tmp/home paths; compiler dirs initially empty. Retain ALL mode import paths until readiness and decoding finish. No relocation/recycling/automatic cleanup.
6. Package adapted deployer/followup artifacts with independent external ledger pins, review them and rerun their local tests. Candidate source archive alone is not deployable. Full panel/evaluation also needs actual Git authority `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`; archive is not that object. P1 does not need frozen_reference, so do not block P1 merely on later evaluation transport.

## Exact existing command interfaces

These are reference interfaces for the controller's later reviewed package, not instructions to execute this candidate now:

```sh
# Local unchanged deployer tests and ledger verification (no submission):
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s notes/mvp_execution_20260916/final_p1 -p test_deploy.py -v
bash -n notes/mvp_execution_20260916/final_p1/p1.sbatch
(cd notes/mvp_execution_20260916/final_p1/release && sha256sum -c release.sha256)
# Existing host entry, ONLY in actual allocated pinned release:
python deploy.py host
# Existing post-P1 interface. Actual inputs/output mandatory, no synthetic receipt:
python3 followup.py --inputs "$ACTUAL_CONTROLLER_INPUTS" --mode real_hyena_step0_three_tiers --output "$FRESH_ABSOLUTE_OUTPUT"
# Same interface plus --execute only in reviewed SIF/allocation and a fresh output.
```

`deploy.py:container_stage` already invokes stage_checkpoint with exact source `/source/seed123/epoch_05`, destination `/mvp/inputs/b5_seed123_epoch05`, three expected hashes, quota record/hash/copy/reserve. Its next call is real `make_launch_readiness.py prepare`; use precisely the original input arguments plus the profile/seed change above. Its `container_p1` records absolute `sys.executable` and full `p1.argv.json`, then calls:

```text
<actual SIF sys.executable> /work/repo/revision/scripts/mvp/run_bounded.py
 --name real_tokenizer --bindings /mvp/probe/p1_control/prepared.json
 --bindings-file-sha256 <actual prepared file hash>
 --command-file /mvp/probe/p1_control/p1.argv.json --command-sha256 <actual argv hash>
 --output-dir /mvp/probe/real_tokenizer --timeout-seconds 900
```

Angle-bracket values are necessarily measured SIF outputs, deliberately not fabricated executable argv. No locally generated binding digest can replace them. `make_launch_readiness` creates the actual canonical digest including profile, seed, role, recipe, absolute source paths, SIF/input identities, installed packages and consumed/tooling manifests.

Six gates in sequence, each fresh at the same development binding: real_tokenizer; real_hyena_step0_three_tiers; generation_cache_parity; bf16_compile; four_rank_smoke; resource_budget. First three use direct recorded Python. Last three use the existing exact prefix:

```text
<actual SIF sys.executable> -m accelerate.commands.launch --multi_gpu
 --num_machines 1 --num_processes 4 --mixed_precision bf16
 --config_file /work/repo/revision/scripts/mvp/accelerate_mvp.json --max_restarts 0
 /work/repo/revision/scripts/mvp/real_probes.py
```

Native bound args come from actual P1 argv, changing only mode/scratch and resource arguments where required. Use run_bounded for every mode, no warmup bypass. Only after six genuine wrappers+DONE, no FAILED, matching imports/assets/bindings and reviewed terminal accounting: `make_launch_readiness.py generate`, then `check` with **the same** input flags/profile/seed/pins/manifest. No validate-only shortcut. Separately reviewed `run_training.py` requires actual readiness, reviewed argv and bound schema2 authorization; this preparation does not launch training.

## Budget reconciliation and missing controller decisions

Concrete existing authority is `final_p1/controller_execution_scope.json`: one old S0/P11800s allocation,0.5node-h/2GPU-h, cumulative diagnostic cap5node-h/20GPU-h, training_authorized=false. Earlier `MVP_DEPLOY_OPS` called5node-h a proposal; the later execution scope is the concrete controller receipt. The5node-h ceiling includes startup, transfers, copying, hashing, failed attempts and teardown, not just GPU compute. Old3163545 was recorded cancelled/zero allocation; current3163986 pending/elapsed0 is a controller snapshot, not exhaustive campaign accounting. The old one-job approval does not automatically authorize a replacement or followups.

No approved numeric development training cap was found in these current receipts. Historical1154GPU-h account balance and B5 reserve arithmetic are capacity/history, not a seed123 training authorization. Do NOT derive `max_training_node_hours` from20GPU-h or a legacy three-seed cap. Request a distinct numeric schema2 cap and wall only from controller's existing approved envelope/decision, then bind it to actual prepared digest.

Before new action controller must supply:

- Full exact source snapshot/aggregate decoder pin and adapted deployment review acceptance.
- New release namespace and stable absolute cache/mount layout; actual staging existence/ownership/quota and live B5 exclusions.
- Complete timestamped diagnostic allocation accounting, pending reservations and current account/B5 reserve/margin. Preserve B5 eight pending links unless its own legitimate lifecycle changes them; no additional seeds.
- Explicit one-replacement S0/P1 authorization within cumulative5node-h, and per-followup bounded outer/inner limits. A proposed1800s first allocation is the prior reviewed envelope, NOT a newly approved cap or performance claim.
- Schema2 controller_record_reference, bindings_sha256, positive `max_training_node_hours`, `max_wall_seconds`, `max_probe_wall_seconds`, `launcher_overhead_seconds`, safety_margin>1,0<memory_headroom_fraction<1, nonnegative **actual** prior_smoke_node_hours, intent and margin rationale. Candidate leaves these unresolved instead of inventing them.

Resource projection must remain `(250 * worst_rank_tier_steady + entire_probe_elapsed) * margin + launcher_allowance`, one run/3600 for development. Measure all three full tiers, cold+two steady updates×24microbatches/rank, four disjoint ranks, label8192/v3 maximum coverage, validation/publication/revalidation, memory headroom and no steady recompilation. No runtime/VRAM/throughput estimate is certified here.

## Controller-only replacement and next safe step

Old3163986 may be replaced only AFTER the new release is reviewed and a fresh live owner/PENDING/elapsed0/no allocated TRES recheck. Controller alone performs any cancellation, confirms terminal cancellation, reconciles all own jobs/B5/accounting, then considers one new submission. If RUNNING, race, terminal output not reviewed, unclear scheduler state or evidence moved: STOP; no automatic cancellation, retry, second allocation or input recovery. Keep old logs/receipts unchanged and never relabel them seed123.

**Next single safe step:** controller locally verifies `candidate.sha256`, compares its103-file snapshot to scoped TRAIN/EVAL/PANEL/recovery approvals, and authorizes the minimal new deployer/followup adaptation above. Do not transfer/cancel/submit merely because this candidate exists. Missing training cap does not justify fabricating resource authorization; P1 may be separately scoped once its own review/budget/queue gates are met.

Completed here: targeted context/authority/closure investigation, local candidate snapshot/config/binding-request/budget-request/package recipe and hash ledger. Blocked here: executable new deployment package review, fresh staging/accounting, numeric training authorization, all real GPU gates. These are explicit controller followups, not unattended worker TODOs.

Graft context first; estimated reported savings across calls:91010 tokens. No broad rediscovery or large-data scan.
