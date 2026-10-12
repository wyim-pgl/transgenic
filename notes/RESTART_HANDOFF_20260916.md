# Restart handoff — 2026-09-16 10:41 PDT

> **Superseded operational status:** read [12:51 recovery](RESTART_RECOVERY_20260916_1251.md) first. Phase-2 partial worktrees were recovered and three implementation agents dispatched. Statements below saying “not dispatched” or “no local agents” are historical, not current. Technical background remains useful.

Read this after `.atl/resume.md`, then execute `.atl/todo.md`. This supersedes the Sep15 handoff and all archived “current/next/pending” blocks. Remote observations are snapshots, not a live monitor.

## 1. Exact stopping point and latest author request

The author requested **all three following workstreams through subagents**:
1. actual Hyena/tokenizer/SIF/GPU preflight, launch-readiness, then MVP training;
2. decoding, independent long-read scoring, and bootstrap;
3. full nine-species blastx artifact audit, HSP acceptance, and genomic phase projection.

This request authorizes proceeding with those exploratory tasks and multi-agent orchestration. [Authorization note](b6_parallel_execution_authorization_20260916.md). It is not A46/full-production approval, permission to restart deferred B5 seeds456/789, or permission to mutate existing seed123 jobs/checkouts.

**Those new workstreams have NOT been dispatched yet.** No SubagentWorkflow was launched. The author interrupted to invoke `/advisor`, then requested this restart update. Advisor returned **use limit reached (5/5)**; no new strategic advice was obtained. Do not describe that failed call as approval/review. No local monitor or agent/workflow is running; old completed agent handles may be gone. Remote seed123 training continues independently.

Next controller should launch three scoped owners in parallel for live prerequisites/execution designs, independently review the designs/scripts, then dispatch bounded implementation/execution stages. Do not repeat completed CPU implementation. Gate GPU fine-tuning on actual readiness results; decoding/scoring scripts and blastx auditing can prepare concurrently. Assign one DeltaAI job writer, and disjoint pronghorn namespaces for long-read and blastx work. Before any submission, inspect live jobs and output identities to avoid duplicates. Record per-job ownership/resources/receipt paths, including pending Slurm states rather than polling tightly.

## 2. Durable code and isolation

| Work | Durable branch | Exact commit |
|---|---|---|
| Latest cumulative MVP: data/model/dataset, metrics, weights-init, fixed runtime, atomic finalizer, exploratory CLI, GPU topology gate | `pi-agent-5b846a12-93fa-4c3` | `f847cb80c7750326f0781f81481891ab0c9ea309` |
| Latest blastx engine/fixture launcher | `pi-agent-cb2fd26a-e84c-456` | `950cf4cd9ef55db276ffa1b713d8925ed6dd6dd2` |
| Main (unchanged code) | `main` | `9a8c8e1ba68286e845fb99be3cb76112a8bcd186` |

Both implementation heads were verified with `git show-ref` at restart. **Neither is merged into main or pushed by this session.** Main contains valuable uncommitted protocol/panel/notes/graft config; preserve it. Main's graft graph does not include these branch implementations. New implementation work must start in a fresh isolated worktree from the relevant exact head; build/query graft there before source exploration.

Some older temporary worktrees remain registered (unlike the earlier Sep15 snapshot): `b6-mvp-fixed-runtime-review1` at a314bf9, `won/b6-mvp-finalize` at0a8120b, `b6-mvp-weights-init-review1` at3a63314, and `/tmp/pi-agent-round2-f6b1e77` detached at1c47ae5. They are stale, not the latest source and not evidence of active agents. Do not clean them blindly. Verify `git worktree list` before creating/repairing isolation.

Latest controller export: `/tmp/b6-topology-controller-f847cb8`. This is a **git archive, not a checkout**. Its lifetime is not guaranteed; recreate from Git if absent. The authoritative code is the branch. Main-path handoff docs may be uncommitted; copy a brief to neutral `/tmp/…` before delegation if isolation prohibits reading main. Earlier agent stopped solely because its brief was absent from its isolated checkout; no code was lost.

## 3. Completed CPU implementation — do not redo

The new path leaves B5 model/trainer behavior intact. Entry point: `train/train_HyenaTransgenic_MVP.py`; helper files under `src/transgenic/training/`: `mvp_cli.py`, `mvp_cli_runtime.py`, `mvp_metrics.py`, `weights_init.py`, `mvp_runtime.py`, `mvp_finalize.py`.

- Strict B5 evidence0→MVP8 weights-only initialization from concrete checkpoint; source authority is parent `run_config.json` recipe, not nonexistent checkpoint config. Safetensors aliases preserved; exactly missing projection is zeroed. No optimizer/RNG/pending-gradient restore. Failed target poisoned/discarded. Parent `runtime_identity.txt` hashed as recorded provenance.
- Training loss stays inherited composite. Paired evaluation primary is **pure weighted CE**, secondary actual composite. CE denominator is target-weight mass, not token count. Primary aggregate is **equal-weight evidence-positive tile mean of off−on differences**, not pooled token CE. No second label shift.
- Fixed250 successful optimizer updates,12 warmup, batch1,4ranks×24 accumulation, backbone LR2e−5/projection2e−4, AdamW wd.02, clipping1. Cross-epoch gradient carry; nonfinite/skipped step fails. Standard post-step LambdaLR-equivalent schedule: update1 uses LR0; update13 peak; update250 peak/238; after250 zero. This convention is explicit in tests, not the rejected shifted table.
- DDP `find_unused_parameters=True`; exact optimizer group identity/order; prepared model/optimizer, unprepared rank-sharded loaders. No double sharding.
- Canonical `aug_seed=20260914+seed`; private workers0=aug_seed+rank, trainworker=aug_seed+1,000,000+rank, validworker=aug_seed+2,000,000+rank, sampler seed=seed. Separate generator objects. These exact maps are tested through real execute→train wiring.
- Bound-forward compile before DDP; no silent eager fallback. Paired evaluation uses the same prepared/compiled model, not a different eager evaluation path. Saving alone unwraps known wrappers.
- Bounded tensor record/status transport; exact RN inventory and padding duplicate validation; complete returned initializer audit schema/digest; all-rank coordination. Rank0 sibling-stage safetensors/config/metadata/records/summary/manifests, marker last, Linux `renameat2(RENAME_NOREPLACE)`. Post-rename parent-fsync failure preserves complete artifact and raises committed durability uncertainty.
- CLI separates technical candidate status from exploratory launch-readiness and production permission. No public resume/eager/test-budget/production override. Real execution lacks the mandatory readiness record and remains blocked. Internal finalizer `identity_scope="production"` means fixed250×96 profile only; CLI separately labels `EXPLORATORY_MVP_NOT_FOR_PRODUCTION` and `production_authorized=false`.
- Canonical verified paths consumed, actual initialization audit compared to readiness, identity checks at open boundaries and **after evaluation/before publication**. Consumed-code manifest covers exact source tree+entrypoint; receipt schemas and all evidence split counts fail closed.
- Public runtime requires **single-node MULTI_GPU4**, dense rank/local-rank/device0..3, four visible distinct non-MIG GPU UUIDs on every rank, native bf16 support; measured topology recorded. This restrictive topology must be verified in actual deployment; CPU test topology facts do not certify GPUs.

### Verification evidence

Controller independently exported latestf847cb8 and reran:
- CLI+finalizer+fixed-runtime pytest: **107 passed,407.16s**;
- weights-init+metrics+model+dataset unittest: **46 passed,52.661s**;
- total **153 passed**. Diff whitespace check clean.

CPU Torch2.6.0+cu124 / Transformers4.46.3 / Accelerate0.34.2 / safetensors0.5.2 / DuckDB1.5.5. Real CPU Gloo1/2/4-rank probes and actual tiny TransGenic wrappers were used; external Hyena/tokenizer factories were substituted. Compile integration used an eager counting backend. **No actual Hyena400M, production tokenizer/SIF, GPU, bf16 numerics, Inductor, or production filesystem certification.** Initial data86 suite passed earlier in a separate environment; do not sum it into a unified latest run.

Independent per-increment reviews passed after scoped fixes. Whole-path review found one medium GPU-topology gap; final fix f847cb8 scoped re-review **PASS/APPROVE**, no residual finding. [Integrated review](b6_mvp_integrated_review_20260915.md); controller reports: [CLI](b6_mvp_cli_controller_20260915.md), [finalizer](b6_mvp_finalize_controller_20260915.md), [runtime](b6_mvp_runtime_controller_20260915.md), [weights](b6_mvp_weights_init_controller_20260915.md), [metrics](b6_mvp_metrics_controller_20260915.md).

One optional broad-suite run on an earlier runtime increment had698passed/24failed, including missing seqkit and external .fai; other failures untriaged. **No whole-repository green claim.** Tests/run recipes in those reports are the reproduction source.

## 4. Live DeltaAI observation — 09-16 10:41 PDT

[Raw accounting/status](restart_status_20260916.txt); [checkpoint metadata](restart_seed123_20260916.json.txt).

- Authentication works via `ssh pronghorn 'ssh dtai …'` BatchMode.
- Existing B5 seed123 job **3131537 RUNNING gh086**, elapsed4:39:42. Following3131538 and3149356–3149363 PENDING Dependency.
- `repo_seeds` remains **8f049eb3ba72ba7541076fe9fc8f6019b90c0f7f**; no pull/edit/job mutation performed.
- `best` now resolves to `/work/nvme/bilv/wcyim/transgenic/runs/seed123/epoch_05` (epoch04 was an older snapshot).
- Epoch5 complete: eval_loss **0.6394507884979248**, train_loss0.6142173409461975, global_step10355, bad_epochs0, best_epoch5. Next epoch training underway. `latest_state` observation: epoch5, step5865, global_step10600. DONE/FAILED markers absent.
- Existing runtime record: NGC26.08, Torch2.14.0a0+4fdf77b940.nv26.08, GPUs4/CPUs64, nonreentrant, compile on. Recorded SIF hash `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6`. Expected SIF location `/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif` from prior launch records; **reconfirm actual file and fresh hash**.

For MVP initialization resolve and hash a **concrete** checkpoint, bind it in readiness, and use the same initialization across MVP seeds. Do not silently follow mutable `best` as B5 improves. No actual epoch05 weights/SIF readiness verification has been done. Seeds456/789 remain author-deferred, not to restart implicitly.

## 5. Full MVP data candidate — technically accepted, not production training

Pronghorn root `R=/data/gpfs/assoc/pgl/data/Transgenic`; final DB `$R/evidence/mvp_full_20260915/fullc0lite.db`.

- Final size1,068,773,376B; independently rehashed SHA256 `ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d`.
- Job6164118 allocation/batch COMPLETED0:0, payload0, elapsed2:22:10. `.extern OUT_OF_MEMORY0:125` is unresolved separately; no claim of completely clean accounting.
- Counts reconciled:1,050,677 BAM alignments−3,852 supplementary=1,046,825members; minus60,444collapsed=986,381molecules; retained733,812+panel67,293+test113,262+unmapped72,014=986,381.
- Panel2079 protected, retained panel/test overlap0, valid evidence1442/1447. Five no-evidence valid tiles explicitly retained. Four train/valid forward/RC raster checks PASS. These are MVP limitations, not full C0/A11/secondary-alignment certification.
- Ordered Athaliana inventories: train7635+valid1447=9082, source `geneList` and evidence `corpus_tile` exactly matched on RN/chrom/coords/split/RC. Test count1388 also reconciled. Corpus23.5GB not rehashed again during consolidation; freeze/prebuild/postbuilder receipts bind its hash.

Local canonical records (preserve exact bytes, do not rewrite merely to alter dates):

| File | SHA256 |
|---|---|
| `notes/b6_mvp_technical_candidate_20260915.json` | `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e` |
| `notes/b6_mvp_candidate_inventories_20260915.json` | `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98` |
| `notes/b6_mvp_candidate_receipts_20260915.json` | `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac` |

Receipts companion binds53receipts including41 freshly rehashed remote files and measured Git objects. Controller checked canonical JSON, all three hashes, companion sizes/digests and inventory count. Decision technical_acceptance=true; production_training_accepted/production_authorized/author_protocol_freeze/scientific_approval=false. [Audit/scope ruling](b6_mvp_candidate_audit_20260915.md). Actual launch-readiness JSON does not exist; it must be generated from measured runtime/preflight receipts, never placeholders.

## 6. Blastx and scientific downstream work

Nine species have both source DONE observed. Sitalica pilot6163851 completed earlier; remaining8array6164049_0–7 all allocation/batch COMPLETED0:0 reconfirmed10:41PDT, none queued. Every completed array's extern anomaly persists. **Do not resubmit blastx.**

Paths: `$R/evidence/est_blastx/sitalica_pilot_20260914/`, `$R/evidence/est_blastx/remaining8_20260915/<species>/{swissprot,orthodb}`. Snapshot launcher `$R/worktrees/b6-est-blastx-950cf4c`. Per-source provenance/DONE/raw13/timing and species batch.exitcode remain inputs for the requested **full output audit**, not already fully audited just because DONE exists. Preserve same-species/descendant exclusion before search and lineage limitations.

Not implemented/executed: raw HSP≥30aa final acceptance, frameshift-aware genomic phase projection and path-A disagreement handling. No phase evidence should be inferred from raw HSP count. Follow approved route-B protocol/spec, resolve details with real source definitions and tests before production processing. Audit large raw outputs on compute jobs with reviewed scripts/resources, not heavy login-node scans.

MVP training has not run. Decoder, independent A-ONT1/A-HiFi alignments/scoring, and paired bootstrap/GO-STOP-INCONCLUSIVE remain to implement/execute. Approved MVP spec on f847cb8 is `revision/protocols/B6_MVP_PILOT_v1.md`; use its exact commands/statistics/50locus panel/three-seed decisions. Paired primary convention is user-approved pure weighted CE+composite logging, delta off−on, δ.005nats; do not change thresholds/estimands. Keep input evidence disjoint from independent scoring evidence. Absence of independent evidence is not a negative biological result. The legacy long-read driver rejects validation-only runs; make a reviewed scoring-only path, not a role-bypass in the training driver.

## 7. Operational reuse and remaining decisions

- Local test interpreter `/home/wyim/miniconda3/envs/pdb/bin/python`; existing compatible isolated deps `/tmp/b6-runtime-controller-227c6e3/.testdeps` (Accelerate0.34.2, pytest8.3.5, psutil, iniconfig, pluggy) and `/tmp/b6-dataset-verify.D2Is9a/.testdeps` (DuckDB1.5.5). Recreate if absent. Probe subprocesses may reset PYTHONPATH to checkout-local `.testdeps`; link/install there explicitly. Do not alter shared environments or mix Python3.8 binary packages with3.12.
- Pronghorn container/runtime paths are in Sep15 handoff §6. DuckDB there is user-site; do not hide it via PYTHONNOUSERSITE. CPU/large scans belong on compute allocations.
- All timestamps in human-facing records PDT. Metadata timestamps without zone retain their original ambiguity; do not silently relabel remoteCDT asPDT.
- B6/A46 final freeze, main integration/push, deferred456/789, manuscript/wiki/scoring-universe decisions remain separate. Preserve old unresolved issue history; do not blindly execute archived tasks.
- User's request is to use subagents for the full next phase. A workflow is explicitly allowed now, but none was started. Do not rely on previously completed handles or imply a background monitor exists.

## 8. Restart documentation preservation

Old active files were archived byte-for-byte before consolidation:
- `.atl/archive/resume_before_20260916.md`: SHA256 `4eb28de3c6c74436a40071719247bdbccec24250ec9244703a729e45abd391ef`.
- `.atl/archive/todo_before_20260916.md`: SHA256 `2f25fe5542a2eb93dfe9adeb1519dae4f032eb03688f8a93c0fd2e84767c824c`.

Historical uncompleted checkboxes may be superseded; current `.atl/todo.md` is authoritative. Recurrence lessons are `.atl/quarantine.md` §1n; root `quarantine.md` points there. `notes/RESTART_HANDOFF_20260915.md` remains historical. This refresh changed documentation only and took read-only remote observations: no job/production file modification, no merge/push.
