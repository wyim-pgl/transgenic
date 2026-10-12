# TransGenic restart handoff — 2026-09-15 15:13 PDT

> **SUPERSEDED:** Current restart authority is [RESTART_HANDOFF_20260916.md](RESTART_HANDOFF_20260916.md). The text below is historical; its pending metric/authentication/implementation/build states have been superseded. Preserve as evidence, not as an execution checklist.

**Read this before acting.** This supersedes older “current” blocks in `.atl/`. Historical logs remain preserved. All times here are PDT. Snapshot evidence: [remote status](restart_status_20260915.txt), [DeltaAI authentication](restart_deltaai_20260915.txt). This is a point-in-time snapshot, not a live monitor.

## 1. Author approvals and the open decision

Approved in this session: EST DIAMOND blastx route B execution; exploratory MVP; separate worktrees; remaining eight-species blastx first; independent tasks in parallel; `graft init --yes`. These are **not blanket A46 freeze approval**, not permission to restart deferred seeds 456/789, and not evidence that GPU training has started.

**Unanswered author question:** current model `outputs.loss` includes weighted CE + `0.05 *` detached digit/structure ratio penalty (and an auxiliary term if transcript counts are passed). MVP spec calls the paired loss weighted CE/nats and uses a 0.005-nats threshold. Proposed resolution: preserve existing training loss, compute pure weighted CE for paired validation, also log inherited composite loss. **Author has not approved this resolution. Do not change primary metric/threshold silently.** Trainer implementation has not started; planning is complete. Graft/restart requests are not metric approval.

MVP v1.2 approved bytes: `revision/protocols/B6_MVP_PILOT_v1.md`, SHA256 `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`. Panel SHA256 `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`: 2,079 loci (MVP core20 + MVP30 + pilot300 + final1,729). Full panel is protected in C0-lite; evaluation still uses MVP50.

## 2. Live remote work — do not duplicate submissions

Remote root `R=/data/gpfs/assoc/pgl/data/Transgenic` on `pronghorn`. SSH is working. Slurm jobs survive this local session.

### EST blastx

Each species: Swiss-Prot then OrthoDB; 16 CPUs, 64G, 12h cap; array concurrency2. Core flags: frameshift15, evalue1e-5, min-orf30, max-target-seqs25, max-hsps1. Same-species/descendant proteins are removed **before search** by making filtered per-species DBs. Raw13 preserves BTOP; downstream HSP≥30-aa acceptance and genomic phase projection **not implemented/executed**.

| Species | Job | State at snapshot | Both sources DONE? |
|---|---|---|---|
| Sitalica | 6163851 | COMPLETED, 21m54s | yes |
| Athaliana | 6164049_0 | COMPLETED, 1h03m56s | yes |
| Bdistachyon | 6164049_1 | COMPLETED, 33m41s | yes |
| Gmax | 6164049_2 | COMPLETED, 1h20m44s | yes |
| Osativa | 6164049_3 | RUNNING, 1h07m | Swiss DONE; Ortho running |
| Ppatens | 6164049_4 | RUNNING, 17m | Swiss DONE; Ortho running |
| Ptrichocarpa | 6164049_5 | PENDING array limit | not started |
| Sbicolor | 6164049_6 | PENDING array limit | not started |
| Vvinifera | 6164049_7 | PENDING array limit | not started |

Paths:
- Sitalica: `$R/evidence/est_blastx/sitalica_pilot_20260914/` (suffix is preparation date; executed Sep15).
- Remaining8: `$R/evidence/est_blastx/remaining8_20260915/<species>/{swissprot,orthodb}/`.
- Array logs: `remaining8_20260915/logs/6164049_<index>.{out,err}`; per-species `batch.exitcode`, timing files, each source `DONE`, `provenance.json`, `*.raw13.tsv` and `blastx.log`.
- Inputs/operational wrapper: `notes/b6_blastx_eight_inputs_20260915.json`, `notes/b6_eight_species_blastx_20260915.sbatch`.
- Reviewed remote launcher snapshot: `$R/worktrees/b6-est-blastx-950cf4c` (exported source, not a Git worktree).

Completed raw HSP counts (Swiss / Ortho): Sitalica 290,009 / 1,371,252; Athaliana 4,672,257 / 20,071,843; Bdistachyon 1,363,835 / 4,518,360; Gmax 6,825,751 / 22,401,265. These include short HSPs; **not accepted phase evidence counts**. Athaliana Ortho additional species exclusion count0 is consistent with its prior A19 exclusion from the source.

All completed allocation/batch records above are `COMPLETED 0:0`, with per-species exit0 and both DONE. Their `.extern` steps show `OUT_OF_MEMORY 0:125`, repeatedly including very small successful diagnostics. Record separately: neither silently clean cluster status nor proof that the application OOMed. Cause remains unresolved.

Frozen candidate lineages prepared by6163397: Ortho408 observed taxids, Swiss2,208 (2,015 newly resolved). Structural audit: no missing taxids/conflicts. Ortho lineage SHA `812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5`; Swiss `402bca970569cc2c0192ba185c12c8fcaed99ecafd71f46a5e19f4f2789615ce`. Structural completeness is not independent biological-ancestry/merged-ID proof; retain this limitation in downstream audits. See [lineage note](b6_blastx_lineage_preflight_20260914.md).

### Full MVP C0-lite build

**6164118 RUNNING**, cpu-51, elapsed47m at snapshot; 2CPU/64G/12h, no requeue. `$R/evidence/mvp_full_20260915/`, intended final `fullc0lite.db`.

Input hashes passed; builder is running. At snapshot `builder_result.json`, `qc.json`, `raster_results.json`, `SUCCESS.json`, `payload.exitcode` were **absent**. Empty stdout/stderr does not prove a stall; builder is quiet. Partial DB/WAL is expected while writing. **Never open it as the final artifact, delete its WAL, or relabel it complete.**

Completion gate: accounting + payload exit0; published DB; both retained-protected overlap QC counts0; `valid_tiles_with_evidence > 0`; full-source record count; before/after source identities; evidence-bearing train/valid forward/RC raster checks; `SUCCESS.json`. Even then record/review artifact identity before training. See [full build note](b6_mvp_full_build_20260915.md).

Prior sample6164093 succeeded: 10,000 records, builder195.87s, peak149,104KiB, DB16,527,360B. Sample valid-evidence count0 due geographic first-record sampling; train RC/N/off checked, valid only empty-source path. **Sample is NOT_FOR_TRAINING and not a full-data leakage certificate.**

## 3. Saved implementation branches — not merged into main

Local main stays `9a8c8e1`. `git worktree list` currently lists only main: agent worktree directories were cleaned, **branches/commits survive**. Do not reuse `/tmp/pi-agent-*` paths or reimplement completed increments.

| Work | Durable branch | HEAD | Tests + review |
|---|---|---|---|
| CPU builder/rasterizer | `pi-agent-611e8432-78e4-474` | `abca40f` | controller86 tests PASS; independent spec/quality PASS |
| Model projection/generation | `pi-agent-1765f0e2-cf6c-4ed` | `0514b63` | controller11 CPU tests PASS; independent PASS |
| Dataset/collate (includes both above) | `pi-agent-a92f1af6-926c-40f` | `c7fa759` | controller20 dataset+model tests PASS; independent PASS |
| Final blastx launcher | `pi-agent-cb2fd26a-e84c-456` | `950cf4c` | controller26 tests PASS; scoped review GO; real fixture6163396 PASS |

MVP ancestry verified: `5390864` copied approved prerequisite docs/data → `6e26a70` CPU implementation → `abca40f` report → `0514b63` model → `ad07302` dataset → `c7fa759` report. Last branch is the correct starting point for trainer work. Blastx history: `278cca7` → `5e0ab24` → `cc9735f` → `0715940` getseq correction → `950cf4c` version correction.

Reports are committed on those branches, not necessarily files in main. Retrieve e.g.:
```bash
git show c7fa759:notes/b6_mvp_dataset_implementation_report_20260915.md
git show 0514b63:notes/b6_mvp_model_implementation_report_20260915.md
git show abca40f:notes/b6_mvp_data_implementation_report_20260914.md
git show 950cf4c:notes/b6_blastx_implementation_report_20260914.md
```

Main contains valuable uncommitted A46/B6 English/Korean docs, MVP spec, panel TSVs/scripts, notes, `.atl/`, and new graft config. Branches also copied some of these prerequisite docs. **Do not reset/clean main or blindly merge a whole branch over it.** Make a new isolated worktree from c7fa759 for subsequent MVP development; integrate carefully only when requested. No push/merge performed. No running background agents or local recurring monitor remain; do not rely on old agent handles after restart.

## 4. Actual implementation boundary

Complete/reviewed: C0-lite builder and NumPy8-track rasterizer; model optional bias-free projection at per-base encoder states; forward/generate track propagation; dataset species filtering, separate worker readonly evidence connection, private augmentation RNG; 9-tuple evidence batches, optional tenth `rn`/`has_evidence` metadata; DNA-left padding.

Not complete: trainer CLI/loop, audited init-from, exact fixed-step scheduler/DDP validation, paired evaluator, final checkpoint publication, decoding/scoring/bootstrap, A-ONT1/A-HiFi MVP scoring alignments, phase projection, full B6 corpus/freeze/training. No MVP optimizer step or GPU run has been launched.

Testing limitations: model tests use actual wrappers/LED/GenerationMixin but a tiny substituted Hyena factory, local torch2.6/Transformers4.46.3. Not real Hyena400M, production4.57.6 SIF, bf16, compile, or DDP certification. Dataset uses real tiny DuckDB + tokenizer substitutes; real tokenizer alignment/actual forked DuckDB workers remain gates. Data suite86 and integration20 ran in separate environments; do not call that one unified full-suite pass.

## 5. Trainer plan recovered from planning agent (no implementation)

After metric decision:
1. Preserve default B5 behavior/recipes/deployment; explicit MVP path. Extend trainer/runtime or narrowly scoped helper; design brief before edits.
2. MVP Athaliana/K8/batch1/seeds1,2,3; aug_seed20260914+seed; dedicated loader-worker and sampler RNG; always pass tracks tensor; validation eval_fraction1 + tenth metadata.
3. Weights-only `--init-from`, mutually exclusive with resume; MVP resume deferred because exact worker-augmentation restart is unproven. Resolve `best` symlink to immutable checkpoint first. B5 checkpoint has safetensors/meta/Accelerate state, **not necessarily config.json**; architecture authority is enclosing run_config.json recipe/defaults. Compare semantic architecture and any config before alias-aware load; allow only missing projection, re-zero only when missing. Fresh optimizer/scheduler, not B5 optimizer restore.
4. LR backbone2e-5/projection2e-4, wd.02, clipping1.0; effective batch96 (4ranks ×24 accumulation ×batch1). N250 successful optimizer updates; warmup12. Test real Accelerate scheduler ticks at world sizes1/2/4 rather than assuming multiplier. Carry partial accumulation across epochs; stop exactlyN; MVP no early stop. Smoke budgets get distinct identities.
5. Compile failures abort MVP; explicit eager rerun requires new identity/directory and budget confirmation. B5 behavior unchanged.
6. Paired off/on eval same **current** backbone under eval/no-grad; dedup actual rn across DDP padding; reject missing/inconsistent/nonfinite records and empty evidence-positive set. Metric definition remains author-gated above.
7. Atomic final weights/config/metadata and paired summary; all-rank save/barriers, rank0 publication, TRAINING_DONE only after success. Record source checkpoint/config, DB/panel/SIF/code hashes and complete runtime/recipe identity.
8. Tests: alias-safe init/config drift, LR trace/exactN, cross-epoch accumulation, RNG separation, padded DDP pair dedup, artifacts, legacy B5 regressions. Worktree-local dependency installs only; no shared upgrades.

## 6. Runtimes and key identities

Pronghorn CPU container: `/apps/singularity/3.6.1/bin/singularity exec -B /data/gpfs /data/gpfs/home/wyim/scratch/singularity/micromamba.sif /data/gpfs/home/wyim/micromamba/bin/python`. PYTHONPATH adds `$R/worktrees/mvp-cpu-deps-20260914/site` for pysam0.23.3 and approved/src. **Unset PYTHONNOUSERSITE:** DuckDB1.5.5 is user-site. NumPy2.3.1. Host `/usr/bin/time` exists; inside container it does not—use stdlib resource/monotonic internally.

Local model/test interpreter `/home/wyim/miniconda3/envs/pdb/bin/python` (torch2.6/Transformers4.46.3); use worktree-local duckdb1.5.5 target for dataset tests, stdlib unittest. Never prepend Python3.8 binary-package directories to Python3.12 PYTHONPATH.

Input identities reverified on compute:
- Corpus `$R/training_input/db/b5_full_a40_v1.db` SHA `6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222`.
- Athaliana BAM SHA `f4598638a377d2ed6a89a9a01242f9872207ab6e55c98da5fe3e104764f0760e`.
- EST-alignment reference and corpus build_manifest FASTA SHA both `63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973` (byte identity now measured, not merely stated).

DeltaAI BatchMode via pronghorn **fails Kerberos/Duo authentication at restart snapshot**. Seed123 current epoch/status is unknown; old epoch4/best.6525 is historical. Ask author to re-establish `ssh -fN dtai` on pronghorn/login-0 before GPU/checkpoint checks. No seed123 checkout/jobs or deferred456/789 changes.

## 7. Graft and first actions after restart

`graft init --yes` completed in main: 233 Python files,2,478nodes/5,571edges; local graph git-ignored. Generated AGENTS.md/GEMINI.md, .claude settings/helpers/skill, .mcp.json, .gemini/settings.json, opencode.json, .ignore and .gitignore change are uncommitted. Home Codex config/hooks reported unchanged. Follow AGENTS graph-first guidance, **but main graph does not contain unmerged MVP/blastx branch code**. Build graph in the chosen implementation worktree before searching those features; Markdown handoffs may be unindexed.

First actions:
```bash
# READ ONLY, not resubmission
ssh -o BatchMode=yes pronghorn 'squeue -j 6164049,6164118 -o "%i %j %T %M %R"; sacct -j 6164049,6164118 --format=JobID,State,ExitCode,Elapsed,MaxRSS'
git status --short
git show-ref --verify refs/heads/pi-agent-a92f1af6-926c-40f
git show-ref --verify refs/heads/pi-agent-cb2fd26a-e84c-456
```

Inspect per-source DONE/provenance and full-build SUCCESS/QC as above, then address the unanswered metric question. Cluster jobs keep running during restart. No new jobs, commits, merges, or pushes were performed for this handoff refresh.
