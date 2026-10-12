# B6 3-track discovery synthesis — 2026-09-16 (controller)

Phase 1 of the authorized subagent orchestration ([authorization](b6_parallel_execution_authorization_20260916.md)) is complete: three scoped owners collected live read-only facts and execution designs; each was independently adversarially reviewed. All three verdicts: **APPROVE_WITH_FIXES**. No Slurm submission, no remote writes, no worktree creation, seed123 untouched. Raw designs/reviews: workflow `wf_fb9feec892e4` journal (`/tmp/pi-subagents-1000/.../wf_fb9feec892e4.workflow.jsonl` — tmp lifetime not guaranteed; this note is the durable summary).

## Track A — GPU/SIF preflight → launch-readiness → MVP training

New measured facts (evidence commands in journal):
- SIF `/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif` exists, 11,120,398,336 B, fresh sha256 = `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6` — **matches recorded hash**.
- Concrete epoch_05 candidate binding measured: `epoch_05/model.safetensors` 2,091,010,540 B, sha256 `6e5e408681dd5356548a8f4b93d28e561c22d885668a4e0015d3e314e28f8d22`; parent `run_config.json` sha `c3ca0b4d…`, `runtime_identity.txt` sha `44b6f578…`. eval_loss 0.6394507885, global_step 10355. seed123 was mid-epoch-6 at observation — **epoch_06 may supersede; pinning choice is an author decision**.
- QOS `bilv-dtai-gh`: no MaxJobs/MaxSubmit/TRES limits; ghx4 MaxWall 2-00:00:00, heavily utilized. New 4-GPU jobs permitted by policy; constraint is queue contention only.
- **Confirmed gap:** MVP python code at f847cb8 has NO CUDA warmup/retry and NO signal handling; and mvp_cli has **no public resume** — wrapper SIGTERM cannot produce a graceful save (walltime overrun = total loss of that seed's run; trap must record failure atomically, never fake completion).

Design: new disjoint DeltaAI namespace `/work/nvme/bilv/wcyim/transgenic/mvp/{preflight,readiness,runs}`; immutable git-archive snapshot of f847cb8; P1 identity / P2 step0 determinism / P3 generation parity / P4 compile+topology+memory smoke jobs; readiness record binding only measured receipts; then 3 seeds × 250 updates.

Reviewer must-fix (all accepted):
1. Readiness schema is resolvable NOW from `mvp_cli_runtime.py:175-236` (schema_version 1, authorized_scope `exploratory_mvp`, source_checkpoint binds seed123 identity, exact bindings digests, fail-closed evidence set) — design generator to it before writing scripts.
2. Checkpoint copy must be staged in a layout `weights_init.py`/`_validate_initialization_readiness` accepts (weights + sidecar run_config.json + runtime_identity.txt, re-verified hashes), or bind the original epoch dir read-only with proof B5 cannot mutate a completed epoch dir.
3. `--validate-only` skips the readiness requirement — verify readiness validation path explicitly (direct `validate_launch_readiness()` check).
4. Stage the FULL mvp_cli input set to DeltaAI (db, evidence-db, panel, config, init-from, sif, acceptance/inventory/receipts JSONs `faab8387/9f790196/73204a4e`), each with transfer route + post-copy re-hash.
5. SIGTERM semantics corrected as above; only CUDA warmup+retry is wrapper-fixable.
6. 2.09 GB checkpoint copy / large hashing = compute-job or data-mover, never login node.
7. Measure /work/nvme quota headroom before staging.
8. P2 GPU bit-identity gate needs an explicit determinism config (deterministic algorithms, CUBLAS_WORKSPACE_CONFIG) or a pre-declared measured tolerance — no ambiguous gates.

## Track B — decoding, independent long-read scoring, paired bootstrap

New facts:
- Spec contract extracted verbatim from `f847cb8:revision/protocols/B6_MVP_PILOT_v1.md` (50-locus panel; 300 decodes; tile bootstrap 2000 reps rng(20260914); δ_loss 0.005 nats, δ_F1 2.0 pp; STOP-first §5.3 rule; downgrade clauses). Panel `b6_panels_v1.tsv` **is committed on f847cb8**, sha `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` (matches working tree) — gates pin the commit-bound hash.
- A-ONT1 (SRR28341943/44/45 FLIC) present on pronghorn with DONE+md5. **A-HiFi (PRJEB77203 ERR13994458) is ABSENT from pronghorn** — fetching + ccs/lima/refine is a new multi-hour/tens-of-GB task, or scoring is ONT-only.
- No §3.3 orientation audit exists for the FLIC runs (orientation_audit_v1/Athaliana has 10 datasets, no FLIC).
- Legacy `longread_align.sbatch` Gate 1 refuses b1_validation_only runs — scoring-only driver confirmed necessary. mvp_metrics.py paired primary exists; decode/score/bootstrap code does NOT exist on f847cb8.
- Proposed disjoint namespace: `$R/evidence/mvp_scoring_20260916/` (verified no collision).

Reviewer must-fix (all accepted):
1. Pass-1 prefilter interval: use the 30,720-nt **centred evaluation window** (shared window-derivation function + unit test) ±10 kb — NOT gene±10.24 kb (silently drops reads supporting eval-window junctions; can flip STOP↔INCONCLUSIVE). Resolve 10,240-vs-10,000 explicitly.
2. Pin the two restriction scopes exactly (read-supported junctions in panel *windows*; predicted/recall denominators against panel *genes*) with unit-tested predicates sharing fix-1's window code.
3. Roles/md5 gate must bind to the file the job reads (`$R/evidence/DATASET_ROLES.longread_v1.tsv` vs repo copy — declare canonical, assert format/columns, fresh md5sum of fa.gz).
4. Track-B orientation audit must reproduce the legacy DONE-key contract (FA_MD5:REF_MD5:GFF_MD5:scorer + audit.json) in the new namespace; precondition-test the tool tolerates a non-default root without touching orientation_audit_v1/.
5. Panel-TSV premise corrected (committed, commit-bound hash is trust anchor).

## Track C — blastx audit → HSP acceptance → phase projection

New facts:
- All 18 species×source units complete: DONE + provenance.json + raw13 + commands.json + logs; batch.exitcode=0 everywhere; sacct reconfirms allocation/batch COMPLETED 0:0 for 6163851 + 6164049_0-7 and `.extern OUT_OF_MEMORY 0:125` on ALL 9 jobs (**recorded, still unresolved — no clean-accounting claim**). Total raw13 ≈ 16.4 GB. **No resubmission needed or permitted.**
- Engine at 950cf4c confirmed: raw13 13-field with btop, `--frameshift 15 --evalue 1e-5 --min-orf 30`, hsps_lt_30aa counted but preserved; DONE/FAILED.json semantics; launcher itself disclaims acceptance.
- Route-B authority: `B6_EVIDENCE_MODE_WORKPLAN_v1.md` §2.6 (frame projection across introns mod 3, 6 channels, route-A disagreement→0) and §2.2 est_protein_hsp schema; **§4.6 keeps WP1 (the whole route-B chain) BLOCKED until explicit A46 approval** — only exploratory MVP pre-permitted.
- Proposed namespace: `$R/evidence/est_phase_routeB_20260916/{audit,acceptance,projection}`; stages C1 audit (compute job, fail-closed hash/count reconciliation vs provenance) → C2 HSP≥30aa acceptance (BTOP frameshift decomposition, RICH17 real-DIAMOND fixture to pin '/' and '\\' semantics) → C3 projection (hand-computed multi-exon fixtures, FLAG×ts strand rule).

Reviewer must-fix (all accepted):
1. **C2 production (not just C3) sits behind the same author/A46 gate** — only implementation+fixtures pre-permitted.
2. Pin the operational "HSP ≥ 30 aa" definition (raw13 col 9 semantics under --frameshift are nonstandard: alignment length vs ungapped aa vs per-frame sub-span) and verify against the RICH17 fixture.
3. C1 must explicitly enumerate `preflight_20260914/` and tmp/ dirs with fail-closed DONE/FAILED/neither disposition.
4. Handle the two provenance layouts (pilot: query.sha256 at species root; remaining8: inputs.json+inputs.sha256 at batch root); unresolvable query file = hard FAIL.
5. Bind the strand-composition rule to the exact quarantine record (§1m ts:A/FLAG16 row), not a mis-cited section.

## Author decisions required before next dispatch

1. **Checkpoint pinning (A):** pin epoch_05 now (hashes above) or wait for the running epoch_06 / later B5-best? Waiting couples MVP start to the 9-job B5 chain.
2. **Slurm submission authorization:** exits read-only discovery. (a) DeltaAI preflight smokes P1–P4 (~5 node-h); (b) pronghorn compute jobs: C1 audit (≤8 CPU/16 GB/6–12 h), Track-B alignment array + orientation audit, fixture jobs.
3. **A-HiFi (B):** fetch+CCS/lima/refine ERR13994458 (tens of GB, compute budget) or proceed ONT-only? If ONT-only: does ONT-only local alignment satisfy the §5.3 availability clause, or does any prospective STOP downgrade to INCONCLUSIVE?
4. **C2/C3 production scope (C):** does the 2026-09-16 authorization permit production acceptance/projection over 9 species now, or implementation+fixtures only with production behind A46? (Reviewer reads the workplan as the latter.)
5. **Budget (A):** 3 MVP training jobs (est. 10–40 node-h each pending P4 measurement) on bilv-dtai-gh while the B5 chain consumes the same account.
6. **DB staging route:** fullc0lite.db (1.07 GB) + full input set pronghorn→DeltaAI transfer route and target path.
7. **Spec path drift (B):** confirm `mvp_full_20260915/fullc0lite.db` formally supersedes the spec's `evidence/mvp/athaliana_est_c0lite_v1.db`.
8. **Orientation audit (B):** run §3.3 audit for the 3 FLIC runs into the new namespace, or documented no--uf default?

## What can proceed WITHOUT further author input (per existing authorization)

- Code implementation + CPU tests in fresh isolated worktrees (no Slurm, no remote writes): Track A readiness-generator + Slurm wrapper scripts designed to the exact mvp_cli_runtime schema; Track B scoring-only driver, junction table, mvp_score.py/bootstrap/decision module + fixtures; Track C audit script + est_phase_accept.py + est_phase_project.py + unit fixtures. Each incorporating its must-fix list, followed by independent review.
- Cheap read-only lookups: HF offline cache location for hyenadna (from the 8f049eb B5 launch script), /work/nvme quota headroom (df, read-only).

No production claims are made by this note. `.extern OOM` remains untriaged. Thresholds (250/12, δ 0.005 nats, seeds 1/2/3) unchanged.
