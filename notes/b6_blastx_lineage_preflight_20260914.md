# B6 EST blastx lineage preflight — 2026-09-14

**Restart update (2026-09-15 15:13 PDT):** preparation6163397 finished allocation/batch0:0 (8m18s), both DONE and independent structural audit passed. Ortho408 observed taxids, Swiss2208 (2015 newly resolved), unresolved0/conflicts0. Ortho lineageSHA812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5; Swiss402bca970569cc2c0192ba185c12c8fcaed99ecafd71f46a5e19f4f2789615ce. Ancestry-authority limitation below remains; structural audit is not independent biological proof. Real searches subsequently authorized/launched: see [current handoff](RESTART_HANDOFF_20260915.md). Pending/running statements below describe the original preparation handoff, not current search state.

## Scope and current disposition

Authorized **taxonomy preparation only**. Submitted exactly one CPU job, **6163397**, named `b6-lineage-preflight-20260914`. No EST alignments, DIAMOND database builds, arrays, full protein downloads, production searches, B5 changes, or repository code modifications were performed. This is the only local file written by this executor.

**Latest bounded check: 2026-09-14 17:48:58 -07:00.** Allocation, batch, and extern were RUNNING (elapsed 00:01:12). Reported `0:0` values are provisional for running steps, not completed exit results. `batch.exitcode`, both source preparation provenance files, both candidate lineage files, and the independent lineage audit were not yet present. **Preparation and lineage readiness remain pending.** No recurring monitor or follow-on job was launched.

Remote root `R=/data/gpfs/assoc/pgl/data/Transgenic`.
New isolated output `D=$R/evidence/est_blastx/preflight_20260914`.

## Canonical source identity: verified before submission

Read existing small build receipts, not large source bytes, on the login node:

- `$R/evidence/protein/diamond/PROVENANCE.txt`
- `$R/evidence/protein/diamond/DB.sha256`
- `$R/evidence/protein/diamond/build_dbs.sh`
- `$R/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz.sha256`

OrthoDB `PROVENANCE.txt` and `DB.sha256` both identify the exact **already-filtered** source and agree with the authorized expected hash. `build_dbs.sh` confirms that input, rather than the original unfiltered partition. The Swiss-Prot FASTA-specific sidecar agrees with its authorized hash; the build's `DB.sha256` itself contains DMND hashes, not the Swiss FASTA hash. No conflicting receipt was found. Both input realpaths equal their stated paths under R.

| Source | Compressed FASTA path | Authorized and receipt SHA256 |
|---|---|---|
| OrthoDB | `$R/evidence/protein/odb12_Viridiplantae.filtered.fa.gz` | `fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f` |
| Swiss-Prot | `$R/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz` | `8d66a18e2ad350fd640f1fb97a180a8091c544efd33351e5f451bb03e11379c9` |

**Independent actual-byte attestation on compute: both PASS.** Before calling the launcher, the batch ran `sha256sum --check "$D/expected-sources.sha256"`; at the latest check `$D/source-hash-attestation.txt` contained:

```text
/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein/odb12_Viridiplantae.filtered.fa.gz: OK
/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz: OK
```

These expectations were not replaced with hashes of current input bytes. The reviewed launcher additionally checks each source hash before and after preparation; those later checks are still pending. Original build provenance was not altered; copies are retained in `$D/receipts/`.

## Reviewed snapshot and baseline cache

Reviewed local branch `pi-agent-cb2fd26a-e84c-456` resolved to commit `950cf4cd9ef55db276ffa1b713d8925ed6dd6dd2`.
Remote snapshot `$R/worktrees/b6-est-blastx-950cf4c` is an exported tree, not a Git worktree with repository metadata; its `SOURCE_COMMIT.txt` reads `950cf4c`. Direct SHA256 comparisons against `git show` of the reviewed branch matched:

- `revision/scripts/evidence/est_blastx.py`: `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291`
- `revision/scripts/64_orthodb_filter.py`: `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302`

Compute-side checks of both files and the baseline cache also passed at job startup. No source checkout, replacement, or launcher modification was made.

Using pronghorn's SSH connection to gpu, checked and copied only:

`gpu:/home/pgl/scratch1/wyim/transgenic_data/orthodb_filtered/lineage_ncbi.tsv`

into the newly created `$D/lineage_ncbi.baseline.tsv` using pronghorn-side `scp -p`. Source and destination were each confirmed **52,988 bytes**, SHA256:

`812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5`

The existing gpu cache was not modified. Each source receives this same baseline independently, so new/cache-hit counts are attributable per source. The reviewed CLI copies the supplied cache into a new source output directory before its mutating resolver is called.

## Submission and exact preparation commands

Python: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python`.

Stored operational batch script `$D/prepare.sbatch`, SHA256 `dfe295d76d9f4f15a5b153e6dcd4fb1e9fbd658aa1a6d59539c92bb8c33bf318`; syntax checked with `bash -n` before submission.

Submitted with `sbatch --parsable "$D/prepare.sbatch"`; receipt return code 0, stdout `6163397\n`, stderr empty, saved in `$D/sbatch-receipt.json`; exact ID retained in `$D/jobid`.

Allocation: account **and** partition `cpu-s1-pgl-0`; one node, one task, 4 CPUs, `--mem=16G`, `--time=01:00:00`, `--no-requeue`. GPFS working directory D, `TMPDIR=$D/tmp`, `XDG_CACHE_HOME=$D/xdg-cache`, stdout `$D/logs/slurm-%j.out`, stderr `$D/logs/slurm-%j.err`. `PYTHONDONTWRITEBYTECODE=1`; `OMP_NUM_THREADS=4`. Strict fail-fast shell; exclusive new-directory submission lock `$D/submission.lock`, in-job `flock -n` on `$D/execution.lock`, no-clobber result files, exit-code trap. The job started on **cpu-50 at 17:47:47 -07:00**.

With `W=$R/worktrees/b6-est-blastx-950cf4c`, executes these actual reviewed CLI commands **sequentially**:

```bash
"$PY" "$W/revision/scripts/evidence/est_blastx.py" prepare-lineage \
  --source-kind orthodb \
  --source "$R/evidence/protein/odb12_Viridiplantae.filtered.fa.gz" \
  --source-sha256 fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f \
  --cache "$D/lineage_ncbi.baseline.tsv" --fetch-missing \
  --out-dir "$D/orthodb"
"$PY" "$W/revision/scripts/evidence/est_blastx.py" prepare-lineage \
  --source-kind swissprot \
  --source "$R/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz" \
  --source-sha256 8d66a18e2ad350fd640f1fb97a180a8091c544efd33351e5f451bb03e11379c9 \
  --cache "$D/lineage_ncbi.baseline.tsv" --fetch-missing \
  --out-dir "$D/swissprot"
```

`--fetch-missing` invokes the existing helper's NCBI taxonomy efetch resolver for taxids absent from the candidate cache, including single-ID merged-taxid fallback. It does not fetch proteins or perform alignments. New candidates, if successful, will be `$D/{orthodb,swissprot}/lineage.tsv` with separate provenance and DONE files. They are not approved production lineages merely because DONE exists.

Duplicate checks: the initial valid `squeue -u "$USER"` listing showed no lineage preflight. The submission helper mistakenly used `-u pgl`; Slurm printed `Invalid user: pgl` but returned success, so the stored initial `duplicate-check.txt` is not reliable evidence. A subsequent valid check with actual user **wyim** found exactly one relevant job, **6163397**, RUNNING, and was stored as `$D/duplicate-check-correct-user.txt`. The new isolated directory and exclusive submission lock also prevented this executor from submitting twice. No resubmission occurred.

## Independent lineage verification: planned in same job, pending

After both preparations succeed, an inline standard-library audit in the operational batch (not a repository code edit) independently rescans source FASTA headers and compares every observed-taxid sequence count against preparation provenance. It independently parses each candidate TSV and checks:

- every observed taxid exists as the first entry of its lineage;
- positive IDs, no duplicate cache keys or ancestry cycles;
- terminal root 1 or cellular organisms 131567, and Viridiplantae 33090;
- exact ancestry suffix agreement wherever another ancestor has its own cache row;
- no changes to baseline rows for observed taxids;
- candidate SHA256 equals its preparation provenance.

It records source sequence/taxid counts, baseline cache-hit observed taxids, newly resolved observed taxids, added cache keys, unresolved IDs, structural failures and cached-ancestor conflicts in `$D/independent_lineage_audit.json`. Newly resolved counts are set differences against the verified baseline, not a claim about the number of HTTP requests; the launcher's `fetched: true` is only a flag. Any structural failure exits nonzero, retaining candidates and audit for diagnosis.

**Important limitation:** complete biological ancestry and authoritative merged-ID identity cannot be independently proven from these TSVs alone. They contain ordered chains but no authoritative per-edge parent records or explicit merged-ID metadata; available suffix consistency does not prove that an intermediate ancestor was never omitted. The launcher's merged-ID fallback preserves a requested historical ID before the returned canonical chain, but the candidate TSV does not label that relationship. Existing cached IDs are not refreshed. Accordingly, even a later structural pass does not satisfy a strong independent complete-ancestry/merged-ID attestation by itself. Retain this as a production-readiness gate rather than declaring full readiness without additional evidence.

| Source | Actual source bytes | Candidate lineage SHA256 | Cached/newly resolved counts | Independent ancestry audit | Production readiness |
|---|---|---|---|---|---|
| OrthoDB | PASS against canonical expectation | Pending | Pending | Pending; authority limitation above | Not attested |
| Swiss-Prot | PASS against canonical expectation | Pending | Pending | Pending; authority limitation above | Not attested |

## Execution anomalies and resource interpretation

The user-reported real synthetic fixture **6163396**, using this remote snapshot, passed all six query cases for both sources, with allocation and batch COMPLETED `0:0`, but extern OOM `0:125`. Similar extern anomalies occurred in successful diagnostics; the existing DB build 6163304 also had allocation/batch success with extern OOM. These are recorded as anomalies, **not evidence of OOM-safe production search sizing**. This preparation allocation does not measure production blastx memory/runtime.

Setup-only failures were recovered before successful submission: login `python3` was unavailable (switched to the required explicit interpreter); old remote Git rejected `-C` (export identity was established through SOURCE_COMMIT and file hashes); first new-directory creation needed missing parents created; an initial local shell-quoting failure prevented batch generation/submission and attempted root-level redirections failed with permission denied. The corrected generated script passed `bash -n`, was read back after submission, and is retained at D. These failures did not produce an earlier job or modify existing provenance.

Remote bounded-check evidence: `$D/accounting-bounded-check.txt`, corrected duplicate check, submission receipt, source attestation, logs, expected hash files, and copied canonical receipts. No completion or candidate hashes were inferred from absent outputs. Future retrieval must check allocation **and** batch/extern accounting, `batch.exitcode`, both provenance/DONE or FAILED files, source attestation, candidate hashes, and the audit. No production search may be inferred as approved by this pending preparation record.
