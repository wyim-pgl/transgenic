# B6 EST blastx implementation handoff — 2026-09-14

## Status and isolation

**Ready for controller code review; real-engine validation still required. No remote job submitted, no production input built, no push/merge.**

- Worktree: `/tmp/pi-agent-462c26cc-1192-466-24b9a1b8`.
- Branch: `b6-est-blastx-reviewed-20260914`.
- Implementation commit: `278cca7fcaf15ecac828db74b9197f427e2494f2`.
- Lineage compatibility correction: `5e0ab24` (accept NCBI cellular-organisms ancestor when taxid 1 is omitted).
- This report is committed separately on the same branch. Commit author: Won Yim `<wyim@unr.edu>`.
- Copied only the exact execution brief and current English `B6_EVIDENCE_MODE_WORKPLAN_v1.md` reference from the main checkout. The protocol diff is that reference synchronization, not new protocol authorship/approval. No MVP implementation, model, training, or seed123 files changed. No main-checkout writes.

## Files and behavior

- `revision/scripts/evidence/est_blastx.py`: stdlib `prepare-lineage` and `run` commands; streaming source filter and explicit subprocess argv.
- `revision/scripts/evidence/est_blastx_fixture.py`: synthetic **real DIAMOND** integration gate; no stub engine and no production-input bypass flag.
- `revision/scripts/evidence/est_blastx_fixture.sbatch`: fixture only, CPU account/partition `cpu-s1-pgl-0`, 4 CPUs, 8G, 30 minutes, explicit GPFS temporary directory, no array.
- `revision/scripts/tests/test_est_blastx.py`, `test_est_blastx_fixture.py`: launcher and independently hand-derived traceback contracts.

Reuses `64_orthodb_filter.py` through importlib: FASTA/header helpers, `read_lineage`, `excluded_taxids`, and merged-ID-aware `fetch_lineage`. Preparation scans all observed taxids and copies an optional existing cache into a **new** work directory before explicit fetching. Search never fetches taxonomy. Every observed taxid needs its own ID in its frozen lineage and a root/cellular-organisms ancestor; these structural checks do not prove biological completeness, which remains a reviewed-cache requirement.

Search checks expected source/query/lineage/role SHA256, source basename, the exact nine species, primary screened query path, A21 DONE/JSON summary, a unique matching training-eligible dataset role, and actual nucleotide records >=100 nt. Rice uses species root **4530**, not 39947. Source and lineage hashes are checked around filtering and all four reviewed inputs again after search. Preserve original inputs in immutable storage; the launcher does not archive a second copy of the large EST query file.

Each invocation handles one species and one source. It retains full source FASTA headers/IDs and builds a fresh species-filtered DB before searching; it never uses historical `.dmnd` accession selection or `--seqidlist`. Existing A19 exact-match exclusions remain intact because production OrthoDB starts from the previously filtered FASTA.

Both build/search use `--no-parse-seqids`. Search tokens include `--frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25 --max-hsps 1`. Raw13 columns are:

```text
qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop
```

Raw 1-based endpoints are unchanged. Short HSPs are counted, not removed: min-ORF does not impose the downstream >=30-aa HSP acceptance rule. No phase rasterization or genomic projection is implemented.

Outputs include filtered FASTA, fresh DB, raw hits, exact commands, engine logs, per-taxid/sequence retained/excluded counts, source/lineage/FASTA/DB/query/role/output hashes, engine version/binary hash, and script/helper hashes. Existing directories/files are refused; errors leave `FAILED.json`, never `DONE`. TERM/INT terminates the engine process group. Uncatchable SIGKILL/OOM can leave an unfinished directory without FAILED; **absence of DONE always means incomplete**. Failed directories are retained for diagnosis and never resumed/overwritten.

## Verified OrthoDB source

Read only `build_dbs.sh` using SSH `BatchMode=yes`; no scans, hashes or jobs on the login node. The build command names:

```text
/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein/odb12_Viridiplantae.filtered.fa.gz
```

Swiss source:

```text
/data/gpfs/assoc/pgl/data/Transgenic/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz
```

The remote shell lacked `python3` on PATH during the first read attempt; shell built-in line reading succeeded. Fixture wrapper therefore names an explicit interpreter already referenced elsewhere in this repository: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python`. Controller must verify that executable on compute before submission.

## Test evidence

Local Python: **3.8.20**.

```sh
python3 -m pytest -q tests/test_orthodb_filter.py \
  revision/scripts/tests/test_est_blastx.py \
  revision/scripts/tests/test_est_blastx_fixture.py
# 22 passed in 0.27s
python3 -m py_compile revision/scripts/evidence/est_blastx.py \
  revision/scripts/evidence/est_blastx_fixture.py
bash -n revision/scripts/evidence/est_blastx_fixture.sbatch
git diff --check
# all exit 0
```

Observed red→green cycles: seven initial launcher tests failed before implementation; four fixture/replay tests failed before fixture implementation; missing-wrapper test failed before wrapper creation; SIGTERM regression failed because an engine descendant survived, then passed after process-group cleanup; valid NCBI cellular-root lineage failed the overstrict taxid-1 check, then passed after correction. Additional tests cover guard success/failure, short queries, empty filtered source, overwrite refusal, subprocess nonzero propagation, and the existing merged-taxid resolver. The process-control tests use real Python child processes, **not DIAMOND**, and are not reported as engine evidence.

**Real DIAMOND: NOT RUN.** `command -v diamond` found no executable locally. No mock/stub output is counted as a real-engine pass. CLI/version/fixture alignment behavior must pass on the specified 2.1.24.178 installation before a production pilot.

The fixture creates a deterministic diverse 180-aa protein, same-species/root/descendant/merged subjects and a retained cross-species subject, separately in Swiss/Ortho header forms. Queries: ordinary, reverse complement, +1 nt, -1 nt, 3-aa insertion, 3-aa deletion. Each source must yield all six positive controls. `getseq` verifies the built DB contains only the retained full subject ID and expected sequence.

Rich17 adds `cigar qseq_gapped sseq_gapped qseq_translated`. BTOP replay checks input codons, subject residues, both endpoints, initial frame/strand, CIGAR, gapped strings, translated string, edit-spanning anchors, and expected internal frameshift direction/count. A nonempty hit file alone cannot pass. Exact v2.1.24 upstream source was narrowly inspected to avoid guessing syntax: `src/basic/match.h`, `src/output/blast_tab_format.cpp`, `src/output/sam_format.cpp` at `https://github.com/bbuchfink/diamond/tree/v2.1.24`. Frameshift BTOP pairs are `\-` (+1 nt) and `/-` (-1 nt); real-engine characterization is still pending.

## Proposed fixture submission — controller only

Stage the reviewed branch in a **new isolated remote checkout**, including `revision/scripts/64_orthodb_filter.py`; do not touch the running seed123 checkout. Proposed exact paths:

```text
checkout: /data/gpfs/assoc/pgl/data/Transgenic/worktrees/b6-est-blastx-reviewed-20260914
output:   /data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx_fixture/20260914_review01
tmp:      /data/gpfs/assoc/pgl/data/Transgenic/tmp/b6_est_blastx_fixture
logs:     /data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx_fixture/logs
```

Controller creates tmp/log parent directories, verifies checkout/helper/interpreter and engine version, and confirms the output directory does not exist. **Not submitted by implementer:**

```sh
sbatch \
  --chdir=/data/gpfs/assoc/pgl/data/Transgenic/worktrees/b6-est-blastx-reviewed-20260914 \
  --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx_fixture/logs/%x_%j.out \
  /data/gpfs/assoc/pgl/data/Transgenic/worktrees/b6-est-blastx-reviewed-20260914/revision/scripts/evidence/est_blastx_fixture.sbatch \
  /data/gpfs/assoc/pgl/data/Transgenic/worktrees/b6-est-blastx-reviewed-20260914 \
  /data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx_fixture/20260914_review01 \
  /data/gpfs/assoc/pgl/data/Transgenic/tmp/b6_est_blastx_fixture
```

Require allocation/batch/extern exit status plus top-level and both per-source DONE, provenance, logs, raw17, replay results and DB getseq outputs. A fixture failure is a blocking characterization result, not permission to bypass checks.

## Remaining production requirements and stage paths

1. Controller review of this diff and successful real fixture. Fixture allocation is **not** a production resource estimate.
2. On compute: verify the existing source checksums/provenance and approved A19 filtering history. Historical DB job6163304 had an extern OOM despite batch success; existing DONE alone is insufficient. This launcher rebuilds from FASTA, not those DB files.
3. Prepare complete frozen lineages for **every observed subject taxid**, separately per source. Copy the old GPU cache only under controller oversight, or explicitly fetch into new candidate directories. Review lineage provenance/completeness and freeze SHA256 before filtering/search. No live descendant snippets.
4. Verify/query-hash each `evidence/est/{SP}/univec/est.univec.fa.gz`, DONE/summary and `evidence/DATASET_ROLES.est_v1.tsv`. Retain immutable source, query and lineage files for later BTOP reconstruction. Expected hashes must come from reviewer-approved compute preflight, not be silently calculated as approval inside the launcher.
5. Suggested new sequential stage paths under `evidence/est_blastx/`: `lineages/orthodb_review01`, `lineages/swissprot_review01`, then `pilot01/{SP}/orthodb` and `pilot01/{SP}/swissprot`; every directory must be new. A candidate's usable cache is `<lineage-stage>/lineage.tsv` and its hash is in `<lineage-stage>/provenance.json`.
6. Compute-stage command shapes (explicit values/hashes/resources supplied by controller):

```sh
python est_blastx.py prepare-lineage \
  --source-kind orthodb --source APPROVED_PREFILTERED_FASTA \
  --source-sha256 REVIEWED_SOURCE_SHA256 \
  --cache COPIED_COMPLETE_CACHE --out-dir NEW_LINEAGE_STAGE
# Optionally add --fetch-missing here only, with explicit approval.

python est_blastx.py run \
  --species Athaliana --source-kind orthodb \
  --source APPROVED_PREFILTERED_FASTA --source-sha256 REVIEWED_SOURCE_SHA256 \
  --query SCREENED_PRIMARY_EST --query-sha256 REVIEWED_QUERY_SHA256 \
  --lineage FROZEN_LINEAGE_STAGE/lineage.tsv --lineage-sha256 REVIEWED_LINEAGE_SHA256 \
  --roles DATASET_ROLES.est_v1.tsv --roles-sha256 REVIEWED_ROLES_SHA256 \
  --diamond /data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond \
  --threads APPROVED_THREADS --tmpdir EXISTING_GPFS_TMPDIR --out-dir NEW_SPECIES_SOURCE_OUTPUT
```

7. Measure source scan/filter/makedb/search elapsed time, peak memory, temporary disk and query throughput in a controller-approved sequential pilot before scheduling more species. No nine-species array and no guessed 48-hour production allocation supplied here.
8. Later increments own >=30-aa acceptance, BTOP-derived internal frame blocks, genome projection, leakage closure and A37 paired reporting. This increment does not claim those are complete or freeze A46.

## Scoped reviewer-blocker correction — getseq CLI

Based on detached `cc9735f` (`pi-agent-462c26cc-1192-466`), not main, in isolated worktree `/tmp/pi-agent-8d1bf61b-edb0-445-b3eba4d3`. Removed `--no-parse-seqids` from the fixture's `getseq` argv only: DIAMOND v2.1.24 does not support that option for this subcommand. Both `makedb` and `blastx` flags, the full-ID/sequence DB assertion, search-ID checks, traceback validation and all production guards are unchanged.

Added `TracebackTests.test_getseq_argv_uses_supported_options`, which runs fixture setup/filtering and intercepts the emitted getseq argv at the execution boundary, requiring the executable, subcommand, DB and output arguments with no extra option. Engine-version/build/search calls are patched solely to reach this boundary; a sentinel stops execution before database inspection. This is an argv contract test, not a real-engine pass or a replacement for either source's full-ID assertion.

Exact local covering commands/results (Python 3.8.20):

```sh
python3 -m pytest -q revision/scripts/tests/test_est_blastx_fixture.py::TracebackTests::test_getseq_argv_uses_supported_options
# Before fix: 1 failed in 0.30s, exit 1.
# Intended assertion: first extra argv element 6 was '--no-parse-seqids'.
# After fix: 1 passed in 0.04s, exit 0.
python3 -m pytest -q tests/test_orthodb_filter.py revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py
# 23 passed in 0.62s, exit 0 (previous 22 tests plus new regression).
python3 -m py_compile revision/scripts/evidence/est_blastx.py revision/scripts/evidence/est_blastx_fixture.py
bash -n revision/scripts/evidence/est_blastx_fixture.sbatch
git diff --check
# All exit 0.
```

No remote jobs or real DIAMOND execution performed for this correction. Controller scoped re-review and successful real fixture on 2.1.24.178 remain required before a production pilot. Production prerequisites are unchanged: complete frozen lineages for every observed taxid, independently reviewed for provenance/completeness; independently approved input identity/checksums for source, screened query, lineage and dataset roles, plus A19 filtering history and A21 screening evidence. No validation has been relaxed and no input identity or lineage approval is inferred from these local tests.

## Scoped version-guard correction — job 6163393

### Inline root-cause review before implementation

Inspected detached base `071594049b73152778b58cc2908cf5eb539d71d8` in isolated worktree `/tmp/pi-agent-cb2fd26a-e84c-456-21f9f94f`. `engine_version()` actually uses exact string equality (not a substring test), requiring `diamond version 2.1.24.178`. Both production `run()` and `run_fixture()` call it before building/searching. Existing launcher tests do not exercise this guard; the getseq contract test patches it out.

Controller-supplied real-job evidence (not independently executed here): job6163393 failed in this guard before search. The SAME binary reports `diamond version 2.1.24` for `version`, and first help line `diamond v2.1.24.178 (C) Max Planck Society ...`; reported binary SHA256 is `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`. Sacct: batch FAILED 1:0, MaxRSS1496K; extern OUT_OF_MEMORY 0:125, MaxRSS1480K. The primary logged failure is the version guard; these statuses do not establish a memory-sizing cause and no resource increase is proposed.

Minimal correction planned: parse the version response and first help banner from the same explicit executable, require the exact reviewed four-part build `2.1.24.178` in help and a matching three-part release (or identical four-part build) in version, reject malformed/incompatible/conflicting tokens, and retain both identity lines in existing `diamond_version` provenance. Command failures remain failures. Add realistic command-output regression coverage before implementation; no fixture alignment-rule or Slurm changes.

### Reproduced evidence and correction

Local regression uses a clearly labeled executable CLI double, not DIAMOND: `version` emits `diamond version 2.1.24`; `help` emits `diamond v2.1.24.178 (C) Max Planck Society for the Advancement of Science` followed by usage text. The guard's actual subprocess path is exercised. Before implementation this reproduced `ValueError: unreviewed DIAMOND version: 'diamond version 2.1.24'; require 2.1.24.178`. Two further regression methods exposed acceptance of a conflicting help build and failure to check a failing help command.

The fix uses anchored token parsing, not substring matching. It always checks both `[absolute_diamond_path, 'version']` and `[absolute_diamond_path, 'help']`, captures stdout/stderr, and propagates nonzero exits. Only release `2.1.24` or full build `2.1.24.178` paired with help build exactly `2.1.24.178` is accepted. Existing `diamond_version` provenance remains a string, now containing the version response followed by the help's first identity line; existing binary SHA256 recording is unchanged. Full help usage text is not stored. No fixture code, alignment assertions, search parameters or resource settings changed.

Exact local commands/results, Python 3.8.20:

```sh
python3 -m pytest -q revision/scripts/tests/test_est_blastx.py -k engine_version
# RED: 3 failed, 14 deselected in 0.26s (exit 1), before production edit.
# GREEN: 3 passed, 14 deselected in 0.75s (exit 0).
python3 -m pytest -q tests/test_orthodb_filter.py revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py
# 26 passed in 1.25s (exit 0).
```

Coverage includes compatible short/full versions, stdout/stderr banners, incompatible releases, conflicting full builds, prefix collisions (`.1780`), suffixed builds, absent/malformed/non-first banners, extra version tokens/lines, nonzero version/help commands and invalid executable paths. `command -v diamond` found no local executable: no real binary/hash verification or alignment run is claimed here, and no remote submission occurred. The controller-provided job evidence above remains distinct from these local contract tests. Ready for scoped diff review, then controller-only real fixture rerun in a fresh output directory; real-engine fixture success is still required before any production pilot.

Additional checks: `python3 -m py_compile revision/scripts/evidence/est_blastx.py revision/scripts/evidence/est_blastx_fixture.py` and `bash -n revision/scripts/evidence/est_blastx_fixture.sbatch` exited 0. Initial `git diff --check` flagged a trailing blank line in this report; removed before final verification. Commit author: Won Yim `<wyim@unr.edu>`.
