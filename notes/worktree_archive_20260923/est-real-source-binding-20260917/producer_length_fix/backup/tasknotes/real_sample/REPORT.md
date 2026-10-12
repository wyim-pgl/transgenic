# Actual-source sample binder — implementation handoff

## Latest narrow review revision

I1/I2/I3 have been addressed locally; see [REVIEW_FIXES.md](REVIEW_FIXES.md) for precise behavior, red/green evidence, current source hashes and limitations. The requested focused baseline plus 12 new regressions is **321 passed**; unchanged reviewer boundary checks are **7 passed**. Manifest SHA256 remains unchanged. No commit/deployment/job occurred; controller independent verification and real-data acceptance remain pending. The implementation description below is the original handoff, superseded where the linked revision specifies stronger provenance binding, unconditional abort group cleanup and bounded stderr capture.

Owned files: `revision/scripts/evidence/est_real_sample.py`, `tests/test_est_real_sample.py`, and this directory only. No commits, deployment, remote writes, jobs, alignments, or searches. Existing HSP/projector/conflict/molecule/trainer modules unchanged. Independent advisor was attempted but unavailable (client/model-version incompatibility); controller review remains required.

## Current status

Implemented actual streaming paths, not a synthetic-only primitive. Local file integration tests invoke a test-only explicit Python SAM-reader command; this hook is absent from CLI/manifest. Production CLI requires `SLURM_JOB_ID`, a SHA256-bound manifest, a fresh exclusive output directory, and a manifest-SHA256-pinned absolute samtools binary. Production command is exactly `[samtools, 'view', '-h', bam]`, no shell.

Actual source compute replay **not yet run**. COMPLETE means bounded selected-QNAME traversal only, not successful biology or training approval. Source-bound noncanonical DNA/protein (including X), trace failures, missing geometry, self-overlap -1 events, and nonprimary records remain explicit diagnostics. Reported length comparison, validated whole-HSP length acceptance, and geometry success are distinct. Nothing calls the conflict resolver, invents molecule/library/mate identities, or sets complete-HSP/molecule gates. A11 is unassessable; training_eligible/full_c0/raster/molecule_counts_complete/route_a_complete are false.

## Manifest and runtime

`build_manifest.py` consumes the controller's exact path→text capture and writes exclusively. `sitalica.manifest.json` was generated from main `notes/seed123_execution_20260917/est_source_metadata.txt`, not hand-invented hashes.

Exact manifest file SHA256: `7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c`.

- Python: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11`, captured version 3.11.14. Module is stdlib plus existing evidence modules; no new dependencies. Syntax compatible with Python >=3.8; 3.11 preferred, tests run on 3.12.
- Read-only BAM reader: same environment's `samtools`, version 1.21 / htslib 1.23, SHA256 `a579aed31a13841cc0950a487b59729355d0b9fbcc39293e0b90b960133217bc`.
- Original BAM producer samtools 1.3.1 remains separately declared, not claimed identical to reader.
- Proposal: `compute_proposal.sbatch`, 2 CPUs / 8 GiB / 30 minutes, internal deadline 1740 s. Controller supplies account/partition/log paths and deploys only after review.

After deployment (not executed here):

```bash
sbatch [site options] compute_proposal.sbatch CODE_ROOT ABS_MANIFEST \
  7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c ABS_NEW_OUT
```

## Source closures and selection

Frozen selection: first 16 distinct exact QNAMEs in the first at most 10,000 physical orthodb raw lines (stop once 16 found); fewer if prefix/EOF has fewer. No outcome filtering. Actual prefix line count/hash and ordered selected names are recorded.

Both raw13 files are scanned to EOF, retaining every selected row, including competitors and below-30 HSPs. Raw physical ordinal/exact base64 bytes/hash are emitted immediately, before downstream joins, so later failure preserves partial extraction. Full-file SHA256 is matched against captured C1 provenance. Search records link the source/arm/species/reference/query/protein/raw digests; provenance/commands files are themselves freshly SHA-checked against capture.

Both filtered protein FASTAs are fully streamed and hashed (5.55 GB orthodb / 21 MB SwissProt), selecting only requested exact IDs. No whole-protein record list. Selected duplicate/missing protein, screened, original IDs and ambiguous/missing report joins fail closed. Report-produced naming derives from retained interval count, so a one-survivor split keeps its unsuffixed accession; multi-part relationships are not inferred from suffix alone. The selected screened sequence and UniVec header must agree with the report interval and pinned original subsequence.

BAM is freshly SHA256/MD5 hashed, MD5 bound to alignment provenance. Whole `samtools view -h` is consumed without materializing whole SAM. Every selected QNAME record and all headers are retained; primary FLAG/POS/CIGAR/SEQ/ts and actual screened identity feed existing projection. Reference FASTA is freshly SHA256/MD5 hashed and its complete contig lengths must match the SAM dictionary. Only successful EOF, successful subprocess exit, and final unchanged source stats permit selected-search/BAM completeness flags. This is completeness of stored source files, **not** of all possible alignments/secondary placements or full A11.

Historical pins: reference MD5 `d8f77d532d8c87f314f6175146fa6728`; original EST MD5 `4b0f262bc3a0a72f9e290035ac3c67ff`; BAM MD5 `01b75b0ddd6ccbb7759bf781e7c1b4b6`. SHA values for raw/protein/query come directly from captured JSON. UniVec report has no captured historical hash: manifest contains no placeholder; fresh hash/stat and pinned-FASTA cross-checks are recorded, with `receipt_bound=false`. This does not certify historical report provenance or molecule closure.

States are separate: receipt_bound (only after verifying pinned metadata field), fresh_whole_file_verified (fresh EOF digest), expected_hash_validated, extracted_bytes_verified. Before/after device/inode/size/mtime/ctime are retained. Member/search/HSP/placement IDs hash domain-separated canonical JSON and retain preimages, including species/reference/source/arm.

## Bounds and outputs

At most 40 GiB charged reads (whole hash passes + extraction passes + external BAM pass + decoded SAM), 256 MiB output, one million charged expanded base occurrences, 1 MiB physical lines, 1 MiB selected sequences, 32 MiB charged retained payload and 10,000 selected raw/SAM records. Expansion charge follows BTOP tokens before replay, not untrusted reported length; projection charges its events again conservatively. Limits may be reduced, never raised. Output limit must be at least 2 MiB for final-status reserve. Reference sequences are length-streamed, never retained whole.

Output is streaming `diagnostics.jsonl`, atomic `report.json`, then a last-written COMPLETE/FAILED/INCOMPLETE marker containing report SHA256. Report binds diagnostics SHA256 and exact CLI manifest-file SHA, separately from canonical object SHA. Cap/timeout/nonzero-reader/mutation failure keeps partial outputs and never sets complete flags. SAM subprocess process group is killed/reaped on interruption. No automatic rerun or overwrite.

## Verification and remaining acceptance

TDD observed: initial 9 failures because binder module absent; then green. Receipt mismatch regression failed COMPLETE vs required FAILED before receipt validation fix. Trace-expansion regression failed COMPLETE vs required INCOMPLETE before switching budget enforcement to actual BTOP tokens. Additional realistic fixtures cover joins, 1-survivor/parts, selected duplicates, missing targets, report mismatch, N/X, invalid trace, -1 overlap, below30, reverse SEQ identity, nonprimary/no-placement, caps, nonzero EOF, timeout cleanup, source mutation, output collision, scope/domain ID collisions, CLI compute enforcement and unknown A11.

Latest combined verification: `verification.txt`; 278 existing baseline tests + 31 binder tests = **309 passed**. `git diff --check` passed. Exact baseline includes root HSP/conflicts/integration and revision HSP-expanded/project/molecule/blastx-fixture/evidence-support/UniVec tests. This is not real-data compute acceptance.

Pending controller actions: independent review; deploy only owned files after approval; submit the bounded proposal; inspect actual selected QNAME/raw counts and dispositions, verified EOF/hash closure and at least actual replay+geometry or precise source-bound failure. Do not treat synthetic success as actual-source scientific acceptance. No full C0/A11/route-A certification is requested or produced.
