# Independent EST bounded-sample review — 2026-09-17

## Verdict: Important reproducible fixes required before exact deployment acceptance

No Critical scientific false-approval was found: all examined paths keep A11 unassessable and training/full-C0/route-A/molecule completeness false. The core selected-member extraction/replay design is suitable for this first Sitalica diagnostic step. However, search/HSP provenance identity and the claimed process-group cleanup have reproducible gaps. Reader stderr also fails the requested precise error-evidence requirement. Fix/recheck the narrow issues below; do not rerun existing searches/alignments or broaden to nine species.

Reviewed `/tmp/est-real-source-binding-20260917`, HEAD `fca911e58fde44124b9359f9adc03a9d20059599`, including the complete new binder, complete 31-case test file, builder, actual manifest, handoff and Slurm proposal. The new files are untracked, not part of that HEAD; exact review-time hashes are in `est_sample_review_evidence/metadata_and_source_ledger.txt`. No reviewed source, index or HEAD was changed. Existing `.gitignore`/`.ignore` state was preserved. No remote command, job, real large-file scan, source search, alignment, or subagent was run. Local tests used tiny synthetic files only.

## Findings

### I1 — Important: different search provenance produces identical search and HSP IDs

`revision/scripts/evidence/est_real_sample.py:288-290,389-393` hashes species/arm/reference/source and raw/protein/query bytes into `search_id`, then derives HSP IDs from that ID plus raw-row ordinal/hash. Neither `c1_provenance` nor `c1_commands` is consumed here. With all sequence/raw bytes unchanged, two different SHA-pinned provenance/commands files (different engine identity/search settings) yield COMPLETE runs with **identical search and HSP IDs**. Raw receipt validation succeeds in both runs.

Reproduction: `test_reproduce_search_and_hsp_identity_collision`; recorded in `adversarial.txt`. The canonical manifest hashes differ. Thus the complete report/manifest bundle *does* retain enough information to distinguish these two runs; this is not loss of all provenance. But the advertised standalone search/HSP ID is not a full-search-contract identity. A downstream join/deduplication by those IDs would collapse different search provenance. Manifest binding points from report to artifacts, not from the search ID back to a unique manifest, so it does not repair that identity contract.

Minimal fix: add the verified `c1_provenance` and `c1_commands` SHA256s to the search preimage, validate those references, and retain their preimages. The pinned provenance already contains DIAMOND hash/version, database hash, launcher and input lineage hashes, so those digests transitively bind engine identity without rereading/rerunning DIAMOND. Alternatively explicitly redefine identifiers as content-only and require `(manifest_sha256, search_id/hsp_id)` everywhere, but that changes the stated full-HSP identity contract and is less direct. Regression should require differing search/HSP IDs when provenance alone changes.

### I2 — Important bounded-cleanup gap: descendants survive if the group leader has exited

`est_real_sample.py:438-445`: cleanup calls `killpg` only when `proc.poll() is None`. A helper that spawns a child inheriting stdout and exits immediately leaves the parent reader blocked awaiting EOF. The wall alarm fires; report becomes INCOMPLETE, but `poll()` sees the exited leader and skips killing the still-live process group. Our fixture verified the descendant was alive after return, then explicitly killed it. This contradicts the unconditional process-group cleanup claim.

Reproduction: `test_reproduce_exited_parent_descendant_cleanup_gap`. Normal pinned `samtools view` is not demonstrated to spawn descendants, so this is a boundary/cleanup defect, not evidence that the proposed actual command currently leaks workers.

Minimal fix: attempt process-group cleanup on abort even if the leader has exited, tolerate `ProcessLookupError`, then reap the leader; ensure cleanup exceptions cannot bypass status publication. Keep the regression bounded and always clean its child. Existing timeout tests cover only a sleeping live leader, not this case.

### I3 — Important diagnostic requirement: actual reader errors are discarded

`est_real_sample.py:343` directs samtools stderr to DEVNULL. A reader that writes a unique corruption diagnostic and exits 7 produces only `samtools/helper nonzero exit: 7`; none of the actual error bytes survive. The FAILED status is honest and cannot become a biological negative, but it cannot establish the requested precise source-bound failure cause. Slurm stderr cannot recover bytes already discarded inside this subprocess.

Reproduction: `test_reproduce_stderr_loss`. Minimal fix: concurrently drain stderr into a bounded retained diagnostic buffer/file, record truncation/byte count/hash, and charge retained/output usage. Do not replace DEVNULL with an undrained PIPE (deadlock) or an unlimited log (cap bypass). No extra search/realignment is needed.

## What passed and what the pins actually establish

- Regenerated the manifest in memory from the controller's exact local `est_source_metadata.txt`; it equals the checked-in-to-handoff JSON object. Exact manifest SHA256 is `7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c`; capture SHA256 is `f814b5665f60421826eec838bd259f3728ec793d58e1c6ea4a3aa4870769b998`.
- Checked both provenance and commands SHA256s against captured text; both raw/protein SHA256s and both provenance query hashes match manifest pins. Historical original/reference/BAM/screened MD5s and paths agree with captured alignment provenance. The captured samtools reader SHA/version agrees with the manifest; original producer 1.3.1 is correctly separate from reader 1.21/htslib 1.23. These are **capture-to-manifest checks, not fresh remote-file verification**.
- Runtime hashes every declared file to EOF, checks provided SHA256/MD5, then checks digest receipt fields in SHA-pinned metadata. `receipt_bound`, `fresh_whole_file_verified`, `expected_hash_validated` and `extracted_bytes_verified` are distinct. The screened SHA receipt is to orthodb provenance; the builder also checks SwissProt query SHA agreement. UniVec report intentionally has no historical pin and cannot claim historical authenticity; its fresh hash and sequence joins are appropriate for this bounded diagnostic scope.
- First distinct orthodb QNAME prefix selection has no outcome filtering. Both raw sources are traversed to EOF, retaining every selected row, competitors and below-30 HSPs; raw exact bytes and ordinals survive later join failures. Protein/reference FASTAs are streamed, not loaded wholesale. Matching protein/EST IDs are exact; duplicate selected IDs and missing targets fail closed.
- UniVec relation names derive from report intervals, including one-survivor unsuffixed splits, not suffix stripping. Pinned original subsequence, screened sequence length and UniVec header must agree. This is member identity, not molecule/library/mate identity.
- Every selected SAM row and header is retained, including nonprimary/unmapped diagnostics. Successful reader exit and exact full reference/SAM contig-length dictionary agreement are required. Primary SEQ is mandatory here; existing projector checks reverse complement, FLAG range, full hard/soft-clipped CIGAR geometry, positions, contig bounds and screened identity. Duplicate ts and missing SEQ are explicit geometry failures; below-30 HSP length rejection is distinct from geometric success. Noncanonical bases/residues are source-bound replay failures rather than invented biological negatives.
- Existing -1 frameshift self-overlap stays explicitly `self_overlap_policy_unresolved`; no conflict resolver, production A11 or route-A path is invoked. Missing placements remain a diagnostic, not a biological zero. Stored BAM EOF is not completeness of all possible placements (historical command used `--secondary=no`).
- The 40 GiB charged-read, 256 MiB output, 1M expansion, retained/record/line/sequence limits are enforced fail-closed. Full BAM pass is charged in addition to hashing and decoded SAM; BTOP expansion is charged before expansion rather than trusting reported HSP length. Limits are application accounting, not an exact RSS or physical-I/O meter; Slurm remains the hard memory/time boundary.
- Mutation checks compare device/inode/size/mtime/ctime across hash/read/finalization; tested mutation never yields COMPLETE. Hash/stat checks presume ordinary immutable inputs, not protection from an adversarial privileged filesystem writer. No output reuse is allowed. Diagnostics hash -> atomic report -> last marker containing report SHA is sound for normal filesystem operation, and was independently checked. Filesystem failure/SIGKILL can prevent any final marker; controller must treat missing/mismatched markers as INCOMPLETE, never infer success from partial diagnostics.

## Deployment conditions (not implemented or guaranteed by this review)

1. **Immutable full package ledger and runtime verification.** `est_real_sample.py:261-263` records only the current on-disk `est_hsp_contract.py` hash. It does not bind the binder itself, projector, expanded interpreter/genetic code, parser, or loaded module bytes. The local relative import closure is five files (listed in the evidence ledger). Package initializer if present, launcher, manifest, wrapper and controller ledger also need binding. A HEAD alone does not identify these untracked changes.
2. Minimal controller wrapper: stage only reviewed closure + manifest into a new content-addressed, non-mutating directory; bind its ledger digest externally; check all file SHA256s before Python starts; invoke the absolute captured Python with isolated imports/no user-site or CWD shadowing; check loaded module origins against that directory; record actual interpreter/version/hash and reader/library environment; run once; verify the same ledger after exit, including failure paths. Publish a separate controller acceptance receipt only after package/runtime checks and report/diagnostics/marker hashes agree. Preserve the worker's original report. A wrapper that uses `exec` without a post-run verifier cannot meet the after-check requirement. Review-time `metadata_and_source_ledger.txt` is **not** such a deployment guarantee.
3. **Clean environment.** Current proposal resets PYTHONPATH and suppresses bytecode writes, but does not neutralize inherited PYTHONHOME, user site, CWD import shadowing, BASH_ENV, LD_PRELOAD or library environment. Use a controlled job environment, absolute interpreter/reader, fixed package root, and verify actual imports. Do not activate/change global environments, profiles, B5 or MVP work. Stdlib-only code does not by itself guarantee module/runtime identity.
4. **One bounded compute run only.** Controller supplies `--no-requeue --nodes=1 --ntasks=1`, account/partition and exclusive log/output locations, preserving proposed 2 CPUs, 8 GiB and 30 minutes/internal 1740 seconds. The sbatch file does not itself set no-requeue/nodes/tasks. No auto-resubmission, cap increase, login-node scan, new DIAMOND run or alignment. Captured orthodb FASTA size is 5,547,417,703 bytes; scanning belongs only inside the authorized allocation. Keep failed partial output and record exact job/deployed hashes.
5. After fixes and source recheck, actual compute results still require observed selected names/counts, raw EOF counts, fresh pins/receipt checks, successful reader EOF, dispositions and actual replay+geometry or a precise source-bound failure. A11 remains unassessable; route A remains false. Nothing here certifies production acceptance or requires the full nine-species campaign.

## Executed local evidence

`est_sample_review_evidence/focused309.txt`: **309 passed in 22.70s** using `/tmp/mvp-final-source-env/bin/python` (Python 3.12 environment) with `PYTHONDONTWRITEBYTECODE=1`, `pytest -p no:cacheprovider`, and these exact files:

```
tests/test_est_hsp_contract.py
tests/test_est_phase_conflicts.py
tests/test_est_approved_contract_integration.py
revision/scripts/tests/test_est_hsp_expanded.py
revision/scripts/tests/test_est_phase_project.py
revision/scripts/tests/test_est_molecule_closure.py
revision/scripts/tests/test_est_blastx_fixture.py
revision/scripts/tests/test_evidence_support.py
revision/scripts/tests/test_univec_trim.py
tests/test_est_real_sample.py
```

This is 278 existing focused cases plus 31 new binder cases, **not 31 actual-data validations**. Independent `est_sample_review_evidence/test_adversarial.py`: **10 passed in 5.76s**, covering three reproduced defects plus SAM boundaries, output cap/marker integrity, MD5 receipt match/mismatch and CLI manifest refusal. Defect tests intentionally assert the observed bad behavior; their passing is not a repaired implementation. Local pytest fixture artifacts remain under its temporary run directories.

Initial review harness attempts used a mistaken UniVec filename, then default Python 3.8 (incompatible with one existing test module), and system Python lacking pytest; the final run uses the available 3.12 test environment, without dependency installation. `adversarial_initial.txt` preserves an initial reviewer-test failure: the searched error literal appeared in the recorded command itself; the corrected fixture constructs stderr dynamically and confirms actual emitted bytes are lost. No implementation change was made to get green results. `git diff --check` passed; worktree status/HEAD remained unchanged. Advisor consultation was attempted once but unavailable due client/model-version incompatibility; no advisor endorsement is claimed.

Graft-first context was used in the selected worktree (reported cumulative estimate: 910,923 tokens saved). Source corrections, deployment and actual-data acceptance remain controller work, not completed review actions.
