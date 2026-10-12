# Independent bounded ONT fixture review

## Disposition

**Approve the bounded fixture design, conditional on the narrow operational amendments and runtime/containment evidence below. The present candidate is not submission-ready. ONTF4 real-tool execution remains NOT DONE.** This is not a controller authorization, full ONT approval, or resolution of the historical externOOM gate.

The mandatory dedicated 256-MiB filesystem hard quota is an overstrict, newly self-imposed gate for this seven-read, trusted-tool fixture. It is not necessary to preserve the scientific contract. Removing it requires an explicit controller amendment, an honest replacement approval schema, corresponding fixture/plan/sbatch changes and a newly retained manifest hash. Do not put an invented quota value in the existing approval to make line 244 pass.

## Review evidence and limits

Read the complete candidate PLAN.md, fixture.py, fixture.sbatch, runtime.json, manifest, test_fixture.py and stub_tool.py; complete package gate.py/wrapper.py/source_pins.json; complete executed frozen driver, sampler, orientation audit, junction CLI, mvp_windows and 59_evidence_support.py. Also inspected mvp_bootstrap.py, which is not executed by this fixture. No full statistics/scoring invocation is part of this test. Exact nine-file closure and all 15 manifest entries verified. No source or fixture edits, remote commands, installations, jobs or real alignments were performed.

Review-only runner changes the tests' temporary-directory destination to review/ and nothing about their assertions or fixture paths. Bytecode writes are disabled. All candidate files outside review/ have identical SHA256 before/after tests, retained in identities.json. The test runner has 90-second external wall timeout, 60-second per-process CPU, 1-GiB address-space and 4-MiB file limits.

- `/usr/bin/python3 -I -B review/run_local_tests.py`: six tests passed, 8.660 seconds, exit 0. See local_tests_system_python.txt. Local interpreter is Python 3.10.12, not the approved remote runtime.
- Initial PATH Python was 3.8: four passed, two import failures at `tuple | None`. See local_tests.txt. This is not real-runtime evidence. The fixture's advertised >=3.9 check is too permissive: frozen annotations require >=3.10. Correcting that guard/message is a small accuracy fix, not a blocker for the already pinned 3.11 runtime.
- `sha256sum --strict -c MANIFEST.sha256`: all 15 pass, manifest_check.txt. `bash -n fixture.sbatch`: exit 0.
- Advisor consultation failed because the installed client does not support its configured model. No advisor approval is claimed.

The six tests do not execute `fixture.run`, its approval/resource checks, production `wrapper.execute`, timeout failures or final receipt publication. The stub substitutes text for BAM/index and scripts for all three tools. Passing tests establish local plumbing only.

## Frozen path and scientific boundary

`fixture.py:136-222` genuinely calls the frozen CLI main functions and unchanged `gate.verify_alignment`. Read validation, reference byte checks, window geometry, selection, audit rules, alignment commands, junction parser and thresholds are reused, not reimplemented.

The panel seam is process-local and delegates to the frozen explicit-hash loader, preserving schema, counts and duplicate checks. The reference seam deliberately replaces Git92 authority with synthetic authority, while retaining actual FASTA/GFF MD5 comparisons. The enum-shaped read root passes actual role/path/compressed-MD5 checks but is not accession evidence. The junction bootstrap changes its OS argv, and that limitation is correctly declared. The postflight's second driver load shares the same imported windows module, so the seam remains effective. Default frozen panel rejection is tested in a fresh process.

The two-row precommitted oracle is sound for this scope: inclusive junctions 20500..20699 with three distinct ONT names and support=true; 30500..30699 with two and support=false. Exact equality excludes other rows and the +9 decoy. Six selected names include the unspliced read and exclude the outside-contig read. Do not alter this oracle after observing aligner output. The table oracle plus selected-name check does not independently prove that the unspliced read was accepted by the BAM parser rather than filtered later. Any claim about its actual acceptance should additionally cite the real BAM/QC, or remain limited to selection and absence of extra junction rows.

Orientation must remain UNRESOLVED/exit 4/no-uf with min_eligible=1000 and threshold=.95 unchanged. Overlapping 50 test IDs are not 50 biological loci. The absence of HiFi input is not biological HiFi zero. Raw fixture-bound `panel_sha256_is_frozen=true` is not canonical attestation.

The junction CLI uses `sam_lines`, `parse_sam`, whole-read `alignment_rejection`, transcript-strand composition and ONT3/HiFi2 thresholds from the frozen scorer. It does NOT invoke the scorer's cs-anchor/correction, molecule assignment, complete-chain/callability or B1 execution logic. The real run must not be described as validation of those rules.

## Minimal operational changes before submission

### 1. Replace the invented quota gate explicitly

Recommended least-work amendment: retain the fresh private OUT, fixed tiny inputs, trusted pinned finite pipeline, no retries, per-file hard 4 MiB during workload, TMPDIR/CWD beneath OUT, 4 CPUs/4 GiB/360 seconds and aggregate output review threshold of 256 MiB. Record the resolved scratch and log filesystem, mount/type, available bytes/inodes at start, ownership and absence of overlap. Controller explicitly accepts a **non-hard aggregate disk bound** for this fixture alone.

Use truthful approval fields such as `output_policy: finite_fixture_soft_aggregate`, `aggregate_review_limit_bytes: 268435456`, `aggregate_hard_limit_bytes: null`, with evidence references. Amend the equality check at fixture.py:244 and quota language in PLAN/sbatch; regenerate manifest and approval bindings. Do not merely change the narrative while the executable still requires a hard quota attestation.

The existing postcheck is logical byte size under OUT before final receipt/COMPLETE; it excludes Slurm logs, final publication growth, allocated-block overhead, and transient/deleted files. Call it a post-run threshold, not a hard cap. Controller's final review should include the complete output and scheduler logs, report logical and allocated sizes, and reject threshold excess. Free filesystem capacity is an observation, not a reservation. No guarantee against arbitrary malfunction or malicious tools is being claimed.

Optional alternative: an already available tmpfs charged to the same verified 4-GiB job memory cgroup can bound combined RAM plus resident tmpfs usage, without provisioning a new 256-MiB disk quota. Evidence must identify mount and paths, cgroup membership of writers and descendants, effective memory.max (or v1 equivalent), swap limit/accounting, and confirm pages are charged to that job rather than a shared/preexisting writer. Any external logs or copies remain separately accounted. With swap enabled, 4-GiB memory.max alone is not a 4-GiB total tmpfs storage bound. A tmpfs mount size alone may also be shared, not dedicated. This is an optional verified containment choice, not a required new mount/job/wrapper. Do not label it a 256-MiB cap or treat tmpfs OOM as acceptable completion.

### 2. Make resource and failure claims exact

`fixture.sbatch:25` uses Bash `ulimit -f 8192`. In ordinary non-POSIX Bash units are 1024 bytes, so this is 8 MiB, not the documented outer 4 MiB. Set/verify a 4194304-byte hard and soft limit in the actual launch environment if the plan promises it from startup. `fixture.py:272` correctly sets 4 MiB for the workload and descendants. Neither mechanism necessarily limits Slurm's separately opened log files.

`wrapper.child` creates a separate session and cleans that process group on normal return, exceptions and its 240-second wait timeout. However, fixture.run does not install the production wrapper's SIGTERM/SIGINT handlers. An outer timeout signal can terminate the harness without executing Python finally, leaving its separate driver group for Slurm to reap. Preflight failures before line 275 also have no fixture FAILED receipt; a hard kill may leave partial evidence. These are not false successful runs under the stated acceptance conjunction, but **300 seconds is not independently proven descendant cleanup**. Minimal acceptable policy: retain verified scheduler cgroup containment/reaping at 360 seconds, explicitly describe the outer timeout as best effort and require all job/batch/extern states and diagnostics. Alternatively narrowly add signal-to-exception cleanup in the fixture, without touching frozen code. No successful marker alone can override timeout, signal, incomplete receipt, FAILED, OOM or unexplained extern status.

A failure after COMPLETE may leave that marker because the fixture exception handler does not withdraw it. Current acceptance of matching receipt hash plus no FAILED plus final stdout plus clean accounting prevents marker-only acceptance. Keep that conjunction; withdrawing the marker would be a useful small hardening, not grounds for a new wrapper project.

### 3. Supply concrete runtime evidence, not an impossible global proof

runtime.json pins plausible resolved paths and versions, and the recorded read-only probe agrees. The chosen Python3.11 avoids the observed local 3.8 incompatibility. Current metadata proves executable identities only. Before setting runtime_closure_approved, attach a bounded read-only record of:

- Host/node, OS/architecture, time, exact executable realpaths/symlink targets, hashes, versions and PATH resolution for minimap2, samtools and seqkit. Keep their shared RNASEQ bin-directory linkage explicit, including nested samtools view.
- Loader/interpreter and resolved shared-object paths for each dynamic executable and relevant Python extension modules, with hashes or a controller-approved immutable environment/package manifest binding those files. Record static linkage explicitly where applicable. Conda package names/version/build/channel records identify provenance but alone do not prove installed library bytes.
- Python sys.executable, version, prefix/base_prefix, sys.path and actual module origins for the harness/helper imports under both isolated launcher and ordinary frozen helper startup. Bind the relevant standard-library/extension installation through a verified environment manifest or an import-origin/hash inventory; inspect system site initialization, .pth, sitecustomize and usercustomize effects. No NumPy/BLAS runtime approval is necessary merely because unused mvp_bootstrap.py is in the source closure.
- Effective environment at top-level launch and child launch, particularly LD_PRELOAD, LD_LIBRARY_PATH, LD_AUDIT and Python startup variables. The harness cleans child env, but sbatch only unsets PYTHONPATH/PYTHONHOME; `-I` does not neutralize the native loader or system site hooks. Sanitize or explicitly approve the top-level startup environment before executing imports.
- Ownership/write permissions and controller basis for no mutation during the one job, plus compute-side confirmation that relevant paths and resolution match the recorded login observations. Identify the scheduler/timeout executables used for operational containment as well.

These records support a narrow trusted-runtime decision, not proof of all possible code paths or malicious-runtime resistance. They can be attached to the separately pinned controller approval without enlarging the frozen nine-file closure. No installs or remote probing were done in this review.

## Exactly what a successful real run would validate

With the amendments and evidence satisfied, one 4-CPU/4-GiB/6-minute job may validate interoperability of the specified real minimap2/seqkit/samtools/Python runtime with this frozen synthetic ONT path: sampling all seven reads, unresolved orientation, pass1 selection, no-uf pass2, real BAM sort/index/view, frozen junction table and fixed threshold oracle, linked command/exit records, byte identities and postflight consistency. The nine-entry ledger is eight driver subprocesses plus the audit; samtools view is real but not a separate ledger entry. Scheduler evidence validates only this allocation's bounded behavior.

It does not validate canonical reference/read authority, full50 callability, PASS/-uf, HiFi/pooling, full-run memory/throughput, all adversarial parser cases, production wrapper authorization, biological conclusions, training eligibility or missingness policy. Historical externOOM remains a separate full-ONT blocker even if this tiny job succeeds. F4 closure is a subsequent controller decision limited to the declared seam-based sub-scope.

## Work record

Completed: independent path/source review, six local tests on compatible local Python, manifest/shell validation, nonmutation evidence, quota proportionality decision and exact remaining metadata list. Blocked pending controller action: explicit operational amendment and re-pinning; runtime and scheduler containment evidence; authorization, real run and independent result/accounting acceptance. No additional review-side implementation TODO remains.
