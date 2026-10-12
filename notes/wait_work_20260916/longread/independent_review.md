# Independent Task 3 review — first orientation pilot

## Verdict: APPROVE bounded pilot only

Approve the reviewed sbatch's **one SRR28341943 / 10,000-record orientation pilot**, conditional on the controller checks below. This is not PILOT_COMPLETE, approval of `verify_alignment`, full independent structural availability, full-run execution, ONT-only policy, or training authorization. No actual pilot has run in the evidence supplied. No SSH, jobs, source edits, or subagents were used for this review.

Reviewed main-directory artifacts against actual Git objects at `b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6`, not main's implementation. All nine `source_pins.json` entries matched both `git show b660…:<path>` and the frozen exported bytes. Actual Git92 authority blob matched `authority.tsv` and hardcoded SHA256 `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. Graft was consulted before extra source inspection (it auto-refreshed its index; no tracked graft diff was observed).

## Findings and acceptance conditions

### MEDIUM — COMPLETE alone is not a successful pilot

`gate.py:207–214` writes COMPLETE **before** `verify_publication`, which performs substantial fresh hashing. A normal verification exception leaves COMPLETE plus FAILED.json. A scheduler kill during that verification can leave COMPLETE with no FAILED and no final JSON stdout. Thus REPORT's assertion that a timeout implies “no COMPLETE” is not universally true.

Reproduction by control-flow inspection: interrupt the worker after line 209, before line 211 returns; or force `verify_publication` to raise after COMPLETE creation. Neither requires a failed external alignment command. Do not run this interruption on actual data.

**Binding condition for approval:** controller must require all of: independently pinned release, successful workload/scheduler accounting (inspect batch/extern, reject timeout/OOM/nonzero), complete parseable final worker stdout equal to receipt.json, absence of FAILED, matching COMPLETE receipt hash, and successful compute-side `verify_publication`. Never publish PILOT_COMPLETE from marker existence or audit PASS alone. This condition is already compatible with REPORT's controller acceptance procedure; marker-only automation would be blocked. Fix publication ordering before treating this wrapper as a standalone completion authority.

### LOW — failure capture and isolation are controller-dependent at the boundaries

`gate.py:166–178` preflight/hash failures precede the try block; output may not exist, or may exist without FAILED. `subprocess.run` launch exceptions cannot have a returned exit code. Scheduler termination can prevent FAILED entirely. Preserve Slurm stderr/accounting and classify these as incomplete, not missing-negative evidence.

`gate.py:173–175` refuses an existing final output directory but does not reject symlinked ancestors or require OUT outside RELEASE. Reproduction: an OUT beneath a symlinked parent passes the leaf `is_symlink()` test. Controller must select new absolute real directories, reject redirected ancestors, keep OUT outside the source release and canonical inputs, and verify release freshness before/after copying. Runtime deployment checks remain controller work, not certified here.

### Deferred full-execution gate — not covered by this approval

`gate.py:102–138` is an untested full-output consistency helper, not an operational full alignment gate. It does not completely validate the first two orientation argv entries; it relies on a trusted external execution receipt. Frozen full driver `mvp_align_panel_longreads.py:181–200` uses bare minimap2/samtools/seqkit and the junction CLI defaults to PATH samtools (`mvp_junction_table.py:193`). The pilot's absolute binary paths do not establish that later command graph's executable identities. Full execution requires a separately reviewed environment, actual tool fixture, stdout/exit/recursive inventory integration, and full postflight tests. None is needed merely to launch this restricted orientation experiment; none is implicitly approved.

## Verified connection contracts

- `gate.py:44–60`: consumed reads must be exactly canonical ROOT/evidence/ont/Athaliana_FLIC_PRJNA1087576/RUN.fa.gz, without source symlink redirection. Exactly one run row must bind dataset, Athaliana/reference and b1_validation_only. Fresh consumed compressed FASTA MD5 is checked against **local_fa_md5**, not ENA's FASTQ `source_checksum`. Roles themselves are SHA256-bound and rechecked. ROOT is fixed in the reviewed sbatch, not inferred from a repository role copy.
- `gate.py:62–70`: fixed Git92 blob digest, unique Athaliana row, actual FASTA/GFF MD5 and SHA256; arbitrary caller-provided reference MD5 cannot replace authority. Supplied observations are metadata/small receipts, not a fresh remote hash or recount performed by this reviewer.
- `gate.py:72–82,168–172`: frozen file pins checked before driver import, imported windows module path checked, actual consumed panel checked by frozen loader. `frozen/src/transgenic/evidence/mvp_windows.py:94–172` binds panel bytes/counts and centered 30,720-nt half-open windows, odd padding genomic-right, no evaluation-window clamp; prefilter adds ±10,000. Pilot does not perform full window/genome evidence coverage validation.
- `gate.py:190–205` and frozen `68_sample_fasta.py:27–68`: sampler's actual CLI accepts --fasta/--out/--n/--md5; deterministic sample seeded from fresh input MD5, two complete decompression scans, exact sampled=10000 and run_records=roles expected count checks. Observed first input is 1,915,883,586 compressed bytes / expected 6,952,504 records. Hashing adds further full compressed-byte reads; the job is not merely a cheap 10,000-record prefix read.
- Frozen `mvp_align_panel_longreads.py:100–106`: actual pilot argv is absolute minimap2, `-t 4 -ax splice -k14 -G 200000 --secondary=no --cs=long`, full fixed reference and retained sample.fa; **no -uf**. No pass1/full-run alignment or junction scoring is called by pilot.
- Frozen `67_orientation_audit.py:271–344` accepts the exact scorer argv. It returns 0 for PASS and FAIL, 4 for UNRESOLVED. `gate.py:186,202–205` accepts 0/4 and then verifies the frozen count-derived decision. DONE binds read/reference/GFF/**67 scorer** MD5s, run, and audit JSON MD5. Frozen `read_audit:65–80` enforces MAPQ20, minimum1000, threshold .95, count consistency and only-PASS use_uf. FAIL/UNRESOLVED may be a successfully completed orientation experiment, never proof of absent independent evidence. The frozen scorer determines rc4 semantics; the wrapper does not independently require rc4 iff status UNRESOLVED.
- `orientation_pilot.sbatch:2–28`: one job, 4 CPUs, 8G, 30 minutes, no array/retry/GPU. Explicit sylvan Python, RNASEQ_bch709 binaries, PYTHONPATH unset and user site disabled. Reported versions are minimap2 2.28-r1209, samtools 1.3.1, seqkit 2.13.0; pilot captures actual versions and binary identities, but does not enforce expected version strings. Controller must compare runtime observations to approved environment; no live binary certification is claimed here.
- `gate.py:84–100,143–163,180–210`: external argv/stdout/stderr/returned exit captured, retained sample/SAM/audit/DONE recursively inventoried by relative path, input/source/tool bytes rechecked. Final receipt always orientation_sample / UNASSESSABLE / training_eligible=false / decision=null. `require_final_coverage` rejects unconditionally. No training filter bypass or HiFi substitution exists.

## Remaining gates, explicitly open

1. Controller runtime release/copy/hash freshness, isolated fresh OUT, live ownership/queue/overlap/budget checks; no submission performed here.
2. Actual compute-side canonical read/reference hashes, full-run record recount, exact environment compatibility, resource measurements, worker stdout and all Slurm accounting. Thirty minutes is a bound, not demonstrated adequacy; no automatic retry.
3. Post-run conjunction above and compute-side revalidation, with retained failed/incomplete artifacts if any check fails.
4. Separately reviewed full-run alignment wrapper/postflight, scratch and throughput budget, actual selected-read/BAM/junction tool fixture. `verify_alignment` is not certified by 11 existing gate tests.
5. All run/locus evidence coverage, per-run and pooled repeated-alignment scoring with ONT3/HiFi2 thresholds, exact scorer identities and established window/gene-denominator distinction. A sample or one run cannot fill this gate.
6. HiFi acquisition/identity/role availability or explicitly authorized ONT-only policy; missing HiFi is not zero support. No policy selected here.
7. Finalizer/checkpoint-bound 300 paired predictions plus 50 baseline, counts producer integration, unchanged score/bootstrap, withholding/duplicate-isoform policy and final decision authorization. No input-training or STOP/GO authorization follows from this review.

## Verification evidence and reviewed hashes

Independent checks: all nine Git/export/source-pin comparisons passed; Git92 blob comparison passed; `bash -n orientation_pilot.sbatch` passed. Inspected existing test source rather than broadly rerunning controller tests. Controller-reported **11 tests in 1.349 s** remains controller evidence; those tests include a real frozen scorer synthetic PASS fixture but not actual minimap2, full pilot execution, full postflight, or a publication-success integration fixture. Advisor attempt failed due client/model-version incompatibility; no advisor endorsement is claimed.

SHA256 at review:

| Artifact | SHA256 |
|---|---|
| gate.py | 0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f |
| orientation_pilot.sbatch | e54378e5f4d86659d6ba44d4fe8ef88c53ca6b57b6ff5f0e939519268bb39832 |
| source_pins.json | 38c56c608336ef3fdcb83eaa5eb35a355ddc77ea1e8aff8387160603bf88ce34 |
| authority.tsv | 3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783 |
| test_gate.py | f3b39f5358bfcb25684adb7959fd21af970185df1a71ced3e0c3c2427361adab |
| REPORT.md | db0e3a31173108a456e5b242b8c555dec034a7acf6691a53cbffe11a4bee0666 |

Approval is for these bytes and this bounded scope, subject to the explicit controller conditions; subsequent edits require freshness/diff review.
