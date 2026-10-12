# ONTF4 tiny real-tool fixture candidate

## Disposition and ownership

Prepared locally only. **ONTF4 OPEN: no real-tool fixture has run.** The controller must review this candidate before any deployment or single bounded job. No canonical read/reference was hashed, no alignment ran on login, no remote write/install/submission occurred. No full ONT authorization. Existing externOOM is unresolved and remains a separate blocker.

All new artifacts are under `notes/auth_resume_20260917/ont_fixture/`. Reviewed wrapper, old gate, source pins and nine frozen files were copied byte-identically into `package/`; originals are not edited. `package/frozen/` contains exactly the nine pinned files plus necessary directories, not a widened source closure. Harness code is outside that closure. No X/missingness, pooling, thresholds, training or source-acquisition policy change.

Read: recovery `ont_full/FIX_REPORT.md`, `ONT_FIX_CONTROLLER_REVIEW.md`, old longread `REPORT.md`, `independent_review.md`, `source_pins.json`, actual frozen driver, junction CLI, windows loader, old gate scientific postflight and wrapper helpers. Advisor planning was attempted but failed due installed client/model incompatibility; no independent approval claimed.

## Design chosen and explicit test seams

Use the original driver `main(argv)` and actual CLI parser in an isolated test launcher, not a reimplementation of the eight-command pipeline. It consumes seven synthetic reads, a 64,000-base two-contig synthetic reference, synthetic GFF, and a 50-row synthetic panel. Generated input bytes are below 150,000 bytes. All synthetic outputs remain outside canonical data and the immutable release.

Seams are necessary because frozen canonical hashes cannot truthfully accept invented inputs:

1. Fresh synthetic ROOT uses the frozen SRR28341943 enum and directory shape solely to exercise frozen validation/ONT routing. The role row is explicitly marked SYNTHETIC_TEST_ONLY. This is not that accession's reads or scientific authority. Unchanged `validate_read` and `gate.read_binding` still perform real schema, role, filename and compressed-MD5 checks against this test root.
2. `panel_seam` delegates to the unchanged `load_panel` explicit fixture-hash API, preserving its field validation, counts 20+30, duplicate-ID checks and windows. It changes the in-memory panel hash label only for the fixture process. Fifty different TEST_ONLY gene IDs intentionally share one safe 21-kb extent. They do not represent 50 biological loci or canonical callability. No panel bytes inside `frozen/` change.
3. The driver is launched via `fixture.py entry ... driver -- <original argument tail>`. All actual minimap2, seqkit, samtools, sampler and audit invocations retain their original argv. The nested junction Python invocation alone is bootstrapped through `fixture.py entry ... junction -- <original argument tail>` to install the same panel seam before invoking unchanged junction `main`. Both logical and actual argv and returned exits are retained in `nested_execution.jsonl`. This is deliberately NOT a claim of identical top-level OS argv to a direct production script launch.
4. `gate.verify_alignment` is called unchanged with a test-only replacement for `reference_binding` that validates the tiny synthetic authority bytes and actual FASTA/GFF MD5s. It does not pass or weaken the canonical Git92 authority check in production. Every remaining scientific postflight check runs against real output bytes.
5. The production wrapper `execute` is not called: its canonical ROOT/approval semantics must not be misrepresented. Its unchanged exact-closure checks, identity helpers and process-group child/timeout cleanup are reused. A separate fixture receipt records scope and seams. This candidate therefore tests the frozen pipeline/postflight path, not the whole production authorization wrapper.

Raw frozen labels are preserved. In particular the junction QC's `panel_sha256_is_frozen=true` only reflects its comparison to the fixture-bound in-memory constant. It is **not** canonical-panel attestation. The outer receipt and this document override no raw artifact; they explicitly identify that limitation. `reference_identity_status=UNVERIFIED`, `scientific_input_validation=false`, `coverage_policy=PENDING`, training=false and decision=null remain untouched.

If controller requires absolutely unchanged OS-level Python launch argv, this seam-based candidate is insufficient: do not silently declare F4 closed. A separate reviewed approach would be needed. Do not modify frozen code or widen its pins to force acceptance.

## Fixed oracle, before any aligner run

Deterministic RNG seed 20260917, compressed FASTA mtime zero. One unique 60-kb fixture chromosome and one 4-kb outside chromosome. Two canonical GT-AG introns, 200 nt each, with 500-nt exact exons on either side and boundary bases chosen to avoid simple junction sliding.

| Junction, zero-based inclusive | Input distinct names | Expected n_ONT | Supported |
| --- | --- | ---: | --- |
| fixture_chr:20500..20699, + | positive_0, positive_1, positive_2 | 3 | true |
| fixture_chr:30500..30699, + | below_threshold_0, below_threshold_1 | 2 | false |

Exact table equality requires those two rows only, n_HiFi=0 for this synthetic input only. Shifted +9 decoy 20509..20708 must not appear. One contiguous unspliced read remains selected but contributes no intron. One outside-contig read must be absent from the selected names. This tests a positive junction, a below-threshold negative, absence of invented decoys, selection and unspliced handling. It is not an exhaustive parser adversarial suite.

The actual sampler requests 10,000 but sees all seven records. Frozen orientation MUST be UNRESOLVED, exit 4, use_uf=false. This tiny fixture deliberately cannot exercise PASS/-uf: that is an explicit coverage limit, not a reason to relax min_eligible=1000. No synthetic count is interpreted as biological missingness.

Eight original driver ledger commands: sample; no-uf orientation minimap2; full-fixture pass1 PAF; seqkit selected-ID extraction; frozen pass2 splice SAM; samtools sort; samtools index; frozen junction CLI. The actual audit CLI between commands 2 and 3 is separately captured, making nine subprocess ledger entries. The junction CLI's own samtools-view subprocess is real and covered by the pinned PATH/tool and frozen parser, but is not separately logged by this harness.

## Intended runtime, read-only observations

`runtime.json` records resolved paths/hashes/version expectations obtained read-only on pronghorn login-0 at 2026-09-17T16:03:40-0700; `readonly_runtime_probe.txt` retains output. RNASEQ_bch709 minimap2 2.28-r1209, samtools 1.3.1 / htslib 1.3.1, seqkit 2.13.0; sylvan resolved Python3.11 binary reports 3.11.14. No local 2.22/1.12/2.3 tool substitute was used.

Compute launcher verifies exact interpreter path, four executable hashes, first version lines and zero exits before alignment, then rehashes executables afterward. Frozen PATH resolves only the observed RNASEQ binary directory, including nested samtools. Clean subprocess environment prevents user PYTHONPATH/site injection; unchanged non-isolated helpers still rely on the approved external interpreter installation. Linked libraries, Python standard library/site initialization and installation immutability still require independent controller runtime-closure approval. Binary hashes alone do not satisfy that approval.

## Proposed one-job resource bound

Candidate `fixture.sbatch`: single node/task, 4 CPUs, 4 GiB, 6 minutes, no GPU, array, retry or requeue. Maximum allocation budget 0.4 CPU-hours. This is a proposed upper allocation, not measured resource adequacy.

- srun binds four hardware threads; harness verifies affinity count and allocation environment.
- Outer timeout 300 seconds, 10-second TERM-to-KILL grace, within 360-second Slurm bound. Driver group deadline 240 seconds plus 5-second cleanup. Each version flag timeout 10 seconds. A CPU-time rlimit of 240 seconds per process is a secondary safety net, not aggregate CPU accounting.
- Slurm must actually enforce a 4-GiB aggregate memory cgroup. No misleading RLIMIT_AS is used for Go seqkit's address-space reservation.
- Per-file hard limit 4 MiB after preflight and inherited by tools; outer shell also limits files to 4 MiB. Aggregate output audited below 256 MiB. **A separately verified dedicated filesystem hard quota of 256 MiB, including Slurm logs, is mandatory for a true disk upper bound.** A final size check or `--tmp` request is not hard enforcement. Controller must provision/verify the quota outside this task or block submission.
- Scheduler cgroup must contain and reap escaped descendants; process groups alone do not certify containment. Live batch/extern accounting and any OOM event remain controller acceptance gates.

The sbatch logs `scontrol show job -o`; harness allocation/environment checks are not a substitute for the controller checking actual owner/node/resources/cgroups. Release and output must be fresh real absolute directories, immutable release outside output/canonical data. Controller must confirm no overlap and no concurrent work exceeding approved budget.

## Package and review-only submission procedure

Deploy only the files named in `MANIFEST.sha256` plus that manifest, retaining relative layout. It contains harness, runtime expectations, sbatch and exact frozen package. Tests, stubs and local logs are deliberately excluded. Controller independently retains manifest SHA256, verifies no unlisted runtime source shadows and verifies hashes after staging; don't mutate the historical release to obtain this package.

No approval JSON is supplied by this worker. After independent code/runtime/containment review, controller authors a separately pinned approval with:

```
scope: TEST_ONLY_REAL_TOOL_FIXTURE
runtime_closure_approved: true
scratch_hard_limit_bytes: 268435456
output: <fresh absolute OUT beneath verified quota>
cpus: 4
memory_mib: 4096
wall_seconds: 360
manifest_sha256: <independently retained MANIFEST.sha256 digest>
```

These fields are attestations, not automatic runtime certification. Include controller evidence references as extra fields. Export RELEASE, OUT, MANIFEST_SHA256, ONT_FIXTURE_APPROVAL and ONT_FIXTURE_APPROVAL_SHA256. Review the manifest hash and candidate sbatch; only then controller may perform one bounded `sbatch --parsable` with explicit stdout/stderr paths under the same quota but outside fresh OUT. No submission command was executed here.

## Required evidence and acceptance conjunction

Retrieve complete output tree, Slurm stdout/stderr, original release/manifest/approval identities, scontrol output, and sacct job/batch/extern State, ExitCode, DerivedExitCode if available, Elapsed, TotalCPU, AllocCPUS, ReqMem, MaxRSS and cgroup/OOM diagnostics. Reject any TIMEOUT/OOM/CANCELLED/nonzero or unexplained extern event. No automatic retry or budget increase.

Require all of:

1. Submitted/staged package and runtime match independently retained identities and runtime-closure approval. Four CPUs, memory/quota bounds and containment actually enforced.
2. Workload and all scheduler components successful; final one-line harness JSON in Slurm stdout equals final receipt. scontrol/manifest output preceding that line is not driver stdout.
3. No FAILED anywhere; COMPLETE hash equals receipt hash. Marker alone is never sufficient. A termination during final publication can leave partial evidence; classify incomplete.
4. `driver.stdout` parses as exact `alignment.json`, driver exit 0, original stdout/stderr identities remain unchanged, exact recursive scored-output inventory and MD5 links pass unchanged gate verification.
5. Fixed two-row oracle, selected names, audit UNRESOLVED/no-uf, eight frozen logical commands plus actual audit, corresponding real executable exits and declared junction seam. Source, input and tool identities unchanged. Fixture diagnostics inventory includes synthetic input, ledger and runtime observations; receipt/COMPLETE are excluded from recursive self-hashing.
6. Recheck published diagnostics hashes, release manifest, raw outputs and actual accounting independently. Receipt status `FIXTURE_WORKLOAD_PASS_PENDING_CONTROLLER` is not controller acceptance.

Potential real alignment differences are failures against the precommitted oracle, not permission to update expected rows after seeing the output. Keep failure artifacts and review cause. This tiny job does not test full-run memory/throughput, canonical input authority, full50 callability, PASS/-uf, pooling/HiFi or production wrapper acceptance.

## Work record and remaining TODO

Done: unchanged-source package, deterministic fixture generator, isolated frozen CLI bootstrap, explicit seams, real postflight/oracle checks, captured actual argv/exits, input/source/tool/stdout/stderr/output identity checks, read-only runtime resolution, candidate sbatch, local preparation tests.

Local tests include deterministic byte identity, tiny bound, refusal to overwrite, source/closure pins, real frozen no-execute CLI validation, default canonical-panel rejection, actual junction --help imports, and a complete **stub-tool** plumbing run through the real frozen driver/scorer/postflight including stdout-mutation rejection. Stub BAM is text, not a real BAM. These tests are never real-tool evidence and do not close F4. `green_tests.txt` records local results. `red_tests.txt` records the initial missing-harness failure before implementation.

Pending controller: independent candidate review, external Python/runtime closure, immutable deployment, hard quota and memory/CPU cgroup proof, fresh allocation ownership/overlap review, single bounded job if approved, independent output/scientific/accounting review. Historical externOOM remains unresolved. Only controller can decide whether this seam-limited real fixture closes the intended ONTF4 sub-scope. Full ONT remains blocked regardless.
